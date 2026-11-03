// File-based rendering for the Premiere GPU filter (included at the end of
// frame_blend_gpu.cpp). The output frame's media time is the clip's in point
// plus the render's clip time; the core's kernel is evaluated at the file's own
// exact frame rate and every tap is a frame decoded from the file, uploaded to
// the device once and blended there.

namespace {

// Device copy of a host-decoded frame (caller holds slot_mutex). Uploads it
// into the least recently used slot not needed by the current render. Slots
// match the decoded frame itself: indices in a gap of a variable frame rate
// recording share their held frame.
Instance::Slot* upload_frame(Instance& inst, std::shared_ptr<const fb::media::Frame> frame, bool cuda, size_t bytes,
                             size_t capacity, bool& uploaded) {
    uploaded = false;
    const int64_t index = frame->index;
    for (Instance::Slot& slot : inst.slots) {
        if (slot.host == frame) {
            slot.stamp = inst.render_stamp;
            return &slot;
        }
    }
    if (frame->nv12.size() != bytes) {
        return nullptr;
    }
    Instance::Slot* target = nullptr;
    if (inst.slots.size() < capacity) {
        void* memory = nullptr;
        if (inst.device->AllocateDeviceMemory(inst.device_index, bytes, &memory) != suiteError_NoError || !memory) {
            return nullptr;
        }
        inst.slots.push_back(Instance::Slot{-1, memory, cuda ? fb::gpu::cuda_event_create(inst.info.outContextHandle) : nullptr,
                                            0, nullptr});
        target = &inst.slots.back();
    } else {
        for (Instance::Slot& slot : inst.slots) {
            if (slot.stamp != inst.render_stamp && (!target || slot.stamp < target->stamp)) {
                target = &slot;
            }
        }
        if (!target) {
            return nullptr;
        }
    }
    // The previous upload into this slot must be done before its host frame
    // goes back to the reader to be decoded into again.
    if (cuda) {
        fb::gpu::cuda_event_wait(inst.info.outContextHandle, target->uploaded);
    }
    target->host.reset();
    const bool ok = cuda ? fb::gpu::cuda_upload(inst.info.outContextHandle, inst.info.outCommandQueueHandle,
                                                target->memory, frame->nv12.data(), bytes) &&
                               (!target->uploaded || fb::gpu::cuda_event_record(inst.info.outContextHandle,
                                                                                target->uploaded,
                                                                                inst.info.outCommandQueueHandle))
                         : fb::gpu::opencl_upload(inst.info.outCommandQueueHandle, target->memory, frame->nv12.data(),
                                                  bytes);
    if (!ok) {
        target->index = -1;
        return nullptr;
    }
    target->index = index;
    target->stamp = inst.render_stamp;
    target->host = std::move(frame);
    uploaded = true;
    return target;
}

// Rate at which the clip plays the file's frames: the file's exact rate times
// the clip's constant speed. Premiere's speed is relative to its own reading of
// the stream rate (a 600 fps recording read as 600.024 makes 200 % arrive as
// 1.99992), so it is converted to the exact rate and snapped to the 0.01 %
// steps of the Speed/Duration dialog: 200 % of 600 fps plays 1200 frames per
// second, and an output frame at 30 fps then blends 40 of them.
FbRational playback_rate(const Instance& inst, FbRational fps) {
    double speed = inst.speed;
    if (inst.stream_ticks_per_frame > 0 && fps.num > 0 && fps.den > 0) {
        speed *= static_cast<double>(kTicksPerSecond) / static_cast<double>(inst.stream_ticks_per_frame) /
                 fb::to_double(fps);
    }
    if (!std::isfinite(speed) || speed <= 0.0 || speed > 1000.0) {
        fb::debug::line("GPU_ERROR playback rate: speed=%g stream_ticks=%lld fps=%s; using the file's rate", inst.speed,
                        static_cast<long long>(inst.stream_ticks_per_frame), fb::text(fps).c_str());
        speed = 1.0;
    }
    const int64_t steps = std::max<int64_t>(1, std::llround(speed * 10000.0));
    int64_t num = fps.num * steps;
    int64_t den = fps.den * 10000;
    const int64_t g = std::gcd(num, den);
    return fb::rational(num / g, den / g);
}

// ---- time remapping ----

// Media speed of the constant segment from key i to key i + 1.
double remap_slope(const std::vector<RemapKey>& keys, size_t i) {
    return (keys[i + 1].position - keys[i].position) / (keys[i + 1].time - keys[i].time);
}

// Media position at clip time t (seconds). Constant speed between keyframes;
// a ramp (type 7 to type 8) blends the speeds of its neighbouring segments
// with a cubic Hermite curve, which for Premiere's default linear speed change
// is exactly its quadratic. Outside the keyframes the end speeds continue.
double remap_position(const std::vector<RemapKey>& keys, double t) {
    const size_t n = keys.size();
    if (t <= keys.front().time) {
        return keys.front().position + remap_slope(keys, 0) * (t - keys.front().time);
    }
    if (t >= keys.back().time) {
        return keys.back().position + remap_slope(keys, n - 2) * (t - keys.back().time);
    }
    size_t i = 0;
    while (i + 2 < n && keys[i + 1].time <= t) {
        ++i;
    }
    const RemapKey& a = keys[i];
    const RemapKey& b = keys[i + 1];
    const double span = b.time - a.time;
    const double s = (t - a.time) / span;
    if (a.type == kRemapRampStart && b.type == kRemapRampEnd) {
        const double v0 = i > 0 ? remap_slope(keys, i - 1) : remap_slope(keys, i);
        const double v1 = i + 2 < n ? remap_slope(keys, i + 1) : remap_slope(keys, i);
        const double s2 = s * s, s3 = s2 * s;
        return (2 * s3 - 3 * s2 + 1) * a.position + (s3 - 2 * s2 + s) * span * v0 + (-2 * s3 + 3 * s2) * b.position +
               (s3 - s2) * span * v1;
    }
    return a.position + (b.position - a.position) * s;
}

// Clip time whose media position is closest to `position`, searched over the
// keyframes' range (1 ms steps, then refined). Used once per instance to find
// how the curve's clip time relates to the sequence (e.g. a trimmed head).
double remap_time_of(const std::vector<RemapKey>& keys, double position, double around) {
    const double first = keys.front().time - 1.0;
    const double last = keys.back().time + 1.0;
    double best = around, best_error = std::fabs(remap_position(keys, around) - position);
    for (double t = first; t <= last; t += 0.001) {
        const double error = std::fabs(remap_position(keys, t) - position);
        if (error < best_error - 1e-12 || (std::fabs(error - best_error) <= 1e-12 && std::fabs(t - around) < std::fabs(best - around))) {
            best = t;
            best_error = error;
        }
    }
    double step = 0.0005;
    for (int k = 0; k < 40; ++k, step *= 0.5) {
        for (double t : {best - step, best + step}) {
            const double error = std::fabs(remap_position(keys, t) - position);
            if (error < best_error) {
                best = t;
                best_error = error;
            }
        }
    }
    return best;
}

// The kernel's inputs for one output frame: R = rate / output source frames
// wide, centred on `center` (file frames).
struct Window {
    FbRational rate{0, 1};
    FbRational output{0, 1};
    FbRational center{0, 1};
};

// Constant speed: Premiere hands effects the clip time as the position in the
// file itself, with speed and reverse applied (it counts down on a reversed
// clip) and the in point included (host test 2026-10-05, numbered 600 fps
// clip trimmed to start at 1.97 s). The window is the playback rate over the
// output rate, centred on that position.
bool constant_speed_window(const Instance& inst, const PrGPUFilterRenderParams* params, FbRational fps, Window& w) {
    w.output = fb::rational(kTicksPerSecond, params->inRenderTicksPerFrame);
    w.rate = playback_rate(inst, fps);
    return fb_center_for_output_frame(fps, w.output, fb::rational(0, 1),
                                      fb::rational(params->inClipTime, params->inRenderTicksPerFrame), &w.center) == FB_OK;
}

// Time remapping: the output frame spans one sequence frame of clip time; the
// curve maps that span to the media it covers, which is the blend window. On
// a constant stretch this is the constant-speed window; through a ramp it
// widens or narrows smoothly. The curve is checked against the position
// Premiere hands this render; false when they disagree.
bool remap_window(Instance& inst, const PrGPUFilterRenderParams* params, FbRational fps, Window& w,
                  std::string& error) {
    const double ticks = static_cast<double>(kTicksPerSecond);
    const double premiere = static_cast<double>(params->inClipTime) / ticks;
    const double sequence = static_cast<double>(params->inSequenceTime - inst.track_item_start) / ticks;
    // Premiere counts media seconds at its own reading of the stream rate.
    const double media_fps = inst.stream_ticks_per_frame > 0 ? ticks / static_cast<double>(inst.stream_ticks_per_frame)
                                                             : fb::to_double(fps);
    const double tolerance = 0.25 / media_fps;  // a quarter of a recorded frame
    if (!inst.remap_offset_known) {
        inst.remap_offset = remap_time_of(inst.remap, premiere, sequence) - sequence;
        inst.remap_offset_known = true;
    }
    const double t = sequence + inst.remap_offset;
    const double position = remap_position(inst.remap, t);
    if (!std::isfinite(position) || std::fabs(position - premiere) > tolerance) {
        error = "time remapping curve gives " + std::to_string(position) + " s where Premiere shows " +
                std::to_string(premiere) + " s";
        return false;
    }
    const double half = 0.5 * static_cast<double>(params->inRenderTicksPerFrame) / ticks;
    const double a = remap_position(inst.remap, t - half);
    const double b = remap_position(inst.remap, t + half);
    // Width in recorded frames, in 1/10000 steps; under a frame (a freeze or a
    // crawl) the kernel passes the nearest frame through.
    const int64_t width = std::max<int64_t>(1, std::llround(std::fabs(b - a) * media_fps * 10000.0));
    const int64_t g = std::gcd(width, static_cast<int64_t>(10000));
    w.rate = fb::rational(width / g, 10000 / g);
    w.output = fb::rational(1, 1);
    // Centre: Premiere's position, moved to the middle of the span (they only
    // differ through a ramp), in ticks so the centre stays an exact fraction.
    const PrTime center_ticks = params->inClipTime + static_cast<PrTime>(std::llround((0.5 * (a + b) - position) * ticks));
    return fb_center_for_output_frame(fps, fb::rational(kTicksPerSecond, 1), fb::rational(0, 1),
                                      fb::rational(center_ticks, 1), &w.center) == FB_OK;
}

prSuiteError render_file(Instance& inst, const PrGPUFilterRenderParams* params, const PPixHand* frames,
                         csSDK_size_t frame_count, PPixHand* out_frame, uint64_t render_id, double t_start) {
    fb::media::Reader& reader = *inst.reader;
    const fb::media::Info& info = reader.info();
    if (params->inRenderTicksPerFrame <= 0 || frame_count < 1 || !frames[0]) {
        return suiteError_Fail;
    }
    Window window;
    std::string remap_error;
    if (inst.remap.empty() ? !constant_speed_window(inst, params, info.fps, window)
                           : !remap_window(inst, params, info.fps, window, remap_error)) {
        if (!remap_error.empty()) {
            fb::debug::line("GPU_ERROR id=%" PRIu64 " %s; passing Premiere's frame through", render_id,
                            remap_error.c_str());
            return render_passthrough(inst, frames, out_frame);
        }
        return suiteError_Fail;
    }
    const FbRational output = window.output;
    const FbRational rate = window.rate;
    const FbRational center = window.center;

    PrPixelFormat format = PrPixelFormat_Invalid;
    prRect bounds{};
    csSDK_uint32 par_num = 1;
    csSDK_uint32 par_den = 1;
    prFieldType field = prFieldsNone;
    inst.ppix->GetPixelFormat(frames[0], &format);
    inst.ppix->GetBounds(frames[0], &bounds);
    inst.ppix->GetPixelAspectRatio(frames[0], &par_num, &par_den);
    inst.ppix2->GetFieldOrder(frames[0], &field);
    const int width = bounds.right - bounds.left;
    const int height = bounds.bottom - bounds.top;
    if ((format != PrPixelFormat_GPU_BGRA_4444_32f && format != PrPixelFormat_GPU_BGRA_4444_16f) || width <= 0 ||
        height <= 0) {
        return suiteError_Fail;
    }
    const fb::gpu::Element element =
        format == PrPixelFormat_GPU_BGRA_4444_32f ? fb::gpu::Element::F32 : fb::gpu::Element::F16;
    const bool cuda = inst.info.outDeviceFramework == PrGPUDeviceFramework_CUDA;
    void* context = inst.info.outContextHandle;
    void* queue = inst.info.outCommandQueueHandle;
    const size_t frame_bytes = static_cast<size_t>(info.width) * info.height * 3 / 2;
    int32_t max_taps = 0;
    fb_max_taps(rate, output, &max_taps);

    // Decode (blocking) without holding the slot lock, so concurrent renders
    // of neighbouring frames overlap; then upload under the lock.
    std::map<int64_t, std::shared_ptr<const fb::media::Frame>> prefetched;
    {
        std::vector<FbTap> wanted;
        fb::gather(rate, output, center, 0, info.frames > 0 ? info.frames - 1 : INT64_MAX / 4,
                   [&](int64_t index) {
                       auto frame = reader.get(index);
                       if (frame) {
                           prefetched[index] = std::move(frame);
                       }
                       return prefetched.count(index) > 0;
                   },
                   wanted);
    }
    std::lock_guard<std::mutex> lock(inst.slot_mutex);
    ++inst.render_stamp;
    std::vector<FbTap> taps;
    // Decoded frames used by this render, and their device images: NVDEC
    // surfaces directly, host frames through an uploaded slot.
    std::map<int64_t, std::shared_ptr<const fb::media::Frame>> decoded;
    std::map<int64_t, fb::gpu::Image> images;
    const int64_t last = info.frames > 0 ? info.frames - 1 : INT64_MAX / 4;
    const int32_t status = fb::gather(
        rate, output, center, 0, last,
        [&](int64_t index) {
            auto found = prefetched.find(index);
            std::shared_ptr<const fb::media::Frame> frame =
                found != prefetched.end() ? found->second : reader.get(index);
            if (!frame) {
                return false;
            }
            if (frame->on_device()) {
                images[index] = fb::gpu::Image{reinterpret_cast<void*>(frame->luma), 0, frame->pitch,
                                               reinterpret_cast<void*>(frame->chroma)};
            } else {
                bool uploaded = false;
                Instance::Slot* slot =
                    upload_frame(inst, frame, cuda, frame_bytes, static_cast<size_t>(max_taps) + 8, uploaded);
                if (!slot) {
                    return false;
                }
                images[index] = fb::gpu::Image{slot->memory, 0, info.width, nullptr};
            }
            decoded[index] = std::move(frame);
            return true;
        },
        taps);
    if (status != FB_OK) {
        // Premiere shows a failed render as black; its own frame is better.
        fb::debug::line("GPU_ERROR id=%" PRIu64 " file render: no frame decoded around %s (%s; %s); passing Premiere's "
                        "frame through",
                        render_id, fb::text(center).c_str(), fb_error_message(status), reader.stats().c_str());
        fb::debug::flush();
        return render_passthrough(inst, frames, out_frame);
    }

    if (inst.device->CreateGPUPPix(inst.device_index, format, width, height, par_num, par_den, field, out_frame) !=
            suiteError_NoError ||
        !*out_frame) {
        return suiteError_Fail;
    }
    void* out_data = nullptr;
    csSDK_int32 out_pitch = 0;
    inst.device->GetGPUPPixData(*out_frame, &out_data);
    inst.ppix->GetRowBytes(*out_frame, &out_pitch);
    const fb::gpu::Image output_image{out_data, 0, out_pitch};
    std::vector<fb::gpu::Tap> gpu_taps;
    for (const FbTap& tap : taps) {
        gpu_taps.push_back({images.at(tap.index), tap.weight});
    }
    const fb::gpu::Yuv yuv = fb::gpu::yuv_coefficients(info.full_range, info.bt601);
    std::string error;
    bool ok = cuda ? fb::gpu::cuda_blend_nv12(context, queue, output_image, gpu_taps.data(),
                                              static_cast<int>(gpu_taps.size()), width, height, info.width,
                                              info.height, yuv, element, error)
                   : fb::gpu::opencl_blend_nv12(queue, output_image, gpu_taps.data(), static_cast<int>(gpu_taps.size()),
                                                width, height, info.width, info.height, yuv, element, error);
    if (!ok) {
        fb::debug::line("GPU_ERROR id=%" PRIu64 " file render: %s", render_id, error.c_str());
        fb::debug::flush();
        inst.ppix->Dispose(*out_frame);
        *out_frame = nullptr;
        return suiteError_Fail;
    }
    if (fb::debug::enabled()) {
        // Memory: the readers' frames against the process as a whole.
        PROCESS_MEMORY_COUNTERS_EX memory{};
        K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory));
        fb::debug::line("GPU_FILE id=%" PRIu64 " taps=%zu ms=%.1f process_private_mb=%llu %s", render_id, taps.size(),
                        fb::debug::now_ms() - t_start, static_cast<unsigned long long>(memory.PrivateUsage >> 20),
                        reader.stats().c_str());
    }

    return suiteError_NoError;
}

}  // namespace
