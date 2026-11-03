// Adobe effect-API adapter.
//
// Premiere Pro (CPU render path, PF_Cmd_RENDER): reads the clip's media frame
// rate and the sequence rate from the host, asks the core for the kernel of
// the output frame, checks out every covered source frame and applies the
// weights. (The GPU filter, frame_blend_gpu.cpp, decodes the file instead.)
//
// After Effects (SmartFX, PF_Cmd_SMART_PRE_RENDER / PF_Cmd_SMART_RENDER): the
// layer's real source rate comes from its footage interpretation (or the
// precomp's frame rate); pre-render checks out every recorded frame the output
// frame covers, at the frame's middle, and render blends them in 8, 16 or 32
// bits per channel. After Effects hands effects any source time, so no file is
// decoded here.

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h"
#include "AE_GeneralPlug.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectPixelFormat.h"
#include "PrSDKPixelFormat.h"
#include "PrSDKAESupport.h"
#include "SPBasic.h"

#include "fb_debug.h"  // debug log (off unless FrameBlend-log.txt exists)
#include "fb_plan.h"
#include "frame_blend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int64_t kTicksPerSecond = 254016000000LL;

// No parameters: in Premiere the clip's Time Interpolation switches the blend
// (GPU filter, frame_blend_gpu.cpp). This CPU path cannot read that setting
// and always blends.
enum { PARAM_INPUT = 0, PARAM_COUNT };

enum class Format { Float32, Byte8, Unknown };

template <class Suite>
struct ScopedSuite {
    PF_InData* in_data;
    const char* name;
    int32_t version;
    Suite* suite = nullptr;
    ScopedSuite(PF_InData* in, const char* n, int32_t v) : in_data(in), name(n), version(v) {
        const void* p = nullptr;
        if (in_data && in_data->pica_basicP && in_data->pica_basicP->AcquireSuite(name, version, &p) == kSPNoError) {
            suite = static_cast<Suite*>(const_cast<void*>(p));
        }
    }
    ~ScopedSuite() {
        if (suite) {
            in_data->pica_basicP->ReleaseSuite(name, version);
        }
    }
};

Format format_of(PF_InData* in_data, const PF_EffectWorld* world) {
    PF_PixelFormat fmt = PF_PixelFormat_INVALID;
    if (in_data->appl_id == kAppID_Premiere) {
        ScopedSuite<PF_PixelFormatSuite1> s(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
        PrPixelFormat pr = PrPixelFormat_Invalid;
        if (!s.suite || s.suite->GetPixelFormat(const_cast<PF_EffectWorld*>(world), &pr) != PF_Err_NONE) {
            return Format::Unknown;
        }
        fmt = static_cast<PF_PixelFormat>(pr);
    } else {
        ScopedSuite<PF_WorldSuite2> s(in_data, kPFWorldSuite, kPFWorldSuiteVersion2);
        if (!s.suite || s.suite->PF_GetPixelFormat(world, &fmt) != PF_Err_NONE) {
            return Format::Unknown;
        }
    }
    switch (fmt) {
        case PF_PixelFormat_ARGB128:
        case PrPixelFormat_BGRA_4444_32f:
            return Format::Float32;
        case PF_PixelFormat_ARGB32:
        case PrPixelFormat_BGRA_4444_8u:
            return Format::Byte8;
        default:
            return Format::Unknown;
    }
}

struct Rates {
    FbRational source{0, 1};
    FbRational output{0, 1};
};

// Output: the sequence rate (time_scale / time_step). Source: the clip's media
// rate as Premiere interprets it; without it (After Effects) the rates are
// equal and the frame passes through.
Rates read_rates(PF_InData* in_data) {
    Rates rates;
    rates.output = fb::rational(in_data->time_scale, in_data->time_step);
    rates.source = rates.output;
    if (in_data->appl_id == kAppID_Premiere) {
        ScopedSuite<PF_UtilitySuite> util(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion);
        PrTime ticks_per_frame = 0;
        if (util.suite && util.suite->GetMediaFrameRate(in_data->effect_ref, &ticks_per_frame) == PF_Err_NONE &&
            ticks_per_frame > 0) {
            rates.source = fb::rational(kTicksPerSecond, ticks_per_frame);
        }
    }
    return rates;
}

struct Checkout {
    PF_InData* in_data = nullptr;
    PF_ParamDef param{};
    bool held = false;

    Checkout() = default;
    Checkout(const Checkout&) = delete;
    Checkout& operator=(const Checkout&) = delete;
    Checkout(Checkout&& o) noexcept : in_data(o.in_data), param(o.param), held(o.held) { o.held = false; }
    ~Checkout() {
        if (held) {
            PF_CHECKIN_PARAM(in_data, &param);
        }
    }
    const PF_LayerDef& layer() const { return param.u.ld; }
};

struct CheckoutTime {
    A_long value = 0;
    A_long step = 0;
    A_u_long scale = 0;
};

// Checks out the input layer at output position `num/den` output frames.
bool checkout_at(PF_InData* in_data, FbRational position, Checkout& out, CheckoutTime& used) {
    // Express the time exactly in a finer time scale when it fits in 32 bits.
    int64_t scale = static_cast<int64_t>(in_data->time_scale) * position.den;
    int64_t value = static_cast<int64_t>(in_data->time_step) * position.num;
    int64_t step = static_cast<int64_t>(in_data->time_step) * position.den;
    if (scale > INT32_MAX || value > INT32_MAX || value < INT32_MIN || step > INT32_MAX) {
        scale = in_data->time_scale;
        step = in_data->time_step;
        value = static_cast<int64_t>(std::llround(fb::to_double(position) * static_cast<double>(in_data->time_step)));
    }
    AEFX_CLR_STRUCT(out.param);
    out.in_data = in_data;
    used = {static_cast<A_long>(value), static_cast<A_long>(step), static_cast<A_u_long>(scale)};
    if (PF_CHECKOUT_PARAM(in_data, PARAM_INPUT, used.value, used.step, used.scale, &out.param) != PF_Err_NONE) {
        return false;
    }
    out.held = true;
    return out.param.u.ld.data != nullptr;
}

PF_Err fail(PF_OutData* out_data, const char* what, int32_t code) {
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "Frame Blend: %s (%s)", what,
                  fb_error_message(code));
    out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
    return PF_Err_NONE;  // the host shows return_msg
}

// ---- Debug log ----------------------------------------------------------------

std::atomic<uint64_t> g_render_id{0};

std::string host_timing_info(PF_InData* in_data) {
    char text[768] = {};
    if (in_data->appl_id != kAppID_Premiere) {
        std::snprintf(text, sizeof(text), "appl=%d", static_cast<int>(in_data->appl_id));
        return text;
    }
    ScopedSuite<PF_UtilitySuite> util(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion);
    if (!util.suite) {
        return "utility_suite=none";
    }
    PrTime media_rate = -1, original_rate = -1, sequence_time = -1;
    A_long clip_start = -1, clip_duration = -1, unscaled_start = -1, unscaled_duration = -1, item_start = -1;
    A_long media_frame = -1;
    PF_TimeDisplay display{};
    double speed = NAN;
    PF_ProgPtr ref = in_data->effect_ref;
    util.suite->GetMediaFrameRate(ref, &media_rate);
    util.suite->GetOriginalClipFrameRate(ref, &original_rate);
    util.suite->GetClipStart(ref, &clip_start);
    util.suite->GetClipDuration(ref, &clip_duration);
    util.suite->GetUnscaledClipStart(ref, &unscaled_start);
    util.suite->GetUnscaledClipDuration(ref, &unscaled_duration);
    util.suite->GetTrackItemStart(ref, &item_start);
    util.suite->GetMediaTimecode(ref, &media_frame, &display);
    util.suite->GetClipSpeed(ref, &speed);
    util.suite->GetSequenceTime(ref, &sequence_time);
    std::snprintf(text, sizeof(text),
                  "media_ticks_per_frame=%lld original_ticks_per_frame=%lld clip_start=%d clip_duration=%d "
                  "unscaled_clip_start=%d unscaled_clip_duration=%d track_item_start=%d media_timecode_frame=%d "
                  "clip_speed=%.6f sequence_time=%lld",
                  static_cast<long long>(media_rate), static_cast<long long>(original_rate), clip_start, clip_duration,
                  unscaled_start, unscaled_duration, item_start, media_frame, speed,
                  static_cast<long long>(sequence_time));
    return text;
}

struct Attempt {
    int64_t index = 0;
    CheckoutTime time;
    bool ok = false;
};

PF_Err render(PF_InData* in_data, PF_OutData* out_data, PF_LayerDef* output) {
    const double t_start = fb::debug::now_ms();
    const uint64_t render_id = ++g_render_id;
    if (!output || !output->data || in_data->time_scale == 0 || in_data->time_step <= 0) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    const Format format = format_of(in_data, output);
    if (format == Format::Unknown) {
        return fail(out_data, "unsupported pixel format", FB_ERR_BAD_ARGS);
    }
    Rates rates = read_rates(in_data);
    // Output frame index of the current time within the layer.
    const FbRational output_frame = fb::rational(in_data->current_time, in_data->time_step);
    FbRational center{};
    int32_t status = fb_center_for_output_frame(rates.source, rates.output, fb::rational(0, 1), output_frame, &center);
    if (status != FB_OK) {
        return fail(out_data, "cannot place the output frame", status);
    }

    if (fb::debug::enabled()) {
        fb::debug::line("CPU_RENDER id=%" PRIu64 " appl=%d reading host clip info", render_id,
                        static_cast<int>(in_data->appl_id));
        fb::debug::line("RENDER id=%" PRIu64 " host=premiere-cpu t_ms=%.3f current_time=%d time_step=%d time_scale=%u total_time=%d "
                        "src_fps=%s out_fps=%s center=%s size=%dx%d format=%s quality=%d %s",
                        render_id, t_start, in_data->current_time, in_data->time_step, in_data->time_scale, in_data->total_time,
                        fb::text(rates.source).c_str(), fb::text(rates.output).c_str(), fb::text(center).c_str(),
                        output->width, output->height, format == Format::Float32 ? "32f" : "8u",
                        static_cast<int>(in_data->quality), host_timing_info(in_data).c_str());
    }

    std::map<int64_t, Checkout> frames;
    std::vector<Attempt> attempts;
    auto fetch = [&](int64_t index) {
        FbRational position{};
        if (fb_output_position_of_source_frame(rates.source, rates.output, fb::rational(0, 1), index, &position) != FB_OK) {
            return false;
        }
        Checkout checkout;
        CheckoutTime used;
        bool ok = checkout_at(in_data, position, checkout, used);
        if (ok) {
            const PF_LayerDef& layer = checkout.layer();
            ok = layer.width == output->width && layer.height == output->height &&
                 format_of(in_data, &checkout.layer()) == format;
        }
        attempts.push_back({index, used, ok});
        if (ok) {
            frames.emplace(index, std::move(checkout));
        }
        return ok;
    };
    std::vector<FbTap> taps;
    status = fb::gather(rates.source, rates.output, center, INT64_MIN / 4, INT64_MAX / 4, fetch, taps);
    if (status != FB_OK) {
        fb::debug::line("ERROR id=%" PRIu64 " no source frame could be checked out", render_id);
        fb::debug::flush();
        return fail(out_data, "no source frame could be checked out", status);
    }
    const double t_fetched = fb::debug::now_ms();

    const int32_t row_elements = output->width * 4;
    std::vector<ptrdiff_t> strides;
    std::vector<float> weights;
    for (const FbTap& tap : taps) {
        strides.push_back(frames.at(tap.index).layer().rowbytes);
        weights.push_back(tap.weight);
    }
    if (format == Format::Float32) {
        std::vector<const float*> sources;
        for (const FbTap& tap : taps) {
            sources.push_back(reinterpret_cast<const float*>(frames.at(tap.index).layer().data));
        }
        status = fb_apply_f32(reinterpret_cast<float*>(output->data), output->rowbytes, sources.data(), strides.data(),
                              weights.data(), static_cast<int32_t>(taps.size()), row_elements, output->height);
    } else {
        std::vector<const uint8_t*> sources;
        for (const FbTap& tap : taps) {
            sources.push_back(reinterpret_cast<const uint8_t*>(frames.at(tap.index).layer().data));
        }
        status = fb_apply_u8(reinterpret_cast<uint8_t*>(output->data), output->rowbytes, sources.data(), strides.data(),
                             weights.data(), static_cast<int32_t>(taps.size()), row_elements, output->height);
    }
    const double t_done = fb::debug::now_ms();
    if (status != FB_OK) {
        return fail(out_data, "blend failed", status);
    }

    // ---- Debug log (after timing) ----
    if (!fb::debug::enabled()) {
        return PF_Err_NONE;
    }
    const fb::debug::Element element = format == Format::Float32 ? fb::debug::Element::F32 : fb::debug::Element::U8;
    for (const Attempt& attempt : attempts) {
        const FbTap* tap = nullptr;
        int k = -1;
        for (size_t i = 0; i < taps.size(); ++i) {
            if (taps[i].index == attempt.index) {
                tap = &taps[i];
                k = static_cast<int>(i);
            }
        }
        fb::debug::Probe probe;
        if (attempt.ok) {
            const PF_LayerDef& layer = frames.at(attempt.index).layer();
            probe = fb::debug::analyse(fb::debug::sample_host(layer.data, layer.rowbytes, layer.width, layer.height, element),
                                       layer.width, element);
        }
        fb::debug::line("TAP host=%s id=%" PRIu64 " k=%d/%zu src=%lld req_time=%d/%u step=%d weight=%s status=%s checksum=%016" PRIx64
                        " code_top=%d code_bottom=%d",
                        "premiere-cpu", render_id, k, taps.size(), static_cast<long long>(attempt.index), attempt.time.value,
                        attempt.time.scale, attempt.time.step,
                        tap ? (std::to_string(tap->weight_num) + "/" + std::to_string(tap->weight_den)).c_str() : "-",
                        attempt.ok ? "ok" : "fail", probe.checksum, probe.code_top, probe.code_bottom);
    }
    const fb::debug::Probe out_probe = fb::debug::analyse(
        fb::debug::sample_host(output->data, output->rowbytes, output->width, output->height, element), output->width,
        element);
    fb::debug::line("DONE host=%s id=%" PRIu64 " api=cpu taps=%zu fetch_ms=%.3f blend_ms=%.3f total_ms=%.3f out_checksum=%016" PRIx64,
                    "premiere-cpu", render_id, taps.size(), t_fetched - t_start, t_done - t_fetched, t_done - t_start, out_probe.checksum);
    fb::debug::flush();
    return PF_Err_NONE;
}

// ---- After Effects (SmartFX) ------------------------------------------------

// Suite versions present in every After Effects since 2020 (17.0).
using LayerSuite = AEGP_LayerSuite8;
using ItemSuite = AEGP_ItemSuite8;
using CompSuite = AEGP_CompSuite10;

struct LayerSource {
    FbRational fps{0, 1};   // rate of the layer's source (footage or precomp)
    int64_t frames = 0;     // source frames, 0 when unknown
};

// AEGP calls read the project; render threads (Multi-Frame Rendering) take
// turns.
std::mutex g_aegp_mutex;

// The layer's source rate: the footage's interpretation ("Assume this frame
// rate" when set, the file's rate otherwise) or the precomp's frame rate.
// False for sources without a rate (solids, stills, text, shape layers).
bool layer_source(PF_InData* in_data, LayerSource& out) {
    std::lock_guard<std::mutex> lock(g_aegp_mutex);
    ScopedSuite<AEGP_PFInterfaceSuite1> pf(in_data, kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1);
    ScopedSuite<LayerSuite> layers(in_data, kAEGPLayerSuite, kAEGPLayerSuiteVersion8);
    ScopedSuite<ItemSuite> items(in_data, kAEGPItemSuite, kAEGPItemSuiteVersion8);
    ScopedSuite<AEGP_FootageSuite5> footage(in_data, kAEGPFootageSuite, kAEGPFootageSuiteVersion5);
    ScopedSuite<CompSuite> comps(in_data, kAEGPCompSuite, kAEGPCompSuiteVersion10);
    if (!pf.suite || !layers.suite || !items.suite || !footage.suite || !comps.suite) {
        fb::debug::line("AE_SOURCE suites pf=%d layer=%d item=%d footage=%d comp=%d", pf.suite != nullptr,
                        layers.suite != nullptr, items.suite != nullptr, footage.suite != nullptr, comps.suite != nullptr);
        return false;
    }
    AEGP_LayerH layer = nullptr;
    AEGP_ItemH item = nullptr;
    AEGP_ItemType type = AEGP_ItemType_NONE;
    const A_Err e1 = pf.suite->AEGP_GetEffectLayer(in_data->effect_ref, &layer);
    const A_Err e2 = layer ? layers.suite->AEGP_GetLayerSourceItem(layer, &item) : -1;
    const A_Err e3 = item ? items.suite->AEGP_GetItemType(item, &type) : -1;
    if (e1 || e2 || e3 || !layer || !item) {
        fb::debug::line("AE_SOURCE effect_layer_err=%d layer=%p source_item_err=%d item=%p item_type_err=%d type=%d",
                        static_cast<int>(e1), static_cast<void*>(layer), static_cast<int>(e2), static_cast<void*>(item),
                        static_cast<int>(e3), static_cast<int>(type));
        return false;
    }
    A_FpLong fps = 0.0;
    if (type == AEGP_ItemType_FOOTAGE) {
        AEGP_FootageInterp interp{};
        if (footage.suite->AEGP_GetFootageInterpretation(item, FALSE, &interp)) {
            return false;
        }
        fps = interp.conform_fpsF > 0.0 ? interp.conform_fpsF : interp.native_fpsF;
    } else if (type == AEGP_ItemType_COMP) {
        AEGP_CompH comp = nullptr;
        if (comps.suite->AEGP_GetCompFromItem(item, &comp) || !comp || comps.suite->AEGP_GetCompFramerate(comp, &fps)) {
            return false;
        }
    }
    if (!(fps > 0.0) || fps > 1.0e6) {
        return false;
    }
    // After Effects keeps the rate it read from the file (a 600 fps MP4 can be
    // 600.024002) and places source frame i at i / fps: integer and NTSC rates
    // exactly, others to a thousandth of a frame per second.
    if (fb_rational_from_fps(fps, &out.fps) != FB_OK) {
        const int64_t milli = std::llround(fps * 1000.0);
        const int64_t g = std::gcd(milli, int64_t{1000});
        out.fps = fb::rational(milli / g, 1000 / g);
    }
    A_Time duration{};
    if (!items.suite->AEGP_GetItemDuration(item, &duration) && duration.scale > 0 && duration.value > 0) {
        out.frames = static_cast<int64_t>(std::floor(static_cast<double>(duration.value) / duration.scale * fps + 0.5));
    }
    return true;
}

// Checked-out source frames of one output frame, from pre-render to render.
struct AePlan {
    std::vector<float> weights;  // per checkout id
};

void delete_plan(void* data) {
    delete static_cast<AePlan*>(data);
}

// Time of source frame `index`: its middle, so the host's rounding to frames
// cannot land on a neighbour. In the effect's own time scale (in_data->
// time_scale, e.g. 30720 for a 30 fps comp) when a source frame spans at least
// 8 of its units; otherwise in 10 us units (one frame of 1200 fps is 833 us),
// which keep times within 32 bits for six hours of source.
bool source_frame_time(FbRational fps, int64_t index, A_u_long host_scale, A_long& value, A_long& step,
                       A_u_long& scale) {
    const double seconds_per_frame = static_cast<double>(fps.den) / static_cast<double>(fps.num);
    const double units = host_scale > 0 && seconds_per_frame * host_scale >= 8.0 ? static_cast<double>(host_scale)
                                                                                 : 100000.0;
    const double t = (static_cast<double>(index) + 0.5) * seconds_per_frame * units;
    if (!(t >= 0.0) || t > static_cast<double>(INT32_MAX)) {
        return false;
    }
    value = static_cast<A_long>(std::llround(t));
    step = std::max<A_long>(1, static_cast<A_long>(std::llround(seconds_per_frame * units)));
    scale = static_cast<A_u_long>(units);
    return true;
}

PF_LRect intersect(const PF_LRect& a, const PF_LRect& b) {
    PF_LRect r{std::max(a.left, b.left), std::max(a.top, b.top), std::min(a.right, b.right), std::min(a.bottom, b.bottom)};
    if (r.right <= r.left || r.bottom <= r.top) {
        r = PF_LRect{0, 0, 0, 0};
    }
    return r;
}

PF_LRect unite(const PF_LRect& a, const PF_LRect& b) {
    if (a.right <= a.left || a.bottom <= a.top) {
        return b;
    }
    if (b.right <= b.left || b.bottom <= b.top) {
        return a;
    }
    return PF_LRect{std::min(a.left, b.left), std::min(a.top, b.top), std::max(a.right, b.right), std::max(a.bottom, b.bottom)};
}

PF_Err smart_pre_render(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra) {
    // One output frame spans local_time_step of layer time (time stretch
    // included; negative for a reversed layer).
    A_long step = in_data->local_time_step ? in_data->local_time_step : in_data->time_step;
    step = step < 0 ? -step : step;
    std::vector<FbTap> taps;
    LayerSource source;
    if (step > 0 && in_data->time_scale > 0 && layer_source(in_data, source)) {
        const FbRational output = fb::rational(in_data->time_scale, step);
        FbRational center{};
        const int64_t last = source.frames > 0 ? source.frames - 1 : INT64_MAX / 4;
        if (fb_center_for_output_frame(source.fps, output, fb::rational(0, 1), fb::rational(in_data->current_time, step),
                                       &center) != FB_OK ||
            fb::kernel(source.fps, output, center, 0, last, taps) != FB_OK) {
            taps.clear();
        }
    }
    auto plan = std::make_unique<AePlan>();
    PF_LRect result{0, 0, 0, 0}, max_result{0, 0, 0, 0};
    bool first = true;
    auto checkout = [&](A_long id, A_long value, A_long time_step, A_u_long scale) {
        PF_CheckoutResult got{};
        PF_Err err = extra->cb->checkout_layer(in_data->effect_ref, PARAM_INPUT, id, &extra->input->output_request, value,
                                               time_step, scale, &got);
        if (!err) {
            result = first ? got.result_rect : intersect(result, got.result_rect);
            max_result = unite(max_result, got.max_result_rect);
            first = false;
        }
        return err;
    };
    A_long value = 0, time_step = 0;
    A_u_long scale = 0;
    bool timed = !taps.empty();
    for (const FbTap& tap : taps) {
        timed = timed && source_frame_time(source.fps, tap.index, in_data->time_scale, value, time_step, scale);
    }
    if (timed) {
        for (size_t k = 0; k < taps.size(); ++k) {
            source_frame_time(source.fps, taps[k].index, in_data->time_scale, value, time_step, scale);
            if (PF_Err err = checkout(static_cast<A_long>(k), value, time_step, scale)) {
                return err;
            }
            plan->weights.push_back(taps[k].weight);
        }
    } else {
        // No source rate (solid, still, text): the frame passes through.
        if (PF_Err err = checkout(0, in_data->current_time, in_data->time_step, in_data->time_scale)) {
            return err;
        }
        plan->weights.push_back(1.0f);
    }
    extra->output->result_rect = result;
    extra->output->max_result_rect = max_result;
    extra->output->solid = FALSE;
    extra->output->pre_render_data = plan.release();
    extra->output->delete_pre_render_data_func = delete_plan;
    return PF_Err_NONE;
}

PF_Err smart_render(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    const auto* plan = static_cast<const AePlan*>(extra->input->pre_render_data);
    PF_EffectWorld* output = nullptr;
    if (!plan || plan->weights.empty()) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    // Source frames first, then the output (the order of Adobe's samples).
    std::vector<PF_EffectWorld*> inputs(plan->weights.size(), nullptr);
    for (size_t k = 0; k < inputs.size(); ++k) {
        const PF_Err err = extra->cb->checkout_layer_pixels(in_data->effect_ref, static_cast<A_long>(k), &inputs[k]);
        if (err || !inputs[k] || !inputs[k]->data) {
            fb::debug::line("AE_RENDER k=%zu checkout_layer_pixels err=%d world=%p", k, static_cast<int>(err),
                            static_cast<void*>(inputs[k]));
            inputs[k] = nullptr;
        }
    }
    if (PF_Err err = extra->cb->checkout_output(in_data->effect_ref, &output)) {
        fb::debug::line("AE_RENDER checkout_output err=%d", static_cast<int>(err));
        return err;
    }
    if (!output || !output->data || output->width <= 0 || output->height <= 0) {
        fb::debug::line("AE_RENDER empty output %p", static_cast<void*>(output));
        return PF_Err_NONE;
    }
    ScopedSuite<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2);
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    if (!worlds.suite || worlds.suite->PF_GetPixelFormat(output, &format)) {
        fb::debug::line("AE_RENDER no pixel format suite=%p", static_cast<void*>(worlds.suite));
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    const int pixel_bytes = format == PF_PixelFormat_ARGB128 ? 16 : format == PF_PixelFormat_ARGB64 ? 8 : 4;

    // Every source frame covers the output rectangle (pre-render asked for the
    // intersection); origins place them in layer coordinates.
    std::vector<const uint8_t*> sources;
    std::vector<ptrdiff_t> strides;
    std::vector<float> weights;
    for (size_t k = 0; k < inputs.size(); ++k) {
        PF_EffectWorld* world = inputs[k];
        if (!world) {
            continue;
        }
        const A_long dx = output->origin_x - world->origin_x;
        const A_long dy = output->origin_y - world->origin_y;
        if (dx < 0 || dy < 0 || dx + output->width > world->width || dy + output->height > world->height) {
            continue;
        }
        sources.push_back(reinterpret_cast<const uint8_t*>(world->data) + static_cast<ptrdiff_t>(dy) * world->rowbytes +
                          static_cast<ptrdiff_t>(dx) * pixel_bytes);
        strides.push_back(world->rowbytes);
        weights.push_back(plan->weights[k]);
    }
    if (sources.empty()) {
        return fail(out_data, "no source frame could be checked out", FB_ERR_AVAILABILITY);
    }
    // Frames that could not be checked out (past the source's end) leave
    // their weight to the others.
    float total = 0.0f;
    for (float w : weights) {
        total += w;
    }
    if (total > 0.0f && std::fabs(total - 1.0f) > 1e-6f) {
        for (float& w : weights) {
            w /= total;
        }
    }
    const int32_t row_elements = output->width * 4;
    const int32_t count = static_cast<int32_t>(sources.size());
    int32_t status = FB_OK;
    if (format == PF_PixelFormat_ARGB128) {
        std::vector<const float*> planes;
        for (const uint8_t* p : sources) planes.push_back(reinterpret_cast<const float*>(p));
        status = fb_apply_f32(reinterpret_cast<float*>(output->data), output->rowbytes, planes.data(), strides.data(),
                              weights.data(), count, row_elements, output->height);
    } else if (format == PF_PixelFormat_ARGB64) {
        std::vector<const uint16_t*> planes;
        for (const uint8_t* p : sources) planes.push_back(reinterpret_cast<const uint16_t*>(p));
        status = fb_apply_u16(reinterpret_cast<uint16_t*>(output->data), output->rowbytes, planes.data(), strides.data(),
                              weights.data(), count, row_elements, output->height, PF_MAX_CHAN16);
    } else {
        status = fb_apply_u8(reinterpret_cast<uint8_t*>(output->data), output->rowbytes, sources.data(), strides.data(),
                             weights.data(), count, row_elements, output->height);
    }
    return status == FB_OK ? PF_Err_NONE : fail(out_data, "blend failed", status);
}

PF_Err global_setup(PF_InData* in_data, PF_OutData* out_data) {
    fb::debug::line("LOAD plugin=FrameBlend.aex appl=%d log=%s", static_cast<int>(in_data->appl_id), fb::debug::log_path());
    fb::debug::flush();
    // Must match the PiPL (FrameBlendPiPL.r). Premiere ignores the SmartFX
    // flags and renders through PF_Cmd_RENDER.
    static_assert(PF_VERSION(1, 4, 1, PF_Stage_DEVELOP, 0) == 657408, "PiPL AE_Effect_Version");
    static_assert((PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_PIX_INDEPENDENT |
                   PF_OutFlag_DEEP_COLOR_AWARE) == 33555462, "PiPL AE_Effect_Global_OutFlags");
    static_assert((PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                   PF_OutFlag2_SUPPORTS_THREADED_RENDERING) == 134222848, "PiPL AE_Effect_Global_OutFlags_2");
    out_data->my_version = PF_VERSION(1, 4, 1, PF_Stage_DEVELOP, 0);
    out_data->out_flags = PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_PIX_INDEPENDENT |
                          PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                           PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
    if (in_data->appl_id != kAppID_Premiere) {
        return PF_Err_NONE;
    }
    ScopedSuite<PF_PixelFormatSuite1> pixels(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
    if (!pixels.suite) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }
    PF_Err err = pixels.suite->ClearSupportedPixelFormats(in_data->effect_ref);
    if (!err) err = pixels.suite->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
    if (!err) err = pixels.suite->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
    // Ask for checked-out source frames in the output format (element-wise
    // blending needs one format). Optional: Premiere 2023 rejects it during
    // global setup, and render() refuses frames whose format differs anyway.
    PF_Err match = PF_Err_NONE;
    ScopedSuite<PF_UtilitySuite> util(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion);
    if (!err && util.suite) {
        try {
            match = util.suite->EffectWantsCheckedOutFramesToMatchRenderPixelFormat(in_data->effect_ref);
        } catch (...) {
            match = -1;
        }
    }
    fb::debug::line("GLOBAL_SETUP pixel_formats_err=%d utility_suite=%d match_render_format_err=%d", static_cast<int>(err),
                    util.suite ? 1 : 0, static_cast<int>(match));
    fb::debug::flush();
    return err;
}

PF_Err params_setup(PF_InData* in_data, PF_OutData* out_data) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    out_data->num_params = PARAM_COUNT;
    return PF_Err_NONE;
}

}  // namespace

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                       PF_LayerDef* output, void* extra) {
    try {
        switch (cmd) {
            case PF_Cmd_GLOBAL_SETUP:
                return global_setup(in_data, out_data);
            case PF_Cmd_PARAMS_SETUP:
                return params_setup(in_data, out_data);
            case PF_Cmd_RENDER:
                return render(in_data, out_data, output);
            case PF_Cmd_SMART_PRE_RENDER:
                return smart_pre_render(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
            case PF_Cmd_SMART_RENDER:
                return smart_render(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
            default:
                return PF_Err_NONE;
        }
    } catch (const std::bad_alloc&) {
        return PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}
