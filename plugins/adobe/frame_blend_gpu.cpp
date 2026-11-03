// Premiere Pro GPU filter for the same effect (matched through the module's
// PiPL). The weights from the core are applied on Premiere's GPU device with
// CUDA or OpenCL. When this filter declines an instance, Premiere renders it
// through the effect API (frame_blend_ae.cpp).
//
// Premiere only hands effects frames on the sequence's own frame grid (host
// test, 2026-10-05: 25 requested source times returned 2 distinct frames), so
// the filter decodes the clip's file itself (fb_media_reader) and blends the
// recording's real frames. It falls back to Premiere's frames when the clip
// is retimed (speed, reverse, time remapping) or its file cannot be decoded.

#define NOMINMAX
#include <windows.h>

#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKGPUFilter.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPPix2Suite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKVideoSegmentProperties.h"
#include "PrSDKVideoSegmentSuite.h"
#include "SPBasic.h"

#include "fb_debug.h"  // debug log (off unless FrameBlend-log.txt exists)
#include "fb_gpu.h"
#include "fb_media_reader.h"
#include "fb_plan.h"
#include "frame_blend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kTicksPerSecond = 254016000000LL;
// Frames a reversed clip decodes ahead of its first displayed frame (the
// reader warms up forward from the index it is given).
constexpr int64_t kReverseWarmup = 240;
// MediaNode::StreamTimeInterpolationType values: 0 Frame Sampling, 1 Frame
// Blending, 2 Optical Flow (set through QE and read back, 2026-10-06).
constexpr int kInterpolationFrameBlending = 1;
// Video segment suite versions tried, newest first. Version 7 added
// AcquireOperatorOwnerNodeID; with an older suite (Premiere before 2023 may
// only have one) the clip owning the effect is found by walking the timeline's
// segments instead (find_owner). Every other function used is in version 4.
// Requesting the newest (9) would decline the GPU path in Premiere 2023.
constexpr int32_t kVideoSegmentSuiteVersions[] = {kPrSDKVideoSegmentSuiteVersion7, kPrSDKVideoSegmentSuiteVersion6,
                                                  kPrSDKVideoSegmentSuiteVersion5, kPrSDKVideoSegmentSuiteVersion4};

// One keyframe of a clip's time remapping. Premiere keeps time remapping as a
// curve from clip time (ticks from the clip's untrimmed start) to a position
// in the file (seconds), in ClipNode::TimeRemapping as "time,position,type,
// ...;" entries: the speed is constant between keyframes, except from a
// keyframe of type 7 to the following one of type 8, where it changes
// gradually (a ramp); other keyframes have type 6 (read from Premiere 2023 and
// the project file it saves, 2026-10-05).
struct RemapKey {
    double time = 0.0;      // clip time, seconds
    double position = 0.0;  // media time, seconds
    int type = 0;
};
constexpr int kRemapRampStart = 7;
constexpr int kRemapRampEnd = 8;

// Keyframes of ClipNode::TimeRemapping: its fields are separated by white
// space, and the keyframe list is the one with ';'. False without at least two
// keyframes (time remapping not in use) or when the list cannot be read.
bool parse_time_remapping(const std::string& value, std::vector<RemapKey>& keys) {
    keys.clear();
    std::istringstream fields(value);
    std::string field, list;
    while (fields >> field) {
        if (field.find(';') != std::string::npos) {
            list = field;
        }
    }
    std::istringstream entries(list);
    std::string entry;
    while (std::getline(entries, entry, ';')) {
        if (entry.empty()) {
            continue;
        }
        char* end = nullptr;
        RemapKey key;
        const long long ticks = std::strtoll(entry.c_str(), &end, 10);
        if (*end != ',') {
            return false;
        }
        key.position = std::strtod(end + 1, &end);
        if (*end != ',') {
            return false;
        }
        key.type = static_cast<int>(std::strtol(end + 1, &end, 10));
        key.time = static_cast<double>(ticks) / static_cast<double>(kTicksPerSecond);
        if (!keys.empty() && key.time <= keys.back().time) {
            return false;
        }
        keys.push_back(key);
    }
    return keys.size() >= 2;
}

struct Instance {
    SPBasicSuite* sp = nullptr;
    PrSDKGPUDeviceSuite* device = nullptr;
    PrSDKPPixSuite* ppix = nullptr;
    PrSDKPPix2Suite* ppix2 = nullptr;
    PrSDKVideoSegmentSuite* segments = nullptr;
    int32_t segments_version = 0;
    PrSDKMemoryManagerSuite* memory = nullptr;
    csSDK_uint32 device_index = 0;
    PrGPUDeviceInfo info{};
    PrTimelineID timeline = 0;
    csSDK_int32 node = 0;
    FbRational source{0, 1};

    // Reading the clip's file directly.
    std::string media_path;
    PrTime in_point = 0;
    PrTime stream_ticks_per_frame = 0;  // Premiere's reading of the stream rate
    double speed = 1.0;                 // constant speed (Speed/Duration), 0.5 = 50 %
    bool backwards = false;             // Reverse Speed
    std::vector<RemapKey> remap;        // time remapping curve (empty: constant speed)
    PrTime track_item_start = 0;        // the clip's start in the sequence
    double remap_offset = 0.0;          // clip time - (sequence time - start), found on the first render
    bool remap_offset_known = false;
    bool retimed = false;               // unreadable retiming (frame hold, unknown curve): Premiere's frames
    int interpolation = -1;             // the clip's Time Interpolation (MediaNode::StreamTimeInterpolationType), -1 unknown
    std::shared_ptr<fb::media::Reader> reader;
    std::string file_status = "not tried";

    // Decoded frames uploaded to the device (NV12), reused across renders.
    struct Slot {
        int64_t index = -1;
        void* memory = nullptr;
        void* uploaded = nullptr;  // CUDA event: the upload from `host` is complete
        uint64_t stamp = 0;
        std::shared_ptr<const fb::media::Frame> host;  // decoded frame, kept until its upload is done
    };
    std::mutex slot_mutex;
    std::vector<Slot> slots;
    uint64_t render_stamp = 0;

    ~Instance() {
        if (!sp) {
            return;
        }
        for (Slot& slot : slots) {
            fb::gpu::cuda_event_wait(info.outContextHandle, slot.uploaded);
            fb::gpu::cuda_event_destroy(info.outContextHandle, slot.uploaded);
            slot.host.reset();
            if (slot.memory && device) {
                device->FreeDeviceMemory(device_index, slot.memory);
            }
        }
        if (device) sp->ReleaseSuite(kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion);
        if (ppix) sp->ReleaseSuite(kPrSDKPPixSuite, kPrSDKPPixSuiteVersion);
        if (ppix2) sp->ReleaseSuite(kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion);
        if (segments) sp->ReleaseSuite(kPrSDKVideoSegmentSuite, segments_version);
        if (memory) sp->ReleaseSuite(kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion);
    }
};

// The clip's Time Interpolation (right-click > Time Interpolation) switches
// the blend: Frame Blending blends; with Frame Sampling or Optical Flow the
// effect passes Premiere's own frame through. Premiere rebuilds the effect
// instances when the setting changes. A Premiere that does not report the
// setting blends every clip with the effect.
bool blend_enabled(const Instance& inst) {
    return inst.interpolation == kInterpolationFrameBlending || inst.interpolation < 0;
}

template <class Suite>
bool acquire(SPBasicSuite* sp, const char* name, int32_t version, Suite*& out) {
    const void* p = nullptr;
    if (sp->AcquireSuite(name, version, &p) != kSPNoError || !p) {
        return false;
    }
    out = static_cast<Suite*>(const_cast<void*>(p));
    return true;
}

bool node_property(Instance& inst, csSDK_int32 node, const char* key, std::string& value) {
    PrMemoryPtr buffer = nullptr;
    if (inst.segments->GetNodeProperty(node, key, &buffer) != suiteError_NoError || !buffer) {
        return false;
    }
    value = reinterpret_cast<const char*>(buffer);
    inst.memory->PrDisposePtr(buffer);
    return true;
}

std::string node_type(Instance& inst, csSDK_int32 node) {
    char type[kMaxNodeTypeStringSize] = {};
    prPluginID hash{};
    csSDK_int32 flags = 0;
    inst.segments->GetNodeInfo(node, type, &hash, &flags);
    return type;
}

// Debug log: every property of a node as key=value pairs.
thread_local std::string* g_property_text = nullptr;
prSuiteError collect_property(csSDK_int32, const char* key, const prUTF8Char* value) {
    if (g_property_text) {
        *g_property_text += std::string(" ") + key + "=" + reinterpret_cast<const char*>(value);
    }
    return suiteError_NoError;
}
std::string node_properties(Instance& inst, csSDK_int32 node) {
    std::string text;
    g_property_text = &text;
    inst.segments->IterateNodeProperties(node, collect_property, 0);
    g_property_text = nullptr;
    return text;
}

// The node (acquired) among `node` and its inputs that has an operator with
// this RuntimeInstanceID, or 0.
csSDK_int32 node_with_operator(Instance& inst, csSDK_int32 node, const std::string& instance_id, int depth) {
    csSDK_int32 operators = 0;
    inst.segments->GetNodeOperatorCount(node, &operators);
    for (csSDK_int32 k = 0; k < operators; ++k) {
        csSDK_int32 op = 0;
        if (inst.segments->AcquireOperatorNodeID(node, k, &op) != suiteError_NoError) {
            continue;
        }
        std::string value;
        const bool match = node_property(inst, op, kVideoSegmentProperty_Effect_RuntimeInstanceID, value) &&
                           value == instance_id;
        inst.segments->ReleaseVideoNodeID(op);
        if (match) {
            return node;
        }
    }
    csSDK_int32 inputs = 0;
    if (depth >= 8 || inst.segments->GetNodeInputCount(node, &inputs) != suiteError_NoError) {
        return 0;
    }
    for (csSDK_int32 i = 0; i < inputs; ++i) {
        csSDK_int32 input = 0;
        PrTime offset = 0;
        if (inst.segments->AcquireInputNodeID(node, i, &offset, &input) != suiteError_NoError) {
            continue;
        }
        const csSDK_int32 found = node_with_operator(inst, input, instance_id, depth + 1);
        if (found != input) {
            inst.segments->ReleaseVideoNodeID(input);
        }
        if (found) {
            return found;
        }
    }
    return 0;
}

// Test hook: FRAMEBLEND_SEGMENT_SCAN=1 takes the older-Premiere
// path (find_owner's segment walk) on a current Premiere.
bool force_segment_scan() {
    char value[8] = {};
    return GetEnvironmentVariableA("FRAMEBLEND_SEGMENT_SCAN", value, sizeof(value)) > 0 && value[0] == '1';
}

// The clip node carrying this effect (acquired). With suite version 7 or
// later the host says; before, the timeline's segments are walked for the
// node whose operators include this effect instance (EffectNode::
// RuntimeInstanceID, unique per instance).
bool find_owner(Instance& inst, csSDK_int32& owner) {
    if (inst.segments_version >= kPrSDKVideoSegmentSuiteVersion7 && !force_segment_scan()) {
        return inst.segments->AcquireOperatorOwnerNodeID(inst.node, &owner) == suiteError_NoError && owner;
    }
    std::string instance_id;
    if (!node_property(inst, inst.node, kVideoSegmentProperty_Effect_RuntimeInstanceID, instance_id) ||
        instance_id.empty()) {
        return false;
    }
    csSDK_int32 segments_id = 0;
    if (inst.segments->AcquireVideoSegmentsID(inst.timeline, &segments_id) != suiteError_NoError) {
        return false;
    }
    csSDK_int32 count = 0;
    inst.segments->GetSegmentCount(segments_id, &count);
    owner = 0;
    for (csSDK_int32 s = 0; s < count && !owner; ++s) {
        PrTime start = 0, end = 0, offset = 0;
        prPluginID hash{};
        csSDK_int32 node = 0;
        if (inst.segments->GetSegmentInfo(segments_id, s, &start, &end, &offset, &hash) != suiteError_NoError ||
            inst.segments->AcquireNodeID(segments_id, &hash, &node) != suiteError_NoError) {
            continue;
        }
        owner = node_with_operator(inst, node, instance_id, 0);
        if (owner != node) {
            inst.segments->ReleaseVideoNodeID(node);
        }
    }
    inst.segments->ReleaseVideoSegmentsID(segments_id);
    fb::debug::line("GPU_OWNER_SCAN effect=%d instance=%s segments=%d owner=%d suite=%d", inst.node,
                    instance_id.c_str(), count, owner, inst.segments_version);
    return owner != 0;
}

// The clip's media rate, file and in point: the effect node's owner is the
// clip node; its input chain leads to the media node carrying StreamFrameRate
// (ticks per frame), MediaInstanceString and InPointMediaTimeAsTicks.
bool read_source_rate(Instance& inst) {
    csSDK_int32 owner = 0;
    if (!find_owner(inst, owner)) {
        fb::debug::line("GPU_NODE effect=%d owner=none", inst.node);
        return false;
    }
    if (fb::debug::enabled()) {
        fb::debug::line("GPU_NODE effect=%d type=%s props:%s", inst.node, node_type(inst, inst.node).c_str(),
                        node_properties(inst, inst.node).c_str());
    }
    csSDK_int32 current = owner;
    bool found = false;
    for (int depth = 0; depth < 4 && current && !found; ++depth) {
        if (fb::debug::enabled()) {
            fb::debug::line("GPU_NODE depth=%d id=%d type=%s props:%s", depth, current, node_type(inst, current).c_str(),
                            node_properties(inst, current).c_str());
        }
        std::string value;
        if ((node_property(inst, current, kVideoSegmentProperty_Clip_ClipSpeed, value) ||
             node_property(inst, current, kVideoSegmentProperty_Media_ClipSpeed, value)) &&
            std::strtod(value.c_str(), nullptr) != 0.0) {
            inst.speed = std::fabs(std::strtod(value.c_str(), nullptr));
        }
        if ((node_property(inst, current, kVideoSegmentProperty_Clip_ClipBackwards, value) ||
             node_property(inst, current, kVideoSegmentProperty_Media_ClipBackwards, value)) &&
            value != "0" && value != "false") {
            inst.backwards = true;
        }
        if (node_property(inst, current, kVideoSegmentProperty_Clip_TimeRemapping, value) &&
            value.find(';') != std::string::npos && !parse_time_remapping(value, inst.remap)) {
            inst.retimed = true;
        }
        if (node_property(inst, current, kVideoSegmentProperty_Clip_FrameHoldAtTime, value) && !value.empty()) {
            inst.retimed = true;
        }
        if (node_property(inst, current, kVideoSegmentProperty_Clip_TrackItemStartAsTicks, value)) {
            inst.track_item_start = std::strtoll(value.c_str(), nullptr, 10);
        }
        if (node_property(inst, current, kVideoSegmentProperty_Media_InstanceString, value)) {
            inst.media_path = value;
        }
        if (node_property(inst, current, kVideoSegmentProperty_Media_StreamTimeInterpolationType, value)) {
            inst.interpolation = static_cast<int>(std::strtol(value.c_str(), nullptr, 10));
        }
        if (node_property(inst, current, kVideoSegmentProperty_Media_InPointMediaTimeAsTicks, value)) {
            inst.in_point = std::strtoll(value.c_str(), nullptr, 10);
        }
        std::string ticks;
        if (node_property(inst, current, kVideoSegmentProperty_Media_StreamFrameRate, ticks)) {
            const long long ticks_per_frame = std::strtoll(ticks.c_str(), nullptr, 10);
            // While a project opens, Premiere first builds instances whose
            // media is still loading, with the rate given as INT64_MAX ticks.
            if (ticks_per_frame > 0 && ticks_per_frame < kTicksPerSecond) {
                inst.stream_ticks_per_frame = ticks_per_frame;
                inst.source = fb::rational(kTicksPerSecond, ticks_per_frame);
                found = true;
            }
        }
        csSDK_int32 input = 0;
        PrTime offset = 0;
        const bool has_input = !found && inst.segments->AcquireInputNodeID(current, 0, &offset, &input) == suiteError_NoError;
        inst.segments->ReleaseVideoNodeID(current);
        current = has_input ? input : 0;
    }
    if (current) {
        inst.segments->ReleaseVideoNodeID(current);
    }
    return found;
}

std::wstring widen(const std::string& utf8) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    std::wstring wide(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) {
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), n);
    }
    return wide;
}

void open_file(Instance& inst) {
    if (inst.retimed) {
        inst.file_status = "clip has a frame hold or an unreadable time remapping; using Premiere's frames";
        return;
    }
    if (inst.media_path.empty()) {
        inst.file_status = "no media path; using Premiere's frames";
        return;
    }
    std::string error;
    // H.264: software decoding. Premiere decodes the same clip on the GPU's
    // video decoder for its own frames, and sharing NVDEC halved its throughput
    // (host test 2026-10-05: about 300 frames/s with NVDEC busy, 780 on the CPU);
    // the decoder also refuses level 6.1 (1130 fps 720p recordings). Decoded
    // into page-locked memory of Premiere's CUDA context, so uploads to the
    // device are enqueued without copying through the driver.
    // Other codecs (HEVC, AV1): NVDEC. Software HEVC is far too slow for high
    // rates (1200 fps 720p: 620 frames/s on the CPU, 2577 on an RTX 2060's
    // decoder, 2026-10-22).
    const bool cuda = inst.info.outDeviceFramework == PrGPUDeviceFramework_CUDA;
    const std::wstring path = widen(inst.media_path);
    const bool nvdec = cuda && fb::media::video_codec(path) != "h264";
    inst.reader = fb::media::Reader::open(path, nvdec ? inst.info.outContextHandle : nullptr,
                                          cuda && !nvdec ? inst.info.outContextHandle : nullptr, error);
    if (!inst.reader) {
        inst.file_status = "cannot decode file (" + error + "); using Premiere's frames";
        return;
    }
    const fb::media::Info& info = inst.reader->info();
    // Decode from where the clip starts right away. A reversed clip starts at
    // the end of its range, and Premiere gives its in point from the file's end.
    FbRational in_frames{};
    if (fb_center_for_output_frame(info.fps, fb::rational(kTicksPerSecond, 1), fb::rational(0, 1),
                                   fb::rational(inst.in_point, kTicksPerSecond), &in_frames) == FB_OK) {
        const int64_t in_index = in_frames.num / in_frames.den;
        inst.reader->prefetch(inst.backwards ? std::max<int64_t>(0, info.frames - in_index - kReverseWarmup) : in_index);
    }
    inst.file_status = "decoding " + std::to_string(info.width) + "x" + std::to_string(info.height) + " fps=" +
                       fb::text(info.fps) + " frames=" + std::to_string(info.frames) +
                       " speed=" + std::to_string(inst.speed) + (inst.backwards ? " reversed" : "") +
                       (info.full_range ? " full" : " limited") + (info.bt601 ? " bt601" : " bt709") + " " +
                       info.codec + (info.nvdec ? " nvdec" : " software");
}

prSuiteError create_instance(PrGPUFilterInstance* data) {
    auto* inst = new Instance();
    inst->sp = data->piSuites->utilFuncs->getSPBasicSuite();
    inst->device_index = data->inDeviceIndex;
    inst->timeline = data->inTimelineID;
    inst->node = data->inNodeID;
    if (inst->sp) {
        for (int32_t version : kVideoSegmentSuiteVersions) {
            if (acquire(inst->sp, kPrSDKVideoSegmentSuite, version, inst->segments)) {
                inst->segments_version = version;
                break;
            }
        }
    }
    const bool suites = inst->sp && inst->segments &&
                        acquire(inst->sp, kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion, inst->device) &&
                        acquire(inst->sp, kPrSDKPPixSuite, kPrSDKPPixSuiteVersion, inst->ppix) &&
                        acquire(inst->sp, kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion, inst->ppix2) &&
                        acquire(inst->sp, kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion, inst->memory);
    bool ok = suites && inst->device->GetDeviceInfo(kPrSDKGPUDeviceSuiteVersion, inst->device_index, &inst->info) ==
                            suiteError_NoError;
    const PrGPUDeviceFramework framework = inst->info.outDeviceFramework;
    ok = ok && (framework == PrGPUDeviceFramework_CUDA || framework == PrGPUDeviceFramework_OpenCL);
    ok = ok && read_source_rate(*inst);
    if (ok && blend_enabled(*inst)) {
        open_file(*inst);
    } else if (ok) {
        inst->file_status = "Time Interpolation is not Frame Blending; passing Premiere's frames through";
    }
    fb::debug::line("GPU_INSTANCE node=%d device=%u framework=%d segment_suite=%d interpolation=%d accepted=%d src_fps=%s file=%s in_point=%lld "
                    "retimed=%d remap_keys=%zu file_status=%s inst=%p",
                    inst->node, inst->device_index, static_cast<int>(framework), inst->segments_version, inst->interpolation, ok ? 1 : 0,
                    fb::text(inst->source).c_str(), inst->media_path.c_str(), static_cast<long long>(inst->in_point),
                    inst->retimed ? 1 : 0, inst->remap.size(), inst->file_status.c_str(), static_cast<void*>(inst));
    fb::debug::flush();
    if (!ok) {
        delete inst;  // Premiere renders this instance through the effect API instead.
        return suiteError_Fail;
    }
    data->ioPrivatePluginData = inst;
    data->outIsRealtime = kPrTrue;
    return suiteError_NoError;
}

prSuiteError dispose_instance(PrGPUFilterInstance* data) {
    fb::debug::line("GPU_DISPOSE inst=%p t_ms=%.3f", data->ioPrivatePluginData, fb::debug::now_ms());
    delete static_cast<Instance*>(data->ioPrivatePluginData);
    data->ioPrivatePluginData = nullptr;
    return suiteError_NoError;
}

// Output rate from the render's frame duration; centre from the clip time.
struct Plan {
    FbRational output{0, 1};
    FbRational center{0, 1};
    std::vector<FbTap> taps;
};

bool plan(const Instance& inst, const PrGPUFilterRenderParams* params, Plan& out) {
    if (params->inRenderTicksPerFrame <= 0) {
        return false;
    }
    out.output = fb::rational(kTicksPerSecond, params->inRenderTicksPerFrame);
    const FbRational frame = fb::rational(params->inClipTime, params->inRenderTicksPerFrame);
    return fb_center_for_output_frame(inst.source, out.output, fb::rational(0, 1), frame, &out.center) == FB_OK &&
           fb::kernel(inst.source, out.output, out.center, INT64_MIN / 4, INT64_MAX / 4, out.taps) == FB_OK;
}

// Sequence time at which source frame `index` is shown, relative to this render.
bool sequence_time_of(const Instance& inst, const Plan& p, const PrGPUFilterRenderParams* params, int64_t index,
                      PrTime& out) {
    FbRational position{};
    if (fb_output_position_of_source_frame(inst.source, p.output, fb::rational(0, 1), index, &position) != FB_OK) {
        return false;
    }
    const double clip_ticks = static_cast<double>(position.num) / static_cast<double>(position.den) *
                              static_cast<double>(params->inRenderTicksPerFrame);
    out = params->inSequenceTime + static_cast<PrTime>(std::llround(clip_ticks)) - params->inClipTime;
    return true;
}


// Blending off: a copy of the current frame (one tap of weight 1).
prSuiteError render_passthrough(Instance& inst, const PPixHand* frames, PPixHand* out_frame) {
    PrPixelFormat format = PrPixelFormat_Invalid;
    prRect bounds{};
    csSDK_uint32 par_num = 1, par_den = 1;
    prFieldType field = prFieldsNone;
    inst.ppix->GetPixelFormat(frames[0], &format);
    inst.ppix->GetBounds(frames[0], &bounds);
    inst.ppix->GetPixelAspectRatio(frames[0], &par_num, &par_den);
    inst.ppix2->GetFieldOrder(frames[0], &field);
    const int width = bounds.right - bounds.left;
    const int height = bounds.bottom - bounds.top;
    if (inst.device->CreateGPUPPix(inst.device_index, format, width, height, par_num, par_den, field, out_frame) !=
            suiteError_NoError ||
        !*out_frame) {
        return suiteError_Fail;
    }
    auto image = [&](PPixHand frame) {
        void* data = nullptr;
        csSDK_int32 pitch = 0;
        inst.device->GetGPUPPixData(frame, &data);
        inst.ppix->GetRowBytes(frame, &pitch);
        return fb::gpu::Image{data, 0, pitch};
    };
    const fb::gpu::Tap tap{image(frames[0]), 1.0f};
    const fb::gpu::Element element =
        format == PrPixelFormat_GPU_BGRA_4444_32f ? fb::gpu::Element::F32 : fb::gpu::Element::F16;
    std::string error;
    const bool cuda = inst.info.outDeviceFramework == PrGPUDeviceFramework_CUDA;
    const bool ok = cuda ? fb::gpu::cuda_blend(inst.info.outContextHandle, inst.info.outCommandQueueHandle,
                                               image(*out_frame), &tap, 1, width * 4, height, element, error)
                         : fb::gpu::opencl_blend(inst.info.outCommandQueueHandle, image(*out_frame), &tap, 1, width * 4,
                                                 height, element, error);
    if (!ok) {
        inst.ppix->Dispose(*out_frame);
        *out_frame = nullptr;
        return suiteError_Fail;
    }
    return suiteError_NoError;
}

prSuiteError frame_dependencies(PrGPUFilterInstance* data, const PrGPUFilterRenderParams* params, csSDK_int32* query,
                                PrGPUFilterFrameDependency* out) {
    const auto* inst = static_cast<const Instance*>(data->ioPrivatePluginData);
    if (inst->reader || !blend_enabled(*inst)) {
        return suiteError_NotImplemented;  // frames come from the file; only the current frame is needed
    }
    Plan p;
    if (!plan(*inst, params, p) || *query < 0 || static_cast<size_t>(*query) >= p.taps.size()) {
        return suiteError_NotImplemented;
    }
    const FbTap& tap = p.taps[static_cast<size_t>(*query)];
    out->outDependencyType = PrGPUDependency_InputFrame;
    out->outTrackID = 0;
    if (!sequence_time_of(*inst, p, params, tap.index, out->outSequenceTime)) {
        return suiteError_Fail;
    }
    fb::debug::line("GPU_DEPENDENCY clip_time=%lld seq_time=%lld query=%d src=%lld dep_seq_time=%lld",
                    static_cast<long long>(params->inClipTime), static_cast<long long>(params->inSequenceTime), *query,
                    static_cast<long long>(tap.index), static_cast<long long>(out->outSequenceTime));
    if (static_cast<size_t>(*query) + 1 < p.taps.size()) {
        ++*query;
    }
    return suiteError_NoError;
}

std::atomic<uint64_t> g_render_id{0};

prSuiteError render_file(Instance& inst, const PrGPUFilterRenderParams* params, const PPixHand* frames,
                         csSDK_size_t frame_count, PPixHand* out_frame, uint64_t render_id, double t_start);

prSuiteError render(PrGPUFilterInstance* data, const PrGPUFilterRenderParams* params, const PPixHand* frames,
                    csSDK_size_t frame_count, PPixHand* out_frame) {
    const double t_start = fb::debug::now_ms();
    const uint64_t render_id = ++g_render_id;
    auto* inst = static_cast<Instance*>(data->ioPrivatePluginData);
    if (fb::debug::enabled()) {
        // Which path renders this frame: "file" blends the recording, "pass"
        // copies Premiere's frame (Time Interpolation is not Frame Blending),
        // "host" blends the frames Premiere hands over.
        fb::debug::line("GPU_RENDER id=%" PRIu64 " path=%s quality=%d downsample=%.2f,%.2f size=%ux%u clip_time=%lld "
                        "frames=%zu",
                        render_id, frame_count >= 1 && frames[0] && !blend_enabled(*inst) ? "pass"
                                   : inst->reader                                        ? "file"
                                                                                         : "host",
                        static_cast<int>(params->inQuality), params->inDownsampleFactorX, params->inDownsampleFactorY,
                        params->inRenderWidth, params->inRenderHeight, static_cast<long long>(params->inClipTime),
                        static_cast<size_t>(frame_count));
    }
    if (frame_count >= 1 && frames[0] && !blend_enabled(*inst)) {
        return render_passthrough(*inst, frames, out_frame);
    }
    if (inst->reader) {
        return render_file(*inst, params, frames, frame_count, out_frame, render_id, t_start);
    }
    Plan p;
    if (!plan(*inst, params, p) || frame_count != p.taps.size() + 1 || !frames[0]) {
        fb::debug::line("GPU_ERROR id=%" PRIu64 " clip_time=%lld frames=%zu expected=%zu", render_id,
                        static_cast<long long>(params->inClipTime), static_cast<size_t>(frame_count), p.taps.size() + 1);
        fb::debug::flush();
        return suiteError_Fail;
    }
    // inFrames[1 + k] holds original tap k; a missing frame marks the source edge.
    const std::vector<FbTap> requested = p.taps;
    auto frame_of = [&](int64_t index) -> PPixHand {
        for (size_t k = 0; k < requested.size(); ++k) {
            if (requested[k].index == index) {
                return frames[1 + k];
            }
        }
        return nullptr;
    };
    std::vector<FbTap> taps;
    if (fb::gather(inst->source, p.output, p.center, INT64_MIN / 4, INT64_MAX / 4,
                   [&](int64_t index) { return frame_of(index) != nullptr; }, taps) != FB_OK) {
        return suiteError_Fail;
    }

    PrPixelFormat format = PrPixelFormat_Invalid;
    prRect bounds{};
    csSDK_int32 row_bytes = 0;
    csSDK_uint32 par_num = 1;
    csSDK_uint32 par_den = 1;
    prFieldType field = prFieldsNone;
    inst->ppix->GetPixelFormat(frames[0], &format);
    inst->ppix->GetBounds(frames[0], &bounds);
    inst->ppix->GetPixelAspectRatio(frames[0], &par_num, &par_den);
    inst->ppix2->GetFieldOrder(frames[0], &field);
    const int width = bounds.right - bounds.left;
    const int height = bounds.bottom - bounds.top;
    if ((format != PrPixelFormat_GPU_BGRA_4444_32f && format != PrPixelFormat_GPU_BGRA_4444_16f) || width <= 0 ||
        height <= 0) {
        fb::debug::line("GPU_ERROR id=%" PRIu64 " unsupported frame format=%u size=%dx%d", render_id,
                        static_cast<unsigned>(format), width, height);
        return suiteError_Fail;
    }
    // Every blended frame must share frame 0's format and size.
    for (const FbTap& tap : taps) {
        PrPixelFormat tap_format = PrPixelFormat_Invalid;
        prRect tap_bounds{};
        inst->ppix->GetPixelFormat(frame_of(tap.index), &tap_format);
        inst->ppix->GetBounds(frame_of(tap.index), &tap_bounds);
        if (tap_format != format || tap_bounds.right - tap_bounds.left != width ||
            tap_bounds.bottom - tap_bounds.top != height) {
            fb::debug::line("GPU_ERROR id=%" PRIu64 " source frame %lld differs in format or size", render_id,
                            static_cast<long long>(tap.index));
            return suiteError_Fail;
        }
    }
    const fb::gpu::Element element =
        format == PrPixelFormat_GPU_BGRA_4444_32f ? fb::gpu::Element::F32 : fb::gpu::Element::F16;
    if (inst->device->CreateGPUPPix(inst->device_index, format, width, height, par_num, par_den, field, out_frame) !=
            suiteError_NoError ||
        !*out_frame) {
        return suiteError_Fail;
    }
    auto device_image = [&](PPixHand frame) {
        void* data_ptr = nullptr;
        csSDK_int32 pitch = 0;
        inst->device->GetGPUPPixData(frame, &data_ptr);
        inst->ppix->GetRowBytes(frame, &pitch);
        return fb::gpu::Image{data_ptr, 0, pitch};
    };
    std::vector<fb::gpu::Tap> gpu_taps;
    for (const FbTap& tap : taps) {
        gpu_taps.push_back({device_image(frame_of(tap.index)), tap.weight});
    }
    const fb::gpu::Image output = device_image(*out_frame);
    const bool cuda = inst->info.outDeviceFramework == PrGPUDeviceFramework_CUDA;
    void* context = inst->info.outContextHandle;
    void* queue = inst->info.outCommandQueueHandle;
    std::string error;
    bool ok = cuda ? fb::gpu::cuda_blend(context, queue, output, gpu_taps.data(), static_cast<int>(gpu_taps.size()),
                                         width * 4, height, element, error)
                   : fb::gpu::opencl_blend(queue, output, gpu_taps.data(), static_cast<int>(gpu_taps.size()), width * 4,
                                           height, element, error);
    if (!ok) {
        fb::debug::line("GPU_ERROR id=%" PRIu64 " %s", render_id, error.c_str());
        fb::debug::flush();
        inst->ppix->Dispose(*out_frame);
        *out_frame = nullptr;
        return suiteError_Fail;
    }

    return suiteError_NoError;
}

prSuiteError precompute(PrGPUFilterInstance*, const PrGPUFilterRenderParams*, csSDK_int32, PPixHand) {
    return suiteError_NotImplemented;
}

}  // namespace

extern "C" __declspec(dllexport) prSuiteError xGPUFilterEntry(csSDK_uint32 host_version, csSDK_int32* index,
                                                               prBool startup, piSuitesPtr suites, PrGPUFilter* filter,
                                                               PrGPUFilterInfo* info) {
    // Load FFmpeg in the background at startup, so applying the effect does
    // not wait for ~117 MB of DLLs to be read.
    static std::once_flag preload;
    if (!startup) {
        return suiteError_NoError;
    }
    std::call_once(preload, [] {
        std::thread([] {
            std::string error;
            fb::media::load_ffmpeg(error);
        }).detach();
    });
    if (*index != 0) {
        return suiteError_InvalidParms;
    }
    filter->CreateInstance = create_instance;
    filter->DisposeInstance = dispose_instance;
    filter->GetFrameDependencies = frame_dependencies;
    filter->Precompute = precompute;
    filter->Render = render;
    info->outInterfaceVersion = PrSDKGPUFilterInterfaceVersion;
    info->outMatchName = PrSDKString();  // the module's PiPL match name
    fb::debug::line("GPU_ENTRY host_interface=%u", host_version);
    fb::debug::flush();
    (void)suites;
    return suiteError_NoError;
}

#include "frame_blend_gpu_file.inl"
