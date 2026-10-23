// Reads every frame of a recording directly from its file with FFmpeg,
// because Premiere only hands effects frames on the sequence's frame grid.
//
// With a CUDA context, frames are decoded by NVDEC into that context (8-bit
// NV12 decoder surfaces; nothing is copied to host memory). Otherwise, or when
// NVDEC cannot decode the stream, multi-threaded software decoding produces
// compact host NV12 (Y plane `width x height`, then the interleaved UV plane
// `width x height/2`). A worker thread decodes sequentially and reads ahead of
// the latest request; it seeks only for backward or distant jumps.
#pragma once

#include "frame_blend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <string>
#include <vector>

namespace fb::media {

struct Info {
    int width = 0;
    int height = 0;
    FbRational fps{0, 1};   // the stream's exact frame rate (e.g. 720/1)
    int64_t frames = 0;     // from the container
    bool full_range = true;
    bool bt601 = false;     // BT.709 otherwise
    bool nvdec = false;     // frames live on the device
    std::string codec;
};

// Allocator of the buffers software frames are decoded into: page-locked
// memory of a CUDA context when one is given (uploads from it are direct
// transfers that do not hold up the render), ordinary memory otherwise or
// when page-locked memory runs out. Each block remembers which it is.
struct HostMemory {
    void* cuda_context = nullptr;
};

void* host_allocate(const HostMemory* memory, size_t bytes);
void host_free(const HostMemory* memory, void* p);

template <class T>
struct HostAllocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    const HostMemory* memory = nullptr;

    HostAllocator() = default;
    explicit HostAllocator(const HostMemory* m) : memory(m) {}
    template <class U>
    HostAllocator(const HostAllocator<U>& other) : memory(other.memory) {}

    T* allocate(size_t n) { return static_cast<T*>(host_allocate(memory, n * sizeof(T))); }
    void deallocate(T* p, size_t) { host_free(memory, p); }

    template <class U>
    bool operator==(const HostAllocator<U>& other) const { return memory == other.memory; }
    template <class U>
    bool operator!=(const HostAllocator<U>& other) const { return memory != other.memory; }
};

using HostBuffer = std::vector<uint8_t, HostAllocator<uint8_t>>;

struct Frame {
    int64_t index = 0;
    int64_t timestamp_hns = 0;
    // Software decoding: compact host NV12.
    HostBuffer nv12;
    // NVDEC: device planes (CUdeviceptr) in the reader's CUDA context.
    unsigned long long luma = 0;
    unsigned long long chroma = 0;
    int64_t pitch = 0;
    std::shared_ptr<void> hold;  // keeps the decoder surface alive

    bool on_device() const { return luma != 0; }
};

class Reader {
public:
    // One reader per (file, CUDA context, host memory context), shared
    // between effect instances. `cuda_context` decodes with NVDEC into that
    // context; `pinned_context` decodes in software into page-locked memory
    // of that context. Null when the file cannot be opened; `error` says why.
    static std::shared_ptr<Reader> open(const std::wstring& path, void* cuda_context, void* pinned_context,
                                        std::string& error);
    virtual ~Reader() = default;

    virtual const Info& info() const = 0;

    // Blocks until frame `index` (0-based, presentation order) is decoded.
    // Null beyond the file's end or after a decoding error.
    virtual std::shared_ptr<const Frame> get(int64_t index) = 0;

    // Starts reading ahead from `index` before anything is requested (the
    // clip's in point when the effect is applied), so playback starts warm.
    virtual void prefetch(int64_t index) = 0;

    // Decoder statistics for the temporary logs.
    virtual std::string stats() const = 0;
};

// Loads the FFmpeg DLLs from the plugin's own folder (hosts do not search it).
bool load_ffmpeg(std::string& error);

// FFmpeg name of the file's video codec ("h264", "hevc", ...), empty when the
// file cannot be read.
std::string video_codec(const std::wstring& path);

}  // namespace fb::media
