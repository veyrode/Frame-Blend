// GPU application of core weights on the host's own CUDA stream or OpenCL
// queue. nvcuda.dll and OpenCL.dll are loaded on first use, so the plugins
// load on machines that have only one (or neither) of them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fb::gpu {

enum class Element { F32, F16 };

inline size_t element_size(Element e) {
    return e == Element::F32 ? 4 : 2;
}

// A device image: `base` is a CUdeviceptr (as a pointer) for CUDA or a cl_mem
// for OpenCL; `offset` is the byte offset of the first element used.
struct Image {
    void* base = nullptr;
    int64_t offset = 0;
    int64_t pitch = 0;
    // NV12 only: separate UV plane (CUDA decoder surfaces). Null: the UV rows
    // follow the luma rows in the same buffer.
    void* chroma = nullptr;
};

struct Tap {
    Image image;
    float weight = 0.0f;
};

constexpr int kMaxCudaTaps = 128;

// Launches dst = sum weight * src on `stream`. `context` is pushed around the
// launch when non-null; otherwise the thread's current context is used.
// Asynchronous: returns once enqueued.
bool cuda_blend(void* context, void* stream, const Image& dst, const Tap* taps, int count, int row_elements, int rows,
                Element element, std::string& error);

// Same on an OpenCL command queue with cl_mem buffers.
bool opencl_blend(void* queue, const Image& dst, const Tap* taps, int count, int row_elements, int rows,
                  Element element, std::string& error);

// YUV -> RGB coefficients for the NV12 kernels (layout mirrors FbYuv in fb_blend.cu).
struct Yuv {
    float y_scale, y_offset, c_scale;
    float r_cr, g_cb, g_cr, b_cb;
};
Yuv yuv_coefficients(bool full_range, bool bt601);

// Blends decoded NV12 frames: each tap's image is one NV12 frame (`src_h` luma
// rows and `src_h / 2` UV rows, pitch `image.pitch`; UV at `image.chroma` or,
// when that is null, right after the luma rows; OpenCL supports only the latter). Output is
// BGRA (32f or 16f) of `width` x `height`, alpha 1. Up to kMaxCudaTaps taps on
// CUDA; any number on OpenCL (32 per pass).
bool cuda_blend_nv12(void* context, void* stream, const Image& dst, const Tap* taps, int count, int width, int height,
                     int src_w, int src_h, const Yuv& yuv, Element element, std::string& error);
bool opencl_blend_nv12(void* queue, const Image& dst, const Tap* taps, int count, int width, int height, int src_w,
                       int src_h, const Yuv& yuv, Element element, std::string& error);

// Host -> device copies into buffers allocated by the host application.
bool cuda_upload(void* context, void* stream, void* device, const void* host, size_t bytes);
bool opencl_upload(void* queue, void* device, const void* host, size_t bytes);

// CUDA helpers for the NVDEC reader: a non-blocking stream in `context`, and
// a synchronous copy of `rows` pitched rows (row_bytes each) to host memory.
// Page-locked host memory, usable from every context: an upload from it is a
// direct transfer that cuda_upload only enqueues. Null without CUDA.
void* cuda_host_alloc(void* context, size_t bytes);
void cuda_host_free(void* context, void* host);

// Events marking the end of the work enqueued on a stream so far (null
// without CUDA; wait returns true for a null event).
void* cuda_event_create(void* context);
bool cuda_event_record(void* context, void* event, void* stream);
bool cuda_event_wait(void* context, void* event);
void cuda_event_destroy(void* context, void* event);
void* cuda_create_stream(void* context);
bool cuda_download(void* context, unsigned long long src, int64_t pitch, int64_t row_bytes, int rows, void* host);

// Waits until everything enqueued on the stream/queue has completed.
bool cuda_sync(void* context, void* stream);
bool opencl_finish(void* queue);

}  // namespace fb::gpu
