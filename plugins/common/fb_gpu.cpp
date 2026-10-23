#include "fb_gpu.h"

#include "fb_blend_cl.h"   // kFbBlendCl: OpenCL source (generated at build time)
#include "fb_blend_ptx.h"  // kFbBlendPtx: PTX from fb_blend.cu (generated at build time)

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace fb::gpu {
namespace {

// ---- CUDA driver API, loaded from nvcuda.dll -------------------------------

using CUresult = int;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;
using CUevent = struct CUevent_st*;

struct Cuda {
    CUresult (__stdcall* cuInit)(unsigned int) = nullptr;
    CUresult (__stdcall* cuCtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (__stdcall* cuCtxPushCurrent)(CUcontext) = nullptr;
    CUresult (__stdcall* cuCtxPopCurrent)(CUcontext*) = nullptr;
    CUresult (__stdcall* cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (__stdcall* cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (__stdcall* cuModuleUnload)(CUmodule) = nullptr;
    CUresult (__stdcall* cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                         CUstream, void**, void**) = nullptr;
    CUresult (__stdcall* cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (__stdcall* cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream) = nullptr;
    CUresult (__stdcall* cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (__stdcall* cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (__stdcall* cuGetErrorString)(CUresult, const char**) = nullptr;
    // Optional (page-locked uploads): absent entries leave them unused.
    CUresult (__stdcall* cuMemHostAlloc)(void**, size_t, unsigned int) = nullptr;
    CUresult (__stdcall* cuMemFreeHost)(void*) = nullptr;
    CUresult (__stdcall* cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (__stdcall* cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (__stdcall* cuEventSynchronize)(CUevent) = nullptr;
    CUresult (__stdcall* cuEventDestroy)(CUevent) = nullptr;
    bool ok = false;
    bool pinned = false;
};

template <class F>
bool resolve(HMODULE module, const char* name, F& fn) {
    fn = reinterpret_cast<F>(GetProcAddress(module, name));
    return fn != nullptr;
}

Cuda& cuda() {
    static Cuda api;
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE m = LoadLibraryA("nvcuda.dll");
        if (!m) {
            return;
        }
        api.ok = resolve(m, "cuInit", api.cuInit) && resolve(m, "cuCtxGetCurrent", api.cuCtxGetCurrent) &&
                 resolve(m, "cuCtxPushCurrent_v2", api.cuCtxPushCurrent) &&
                 resolve(m, "cuCtxPopCurrent_v2", api.cuCtxPopCurrent) &&
                 resolve(m, "cuModuleLoadData", api.cuModuleLoadData) &&
                 resolve(m, "cuModuleGetFunction", api.cuModuleGetFunction) &&
                 resolve(m, "cuModuleUnload", api.cuModuleUnload) &&
                 resolve(m, "cuLaunchKernel", api.cuLaunchKernel) &&
                 resolve(m, "cuStreamSynchronize", api.cuStreamSynchronize) &&
                 resolve(m, "cuMemcpyHtoDAsync_v2", api.cuMemcpyHtoDAsync) &&
                 resolve(m, "cuMemcpyDtoH_v2", api.cuMemcpyDtoH) && resolve(m, "cuStreamCreate", api.cuStreamCreate) &&
                 resolve(m, "cuGetErrorString", api.cuGetErrorString) && api.cuInit(0) == 0;
        api.pinned = api.ok && resolve(m, "cuMemHostAlloc", api.cuMemHostAlloc) &&
                     resolve(m, "cuMemFreeHost", api.cuMemFreeHost) && resolve(m, "cuEventCreate", api.cuEventCreate) &&
                     resolve(m, "cuEventRecord", api.cuEventRecord) &&
                     resolve(m, "cuEventSynchronize", api.cuEventSynchronize) &&
                     resolve(m, "cuEventDestroy_v2", api.cuEventDestroy);
    });
    return api;
}

std::string cuda_error(const char* what, CUresult r) {
    const char* text = nullptr;
    if (cuda().cuGetErrorString) {
        cuda().cuGetErrorString(r, &text);
    }
    return std::string(what) + " failed: CUDA error " + std::to_string(r) + (text ? std::string(" ") + text : "");
}

struct CudaKernels {
    std::string error;  // non-empty: loading failed once for this context; not retried
    CUfunction f32 = nullptr;
    CUfunction f16 = nullptr;
    CUfunction f32x4 = nullptr;
    CUfunction f16x4 = nullptr;
    CUfunction nv12_f32 = nullptr;
    CUfunction nv12_f16 = nullptr;
};

// Must mirror FbTaps in fb_blend.cu.
struct CudaTapTable {
    unsigned long long src[kMaxCudaTaps];
    long long pitch[kMaxCudaTaps];
    float weight[kMaxCudaTaps];
};

std::mutex g_cuda_mutex;
std::map<CUcontext, CudaKernels> g_cuda_kernels;

// Pushes `context` for the lifetime of the scope when it is non-null;
// `ok` is false when that push failed.
struct ContextScope {
    bool pushed = false;
    bool ok = true;
    explicit ContextScope(void* context) {
        if (context) {
            pushed = cuda().cuCtxPushCurrent(static_cast<CUcontext>(context)) == 0;
            ok = pushed;
        }
    }
    ~ContextScope() {
        if (pushed) {
            CUcontext popped = nullptr;
            cuda().cuCtxPopCurrent(&popped);
        }
    }
};

bool cuda_kernels(CudaKernels& out, std::string& error) {
    CUcontext current = nullptr;
    if (cuda().cuCtxGetCurrent(&current) != 0 || !current) {
        error = "no current CUDA context";
        return false;
    }
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    auto it = g_cuda_kernels.find(current);
    if (it == g_cuda_kernels.end()) {
        CudaKernels k;
        CUmodule module = nullptr;
        CUresult r = cuda().cuModuleLoadData(&module, kFbBlendPtx);
        if (r != 0) {
            k.error = cuda_error("cuModuleLoadData", r);
        } else if ((r = cuda().cuModuleGetFunction(&k.f32, module, "fb_blend_f32")) != 0 ||
                   (r = cuda().cuModuleGetFunction(&k.f16, module, "fb_blend_f16")) != 0 ||
                   (r = cuda().cuModuleGetFunction(&k.f32x4, module, "fb_blend_f32x4")) != 0 ||
                   (r = cuda().cuModuleGetFunction(&k.f16x4, module, "fb_blend_f16x4")) != 0 ||
                   (r = cuda().cuModuleGetFunction(&k.nv12_f32, module, "fb_blend_nv12_f32")) != 0 ||
                   (r = cuda().cuModuleGetFunction(&k.nv12_f16, module, "fb_blend_nv12_f16")) != 0) {
            k.error = cuda_error("cuModuleGetFunction", r);
            cuda().cuModuleUnload(module);
        }
        it = g_cuda_kernels.emplace(current, k).first;
    }
    out = it->second;
    error = out.error;
    return out.error.empty();
}

// ---- OpenCL, loaded from OpenCL.dll ----------------------------------------

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_mem = struct _cl_mem*;
using cl_context = struct _cl_context*;
using cl_device_id = struct _cl_device_id*;
using cl_command_queue = struct _cl_command_queue*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

constexpr cl_uint CL_QUEUE_CONTEXT = 0x1090;
constexpr cl_uint CL_QUEUE_DEVICE = 0x1091;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr uint64_t CL_MEM_READ_WRITE = 1;
constexpr cl_uint CL_TRUE = 1;

struct OpenCl {
    cl_int (__stdcall* clGetCommandQueueInfo)(cl_command_queue, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_program (__stdcall* clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_int (__stdcall* clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
    cl_int (__stdcall* clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_kernel (__stdcall* clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_int (__stdcall* clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
    cl_int (__stdcall* clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                               const size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_mem (__stdcall* clCreateBuffer)(cl_context, uint64_t, size_t, void*, cl_int*) = nullptr;
    cl_int (__stdcall* clReleaseMemObject)(cl_mem) = nullptr;
    cl_int (__stdcall* clFinish)(cl_command_queue) = nullptr;
    cl_int (__stdcall* clReleaseProgram)(cl_program) = nullptr;
    cl_int (__stdcall* clReleaseEvent)(cl_event) = nullptr;
    cl_int (__stdcall* clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint,
                                             const cl_event*, cl_event*) = nullptr;
    bool ok = false;
};

OpenCl& opencl() {
    static OpenCl api;
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE m = LoadLibraryA("OpenCL.dll");
        if (!m) {
            return;
        }
        api.ok = resolve(m, "clGetCommandQueueInfo", api.clGetCommandQueueInfo) &&
                 resolve(m, "clCreateProgramWithSource", api.clCreateProgramWithSource) &&
                 resolve(m, "clBuildProgram", api.clBuildProgram) &&
                 resolve(m, "clGetProgramBuildInfo", api.clGetProgramBuildInfo) &&
                 resolve(m, "clCreateKernel", api.clCreateKernel) && resolve(m, "clSetKernelArg", api.clSetKernelArg) &&
                 resolve(m, "clEnqueueNDRangeKernel", api.clEnqueueNDRangeKernel) &&
                 resolve(m, "clCreateBuffer", api.clCreateBuffer) &&
                 resolve(m, "clReleaseMemObject", api.clReleaseMemObject) &&
                 resolve(m, "clFinish", api.clFinish) &&
                 resolve(m, "clReleaseProgram", api.clReleaseProgram) && resolve(m, "clReleaseEvent", api.clReleaseEvent) &&
                 resolve(m, "clEnqueueWriteBuffer", api.clEnqueueWriteBuffer);
    });
    return api;
}

constexpr int kClPass = 32;  // FB_CL_PASS in fb_blend.cl

std::mutex g_cl_mutex;  // guards the cache and kernel argument setting
struct ClKernel {
    cl_kernel kernel = nullptr;
    cl_kernel nv12 = nullptr;
    std::string error;  // non-empty: the build failed once for this device; not retried
};
std::map<std::pair<cl_context, cl_device_id>, ClKernel> g_cl_kernels;

std::string build_log(cl_program program, cl_device_id device) {
    size_t size = 0;
    opencl().clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    std::string log(size, ' ');
    if (size) {
        opencl().clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
    }
    return log;
}

ClKernel build_kernel(cl_context context, cl_device_id device) {
    ClKernel result;
    cl_int status = 0;
    const char* source = kFbBlendCl;
    const size_t length = std::strlen(kFbBlendCl);
    cl_program program = opencl().clCreateProgramWithSource(context, 1, &source, &length, &status);
    if (status != 0) {
        result.error = "clCreateProgramWithSource failed: " + std::to_string(status);
        return result;
    }
    // No -cl-mad-enable / fast-math: every multiply and add is rounded separately.
    status = opencl().clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (status != 0) {
        result.error = "clBuildProgram failed: " + std::to_string(status) + " " + build_log(program, device);
    } else {
        result.kernel = opencl().clCreateKernel(program, "fb_blend", &status);
        cl_int nv12_status = 0;
        result.nv12 = status == 0 ? opencl().clCreateKernel(program, "fb_blend_nv12", &nv12_status) : nullptr;
        if (status != 0 || nv12_status != 0) {
            result.error = "clCreateKernel failed: " + std::to_string(status != 0 ? status : nv12_status);
            result.kernel = nullptr;
            result.nv12 = nullptr;
        }
    }
    opencl().clReleaseProgram(program);  // a kernel keeps its program alive
    return result;
}

cl_kernel opencl_kernel(cl_command_queue queue, cl_context& context, std::string& error, bool nv12 = false) {
    cl_device_id device = nullptr;
    if (opencl().clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr) != 0 ||
        opencl().clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr) != 0) {
        error = "clGetCommandQueueInfo failed";
        return nullptr;
    }
    auto it = g_cl_kernels.find({context, device});
    if (it == g_cl_kernels.end()) {
        it = g_cl_kernels.emplace(std::make_pair(context, device), build_kernel(context, device)).first;
    }
    error = it->second.error;
    return nv12 ? it->second.nv12 : it->second.kernel;
}

bool fits_int(int64_t v) {
    return v >= INT32_MIN && v <= INT32_MAX;
}

}  // namespace

bool cuda_blend(void* context, void* stream, const Image& dst, const Tap* taps, int count, int row_elements, int rows,
                Element element, std::string& error) {
    if (!cuda().ok) {
        error = "nvcuda.dll is not available";
        return false;
    }
    if (count <= 0 || count > kMaxCudaTaps) {
        error = "tap count outside 1.." + std::to_string(kMaxCudaTaps);
        return false;
    }
    ContextScope scope(context);
    if (!scope.ok) {
        error = "cuCtxPushCurrent failed";
        return false;
    }
    CudaKernels kernels;
    if (!cuda_kernels(kernels, error)) {
        return false;
    }
    CudaTapTable table{};
    for (int k = 0; k < count; ++k) {
        table.src[k] = reinterpret_cast<CUdeviceptr>(taps[k].image.base) + taps[k].image.offset;
        table.pitch[k] = taps[k].image.pitch;
        table.weight[k] = taps[k].weight;
    }
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(dst.base) + dst.offset;
    long long out_pitch = dst.pitch;
    // One pixel (4 elements) per thread when every row start is vector aligned.
    const unsigned long long vector_bytes = 4 * element_size(element);
    bool vector = row_elements % 4 == 0 && out % vector_bytes == 0 && out_pitch % vector_bytes == 0;
    for (int k = 0; k < count && vector; ++k) {
        vector = table.src[k] % vector_bytes == 0 && table.pitch[k] % static_cast<long long>(vector_bytes) == 0;
    }
    int width = vector ? row_elements / 4 : row_elements;
    void* args[] = {&out, &out_pitch, &width, &rows, &count, &table};
    CUfunction function = element == Element::F32 ? (vector ? kernels.f32x4 : kernels.f32)
                                                  : (vector ? kernels.f16x4 : kernels.f16);
    const unsigned bx = 64;
    const unsigned by = 4;
    const CUresult r = cuda().cuLaunchKernel(function, (width + bx - 1) / bx, (rows + by - 1) / by, 1, bx, by, 1, 0,
                                             static_cast<CUstream>(stream), args, nullptr);
    if (r != 0) {
        error = cuda_error("cuLaunchKernel", r);
        return false;
    }
    return true;
}

bool opencl_blend(void* queue_handle, const Image& dst, const Tap* taps, int count, int row_elements, int rows,
                  Element element, std::string& error) {
    if (!opencl().ok) {
        error = "OpenCL.dll is not available";
        return false;
    }
    if (count <= 0 || row_elements % 4 != 0) {
        error = "no taps or not RGBA rows";
        return false;
    }
    int row_pixels = row_elements / 4;
    const int64_t float_pitch = static_cast<int64_t>(row_elements) * 4;
    if (!fits_int(dst.offset) || !fits_int(dst.pitch) || !fits_int(float_pitch)) {
        error = "image too large for 32-bit offsets";
        return false;
    }
    for (int k = 0; k < count; ++k) {
        if (!fits_int(taps[k].image.offset) || !fits_int(taps[k].image.pitch)) {
            error = "image too large for 32-bit offsets";
            return false;
        }
    }
    auto queue = static_cast<cl_command_queue>(queue_handle);
    std::lock_guard<std::mutex> lock(g_cl_mutex);
    cl_context context = nullptr;
    cl_kernel kernel = opencl_kernel(queue, context, error);
    if (!kernel) {
        return false;
    }
    const int passes = (count + kClPass - 1) / kClPass;
    // Intermediate passes keep a float accumulator: the float destination itself,
    // or a scratch buffer when the destination is half.
    cl_mem scratch = nullptr;
    if (passes > 1 && element == Element::F16) {
        cl_int status = 0;
        scratch = opencl().clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(float_pitch) * rows, nullptr, &status);
        if (status != 0) {
            error = "clCreateBuffer failed: " + std::to_string(status);
            return false;
        }
    }
    bool ok = true;
    cl_event previous = nullptr;
    for (int pass = 0; pass < passes && ok; ++pass) {
        const bool last = pass == passes - 1;
        const int first_tap = pass * kClPass;
        const int n = std::min(kClPass, count - first_tap);
        cl_mem out_mem = static_cast<cl_mem>(dst.base);
        int out_offset = static_cast<int>(dst.offset);
        int out_pitch = static_cast<int>(dst.pitch);
        int out_half = element == Element::F16 ? 1 : 0;
        if (!last && scratch) {
            out_mem = scratch;
            out_offset = 0;
            out_pitch = static_cast<int>(float_pitch);
            out_half = 0;
        }
        cl_mem acc_mem = scratch ? scratch : static_cast<cl_mem>(dst.base);
        int acc_offset = scratch ? 0 : static_cast<int>(dst.offset);
        int acc_pitch = scratch ? static_cast<int>(float_pitch) : static_cast<int>(dst.pitch);
        int use_acc = pass > 0 ? 1 : 0;
        int half_src = element == Element::F16 ? 1 : 0;
        cl_uint a = 0;
        auto arg = [&](size_t size, const void* value) {
            ok = ok && opencl().clSetKernelArg(kernel, a++, size, value) == 0;
        };
        arg(sizeof(cl_mem), &out_mem);
        arg(sizeof(int), &out_offset);
        arg(sizeof(int), &out_pitch);
        arg(sizeof(int), &out_half);
        arg(sizeof(cl_mem), &acc_mem);
        arg(sizeof(int), &acc_offset);
        arg(sizeof(int), &acc_pitch);
        arg(sizeof(int), &use_acc);
        arg(sizeof(int), &row_pixels);
        arg(sizeof(int), &rows);
        arg(sizeof(int), &n);
        arg(sizeof(int), &half_src);
        for (int k = 0; k < kClPass; ++k) {
            const Tap& tap = taps[first_tap + std::min(k, n - 1)];
            cl_mem mem = static_cast<cl_mem>(tap.image.base);
            int offset = static_cast<int>(tap.image.offset);
            int pitch = static_cast<int>(tap.image.pitch);
            float weight = k < n ? tap.weight : 0.0f;
            arg(sizeof(cl_mem), &mem);
            arg(sizeof(int), &offset);
            arg(sizeof(int), &pitch);
            arg(sizeof(float), &weight);
        }
        if (!ok) {
            error = "clSetKernelArg failed";
            break;
        }
        const size_t global[2] = {static_cast<size_t>(row_pixels), static_cast<size_t>(rows)};
        // Each pass waits for the previous one, which also holds on out-of-order queues.
        cl_event done = nullptr;
        const cl_int status = opencl().clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr,
                                                              previous ? 1 : 0, previous ? &previous : nullptr,
                                                              last ? nullptr : &done);
        if (previous) {
            opencl().clReleaseEvent(previous);
            previous = nullptr;
        }
        if (status != 0) {
            ok = false;
            error = "clEnqueueNDRangeKernel failed: " + std::to_string(status);
        }
        previous = done;
    }
    if (previous) {
        opencl().clReleaseEvent(previous);
    }
    if (scratch) {
        opencl().clReleaseMemObject(scratch);  // freed once the enqueued passes complete
    }
    return ok;
}

Yuv yuv_coefficients(bool full_range, bool bt601) {
    const double kr = bt601 ? 0.299 : 0.2126;
    const double kb = bt601 ? 0.114 : 0.0722;
    const double kg = 1.0 - kr - kb;
    Yuv k{};
    k.y_scale = static_cast<float>(full_range ? 1.0 / 255.0 : 1.0 / 219.0);
    k.y_offset = full_range ? 0.0f : 16.0f;
    k.c_scale = static_cast<float>(full_range ? 1.0 / 255.0 : 1.0 / 224.0);
    k.r_cr = static_cast<float>(2.0 * (1.0 - kr));
    k.b_cb = static_cast<float>(2.0 * (1.0 - kb));
    k.g_cb = static_cast<float>(2.0 * kb * (1.0 - kb) / kg);
    k.g_cr = static_cast<float>(2.0 * kr * (1.0 - kr) / kg);
    return k;
}

bool cuda_blend_nv12(void* context, void* stream, const Image& dst, const Tap* taps, int count, int width, int height,
                     int src_w, int src_h, const Yuv& yuv, Element element, std::string& error) {
    if (!cuda().ok) {
        error = "nvcuda.dll is not available";
        return false;
    }
    if (count <= 0 || count > kMaxCudaTaps) {
        error = "tap count outside 1.." + std::to_string(kMaxCudaTaps);
        return false;
    }
    ContextScope scope(context);
    if (!scope.ok) {
        error = "cuCtxPushCurrent failed";
        return false;
    }
    CudaKernels kernels;
    if (!cuda_kernels(kernels, error)) {
        return false;
    }
    // Must mirror FbNv12Taps in fb_blend.cu.
    struct {
        unsigned long long luma[kMaxCudaTaps];
        unsigned long long chroma[kMaxCudaTaps];
        long long pitch[kMaxCudaTaps];
        float weight[kMaxCudaTaps];
    } table{};
    for (int k = 0; k < count; ++k) {
        const Image& image = taps[k].image;
        table.luma[k] = reinterpret_cast<CUdeviceptr>(image.base) + image.offset;
        table.chroma[k] = image.chroma ? reinterpret_cast<CUdeviceptr>(image.chroma)
                                       : table.luma[k] + static_cast<CUdeviceptr>(image.pitch) * src_h;
        table.pitch[k] = image.pitch;
        table.weight[k] = taps[k].weight;
    }
    CUdeviceptr out = reinterpret_cast<CUdeviceptr>(dst.base) + dst.offset;
    long long out_pitch = dst.pitch;
    Yuv coefficients = yuv;
    void* args[] = {&out, &out_pitch, &width, &height, &count, &src_w, &src_h, &coefficients, &table};
    const unsigned bx = 32;
    const unsigned by = 8;
    const CUresult r = cuda().cuLaunchKernel(element == Element::F32 ? kernels.nv12_f32 : kernels.nv12_f16,
                                             (width + bx - 1) / bx, (height + by - 1) / by, 1, bx, by, 1, 0,
                                             static_cast<CUstream>(stream), args, nullptr);
    if (r != 0) {
        error = cuda_error("cuLaunchKernel", r);
        return false;
    }
    return true;
}

bool opencl_blend_nv12(void* queue_handle, const Image& dst, const Tap* taps, int count, int width, int height,
                       int src_w, int src_h, const Yuv& yuv, Element element, std::string& error) {
    if (!opencl().ok) {
        error = "OpenCL.dll is not available";
        return false;
    }
    if (count <= 0 || !fits_int(dst.offset) || !fits_int(dst.pitch)) {
        error = "bad arguments";
        return false;
    }
    auto queue = static_cast<cl_command_queue>(queue_handle);
    std::lock_guard<std::mutex> lock(g_cl_mutex);
    cl_context context = nullptr;
    cl_kernel kernel = opencl_kernel(queue, context, error, true);
    if (!kernel) {
        return false;
    }
    const int passes = (count + kClPass - 1) / kClPass;
    const int float_pitch = width * 16;
    cl_mem scratch = nullptr;
    if (passes > 1) {
        cl_int status = 0;
        scratch = opencl().clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(float_pitch) * height,
                                          nullptr, &status);
        if (status != 0) {
            error = "clCreateBuffer failed: " + std::to_string(status);
            return false;
        }
    }
    bool ok = true;
    cl_event previous = nullptr;
    for (int pass = 0; pass < passes && ok; ++pass) {
        const bool last_pass = pass == passes - 1;
        const int first_tap = pass * kClPass;
        const int n = std::min(kClPass, count - first_tap);
        cl_mem out_mem = last_pass ? static_cast<cl_mem>(dst.base) : scratch;
        int out_offset = last_pass ? static_cast<int>(dst.offset) : 0;
        int out_pitch = last_pass ? static_cast<int>(dst.pitch) : float_pitch;
        int out_half = last_pass && element == Element::F16 ? 1 : 0;
        cl_mem acc_mem = scratch ? scratch : static_cast<cl_mem>(dst.base);
        int acc_offset = 0;
        int acc_pitch = float_pitch;
        int use_acc = pass > 0 ? 1 : 0;
        int last = last_pass ? 1 : 0;
        cl_uint a = 0;
        auto arg = [&](size_t size, const void* value) {
            ok = ok && opencl().clSetKernelArg(kernel, a++, size, value) == 0;
        };
        arg(sizeof(cl_mem), &out_mem);
        arg(sizeof(int), &out_offset);
        arg(sizeof(int), &out_pitch);
        arg(sizeof(int), &out_half);
        arg(sizeof(cl_mem), &acc_mem);
        arg(sizeof(int), &acc_offset);
        arg(sizeof(int), &acc_pitch);
        arg(sizeof(int), &use_acc);
        arg(sizeof(int), &width);
        arg(sizeof(int), &height);
        arg(sizeof(int), &n);
        arg(sizeof(int), &src_w);
        arg(sizeof(int), &src_h);
        arg(sizeof(int), &last);
        const float coefficients[7] = {yuv.y_scale, yuv.y_offset, yuv.c_scale, yuv.r_cr, yuv.g_cb, yuv.g_cr, yuv.b_cb};
        for (float c : coefficients) {
            arg(sizeof(float), &c);
        }
        for (int k = 0; k < kClPass; ++k) {
            const Tap& tap = taps[first_tap + std::min(k, n - 1)];
            cl_mem mem = static_cast<cl_mem>(tap.image.base);
            int offset = 0;
            int pitch = static_cast<int>(tap.image.pitch);
            float weight = k < n ? tap.weight : 0.0f;
            arg(sizeof(cl_mem), &mem);
            arg(sizeof(int), &offset);
            arg(sizeof(int), &pitch);
            arg(sizeof(float), &weight);
        }
        if (!ok) {
            error = "clSetKernelArg failed";
            break;
        }
        const size_t global[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
        cl_event done = nullptr;
        const cl_int status = opencl().clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr,
                                                              previous ? 1 : 0, previous ? &previous : nullptr,
                                                              last_pass ? nullptr : &done);
        if (previous) {
            opencl().clReleaseEvent(previous);
            previous = nullptr;
        }
        if (status != 0) {
            ok = false;
            error = "clEnqueueNDRangeKernel failed: " + std::to_string(status);
        }
        previous = done;
    }
    if (previous) {
        opencl().clReleaseEvent(previous);
    }
    if (scratch) {
        opencl().clReleaseMemObject(scratch);
    }
    return ok;
}

bool cuda_upload(void* context, void* stream, void* device, const void* host, size_t bytes) {
    if (!cuda().ok) {
        return false;
    }
    ContextScope scope(context);
    // From pageable memory the copy is staged before this call returns.
    return scope.ok && cuda().cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(device), host, bytes,
                                                static_cast<CUstream>(stream)) == 0;
}

bool opencl_upload(void* queue, void* device, const void* host, size_t bytes) {
    return opencl().ok &&
           opencl().clEnqueueWriteBuffer(static_cast<cl_command_queue>(queue), static_cast<cl_mem>(device), CL_TRUE, 0,
                                         bytes, host, 0, nullptr, nullptr) == 0;
}

void* cuda_host_alloc(void* context, size_t bytes) {
    if (!cuda().pinned) {
        return nullptr;
    }
    ContextScope scope(context);
    void* host = nullptr;
    constexpr unsigned kPortable = 0x01;  // CU_MEMHOSTALLOC_PORTABLE: page-locked for every context
    return scope.ok && cuda().cuMemHostAlloc(&host, bytes, kPortable) == 0 ? host : nullptr;
}

void cuda_host_free(void* context, void* host) {
    if (!host || !cuda().pinned) {
        return;
    }
    ContextScope scope(context);
    cuda().cuMemFreeHost(host);
}

void* cuda_event_create(void* context) {
    if (!cuda().pinned) {
        return nullptr;
    }
    ContextScope scope(context);
    CUevent event = nullptr;
    constexpr unsigned kDisableTiming = 0x02;  // CU_EVENT_DISABLE_TIMING
    return scope.ok && cuda().cuEventCreate(&event, kDisableTiming) == 0 ? event : nullptr;
}

bool cuda_event_record(void* context, void* event, void* stream) {
    if (!event) {
        return false;
    }
    ContextScope scope(context);
    return scope.ok && cuda().cuEventRecord(static_cast<CUevent>(event), static_cast<CUstream>(stream)) == 0;
}

bool cuda_event_wait(void* context, void* event) {
    if (!event) {
        return true;
    }
    ContextScope scope(context);
    return scope.ok && cuda().cuEventSynchronize(static_cast<CUevent>(event)) == 0;
}

void cuda_event_destroy(void* context, void* event) {
    if (!event) {
        return;
    }
    ContextScope scope(context);
    cuda().cuEventDestroy(static_cast<CUevent>(event));
}

void* cuda_create_stream(void* context) {
    if (!cuda().ok) {
        return nullptr;
    }
    ContextScope scope(context);
    CUstream stream = nullptr;
    return scope.ok && cuda().cuStreamCreate(&stream, 1 /* CU_STREAM_NON_BLOCKING */) == 0 ? stream : nullptr;
}

bool cuda_download(void* context, unsigned long long src, int64_t pitch, int64_t row_bytes, int rows, void* host) {
    if (!cuda().ok) {
        return false;
    }
    ContextScope scope(context);
    if (!scope.ok) {
        return false;
    }
    std::vector<uint8_t> block(static_cast<size_t>(pitch) * rows);
    if (cuda().cuMemcpyDtoH(block.data(), src, block.size()) != 0) {
        return false;
    }
    for (int r = 0; r < rows; ++r) {
        std::memcpy(static_cast<uint8_t*>(host) + r * row_bytes, block.data() + r * pitch, static_cast<size_t>(row_bytes));
    }
    return true;
}

bool cuda_sync(void* context, void* stream) {
    if (!cuda().ok) {
        return false;
    }
    ContextScope scope(context);
    return scope.ok && cuda().cuStreamSynchronize(static_cast<CUstream>(stream)) == 0;
}

bool opencl_finish(void* queue) {
    return opencl().ok && opencl().clFinish(static_cast<cl_command_queue>(queue)) == 0;
}

}  // namespace fb::gpu
