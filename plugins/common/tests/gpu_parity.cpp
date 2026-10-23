// GPU/CPU parity and kernel timing for fb_gpu, outside any host.
//
// For several rate pairs, the core computes the kernel; random 1080p RGBA
// frames (with padded, offset rows) are blended by the core's CPU path and by
// the CUDA and OpenCL paths, and the results are compared element by element.
// Half-float runs compare against the CPU result rounded to half (RTE).

#define NOMINMAX
#include <windows.h>

#include "fb_gpu.h"
#include "fb_plan.h"
#include "frame_blend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr int kRowElements = kWidth * 4;
constexpr int kPadBytes = 256;     // extra bytes per row
constexpr int kOffsetRows = 3;     // images start a few rows into their buffer

double now_ms() {
    static const double f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return static_cast<double>(x.QuadPart); }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return 1000.0 * static_cast<double>(c.QuadPart) / f;
}

uint16_t to_half(float f) {  // RTE, as in fb_debug.cpp
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, abs = x & 0x7fffffff;
    if (abs >= 0x477ff000) return static_cast<uint16_t>(sign | 0x7c00);
    if (abs < 0x38800000) return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(std::fabs(f) * 16777216.0f)));
    uint32_t half = (((abs >> 23) - 112) << 10) | ((abs & 0x7fffff) >> 13);
    const uint32_t rest = abs & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
}

float from_half(uint16_t h) {
    const int e = (h >> 10) & 31;
    const uint32_t m = h & 1023;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
    return (h >> 15) ? -v : v;
}

// ---- minimal driver-API / OpenCL setup for the test ----

using CUresult = int;
struct Cu {
    CUresult (__stdcall* cuInit)(unsigned);
    CUresult (__stdcall* cuDeviceGet)(int*, int);
    CUresult (__stdcall* cuDeviceGetName)(char*, int, int);
    CUresult (__stdcall* cuCtxCreate)(void**, unsigned, int);
    CUresult (__stdcall* cuMemAlloc)(unsigned long long*, size_t);
    CUresult (__stdcall* cuMemcpyHtoD)(unsigned long long, const void*, size_t);
    CUresult (__stdcall* cuMemcpyDtoH)(void*, unsigned long long, size_t);
    CUresult (__stdcall* cuStreamCreate)(void**, unsigned);
    CUresult (__stdcall* cuMemFree)(unsigned long long);
};

struct Cl {
    int (__stdcall* clGetPlatformIDs)(unsigned, void**, unsigned*);
    int (__stdcall* clGetPlatformInfo)(void*, unsigned, size_t, void*, size_t*);
    int (__stdcall* clGetDeviceIDs)(void*, uint64_t, unsigned, void**, unsigned*);
    int (__stdcall* clGetDeviceInfo)(void*, unsigned, size_t, void*, size_t*);
    void* (__stdcall* clCreateContext)(const intptr_t*, unsigned, void* const*, void*, void*, int*);
    void* (__stdcall* clCreateCommandQueue)(void*, void*, uint64_t, int*);
    void* (__stdcall* clCreateBuffer)(void*, uint64_t, size_t, void*, int*);
    int (__stdcall* clEnqueueWriteBuffer)(void*, void*, unsigned, size_t, size_t, const void*, unsigned, void*, void*);
    int (__stdcall* clEnqueueReadBuffer)(void*, void*, unsigned, size_t, size_t, void*, unsigned, void*, void*);
    int (__stdcall* clReleaseMemObject)(void*);
};

template <class T>
bool load(HMODULE m, const char* name, T& fn) {
    fn = reinterpret_cast<T>(GetProcAddress(m, name));
    return fn != nullptr;
}

struct Case {
    const char* name;
    FbRational source;
    FbRational output;
    FbRational center;
};

struct Frames {
    size_t pitch_bytes = 0;
    size_t buffer_bytes = 0;
    size_t offset_bytes = 0;
    std::vector<std::vector<uint8_t>> sources;  // padded host copies
};

// Host images with padded rows; element-wise contents in [0, 1].
Frames make_frames(size_t count, bool half, uint32_t seed) {
    Frames f;
    const size_t element = half ? 2 : 4;
    f.pitch_bytes = kRowElements * element + kPadBytes;
    f.offset_bytes = kOffsetRows * f.pitch_bytes;
    f.buffer_bytes = f.offset_bytes + f.pitch_bytes * kHeight;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t k = 0; k < count; ++k) {
        std::vector<uint8_t> buffer(f.buffer_bytes, 0xAB);
        for (int y = 0; y < kHeight; ++y) {
            uint8_t* row = buffer.data() + f.offset_bytes + y * f.pitch_bytes;
            for (int x = 0; x < kRowElements; ++x) {
                const float v = dist(rng);
                if (half) {
                    const uint16_t h = to_half(v);
                    std::memcpy(row + x * 2, &h, 2);
                } else {
                    std::memcpy(row + x * 4, &v, 4);
                }
            }
        }
        f.sources.push_back(std::move(buffer));
    }
    return f;
}

// CPU reference (the core) in float for the given frames.
std::vector<float> cpu_reference(const Frames& f, const std::vector<FbTap>& taps, bool half) {
    std::vector<std::vector<float>> floats;
    std::vector<const float*> planes;
    std::vector<ptrdiff_t> strides;
    for (const auto& buffer : f.sources) {
        if (half) {
            std::vector<float> v(static_cast<size_t>(kRowElements) * kHeight);
            for (int y = 0; y < kHeight; ++y) {
                const uint8_t* row = buffer.data() + f.offset_bytes + y * f.pitch_bytes;
                for (int x = 0; x < kRowElements; ++x) {
                    uint16_t h = 0;
                    std::memcpy(&h, row + x * 2, 2);
                    v[static_cast<size_t>(y) * kRowElements + x] = from_half(h);
                }
            }
            floats.push_back(std::move(v));
            planes.push_back(floats.back().data());
            strides.push_back(kRowElements * 4);
        } else {
            planes.push_back(reinterpret_cast<const float*>(buffer.data() + f.offset_bytes));
            strides.push_back(static_cast<ptrdiff_t>(f.pitch_bytes));
        }
    }
    std::vector<float> weights;
    for (const FbTap& t : taps) weights.push_back(t.weight);
    std::vector<float> out(static_cast<size_t>(kRowElements) * kHeight);
    fb_apply_f32(out.data(), kRowElements * 4, planes.data(), strides.data(), weights.data(),
                 static_cast<int32_t>(taps.size()), kRowElements, kHeight);
    return out;
}

struct Result {
    double max_abs = 0;
    long long differing = 0;
};

Result compare(const std::vector<uint8_t>& gpu, const Frames& f, const std::vector<float>& cpu, bool half) {
    Result r;
    for (int y = 0; y < kHeight; ++y) {
        const uint8_t* row = gpu.data() + f.offset_bytes + y * f.pitch_bytes;
        for (int x = 0; x < kRowElements; ++x) {
            const float reference = cpu[static_cast<size_t>(y) * kRowElements + x];
            float value = 0;
            if (half) {
                uint16_t h = 0;
                std::memcpy(&h, row + x * 2, 2);
                value = from_half(h);
                const float rounded = from_half(to_half(reference));
                r.differing += h != to_half(reference);
                r.max_abs = std::max(r.max_abs, static_cast<double>(std::fabs(value - rounded)));
            } else {
                std::memcpy(&value, row + x * 4, 4);
                r.differing += value != reference;
                r.max_abs = std::max(r.max_abs, static_cast<double>(std::fabs(value - reference)));
            }
        }
    }
    return r;
}

}  // namespace

int main() {
    const Case cases[] = {
        {"240->30", {240, 1}, {30, 1}, {80, 1}},
        {"600->30", {600, 1}, {30, 1}, {200, 1}},
        {"960->30", {960, 1}, {30, 1}, {10752, 1}},
        {"960->50 phase 1/5", {960, 1}, {50, 1}, {53856, 5}},
        {"2000->30", {2000, 1}, {30, 1}, {2000, 3}},
    };
    int failures = 0;

    Cu cu{};
    HMODULE nv = LoadLibraryA("nvcuda.dll");
    void* cu_stream = nullptr;
    bool have_cuda = nv && load(nv, "cuInit", cu.cuInit) && load(nv, "cuDeviceGet", cu.cuDeviceGet) &&
                     load(nv, "cuDeviceGetName", cu.cuDeviceGetName) && load(nv, "cuCtxCreate_v2", cu.cuCtxCreate) &&
                     load(nv, "cuMemAlloc_v2", cu.cuMemAlloc) && load(nv, "cuMemcpyHtoD_v2", cu.cuMemcpyHtoD) &&
                     load(nv, "cuMemcpyDtoH_v2", cu.cuMemcpyDtoH) && load(nv, "cuStreamCreate", cu.cuStreamCreate) &&
                     load(nv, "cuMemFree_v2", cu.cuMemFree);
    if (have_cuda) {
        int dev = 0;
        void* ctx = nullptr;
        char name[256] = {};
        have_cuda = cu.cuInit(0) == 0 && cu.cuDeviceGet(&dev, 0) == 0 && cu.cuCtxCreate(&ctx, 0, dev) == 0 &&
                    cu.cuStreamCreate(&cu_stream, 1 /* non-blocking */) == 0;
        cu.cuDeviceGetName(name, sizeof(name), dev);
        std::printf("CUDA device: %s\n", name);
    }

    Cl cl{};
    HMODULE ocl = LoadLibraryA("OpenCL.dll");
    void* cl_context = nullptr;
    void* cl_queue = nullptr;
    bool have_cl = ocl && load(ocl, "clGetPlatformIDs", cl.clGetPlatformIDs) &&
                   load(ocl, "clGetPlatformInfo", cl.clGetPlatformInfo) && load(ocl, "clGetDeviceIDs", cl.clGetDeviceIDs) &&
                   load(ocl, "clGetDeviceInfo", cl.clGetDeviceInfo) && load(ocl, "clCreateContext", cl.clCreateContext) &&
                   load(ocl, "clCreateCommandQueue", cl.clCreateCommandQueue) && load(ocl, "clCreateBuffer", cl.clCreateBuffer) &&
                   load(ocl, "clEnqueueWriteBuffer", cl.clEnqueueWriteBuffer) &&
                   load(ocl, "clEnqueueReadBuffer", cl.clEnqueueReadBuffer) &&
                   load(ocl, "clReleaseMemObject", cl.clReleaseMemObject);
    if (have_cl) {
        void* platforms[8] = {};
        unsigned count = 0;
        have_cl = false;
        cl.clGetPlatformIDs(8, platforms, &count);
        for (unsigned p = 0; p < count && !have_cl; ++p) {
            void* device = nullptr;
            unsigned devices = 0;
            if (cl.clGetDeviceIDs(platforms[p], 4 /* GPU */, 1, &device, &devices) != 0 || !devices) continue;
            char pname[256] = {}, dname[256] = {};
            cl.clGetPlatformInfo(platforms[p], 0x0902, sizeof(pname), pname, nullptr);
            cl.clGetDeviceInfo(device, 0x102B, sizeof(dname), dname, nullptr);
            const intptr_t props[] = {0x1084 /* CL_CONTEXT_PLATFORM */, reinterpret_cast<intptr_t>(platforms[p]), 0};
            int status = 0;
            cl_context = cl.clCreateContext(props, 1, &device, nullptr, nullptr, &status);
            if (status == 0) cl_queue = cl.clCreateCommandQueue(cl_context, device, 0, &status);
            have_cl = status == 0 && cl_queue;
            std::printf("OpenCL device: %s / %s\n", pname, dname);
        }
    }
    std::printf("frames %dx%d RGBA, row padding %d bytes, image offset %d rows\n\n", kWidth, kHeight, kPadBytes, kOffsetRows);
    std::printf("%-20s %-6s %-5s %4s %14s %12s %10s\n", "case", "api", "type", "taps", "max|gpu-cpu|", "differing", "ms/frame");

    for (const Case& c : cases) {
        std::vector<FbTap> taps;
        if (fb::kernel(c.source, c.output, c.center, INT64_MIN / 4, INT64_MAX / 4, taps) != FB_OK) {
            std::printf("%s: kernel failed\n", c.name);
            return 1;
        }
        for (bool half : {false, true}) {
            const Frames f = make_frames(taps.size(), half, 1234u + static_cast<uint32_t>(taps.size()));
            const std::vector<float> cpu = cpu_reference(f, taps, half);
            const auto element = half ? fb::gpu::Element::F16 : fb::gpu::Element::F32;
            std::vector<uint8_t> gpu(f.buffer_bytes);
            if (have_cuda) {
                std::vector<unsigned long long> mem(taps.size() + 1);
                for (auto& m : mem) cu.cuMemAlloc(&m, f.buffer_bytes);
                for (size_t k = 0; k < taps.size(); ++k) cu.cuMemcpyHtoD(mem[k], f.sources[k].data(), f.buffer_bytes);
                std::vector<fb::gpu::Tap> gt;
                for (size_t k = 0; k < taps.size(); ++k)
                    gt.push_back({{reinterpret_cast<void*>(mem[k]), static_cast<int64_t>(f.offset_bytes), static_cast<int64_t>(f.pitch_bytes)}, taps[k].weight});
                const fb::gpu::Image dst{reinterpret_cast<void*>(mem.back()), static_cast<int64_t>(f.offset_bytes), static_cast<int64_t>(f.pitch_bytes)};
                std::string error;
                bool ok = fb::gpu::cuda_blend(nullptr, cu_stream, dst, gt.data(), static_cast<int>(gt.size()), kRowElements, kHeight, element, error) &&
                          fb::gpu::cuda_sync(nullptr, cu_stream);
                const int reps = 50;
                const double t0 = now_ms();
                for (int i = 0; i < reps && ok; ++i)
                    ok = fb::gpu::cuda_blend(nullptr, cu_stream, dst, gt.data(), static_cast<int>(gt.size()), kRowElements, kHeight, element, error);
                ok = ok && fb::gpu::cuda_sync(nullptr, cu_stream);
                const double ms = (now_ms() - t0) / reps;
                cu.cuMemcpyDtoH(gpu.data(), mem.back(), f.buffer_bytes);
                for (auto m : mem) cu.cuMemFree(m);
                const Result r = compare(gpu, f, cpu, half);
                std::printf("%-20s %-6s %-5s %4zu %14.9g %12lld %10.3f %s\n", c.name, "cuda", half ? "16f" : "32f", taps.size(),
                            r.max_abs, r.differing, ms, ok ? "" : error.c_str());
                failures += !ok || r.differing != 0;
            }
            if (have_cl) {
                std::vector<void*> mem(taps.size() + 1);
                for (auto& m : mem) m = cl.clCreateBuffer(cl_context, 1, f.buffer_bytes, nullptr, nullptr);
                for (size_t k = 0; k < taps.size(); ++k) cl.clEnqueueWriteBuffer(cl_queue, mem[k], 1, 0, f.buffer_bytes, f.sources[k].data(), 0, nullptr, nullptr);
                std::vector<fb::gpu::Tap> gt;
                for (size_t k = 0; k < taps.size(); ++k)
                    gt.push_back({{mem[k], static_cast<int64_t>(f.offset_bytes), static_cast<int64_t>(f.pitch_bytes)}, taps[k].weight});
                const fb::gpu::Image dst{mem.back(), static_cast<int64_t>(f.offset_bytes), static_cast<int64_t>(f.pitch_bytes)};
                std::string error;
                bool ok = fb::gpu::opencl_blend(cl_queue, dst, gt.data(), static_cast<int>(gt.size()), kRowElements, kHeight, element, error) &&
                          fb::gpu::opencl_finish(cl_queue);
                const int reps = 50;
                const double t0 = now_ms();
                for (int i = 0; i < reps && ok; ++i)
                    ok = fb::gpu::opencl_blend(cl_queue, dst, gt.data(), static_cast<int>(gt.size()), kRowElements, kHeight, element, error);
                ok = ok && fb::gpu::opencl_finish(cl_queue);
                const double ms = (now_ms() - t0) / reps;
                cl.clEnqueueReadBuffer(cl_queue, mem.back(), 1, 0, f.buffer_bytes, gpu.data(), 0, nullptr, nullptr);
                for (auto m : mem) cl.clReleaseMemObject(m);
                const Result r = compare(gpu, f, cpu, half);
                std::printf("%-20s %-6s %-5s %4zu %14.9g %12lld %10.3f %s\n", c.name, "opencl", half ? "16f" : "32f", taps.size(),
                            r.max_abs, r.differing, ms, ok ? "" : error.c_str());
                failures += !ok || r.differing != 0;
            }
        }
    }
    std::printf("\n%s\n", failures ? "FAILED" : "all GPU results identical to the core's CPU results");
    return failures ? 1 : 0;
}
