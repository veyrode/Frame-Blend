// GPU/CPU parity of the NV12 blend kernels (Premiere file path): random 8-bit
// NV12 frames are converted and blended by CUDA (separate luma/chroma planes,
// as NVDEC surfaces) and OpenCL (compact frames, as uploads), and compared
// with the same conversion done on the CPU plus the core's fb_apply_f32.

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

double now_ms() {
    static const double f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return static_cast<double>(x.QuadPart); }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return 1000.0 * static_cast<double>(c.QuadPart) / f;
}

uint16_t to_half(float f) {
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

// Same operations, in the same order, as fb_blend_nv12 in fb_blend.cu.
float chroma(const uint8_t* uv, int pitch, int cw, int ch, float cx, float cy, int c) {
    const int x0 = std::max(0, std::min(cw - 1, static_cast<int>(std::floor(cx))));
    const int y0 = std::max(0, std::min(ch - 1, static_cast<int>(std::floor(cy))));
    const int x1 = std::min(cw - 1, x0 + 1), y1 = std::min(ch - 1, y0 + 1);
    const float fx = std::fmin(std::fmax(cx - std::floor(cx), 0.0f), 1.0f);
    const float fy = std::fmin(std::fmax(cy - std::floor(cy), 0.0f), 1.0f);
    const float a = uv[y0 * pitch + 2 * x0 + c], b = uv[y0 * pitch + 2 * x1 + c];
    const float d0 = uv[y1 * pitch + 2 * x0 + c], d1 = uv[y1 * pitch + 2 * x1 + c];
    const float top = a + fx * (b - a), bottom = d0 + fx * (d1 - d0);
    return top + fy * (bottom - top);
}

void convert(const std::vector<uint8_t>& nv12, int w, int h, const fb::gpu::Yuv& k, int ow, int oh, float* out) {
    const uint8_t* uv = nv12.data() + static_cast<size_t>(w) * h;
    for (int y = 0; y < oh; ++y) {
        const int sy = oh == h ? y : std::min(h - 1, static_cast<int>((static_cast<int64_t>(y) * h) / oh));
        for (int x = 0; x < ow; ++x) {
            const int sx = ow == w ? x : std::min(w - 1, static_cast<int>((static_cast<int64_t>(x) * w) / ow));
            const float cx = 0.5f * sx, cy = 0.5f * sy - 0.25f;
            const float yy = (static_cast<float>(nv12[static_cast<size_t>(sy) * w + sx]) - k.y_offset) * k.y_scale;
            const float cb = (chroma(uv, w, w / 2, h / 2, cx, cy, 0) - 128.0f) * k.c_scale;
            const float cr = (chroma(uv, w, w / 2, h / 2, cx, cy, 1) - 128.0f) * k.c_scale;
            float* p = out + (static_cast<size_t>(y) * ow + x) * 4;
            p[0] = yy + k.b_cb * cb;
            p[1] = (yy - k.g_cb * cb) - k.g_cr * cr;
            p[2] = yy + k.r_cr * cr;
            p[3] = 1.0f;
        }
    }
}

template <class T>
bool load(HMODULE m, const char* name, T& fn) {
    fn = reinterpret_cast<T>(GetProcAddress(m, name));
    return fn != nullptr;
}

}  // namespace

int main() {
    const int w = 1280, h = 720;
    struct Case { const char* name; FbRational src, out, center; int ow, oh; bool full; };
    const Case cases[] = {
        {"720->30 full range", {720, 1}, {30, 1}, {720, 1}, w, h, true},
        {"600->30 limited", {600, 1}, {30, 1}, {200, 1}, w, h, false},
        {"960->30 half-res out", {960, 1}, {30, 1}, {320, 1}, w / 2, h / 2, true},
        {"2000->30 (68 taps)", {2000, 1}, {30, 1}, {2000, 3}, w, h, true},
    };
    // CUDA context, streams and memory.
    HMODULE nv = LoadLibraryA("nvcuda.dll");
    int (__stdcall* cuInit)(unsigned) = nullptr;
    int (__stdcall* cuDeviceGet)(int*, int) = nullptr;
    int (__stdcall* cuCtxCreate)(void**, unsigned, int) = nullptr;
    int (__stdcall* cuMemAlloc)(unsigned long long*, size_t) = nullptr;
    int (__stdcall* cuMemFree)(unsigned long long) = nullptr;
    int (__stdcall* cuMemcpyHtoD)(unsigned long long, const void*, size_t) = nullptr;
    int (__stdcall* cuMemcpyDtoH)(void*, unsigned long long, size_t) = nullptr;
    void* ctx = nullptr;
    bool have_cuda = nv && load(nv, "cuInit", cuInit) && load(nv, "cuDeviceGet", cuDeviceGet) &&
                     load(nv, "cuCtxCreate_v2", cuCtxCreate) && load(nv, "cuMemAlloc_v2", cuMemAlloc) &&
                     load(nv, "cuMemFree_v2", cuMemFree) && load(nv, "cuMemcpyHtoD_v2", cuMemcpyHtoD) &&
                     load(nv, "cuMemcpyDtoH_v2", cuMemcpyDtoH);
    int dev = 0;
    have_cuda = have_cuda && cuInit(0) == 0 && cuDeviceGet(&dev, 0) == 0 && cuCtxCreate(&ctx, 0, dev) == 0;
    void* stream = have_cuda ? fb::gpu::cuda_create_stream(nullptr) : nullptr;
    // OpenCL context on the first GPU.
    HMODULE ocl = LoadLibraryA("OpenCL.dll");
    int (__stdcall* clGetPlatformIDs)(unsigned, void**, unsigned*) = nullptr;
    int (__stdcall* clGetDeviceIDs)(void*, uint64_t, unsigned, void**, unsigned*) = nullptr;
    void* (__stdcall* clCreateContext)(const intptr_t*, unsigned, void* const*, void*, void*, int*) = nullptr;
    void* (__stdcall* clCreateCommandQueue)(void*, void*, uint64_t, int*) = nullptr;
    void* (__stdcall* clCreateBuffer)(void*, uint64_t, size_t, void*, int*) = nullptr;
    int (__stdcall* clEnqueueReadBuffer)(void*, void*, unsigned, size_t, size_t, void*, unsigned, void*, void*) = nullptr;
    int (__stdcall* clReleaseMemObject)(void*) = nullptr;
    void* cl_ctx = nullptr;
    void* queue = nullptr;
    if (ocl && load(ocl, "clGetPlatformIDs", clGetPlatformIDs) && load(ocl, "clGetDeviceIDs", clGetDeviceIDs) &&
        load(ocl, "clCreateContext", clCreateContext) && load(ocl, "clCreateCommandQueue", clCreateCommandQueue) &&
        load(ocl, "clCreateBuffer", clCreateBuffer) && load(ocl, "clEnqueueReadBuffer", clEnqueueReadBuffer) &&
        load(ocl, "clReleaseMemObject", clReleaseMemObject)) {
        void* platforms[8] = {};
        unsigned n = 0;
        clGetPlatformIDs(8, platforms, &n);
        for (unsigned p = 0; p < n && !queue; ++p) {
            void* device = nullptr;
            unsigned count = 0;
            if (clGetDeviceIDs(platforms[p], 4, 1, &device, &count) != 0 || !count) continue;
            const intptr_t props[] = {0x1084, reinterpret_cast<intptr_t>(platforms[p]), 0};
            int status = 0;
            cl_ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &status);
            if (status == 0) queue = clCreateCommandQueue(cl_ctx, device, 0, &status);
        }
    }
    int failures = 0;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> byte(0, 255);
    std::printf("%-22s %-6s %-4s %4s %14s %10s %9s\n", "case", "api", "out", "taps", "max|gpu-cpu|", "differing", "ms");
    for (const Case& c : cases) {
        std::vector<FbTap> taps;
        fb::kernel(c.src, c.out, c.center, INT64_MIN / 4, INT64_MAX / 4, taps);
        const fb::gpu::Yuv k = fb::gpu::yuv_coefficients(c.full, false);
        const size_t frame_bytes = static_cast<size_t>(w) * h * 3 / 2;
        std::vector<std::vector<uint8_t>> frames(taps.size(), std::vector<uint8_t>(frame_bytes));
        for (auto& f : frames) for (auto& b : f) b = static_cast<uint8_t>(byte(rng));
        // CPU reference.
        const size_t n = static_cast<size_t>(c.ow) * c.oh * 4;
        std::vector<std::vector<float>> rgb(taps.size(), std::vector<float>(n));
        std::vector<const float*> planes;
        std::vector<ptrdiff_t> strides;
        std::vector<float> weights;
        for (size_t t = 0; t < taps.size(); ++t) {
            convert(frames[t], w, h, k, c.ow, c.oh, rgb[t].data());
            planes.push_back(rgb[t].data());
            strides.push_back(c.ow * 16);
            weights.push_back(taps[t].weight);
        }
        std::vector<float> cpu(n);
        fb_apply_f32(cpu.data(), c.ow * 16, planes.data(), strides.data(), weights.data(), static_cast<int32_t>(taps.size()),
                     c.ow * 4, c.oh);
        for (bool half : {false, true}) {
            const fb::gpu::Element e = half ? fb::gpu::Element::F16 : fb::gpu::Element::F32;
            const int64_t out_pitch = static_cast<int64_t>(c.ow) * 4 * (half ? 2 : 4);
            std::vector<uint8_t> out(static_cast<size_t>(out_pitch) * c.oh);
            auto compare = [&](const char* api, double ms, bool ok, const std::string& error) {
                double max_abs = 0.0;
                long long differing = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (i % 4 == 3) continue;
                    float g = 0.0f, ref = cpu[i];
                    if (half) {
                        uint16_t hv = 0;
                        std::memcpy(&hv, out.data() + 2 * i, 2);
                        differing += hv != to_half(ref);
                        const int ex = (hv >> 10) & 31;
                        const uint32_t m = hv & 1023;
                        g = ex == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), ex - 25);
                        if (hv >> 15) g = -g;
                        const uint16_t rh = to_half(ref);
                        const int rex = (rh >> 10) & 31;
                        const uint32_t rm = rh & 1023;
                        ref = rex == 0 ? std::ldexp(static_cast<float>(rm), -24) : std::ldexp(static_cast<float>(rm | 1024), rex - 25);
                        if (rh >> 15) ref = -ref;
                    } else {
                        std::memcpy(&g, out.data() + 4 * i, 4);
                        differing += g != ref;
                    }
                    max_abs = std::max(max_abs, static_cast<double>(std::fabs(g - ref)));
                }
                std::printf("%-22s %-6s %-4s %4zu %14.9g %10lld %9.3f %s\n", c.name, api, half ? "16f" : "32f", taps.size(),
                            max_abs, differing, ms, ok ? "" : error.c_str());
                failures += !ok || differing != 0;
            };
            if (have_cuda) {
                // Separate planes, padded pitch: like NVDEC surfaces.
                const int64_t pitch = w + 256;
                std::vector<unsigned long long> luma(taps.size()), chroma_planes(taps.size());
                std::vector<fb::gpu::Tap> gt;
                for (size_t t = 0; t < taps.size(); ++t) {
                    cuMemAlloc(&luma[t], static_cast<size_t>(pitch) * h);
                    cuMemAlloc(&chroma_planes[t], static_cast<size_t>(pitch) * h / 2);
                    for (int y = 0; y < h; ++y) cuMemcpyHtoD(luma[t] + y * pitch, frames[t].data() + static_cast<size_t>(y) * w, w);
                    for (int y = 0; y < h / 2; ++y)
                        cuMemcpyHtoD(chroma_planes[t] + y * pitch, frames[t].data() + static_cast<size_t>(w) * h + static_cast<size_t>(y) * w, w);
                    gt.push_back({fb::gpu::Image{reinterpret_cast<void*>(luma[t]), 0, pitch, reinterpret_cast<void*>(chroma_planes[t])},
                                  taps[t].weight});
                }
                unsigned long long dst = 0;
                cuMemAlloc(&dst, out.size());
                std::string error;
                const fb::gpu::Image d{reinterpret_cast<void*>(dst), 0, out_pitch};
                bool ok = fb::gpu::cuda_blend_nv12(nullptr, stream, d, gt.data(), static_cast<int>(gt.size()), c.ow, c.oh, w,
                                                   h, k, e, error) && fb::gpu::cuda_sync(nullptr, stream);
                const double t0 = now_ms();
                for (int i = 0; i < 20 && ok; ++i)
                    ok = fb::gpu::cuda_blend_nv12(nullptr, stream, d, gt.data(), static_cast<int>(gt.size()), c.ow, c.oh, w, h,
                                                  k, e, error);
                ok = ok && fb::gpu::cuda_sync(nullptr, stream);
                const double ms = (now_ms() - t0) / 20;
                cuMemcpyDtoH(out.data(), dst, out.size());
                compare("cuda", ms, ok, error);
                for (size_t t = 0; t < taps.size(); ++t) { cuMemFree(luma[t]); cuMemFree(chroma_planes[t]); }
                cuMemFree(dst);
            }
            if (queue) {
                std::vector<void*> mem;
                std::vector<fb::gpu::Tap> gt;
                for (size_t t = 0; t < taps.size(); ++t) {
                    void* m = clCreateBuffer(cl_ctx, 1, frame_bytes, nullptr, nullptr);
                    fb::gpu::opencl_upload(queue, m, frames[t].data(), frame_bytes);
                    mem.push_back(m);
                    gt.push_back({fb::gpu::Image{m, 0, w, nullptr}, taps[t].weight});
                }
                void* dst = clCreateBuffer(cl_ctx, 1, out.size(), nullptr, nullptr);
                std::string error;
                const fb::gpu::Image d{dst, 0, out_pitch};
                bool ok = fb::gpu::opencl_blend_nv12(queue, d, gt.data(), static_cast<int>(gt.size()), c.ow, c.oh, w, h, k, e,
                                                     error) && fb::gpu::opencl_finish(queue);
                const double t0 = now_ms();
                for (int i = 0; i < 20 && ok; ++i)
                    ok = fb::gpu::opencl_blend_nv12(queue, d, gt.data(), static_cast<int>(gt.size()), c.ow, c.oh, w, h, k, e, error);
                ok = ok && fb::gpu::opencl_finish(queue);
                const double ms = (now_ms() - t0) / 20;
                clEnqueueReadBuffer(queue, dst, 1, 0, out.size(), out.data(), 0, nullptr, nullptr);
                compare("opencl", ms, ok, error);
                for (void* m : mem) clReleaseMemObject(m);
                clReleaseMemObject(dst);
            }
        }
    }
    std::printf("\n%s\n", failures ? "FAILED" : "all NV12 GPU results identical to the CPU reference");
    return failures ? 1 : 0;
}
