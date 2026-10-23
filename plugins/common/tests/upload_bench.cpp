// CPU cost of uploading decoded 720p NV12 frames to the device the way the
// Premiere filter does (cuda_upload on a stream), from ordinary memory and
// from page-locked memory (fb::gpu::cuda_host_alloc).
//
//   upload_bench [frames]

#define NOMINMAX
#include <windows.h>

#include "fb_gpu.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using CuResult = int;

struct Driver {
    CuResult(__stdcall* init)(unsigned) = nullptr;
    CuResult(__stdcall* device_get)(int*, int) = nullptr;
    CuResult(__stdcall* ctx_create)(void**, unsigned, int) = nullptr;
    CuResult(__stdcall* ctx_pop)(void**) = nullptr;
    CuResult(__stdcall* ctx_push)(void*) = nullptr;
    CuResult(__stdcall* mem_alloc)(unsigned long long*, size_t) = nullptr;
};

double cpu_seconds() {
    FILETIME creation, exit, kernel, user;
    GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
    auto value = [](FILETIME t) { return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return (value(kernel) + value(user)) / 1e7;
}

double wall_seconds() {
    LARGE_INTEGER now, frequency;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    return static_cast<double>(now.QuadPart) / static_cast<double>(frequency.QuadPart);
}

}  // namespace

int main(int argc, char** argv) {
    const int frames = argc > 1 ? std::atoi(argv[1]) : 2400;
    const size_t bytes = 1280 * 720 * 3 / 2;
    HMODULE nv = LoadLibraryA("nvcuda.dll");
    Driver d;
    d.init = reinterpret_cast<decltype(d.init)>(GetProcAddress(nv, "cuInit"));
    d.device_get = reinterpret_cast<decltype(d.device_get)>(GetProcAddress(nv, "cuDeviceGet"));
    d.ctx_create = reinterpret_cast<decltype(d.ctx_create)>(GetProcAddress(nv, "cuCtxCreate_v2"));
    d.ctx_pop = reinterpret_cast<decltype(d.ctx_pop)>(GetProcAddress(nv, "cuCtxPopCurrent_v2"));
    d.ctx_push = reinterpret_cast<decltype(d.ctx_push)>(GetProcAddress(nv, "cuCtxPushCurrent_v2"));
    d.mem_alloc = reinterpret_cast<decltype(d.mem_alloc)>(GetProcAddress(nv, "cuMemAlloc_v2"));
    int device = 0;
    void* context = nullptr;
    if (d.init(0) != 0 || d.device_get(&device, 0) != 0 || d.ctx_create(&context, 0, device) != 0) {
        std::printf("no CUDA context\n");
        return 1;
    }
    std::vector<unsigned long long> slots(32);
    for (auto& slot : slots) {
        d.mem_alloc(&slot, bytes);
    }
    void* popped = nullptr;
    d.ctx_pop(&popped);
    void* stream = fb::gpu::cuda_create_stream(context);

    // Sources as the reader keeps them: many distinct frames in memory.
    const int sources = 256;
    std::vector<std::vector<uint8_t>> ordinary(sources, std::vector<uint8_t>(bytes, 7));
    std::vector<uint8_t*> pinned(sources);
    for (auto& p : pinned) {
        p = static_cast<uint8_t*>(fb::gpu::cuda_host_alloc(context, bytes));
        std::memset(p, 7, bytes);
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (bool use_pinned : {false, true}) {
            const double c0 = cpu_seconds(), w0 = wall_seconds();
            for (int i = 0; i < frames; ++i) {
                const void* src = use_pinned ? static_cast<const void*>(pinned[i % sources]) : ordinary[i % sources].data();
                fb::gpu::cuda_upload(context, stream, reinterpret_cast<void*>(slots[i % slots.size()]), src, bytes);
            }
            const double c_enqueued = cpu_seconds(), w_enqueued = wall_seconds();
            fb::gpu::cuda_sync(context, stream);
            const double w1 = wall_seconds();
            std::printf("%s %-9s: %d frames, CPU %.1f ms per 24 frames, enqueue wall %.1f ms per 24, total wall %.1f ms per 24 (%.1f GB/s)\n",
                        pass ? "measure" : "warm-up", use_pinned ? "pinned" : "ordinary", frames,
                        (c_enqueued - c0) * 1000.0 * 24 / frames, (w_enqueued - w0) * 1000.0 * 24 / frames,
                        (w1 - w0) * 1000.0 * 24 / frames, frames * static_cast<double>(bytes) / (w1 - w0) / 1e9);
        }
    }
    return 0;
}
