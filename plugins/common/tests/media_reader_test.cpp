// Frame accuracy and speed of fb_media_reader on the numbered test clips
// (tools/make_numbered_clips.py): every decoded frame must carry its own index
// in the 16-cell code band, for sequential reads and after seeks.
//
//   media_reader_test [--nvdec | --pinned] [--limit N] <clip> [more clips...]
//
// --nvdec decodes into a CUDA context created here (as Premiere's would be);
// --pinned decodes in software into page-locked memory of such a context;
// --limit N reads only the first N frames and skips the code checks (speed
// measurements on ordinary recordings); --vfr checks that every index of a
// variable frame rate recording shows a frame (its own or a held earlier one).
//
//   media_reader_test [--pinned] --budget <clip> [more clips...]
//
// opens every clip at once, as a project opening in Premiere does, and samples the memory the readers hold; then plays the first clip,
// lets its decoder close while idle and reads on after a seek (numbered clips:
// every frame's code is checked), then plays the second clip.

#define NOMINMAX
#include <windows.h>

#include <psapi.h>

#include "fb_gpu.h"
#include "fb_media_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

void* g_context = nullptr;
void* g_pinned = nullptr;

// Code from luma at the cell centres (white cells: luma well above mid grey).
int code_of(const fb::media::Frame& frame, int width, int height) {
    const int y = static_cast<int>(std::lround(60.0 / 1080.0 * height));
    std::vector<uint8_t> row(static_cast<size_t>(width));
    if (frame.on_device()) {
        if (!fb::gpu::cuda_download(g_context, frame.luma + static_cast<unsigned long long>(y) * frame.pitch, frame.pitch,
                                    width, 1, row.data())) {
            return -1;
        }
    } else {
        std::copy_n(frame.nv12.data() + static_cast<size_t>(y) * width, width, row.data());
    }
    int code = 0;
    for (int bit = 0; bit < 16; ++bit) {
        const int x = static_cast<int>(std::lround((120.0 * bit + 60.0) / 1920.0 * width));
        if (row[x] > 128) {
            code |= 1 << bit;
        }
    }
    return code;
}

void* create_cuda_context() {
    HMODULE nv = LoadLibraryA("nvcuda.dll");
    if (!nv) {
        return nullptr;
    }
    using Init = int(__stdcall*)(unsigned);
    using DeviceGet = int(__stdcall*)(int*, int);
    using CtxCreate = int(__stdcall*)(void**, unsigned, int);
    using CtxPop = int(__stdcall*)(void**);
    auto init = reinterpret_cast<Init>(GetProcAddress(nv, "cuInit"));
    auto device_get = reinterpret_cast<DeviceGet>(GetProcAddress(nv, "cuDeviceGet"));
    auto create = reinterpret_cast<CtxCreate>(GetProcAddress(nv, "cuCtxCreate_v2"));
    auto pop = reinterpret_cast<CtxPop>(GetProcAddress(nv, "cuCtxPopCurrent_v2"));
    int device = 0;
    void* context = nullptr;
    if (!init || init(0) != 0 || device_get(&device, 0) != 0 || create(&context, 0, device) != 0) {
        return nullptr;
    }
    void* popped = nullptr;
    pop(&popped);  // like a host: the context is pushed only around our calls
    return context;
}

long long private_mb() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters));
    return static_cast<long long>(counters.PrivateUsage >> 20);
}

void memory_line(const char* what, double t0) {
    std::printf("  %6.1f s  %-22s process private=%lld MB  readers hold=%lld MB (budget %lld MB)\n", (now_ms() - t0) / 1000.0,
                what, private_mb(), static_cast<long long>(fb::media::frame_memory_bytes() >> 20),
                static_cast<long long>(fb::media::frame_memory_budget() >> 20));
}

// Plays `count` frames from `start`; numbered clips also check each code.
int play(fb::media::Reader& reader, bool numbered, int64_t start, int64_t count, const char* what) {
    const fb::media::Info& info = reader.info();
    count = std::min(count, info.frames - start);
    int wrong = 0;
    const double t = now_ms();
    for (int64_t i = start; i < start + count; ++i) {
        auto frame = reader.get(i);
        const bool bad = !frame || frame->index > i || frame->index < i - 1000 ||
                         (numbered && code_of(*frame, info.width, info.height) != static_cast<int>(i));
        if (bad && wrong++ < 5) {
            std::printf("    %s frame %lld: %s\n", what, static_cast<long long>(i),
                        frame ? std::to_string(frame->index).c_str() : "missing");
        }
    }
    std::printf("    %s: frames %lld-%lld, %d wrong, %.0f frames/s; %s\n", what, static_cast<long long>(start),
                static_cast<long long>(start + count - 1), wrong, 1000.0 * static_cast<double>(count) / (now_ms() - t),
                reader.stats().c_str());
    return wrong;
}

int budget_test(const std::vector<std::wstring>& paths) {
    const double t0 = now_ms();
    memory_line("start", t0);
    std::vector<std::shared_ptr<fb::media::Reader>> readers;
    std::vector<bool> numbered;
    for (const std::wstring& path : paths) {
        std::string error;
        auto reader = fb::media::Reader::open(path, g_context, g_pinned, error);
        if (!reader) {
            std::printf("  %ls: open failed: %s\n", path.c_str(), error.c_str());
            return 1;
        }
        readers.push_back(reader);
        numbered.push_back(path.find(L"numbered") != std::wstring::npos);
    }
    std::printf("  %zu readers opened (nothing decoded until asked)\n", readers.size());
    for (int s = 0; s < 12; ++s) {
        memory_line(s == 0 ? "opened" : "", t0);
        Sleep(500);
    }
    int wrong = play(*readers[0], numbered[0], 0, 3000, "play clip 1");
    memory_line("after playing clip 1", t0);
    Sleep(4000);  // past the idle time that closes the decoders
    memory_line("idle 4 s", t0);
    const int64_t later = std::max<int64_t>(0, readers[0]->info().frames - 400);
    wrong += play(*readers[0], numbered[0], later, 300, "clip 1 after its decoder closed");
    // More clips played one after the other than the budget holds: the idle
    // readers (those played earlier) give memory back.
    for (size_t k = 1; k < std::min<size_t>(readers.size(), 7); ++k) {
        const std::string what = "play clip " + std::to_string(k + 1);
        wrong += play(*readers[k], numbered[k], 0, 3000, what.c_str());
        memory_line(("after " + what).c_str(), t0);
    }
    if (readers.size() > 1) {
        wrong += play(*readers[0], numbered[0], 100, 300, "back to clip 1");
    }
    memory_line("end", t0);
    readers.clear();
    memory_line("readers closed", t0);
    const bool freed = fb::media::frame_memory_bytes() == 0;
    std::printf("  frame memory all freed: %s\n%s\n", freed ? "yes" : "NO", wrong || !freed ? "FAILED" : "budget test passed");
    return wrong || !freed ? 1 : 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    int failures = 0;
    int64_t limit = -1;
    bool vfr = false;
    bool budget = false;
    std::vector<std::wstring> budget_paths;
    for (int a = 1; a < argc; ++a) {
        const std::wstring arg = argv[a];
        if (arg == L"--budget") {
            budget = true;
            continue;
        }
        if (budget && arg.rfind(L"--", 0) != 0) {
            budget_paths.push_back(arg);
            continue;
        }
        if (arg == L"--pinned") {
            g_pinned = create_cuda_context();
            std::printf("CUDA context for page-locked memory: %s\n", g_pinned ? "created" : "FAILED");
            continue;
        }
        if (arg == L"--nvdec") {
            g_context = create_cuda_context();
            std::printf("CUDA context for NVDEC: %s\n", g_context ? "created" : "FAILED");
            continue;
        }
        if (arg == L"--vfr") {
            vfr = true;
            continue;
        }
        if (arg == L"--limit" && a + 1 < argc) {
            limit = _wtoi64(argv[++a]);
            continue;
        }
        std::string error;
        const double open_start = now_ms();
        auto reader = fb::media::Reader::open(arg, g_context, g_pinned, error);
        std::printf("  open_ms=%.1f (FFmpeg load, probe, first frame)\n", now_ms() - open_start);
        if (!reader) {
            std::printf("%ls: open failed: %s\n", argv[a], error.c_str());
            ++failures;
            continue;
        }
        const fb::media::Info& info = reader->info();
        if (vfr) {
            // Every index up to the end shows a frame: its own, or the one
            // before a gap in the recording (held), never a later one.
            int64_t held = 0, bad = 0, previous = -1, end = -1;
            const double t0 = now_ms();
            for (int64_t i = 0; i < info.frames; ++i) {
                auto frame = reader->get(i);
                if (!frame) {
                    end = i;
                    break;
                }
                const bool wrong = frame->index > i || frame->index < previous;
                if (wrong && bad++ < 5) {
                    std::printf("  index %lld: frame %lld\n", static_cast<long long>(i), static_cast<long long>(frame->index));
                }
                held += frame->index != i;
                previous = frame->index;
            }
            std::printf("%ls\n  fps=%lld/%lld frames=%lld: %lld indices read in %.0f ms, %lld held, %lld wrong, end at %lld; %s\n",
                        argv[a], static_cast<long long>(info.fps.num), static_cast<long long>(info.fps.den),
                        static_cast<long long>(info.frames), static_cast<long long>(end < 0 ? info.frames : end),
                        now_ms() - t0, static_cast<long long>(held), static_cast<long long>(bad),
                        static_cast<long long>(end), reader->stats().c_str());
            // Scrubbing: 25-index windows at random places, after seeks.
            std::mt19937 rng(7);
            const int64_t last = (end < 0 ? info.frames : end) - 30;
            std::uniform_int_distribution<int64_t> pick(0, std::max<int64_t>(0, last));
            int64_t window_bad = 0;
            for (int s = 0; s < 40; ++s) {
                const int64_t start = pick(rng);
                for (int64_t i = start; i < start + 25; ++i) {
                    auto frame = reader->get(i);
                    const bool wrong = !frame || frame->index > i || frame->index < i - 1000;
                    if (wrong && window_bad++ < 5) {
                        std::printf("  window index %lld: %s; %s\n", static_cast<long long>(i),
                                    frame ? std::to_string(frame->index).c_str() : "missing", reader->stats().c_str());
                    }
                }
            }
            std::printf("  40 random 25-index windows: %lld wrong; %s\n", static_cast<long long>(window_bad),
                        reader->stats().c_str());
            failures += static_cast<int>(bad + window_bad) + (end >= 0 && end < info.frames - 64 ? 1 : 0);
            continue;
        }
        const int64_t count = limit > 0 ? std::min(limit, info.frames) : info.frames;
        int wrong = 0;
        const double t0 = now_ms();
        for (int64_t i = 0; i < count; ++i) {
            auto frame = reader->get(i);
            const bool ok = frame && frame->index == i && (limit > 0 || code_of(*frame, info.width, info.height) == i);
            if (!ok && wrong++ < 5) {
                std::printf("  frame %lld: %s\n", static_cast<long long>(i),
                            frame ? ("code " + std::to_string(code_of(*frame, info.width, info.height))).c_str() : "missing");
            }
        }
        const double seq_ms = now_ms() - t0;
        std::printf("%ls\n  %dx%d fps=%lld/%lld frames=%lld %s %s %s %s\n", argv[a], info.width, info.height,
                    static_cast<long long>(info.fps.num), static_cast<long long>(info.fps.den),
                    static_cast<long long>(info.frames), info.codec.c_str(), info.full_range ? "full" : "limited",
                    info.bt601 ? "bt601" : "bt709", info.nvdec ? "nvdec" : "software");
        std::printf("  sequential: %lld frames, %d wrong, %.1f frames/s\n", static_cast<long long>(count), wrong,
                    1000.0 * static_cast<double>(count) / seq_ms);
        failures += wrong;
        if (limit > 0) {
            continue;
        }
        // Random windows like scrubbing: 25 consecutive frames at random places.
        std::mt19937 rng(7);
        std::uniform_int_distribution<int64_t> pick(0, std::max<int64_t>(0, info.frames - 30));
        int seek_wrong = 0;
        const double t1 = now_ms();
        for (int s = 0; s < 20; ++s) {
            const int64_t start = pick(rng);
            for (int64_t i = start; i < start + 25; ++i) {
                auto frame = reader->get(i);
                const bool bad = !frame || code_of(*frame, info.width, info.height) != static_cast<int>(i);
                if (bad && seek_wrong < 5) {
                    std::printf("  window frame %lld: %s; %s\n", static_cast<long long>(i),
                                frame ? ("code " + std::to_string(code_of(*frame, info.width, info.height))).c_str() : "missing",
                                reader->stats().c_str());
                }
                seek_wrong += bad;
            }
        }
        std::printf("  20 random 25-frame windows: %d wrong, %.1f ms per window; %s\n", seek_wrong,
                    (now_ms() - t1) / 20.0, reader->stats().c_str());
        // Reverse playback (a reversed clip): 25-frame windows, each starting
        // 24 frames below the previous one, from the end of the file.
        int reverse_wrong = 0;
        int windows = 0;
        const double t2 = now_ms();
        for (int64_t start = info.frames - 25; start >= 0 && windows < 120; start -= 24, ++windows) {
            for (int64_t i = start; i < start + 25; ++i) {
                auto frame = reader->get(i);
                const bool bad = !frame || code_of(*frame, info.width, info.height) != static_cast<int>(i);
                if (bad && reverse_wrong < 5) {
                    std::printf("  reverse frame %lld: %s; %s\n", static_cast<long long>(i),
                                frame ? ("code " + std::to_string(code_of(*frame, info.width, info.height))).c_str() : "missing",
                                reader->stats().c_str());
                }
                reverse_wrong += bad;
            }
        }
        std::printf("  %d reverse 25-frame windows: %d wrong, %.1f ms per window; %s\n", windows, reverse_wrong,
                    (now_ms() - t2) / std::max(1, windows), reader->stats().c_str());
        seek_wrong += reverse_wrong;
        const bool end_ok = !reader->get(info.frames);
        std::printf("  frame %lld (past the end) refused: %s\n", static_cast<long long>(info.frames), end_ok ? "yes" : "NO");
        failures += seek_wrong + (end_ok ? 0 : 1);
    }
    if (budget) {
        return budget_paths.empty() ? 1 : budget_test(budget_paths);
    }
    std::printf("%s\n", failures ? "FAILED" : "all frames decoded at their own index");
    return failures ? 1 : 0;
}
