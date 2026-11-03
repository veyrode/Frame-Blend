// Debug log and image probes; see fb_debug.h.
#include "fb_debug.h"

#include "frame_blend.h"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace fb::debug {
namespace {

std::mutex g_mutex;
FILE* g_file = nullptr;
bool g_tried = false;
char g_path[MAX_PATH * 2] = {};

// Logging is off unless FrameBlend-log.txt already exists beside the plugin
// binary or in %TEMP%: the log is only ever appended to, never created, so a
// release install writes nothing. Looked up once per host session.
FILE* open_existing(const char* path) {
    return GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES ? nullptr : std::fopen(path, "ab");
}

void open_log() {
    if (g_file || g_tried) {
        return;
    }
    g_tried = true;
    HMODULE module = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&open_log), &module);
    char dir[MAX_PATH] = {};
    if (GetModuleFileNameA(module, dir, MAX_PATH)) {
        char* slash = std::strrchr(dir, '\\');
        if (slash) {
            slash[1] = '\0';
            std::snprintf(g_path, sizeof(g_path), "%sFrameBlend-log.txt", dir);
            g_file = open_existing(g_path);
        }
    }
    if (!g_file) {
        char temp[MAX_PATH] = {};
        GetTempPathA(MAX_PATH, temp);
        std::snprintf(g_path, sizeof(g_path), "%sFrameBlend-log.txt", temp);
        g_file = open_existing(g_path);
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    const int exponent = (h >> 10) & 31;
    const uint32_t mantissa = h & 1023;
    float v = 0.0f;
    if (exponent == 0) {
        v = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        v = mantissa ? NAN : INFINITY;
    } else {
        v = std::ldexp(static_cast<float>(mantissa | 1024), exponent - 25);
    }
    return sign ? -v : v;
}

// Round-to-nearest-even float -> half, as __float2half_rn / vstore_half_rte.
uint16_t float_to_half(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const uint32_t abs = x & 0x7fffffff;
    if (abs >= 0x7f800000) {
        return static_cast<uint16_t>(sign | 0x7c00 | (abs > 0x7f800000 ? 0x200 : 0));
    }
    if (abs >= 0x477ff000) {  // rounds to >= 65520: overflow
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    if (abs < 0x38800000) {  // subnormal half
        const float scaled = std::fabs(f) * 16777216.0f;  // 2^24
        return static_cast<uint16_t>(sign | static_cast<uint32_t>(std::nearbyint(scaled)));
    }
    uint32_t mantissa = abs & 0x7fffff;
    uint32_t exponent = (abs >> 23) - 112;
    uint32_t half = (exponent << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1))) {
        ++half;
    }
    return static_cast<uint16_t>(sign | half);
}

float value_at(const uint8_t* row, int index, Element element) {
    switch (element) {
        case Element::F32: {
            float v = 0.0f;
            std::memcpy(&v, row + static_cast<size_t>(index) * 4, 4);
            return v;
        }
        case Element::U8:
            return row[index] / 255.0f;
        case Element::F16: {
            uint16_t h = 0;
            std::memcpy(&h, row + static_cast<size_t>(index) * 2, 2);
            return half_to_float(h);
        }
    }
    return 0.0f;
}

// Code cells: 16 squares of 120 px across a 1920 px frame, centres 60 px from the top.
int read_code(const uint8_t* row, int width, Element element) {
    int code = 0;
    for (int bit = 0; bit < 16; ++bit) {
        const int x = std::min(width - 1, static_cast<int>(std::lround((120.0 * bit + 60.0) / 1920.0 * width)));
        float sum = 0.0f;
        for (int c = 0; c < 3; ++c) {
            sum += value_at(row, x * 4 + c, element);
        }
        if (sum / 3.0f > 0.5f) {
            code |= 1 << bit;
        }
    }
    return code;
}

}  // namespace

size_t element_bytes(Element element) {
    return element == Element::F32 ? 4 : element == Element::F16 ? 2 : 1;
}

void line(const char* format, ...) {
    std::lock_guard<std::mutex> lock(g_mutex);
    open_log();
    if (!g_file) {
        return;
    }
    va_list args;
    va_start(args, format);
    std::vfprintf(g_file, format, args);
    va_end(args);
    std::fputc('\n', g_file);
    // Written through at once: when a host crashes, the last line shows where.
    std::fflush(g_file);
}

void flush() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_file) {
        std::fflush(g_file);
    }
}

bool enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    open_log();
    return g_file != nullptr;
}

const char* log_path() {
    std::lock_guard<std::mutex> lock(g_mutex);
    open_log();
    return g_path;
}

double now_ms() {
    static const double frequency = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return 1000.0 * static_cast<double>(counter.QuadPart) / frequency;
}

std::vector<int> sample_rows(int height) {
    const int code_row = std::min(height - 1, static_cast<int>(std::lround(60.0 / 1080.0 * height)));
    std::vector<int> rows = {code_row, height - 1 - code_row};
    for (int i = 0; i < 8; ++i) {
        rows.push_back(static_cast<int>((static_cast<int64_t>(2 * i + 1) * height) / 16));
    }
    return rows;
}

Sample sample_host(const void* row0, int64_t pitch, int width, int height, Element element) {
    Sample s;
    s.row_bytes = static_cast<int64_t>(width) * 4 * static_cast<int64_t>(element_bytes(element));
    const std::vector<int> rows = sample_rows(height);
    s.bytes.resize(static_cast<size_t>(s.row_bytes) * rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        std::memcpy(s.bytes.data() + i * s.row_bytes, static_cast<const uint8_t*>(row0) + rows[i] * pitch,
                    static_cast<size_t>(s.row_bytes));
    }
    s.valid = true;
    return s;
}

Probe analyse(const Sample& sample, int width, Element element) {
    Probe p;
    if (!sample.valid || sample.bytes.size() < static_cast<size_t>(sample.row_bytes) * 2) {
        return p;
    }
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t b : sample.bytes) {
        hash = (hash ^ b) * 1099511628211ULL;
    }
    p.checksum = hash;
    p.code_top = read_code(sample.bytes.data(), width, element);
    p.code_bottom = read_code(sample.bytes.data() + sample.row_bytes, width, element);
    return p;
}

Parity parity(const std::vector<const Sample*>& sources, const std::vector<float>& weights, const Sample& output,
              int width, Element element) {
    Parity result;
    if (!output.valid || sources.empty() || sources.size() != weights.size()) {
        return result;
    }
    const int64_t row_elements = static_cast<int64_t>(width) * 4;
    const int64_t rows = static_cast<int64_t>(output.bytes.size()) / output.row_bytes;
    const int64_t n = row_elements * rows;
    // Sources as contiguous float rows for the core's CPU path.
    std::vector<std::vector<float>> floats(sources.size(), std::vector<float>(static_cast<size_t>(n)));
    std::vector<const float*> planes;
    std::vector<ptrdiff_t> strides;
    for (size_t k = 0; k < sources.size(); ++k) {
        if (!sources[k]->valid) {
            return result;
        }
        for (int64_t i = 0; i < n; ++i) {
            const int64_t r = i / row_elements;
            floats[k][static_cast<size_t>(i)] =
                value_at(sources[k]->bytes.data() + r * sources[k]->row_bytes, static_cast<int>(i % row_elements), element);
        }
        planes.push_back(floats[k].data());
        strides.push_back(static_cast<ptrdiff_t>(row_elements * 4));
    }
    std::vector<float> cpu(static_cast<size_t>(n));
    if (fb_apply_f32(cpu.data(), static_cast<ptrdiff_t>(row_elements * 4), planes.data(), strides.data(), weights.data(),
                     static_cast<int32_t>(planes.size()), static_cast<int32_t>(row_elements),
                     static_cast<int32_t>(rows)) != FB_OK) {
        return result;
    }
    result.max_abs = 0.0;
    result.max_abs_after_rounding = 0.0;
    result.elements = n;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = i / row_elements;
        const float gpu = value_at(output.bytes.data() + r * output.row_bytes, static_cast<int>(i % row_elements), element);
        const float reference = cpu[static_cast<size_t>(i)];
        const float rounded = element == Element::F16 ? half_to_float(float_to_half(reference)) : reference;
        result.max_abs = std::max(result.max_abs, static_cast<double>(std::fabs(gpu - reference)));
        const double after = std::fabs(static_cast<double>(gpu) - rounded);
        result.max_abs_after_rounding = std::max(result.max_abs_after_rounding, after);
        result.differing += after != 0.0;
    }
    return result;
}

}  // namespace fb::debug
