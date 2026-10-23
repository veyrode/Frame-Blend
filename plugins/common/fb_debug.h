// Debug log and image probes used by the host frame-access and timing tests.
//
// Lines go to FrameBlend-log.txt next to the plugin binary, or in %TEMP%, but
// only when that file already exists (create an empty one to turn logging on;
// tools/install_plugins.ps1 does for development installs).
//
// Each image is probed on a few sampled rows only, so GPU read-back stays
// cheap: the rows holding the 16-bit frame code
// burned into the synthetic clips (tools/make_numbered_clips.py) under both
// row orders (first memory row = top, or = bottom), plus 8 evenly spaced rows.
// The checksum is 64-bit FNV-1a over all sampled bytes.
#pragma once

#include <cstdint>
#include <vector>

namespace fb::debug {

enum class Element { F32, F16, U8 };

size_t element_bytes(Element element);

// printf-style; appends a newline. Thread safe.
void line(const char* format, ...);
void flush();
// Whether lines are written (the log file exists).
bool enabled();
const char* log_path();

// Milliseconds from an arbitrary fixed origin (QueryPerformanceCounter).
double now_ms();

// Memory-order row indices sampled in an image of `height` rows:
// [0] code row if row 0 is the top, [1] code row if row 0 is the bottom,
// [2..9] evenly spaced rows.
std::vector<int> sample_rows(int height);

// Sampled rows of one image, concatenated in sample_rows() order.
struct Sample {
    std::vector<uint8_t> bytes;
    int64_t row_bytes = 0;
    bool valid = false;
};

Sample sample_host(const void* row0, int64_t pitch, int width, int height, Element element);

struct Probe {
    uint64_t checksum = 0;
    int code_top = -1;
    int code_bottom = -1;
};

Probe analyse(const Sample& sample, int width, Element element);

// Largest |gpu - cpu| over the sampled rows, where cpu is the core's CPU
// blend (fb_apply_f32) of the sampled source rows. For half output the CPU
// value is compared both directly and after round-to-nearest-even to half
// (`max_abs_after_rounding`). `differing` counts elements with any difference
// after that rounding.
struct Parity {
    double max_abs = -1.0;
    double max_abs_after_rounding = -1.0;
    int64_t differing = 0;
    int64_t elements = 0;
};

Parity parity(const std::vector<const Sample*>& sources, const std::vector<float>& weights, const Sample& output,
              int width, Element element);

}  // namespace fb::debug
