// Host-independent planning helpers shared by the OFX and Adobe adapters.
//
// The weights always come from the core (fb_kernel). Adapters only decide
// which source frames the host can deliver: when a fetch fails, the frame is
// treated as beyond the physical source edge, the availability range is
// narrowed and the core recomputes (discard and renormalize).
#pragma once

#include "frame_blend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fb {

inline FbRational rational(int64_t num, int64_t den) {
    return FbRational{num, den};
}

inline double to_double(FbRational value) {
    return static_cast<double>(value.num) / static_cast<double>(value.den);
}

inline std::string text(FbRational value) {
    char buffer[64];
    if (value.den == 1) {
        std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value.num));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%lld/%lld", static_cast<long long>(value.num),
                      static_cast<long long>(value.den));
    }
    return buffer;
}

// Kernel for one output frame, sized by the core's own bound.
inline int32_t kernel(FbRational source_fps, FbRational output_fps, FbRational center,
                      int64_t first_available, int64_t last_available, std::vector<FbTap>& taps) {
    int32_t capacity = 0;
    int32_t status = fb_max_taps(source_fps, output_fps, &capacity);
    if (status != FB_OK) {
        return status;
    }
    taps.resize(static_cast<size_t>(capacity));
    int32_t count = 0;
    status = fb_kernel(source_fps, output_fps, center, first_available, last_available, taps.data(), capacity, &count);
    taps.resize(status == FB_OK ? static_cast<size_t>(count) : 0);
    return status;
}

// Computes the kernel and fetches every tap through `fetch(index) -> bool`.
// A failed fetch marks the edge of the available source: the range is
// narrowed on that side of the centre and the kernel recomputed by the core.
// `fetch` is called at most once per index; the caller caches what it got.
template <class Fetch>
int32_t gather(FbRational source_fps, FbRational output_fps, FbRational center, int64_t first_available,
               int64_t last_available, Fetch&& fetch, std::vector<FbTap>& taps) {
    std::vector<int64_t> tried;
    for (;;) {
        const int32_t status = kernel(source_fps, output_fps, center, first_available, last_available, taps);
        if (status != FB_OK) {
            return status;
        }
        bool narrowed = false;
        for (const FbTap& tap : taps) {
            bool seen = false;
            for (int64_t index : tried) {
                seen = seen || index == tap.index;
            }
            if (seen) {
                continue;
            }
            tried.push_back(tap.index);
            if (fetch(tap.index)) {
                continue;
            }
            // Positions are compared exactly: index * den vs num.
            const bool before_center = static_cast<long double>(tap.index) * center.den < static_cast<long double>(center.num);
            if (before_center) {
                first_available = tap.index + 1;
            } else {
                last_available = tap.index - 1;
            }
            narrowed = true;
            break;
        }
        if (!narrowed) {
            return FB_OK;
        }
        if (first_available > last_available) {
            return FB_ERR_AVAILABILITY;
        }
    }
}

}  // namespace fb
