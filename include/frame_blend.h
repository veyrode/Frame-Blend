// C interface of the shared frame-blending core (src/lib.rs).
//
// The core owns the kernel: support, exact rational weights and edge
// renormalization. Adapters fetch source frames and apply the returned
// weights; they never compute weights themselves.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FB_OK = 0,
    FB_ERR_BAD_ARGS = 1,
    FB_ERR_RATE = 2,
    FB_ERR_AVAILABILITY = 3,
    FB_ERR_CAPACITY = 4,
    FB_ERR_OVERFLOW = 5
};

typedef struct FbRational {
    int64_t num;
    int64_t den;
} FbRational;

typedef struct FbTap {
    int64_t index;      // absolute source frame index
    int64_t weight_num; // exact weight = weight_num / weight_den
    int64_t weight_den;
    float weight;       // the exact weight rounded to float, applied to pixels
} FbTap;

// Exact rate behind a floating-point host rate (29.97002997 -> 30000/1001).
int32_t fb_rational_from_fps(double fps, FbRational* out);

// Tap capacity fb_kernel may need for these rates.
int32_t fb_max_taps(FbRational source_fps, FbRational output_fps, int32_t* out);

// origin + output_frame * (source_fps / output_fps)
int32_t fb_center_for_output_frame(FbRational source_fps, FbRational output_fps,
                                   FbRational source_origin, FbRational output_frame,
                                   FbRational* out);

// (source_frame - origin) / (source_fps / output_fps), in output frames.
int32_t fb_output_position_of_source_frame(FbRational source_fps, FbRational output_fps,
                                           FbRational source_origin, int64_t source_frame,
                                           FbRational* out);

// Sample-cell overlap kernel for the output frame centred on source position
// `center`. Frames outside [first_available, last_available] are discarded and
// the remaining weights renormalized. Pass-through (one tap, weight 1) when
// output_fps >= source_fps.
int32_t fb_kernel(FbRational source_fps, FbRational output_fps, FbRational center,
                  int64_t first_available, int64_t last_available,
                  FbTap* taps, int32_t capacity, int32_t* count);

// CPU fallback: dst = sum_k weights[k] * srcs[k], f32 accumulation in tap
// order with separately rounded multiply and add. Row strides are in bytes.
int32_t fb_apply_f32(float* dst, ptrdiff_t dst_row_bytes,
                     const float* const* srcs, const ptrdiff_t* src_row_bytes,
                     const float* weights, int32_t count,
                     int32_t row_elements, int32_t rows);

int32_t fb_apply_u8(uint8_t* dst, ptrdiff_t dst_row_bytes,
                    const uint8_t* const* srcs, const ptrdiff_t* src_row_bytes,
                    const float* weights, int32_t count,
                    int32_t row_elements, int32_t rows);

// 16-bit variant (After Effects 16 bpc): codes are rounded and clamped to
// 0..max (32768 there).
int32_t fb_apply_u16(uint16_t* dst, ptrdiff_t dst_row_bytes,
                     const uint16_t* const* srcs, const ptrdiff_t* src_row_bytes,
                     const float* weights, int32_t count,
                     int32_t row_elements, int32_t rows, uint16_t max);

const char* fb_error_message(int32_t code);

#ifdef __cplusplus
}
#endif
