// CUDA weight application. Compiled to PTX at build time and loaded through
// the driver API, so the plugins carry no CUDA runtime dependency.
//
// Each thread computes one element: acc = sum_k weight[k] * src[k], in tap
// order, with separately rounded multiply and add (__fmul_rn/__fadd_rn), the
// same arithmetic as the core's CPU path (src/blend.rs). Weights come from the
// core; this code never computes them.

#include <cuda_fp16.h>

#define FB_MAX_TAPS 128

struct FbTaps {
    unsigned long long src[FB_MAX_TAPS];  // device address of row 0 of each source
    long long pitch[FB_MAX_TAPS];         // bytes between rows
    float weight[FB_MAX_TAPS];
};

extern "C" __global__ void fb_blend_f32(unsigned long long dst, long long dst_pitch, int row_elements, int rows,
                                        int count, FbTaps taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row_elements || y >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) {
        const float* row = reinterpret_cast<const float*>(taps.src[k] + static_cast<long long>(y) * taps.pitch[k]);
        acc = __fadd_rn(acc, __fmul_rn(taps.weight[k], row[x]));
    }
    reinterpret_cast<float*>(dst + static_cast<long long>(y) * dst_pitch)[x] = acc;
}

extern "C" __global__ void fb_blend_f16(unsigned long long dst, long long dst_pitch, int row_elements, int rows,
                                        int count, FbTaps taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row_elements || y >= rows) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) {
        const __half* row = reinterpret_cast<const __half*>(taps.src[k] + static_cast<long long>(y) * taps.pitch[k]);
        acc = __fadd_rn(acc, __fmul_rn(taps.weight[k], __half2float(row[x])));
    }
    reinterpret_cast<__half*>(dst + static_cast<long long>(y) * dst_pitch)[x] = __float2half_rn(acc);
}

// Vectorised variants: one thread per pixel (4 elements), used when every
// row start and pitch is aligned to the vector size. Same arithmetic per element.
extern "C" __global__ void fb_blend_f32x4(unsigned long long dst, long long dst_pitch, int row_pixels, int rows,
                                          int count, FbTaps taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row_pixels || y >= rows) {
        return;
    }
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = 0; k < count; ++k) {
        const float4 v = reinterpret_cast<const float4*>(taps.src[k] + static_cast<long long>(y) * taps.pitch[k])[x];
        const float w = taps.weight[k];
        acc.x = __fadd_rn(acc.x, __fmul_rn(w, v.x));
        acc.y = __fadd_rn(acc.y, __fmul_rn(w, v.y));
        acc.z = __fadd_rn(acc.z, __fmul_rn(w, v.z));
        acc.w = __fadd_rn(acc.w, __fmul_rn(w, v.w));
    }
    reinterpret_cast<float4*>(dst + static_cast<long long>(y) * dst_pitch)[x] = acc;
}

extern "C" __global__ void fb_blend_f16x4(unsigned long long dst, long long dst_pitch, int row_pixels, int rows,
                                          int count, FbTaps taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= row_pixels || y >= rows) {
        return;
    }
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = 0; k < count; ++k) {
        const uint2 raw = reinterpret_cast<const uint2*>(taps.src[k] + static_cast<long long>(y) * taps.pitch[k])[x];
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
        const float w = taps.weight[k];
        a0 = __fadd_rn(a0, __fmul_rn(w, lo.x));
        a1 = __fadd_rn(a1, __fmul_rn(w, lo.y));
        a2 = __fadd_rn(a2, __fmul_rn(w, hi.x));
        a3 = __fadd_rn(a3, __fmul_rn(w, hi.y));
    }
    __half2 lo = __floats2half2_rn(a0, a1);
    __half2 hi = __floats2half2_rn(a2, a3);
    uint2 out;
    out.x = *reinterpret_cast<unsigned int*>(&lo);
    out.y = *reinterpret_cast<unsigned int*>(&hi);
    reinterpret_cast<uint2*>(dst + static_cast<long long>(y) * dst_pitch)[x] = out;
}

// ---- Blending straight from decoded 8-bit NV12 frames (Premiere file path) ----
//
// Each tap is one NV12 frame: `src_h` rows of luma and `src_h / 2` rows of
// interleaved UV, both with pitch `pitch`, at independent addresses (decoder
// surfaces) or with UV directly after the luma rows (uploaded frames).
// Every tap is converted to RGB first (BT.709 or BT.601, full or limited
// range, chroma upsampled bilinearly with H.264's left siting), then the
// weights are applied in tap order exactly as in fb_blend_f32. Output is BGRA,
// alpha 1; a smaller output samples the source at the nearest position.

struct FbNv12Taps {
    unsigned long long luma[FB_MAX_TAPS];
    unsigned long long chroma[FB_MAX_TAPS];
    long long pitch[FB_MAX_TAPS];
    float weight[FB_MAX_TAPS];
};

struct FbYuv {
    float y_scale, y_offset, c_scale;
    float r_cr, g_cb, g_cr, b_cb;
};

__device__ __forceinline__ float fb_chroma(const unsigned char* uv, long long pitch, int cw, int ch, float cx, float cy,
                                           int component) {
    const int x0 = max(0, min(cw - 1, static_cast<int>(floorf(cx))));
    const int y0 = max(0, min(ch - 1, static_cast<int>(floorf(cy))));
    const int x1 = min(cw - 1, x0 + 1);
    const int y1 = min(ch - 1, y0 + 1);
    const float fx = fminf(fmaxf(cx - floorf(cx), 0.0f), 1.0f);
    const float fy = fminf(fmaxf(cy - floorf(cy), 0.0f), 1.0f);
    const float a = uv[y0 * pitch + 2 * x0 + component];
    const float b = uv[y0 * pitch + 2 * x1 + component];
    const float c = uv[y1 * pitch + 2 * x0 + component];
    const float d = uv[y1 * pitch + 2 * x1 + component];
    const float top = __fadd_rn(a, __fmul_rn(fx, __fsub_rn(b, a)));
    const float bottom = __fadd_rn(c, __fmul_rn(fx, __fsub_rn(d, c)));
    return __fadd_rn(top, __fmul_rn(fy, __fsub_rn(bottom, top)));
}

__device__ __forceinline__ float3 fb_nv12_rgb(unsigned long long luma_base, unsigned long long chroma_base,
                                              long long pitch, int src_w, int src_h, int sx, int sy, const FbYuv& k) {
    const unsigned char* luma = reinterpret_cast<const unsigned char*>(luma_base);
    const unsigned char* uv = reinterpret_cast<const unsigned char*>(chroma_base);
    const float cx = 0.5f * sx;                 // left siting: chroma column i sits on luma column 2i
    const float cy = 0.5f * sy - 0.25f;         // vertically centred between luma rows 2j and 2j+1
    const float y = __fmul_rn(__fsub_rn(static_cast<float>(luma[sy * pitch + sx]), k.y_offset), k.y_scale);
    const float cb = __fmul_rn(__fsub_rn(fb_chroma(uv, pitch, src_w / 2, src_h / 2, cx, cy, 0), 128.0f), k.c_scale);
    const float cr = __fmul_rn(__fsub_rn(fb_chroma(uv, pitch, src_w / 2, src_h / 2, cx, cy, 1), 128.0f), k.c_scale);
    float3 rgb;
    rgb.x = __fadd_rn(y, __fmul_rn(k.r_cr, cr));
    rgb.y = __fsub_rn(__fsub_rn(y, __fmul_rn(k.g_cb, cb)), __fmul_rn(k.g_cr, cr));
    rgb.z = __fadd_rn(y, __fmul_rn(k.b_cb, cb));
    return rgb;
}

template <bool kHalf>
__device__ void fb_blend_nv12(unsigned long long dst, long long dst_pitch, int width, int height, int count,
                              int src_w, int src_h, FbYuv k, const FbNv12Taps& taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const int sx = width == src_w ? x : min(src_w - 1, static_cast<int>((static_cast<long long>(x) * src_w) / width));
    const int sy = height == src_h ? y : min(src_h - 1, static_cast<int>((static_cast<long long>(y) * src_h) / height));
    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int t = 0; t < count; ++t) {
        const float3 rgb = fb_nv12_rgb(taps.luma[t], taps.chroma[t], taps.pitch[t], src_w, src_h, sx, sy, k);
        const float w = taps.weight[t];
        r = __fadd_rn(r, __fmul_rn(w, rgb.x));
        g = __fadd_rn(g, __fmul_rn(w, rgb.y));
        b = __fadd_rn(b, __fmul_rn(w, rgb.z));
    }
    if (kHalf) {
        __half* out = reinterpret_cast<__half*>(dst + static_cast<long long>(y) * dst_pitch) + 4 * x;
        out[0] = __float2half_rn(b);
        out[1] = __float2half_rn(g);
        out[2] = __float2half_rn(r);
        out[3] = __float2half_rn(1.0f);
    } else {
        float* out = reinterpret_cast<float*>(dst + static_cast<long long>(y) * dst_pitch) + 4 * x;
        out[0] = b;
        out[1] = g;
        out[2] = r;
        out[3] = 1.0f;
    }
}

extern "C" __global__ void fb_blend_nv12_f32(unsigned long long dst, long long dst_pitch, int width, int height,
                                             int count, int src_w, int src_h, FbYuv k, FbNv12Taps taps) {
    fb_blend_nv12<false>(dst, dst_pitch, width, height, count, src_w, src_h, k, taps);
}

extern "C" __global__ void fb_blend_nv12_f16(unsigned long long dst, long long dst_pitch, int width, int height,
                                             int count, int src_w, int src_h, FbYuv k, FbNv12Taps taps) {
    fb_blend_nv12<true>(dst, dst_pitch, width, height, count, src_w, src_h, k, taps);
}
