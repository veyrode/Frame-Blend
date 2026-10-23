// OpenCL weight application, compiled at run time on the host's context.
//
// Each work-item computes one RGBA pixel: acc = sum_k weight[k] * src[k], in
// tap order. Contraction is disabled so every multiply and add is rounded
// separately, matching the core's CPU path (src/blend.rs) element by element.
//
// OpenCL cannot receive an array of buffers, so each launch applies up to
// FB_CL_PASS taps passed as separate arguments; longer kernels run several
// passes that carry the float accumulator in memory (bit-identical to keeping
// it in a register).

#pragma OPENCL FP_CONTRACT OFF

// 32 taps x (pointer + 3 x 32-bit) plus the fixed arguments stay below the
// 1024-byte parameter size every OpenCL device must support.
#define FB_CL_PASS 32

#define TAP_ARGS(i) , __global const uchar* s##i, int o##i, int p##i, float w##i

inline float4 load_pixel(__global const uchar* base, int offset, int pitch, int x, int y, int half_src) {
    __global const uchar* row = base + offset + (long)y * pitch;
    return half_src ? vload_half4(x, (__global const half*)row) : vload4(x, (__global const float*)row);
}

#define TAP(i) if (i < count) { acc = acc + w##i * load_pixel(s##i, o##i, p##i, x, y, half_src); }

// acc_in: optional float accumulator from a previous pass (same layout as the
// float destination of that pass). dst receives float (dst_half = 0) or half.
__kernel void fb_blend(__global uchar* dst, int dst_offset, int dst_pitch, int dst_half,
                       __global const uchar* acc_in, int acc_offset, int acc_pitch, int use_acc,
                       int row_pixels, int rows, int count, int half_src
                       TAP_ARGS(0) TAP_ARGS(1) TAP_ARGS(2) TAP_ARGS(3) TAP_ARGS(4) TAP_ARGS(5) TAP_ARGS(6) TAP_ARGS(7)
                       TAP_ARGS(8) TAP_ARGS(9) TAP_ARGS(10) TAP_ARGS(11) TAP_ARGS(12) TAP_ARGS(13) TAP_ARGS(14) TAP_ARGS(15)
                       TAP_ARGS(16) TAP_ARGS(17) TAP_ARGS(18) TAP_ARGS(19) TAP_ARGS(20) TAP_ARGS(21) TAP_ARGS(22) TAP_ARGS(23)
                       TAP_ARGS(24) TAP_ARGS(25) TAP_ARGS(26) TAP_ARGS(27) TAP_ARGS(28) TAP_ARGS(29) TAP_ARGS(30) TAP_ARGS(31)) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= row_pixels || y >= rows) {
        return;
    }
    float4 acc = use_acc ? vload4(x, (__global const float*)(acc_in + acc_offset + (long)y * acc_pitch))
                         : (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    TAP(0) TAP(1) TAP(2) TAP(3) TAP(4) TAP(5) TAP(6) TAP(7) TAP(8) TAP(9)
    TAP(10) TAP(11) TAP(12) TAP(13) TAP(14) TAP(15) TAP(16) TAP(17) TAP(18) TAP(19)
    TAP(20) TAP(21) TAP(22) TAP(23) TAP(24) TAP(25) TAP(26) TAP(27) TAP(28) TAP(29)
    TAP(30) TAP(31)
    __global uchar* row = dst + dst_offset + (long)y * dst_pitch;
    if (dst_half) {
        vstore_half4_rte(acc, x, (__global half*)row);
    } else {
        vstore4(acc, x, (__global float*)row);
    }
}

// ---- Blending straight from decoded 8-bit NV12 frames (Premiere file path) ----
// Same conversion as fb_blend_nv12_* in fb_blend.cu. Each tap buffer holds one
// compact NV12 frame (luma rows, then UV rows, same pitch); o##i is unused.
// The accumulator and output hold B, G, R, A floats per pixel.

inline float fb_chroma(__global const uchar* uv, int pitch, int cw, int ch, float cx, float cy, int component) {
    const int x0 = max(0, min(cw - 1, (int)floor(cx)));
    const int y0 = max(0, min(ch - 1, (int)floor(cy)));
    const int x1 = min(cw - 1, x0 + 1);
    const int y1 = min(ch - 1, y0 + 1);
    const float fx = clamp(cx - floor(cx), 0.0f, 1.0f);
    const float fy = clamp(cy - floor(cy), 0.0f, 1.0f);
    const float a = uv[(long)y0 * pitch + 2 * x0 + component];
    const float b = uv[(long)y0 * pitch + 2 * x1 + component];
    const float c = uv[(long)y1 * pitch + 2 * x0 + component];
    const float d = uv[(long)y1 * pitch + 2 * x1 + component];
    const float top = a + fx * (b - a);
    const float bottom = c + fx * (d - c);
    return top + fy * (bottom - top);
}

inline float4 fb_nv12_bgr(__global const uchar* luma, int pitch, int src_w, int src_h, int sx, int sy, float y_scale,
                          float y_offset, float c_scale, float r_cr, float g_cb, float g_cr, float b_cb) {
    __global const uchar* uv = luma + (long)pitch * src_h;
    const float cx = 0.5f * sx;
    const float cy = 0.5f * sy - 0.25f;
    const float y = ((float)luma[(long)sy * pitch + sx] - y_offset) * y_scale;
    const float cb = (fb_chroma(uv, pitch, src_w / 2, src_h / 2, cx, cy, 0) - 128.0f) * c_scale;
    const float cr = (fb_chroma(uv, pitch, src_w / 2, src_h / 2, cx, cy, 1) - 128.0f) * c_scale;
    return (float4)(y + b_cb * cb, (y - g_cb * cb) - g_cr * cr, y + r_cr * cr, 1.0f);
}

#define NV12_TAP(i) if (i < count) { acc = acc + w##i * fb_nv12_bgr(s##i, p##i, src_w, src_h, sx, sy, y_scale, y_offset, c_scale, r_cr, g_cb, g_cr, b_cb); }

__kernel void fb_blend_nv12(__global uchar* dst, int dst_offset, int dst_pitch, int dst_half,
                            __global const uchar* acc_in, int acc_offset, int acc_pitch, int use_acc,
                            int width, int height, int count, int src_w, int src_h, int last,
                            float y_scale, float y_offset, float c_scale, float r_cr, float g_cb, float g_cr, float b_cb
                            TAP_ARGS(0) TAP_ARGS(1) TAP_ARGS(2) TAP_ARGS(3) TAP_ARGS(4) TAP_ARGS(5) TAP_ARGS(6) TAP_ARGS(7)
                            TAP_ARGS(8) TAP_ARGS(9) TAP_ARGS(10) TAP_ARGS(11) TAP_ARGS(12) TAP_ARGS(13) TAP_ARGS(14) TAP_ARGS(15)
                            TAP_ARGS(16) TAP_ARGS(17) TAP_ARGS(18) TAP_ARGS(19) TAP_ARGS(20) TAP_ARGS(21) TAP_ARGS(22) TAP_ARGS(23)
                            TAP_ARGS(24) TAP_ARGS(25) TAP_ARGS(26) TAP_ARGS(27) TAP_ARGS(28) TAP_ARGS(29) TAP_ARGS(30) TAP_ARGS(31)) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int sx = width == src_w ? x : min(src_w - 1, (int)(((long)x * src_w) / width));
    const int sy = height == src_h ? y : min(src_h - 1, (int)(((long)y * src_h) / height));
    float4 acc = use_acc ? vload4(x, (__global const float*)(acc_in + acc_offset + (long)y * acc_pitch))
                         : (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    NV12_TAP(0) NV12_TAP(1) NV12_TAP(2) NV12_TAP(3) NV12_TAP(4) NV12_TAP(5) NV12_TAP(6) NV12_TAP(7)
    NV12_TAP(8) NV12_TAP(9) NV12_TAP(10) NV12_TAP(11) NV12_TAP(12) NV12_TAP(13) NV12_TAP(14) NV12_TAP(15)
    NV12_TAP(16) NV12_TAP(17) NV12_TAP(18) NV12_TAP(19) NV12_TAP(20) NV12_TAP(21) NV12_TAP(22) NV12_TAP(23)
    NV12_TAP(24) NV12_TAP(25) NV12_TAP(26) NV12_TAP(27) NV12_TAP(28) NV12_TAP(29) NV12_TAP(30) NV12_TAP(31)
    if (last) {
        acc.w = 1.0f;
    }
    __global uchar* row = dst + dst_offset + (long)y * dst_pitch;
    if (dst_half) {
        vstore_half4_rte(acc, x, (__global half*)row);
    } else {
        vstore4(acc, x, (__global float*)row);
    }
}
