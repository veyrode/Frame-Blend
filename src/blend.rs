//! CPU application of kernel weights to pixel buffers.
//!
//! Every element is `sum_k w[k] * src[k]`, accumulated in f32 in tap order with
//! a separately rounded multiply and add (no fused multiply-add). The GPU
//! kernels in the adapters use the same order and rounding so their output
//! matches this path. Channel order is irrelevant: each element is independent.

use std::thread;

/// One strided plane: `rows` rows of `row_elements` elements, `stride` elements apart.
#[derive(Clone, Copy)]
pub struct Plane<T> {
    pub ptr: *const T,
    pub stride: isize,
}

unsafe impl<T> Send for Plane<T> {}
unsafe impl<T> Sync for Plane<T> {}

#[derive(Clone, Copy)]
struct DstPlane<T> {
    ptr: *mut T,
    stride: isize,
}

unsafe impl<T> Send for DstPlane<T> {}
unsafe impl<T> Sync for DstPlane<T> {}

#[inline(always)]
fn accumulate(acc: f32, weight: f32, value: f32) -> f32 {
    // Written as two operations so the compiler cannot contract them.
    let product = weight * value;
    acc + product
}

/// Rounds a blended 8-bit value to the nearest code, halves away from zero.
#[inline(always)]
pub fn quantize_u8(value: f32) -> u8 {
    let rounded = (value + 0.5).floor();
    rounded.clamp(0.0, 255.0) as u8
}

/// Rounds a blended 16-bit value to the nearest code in `0..=max`, halves away
/// from zero (After Effects' 16 bpc pixels run from 0 to 32768).
#[inline(always)]
pub fn quantize_u16(value: f32, max: f32) -> u16 {
    let rounded = (value + 0.5).floor();
    rounded.clamp(0.0, max) as u16
}

fn worker_count(rows: usize) -> usize {
    let available = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    available.min(rows.max(1)).min(32)
}

fn run_rows<F>(rows: usize, work: F)
where
    F: Fn(usize, usize) + Sync,
{
    let workers = worker_count(rows);
    if workers <= 1 {
        work(0, rows);
        return;
    }
    let chunk = rows.div_ceil(workers);
    thread::scope(|scope| {
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            let work = &work;
            scope.spawn(move || work(start, end));
        }
    });
}

/// # Safety
/// Every plane must cover `rows` rows of `row_elements` readable elements, the
/// destination must be writable for the same extent and must not alias a source.
pub unsafe fn apply_f32(dst: *mut f32, dst_stride: isize, srcs: &[Plane<f32>], weights: &[f32], row_elements: usize, rows: usize) {
    let dst = DstPlane { ptr: dst, stride: dst_stride };
    run_rows(rows, |start, end| {
        let dst = dst;
        for y in start..end {
            let out = std::slice::from_raw_parts_mut(dst.ptr.offset(y as isize * dst.stride), row_elements);
            out.fill(0.0);
            for (plane, &w) in srcs.iter().zip(weights) {
                let row = std::slice::from_raw_parts(plane.ptr.offset(y as isize * plane.stride), row_elements);
                for (o, &s) in out.iter_mut().zip(row) {
                    *o = accumulate(*o, w, s);
                }
            }
        }
    });
}

/// 8-bit variant: accumulates in f32 on code values 0..255, then rounds.
///
/// # Safety
/// Same contract as [`apply_f32`].
pub unsafe fn apply_u8(dst: *mut u8, dst_stride: isize, srcs: &[Plane<u8>], weights: &[f32], row_elements: usize, rows: usize) {
    let dst = DstPlane { ptr: dst, stride: dst_stride };
    run_rows(rows, |start, end| {
        let dst = dst;
        let mut acc = vec![0.0f32; row_elements];
        for y in start..end {
            acc.fill(0.0);
            for (plane, &w) in srcs.iter().zip(weights) {
                let row = std::slice::from_raw_parts(plane.ptr.offset(y as isize * plane.stride), row_elements);
                for (a, &s) in acc.iter_mut().zip(row) {
                    *a = accumulate(*a, w, s as f32);
                }
            }
            let out = std::slice::from_raw_parts_mut(dst.ptr.offset(y as isize * dst.stride), row_elements);
            for (o, &a) in out.iter_mut().zip(&acc) {
                *o = quantize_u8(a);
            }
        }
    });
}

/// 16-bit variant: accumulates in f32 on code values, then rounds and clamps
/// to `0..=max`.
///
/// # Safety
/// Same contract as [`apply_f32`].
pub unsafe fn apply_u16(dst: *mut u16, dst_stride: isize, srcs: &[Plane<u16>], weights: &[f32], row_elements: usize,
                        rows: usize, max: u16) {
    let dst = DstPlane { ptr: dst, stride: dst_stride };
    let max = max as f32;
    run_rows(rows, |start, end| {
        let dst = dst;
        let mut acc = vec![0.0f32; row_elements];
        for y in start..end {
            acc.fill(0.0);
            for (plane, &w) in srcs.iter().zip(weights) {
                let row = std::slice::from_raw_parts(plane.ptr.offset(y as isize * plane.stride), row_elements);
                for (a, &s) in acc.iter_mut().zip(row) {
                    *a = accumulate(*a, w, s as f32);
                }
            }
            let out = std::slice::from_raw_parts_mut(dst.ptr.offset(y as isize * dst.stride), row_elements);
            for (o, &a) in out.iter_mut().zip(&acc) {
                *o = quantize_u16(a, max);
            }
        }
    });
}

/// Safe convenience wrapper over contiguous buffers of equal length.
pub fn blend_f32(dst: &mut [f32], srcs: &[&[f32]], weights: &[f32]) {
    assert_eq!(srcs.len(), weights.len());
    assert!(srcs.iter().all(|s| s.len() == dst.len()));
    let planes: Vec<Plane<f32>> = srcs.iter().map(|s| Plane { ptr: s.as_ptr(), stride: 0 }).collect();
    unsafe { apply_f32(dst.as_mut_ptr(), 0, &planes, weights, dst.len(), 1) }
}

/// Safe convenience wrapper over contiguous 8-bit buffers of `rows` rows.
pub fn blend_u8(dst: &mut [u8], srcs: &[&[u8]], weights: &[f32], rows: usize) {
    assert_eq!(srcs.len(), weights.len());
    assert!(srcs.iter().all(|s| s.len() == dst.len()));
    assert!(rows > 0 && dst.len() % rows == 0);
    let row_elements = dst.len() / rows;
    let planes: Vec<Plane<u8>> =
        srcs.iter().map(|s| Plane { ptr: s.as_ptr(), stride: row_elements as isize }).collect();
    unsafe { apply_u8(dst.as_mut_ptr(), row_elements as isize, &planes, weights, row_elements, rows) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_blend_is_weighted_sum_in_tap_order() {
        let a = [0.0f32, 1.0, 0.5, 0.25];
        let b = [1.0f32, 1.0, 0.0, 0.75];
        let mut out = [9.0f32; 4];
        blend_f32(&mut out, &[&a, &b], &[0.25, 0.75]);
        assert_eq!(out, [0.75, 1.0, 0.125, 0.625]);
    }

    #[test]
    fn u8_blend_rounds_to_nearest_and_threads_rows() {
        let rows = 37;
        let a = vec![10u8; rows * 5];
        let b = vec![11u8; rows * 5];
        let mut out = vec![0u8; rows * 5];
        blend_u8(&mut out, &[&a, &b], &[0.5, 0.5], rows);
        assert!(out.iter().all(|&v| v == 11)); // 10.5 rounds up
        blend_u8(&mut out, &[&a, &b], &[0.75, 0.25], rows);
        assert!(out.iter().all(|&v| v == 10)); // 10.25
        let full = vec![255u8; rows * 5];
        blend_u8(&mut out, &[&full], &[1.0], rows);
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn u16_blend_rounds_and_clamps_to_max() {
        let a = [1000u16, 32768, 0, 7];
        let b = [1001u16, 32768, 0, 8];
        let mut out = [0u16; 4];
        let planes = [Plane { ptr: a.as_ptr(), stride: 4 }, Plane { ptr: b.as_ptr(), stride: 4 }];
        unsafe { apply_u16(out.as_mut_ptr(), 4, &planes, &[0.5, 0.5], 4, 1, 32768) };
        assert_eq!(out, [1001, 32768, 0, 8]); // 1000.5 and 7.5 round up
        unsafe { apply_u16(out.as_mut_ptr(), 4, &planes[..1], &[1.5], 4, 1, 32768) };
        assert_eq!(out, [1500, 32768, 0, 11]); // clamped to the maximum
    }

    #[test]
    fn strided_rows_leave_padding_untouched() {
        // Two rows of 3 elements stored with stride 4.
        let a = [1.0f32, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, -1.0];
        let b = [3.0f32, 2.0, 1.0, -1.0, 0.0, 1.0, 2.0, -1.0];
        let mut out = [7.0f32; 8];
        let planes = [Plane { ptr: a.as_ptr(), stride: 4 }, Plane { ptr: b.as_ptr(), stride: 4 }];
        unsafe { apply_f32(out.as_mut_ptr(), 4, &planes, &[0.5, 0.5], 3, 2) };
        assert_eq!(out, [2.0, 2.0, 2.0, 7.0, 2.0, 3.0, 4.0, 7.0]);
    }
}
