//! Frame-blending core shared by the Adobe adapter and the offline CLI.
//!
//! The kernel (support, exact rational weights, edge renormalization) is
//! defined once in [`kernel`]; hosts only fetch frames and apply the weights.
//! Blending operates directly on the encoded (gamma) pixel values.

pub mod blend;
pub mod kernel;
pub mod rational;

use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use kernel::{Availability, KernelError};
use rational::Rational;

pub const FB_OK: i32 = 0;
pub const FB_ERR_BAD_ARGS: i32 = 1;
pub const FB_ERR_RATE: i32 = 2;
pub const FB_ERR_AVAILABILITY: i32 = 3;
pub const FB_ERR_CAPACITY: i32 = 4;
pub const FB_ERR_OVERFLOW: i32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FbRational {
    pub num: i64,
    pub den: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FbTap {
    pub index: i64,
    pub weight_num: i64,
    pub weight_den: i64,
    pub weight: f32,
}

fn to_rational(value: FbRational) -> Option<Rational> {
    Rational::new(value.num, value.den)
}

fn from_rational(value: Rational) -> FbRational {
    FbRational { num: value.numer(), den: value.denom() }
}

fn kernel_error(error: KernelError) -> i32 {
    match error {
        KernelError::NonPositiveRate => FB_ERR_RATE,
        KernelError::EmptyAvailability => FB_ERR_AVAILABILITY,
    }
}

/// Runs `body`, turning arithmetic overflow panics into an error code so no
/// panic unwinds into host C++ code.
fn guarded(body: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(FB_ERR_OVERFLOW)
}

/// Exact rate behind a floating-point host frame rate (e.g. 29.97 -> 30000/1001).
#[no_mangle]
pub extern "C" fn fb_rational_from_fps(fps: f64, out: *mut FbRational) -> i32 {
    if out.is_null() {
        return FB_ERR_BAD_ARGS;
    }
    match Rational::from_frame_rate_f64(fps) {
        Some(rate) => {
            unsafe { *out = from_rational(rate) };
            FB_OK
        }
        None => FB_ERR_RATE,
    }
}

/// Capacity a caller must provide to [`fb_kernel`] for these rates.
#[no_mangle]
pub extern "C" fn fb_max_taps(source_fps: FbRational, output_fps: FbRational, out: *mut i32) -> i32 {
    if out.is_null() {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| {
        let (Some(src), Some(dst)) = (to_rational(source_fps), to_rational(output_fps)) else {
            return FB_ERR_BAD_ARGS;
        };
        match kernel::max_taps(src, dst) {
            Ok(n) => match i32::try_from(n) {
                Ok(n) => {
                    unsafe { *out = n };
                    FB_OK
                }
                Err(_) => FB_ERR_OVERFLOW,
            },
            Err(e) => kernel_error(e),
        }
    })
}

/// Source position of output frame `output_frame` when output frame 0 is
/// centred on source position `source_origin`: `origin + output_frame * R`.
#[no_mangle]
pub extern "C" fn fb_center_for_output_frame(
    source_fps: FbRational,
    output_fps: FbRational,
    source_origin: FbRational,
    output_frame: FbRational,
    out: *mut FbRational,
) -> i32 {
    if out.is_null() {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| {
        let (Some(src), Some(dst), Some(origin), Some(frame)) = (
            to_rational(source_fps),
            to_rational(output_fps),
            to_rational(source_origin),
            to_rational(output_frame),
        ) else {
            return FB_ERR_BAD_ARGS;
        };
        match kernel::rate_ratio(src, dst) {
            Ok(ratio) => {
                unsafe { *out = from_rational(origin + frame * ratio) };
                FB_OK
            }
            Err(e) => kernel_error(e),
        }
    })
}

/// Inverse of [`fb_center_for_output_frame`]: the output-frame position (in
/// output frames, possibly fractional) at which source frame `source_frame` lies.
#[no_mangle]
pub extern "C" fn fb_output_position_of_source_frame(
    source_fps: FbRational,
    output_fps: FbRational,
    source_origin: FbRational,
    source_frame: i64,
    out: *mut FbRational,
) -> i32 {
    if out.is_null() {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| {
        let (Some(src), Some(dst), Some(origin)) =
            (to_rational(source_fps), to_rational(output_fps), to_rational(source_origin))
        else {
            return FB_ERR_BAD_ARGS;
        };
        match kernel::rate_ratio(src, dst) {
            Ok(ratio) => {
                unsafe { *out = from_rational((Rational::integer(source_frame) - origin) / ratio) };
                FB_OK
            }
            Err(e) => kernel_error(e),
        }
    })
}

/// Production kernel for the output frame centred on source position `center`.
/// Frames outside `[first_available, last_available]` are discarded and the
/// remaining weights renormalized. Taps are written in increasing index order.
#[no_mangle]
pub extern "C" fn fb_kernel(
    source_fps: FbRational,
    output_fps: FbRational,
    center: FbRational,
    first_available: i64,
    last_available: i64,
    taps: *mut FbTap,
    capacity: i32,
    count: *mut i32,
) -> i32 {
    if taps.is_null() || count.is_null() || capacity <= 0 {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| {
        let (Some(src), Some(dst), Some(c)) = (to_rational(source_fps), to_rational(output_fps), to_rational(center))
        else {
            return FB_ERR_BAD_ARGS;
        };
        let available = Availability { first: first_available, last: last_available };
        let result = match kernel::kernel(src, dst, c, available) {
            Ok(result) => result,
            Err(e) => return kernel_error(e),
        };
        if result.len() > capacity as usize {
            return FB_ERR_CAPACITY;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(taps, result.len()) };
        for (slot, tap) in out.iter_mut().zip(&result) {
            *slot = FbTap {
                index: tap.index,
                weight_num: tap.weight.numer(),
                weight_den: tap.weight.denom(),
                weight: tap.weight.to_f64() as f32,
            };
        }
        unsafe { *count = result.len() as i32 };
        FB_OK
    })
}

fn apply_args_ok<T>(dst: *mut T, srcs: *const *const T, strides: *const isize, weights: *const f32, count: i32, row_elements: i32, rows: i32) -> bool {
    !dst.is_null() && !srcs.is_null() && !strides.is_null() && !weights.is_null() && count > 0 && row_elements > 0 && rows > 0
}

/// CPU fallback: `dst = sum_k weights[k] * srcs[k]` over `rows` rows of
/// `row_elements` floats. Strides are in bytes and may be negative.
#[no_mangle]
pub extern "C" fn fb_apply_f32(
    dst: *mut f32,
    dst_row_bytes: isize,
    srcs: *const *const f32,
    src_row_bytes: *const isize,
    weights: *const f32,
    count: i32,
    row_elements: i32,
    rows: i32,
) -> i32 {
    if !apply_args_ok(dst, srcs, src_row_bytes, weights, count, row_elements, rows) {
        return FB_ERR_BAD_ARGS;
    }
    let element = std::mem::size_of::<f32>() as isize;
    if dst_row_bytes % element != 0 {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| unsafe {
        let n = count as usize;
        let ptrs = std::slice::from_raw_parts(srcs, n);
        let strides = std::slice::from_raw_parts(src_row_bytes, n);
        if ptrs.iter().any(|p| p.is_null()) || strides.iter().any(|s| s % element != 0) {
            return FB_ERR_BAD_ARGS;
        }
        let planes: Vec<blend::Plane<f32>> =
            ptrs.iter().zip(strides).map(|(&ptr, &s)| blend::Plane { ptr, stride: s / element }).collect();
        let weights = std::slice::from_raw_parts(weights, n);
        blend::apply_f32(dst, dst_row_bytes / element, &planes, weights, row_elements as usize, rows as usize);
        FB_OK
    })
}

/// 8-bit CPU fallback; accumulates in f32 and rounds to the nearest code.
#[no_mangle]
pub extern "C" fn fb_apply_u8(
    dst: *mut u8,
    dst_row_bytes: isize,
    srcs: *const *const u8,
    src_row_bytes: *const isize,
    weights: *const f32,
    count: i32,
    row_elements: i32,
    rows: i32,
) -> i32 {
    if !apply_args_ok(dst, srcs, src_row_bytes, weights, count, row_elements, rows) {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| unsafe {
        let n = count as usize;
        let ptrs = std::slice::from_raw_parts(srcs, n);
        if ptrs.iter().any(|p| p.is_null()) {
            return FB_ERR_BAD_ARGS;
        }
        let strides = std::slice::from_raw_parts(src_row_bytes, n);
        let planes: Vec<blend::Plane<u8>> =
            ptrs.iter().zip(strides).map(|(&ptr, &s)| blend::Plane { ptr, stride: s }).collect();
        let weights = std::slice::from_raw_parts(weights, n);
        blend::apply_u8(dst, dst_row_bytes, &planes, weights, row_elements as usize, rows as usize);
        FB_OK
    })
}

/// 16-bit CPU fallback (After Effects 16 bpc: codes 0..=`max`, 32768 there);
/// accumulates in f32 and rounds to the nearest code. Strides are in bytes.
#[no_mangle]
pub extern "C" fn fb_apply_u16(
    dst: *mut u16,
    dst_row_bytes: isize,
    srcs: *const *const u16,
    src_row_bytes: *const isize,
    weights: *const f32,
    count: i32,
    row_elements: i32,
    rows: i32,
    max: u16,
) -> i32 {
    if !apply_args_ok(dst, srcs, src_row_bytes, weights, count, row_elements, rows) {
        return FB_ERR_BAD_ARGS;
    }
    let element = std::mem::size_of::<u16>() as isize;
    if dst_row_bytes % element != 0 {
        return FB_ERR_BAD_ARGS;
    }
    guarded(|| unsafe {
        let n = count as usize;
        let ptrs = std::slice::from_raw_parts(srcs, n);
        let strides = std::slice::from_raw_parts(src_row_bytes, n);
        if ptrs.iter().any(|p| p.is_null()) || strides.iter().any(|s| s % element != 0) {
            return FB_ERR_BAD_ARGS;
        }
        let planes: Vec<blend::Plane<u16>> =
            ptrs.iter().zip(strides).map(|(&ptr, &s)| blend::Plane { ptr, stride: s / element }).collect();
        let weights = std::slice::from_raw_parts(weights, n);
        blend::apply_u16(dst, dst_row_bytes / element, &planes, weights, row_elements as usize, rows as usize, max);
        FB_OK
    })
}

#[no_mangle]
pub extern "C" fn fb_error_message(code: i32) -> *const c_char {
    let text: &'static [u8] = match code {
        FB_OK => b"ok\0",
        FB_ERR_BAD_ARGS => b"bad arguments\0",
        FB_ERR_RATE => b"frame rates must be positive\0",
        FB_ERR_AVAILABILITY => b"no source frames are available\0",
        FB_ERR_CAPACITY => b"tap buffer too small\0",
        FB_ERR_OVERFLOW => b"exact arithmetic overflow\0",
        _ => b"unknown error\0",
    };
    text.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(num: i64, den: i64) -> FbRational {
        FbRational { num, den }
    }

    #[test]
    fn c_abi_kernel_matches_rust_kernel() {
        let mut taps = [FbTap { index: 0, weight_num: 0, weight_den: 1, weight: 0.0 }; 64];
        let mut count = 0;
        let status = fb_kernel(fr(960, 1), fr(50, 1), fr(53856, 5), 0, 38790, taps.as_mut_ptr(), 64, &mut count);
        assert_eq!(status, FB_OK);
        let expected = kernel::kernel(
            Rational::integer(960),
            Rational::integer(50),
            Rational::new(53856, 5).unwrap(),
            Availability { first: 0, last: 38790 },
        )
        .unwrap();
        assert_eq!(count as usize, expected.len());
        for (a, b) in taps.iter().zip(&expected) {
            assert_eq!(a.index, b.index);
            assert_eq!((a.weight_num, a.weight_den), (b.weight.numer(), b.weight.denom()));
            assert_eq!(a.weight, b.weight.to_f64() as f32);
        }
        let mut small = [taps[0]; 2];
        assert_eq!(fb_kernel(fr(960, 1), fr(50, 1), fr(0, 1), 0, 100, small.as_mut_ptr(), 2, &mut count), FB_ERR_CAPACITY);
        assert_eq!(fb_kernel(fr(0, 1), fr(50, 1), fr(0, 1), 0, 100, taps.as_mut_ptr(), 64, &mut count), FB_ERR_RATE);
        assert_eq!(fb_kernel(fr(1, 0), fr(50, 1), fr(0, 1), 0, 100, taps.as_mut_ptr(), 64, &mut count), FB_ERR_BAD_ARGS);
    }

    #[test]
    fn c_abi_positions_round_trip() {
        let mut centre = fr(0, 1);
        assert_eq!(fb_center_for_output_frame(fr(960, 1), fr(50, 1), fr(10752, 1), fr(3, 1), &mut centre), FB_OK);
        assert_eq!((centre.num, centre.den), (54048, 5)); // 10752 + 3 * 96/5
        let mut pos = fr(0, 1);
        assert_eq!(fb_output_position_of_source_frame(fr(240, 1), fr(30, 1), fr(0, 1), 12, &mut pos), FB_OK);
        assert_eq!((pos.num, pos.den), (3, 2));
        let mut max = 0;
        assert_eq!(fb_max_taps(fr(600, 1), fr(30, 1), &mut max), FB_OK);
        assert_eq!(max, 21);
        let mut rate = fr(0, 1);
        assert_eq!(fb_rational_from_fps(29.97002997002997, &mut rate), FB_OK);
        assert_eq!((rate.num, rate.den), (30000, 1001));
    }

    #[test]
    fn c_abi_overflow_is_reported_not_unwound() {
        let mut centre = fr(0, 1);
        let huge = fr(i64::MAX, 1);
        assert_eq!(fb_center_for_output_frame(fr(960, 1), fr(7, 1), huge, huge, &mut centre), FB_ERR_OVERFLOW);
    }

    #[test]
    fn c_abi_apply_f32_and_u8() {
        let a = [0.0f32, 1.0, 0.5, 0.25];
        let b = [1.0f32, 1.0, 0.0, 0.75];
        let mut out = [0.0f32; 4];
        let srcs = [a.as_ptr(), b.as_ptr()];
        let strides = [16isize, 16];
        let weights = [0.25f32, 0.75];
        assert_eq!(fb_apply_f32(out.as_mut_ptr(), 16, srcs.as_ptr(), strides.as_ptr(), weights.as_ptr(), 2, 4, 1), FB_OK);
        assert_eq!(out, [0.75, 1.0, 0.125, 0.625]);
        let a8 = [0u8, 255, 100, 7];
        let b8 = [255u8, 255, 101, 8];
        let mut out8 = [0u8; 4];
        let srcs8 = [a8.as_ptr(), b8.as_ptr()];
        let half = [0.5f32, 0.5];
        assert_eq!(fb_apply_u8(out8.as_mut_ptr(), 4, srcs8.as_ptr(), [4isize, 4].as_ptr(), half.as_ptr(), 2, 4, 1), FB_OK);
        assert_eq!(out8, [128, 255, 101, 8]);
    }
}
