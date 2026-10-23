//! The production blend kernel: sample-cell overlap, computed exactly.
//!
//! Source frame `j` occupies the cell `[j - 1/2, j + 1/2]` in source-frame units.
//! An output frame whose centre maps to source position `c` covers
//! `[c - R/2, c + R/2]`, where `R = source_fps / output_fps`, and gives frame `j`
//! the weight `|cell(j) ∩ window| / R`. The weights sum to exactly 1.
//!
//! When `output_fps >= source_fps` nothing is blended: the frame nearest to `c`
//! is passed through. Taps outside the available source range are discarded
//! and the remaining weights renormalized to sum to 1.

use crate::rational::Rational;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tap {
    pub index: i64,
    pub weight: Rational,
}

/// Inclusive range of source frames the host can actually deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Availability {
    pub first: i64,
    pub last: i64,
}

impl Availability {
    pub const UNBOUNDED: Availability = Availability { first: i64::MIN / 4, last: i64::MAX / 4 };

    pub fn contains(self, index: i64) -> bool {
        self.first <= index && index <= self.last
    }

    fn clamp(self, index: i64) -> i64 {
        index.clamp(self.first, self.last)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    NonPositiveRate,
    EmptyAvailability,
}

/// `R = source_fps / output_fps`.
pub fn rate_ratio(source_fps: Rational, output_fps: Rational) -> Result<Rational, KernelError> {
    if !source_fps.is_positive() || !output_fps.is_positive() {
        return Err(KernelError::NonPositiveRate);
    }
    Ok(source_fps / output_fps)
}

/// Source position of a time measured from source frame 0's timestamp.
pub fn center_at_time(time_seconds: Rational, source_fps: Rational) -> Rational {
    time_seconds * source_fps
}

/// Upper bound on the number of taps any kernel for these rates can return.
pub fn max_taps(source_fps: Rational, output_fps: Rational) -> Result<usize, KernelError> {
    let ratio = rate_ratio(source_fps, output_fps)?;
    if ratio <= Rational::ONE {
        return Ok(1);
    }
    Ok(ratio.ceil() as usize + 1)
}

fn nearest(center: Rational) -> i64 {
    (center + Rational::HALF).floor()
}

/// Unrestricted sample-cell overlap weights (no edge handling).
pub fn window(source_fps: Rational, output_fps: Rational, center: Rational) -> Result<Vec<Tap>, KernelError> {
    let ratio = rate_ratio(source_fps, output_fps)?;
    if ratio <= Rational::ONE {
        return Ok(vec![Tap { index: nearest(center), weight: Rational::ONE }]);
    }
    let half = ratio * Rational::HALF;
    let (lo, hi) = (center - half, center + half);
    // Cell j overlaps the open window iff lo - 1/2 < j < hi + 1/2.
    let first = (lo - Rational::HALF).floor() + 1;
    let last = (hi + Rational::HALF).ceil() - 1;
    let mut taps = Vec::with_capacity((last - first + 1).max(0) as usize);
    for index in first..=last {
        let j = Rational::integer(index);
        let overlap = hi.min(j + Rational::HALF) - lo.max(j - Rational::HALF);
        if overlap.is_positive() {
            taps.push(Tap { index, weight: overlap / ratio });
        }
    }
    Ok(taps)
}

/// Production kernel with edge handling: taps outside `available` are
/// discarded and the rest renormalized. If no tap of the window is available
/// (the output lies entirely beyond the source), the nearest available frame
/// is used alone.
pub fn kernel(
    source_fps: Rational,
    output_fps: Rational,
    center: Rational,
    available: Availability,
) -> Result<Vec<Tap>, KernelError> {
    if available.first > available.last {
        return Err(KernelError::EmptyAvailability);
    }
    let taps = window(source_fps, output_fps, center)?;
    let kept: Vec<Tap> = taps.into_iter().filter(|t| available.contains(t.index)).collect();
    if kept.is_empty() {
        return Ok(vec![Tap { index: available.clamp(nearest(center)), weight: Rational::ONE }]);
    }
    let mass = kept.iter().fold(Rational::ZERO, |acc, t| acc + t.weight);
    if mass == Rational::ONE {
        return Ok(kept);
    }
    Ok(kept.into_iter().map(|t| Tap { index: t.index, weight: t.weight / mass }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    fn sum(taps: &[Tap]) -> Rational {
        taps.iter().fold(Rational::ZERO, |acc, t| acc + t.weight)
    }

    /// Weights relative to `origin`, as (offset, weight) pairs.
    fn relative(taps: &[Tap], origin: i64) -> Vec<(i64, Rational)> {
        taps.iter().map(|t| (t.index - origin, t.weight)).collect()
    }

    fn trapezoid(half_width: i64, endpoint: Rational, interior: Rational) -> Vec<(i64, Rational)> {
        (-half_width..=half_width)
            .map(|k| (k, if k.abs() == half_width { endpoint } else { interior }))
            .collect()
    }

    #[test]
    fn exact_960_to_30() {
        let taps = window(r(960, 1), r(30, 1), r(10752, 1)).unwrap();
        assert_eq!(relative(&taps, 10752), trapezoid(16, r(1, 64), r(1, 32)));
        assert_eq!(sum(&taps), Rational::ONE);
        // Every output frame of 960->30 has the same phase.
        for n in 0..300 {
            let c = r(10752, 1) + r(32, 1) * Rational::integer(n);
            assert_eq!(relative(&window(r(960, 1), r(30, 1), c).unwrap(), c.floor()), trapezoid(16, r(1, 64), r(1, 32)));
        }
    }

    #[test]
    fn exact_960_to_60() {
        let taps = window(r(960, 1), r(60, 1), r(10752, 1)).unwrap();
        assert_eq!(relative(&taps, 10752), trapezoid(8, r(1, 32), r(1, 16)));
        assert_eq!(sum(&taps), Rational::ONE);
    }

    /// Must equal the exact sample-cell table measured on VEGAS Pro renders (960 -> 50 fps).
    #[test]
    fn exact_960_to_50_five_phases() {
        let interior = r(5, 96);
        let expected: [(i64, Rational, Rational); 5] = [
            (-10, r(1, 192), r(1, 192)),
            (-9, r(3, 64), r(1, 64)),
            (-9, r(7, 192), r(5, 192)),
            (-9, r(5, 192), r(7, 192)),
            (-9, r(1, 64), r(3, 64)),
        ];
        let ratio = r(96, 5);
        for n in 0..10 {
            let c = r(10752, 1) + ratio * Rational::integer(n);
            let phase = (n % 5) as usize;
            assert_eq!(c.fract(), r(phase as i64, 5));
            let taps = window(r(960, 1), r(50, 1), c).unwrap();
            let (left, left_w, right_w) = expected[phase];
            let origin = c.floor();
            let rel = relative(&taps, origin);
            assert_eq!(rel.first().unwrap(), &(left, left_w), "phase {phase}");
            assert_eq!(rel.last().unwrap(), &(10, right_w), "phase {phase}");
            assert_eq!(rel.len() as i64, 10 - left + 1);
            for &(_, w) in &rel[1..rel.len() - 1] {
                assert_eq!(w, interior);
            }
            assert_eq!(sum(&taps), Rational::ONE);
        }
    }

    /// Weights sum to 1, the support is exactly the cells the window touches,
    /// and the kernel repeats (translated by num(R) frames) every den(R) outputs.
    fn check_rate(source: i64, output: i64, expected_period: i64) {
        let (src, out) = (r(source, 1), r(output, 1));
        let ratio = rate_ratio(src, out).unwrap();
        assert_eq!(ratio.denom(), expected_period);
        let bound = max_taps(src, out).unwrap();
        for n in 0..(3 * expected_period + 7) {
            let c = center_at_time(Rational::integer(n) / out, src);
            let taps = window(src, out, c).unwrap();
            assert_eq!(sum(&taps), Rational::ONE);
            assert!(taps.len() <= bound);
            assert!(taps.iter().all(|t| t.weight.is_positive()));
            let (lo, hi) = (c - ratio * Rational::HALF, c + ratio * Rational::HALF);
            assert!(Rational::integer(taps[0].index) + Rational::HALF > lo);
            assert!(Rational::integer(taps[0].index - 1) + Rational::HALF <= lo);
            let last = taps.last().unwrap().index;
            assert!(Rational::integer(last) - Rational::HALF < hi);
            assert!(Rational::integer(last + 1) - Rational::HALF >= hi);
            for pair in taps.windows(2) {
                assert_eq!(pair[1].index, pair[0].index + 1);
            }
            let later = window(src, out, c + ratio * Rational::integer(expected_period)).unwrap();
            let shift = ratio.numer();
            assert_eq!(later, taps.iter().map(|t| Tap { index: t.index + shift, weight: t.weight }).collect::<Vec<_>>());
        }
        // The period is minimal: phases 0..period are all distinct.
        let phases: std::collections::HashSet<Rational> =
            (0..expected_period).map(|n| (ratio * Rational::integer(n)).fract()).collect();
        assert_eq!(phases.len() as i64, expected_period);
    }

    #[test]
    fn rates_240_600_to_30_240_to_60_120_to_50() {
        check_rate(240, 30, 1);
        check_rate(600, 30, 1);
        check_rate(240, 60, 1);
        check_rate(120, 50, 5);
        check_rate(960, 50, 5);
        // 240->30: R = 8, offsets -4..+4, endpoints 1/16, interiors 1/8.
        let taps = window(r(240, 1), r(30, 1), r(80, 1)).unwrap();
        assert_eq!(relative(&taps, 80), trapezoid(4, r(1, 16), r(1, 8)));
        // 600->30: R = 20, offsets -10..+10, endpoints 1/40, interiors 1/20.
        let taps = window(r(600, 1), r(30, 1), r(200, 1)).unwrap();
        assert_eq!(relative(&taps, 200), trapezoid(10, r(1, 40), r(1, 20)));
        // 240->60: R = 4, offsets -2..+2, endpoints 1/8, interiors 1/4.
        let taps = window(r(240, 1), r(60, 1), r(40, 1)).unwrap();
        assert_eq!(relative(&taps, 40), trapezoid(2, r(1, 8), r(1, 4)));
        // 120->50: R = 12/5; output 1 has phase 2/5 and covers [6/5, 18/5].
        let taps = window(r(120, 1), r(50, 1), r(12, 5)).unwrap();
        assert_eq!(relative(&taps, 0), vec![(1, r(1, 8)), (2, r(5, 12)), (3, r(5, 12)), (4, r(1, 24))]);
    }

    #[test]
    fn ntsc_rates_are_exact() {
        let src = r(240000, 1001);
        let out = r(30000, 1001);
        assert_eq!(rate_ratio(src, out).unwrap(), r(8, 1));
        let taps = window(src, out, r(8, 1)).unwrap();
        assert_eq!(relative(&taps, 8), trapezoid(4, r(1, 16), r(1, 8)));
        // 240 fps source in a 29.97 timeline: R = 1001/125, period 125.
        check_rate_rational(r(240, 1), out, 125);
    }

    fn check_rate_rational(src: Rational, out: Rational, period: i64) {
        let ratio = rate_ratio(src, out).unwrap();
        assert_eq!(ratio.denom(), period);
        for n in 0..(2 * period) {
            let c = ratio * Rational::integer(n);
            assert_eq!(sum(&window(src, out, c).unwrap()), Rational::ONE);
        }
    }

    #[test]
    fn edge_renormalization_at_source_start() {
        // 960->30 centred on source frame 0: taps -16..-1 are unavailable.
        let taps = kernel(r(960, 1), r(30, 1), Rational::ZERO, Availability { first: 0, last: 38790 }).unwrap();
        // Remaining mass: 15 * 1/32 + 1/32 (frame 0) + 1/64 = 33/64.
        let expected: Vec<(i64, Rational)> = (0..=16)
            .map(|k| (k, if k == 16 { r(1, 64) / r(33, 64) } else { r(1, 32) / r(33, 64) }))
            .collect();
        assert_eq!(relative(&taps, 0), expected);
        assert_eq!(sum(&taps), Rational::ONE);
        assert_eq!(taps[0].weight, r(2, 33));
        assert_eq!(taps[16].weight, r(1, 33));
    }

    #[test]
    fn edge_renormalization_at_source_end() {
        // 240->30 with the source ending at frame 99: window of c = 98 is 94..102.
        let taps = kernel(r(240, 1), r(30, 1), r(98, 1), Availability { first: 0, last: 99 }).unwrap();
        // Kept: 94 (1/16), 95..99 (5 * 1/8) -> mass 11/16.
        assert_eq!(
            relative(&taps, 0),
            vec![(94, r(1, 11)), (95, r(2, 11)), (96, r(2, 11)), (97, r(2, 11)), (98, r(2, 11)), (99, r(2, 11))]
        );
        assert_eq!(sum(&taps), Rational::ONE);
        // Clip handles: frames outside the timeline in/out but inside the clip stay in use.
        let inside = kernel(r(240, 1), r(30, 1), r(98, 1), Availability { first: 0, last: 1000 }).unwrap();
        assert_eq!(inside, window(r(240, 1), r(30, 1), r(98, 1)).unwrap());
    }

    #[test]
    fn fully_unavailable_window_uses_nearest_available_frame() {
        let taps = kernel(r(240, 1), r(30, 1), r(200, 1), Availability { first: 0, last: 99 }).unwrap();
        assert_eq!(taps, vec![Tap { index: 99, weight: Rational::ONE }]);
        assert_eq!(
            kernel(r(240, 1), r(30, 1), r(0, 1), Availability { first: 5, last: 4 }),
            Err(KernelError::EmptyAvailability)
        );
    }

    #[test]
    fn output_rate_at_or_above_source_passes_through_nearest_frame() {
        for (src, out) in [(r(30, 1), r(30, 1)), (r(30, 1), r(60, 1)), (r(24000, 1001), r(24, 1))] {
            for n in 0..20 {
                let t = Rational::integer(n) / out;
                let c = center_at_time(t, src);
                let taps = kernel(src, out, c, Availability::UNBOUNDED).unwrap();
                assert_eq!(taps.len(), 1);
                assert_eq!(taps[0].weight, Rational::ONE);
                assert_eq!(taps[0].index, (c + Rational::HALF).floor());
            }
        }
        // 30 -> 60: output frames 0,1,2,3 show source frames 0,1,1,2 (ties round up).
        let picks: Vec<i64> = (0..4)
            .map(|n| kernel(r(30, 1), r(60, 1), r(n, 2), Availability::UNBOUNDED).unwrap()[0].index)
            .collect();
        assert_eq!(picks, vec![0, 1, 1, 2]);
        // Pass-through clamps to the available range.
        assert_eq!(kernel(r(30, 1), r(30, 1), r(-3, 1), Availability { first: 0, last: 9 }).unwrap()[0].index, 0);
        assert_eq!(max_taps(r(30, 1), r(60, 1)).unwrap(), 1);
    }

    #[test]
    fn rejects_non_positive_rates() {
        assert_eq!(window(r(0, 1), r(30, 1), Rational::ZERO), Err(KernelError::NonPositiveRate));
        assert_eq!(window(r(30, 1), r(-30, 1), Rational::ZERO), Err(KernelError::NonPositiveRate));
    }
}
