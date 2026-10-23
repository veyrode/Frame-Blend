//! Exact rational numbers for frame rates, frame positions and kernel weights.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;

/// A reduced fraction with a positive denominator.
///
/// Arithmetic uses 128-bit intermediates and panics if a reduced result does
/// not fit in 64 bits; realistic frame rates and positions stay far below that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i64,
    den: i64,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Rational {
    pub const ZERO: Rational = Rational { num: 0, den: 1 };
    pub const ONE: Rational = Rational { num: 1, den: 1 };
    pub const HALF: Rational = Rational { num: 1, den: 2 };

    /// Returns `None` for a zero denominator or a value outside 64-bit range.
    pub fn new(num: i64, den: i64) -> Option<Rational> {
        Self::from_i128(num as i128, den as i128)
    }

    fn from_i128(num: i128, den: i128) -> Option<Rational> {
        if den == 0 {
            return None;
        }
        let g = gcd(num, den).max(1);
        let (mut n, mut d) = (num / g, den / g);
        if d < 0 {
            n = -n;
            d = -d;
        }
        Some(Rational {
            num: i64::try_from(n).ok()?,
            den: i64::try_from(d).ok()?,
        })
    }

    fn reduced(num: i128, den: i128) -> Rational {
        Self::from_i128(num, den).expect("rational arithmetic overflowed 64 bits")
    }

    pub fn integer(value: i64) -> Rational {
        Rational { num: value, den: 1 }
    }

    pub fn numer(self) -> i64 {
        self.num
    }

    pub fn denom(self) -> i64 {
        self.den
    }

    pub fn is_positive(self) -> bool {
        self.num > 0
    }

    pub fn floor(self) -> i64 {
        self.num.div_euclid(self.den)
    }

    pub fn ceil(self) -> i64 {
        -((-self.num).div_euclid(self.den))
    }

    /// Fractional part in [0, 1).
    pub fn fract(self) -> Rational {
        self - Rational::integer(self.floor())
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn min(self, other: Rational) -> Rational {
        if self <= other { self } else { other }
    }

    pub fn max(self, other: Rational) -> Rational {
        if self >= other { self } else { other }
    }

    /// Recovers the exact rate behind a floating-point frame rate reported by a
    /// host (for example 29.97002997 -> 30000/1001). Integer and NTSC (k*1000/1001)
    /// rates are matched first; otherwise the closest fraction with denominator
    /// at most 1001 within 1e-9 relative error is returned.
    pub fn from_frame_rate_f64(fps: f64) -> Option<Rational> {
        if !fps.is_finite() || fps <= 0.0 || fps > 1.0e7 {
            return None;
        }
        let tolerance = fps * 1.0e-7;
        let rounded = fps.round();
        if rounded >= 1.0 && (fps - rounded).abs() <= tolerance {
            return Some(Rational::integer(rounded as i64));
        }
        let ntsc_k = (fps * 1001.0 / 1000.0).round();
        if ntsc_k >= 1.0 && (fps - ntsc_k * 1000.0 / 1001.0).abs() <= tolerance {
            return Rational::new(ntsc_k as i64 * 1000, 1001);
        }
        // Continued-fraction convergents with a bounded denominator.
        let (mut h0, mut h1, mut k0, mut k1) = (0i64, 1i64, 1i64, 0i64);
        let mut x = fps;
        for _ in 0..32 {
            let a = x.floor();
            let a_i = a as i64;
            let h2 = a_i.checked_mul(h1)?.checked_add(h0)?;
            let k2 = a_i.checked_mul(k1)?.checked_add(k0)?;
            if k2 > 1001 {
                break;
            }
            h0 = h1;
            h1 = h2;
            k0 = k1;
            k1 = k2;
            if (fps - h1 as f64 / k1 as f64).abs() <= fps * 1.0e-9 {
                return Rational::new(h1, k1);
            }
            let rem = x - a;
            if rem <= 0.0 {
                break;
            }
            x = 1.0 / rem;
        }
        None
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.num as i128 * other.den as i128).cmp(&(other.num as i128 * self.den as i128))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Add for Rational {
    type Output = Rational;
    fn add(self, rhs: Rational) -> Rational {
        Rational::reduced(
            self.num as i128 * rhs.den as i128 + rhs.num as i128 * self.den as i128,
            self.den as i128 * rhs.den as i128,
        )
    }
}

impl Sub for Rational {
    type Output = Rational;
    fn sub(self, rhs: Rational) -> Rational {
        self + (-rhs)
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational { num: -self.num, den: self.den }
    }
}

impl Mul for Rational {
    type Output = Rational;
    fn mul(self, rhs: Rational) -> Rational {
        Rational::reduced(self.num as i128 * rhs.num as i128, self.den as i128 * rhs.den as i128)
    }
}

impl Div for Rational {
    type Output = Rational;
    fn div(self, rhs: Rational) -> Rational {
        assert!(rhs.num != 0, "rational division by zero");
        Rational::reduced(self.num as i128 * rhs.den as i128, self.den as i128 * rhs.num as i128)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Parses `a`, `a/b` or an exact decimal such as `11.2` or `-0.125`.
impl FromStr for Rational {
    type Err = String;
    fn from_str(text: &str) -> Result<Rational, String> {
        let text = text.trim();
        let bad = || format!("not an exact rational: {text:?}");
        if let Some((n, d)) = text.split_once('/') {
            let n: i64 = n.trim().parse().map_err(|_| bad())?;
            let d: i64 = d.trim().parse().map_err(|_| bad())?;
            return Rational::new(n, d).ok_or_else(bad);
        }
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (whole, frac) = digits.split_once('.').unwrap_or((digits, ""));
        if (whole.is_empty() && frac.is_empty())
            || !whole.chars().all(|c| c.is_ascii_digit())
            || !frac.chars().all(|c| c.is_ascii_digit())
            || frac.len() > 18
        {
            return Err(bad());
        }
        let scale = 10i128.pow(frac.len() as u32);
        let whole: i128 = if whole.is_empty() { 0 } else { whole.parse().map_err(|_| bad())? };
        let frac_value: i128 = if frac.is_empty() { 0 } else { frac.parse().map_err(|_| bad())? };
        let num = whole.checked_mul(scale).and_then(|v| v.checked_add(frac_value)).ok_or_else(bad)?;
        Rational::from_i128(if negative { -num } else { num }, scale).ok_or_else(bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    #[test]
    fn reduces_and_orders() {
        assert_eq!(r(6, -4), r(-3, 2));
        assert_eq!(r(-3, 2).floor(), -2);
        assert_eq!(r(-3, 2).ceil(), -1);
        assert_eq!(r(7, 2).fract(), r(1, 2));
        assert!(r(1, 3) < r(1, 2));
        assert_eq!(r(1, 3) + r(1, 6), r(1, 2));
        assert_eq!(r(96, 5) * r(5, 1), r(96, 1));
        assert_eq!(r(960, 1) / r(50, 1), r(96, 5));
    }

    #[test]
    fn parses_exact_decimals_and_fractions() {
        assert_eq!("11.2".parse::<Rational>().unwrap(), r(56, 5));
        assert_eq!("30000/1001".parse::<Rational>().unwrap(), r(30000, 1001));
        assert_eq!("-0.125".parse::<Rational>().unwrap(), r(-1, 8));
        assert_eq!("10".parse::<Rational>().unwrap(), r(10, 1));
        assert!("1e3".parse::<Rational>().is_err());
        assert!("1/0".parse::<Rational>().is_err());
    }

    #[test]
    fn recovers_host_frame_rates() {
        assert_eq!(Rational::from_frame_rate_f64(30.0), Some(r(30, 1)));
        assert_eq!(Rational::from_frame_rate_f64(30000.0 / 1001.0), Some(r(30000, 1001)));
        assert_eq!(Rational::from_frame_rate_f64(29.97002997), Some(r(30000, 1001)));
        assert_eq!(Rational::from_frame_rate_f64(59.94005994005994), Some(r(60000, 1001)));
        assert_eq!(Rational::from_frame_rate_f64(23.976023976), Some(r(24000, 1001)));
        assert_eq!(Rational::from_frame_rate_f64(600.0), Some(r(600, 1)));
        assert_eq!(Rational::from_frame_rate_f64(12.5), Some(r(25, 2)));
        assert_eq!(Rational::from_frame_rate_f64(0.0), None);
        assert_eq!(Rational::from_frame_rate_f64(f64::NAN), None);
    }
}
