//! Tolerance comparator for validating Rust op outputs against the reference
//! trace (build-plan.md, step 2 and the "Parity bar" section).
//!
//! Comparisons use a tight relative/absolute tolerance rather than
//! bit-exactness: the int8 GEMM path and threaded reductions make exact
//! matching brittle, but a tight epsilon still surfaces real bugs. An element
//! is "close" when
//!
//! ```text
//! |actual - expected| <= atol + rtol * |expected|
//! ```
//!
//! which is the same rule as numpy's `allclose`.

use std::fmt;

/// Relative and absolute tolerances for a comparison.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub rtol: f32,
    pub atol: f32,
}

impl Tolerance {
    pub fn new(rtol: f32, atol: f32) -> Tolerance {
        Tolerance { rtol, atol }
    }
}

impl Default for Tolerance {
    /// A tight default: loose enough to absorb reduction-order differences,
    /// tight enough that a genuine op bug still trips it.
    fn default() -> Tolerance {
        Tolerance {
            rtol: 1e-3,
            atol: 1e-5,
        }
    }
}

/// The worst single element in a comparison.
#[derive(Clone, Copy, Debug)]
pub struct Mismatch {
    pub index: usize,
    pub actual: f32,
    pub expected: f32,
    pub abs_err: f32,
    pub rel_err: f32,
}

/// Summary of comparing two equal-length `f32` slices.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub len: usize,
    pub num_mismatched: usize,
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    /// The first element (in order) that exceeded tolerance, if any.
    pub first_mismatch: Option<Mismatch>,
    tol: Tolerance,
}

impl Comparison {
    /// Whether every element was within tolerance.
    pub fn all_close(&self) -> bool {
        self.num_mismatched == 0
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} elements exceed tolerance (rtol={:e}, atol={:e}); \
             max abs err {:e}, max rel err {:e}",
            self.num_mismatched,
            self.len,
            self.tol.rtol,
            self.tol.atol,
            self.max_abs_err,
            self.max_rel_err
        )?;
        if let Some(m) = self.first_mismatch {
            write!(
                f,
                "; first at [{}]: actual={}, expected={} (abs {:e}, rel {:e})",
                m.index, m.actual, m.expected, m.abs_err, m.rel_err
            )?;
        }
        Ok(())
    }
}

/// Error returned when two slices cannot be compared at all.
#[derive(Debug)]
pub enum CompareError {
    /// The slices had different lengths.
    LengthMismatch { actual: usize, expected: usize },
}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareError::LengthMismatch { actual, expected } => write!(
                f,
                "length mismatch: actual has {actual} elements, expected has {expected}"
            ),
        }
    }
}

impl std::error::Error for CompareError {}

/// Are two values within tolerance? Handles NaN (only NaN≈NaN) and infinities
/// (equal only when identical, including sign).
fn is_close(actual: f32, expected: f32, tol: Tolerance) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if actual.is_infinite() || expected.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= tol.atol + tol.rtol * expected.abs()
}

/// Compare two `f32` slices element-wise under `tol`, returning error
/// statistics. Errors only when the lengths differ; otherwise inspect
/// [`Comparison::all_close`].
pub fn compare_f32(
    actual: &[f32],
    expected: &[f32],
    tol: Tolerance,
) -> Result<Comparison, CompareError> {
    if actual.len() != expected.len() {
        return Err(CompareError::LengthMismatch {
            actual: actual.len(),
            expected: expected.len(),
        });
    }

    let mut num_mismatched = 0;
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut first_mismatch = None;

    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Skip non-finite pairs from the error magnitudes (they would poison
        // the max), but still count them as mismatched when they disagree.
        if a.is_finite() && e.is_finite() {
            let abs_err = (a - e).abs();
            let rel_err = if e != 0.0 { abs_err / e.abs() } else { abs_err };
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
        }

        if !is_close(a, e, tol) {
            num_mismatched += 1;
            if first_mismatch.is_none() {
                let abs_err = (a - e).abs();
                let rel_err = if e != 0.0 { abs_err / e.abs() } else { abs_err };
                first_mismatch = Some(Mismatch {
                    index,
                    actual: a,
                    expected: e,
                    abs_err,
                    rel_err,
                });
            }
        }
    }

    Ok(Comparison {
        len: actual.len(),
        num_mismatched,
        max_abs_err,
        max_rel_err,
        first_mismatch,
        tol,
    })
}

/// Assert that `actual` matches `expected` within `tol`, panicking with a
/// detailed diff otherwise. Convenience for op-level fixture tests.
#[track_caller]
pub fn assert_close(actual: &[f32], expected: &[f32], tol: Tolerance) {
    match compare_f32(actual, expected, tol) {
        Err(e) => panic!("{e}"),
        Ok(cmp) if !cmp.all_close() => panic!("{cmp}"),
        Ok(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_are_close() {
        let cmp = compare_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], Tolerance::default()).unwrap();
        assert!(cmp.all_close());
        assert_eq!(cmp.num_mismatched, 0);
        assert_eq!(cmp.max_abs_err, 0.0);
    }

    #[test]
    fn small_differences_within_tolerance() {
        // Within rtol=1e-3 of 1000.0.
        let cmp = compare_f32(&[1000.5], &[1000.0], Tolerance::default()).unwrap();
        assert!(cmp.all_close());
    }

    #[test]
    fn large_differences_flagged() {
        let cmp = compare_f32(&[1.0, 5.0], &[1.0, 2.0], Tolerance::default()).unwrap();
        assert!(!cmp.all_close());
        assert_eq!(cmp.num_mismatched, 1);
        let first = cmp.first_mismatch.unwrap();
        assert_eq!(first.index, 1);
        assert_eq!(first.expected, 2.0);
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(matches!(
            compare_f32(&[1.0], &[1.0, 2.0], Tolerance::default()),
            Err(CompareError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn nan_only_matches_nan() {
        let tol = Tolerance::default();
        assert!(is_close(f32::NAN, f32::NAN, tol));
        assert!(!is_close(f32::NAN, 1.0, tol));
        assert!(!is_close(1.0, f32::NAN, tol));
    }

    #[test]
    fn infinities_match_by_sign() {
        let tol = Tolerance::default();
        assert!(is_close(f32::INFINITY, f32::INFINITY, tol));
        assert!(!is_close(f32::INFINITY, f32::NEG_INFINITY, tol));
        assert!(!is_close(f32::INFINITY, 1e30, tol));
    }

    #[test]
    #[should_panic(expected = "exceed tolerance")]
    fn assert_close_panics_on_mismatch() {
        assert_close(&[1.0], &[2.0], Tolerance::default());
    }
}
