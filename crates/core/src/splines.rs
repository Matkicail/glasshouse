//! B-spline basis evaluation (Cox–de Boor). The one spline kernel: the GLM's spline terms
//! use it now; penalised smooths and monotone constraints will reuse it later.
//!
//! Given a full knot vector (boundary knots repeated `degree + 1` times, interior knots in
//! between) and a degree, each row of the design is the value of every basis function at
//! that `x`. Off the boundary the basis is evaluated at the clamped value — extrapolating a
//! polynomial tail silently is how spline models go wrong quietly, so we do the safe thing
//! and say so in the docs.

use crate::error::{all_values, GlassError};

/// Number of basis functions for a full knot vector and degree.
#[must_use]
pub fn n_basis(knots: &[f64], degree: usize) -> usize {
    knots.len().saturating_sub(degree + 1)
}

/// Evaluate the B-spline design matrix, row-major `x.len() x n_basis`.
///
/// # Errors
/// Non-finite `x`; a knot vector that is too short, unsorted, or has equal boundaries;
/// `degree` of 0 (use the raw column instead).
pub fn bspline_design(x: &[f64], knots: &[f64], degree: usize) -> Result<Vec<f64>, GlassError> {
    validate(x, knots, degree)?;
    let p = n_basis(knots, degree);
    let lo = knots[degree];
    let hi = knots[knots.len() - degree - 1];
    let mut out = vec![0.0; x.len() * p];
    let mut work = vec![0.0; p + degree + 1];
    for (row, &xv) in x.iter().enumerate() {
        let t = xv.clamp(lo, hi);
        cox_de_boor(t, knots, degree, &mut work);
        out[row * p..(row + 1) * p].copy_from_slice(&work[..p]);
    }
    Ok(out)
}

/// One row of the basis at `t`, written into `values[..n_basis]`.
///
/// The standard triangular recursion: start from the degree-0 indicator functions and build
/// up. `t` must already be inside `[knots[degree], knots[len - degree - 1]]`.
// Exact float comparison is the point: knot identity decides interval membership.
#[allow(clippy::float_cmp)]
fn cox_de_boor(t: f64, knots: &[f64], degree: usize, values: &mut [f64]) {
    let n = knots.len() - 1; // number of degree-0 functions
    values.fill(0.0);
    // degree 0: indicators. The last interval is closed on the right so t = hi belongs somewhere.
    let mut k = usize::MAX;
    for i in 0..n {
        if (knots[i] <= t && t < knots[i + 1])
            || (t == knots[i + 1] && knots[i] < knots[i + 1] && i + 1 == last_upper(knots))
        {
            k = i;
            break;
        }
    }
    if k == usize::MAX {
        return; // t outside every interval (cannot happen after clamping)
    }
    values[k] = 1.0;
    for d in 1..=degree {
        for i in k.saturating_sub(d)..=k {
            let left = {
                let den = knots[i + d] - knots[i];
                if den > 0.0 {
                    (t - knots[i]) / den * values[i]
                } else {
                    0.0
                }
            };
            let right = if i + d + 1 < knots.len() {
                let den = knots[i + d + 1] - knots[i + 1];
                if den > 0.0 && i + 1 < values.len() {
                    (knots[i + d + 1] - t) / den * values[i + 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            values[i] = left + right;
        }
    }
}

/// Index i such that `knots[i]` is the upper boundary's first occurrence minus one — the
/// last half-open interval that must absorb `t == hi`.
// Exact float comparison is the point: the boundary knot's first occurrence.
#[allow(clippy::float_cmp)]
fn last_upper(knots: &[f64]) -> usize {
    let hi = knots[knots.len() - 1];
    knots
        .iter()
        .position(|&v| v == hi)
        .unwrap_or(knots.len() - 1)
}

fn validate(x: &[f64], knots: &[f64], degree: usize) -> Result<(), GlassError> {
    if degree == 0 {
        return Err(GlassError::BadArgument {
            name: "degree",
            problem: "a degree-0 spline is a step function",
            fix: "use degree 3 (cubic), or bin the column instead",
        });
    }
    if knots.len() < 2 * (degree + 1) {
        return Err(GlassError::BadArgument {
            name: "knots",
            problem: "too few knots for this degree",
            fix: "the full knot vector needs the boundaries repeated degree + 1 times",
        });
    }
    all_values(
        "x",
        x,
        "must be finite",
        "NaN or inf cannot sit on a spline",
        f64::is_finite,
    )?;
    all_values(
        "knots",
        knots,
        "must be finite",
        "NaN or inf in the knot vector",
        f64::is_finite,
    )?;
    if knots.windows(2).any(|w| w[1] < w[0]) {
        return Err(GlassError::BadArgument {
            name: "knots",
            problem: "must be non-decreasing",
            fix: "sort the interior knots; boundaries first and last",
        });
    }
    let (lo, hi) = (knots[degree], knots[knots.len() - degree - 1]);
    if hi <= lo {
        return Err(GlassError::BadArgument {
            name: "knots",
            problem: "the boundary knots are equal",
            fix: "the column is constant on the training rows; drop it instead of splining it",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_knots(interior: &[f64], lo: f64, hi: f64, degree: usize) -> Vec<f64> {
        let mut k = vec![lo; degree + 1];
        k.extend_from_slice(interior);
        k.extend(std::iter::repeat_n(hi, degree + 1));
        k
    }

    #[test]
    fn partition_of_unity_and_local_support() {
        let knots = full_knots(&[0.3, 0.5, 0.8], 0.0, 1.0, 3);
        let x: Vec<f64> = (0..=100).map(|i| f64::from(i) / 100.0).collect();
        let p = n_basis(&knots, 3);
        let design = bspline_design(&x, &knots, 3).unwrap();
        for row in 0..x.len() {
            let s: f64 = design[row * p..(row + 1) * p].iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "row {row} sums to {s}");
            let nonzero = design[row * p..(row + 1) * p]
                .iter()
                .filter(|v| **v > 0.0)
                .count();
            assert!(nonzero <= 4, "cubic basis has at most 4 active functions");
        }
    }

    #[test]
    fn clamps_outside_the_boundaries() {
        let knots = full_knots(&[0.5], 0.0, 1.0, 3);
        let inside = bspline_design(&[0.0, 1.0], &knots, 3).unwrap();
        let outside = bspline_design(&[-5.0, 7.0], &knots, 3).unwrap();
        assert_eq!(inside, outside);
    }

    #[test]
    fn refuses_bad_knots() {
        assert!(bspline_design(&[0.5], &[0.0, 0.0, 1.0, 1.0], 3).is_err());
        let unsorted = full_knots(&[0.8, 0.3], 0.0, 1.0, 3);
        assert!(bspline_design(&[0.5], &unsorted, 3).is_err());
        let constant = full_knots(&[], 1.0, 1.0, 3);
        assert!(bspline_design(&[0.5], &constant, 3).is_err());
    }
}
