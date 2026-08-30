//! Deviance-based metrics. Weighted, exposure-aware, and the same formulas the fitters use.
//!
//! Notation: `y` observed, `mu` predicted mean, `w` sample weights (`None` = all ones).
//! Mean deviance is `sum(w_i * d(y_i, mu_i)) / sum(w_i)` — the convention scikit-learn's
//! `mean_poisson_deviance` uses, so the golden tests compare like with like.

use crate::error::{all_values, same_length, GlassError};

/// Poisson unit deviance `2 * (y * ln(y / mu) - (y - mu))`, with `y ln y -> 0` as `y -> 0`.
///
/// Computed as `y * (ln y - ln mu)` rather than `y * ln(y / mu)`: for denormal `y` the ratio
/// underflows to 0 and `ln(0) = -inf` (found by hypothesis on day one).
#[inline]
#[must_use]
pub fn poisson_unit_deviance(y: f64, mu: f64) -> f64 {
    let ylog = if y > 0.0 { y * (y.ln() - mu.ln()) } else { 0.0 };
    2.0 * (ylog - (y - mu))
}

/// Weighted mean Poisson deviance.
///
/// # Errors
/// - lengths differ
/// - any `y < 0`, any `mu <= 0`, any `w < 0`, or a non-finite value
/// - empty input, or all weights zero
pub fn poisson_deviance(y: &[f64], mu: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    weighted_mean_deviance("poisson", y, mu, w, poisson_unit_deviance, |v| v >= 0.0)
}

/// The shared weighted-mean-of-unit-deviances path. One implementation for every family.
fn weighted_mean_deviance(
    family: &'static str,
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
    unit: impl Fn(f64, f64) -> f64,
    y_ok: impl Fn(f64) -> bool,
) -> Result<f64, GlassError> {
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    same_length("y", y, "mu", mu)?;
    all_values(
        "y",
        y,
        "must be finite and inside the family's support",
        match family {
            "poisson" => "poisson needs y >= 0: check for negatives or NaN",
            _ => "check the family's support in docs/methods",
        },
        |v| v.is_finite() && y_ok(v),
    )?;
    all_values(
        "mu",
        mu,
        "must be finite and > 0",
        "predictions on the mean scale must be positive; did you pass the linear predictor?",
        |v| v.is_finite() && v > 0.0,
    )?;

    let (total, weight_sum) = match w {
        None => {
            let total: f64 = y.iter().zip(mu).map(|(&yi, &mi)| unit(yi, mi)).sum();
            #[allow(clippy::cast_precision_loss)]
            let n = y.len() as f64;
            (total, n)
        }
        Some(w) => {
            same_length("y", y, "sample_weight", w)?;
            all_values(
                "sample_weight",
                w,
                "must be finite and >= 0",
                "negative or NaN weights are not weights",
                |v| v.is_finite() && v >= 0.0,
            )?;
            let total: f64 = y
                .iter()
                .zip(mu)
                .zip(w)
                .map(|((&yi, &mi), &wi)| wi * unit(yi, mi))
                .sum();
            (total, w.iter().sum())
        }
    };
    if weight_sum <= 0.0 {
        return Err(GlassError::InvalidValues {
            name: "sample_weight",
            count: y.len(),
            rule: "must sum to more than zero",
            fix: "every weight is zero, so there is nothing to average",
        });
    }
    Ok(total / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_at_perfect_fit() {
        let y = [0.0, 1.0, 2.0, 3.5];
        assert!(poisson_deviance(&y, &y.map(|v| v.max(1e-12)), None).unwrap() < 1e-9);
        let mu = [0.5, 1.0, 2.0, 3.5];
        assert!(poisson_deviance(&mu, &mu, None).unwrap().abs() < 1e-12);
    }

    #[test]
    fn denormal_y_is_finite() {
        let d = poisson_deviance(&[5e-324], &[2.0], None).unwrap();
        assert!(d.is_finite() && d >= 0.0, "{d}");
    }

    #[test]
    fn weights_scale_invariant() {
        let y = [0.0, 1.0, 2.0, 3.0];
        let mu = [0.5, 1.2, 1.8, 3.3];
        let w = [1.0, 2.0, 0.5, 4.0];
        let w2 = w.map(|v| v * 7.0);
        let a = poisson_deviance(&y, &mu, Some(&w)).unwrap();
        let b = poisson_deviance(&y, &mu, Some(&w2)).unwrap();
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_input_with_reasons() {
        let err = poisson_deviance(&[1.0, -1.0], &[1.0, 1.0], None).unwrap_err();
        assert!(err.to_string().contains("1 row(s)"), "{err}");
        let err = poisson_deviance(&[1.0], &[1.0, 1.0], None).unwrap_err();
        assert!(matches!(err, GlassError::LengthMismatch { .. }));
        let err = poisson_deviance(&[1.0], &[0.0], None).unwrap_err();
        assert!(err.to_string().contains("mu"), "{err}");
    }
}
