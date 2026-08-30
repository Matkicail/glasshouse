//! Plain regression errors, weighted. "How far off, on average?"
//!
//! These are the metrics everyone knows; the deviances in `metrics` are the ones a GLM
//! actually minimises. On skewed, heavy-tailed targets these are dominated by a handful of
//! rows — read them next to the family deviance, not instead of it.

use crate::error::{all_values, same_length, GlassError};

/// Which error to average.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionMetric {
    /// Root of the weighted mean squared error.
    Rmse,
    /// Weighted mean absolute error.
    Mae,
    /// Weighted mean absolute percentage error `|y - mu| / |y|`. Needs `y != 0`.
    Mape,
    /// Symmetric MAPE `2 |y - mu| / (|y| + |mu|)`, in `[0, 2]`. Needs `|y| + |mu| != 0`.
    Smape,
    /// Weighted mean squared log error `(ln(1 + y) - ln(1 + mu))^2`. Needs `y, mu > -1`.
    Msle,
    /// Coefficient of determination `1 - SSE / SST` (weighted). 1 is perfect, 0 is the mean.
    R2,
}

/// Compute one regression metric.
///
/// # Errors
/// Lengths differ; non-finite values; bad weights; a row outside the metric's domain (zero
/// `y` for MAPE, `y <= -1` for MSLE, constant `y` for R²).
pub fn regression(
    metric: RegressionMetric,
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
) -> Result<f64, GlassError> {
    validate(metric, y, mu, w)?;
    let weight = |i: usize| w.map_or(1.0, |w| w[i]);
    let total_w: f64 = (0..y.len()).map(weight).sum();
    let mean = |f: &dyn Fn(f64, f64) -> f64| -> f64 {
        y.iter()
            .zip(mu)
            .enumerate()
            .map(|(i, (&yi, &mi))| weight(i) * f(yi, mi))
            .sum::<f64>()
            / total_w
    };
    Ok(match metric {
        RegressionMetric::Rmse => mean(&|a, b| (a - b) * (a - b)).sqrt(),
        RegressionMetric::Mae => mean(&|a, b| (a - b).abs()),
        RegressionMetric::Mape => mean(&|a, b| (a - b).abs() / a.abs()),
        RegressionMetric::Smape => mean(&|a, b| 2.0 * (a - b).abs() / (a.abs() + b.abs())),
        RegressionMetric::Msle => mean(&|a, b| {
            let d = a.ln_1p() - b.ln_1p();
            d * d
        }),
        RegressionMetric::R2 => {
            let ybar = y
                .iter()
                .enumerate()
                .map(|(i, &v)| weight(i) * v)
                .sum::<f64>()
                / total_w;
            let sse = mean(&|a, b| (a - b) * (a - b));
            let sst = mean(&|a, _| (a - ybar) * (a - ybar));
            1.0 - sse / sst
        }
    })
}

fn validate(
    metric: RegressionMetric,
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
) -> Result<(), GlassError> {
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    same_length("y", y, "mu", mu)?;
    all_values(
        "y",
        y,
        "must be finite",
        "NaN or inf in the outcome",
        f64::is_finite,
    )?;
    all_values(
        "mu",
        mu,
        "must be finite",
        "NaN or inf in the predictions",
        f64::is_finite,
    )?;
    if let Some(w) = w {
        same_length("y", y, "sample_weight", w)?;
        all_values(
            "sample_weight",
            w,
            "must be finite and >= 0",
            "negative or NaN weights are not weights",
            |v| v.is_finite() && v >= 0.0,
        )?;
        if w.iter().sum::<f64>() <= 0.0 {
            return Err(GlassError::InvalidValues {
                name: "sample_weight",
                count: w.len(),
                rule: "must sum to more than zero",
                fix: "every weight is zero, so there is nothing to average",
            });
        }
    }
    match metric {
        RegressionMetric::Mape => all_values(
            "y",
            y,
            "must be non-zero for MAPE",
            "a percentage error of a zero outcome is infinite; use sMAPE or MAE",
            |v| v != 0.0,
        ),
        RegressionMetric::Smape => {
            let bad = y
                .iter()
                .zip(mu)
                .filter(|(&a, &b)| a.abs() + b.abs() == 0.0)
                .count();
            if bad > 0 {
                return Err(GlassError::InvalidValues {
                    name: "y",
                    count: bad,
                    rule: "and mu must not both be zero for sMAPE",
                    fix: "0/0 is undefined; drop those rows or use MAE",
                });
            }
            Ok(())
        }
        RegressionMetric::Msle => {
            all_values(
                "y",
                y,
                "must be > -1 for MSLE",
                "ln(1 + y) needs y > -1; MSLE is meant for non-negative targets",
                |v| v > -1.0,
            )?;
            all_values(
                "mu",
                mu,
                "must be > -1 for MSLE",
                "ln(1 + mu) needs mu > -1",
                |v| v > -1.0,
            )
        }
        RegressionMetric::R2 => {
            // "constant" means exactly equal: SST would be exactly zero.
            #[allow(clippy::float_cmp)]
            let constant = y.iter().all(|&v| v == y[0]);
            if constant {
                return Err(GlassError::InvalidValues {
                    name: "y",
                    count: y.len(),
                    rule: "must vary for R² to mean anything",
                    fix: "y is constant, so there is no variance to explain",
                });
            }
            Ok(())
        }
        RegressionMetric::Rmse | RegressionMetric::Mae => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_values() {
        let y = [1.0, 2.0, 4.0];
        let mu = [1.0, 3.0, 2.0];
        let r = |m| regression(m, &y, &mu, None).unwrap();
        assert!((r(RegressionMetric::Mae) - 1.0).abs() < 1e-15);
        assert!((r(RegressionMetric::Rmse) - (5.0f64 / 3.0).sqrt()).abs() < 1e-15);
        assert!((r(RegressionMetric::Mape) - (0.5 + 0.5) / 3.0).abs() < 1e-15);
        assert!((r(RegressionMetric::Smape) - (0.4 + 2.0 / 3.0) / 3.0).abs() < 1e-15);
        // mean 7/3, sst = (16/9 + 1/9 + 25/9)/3 = 14/9, sse = 5/3
        assert!((r(RegressionMetric::R2) - (1.0 - (5.0 / 3.0) / (14.0 / 9.0))).abs() < 1e-14);
    }

    #[test]
    fn refuses_domain_errors() {
        assert!(regression(RegressionMetric::Mape, &[0.0, 1.0], &[1.0, 1.0], None).is_err());
        assert!(regression(RegressionMetric::Smape, &[0.0, 1.0], &[0.0, 1.0], None).is_err());
        assert!(regression(RegressionMetric::Msle, &[-2.0, 1.0], &[1.0, 1.0], None).is_err());
        assert!(regression(RegressionMetric::R2, &[1.0, 1.0], &[1.0, 2.0], None).is_err());
    }
}
