//! Deviance-based metrics. Weighted, exposure-aware, and the same formulas the fitters use.
//!
//! Notation: `y` observed, `mu` predicted mean, `w` sample weights (`None` = all ones).
//! Mean deviance is `sum(w_i * d(y_i, mu_i)) / sum(w_i)` — the convention scikit-learn's
//! `mean_tweedie_deviance` uses, so the golden tests compare like with like.

use crate::error::{all_values, same_length, GlassError};
use crate::family::Family;

/// Weighted mean deviance of `mu` against `y` under `family`.
///
/// # Errors
/// Lengths differ; `y` or `mu` outside the family's support or non-finite; negative, non-finite
/// or all-zero weights; empty input.
pub fn deviance(
    family: Family,
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
) -> Result<f64, GlassError> {
    validate(family, y, mu, w)?;
    let unit = |yi: f64, mi: f64| family.unit_deviance(yi, mi);
    let (total, weight_sum) = match w {
        None => {
            let total: f64 = y.iter().zip(mu).map(|(&yi, &mi)| unit(yi, mi)).sum();
            #[allow(clippy::cast_precision_loss)]
            let n = y.len() as f64;
            (total, n)
        }
        Some(w) => {
            let total: f64 = y
                .iter()
                .zip(mu)
                .zip(w)
                .map(|((&yi, &mi), &wi)| wi * unit(yi, mi))
                .sum();
            (total, w.iter().sum())
        }
    };
    Ok(total / weight_sum)
}

/// D², "deviance explained": `1 - deviance(y, mu) / deviance(y, weighted mean of y)`.
///
/// 1 is perfect, 0 is "no better than predicting the mean", negative is worse than the mean.
/// This is the family-consistent pseudo-R²; scikit-learn calls it `d2_tweedie_score`.
///
/// # Errors
/// Everything [`deviance`] rejects, plus a constant `y` (the null deviance is 0, so D² is
/// undefined — there is nothing to explain).
pub fn d2(family: Family, y: &[f64], mu: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    let fitted = deviance(family, y, mu, w)?;
    let ybar = weighted_mean(y, w);
    let null: f64 = match w {
        None => {
            #[allow(clippy::cast_precision_loss)]
            let n = y.len() as f64;
            y.iter()
                .map(|&yi| family.unit_deviance(yi, ybar))
                .sum::<f64>()
                / n
        }
        Some(w) => {
            y.iter()
                .zip(w)
                .map(|(&yi, &wi)| wi * family.unit_deviance(yi, ybar))
                .sum::<f64>()
                / w.iter().sum::<f64>()
        }
    };
    if null <= 0.0 {
        return Err(GlassError::InvalidValues {
            name: "y",
            count: y.len(),
            rule: "must vary for D² to mean anything",
            fix: "y is constant, so the null deviance is 0 and there is nothing to explain",
        });
    }
    Ok(1.0 - fitted / null)
}

fn weighted_mean(y: &[f64], w: Option<&[f64]>) -> f64 {
    match w {
        None => {
            #[allow(clippy::cast_precision_loss)]
            let n = y.len() as f64;
            y.iter().sum::<f64>() / n
        }
        Some(w) => y.iter().zip(w).map(|(&yi, &wi)| yi * wi).sum::<f64>() / w.iter().sum::<f64>(),
    }
}

/// The one validation path: lengths, supports, weights. Fails early and clearly.
fn validate(family: Family, y: &[f64], mu: &[f64], w: Option<&[f64]>) -> Result<(), GlassError> {
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    same_length("y", y, "mu", mu)?;
    all_values(
        "y",
        y,
        "must be finite and inside the family's support",
        family.y_rule(),
        |v| v.is_finite() && family.accepts_y(v),
    )?;
    all_values(
        "mu",
        mu,
        "must be finite and a valid mean for the family",
        family.mu_rule(),
        |v| v.is_finite() && family.accepts_mu(v),
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Family; 5] = [
        Family::Gaussian,
        Family::Poisson,
        Family::Gamma,
        Family::Tweedie { power: 1.5 },
        Family::Binomial,
    ];

    #[test]
    fn zero_at_perfect_fit() {
        let mu = [0.2, 0.5, 0.9];
        for f in ALL {
            assert!(
                deviance(f, &mu, &mu, None).unwrap().abs() < 1e-15,
                "{}",
                f.name()
            );
        }
    }

    #[test]
    fn denormal_y_is_finite() {
        let d = deviance(Family::Poisson, &[5e-324], &[2.0], None).unwrap();
        assert!(d.is_finite() && d >= 0.0, "{d}");
    }

    #[test]
    fn weights_scale_invariant() {
        let y = [0.1, 0.4, 0.6, 0.9];
        let mu = [0.2, 0.3, 0.7, 0.8];
        let w = [1.0, 2.0, 0.5, 4.0];
        let w7 = w.map(|v| v * 7.0);
        for f in ALL {
            let a = deviance(f, &y, &mu, Some(&w)).unwrap();
            let b = deviance(f, &y, &mu, Some(&w7)).unwrap();
            assert!((a - b).abs() < 1e-12, "{}", f.name());
        }
    }

    #[test]
    fn d2_is_one_at_perfect_fit_and_zero_at_mean() {
        let y = [1.0, 2.0, 3.0, 6.0];
        let mean = [3.0; 4];
        for f in ALL.iter().filter(|f| **f != Family::Binomial) {
            assert!(
                (d2(*f, &y, &y, None).unwrap() - 1.0).abs() < 1e-12,
                "{}",
                f.name()
            );
            assert!(
                d2(*f, &y, &mean, None).unwrap().abs() < 1e-12,
                "{}",
                f.name()
            );
        }
    }

    #[test]
    fn d2_refuses_constant_y() {
        let err = d2(Family::Poisson, &[2.0, 2.0], &[1.0, 3.0], None).unwrap_err();
        assert!(err.to_string().contains("constant"), "{err}");
    }

    #[test]
    fn refuses_bad_input_with_reasons() {
        let err = deviance(Family::Poisson, &[1.0, -1.0], &[1.0, 1.0], None).unwrap_err();
        assert!(err.to_string().contains("1 row(s)"), "{err}");
        let err = deviance(Family::Poisson, &[1.0], &[1.0, 1.0], None).unwrap_err();
        assert!(matches!(err, GlassError::LengthMismatch { .. }));
        let err = deviance(Family::Gamma, &[0.0], &[1.0], None).unwrap_err();
        assert!(err.to_string().contains("tweedie"), "{err}");
        let err = deviance(Family::Binomial, &[1.0], &[1.0], None).unwrap_err();
        assert!(err.to_string().contains("probabilities"), "{err}");
    }
}
