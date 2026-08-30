//! The family table. Each row knows its unit deviance and which `y` it accepts.
//!
//! Adding a distribution means adding a row here — nothing else in the metrics moves.
//! The actuarial families (Poisson, gamma, Tweedie) are rows like any other.

use crate::error::GlassError;

/// An exponential-dispersion family. Selects the deviance formula and the support of `y`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Squared error. `y` any real number.
    Gaussian,
    /// Counts and rates. `y >= 0`.
    Poisson,
    /// Positive, right-skewed amounts (severity). `y > 0`.
    Gamma,
    /// Compound Poisson–gamma for `1 < p < 2` (pure premium); `p = 0, 1, 2` are gaussian,
    /// poisson, gamma; `p < 0` and `p >= 2` are allowed as in scikit-learn; `0 < p < 1` is not a
    /// valid Tweedie power.
    Tweedie {
        /// The variance power `p` in `Var(Y) = phi * mu^p`.
        power: f64,
    },
    /// Binary outcomes or proportions. `0 <= y <= 1`.
    Binomial,
}

impl Family {
    /// Look a family up by name. `power` is only read for `"tweedie"`.
    ///
    /// # Errors
    /// Unknown name, missing power for tweedie, or a power in `(0, 1)`.
    pub fn parse(name: &str, power: Option<f64>) -> Result<Self, GlassError> {
        match name {
            "gaussian" | "normal" => Ok(Self::Gaussian),
            "poisson" => Ok(Self::Poisson),
            "gamma" => Ok(Self::Gamma),
            "binomial" | "bernoulli" => Ok(Self::Binomial),
            "tweedie" => match power {
                None => Err(GlassError::BadArgument {
                    name: "power",
                    problem: "tweedie needs a variance power",
                    fix: "pass power=1.5 for pure premium, or 0/1/2 for gaussian/poisson/gamma",
                }),
                Some(p) if (0.0 < p && p < 1.0) || !p.is_finite() => Err(GlassError::BadArgument {
                    name: "power",
                    problem: "no Tweedie distribution exists for 0 < power < 1",
                    fix: "use power <= 0, power = 1, or power >= 1 (1.5 is the usual choice)",
                }),
                Some(p) => Ok(Self::Tweedie { power: p }),
            },
            _ => Err(GlassError::BadArgument {
                name: "family",
                problem: "unknown family",
                fix: "one of: gaussian, poisson, gamma, tweedie, binomial",
            }),
        }
    }

    /// The name, for error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Poisson => "poisson",
            Self::Gamma => "gamma",
            Self::Tweedie { .. } => "tweedie",
            Self::Binomial => "binomial",
        }
    }

    /// Is `y` inside this family's support? (Finite is checked separately.)
    #[must_use]
    pub fn accepts_y(self, y: f64) -> bool {
        match self {
            Self::Gaussian => true,
            Self::Poisson => y >= 0.0,
            Self::Gamma => y > 0.0,
            Self::Tweedie { power } => match power {
                p if p <= 0.0 => true,
                p if p < 2.0 => y >= 0.0,
                _ => y > 0.0,
            },
            Self::Binomial => (0.0..=1.0).contains(&y),
        }
    }

    /// Plain-English rule for `y`, used in the error when `accepts_y` fails.
    #[must_use]
    pub fn y_rule(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian accepts any finite y",
            Self::Poisson => "poisson needs y >= 0: check for negatives or NaN",
            Self::Gamma => {
                "gamma needs y > 0: zeros belong to tweedie(power in (1, 2)) or a two-part model"
            }
            Self::Tweedie { power } => match power {
                p if p <= 0.0 => "tweedie with power <= 0 accepts any finite y",
                p if p < 2.0 => "tweedie with power in [1, 2) needs y >= 0",
                _ => "tweedie with power >= 2 needs y > 0",
            },
            Self::Binomial => "binomial needs 0 <= y <= 1 (a label or a proportion)",
        }
    }

    /// Is `mu` a valid mean for this family?
    #[must_use]
    pub fn accepts_mu(self, mu: f64) -> bool {
        match self {
            Self::Gaussian | Self::Tweedie { power: 0.0 } => true,
            Self::Binomial => mu > 0.0 && mu < 1.0,
            _ => mu > 0.0,
        }
    }

    /// Plain-English rule for `mu`.
    #[must_use]
    pub fn mu_rule(self) -> &'static str {
        match self {
            Self::Gaussian | Self::Tweedie { power: 0.0 } => "accepts any finite mu",
            Self::Binomial => "binomial needs 0 < mu < 1: pass probabilities, not labels or logits",
            _ => "predictions on the mean scale must be > 0; did you pass the linear predictor?",
        }
    }

    /// The unit deviance `d(y, mu)`: twice the log-likelihood gap between the saturated model
    /// and the fitted mean. Zero when `y == mu`, otherwise positive.
    ///
    /// Limits `y ln y -> 0` as `y -> 0` are taken. Logs are computed as differences of logs, not
    /// logs of ratios, so denormal values cannot underflow to `ln 0`.
    #[must_use]
    pub fn unit_deviance(self, y: f64, mu: f64) -> f64 {
        match self {
            Self::Gaussian => (y - mu) * (y - mu),
            Self::Poisson => 2.0 * (xlogx_minus(y, mu) - (y - mu)),
            Self::Gamma => 2.0 * ((mu.ln() - y.ln()) + (y - mu) / mu),
            Self::Tweedie { power } => tweedie_unit_deviance(power, y, mu),
            Self::Binomial => 2.0 * (xlogx_minus(y, mu) + xlogx_minus(1.0 - y, 1.0 - mu)),
        }
    }
}

/// `y * (ln y - ln mu)`, with the value 0 at `y == 0`.
#[inline]
fn xlogx_minus(y: f64, mu: f64) -> f64 {
    if y > 0.0 {
        y * (y.ln() - mu.ln())
    } else {
        0.0
    }
}

/// Tweedie unit deviance for general `p`, with the three special powers delegated so that
/// `tweedie(1)` is bit-for-bit `poisson`, etc.
// Exact comparison is the point here: only the exact special powers get the special formula.
#[allow(clippy::float_cmp)]
#[inline]
fn tweedie_unit_deviance(p: f64, y: f64, mu: f64) -> f64 {
    if p == 0.0 {
        return Family::Gaussian.unit_deviance(y, mu);
    }
    if p == 1.0 {
        return Family::Poisson.unit_deviance(y, mu);
    }
    if p == 2.0 {
        return Family::Gamma.unit_deviance(y, mu);
    }
    let a = 1.0 - p;
    let b = 2.0 - p;
    // For p < 0 ("extreme stable") y may be negative and the y-term uses max(y, 0), the same
    // convention as scikit-learn. For 1 < p < 2 the term is 0 at y = 0; for p > 2 the support
    // excludes 0. So `y_plus == 0` is exactly the case where the term vanishes.
    let y_plus = y.max(0.0);
    let y_term = if y_plus == 0.0 {
        0.0
    } else {
        y_plus.powf(b) / (a * b)
    };
    // The three terms cancel at y == mu; rounding can leave -1e-16, so clamp at the true floor.
    (2.0 * (y_term - y * mu.powf(a) / a + mu.powf(b) / b)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bit-for-bit equality is what this test is asserting.
    #[allow(clippy::float_cmp)]
    #[test]
    fn special_powers_match_named_families() {
        for (y, mu) in [(0.0, 0.7), (1.0, 1.3), (2.5, 2.0)] {
            let t = |p: f64| Family::Tweedie { power: p }.unit_deviance(y, mu);
            assert_eq!(t(0.0), Family::Gaussian.unit_deviance(y, mu));
            assert_eq!(t(1.0), Family::Poisson.unit_deviance(y, mu));
            if y > 0.0 {
                assert_eq!(t(2.0), Family::Gamma.unit_deviance(y, mu));
            }
        }
    }

    #[test]
    fn zero_at_perfect_fit() {
        for f in [
            Family::Gaussian,
            Family::Poisson,
            Family::Gamma,
            Family::Tweedie { power: 1.5 },
            Family::Binomial,
        ] {
            assert!(f.unit_deviance(0.4, 0.4).abs() < 1e-15, "{}", f.name());
        }
    }

    #[test]
    fn parse_rejects_bad_power() {
        assert!(Family::parse("tweedie", Some(0.5)).is_err());
        assert!(Family::parse("tweedie", None).is_err());
        assert!(Family::parse("weibull", None).is_err());
        assert_eq!(
            Family::parse("tweedie", Some(1.5)).unwrap(),
            Family::Tweedie { power: 1.5 }
        );
    }
}
