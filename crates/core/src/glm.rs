//! Generalised linear model fitting by IRLS (iteratively reweighted least squares).
//!
//! Each iteration linearises the link at the current mean, solves a weighted least-squares
//! problem, and checks the deviance went down; if it did not, the step is halved (up to a
//! limit). Every iteration is recorded in a trace, so a fit that misbehaves can be read, not
//! re-run — the "logs that explain why" rule.
//!
//! The design matrix `x` is dense, row-major, `n x p`, and already contains the intercept
//! column if one is wanted. `offset` is added to the linear predictor (log exposure for a
//! rate model). `w` are prior weights.

use crate::error::{all_values, same_length, GlassError};
use crate::family::Family;
use crate::linalg::Square;
use crate::link::Link;

/// One IRLS iteration, for the trace.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRow {
    pub iteration: usize,
    pub deviance: f64,
    /// Step-halvings needed before the deviance decreased.
    pub halvings: usize,
    /// Largest absolute coefficient change this iteration.
    pub max_step: f64,
}

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// Relative deviance change below `tol`.
    Converged,
    /// `max_iter` reached with the deviance still moving.
    MaxIter,
    /// Even a tiny step increased the deviance: sitting at the optimum already, or the
    /// problem is separated / ill-posed.
    NoImprovement,
}

/// Everything a fitted GLM knows.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmFit {
    pub family: Family,
    pub link: Link,
    pub coef: Vec<f64>,
    /// Fitted means on the response scale.
    pub mu: Vec<f64>,
    pub deviance: f64,
    /// Deviance of the intercept-only model with the same offset and weights.
    pub null_deviance: f64,
    /// Pearson estimate `sum(w (y-mu)^2 / V(mu)) / (n - p)`, or exactly 1 for Poisson and
    /// binomial.
    pub dispersion: f64,
    /// `dispersion * (X' W X)^{-1}` at convergence, row-major `p x p`.
    pub cov: Vec<f64>,
    /// HC1 sandwich covariance `(X'WX)^{-1} (sum_i s_i s_i') (X'WX)^{-1} * n/(n-p)`, where
    /// `s_i = x_i * w_i (y_i - mu_i) (dmu/deta) / V(mu_i)` is row i's score. Robust to a
    /// mis-specified variance function (over-dispersion); the dispersion cancels out.
    pub cov_robust: Vec<f64>,
    pub n_rows: usize,
    pub n_features: usize,
    pub iterations: usize,
    pub stop: Stop,
    pub trace: Vec<TraceRow>,
}

/// Knobs for the solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub max_iter: usize,
    /// Relative deviance change that counts as converged.
    pub tol: f64,
    pub max_halvings: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
            max_halvings: 20,
        }
    }
}

/// The inputs of a fit, borrowed. `x` is row-major `n_rows x n_features`.
#[derive(Debug, Clone, Copy)]
pub struct Data<'a> {
    pub x: &'a [f64],
    pub n_rows: usize,
    pub n_features: usize,
    pub y: &'a [f64],
    pub weights: Option<&'a [f64]>,
    pub offset: Option<&'a [f64]>,
}

impl Data<'_> {
    #[inline]
    fn weight(&self, i: usize) -> f64 {
        self.weights.map_or(1.0, |w| w[i])
    }

    #[inline]
    fn offset_at(&self, i: usize) -> f64 {
        self.offset.map_or(0.0, |o| o[i])
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.n_features..(i + 1) * self.n_features]
    }
}

/// Fit by IRLS.
///
/// # Errors
/// Bad shapes; `y` outside the family's support; non-finite or negative weights; a
/// rank-deficient design; `max_iter == 0`.
pub fn fit(
    family: Family,
    link: Link,
    data: Data<'_>,
    settings: Settings,
) -> Result<GlmFit, GlassError> {
    validate(family, data, settings)?;
    let state = irls(family, link, data, settings)?;

    // information matrix at the final mean, for the covariance
    let ww = working_weights(family, link, data, &state.eta, &state.mu);
    let mut xtwx = Square::zeros(data.n_features);
    cross_product(data, &ww, &mut xtwx);
    let inv = Square::inverse_with(&xtwx.cholesky()?);
    let dispersion = if family.fixed_dispersion() {
        1.0
    } else {
        let pearson: f64 = (0..data.n_rows)
            .map(|i| {
                data.weight(i) * (data.y[i] - state.mu[i]).powi(2) / family.variance(state.mu[i])
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let dof = (data.n_rows - data.n_features) as f64;
        pearson / dof
    };
    let cov = inv.data.iter().map(|v| v * dispersion).collect();
    let cov_robust = sandwich(family, link, data, &state.eta, &state.mu, &inv);

    // the intercept-only model with the same offset and weights: what a model must beat
    let ones = vec![1.0; data.n_rows];
    let null = irls(
        family,
        link,
        Data {
            x: &ones,
            n_features: 1,
            ..data
        },
        settings,
    )?;

    Ok(GlmFit {
        family,
        link,
        coef: state.coef,
        mu: state.mu,
        deviance: state.deviance,
        null_deviance: null.deviance,
        dispersion,
        cov,
        cov_robust,
        n_rows: data.n_rows,
        n_features: data.n_features,
        iterations: state.trace.len(),
        stop: state.stop,
        trace: state.trace,
    })
}

/// The state IRLS carries between iterations.
struct State {
    coef: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    deviance: f64,
    trace: Vec<TraceRow>,
    stop: Stop,
}

/// The IRLS loop: linearise, solve, step-halve until the deviance drops, repeat.
fn irls(
    family: Family,
    link: Link,
    data: Data<'_>,
    settings: Settings,
) -> Result<State, GlassError> {
    // starting values: R's mustart, then eta from the link, coefficients at zero
    let mu: Vec<f64> = (0..data.n_rows)
        .map(|i| family.mu_start(data.y[i], data.weight(i)))
        .collect();
    let eta: Vec<f64> = mu.iter().map(|&m| link.link(m)).collect();
    let deviance = total_deviance(family, data, &mu);
    let mut state = State {
        coef: vec![0.0; data.n_features],
        eta,
        mu,
        deviance,
        trace: Vec::new(),
        stop: Stop::MaxIter,
    };
    let mut xtwx = Square::zeros(data.n_features);

    for iteration in 1..=settings.max_iter {
        // working response and weights at the current mean
        let ww = working_weights(family, link, data, &state.eta, &state.mu);
        let z: Vec<f64> = (0..data.n_rows)
            .map(|i| {
                (state.eta[i] - data.offset_at(i))
                    + (data.y[i] - state.mu[i]) / link.mu_eta(state.eta[i])
            })
            .collect();
        let proposal = weighted_least_squares(data, &z, &ww, &mut xtwx)?;

        // step-halving: shrink toward the previous coefficients until the deviance drops
        let mut halvings = 0;
        let mut fraction = 1.0;
        let accepted = loop {
            let cand: Vec<f64> = (0..data.n_features)
                .map(|j| state.coef[j] + fraction * (proposal[j] - state.coef[j]))
                .collect();
            let cand_eta = linear_predictor(data, &cand);
            let cand_mu: Vec<f64> = cand_eta.iter().map(|&e| link.inverse(e)).collect();
            let inside = cand_mu
                .iter()
                .all(|&m| m.is_finite() && family.accepts_mu(m));
            let cand_dev = if inside {
                total_deviance(family, data, &cand_mu)
            } else {
                f64::INFINITY
            };
            // iteration 1 starts from coef = 0, which is not a real fit: always accept it
            if cand_dev.is_finite() && (iteration == 1 || cand_dev <= state.deviance) {
                break Some((cand, cand_eta, cand_mu, cand_dev));
            }
            halvings += 1;
            if halvings > settings.max_halvings {
                break None;
            }
            fraction /= 2.0;
        };
        let Some((coef, eta, mu, deviance)) = accepted else {
            state.stop = Stop::NoImprovement;
            state.trace.push(TraceRow {
                iteration,
                deviance: state.deviance,
                halvings,
                max_step: 0.0,
            });
            break;
        };
        let max_step = coef
            .iter()
            .zip(&state.coef)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let rel_change = (state.deviance - deviance).abs() / (deviance.abs() + 0.1);
        state.coef = coef;
        state.eta = eta;
        state.mu = mu;
        state.deviance = deviance;
        state.trace.push(TraceRow {
            iteration,
            deviance,
            halvings,
            max_step,
        });
        if iteration > 1 && rel_change < settings.tol {
            state.stop = Stop::Converged;
            break;
        }
    }
    Ok(state)
}

/// HC1: `B = sum_i s_i s_i'`, then `inv B inv * n / (n - p)`.
fn sandwich(
    family: Family,
    link: Link,
    data: Data<'_>,
    eta: &[f64],
    mu: &[f64],
    inv: &Square,
) -> Vec<f64> {
    let p = data.n_features;
    let mut meat = Square::zeros(p);
    for i in 0..data.n_rows {
        let score_i =
            data.weight(i) * (data.y[i] - mu[i]) * link.mu_eta(eta[i]) / family.variance(mu[i]);
        let row = data.row(i);
        for a in 0..p {
            for b in 0..p {
                let v = meat.get(a, b) + score_i * score_i * row[a] * row[b];
                meat.set(a, b, v);
            }
        }
    }
    // inv * meat * inv
    let mut tmp = Square::zeros(p);
    for a in 0..p {
        for b in 0..p {
            let v: f64 = (0..p).map(|k| inv.get(a, k) * meat.get(k, b)).sum();
            tmp.set(a, b, v);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let scale = data.n_rows as f64 / (data.n_rows - p) as f64;
    let mut out = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..p {
            out[a * p + b] = scale * (0..p).map(|k| tmp.get(a, k) * inv.get(k, b)).sum::<f64>();
        }
    }
    out
}

/// `w * (d mu / d eta)^2 / V(mu)` per row.
fn working_weights(
    family: Family,
    link: Link,
    data: Data<'_>,
    eta: &[f64],
    mu: &[f64],
) -> Vec<f64> {
    (0..data.n_rows)
        .map(|i| {
            let d = link.mu_eta(eta[i]);
            data.weight(i) * d * d / family.variance(mu[i])
        })
        .collect()
}

/// `eta = X beta + offset`.
fn linear_predictor(data: Data<'_>, coef: &[f64]) -> Vec<f64> {
    (0..data.n_rows)
        .map(|i| {
            let dot: f64 = data.row(i).iter().zip(coef).map(|(a, b)| a * b).sum();
            dot + data.offset_at(i)
        })
        .collect()
}

/// `X' W X` into `out` (symmetric, computed once per pair).
// Index loops are the readable form of a matrix kernel; iterators would hide the (a, b) pair.
#[allow(clippy::needless_range_loop)]
fn cross_product(data: Data<'_>, ww: &[f64], out: &mut Square) {
    let p = data.n_features;
    out.data.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..data.n_rows {
        let row = data.row(i);
        for a in 0..p {
            let ra = ww[i] * row[a];
            for b in a..p {
                let v = out.get(a, b) + ra * row[b];
                out.set(a, b, v);
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            let v = out.get(b, a);
            out.set(a, b, v);
        }
    }
}

/// Solve `(X' W X) beta = X' W z`.
fn weighted_least_squares(
    data: Data<'_>,
    z: &[f64],
    ww: &[f64],
    xtwx: &mut Square,
) -> Result<Vec<f64>, GlassError> {
    cross_product(data, ww, xtwx);
    let mut rhs = vec![0.0; data.n_features];
    for i in 0..data.n_rows {
        let wz = ww[i] * z[i];
        for (a, &xa) in data.row(i).iter().enumerate() {
            rhs[a] += xa * wz;
        }
    }
    let chol = xtwx.cholesky()?;
    Ok(Square::solve_with(&chol, &rhs))
}

/// Total weighted deviance (not the mean: the fitter compares sums).
fn total_deviance(family: Family, data: Data<'_>, mu: &[f64]) -> f64 {
    data.y
        .iter()
        .zip(mu)
        .enumerate()
        .map(|(i, (&yi, &mi))| data.weight(i) * family.unit_deviance(yi, mi))
        .sum()
}

fn validate(family: Family, data: Data<'_>, settings: Settings) -> Result<(), GlassError> {
    let Data {
        x,
        n_rows,
        n_features,
        y,
        weights,
        offset,
    } = data;
    if n_rows == 0 || n_features == 0 {
        return Err(GlassError::Empty { name: "X" });
    }
    if x.len() != n_rows * n_features {
        return Err(GlassError::LengthMismatch {
            left: "X",
            left_len: x.len(),
            right: "n_rows * n_features",
            right_len: n_rows * n_features,
        });
    }
    if n_rows <= n_features {
        return Err(GlassError::BadArgument {
            name: "X",
            problem: "needs more rows than features",
            fix: "a GLM with p coefficients needs at least p + 1 rows to estimate dispersion",
        });
    }
    if y.len() != n_rows {
        return Err(GlassError::LengthMismatch {
            left: "y",
            left_len: y.len(),
            right: "X rows",
            right_len: n_rows,
        });
    }
    all_values(
        "X",
        x,
        "must be finite",
        "NaN or inf in the design",
        f64::is_finite,
    )?;
    all_values(
        "y",
        y,
        "must be finite and inside the family's support",
        family.y_rule(),
        |v| v.is_finite() && family.accepts_y(v),
    )?;
    if let Some(w) = weights {
        same_length("y", y, "sample_weight", w)?;
        all_values(
            "sample_weight",
            w,
            "must be finite and >= 0",
            "negative or NaN weights are not weights",
            |v| v.is_finite() && v >= 0.0,
        )?;
    }
    if let Some(o) = offset {
        same_length("y", y, "offset", o)?;
        all_values(
            "offset",
            o,
            "must be finite",
            "an offset is on the link scale: log(exposure), not exposure",
            f64::is_finite,
        )?;
    }
    if settings.max_iter == 0 {
        return Err(GlassError::BadArgument {
            name: "max_iter",
            problem: "must be at least 1",
            fix: "the default is 100",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-column design: intercept + one feature.
    fn design(feature: &[f64]) -> Vec<f64> {
        feature.iter().flat_map(|&v| [1.0, v]).collect()
    }

    fn data<'a>(x: &'a [f64], p: usize, y: &'a [f64]) -> Data<'a> {
        Data {
            x,
            n_rows: y.len(),
            n_features: p,
            y,
            weights: None,
            offset: None,
        }
    }

    #[test]
    fn gaussian_identity_is_least_squares() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = [1.0, 3.0, 5.0, 7.0, 9.0]; // exactly 1 + 2 x
        let fit = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 2, &y),
            Settings::default(),
        )
        .unwrap();
        assert!((fit.coef[0] - 1.0).abs() < 1e-12 && (fit.coef[1] - 2.0).abs() < 1e-12);
        assert!(fit.deviance < 1e-20);
    }

    #[test]
    fn poisson_log_is_balanced_and_beats_null() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let fit = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            Settings::default(),
        )
        .unwrap();
        assert_eq!(fit.stop, Stop::Converged);
        let total_y: f64 = y.iter().sum();
        let total_mu: f64 = fit.mu.iter().sum();
        assert!(
            (total_y - total_mu).abs() < 1e-8,
            "balance: {total_y} vs {total_mu}"
        );
        assert!(fit.deviance < fit.null_deviance);
        assert!(fit
            .trace
            .windows(2)
            .all(|p| p[1].deviance <= p[0].deviance + 1e-12));
    }

    #[test]
    fn refuses_collinear_design() {
        let x: Vec<f64> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|&v| [1.0, v, 2.0 * v])
            .collect();
        let y = [1.0, 2.0, 3.0, 4.0];
        let err = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 3, &y),
            Settings::default(),
        )
        .unwrap_err();
        assert!(matches!(err, GlassError::Singular { column: 2 }), "{err}");
    }
}
