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
//!
//! Every pass over the rows runs in parallel over fixed-size chunks (rayon). Partial sums are
//! formed per chunk and combined in chunk order, so a fit is bit-for-bit the same whatever
//! the thread count — reproducible first, fast second.

use rayon::prelude::*;

use crate::error::{all_values, same_length, GlassError};
use crate::family::Family;
use crate::linalg::Square;
use crate::link::Link;

/// Rows per parallel chunk. Small enough to feed every core on a modest fold, large enough
/// that the per-chunk `p x p` partials are noise next to the row work.
const CHUNK: usize = 4096;

/// One IRLS iteration, for the trace.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRow {
    pub iteration: usize,
    /// The objective after the step: the deviance, plus `beta' S beta` for a penalised fit.
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
    /// Even a tiny step raised the objective by more than the tolerance's worth of noise:
    /// the problem is separated / ill-posed (a solver merely sitting at its optimum
    /// converges instead).
    NoImprovement,
}

/// The inference a fitted GLM carries: the null model, the dispersion and the covariances.
/// Skipped (`GlmFit::inference == None`) when `Settings::inference` is off — a lambda search
/// evaluates hundreds of fits and reads only the deviance and the edf.
#[derive(Debug, Clone, PartialEq)]
pub struct Inference {
    /// Deviance of the intercept-only model with the same offset and weights.
    pub null_deviance: f64,
    /// Pearson estimate `sum(w (y-mu)^2 / V(mu)) / (n - edf)`, or exactly 1 for Poisson and
    /// binomial.
    pub dispersion: f64,
    /// `dispersion * (X'WX + S)^{-1}` at convergence, row-major `p x p` (`S = 0` when
    /// unpenalised; for a penalised fit this is the Bayesian posterior covariance, mgcv's
    /// convention).
    pub cov: Vec<f64>,
    /// HC1 sandwich covariance `(X'WX)^{-1} (sum_i s_i s_i') (X'WX)^{-1} * n/(n-p)`, where
    /// `s_i = x_i * w_i (y_i - mu_i) (dmu/deta) / V(mu_i)` is row i's score. Robust to a
    /// mis-specified variance function (over-dispersion); the dispersion cancels out.
    pub cov_robust: Vec<f64>,
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
    /// Effective degrees of freedom `tr((X'WX + S)^{-1} X'WX)`: how many coefficients the
    /// fit really spends. Equals `n_features` exactly when there is no penalty.
    pub edf: f64,
    /// `None` when the fit was asked to skip inference (see [`Settings::inference`]).
    pub inference: Option<Inference>,
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
    /// Fit the null model and form the covariances. Off, the fit returns coefficients,
    /// deviance and edf only — all a smoothing-parameter search needs, at a third of the
    /// row work.
    pub inference: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
            max_halvings: 20,
            inference: true,
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

/// Fit by IRLS, optionally with a quadratic penalty `beta' S beta` on the coefficients.
///
/// `penalty` is a row-major symmetric `p x p` matrix `S`, already scaled by the smoothing
/// parameter. Zero rows and columns (the intercept, plain linear terms) leave those
/// coefficients unpenalised; `None` is the ordinary GLM. The penalised fit minimises the
/// penalised deviance `D + beta' S beta` — the PIRLS fixed point.
///
/// `warm_start` are coefficients to start from instead of R's `mustart` — a converged fit at
/// a neighbouring smoothing parameter, typically, which lands within a couple of iterations.
/// A warm start is a real fit, so its first step is judged like every other, and it may
/// converge at iteration 1.
///
/// # Errors
/// Bad shapes; `y` outside the family's support; non-finite or negative weights; a
/// rank-deficient design; a penalty that is not symmetric; a warm start of the wrong length
/// or whose mean leaves the family's support; `max_iter == 0`.
pub fn fit(
    family: Family,
    link: Link,
    data: Data<'_>,
    penalty: Option<&[f64]>,
    warm_start: Option<&[f64]>,
    settings: Settings,
) -> Result<GlmFit, GlassError> {
    validate(family, data, penalty, warm_start, settings)?;
    let state = irls(family, link, data, penalty, warm_start, settings)?;

    // information matrix at the final mean, for the covariance and the effective dof
    let ww = working_weights(family, link, data, &state.eta, &state.mu);
    let xtwx = cross_product(data, &ww);
    let mut bread = xtwx.clone();
    add_penalty(&mut bread, penalty);
    let inv = Square::inverse_with(&bread.cholesky()?);
    #[allow(clippy::cast_precision_loss)]
    let edf = if penalty.is_none() {
        data.n_features as f64
    } else {
        // tr((X'WX + S)^{-1} X'WX): what the penalised fit actually spends
        (0..data.n_features)
            .map(|a| {
                (0..data.n_features)
                    .map(|k| inv.get(a, k) * xtwx.get(k, a))
                    .sum::<f64>()
            })
            .sum()
    };
    let inference = if settings.inference {
        Some(inference(family, link, data, &state, edf, &inv, settings)?)
    } else {
        None
    };

    Ok(GlmFit {
        family,
        link,
        coef: state.coef,
        mu: state.mu,
        deviance: state.deviance,
        edf,
        inference,
        n_rows: data.n_rows,
        n_features: data.n_features,
        iterations: state.trace.len(),
        stop: state.stop,
        trace: state.trace,
    })
}

/// The null model, the dispersion and both covariances at a converged fit.
fn inference(
    family: Family,
    link: Link,
    data: Data<'_>,
    state: &State,
    edf: f64,
    inv: &Square,
    settings: Settings,
) -> Result<Inference, GlassError> {
    let dispersion = if family.fixed_dispersion() {
        1.0
    } else {
        let pearson = chunk_sum(data.n_rows, |i| {
            data.weight(i) * (data.y[i] - state.mu[i]).powi(2) / family.variance(state.mu[i])
        });
        #[allow(clippy::cast_precision_loss)]
        let dof = data.n_rows as f64 - edf;
        pearson / dof
    };
    let cov = inv.data.iter().map(|v| v * dispersion).collect();
    let cov_robust = sandwich(family, link, data, &state.eta, &state.mu, inv);

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
        None,
        None,
        settings,
    )?;
    Ok(Inference {
        null_deviance: null.deviance,
        dispersion,
        cov,
        cov_robust,
    })
}

/// The state IRLS carries between iterations.
struct State {
    coef: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    deviance: f64,
    /// `deviance + beta' S beta`: what a penalised fit minimises. Equal to the deviance when
    /// there is no penalty.
    objective: f64,
    trace: Vec<TraceRow>,
    stop: Stop,
}

/// Where the loop starts: the supplied coefficients, or R's `mustart` with `coef = 0`.
///
/// A cold start's mean is not the mean of any coefficient vector, so iteration 1 is always
/// accepted; a warm start is a real fit and is judged from its first step. `warmed` says
/// which.
fn start(
    family: Family,
    link: Link,
    data: Data<'_>,
    penalty: Option<&[f64]>,
    warm_start: Option<&[f64]>,
) -> Result<(State, bool), GlassError> {
    let (coef, eta, mu, warmed) = if let Some(coef) = warm_start {
        let eta = linear_predictor(data, coef);
        let mu = per_row(data.n_rows, |i| link.inverse(eta[i]));
        if !mu.iter().all(|&m| m.is_finite() && family.accepts_mu(m)) {
            return Err(GlassError::BadArgument {
                name: "warm_start",
                problem: "gives a mean outside the family's support",
                fix: "start from a fit of the same family and link, or pass None",
            });
        }
        (coef.to_vec(), eta, mu, true)
    } else {
        let mu = per_row(data.n_rows, |i| family.mu_start(data.y[i], data.weight(i)));
        let eta = per_row(data.n_rows, |i| link.link(mu[i]));
        (vec![0.0; data.n_features], eta, mu, false)
    };
    let deviance = total_deviance(family, data, &mu);
    let objective = deviance + quad_form(penalty, &coef);
    let state = State {
        coef,
        eta,
        mu,
        deviance,
        objective,
        trace: Vec::new(),
        stop: Stop::MaxIter,
    };
    Ok((state, warmed))
}

/// The IRLS loop: linearise, solve, step-halve until the objective drops, repeat.
fn irls(
    family: Family,
    link: Link,
    data: Data<'_>,
    penalty: Option<&[f64]>,
    warm_start: Option<&[f64]>,
    settings: Settings,
) -> Result<State, GlassError> {
    let (mut state, warmed) = start(family, link, data, penalty, warm_start)?;

    for iteration in 1..=settings.max_iter {
        // working response and weights at the current mean
        let ww = working_weights(family, link, data, &state.eta, &state.mu);
        let z = per_row(data.n_rows, |i| {
            (state.eta[i] - data.offset_at(i))
                + (data.y[i] - state.mu[i]) / link.mu_eta(state.eta[i])
        });
        let proposal = weighted_least_squares(data, &z, &ww, penalty)?;

        // step-halving: shrink toward the previous coefficients until the objective drops.
        // A change smaller than the convergence tolerance is not a change: a step that
        // moves the objective by less than that, either way, is the rounding noise of a
        // solver already at its optimum, and is accepted so the loop can stop there.
        let slack = settings.tol * (state.objective.abs() + 0.1);
        let mut halvings = 0;
        let mut fraction = 1.0;
        let accepted = loop {
            let cand: Vec<f64> = (0..data.n_features)
                .map(|j| state.coef[j] + fraction * (proposal[j] - state.coef[j]))
                .collect();
            let cand_eta = linear_predictor(data, &cand);
            let cand_mu = per_row(data.n_rows, |i| link.inverse(cand_eta[i]));
            let inside = cand_mu
                .iter()
                .all(|&m| m.is_finite() && family.accepts_mu(m));
            let cand_dev = if inside {
                total_deviance(family, data, &cand_mu)
            } else {
                f64::INFINITY
            };
            let cand_obj = cand_dev + quad_form(penalty, &cand);
            let first_cold_step = iteration == 1 && !warmed;
            if cand_obj.is_finite() && (first_cold_step || cand_obj <= state.objective + slack) {
                break Some((cand, cand_eta, cand_mu, cand_dev, cand_obj));
            }
            halvings += 1;
            if halvings > settings.max_halvings {
                break None;
            }
            fraction /= 2.0;
        };
        let Some((coef, eta, mu, deviance, objective)) = accepted else {
            state.stop = Stop::NoImprovement;
            state.trace.push(TraceRow {
                iteration,
                deviance: state.objective,
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
        let rel_change = (state.objective - objective).abs() / (objective.abs() + 0.1);
        state.coef = coef;
        state.eta = eta;
        state.mu = mu;
        state.deviance = deviance;
        state.objective = objective;
        state.trace.push(TraceRow {
            iteration,
            deviance: objective,
            halvings,
            max_step,
        });
        if (iteration > 1 || warmed) && rel_change < settings.tol {
            state.stop = Stop::Converged;
            break;
        }
    }
    Ok(state)
}

/// HC1: `B = X' diag(s_i^2) X`, then `inv B inv * n / (n - p)`.
fn sandwich(
    family: Family,
    link: Link,
    data: Data<'_>,
    eta: &[f64],
    mu: &[f64],
    inv: &Square,
) -> Vec<f64> {
    let p = data.n_features;
    let score_sq = per_row(data.n_rows, |i| {
        let s = data.weight(i) * (data.y[i] - mu[i]) * link.mu_eta(eta[i]) / family.variance(mu[i]);
        s * s
    });
    let meat = cross_product(data, &score_sq);
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

// ---------------------------------------------------------------- row passes (parallel)

/// The row ranges `[lo, hi)` of the fixed-size chunks, in order.
fn chunks(n_rows: usize) -> impl IndexedParallelIterator<Item = (usize, usize)> {
    (0..n_rows.div_ceil(CHUNK))
        .into_par_iter()
        .map(move |c| (c * CHUNK, ((c + 1) * CHUNK).min(n_rows)))
}

/// One value per row, computed in parallel; the order of the output is the order of the rows.
fn per_row(n_rows: usize, f: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
    let mut out = vec![0.0; n_rows];
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, chunk)| {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = f(c * CHUNK + k);
            }
        });
    out
}

/// `sum_i f(i)`: sequential within each chunk, chunk partials added in chunk order.
fn chunk_sum(n_rows: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    let partials: Vec<f64> = chunks(n_rows)
        .map(|(lo, hi)| (lo..hi).map(&f).sum::<f64>())
        .collect();
    partials.iter().sum()
}

/// `w * (d mu / d eta)^2 / V(mu)` per row.
fn working_weights(
    family: Family,
    link: Link,
    data: Data<'_>,
    eta: &[f64],
    mu: &[f64],
) -> Vec<f64> {
    per_row(data.n_rows, |i| {
        let d = link.mu_eta(eta[i]);
        data.weight(i) * d * d / family.variance(mu[i])
    })
}

/// `eta = X beta + offset`.
fn linear_predictor(data: Data<'_>, coef: &[f64]) -> Vec<f64> {
    per_row(data.n_rows, |i| {
        let dot: f64 = data.row(i).iter().zip(coef).map(|(a, b)| a * b).sum();
        dot + data.offset_at(i)
    })
}

/// `X' W X` (symmetric, computed once per pair): chunk partials, added in chunk order.
// Index loops are the readable form of a matrix kernel; iterators would hide the (a, b) pair.
#[allow(clippy::needless_range_loop)]
fn cross_product(data: Data<'_>, ww: &[f64]) -> Square {
    let p = data.n_features;
    let partials: Vec<Square> = chunks(data.n_rows)
        .map(|(lo, hi)| {
            let mut part = Square::zeros(p);
            for i in lo..hi {
                let row = data.row(i);
                for a in 0..p {
                    let ra = ww[i] * row[a];
                    for b in a..p {
                        let v = part.get(a, b) + ra * row[b];
                        part.set(a, b, v);
                    }
                }
            }
            part
        })
        .collect();
    let mut out = Square::zeros(p);
    for part in &partials {
        for (v, add) in out.data.iter_mut().zip(&part.data) {
            *v += add;
        }
    }
    for a in 0..p {
        for b in 0..a {
            let v = out.get(b, a);
            out.set(a, b, v);
        }
    }
    out
}

/// `X' W z`: chunk partials, added in chunk order.
fn weighted_rhs(data: Data<'_>, z: &[f64], ww: &[f64]) -> Vec<f64> {
    let p = data.n_features;
    let partials: Vec<Vec<f64>> = chunks(data.n_rows)
        .map(|(lo, hi)| {
            let mut part = vec![0.0; p];
            for i in lo..hi {
                let wz = ww[i] * z[i];
                for (acc, &xa) in part.iter_mut().zip(data.row(i)) {
                    *acc += xa * wz;
                }
            }
            part
        })
        .collect();
    let mut out = vec![0.0; p];
    for part in &partials {
        for (v, add) in out.iter_mut().zip(part) {
            *v += add;
        }
    }
    out
}

/// `m += S` (no-op without a penalty).
fn add_penalty(m: &mut Square, penalty: Option<&[f64]>) {
    if let Some(s) = penalty {
        for (v, add) in m.data.iter_mut().zip(s) {
            *v += add;
        }
    }
}

/// `beta' S beta`: the wiggliness the penalty charges.
fn quad_form(penalty: Option<&[f64]>, coef: &[f64]) -> f64 {
    let Some(s) = penalty else {
        return 0.0;
    };
    let p = coef.len();
    (0..p)
        .map(|a| coef[a] * (0..p).map(|b| s[a * p + b] * coef[b]).sum::<f64>())
        .sum()
}

/// Solve `(X' W X + S) beta = X' W z`.
fn weighted_least_squares(
    data: Data<'_>,
    z: &[f64],
    ww: &[f64],
    penalty: Option<&[f64]>,
) -> Result<Vec<f64>, GlassError> {
    let mut xtwx = cross_product(data, ww);
    add_penalty(&mut xtwx, penalty);
    let rhs = weighted_rhs(data, z, ww);
    let chol = xtwx.cholesky()?;
    Ok(Square::solve_with(&chol, &rhs))
}

/// Total weighted deviance (not the mean: the fitter compares sums).
fn total_deviance(family: Family, data: Data<'_>, mu: &[f64]) -> f64 {
    chunk_sum(data.n_rows, |i| {
        data.weight(i) * family.unit_deviance(data.y[i], mu[i])
    })
}

fn validate(
    family: Family,
    data: Data<'_>,
    penalty: Option<&[f64]>,
    warm_start: Option<&[f64]>,
    settings: Settings,
) -> Result<(), GlassError> {
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
    if let Some(s) = penalty {
        validate_penalty(s, n_features)?;
    }
    if let Some(c) = warm_start {
        if c.len() != n_features {
            return Err(GlassError::LengthMismatch {
                left: "warm_start",
                left_len: c.len(),
                right: "X columns",
                right_len: n_features,
            });
        }
        all_values(
            "warm_start",
            c,
            "must be finite",
            "NaN or inf in the starting coefficients",
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

/// A penalty must be a finite, symmetric `p x p` matrix.
fn validate_penalty(s: &[f64], n_features: usize) -> Result<(), GlassError> {
    if s.len() != n_features * n_features {
        return Err(GlassError::LengthMismatch {
            left: "penalty",
            left_len: s.len(),
            right: "n_features * n_features",
            right_len: n_features * n_features,
        });
    }
    all_values(
        "penalty",
        s,
        "must be finite",
        "NaN or inf in the penalty matrix",
        f64::is_finite,
    )?;
    for a in 0..n_features {
        for b in 0..a {
            let (ab, ba) = (s[a * n_features + b], s[b * n_features + a]);
            if (ab - ba).abs() > 1e-8 * ab.abs().max(ba.abs()).max(1.0) {
                return Err(GlassError::BadArgument {
                    name: "penalty",
                    problem: "must be symmetric",
                    fix: "a quadratic penalty is beta' S beta with S = S'; symmetrise S",
                });
            }
        }
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

    /// The objective never rises by more than the convergence tolerance's worth of noise.
    fn descends(trace: &[TraceRow]) -> bool {
        trace
            .windows(2)
            .all(|p| p[1].deviance <= p[0].deviance + 1e-10 * (p[0].deviance.abs() + 0.1))
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
            None,
            None,
            Settings::default(),
        )
        .unwrap();
        assert!((fit.coef[0] - 1.0).abs() < 1e-12 && (fit.coef[1] - 2.0).abs() < 1e-12);
        assert!(fit.deviance < 1e-20);
        assert!((fit.edf - 2.0).abs() < 1e-12);
    }

    #[test]
    fn poisson_log_is_balanced_and_beats_null() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let fit = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            None,
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
        assert!(fit.deviance < fit.inference.as_ref().unwrap().null_deviance);
        assert!(descends(&fit.trace));
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
            None,
            None,
            Settings::default(),
        )
        .unwrap_err();
        assert!(matches!(err, GlassError::Singular { column: 2 }), "{err}");
    }

    #[test]
    fn zero_penalty_matches_the_unpenalised_fit() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let plain = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            None,
            Settings::default(),
        )
        .unwrap();
        let zeros = vec![0.0; 4];
        let pen = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            Some(&zeros),
            None,
            Settings::default(),
        )
        .unwrap();
        assert!(plain
            .coef
            .iter()
            .zip(&pen.coef)
            .all(|(a, b)| (a - b).abs() < 1e-10));
        assert!((pen.edf - 2.0).abs() < 1e-8, "edf {}", pen.edf);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn ridge_gaussian_matches_the_closed_form_and_shrinks_edf() {
        let feats = [0.0, 1.0, 2.0, 3.0, 4.0];
        let x = design(&feats);
        let y = [1.2, 2.9, 5.3, 6.8, 9.1];
        // penalise the slope only; the intercept stays free
        let s = vec![0.0, 0.0, 0.0, 3.0];
        let shrunk = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 2, &y),
            Some(&s),
            None,
            Settings::default(),
        )
        .unwrap();
        // closed form beta = (X'X + S)^{-1} X'y, computed here rather than typed in
        let mut xtx = Square::zeros(2);
        let mut rhs = [0.0; 2];
        for (i, &v) in feats.iter().enumerate() {
            let row = [1.0, v];
            for a in 0..2 {
                rhs[a] += row[a] * y[i];
                for b in 0..2 {
                    xtx.set(a, b, xtx.get(a, b) + row[a] * row[b]);
                }
            }
        }
        xtx.set(1, 1, xtx.get(1, 1) + 3.0);
        let beta = Square::solve_with(&xtx.cholesky().unwrap(), &rhs);
        assert!(
            (shrunk.coef[0] - beta[0]).abs() < 1e-10 && (shrunk.coef[1] - beta[1]).abs() < 1e-10
        );
        // a stronger penalty spends fewer effective dof; every penalised fit spends < 2
        let s_loose = vec![0.0, 0.0, 0.0, 0.3];
        let loose = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 2, &y),
            Some(&s_loose),
            None,
            Settings::default(),
        )
        .unwrap();
        assert!(
            1.0 < shrunk.edf && shrunk.edf < loose.edf && loose.edf < 2.0,
            "edf {} vs {}",
            shrunk.edf,
            loose.edf
        );
    }

    #[test]
    fn penalised_poisson_keeps_the_unpenalised_intercept_balanced() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        // the penalty's intercept row and column are zero, so the intercept's score
        // equation still holds exactly: total predicted equals total actual
        let s = vec![0.0, 0.0, 0.0, 5.0];
        let f = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            Some(&s),
            None,
            Settings::default(),
        )
        .unwrap();
        let (ty, tm): (f64, f64) = (y.iter().sum(), f.mu.iter().sum());
        assert!((ty - tm).abs() < 1e-8, "balance: {ty} vs {tm}");
        let free = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            None,
            Settings::default(),
        )
        .unwrap();
        assert!(f.coef[1].abs() < free.coef[1].abs(), "slope must shrink");
        assert!(
            f.deviance > free.deviance,
            "shrinkage costs training deviance"
        );
    }

    #[test]
    fn refuses_an_asymmetric_penalty() {
        let x = design(&[0.0, 1.0, 2.0, 3.0]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let s = vec![1.0, 0.5, 0.0, 1.0];
        let err = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 2, &y),
            Some(&s),
            None,
            Settings::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("symmetric"), "{err}");
    }

    #[test]
    fn warm_start_from_the_optimum_converges_at_once() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let cold = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            None,
            Settings::default(),
        )
        .unwrap();
        let warm = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            Some(&cold.coef),
            Settings::default(),
        )
        .unwrap();
        assert_eq!(warm.stop, Stop::Converged);
        assert_eq!(warm.iterations, 1, "{:?}", warm.trace);
        assert!(cold
            .coef
            .iter()
            .zip(&warm.coef)
            .all(|(a, b)| (a - b).abs() < 1e-10));
        assert!((cold.deviance - warm.deviance).abs() < 1e-10);
    }

    #[test]
    fn warm_start_is_judged_like_any_other_step() {
        // a warm start far from the optimum must not be force-accepted: the trace can only
        // go down from the starting objective
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let far = [3.0, -1.0];
        let start_data = data(&x, 2, &y);
        let eta = linear_predictor(start_data, &far);
        let mu: Vec<f64> = eta.iter().map(|&e| Link::Log.inverse(e)).collect();
        let start_dev = total_deviance(Family::Poisson, start_data, &mu);
        let warm = fit(
            Family::Poisson,
            Link::Log,
            start_data,
            None,
            Some(&far),
            Settings::default(),
        )
        .unwrap();
        assert!(warm.trace[0].deviance <= start_dev);
        assert!(descends(&warm.trace));
        assert_eq!(warm.stop, Stop::Converged);
    }

    #[test]
    fn warm_start_is_refused_when_it_leaves_the_support_or_has_the_wrong_length() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        // identity link: these coefficients put every mean below zero
        let bad = [-10.0, 0.0];
        let err = fit(
            Family::Poisson,
            Link::Identity,
            data(&x, 2, &y),
            None,
            Some(&bad),
            Settings::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("warm_start"), "{err}");
        let short = [0.0];
        let err = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            None,
            Some(&short),
            Settings::default(),
        )
        .unwrap_err();
        assert!(matches!(err, GlassError::LengthMismatch { .. }), "{err}");
    }

    #[test]
    // Exact equality is the claim: the lean fit walks the identical path and only skips
    // the inference afterwards.
    #[allow(clippy::float_cmp)]
    fn lean_fit_matches_the_full_fit_and_carries_no_inference() {
        let x = design(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = [1.0, 1.0, 3.0, 4.0, 8.0, 12.0];
        let s = vec![0.0, 0.0, 0.0, 2.0];
        let full = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            Some(&s),
            None,
            Settings::default(),
        )
        .unwrap();
        let lean = fit(
            Family::Poisson,
            Link::Log,
            data(&x, 2, &y),
            Some(&s),
            None,
            Settings {
                inference: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert!(lean.inference.is_none() && full.inference.is_some());
        assert_eq!(lean.coef, full.coef);
        assert_eq!(lean.deviance, full.deviance);
        assert_eq!(lean.edf, full.edf);
        assert_eq!(lean.trace, full.trace);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn chunked_row_sums_match_a_sequential_pass() {
        // more rows than one chunk, so the partial sums really are combined
        let n = 3 * CHUNK + 17;
        let mut seed: u64 = 12345;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            u
        };
        let feats: Vec<f64> = (0..n).map(|_| next()).collect();
        let y: Vec<f64> = feats
            .iter()
            .map(|&v| 1.0 + 2.0 * v + (next() - 0.5))
            .collect();
        let x = design(&feats);
        let f = fit(
            Family::Gaussian,
            Link::Identity,
            data(&x, 2, &y),
            None,
            None,
            Settings::default(),
        )
        .unwrap();
        // closed form, accumulated row by row in one sequential pass
        let mut xtx = Square::zeros(2);
        let mut rhs = [0.0; 2];
        for (i, &v) in feats.iter().enumerate() {
            let row = [1.0, v];
            for a in 0..2 {
                rhs[a] += row[a] * y[i];
                for b in 0..2 {
                    xtx.set(a, b, xtx.get(a, b) + row[a] * row[b]);
                }
            }
        }
        let beta = Square::solve_with(&xtx.cholesky().unwrap(), &rhs);
        for a in 0..2 {
            assert!(
                (f.coef[a] - beta[a]).abs() < 1e-10 * beta[a].abs().max(1.0),
                "{:?} vs {:?}",
                f.coef,
                beta
            );
        }
        let rss: f64 = (0..n).map(|i| (y[i] - f.mu[i]).powi(2)).sum();
        assert!(
            (f.deviance - rss).abs() < 1e-9 * rss,
            "{} vs {rss}",
            f.deviance
        );
    }
}
