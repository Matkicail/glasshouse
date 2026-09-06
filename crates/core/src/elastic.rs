//! Elastic-net penalties by coordinate descent: the weighted least squares step of a
//! lasso / ridge / elastic-net GLM, solved one coefficient at a time with soft-thresholding
//! (Friedman, Hastie & Tibshirani, "Regularization paths for generalized linear models via
//! coordinate descent", *J. Statistical Software* 33, 2010 — the glmnet algorithm).
//!
//! The penalty is glmnet's and glum's: in the units the outer loop works in (the deviance),
//! `2 * sum(w) * alpha * (l1 * sum|b_j| + (1 - l1) / 2 * sum b_j^2)` over the penalised
//! columns, so `alpha` means the same number here as there. The intercept and any column
//! not marked penalised are free.
//!
//! With `groups`, columns sharing a group id are penalised together (Yuan & Lin's group
//! lasso): `sum|b_j|` becomes `sum_g sqrt(p_g) ||b_g||_2`, so a one-hot factor or a spline
//! term leaves the model whole or stays whole. A group is updated as a block by one
//! majorisation step (Breheny & Huang, "Group descent algorithms for nonconvex penalized
//! linear and logistic regression with grouped predictors", *Statistics and Computing* 25,
//! 2015): a gradient step with the group's largest Gram eigenvalue as the step bound, then
//! the group soft-threshold. A group of one column is the plain lasso update, exactly.

use crate::error::GlassError;
use crate::par::{axpy, chunk_sum};

/// The penalty: glmnet's `alpha` (the overall strength, called lambda there) and `l1_ratio`
/// (1 = lasso, 0 = ridge), applied to the marked columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticNet<'a> {
    pub alpha: f64,
    pub l1_ratio: f64,
    /// One flag per design column; `false` leaves that coefficient unpenalised.
    pub penalised: &'a [bool],
    /// One group id per design column for a group lasso; `None` penalises column by column.
    pub groups: Option<&'a [usize]>,
}

impl ElasticNet<'_> {
    /// The penalty in deviance units for these coefficients.
    #[must_use]
    pub fn value(&self, coef: &[f64], weight_sum: f64) -> f64 {
        let l1: f64 = units_of(self.penalised, self.groups)
            .iter()
            .filter(|u| self.penalised[u[0]])
            .map(|u| {
                #[allow(clippy::cast_precision_loss)]
                let weight = (u.len() as f64).sqrt();
                weight * u.iter().map(|&j| coef[j] * coef[j]).sum::<f64>().sqrt()
            })
            .sum();
        let l2: f64 = coef
            .iter()
            .zip(self.penalised)
            .filter(|(_, &pen)| pen)
            .map(|(b, _)| b * b)
            .sum();
        2.0 * weight_sum * self.alpha * (self.l1_ratio * l1 + 0.5 * (1.0 - self.l1_ratio) * l2)
    }

    /// `d^2 / d b_j^2` of half the penalty: what the ridge part adds to the diagonal of the
    /// information matrix for a penalised column.
    #[must_use]
    pub fn ridge_diagonal(&self, weight_sum: f64) -> f64 {
        weight_sum * self.alpha * (1.0 - self.l1_ratio)
    }
}

/// Knobs for the inner loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CdSettings {
    /// Converged when no coefficient moved by more than `tol * (1 + max|b|)` in a sweep.
    pub tol: f64,
    pub max_sweeps: usize,
}

impl Default for CdSettings {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            max_sweeps: 10_000,
        }
    }
}

/// One weighted least-squares subproblem: the design transposed (row-major `p x n_rows`,
/// so a coefficient's column is contiguous), the working weights and response, and the
/// penalty. `scale` is `sum(w) * alpha` (see the module docs).
#[derive(Debug, Clone, Copy)]
pub struct CdProblem<'a> {
    pub xt: &'a [f64],
    pub n_rows: usize,
    pub ww: &'a [f64],
    pub z: &'a [f64],
    pub scale: f64,
    pub l1_ratio: f64,
    pub penalised: &'a [bool],
    pub groups: Option<&'a [usize]>,
}

impl CdProblem<'_> {
    fn col(&self, j: usize) -> &[f64] {
        &self.xt[j * self.n_rows..(j + 1) * self.n_rows]
    }
}

/// The units the descent moves one at a time: penalised columns that share a group id form
/// one unit, in order of first appearance; every other column is a unit of its own.
#[must_use]
pub fn units_of(penalised: &[bool], groups: Option<&[usize]>) -> Vec<Vec<usize>> {
    let mut units: Vec<Vec<usize>> = Vec::new();
    let mut of_group: Vec<(usize, usize)> = Vec::new(); // (group id, unit index)
    for (j, &pen) in penalised.iter().enumerate() {
        match groups {
            Some(g) if pen => {
                if let Some(&(_, k)) = of_group.iter().find(|&&(id, _)| id == g[j]) {
                    units[k].push(j);
                } else {
                    of_group.push((g[j], units.len()));
                    units.push(vec![j]);
                }
            }
            _ => units.push(vec![j]),
        }
    }
    units
}

/// Minimise `1/2 sum ww_i (z_i - x_i b)^2 + scale * (l1 * sum|b_j| + (1 - l1)/2 * sum b_j^2)`
/// over the penalised `j`, the rest free, starting from `start`.
///
/// Cyclic sweeps over every coefficient, then over the active set until it settles, then a
/// full sweep to confirm. The intercept (an unpenalised constant column) is never swept:
/// it is kept at its closed form, the weighted mean of the residual, after every update,
/// which is the same as running the descent on centred columns (glmnet does this too).
/// Without it a frequent 0/1 column and the intercept are nearly collinear and the descent
/// crawls.
///
/// # Errors
/// The sweep cap is hit without settling.
pub fn coordinate_descent(
    cd: &CdProblem<'_>,
    start: &[f64],
    settings: CdSettings,
) -> Result<Vec<f64>, GlassError> {
    let (n_rows, ww) = (cd.n_rows, cd.ww);
    let Prepared {
        weight_sum,
        intercept,
        means,
        norms,
        mut coef,
        mut resid,
    } = prepare(cd, start);
    let threshold = cd.scale * cd.l1_ratio;
    let ridge = cd.scale * (1.0 - cd.l1_ratio);
    let units = units_of(cd.penalised, cd.groups);
    // a block's gradient step is bounded by its largest centred Gram eigenvalue
    let bounds: Vec<f64> = units
        .iter()
        .map(|u| {
            if u.len() == 1 {
                norms[u[0]]
            } else {
                largest_eigenvalue(&centred_gram(cd, u, &means, weight_sum), u.len()) * 1.05
            }
        })
        .collect();
    // move coefficient j by `step`, keeping the residual and the intercept current
    let apply = |coef: &mut Vec<f64>, resid: &mut Vec<f64>, j: usize, step: f64| {
        // resid -= step * (x_j - mean_j): the intercept absorbs step * mean_j
        let (col, mean) = (cd.col(j), means[j]);
        for (r, &x) in resid.iter_mut().zip(col) {
            *r -= step * (x - mean);
        }
        if let Some(k) = intercept {
            coef[k] -= step * mean / cd.col(k)[0];
        }
        coef[j] += step;
    };
    let mut sweeps = 0;
    let sweep = |coef: &mut Vec<f64>, resid: &mut Vec<f64>, only_active: bool| -> f64 {
        let mut max_step: f64 = 0.0;
        for (unit, &bound) in units.iter().zip(&bounds) {
            let j0 = unit[0];
            let pen = cd.penalised[j0];
            if Some(j0) == intercept
                || bound == 0.0
                || (only_active && pen && unit.iter().all(|&j| coef[j] == 0.0))
            {
                continue;
            }
            // with the intercept optimal the residual has zero weighted mean, so the
            // gradient on the raw column equals the gradient on the centred one
            if unit.len() == 1 {
                let j = j0;
                let rho =
                    chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i] * resid[i]) + norms[j] * coef[j];
                let new = if pen {
                    soft_threshold(rho, threshold) / (norms[j] + ridge)
                } else {
                    rho / norms[j]
                };
                let step = new - coef[j];
                if step != 0.0 {
                    apply(coef, resid, j, step);
                    max_step = max_step.max(step.abs());
                }
                continue;
            }
            // a group: one majorisation step, then the group soft-threshold
            let target: Vec<f64> = unit
                .iter()
                .map(|&j| coef[j] + chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i] * resid[i]) / bound)
                .collect();
            let norm = target.iter().map(|t| t * t).sum::<f64>().sqrt();
            #[allow(clippy::cast_precision_loss)]
            let radius = threshold * (unit.len() as f64).sqrt() / bound;
            let shrink = if norm > radius {
                (1.0 - radius / norm) / (1.0 + ridge / bound)
            } else {
                0.0
            };
            for (&j, t) in unit.iter().zip(&target) {
                let step = t * shrink - coef[j];
                if step != 0.0 {
                    apply(coef, resid, j, step);
                    max_step = max_step.max(step.abs());
                }
            }
        }
        max_step
    };
    loop {
        let largest = coef.iter().fold(0.0_f64, |m, b| m.max(b.abs()));
        let tol = settings.tol * (1.0 + largest);
        sweeps += 1;
        if sweep(&mut coef, &mut resid, false) <= tol {
            return Ok(coef);
        }
        // the active set alone, until it settles; then confirm with a full sweep
        while sweep(&mut coef, &mut resid, true) > tol {
            sweeps += 1;
            if sweeps > settings.max_sweeps {
                return Err(sweep_cap());
            }
        }
        if sweeps > settings.max_sweeps {
            return Err(sweep_cap());
        }
    }
}

/// What a descent starts from: the intercept column, the weighted column means (zero when
/// there is no intercept to absorb them), the centred norms, and the residual of `start`
/// with the intercept already at its optimum.
struct Prepared {
    weight_sum: f64,
    intercept: Option<usize>,
    means: Vec<f64>,
    norms: Vec<f64>,
    coef: Vec<f64>,
    resid: Vec<f64>,
}

fn prepare(cd: &CdProblem<'_>, start: &[f64]) -> Prepared {
    let (n_rows, ww, p) = (cd.n_rows, cd.ww, start.len());
    let weight_sum: f64 = chunk_sum(n_rows, |i| ww[i]);
    let intercept = (0..p).find(|&j| !cd.penalised[j] && is_constant(cd.col(j)));
    let means: Vec<f64> = (0..p)
        .map(|j| match intercept {
            Some(_) => chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i]) / weight_sum,
            None => 0.0,
        })
        .collect();
    let norms: Vec<f64> = (0..p)
        .map(|j| {
            chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i] * cd.col(j)[i])
                - weight_sum * means[j] * means[j]
        })
        .collect();
    let mut coef = start.to_vec();
    let mut resid = cd.z.to_vec();
    for (j, &b) in coef.iter().enumerate() {
        if b != 0.0 {
            axpy(&mut resid, cd.col(j), -b);
        }
    }
    if let Some(k) = intercept {
        let shift = chunk_sum(n_rows, |i| ww[i] * resid[i]) / weight_sum;
        coef[k] += shift / cd.col(k)[0];
        for r in &mut resid {
            *r -= shift;
        }
    }
    Prepared {
        weight_sum,
        intercept,
        means,
        norms,
        coef,
        resid,
    }
}

/// The centred, weighted Gram matrix of one group's columns, row-major `p_g x p_g`.
fn centred_gram(cd: &CdProblem<'_>, unit: &[usize], means: &[f64], weight_sum: f64) -> Vec<f64> {
    let m = unit.len();
    let mut gram = vec![0.0; m * m];
    for (a, &j) in unit.iter().enumerate() {
        for (b, &k) in unit.iter().enumerate().skip(a) {
            let raw = chunk_sum(cd.n_rows, |i| cd.ww[i] * cd.col(j)[i] * cd.col(k)[i]);
            let value = raw - weight_sum * means[j] * means[k];
            gram[a * m + b] = value;
            gram[b * m + a] = value;
        }
    }
    gram
}

/// The largest eigenvalue of a small symmetric positive semi-definite matrix, by power
/// iteration from the all-ones vector.
fn largest_eigenvalue(gram: &[f64], m: usize) -> f64 {
    let mut v = vec![1.0; m];
    let mut value = 0.0;
    for _ in 0..100 {
        let next: Vec<f64> = (0..m)
            .map(|a| (0..m).map(|b| gram[a * m + b] * v[b]).sum())
            .collect();
        let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        let estimate = next.iter().zip(&v).map(|(n, x)| n * x).sum::<f64>()
            / v.iter().map(|x| x * x).sum::<f64>();
        v = next.iter().map(|x| x / norm).collect();
        if (estimate - value).abs() <= 1e-12 * estimate.abs() {
            return estimate;
        }
        value = estimate;
    }
    value
}

#[allow(clippy::float_cmp)] // an intercept column is exactly constant, or it is not one
fn is_constant(col: &[f64]) -> bool {
    col.first()
        .is_some_and(|&c| c != 0.0 && col.iter().all(|&x| x == c))
}

fn sweep_cap() -> GlassError {
    GlassError::BadArgument {
        name: "elastic_net",
        problem: "coordinate descent did not settle within the sweep cap",
        fix: "standardise the columns, loosen cd_tol, or raise cd_max_sweeps",
    }
}

#[inline]
fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Columns of a small design with an intercept, transposed.
    fn problem() -> (Vec<f64>, usize, Vec<f64>) {
        let n = 8;
        let x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x2 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut xt = vec![1.0; n];
        xt.extend_from_slice(&x1);
        xt.extend_from_slice(&x2);
        let z: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + 0.5 * x1[i] + 0.1 * x2[i] + [0.1, -0.2, 0.05, 0.0, -0.1, 0.2, -0.05, 0.1][i]
            })
            .collect();
        (xt, n, z)
    }

    fn gradient(xt: &[f64], n: usize, ww: &[f64], z: &[f64], coef: &[f64]) -> Vec<f64> {
        let p = coef.len();
        (0..p)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let fitted: f64 = (0..p).map(|k| xt[k * n + i] * coef[k]).sum();
                        -ww[i] * xt[j * n + i] * (z[i] - fitted)
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn unpenalised_columns_reach_least_squares() {
        let (xt, n, z) = problem();
        let ww = vec![1.0; n];
        let coef = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale: 1.0,
                l1_ratio: 0.5,
                penalised: &[false, false, false],
                groups: None,
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        for g in gradient(&xt, n, &ww, &z, &coef) {
            assert!(g.abs() < 1e-8, "{g}");
        }
    }

    #[test]
    fn lasso_satisfies_the_subgradient_conditions_and_zeroes_a_weak_column() {
        let (xt, n, z) = problem();
        let ww = vec![1.0; n];
        let (scale, l1) = (2.0, 1.0);
        let penalised = [false, true, true];
        let coef = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale,
                l1_ratio: l1,
                penalised: &penalised,
                groups: None,
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        assert!(
            coef[2].to_bits() == 0,
            "the weak +/-1 column is switched off: {coef:?}"
        );
        assert!(coef[1] > 0.0);
        let g = gradient(&xt, n, &ww, &z, &coef);
        assert!(g[0].abs() < 1e-8, "intercept stationary");
        assert!(
            (g[1] + scale * l1 * coef[1].signum()).abs() < 1e-8,
            "active: g = -scale*sign"
        );
        assert!(g[2].abs() <= scale * l1 + 1e-12, "inactive: |g| <= scale");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn ridge_is_the_closed_form() {
        let (xt, n, z) = problem();
        let ww: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * i as f64).collect();
        let scale = 2.0;
        let penalised = [false, true, true];
        let coef = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale,
                l1_ratio: 0.0,
                penalised: &penalised,
                groups: None,
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        // (X'WX + scale * diag(0,1,1)) b = X'Wz, written out
        let mut h = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for i in 0..n {
            for a in 0..3 {
                rhs[a] += ww[i] * xt[a * n + i] * z[i];
                for b in 0..3 {
                    h[a][b] += ww[i] * xt[a * n + i] * xt[b * n + i];
                }
            }
        }
        h[1][1] += scale;
        h[2][2] += scale;
        for a in 0..3 {
            let lhs: f64 = (0..3).map(|b| h[a][b] * coef[b]).sum();
            assert!((lhs - rhs[a]).abs() < 1e-8, "{coef:?}");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn a_frequent_level_next_to_the_intercept_settles_quickly() {
        // 95 % ones in the penalised column: nearly collinear with the intercept, the case
        // that made the plain descent crawl. Ridge only, so the closed form is exact.
        let n = 400;
        let ones: Vec<f64> = (0..n)
            .map(|i| if i % 20 == 0 { 0.0 } else { 1.0 })
            .collect();
        let mut xt = vec![1.0; n];
        xt.extend_from_slice(&ones);
        let z: Vec<f64> = (0..n)
            .map(|i| 0.5 + 0.3 * ones[i] + ((i * 7 % 11) as f64 - 5.0) * 0.01)
            .collect();
        let ww = vec![1.0; n];
        let scale = 3.0;
        let coef = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale,
                l1_ratio: 0.0,
                penalised: &[false, true],
                groups: None,
            },
            &[0.0; 2],
            CdSettings {
                tol: 1e-12,
                max_sweeps: 50,
            },
        )
        .unwrap();
        let mut h = [[0.0; 2]; 2];
        let mut rhs = [0.0; 2];
        for i in 0..n {
            for a in 0..2 {
                rhs[a] += xt[a * n + i] * z[i];
                for b in 0..2 {
                    h[a][b] += xt[a * n + i] * xt[b * n + i];
                }
            }
        }
        h[1][1] += scale;
        for a in 0..2 {
            let lhs: f64 = (0..2).map(|b| h[a][b] * coef[b]).sum();
            assert!((lhs - rhs[a]).abs() < 1e-8, "{coef:?}");
        }
    }

    #[test]
    fn singleton_groups_are_the_plain_lasso() {
        let (xt, n, z) = problem();
        let ww = vec![1.0; n];
        let penalised = [false, true, true];
        let plain = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale: 0.4,
                l1_ratio: 0.8,
                penalised: &penalised,
                groups: None,
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        let grouped = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale: 0.4,
                l1_ratio: 0.8,
                penalised: &penalised,
                groups: Some(&[0, 1, 2]),
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        for (a, b) in plain.iter().zip(&grouped) {
            assert!((a - b).abs() < 1e-12, "{plain:?} vs {grouped:?}");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn a_group_meets_the_group_lasso_conditions_and_leaves_whole() {
        // intercept, a strong two-column group, and a weak two-column group
        let n = 60;
        let cols: Vec<Vec<f64>> = (1..=4)
            .map(|c| {
                (0..n)
                    .map(|i| (((i * 7 + c * 13) % 17) as f64 - 8.0) / 8.0)
                    .collect()
            })
            .collect();
        let mut xt = vec![1.0; n];
        for c in &cols {
            xt.extend_from_slice(c);
        }
        let z: Vec<f64> = (0..n)
            .map(|i| {
                0.3 + 1.5 * cols[0][i] - 1.0 * cols[1][i]
                    + 0.02 * cols[2][i]
                    + ((i % 5) as f64 - 2.0) * 0.05
            })
            .collect();
        let ww: Vec<f64> = (0..n).map(|i| 0.5 + (i % 3) as f64 * 0.25).collect();
        let penalised = [false, true, true, true, true];
        let groups = [0, 1, 1, 2, 2];
        let scale = 3.0;
        let coef = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale,
                l1_ratio: 1.0,
                penalised: &penalised,
                groups: Some(&groups),
            },
            &[0.0; 5],
            CdSettings {
                tol: 1e-12,
                max_sweeps: 100_000,
            },
        )
        .unwrap();
        assert!(
            coef[3] == 0.0 && coef[4] == 0.0,
            "the weak group leaves whole: {coef:?}"
        );
        assert!(
            coef[1] != 0.0 && coef[2] != 0.0,
            "the strong group stays whole: {coef:?}"
        );
        // KKT: an active group's gradient is scale * sqrt(p_g) * b_g / ||b_g||; an inactive
        // group's gradient norm is at most scale * sqrt(p_g); the intercept is stationary
        let g = gradient(&xt, n, &ww, &z, &coef);
        assert!(g[0].abs() < 1e-8, "intercept: {}", g[0]);
        let norm = (coef[1] * coef[1] + coef[2] * coef[2]).sqrt();
        let radius = scale * 2.0_f64.sqrt();
        for j in [1, 2] {
            assert!(
                (g[j] + radius * coef[j] / norm).abs() < 1e-7,
                "active KKT at {j}: {}",
                g[j]
            );
        }
        assert!(
            (g[3] * g[3] + g[4] * g[4]).sqrt() <= radius + 1e-9,
            "inactive KKT"
        );
    }

    #[test]
    fn a_warm_start_at_the_solution_is_left_alone() {
        let (xt, n, z) = problem();
        let ww = vec![1.0; n];
        let penalised = [false, true, true];
        let first = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale: 0.3,
                l1_ratio: 0.7,
                penalised: &penalised,
                groups: None,
            },
            &[0.0; 3],
            CdSettings::default(),
        )
        .unwrap();
        let again = coordinate_descent(
            &CdProblem {
                xt: &xt,
                n_rows: n,
                ww: &ww,
                z: &z,
                scale: 0.3,
                l1_ratio: 0.7,
                penalised: &penalised,
                groups: None,
            },
            &first,
            CdSettings::default(),
        )
        .unwrap();
        for (a, b) in first.iter().zip(&again) {
            assert!((a - b).abs() < 1e-9, "{first:?} vs {again:?}");
        }
    }
}
