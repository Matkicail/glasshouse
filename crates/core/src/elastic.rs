//! Elastic-net penalties by coordinate descent: the weighted least squares step of a
//! lasso / ridge / elastic-net GLM, solved one coefficient at a time with soft-thresholding
//! (Friedman, Hastie & Tibshirani, "Regularization paths for generalized linear models via
//! coordinate descent", *J. Statistical Software* 33, 2010 — the glmnet algorithm).
//!
//! The penalty is glmnet's and glum's: in the units the outer loop works in (the deviance),
//! `2 * sum(w) * alpha * (l1 * sum|b_j| + (1 - l1) / 2 * sum b_j^2)` over the penalised
//! columns, so `alpha` means the same number here as there. The intercept and any column
//! not marked penalised are free.

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
}

impl ElasticNet<'_> {
    /// The penalty in deviance units for these coefficients.
    #[must_use]
    pub fn value(&self, coef: &[f64], weight_sum: f64) -> f64 {
        let (mut l1, mut l2) = (0.0, 0.0);
        for (b, &pen) in coef.iter().zip(self.penalised) {
            if pen {
                l1 += b.abs();
                l2 += b * b;
            }
        }
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
            tol: 1e-10,
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
}

impl CdProblem<'_> {
    fn col(&self, j: usize) -> &[f64] {
        &self.xt[j * self.n_rows..(j + 1) * self.n_rows]
    }
}

/// Minimise `1/2 sum ww_i (z_i - x_i b)^2 + scale * (l1 * sum|b_j| + (1 - l1)/2 * sum b_j^2)`
/// over the penalised `j`, the rest free, starting from `start`.
///
/// Cyclic sweeps over every coefficient, then over the active set until it settles, then a
/// full sweep to confirm.
///
/// # Errors
/// The sweep cap is hit without settling.
pub fn coordinate_descent(
    cd: &CdProblem<'_>,
    start: &[f64],
    settings: CdSettings,
) -> Result<Vec<f64>, GlassError> {
    let (n_rows, ww, p) = (cd.n_rows, cd.ww, start.len());
    let norms: Vec<f64> = (0..p)
        .map(|j| chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i] * cd.col(j)[i]))
        .collect();
    let mut coef = start.to_vec();
    // residual z - X b, kept current as coefficients move
    let mut resid = cd.z.to_vec();
    for (j, &b) in coef.iter().enumerate() {
        if b != 0.0 {
            axpy(&mut resid, cd.col(j), -b);
        }
    }
    let threshold = cd.scale * cd.l1_ratio;
    let ridge = cd.scale * (1.0 - cd.l1_ratio);
    let mut sweeps = 0;
    let sweep = |coef: &mut Vec<f64>, resid: &mut Vec<f64>, only_active: bool| -> f64 {
        let mut max_step: f64 = 0.0;
        for j in 0..p {
            if norms[j] == 0.0 || (only_active && cd.penalised[j] && coef[j] == 0.0) {
                continue;
            }
            let rho = chunk_sum(n_rows, |i| ww[i] * cd.col(j)[i] * resid[i]) + norms[j] * coef[j];
            let new = if cd.penalised[j] {
                soft_threshold(rho, threshold) / (norms[j] + ridge)
            } else {
                rho / norms[j]
            };
            let step = new - coef[j];
            if step != 0.0 {
                axpy(resid, cd.col(j), -step);
                coef[j] = new;
                max_step = max_step.max(step.abs());
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
