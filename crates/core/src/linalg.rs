//! The little linear algebra a dense GLM needs: symmetric positive-definite solve and inverse
//! by Cholesky. `p` (features) is small; `n` (rows) is where the time goes, and that is a
//! plain loop in `glm.rs`.

use crate::error::GlassError;

/// Row-major square matrix, `p x p`.
#[derive(Debug, Clone, PartialEq)]
pub struct Square {
    pub p: usize,
    pub data: Vec<f64>,
}

impl Square {
    #[must_use]
    pub fn zeros(p: usize) -> Self {
        Self {
            p,
            data: vec![0.0; p * p],
        }
    }

    #[inline]
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.p + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.p + j] = v;
    }

    /// Lower Cholesky factor `L` with `L L^T = self`.
    ///
    /// # Errors
    /// The matrix is not positive definite — for a GLM that means collinear (or constant)
    /// columns in the design; the message names the offending column index.
    pub fn cholesky(&self) -> Result<Self, GlassError> {
        let dim = self.p;
        let mut low = Self::zeros(dim);
        for j in 0..dim {
            let mut diag = self.get(j, j);
            for k in 0..j {
                diag -= low.get(j, k) * low.get(j, k);
            }
            // Relative tolerance: a pivot that has lost all its digits is a dependent column.
            if diag <= 1e-12 * self.get(j, j).abs().max(1e-300) {
                return Err(GlassError::Singular { column: j });
            }
            let pivot = diag.sqrt();
            low.set(j, j, pivot);
            for i in (j + 1)..dim {
                let mut acc = self.get(i, j);
                for k in 0..j {
                    acc -= low.get(i, k) * low.get(j, k);
                }
                low.set(i, j, acc / pivot);
            }
        }
        Ok(low)
    }

    /// Solve `self x = b` via a Cholesky factor `l` of `self`.
    // Index loops are the readable form of a triangular solve.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn solve_with(factor: &Self, rhs: &[f64]) -> Vec<f64> {
        let dim = factor.p;
        // forward: L y = rhs
        let mut fwd = vec![0.0; dim];
        for i in 0..dim {
            let mut acc = rhs[i];
            for k in 0..i {
                acc -= factor.get(i, k) * fwd[k];
            }
            fwd[i] = acc / factor.get(i, i);
        }
        // backward: L^T x = y
        let mut sol = vec![0.0; dim];
        for i in (0..dim).rev() {
            let mut acc = fwd[i];
            for k in (i + 1)..dim {
                acc -= factor.get(k, i) * sol[k];
            }
            sol[i] = acc / factor.get(i, i);
        }
        sol
    }

    /// Inverse via a Cholesky factor `l` of `self`.
    #[must_use]
    pub fn inverse_with(factor: &Self) -> Self {
        let dim = factor.p;
        let mut inv = Self::zeros(dim);
        for j in 0..dim {
            let mut unit = vec![0.0; dim];
            unit[j] = 1.0;
            let col = Self::solve_with(factor, &unit);
            for (i, v) in col.into_iter().enumerate() {
                inv.set(i, j, v);
            }
        }
        inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_spd_system() {
        let a = Square {
            p: 2,
            data: vec![4.0, 2.0, 2.0, 3.0],
        };
        let l = a.cholesky().unwrap();
        let x = Square::solve_with(&l, &[2.0, 1.0]);
        assert!((x[0] - 0.5).abs() < 1e-14 && x[1].abs() < 1e-14, "{x:?}");
        let inv = Square::inverse_with(&l);
        assert!((inv.get(0, 0) - 0.375).abs() < 1e-14);
    }

    #[test]
    fn detects_collinearity() {
        let a = Square {
            p: 2,
            data: vec![1.0, 1.0, 1.0, 1.0],
        };
        assert!(matches!(
            a.cholesky(),
            Err(GlassError::Singular { column: 1 })
        ));
    }
}
