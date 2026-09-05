//! Monotone (shape) constraints on a run of coefficients, and the active-set QP that each
//! IRLS step of a constrained fit solves.
//!
//! A cubic B-spline is non-decreasing when its coefficients are non-decreasing along the
//! knots, so "this effect must not fall" is a chain of inequalities `beta[c1] <= beta[c2] <=
//! ...` on the columns of that smooth. Each IRLS step then minimises the usual quadratic
//! `1/2 beta' H beta - b' beta` (with `H = X'WX + S`, `b = X'Wz`) subject to those
//! inequalities. The primal active-set method below keeps a feasible iterate; an active
//! constraint is a *tie* (two adjacent coefficients equal, or a leading run held at zero),
//! and a tied run is just one merged column of the design, so every subproblem is an
//! ordinary Cholesky solve on a reduced matrix. Ties are added when a step would cross
//! them and released when their multiplier says the objective would rather move apart.
//!
//! Isotonic regression is the special case `H = I`, which is how the tests check it.

use crate::error::GlassError;
use crate::linalg::Square;

/// One ordered run of coefficients that must not decrease (or not increase) along it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    /// Design columns in the order the constraint runs.
    pub columns: Vec<usize>,
    /// `true`: non-decreasing along `columns`; `false`: non-increasing.
    pub increasing: bool,
    /// The run starts at an implicit coefficient fixed at zero (the dropped first basis
    /// column of a spline), so the first coefficient is itself constrained against zero.
    pub anchored: bool,
}

impl Chain {
    fn sign(&self) -> f64 {
        if self.increasing {
            1.0
        } else {
            -1.0
        }
    }
}

/// How one chain's columns are currently tied: consecutive runs of positions, the first of
/// which may be held at zero.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Partition {
    /// Half-open position ranges `[a, b)` into `chain.columns`, in order, covering it.
    runs: Vec<(usize, usize)>,
    /// The first run is tied to zero (only possible for an anchored chain).
    zero: bool,
}

/// The tie structure of `beta` under `chains`: exact equal neighbours are tied, an exactly
/// zero leading run of an anchored chain is held at zero. Exact equality is the point: a
/// solve returns tied coefficients bit-identical, and a convex combination of two tied
/// pairs stays tied.
// Exact float comparison is the point here (see above).
#[allow(clippy::float_cmp)]
fn partition_of(chain: &Chain, beta: &[f64]) -> Partition {
    let m = chain.columns.len();
    let mut runs = Vec::new();
    let mut start = 0;
    for k in 1..m {
        if beta[chain.columns[k]] != beta[chain.columns[k - 1]] {
            runs.push((start, k));
            start = k;
        }
    }
    runs.push((start, m));
    let zero = chain.anchored && beta[chain.columns[0]] == 0.0;
    Partition { runs, zero }
}

/// The free parameters of the reduced problem: each is the set of design columns it
/// expands to. Columns held at zero belong to none.
fn groups(chains: &[Chain], parts: &[Partition], p: usize) -> Vec<Vec<usize>> {
    let mut in_chain = vec![false; p];
    for c in chains {
        for &col in &c.columns {
            in_chain[col] = true;
        }
    }
    let mut out: Vec<Vec<usize>> = (0..p).filter(|&j| !in_chain[j]).map(|j| vec![j]).collect();
    for (c, part) in chains.iter().zip(parts) {
        for (r, &(a, b)) in part.runs.iter().enumerate() {
            if r == 0 && part.zero {
                continue;
            }
            out.push(c.columns[a..b].to_vec());
        }
    }
    out
}

/// `E' M E` for the expansion `E` implied by `groups`.
#[must_use]
pub fn reduce(m: &Square, groups: &[Vec<usize>]) -> Square {
    let q = groups.len();
    let mut out = Square::zeros(q);
    for (a, ga) in groups.iter().enumerate() {
        for (b, gb) in groups.iter().enumerate() {
            let v: f64 = ga
                .iter()
                .map(|&i| gb.iter().map(|&j| m.get(i, j)).sum::<f64>())
                .sum();
            out.set(a, b, v);
        }
    }
    out
}

/// `E M E'`: a reduced `q x q` matrix spread back over the `p` design columns (tied columns
/// share entries; columns held at zero get zeros).
#[must_use]
pub fn expand(m: &Square, groups: &[Vec<usize>], p: usize) -> Square {
    let mut out = Square::zeros(p);
    for (a, ga) in groups.iter().enumerate() {
        for (b, gb) in groups.iter().enumerate() {
            let v = m.get(a, b);
            for &i in ga {
                for &j in gb {
                    out.set(i, j, v);
                }
            }
        }
    }
    out
}

fn reduce_vector(v: &[f64], groups: &[Vec<usize>]) -> Vec<f64> {
    groups
        .iter()
        .map(|g| g.iter().map(|&i| v[i]).sum())
        .collect()
}

fn expand_vector(v: &[f64], groups: &[Vec<usize>], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p];
    for (a, g) in groups.iter().enumerate() {
        for &i in g {
            out[i] = v[a];
        }
    }
    out
}

/// The groups the final coefficients imply (for the edf and the covariance of a fit).
#[must_use]
pub fn groups_at(chains: &[Chain], beta: &[f64]) -> Vec<Vec<usize>> {
    let parts: Vec<Partition> = chains.iter().map(|c| partition_of(c, beta)).collect();
    groups(chains, &parts, beta.len())
}

/// Is `beta` inside every chain's constraints (to a small slack for rounding)?
#[must_use]
pub fn feasible(chains: &[Chain], beta: &[f64]) -> bool {
    chains.iter().all(|c| {
        let s = c.sign();
        let tol = 1e-12 * (1.0 + c.columns.iter().map(|&j| beta[j].abs()).fold(0.0, f64::max));
        let anchor_ok = !c.anchored || s * beta[c.columns[0]] >= -tol;
        anchor_ok
            && c.columns
                .windows(2)
                .all(|w| s * (beta[w[1]] - beta[w[0]]) >= -tol)
    })
}

/// Minimise `1/2 beta' h beta - b' beta` subject to the chains, starting from the feasible
/// `start`. `h` must be positive definite.
///
/// # Errors
/// `start` violates a constraint; `h` is not positive definite; the active set failed to
/// settle (which the theory says cannot happen for a positive-definite `h`).
pub fn solve_constrained(
    hess: &Square,
    rhs: &[f64],
    chains: &[Chain],
    start: &[f64],
) -> Result<Vec<f64>, GlassError> {
    if !feasible(chains, start) {
        return Err(GlassError::BadArgument {
            name: "start",
            problem: "violates a monotone constraint",
            fix: "start the constrained solve from a feasible point (zeros always are)",
        });
    }
    let dim = rhs.len();
    let mut beta = start.to_vec();
    let mut parts: Vec<Partition> = chains.iter().map(|c| partition_of(c, &beta)).collect();
    let n_constraints: usize = chains
        .iter()
        .map(|c| c.columns.len() - usize::from(!c.anchored))
        .sum();
    for _ in 0..(20 + 4 * n_constraints * n_constraints) {
        let grp = groups(chains, &parts, dim);
        let h_r = reduce(hess, &grp);
        let b_r = reduce_vector(rhs, &grp);
        let target = expand_vector(&Square::solve_with(&h_r.cholesky()?, &b_r), &grp, dim);
        if let Some((chain_i, tie, step)) = first_blocking(chains, &parts, &beta, &target) {
            // move as far as the blocking constraint allows, then add it as a tie
            for j in 0..dim {
                beta[j] += step * (target[j] - beta[j]);
            }
            add_tie(&chains[chain_i], &mut parts[chain_i], tie, &mut beta);
        } else {
            beta = target;
            let gradient: Vec<f64> = (0..dim)
                .map(|i| (0..dim).map(|k| hess.get(i, k) * beta[k]).sum::<f64>() - rhs[i])
                .collect();
            let Some((chain_i, tie)) = most_negative_multiplier(chains, &parts, &gradient) else {
                return Ok(beta);
            };
            release_tie(&mut parts[chain_i], tie);
        }
    }
    Err(GlassError::BadArgument {
        name: "monotone",
        problem: "the active-set solve did not settle",
        fix: "this should not happen with a positive-definite system; please report it",
    })
}

/// A tie, addressed inside one chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tie {
    /// The first run held at zero.
    Anchor,
    /// Positions `k - 1` and `k` equal (`k` is a run boundary when active).
    Between(usize),
}

/// Along the segment from `beta` (feasible) to `target`, the first inactive constraint that
/// would be crossed: `(chain, tie, fraction of the step allowed)`. `None` means the whole
/// step is feasible.
fn first_blocking(
    chains: &[Chain],
    parts: &[Partition],
    beta: &[f64],
    target: &[f64],
) -> Option<(usize, Tie, f64)> {
    let mut best: Option<(usize, Tie, f64)> = None;
    let mut consider = |chain_i: usize, tie: Tie, now: f64, after: f64| {
        // constraint value `now >= 0`; if it goes negative along the step, cut the step
        if after < 0.0 {
            let t = if now <= 0.0 { 0.0 } else { now / (now - after) };
            if best.is_none_or(|(_, _, bt)| t < bt) {
                best = Some((chain_i, tie, t));
            }
        }
    };
    for (chain_i, (c, part)) in chains.iter().zip(parts).enumerate() {
        let s = c.sign();
        if c.anchored && !part.zero {
            let j = c.columns[0];
            consider(chain_i, Tie::Anchor, s * beta[j], s * target[j]);
        }
        for w in part.runs.windows(2) {
            let k = w[1].0; // boundary position between two runs: constraint k-1 -> k
            let (lo, hi) = (c.columns[k - 1], c.columns[k]);
            consider(
                chain_i,
                Tie::Between(k),
                s * (beta[hi] - beta[lo]),
                s * (target[hi] - target[lo]),
            );
        }
    }
    best
}

/// Make a tie active and set `beta` on it exactly.
fn add_tie(chain: &Chain, part: &mut Partition, tie: Tie, beta: &mut [f64]) {
    match tie {
        Tie::Anchor => {
            part.zero = true;
            let (a, b) = part.runs[0];
            for &j in &chain.columns[a..b] {
                beta[j] = 0.0;
            }
        }
        Tie::Between(k) => {
            let r = part
                .runs
                .iter()
                .position(|&(a, _)| a == k)
                .expect("a run boundary");
            let (a0, _) = part.runs[r - 1];
            let (_, b1) = part.runs[r];
            part.runs.splice(r - 1..=r, [(a0, b1)]);
            let value = if r == 1 && part.zero {
                0.0
            } else {
                beta[chain.columns[a0]]
            };
            for &j in &chain.columns[a0..b1] {
                beta[j] = value;
            }
        }
    }
}

/// Split a run (or free the zero run).
fn release_tie(part: &mut Partition, tie: Tie) {
    match tie {
        Tie::Anchor => part.zero = false,
        Tie::Between(k) => {
            let r = part
                .runs
                .iter()
                .position(|&(a, b)| a < k && k < b)
                .expect("an interior position");
            let (a, b) = part.runs[r];
            part.runs.splice(r..=r, [(a, k), (k, b)]);
        }
    }
}

/// At a reduced optimum, the KKT multiplier of every active tie, and the most negative one
/// if any is below tolerance. For a free run the multiplier at position `k` is minus the
/// signed prefix sum of the gradient over the run up to `k`; for the zero run it is the
/// signed suffix sum (the anchor's multiplier is the whole sum).
fn most_negative_multiplier(
    chains: &[Chain],
    parts: &[Partition],
    gradient: &[f64],
) -> Option<(usize, Tie)> {
    let scale = 1.0 + gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = 1e-10 * scale;
    let mut worst: Option<(usize, Tie, f64)> = None;
    let mut consider = |chain_i: usize, tie: Tie, mu: f64| {
        if mu < -tol && worst.is_none_or(|(_, _, w)| mu < w) {
            worst = Some((chain_i, tie, mu));
        }
    };
    for (chain_i, (c, part)) in chains.iter().zip(parts).enumerate() {
        let s = c.sign();
        for (r, &(a, b)) in part.runs.iter().enumerate() {
            let g: Vec<f64> = c.columns[a..b].iter().map(|&j| gradient[j]).collect();
            if r == 0 && part.zero {
                let mut suffix = 0.0;
                for k in (0..g.len()).rev() {
                    suffix += g[k];
                    let tie = if k == 0 {
                        Tie::Anchor
                    } else {
                        Tie::Between(a + k)
                    };
                    consider(chain_i, tie, s * suffix);
                }
            } else {
                let mut prefix = 0.0;
                for (k, gk) in g.iter().take(g.len() - 1).enumerate() {
                    prefix += gk;
                    consider(chain_i, Tie::Between(a + k + 1), -s * prefix);
                }
            }
        }
    }
    worst.map(|(i, t, _)| (i, t))
}

#[cfg(test)]
// Exact equality is the claim in these tests: tied coefficients are bit-identical.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn identity(p: usize) -> Square {
        let mut m = Square::zeros(p);
        for i in 0..p {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Pool-adjacent-violators: the textbook isotonic regression, for the reference.
    fn pava(y: &[f64]) -> Vec<f64> {
        let mut blocks: Vec<(f64, usize)> = Vec::new(); // (mean, size)
        for &v in y {
            blocks.push((v, 1));
            while blocks.len() >= 2 {
                let (m2, n2) = blocks[blocks.len() - 1];
                let (m1, n1) = blocks[blocks.len() - 2];
                if m1 <= m2 {
                    break;
                }
                #[allow(clippy::cast_precision_loss)]
                let merged = (m1 * n1 as f64 + m2 * n2 as f64) / (n1 + n2) as f64;
                blocks.truncate(blocks.len() - 2);
                blocks.push((merged, n1 + n2));
            }
        }
        blocks
            .iter()
            .flat_map(|&(m, n)| std::iter::repeat_n(m, n))
            .collect()
    }

    fn chain(p: usize, increasing: bool, anchored: bool) -> Chain {
        Chain {
            columns: (0..p).collect(),
            increasing,
            anchored,
        }
    }

    #[test]
    fn isotonic_regression_matches_pava() {
        let y = [1.0, 3.0, 2.0, 2.5, 6.0, 5.0, 4.0, 7.0];
        let sol = solve_constrained(&identity(8), &y, &[chain(8, true, false)], &[0.0; 8]).unwrap();
        let reference = pava(&y);
        for (a, b) in sol.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-12, "{sol:?} vs {reference:?}");
        }
    }

    #[test]
    fn decreasing_is_the_mirror_of_increasing() {
        let y = [1.0, 3.0, 2.0, 2.5, 6.0, 5.0, 4.0, 7.0];
        let neg: Vec<f64> = y.iter().map(|v| -v).collect();
        let inc = solve_constrained(&identity(8), &y, &[chain(8, true, false)], &[0.0; 8]).unwrap();
        let dec =
            solve_constrained(&identity(8), &neg, &[chain(8, false, false)], &[0.0; 8]).unwrap();
        for (a, b) in inc.iter().zip(&dec) {
            assert!((a + b).abs() < 1e-12);
        }
    }

    #[test]
    fn anchored_chain_holds_a_negative_start_at_zero() {
        // isotonic with beta_0 >= 0: the first two pool below zero and are held there
        let y = [-2.0, -1.0, 0.5, 2.0];
        let sol = solve_constrained(&identity(4), &y, &[chain(4, true, true)], &[0.0; 4]).unwrap();
        assert_eq!(sol[0], 0.0);
        assert_eq!(sol[1], 0.0);
        assert!(
            (sol[2] - 0.5).abs() < 1e-12 && (sol[3] - 2.0).abs() < 1e-12,
            "{sol:?}"
        );
        assert_eq!(
            groups_at(&[chain(4, true, true)], &sol),
            vec![vec![2], vec![3]]
        );
    }

    #[test]
    fn an_already_feasible_optimum_is_left_alone() {
        let y = [0.5, 1.0, 2.0, 3.0];
        let sol = solve_constrained(&identity(4), &y, &[chain(4, true, true)], &[0.0; 4]).unwrap();
        for (a, b) in sol.iter().zip(&y) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn kkt_conditions_hold_on_a_correlated_system() {
        // a random-ish SPD system with a chain over columns 1..5 (column 0 free)
        let dim = 6;
        let mut basis = Square::zeros(dim);
        let mut seed: u64 = 7;
        let mut next = move || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            u - 0.5
        };
        for i in 0..dim {
            for j in 0..dim {
                basis.set(i, j, next());
            }
        }
        let mut hess = Square::zeros(dim);
        for i in 0..dim {
            for j in 0..dim {
                let v: f64 = (0..dim).map(|k| basis.get(k, i) * basis.get(k, j)).sum();
                hess.set(i, j, v + if i == j { 0.5 } else { 0.0 });
            }
        }
        let rhs: Vec<f64> = (0..dim).map(|_| 3.0 * next()).collect();
        let chains = [Chain {
            columns: vec![1, 2, 3, 4, 5],
            increasing: true,
            anchored: true,
        }];
        let beta = solve_constrained(&hess, &rhs, &chains, &[0.0; 6]).unwrap();
        assert!(feasible(&chains, &beta), "{beta:?}");
        // stationarity: gradient = A' mu with mu >= 0 and mu_i * constraint_i = 0
        let grad: Vec<f64> = (0..dim)
            .map(|i| (0..dim).map(|k| hess.get(i, k) * beta[k]).sum::<f64>() - rhs[i])
            .collect();
        assert!(grad[0].abs() < 1e-10, "free column is stationary: {grad:?}");
        // multipliers of the chain constraints, solved from the gradient: mu_k = mu_{k-1} - g_k
        let cols = &chains[0].columns;
        let mut mu = vec![0.0; cols.len() + 1]; // mu[0] anchor, mu[k] between k-1 and k
        mu[0] = grad.iter().skip(1).sum::<f64>(); // whole-chain sum: the constraint past the end is inactive
        for k in 0..cols.len() {
            mu[k + 1] = mu[k] - grad[cols[k]];
        }
        assert!(mu[cols.len()].abs() < 1e-10, "past the end: {mu:?}");
        for k in 0..cols.len() {
            let value = if k == 0 {
                beta[cols[0]]
            } else {
                beta[cols[k]] - beta[cols[k - 1]]
            };
            assert!(mu[k] >= -1e-10, "dual feasible: {mu:?}");
            assert!(
                mu[k] * value < 1e-10,
                "complementary slackness: {mu:?} {beta:?}"
            );
        }
    }

    #[test]
    fn refuses_an_infeasible_start() {
        let err = solve_constrained(
            &identity(3),
            &[1.0; 3],
            &[chain(3, true, false)],
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
        assert!(err.to_string().contains("monotone"), "{err}");
    }

    #[test]
    fn reduce_and_expand_round_trip_tied_columns() {
        let h = identity(3);
        let g = vec![vec![0, 1], vec![2]];
        let r = reduce(&h, &g);
        assert_eq!(r.data, vec![2.0, 0.0, 0.0, 1.0]);
        let e = expand(&r, &g, 3);
        assert_eq!(e.get(0, 1), 2.0);
        assert_eq!(e.get(2, 2), 1.0);
        assert_eq!(e.get(0, 2), 0.0);
    }
}
