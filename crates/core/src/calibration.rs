//! Calibration: "can I trust the number, not just the order?"
//!
//! Bin rows by predicted value (weighted quantiles, ties kept whole), then compare the mean
//! prediction to the mean outcome in each bin. A calibrated model has actual / expected ≈ 1
//! everywhere; the balance property is the same check on the whole portfolio at once.

use crate::error::{all_values, same_length, GlassError};
use crate::ranking::sorted_tie_groups;

/// One row of the calibration table.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBin {
    /// Number of rows in the bin.
    pub n_rows: usize,
    /// Total weight (exposure) in the bin.
    pub weight: f64,
    /// Weighted mean of `mu`.
    pub predicted: f64,
    /// Weighted mean of `y`.
    pub actual: f64,
    /// `actual / predicted`. 1 is calibrated, > 1 the model under-predicts here.
    pub actual_over_expected: f64,
}

/// Actual-over-expected by prediction bin.
///
/// Rows are sorted by `mu`; bin edges fall at equal shares of total weight, so with unit
/// weights and `n_bins = 10` these are the usual deciles. Rows with the same `mu` always land
/// in the same bin, so a bin can end up larger than its share — never split a tie.
///
/// # Errors
/// Lengths differ; non-finite `y` or `mu` (`mu` may be negative — gaussian — but `sum(w * mu)`
/// must be non-zero for the ratio); `w` not finite and `> 0`; `n_bins == 0`.
pub fn calibration_table(
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
    n_bins: usize,
) -> Result<Vec<CalibrationBin>, GlassError> {
    binned_table(mu, y, mu, w, n_bins)
}

/// The calibration table binned by any `key` instead of the prediction: actual / expected by
/// a *feature* (equal-weight bins of the feature, ties whole). This is the view that finds the
/// segment a model gets wrong.
///
/// # Errors
/// As [`calibration_table`], plus a non-finite `key` or a length mismatch.
pub fn binned_table(
    key: &[f64],
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
    n_bins: usize,
) -> Result<Vec<CalibrationBin>, GlassError> {
    validate(y, mu, w)?;
    let index = bin_index(key, w, n_bins)?;
    let n = index.iter().max().map_or(0, |m| m + 1);
    grid_table(&index, n, &vec![0; key.len()], 1, y, mu, w)
}

/// Which equal-weight bin each row of `key` falls in, `0..n_bins`, ties kept whole.
///
/// Rows are sorted by `key`; bin edges are fixed at equal shares of the total weight; a tie
/// group belongs to the bin its weight midpoint falls in, so a bin can be larger than its
/// share, a group that spans several edges takes them all, and the number of bins used can
/// be smaller than `n_bins` (all tied: one bin). This is the one binning rule behind the
/// calibration table, A/E by feature and A/E by two features.
///
/// # Errors
/// `key` non-finite; `w` (if given) not the same length as `key`; `n_bins == 0`.
pub fn bin_index(key: &[f64], w: Option<&[f64]>, n_bins: usize) -> Result<Vec<usize>, GlassError> {
    all_values(
        "key",
        key,
        "must be finite",
        "NaN or inf in the binning column",
        f64::is_finite,
    )?;
    if let Some(w) = w {
        same_length("key", key, "sample_weight", w)?;
    }
    if n_bins == 0 {
        return Err(GlassError::BadArgument {
            name: "n_bins",
            problem: "must be at least 1",
            fix: "10 gives deciles; use fewer bins on small data",
        });
    }
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(key);
    let total_w: f64 = (0..key.len()).map(weight).sum();
    #[allow(clippy::cast_precision_loss)]
    let bin_width = total_w / n_bins as f64;
    let mut index = vec![0; key.len()];
    let mut bin = 0;
    let mut bin_rows = 0;
    let mut cum_w = 0.0;
    let mut next_edge = bin_width;
    for (start, end) in groups {
        let rows = &order[start..end];
        let group_w: f64 = rows.iter().map(|&r| weight(r)).sum();
        // A tie group belongs to the bin its midpoint falls in; open a new bin when the
        // midpoint has crossed the current edge (and the current bin is not empty). A group
        // heavy enough to cross several edges at once skips them all: the edges are fixed,
        // so the rows after it are still cut at equal shares, never into near-empty bins.
        let midpoint = cum_w + group_w / 2.0;
        if midpoint >= next_edge && bin_rows > 0 && bin + 1 < n_bins {
            bin += 1;
            bin_rows = 0;
            while midpoint >= next_edge {
                next_edge += bin_width;
            }
        }
        for &r in rows {
            index[r] = bin;
        }
        bin_rows += rows.len();
        cum_w += group_w;
    }
    Ok(index)
}

/// Actual / expected on a grid: cell `(i, j)` (row-major, `i * n_b + j`) holds the rows whose
/// `index_a` is `i` and `index_b` is `j`. With `n_b == 1` this is the one-way table; with two
/// features it is the interaction view that finds the segment a one-way A/E averages away.
/// An empty cell has `n_rows == 0`, zero weight and NaN means.
///
/// # Errors
/// As [`calibration_table`]; an index out of range or of the wrong length.
pub fn grid_table(
    index_a: &[usize],
    n_a: usize,
    index_b: &[usize],
    n_b: usize,
    y: &[f64],
    mu: &[f64],
    w: Option<&[f64]>,
) -> Result<Vec<CalibrationBin>, GlassError> {
    validate(y, mu, w)?;
    same_length("index_a", index_a, "y", y)?;
    same_length("index_b", index_b, "y", y)?;
    if index_a.iter().any(|&i| i >= n_a) || index_b.iter().any(|&j| j >= n_b) {
        return Err(GlassError::BadArgument {
            name: "index",
            problem: "a bin index is outside the grid",
            fix: "indices must be < n_a and < n_b",
        });
    }
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let mut cells = vec![Accumulator::default(); n_a * n_b];
    for (row, (&i, &j)) in index_a.iter().zip(index_b).enumerate() {
        cells[i * n_b + j].add(y[row], mu[row], weight(row));
    }
    Ok(cells.iter().map(Accumulator::finish).collect())
}

/// One row of a double-lift table.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleLiftBin {
    pub n_rows: usize,
    pub weight: f64,
    /// Weighted mean of `mu_a / mu_b` in the bin (the sort key).
    pub ratio: f64,
    pub actual: f64,
    pub predicted_a: f64,
    pub predicted_b: f64,
}

/// Double lift: sort rows by `mu_a / mu_b`, bin by equal weight, and in each bin compare both
/// models' mean prediction to the mean outcome. Where the two models disagree most, which one
/// is closer to reality? The most defensible chart in a model review.
///
/// # Errors
/// As [`calibration_table`], plus `mu_b` must be `> 0` (it is a divisor).
pub fn double_lift_table(
    y: &[f64],
    mu_a: &[f64],
    mu_b: &[f64],
    w: Option<&[f64]>,
    n_bins: usize,
) -> Result<Vec<DoubleLiftBin>, GlassError> {
    validate(y, mu_a, w)?;
    same_length("mu_a", mu_a, "mu_b", mu_b)?;
    all_values(
        "mu_b",
        mu_b,
        "must be finite and > 0",
        "the double lift sorts by mu_a / mu_b, so the second model's predictions must be positive",
        |v| v.is_finite() && v > 0.0,
    )?;
    if n_bins == 0 {
        return Err(GlassError::BadArgument {
            name: "n_bins",
            problem: "must be at least 1",
            fix: "10 gives deciles; use fewer bins on small data",
        });
    }
    let ratio: Vec<f64> = mu_a.iter().zip(mu_b).map(|(a, b)| a / b).collect();
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(&ratio);
    let total_w: f64 = (0..y.len()).map(weight).sum();
    #[allow(clippy::cast_precision_loss)]
    let bin_width = total_w / n_bins as f64;
    let mut bins: Vec<DoubleLiftBin> = Vec::with_capacity(n_bins);
    let mut cur = DoubleAcc::default();
    let mut cum_w = 0.0;
    let mut next_edge = bin_width;
    for (start, end) in groups {
        let rows = &order[start..end];
        let group_w: f64 = rows.iter().map(|&r| weight(r)).sum();
        let midpoint = cum_w + group_w / 2.0;
        while midpoint >= next_edge && cur.n_rows > 0 && bins.len() + 1 < n_bins {
            bins.push(cur.finish());
            cur = DoubleAcc::default();
            next_edge += bin_width;
        }
        for &r in rows {
            cur.add(y[r], mu_a[r], mu_b[r], ratio[r], weight(r));
        }
        cum_w += group_w;
    }
    if cur.n_rows > 0 {
        bins.push(cur.finish());
    }
    Ok(bins)
}

#[derive(Default)]
struct DoubleAcc {
    n_rows: usize,
    weight: f64,
    wy: f64,
    wa: f64,
    wb: f64,
    wr: f64,
}

impl DoubleAcc {
    fn add(&mut self, actual: f64, pred_a: f64, pred_b: f64, ratio: f64, weight: f64) {
        self.n_rows += 1;
        self.weight += weight;
        self.wy += weight * actual;
        self.wa += weight * pred_a;
        self.wb += weight * pred_b;
        self.wr += weight * ratio;
    }

    fn finish(&self) -> DoubleLiftBin {
        DoubleLiftBin {
            n_rows: self.n_rows,
            weight: self.weight,
            ratio: self.wr / self.weight,
            actual: self.wy / self.weight,
            predicted_a: self.wa / self.weight,
            predicted_b: self.wb / self.weight,
        }
    }
}

/// The balance property: `sum(w * y) / sum(w * mu)`. 1 means the model reproduces the total.
///
/// A GLM with the canonical link is balanced on its training data by construction; anything
/// trained by gradient descent usually is not — and a price that is 3 % under on the whole
/// book is a problem no Gini will show you.
///
/// # Errors
/// As [`calibration_table`].
pub fn balance(y: &[f64], mu: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    validate(y, mu, w)?;
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let actual: f64 = y.iter().enumerate().map(|(i, &v)| weight(i) * v).sum();
    let expected: f64 = mu.iter().enumerate().map(|(i, &v)| weight(i) * v).sum();
    Ok(actual / expected)
}

#[derive(Default, Clone)]
struct Accumulator {
    n_rows: usize,
    weight: f64,
    wy: f64,
    wmu: f64,
}

impl Accumulator {
    fn add(&mut self, y: f64, mu: f64, w: f64) {
        self.n_rows += 1;
        self.weight += w;
        self.wy += w * y;
        self.wmu += w * mu;
    }

    fn finish(&self) -> CalibrationBin {
        CalibrationBin {
            n_rows: self.n_rows,
            weight: self.weight,
            predicted: self.wmu / self.weight,
            actual: self.wy / self.weight,
            actual_over_expected: self.wy / self.wmu,
        }
    }
}

fn validate(y: &[f64], mu: &[f64], w: Option<&[f64]>) -> Result<(), GlassError> {
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
            "must be finite and > 0",
            "a zero weight contributes nothing; drop those rows",
            |v| v.is_finite() && v > 0.0,
        )?;
    }
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let expected: f64 = mu.iter().enumerate().map(|(i, &v)| weight(i) * v).sum();
    if expected == 0.0 {
        return Err(GlassError::InvalidValues {
            name: "mu",
            count: mu.len(),
            rule: "must not sum to zero",
            fix: "actual / expected needs a non-zero expected total",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deciles_with_unit_weights() {
        let mu: Vec<f64> = (0..100).map(f64::from).collect();
        let y = mu.clone();
        let bins = calibration_table(&y, &mu, None, 10).unwrap();
        assert_eq!(bins.len(), 10);
        assert!(bins.iter().all(|b| b.n_rows == 10));
        assert!(bins
            .iter()
            .all(|b| (b.actual_over_expected - 1.0).abs() < 1e-12));
        assert!((bins[0].predicted - 4.5).abs() < 1e-12);
    }

    #[test]
    fn ties_never_split() {
        let mu = [1.0, 1.0, 1.0, 2.0];
        let y = [1.0, 1.0, 1.0, 2.0];
        let bins = calibration_table(&y, &mu, None, 2).unwrap();
        assert_eq!(bins.len(), 2);
        assert_eq!(bins[0].n_rows, 3);
        assert_eq!(bins[1].n_rows, 1);
    }

    #[test]
    fn all_tied_is_one_bin() {
        let bins = calibration_table(&[1.0, 2.0], &[0.5, 0.5], None, 10).unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].n_rows, 2);
    }

    #[test]
    fn bin_index_reproduces_the_binned_table() {
        let key = [3.0, 1.0, 2.0, 2.0, 5.0, 4.0, 2.0, 0.5];
        let w = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0];
        let y = [1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 1.0];
        let mu = [1.0, 1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 0.5];
        let idx = bin_index(&key, Some(&w), 3).unwrap();
        // ties (the three 2.0s) share a bin; sorted order maps to non-decreasing bins
        assert_eq!(idx[2], idx[3]);
        assert_eq!(idx[3], idx[6]);
        assert!(idx[7] <= idx[1] && idx[1] <= idx[2] && idx[2] <= idx[0]);
        let table = binned_table(&key, &y, &mu, Some(&w), 3).unwrap();
        let n = idx.iter().max().unwrap() + 1;
        assert_eq!(table.len(), n);
        let by_hand: Vec<f64> = (0..n)
            .map(|b| {
                (0..8)
                    .filter(|&r| idx[r] == b)
                    .map(|r| w[r] * y[r])
                    .sum::<f64>()
            })
            .collect();
        for (b, t) in table.iter().enumerate() {
            assert!((t.actual * t.weight - by_hand[b]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_heavy_tie_group_takes_the_edges_it_spans() {
        // 60 rows tied at 0, then 40 distinct values: deciles must give the tie one bin and
        // cut the remaining 40 rows at the fixed edges 70, 80, 90 -> four bins of 10
        let key: Vec<f64> = (0..100)
            .map(|i| if i < 60 { 0.0 } else { f64::from(i) })
            .collect();
        let idx = bin_index(&key, None, 10).unwrap();
        let mut counts = vec![0; 10];
        for &b in &idx {
            counts[b] += 1;
        }
        assert_eq!(&counts[..5], &[60, 10, 10, 10, 10], "{counts:?}");
        assert!(counts[5..].iter().all(|&c| c == 0));
    }

    #[test]
    fn grid_marginals_are_the_one_way_tables() {
        let a = [0, 0, 1, 1, 2, 2, 0, 1];
        let b = [0, 1, 0, 1, 0, 1, 1, 1];
        let y = [1.0, 2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0];
        let mu = [1.0, 1.0, 1.5, 1.5, 2.0, 1.0, 1.5, 0.5];
        let w = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0];
        let grid = grid_table(&a, 3, &b, 2, &y, &mu, Some(&w)).unwrap();
        assert_eq!(grid.len(), 6);
        let rows = grid_table(&a, 3, &[0; 8], 1, &y, &mu, Some(&w)).unwrap();
        for i in 0..3 {
            let wt: f64 = (0..2).map(|j| grid[i * 2 + j].weight).sum();
            let wy: f64 = (0..2)
                .map(|j| grid[i * 2 + j].weight * grid[i * 2 + j].actual)
                .sum();
            assert!((wt - rows[i].weight).abs() < 1e-12);
            assert!((wy / wt - rows[i].actual).abs() < 1e-12);
        }
        // an empty cell is honest about it
        let sparse = grid_table(&[0, 0], 2, &[0, 0], 2, &[1.0, 1.0], &[1.0, 1.0], None).unwrap();
        assert_eq!(sparse[3].n_rows, 0);
        assert!(sparse[3].actual_over_expected.is_nan());
        assert!(grid_table(&[0, 2], 2, &[0, 0], 1, &[1.0, 1.0], &[1.0, 1.0], None).is_err());
    }

    #[test]
    fn balance_is_overall_ratio() {
        let b = balance(&[1.0, 3.0], &[2.0, 2.0], Some(&[1.0, 3.0])).unwrap();
        assert!((b - 10.0 / 8.0).abs() < 1e-15, "{b}");
    }

    #[test]
    fn refuses_bad_input() {
        assert!(calibration_table(&[1.0], &[1.0], None, 0).is_err());
        assert!(balance(&[1.0, 1.0], &[1.0, -1.0], None).is_err());
        assert!(balance(&[1.0], &[f64::NAN], None).is_err());
    }
}
