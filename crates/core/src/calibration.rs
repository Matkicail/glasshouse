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
    validate(y, mu, w)?;
    if n_bins == 0 {
        return Err(GlassError::BadArgument {
            name: "n_bins",
            problem: "must be at least 1",
            fix: "10 gives deciles; use fewer bins on small data",
        });
    }
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(mu);
    let total_w: f64 = (0..y.len()).map(weight).sum();
    #[allow(clippy::cast_precision_loss)]
    let bin_width = total_w / n_bins as f64;

    let mut bins: Vec<CalibrationBin> = Vec::with_capacity(n_bins);
    let mut current = Accumulator::default();
    let mut cum_w = 0.0;
    let mut next_edge = bin_width;
    for (start, end) in groups {
        let rows = &order[start..end];
        let group_w: f64 = rows.iter().map(|&r| weight(r)).sum();
        // A tie group belongs to the bin its midpoint falls in; open a new bin when the
        // midpoint has crossed the current edge (and the current bin is not empty).
        let midpoint = cum_w + group_w / 2.0;
        while midpoint >= next_edge && current.n_rows > 0 && bins.len() + 1 < n_bins {
            bins.push(current.finish());
            current = Accumulator::default();
            next_edge += bin_width;
        }
        for &r in rows {
            current.add(y[r], mu[r], weight(r));
        }
        cum_w += group_w;
    }
    if current.n_rows > 0 {
        bins.push(current.finish());
    }
    Ok(bins)
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

#[derive(Default)]
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
