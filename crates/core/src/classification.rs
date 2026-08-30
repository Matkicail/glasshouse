//! Binary classification metrics. Weighted, and split by what they score:
//! a *decision* (threshold metrics from the confusion counts), a *ranking* (ROC-AUC, average
//! precision, KS), or the *probabilities themselves* (log-loss and Brier, which are the
//! binomial and gaussian deviances and live in `metrics`).

use crate::error::{all_values, same_length, GlassError};
use crate::ranking::sorted_tie_groups;

/// Weighted confusion counts at a threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confusion {
    pub tp: f64,
    pub fp: f64,
    pub fn_: f64,
    pub tn: f64,
}

impl Confusion {
    /// Count (with weights) how `score >= threshold` decisions land against 0/1 labels.
    ///
    /// # Errors
    /// Lengths differ; labels not exactly 0 or 1; non-finite scores; bad weights.
    pub fn at(
        y: &[f64],
        score: &[f64],
        w: Option<&[f64]>,
        threshold: f64,
    ) -> Result<Self, GlassError> {
        validate_labels(y, score, w)?;
        let mut c = Self {
            tp: 0.0,
            fp: 0.0,
            fn_: 0.0,
            tn: 0.0,
        };
        for (i, (&yi, &si)) in y.iter().zip(score).enumerate() {
            let wi = w.map_or(1.0, |w| w[i]);
            match (is_positive(yi), si >= threshold) {
                (true, true) => c.tp += wi,
                (false, true) => c.fp += wi,
                (true, false) => c.fn_ += wi,
                (false, false) => c.tn += wi,
            }
        }
        Ok(c)
    }

    #[must_use]
    pub fn accuracy(&self) -> f64 {
        (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn_)
    }

    /// Mean of recall on each class. 0.5 is guessing, whatever the class balance.
    #[must_use]
    pub fn balanced_accuracy(&self) -> f64 {
        self.recall()
            .midpoint(zero_if_nan(self.tn / (self.tn + self.fp)))
    }

    /// Of everything flagged, how much was real. 0 when nothing was flagged (scikit-learn's
    /// `zero_division=0` convention).
    #[must_use]
    pub fn precision(&self) -> f64 {
        zero_if_nan(self.tp / (self.tp + self.fp))
    }

    /// Of everything real, how much was flagged.
    #[must_use]
    pub fn recall(&self) -> f64 {
        zero_if_nan(self.tp / (self.tp + self.fn_))
    }

    /// Harmonic mean of precision and recall.
    #[must_use]
    pub fn f1(&self) -> f64 {
        zero_if_nan(2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn_))
    }

    /// Matthews correlation (phi): high only when all four cells are right. 0 when any
    /// marginal is empty (undefined; scikit-learn returns 0 there too).
    #[must_use]
    pub fn mcc(&self) -> f64 {
        let Self { tp, fp, fn_, tn } = *self;
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        zero_if_nan((tp * tn - fp * fn_) / denom).clamp(-1.0, 1.0)
    }
}

/// Labels are validated to be exactly 0 or 1, so exact comparison is the semantics here.
#[allow(clippy::float_cmp)]
#[inline]
fn is_positive(label: f64) -> bool {
    label == 1.0
}

#[allow(clippy::float_cmp)]
#[inline]
fn is_label(v: f64) -> bool {
    v == 0.0 || v == 1.0
}

/// Bounded metrics can land a rounding error outside [0, 1] (seen on macOS CI:
/// 1.0000000000000002); the true value cannot, so clamp to the real range.
#[inline]
fn unit(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

fn zero_if_nan(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// Area under the ROC curve, weighted, ties as one step (trapezoid). 0.5 is random.
///
/// # Errors
/// As [`Confusion::at`], plus both classes must be present.
pub fn roc_auc(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    let (pos, neg) = validate_ranking(y, score, w)?;
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(score);
    // walk from the highest score down: each tie group moves (fpr, tpr) by one step
    let (mut tp, mut fp, mut area) = (0.0, 0.0, 0.0);
    for &(start, end) in groups.iter().rev() {
        let (mut dtp, mut dfp) = (0.0, 0.0);
        for &row in &order[start..end] {
            if is_positive(y[row]) {
                dtp += weight(row);
            } else {
                dfp += weight(row);
            }
        }
        let (tpr0, fpr0) = (tp / pos, fp / neg);
        tp += dtp;
        fp += dfp;
        let (tpr1, fpr1) = (tp / pos, fp / neg);
        area += (fpr1 - fpr0) * (tpr0 + tpr1) / 2.0;
    }
    Ok(unit(area))
}

/// Average precision: the area under the precision–recall curve as a step function, the same
/// definition as scikit-learn's `average_precision_score`. The baseline is the positive rate.
///
/// # Errors
/// As [`roc_auc`].
pub fn average_precision(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    let (pos, _) = validate_ranking(y, score, w)?;
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(score);
    let (mut tp, mut fp, mut ap, mut prev_recall) = (0.0, 0.0, 0.0, 0.0);
    for &(start, end) in groups.iter().rev() {
        for &row in &order[start..end] {
            if is_positive(y[row]) {
                tp += weight(row);
            } else {
                fp += weight(row);
            }
        }
        let recall = tp / pos;
        let precision = tp / (tp + fp);
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    Ok(unit(ap))
}

/// Kolmogorov–Smirnov: the largest gap between the score distributions of the two classes.
/// 0 is no separation, 1 is perfect.
///
/// # Errors
/// As [`roc_auc`].
pub fn ks(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    let (pos, neg) = validate_ranking(y, score, w)?;
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let (order, groups) = sorted_tie_groups(score);
    let (mut cum_pos, mut cum_neg, mut best) = (0.0, 0.0, 0.0_f64);
    for (start, end) in groups {
        for &row in &order[start..end] {
            if is_positive(y[row]) {
                cum_pos += weight(row);
            } else {
                cum_neg += weight(row);
            }
        }
        best = best.max((cum_pos / pos - cum_neg / neg).abs());
    }
    Ok(unit(best))
}

fn validate_labels(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<(), GlassError> {
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    same_length("y", y, "score", score)?;
    all_values(
        "y",
        y,
        "must be exactly 0 or 1",
        "binary metrics need hard labels; encode the positive class as 1",
        is_label,
    )?;
    all_values(
        "score",
        score,
        "must be finite",
        "NaN or inf cannot be scored",
        f64::is_finite,
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
    }
    Ok(())
}

/// Labels + both classes present (weighted). Returns (total positive weight, total negative).
fn validate_ranking(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<(f64, f64), GlassError> {
    validate_labels(y, score, w)?;
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let pos: f64 = y
        .iter()
        .enumerate()
        .filter(|(_, &v)| is_positive(v))
        .map(|(i, _)| weight(i))
        .sum();
    let neg: f64 = y
        .iter()
        .enumerate()
        .filter(|(_, &v)| !is_positive(v))
        .map(|(i, _)| weight(i))
        .sum();
    if pos <= 0.0 || neg <= 0.0 {
        return Err(GlassError::InvalidValues {
            name: "y",
            count: y.len(),
            rule: "must contain both classes (with positive weight)",
            fix: "a ranking metric needs positives and negatives to separate",
        });
    }
    Ok((pos, neg))
}

#[cfg(test)]
mod tests {
    use super::*;

    const Y: [f64; 6] = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    const S: [f64; 6] = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1];

    #[test]
    fn confusion_at_half() {
        let c = Confusion::at(&Y, &S, None, 0.5).unwrap();
        assert_eq!((c.tp, c.fp, c.fn_, c.tn), (2.0, 1.0, 1.0, 2.0));
        assert!((c.accuracy() - 4.0 / 6.0).abs() < 1e-15);
        assert!((c.precision() - 2.0 / 3.0).abs() < 1e-15);
        assert!((c.recall() - 2.0 / 3.0).abs() < 1e-15);
        assert!((c.f1() - 2.0 / 3.0).abs() < 1e-15);
        assert!((c.mcc() - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn perfect_and_random_ranking() {
        assert!((roc_auc(&[0.0, 1.0], &[0.1, 0.9], None).unwrap() - 1.0).abs() < 1e-15);
        assert!((roc_auc(&[0.0, 1.0], &[0.5, 0.5], None).unwrap() - 0.5).abs() < 1e-15);
        assert!((average_precision(&[0.0, 1.0], &[0.1, 0.9], None).unwrap() - 1.0).abs() < 1e-15);
        assert!((ks(&[0.0, 1.0], &[0.1, 0.9], None).unwrap() - 1.0).abs() < 1e-15);
        assert!(ks(&[0.0, 1.0], &[0.5, 0.5], None).unwrap().abs() < 1e-15);
    }

    #[test]
    #[allow(clippy::float_cmp)] // asserting the exact 0.0 convention
    fn undefined_mcc_is_zero_like_sklearn() {
        let c = Confusion::at(&[1.0, 1.0], &[0.9, 0.8], None, 0.5).unwrap();
        assert_eq!(c.mcc(), 0.0);
    }

    #[test]
    fn refuses_soft_labels_and_one_class() {
        assert!(Confusion::at(&[0.5, 1.0], &[0.1, 0.9], None, 0.5).is_err());
        assert!(roc_auc(&[1.0, 1.0], &[0.1, 0.9], None).is_err());
    }
}
