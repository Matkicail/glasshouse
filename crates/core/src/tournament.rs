//! Win sets and the tournament: what happens when several models price the same risks and
//! every risk goes to the cheapest.
//!
//! Row `i` is won by the model(s) whose prediction is the lowest; a tie is split equally
//! among the tied models. Each model's win set is then summed: the weight it won (its
//! share of the market), what it charged there (`sum w * p_m`) and what it actually cost
//! (`sum w * y`). A model that wins business it under-prices has `actual > predicted` on
//! its win set: adverse selection, the thing a pricing review is trying to catch. Two
//! models make the pairwise win sets; all of them make the tournament.

use crate::error::{all_values, same_length, GlassError};

/// One model's win set, summed.
#[derive(Debug, Clone, PartialEq)]
pub struct WinSet {
    /// Weight won (shares of the tied rows included).
    pub weight: f64,
    /// `sum w * p` over the rows won: what the model would have charged.
    pub predicted: f64,
    /// `sum w * y` over the rows won: what those rows actually cost.
    pub actual: f64,
}

/// Every row goes to the model(s) with the lowest prediction, ties split equally; one
/// [`WinSet`] per model, in the order given.
///
/// # Errors
/// No models; lengths differ; non-finite `y` or predictions; non-finite or negative
/// weights.
// Exact equality decides a tie, on purpose: rounding-noise "ties" would be broken anyway.
#[allow(clippy::float_cmp)]
pub fn win_sets(
    y: &[f64],
    predictions: &[&[f64]],
    w: Option<&[f64]>,
) -> Result<Vec<WinSet>, GlassError> {
    validate(y, predictions, w)?;
    let mut out = vec![
        WinSet {
            weight: 0.0,
            predicted: 0.0,
            actual: 0.0,
        };
        predictions.len()
    ];
    for i in 0..y.len() {
        let wi = w.map_or(1.0, |w| w[i]);
        let lowest = predictions
            .iter()
            .map(|p| p[i])
            .fold(f64::INFINITY, f64::min);
        #[allow(clippy::cast_precision_loss)]
        let n_tied = predictions.iter().filter(|p| p[i] == lowest).count() as f64;
        for (m, p) in predictions.iter().enumerate() {
            if p[i] == lowest {
                let share = wi / n_tied;
                out[m].weight += share;
                out[m].predicted += share * p[i];
                out[m].actual += share * y[i];
            }
        }
    }
    Ok(out)
}

fn validate(y: &[f64], predictions: &[&[f64]], w: Option<&[f64]>) -> Result<(), GlassError> {
    if predictions.is_empty() {
        return Err(GlassError::Empty {
            name: "predictions",
        });
    }
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    all_values("y", y, "must be finite", "NaN or inf in y", f64::is_finite)?;
    for p in predictions {
        same_length("y", y, "prediction", p)?;
        all_values(
            "prediction",
            p,
            "must be finite",
            "NaN or inf in a model's predictions",
            f64::is_finite,
        )?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheapest_wins_and_ties_split() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let a = [0.5, 3.0, 2.0, 4.0];
        let b = [1.0, 2.0, 2.0, 5.0];
        let w = [1.0, 1.0, 2.0, 1.0];
        let sets = win_sets(&y, &[&a, &b], Some(&w)).unwrap();
        // row 0 -> a, row 1 -> b, row 2 tied (weight 2 split 1/1), row 3 -> a
        assert!((sets[0].weight - 3.0).abs() < 1e-12 && (sets[1].weight - 2.0).abs() < 1e-12);
        assert!((sets[0].predicted - (0.5 + 2.0 + 4.0)).abs() < 1e-12);
        assert!((sets[0].actual - (1.0 + 3.0 + 4.0)).abs() < 1e-12);
        assert!((sets[1].predicted - (2.0 + 2.0)).abs() < 1e-12);
        assert!((sets[1].actual - (2.0 + 3.0)).abs() < 1e-12);
        // the win sets partition the market
        let total_w: f64 = sets.iter().map(|s| s.weight).sum();
        let total_y: f64 = sets.iter().map(|s| s.actual).sum();
        assert!((total_w - 5.0).abs() < 1e-12 && (total_y - 13.0).abs() < 1e-12);
    }

    #[test]
    fn one_model_takes_the_whole_market() {
        let y = [1.0, 2.0];
        let p = [3.0, 4.0];
        let sets = win_sets(&y, &[&p], None).unwrap();
        assert_eq!(sets.len(), 1);
        assert!((sets[0].weight - 2.0).abs() < 1e-12 && (sets[0].predicted - 7.0).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_input() {
        let y = [1.0, 2.0];
        assert!(win_sets(&y, &[], None).is_err());
        assert!(win_sets(&y, &[&[1.0]], None).is_err());
        assert!(win_sets(&y, &[&[1.0, f64::NAN]], None).is_err());
        assert!(win_sets(&y, &[&[1.0, 2.0]], Some(&[1.0, -1.0])).is_err());
    }
}
