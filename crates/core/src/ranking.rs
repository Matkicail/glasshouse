//! Ranking metrics: Lorenz curve and Gini. "Does the model sort risk from low to high?"
//!
//! Rows are ranked by `score` ascending. The x-axis accumulates `w` (exposure, or 1 per row),
//! the y-axis accumulates `y` (actual). A model that ranks well pushes the curve below the
//! diagonal; Gini is twice the area between the curve and the diagonal, so 0 = random order,
//! and the maximum is reached by ranking on the actual outcome itself.
//!
//! Ties in `score` are grouped into one straight segment, so the answer never depends on the
//! order rows happen to arrive in.

use crate::error::{all_values, same_length, GlassError};

/// Gini index of the Lorenz curve obtained by ranking on `score`.
///
/// Positive means higher scores go with higher `y`. For a 0/1 `y` with unit weights the
/// *normalised* Gini equals `2 * AUC - 1` (Somers' D / accuracy ratio); the raw one is that
/// times `1 - prevalence`, because the perfect Lorenz curve is a triangle of that height.
///
/// # Errors
/// Lengths differ; non-finite values; negative `y` or `w`; `sum(y)` or `sum(w)` not positive.
pub fn gini(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    validate(y, score, w)?;
    Ok(gini_unchecked(y, score, w))
}

/// Gini of `score` divided by the Gini of the perfect ranking (by `y / w`). 1 is perfect.
///
/// Use this to compare across datasets or sample sizes: the raw Gini's ceiling depends on how
/// concentrated `y` is, this one is scale-free.
///
/// # Errors
/// As [`gini`], plus a constant `y / w` (nothing to rank).
pub fn normalized_gini(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<f64, GlassError> {
    validate(y, score, w)?;
    let perfect_key: Vec<f64> = match w {
        None => y.to_vec(),
        Some(w) => y.iter().zip(w).map(|(&yi, &wi)| yi / wi).collect(),
    };
    let ceiling = gini_unchecked(y, &perfect_key, w);
    if ceiling <= 0.0 {
        return Err(GlassError::InvalidValues {
            name: "y",
            count: y.len(),
            rule: "must vary (per unit of weight) for a normalised Gini to mean anything",
            fix: "every row has the same rate, so no ranking can beat random",
        });
    }
    Ok(gini_unchecked(y, score, w) / ceiling)
}

/// Sort ascending by `score`, walk the curve, integrate with trapezoids, tie groups as one step.
// A tie is exact equality of scores by definition, so the strict float comparison is the point.
#[allow(clippy::float_cmp)]
fn gini_unchecked(y: &[f64], score: &[f64], w: Option<&[f64]>) -> f64 {
    let n_rows = y.len();
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
    let weight = |row: usize| w.map_or(1.0, |w| w[row]);
    let total_y: f64 = y.iter().sum();
    let total_w: f64 = (0..n_rows).map(weight).sum();

    let mut area = 0.0;
    let (mut cum_x, mut cum_y) = (0.0, 0.0);
    let mut start = 0;
    while start < n_rows {
        // one tie group [start, end)
        let mut end = start + 1;
        while end < n_rows && score[order[end]] == score[order[start]] {
            end += 1;
        }
        let (mut dx, mut dy) = (0.0, 0.0);
        for &row in &order[start..end] {
            dx += weight(row);
            dy += y[row];
        }
        let x0 = cum_x / total_w;
        let y0 = cum_y / total_y;
        cum_x += dx;
        cum_y += dy;
        let x1 = cum_x / total_w;
        let y1 = cum_y / total_y;
        area += (x1 - x0) * (y0 + y1) / 2.0;
        start = end;
    }
    1.0 - 2.0 * area
}

fn validate(y: &[f64], score: &[f64], w: Option<&[f64]>) -> Result<(), GlassError> {
    if y.is_empty() {
        return Err(GlassError::Empty { name: "y" });
    }
    same_length("y", y, "score", score)?;
    all_values(
        "y",
        y,
        "must be finite and >= 0",
        "Gini ranks a non-negative outcome (claims, losses, labels); negatives have no Lorenz curve",
        |v| v.is_finite() && v >= 0.0,
    )?;
    all_values(
        "score",
        score,
        "must be finite",
        "NaN or inf cannot be ranked",
        f64::is_finite,
    )?;
    if y.iter().sum::<f64>() <= 0.0 {
        return Err(GlassError::InvalidValues {
            name: "y",
            count: y.len(),
            rule: "must sum to more than zero",
            fix: "there are no positive outcomes to rank",
        });
    }
    if let Some(w) = w {
        same_length("y", y, "sample_weight", w)?;
        all_values(
            "sample_weight",
            w,
            "must be finite and > 0",
            "exposure of zero has no rate; drop those rows or use the unweighted Gini",
            |v| v.is_finite() && v > 0.0,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking_two_rows() {
        // half the exposure carries all the loss: Lorenz is the triangle, Gini = 0.5
        let g = gini(&[0.0, 1.0], &[0.1, 0.9], None).unwrap();
        assert!((g - 0.5).abs() < 1e-15, "{g}");
        assert!((normalized_gini(&[0.0, 1.0], &[0.1, 0.9], None).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn reversed_ranking_is_negative() {
        let g = gini(&[0.0, 1.0], &[0.9, 0.1], None).unwrap();
        assert!((g + 0.5).abs() < 1e-15, "{g}");
    }

    #[test]
    fn all_tied_is_zero() {
        let g = gini(&[1.0, 2.0, 3.0], &[0.5, 0.5, 0.5], None).unwrap();
        assert!(g.abs() < 1e-15, "{g}");
    }

    #[test]
    fn ties_ignore_row_order() {
        let y = [1.0, 0.0, 3.0, 0.0];
        let s = [0.2, 0.2, 0.9, 0.2];
        let a = gini(&y, &s, None).unwrap();
        let b = gini(&[0.0, 0.0, 1.0, 3.0], &[0.2, 0.2, 0.2, 0.9], None).unwrap();
        assert!((a - b).abs() < 1e-15);
    }

    #[test]
    fn refuses_zero_exposure_and_no_losses() {
        assert!(gini(&[0.0, 1.0], &[0.1, 0.9], Some(&[1.0, 0.0])).is_err());
        assert!(gini(&[0.0, 0.0], &[0.1, 0.9], None).is_err());
        assert!(normalized_gini(&[1.0, 1.0], &[0.1, 0.9], None).is_err());
    }
}
