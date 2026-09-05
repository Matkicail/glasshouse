// PyO3 extracts arguments by value; clippy::pedantic cannot see that, so this one lint is off here.
#![allow(clippy::needless_pass_by_value)]

//! `glasshouse._core` — the Python-facing surface of the Rust core.
//!
//! Rule: bindings own nothing. Each function converts arguments (zero-copy `NumPy` views),
//! calls `glasshouse_core`, and maps `GlassError` to `ValueError`. No `if` about numbers here.

use glasshouse_core::classification::Confusion;
use glasshouse_core::constraints::Chain;
use glasshouse_core::elastic::ElasticNet;
use glasshouse_core::glm::{self, Data, Penalty, Settings, Stop};
use glasshouse_core::regression::{self, RegressionMetric};
use glasshouse_core::Link;
use glasshouse_core::{
    calibration, classification, metrics, ranking, splines, tournament, Family, GlassError,
};
use numpy::PyReadonlyArray2;
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py(err: GlassError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

type Arr<'a> = PyReadonlyArray1<'a, f64>;

/// The optional-weights dance, once: borrow the slice out of an optional array.
fn opt_slice<'py>(arr: Option<&'py Arr<'py>>) -> PyResult<Option<&'py [f64]>> {
    arr.map(numpy::PyReadonlyArray1::as_slice)
        .transpose()
        .map_err(Into::into)
}

/// Weighted mean deviance under `family`. See `glasshouse.metrics.deviance`.
#[pyfunction]
#[pyo3(signature = (family, y, mu, sample_weight=None, power=None))]
fn deviance(
    family: &str,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    power: Option<f64>,
) -> PyResult<f64> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let w = opt_slice(sample_weight.as_ref())?;
    metrics::deviance(fam, y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// D² (deviance explained) under `family`. See `glasshouse.metrics.d2`.
#[pyfunction]
#[pyo3(signature = (family, y, mu, sample_weight=None, power=None))]
fn d2(
    family: &str,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    power: Option<f64>,
) -> PyResult<f64> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let w = opt_slice(sample_weight.as_ref())?;
    metrics::d2(fam, y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// Gini of the Lorenz curve ranked by `score`. See `glasshouse.metrics.gini`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn gini(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    ranking::gini(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// Gini divided by the perfect-ranking Gini. See `glasshouse.metrics.normalized_gini`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn normalized_gini(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    ranking::normalized_gini(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// Calibration table as a dict of columns. See `glasshouse.metrics.calibration_table`.
#[pyfunction]
#[pyo3(signature = (y, mu, sample_weight=None, n_bins=10))]
fn calibration_table<'py>(
    py: Python<'py>,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let w = opt_slice(sample_weight.as_ref())?;
    let bins =
        calibration::calibration_table(y.as_slice()?, mu.as_slice()?, w, n_bins).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("n_rows", bins.iter().map(|b| b.n_rows).collect::<Vec<_>>())?;
    out.set_item("weight", bins.iter().map(|b| b.weight).collect::<Vec<_>>())?;
    out.set_item(
        "predicted",
        bins.iter().map(|b| b.predicted).collect::<Vec<_>>(),
    )?;
    out.set_item("actual", bins.iter().map(|b| b.actual).collect::<Vec<_>>())?;
    out.set_item(
        "actual_over_expected",
        bins.iter()
            .map(|b| b.actual_over_expected)
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// Calibration-style table binned by any column. See `glasshouse.residuals.ae_by_feature`.
#[pyfunction]
#[pyo3(signature = (key, y, mu, sample_weight=None, n_bins=10))]
fn binned_table<'py>(
    py: Python<'py>,
    key: Arr<'_>,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let w = opt_slice(sample_weight.as_ref())?;
    let bins = calibration::binned_table(key.as_slice()?, y.as_slice()?, mu.as_slice()?, w, n_bins)
        .map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("n_rows", bins.iter().map(|b| b.n_rows).collect::<Vec<_>>())?;
    out.set_item("weight", bins.iter().map(|b| b.weight).collect::<Vec<_>>())?;
    out.set_item(
        "predicted",
        bins.iter().map(|b| b.predicted).collect::<Vec<_>>(),
    )?;
    out.set_item("actual", bins.iter().map(|b| b.actual).collect::<Vec<_>>())?;
    out.set_item(
        "actual_over_expected",
        bins.iter()
            .map(|b| b.actual_over_expected)
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// Deviance or Pearson residuals. See `glasshouse.residuals`.
#[pyfunction]
#[pyo3(signature = (kind, family, y, mu, sample_weight=None, power=None))]
fn residuals(
    kind: &str,
    family: &str,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    power: Option<f64>,
) -> PyResult<Vec<f64>> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let w = opt_slice(sample_weight.as_ref())?;
    match kind {
        "deviance" => metrics::deviance_residuals(fam, y.as_slice()?, mu.as_slice()?, w),
        "pearson" => metrics::pearson_residuals(fam, y.as_slice()?, mu.as_slice()?, w),
        other => {
            return Err(PyValueError::new_err(format!(
                "kind: unknown residual {other:?} — one of: deviance, pearson"
            )))
        }
    }
    .map_err(to_py)
}

/// Overall actual / expected. See `glasshouse.metrics.balance`.
#[pyfunction]
#[pyo3(signature = (y, mu, sample_weight=None))]
fn balance(y: Arr<'_>, mu: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    calibration::balance(y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// Confusion counts and every threshold metric at once. See `glasshouse.metrics.threshold_metrics`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None, threshold=0.5))]
fn threshold_metrics<'py>(
    py: Python<'py>,
    y: Arr<'_>,
    score: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let w = opt_slice(sample_weight.as_ref())?;
    let c = Confusion::at(y.as_slice()?, score.as_slice()?, w, threshold).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("tp", c.tp)?;
    out.set_item("fp", c.fp)?;
    out.set_item("fn", c.fn_)?;
    out.set_item("tn", c.tn)?;
    out.set_item("accuracy", c.accuracy())?;
    out.set_item("balanced_accuracy", c.balanced_accuracy())?;
    out.set_item("precision", c.precision())?;
    out.set_item("recall", c.recall())?;
    out.set_item("f1", c.f1())?;
    out.set_item("mcc", c.mcc())?;
    Ok(out)
}

/// See `glasshouse.metrics.roc_auc`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn roc_auc(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    classification::roc_auc(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// See `glasshouse.metrics.average_precision`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn average_precision(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    classification::average_precision(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// ROC curve points. See `glasshouse.curves.roc`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn roc_curve(
    y: Arr<'_>,
    score: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let w = opt_slice(sample_weight.as_ref())?;
    classification::roc_curve(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// Precision-recall curve points. See `glasshouse.curves.pr`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn pr_curve(
    y: Arr<'_>,
    score: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let w = opt_slice(sample_weight.as_ref())?;
    classification::pr_curve(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// See `glasshouse.metrics.ks`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn ks(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = opt_slice(sample_weight.as_ref())?;
    classification::ks(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// One weighted regression error by name. See `glasshouse.regression`.
#[pyfunction]
#[pyo3(signature = (metric, y, mu, sample_weight=None))]
fn regression_metric(
    metric: &str,
    y: Arr<'_>,
    mu: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<f64> {
    let which = match metric {
        "rmse" => RegressionMetric::Rmse,
        "mae" => RegressionMetric::Mae,
        "mape" => RegressionMetric::Mape,
        "smape" => RegressionMetric::Smape,
        "msle" => RegressionMetric::Msle,
        "r2" => RegressionMetric::R2,
        other => {
            return Err(PyValueError::new_err(format!(
                "metric: unknown regression metric {other:?} — one of: rmse, mae, mape, smape, msle, r2"
            )))
        }
    };
    let w = opt_slice(sample_weight.as_ref())?;
    regression::regression(which, y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// Fit a GLM by IRLS. See `glasshouse.glm.GLM`. Returns a dict of results.
/// `penalty` is a row-major symmetric `p x p` matrix, already scaled by the smoothing
/// parameter — the penalised deviance `D + beta' S beta` is what gets minimised.
#[pyfunction]
#[pyo3(signature = (family, link, x, y, sample_weight=None, offset=None, power=None, max_iter=100, tol=1e-10, penalty=None, warm_start=None, inference=true, monotone=None, elastic_net=None))]
#[allow(clippy::too_many_arguments)]
fn glm_fit<'py>(
    py: Python<'py>,
    family: &str,
    link: &str,
    x: PyReadonlyArray2<'_, f64>,
    y: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    offset: Option<Arr<'_>>,
    power: Option<f64>,
    max_iter: usize,
    tol: f64,
    penalty: Option<PyReadonlyArray2<'_, f64>>,
    warm_start: Option<Arr<'_>>,
    inference: bool,
    monotone: Option<Vec<(Vec<usize>, bool, bool)>>,
    elastic_net: Option<(f64, f64, Vec<bool>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let link_fn = Link::parse(link).map_err(to_py)?;
    let shape = x.shape().to_vec();
    let w = opt_slice(sample_weight.as_ref())?;
    let o = opt_slice(offset.as_ref())?;
    let pen = match (penalty.as_ref(), elastic_net.as_ref()) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "pass either a penalty matrix or an elastic_net, not both",
            ))
        }
        (Some(arr), None) => Penalty::Quadratic(arr.as_slice()?),
        (None, Some((alpha, l1_ratio, penalised))) => Penalty::ElasticNet(ElasticNet {
            alpha: *alpha,
            l1_ratio: *l1_ratio,
            penalised,
        }),
        (None, None) => Penalty::None,
    };
    let data = Data {
        x: x.as_slice()?,
        n_rows: shape[0],
        n_features: shape[1],
        y: y.as_slice()?,
        weights: w,
        offset: o,
    };
    let start = opt_slice(warm_start.as_ref())?;
    let chains: Vec<Chain> = monotone
        .unwrap_or_default()
        .into_iter()
        .map(|(columns, increasing, anchored)| Chain {
            columns,
            increasing,
            anchored,
        })
        .collect();
    let settings = Settings {
        max_iter,
        tol,
        inference,
        ..Settings::default()
    };
    let fit = glm::fit(fam, link_fn, data, pen, start, &chains, settings).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("coef", fit.coef)?;
    out.set_item("mu", fit.mu)?;
    out.set_item("deviance", fit.deviance)?;
    out.set_item("edf", fit.edf)?;
    if let Some(inf) = fit.inference {
        out.set_item("null_deviance", inf.null_deviance)?;
        out.set_item("dispersion", inf.dispersion)?;
        out.set_item("cov", inf.cov)?;
        out.set_item("cov_robust", inf.cov_robust)?;
    }
    out.set_item("n_rows", fit.n_rows)?;
    out.set_item("n_features", fit.n_features)?;
    out.set_item("iterations", fit.iterations)?;
    out.set_item(
        "stop",
        match fit.stop {
            Stop::Converged => "converged",
            Stop::MaxIter => "max_iter",
            Stop::NoImprovement => "no_improvement",
        },
    )?;
    out.set_item(
        "trace_iteration",
        fit.trace.iter().map(|t| t.iteration).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "trace_deviance",
        fit.trace.iter().map(|t| t.deviance).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "trace_halvings",
        fit.trace.iter().map(|t| t.halvings).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "trace_max_step",
        fit.trace.iter().map(|t| t.max_step).collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// The elastic-net alpha at which every penalised coefficient is zero. See `glasshouse.glm`.
#[pyfunction]
#[pyo3(signature = (family, link, x, y, sample_weight=None, offset=None, power=None, l1_ratio=0.5, penalised=None))]
#[allow(clippy::too_many_arguments)]
fn glm_alpha_max(
    family: &str,
    link: &str,
    x: PyReadonlyArray2<'_, f64>,
    y: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    offset: Option<Arr<'_>>,
    power: Option<f64>,
    l1_ratio: f64,
    penalised: Option<Vec<bool>>,
) -> PyResult<f64> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let link_fn = Link::parse(link).map_err(to_py)?;
    let shape = x.shape().to_vec();
    let w = opt_slice(sample_weight.as_ref())?;
    let o = opt_slice(offset.as_ref())?;
    let data = Data {
        x: x.as_slice()?,
        n_rows: shape[0],
        n_features: shape[1],
        y: y.as_slice()?,
        weights: w,
        offset: o,
    };
    let mask = penalised.unwrap_or_else(|| vec![true; shape[1]]);
    glm::alpha_max(fam, link_fn, data, l1_ratio, &mask, Settings::default()).map_err(to_py)
}

/// Every row to the cheapest model, ties split. See `glasshouse.tournament`.
#[pyfunction]
#[pyo3(signature = (y, predictions, sample_weight=None))]
fn win_sets<'py>(
    py: Python<'py>,
    y: Arr<'_>,
    predictions: Vec<Arr<'_>>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = opt_slice(sample_weight.as_ref())?;
    let slices: Vec<&[f64]> = predictions
        .iter()
        .map(|p| p.as_slice())
        .collect::<Result<_, _>>()?;
    let sets = tournament::win_sets(y.as_slice()?, &slices, w).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("weight", sets.iter().map(|s| s.weight).collect::<Vec<_>>())?;
    out.set_item(
        "predicted",
        sets.iter().map(|s| s.predicted).collect::<Vec<_>>(),
    )?;
    out.set_item("actual", sets.iter().map(|s| s.actual).collect::<Vec<_>>())?;
    Ok(out)
}

/// Lorenz curve points. See `glasshouse.curves.lorenz`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn lorenz_curve(
    y: Arr<'_>,
    score: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let w = opt_slice(sample_weight.as_ref())?;
    ranking::lorenz_curve(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// Double-lift table as a dict of columns. See `glasshouse.curves.double_lift`.
#[pyfunction]
#[pyo3(signature = (y, mu_a, mu_b, sample_weight=None, n_bins=10))]
fn double_lift_table<'py>(
    py: Python<'py>,
    y: Arr<'_>,
    mu_a: Arr<'_>,
    mu_b: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
    n_bins: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let w = opt_slice(sample_weight.as_ref())?;
    let bins = calibration::double_lift_table(
        y.as_slice()?,
        mu_a.as_slice()?,
        mu_b.as_slice()?,
        w,
        n_bins,
    )
    .map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("n_rows", bins.iter().map(|b| b.n_rows).collect::<Vec<_>>())?;
    out.set_item("weight", bins.iter().map(|b| b.weight).collect::<Vec<_>>())?;
    out.set_item("ratio", bins.iter().map(|b| b.ratio).collect::<Vec<_>>())?;
    out.set_item("actual", bins.iter().map(|b| b.actual).collect::<Vec<_>>())?;
    out.set_item(
        "predicted_a",
        bins.iter().map(|b| b.predicted_a).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "predicted_b",
        bins.iter().map(|b| b.predicted_b).collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// B-spline design matrix, row-major. See `glasshouse.encoders.BSpline`.
#[pyfunction]
#[pyo3(signature = (x, knots, degree))]
fn bspline_design(x: Arr<'_>, knots: Vec<f64>, degree: usize) -> PyResult<(Vec<f64>, usize)> {
    let design = splines::bspline_design(x.as_slice()?, &knots, degree).map_err(to_py)?;
    Ok((design, splines::n_basis(&knots, degree)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bspline_design, m)?)?;
    m.add_function(wrap_pyfunction!(lorenz_curve, m)?)?;
    m.add_function(wrap_pyfunction!(win_sets, m)?)?;
    m.add_function(wrap_pyfunction!(double_lift_table, m)?)?;
    m.add_function(wrap_pyfunction!(glm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(glm_alpha_max, m)?)?;
    m.add_function(wrap_pyfunction!(regression_metric, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(ks, m)?)?;
    m.add_function(wrap_pyfunction!(roc_curve, m)?)?;
    m.add_function(wrap_pyfunction!(pr_curve, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_table, m)?)?;
    m.add_function(wrap_pyfunction!(binned_table, m)?)?;
    m.add_function(wrap_pyfunction!(residuals, m)?)?;
    m.add_function(wrap_pyfunction!(balance, m)?)?;
    m.add_function(wrap_pyfunction!(deviance, m)?)?;
    m.add_function(wrap_pyfunction!(d2, m)?)?;
    m.add_function(wrap_pyfunction!(gini, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_gini, m)?)?;
    Ok(())
}
