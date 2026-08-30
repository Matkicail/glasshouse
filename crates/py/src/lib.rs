// PyO3 extracts arguments by value; clippy::pedantic cannot see that, so this one lint is off here.
#![allow(clippy::needless_pass_by_value)]

//! `glasshouse._core` — the Python-facing surface of the Rust core.
//!
//! Rule: bindings own nothing. Each function converts arguments (zero-copy `NumPy` views),
//! calls `glasshouse_core`, and maps `GlassError` to `ValueError`. No `if` about numbers here.

use glasshouse_core::classification::Confusion;
use glasshouse_core::glm::{self, Data, Settings, Stop};
use glasshouse_core::regression::{self, RegressionMetric};
use glasshouse_core::Link;
use glasshouse_core::{calibration, classification, metrics, ranking, Family, GlassError};
use numpy::PyReadonlyArray2;
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py(err: GlassError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

type Arr<'a> = PyReadonlyArray1<'a, f64>;

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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    metrics::d2(fam, y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// Gini of the Lorenz curve ranked by `score`. See `glasshouse.metrics.gini`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn gini(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    ranking::gini(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// Gini divided by the perfect-ranking Gini. See `glasshouse.metrics.normalized_gini`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn normalized_gini(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    classification::roc_auc(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// See `glasshouse.metrics.average_precision`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn average_precision(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    classification::average_precision(y.as_slice()?, score.as_slice()?, w).map_err(to_py)
}

/// See `glasshouse.metrics.ks`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn ks(y: Arr<'_>, score: Arr<'_>, sample_weight: Option<Arr<'_>>) -> PyResult<f64> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    regression::regression(which, y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

/// Fit a GLM by IRLS. See `glasshouse.glm.GLM`. Returns a dict of results.
#[pyfunction]
#[pyo3(signature = (family, link, x, y, sample_weight=None, offset=None, power=None, max_iter=100, tol=1e-10))]
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
) -> PyResult<Bound<'py, PyDict>> {
    let fam = Family::parse(family, power).map_err(to_py)?;
    let link_fn = Link::parse(link).map_err(to_py)?;
    let shape = x.shape().to_vec();
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    let o = offset.as_ref().map(|a| a.as_slice()).transpose()?;
    let data = Data {
        x: x.as_slice()?,
        n_rows: shape[0],
        n_features: shape[1],
        y: y.as_slice()?,
        weights: w,
        offset: o,
    };
    let settings = Settings {
        max_iter,
        tol,
        ..Settings::default()
    };
    let fit = glm::fit(fam, link_fn, data, settings).map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("coef", fit.coef)?;
    out.set_item("mu", fit.mu)?;
    out.set_item("deviance", fit.deviance)?;
    out.set_item("null_deviance", fit.null_deviance)?;
    out.set_item("dispersion", fit.dispersion)?;
    out.set_item("cov", fit.cov)?;
    out.set_item("cov_robust", fit.cov_robust)?;
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

/// Lorenz curve points. See `glasshouse.curves.lorenz`.
#[pyfunction]
#[pyo3(signature = (y, score, sample_weight=None))]
fn lorenz_curve(
    y: Arr<'_>,
    score: Arr<'_>,
    sample_weight: Option<Arr<'_>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lorenz_curve, m)?)?;
    m.add_function(wrap_pyfunction!(double_lift_table, m)?)?;
    m.add_function(wrap_pyfunction!(glm_fit, m)?)?;
    m.add_function(wrap_pyfunction!(regression_metric, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(ks, m)?)?;
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
