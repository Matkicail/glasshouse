// PyO3 extracts arguments by value; clippy::pedantic cannot see that, so this one lint is off here.
#![allow(clippy::needless_pass_by_value)]

//! `glasshouse._core` — the Python-facing surface of the Rust core.
//!
//! Rule: bindings own nothing. Each function converts arguments (zero-copy `NumPy` views),
//! calls `glasshouse_core`, and maps `GlassError` to `ValueError`. No `if` about numbers here.

use glasshouse_core::classification::Confusion;
use glasshouse_core::regression::{self, RegressionMetric};
use glasshouse_core::{calibration, classification, metrics, ranking, Family, GlassError};
use numpy::PyReadonlyArray1;
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regression_metric, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(ks, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_table, m)?)?;
    m.add_function(wrap_pyfunction!(balance, m)?)?;
    m.add_function(wrap_pyfunction!(deviance, m)?)?;
    m.add_function(wrap_pyfunction!(d2, m)?)?;
    m.add_function(wrap_pyfunction!(gini, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_gini, m)?)?;
    Ok(())
}
