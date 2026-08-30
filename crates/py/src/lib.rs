// PyO3 extracts arguments by value; clippy::pedantic cannot see that, so this one lint is off here.
#![allow(clippy::needless_pass_by_value)]

//! `glasshouse._core` — the Python-facing surface of the Rust core.
//!
//! Rule: bindings own nothing. Each function converts arguments (zero-copy `NumPy` views),
//! calls `glasshouse_core`, and maps `GlassError` to `ValueError`. No `if` about numbers here.

use glasshouse_core::{metrics, Family, GlassError};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deviance, m)?)?;
    m.add_function(wrap_pyfunction!(d2, m)?)?;
    Ok(())
}
