// PyO3 extracts arguments by value; clippy::pedantic cannot see that, so this one lint is off here.
#![allow(clippy::needless_pass_by_value)]

//! `glasshouse._core` — the Python-facing surface of the Rust core.
//!
//! Rule: bindings own nothing. Each function converts arguments (zero-copy `NumPy` views),
//! calls `glasshouse_core`, and maps `GlassError` to `ValueError`. No `if` about numbers here.

use glasshouse_core::{metrics, GlassError};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py(err: GlassError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Weighted mean Poisson deviance. See `glasshouse.metrics.poisson_deviance` for the docs.
#[pyfunction]
#[pyo3(signature = (y, mu, sample_weight=None))]
fn poisson_deviance(
    y: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>,
    sample_weight: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<f64> {
    let w = sample_weight.as_ref().map(|a| a.as_slice()).transpose()?;
    metrics::poisson_deviance(y.as_slice()?, mu.as_slice()?, w).map_err(to_py)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(poisson_deviance, m)?)?;
    Ok(())
}
