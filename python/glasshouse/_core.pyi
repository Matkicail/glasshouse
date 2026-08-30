"""Type stubs for the Rust extension. Keep in step with crates/py/src/lib.rs."""

import numpy as np
import numpy.typing as npt

def deviance(
    family: str,
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    power: float | None = None,
) -> float: ...
def d2(
    family: str,
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    power: float | None = None,
) -> float: ...
def gini(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def normalized_gini(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def calibration_table(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]: ...
def balance(
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def threshold_metrics(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]: ...
def roc_auc(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def average_precision(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def ks(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
def regression_metric(
    metric: str,
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> float: ...
