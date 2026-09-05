"""Type stubs for the Rust extension. Keep in step with crates/py/src/lib.rs."""

from typing import Any

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
def glm_fit(
    family: str,
    link: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    offset: npt.NDArray[np.float64] | None = None,
    power: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    penalty: npt.NDArray[np.float64] | None = None,
    warm_start: npt.NDArray[np.float64] | None = None,
    inference: bool = True,
) -> dict[str, Any]: ...
def lorenz_curve(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> tuple[list[float], list[float]]: ...
def double_lift_table(
    y: npt.NDArray[np.float64],
    mu_a: npt.NDArray[np.float64],
    mu_b: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]: ...
def binned_table(
    key: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]: ...
def residuals(
    kind: str,
    family: str,
    y: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
    power: float | None = None,
) -> list[float]: ...
def roc_curve(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> tuple[list[float], list[float], list[float]]: ...
def pr_curve(
    y: npt.NDArray[np.float64],
    score: npt.NDArray[np.float64],
    sample_weight: npt.NDArray[np.float64] | None = None,
) -> tuple[list[float], list[float], list[float]]: ...
def bspline_design(
    x: npt.NDArray[np.float64],
    knots: list[float],
    degree: int,
) -> tuple[list[float], int]: ...
