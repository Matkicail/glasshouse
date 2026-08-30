"""Metrics that tell the truth. Every one takes ``sample_weight``.

The numbers are computed in Rust (``glasshouse._core``); this module only converts inputs to
contiguous float64 arrays and documents what each metric is for. Weight semantics — sample
weights vs frequency weights — are explained once in ``docs/methods.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse import _core

ArrayLike = Any  # anything ``np.asarray`` accepts: list, NumPy, pandas/Polars Series, pyarrow


def _f64(x: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``x`` as a contiguous 1-D float64 array, or explain why it cannot be one."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1:
        msg = f"{name} must be 1-D, got shape {arr.shape}; pass one column at a time"
        raise ValueError(msg)
    return arr


def poisson_deviance(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean Poisson deviance: ``mean_w(2 * (y * ln(y / mu) - (y - mu)))``.

    What it is for: "did the model fit the counts it said it was modelling". It is the same
    function a Poisson GLM minimises, so it is the honest score for the mean. It says nothing
    about ranking (see Gini) or business value — compare it against the null model's deviance.

    Parameters
    ----------
    y : array-like of shape (n,)
        Observed counts (or rates). Must be ``>= 0``.
    mu : array-like of shape (n,)
        Predicted means, ``> 0``. Pass the mean scale, not the linear predictor.
    sample_weight : array-like of shape (n,), optional
        Non-negative weights (exposure, for a rate model). ``None`` means all ones.

    Returns
    -------
    float
        ``sum(w * d(y, mu)) / sum(w)``. Zero when ``mu == y`` everywhere.

    Raises
    ------
    ValueError
        With the row count and the fix, if lengths differ or values are outside the support.

    Examples
    --------
    >>> from glasshouse.metrics import poisson_deviance
    >>> round(poisson_deviance([0, 1, 2], [0.5, 1.0, 2.5]), 4)
    0.3691
    """
    w = None if sample_weight is None else _f64(sample_weight, "sample_weight")
    return float(_core.poisson_deviance(_f64(y, "y"), _f64(mu, "mu"), w))


__all__ = ["poisson_deviance"]
