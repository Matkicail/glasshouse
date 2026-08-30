"""Metrics that tell the truth. Every one takes ``sample_weight``.

The numbers are computed in Rust (``glasshouse._core``); this module only converts inputs to
contiguous float64 arrays and documents what each metric is for. Weight semantics — sample
weights vs frequency weights — are explained once in ``docs/methods.md``.

Deviance in one paragraph
-------------------------
The deviance is twice the log-likelihood gap between a perfect model (one that predicts each
``y`` exactly) and yours, under the distribution you say the data follows. It is what a GLM of
that family minimises, so it is the honest, consistent score for the mean. Lower is better,
zero is perfect. Read it next to the null model's deviance: ``d2`` does that division for you.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from glasshouse import _core

ArrayLike = Any  # anything ``np.asarray`` accepts: list, NumPy, pandas/Polars Series, pyarrow
FamilyName = Literal["gaussian", "poisson", "gamma", "tweedie", "binomial"]


def _f64(x: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``x`` as a contiguous 1-D float64 array, or explain why it cannot be one."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1:
        msg = f"{name} must be 1-D, got shape {arr.shape}; pass one column at a time"
        raise ValueError(msg)
    return arr


def _weights(sample_weight: ArrayLike | None) -> npt.NDArray[np.float64] | None:
    return None if sample_weight is None else _f64(sample_weight, "sample_weight")


def deviance(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    family: FamilyName,
    sample_weight: ArrayLike | None = None,
    power: float | None = None,
) -> float:
    """Weighted mean deviance of predictions ``mu`` against observations ``y``.

    What it is for: "did the model fit the distribution I said the data has". It is the same
    function the matching GLM minimises. It says nothing about ranking (see Gini) or about
    business value — two models with equal deviance can price the tails very differently.

    Parameters
    ----------
    y : array-like of shape (n,)
        Observed values, inside the family's support (poisson ``>= 0``, gamma ``> 0``,
        binomial in ``[0, 1]``, gaussian anything).
    mu : array-like of shape (n,)
        Predicted means. Mean scale, not the linear predictor; probabilities for binomial.
    family : {"gaussian", "poisson", "gamma", "tweedie", "binomial"}
        Which unit deviance to use. Pick the family you trained with.
    sample_weight : array-like of shape (n,), optional
        Non-negative weights (exposure for a rate model, claim count for severity).
        ``None`` means all ones.
    power : float, optional
        Tweedie variance power. Required for ``family="tweedie"``; ignored otherwise.
        ``1 < power < 2`` is the compound Poisson-gamma used for pure premium; ``0``, ``1``,
        ``2`` reproduce gaussian, poisson, gamma exactly.

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
    >>> from glasshouse.metrics import deviance
    >>> round(deviance([0, 1, 2], [0.5, 1.0, 2.5], family="poisson"), 4)
    0.3691
    >>> round(deviance([10, 20], [12, 18], family="tweedie", power=1.5), 4)
    0.0774
    """
    w = _weights(sample_weight)
    return float(_core.deviance(family, _f64(y, "y"), _f64(mu, "mu"), w, power))


def d2(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    family: FamilyName,
    sample_weight: ArrayLike | None = None,
    power: float | None = None,
) -> float:
    """D², "deviance explained": ``1 - deviance(y, mu) / deviance(y, mean(y))``.

    What it is for: the family-consistent pseudo-R². 1 is perfect, 0 means "no better than
    predicting the (weighted) mean of ``y``", negative means worse than that. The null model
    here is the intercept-only model, so this *is* the "vs naive" comparison for a GLM.

    When it lies: like R², it rewards fitting the training data; report it on held-out data.
    It is undefined when ``y`` is constant (nothing to explain) and the function says so.

    Parameters are those of :func:`deviance`.

    Examples
    --------
    >>> from glasshouse.metrics import d2
    >>> y = [1, 2, 3, 6]
    >>> d2(y, y, family="poisson")
    1.0
    >>> round(d2(y, [3, 3, 3, 3], family="poisson"), 12)
    0.0
    """
    w = _weights(sample_weight)
    return float(_core.d2(family, _f64(y, "y"), _f64(mu, "mu"), w, power))


def gaussian_deviance(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean squared error, as a deviance. See :func:`deviance`."""
    return deviance(y, mu, family="gaussian", sample_weight=sample_weight)


def poisson_deviance(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean Poisson deviance (counts, rates with exposure). See :func:`deviance`."""
    return deviance(y, mu, family="poisson", sample_weight=sample_weight)


def gamma_deviance(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean gamma deviance (positive amounts, severity). See :func:`deviance`."""
    return deviance(y, mu, family="gamma", sample_weight=sample_weight)


def tweedie_deviance(
    y: ArrayLike, mu: ArrayLike, power: float, sample_weight: ArrayLike | None = None
) -> float:
    """Weighted mean Tweedie deviance at variance ``power`` (pure premium). See :func:`deviance`."""
    return deviance(y, mu, family="tweedie", sample_weight=sample_weight, power=power)


def binomial_deviance(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean binomial deviance — twice the log-loss for 0/1 labels. See :func:`deviance`."""
    return deviance(y, mu, family="binomial", sample_weight=sample_weight)


__all__ = [
    "binomial_deviance",
    "d2",
    "deviance",
    "gamma_deviance",
    "gaussian_deviance",
    "poisson_deviance",
    "tweedie_deviance",
]
