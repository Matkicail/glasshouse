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


def gini(y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Gini index: how well ``score`` sorts risk from low to high.

    What it is for: pricing, credit, any place you act on the *ordering* of predictions.
    Rows are ranked by ``score``; the Lorenz curve accumulates exposure (``sample_weight``)
    on the x-axis and actual ``y`` on the y-axis. Gini is twice the area between that curve
    and the diagonal: 0 is random order, negative is backwards, and the ceiling depends on how
    concentrated ``y`` is (see :func:`normalized_gini`). For a 0/1 ``y`` the raw Gini is
    ``(2 * AUC - 1) * (1 - prevalence)``; the *normalised* one is exactly ``2 * AUC - 1``.

    When it lies: it assumes the data should follow a Lorenz curve, and most data doesn't;
    it needs sample size; and it is blind to calibration — double every prediction and Gini
    does not move. In fraud with lots of unlabelled fraud it can look great while missing
    the real problem: read it next to MCC / PR-AUC and calibration, never alone.

    Parameters
    ----------
    y : array-like of shape (n,)
        Actual outcome per row, ``>= 0`` (claim counts, losses, 0/1 labels).
    score : array-like of shape (n,)
        The ranking key: predicted *rate* or probability. Ties are grouped, so row order
        never matters.
    sample_weight : array-like of shape (n,), optional
        Exposure per row, ``> 0``. ``None`` means one unit per row.

    Examples
    --------
    >>> from glasshouse.metrics import gini
    >>> gini([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    0.5
    >>> gini([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    -0.5
    """
    w = _weights(sample_weight)
    return float(_core.gini(_f64(y, "y"), _f64(score, "score"), w))


def normalized_gini(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    """:func:`gini` of ``score`` divided by the Gini of the perfect ranking. 1 is perfect.

    What it is for: comparing models across datasets or sample sizes. The raw Gini's ceiling
    depends on how concentrated ``y`` is (a 0.3 can be excellent on sparse claims and poor on
    a dense target); dividing by the best achievable Gini makes it scale-free, and it is your
    safety net on noise — the same Kaggle "normalized Gini" used in insurance competitions.
    For 0/1 labels it is exactly ``2 * AUC - 1`` (the accuracy ratio / Somers' D).

    When it lies: exactly as :func:`gini`; a ratio inherits all of it.

    Examples
    --------
    >>> from glasshouse.metrics import normalized_gini
    >>> normalized_gini([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    1.0
    """
    w = _weights(sample_weight)
    return float(_core.normalized_gini(_f64(y, "y"), _f64(score, "score"), w))


__all__ = [
    "binomial_deviance",
    "d2",
    "deviance",
    "gamma_deviance",
    "gaussian_deviance",
    "gini",
    "normalized_gini",
    "poisson_deviance",
    "tweedie_deviance",
]
