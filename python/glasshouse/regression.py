"""Plain regression errors, weighted: "how far off, on average?".

These are the numbers everyone already knows, so they are the easiest to explain and the
easiest to be misled by. On skewed, heavy-tailed targets a handful of rows dominate every one
of them; the family deviance in :mod:`glasshouse.metrics` is the score a GLM actually
minimises. Report both.
"""

from __future__ import annotations

from glasshouse import _core
from glasshouse.arrays import ArrayLike
from glasshouse.metrics import _f64, _weights


def _run(metric: str, y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None) -> float:
    w = _weights(sample_weight)
    return float(_core.regression_metric(metric, _f64(y, "y"), _f64(mu, "mu"), w))


def rmse(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Root mean squared error. Big misses hurt more than small ones, quadratically.

    What it is for: plain regression where a large error really is worse than several small
    ones, and you want the answer in the units of ``y``.

    When it lies: a single outlier can own it; on heavy-tailed targets it tracks the tail,
    not the typical row. See :func:`mae`.

    Examples
    --------
    >>> from glasshouse.regression import rmse
    >>> round(rmse([1, 2, 4], [1, 3, 2]), 4)
    1.291
    """
    return _run("rmse", y, mu, sample_weight)


def mae(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean absolute error. Every unit of miss costs the same.

    What it is for: when errors cost linearly, and as the robust partner to :func:`rmse` —
    if RMSE is much larger than MAE, a few rows are doing the damage.

    When it lies: it is minimised by the median, not the mean, so a model that is unbiased on
    total can look worse than one that is not.

    Examples
    --------
    >>> from glasshouse.regression import mae
    >>> mae([1, 2, 4], [1, 3, 2])
    1.0
    """
    return _run("mae", y, mu, sample_weight)


def mape(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean absolute percentage error ``|y - mu| / |y|``. 0.1 means "10 % off on average".

    What it is for: forecasting, where a percentage is what the business asks for.

    When it lies: it blows up at ``y == 0`` (refused here, on purpose), punishes over-forecasts
    more than under-forecasts of the same size, and is dominated by small ``y``. Prefer
    :func:`smape` or :func:`mae` unless someone insists.

    Examples
    --------
    >>> from glasshouse.regression import mape
    >>> round(mape([1, 2, 4], [1, 3, 2]), 4)
    0.3333
    """
    return _run("mape", y, mu, sample_weight)


def smape(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Symmetric MAPE ``2 |y - mu| / (|y| + |mu|)``, in ``[0, 2]``.

    What it is for: a percentage error that treats over- and under-forecasts alike and
    survives zeros in ``y`` (as long as ``mu`` is not also zero there).

    When it lies: it is bounded at 2, so a forecast of 0 against any positive actual scores
    the maximum regardless of size; and "symmetric" is generous — it still favours
    over-forecasting slightly.

    Examples
    --------
    >>> from glasshouse.regression import smape
    >>> round(smape([1, 2, 4], [1, 3, 2]), 4)
    0.3556
    """
    return _run("smape", y, mu, sample_weight)


def msle(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean squared log error ``(ln(1 + y) - ln(1 + mu))²``: relative errors on a log scale.

    What it is for: non-negative targets that span orders of magnitude, where being 2x off
    should cost the same at 10 as at 10 000.

    When it lies: it penalises under-prediction more than over-prediction, and the ``1 +``
    makes it scale-dependent for small values.

    Examples
    --------
    >>> from glasshouse.regression import msle
    >>> round(msle([1, 2, 4], [1, 3, 2]), 4)
    0.1146
    """
    return _run("msle", y, mu, sample_weight)


def r2(y: ArrayLike, mu: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Coefficient of determination ``1 - SSE / SST``. 1 is perfect, 0 is "predict the mean".

    What it is for: the squared-error cousin of :func:`glasshouse.metrics.d2` — the "vs naive"
    number for plain least squares. Negative means worse than the weighted mean of ``y``.

    When it lies: like every squared-error score, on skewed targets; and on training data it
    only ever goes up as you add features. Report it held-out.

    Examples
    --------
    >>> from glasshouse.regression import r2
    >>> round(r2([1, 2, 4], [1, 3, 2]), 4)
    -0.0714
    """
    return _run("r2", y, mu, sample_weight)


__all__ = ["mae", "mape", "msle", "r2", "rmse", "smape"]
