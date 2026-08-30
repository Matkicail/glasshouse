"""The scorecard: the panel, not the number — and always against a naive baseline.

The thing people most often don't know is whether their model is actually helping. So every
card carries a naive row, free, without asking: the weighted mean of ``y`` for a regression
or GLM family, the class prior for binomial. Every metric is reported for both, and
:func:`compare` reads two cards side by side with a direction-aware "better" column.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glasshouse import classification as clf
from glasshouse import metrics, regression
from glasshouse.arrays import ArrayLike
from glasshouse.metrics import CalibrationTable, FamilyName, _f64, _weights

# Which way is "good" for each metric name. Anything not listed is a diagnostic, not a score.
HIGHER_IS_BETTER: Mapping[str, bool] = {
    "deviance": False,
    "d2": True,
    "gini": True,
    "normalized_gini": True,
    "rmse": False,
    "mae": False,
    "r2": True,
    "mcc": True,
    "f1": True,
    "roc_auc": True,
    "average_precision": True,
    "ks": True,
    "log_loss": False,
    "brier": False,
}

# Distance from 1 is what matters for balance; treated specially in compare().
_TARGET_ONE = frozenset({"balance"})


@dataclass(frozen=True)
class Scorecard:
    """One model's panel, plus the same panel for the naive baseline.

    ``metrics`` and ``naive`` have the same keys; ``calibration`` is the model's A/E table.
    Print it, or ``to_dict()`` it for a report.
    """

    family: str
    n_rows: int
    weight_sum: float
    metrics: dict[str, float]
    naive: dict[str, float]
    calibration: CalibrationTable
    naive_prediction: float
    label: str = "model"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (JSON-ready) with the calibration table as columns."""
        return {
            "label": self.label,
            "family": self.family,
            "n_rows": self.n_rows,
            "weight_sum": self.weight_sum,
            "naive_prediction": self.naive_prediction,
            "metrics": dict(self.metrics),
            "naive": dict(self.naive),
            "calibration": {
                "n_rows": self.calibration.n_rows.tolist(),
                "weight": self.calibration.weight.tolist(),
                "predicted": self.calibration.predicted.tolist(),
                "actual": self.calibration.actual.tolist(),
                "actual_over_expected": self.calibration.actual_over_expected.tolist(),
            },
            **self.extra,
        }

    def __str__(self) -> str:
        """Fixed-width table: metric, model, naive, and whether the model beats naive."""
        head = f"{self.label} vs naive  (family={self.family}, n={self.n_rows})"
        lines = [head, f"{'metric':<20}{self.label:>14}{'naive':>14}  better?"]
        for name, value in self.metrics.items():
            verdict = _verdict(name, value, self.naive[name])
            lines.append(f"{name:<20}{value:>14.5g}{self.naive[name]:>14.5g}  {verdict}")
        return "\n".join(lines)


def _verdict(name: str, a: float, b: float) -> str:
    """Is a better than b on this metric? 'yes' / 'no' / 'tie' / '-' when it isn't a score."""
    if name in _TARGET_ONE:
        da, db = abs(a - 1.0), abs(b - 1.0)
        return "tie" if np.isclose(da, db) else ("yes" if da < db else "no")
    if name not in HIGHER_IS_BETTER:
        return "-"
    if np.isclose(a, b):
        return "tie"
    return "yes" if (a > b) == HIGHER_IS_BETTER[name] else "no"


def naive_prediction(
    y: ArrayLike, *, family: FamilyName, sample_weight: ArrayLike | None = None
) -> float:
    """Return the constant a model must beat: the weighted mean of ``y``.

    For every family this is the intercept-only model (the mean on the response scale); for
    binomial it is the class prior, which is also what "guess with the class distribution"
    scores in expectation. Forecasting's last-value naive needs a time order and lives with
    the split registry, not here.

    Examples
    --------
    >>> from glasshouse.scorecard import naive_prediction
    >>> naive_prediction([0, 1, 1, 1], family="binomial")
    0.75
    """
    _ = family  # every family shares the same naive today; the parameter keeps the door open
    yy = _f64(y, "y")
    w = _weights(sample_weight)
    return float(np.average(yy, weights=w))


def scorecard(  # noqa: PLR0913 — keyword-only knobs, all optional, one obvious entry point
    y: ArrayLike,
    pred: ArrayLike,
    *,
    family: FamilyName,
    sample_weight: ArrayLike | None = None,
    power: float | None = None,
    n_bins: int = 10,
    threshold: float = 0.5,
    label: str = "model",
) -> Scorecard:
    """Score ``pred`` on the full panel for ``family``, and the naive baseline on the same panel.

    Regression / GLM families report: deviance, d2, gini, normalized_gini, balance, rmse, mae,
    r2. Binomial reports: log_loss, brier, roc_auc, average_precision, ks, mcc, f1, balance.
    Both come with the calibration table.

    Parameters
    ----------
    y, pred : array-like of shape (n,)
        Outcome and prediction on the same scale (probabilities for binomial; rates with
        exposure as ``sample_weight``, or totals without).
    family : {"gaussian", "poisson", "gamma", "tweedie", "binomial"}
        The family you trained with; picks the deviance and the panel.
    sample_weight, power, n_bins, threshold
        Passed to the underlying metrics. ``threshold`` only matters for binomial.
    label : str
        Name shown in the printed card and in :func:`compare`.

    Examples
    --------
    >>> from glasshouse.scorecard import scorecard
    >>> card = scorecard([0, 1, 2, 3], [0.5, 1.0, 2.0, 2.5], family="poisson")
    >>> round(card.metrics["d2"], 3), round(card.naive["d2"], 3)
    (0.757, 0.0)
    """
    yy, pp = _f64(y, "y"), _f64(pred, "pred")
    w = _weights(sample_weight)
    base = naive_prediction(yy, family=family, sample_weight=w)
    naive_pred = np.full_like(yy, base)
    panel = _binomial_panel(threshold) if family == "binomial" else _family_panel(family, power)

    def run(p: Any) -> dict[str, float]:
        return {name: fn(yy, p, w) for name, fn in panel.items()}

    return Scorecard(
        family=family,
        n_rows=int(yy.shape[0]),
        weight_sum=float(yy.shape[0] if w is None else w.sum()),
        metrics=run(pp),
        naive=run(naive_pred),
        calibration=metrics.calibration_table(yy, pp, sample_weight=w, n_bins=n_bins),
        naive_prediction=base,
        label=label,
    )


Metric = Callable[[Any, Any, Any], float]


def _family_panel(family: FamilyName, power: float | None) -> dict[str, Metric]:
    def dev(y: Any, p: Any, w: Any) -> float:
        return metrics.deviance(y, p, family=family, sample_weight=w, power=power)

    def d2(y: Any, p: Any, w: Any) -> float:
        return metrics.d2(y, p, family=family, sample_weight=w, power=power)

    return {
        "deviance": dev,
        "d2": d2,
        "gini": _safe_gini(metrics.gini),
        "normalized_gini": _safe_gini(metrics.normalized_gini),
        "balance": metrics.balance,
        "rmse": regression.rmse,
        "mae": regression.mae,
        "r2": regression.r2,
    }


def _binomial_panel(threshold: float) -> dict[str, Metric]:
    def mcc(y: Any, p: Any, w: Any) -> float:
        return clf.mcc(y, p, w, threshold)

    def f1(y: Any, p: Any, w: Any) -> float:
        return clf.f1(y, p, w, threshold)

    return {
        "log_loss": clf.log_loss,
        "brier": clf.brier,
        "roc_auc": clf.roc_auc,
        "average_precision": clf.average_precision,
        "ks": clf.ks,
        "mcc": mcc,
        "f1": f1,
        "balance": metrics.balance,
    }


def _safe_gini(fn: Metric) -> Metric:
    """Gini needs a non-negative outcome; for a gaussian target it is not defined — say NaN."""

    def wrapped(y: Any, p: Any, w: Any) -> float:
        yy = np.asarray(y, dtype=np.float64)
        if yy.min() < 0.0:
            return float("nan")
        return fn(y, p, w)

    return wrapped


@dataclass(frozen=True)
class Comparison:
    """Two cards side by side. ``better`` says which label wins each metric, or 'tie' / '-'."""

    left: str
    right: str
    rows: list[tuple[str, float, float, str]]

    def __str__(self) -> str:
        """Fixed-width table: metric, left, right, winner."""
        lines = [f"{'metric':<20}{self.left:>14}{self.right:>14}  better"]
        lines += [f"{n:<20}{a:>14.5g}{b:>14.5g}  {v}" for n, a, b, v in self.rows]
        return "\n".join(lines)


def compare(a: Scorecard, b: Scorecard) -> Comparison:
    """Put two scorecards side by side, metric by metric, with a direction-aware verdict.

    Both cards must come from the same family (otherwise the metrics don't mean the same
    thing) and the same data size (a hint they were scored on the same split).

    Examples
    --------
    >>> from glasshouse.scorecard import scorecard, compare
    >>> y = [0, 1, 2, 3]
    >>> a = scorecard(y, [0.5, 1.0, 2.0, 2.5], family="poisson", label="a")
    >>> b = scorecard(y, [1.5, 1.5, 1.5, 1.5], family="poisson", label="b")
    >>> [row[3] for row in compare(a, b).rows if row[0] == "d2"]
    ['a']
    """
    if a.family != b.family:
        msg = (
            f"cannot compare family={a.family!r} with family={b.family!r}: "
            "the deviances mean different things — score both models with the same family"
        )
        raise ValueError(msg)
    if a.n_rows != b.n_rows:
        msg = (
            f"{a.label} was scored on {a.n_rows} rows and {b.label} on {b.n_rows}: "
            "compare models on the same split"
        )
        raise ValueError(msg)
    rows: list[tuple[str, float, float, str]] = []
    for name in a.metrics:
        verdict = _verdict(name, a.metrics[name], b.metrics[name])
        winner = {"yes": a.label, "no": b.label}.get(verdict, verdict)
        rows.append((name, a.metrics[name], b.metrics[name], winner))
    return Comparison(left=a.label, right=b.label, rows=rows)


__all__ = [
    "HIGHER_IS_BETTER",
    "Comparison",
    "Scorecard",
    "compare",
    "naive_prediction",
    "scorecard",
]
