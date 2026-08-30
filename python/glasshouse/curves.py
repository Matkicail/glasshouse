"""The four curves, as data. Plots render these; reports serialise them; nothing else computes them.

Every function returns a frozen dataclass of plain arrays with a ``to_dict`` — the JSON
contract that :mod:`glasshouse.plots` (Python) and the HTML report (TypeScript) both read,
so the two can never disagree about a number.

- :func:`lorenz` — cumulative exposure vs cumulative outcome, ranked by score (Gini's curve).
- :func:`lift` — actual vs predicted by prediction decile (the calibration table, drawn).
- :func:`double_lift` — two models, ranked by the ratio of their predictions: where they
  disagree most, which is closer to the truth?
- :func:`calibration` — the reliability table with a perfect-line reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse import _core, metrics
from glasshouse.arrays import F64, ArrayLike, to_vector


def _w(sample_weight: ArrayLike | None) -> F64 | None:
    return None if sample_weight is None else to_vector(sample_weight, "sample_weight")


@dataclass(frozen=True)
class Lorenz:
    """Points of the Lorenz curve plus the Gini they integrate to."""

    x: F64  # cumulative share of weight, ranked by score ascending
    y: F64  # cumulative share of outcome
    gini: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {"kind": "lorenz", **_lists(asdict(self))}


@dataclass(frozen=True)
class Lift:
    """Actual vs predicted by prediction bin (equal weight), lowest predictions first."""

    bin: npt.NDArray[np.int64]
    weight: F64
    predicted: F64
    actual: F64
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {"kind": "lift", **_lists(asdict(self))}


@dataclass(frozen=True)
class DoubleLift:
    """Two models compared where they disagree: bins by ``a / b``, both predictions vs actual."""

    bin: npt.NDArray[np.int64]
    weight: F64
    ratio: F64
    actual: F64
    predicted_a: F64
    predicted_b: F64
    label_a: str
    label_b: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {"kind": "double_lift", **_lists(asdict(self))}


@dataclass(frozen=True)
class Calibration:
    """Reliability: mean prediction vs mean outcome per bin, with the actual/expected ratio."""

    bin: npt.NDArray[np.int64]
    weight: F64
    predicted: F64
    actual: F64
    actual_over_expected: F64
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {"kind": "calibration", **_lists(asdict(self))}


def lorenz(
    y: ArrayLike, score: ArrayLike, sample_weight: ArrayLike | None = None, label: str = "model"
) -> Lorenz:
    """Lorenz curve ranked by ``score``; ties are one point, so row order never matters.

    Examples
    --------
    >>> from glasshouse.curves import lorenz
    >>> c = lorenz([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    >>> c.x.tolist(), c.y.tolist(), c.gini
    ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.5, 1.0], 0.5)
    """
    yy, ss, w = to_vector(y, "y"), to_vector(score, "score"), _w(sample_weight)
    xs, ys = _core.lorenz_curve(yy, ss, w)
    return Lorenz(
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        gini=metrics.gini(yy, ss, w),
        label=label,
    )


def lift(
    y: ArrayLike,
    mu: ArrayLike,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
    label: str = "model",
) -> Lift:
    """Actual vs predicted by prediction decile — the calibration table as a chart.

    Examples
    --------
    >>> from glasshouse.curves import lift
    >>> c = lift([0, 1, 1, 2, 2, 4], [0.5, 0.5, 1.5, 1.5, 3.0, 3.0], n_bins=3)
    >>> c.predicted.tolist(), c.actual.tolist()
    ([0.5, 1.5, 3.0], [0.5, 1.5, 3.0])
    """
    t = metrics.calibration_table(y, mu, sample_weight=sample_weight, n_bins=n_bins)
    return Lift(
        bin=np.arange(len(t), dtype=np.int64),
        weight=t.weight,
        predicted=t.predicted,
        actual=t.actual,
        label=label,
    )


def double_lift(  # noqa: PLR0913, PLR0917 — two models, two labels; all trailing args optional
    y: ArrayLike,
    mu_a: ArrayLike,
    mu_b: ArrayLike,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
    label_a: str = "a",
    label_b: str = "b",
) -> DoubleLift:
    """Rank rows by ``mu_a / mu_b``, bin by equal weight, compare both to the outcome.

    In the bins where the models disagree most (far left, far right), the one whose line sits
    on the actual line is the one to trust there. Between equal-deviance models this is the
    chart that settles it.

    Examples
    --------
    >>> from glasshouse.curves import double_lift
    >>> c = double_lift([1, 2, 3, 4], [1, 2, 3, 4], [2, 2, 2, 2], n_bins=2)
    >>> c.actual.tolist(), c.predicted_a.tolist(), c.predicted_b.tolist()
    ([1.5, 3.5], [1.5, 3.5], [2.0, 2.0])
    """
    raw = _core.double_lift_table(
        to_vector(y, "y"),
        to_vector(mu_a, "mu_a"),
        to_vector(mu_b, "mu_b"),
        _w(sample_weight),
        n_bins,
    )
    return DoubleLift(
        bin=np.arange(len(raw["weight"]), dtype=np.int64),
        weight=np.asarray(raw["weight"], dtype=np.float64),
        ratio=np.asarray(raw["ratio"], dtype=np.float64),
        actual=np.asarray(raw["actual"], dtype=np.float64),
        predicted_a=np.asarray(raw["predicted_a"], dtype=np.float64),
        predicted_b=np.asarray(raw["predicted_b"], dtype=np.float64),
        label_a=label_a,
        label_b=label_b,
    )


def calibration(
    y: ArrayLike,
    mu: ArrayLike,
    sample_weight: ArrayLike | None = None,
    n_bins: int = 10,
    label: str = "model",
) -> Calibration:
    """Reliability curve: mean prediction vs mean outcome per bin, and their ratio.

    Examples
    --------
    >>> from glasshouse.curves import calibration
    >>> c = calibration([0, 1, 1, 2, 2, 4], [0.5, 0.5, 1.5, 1.5, 3.0, 3.0], n_bins=3)
    >>> c.actual_over_expected.tolist()
    [1.0, 1.0, 1.0]
    """
    t = metrics.calibration_table(y, mu, sample_weight=sample_weight, n_bins=n_bins)
    return Calibration(
        bin=np.arange(len(t), dtype=np.int64),
        weight=t.weight,
        predicted=t.predicted,
        actual=t.actual,
        actual_over_expected=t.actual_over_expected,
        label=label,
    )


def _lists(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


__all__ = [
    "Calibration",
    "DoubleLift",
    "Lift",
    "Lorenz",
    "calibration",
    "double_lift",
    "lift",
    "lorenz",
]
