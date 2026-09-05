"""Residuals: what each model gets wrong, and where.

- :func:`deviance` — ``sign(y - mu) * sqrt(w * d(y, mu))``, the family's own residual. For a
  well-specified model roughly symmetric with unit variance; a skewed histogram or a curved
  QQ plot says the family is wrong.
- :func:`pearson` — ``(y - mu) * sqrt(w / V(mu))``, on the dispersion's scale; the squares
  sum to the Pearson chi-square.
- :func:`ae_by_feature` — actual / expected sliced by a feature: equal-weight bins for a
  numeric column, one row per level for a categorical. This is the table that finds the
  segment a model under- or over-prices. It is the calibration table keyed by the feature
  instead of the prediction, so it uses the same tie policy and the same numbers.
- :func:`ae_by_two` — the same on a grid of two features: the interaction view. A one-way
  A/E averages over the other feature, so a model that is right on average for young
  drivers and right on average for powerful cars can still be badly wrong for young drivers
  in powerful cars; this is the table that shows it. Cells under a weight floor are marked
  thin so the chart can grey them out.

All computed in Rust; this module converts and labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse import _core
from glasshouse.arrays import F64, ArrayLike, to_vector
from glasshouse.metrics import FamilyName, _weights


def deviance(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    family: FamilyName,
    sample_weight: ArrayLike | None = None,
    power: float | None = None,
) -> F64:
    """Deviance residuals, one per row.

    Examples
    --------
    >>> from glasshouse.residuals import deviance
    >>> deviance([0, 2], [1, 1], family="poisson").round(4).tolist()
    [-1.4142, 0.879]
    """
    r = _core.residuals(
        "deviance", family, to_vector(y, "y"), to_vector(mu, "mu"), _weights(sample_weight), power
    )
    return np.asarray(r, dtype=np.float64)


def pearson(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    family: FamilyName,
    sample_weight: ArrayLike | None = None,
    power: float | None = None,
) -> F64:
    """Pearson residuals, one per row.

    Examples
    --------
    >>> from glasshouse.residuals import pearson
    >>> pearson([0, 2], [1, 1], family="poisson").tolist()
    [-1.0, 1.0]
    """
    r = _core.residuals(
        "pearson", family, to_vector(y, "y"), to_vector(mu, "mu"), _weights(sample_weight), power
    )
    return np.asarray(r, dtype=np.float64)


@dataclass(frozen=True)
class AEByFeature:
    """Actual / expected per bin or level of one feature, lowest first."""

    feature: str
    level: list[str]  # bin label ("[18, 25)") or the category
    n_rows: npt.NDArray[np.int64]
    weight: F64
    predicted: F64
    actual: F64
    actual_over_expected: F64
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "kind": "ae_by_feature",
            "feature": self.feature,
            "label": self.label,
            "level": list(self.level),
            "n_rows": self.n_rows.tolist(),
            "weight": self.weight.tolist(),
            "predicted": self.predicted.tolist(),
            "actual": self.actual.tolist(),
            "actual_over_expected": self.actual_over_expected.tolist(),
        }

    def __str__(self) -> str:
        """Fixed-width table."""
        lines = [
            f"{self.label}: A/E by {self.feature}",
            f"{'level':<18}{'weight':>12}{'pred':>12}{'actual':>12}{'A/E':>8}",
        ]
        for lv, w, p, a, r in zip(
            self.level,
            self.weight,
            self.predicted,
            self.actual,
            self.actual_over_expected,
            strict=True,
        ):
            lines.append(f"{lv:<18}{w:>12.4g}{p:>12.4g}{a:>12.4g}{r:>8.3f}")
        return "\n".join(lines)


def ae_by_feature(  # noqa: PLR0913 — one feature, one model, optional knobs
    feature: ArrayLike,
    y: ArrayLike,
    mu: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    name: str = "feature",
    n_bins: int = 10,
    label: str = "model",
) -> AEByFeature:
    """Actual / expected by a feature: equal-weight bins if numeric, one row per level if not.

    Read it as: where ``actual_over_expected`` sits well above 1, the model under-predicts for
    that segment; well below 1, it over-predicts. Bins with little weight are noisy — the
    ``weight`` column is there so you don't over-read them.

    Examples
    --------
    >>> from glasshouse.residuals import ae_by_feature
    >>> t = ae_by_feature(["a", "a", "b", "b"], [1, 3, 2, 2], [2, 2, 2, 2], name="region")
    >>> t.level, t.actual_over_expected.tolist()
    (['a', 'b'], [1.0, 1.0])
    """
    yy, mm, w = to_vector(y, "y"), to_vector(mu, "mu"), _weights(sample_weight)
    index, levels = _axis(feature, w, n_bins, name)
    t = _core.grid_table(index, len(levels), [0] * len(yy), 1, yy, mm, w)
    return AEByFeature(
        feature=name,
        level=levels,
        n_rows=np.asarray(t["n_rows"], dtype=np.int64),
        weight=np.asarray(t["weight"], dtype=np.float64),
        predicted=np.asarray(t["predicted"], dtype=np.float64),
        actual=np.asarray(t["actual"], dtype=np.float64),
        actual_over_expected=np.asarray(t["actual_over_expected"], dtype=np.float64),
        label=label,
    )


@dataclass(frozen=True)
class AEGrid:
    """Actual / expected per cell of a two-feature grid; row ``i`` is ``level_a[i]``."""

    feature_a: str
    feature_b: str
    level_a: list[str]
    level_b: list[str]
    n_rows: npt.NDArray[np.int64]  # (len(level_a), len(level_b))
    weight: F64
    predicted: F64
    actual: F64
    actual_over_expected: F64
    weight_floor: float  # cells lighter than this are too thin to read
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready (nested lists, row-major)."""
        return {
            "kind": "ae_by_two",
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "label": self.label,
            "level_a": list(self.level_a),
            "level_b": list(self.level_b),
            "n_rows": self.n_rows.tolist(),
            "weight": self.weight.tolist(),
            "predicted": self.predicted.tolist(),
            "actual": self.actual.tolist(),
            "actual_over_expected": self.actual_over_expected.tolist(),
            "weight_floor": self.weight_floor,
        }

    def __str__(self) -> str:
        """A/E per cell, rows are ``feature_a``; thin cells shown as ``·``."""
        width = max(8, *(len(lv) + 2 for lv in self.level_b))
        lines = [
            f"{self.label}: A/E by {self.feature_a} (rows) and {self.feature_b} (columns)",
            f"{'':<18}" + "".join(f"{lv:>{width}}" for lv in self.level_b),
        ]
        for i, lv in enumerate(self.level_a):
            cells = "".join(
                f"{'·':>{width}}"
                if self.weight[i, j] < self.weight_floor
                else f"{self.actual_over_expected[i, j]:>{width}.3f}"
                for j in range(len(self.level_b))
            )
            lines.append(f"{lv:<18}{cells}")
        return "\n".join(lines)


def ae_by_two(  # noqa: PLR0913 — two features, one model, the knobs
    feature_a: ArrayLike,
    feature_b: ArrayLike,
    y: ArrayLike,
    mu: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    names: tuple[str, str] = ("a", "b"),
    n_bins: int = 10,
    label: str = "model",
    floor: float = 0.2,
) -> AEGrid:
    """Actual / expected on the grid of two features: bins by weight if numeric, levels if not.

    Each feature is cut exactly as :func:`ae_by_feature` cuts it (the marginals of the grid
    are the one-way tables). A cell is *thin* when its weight is below ``floor`` times the
    average cell's weight; thin cells are kept in the table, with ``weight_floor`` beside
    them, so the reader decides what to trust.

    Examples
    --------
    >>> from glasshouse.residuals import ae_by_two
    >>> g = ae_by_two(["x", "x", "y", "y"], [0.0, 1.0, 0.0, 1.0], [1, 3, 2, 2], [2, 2, 2, 2],
    ...               names=("group", "flag"), n_bins=2)
    >>> g.level_a, g.level_b, g.actual_over_expected.tolist()
    (['x', 'y'], ['[0, 1)', '[1, 1]'], [[0.5, 1.5], [1.0, 1.0]])
    """
    yy, mm, w = to_vector(y, "y"), to_vector(mu, "mu"), _weights(sample_weight)
    idx_a, level_a = _axis(feature_a, w, n_bins, names[0])
    idx_b, level_b = _axis(feature_b, w, n_bins, names[1])
    t = _core.grid_table(idx_a, len(level_a), idx_b, len(level_b), yy, mm, w)
    shape = (len(level_a), len(level_b))
    weight = np.asarray(t["weight"], dtype=np.float64).reshape(shape)
    return AEGrid(
        feature_a=names[0],
        feature_b=names[1],
        level_a=level_a,
        level_b=level_b,
        n_rows=np.asarray(t["n_rows"], dtype=np.int64).reshape(shape),
        weight=weight,
        predicted=np.asarray(t["predicted"], dtype=np.float64).reshape(shape),
        actual=np.asarray(t["actual"], dtype=np.float64).reshape(shape),
        actual_over_expected=np.asarray(t["actual_over_expected"], dtype=np.float64).reshape(shape),
        weight_floor=float(floor * weight.sum() / weight.size),
        label=label,
    )


def _axis(feature: ArrayLike, w: F64 | None, n_bins: int, name: str) -> tuple[list[int], list[str]]:
    """Cut one feature: the bin index per row and a label per bin.

    Numeric: equal-weight bins, ties whole (the Rust rule); a bin is labelled by its lowest
    value and the next bin's lowest value, the last one closed at the maximum. Categorical:
    one bin per level, sorted.
    """
    raw = np.asarray(feature.to_numpy() if hasattr(feature, "to_numpy") else feature)
    if raw.dtype.kind not in "fiub":
        levels, codes = np.unique(raw.astype(str), return_inverse=True)
        return codes.tolist(), levels.tolist()
    key = to_vector(raw, name)
    index = np.asarray(_core.bin_index(key, w, n_bins))
    n = int(index.max()) + 1
    lows = [float(key[index == b].min()) for b in range(n)]
    edges = [*lows, float(key.max())]
    labels = [
        f"[{lo:.4g}, {hi:.4g}{')' if i < n - 1 else ']'}"
        for i, (lo, hi) in enumerate(pairwise(edges))
    ]
    return index.tolist(), labels


__all__ = ["AEByFeature", "AEGrid", "ae_by_feature", "ae_by_two", "deviance", "pearson"]
