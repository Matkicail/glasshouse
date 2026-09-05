"""Profiling the data a report is about, before any model is involved.

A review starts with the data, not the leaderboard: how much of the book sits in each age
band, what the claim rate does across it, how heavy the tail of the outcome is. Two tables
say that:

- :func:`histogram` — even-width bins of one column with the weight in each; for the outcome
  and the exposure.
- :func:`feature_profile` — for one feature, the weight in each bin (even-width for a
  numeric, one row per level for a categorical) with the weighted mean outcome there. The
  one-way view without a model: the bars say where the data is, the line what happens there.

The bins are even in width, not in weight, on purpose: the residuals module's one-way tables
use equal-weight bins so every A/E has the same backing, which makes the weight bars flat by
construction. Here the point is the shape of the distribution, so the width is fixed and the
weight varies. Weighted means follow the one convention in ``docs/methods.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse._rows import as_array
from glasshouse.arrays import F64, ArrayLike, to_vector
from glasshouse.metrics import _weights


@dataclass(frozen=True)
class Histogram:
    """Even-width bins of one column: rows and weight per bin, and a weighted summary."""

    name: str
    level: list[str]  # bin label ("[0, 0.5)"); the last one is the pooled tail when there is one
    edges: list[float]
    n_rows: npt.NDArray[np.int64]
    weight: F64
    summary: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "name": self.name,
            "level": list(self.level),
            "edges": list(self.edges),
            "n_rows": self.n_rows.tolist(),
            "weight": self.weight.tolist(),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class FeatureProfile:
    """Weight and mean outcome per bin or level of one feature."""

    feature: str
    kind: str  # "numeric" | "categorical"
    level: list[str]  # bin label ("[18, 25)") or the category
    edges: list[float] | None  # bin boundaries for a numeric feature
    n_rows: npt.NDArray[np.int64]
    weight: F64
    actual: F64  # weighted mean of y per bin
    n_levels: int  # distinct values seen (more than len(level) when levels were pooled)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "feature": self.feature,
            "kind": self.kind,
            "level": list(self.level),
            "edges": None if self.edges is None else list(self.edges),
            "n_rows": self.n_rows.tolist(),
            "weight": self.weight.tolist(),
            "actual": self.actual.tolist(),
            "n_levels": self.n_levels,
        }

    def __str__(self) -> str:
        """Fixed-width table."""
        lines = [
            f"{self.feature} ({self.kind}, {self.n_levels} distinct)",
            f"{'level':<22}{'rows':>8}{'weight':>12}{'share':>8}{'mean y':>12}",
        ]
        total = self.weight.sum()
        for lv, n, w, a in zip(self.level, self.n_rows, self.weight, self.actual, strict=True):
            lines.append(f"{lv:<22}{n:>8d}{w:>12.4g}{w / total:>8.3f}{a:>12.4g}")
        return "\n".join(lines)


def histogram(
    values: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    name: str = "y",
    n_bins: int = 40,
    tail: float = 0.99,
) -> Histogram:
    """Bin a column into ``n_bins`` even widths and summarise it with its weights.

    The bins span the range up to the weighted ``tail`` quantile; whatever lies above it is
    pooled into one last bin that runs to the maximum. Without that, a claim rate whose
    largest value is a thousand times its 99th percentile draws as a single bar. The
    summary is weighted where a weight is given (mean, std, quantiles); ``zero_share`` is
    the share of the weight sitting on exact zeros, the number a frequency or pure-premium
    reviewer wants first.

    Examples
    --------
    >>> from glasshouse.profile import histogram
    >>> h = histogram([0, 0, 1, 3], [1, 1, 1, 1], name="claims", n_bins=3)
    >>> h.edges, h.n_rows.tolist(), h.summary["zero_share"]
    ([0.0, 1.0, 2.0, 3.0], [2, 1, 1], 0.5)
    >>> heavy = histogram([*range(100), 1000], n_bins=3)  # the 99th percentile is 99
    >>> heavy.level, heavy.n_rows.tolist()
    (['[0, 33)', '[33, 66)', '[66, 99)', '[99, 1000]'], [33, 33, 33, 2])
    """
    v = to_vector(values, name)
    w = _weights(sample_weight)
    _same_length(name, v, w)
    ww = np.ones(len(v)) if w is None else w
    edges = _edges(v, ww, n_bins, tail)
    counts, _ = np.histogram(v, bins=edges)
    weight, _ = np.histogram(v, bins=edges, weights=ww)
    return Histogram(
        name=name,
        level=_labels(edges),
        edges=[float(e) for e in edges],
        n_rows=counts.astype(np.int64),
        weight=np.asarray(weight, dtype=np.float64),
        summary=_summary(v, ww),
    )


def feature_profile(  # noqa: PLR0913 — one feature, the outcome, and the binning knobs
    column: ArrayLike,
    y: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    name: str = "feature",
    n_bins: int = 20,
    max_levels: int = 30,
    tail: float = 0.99,
) -> FeatureProfile:
    """Weight and weighted mean outcome per bin of ``column``.

    A numeric column gets ``n_bins`` even-width bins up to its weighted ``tail`` quantile
    and one pooled bin beyond it (see :func:`histogram`), labelled by their edges. A
    categorical one gets a row per level, heaviest first; past ``max_levels`` the rest are
    pooled into one ``(other)`` row so a postcode does not become a thousand bars.

    Examples
    --------
    >>> from glasshouse.profile import feature_profile
    >>> p = feature_profile(["a", "b", "b", "a", "b"], [1, 0, 2, 3, 4], name="group")
    >>> p.level, p.weight.tolist(), p.actual.tolist()
    (['b', 'a'], [3.0, 2.0], [2.0, 2.0])
    >>> feature_profile([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4], name="x", n_bins=2).level
    ['[1, 2.5)', '[2.5, 4]']
    """
    yy = to_vector(y, "y")
    w = _weights(sample_weight)
    _same_length("y", yy, w)
    ww = np.ones(len(yy)) if w is None else w
    raw = as_array(column)
    if len(raw) != len(yy):
        msg = f"{name} has {len(raw)} rows but y has {len(yy)}"
        raise ValueError(msg)
    if raw.dtype.kind in "fiub":
        return _numeric_profile(to_vector(raw, name), yy, ww, name, _edges(raw, ww, n_bins, tail))
    return _categorical_profile(raw.astype(str), yy, ww, name, max_levels)


def _numeric_profile(key: F64, yy: F64, ww: F64, name: str, edges: F64) -> FeatureProfile:
    counts, _ = np.histogram(key, bins=edges)
    weight, _ = np.histogram(key, bins=edges, weights=ww)
    weighted_y, _ = np.histogram(key, bins=edges, weights=ww * yy)
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = weighted_y / weight
    return FeatureProfile(
        feature=name,
        kind="numeric",
        level=_labels(edges),
        edges=[float(e) for e in edges],
        n_rows=counts.astype(np.int64),
        weight=np.asarray(weight, dtype=np.float64),
        actual=np.asarray(actual, dtype=np.float64),
        n_levels=len(np.unique(key)),
    )


def _categorical_profile(
    labels: npt.NDArray[np.str_], yy: F64, ww: F64, name: str, max_levels: int
) -> FeatureProfile:
    levels, codes = np.unique(labels, return_inverse=True)
    n = np.bincount(codes, minlength=len(levels))
    sw = np.bincount(codes, weights=ww, minlength=len(levels))
    swy = np.bincount(codes, weights=ww * yy, minlength=len(levels))
    order = np.argsort(-sw, kind="stable")
    shown, pooled = order[:max_levels], order[max_levels:]
    level = [str(levels[i]) for i in shown]
    n_rows, weight, weighted_y = n[shown], sw[shown], swy[shown]
    if len(pooled):
        level.append(f"(other: {len(pooled)} levels)")
        n_rows = np.append(n_rows, n[pooled].sum())
        weight = np.append(weight, sw[pooled].sum())
        weighted_y = np.append(weighted_y, swy[pooled].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = weighted_y / weight
    return FeatureProfile(
        feature=name,
        kind="categorical",
        level=level,
        edges=None,
        n_rows=n_rows.astype(np.int64),
        weight=np.asarray(weight, dtype=np.float64),
        actual=np.asarray(actual, dtype=np.float64),
        n_levels=len(levels),
    )


def _edges(v: F64, ww: F64, n_bins: int, tail: float) -> F64:
    """Even-width edges from the minimum to the ``tail`` quantile, then one edge at the maximum."""
    lo, hi, top = float(v.min()), _quantile(v, ww, tail), float(v.max())
    if hi <= lo or hi >= top:
        return np.asarray(np.histogram_bin_edges(v, bins=n_bins), dtype=np.float64)
    return np.append(np.linspace(lo, hi, n_bins + 1), top)


def _labels(edges: F64) -> list[str]:
    return [
        f"[{lo:.4g}, {hi:.4g}{')' if i < len(edges) - 2 else ']'}"
        for i, (lo, hi) in enumerate(pairwise(edges))
    ]


def _quantile(v: F64, ww: F64, p: float) -> float:
    """Return the value where the cumulative weight, on the sorted values, first reaches ``p``."""
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(ww[order]) / ww.sum()
    return float(v[order][min(int(np.searchsorted(cum, p)), len(v) - 1)])


def _summary(v: F64, ww: F64) -> dict[str, float]:
    """Weighted mean, std and quantiles."""
    total = ww.sum()
    lo, hi = float(v.min()), float(v.max())
    mean = min(max(float((ww * v).sum() / total), lo), hi)  # rounding must not put it outside
    std = float(np.sqrt((ww * (v - mean) ** 2).sum() / total))
    quantile = {
        q: _quantile(v, ww, p)
        for q, p in (("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95))
    }
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        **quantile,
        "max": hi,
        "zero_share": float(ww[v == 0.0].sum() / total),
    }


def _same_length(name: str, v: F64, w: F64 | None) -> None:
    if w is not None and len(w) != len(v):
        msg = f"sample_weight has {len(w)} rows but {name} has {len(v)}"
        raise ValueError(msg)


__all__ = ["FeatureProfile", "Histogram", "feature_profile", "histogram"]
