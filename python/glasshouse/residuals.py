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
    raw = np.asarray(feature.to_numpy() if hasattr(feature, "to_numpy") else feature)
    if raw.dtype.kind in "fiub":
        key = to_vector(raw, name)
        t = _core.binned_table(key, yy, mm, w, n_bins)
        edges = _bin_edges(key, w, len(t["weight"]))
        levels = [
            f"[{lo:.4g}, {hi:.4g}{')' if i < len(edges) - 2 else ']'}"
            for i, (lo, hi) in enumerate(pairwise(edges))
        ]
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
    labels = raw.astype(str)
    levels_arr, codes = np.unique(labels, return_inverse=True)
    ww = np.ones(len(yy)) if w is None else w
    sw = np.bincount(codes, weights=ww, minlength=len(levels_arr))
    swy = np.bincount(codes, weights=ww * yy, minlength=len(levels_arr))
    swm = np.bincount(codes, weights=ww * mm, minlength=len(levels_arr))
    return AEByFeature(
        feature=name,
        level=levels_arr.tolist(),
        n_rows=np.bincount(codes, minlength=len(levels_arr)).astype(np.int64),
        weight=sw,
        predicted=swm / sw,
        actual=swy / sw,
        actual_over_expected=swy / swm,
        label=label,
    )


def _bin_edges(key: F64, w: F64 | None, n_bins: int) -> list[float]:
    """Return feature values at the bin boundaries, for labels only.

    The Rust table decides the bins; this reproduces the equal-weight cut points on the sorted
    key so the labels read "[18, 25)".
    """
    order = np.argsort(key, kind="stable")
    ks = key[order]
    ww = np.ones(len(ks)) if w is None else w[order]
    cum = np.cumsum(ww)
    edges = [float(ks[0])]
    for i in range(1, n_bins):
        idx = int(np.searchsorted(cum, cum[-1] * i / n_bins))
        edges.append(float(ks[min(idx, len(ks) - 1)]))
    edges.append(float(ks[-1]))
    return edges


__all__ = ["AEByFeature", "ae_by_feature", "deviance", "pearson"]
