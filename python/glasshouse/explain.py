"""Explaining any model through its predictions: partial dependence and permutation importance.

A GLM is glass-box: its coefficients say how it prices. A boosted tree is not, and the
promise is still to understand it: interpretable when we can, explained when we cannot,
never unexamined. These two tools need nothing but ``predict``, so every model on a report
gets the same picture, and for a GLM they agree with its coefficients (a test checks that).

- **Partial dependence** (Friedman 2001): set one feature to a value on every row, average
  the predictions, repeat along a grid. The curve is what the model says the feature does,
  averaged over how the other features actually co-occur.
- **Permutation importance** (Breiman 2001; Fisher, Rudin & Dominici 2019): shuffle one
  feature on held-out rows and see how much worse the deviance gets. A feature the model
  never uses costs nothing to shuffle.

Both are computed on held-out rows in the bench, so they describe the fitted model on data
it did not see. Both assume features can be varied one at a time; where two features are
tightly bound (age and licence years, say) the picture is of a model asked about rows that
do not exist, and the residuals tab is the honest complement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse._rows import as_array
from glasshouse.arrays import F64, ArrayLike, to_vector
from glasshouse.metrics import FamilyName, deviance


@dataclass(frozen=True)
class PartialDependence:
    """One feature's partial dependence curve for one model: the mean prediction per grid point."""

    feature: str
    kind: str  # "numeric" | "categorical"
    grid: list[float] | list[str]
    effect: F64
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "feature": self.feature,
            "kind": self.kind,
            "grid": list(self.grid),
            "effect": self.effect.tolist(),
            "label": self.label,
        }


@dataclass(frozen=True)
class Importance:
    """Permutation importance for one model: deviance increase per shuffled feature."""

    features: list[str]
    loss: F64
    base_deviance: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready."""
        return {
            "features": list(self.features),
            "loss": self.loss.tolist(),
            "base_deviance": self.base_deviance,
            "label": self.label,
        }

    def __str__(self) -> str:
        """Features by importance, highest first."""
        order = np.argsort(-self.loss)
        lines = [f"{'feature':<20}{'deviance increase':>18}"]
        lines.extend(f"{self.features[i]:<20}{self.loss[i]:>18.6g}" for i in order)
        return "\n".join(lines)


def numeric_grid(column: ArrayLike, n_points: int = 20) -> list[float]:
    """Quantiles of a numeric column, evenly spaced in probability: the grid a curve uses.

    Quantiles rather than an even spacing, so the curve is drawn where the data is and the
    tails do not get a third of the picture for a hundredth of the rows.
    """
    v = to_vector(column, "column")
    return [float(q) for q in np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_points)))]


def is_categorical(column: ArrayLike) -> bool:
    """Strings, objects and categoricals are levels; numbers are a grid."""
    arr = as_array(column)
    return arr.dtype.kind in "OUSb" or str(arr.dtype) == "category"


def partial_dependence(
    model: Any,
    frame: Any,
    feature: str,
    *,
    grid: list[float] | list[str] | None = None,
    n_points: int = 20,
    label: str = "model",
) -> PartialDependence:
    """Average the model's predictions with ``feature`` set to each grid value on every row.

    ``frame`` is the rows to average over (held-out rows, ideally, and a sample of them is
    plenty). ``grid`` defaults to the column's quantiles, or its levels when categorical.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from glasshouse import GLM
    >>> from glasshouse.explain import partial_dependence
    >>> x = np.linspace(0.0, 1.0, 50)
    >>> df = pd.DataFrame({"x": x, "z": x**2})
    >>> y = 1.0 + 2.0 * df.x - 1.0 * df.z
    >>> m = GLM(family="gaussian").fit(df, y)
    >>> pd_x = partial_dependence(m, df, "x", grid=[0.0, 0.5, 1.0])
    >>> pd_x.effect.round(6).tolist()  # slope 2 in x, z held at its mean (about 0.34)
    [0.663265, 1.663265, 2.663265]
    """
    column = frame[feature]
    categorical = is_categorical(column)
    points: list[float] | list[str]
    if categorical:
        levels = sorted({str(v) for v in as_array(column)}) if grid is None else grid
        points = [str(v) for v in levels]
    else:
        values = numeric_grid(column, n_points) if grid is None else grid
        points = [float(v) for v in values]
    effects = []
    for value in points:
        probe = frame.copy()
        probe[feature] = value
        effects.append(float(np.mean(model.predict(probe))))
    return PartialDependence(
        feature=feature,
        kind="categorical" if categorical else "numeric",
        grid=points,
        effect=np.asarray(effects, dtype=np.float64),
        label=label,
    )


def permutation_importance(  # noqa: PLR0913 — a model, its data, the family it is scored with
    model: Any,
    frame: Any,
    y: ArrayLike,
    *,
    family: FamilyName,
    features: list[str],
    power: float | None = None,
    sample_weight: ArrayLike | None = None,
    seed: int = 0,
    label: str = "model",
) -> Importance:
    """Shuffle each feature in turn and measure how much the mean deviance rises.

    Scored the way the report scores: ``y`` and the predictions on the same scale, with the
    exposure as ``sample_weight`` for a rate model. One shuffle per feature, seeded; on a
    few thousand rows that is stable enough to rank, and the bench's fold spread says how
    stable.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from glasshouse import GLM
    >>> from glasshouse.explain import permutation_importance
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"signal": rng.normal(size=400), "noise": rng.normal(size=400)})
    >>> y = 1.0 + 2.0 * df.signal + rng.normal(scale=0.1, size=400)
    >>> imp = permutation_importance(GLM(family="gaussian").fit(df, y), df, y,
    ...                              family="gaussian", features=["signal", "noise"])
    >>> bool(imp.loss[0] > 1.0 and abs(imp.loss[1]) < 0.01)
    True
    """
    yy = to_vector(y, "y")
    w = None if sample_weight is None else to_vector(sample_weight, "sample_weight")
    base = deviance(yy, model.predict(frame), family=family, power=power, sample_weight=w)
    rng = np.random.default_rng(seed)
    loss = []
    for feature in features:
        shuffled = frame.copy()
        shuffled[feature] = as_array(frame[feature])[rng.permutation(len(yy))]
        worse = deviance(yy, model.predict(shuffled), family=family, power=power, sample_weight=w)
        loss.append(worse - base)
    return Importance(
        features=list(features),
        loss=np.asarray(loss, dtype=np.float64),
        base_deviance=float(base),
        label=label,
    )


def coefficients(model: Any) -> dict[str, float] | None:
    """Return a glass-box model's coefficients by term name, intercept first, or ``None``.

    Any model exposing ``feature_names_in_``, ``intercept_`` and ``coef_`` the way the GLM
    does qualifies; the names carry the term structure (``age_s3``, ``Region_target``).
    """
    names = getattr(model, "feature_names_in_", None)
    coef = getattr(model, "coef_", None)
    if names is None or coef is None:
        return None
    values = np.asarray(coef, dtype=np.float64)
    if len(names) == len(values) + 1:
        values = np.concatenate([[float(getattr(model, "intercept_", 0.0))], values])
    if len(names) != len(values):
        return None
    return {str(n): float(v) for n, v in zip(names, values, strict=True)}


IndexArray = npt.NDArray[np.int64]

__all__ = [
    "Importance",
    "PartialDependence",
    "coefficients",
    "is_categorical",
    "numeric_grid",
    "partial_dependence",
    "permutation_importance",
]
