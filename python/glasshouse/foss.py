"""Adapters that put FOSS GLMs behind the bench's Model protocol.

FOSS is the benchmark: these wrappers let glum and scikit-learn run in the same bench, on the
same folds, with the same design matrix, so a comparison is about the solvers and nothing
else. Choices made for fairness, stated plainly:

- Every model (ours included) gets the identical design: one-hot with the first level dropped,
  levels learned on the training fold, numeric columns passed through.
- ``alpha=0`` everywhere, because the comparison is the unpenalized GLM (scikit-learn's
  default is ``alpha=1``, which would be a different model).
- scikit-learn's ``PoissonRegressor`` has no offset, so its adapter fits rates with exposure
  as ``sample_weight`` — for the Poisson deviance this is the standard equivalent form.
- Tolerances are tightened (``tol=1e-8``) so every solver runs to a comparable optimum; fit
  seconds are recorded by the bench either way.

These need the dev dependencies (``glum``, ``scikit-learn``); import errors say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glasshouse.arrays import F64, ArrayLike, columns, to_vector
from glasshouse.encoders import OneHot
from glasshouse.splits import Fold


@dataclass
class FossDesign:
    """The shared design builder: one-hot categoricals (fit on train rows), numerics as-is."""

    onehot: list[str]
    encoders_: dict[str, OneHot] = field(default_factory=dict, repr=False)
    columns_: list[str] = field(default_factory=list, repr=False)

    def fit_transform(self, x: ArrayLike, rows: np.ndarray | None) -> F64:
        """Learn encoders on ``rows`` of ``x`` and return the design for those rows."""
        cols = columns(x)
        if cols is None:
            msg = "the FOSS adapters need a DataFrame with named columns"
            raise ValueError(msg)
        self.columns_ = [str(n) for n, _ in cols]
        unknown = sorted(set(self.onehot) - set(self.columns_))
        if unknown:
            msg = f"onehot names columns that are not in X: {unknown}"
            raise ValueError(msg)
        self.encoders_ = {}
        blocks: list[F64] = []
        for name, col in cols:
            raw = _subset(col, rows)
            if str(name) in self.onehot:
                enc = OneHot(name=str(name))
                block, _ = enc.fit_transform(raw)
                self.encoders_[str(name)] = enc
                blocks.append(block)
            else:
                blocks.append(to_vector(raw, str(name))[:, None])
        return np.ascontiguousarray(np.column_stack(blocks))

    def transform(self, x: ArrayLike) -> F64:
        """Encode new rows with the fitted encoders."""
        cols = columns(x)
        if cols is None or [str(n) for n, _ in cols] != self.columns_:
            msg = f"X must be a DataFrame with the fitted columns {self.columns_}"
            raise ValueError(msg)
        blocks = [
            self.encoders_[str(n)].transform(c)[0]
            if str(n) in self.encoders_
            else to_vector(c, str(n))[:, None]
            for n, c in cols
        ]
        return np.ascontiguousarray(np.column_stack(blocks))


def _subset(col: Any, rows: np.ndarray | None) -> Any:
    if rows is None:
        return col
    arr = np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col)
    return arr[rows]


@dataclass
class GlumPoisson:
    """glum's GeneralizedLinearRegressor (Poisson, log link, alpha=0) behind the protocol."""

    onehot: list[str]
    design_: FossDesign = field(init=False, repr=False)
    model_: Any = field(init=False, repr=False)

    def fit(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        fold: Fold | None = None,
    ) -> GlumPoisson:
        """Fit on the fold's training rows with glum's own offset support."""
        try:
            from glum import GeneralizedLinearRegressor  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover
            msg = "the glum adapter needs glum: uv add --group dev glum"
            raise ImportError(msg) from err
        rows = None if fold is None else fold.train_idx
        design = FossDesign(onehot=self.onehot)
        matrix = design.fit_transform(X, rows)
        yy = to_vector(y, "y")
        yy = yy if rows is None else yy[rows]
        w = None if sample_weight is None else _pick(sample_weight, rows)
        o = None if offset is None else _pick(offset, rows)
        self.design_ = design
        self.model_ = GeneralizedLinearRegressor(
            family="poisson", alpha=0.0, gradient_tol=1e-8
        ).fit(matrix, yy, sample_weight=w, offset=o)
        return self

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Predict the mean, offset included."""
        o = None if offset is None else to_vector(offset, "offset")
        out = self.model_.predict(self.design_.transform(X), offset=o)
        return np.asarray(out, dtype=np.float64)


@dataclass
class SklearnPoisson:
    """scikit-learn's PoissonRegressor (alpha=0) behind the protocol.

    It has no offset, so it is fitted on rates with exposure as the sample weight; predictions
    are rescaled back to the mean by ``exp(offset)``.
    """

    onehot: list[str]
    design_: FossDesign = field(init=False, repr=False)
    model_: Any = field(init=False, repr=False)

    def fit(
        self,
        X: ArrayLike,  # noqa: N803
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        fold: Fold | None = None,
    ) -> SklearnPoisson:
        """Fit on the fold's training rows in the rate-and-weight form."""
        try:
            from sklearn.linear_model import PoissonRegressor  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover
            msg = "the sklearn adapter needs scikit-learn: uv add --group dev scikit-learn"
            raise ImportError(msg) from err
        rows = None if fold is None else fold.train_idx
        design = FossDesign(onehot=self.onehot)
        matrix = design.fit_transform(X, rows)
        yy = to_vector(y, "y")
        yy = yy if rows is None else yy[rows]
        w = None if sample_weight is None else _pick(sample_weight, rows)
        if offset is not None:
            expo = np.exp(_pick(offset, rows))
            yy = yy / expo
            w = expo if w is None else w * expo
        self.design_ = design
        self.model_ = PoissonRegressor(alpha=0.0, tol=1e-8, max_iter=1000).fit(
            matrix, yy, sample_weight=w
        )
        return self

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Predict the mean: the fitted rate times ``exp(offset)``."""
        rate = np.asarray(self.model_.predict(self.design_.transform(X)), dtype=np.float64)
        if offset is not None:
            rate = rate * np.exp(to_vector(offset, "offset"))
        return rate


def _pick(values: ArrayLike, rows: np.ndarray | None) -> F64:
    v = to_vector(values, "values")
    return v if rows is None else v[rows]


__all__ = ["FossDesign", "GlumPoisson", "SklearnPoisson"]
