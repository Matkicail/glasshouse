"""Encoders: fit on the training rows, transform anything, never let a row see its own ``y``.

Three of them, all the same shape — ``fit(x, y=None, sample_weight=None)``, then
``transform(x) -> (matrix, names)``:

- :class:`OneHot` — one 0/1 column per level, reference level dropped. Levels are learned on
  the training rows; an unseen level at transform time is refused by name.
- :class:`TargetEncode` — one column: the smoothed mean of ``y`` per level. The rows it was
  fitted on get **out-of-fold** values (or **cumulative, past-only** values when the data is
  time-ordered), so the model never trains on an encoding that contains its own outcome.
- :class:`Standardize` — ``(x - mean) / std`` with weighted moments from the training rows.

Leakage is a property of the split, not the transform: these only do the right thing if
what you pass to ``fit`` is the training fold. The GLM's ``fold=`` does that for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glasshouse import _core
from glasshouse.arrays import F64, ArrayLike, to_vector


def _labels(x: ArrayLike, name: str) -> np.ndarray:
    """Return any column as an array of string labels."""
    arr = np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x)
    if arr.ndim != 1:
        msg = f"{name} must be one column, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        n = int(np.isnan(arr).sum())
        msg = f"{name} has {n} missing value(s): give missing its own level (e.g. 'unknown') first"
        raise ValueError(msg)
    if arr.dtype.kind == "O" and any(_missing(v) for v in arr):
        n = sum(_missing(v) for v in arr)
        msg = f"{name} has {n} missing value(s): give missing its own level (e.g. 'unknown') first"
        raise ValueError(msg)
    return arr.astype(str)


def _missing(v: object) -> bool:
    """None, or a float NaN hiding in an object column (pandas' missing string)."""
    return v is None or (isinstance(v, float) and np.isnan(v))


@dataclass
class OneHot:
    """One 0/1 column per level of a categorical, with the reference level dropped.

    Parameters
    ----------
    drop_first : bool
        Drop the first level (alphabetically) so the columns are not collinear with an
        intercept. Keep all levels only if you fit without one.
    unknown : {"error", "zero"}
        What to do with a level at ``transform`` that was not seen at ``fit``: refuse (the
        default — a new level is news you want), or encode as all zeros (the reference).

    Examples
    --------
    >>> from glasshouse.encoders import OneHot
    >>> enc = OneHot().fit(["b", "a", "c", "a"])
    >>> m, names = enc.transform(["a", "c"])
    >>> names, m.tolist()
    (['x=b', 'x=c'], [[0.0, 0.0], [0.0, 1.0]])
    """

    drop_first: bool = True
    unknown: str = "error"
    name: str = "x"
    levels_: list[str] = field(default_factory=list, repr=False)

    def fit(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> OneHot:
        """Learn the levels from the training rows (sorted, so the reference is stable)."""
        _ = y, sample_weight
        self.levels_ = sorted(set(_labels(x, self.name).tolist()))
        if len(self.levels_) < 2:  # noqa: PLR2004
            msg = f"{self.name} has a single level {self.levels_}: nothing to encode; drop it"
            raise ValueError(msg)
        return self

    def transform(self, x: ArrayLike) -> tuple[F64, list[str]]:
        """0/1 columns for the kept levels, in level order."""
        labels = _labels(x, self.name)
        kept = self.levels_[1:] if self.drop_first else self.levels_
        unseen = sorted(set(labels.tolist()) - set(self.levels_))
        if unseen and self.unknown == "error":
            msg = (
                f"{self.name} has {len(unseen)} level(s) not seen at fit: {unseen[:5]} — "
                "fit on data that has them, or use unknown='zero' to encode them as the reference"
            )
            raise ValueError(msg)
        out = np.zeros((len(labels), len(kept)), dtype=np.float64)
        for j, level in enumerate(kept):
            out[:, j] = labels == level
        return out, [f"{self.name}={level}" for level in kept]

    def fit_transform(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> tuple[F64, list[str]]:
        """``fit`` then ``transform`` on the same rows."""
        return self.fit(x, y, sample_weight).transform(x)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready state."""
        return {
            "kind": "onehot",
            "name": self.name,
            "drop_first": self.drop_first,
            "unknown": self.unknown,
            "levels": list(self.levels_),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OneHot:
        """Rebuild from :meth:`to_dict`."""
        enc = cls(drop_first=d["drop_first"], unknown=d["unknown"], name=d["name"])
        enc.levels_ = list(d["levels"])
        return enc


@dataclass
class TargetEncode:
    """One column: the smoothed mean of ``y`` per level, ``(sum wy + m·prior) / (sum w + m)``.

    ``fit`` stores two things: the full-training-data table used by ``transform`` on *new*
    rows, and ``training_encoding_`` — the values the *training* rows themselves get, which
    never include the row's own ``y``:

    - ``cumulative=False`` (default): out-of-fold — each training row is encoded from the
      other ``n_inner - 1`` inner folds.
    - ``cumulative=True``: past-only — row ``i`` is encoded from rows strictly before it in
      the order given, with the *running* mean of the past as the prior. A row with no past
      at all (the first in time) is encoded as 0: no information — never the global mean,
      which would contain the future. Use this when the training rows are in time order (a
      ``"time"`` fold guarantees that); it is the only honest encoding for time data.

    Parameters
    ----------
    smoothing : float
        ``m``, the prior's weight in units of exposure. 10 means "a level needs ~10 units of
        weight before its own mean starts to dominate the global one".
    cumulative : bool
        Past-only encoding of the training rows (see above).
    n_inner, seed : int
        The inner folds for out-of-fold encoding.

    Examples
    --------
    >>> from glasshouse.encoders import TargetEncode
    >>> enc = TargetEncode(smoothing=1.0).fit(["a", "a", "b", "b"], [1, 3, 10, 12])
    >>> m, names = enc.transform(["a", "b", "new"])
    >>> names, m.round(3).ravel().tolist()  # 'new' gets the prior (global mean 6.5)
    (['x_te'], [3.5, 9.5, 6.5])
    """

    smoothing: float = 10.0
    cumulative: bool = False
    n_inner: int = 5
    seed: int = 0
    name: str = "x"
    prior_: float = field(default=float("nan"), repr=False)
    table_: dict[str, float] = field(default_factory=dict, repr=False)
    training_encoding_: F64 = field(default_factory=lambda: np.empty(0), repr=False)

    def fit(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> TargetEncode:
        """Learn the per-level table on the training rows and their leakage-free encodings."""
        if y is None:
            msg = f"{self.name}: target encoding needs y"
            raise ValueError(msg)
        labels = _labels(x, self.name)
        yy = to_vector(y, "y")
        w = np.ones(len(yy)) if sample_weight is None else to_vector(sample_weight, "sample_weight")
        if len(labels) != len(yy) or len(w) != len(yy):
            msg = f"{self.name}, y and sample_weight must have the same length"
            raise ValueError(msg)
        self.prior_ = float(np.sum(w * yy) / np.sum(w))
        self.table_ = self._table(labels, yy, w, self.prior_)
        if self.cumulative:
            self.training_encoding_ = self._cumulative(labels, yy, w)
        else:
            self.training_encoding_ = self._out_of_fold(labels, yy, w)
        return self

    def transform(self, x: ArrayLike) -> tuple[F64, list[str]]:
        """Encode new rows from the full training table; unseen levels get the prior."""
        labels = _labels(x, self.name)
        out = np.array([self.table_.get(v, self.prior_) for v in labels], dtype=np.float64)
        return out[:, None], [f"{self.name}_te"]

    def fit_transform(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> tuple[F64, list[str]]:
        """``fit``, then return the *training* (out-of-fold / cumulative) encodings."""
        self.fit(x, y, sample_weight)
        return self.training_encoding_[:, None], [f"{self.name}_te"]

    def _table(self, labels: np.ndarray, y: F64, w: F64, prior: float) -> dict[str, float]:
        levels, codes = np.unique(labels, return_inverse=True)
        sw = np.bincount(codes, weights=w, minlength=len(levels))
        swy = np.bincount(codes, weights=w * y, minlength=len(levels))
        values = (swy + self.smoothing * prior) / (sw + self.smoothing)
        return dict(zip(levels.tolist(), values.tolist(), strict=True))

    def _out_of_fold(self, labels: np.ndarray, y: F64, w: F64) -> F64:
        n = len(y)
        k = min(self.n_inner, n)
        if k < 2:  # noqa: PLR2004
            return np.full(n, self.prior_)
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(n) % k
        out = np.empty(n, dtype=np.float64)
        for i in range(k):
            train, held = fold != i, fold == i
            prior = float(np.sum(w[train] * y[train]) / np.sum(w[train]))
            table = self._table(labels[train], y[train], w[train], prior)
            out[held] = [table.get(v, prior) for v in labels[held]]
        return out

    def _cumulative(self, labels: np.ndarray, y: F64, w: F64) -> F64:
        """Row i sees rows 0..i-1 only. Ties in time are the caller's problem to order."""
        out = np.empty(len(y), dtype=np.float64)
        sw: dict[str, float] = {}
        swy: dict[str, float] = {}
        tot_w = 0.0
        tot_wy = 0.0
        for i, (label, yi, wi) in enumerate(zip(labels.tolist(), y, w, strict=True)):
            prior = tot_wy / tot_w if tot_w > 0 else 0.0  # no past: no information
            out[i] = (swy.get(label, 0.0) + self.smoothing * prior) / (
                sw.get(label, 0.0) + self.smoothing
            )
            sw[label] = sw.get(label, 0.0) + wi
            swy[label] = swy.get(label, 0.0) + wi * yi
            tot_w += wi
            tot_wy += wi * yi
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready state (the training encodings are not needed to predict)."""
        return {
            "kind": "target",
            "name": self.name,
            "smoothing": self.smoothing,
            "cumulative": self.cumulative,
            "n_inner": self.n_inner,
            "seed": self.seed,
            "prior": self.prior_,
            "table": dict(self.table_),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetEncode:
        """Rebuild from :meth:`to_dict`."""
        enc = cls(
            smoothing=d["smoothing"],
            cumulative=d["cumulative"],
            n_inner=d["n_inner"],
            seed=d["seed"],
            name=d["name"],
        )
        enc.prior_ = float(d["prior"])
        enc.table_ = {str(k): float(v) for k, v in d["table"].items()}
        return enc


@dataclass
class Standardize:
    """``(x - mean) / std`` with weighted moments learned on the training rows.

    A GLM does not need it (it is scale-invariant), but penalised fits and anything with a
    learning rate do; and it makes coefficients comparable across features.

    Examples
    --------
    >>> from glasshouse.encoders import Standardize
    >>> m, names = Standardize().fit_transform([1.0, 2.0, 3.0])
    >>> names, m.ravel().round(4).tolist()
    (['x_std'], [-1.2247, 0.0, 1.2247])
    """

    name: str = "x"
    mean_: float = field(default=float("nan"), repr=False)
    std_: float = field(default=float("nan"), repr=False)

    def fit(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> Standardize:
        """Weighted mean and (population) standard deviation of the training rows."""
        _ = y
        v = to_vector(x, self.name)
        w = None if sample_weight is None else to_vector(sample_weight, "sample_weight")
        self.mean_ = float(np.average(v, weights=w))
        self.std_ = float(np.sqrt(np.average((v - self.mean_) ** 2, weights=w)))
        if self.std_ == 0.0:
            msg = f"{self.name} is constant on the training rows: nothing to scale; drop it"
            raise ValueError(msg)
        return self

    def transform(self, x: ArrayLike) -> tuple[F64, list[str]]:
        """Scale with the stored moments."""
        v = to_vector(x, self.name)
        return ((v - self.mean_) / self.std_)[:, None], [f"{self.name}_std"]

    def fit_transform(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> tuple[F64, list[str]]:
        """``fit`` then ``transform`` on the same rows."""
        return self.fit(x, y, sample_weight).transform(x)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready state."""
        return {"kind": "standardize", "name": self.name, "mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Standardize:
        """Rebuild from :meth:`to_dict`."""
        enc = cls(name=d["name"])
        enc.mean_, enc.std_ = float(d["mean"]), float(d["std"])
        return enc


@dataclass
class BSpline:
    """A cubic B-spline expansion of one numeric column: the GLM's smooth term.

    ``df`` columns come out (the first basis function is dropped, R's ``bs()`` convention, so
    the expansion does not fight the intercept). Interior knots sit at quantiles of the
    *training* rows; tied quantiles collapse, so heavily tied data yields fewer columns than
    ``df``. At transform time values outside the training range are clamped to the boundary —
    a polynomial tail extrapolated silently is how spline models go wrong quietly.

    Examples
    --------
    >>> import numpy as np
    >>> from glasshouse.encoders import BSpline
    >>> m, names = BSpline(df=4, name="age").fit_transform(np.linspace(0.0, 1.0, 9))
    >>> m.shape, names[0]
    ((9, 4), 'age_bs1')
    """

    df: int = 6
    degree: int = 3
    name: str = "x"
    knots_: list[float] = field(default_factory=list, repr=False)

    def fit(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> BSpline:
        """Place interior knots at quantiles of the training rows."""
        _ = y, sample_weight
        if self.df <= self.degree:
            msg = (
                f"{self.name}: df must exceed the degree ({self.degree}); df=6 is the usual choice"
            )
            raise ValueError(msg)
        v = to_vector(x, self.name)
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            msg = f"{self.name} is constant on the training rows: nothing to spline; drop it"
            raise ValueError(msg)
        n_interior = self.df - self.degree
        qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        # heavily tied data (a value holding half the column) makes quantiles coincide with
        # each other or with a boundary; duplicate or boundary knots would create degenerate
        # basis columns and a rank-deficient design, so they collapse — you get fewer columns
        # than df, which is the honest answer for data this tied
        interior = sorted({float(q) for q in np.quantile(v, qs)} - {lo, hi})
        self.knots_ = [lo] * (self.degree + 1) + interior + [hi] * (self.degree + 1)
        return self

    def transform(self, x: ArrayLike) -> tuple[F64, list[str]]:
        """Evaluate the basis (Rust), drop the first column, name the rest."""
        v = to_vector(x, self.name)
        flat, p = _core.bspline_design(v, self.knots_, self.degree)
        design = np.asarray(flat, dtype=np.float64).reshape(len(v), p)[:, 1:]
        return np.ascontiguousarray(design), [f"{self.name}_bs{i}" for i in range(1, p)]

    def fit_transform(
        self, x: ArrayLike, y: ArrayLike | None = None, sample_weight: ArrayLike | None = None
    ) -> tuple[F64, list[str]]:
        """``fit`` then ``transform`` on the same rows."""
        return self.fit(x, y, sample_weight).transform(x)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready state."""
        return {
            "kind": "spline",
            "name": self.name,
            "df": self.df,
            "degree": self.degree,
            "knots": list(self.knots_),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BSpline:
        """Rebuild from :meth:`to_dict`."""
        enc = cls(df=d["df"], degree=d["degree"], name=d["name"])
        enc.knots_ = [float(k) for k in d["knots"]]
        return enc


Encoder = OneHot | TargetEncode | Standardize | BSpline

_KINDS: dict[str, type[Encoder]] = {
    "onehot": OneHot,
    "target": TargetEncode,
    "standardize": Standardize,
    "spline": BSpline,
}


def make(kind: str, name: str, **options: Any) -> Encoder:
    """Build an encoder from a term spec string — the one registry.

    Examples
    --------
    >>> from glasshouse.encoders import make
    >>> make("onehot", "Region").name
    'Region'
    """
    if kind not in _KINDS:
        msg = f"unknown term {kind!r} for {name}: one of {sorted(_KINDS)}, or 'linear'"
        raise ValueError(msg)
    return _KINDS[kind](name=name, **options)


def from_dict(d: dict[str, Any]) -> Encoder:
    """Rebuild any encoder from its ``to_dict``."""
    return _KINDS[d["kind"]].from_dict(d)


__all__ = ["BSpline", "Encoder", "OneHot", "Standardize", "TargetEncode", "from_dict", "make"]
