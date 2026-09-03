"""Gradient-boosted trees behind the bench's Model protocol.

A GBM is not a glass box, and that is fine: the job here is understanding and honest
comparison — the same folds, the same scorecard, the same curves as every other model, with
the double lift showing exactly where the trees disagree with the GLM and who was right.

Fairness and leakage choices, stated plainly:

- The offset enters as ``init_score`` (log exposure for a rate model), LightGBM's correct
  mechanism; predictions are ``exp(raw score + offset)``.
- Early stopping needs a validation set, so a seeded fraction of the *training fold only* is
  held out for it. Nothing outside the fold is ever seen.
- Categorical columns use LightGBM's native handling; levels are learned on the training
  rows, and a level unseen at fit becomes missing at predict (LightGBM handles missing),
  never a crash and never a borrowed level.
- Defaults are modest and named (2000 trees cap, learning rate 0.05, 31 leaves, stop after
  50 rounds without improvement); pass others if you disagree — they are printed in
  ``repr`` and land in the report's provenance either way.

Needs the dev dependencies (``lightgbm``, ``pandas``); import errors say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from glasshouse._rows import as_array, subset_column, subset_vector
from glasshouse.arrays import F64, ArrayLike, columns, to_vector
from glasshouse.metrics import FamilyName
from glasshouse.splits import Fold

_OBJECTIVE: dict[str, str] = {"poisson": "poisson", "gamma": "gamma", "tweedie": "tweedie"}


@dataclass
class LightGBM:
    """LightGBM with a GLM family's objective, offsets, and fold-safe early stopping."""

    family: FamilyName = "poisson"
    power: float | None = None
    categorical: list[str] = field(default_factory=list)
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 31
    early_stopping: int = 50
    valid_fraction: float = 0.2
    seed: int = 0
    model_: Any = field(init=False, repr=False)
    levels_: dict[str, list[str]] = field(init=False, repr=False, default_factory=dict)
    columns_: list[str] = field(init=False, repr=False, default_factory=list)
    best_iteration_: int | None = field(init=False, repr=False, default=None)

    def fit(
        self,
        X: ArrayLike,  # noqa: N803 — scikit-learn's spelling
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        fold: Fold | None = None,
    ) -> LightGBM:
        """Fit on the fold's training rows; early-stop on a seeded slice of those rows only."""
        lgb, pd = _imports()
        if self.family not in _OBJECTIVE:
            msg = f"family: LightGBM supports {sorted(_OBJECTIVE)}, not {self.family!r}"
            raise ValueError(msg)
        if self.family == "tweedie" and self.power is None:
            msg = "power: tweedie needs a variance power (1.5 is the usual choice)"
            raise ValueError(msg)
        rows = None if fold is None else fold.train_idx
        frame = self._fit_frame(pd, X, rows)
        yy = subset_vector(y, "y", rows)
        w = None if sample_weight is None else subset_vector(sample_weight, "sample_weight", rows)
        o = None if offset is None else subset_vector(offset, "offset", rows)

        rng = np.random.default_rng(self.seed)
        n = len(yy)
        n_valid = max(1, round(self.valid_fraction * n))
        perm = rng.permutation(n)
        valid, train = perm[:n_valid], perm[n_valid:]

        params: dict[str, Any] = {
            "objective": _OBJECTIVE[self.family],
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "random_state": self.seed,
            "verbose": -1,
        }
        if self.family == "tweedie":
            params["tweedie_variance_power"] = self.power
        model = lgb.LGBMRegressor(**params)
        model.fit(
            frame.iloc[train],
            yy[train],
            sample_weight=None if w is None else w[train],
            init_score=None if o is None else o[train],
            eval_set=[(frame.iloc[valid], yy[valid])],
            eval_sample_weight=None if w is None else [w[valid]],
            eval_init_score=None if o is None else [o[valid]],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )
        self.model_ = model
        self.best_iteration_ = model.best_iteration_
        return self

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Predict the mean: ``exp(raw score + offset)`` — the family's log link."""
        _, pd = _imports()
        frame = self._transform_frame(pd, X)
        raw = np.asarray(self.model_.predict(frame, raw_score=True), dtype=np.float64)
        if offset is not None:
            raw = raw + to_vector(offset, "offset")
        return np.exp(raw)

    # ------------------------------------------------------------------ frames

    def _fit_frame(self, pd: Any, x: ArrayLike, rows: np.ndarray | None) -> Any:
        cols = columns(x)
        if cols is None:
            msg = "LightGBM here needs a DataFrame with named columns"
            raise ValueError(msg)
        self.columns_ = [str(n) for n, _ in cols]
        unknown = sorted(set(self.categorical) - set(self.columns_))
        if unknown:
            msg = f"categorical names columns that are not in X: {unknown}"
            raise ValueError(msg)
        self.levels_ = {}
        data = {}
        for name, col in cols:
            raw = subset_column(col, rows)
            if str(name) in self.categorical:
                labels = np.asarray(raw).astype(str)
                levels = sorted(set(labels.tolist()))
                self.levels_[str(name)] = levels
                data[str(name)] = pd.Categorical(labels, categories=levels)
            else:
                data[str(name)] = to_vector(raw, str(name))
        return pd.DataFrame(data)

    def _transform_frame(self, pd: Any, x: ArrayLike) -> Any:
        cols = columns(x)
        if cols is None or [str(n) for n, _ in cols] != self.columns_:
            msg = f"X must be a DataFrame with the fitted columns {self.columns_}"
            raise ValueError(msg)
        data = {}
        for name, col in cols:
            if str(name) in self.levels_:
                labels = as_array(col).astype(str)
                # unseen levels become missing, which LightGBM handles natively
                data[str(name)] = pd.Categorical(labels, categories=self.levels_[str(name)])
            else:
                data[str(name)] = to_vector(col, str(name))
        return pd.DataFrame(data)


def _imports() -> tuple[Any, Any]:
    try:
        import lightgbm as lgb  # noqa: PLC0415 — dev dependency
        import pandas as pd  # noqa: PLC0415
    except (ImportError, OSError) as err:  # pragma: no cover — OSError: libomp missing on macOS
        msg = (
            "the LightGBM adapter needs lightgbm and pandas (uv add --group dev lightgbm pandas); "
            "on macOS lightgbm also needs OpenMP: brew install libomp"
        )
        raise ImportError(msg) from err
    return lgb, pd


__all__ = ["LightGBM"]
