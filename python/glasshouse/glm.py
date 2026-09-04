"""The GLM: the honest baseline every other model has to beat.

scikit-learn shaped — ``GLM(family=...).fit(X, y, sample_weight=..., offset=...)`` then
``predict`` — with the things a GLM is *for* attached: standard errors, deviance against the
null model, per-row contributions, and a fit trace that explains how it got there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse import _core, encoders
from glasshouse._rows import subset_column
from glasshouse.arrays import F64, ArrayLike, columns, to_matrix, to_vector
from glasshouse.metrics import FamilyName
from glasshouse.splits import Fold

LinkName = str  # "identity" | "log" | "logit"; None means the family's canonical link
_CANONICAL: dict[str, str] = {
    "gaussian": "identity",
    "poisson": "log",
    "gamma": "log",
    "tweedie": "log",
    "binomial": "logit",
}


@dataclass(frozen=True)
class FitTrace:
    """One row per IRLS iteration: what the solver did and why it stopped.

    ``stop`` is ``"converged"`` (relative deviance change below ``tol``), ``"max_iter"``
    (still moving), or ``"no_improvement"`` (no step of any size lowered the deviance —
    already at the optimum, or a separated / ill-posed problem).
    """

    iteration: npt.NDArray[np.int64]
    deviance: F64
    halvings: npt.NDArray[np.int64]
    max_step: F64
    stop: str

    def __str__(self) -> str:
        """Fixed-width table of the iterations."""
        lines = [f"{'iter':>5}{'deviance':>18}{'halvings':>10}{'max_step':>12}"]
        for i, d, h, s in zip(
            self.iteration, self.deviance, self.halvings, self.max_step, strict=True
        ):
            lines.append(f"{i:>5}{d:>18.8g}{h:>10}{s:>12.3g}")
        lines.append(f"stopped: {self.stop}")
        return "\n".join(lines)


def _gcv_minimise(
    fit_at: Callable[[float], tuple[float, float]],
    n_rows: int,
    trace: list[tuple[float, float, float]],
) -> float:
    """1-D GCV minimisation: a coarse log-spaced grid, then a finer pass around the winner.

    GCV is ``n * deviance / (n - edf)^2`` — mgcv's criterion with gamma = 1. The numerator
    rewards fit; the shrinking denominator charges for every effective coefficient spent.
    Each evaluation is appended to ``trace`` as ``(lambda, gcv, edf)``.
    """

    def sweep(grid: F64) -> float:
        best_lam, best = float(grid[0]), float("inf")
        for lam in grid:
            deviance, edf = fit_at(float(lam))
            gcv = n_rows * deviance / (n_rows - edf) ** 2
            trace.append((float(lam), float(gcv), float(edf)))
            if gcv < best:
                best_lam, best = float(lam), float(gcv)
        return best_lam

    coarse = sweep(np.logspace(-4.0, 7.0, 23))
    return sweep(np.logspace(np.log10(coarse) - 0.5, np.log10(coarse) + 0.5, 9))


@dataclass
class GLM:
    """Generalised linear model fitted by IRLS in Rust.

    Parameters
    ----------
    family : {"gaussian", "poisson", "gamma", "tweedie", "binomial"}
        The distribution of ``y``. Picks the deviance, the variance function, and the
        default link.
    link : {"identity", "log", "logit"}, optional
        ``None`` means the family's canonical link (log for counts and amounts, logit for
        probabilities, identity for gaussian). A canonical link makes the fit balanced —
        total predicted equals total actual on the training data.
    power : float, optional
        Tweedie variance power (``1 < power < 2`` for pure premium). Required for tweedie.
    terms : dict, optional
        How to treat named columns of a frame: ``"onehot"``, ``"target"``, ``"spline"``,
        ``"smooth"``, ``"standardize"`` or ``"linear"`` (the default for any column not
        listed) — or a configured encoder instance, e.g. ``{"age": BSpline(df=8)}``. A
        ``"smooth"`` is a penalised spline whose wiggliness is chosen by GCV during ``fit``
        (pin it with ``Smooth(lam=...)``). Encoders are fitted on the training rows only;
        with a time-ordered ``fold`` target encoding is cumulative (past-only) automatically.
        Columns not in a frame cannot have terms.
    fit_intercept : bool
        Prepend a column of ones. Turn off only if your design already has one.
    max_iter, tol : int, float
        IRLS stopping rules. ``tol`` is the relative change in deviance.

    Attributes (after ``fit``)
    --------------------------
    coef_, intercept_, se_, cov_ : the estimates, their standard errors and covariance.
    feature_names_in_ : column names from the frame, or ``x0, x1, ...``.
    deviance_, null_deviance_, dispersion_ : the fit statistics (total, not mean, deviance).
    edf_, lambda_, gcv_ : what the smooths spent, the chosen penalties, the searched grids.
    n_iter_, converged_, trace_ : how the solver got there.

    Examples
    --------
    >>> import numpy as np
    >>> from glasshouse import GLM
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    >>> m = GLM(family="gaussian").fit(X, y)
    >>> np.round(m.intercept_, 6), np.round(m.coef_, 6)
    (np.float64(1.0), array([2.]))
    """

    family: FamilyName
    link: LinkName | None = None
    power: float | None = None
    terms: dict[str, str | encoders.Encoder] | None = None
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-10
    _fit: dict[str, Any] = field(default_factory=dict, repr=False)
    feature_names_in_: list[str] = field(default_factory=list, repr=False)
    encoders_: dict[str, encoders.Encoder] = field(default_factory=dict, repr=False)
    input_columns_: list[str] = field(default_factory=list, repr=False)
    lambda_: dict[str, float] = field(default_factory=dict, repr=False)
    gcv_: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict, repr=False)
    _slices: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        X: ArrayLike,  # noqa: N803 — scikit-learn's spelling
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        fold: Fold | None = None,
    ) -> GLM:
        """Fit by IRLS. Returns ``self`` so calls chain.

        Parameters
        ----------
        X : frame or 2-D array
            Features. Categorical columns need a ``terms`` entry (``"onehot"`` or
            ``"target"``); the data door refuses them otherwise.
        y : 1-D
            Outcome inside the family's support.
        sample_weight : 1-D, optional
            Prior weights (claim counts for a severity model, exposure for a rate model).
        offset : 1-D, optional
            Added to the linear predictor, on the link scale — ``log(exposure)`` for a
            Poisson count model, not the exposure itself.
        fold : Fold, optional
            Fit on ``fold.train_idx`` only. Encoders are fitted on those rows, and if
            ``fold.kind == "time"`` target encoding is cumulative (past-only). Pass the whole
            data and the fold; do not subset by hand.
        """
        yy = to_vector(y, "y")
        w = None if sample_weight is None else to_vector(sample_weight, "sample_weight")
        o = None if offset is None else to_vector(offset, "offset")
        rows = None if fold is None else fold.train_idx
        cumulative = fold is not None and fold.kind == "time"
        matrix, names = self._design_fit(X, yy, w, rows, cumulative=cumulative)
        if rows is not None:
            yy = yy[rows]
            w = None if w is None else w[rows]
            o = None if o is None else o[rows]
        if self.fit_intercept:
            matrix = np.ascontiguousarray(np.column_stack([np.ones(matrix.shape[0]), matrix]))
            names = ["intercept", *names]
        penalty = self._smooth_penalty(matrix, yy, w, o)
        result = _core.glm_fit(
            self.family,
            self._link_name(),
            matrix,
            yy,
            w,
            o,
            self.power,
            self.max_iter,
            self.tol,
            penalty=penalty,
        )
        self._fit = result
        self.feature_names_in_ = names
        return self

    # ------------------------------------------------------------------ design matrices

    def _design_fit(
        self,
        X: ArrayLike,  # noqa: N803
        y: F64,
        w: F64 | None,
        rows: npt.NDArray[np.int64] | None,
        *,
        cumulative: bool,
    ) -> tuple[F64, list[str]]:
        """Build the training design: fit encoders on the training rows, encode them."""
        terms = self.terms or {}
        cols = columns(X)
        if cols is None:
            if terms:
                msg = "terms need named columns: pass a DataFrame, not a bare array"
                raise ValueError(msg)
            matrix, plain_names = to_matrix(X)
            self.input_columns_ = plain_names
            self.encoders_ = {}
            self._slices = {}
            return (matrix if rows is None else matrix[rows]), plain_names
        unknown = sorted(set(terms) - {str(n) for n, _ in cols})
        if unknown:
            msg = f"terms name columns that are not in X: {unknown}"
            raise ValueError(msg)
        self.input_columns_ = [str(n) for n, _ in cols]
        self.encoders_ = {}
        y_train = y if rows is None else y[rows]
        w_train = w if (w is None or rows is None) else w[rows]
        blocks: list[F64] = []
        names: list[str] = []
        self._slices = {}
        for col_name, col in cols:
            block, block_names = self._fit_term(
                str(col_name),
                subset_column(col, rows),
                y_train,
                w_train,
                terms,
                cumulative=cumulative,
            )
            self._slices[str(col_name)] = (len(names), len(names) + len(block_names))
            blocks.append(block)
            names.extend(block_names)
        return np.ascontiguousarray(np.column_stack(blocks)), names

    def _fit_term(
        self,
        name: str,
        raw: Any,
        y_train: F64,
        w_train: F64 | None,
        terms: dict[str, str | encoders.Encoder],
        *,
        cumulative: bool,
    ) -> tuple[F64, list[str]]:
        """One column of the training design: linear passthrough or a fitted encoder."""
        kind = terms.get(name, "linear")
        if kind == "linear":
            return to_vector(raw, name)[:, None], [name]
        if isinstance(kind, str):
            options = {"cumulative": True} if (kind == "target" and cumulative) else {}
            enc = encoders.make(kind, name, **options)
        else:  # a configured encoder instance, e.g. BSpline(df=8); fitted here, on train rows
            enc = kind
            enc.name = name
            if cumulative and isinstance(enc, encoders.TargetEncode):
                enc.cumulative = True
        block, block_names = enc.fit_transform(raw, y_train, w_train)
        self.encoders_[name] = enc
        return block, block_names

    # ------------------------------------------------------------------ smoothing

    def _smooth_penalty(self, matrix: F64, y: F64, w: F64 | None, o: F64 | None) -> F64 | None:
        """Build the combined penalty for the smooth terms (``None`` when there are none).

        Each :class:`~glasshouse.encoders.Smooth` term contributes its second-difference
        penalty, embedded at its columns of the design and scaled by its own lambda — pinned
        by ``Smooth(lam=...)``, otherwise chosen by GCV. With several free smooths the 1-D
        searches sweep twice (coordinate descent). Every (lambda, GCV, edf) evaluated lands
        in ``gcv_``, so the choice can be read, not re-run.
        """
        smooths = {n: e for n, e in self.encoders_.items() if isinstance(e, encoders.Smooth)}
        if not smooths:
            return None
        bases = self._penalty_bases(smooths, matrix.shape[1])

        def combined(lambdas: dict[str, float]) -> F64:
            total = np.zeros((matrix.shape[1], matrix.shape[1]))
            for name, base in bases.items():
                total += lambdas[name] * base
            return np.ascontiguousarray(total)

        def fit_at(lambdas: dict[str, float]) -> tuple[float, float]:
            r = _core.glm_fit(
                self.family,
                self._link_name(),
                matrix,
                y,
                w,
                o,
                self.power,
                self.max_iter,
                self.tol,
                penalty=combined(lambdas),
            )
            return float(r["deviance"]), float(r["edf"])

        lambdas = {n: (e.lam if e.lam is not None else 1.0) for n, e in smooths.items()}
        free = [n for n, e in smooths.items() if e.lam is None]
        self.gcv_ = {n: [] for n in free}
        for _ in range(2 if len(free) > 1 else 1):
            for name in free:
                # `lambdas` is shared state on purpose: each 1-D search sees the others'
                # current values, which is what makes the sweeps coordinate descent
                def fit_at_lam(lam: float, _name: str = name) -> tuple[float, float]:
                    return fit_at({**lambdas, _name: lam})

                lambdas[name] = _gcv_minimise(fit_at_lam, matrix.shape[0], self.gcv_[name])
        self.lambda_ = {n: float(v) for n, v in lambdas.items()}
        return combined(self.lambda_)

    def _penalty_bases(self, smooths: dict[str, encoders.Smooth], p: int) -> dict[str, F64]:
        """Embed each smooth's penalty at its own columns of the full design."""
        shift = 1 if self.fit_intercept else 0
        bases: dict[str, F64] = {}
        for name, enc in smooths.items():
            lo, hi = self._slices[name]
            base = np.zeros((p, p))
            base[lo + shift : hi + shift, lo + shift : hi + shift] = enc.penalty_matrix()
            bases[name] = base
        return bases

    def _design_predict(self, X: ArrayLike) -> F64:  # noqa: N803
        """Build a prediction design with the fitted encoders; check the columns line up."""
        cols = columns(X)
        if cols is None:
            if self.encoders_:
                msg = "this model was fitted with terms on a DataFrame: predict needs one too"
                raise ValueError(msg)
            matrix, _ = to_matrix(X)
            if matrix.shape[1] != len(self.input_columns_):
                msg = (
                    f"X has {matrix.shape[1]} columns but the model was fitted on "
                    f"{len(self.input_columns_)}"
                )
                raise ValueError(msg)
            return self._with_intercept(matrix)
        have = [str(n) for n, _ in cols]
        if have != self.input_columns_:
            msg = f"columns {have} do not match the columns fitted on {self.input_columns_}"
            raise ValueError(msg)
        blocks: list[F64] = []
        for col_name, col in cols:
            enc = self.encoders_.get(str(col_name))
            blocks.append(
                to_vector(col, str(col_name))[:, None] if enc is None else enc.transform(col)[0]
            )
        return self._with_intercept(np.column_stack(blocks))

    def _with_intercept(self, matrix: F64) -> F64:
        if self.fit_intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        return np.ascontiguousarray(matrix)

    def _link_name(self) -> str:
        return self.link if self.link is not None else _CANONICAL[self.family]

    def _require_fit(self) -> dict[str, Any]:
        if not self._fit:
            msg = "this GLM is not fitted yet: call fit(X, y) first"
            raise ValueError(msg)
        return self._fit

    # ------------------------------------------------------------------ estimates

    @property
    def coef_(self) -> F64:
        """Coefficients of the features (the intercept is separate)."""
        coef = np.asarray(self._require_fit()["coef"], dtype=np.float64)
        return coef[1:] if self.fit_intercept else coef

    @property
    def intercept_(self) -> float:
        """The intercept, or 0.0 if none was fitted."""
        coef = np.asarray(self._require_fit()["coef"], dtype=np.float64)
        return float(coef[0]) if self.fit_intercept else 0.0

    @property
    def cov_(self) -> F64:
        """Covariance of all coefficients (intercept first), ``dispersion * (X'WX)^-1``."""
        p = self._require_fit()["n_features"]
        return np.asarray(self._fit["cov"], dtype=np.float64).reshape(p, p)

    @property
    def se_(self) -> F64:
        """Standard errors of all coefficients, intercept first."""
        return np.sqrt(np.diag(self.cov_))

    @property
    def cov_robust_(self) -> F64:
        """HC1 sandwich covariance: robust to a wrong variance function (over-dispersion)."""
        p = self._require_fit()["n_features"]
        return np.asarray(self._fit["cov_robust"], dtype=np.float64).reshape(p, p)

    @property
    def se_robust_(self) -> F64:
        """HC1 robust standard errors, intercept first.

        Use these when you doubt the variance assumption — a Poisson model on over-dispersed
        counts, say. If they are much larger than ``se_``, the model's uncertainty is
        understated and the family (or a quasi-family) deserves a second look.
        """
        return np.sqrt(np.diag(self.cov_robust_))

    @property
    def deviance_(self) -> float:
        """Total weighted deviance of the fit (lower is better; 0 is perfect)."""
        return float(self._require_fit()["deviance"])

    @property
    def null_deviance_(self) -> float:
        """Deviance of the intercept-only model with the same offset and weights."""
        return float(self._require_fit()["null_deviance"])

    @property
    def dispersion_(self) -> float:
        """Pearson dispersion estimate, or exactly 1 for poisson and binomial."""
        return float(self._require_fit()["dispersion"])

    @property
    def edf_(self) -> float:
        """Effective degrees of freedom: ``tr((X'WX + S)^{-1} X'WX)``, what the fit spent.

        Equals the number of coefficients for a plain GLM. A smooth with 9 columns and an
        edf near 3 is behaving like a 3-parameter curve — the penalty declined the rest.
        """
        return float(self._require_fit()["edf"])

    @property
    def n_iter_(self) -> int:
        """IRLS iterations run."""
        return int(self._require_fit()["iterations"])

    @property
    def converged_(self) -> bool:
        """Did the relative deviance change fall below ``tol``."""
        return bool(self._require_fit()["stop"] == "converged")

    @property
    def trace_(self) -> FitTrace:
        """The iteration table: deviance, step-halvings and step size per iteration."""
        r = self._require_fit()
        return FitTrace(
            iteration=np.asarray(r["trace_iteration"], dtype=np.int64),
            deviance=np.asarray(r["trace_deviance"], dtype=np.float64),
            halvings=np.asarray(r["trace_halvings"], dtype=np.int64),
            max_step=np.asarray(r["trace_max_step"], dtype=np.float64),
            stop=str(r["stop"]),
        )

    # ------------------------------------------------------------------ prediction

    def predict_linear(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Return the linear predictor ``eta = X beta + offset`` (link scale)."""
        coef = np.asarray(self._require_fit()["coef"], dtype=np.float64)
        eta = self._design_predict(X) @ coef
        if offset is not None:
            eta = eta + to_vector(offset, "offset")
        return np.asarray(eta, dtype=np.float64)

    def predict(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Return the mean ``mu = g^{-1}(eta)`` on the response scale (probability for binomial)."""
        eta = self.predict_linear(X, offset)
        link = self._link_name()
        if link == "identity":
            return eta
        if link == "log":
            return np.exp(eta)
        return 1.0 / (1.0 + np.exp(-eta))

    def contributions(self, X: ArrayLike) -> tuple[F64, list[str]]:  # noqa: N803
        """Per-row, per-feature contributions ``beta_j * x_ij`` on the link scale, plus names.

        They add up (with the intercept) to ``predict_linear``; for a log link, ``exp`` of a
        contribution is that feature's multiplicative relativity for the row.
        """
        coef = np.asarray(self._require_fit()["coef"], dtype=np.float64)
        return self._design_predict(X) * coef, list(self.feature_names_in_)

    # ------------------------------------------------------------------ reporting

    def summary(self) -> str:
        """Coefficient table with standard errors, plus the fit statistics."""
        r = self._require_fit()
        coef = np.asarray(r["coef"], dtype=np.float64)
        se = self.se_
        lines = [
            f"GLM family={self.family} link={self._link_name()}  n={r['n_rows']}  "
            f"iterations={r['iterations']} ({r['stop']})",
            f"{'term':<24}{'coef':>14}{'se':>14}{'z':>10}{'se_robust':>14}",
        ]
        for name, b, s, sr in zip(self.feature_names_in_, coef, se, self.se_robust_, strict=True):
            z = b / s if s > 0 else float("nan")
            lines.append(f"{name:<24}{b:>14.6g}{s:>14.4g}{z:>10.3f}{sr:>14.4g}")
        lines.append(
            f"deviance={r['deviance']:.6g}  null_deviance={r['null_deviance']:.6g}  "
            f"dispersion={r['dispersion']:.6g}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        """Everything needed to predict again, as plain JSON-able values. Never pickle."""
        r = self._require_fit()
        return {
            "family": self.family,
            "link": self._link_name(),
            "power": self.power,
            "fit_intercept": self.fit_intercept,
            "feature_names_in": list(self.feature_names_in_),
            "input_columns": list(self.input_columns_),
            "terms": {
                k: (v if isinstance(v, str) else v.to_dict()) for k, v in (self.terms or {}).items()
            },
            "encoders": {k: v.to_dict() for k, v in self.encoders_.items()},
            "coef": list(map(float, r["coef"])),
            "cov": list(map(float, r["cov"])),
            "cov_robust": list(map(float, r["cov_robust"])),
            "n_features": int(r["n_features"]),
            "n_rows": int(r["n_rows"]),
            "deviance": float(r["deviance"]),
            "null_deviance": float(r["null_deviance"]),
            "dispersion": float(r["dispersion"]),
            "edf": float(r["edf"]),
            "lambda": {k: float(v) for k, v in self.lambda_.items()},
            "iterations": int(r["iterations"]),
            "stop": str(r["stop"]),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GLM:
        """Rebuild a fitted model from :meth:`to_dict`."""
        model = cls(
            family=payload["family"],
            link=payload["link"],
            power=payload["power"],
            fit_intercept=payload["fit_intercept"],
        )
        model.feature_names_in_ = list(payload["feature_names_in"])
        model.input_columns_ = list(payload["input_columns"])
        model.terms = {
            k: (v if isinstance(v, str) else encoders.from_dict(v))
            for k, v in payload["terms"].items()
        } or None
        model.encoders_ = {k: encoders.from_dict(v) for k, v in payload["encoders"].items()}
        model._fit = {
            key: payload[key]
            for key in (
                "coef",
                "cov",
                "cov_robust",
                "n_features",
                "n_rows",
                "deviance",
                "null_deviance",
                "dispersion",
                "iterations",
                "stop",
            )
        }
        model._fit.update(
            {
                "edf": float(payload.get("edf", payload["n_features"])),
                "mu": [],
                "trace_iteration": [],
                "trace_deviance": [],
                "trace_halvings": [],
                "trace_max_step": [],
            }
        )
        model.lambda_ = {str(k): float(v) for k, v in payload.get("lambda", {}).items()}
        return model


__all__ = ["GLM", "FitTrace"]
