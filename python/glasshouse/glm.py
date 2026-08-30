"""The GLM: the honest baseline every other model has to beat.

scikit-learn shaped — ``GLM(family=...).fit(X, y, sample_weight=..., offset=...)`` then
``predict`` — with the things a GLM is *for* attached: standard errors, deviance against the
null model, per-row contributions, and a fit trace that explains how it got there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from glasshouse import _core
from glasshouse.arrays import F64, ArrayLike, to_matrix, to_vector
from glasshouse.metrics import FamilyName

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
    fit_intercept : bool
        Prepend a column of ones. Turn off only if your design already has one.
    max_iter, tol : int, float
        IRLS stopping rules. ``tol`` is the relative change in deviance.

    Attributes (after ``fit``)
    --------------------------
    coef_, intercept_, se_, cov_ : the estimates, their standard errors and covariance.
    feature_names_in_ : column names from the frame, or ``x0, x1, ...``.
    deviance_, null_deviance_, dispersion_ : the fit statistics (total, not mean, deviance).
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
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-10
    _fit: dict[str, Any] = field(default_factory=dict, repr=False)
    feature_names_in_: list[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        X: ArrayLike,  # noqa: N803 — scikit-learn's spelling
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> GLM:
        """Fit by IRLS. Returns ``self`` so calls chain.

        Parameters
        ----------
        X : frame or 2-D array
            Numeric features. Encode categoricals first; the data door says so if you forget.
        y : 1-D
            Outcome inside the family's support.
        sample_weight : 1-D, optional
            Prior weights (claim counts for a severity model, exposure for a rate model).
        offset : 1-D, optional
            Added to the linear predictor, on the link scale — ``log(exposure)`` for a
            Poisson count model, not the exposure itself.
        """
        matrix, names = to_matrix(X)
        yy = to_vector(y, "y")
        w = None if sample_weight is None else to_vector(sample_weight, "sample_weight")
        o = None if offset is None else to_vector(offset, "offset")
        if self.fit_intercept:
            matrix = np.ascontiguousarray(np.column_stack([np.ones(matrix.shape[0]), matrix]))
            names = ["intercept", *names]
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
        )
        self._fit = result
        self.feature_names_in_ = names
        return self

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

    def _design(self, X: ArrayLike) -> F64:  # noqa: N803
        matrix, names = to_matrix(X)
        expected = self.feature_names_in_[1:] if self.fit_intercept else self.feature_names_in_
        if names != expected and not all(n.startswith("x") for n in names):
            msg = f"columns {names} do not match the columns fitted on {expected}"
            raise ValueError(msg)
        if matrix.shape[1] != len(expected):
            msg = f"X has {matrix.shape[1]} columns but the model was fitted on {len(expected)}"
            raise ValueError(msg)
        if self.fit_intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        return np.ascontiguousarray(matrix)

    def predict_linear(self, X: ArrayLike, offset: ArrayLike | None = None) -> F64:  # noqa: N803
        """Return the linear predictor ``eta = X beta + offset`` (link scale)."""
        coef = np.asarray(self._require_fit()["coef"], dtype=np.float64)
        eta = self._design(X) @ coef
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
        return self._design(X) * coef, list(self.feature_names_in_)

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
            "coef": list(map(float, r["coef"])),
            "cov": list(map(float, r["cov"])),
            "cov_robust": list(map(float, r["cov_robust"])),
            "n_features": int(r["n_features"]),
            "n_rows": int(r["n_rows"]),
            "deviance": float(r["deviance"]),
            "null_deviance": float(r["null_deviance"]),
            "dispersion": float(r["dispersion"]),
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
                "mu": [],
                "trace_iteration": [],
                "trace_deviance": [],
                "trace_halvings": [],
                "trace_max_step": [],
            }
        )
        return model


__all__ = ["GLM", "FitTrace"]
