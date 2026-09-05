"""The GLM: the honest baseline every other model has to beat.

scikit-learn shaped — ``GLM(family=...).fit(X, y, sample_weight=..., offset=...)`` then
``predict`` — with the things a GLM is *for* attached: standard errors, deviance against the
null model, per-row contributions, and a fit trace that explains how it got there.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from glasshouse import _core, encoders, splits
from glasshouse._rows import subset_column
from glasshouse.arrays import F64, ArrayLike, columns, to_matrix, to_vector
from glasshouse.metrics import FamilyName, deviance
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
    (still moving), or ``"no_improvement"`` (every step, however small, raised the deviance
    by more than ``tol``'s worth of noise: a separated / ill-posed problem).
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


AlphaRule = Literal["min", "1se"]


@dataclass(frozen=True)
class AlphaPath:
    """The regularisation path a cross-validated elastic-net walked, one row per alpha.

    ``cv_deviance`` is the held-out mean deviance averaged over the inner folds and
    ``cv_se`` its standard error; ``n_nonzero`` counts the penalised coefficients still in
    the model when fitted on all training rows at that alpha. ``chosen`` is the row the
    rule picked: ``"min"`` is the lowest ``cv_deviance``; ``"1se"`` is the most penalised
    alpha whose ``cv_deviance`` is within one standard error of that minimum, the usual
    choice when a simpler model that scores the same is worth more than the last decimal.
    """

    alphas: F64
    cv_deviance: F64
    cv_se: F64
    n_nonzero: npt.NDArray[np.int64]
    chosen: int
    rule: str

    def __str__(self) -> str:
        """Fixed-width table with the chosen row marked."""
        lines = [f"{'alpha':>12}{'cv deviance':>14}{'se':>12}{'nonzero':>9}"]
        for i, (a, d, s, k) in enumerate(
            zip(self.alphas, self.cv_deviance, self.cv_se, self.n_nonzero, strict=True)
        ):
            mark = " <- " + self.rule if i == self.chosen else ""
            lines.append(f"{a:>12.4g}{d:>14.6g}{s:>12.3g}{k:>9}{mark}")
        return "\n".join(lines)


# A search evaluation: fit at this lambda, starting from these coefficients (or cold), and
# return the deviance, the edf, and the converged coefficients for the next one to start from.
_FitAt = Callable[[float, F64 | None], tuple[float, float, F64]]


def _gcv_minimise(
    fit_at: _FitAt,
    n_rows: int,
    trace: list[tuple[float, float, float]],
    start: F64 | None = None,
) -> tuple[float, F64 | None]:
    """1-D GCV minimisation: a coarse log-spaced grid, then a finer pass around the winner.

    GCV is ``n * deviance / (n - edf)^2`` — mgcv's criterion with gamma = 1. The numerator
    rewards fit; the shrinking denominator charges for every effective coefficient spent.
    Each evaluation is appended to ``trace`` as ``(lambda, gcv, edf)``.

    Neighbouring lambdas have neighbouring optima, so each fit warm-starts from the previous
    one and the fine pass from the coarse winner: the same fixed points, reached in a couple
    of iterations instead of seven. Returns the chosen lambda and its coefficients.
    """

    def sweep(grid: F64, start: F64 | None) -> tuple[float, F64 | None]:
        best_lam, best, best_coef, coef = float(grid[0]), float("inf"), start, start
        for lam in grid:
            deviance, edf, coef = fit_at(float(lam), coef)
            gcv = n_rows * deviance / (n_rows - edf) ** 2
            trace.append((float(lam), float(gcv), float(edf)))
            if gcv < best:
                best_lam, best, best_coef = float(lam), float(gcv), coef
        return best_lam, best_coef

    coarse, coef = sweep(np.logspace(-4.0, 7.0, 23), start)
    return sweep(np.logspace(np.log10(coarse) - 0.5, np.log10(coarse) + 0.5, 9), coef)


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
        (pin it with ``Smooth(lam=...)``). ``Smooth(monotone="increasing")`` (or a
        ``BSpline``) fits the term under a shape constraint: the curve cannot fall.
        Encoders are fitted on the training rows only; with a time-ordered ``fold`` target
        encoding is cumulative (past-only) automatically. Columns not in a frame cannot have
        terms.
    fit_intercept : bool
        Prepend a column of ones. Turn off only if your design already has one.
    max_iter, tol : int, float
        IRLS stopping rules. ``tol`` is the relative change in deviance.
    alpha, l1_ratio : float or "cv", float
        Elastic-net penalty, glmnet's and glum's convention: the objective is the mean
        deviance over 2 plus ``alpha * (l1_ratio * sum|b| + (1 - l1_ratio)/2 * sum b^2)``
        over every column but the intercept, so ``alpha`` means the same number here as
        there. ``l1_ratio=1`` is the lasso (coefficients reach exactly zero), ``0`` is
        ridge. ``alpha="cv"`` walks a path down from the alpha that zeroes everything and
        picks one by ``cv``-fold cross-validation on the training rows (``alpha_rule``:
        ``"1se"`` for the simplest model within one standard error of the best, ``"min"``
        for the best); the path is kept in ``path_``. The columns are penalised on their
        own scale: use ``"standardize"`` terms for a comparable penalty across features.
        Not combinable with ``"smooth"`` or monotone terms.
    cv, alpha_rule, n_alphas, alpha_ratio
        The ``alpha="cv"`` search: inner folds (random k-fold on the training rows; a
        time-ordered ``fold`` is refused, pass ``alpha`` explicitly), the rule, the number
        of alphas on the path and how far down it goes (``alpha_min = alpha_ratio *
        alpha_max``).

    Attributes (after ``fit``)
    --------------------------
    coef_, intercept_, se_, cov_ : the estimates, their standard errors and covariance.
    feature_names_in_ : column names from the frame, or ``x0, x1, ...``.
    deviance_, null_deviance_, dispersion_ : the fit statistics (total, not mean, deviance).
    edf_, lambda_, gcv_ : what the smooths spent, the chosen penalties, the searched grids.
    alpha_, path_ : the elastic-net alpha used, and the cross-validated path if searched.
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
    alpha: float | Literal["cv"] | None = None
    l1_ratio: float = 0.5
    cv: int = 5
    alpha_rule: AlphaRule = "1se"
    n_alphas: int = 50
    alpha_ratio: float = 1e-3
    alpha_: float | None = field(default=None, repr=False)
    path_: AlphaPath | None = field(default=None, repr=False)
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
        elastic_net = self._elastic_net(matrix, yy, w, o, fold, penalty)
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
            monotone=self._monotone_chains(),
            elastic_net=elastic_net,
        )
        self._fit = result
        self.feature_names_in_ = names
        return self

    # ------------------------------------------------------------------ elastic-net

    def _elastic_net(
        self,
        matrix: F64,
        y: F64,
        w: F64 | None,
        o: F64 | None,
        fold: Fold | None,
        penalty: F64 | None,
    ) -> tuple[float, float, list[bool]] | None:
        """Resolve ``alpha`` (searching the path if asked) into the solver's penalty tuple."""
        if self.alpha is None:
            self.alpha_, self.path_ = None, None
            return None
        if penalty is not None or self._monotone_chains() is not None:
            msg = "alpha (elastic-net) cannot be combined with smooth or monotone terms"
            raise ValueError(msg)
        penalised = [not self.fit_intercept] + [True] * (matrix.shape[1] - 1)
        if self.alpha == "cv":
            self.alpha_, self.path_ = self._cv_alpha(matrix, y, w, o, fold, penalised)
        else:
            self.alpha_, self.path_ = float(self.alpha), None
        return (self.alpha_, self.l1_ratio, penalised)

    def _cv_alpha(
        self,
        matrix: F64,
        y: F64,
        w: F64 | None,
        o: F64 | None,
        fold: Fold | None,
        penalised: list[bool],
    ) -> tuple[float, AlphaPath]:
        """Walk the path on inner folds, warm-started, and pick alpha by the rule."""
        if fold is not None and fold.kind == "time":
            msg = (
                "alpha='cv' needs exchangeable rows: inner folds on a time-ordered fold would "
                "let the future score the past; pass alpha explicitly"
            )
            raise ValueError(msg)
        alpha_max = _core.glm_alpha_max(
            self.family, self._link_name(), matrix, y, w, o, self.power, self.l1_ratio, penalised
        )
        alphas = np.geomspace(alpha_max, alpha_max * self.alpha_ratio, self.n_alphas)
        inner = splits.kfold(matrix.shape[0], k=self.cv, seed=0)
        held_out = np.empty((len(inner), len(alphas)))
        for f, inner_fold in enumerate(inner):
            tr, te = inner_fold.train_idx, inner_fold.test_idx
            coef: F64 | None = None
            for a, alpha in enumerate(alphas):
                coef = self._lean_fit(
                    matrix[tr], y[tr], _pick(w, tr), _pick(o, tr), alpha, penalised, coef
                )
                eta = matrix[te] @ coef + (0.0 if o is None else o[te])
                held_out[f, a] = deviance(
                    y[te],
                    self._inverse_link(eta),
                    family=self.family,
                    power=self.power,
                    sample_weight=_pick(w, te),
                )
        cv_mean = held_out.mean(axis=0)
        cv_se = held_out.std(axis=0, ddof=1) / np.sqrt(len(inner))
        best = int(np.argmin(cv_mean))
        chosen = (
            best
            if self.alpha_rule == "min"
            else int(np.flatnonzero(cv_mean <= cv_mean[best] + cv_se[best])[0])
        )
        n_nonzero = np.empty(len(alphas), dtype=np.int64)
        coef = None
        for a, alpha in enumerate(alphas):
            coef = self._lean_fit(matrix, y, w, o, alpha, penalised, coef)
            n_nonzero[a] = int(np.count_nonzero(coef[1:] if self.fit_intercept else coef))
        path = AlphaPath(alphas, cv_mean, cv_se, n_nonzero, chosen, self.alpha_rule)
        return float(alphas[chosen]), path

    def _lean_fit(  # noqa: PLR0913, PLR0917
        self,
        matrix: F64,
        y: F64,
        w: F64 | None,
        o: F64 | None,
        alpha: float,
        penalised: list[bool],
        start: F64 | None,
    ) -> F64:
        r = _core.glm_fit(
            self.family,
            self._link_name(),
            np.ascontiguousarray(matrix),
            y,
            w,
            o,
            self.power,
            self.max_iter,
            self.tol,
            warm_start=start,
            inference=False,
            elastic_net=(alpha, self.l1_ratio, penalised),
        )
        return np.asarray(r["coef"], dtype=np.float64)

    def _inverse_link(self, eta: F64) -> F64:
        link = self._link_name()
        if link == "identity":
            return eta
        if link == "log":
            return np.asarray(np.exp(eta), dtype=np.float64)
        return np.asarray(1.0 / (1.0 + np.exp(-eta)), dtype=np.float64)

    def _lasso(self) -> bool:
        return bool(self.alpha_) and self.l1_ratio > 0.0

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

        def fit_at(lambdas: dict[str, float], start: F64 | None) -> tuple[float, float, F64]:
            # lean: the search reads deviance, edf and coefficients; the null model and the
            # covariances wait for the final fit
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
                warm_start=start,
                inference=False,
                monotone=self._monotone_chains(),
            )
            return float(r["deviance"]), float(r["edf"]), np.asarray(r["coef"], dtype=np.float64)

        lambdas = {n: (e.lam if e.lam is not None else 1.0) for n, e in smooths.items()}
        free = [n for n, e in smooths.items() if e.lam is None]
        self.gcv_ = {n: [] for n in free}
        start: F64 | None = None
        for _ in range(2 if len(free) > 1 else 1):
            for name in free:
                # `lambdas` is shared state on purpose: each 1-D search sees the others'
                # current values, which is what makes the sweeps coordinate descent
                def fit_at_lam(
                    lam: float, start: F64 | None, _name: str = name
                ) -> tuple[float, float, F64]:
                    return fit_at({**lambdas, _name: lam}, start)

                lambdas[name], start = _gcv_minimise(
                    fit_at_lam, matrix.shape[0], self.gcv_[name], start
                )
        self.lambda_ = {n: float(v) for n, v in lambdas.items()}
        return combined(self.lambda_)

    def _monotone_chains(self) -> list[tuple[list[int], bool, bool]] | None:
        """Return the shape constraints as chains of design columns for the solver.

        A spline term with ``monotone`` set gives one chain over its columns: the fitted
        coefficients must not decrease (or increase) along the knots. The chain is anchored
        because the term's first basis column was dropped, so its coefficient is zero and the
        first kept coefficient is constrained against that zero.
        """
        shift = 1 if self.fit_intercept else 0
        chains = []
        for name, enc in self.encoders_.items():
            monotone = getattr(enc, "monotone", None)
            if monotone is None:
                continue
            lo, hi = self._slices[name]
            chains.append((list(range(lo + shift, hi + shift)), monotone == "increasing", True))
        return chains or None

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
        """Covariance of all coefficients (intercept first), ``dispersion * (X'WX)^-1``.

        Not defined for an L1-penalised fit (the selection is part of the estimator); refit
        the selected columns without the penalty if you need standard errors, knowing that
        post-selection inference has its own caveats.
        """
        if self._lasso():
            msg = (
                "standard errors are not defined for an L1-penalised fit: refit the selected "
                "columns with alpha=None for inference (post-selection caveats apply)"
            )
            raise ValueError(msg)
        p = self._require_fit()["n_features"]
        return np.asarray(self._fit["cov"], dtype=np.float64).reshape(p, p)

    @property
    def se_(self) -> F64:
        """Standard errors of all coefficients, intercept first."""
        return np.sqrt(np.diag(self.cov_))

    @property
    def cov_robust_(self) -> F64:
        """HC1 sandwich covariance: robust to a wrong variance function (over-dispersion)."""
        if self._lasso():
            msg = "robust standard errors are not defined for an L1-penalised fit"
            raise ValueError(msg)
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
        return self._inverse_link(self.predict_linear(X, offset))

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
        if self._lasso():
            lines = [
                f"GLM family={self.family} link={self._link_name()}  n={r['n_rows']}  "
                f"alpha={self.alpha_:.4g} l1_ratio={self.l1_ratio}  (no standard errors for an "
                "L1 fit)",
                f"{'term':<24}{'coef':>14}",
            ]
            lines.extend(
                f"{name:<24}{b:>14.6g}"
                for name, b in zip(self.feature_names_in_, coef, strict=True)
            )
            lines.append(f"deviance={r['deviance']:.6g}  null_deviance={r['null_deviance']:.6g}")
            return "\n".join(lines)
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
            "alpha": self.alpha_,
            "l1_ratio": self.l1_ratio,
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
        model.alpha_ = payload.get("alpha")
        model.alpha = model.alpha_
        model.l1_ratio = float(payload.get("l1_ratio", 0.5))
        return model


def _pick(v: F64 | None, rows: npt.NDArray[np.int64]) -> F64 | None:
    return None if v is None else v[rows]


__all__ = ["GLM", "AlphaPath", "FitTrace"]
