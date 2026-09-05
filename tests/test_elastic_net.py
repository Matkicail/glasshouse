"""Elastic-net GLM: golden vs glum at the same alpha, ridge vs the quadratic penalty, the path.

glum minimises ``sum(w d) / (2 sum w) + alpha (l1 sum|b| + (1 - l1)/2 sum b^2)`` with the
intercept free (checked against the closed-form ridge to 1e-16 before this was written), so
the same ``alpha`` and ``l1_ratio`` must give the same coefficients.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from glum import GeneralizedLinearRegressor

from glasshouse import GLM, _core, splits
from glasshouse.encoders import Smooth
from glasshouse.metrics import FamilyName

rng = np.random.default_rng(23)
N, P = 1500, 8
X = rng.normal(size=(N, P))
X[:, 3] *= 4.0  # one column on a bigger scale: the penalty is on the raw scale, like glum
TRUE = np.array([0.6, -0.4, 0.0, 0.05, 0.3, 0.0, 0.0, -0.2])
W = rng.uniform(0.5, 2.0, size=N)
EXPO = rng.uniform(0.3, 1.0, size=N)
FRAME = pd.DataFrame(X, columns=[f"x{j}" for j in range(P)])


def _targets(family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    eta = 0.2 + X @ TRUE
    if family == "gaussian":
        return eta + rng.normal(scale=0.5, size=N), W, None
    if family == "poisson":
        return rng.poisson(np.exp(eta) * EXPO).astype(float), W, np.log(EXPO)
    p = 1.0 / (1.0 + np.exp(-eta))
    return (rng.uniform(size=N) < p).astype(float), W, None


_GLUM = {"gaussian": "normal", "poisson": "poisson", "binomial": "binomial"}


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
@pytest.mark.parametrize("l1_ratio", [1.0, 0.5, 0.0])
def test_golden_vs_glum(family: FamilyName, l1_ratio: float) -> None:
    y, w, offset = _targets(family)
    alpha = 0.02
    ours = GLM(family=family, alpha=alpha, l1_ratio=l1_ratio).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    ref = GeneralizedLinearRegressor(
        family=_GLUM[family],
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        scale_predictors=False,
        gradient_tol=1e-10,
        max_iter=5000,
    ).fit(X, y, sample_weight=w, offset=offset)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-5, atol=1e-7)
    if l1_ratio > 0:
        assert np.array_equal(ours.coef_ == 0.0, ref.coef_ == 0.0), "same sparsity pattern"
    if l1_ratio == 1.0:
        assert np.any(ours.coef_ == 0.0), "the lasso switches something off at this alpha"


def test_ridge_is_the_quadratic_penalty_and_alpha_zero_is_the_plain_glm() -> None:
    y, w, offset = _targets("poisson")
    alpha = 0.05
    ridge = GLM(family="poisson", alpha=alpha, l1_ratio=0.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    design = np.ascontiguousarray(np.column_stack([np.ones(N), X]))
    s = np.zeros((P + 1, P + 1))
    s[1:, 1:] = w.sum() * alpha * np.eye(P)  # deviance units: D + sum(w) alpha ||b||^2
    quad = _core.glm_fit("poisson", "log", design, y, w, offset, penalty=np.ascontiguousarray(s))
    np.testing.assert_allclose(
        np.concatenate([[ridge.intercept_], ridge.coef_]), quad["coef"], rtol=1e-9, atol=1e-12
    )
    assert ridge.edf_ == pytest.approx(quad["edf"], rel=1e-9)
    free = GLM(family="poisson").fit(FRAME, y, sample_weight=w, offset=offset)
    zero = GLM(family="poisson", alpha=0.0, l1_ratio=1.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    np.testing.assert_allclose(zero.coef_, free.coef_, rtol=1e-8, atol=1e-10)


def test_alpha_max_zeroes_everything_and_just_below_it_does_not() -> None:
    y, w, offset = _targets("poisson")
    design = np.ascontiguousarray(np.column_stack([np.ones(N), X]))
    mask = [False] + [True] * P
    a_max = _core.glm_alpha_max("poisson", "log", design, y, w, offset, None, 1.0, mask)
    at = GLM(family="poisson", alpha=a_max, l1_ratio=1.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    assert np.all(at.coef_ == 0.0), at.coef_
    below = GLM(family="poisson", alpha=0.9 * a_max, l1_ratio=1.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    assert np.any(below.coef_ != 0.0)
    assert at.edf_ == pytest.approx(1.0) and below.edf_ > 1.0


def test_cv_path_keeps_the_signal_and_drops_most_noise() -> None:
    n, p = 4000, 30
    x = rng.normal(size=(n, p))
    true = np.zeros(p)
    true[:5] = [1.0, -0.8, 0.6, -0.5, 0.4]
    y = 1.0 + x @ true + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame(x, columns=[f"f{j}" for j in range(p)])
    m = GLM(family="gaussian", alpha="cv", l1_ratio=1.0, cv=5).fit(frame, y)
    assert m.path_ is not None and len(m.path_.alphas) == 50
    assert m.alpha_ == m.path_.alphas[m.path_.chosen]
    assert np.all(np.diff(m.path_.alphas) < 0), "the path runs down from alpha_max"
    assert m.path_.n_nonzero[0] == 0 and m.path_.n_nonzero[-1] > m.path_.n_nonzero[0]
    best = int(np.argmin(m.path_.cv_deviance))
    assert m.path_.chosen <= best, "1se picks a more penalised alpha than the minimum"
    assert m.path_.cv_deviance[m.path_.chosen] <= (
        m.path_.cv_deviance[best] + m.path_.cv_se[best] + 1e-12
    )
    kept = set(np.flatnonzero(m.coef_ != 0.0))
    assert {0, 1, 2, 3, 4} <= kept and len(kept) <= 12, sorted(kept)
    assert "<- 1se" in str(m.path_)
    tightest = GLM(family="gaussian", alpha="cv", l1_ratio=1.0, cv=5, alpha_rule="min").fit(
        frame, y
    )
    assert tightest.alpha_ is not None and m.alpha_ is not None
    assert tightest.alpha_ <= m.alpha_


def test_round_trip_refusals_and_summary() -> None:
    y, w, offset = _targets("poisson")
    m = GLM(family="poisson", alpha=0.02, l1_ratio=1.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(FRAME), m.predict(FRAME), rtol=1e-12)
    assert back.alpha_ == m.alpha_ and back.l1_ratio == 1.0
    assert "no standard errors" in m.summary()
    with pytest.raises(ValueError, match="not defined for an L1"):
        _ = m.se_
    ridge = GLM(family="poisson", alpha=0.02, l1_ratio=0.0).fit(
        FRAME, y, sample_weight=w, offset=offset
    )
    assert ridge.se_.shape == (P + 1,)
    with pytest.raises(ValueError, match="cannot be combined"):
        GLM(family="poisson", alpha=0.1, terms={"x0": Smooth()}).fit(FRAME, y, offset=offset)
    with pytest.raises(ValueError, match="cannot be combined"):
        GLM(family="poisson", alpha=0.1, terms={"x0": Smooth(monotone="increasing")}).fit(
            FRAME, y, offset=offset
        )
    with pytest.raises(ValueError, match="l1_ratio"):
        GLM(family="poisson", alpha=0.1, l1_ratio=1.5).fit(FRAME, y, offset=offset)
    time_fold = splits.time_ordered(np.arange(N), n_folds=2)[0]
    with pytest.raises(ValueError, match="time-ordered"):
        GLM(family="poisson", alpha="cv").fit(FRAME, y, offset=offset, fold=time_fold)
