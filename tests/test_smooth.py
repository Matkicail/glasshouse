"""Penalised smooths: solver golden vs statsmodels GLMGam, GCV behaviour, and properties.

The scale bridge for the golden test: statsmodels penalises the log-likelihood with
``alpha * beta' S beta`` while our solver penalises the deviance (which is ``-2 loglik``
plus a constant), so feeding it ``S = 2 * alpha * cov_der2`` must reproduce GLMGam's
coefficients — and it does, to machine precision. mgcv itself needs R, which this machine
does not have; GLMGam is the reference we can actually run.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.genmod.families import Poisson

from glasshouse import GLM, _core, splits
from glasshouse.encoders import Smooth
from glasshouse.metrics import deviance

rng = np.random.default_rng(21)


def test_golden_vs_statsmodels_glmgam() -> None:
    x = rng.uniform(0.0, 1.0, 500)
    y = rng.poisson(np.exp(-1.0 + np.sin(3.0 * x))).astype(float)
    bs = BSplines(x[:, None], df=[10], degree=[3], include_intercept=False)
    alpha = 2.5
    res = GLMGam(y, np.ones((len(y), 1)), smoother=bs, alpha=[alpha], family=Poisson()).fit()
    design = np.ascontiguousarray(np.column_stack([np.ones(len(y)), bs.basis]))
    p = design.shape[1]
    s = np.zeros((p, p))
    s[1:, 1:] = 2.0 * alpha * bs.smoothers[0].cov_der2
    ours = _core.glm_fit("poisson", "log", design, y, penalty=np.ascontiguousarray(s))
    np.testing.assert_allclose(ours["coef"], res.params, rtol=1e-8, atol=1e-10)
    assert ours["edf"] == pytest.approx(float(np.sum(res.edf)), rel=1e-8)


def test_fixed_lam_gaussian_matches_penalised_normal_equations() -> None:
    """The whole Python path — encoder, embedding, intercept shift — vs the formula."""
    n = 300
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(3.0 * x) + rng.normal(0.0, 0.3, size=n)
    lam = 3.7
    enc = Smooth(df=6, name="x").fit(x)
    basis, _ = enc.transform(x)
    design = np.column_stack([np.ones(n), basis])
    s = np.zeros((design.shape[1],) * 2)
    s[1:, 1:] = lam * enc.penalty_matrix()
    beta = np.linalg.solve(design.T @ design + s, design.T @ y)
    m = GLM(family="gaussian", terms={"x": Smooth(df=6, lam=lam)}).fit(pd.DataFrame({"x": x}), y)
    np.testing.assert_allclose(
        np.concatenate([[m.intercept_], m.coef_]), beta, rtol=1e-9, atol=1e-12
    )
    assert m.lambda_ == {"x": lam}
    assert m.gcv_ == {}  # a pinned lambda is not searched


def test_glm_smooth_term_picks_its_own_wiggliness() -> None:
    n = 4000
    age = rng.uniform(18.0, 80.0, size=n)
    eta = -2.5 + 0.0025 * (age - 45.0) ** 2
    y = rng.poisson(np.exp(eta)).astype(float)
    df = pd.DataFrame({"age": age})
    fold = splits.kfold(n, k=3, seed=0)[0]
    te = fold.test_idx
    linear = GLM(family="poisson").fit(df, y, fold=fold)
    smooth = GLM(family="poisson", terms={"age": "smooth"}).fit(df, y, fold=fold)
    d_lin = deviance(y[te], linear.predict(df.iloc[te]), family="poisson")
    d_smooth = deviance(y[te], smooth.predict(df.iloc[te]), family="poisson")
    assert d_smooth < d_lin
    assert smooth.lambda_["age"] > 0.0
    assert len(smooth.gcv_["age"]) == 32  # 23 coarse + 9 fine: the search is on the record
    assert 2.0 < smooth.edf_ < 10.0  # a curve, not the whole 9-column budget
    # the intercept is unpenalised, so the fit stays balanced on its training rows
    tr = fold.train_idx
    np.testing.assert_allclose(smooth.predict(df.iloc[tr]).sum(), y[tr].sum(), rtol=1e-8)


def test_edf_falls_as_lambda_rises_and_gcv_beats_the_extremes() -> None:
    n = 1500
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.4, size=n)
    df = pd.DataFrame({"x": x})
    edfs = [
        GLM(family="gaussian", terms={"x": Smooth(lam=lam)}).fit(df, y).edf_
        for lam in (1e-3, 1.0, 1e3, 1e6)
    ]
    assert all(a > b for a, b in itertools.pairwise(edfs)), edfs
    # the second-difference penalty leaves a linear trend unpenalised: even a huge lambda
    # keeps the intercept plus roughly one linear direction
    assert 1.0 < edfs[-1] < 3.0

    fold = splits.kfold(n, k=3, seed=1)[0]
    te = fold.test_idx

    def held_out(m: GLM) -> float:
        return deviance(y[te], m.predict(df.iloc[te]), family="gaussian")

    picked = GLM(family="gaussian", terms={"x": "smooth"}).fit(df, y, fold=fold)
    rough = GLM(family="gaussian", terms={"x": Smooth(lam=1e-4)}).fit(df, y, fold=fold)
    flat = GLM(family="gaussian", terms={"x": Smooth(lam=1e7)}).fit(df, y, fold=fold)
    # GCV crushes the over-smoothed fit, and matches the unpenalised one (to within noise
    # on this held-out fold) while spending visibly fewer effective coefficients
    assert held_out(picked) < held_out(flat)
    assert held_out(picked) <= held_out(rough) * 1.02
    assert picked.edf_ < rough.edf_


def test_gcv_trace_rows_reproduce_and_the_winner_is_the_argmin() -> None:
    n = 400
    x = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, size=n)
    df = pd.DataFrame({"x": x})
    m = GLM(family="gaussian", terms={"x": "smooth"}).fit(df, y)
    lam, gcv, edf = m.gcv_["x"][7]  # any searched row must be reproducible
    refit = GLM(family="gaussian", terms={"x": Smooth(lam=lam)}).fit(df, y)
    assert refit.edf_ == pytest.approx(edf, rel=1e-9)
    assert gcv == pytest.approx(n * refit.deviance_ / (n - edf) ** 2, rel=1e-9)
    best = min(m.gcv_["x"], key=lambda row: row[1])
    assert m.lambda_["x"] == pytest.approx(best[0])


def test_two_smooths_get_their_own_lambdas() -> None:
    n = 2500
    a = rng.uniform(0.0, 1.0, size=n)
    b = rng.uniform(0.0, 1.0, size=n)
    y = np.sin(2.0 * np.pi * a) + 0.1 * b + rng.normal(0.0, 0.3, size=n)
    df = pd.DataFrame({"a": a, "b": b})
    m = GLM(family="gaussian", terms={"a": "smooth", "b": "smooth"}).fit(df, y)
    assert set(m.lambda_) == {"a", "b"}
    assert all(len(m.gcv_[k]) == 64 for k in ("a", "b"))  # two coordinate sweeps of 32
    flat = GLM(family="gaussian").fit(df, y)
    assert m.deviance_ < flat.deviance_


def test_round_trips_without_pickle() -> None:
    n = 800
    x = rng.uniform(0.0, 1.0, size=n)
    y = rng.poisson(np.exp(-1.0 + np.sin(3.0 * x))).astype(float)
    df = pd.DataFrame({"x": x})
    m = GLM(family="poisson", terms={"x": "smooth"}).fit(df, y)
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(df), m.predict(df), rtol=1e-12)
    assert back.lambda_ == pytest.approx(m.lambda_)
    assert back.edf_ == pytest.approx(m.edf_)


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="df must exceed the degree"):
        Smooth(df=3).fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="constant"):
        Smooth().fit(np.ones(30))
    # the solver refuses an asymmetric penalty outright (Smooth never builds one)
    x = np.ascontiguousarray(np.column_stack([np.ones(20), np.arange(20.0)]))
    bad = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        _core.glm_fit("gaussian", "identity", x, np.arange(20.0), penalty=np.ascontiguousarray(bad))
