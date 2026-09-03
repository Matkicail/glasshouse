"""B-splines: golden vs scipy's design matrix, the basis properties, and a GLM that bends."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

from glasshouse import GLM, _core, encoders, splits
from glasshouse.encoders import BSpline
from glasshouse.metrics import deviance

rng = np.random.default_rng(15)


def test_golden_vs_scipy_design_matrix() -> None:
    x = rng.uniform(0.0, 10.0, size=500)
    interior = [2.5, 5.0, 7.5]
    knots = [0.0] * 4 + interior + [10.0] * 4
    flat, p = _core.bspline_design(x, knots, 3)
    ours = np.asarray(flat).reshape(len(x), p)
    ref = ScipyBSpline.design_matrix(x, np.asarray(knots), 3).toarray()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_basis_is_a_partition_of_unity_and_local() -> None:
    x = np.linspace(-1.0, 4.0, 301)
    knots = [-1.0] * 4 + [0.5, 1.0, 2.0] + [4.0] * 4
    flat, p = _core.bspline_design(x, knots, 3)
    design = np.asarray(flat).reshape(len(x), p)
    np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-12)
    assert np.all((design >= 0.0) & (design <= 1.0))
    assert np.all((design > 0).sum(axis=1) <= 4)  # cubic: at most 4 active


def test_encoder_places_quantile_knots_and_clamps() -> None:
    v = rng.gamma(2.0, 10.0, size=2000)
    enc = BSpline(df=6, name="age").fit(v)
    interior = enc.knots_[4:-4]
    np.testing.assert_allclose(interior, np.quantile(v, [0.25, 0.5, 0.75]), rtol=1e-12)
    m, names = enc.transform(v)
    assert m.shape == (2000, 6) and names == [f"age_bs{i}" for i in range(1, 7)]
    below, above = enc.transform([v.min() - 100.0])[0], enc.transform([v.min()])[0]
    np.testing.assert_array_equal(below, above)  # clamped, not extrapolated
    back = encoders.from_dict(json.loads(json.dumps(enc.to_dict())))
    np.testing.assert_allclose(back.transform(v)[0], m)


def test_glm_spline_term_bends_where_linear_cannot() -> None:
    n = 8000
    age = rng.uniform(18.0, 80.0, size=n)
    eta = -2.5 + 0.0025 * (age - 45.0) ** 2
    y = rng.poisson(np.exp(eta)).astype(float)
    df = pd.DataFrame({"age": age})
    fold = splits.kfold(n, k=3, seed=0)[0]
    te = fold.test_idx
    linear = GLM(family="poisson").fit(df, y, fold=fold)
    smooth = GLM(family="poisson", terms={"age": "spline"}).fit(df, y, fold=fold)
    custom = GLM(family="poisson", terms={"age": BSpline(df=9)}).fit(df, y, fold=fold)
    d_lin = deviance(y[te], linear.predict(df.iloc[te]), family="poisson")
    d_smooth = deviance(y[te], smooth.predict(df.iloc[te]), family="poisson")
    d_custom = deviance(y[te], custom.predict(df.iloc[te]), family="poisson")
    assert d_smooth < d_lin and d_custom < d_lin
    assert "age_bs1" in smooth.feature_names_in_
    # the fitted curve is a U: higher at the ends than the middle
    grid = pd.DataFrame({"age": np.array([20.0, 45.0, 75.0])})
    mu = smooth.predict(grid)
    assert mu[0] > mu[1] and mu[2] > mu[1]


def test_glm_with_spline_round_trips_without_pickle() -> None:
    n = 800
    age = rng.uniform(0.0, 1.0, size=n)
    y = rng.poisson(np.exp(-1.0 + np.sin(3 * age))).astype(float)
    df = pd.DataFrame({"age": age})
    m = GLM(family="poisson", terms={"age": BSpline(df=5)}).fit(df, y)
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(df), m.predict(df), rtol=1e-12)


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="df must exceed the degree"):
        BSpline(df=3).fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="constant"):
        BSpline().fit(np.ones(50))
    with pytest.raises(ValueError, match="step function"):
        _core.bspline_design(np.array([0.5]), [0.0, 0.0, 1.0, 1.0], 0)


def test_tied_quantiles_collapse_instead_of_breaking_the_design() -> None:
    """A BonusMalus-shaped column: most of the mass on the minimum value."""
    v = np.where(rng.uniform(size=3000) < 0.6, 50.0, rng.uniform(51.0, 150.0, size=3000))
    enc = BSpline(df=6, name="bm").fit(v)
    interior = enc.knots_[4:-4]
    assert len(interior) < 3  # the tied quantiles collapsed
    assert all(50.0 < k < v.max() for k in interior)
    m, names = enc.transform(v)
    assert m.shape[1] == len(names) < 6
    # and a GLM with intercept + this term is full rank
    y = rng.poisson(np.exp(-2.0 + 0.01 * (v - 50.0))).astype(float)
    GLM(family="poisson", terms={"bm": BSpline(df=6)}).fit(pd.DataFrame({"bm": v}), y)
