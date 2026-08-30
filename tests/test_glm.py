"""GLM golden tests vs statsmodels: coefficients, standard errors, deviance — every family."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import statsmodels.api as sm

from glasshouse import GLM
from glasshouse.metrics import balance

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(6)
N = 3000
X = pd.DataFrame(
    {
        "age": rng.uniform(-1, 1, size=N),
        "power": rng.normal(size=N),
        "urban": (rng.uniform(size=N) < 0.4).astype(float),
    }
)
ETA = 0.2 + 0.5 * X["age"] - 0.3 * X["power"] + 0.4 * X["urban"]
EXPOSURE = rng.uniform(0.2, 1.0, size=N)
W = rng.uniform(0.5, 2.0, size=N)


def _sm_family(family: str, link: str, power: float | None) -> sm.families.Family:
    links = {
        "identity": sm.families.links.Identity(),
        "log": sm.families.links.Log(),
        "logit": sm.families.links.Logit(),
    }
    fam = {
        "gaussian": sm.families.Gaussian,
        "poisson": sm.families.Poisson,
        "gamma": sm.families.Gamma,
        "binomial": sm.families.Binomial,
    }
    if family == "tweedie":
        return sm.families.Tweedie(link=links[link], var_power=power)
    return fam[family](link=links[link])


def _targets(family: str) -> tuple[Arr, Arr | None, Arr | None]:
    """(y, sample_weight, offset) for each family, drawn from the true model."""
    if family == "gaussian":
        return (ETA + rng.normal(scale=0.5, size=N)).to_numpy(), W, None
    if family == "poisson":
        return rng.poisson(np.exp(ETA) * EXPOSURE).astype(float), None, np.log(EXPOSURE)
    if family == "gamma":
        return rng.gamma(2.0, np.exp(ETA) / 2.0), W, None
    if family == "tweedie":
        lam, mu = 2.0, np.exp(ETA)
        counts = rng.poisson(lam, size=N)
        return rng.gamma(counts + 1e-12, mu / lam), None, None
    p = 1 / (1 + np.exp(-ETA))
    return (rng.uniform(size=N) < p).astype(float), W, None


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma", "tweedie", "binomial"])
def test_golden_vs_statsmodels(family: str) -> None:
    y, w, offset = _targets(family)
    power = 1.5 if family == "tweedie" else None
    # both solvers pushed to 1e-14 so the test compares optima, not stopping points
    # (at the default tol=1e-10 gamma stops one iteration earlier: 1e-7 apart, same deviance)
    model = GLM(family=family, power=power, tol=1e-14).fit(  # type: ignore[arg-type]
        X, y, sample_weight=w, offset=offset
    )
    ref = sm.GLM(
        y,
        sm.add_constant(X.to_numpy()),
        family=_sm_family(family, model._link_name(), power),
        var_weights=w,
        offset=offset,
    ).fit(tol=1e-14, maxiter=200)
    ours = np.concatenate([[model.intercept_], model.coef_])
    np.testing.assert_allclose(ours, ref.params, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(model.se_, ref.bse, rtol=1e-6)
    assert model.deviance_ == pytest.approx(ref.deviance, rel=1e-8)
    assert model.null_deviance_ == pytest.approx(ref.null_deviance, rel=1e-8)
    assert model.dispersion_ == pytest.approx(ref.scale, rel=1e-6)
    assert model.converged_


def test_canonical_link_is_balanced_and_predict_matches_fit() -> None:
    y, _, offset = _targets("poisson")
    model = GLM(family="poisson").fit(X, y, offset=offset)
    mu = model.predict(X, offset=offset)
    assert balance(y, mu) == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_allclose(mu, model._fit["mu"], rtol=1e-12)
    assert model.deviance_ < model.null_deviance_


def test_trace_explains_the_fit() -> None:
    y, _, offset = _targets("poisson")
    model = GLM(family="poisson").fit(X, y, offset=offset)
    t = model.trace_
    assert t.stop == "converged" and len(t.deviance) == model.n_iter_
    assert np.all(np.diff(t.deviance) <= 1e-9)  # never goes up: that's what halving is for
    assert "stopped: converged" in str(t)
    assert "deviance" in model.summary() and "age" in model.summary()


def test_contributions_add_up_and_names_are_kept() -> None:
    y, w, _ = _targets("gamma")
    model = GLM(family="gamma").fit(X, y, sample_weight=w)
    contrib, names = model.contributions(X)
    assert names == ["intercept", "age", "power", "urban"]
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict_linear(X), rtol=1e-12)


def test_round_trip_without_pickle() -> None:
    y, w, _ = _targets("binomial")
    model = GLM(family="binomial").fit(X, y, sample_weight=w)
    back = GLM.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(back.predict(X), model.predict(X), rtol=1e-12)
    np.testing.assert_allclose(back.se_, model.se_, rtol=1e-12)


def test_fails_early_and_clearly() -> None:
    y, _, _ = _targets("gaussian")
    with pytest.raises(ValueError, match="rank-deficient"):
        GLM(family="gaussian").fit(pd.DataFrame({"a": X["age"], "b": 2 * X["age"]}), y)
    with pytest.raises(ValueError, match="poisson needs y >= 0"):
        GLM(family="poisson").fit(X, y)
    with pytest.raises(ValueError, match="not fitted"):
        GLM(family="poisson").predict(X)
    with pytest.raises(ValueError, match="one of: identity, log, logit"):
        GLM(family="poisson", link="probit").fit(X, np.abs(y))
    with pytest.raises(ValueError, match="do not match"):
        GLM(family="gaussian").fit(X, y).predict(X.rename(columns={"age": "AGE"}))
