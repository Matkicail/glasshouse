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
    p = 1 / (1 + np.exp(-ETA.to_numpy()))
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
    # Robust SEs. Ours use the expected information X'WX as the bread (like R's sandwich for
    # glm, and consistent with se_); statsmodels uses the observed Hessian, which only equals
    # it for a canonical link. So: exact vs statsmodels where links are canonical, and vs the
    # written-out formula for every family.
    np.testing.assert_allclose(model.se_robust_, _hc1_reference(model, X, y, w, offset), rtol=1e-10)
    if family in ("gaussian", "poisson", "binomial"):
        robust = sm.GLM(
            y,
            sm.add_constant(X.to_numpy()),
            family=_sm_family(family, model._link_name(), power),
            var_weights=w,
            offset=offset,
        ).fit(tol=1e-14, maxiter=200, cov_type="HC0")
        # statsmodels' GLM returns the same numbers for HC0 and HC1 (it skips the n/(n-p)
        # small-sample factor that defines HC1); ours is the textbook HC1, so scale their HC0.
        np.testing.assert_allclose(model.se_robust_, robust.bse * np.sqrt(N / (N - 4)), rtol=1e-6)


def _hc1_reference(model: GLM, X: pd.DataFrame, y: Arr, w: Arr | None, offset: Arr | None) -> Arr:  # noqa: N803
    """HC1 with expected-information bread, written out in NumPy: MacKinnon & White (1985)."""
    design = np.column_stack([np.ones(len(y)), X.to_numpy()])
    ww = np.ones(len(y)) if w is None else w
    eta = model.predict_linear(X, offset)
    mu = model.predict(X, offset)
    link = model._link_name()
    mu_eta = {"identity": np.ones_like(eta), "log": np.exp(eta), "logit": mu * (1 - mu)}[link]
    var = {
        "gaussian": np.ones_like(mu),
        "poisson": mu,
        "gamma": mu**2,
        "tweedie": mu**1.5,
        "binomial": mu * (1 - mu),
    }[model.family]
    bread = np.linalg.inv(design.T @ (design * (ww * mu_eta**2 / var)[:, None]))
    score = design * (ww * (y - mu) * mu_eta / var)[:, None]
    meat = score.T @ score
    n, p = design.shape
    cov = bread @ meat @ bread * n / (n - p)
    return np.sqrt(np.diag(cov))


def test_robust_se_exceed_naive_se_under_overdispersion() -> None:
    """Negative-binomial counts fitted as Poisson: the model understates uncertainty."""
    mu = np.exp(ETA.to_numpy())
    y = rng.negative_binomial(n=1.0, p=1.0 / (1.0 + mu)).astype(float)  # variance mu + mu^2
    model = GLM(family="poisson").fit(X, y)
    assert np.all(model.se_robust_ > 1.3 * model.se_)


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
