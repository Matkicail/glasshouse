"""Residuals golden vs statsmodels; A/E by feature agrees with the calibration machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from glasshouse import GLM, _core, metrics, residuals

rng = np.random.default_rng(11)
N = 2000
X = pd.DataFrame(
    {"age": rng.uniform(-1, 1, size=N), "urban": (rng.uniform(size=N) < 0.4).astype(float)}
)
W = rng.uniform(0.5, 2.0, size=N)


@pytest.mark.parametrize(
    ("family", "sm_family", "y"),
    [
        ("poisson", sm.families.Poisson(), rng.poisson(np.exp(0.3 * X.age), size=N).astype(float)),
        (
            "gamma",
            sm.families.Gamma(sm.families.links.Log()),
            rng.gamma(2.0, np.exp(0.3 * X.age) / 2),
        ),
        ("binomial", sm.families.Binomial(), (rng.uniform(size=N) < 0.3).astype(float)),
        ("gaussian", sm.families.Gaussian(), (0.3 * X.age + rng.normal(size=N)).to_numpy()),
    ],
)
def test_golden_vs_statsmodels(family: str, sm_family: sm.families.Family, y: np.ndarray) -> None:
    model = GLM(family=family, tol=1e-14).fit(X, y, sample_weight=W)  # type: ignore[arg-type]
    ref = sm.GLM(y, sm.add_constant(X.to_numpy()), family=sm_family, var_weights=W).fit(tol=1e-14)
    mu = model.predict(X)
    np.testing.assert_allclose(
        residuals.deviance(y, mu, family=family, sample_weight=W),  # type: ignore[arg-type]
        ref.resid_deviance,
        rtol=1e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        residuals.pearson(y, mu, family=family, sample_weight=W),  # type: ignore[arg-type]
        ref.resid_pearson,
        rtol=1e-6,
        atol=1e-9,
    )


def test_residual_identities() -> None:
    y = rng.poisson(1.0, size=N).astype(float)
    mu = np.exp(rng.normal(0, 0.2, size=N))
    d = residuals.deviance(y, mu, family="poisson", sample_weight=W)
    p = residuals.pearson(y, mu, family="poisson", sample_weight=W)
    total = metrics.deviance(y, mu, family="poisson", sample_weight=W) * W.sum()
    assert np.sum(d**2) == pytest.approx(total, rel=1e-10)  # squares sum to the total deviance
    assert np.all(np.sign(d) == np.sign(y - mu)) and np.all(np.sign(p) == np.sign(y - mu))
    assert np.sum(p**2) == pytest.approx(np.sum(W * (y - mu) ** 2 / mu), rel=1e-12)


def test_ae_by_numeric_feature_matches_the_binned_calibration() -> None:
    y = rng.poisson(1.0, size=N).astype(float)
    mu = np.exp(rng.normal(0, 0.2, size=N))
    t = residuals.ae_by_feature(X.age, y, mu, W, name="age", n_bins=5, label="m")
    assert len(t.level) == 5 and t.weight.sum() == pytest.approx(W.sum())
    overall = (t.actual * t.weight).sum() / (t.predicted * t.weight).sum()
    assert overall == pytest.approx(metrics.balance(y, mu, sample_weight=W), rel=1e-12)
    assert t.level[0].startswith("[") and t.level[-1].endswith("]")
    assert "A/E by age" in str(t)


def test_ae_by_categorical_feature_is_the_grouped_ratio() -> None:
    region = rng.choice(["n", "s", "e"], size=N)
    y = rng.poisson(1.0, size=N).astype(float)
    mu = np.exp(rng.normal(0, 0.2, size=N))
    t = residuals.ae_by_feature(region, y, mu, W, name="region")
    assert t.level == ["e", "n", "s"]
    for i, lv in enumerate(t.level):
        m = region == lv
        assert t.actual_over_expected[i] == pytest.approx(
            np.sum(W[m] * y[m]) / np.sum(W[m] * mu[m])
        )
        assert t.n_rows[i] == m.sum()
    payload = t.to_dict()
    assert payload["kind"] == "ae_by_feature" and payload["feature"] == "region"


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="one of: deviance, pearson"):
        _core.residuals("studentised", "poisson", np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="infinite value"):
        residuals.ae_by_feature(np.array([1.0, np.inf]), [1.0, 1.0], [1.0, 1.0], name="x")
