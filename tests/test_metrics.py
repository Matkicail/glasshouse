"""Golden tests vs scikit-learn plus property tests. A metric is a rumour until it matches."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.metrics import (
    d2_tweedie_score,
    log_loss,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
)

from glasshouse.metrics import (
    binomial_deviance,
    d2,
    deviance,
    gamma_deviance,
    gaussian_deviance,
    poisson_deviance,
    tweedie_deviance,
)

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(0)
N = 5000
Y_POS = rng.gamma(2.0, 100.0, size=N)
Y_COUNT = rng.poisson(0.3, size=N).astype(float)
Y_REAL = rng.normal(0.0, 3.0, size=N)
Y_BIN = (rng.uniform(size=N) < 0.1).astype(float)
MU_POS = rng.gamma(2.0, 100.0, size=N)
MU_RATE = rng.gamma(1.2, 0.3, size=N)
MU_REAL = rng.normal(0.0, 3.0, size=N)
MU_PROB = rng.uniform(0.01, 0.99, size=N)
W = rng.uniform(0.01, 1.0, size=N)

# ----------------------------------------------------------------- golden vs scikit-learn


@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
def test_golden_named_families(w: Arr | None) -> None:
    rel = 1e-12
    assert gaussian_deviance(Y_REAL, MU_REAL, w) == pytest.approx(
        mean_squared_error(Y_REAL, MU_REAL, sample_weight=w), rel=rel
    )
    assert poisson_deviance(Y_COUNT, MU_RATE, w) == pytest.approx(
        mean_poisson_deviance(Y_COUNT, MU_RATE, sample_weight=w), rel=rel
    )
    assert gamma_deviance(Y_POS, MU_POS, w) == pytest.approx(
        mean_gamma_deviance(Y_POS, MU_POS, sample_weight=w), rel=rel
    )
    # binomial deviance on 0/1 labels is exactly twice the log-loss
    assert binomial_deviance(Y_BIN, MU_PROB, w) == pytest.approx(
        2.0 * log_loss(Y_BIN, MU_PROB, sample_weight=w), rel=rel
    )


def _tweedie_case(power: float) -> tuple[Arr, Arr]:
    """Pick y/mu inside the support for this power (zeros only below power 2, negatives < 1)."""
    if power < 0.0:
        return Y_REAL, MU_POS  # negative powers: any y, but the mean must be positive
    if power == 0.0:
        return Y_REAL, MU_REAL
    if power < 2.0:
        return Y_COUNT * 50.0, MU_POS
    return Y_POS, MU_POS


@pytest.mark.parametrize("power", [-1.0, 0.0, 1.0, 1.2, 1.5, 1.9, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
def test_golden_tweedie_powers(power: float, w: Arr | None) -> None:
    y, mu = _tweedie_case(power)
    expected = mean_tweedie_deviance(y, mu, sample_weight=w, power=power)
    assert tweedie_deviance(y, mu, power, w) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("power", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
def test_golden_d2_vs_sklearn(power: float, w: Arr | None) -> None:
    y, mu = _tweedie_case(power)
    expected = d2_tweedie_score(y, mu, sample_weight=w, power=power)
    got = d2(y, mu, family="tweedie", power=power, sample_weight=w)
    assert got == pytest.approx(expected, rel=1e-10)


def test_special_powers_equal_named_families() -> None:
    assert tweedie_deviance(Y_REAL, MU_REAL, 0.0) == gaussian_deviance(Y_REAL, MU_REAL)
    assert tweedie_deviance(Y_COUNT, MU_RATE, 1.0) == poisson_deviance(Y_COUNT, MU_RATE)
    assert tweedie_deviance(Y_POS, MU_POS, 2.0) == gamma_deviance(Y_POS, MU_POS)


# ----------------------------------------------------------------- properties

positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)
n_rows = st.integers(min_value=2, max_value=200)
FAMILIES = ["gaussian", "poisson", "gamma", "tweedie", "binomial"]


def _y_for(family: str, n: int) -> st.SearchStrategy[Arr]:
    if family == "gaussian":
        el = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)
    elif family == "poisson":
        el = st.floats(min_value=0.0, max_value=1e3, allow_nan=False)
    elif family == "binomial":
        el = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    else:
        el = positive
    return hnp.arrays(np.float64, n, elements=el)


def _mu_for(family: str, n: int) -> st.SearchStrategy[Arr]:
    if family == "gaussian":
        el = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)
    elif family == "binomial":
        el = st.floats(min_value=1e-6, max_value=1 - 1e-6, allow_nan=False)
    else:
        el = positive
    return hnp.arrays(np.float64, n, elements=el)


@pytest.mark.parametrize("family", FAMILIES)
@settings(max_examples=100, deadline=None)
@given(data=st.data(), c=st.floats(min_value=0.5, max_value=100.0))
def test_properties(family: str, data: st.DataObject, c: float) -> None:
    n = data.draw(n_rows)
    y = data.draw(_y_for(family, n))
    mu = data.draw(_mu_for(family, n))
    w = data.draw(hnp.arrays(np.float64, n, elements=positive))
    power = 1.5 if family == "tweedie" else None

    def dev(yy: Arr, mm: Arr, ww: Arr) -> float:
        return deviance(yy, mm, family=family, sample_weight=ww, power=power)  # type: ignore[arg-type]

    d = dev(y, mu, w)
    assert d >= 0.0  # deviance is never negative
    assert dev(y, mu, w * c) == pytest.approx(d, rel=1e-9)  # scaling weights changes nothing
    perm = np.random.default_rng(1).permutation(n)
    assert dev(y[perm], mu[perm], w[perm]) == pytest.approx(d, rel=1e-9)  # order-free
    # perfect fit is zero (mu is always a valid y; tweedie's power terms cancel to ~1e-15)
    assert dev(mu, mu, w) == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------- fails early and clearly


@pytest.mark.parametrize(
    ("family", "y", "mu", "w", "needle"),
    [
        ("poisson", [1.0, -1.0], [1.0, 1.0], None, "1 row(s)"),
        ("poisson", [1.0], [1.0, 1.0], None, "same length"),
        ("poisson", [1.0, 1.0], [1.0, 0.0], None, "linear predictor"),
        ("poisson", [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], "sum to more than zero"),
        ("poisson", [1.0, np.nan], [1.0, 1.0], None, "1 missing value"),
        ("gamma", [0.0, 1.0], [1.0, 1.0], None, "tweedie"),
        ("binomial", [0.0, 1.0], [0.5, 1.0], None, "probabilities"),
        ("binomial", [0.0, 2.0], [0.5, 0.5], None, "0 <= y <= 1"),
    ],
)
def test_fails_early_and_clearly(
    family: str, y: list[float], mu: list[float], w: list[float] | None, needle: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        deviance(y, mu, family=family, sample_weight=w)  # type: ignore[arg-type]


def test_bad_family_and_power() -> None:
    with pytest.raises(ValueError, match="one of: gaussian"):
        deviance([1.0], [1.0], family="weibull")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs a variance power"):
        deviance([1.0], [1.0], family="tweedie")
    with pytest.raises(ValueError, match="0 < power < 1"):
        tweedie_deviance([1.0], [1.0], power=0.5)


def test_d2_refuses_constant_y() -> None:
    with pytest.raises(ValueError, match="constant"):
        d2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], family="poisson")


def test_refuses_2d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        poisson_deviance([[1.0, 2.0]], [[1.0, 2.0]])
