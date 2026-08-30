"""Golden test vs scikit-learn plus property tests. A metric is a rumour until it matches."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.metrics import mean_poisson_deviance

from glasshouse.metrics import poisson_deviance

rng = np.random.default_rng(0)


def test_golden_vs_sklearn_unweighted() -> None:
    y = rng.poisson(1.5, size=1000).astype(float)
    mu = rng.gamma(2.0, 0.75, size=1000)
    assert poisson_deviance(y, mu) == pytest.approx(mean_poisson_deviance(y, mu), rel=1e-12)


def test_golden_vs_sklearn_weighted() -> None:
    y = rng.poisson(0.3, size=5000).astype(float)
    mu = rng.gamma(1.2, 0.3, size=5000)
    w = rng.uniform(0.01, 1.0, size=5000)  # exposure-like
    expected = mean_poisson_deviance(y, mu, sample_weight=w)
    assert poisson_deviance(y, mu, sample_weight=w) == pytest.approx(expected, rel=1e-12)


positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)
counts = st.floats(min_value=0.0, max_value=1e3, allow_nan=False)
n = st.shared(st.integers(min_value=1, max_value=200), key="n")


@settings(max_examples=200)
@given(
    y=hnp.arrays(np.float64, n, elements=counts),
    mu=hnp.arrays(np.float64, n, elements=positive),
    w=hnp.arrays(np.float64, n, elements=positive),
    c=st.floats(min_value=0.5, max_value=100.0),
)
def test_properties(
    y: npt.NDArray[np.float64], mu: npt.NDArray[np.float64], w: npt.NDArray[np.float64], c: float
) -> None:
    d = poisson_deviance(y, mu, sample_weight=w)
    assert d >= 0.0  # deviance is non-negative
    assert poisson_deviance(y, mu, sample_weight=w * c) == pytest.approx(d, rel=1e-9)  # scale-free
    perm = np.random.default_rng(1).permutation(len(y))
    assert poisson_deviance(y[perm], mu[perm], sample_weight=w[perm]) == pytest.approx(d, rel=1e-9)


def test_zero_at_perfect_fit() -> None:
    mu = np.array([0.2, 1.0, 3.0])
    assert poisson_deviance(mu, mu) == 0.0


@pytest.mark.parametrize(
    ("y", "mu", "w", "needle"),
    [
        ([1.0, -1.0], [1.0, 1.0], None, "1 row(s)"),
        ([1.0], [1.0, 1.0], None, "same length"),
        ([1.0, 1.0], [1.0, 0.0], None, "mu"),
        ([1.0, 1.0], [1.0, 1.0], [0.0, 0.0], "sum to more than zero"),
        ([1.0, np.nan], [1.0, 1.0], None, "1 row(s)"),
    ],
)
def test_fails_early_and_clearly(
    y: list[float], mu: list[float], w: list[float] | None, needle: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        poisson_deviance(y, mu, sample_weight=w)


def test_accepts_2d_refusal() -> None:
    with pytest.raises(ValueError, match="1-D"):
        poisson_deviance([[1.0, 2.0]], [[1.0, 2.0]])
