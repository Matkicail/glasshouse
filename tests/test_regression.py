"""Regression golden tests vs scikit-learn (weighted), sMAPE vs its formula, plus properties."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn import metrics as sk

from glasshouse.regression import mae, mape, msle, r2, rmse, smape

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(4)
N = 3000
Y = rng.gamma(2.0, 50.0, size=N)
MU = Y * rng.lognormal(0.0, 0.3, size=N)
W = rng.uniform(0.1, 2.0, size=N)


@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
def test_golden_vs_sklearn(w: Arr | None) -> None:
    kw = {"sample_weight": w}
    assert rmse(Y, MU, w) == pytest.approx(sk.root_mean_squared_error(Y, MU, **kw), rel=1e-12)
    assert mae(Y, MU, w) == pytest.approx(sk.mean_absolute_error(Y, MU, **kw), rel=1e-12)
    assert mape(Y, MU, w) == pytest.approx(
        sk.mean_absolute_percentage_error(Y, MU, **kw), rel=1e-12
    )
    assert msle(Y, MU, w) == pytest.approx(sk.mean_squared_log_error(Y, MU, **kw), rel=1e-12)
    assert r2(Y, MU, w) == pytest.approx(sk.r2_score(Y, MU, **kw), rel=1e-12)


def test_smape_vs_formula() -> None:
    expected = np.average(2 * np.abs(Y - MU) / (np.abs(Y) + np.abs(MU)), weights=W)
    assert smape(Y, MU, W) == pytest.approx(float(expected), rel=1e-12)


positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)


@settings(max_examples=100, deadline=None)
@given(data=st.data(), c=st.floats(min_value=0.5, max_value=50.0))
def test_properties(data: st.DataObject, c: float) -> None:
    n = data.draw(st.integers(min_value=2, max_value=100))
    y = data.draw(hnp.arrays(np.float64, n, elements=positive))
    mu = data.draw(hnp.arrays(np.float64, n, elements=positive))
    w = data.draw(hnp.arrays(np.float64, n, elements=positive))
    perm = np.random.default_rng(0).permutation(n)
    for fn in (rmse, mae, mape, smape, msle):
        v = fn(y, mu, w)
        assert v >= 0.0
        assert fn(y, y, w) == 0.0  # perfect fit
        assert fn(y, mu, w * c) == pytest.approx(v, rel=1e-9)  # weight scale-free
        assert fn(y[perm], mu[perm], w[perm]) == pytest.approx(v, rel=1e-9)  # order-free
    assert smape(y, mu, w) <= 2.0
    assert smape(y, mu, w) == pytest.approx(smape(mu, y, w), rel=1e-9)  # symmetric
    if y.min() != y.max():
        assert r2(y, y, w) == pytest.approx(1.0, abs=1e-12)
        ybar = np.full(n, np.average(y, weights=w))
        assert r2(y, ybar, w) == pytest.approx(0.0, abs=1e-9)  # the mean scores zero


@pytest.mark.parametrize(
    ("fn", "y", "mu", "needle"),
    [
        (mape, [0.0, 1.0], [1.0, 1.0], "infinite"),
        (smape, [0.0, 1.0], [0.0, 1.0], "0/0"),
        (msle, [-2.0, 1.0], [1.0, 1.0], "MSLE"),
        (r2, [1.0, 1.0], [1.0, 2.0], "constant"),
        (mae, [1.0], [1.0, 2.0], "same length"),
        (rmse, [1.0, np.inf], [1.0, 2.0], "1 infinite value"),
    ],
)
def test_fails_early_and_clearly(fn: object, y: list[float], mu: list[float], needle: str) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        fn(y, mu)  # type: ignore[operator]
