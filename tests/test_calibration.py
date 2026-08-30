"""Calibration golden tests vs sklearn's quantile reliability curve, plus properties."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.calibration import calibration_curve

from glasshouse.metrics import balance, calibration_table

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(2)


def test_golden_vs_sklearn_quantile_bins() -> None:
    # tie-free probabilities and n divisible by n_bins: sklearn's percentile edges and our
    # equal-weight cuts put exactly the same rows in each bin
    prob = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < prob).astype(float)
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    t = calibration_table(y, prob, n_bins=10)
    assert len(t) == 10
    np.testing.assert_allclose(t.actual, frac_pos, rtol=1e-12)
    np.testing.assert_allclose(t.predicted, mean_pred, rtol=1e-12)
    assert t.n_rows.tolist() == [200] * 10


def test_weighted_matches_row_expansion() -> None:
    """Exposure k on one row must equal that row repeated k times (y and mu both rates)."""
    rate = np.array([0.0, 1.0, 1 / 3, 0.0, 1.5])
    mu = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    w = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    k = w.astype(int)
    a = calibration_table(rate, mu, sample_weight=w, n_bins=3)
    b = calibration_table(np.repeat(rate, k), np.repeat(mu, k), n_bins=3)
    np.testing.assert_allclose(a.weight, b.weight)
    np.testing.assert_allclose(a.predicted, b.predicted)
    np.testing.assert_allclose(a.actual, b.actual)
    assert balance(rate, mu, sample_weight=w) == pytest.approx(
        balance(np.repeat(rate, k), np.repeat(mu, k))
    )


def test_bins_partition_the_data_and_reproduce_balance() -> None:
    y = rng.poisson(0.4, size=1000).astype(float)
    mu = rng.gamma(2.0, 0.2, size=1000)
    w = rng.uniform(0.1, 1.0, size=1000)
    t = calibration_table(y, mu, sample_weight=w)
    assert t.n_rows.sum() == 1000
    assert t.weight.sum() == pytest.approx(w.sum())
    overall = (t.actual * t.weight).sum() / (t.predicted * t.weight).sum()
    assert overall == pytest.approx(balance(y, mu, sample_weight=w), rel=1e-12)
    assert np.all(np.diff(t.predicted) >= 0)  # bins are ordered by prediction


positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)


@settings(max_examples=100, deadline=None)
@given(data=st.data(), n_bins=st.integers(min_value=1, max_value=12))
def test_properties(data: st.DataObject, n_bins: int) -> None:
    n = data.draw(st.integers(min_value=1, max_value=80))
    y = data.draw(hnp.arrays(np.float64, n, elements=st.floats(0, 1e3, allow_nan=False)))
    mu = data.draw(hnp.arrays(np.float64, n, elements=positive))
    w = data.draw(hnp.arrays(np.float64, n, elements=positive))
    t = calibration_table(y, mu, sample_weight=w, n_bins=n_bins)
    assert 1 <= len(t) <= n_bins
    assert t.n_rows.sum() == n
    assert t.weight.sum() == pytest.approx(w.sum(), rel=1e-9)
    assert np.all(np.diff(t.predicted) >= -1e-12)
    perm = np.random.default_rng(0).permutation(n)
    t2 = calibration_table(y[perm], mu[perm], sample_weight=w[perm], n_bins=n_bins)
    np.testing.assert_allclose(t.predicted, t2.predicted, rtol=1e-9)  # order-free
    # a perfectly calibrated prediction (mu == y) scores 1 in every bin
    if y.sum() > 0:
        p = calibration_table(y, y, sample_weight=w, n_bins=n_bins)
        ok = p.predicted > 0
        np.testing.assert_allclose(p.actual_over_expected[ok], 1.0, rtol=1e-9)


@pytest.mark.parametrize(
    ("y", "mu", "w", "kw", "needle"),
    [
        ([1.0, 1.0], [1.0, np.nan], None, {}, "1 row(s)"),
        ([1.0], [1.0, 1.0], None, {}, "same length"),
        ([1.0, 1.0], [1.0, 1.0], [1.0, 0.0], {}, "zero weight"),
        ([1.0, 1.0], [1.0, -1.0], None, {}, "non-zero expected total"),
        ([1.0, 1.0], [1.0, 1.0], None, {"n_bins": 0}, "at least 1"),
    ],
)
def test_fails_early_and_clearly(
    y: list[float], mu: list[float], w: list[float] | None, kw: dict[str, int], needle: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        calibration_table(y, mu, sample_weight=w, **kw)
