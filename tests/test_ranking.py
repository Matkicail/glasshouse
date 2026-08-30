"""Gini golden tests: vs sklearn AUC (binary case) and the Kaggle normalized-Gini reference."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.metrics import roc_auc_score

from glasshouse.metrics import gini, normalized_gini

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(1)


def kaggle_gini(actual: Arr, pred: Arr) -> float:
    """The widely used Kaggle reference (Porto Seguro / Allstate), verbatim in spirit.

    Sorts by prediction descending, then by original index (so it is only well-defined without
    ties); Gini is the mean of the cumulative-actual curve minus that of the diagonal.
    """
    n = len(actual)
    order = np.lexsort((np.arange(n), -pred))
    a = actual[order]
    cum = np.cumsum(a) / a.sum()
    return float(cum.sum() / n - (n + 1) / (2 * n))


def kaggle_normalized_gini(actual: Arr, pred: Arr) -> float:
    return kaggle_gini(actual, pred) / kaggle_gini(actual, actual)


def test_golden_binary_vs_sklearn_auc() -> None:
    y = (rng.uniform(size=4000) < 0.15).astype(float)
    score = rng.uniform(size=4000)
    somers_d = 2 * roc_auc_score(y, score) - 1
    # normalised Gini IS the accuracy ratio; the raw one is scaled by the perfect triangle's height
    assert normalized_gini(y, score) == pytest.approx(somers_d, rel=1e-12)
    assert gini(y, score) == pytest.approx(somers_d * (1 - y.mean()), rel=1e-12)


def test_golden_vs_kaggle_reference_counts() -> None:
    y = rng.poisson(0.4, size=3000).astype(float)
    score = rng.uniform(size=3000)  # tie-free, as the reference requires
    # Kaggle's step curve minus its (n+1)/2n diagonal is exactly half our trapezoid Gini.
    assert gini(y, score) == pytest.approx(2 * kaggle_gini(y, score), rel=1e-9)
    assert normalized_gini(y, score) == pytest.approx(kaggle_normalized_gini(y, score), rel=1e-9)


def test_weighted_gini_matches_row_expansion() -> None:
    """An exposure of k on one row must equal that row repeated k times with y split evenly."""
    y = np.array([0.0, 2.0, 1.0, 0.0])
    score = np.array([0.1, 0.9, 0.5, 0.3])
    w = np.array([1.0, 2.0, 3.0, 1.0])
    y_x = np.repeat(y / w, w.astype(int)) * 1.0
    s_x = np.repeat(score, w.astype(int))
    assert gini(y, score, sample_weight=w) == pytest.approx(gini(y_x, s_x), rel=1e-12)


def test_ties_are_order_free() -> None:
    y = np.array([1.0, 0.0, 3.0, 0.0, 2.0])
    s = np.array([0.2, 0.2, 0.9, 0.2, 0.9])
    perm = np.array([3, 0, 4, 1, 2])
    assert gini(y, s) == gini(y[perm], s[perm])


positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)
nonneg = st.floats(min_value=0.0, max_value=1e3, allow_nan=False)


@settings(max_examples=150, deadline=None)
@given(data=st.data(), c=st.floats(min_value=0.5, max_value=50.0))
def test_properties(data: st.DataObject, c: float) -> None:
    n = data.draw(st.integers(min_value=2, max_value=100))
    y = data.draw(hnp.arrays(np.float64, n, elements=nonneg))
    s = data.draw(hnp.arrays(np.float64, n, elements=st.floats(-1e3, 1e3, allow_nan=False)))
    w = data.draw(hnp.arrays(np.float64, n, elements=positive))
    if y.sum() <= 0:
        return
    g = gini(y, s, sample_weight=w)
    assert -1.0 <= g <= 1.0
    assert gini(y, s, sample_weight=w * c) == pytest.approx(g, rel=1e-9)  # weight scale-free
    assert gini(y, s * 3, sample_weight=w) == pytest.approx(g, rel=1e-9)  # monotone in score
    assert gini(y, -s, sample_weight=w) == pytest.approx(-g, abs=1e-12)  # reversed ranking
    perm = np.random.default_rng(0).permutation(n)
    assert gini(y[perm], s[perm], sample_weight=w[perm]) == pytest.approx(g, rel=1e-9)
    # the perfect ranking is the ceiling
    assert gini(y, y / w, sample_weight=w) >= g - 1e-12


@pytest.mark.parametrize(
    ("y", "s", "w", "needle"),
    [
        ([0.0, -1.0], [0.1, 0.2], None, "1 row(s)"),
        ([0.0, 1.0], [0.1, np.inf], None, "cannot be ranked"),
        ([0.0, 0.0], [0.1, 0.2], None, "no positive outcomes"),
        ([0.0, 1.0], [0.1, 0.2], [1.0, 0.0], "exposure of zero"),
        ([1.0], [0.1, 0.2], None, "same length"),
    ],
)
def test_fails_early_and_clearly(
    y: list[float], s: list[float], w: list[float] | None, needle: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        gini(y, s, sample_weight=w)


def test_normalized_refuses_constant_rate() -> None:
    with pytest.raises(ValueError, match="same rate"):
        normalized_gini([1.0, 1.0, 1.0], [0.1, 0.5, 0.9])
