"""Classification golden tests vs scikit-learn (all weighted), plus properties."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn import metrics as sk

from glasshouse.classification import (
    average_precision,
    brier,
    ks,
    log_loss,
    roc_auc,
    threshold_metrics,
)

Arr = npt.NDArray[np.float64]
rng = np.random.default_rng(3)
N = 4000
PROB = rng.beta(0.5, 4.0, size=N)  # imbalanced, rare-event
Y = (rng.uniform(size=N) < PROB).astype(float)
SCORE = np.clip(PROB + rng.normal(0, 0.1, size=N), 1e-6, 1 - 1e-6)
W = rng.uniform(0.1, 2.0, size=N)


@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
def test_golden_threshold_metrics(w: Arr | None, threshold: float) -> None:
    pred = (threshold <= SCORE).astype(float)
    m = threshold_metrics(Y, SCORE, sample_weight=w, threshold=threshold)
    kw = {"sample_weight": w}
    assert m.accuracy == pytest.approx(sk.accuracy_score(Y, pred, **kw), rel=1e-12)
    assert m.balanced_accuracy == pytest.approx(
        sk.balanced_accuracy_score(Y, pred, **kw), rel=1e-12
    )
    assert m.precision == pytest.approx(sk.precision_score(Y, pred, **kw), rel=1e-12)
    assert m.recall == pytest.approx(sk.recall_score(Y, pred, **kw), rel=1e-12)
    assert m.f1 == pytest.approx(sk.f1_score(Y, pred, **kw), rel=1e-12)
    assert m.mcc == pytest.approx(sk.matthews_corrcoef(Y, pred, **kw), rel=1e-10)


@pytest.mark.parametrize("w", [None, W], ids=["unweighted", "weighted"])
def test_golden_ranking_and_probability_metrics(w: Arr | None) -> None:
    kw = {"sample_weight": w}
    assert roc_auc(Y, SCORE, w) == pytest.approx(sk.roc_auc_score(Y, SCORE, **kw), rel=1e-12)
    assert average_precision(Y, SCORE, w) == pytest.approx(
        sk.average_precision_score(Y, SCORE, **kw), rel=1e-12
    )
    assert log_loss(Y, SCORE, w) == pytest.approx(sk.log_loss(Y, SCORE, **kw), rel=1e-12)
    assert brier(Y, SCORE, w) == pytest.approx(sk.brier_score_loss(Y, SCORE, **kw), rel=1e-12)


def test_golden_ks_vs_roc_curve() -> None:
    fpr, tpr, _ = sk.roc_curve(Y, SCORE)
    assert ks(Y, SCORE) == pytest.approx(float(np.max(tpr - fpr)), rel=1e-12)


def test_ties_handled_like_sklearn() -> None:
    score = np.round(SCORE, 1)  # lots of ties
    assert roc_auc(Y, score) == pytest.approx(sk.roc_auc_score(Y, score), rel=1e-12)
    assert average_precision(Y, score) == pytest.approx(
        sk.average_precision_score(Y, score), rel=1e-12
    )


positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False)


@settings(max_examples=100, deadline=None)
@given(data=st.data(), c=st.floats(min_value=0.5, max_value=50.0))
def test_properties(data: st.DataObject, c: float) -> None:
    n = data.draw(st.integers(min_value=2, max_value=100))
    y = data.draw(hnp.arrays(np.float64, n, elements=st.sampled_from([0.0, 1.0])))
    s = data.draw(hnp.arrays(np.float64, n, elements=st.floats(1e-6, 1 - 1e-6, allow_nan=False)))
    w = data.draw(hnp.arrays(np.float64, n, elements=positive))
    if y.min() == y.max():
        return  # one class: ranking metrics refuse, by design
    perm = np.random.default_rng(0).permutation(n)
    for fn in (roc_auc, average_precision, ks):
        v = fn(y, s, w)
        assert 0.0 <= v <= 1.0
        assert fn(y, s, w * c) == pytest.approx(v, rel=1e-9)  # weight scale-free
        assert fn(y[perm], s[perm], w[perm]) == pytest.approx(v, rel=1e-9)  # order-free
        assert fn(y, s * 0.5, w) == pytest.approx(v, rel=1e-9)  # monotone (and exact) in score
    assert roc_auc(y, -s, w) == pytest.approx(1 - roc_auc(y, s, w), abs=1e-12)  # reversal, exact
    m = threshold_metrics(y, s, w)
    assert -1.0 <= m.mcc <= 1.0
    assert m.tp + m.fp + m.fn + m.tn == pytest.approx(w.sum(), rel=1e-9)


@pytest.mark.parametrize(
    ("y", "s", "needle"),
    [
        ([0.5, 1.0], [0.1, 0.9], "exactly 0 or 1"),
        ([1.0, 1.0], [0.1, 0.9], "both classes"),
        ([0.0, 1.0], [0.1, np.nan], "cannot be scored"),
        ([0.0], [0.1, 0.2], "same length"),
    ],
)
def test_fails_early_and_clearly(y: list[float], s: list[float], needle: str) -> None:
    with pytest.raises(ValueError, match=re.escape(needle)):
        roc_auc(y, s)


def test_undefined_ratios_are_zero_like_sklearn() -> None:
    m = threshold_metrics([1.0, 1.0, 0.0], [0.1, 0.2, 0.3])  # nothing flagged at 0.5
    assert (m.precision, m.recall, m.f1, m.mcc) == (0.0, 0.0, 0.0, 0.0)
