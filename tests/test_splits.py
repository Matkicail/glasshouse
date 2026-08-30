"""Folds partition the rows, never overlap, respect the declared kind, and round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from glasshouse.splits import Fold, Splits, grouped, kfold, stratified, time_ordered


def _partition_ok(s: Splits) -> None:
    for f in s:
        assert np.intersect1d(f.train_idx, f.test_idx).size == 0
        assert np.array_equal(
            np.sort(np.concatenate([f.train_idx, f.test_idx])), np.arange(s.n_rows)
        ) or (s.kind == "time")


@settings(max_examples=50, deadline=None)
@given(n=st.integers(5, 200), k=st.integers(2, 5), seed=st.integers(0, 100))
def test_kfold_is_a_partition_and_seeded(n: int, k: int, seed: int) -> None:
    s = kfold(n, k=k, seed=seed)
    _partition_ok(s)
    tests = np.concatenate([f.test_idx for f in s])
    assert np.array_equal(np.sort(tests), np.arange(n))  # every row tested exactly once
    assert kfold(n, k=k, seed=seed)[0].test_idx.tolist() == s[0].test_idx.tolist()  # seeded


def test_time_ordered_never_looks_forward() -> None:
    t = np.repeat(np.arange(2015, 2025), 3).astype(float)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(t))  # rows arrive shuffled; the split must not care
    s = time_ordered(t[perm], n_folds=4)
    assert s.kind == "time"
    for f in s:
        assert t[perm][f.train_idx].max() < t[perm][f.test_idx].min()  # strictly before
        assert len(f.train_idx) >= 0.3 * len(t)
    assert len(s[-1].train_idx) > len(s[0].train_idx)  # expanding window
    covered = np.concatenate([f.test_idx for f in s])
    assert len(np.unique(covered)) == len(covered)  # each row tested at most once


def test_time_ordered_keeps_ties_together_and_refuses_nonsense() -> None:
    t = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3]
    s = time_ordered(t, n_folds=3, min_train_fraction=0.3)
    for f in s:
        assert not {t[i] for i in f.train_idx} & {t[i] for i in f.test_idx}
    with pytest.raises(ValueError, match="nothing is left to test on"):
        time_ordered([5, 5, 5, 5], n_folds=2)
    with pytest.raises(ValueError, match="missing value"):
        time_ordered([1.0, np.nan, 3.0], n_folds=2)
    with pytest.raises(ValueError, match="strictly between"):
        time_ordered([1, 2, 3, 4], min_train_fraction=1.5)


def test_grouped_keeps_entities_whole() -> None:
    groups = np.array(["p1", "p1", "p2", "p2", "p2", "p3", "p4", "p4", "p5", "p6"])
    s = grouped(groups, k=3, seed=1)
    _partition_ok(s)
    for f in s:
        assert not set(groups[f.train_idx]) & set(groups[f.test_idx])
    with pytest.raises(ValueError, match="only 2 rows/groups"):
        grouped(["a", "a", "b"], k=3)


def test_fold_refuses_overlap_and_round_trips() -> None:
    with pytest.raises(ValueError, match="share rows"):
        Fold(np.array([0, 1]), np.array([1, 2]), "random", 0)
    s = kfold(20, k=4, seed=3)
    back = Splits.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.kind == "random" and len(back) == 4
    for a, b in zip(s, back, strict=True):
        assert a.train_idx.tolist() == b.train_idx.tolist()


def test_stratified_keeps_the_class_mix_in_every_fold() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.uniform(size=1000) < 0.05).astype(int)  # 5 % positives
    s = stratified(labels, k=5, seed=1)
    _partition_ok(s)
    tests = np.concatenate([f.test_idx for f in s])
    assert np.array_equal(np.sort(tests), np.arange(1000))
    positives = labels.sum()
    for f in s:
        in_fold = labels[f.test_idx].sum()
        assert abs(in_fold - positives / 5) <= 1  # as even as integers allow
    assert stratified(labels, k=5, seed=1)[2].test_idx.tolist() == s[2].test_idx.tolist()
