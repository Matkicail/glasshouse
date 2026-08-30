"""Splits: where a dataset declares what kind of data it is, once.

Leakage is a property of the split, not the transform. A random k-fold on time-ordered rows
lets every encoder and every model "see the future" no matter how carefully they are fitted
on the training fold. So the split carries a ``kind`` — ``"random"``, ``"time"`` or
``"group"`` — and everything downstream (encoders, naive baselines, the bench) reads it.
Nothing here guesses: you say what the data is.

Folds are plain index arrays, so they can be saved, diffed and reused across models — the
same rows in the same folds, or the comparison means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from glasshouse.arrays import ArrayLike, to_vector

Kind = Literal["random", "time", "group"]
Idx = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Fold:
    """One train/test split. ``kind`` says what the data was declared to be."""

    train_idx: Idx
    test_idx: Idx
    kind: Kind
    number: int

    def __post_init__(self) -> None:
        """Refuse a fold whose sides overlap — that is leakage by construction."""
        if np.intersect1d(self.train_idx, self.test_idx).size:
            msg = f"fold {self.number}: train and test share rows"
            raise ValueError(msg)


@dataclass(frozen=True)
class Splits:
    """A list of folds plus the declaration they came from. Iterate to get the folds."""

    folds: tuple[Fold, ...]
    kind: Kind
    n_rows: int
    spec: dict[str, Any]

    def __iter__(self) -> Any:
        """Yield the folds in order."""
        return iter(self.folds)

    def __len__(self) -> int:
        """Return the number of folds."""
        return len(self.folds)

    def __getitem__(self, i: int) -> Fold:
        """Return fold ``i``."""
        return self.folds[i]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: the spec and every index array. Store it next to the benchmark."""
        return {
            "kind": self.kind,
            "n_rows": self.n_rows,
            "spec": dict(self.spec),
            "folds": [
                {"train": f.train_idx.tolist(), "test": f.test_idx.tolist()} for f in self.folds
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Splits:
        """Rebuild from :meth:`to_dict`."""
        kind: Kind = payload["kind"]
        folds = tuple(
            Fold(
                np.asarray(f["train"], dtype=np.int64),
                np.asarray(f["test"], dtype=np.int64),
                kind,
                i,
            )
            for i, f in enumerate(payload["folds"])
        )
        return cls(folds=folds, kind=kind, n_rows=int(payload["n_rows"]), spec=payload["spec"])


def kfold(n_rows: int, k: int = 5, seed: int = 0) -> Splits:
    """Random k-fold for exchangeable rows: each row is in exactly one test fold.

    Use when rows are independent draws. If they share a policy, a customer, or a clock,
    use :func:`grouped` or :func:`time_ordered` instead — this one will leak.

    Examples
    --------
    >>> from glasshouse.splits import kfold
    >>> s = kfold(10, k=5, seed=0)
    >>> len(s), s.kind, sorted(s[0].test_idx.tolist())
    (5, 'random', [4, 6])
    """
    _check_k(k, n_rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows).astype(np.int64)
    folds = tuple(
        Fold(np.sort(np.setdiff1d(perm, test)), np.sort(test), "random", i)
        for i, test in enumerate(np.array_split(perm, k))
    )
    return Splits(folds, "random", n_rows, {"method": "kfold", "k": k, "seed": seed})


def time_ordered(time: ArrayLike, n_folds: int = 5, min_train_fraction: float = 0.3) -> Splits:
    """Expanding-window folds for time-ordered data: train strictly before test, always.

    Rows are sorted by ``time``; the earliest ``min_train_fraction`` is the first training
    window; the rest is cut into ``n_folds`` consecutive test blocks, each trained on
    everything before it. Rows with the same timestamp never straddle a boundary.

    Invariant: ``train_idx`` and ``test_idx`` are returned **in time order**, not index order,
    so anything cumulative (a past-only target encoding) can treat row order as time.

    Refuses NaN or non-finite times. Does not shuffle, ever.

    Examples
    --------
    >>> from glasshouse.splits import time_ordered
    >>> s = time_ordered([2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022], n_folds=2)
    >>> [(f.train_idx.tolist(), f.test_idx.tolist()) for f in s]
    [([0, 1, 2, 3], [4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7])]
    """
    t = to_vector(time, "time")
    n_rows = len(t)
    if not 0.0 < min_train_fraction < 1.0:
        msg = "min_train_fraction must be strictly between 0 and 1"
        raise ValueError(msg)
    order = np.argsort(t, kind="stable").astype(np.int64)
    ts = t[order]
    first_cut = _next_boundary(ts, int(np.ceil(min_train_fraction * n_rows)))
    if first_cut >= n_rows:
        msg = (
            f"time: the first {min_train_fraction:.0%} of rows share a timestamp with the rest — "
            "nothing is left to test on; lower min_train_fraction or use finer time"
        )
        raise ValueError(msg)
    edges = [first_cut]
    remaining = n_rows - first_cut
    for i in range(1, n_folds):
        edges.append(_next_boundary(ts, first_cut + round(i * remaining / n_folds)))
    edges.append(n_rows)
    folds: list[Fold] = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue  # a block swallowed by a timestamp tie: fewer, honest folds
        folds.append(Fold(order[:lo].copy(), order[lo:hi].copy(), "time", len(folds)))
    if not folds:
        msg = "time: could not form a single fold; are all timestamps equal?"
        raise ValueError(msg)
    spec = {"method": "time_ordered", "n_folds": n_folds, "min_train_fraction": min_train_fraction}
    return Splits(tuple(folds), "time", n_rows, spec)


def stratified(labels: ArrayLike, k: int = 5, seed: int = 0) -> Splits:
    """Stratified k-fold: every fold has (as near as possible) the overall class mix.

    Use for classification with a rare class, so no test fold ends up with three positives.
    Rows are still treated as exchangeable — this is ``kind="random"`` with a balanced draw;
    for time or entity structure use :func:`time_ordered` or :func:`grouped`.

    Examples
    --------
    >>> from glasshouse.splits import stratified
    >>> s = stratified([0, 0, 0, 0, 1, 1, 0, 0, 1, 1], k=2, seed=0)
    >>> [sorted(f.test_idx.tolist()) for f in s]  # two positives in each test fold
    [[0, 3, 7, 8, 9], [1, 2, 4, 5, 6]]
    """
    codes, _ = _codes(labels)
    n_rows = len(codes)
    _check_k(k, n_rows)
    rng = np.random.default_rng(seed)
    row_fold = np.empty(n_rows, dtype=np.int64)
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        rows = rows[rng.permutation(len(rows))]
        # deal this class's rows round-robin, starting at a random fold so small classes
        # do not all land in fold 0
        start = int(rng.integers(k))
        row_fold[rows] = (np.arange(len(rows)) + start) % k
    all_rows = np.arange(n_rows, dtype=np.int64)
    folds = tuple(
        Fold(all_rows[row_fold != i], all_rows[row_fold == i], "random", i) for i in range(k)
    )
    return Splits(folds, "random", n_rows, {"method": "stratified", "k": k, "seed": seed})


def grouped(group: ArrayLike, k: int = 5, seed: int = 0) -> Splits:
    """Group k-fold: every row of a group lands on the same side.

    Use when several rows belong to one policy, customer, or claim — random rows would put
    the same entity on both sides and the score would be a memory test.

    Examples
    --------
    >>> from glasshouse.splits import grouped
    >>> s = grouped(["a", "a", "b", "b", "c", "c"], k=3, seed=0)
    >>> all(len(set(f.test_idx // 2)) == 1 for f in s)  # each fold tests exactly one group
    True
    """
    codes, _ = _codes(group)
    n_rows = len(codes)
    n_groups = int(codes.max()) + 1
    _check_k(k, n_groups)
    rng = np.random.default_rng(seed)
    group_fold = np.empty(n_groups, dtype=np.int64)
    perm = rng.permutation(n_groups)
    for i, block in enumerate(np.array_split(perm, k)):
        group_fold[block] = i
    row_fold = group_fold[codes]
    all_rows = np.arange(n_rows, dtype=np.int64)
    folds = tuple(
        Fold(all_rows[row_fold != i], all_rows[row_fold == i], "group", i) for i in range(k)
    )
    return Splits(folds, "group", n_rows, {"method": "grouped", "k": k, "seed": seed})


def _check_k(k: int, n: int) -> None:
    if k < 2:  # noqa: PLR2004
        msg = "k must be at least 2"
        raise ValueError(msg)
    if k > n:
        msg = f"k={k} folds but only {n} rows/groups to split"
        raise ValueError(msg)


def _next_boundary(sorted_times: npt.NDArray[np.float64], at: int) -> int:
    """Move a cut forward until it no longer splits a run of equal timestamps."""
    n = len(sorted_times)
    while 0 < at < n and sorted_times[at] == sorted_times[at - 1]:
        at += 1
    return at


def _codes(values: ArrayLike) -> tuple[Idx, list[Any]]:
    """Integer codes for arbitrary hashable labels (strings, ints), plus the levels."""
    arr = np.asarray(values.to_numpy() if hasattr(values, "to_numpy") else values)
    levels, codes = np.unique(arr.astype(str), return_inverse=True)
    return codes.astype(np.int64), levels.tolist()


__all__ = ["Fold", "Kind", "Splits", "grouped", "kfold", "stratified", "time_ordered"]
