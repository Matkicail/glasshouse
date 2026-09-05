"""Data profiles: the weight and mean outcome per bin add back up to the whole."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from glasshouse.profile import feature_profile, histogram

rng = np.random.default_rng(7)
N = 2000
AGE = rng.uniform(18.0, 80.0, size=N)
REGION = rng.choice([f"r{i:02d}" for i in range(40)], size=N)
W = rng.uniform(0.1, 1.0, size=N)
Y = rng.poisson(0.1 * np.exp(0.02 * (AGE - 45.0)), size=N) / W


def test_numeric_profile_partitions_the_weight_and_the_outcome() -> None:
    p = feature_profile(AGE, Y, W, name="age", n_bins=12)
    assert p.kind == "numeric" and p.edges is not None
    assert len(p.level) == 13, "12 even bins to the 99th percentile, then the tail to the max"
    assert p.edges[0] == AGE.min() and p.edges[-1] == AGE.max()
    assert np.allclose(np.diff(p.edges[:-1]), np.diff(p.edges[:-1])[0]), "even widths"
    assert p.weight[:-1].sum() / p.weight.sum() == pytest.approx(0.99, abs=0.01)
    assert p.level[-1].startswith("[") and p.level[-1].endswith("]")
    assert np.all(np.diff(p.edges) > 0) and p.n_levels == N
    assert p.n_rows.sum() == N
    assert p.weight.sum() == pytest.approx(W.sum())
    # the bins' weighted means recombine to the overall weighted mean
    assert (p.weight * p.actual).sum() / p.weight.sum() == pytest.approx((W * Y).sum() / W.sum())
    # the outcome rises with age, so the last bin's mean is above the first's
    assert p.actual[-1] > p.actual[0]
    assert p.level[0].startswith("[18")
    assert "age" in str(p)
    # no tail when the 99th percentile is the maximum (few distinct values): plain even bins
    small = feature_profile(np.repeat([1.0, 2.0, 3.0], 100), np.zeros(300), name="k", n_bins=4)
    assert len(small.level) == 4 and small.edges is not None and small.edges[-1] == 3.0


def test_categorical_profile_is_heaviest_first_and_pools_the_tail() -> None:
    p = feature_profile(REGION, Y, W, name="region")
    assert p.kind == "categorical" and p.edges is None and p.n_levels == 40
    assert len(p.level) == 31 and p.level[-1] == "(other: 10 levels)"
    assert np.all(np.diff(p.weight[:-1]) <= 0), "shown levels run heaviest first"
    assert p.weight.sum() == pytest.approx(W.sum()) and p.n_rows.sum() == N
    assert (p.weight * p.actual).sum() / p.weight.sum() == pytest.approx((W * Y).sum() / W.sum())
    full = feature_profile(REGION, Y, W, name="region", max_levels=40)
    assert len(full.level) == 40 and full.level[-1] != "(other: 0 levels)"
    unweighted = feature_profile(REGION, Y, name="region")
    assert unweighted.weight.sum() == N


def test_histogram_summary_is_weighted() -> None:
    h = histogram(Y, W, name="y")
    assert len(h.edges) == 42 and len(h.level) == 41, "40 even bins and the pooled tail"
    assert h.n_rows.sum() == N and h.weight.sum() == pytest.approx(W.sum())
    assert h.edges[-1] == Y.max() and h.edges[-2] < Y.max()
    assert h.summary["mean"] == pytest.approx((W * Y).sum() / W.sum())
    assert h.summary["zero_share"] == pytest.approx(W[Y == 0].sum() / W.sum())
    assert h.summary["min"] == 0.0 and h.summary["max"] == Y.max()
    assert h.summary["q05"] <= h.summary["median"] <= h.summary["q95"]
    # a weighted median: the value where the cumulative weight crosses half
    heavy = histogram([1.0, 2.0, 3.0], [1.0, 1.0, 10.0])
    assert heavy.summary["median"] == 3.0


def test_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        feature_profile([1.0, np.nan], [1.0, 2.0], name="x")
    with pytest.raises(ValueError, match="rows but y"):
        feature_profile([1.0, 2.0, 3.0], [1.0, 2.0], name="x")
    with pytest.raises(ValueError, match="sample_weight has"):
        histogram([1.0, 2.0], [1.0])


@settings(max_examples=100, deadline=None)
@given(
    values=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=60),
    n_bins=st.integers(1, 15),
)
def test_histogram_bins_partition_every_row(values: list[float], n_bins: int) -> None:
    h = histogram(values, name="v", n_bins=n_bins)
    assert h.n_rows.sum() == len(values) and h.weight.sum() == pytest.approx(len(values))
    assert len(h.edges) in (n_bins + 1, n_bins + 2) and len(h.level) == len(h.edges) - 1
    assert np.all(np.diff(h.edges) >= 0)
    assert (h.edges[0] == min(values) and h.edges[-1] == max(values)) or len(set(values)) == 1
    assert h.summary["min"] <= h.summary["mean"] <= h.summary["max"]
