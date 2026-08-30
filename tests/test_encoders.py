"""Encoders fit on train only, never let a row see its own y, and match named references."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from glasshouse import encoders
from glasshouse.encoders import OneHot, Standardize, TargetEncode

rng = np.random.default_rng(7)


def test_onehot_matches_pandas_get_dummies_and_refuses_unseen() -> None:
    x = pd.Series(rng.choice(["north", "south", "east", "west"], size=200))
    m, names = OneHot(name="region").fit_transform(x)
    ref = pd.get_dummies(x, prefix="region", prefix_sep="=", drop_first=True, dtype=float)
    assert names == list(ref.columns)
    np.testing.assert_array_equal(m, ref.to_numpy())
    enc = OneHot(name="region").fit(x)
    with pytest.raises(ValueError, match=r"not seen at fit: \['mars'\]"):
        enc.transform(["north", "mars"])
    m2, _ = OneHot(name="region", unknown="zero").fit(x).transform(["mars"])
    assert m2.sum() == 0.0
    with pytest.raises(ValueError, match="single level"):
        OneHot().fit(["a", "a"])
    with pytest.raises(ValueError, match="missing value"):
        OneHot().fit(pd.Series(["a", None, "b"]))


def test_target_encoding_never_sees_its_own_y() -> None:
    x = rng.choice(["a", "b", "c"], size=300)
    y = rng.normal(size=300)
    enc = TargetEncode(smoothing=2.0, seed=1).fit(x, y)
    y2 = y.copy()
    y2[17] += 1000.0  # move one row's outcome a mile
    enc2 = TargetEncode(smoothing=2.0, seed=1).fit(x, y2)
    # that row's own training encoding cannot have moved (it is out-of-fold)
    assert enc.training_encoding_[17] == enc2.training_encoding_[17]
    # but the full table used for new rows did
    assert enc.table_["a" if x[17] == "a" else x[17]] != enc2.table_[x[17]]


def test_target_encoding_transform_is_the_smoothed_mean() -> None:
    x = np.array(["a", "a", "b", "b", "b"])
    y = np.array([1.0, 3.0, 10.0, 12.0, 14.0])
    w = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    enc = TargetEncode(smoothing=1.0).fit(x, y, sample_weight=w)
    prior = np.sum(w * y) / np.sum(w)
    expected_a = (1 * 1 + 1 * 3 + 1.0 * prior) / (2 + 1.0)
    m, names = enc.transform(["a", "zzz"])
    assert names == ["x_te"]
    assert m[0, 0] == pytest.approx(expected_a)
    assert m[1, 0] == pytest.approx(prior)  # unseen level → prior


def test_cumulative_encoding_only_uses_the_past() -> None:
    x = np.array(["a", "b", "a", "a", "b"])
    y = np.array([1.0, 10.0, 3.0, 5.0, 20.0])
    enc = TargetEncode(smoothing=1.0, cumulative=True).fit(x, y)
    te = enc.training_encoding_
    assert te[0] == 0.0  # nothing before row 0: no information (the global mean is the future)
    # row 2 ('a'): only row 0 ('a', 1.0) is in the past; running prior = mean(1, 10) = 5.5
    assert te[2] == pytest.approx((1.0 + 1.0 * 5.5) / (1.0 + 1.0))
    # row 4 ('b'): past 'b' = row 1 (10); running prior = mean(1, 10, 3, 5) = 4.75
    assert te[4] == pytest.approx((10.0 + 1.0 * 4.75) / (1.0 + 1.0))
    # perturbing the future cannot change the past
    y2 = y.copy()
    y2[4] = 999.0
    te2 = TargetEncode(smoothing=1.0, cumulative=True).fit(x, y2).training_encoding_
    np.testing.assert_array_equal(te[:4], te2[:4])


def test_standardize_matches_sklearn_and_refuses_constant() -> None:
    v = rng.normal(3.0, 2.0, size=500)
    m, names = Standardize(name="age").fit_transform(v)
    ref = StandardScaler().fit_transform(v[:, None])
    np.testing.assert_allclose(m, ref, rtol=1e-12)
    assert names == ["age_std"]
    with pytest.raises(ValueError, match="constant"):
        Standardize().fit([1.0, 1.0, 1.0])


def test_registry_and_round_trip() -> None:
    for spec, x, y in [
        ("onehot", ["a", "b", "c"], None),
        ("target", ["a", "b", "a"], [1.0, 2.0, 3.0]),
        ("standardize", [1.0, 2.0, 4.0], None),
    ]:
        enc = encoders.make(spec, "col").fit(x, y)
        back = encoders.from_dict(json.loads(json.dumps(enc.to_dict())))
        a, _ = enc.transform(x)
        b, _ = back.transform(x)
        np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="unknown term"):
        encoders.make("wavelet", "col")
