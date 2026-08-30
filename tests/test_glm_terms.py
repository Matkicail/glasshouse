"""terms= and fold= on the GLM: encoders fit on train only, and time folds go cumulative."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, splits
from glasshouse.encoders import TargetEncode

rng = np.random.default_rng(8)
N = 4000
REGION_EFFECT = {"north": 0.0, "south": 0.3, "east": -0.2, "west": 0.5}
df = pd.DataFrame(
    {
        "year": rng.integers(2015, 2023, size=N),
        "region": rng.choice(list(REGION_EFFECT), size=N),
        "brand": rng.choice([f"b{i}" for i in range(30)], size=N),  # high-cardinality
        "age": rng.uniform(18, 80, size=N),
    }
)
eta = -2.0 + df.region.map(REGION_EFFECT) + 0.01 * (df.age - 40)
y = rng.poisson(np.exp(eta)).astype(float)


def test_onehot_terms_recover_the_region_effects() -> None:
    m = GLM(family="poisson", terms={"region": "onehot", "brand": "target"}).fit(df, y)
    names = m.feature_names_in_
    assert names == [
        "intercept",
        "year",
        "region=north",
        "region=south",
        "region=west",  # east is the dropped reference level
        "brand_te",
        "age",
    ]
    coef = dict(zip(names, np.r_[m.intercept_, m.coef_], strict=True))
    # relative to east: north +0.2, south +0.5, west +0.7 (within a few SEs)
    assert coef["region=north"] == pytest.approx(0.2, abs=0.15)
    assert coef["region=west"] == pytest.approx(0.7, abs=0.15)
    assert m.predict(df).shape == (N,)


def test_fold_fits_on_train_rows_only_and_predicts_on_test() -> None:
    folds = splits.kfold(N, k=4, seed=0)
    f = folds[1]
    x = df[["region", "age"]]
    m = GLM(family="poisson", terms={"region": "onehot"}).fit(x, y, fold=f)
    assert m._fit["n_rows"] == len(f.train_idx)
    by_hand = GLM(family="poisson", terms={"region": "onehot"}).fit(
        x.iloc[f.train_idx], y[f.train_idx]
    )
    np.testing.assert_allclose(m.coef_, by_hand.coef_, rtol=1e-10)
    pred = m.predict(x.iloc[f.test_idx])
    assert pred.shape == (len(f.test_idx),)


def test_time_fold_makes_target_encoding_cumulative() -> None:
    folds = splits.time_ordered(df.year, n_folds=3)
    f = folds[0]
    x = df[["brand", "age"]]
    m = GLM(family="poisson", terms={"brand": "target"}).fit(x, y, fold=f)
    enc = m.encoders_["brand"]
    assert isinstance(enc, TargetEncode)
    assert enc.cumulative is True  # read from fold.kind, not from the user
    # first training row in time order has nothing before it: no information, not the mean
    assert enc.training_encoding_[0] == 0.0
    m2 = GLM(family="poisson", terms={"brand": "target"}).fit(x, y, fold=splits.kfold(N, k=3)[0])
    enc2 = m2.encoders_["brand"]
    assert isinstance(enc2, TargetEncode) and enc2.cumulative is False


def test_terms_fail_early_and_clearly() -> None:
    with pytest.raises(ValueError, match="encode categoricals first"):
        GLM(family="poisson").fit(df, y)  # strings without a term
    with pytest.raises(ValueError, match="not in X"):
        GLM(family="poisson", terms={"colour": "onehot"}).fit(df, y)
    with pytest.raises(ValueError, match="unknown term"):
        GLM(family="poisson", terms={"region": "wavelet"}).fit(df, y)
    with pytest.raises(ValueError, match="pass a DataFrame"):
        GLM(family="poisson", terms={"x0": "onehot"}).fit(np.ones((10, 1)), np.ones(10))
    m = GLM(family="poisson", terms={"region": "onehot", "brand": "target"}).fit(df, y)
    with pytest.raises(ValueError, match="needs one too"):
        m.predict(df.to_numpy())
    with pytest.raises(ValueError, match="not seen at fit"):
        m.predict(df.assign(region="mars").head())


def test_round_trip_with_encoders() -> None:
    m = GLM(
        family="poisson", terms={"region": "onehot", "brand": "target", "age": "standardize"}
    ).fit(df, y)
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(df), m.predict(df), rtol=1e-12)
    assert back.feature_names_in_ == m.feature_names_in_
