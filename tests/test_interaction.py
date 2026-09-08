"""The interaction term: a tensor-product spline that finds what the main effects cannot."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, splits
from glasshouse.encoders import Interaction
from glasshouse.metrics import deviance

rng = np.random.default_rng(31)
N = 8000
DF = pd.DataFrame(
    {
        "region": rng.choice(["n", "s", "e"], size=N),
        "age": rng.uniform(-1.0, 1.0, size=N),
        "power": rng.uniform(-1.0, 1.0, size=N),
        "Exposure": rng.uniform(0.3, 1.0, size=N),
    }
)
ETA = -1.5 + 0.3 * DF.age + DF.region.map({"n": 0.0, "s": 0.4, "e": -0.3}) + 1.2 * DF.age * DF.power
DF["ClaimNb"] = rng.poisson(np.exp(ETA) * DF.Exposure).astype(float)
COLS = ["region", "age", "power"]
OFFSET = np.log(DF.Exposure.to_numpy())
MAIN: dict[str, Any] = {"region": "onehot", "age": "smooth", "power": "smooth"}


def _held_out(model: GLM, te: np.ndarray) -> float:
    rate = DF.ClaimNb.to_numpy()[te] / DF.Exposure.to_numpy()[te]
    w = DF.Exposure.to_numpy()[te]
    return float(
        deviance(rate, model.predict(DF[COLS].iloc[te]), family="poisson", sample_weight=w)
    )


def test_the_interaction_term_finds_the_surface_the_main_effects_cannot() -> None:
    fold = splits.kfold(N, k=4, seed=0)[0]
    main = GLM(family="poisson", terms=dict(MAIN)).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold
    )
    both = GLM(family="poisson", terms={**MAIN, "age*power": Interaction(df=4)}).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold
    )
    te = fold.test_idx
    assert _held_out(both, te) < _held_out(main, te) * 0.97, (
        _held_out(both, te),
        _held_out(main, te),
    )
    assert both._slices["age*power"][1] - both._slices["age*power"][0] == 16
    assert [n for n in both.feature_names_in_ if n.startswith("age*power")][:2] == [
        "age*power_1_1",
        "age*power_1_2",
    ]
    # the term reads as one bar on "explain a row", and the string form builds the default
    parts, names = both.term_contributions(DF[COLS].iloc[:20])
    assert names == ["intercept", "region", "age", "power", "age*power"]
    np.testing.assert_allclose(
        parts.sum(axis=1), both.predict_linear(DF[COLS].iloc[:20]), rtol=1e-10
    )
    default = GLM(family="poisson", terms={**MAIN, "age*power": "interaction"}).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold
    )
    np.testing.assert_allclose(default.predict(DF[COLS]), both.predict(DF[COLS]), rtol=1e-10)


def test_round_trip_group_lasso_and_refusals() -> None:
    m = GLM(family="poisson", terms={"region": "onehot", "age*power": Interaction(df=4)}).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET
    )
    back = GLM.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(DF[COLS]), m.predict(DF[COLS]), rtol=1e-12)
    assert (
        back.term_contributions(DF[COLS].iloc[:5])[1] == m.term_contributions(DF[COLS].iloc[:5])[1]
    )
    # no signal in the interaction: the group lasso switches the whole 16-column block off
    quiet = DF.copy()
    quiet["ClaimNb"] = rng.poisson(np.exp(-1.5 + 0.5 * quiet.age) * quiet.Exposure).astype(float)
    lasso = GLM(
        family="poisson",
        terms={"age": "standardize", "power": "standardize", "age*power": Interaction(df=4)},
        alpha=0.02,
        l1_ratio=1.0,
        group_lasso=True,
    ).fit(quiet[COLS[1:]], quiet.ClaimNb, offset=OFFSET)
    lo, hi = lasso._slices["age*power"]
    assert np.all(lasso.coef_[lo:hi] == 0.0) and lasso.coef_[lasso._slices["age"][0]] != 0.0
    with pytest.raises(ValueError, match="not in X"):
        GLM(family="poisson", terms={"age*speed": Interaction()}).fit(DF[COLS], DF.ClaimNb)
    with pytest.raises(ValueError, match="pair of columns"):
        Interaction().fit(DF.age)
