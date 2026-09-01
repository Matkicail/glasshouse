"""The LightGBM adapter: fold-safe, offset-correct, and honestly better where trees should be."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.gbdt import LightGBM
from glasshouse.metrics import deviance

try:  # lightgbm needs libomp at load time; without it, skip with the fix named
    import lightgbm  # noqa: F401
except (ImportError, OSError) as _err:
    pytest.skip(
        f"lightgbm unavailable: {_err} (macOS: brew install libomp)", allow_module_level=True
    )

rng = np.random.default_rng(14)
N = 6000
df = pd.DataFrame(
    {
        "region": rng.choice(["n", "s", "e"], size=N),
        "age": rng.uniform(18, 80, size=N),
        "Exposure": rng.uniform(0.2, 1.0, size=N),
    }
)
# a strong U-shape in age: a linear GLM cannot follow it, trees can
age_effect = 0.002 * (df.age - 45) ** 2
eta = -2.5 + age_effect + df.region.map({"n": 0.0, "s": 0.3, "e": -0.2})
df["ClaimNb"] = rng.poisson(np.exp(eta) * df.Exposure).astype(float)
OFFSET = np.log(df.Exposure.to_numpy())
X = df[["region", "age"]]
Y = df.ClaimNb.to_numpy()


def test_offset_is_multiplicative() -> None:
    m = LightGBM(family="poisson", categorical=["region"], n_estimators=50).fit(X, Y, offset=OFFSET)
    mu = m.predict(X, offset=OFFSET)
    doubled = m.predict(X, offset=OFFSET + np.log(2.0))
    np.testing.assert_allclose(doubled, 2.0 * mu, rtol=1e-12)
    assert np.all(mu > 0)


def test_beats_a_linear_glm_on_a_u_shape() -> None:
    fold = splits.kfold(N, k=4, seed=0)[0]
    te = fold.test_idx
    glm = GLM(family="poisson", terms={"region": "onehot"}).fit(X, Y, offset=OFFSET, fold=fold)
    gbm = LightGBM(family="poisson", categorical=["region"], seed=1).fit(
        X, Y, offset=OFFSET, fold=fold
    )
    d_glm = deviance(Y[te], glm.predict(X.iloc[te], offset=OFFSET[te]), family="poisson")
    d_gbm = deviance(Y[te], gbm.predict(X.iloc[te], offset=OFFSET[te]), family="poisson")
    assert d_gbm < d_glm  # the trees find the U-shape the linear term cannot
    assert gbm.best_iteration_ is not None and gbm.best_iteration_ >= 1


def test_respects_the_fold_and_early_stops_inside_it() -> None:
    fold = splits.kfold(N, k=4, seed=2)[1]
    a = LightGBM(family="poisson", categorical=["region"], seed=3).fit(
        X, Y, offset=OFFSET, fold=fold
    )
    by_hand = LightGBM(family="poisson", categorical=["region"], seed=3).fit(
        X.iloc[fold.train_idx], Y[fold.train_idx], offset=OFFSET[fold.train_idx]
    )
    te = fold.test_idx
    np.testing.assert_allclose(
        a.predict(X.iloc[te], offset=OFFSET[te]),
        by_hand.predict(X.iloc[te], offset=OFFSET[te]),
        rtol=1e-10,
    )


def test_unseen_level_predicts_without_crashing() -> None:
    m = LightGBM(family="poisson", categorical=["region"], n_estimators=30).fit(X, Y, offset=OFFSET)
    new = pd.DataFrame({"region": ["mars", "n"], "age": [40.0, 40.0]})
    mu = m.predict(new, offset=np.zeros(2))
    assert np.all(np.isfinite(mu)) and np.all(mu > 0)


def test_runs_in_the_bench_next_to_the_glm() -> None:
    task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
    models = [
        ModelSpec(
            "glm", lambda: GLM(family="poisson", terms={"region": "onehot"}), ["region", "age"]
        ),
        ModelSpec(
            "lightgbm",
            lambda: LightGBM(family="poisson", categorical=["region"], num_leaves=15),
            ["region", "age"],
        ),
    ]
    res = bench.run(df, task, models, splits.kfold(N, k=2, seed=1), dataset="synthetic")
    summ = res.summary()
    assert summ["lightgbm"]["deviance"][0] < summ["glm"]["deviance"][0]
    kinds = [c["kind"] for c in res.doc["curves"]]
    assert "double_lift" in kinds  # the GLM-vs-GBM chart is there


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="named columns"):
        LightGBM(family="poisson").fit(np.ones((10, 2)), np.ones(10))
    with pytest.raises(ValueError, match="not in X"):
        LightGBM(family="poisson", categorical=["colour"]).fit(X, Y)
    with pytest.raises(ValueError, match="variance power"):
        LightGBM(family="tweedie").fit(X, Y)
    with pytest.raises(ValueError, match="LightGBM supports"):
        LightGBM(family="binomial").fit(X, Y)
