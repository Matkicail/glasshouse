"""The FOSS adapters fit the same model our GLM does, on the same design, honestly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.foss import GlumPoisson, SklearnPoisson

rng = np.random.default_rng(13)
N = 4000
df = pd.DataFrame(
    {
        "region": rng.choice(["n", "s", "e", "w"], size=N),
        "age": rng.uniform(-1.0, 1.0, size=N),
        "Exposure": rng.uniform(0.2, 1.0, size=N),
    }
)
eta = -2.0 + 0.4 * df.age + df.region.map({"n": 0.0, "s": 0.4, "e": -0.3, "w": 0.2})
df["ClaimNb"] = rng.poisson(np.exp(eta) * df.Exposure).astype(float)
OFFSET = np.log(df.Exposure.to_numpy())
X = df[["region", "age"]]
Y = df.ClaimNb.to_numpy()


def test_adapters_agree_with_our_glm_at_the_optimum() -> None:
    ours = GLM(family="poisson", terms={"region": "onehot"}, tol=1e-14).fit(X, Y, offset=OFFSET)
    mu = ours.predict(X, offset=OFFSET)
    for adapter in (GlumPoisson(onehot=["region"]), SklearnPoisson(onehot=["region"])):
        fitted = adapter.fit(X, Y, offset=OFFSET)
        np.testing.assert_allclose(fitted.predict(X, offset=OFFSET), mu, rtol=2e-4)


def test_adapters_respect_the_fold() -> None:
    fold = splits.kfold(N, k=4, seed=0)[0]
    a = GlumPoisson(onehot=["region"]).fit(X, Y, offset=OFFSET, fold=fold)
    by_hand = GlumPoisson(onehot=["region"]).fit(
        X.iloc[fold.train_idx], Y[fold.train_idx], offset=OFFSET[fold.train_idx]
    )
    np.testing.assert_allclose(
        a.predict(X.iloc[fold.test_idx], offset=OFFSET[fold.test_idx]),
        by_hand.predict(X.iloc[fold.test_idx], offset=OFFSET[fold.test_idx]),
        rtol=1e-8,
    )


def test_the_three_run_in_one_bench_and_tie_on_deviance() -> None:
    task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
    models = [
        ModelSpec(
            "glasshouse",
            lambda: GLM(family="poisson", terms={"region": "onehot"}),
            ["region", "age"],
        ),
        ModelSpec("glum", lambda: GlumPoisson(onehot=["region"]), ["region", "age"]),
        ModelSpec("sklearn", lambda: SklearnPoisson(onehot=["region"]), ["region", "age"]),
    ]
    res = bench.run(df, task, models, splits.kfold(N, k=2, seed=1), dataset="synthetic")
    summ = res.summary()
    ours = summ["glasshouse"]["deviance"][0]
    for label in ("glum", "sklearn"):
        assert summ[label]["deviance"][0] == pytest.approx(ours, rel=1e-4), label
    assert set(res.doc["scorecards"]) == {"glasshouse", "glum", "sklearn"}


def test_design_fails_early() -> None:
    with pytest.raises(ValueError, match="named columns"):
        GlumPoisson(onehot=["region"]).fit(np.ones((10, 2)), np.ones(10))
    with pytest.raises(ValueError, match="not in X"):
        SklearnPoisson(onehot=["colour"]).fit(X, Y)
