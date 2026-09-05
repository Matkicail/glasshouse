"""Explaining models through predictions: partial dependence and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, explain, report, splits
from glasshouse.bench import ModelSpec, TaskSpec

rng = np.random.default_rng(41)
N = 3000
DF = pd.DataFrame(
    {
        "age": rng.uniform(18.0, 80.0, size=N),
        "power": rng.normal(size=N),
        "noise": rng.normal(size=N),
        "region": rng.choice(["north", "south", "east"], size=N),
        "Exposure": rng.uniform(0.2, 1.0, size=N),
    }
)
ETA = (
    -2.0
    + 0.02 * (DF.age - 45.0)
    + 0.5 * DF.power
    + DF.region.map({"north": 0.0, "south": 0.4, "east": -0.3})
)
DF["ClaimNb"] = rng.poisson(np.exp(ETA) * DF.Exposure).astype(float)
COLS = ["age", "power", "noise", "region"]


def _glm() -> GLM:
    return GLM(family="poisson", terms={"region": "onehot"})


def test_partial_dependence_of_a_glm_is_its_coefficient_on_the_link_scale() -> None:
    m = _glm().fit(DF[COLS], DF.ClaimNb, offset=np.log(DF.Exposure))
    pd_age = explain.partial_dependence(m, DF[COLS], "age", grid=[20.0, 40.0, 60.0])
    # log link: the mean prediction scales by exp(coef * step) between grid points, exactly
    step = np.log(pd_age.effect[1:] / pd_age.effect[:-1]) / 20.0
    coef_age = m.coef_[m.feature_names_in_.index("age") - 1]
    np.testing.assert_allclose(step, coef_age, rtol=1e-10)
    levels = explain.partial_dependence(m, DF[COLS], "region")
    assert levels.kind == "categorical" and levels.grid == ["east", "north", "south"]
    ratio = np.log(levels.effect[2] / levels.effect[0])  # south vs east (the reference level)
    np.testing.assert_allclose(
        ratio, m.coef_[m.feature_names_in_.index("region=south") - 1], rtol=1e-10
    )
    default = explain.partial_dependence(m, DF[COLS], "age")
    assert default.kind == "numeric" and len(default.grid) == 20
    assert list(default.grid) == sorted(default.grid)


def test_permutation_importance_ranks_signal_above_noise() -> None:
    m = _glm().fit(DF[COLS], DF.ClaimNb, offset=np.log(DF.Exposure))
    imp = explain.permutation_importance(
        m,
        DF[COLS],
        DF.ClaimNb / DF.Exposure,
        family="poisson",
        features=COLS,
        sample_weight=DF.Exposure,
    )
    by = dict(zip(imp.features, imp.loss, strict=True))
    assert by["age"] > by["noise"] and by["power"] > by["noise"] and by["region"] > by["noise"]
    assert abs(by["noise"]) < 0.02 * imp.base_deviance
    assert "age" in str(imp)
    coefs = explain.coefficients(m)
    assert coefs is not None and next(iter(coefs)) == "intercept"
    assert explain.coefficients(object()) is None


def test_bench_carries_the_explain_block_and_it_validates() -> None:
    task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
    models = [
        ModelSpec("glm", _glm, COLS),
        ModelSpec("age_only", lambda: GLM(family="poisson"), ["age"]),
    ]
    res = bench.run(
        DF, task, models, splits.kfold(N, k=3, seed=0), features=["age", "region", "noise"]
    )
    doc = res.to_dict()
    report.validate(doc)
    ex = doc["explain"]
    assert set(ex) == {"glm", "age_only"}
    assert [p["feature"] for p in ex["glm"]["partial_dependence"]] == ["age", "region", "noise"]
    assert [p["feature"] for p in ex["age_only"]["partial_dependence"]] == ["age"]
    age = ex["glm"]["partial_dependence"][0]
    assert age["kind"] == "numeric" and len(age["grid"]) == len(age["mean"]) == 20
    assert all(
        lo <= m <= hi for lo, m, hi in zip(age["low"], age["mean"], age["high"], strict=True)
    )
    region = ex["glm"]["partial_dependence"][1]
    assert region["kind"] == "categorical" and region["grid"] == ["east", "north", "south"]
    imp = ex["glm"]["importance"]
    assert imp["features"] == ["age", "region", "noise"]
    assert imp["mean"][0] > imp["mean"][2]
    coef = ex["glm"]["coefficients"]
    assert coef["terms"][0] == "intercept" and coef["relativity"] is not None
    assert coef["relativity"][0] == pytest.approx(np.exp(coef["mean"][0]))
    # no features: no block, and the report still validates
    bare = bench.run(DF, task, models[:1], splits.kfold(N, k=2, seed=0)).to_dict()
    assert "explain" not in bare
    report.validate(bare)
