"""CANN: starts as the GLM, its loss is the Rust deviance, and it learns the interaction a
GLM cannot; the balance is restored and "explain a row" shows the network's column."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, report, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.metrics import FamilyName, deviance

torch = pytest.importorskip("torch", reason="the research track needs torch: uv sync installs it")
from glasshouse.research import CANN  # noqa: E402
from glasshouse.research.cann import deviance_torch  # noqa: E402

rng = np.random.default_rng(21)
N = 8000
DF = pd.DataFrame(
    {
        "region": rng.choice(["n", "s", "e"], size=N),
        "age": rng.uniform(-1.0, 1.0, size=N),
        "power": rng.uniform(-1.0, 1.0, size=N),
        "Exposure": rng.uniform(0.3, 1.0, size=N),
    }
)
# main effects a GLM has, plus a strong age x power interaction it does not
ETA = -1.5 + 0.3 * DF.age + DF.region.map({"n": 0.0, "s": 0.4, "e": -0.3}) + 1.2 * DF.age * DF.power
DF["ClaimNb"] = rng.poisson(np.exp(ETA) * DF.Exposure).astype(float)
COLS = ["region", "age", "power"]
OFFSET = np.log(DF.Exposure.to_numpy())


def _glm() -> GLM:
    return GLM(family="poisson", terms={"region": "onehot"})


@pytest.mark.parametrize(
    ("family", "power", "y", "mu"),
    [
        (
            "poisson",
            None,
            rng.poisson(1.0, size=500).astype(float),
            rng.uniform(0.5, 2.0, size=500),
        ),
        ("gamma", None, rng.gamma(2.0, 1.0, size=500), rng.uniform(0.5, 2.0, size=500)),
        ("gaussian", None, rng.normal(size=500), rng.normal(size=500)),
        (
            "binomial",
            None,
            (rng.uniform(size=500) < 0.3).astype(float),
            rng.uniform(0.05, 0.95, size=500),
        ),
        (
            "tweedie",
            1.5,
            rng.gamma(2.0, 1.0, size=500) * (rng.uniform(size=500) < 0.6),
            rng.uniform(0.5, 2.0, size=500),
        ),
    ],
)
def test_the_torch_loss_is_the_rust_deviance(
    family: FamilyName, power: float | None, y: np.ndarray, mu: np.ndarray
) -> None:
    w = rng.uniform(0.5, 2.0, size=len(y))
    ours = float(
        deviance_torch(
            torch, family, torch.as_tensor(y), torch.as_tensor(mu), torch.as_tensor(w), power
        )
    )
    ref = deviance(y, mu, family=family, power=power, sample_weight=w)
    assert ours == pytest.approx(ref, rel=1e-10)


def test_zero_epochs_is_the_glm_and_training_learns_the_interaction() -> None:
    fold = splits.kfold(N, k=4, seed=0)[0]
    untrained = CANN(glm=_glm, epochs=0).fit(DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold)
    glm = _glm().fit(DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold)
    te = fold.test_idx
    # the network's last layer starts at zero: the same prediction to rounding, before the
    # balance shift, which is itself ~0 because a GLM is balanced on its training rows
    np.testing.assert_allclose(
        untrained.predict(DF[COLS].iloc[te]), glm.predict(DF[COLS].iloc[te]), rtol=1e-6
    )
    assert abs(untrained.shift_) < 1e-6
    trained = CANN(glm=_glm, epochs=200, patience=20, seed=0).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold
    )
    rate = DF.ClaimNb.to_numpy()[te] / DF.Exposure.to_numpy()[te]
    w = DF.Exposure.to_numpy()[te]
    dev_glm = deviance(rate, glm.predict(DF[COLS].iloc[te]), family="poisson", sample_weight=w)
    dev_cann = deviance(rate, trained.predict(DF[COLS].iloc[te]), family="poisson", sample_weight=w)
    assert dev_cann < dev_glm * 0.98, (dev_cann, dev_glm)  # a real gain, on rows it never saw
    assert 1 <= trained.best_epoch_ <= 200 and len(trained.history_) >= trained.best_epoch_
    # the correction is where the interaction is: strongest at the corners of age x power
    corr = trained.correction(DF[COLS].iloc[te])
    ap = (DF.age * DF.power).to_numpy()[te]
    assert np.corrcoef(corr, ap)[0, 1] > 0.5
    # balance on the training rows is restored by the shift
    tr = fold.train_idx
    mu_tr = trained.predict(DF[COLS].iloc[tr], offset=OFFSET[tr])
    assert np.sum(mu_tr) == pytest.approx(DF.ClaimNb.to_numpy()[tr].sum(), rel=1e-9)


def test_explain_a_row_shows_the_network_column_and_the_round_trip_holds() -> None:
    m = CANN(glm=_glm, epochs=5).fit(DF[COLS], DF.ClaimNb, offset=OFFSET)
    parts, names = m.term_contributions(DF[COLS].iloc[:50])
    assert names == ["intercept", "region", "age", "power", "network"]
    np.testing.assert_allclose(parts.sum(axis=1), m.predict_linear(DF[COLS].iloc[:50]), rtol=1e-10)
    back = CANN.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(back.predict(DF[COLS]), m.predict(DF[COLS]), rtol=1e-12)
    assert back.glm_.feature_names_in_ == m.glm_.feature_names_in_
    with pytest.raises(ValueError, match="family"):
        CANN(family="gamma", glm=_glm).fit(DF[COLS], DF.ClaimNb)


def test_the_bench_treats_a_cann_like_any_model_and_the_report_validates() -> None:
    task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
    models = [
        ModelSpec("glm", _glm, COLS),
        ModelSpec("cann", lambda: CANN(glm=_glm, epochs=10, patience=3), COLS),
    ]
    res = bench.run(DF, task, models, splits.kfold(N, k=2, seed=0), features=["age", "region"])
    doc = res.to_dict()
    report.validate(doc)
    assert doc["explain"]["cann"]["coefficients"] is None  # not a glass box
    attr = doc["explain"]["cann"]["attributions"]
    assert attr["terms"][-1] == "network" and len(attr["rows"]) == 30
    assert (
        doc["bench"]["summary"]["cann"]["deviance"]["mean"]
        < doc["bench"]["summary"]["glm"]["deviance"]["mean"]
    )
