"""CANN: starts as the GLM, its loss is the Rust deviance, and it learns the interaction a
GLM cannot; the balance is restored and "explain a row" shows the network's column."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, report, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.metrics import FamilyName, deviance

torch = pytest.importorskip("torch", reason="the research track needs torch: uv sync installs it")
from glasshouse.research import CANN, AdditiveNet, LocalGLMnet  # noqa: E402
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


def _held_out_deviance(model: Any, te: np.ndarray) -> float:
    rate = DF.ClaimNb.to_numpy()[te] / DF.Exposure.to_numpy()[te]
    w = DF.Exposure.to_numpy()[te]
    return float(
        deviance(rate, model.predict(DF[COLS].iloc[te]), family="poisson", sample_weight=w)
    )


def test_the_additive_net_learns_a_marginal_shape_but_not_an_interaction() -> None:
    # a bent marginal in age that the linear GLM term cannot follow, and no interaction
    frame = DF[COLS].copy()
    eta = -1.5 + 1.5 * np.sin(2.5 * frame.age) + frame.region.map({"n": 0.0, "s": 0.4, "e": -0.3})
    y = rng.poisson(np.exp(eta) * DF.Exposure).astype(float)
    fold = splits.kfold(N, k=4, seed=1)[0]
    te = fold.test_idx
    glm = _glm().fit(frame, y, offset=OFFSET, fold=fold)
    add = AdditiveNet(glm=_glm, epochs=200, patience=20).fit(frame, y, offset=OFFSET, fold=fold)
    rate, w = y[te] / DF.Exposure.to_numpy()[te], DF.Exposure.to_numpy()[te]
    dev_glm = deviance(rate, glm.predict(frame.iloc[te]), family="poisson", sample_weight=w)
    dev_add = deviance(rate, add.predict(frame.iloc[te]), family="poisson", sample_weight=w)
    assert dev_add < dev_glm * 0.97, (dev_add, dev_glm)
    parts, names = add.term_contributions(frame.iloc[:300])
    assert names == [
        "intercept",
        "region",
        "age",
        "power",
        "region (net)",
        "age (net)",
        "power (net)",
    ]
    np.testing.assert_allclose(parts.sum(axis=1), add.predict_linear(frame.iloc[:300]), rtol=1e-10)
    # the net's age column carries the bend: it varies with age and with nothing else
    corr_age = parts[:, names.index("age (net)")]
    bend = np.sin(2.5 * frame.age.to_numpy()[:300])
    assert np.std(corr_age) > 0.05 and abs(np.corrcoef(corr_age, bend)[0, 1]) > 0.6
    # on the interaction data the additive net cannot do what the CANN does
    fold2 = splits.kfold(N, k=4, seed=0)[0]
    add2 = AdditiveNet(glm=_glm, epochs=100, patience=10).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold2
    )
    cann = CANN(glm=_glm, epochs=200, patience=20).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold2
    )
    assert _held_out_deviance(cann, fold2.test_idx) < _held_out_deviance(add2, fold2.test_idx)


def test_localglmnet_learns_the_interaction_and_its_attentions_say_where() -> None:
    fold = splits.kfold(N, k=4, seed=0)[0]
    te = fold.test_idx
    glm = _glm().fit(DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold)
    local = LocalGLMnet(glm=_glm, epochs=200, patience=20).fit(
        DF[COLS], DF.ClaimNb, offset=OFFSET, fold=fold
    )
    assert _held_out_deviance(local, te) < _held_out_deviance(glm, te) * 0.985
    beta, names = local.attention(DF[COLS].iloc[te])
    assert names == glm.feature_names_in_[1:] and beta.shape == (len(te), len(names))
    # the coefficient on age moves with power (and the other way round): the interaction
    age_beta = beta[:, names.index("age")]
    assert abs(np.corrcoef(age_beta, DF.power.to_numpy()[te])[0, 1]) > 0.5
    parts, tnames = local.term_contributions(DF[COLS].iloc[:30])
    assert tnames[-3:] == ["region (net)", "age (net)", "power (net)"]
    np.testing.assert_allclose(
        parts.sum(axis=1), local.predict_linear(DF[COLS].iloc[:30]), rtol=1e-10
    )
    with pytest.raises(ValueError, match="localglm"):
        CANN(glm=_glm, epochs=0).fit(DF[COLS], DF.ClaimNb).attention(DF[COLS])
    back = LocalGLMnet.from_dict(json.loads(json.dumps(local.to_dict())))
    np.testing.assert_allclose(back.predict(DF[COLS]), local.predict(DF[COLS]), rtol=1e-12)
    assert back.network == "localglm"


def test_every_network_starts_as_the_glm() -> None:
    glm = _glm().fit(DF[COLS], DF.ClaimNb, offset=OFFSET)
    for model in (
        CANN(glm=_glm, epochs=0),
        AdditiveNet(glm=_glm, epochs=0),
        LocalGLMnet(glm=_glm, epochs=0),
    ):
        m = model.fit(DF[COLS], DF.ClaimNb, offset=OFFSET)
        np.testing.assert_allclose(m.predict(DF[COLS]), glm.predict(DF[COLS]), rtol=1e-9)
    with pytest.raises(ValueError, match="network must be"):
        CANN(glm=_glm, epochs=0, network="kan").fit(DF[COLS], DF.ClaimNb)
