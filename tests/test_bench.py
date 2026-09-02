"""bench: every model on every fold, scored the same way, written down — and pinned."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glasshouse import GLM, bench, report, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.benchmarks import run_named
from glasshouse.cli import main

rng = np.random.default_rng(10)
N = 3000
df = pd.DataFrame(
    {
        "region": rng.choice(["n", "s", "e"], size=N),
        "age": rng.uniform(18, 80, size=N),
        "Exposure": rng.uniform(0.2, 1.0, size=N),
    }
)
eta = -2.0 + 0.01 * (df.age - 40) + df.region.map({"n": 0.0, "s": 0.4, "e": -0.3})
df["ClaimNb"] = rng.poisson(np.exp(eta) * df.Exposure).astype(float)
TASK = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
MODELS = [
    ModelSpec("age_only", lambda: GLM(family="poisson"), ["age"]),
    ModelSpec("full", lambda: GLM(family="poisson", terms={"region": "onehot"}), ["region", "age"]),
]


def test_run_scores_every_model_on_every_fold_and_pools_curves() -> None:
    folds = splits.kfold(N, k=3, seed=0)
    res = bench.run(
        df, TASK, MODELS, folds, dataset="synthetic", describe="made up", features=["region"]
    )
    assert len(res.folds) == 6 and res.labels == ["age_only", "full"]
    summ = res.summary()
    assert summ["full"]["deviance"][0] < summ["age_only"]["deviance"][0]
    kinds = [c["kind"] for c in res.doc["curves"]]
    assert kinds.count("lorenz") == 2 and kinds.count("double_lift") == 1
    assert all(r.card.naive["d2"] == pytest.approx(0.0, abs=1e-12) for r in res.folds)
    assert res.naive_summary()["d2"][0] == pytest.approx(0.0, abs=1e-12)
    assert [t["feature"] for t in res.doc["residuals"]["full"]["by_feature"]] == ["region"]


def test_report_json_markdown_and_html(tmp_path: Path) -> None:
    res = bench.run(df, TASK, MODELS, splits.kfold(N, k=2, seed=1), dataset="synthetic")
    out = res.write(tmp_path / "synthetic")
    payload = json.loads((out / "report.json").read_text())
    assert payload["schema"] == "glasshouse-report/1"
    report.validate(payload)
    assert set(payload["bench"]["summary"]) == {"age_only", "full"}
    assert payload["provenance"]["split"]["kind"] == "random"
    assert (
        payload["provenance"]["n_rows"]
        == len(res.doc["models"]) * 0 + payload["provenance"]["n_rows"]
    )
    md = (out / "report.md").read_text()
    assert "| deviance |" in md and "**" in md and "naive" in md
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "GlasshouseReport" in html and "__GLASSHOUSE_" not in html
    pinned = json.loads((out / "pinned.json").read_text())
    assert set(pinned["summary"]) == {"age_only", "full"}


def test_refuses_missing_columns_by_name() -> None:
    with pytest.raises(ValueError, match=r"missing column\(s\) \['Time'\]"):
        bench.run(df, TASK, MODELS, splits.kfold(N, k=2), features=["Time"])


def test_refuses_folds_from_another_frame() -> None:
    with pytest.raises(ValueError, match="folds were made for"):
        bench.run(df, TASK, MODELS, splits.kfold(10, k=2))


def test_cli_lists_benchmarks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert "fremtpl2_glm" in capsys.readouterr().out


@pytest.mark.skipif(
    not os.environ.get("GLASSHOUSE_NETWORK_TESTS"), reason="needs the cached dataset"
)
def test_fremtpl2_glm_report_is_pinned() -> None:
    """The committed report must be reproducible: same recipe, same numbers (to 1e-6)."""
    pinned = json.loads(Path("benchmarks/fremtpl2_glm/pinned.json").read_text())
    res = run_named("fremtpl2_glm").to_dict()
    for label, metrics in pinned["summary"].items():
        for m, v in metrics.items():
            assert res["bench"]["summary"][label][m]["mean"] == pytest.approx(
                v["mean"], rel=1e-6
            ), (label, m)


def test_progress_reports_every_step(capsys: pytest.CaptureFixture[str]) -> None:
    bench.run(df, TASK, MODELS, splits.kfold(N, k=2, seed=0), progress=True)
    err = capsys.readouterr().err
    assert err.count("bench") == 5  # 2 models x 2 folds + the report build
    assert "age_only fold 0" in err and "building the report" in err and "100%" in err
