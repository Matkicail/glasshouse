"""report.build: the whole document, schema-valid, for every task type; and the fixture the TS
side will render from."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glasshouse import curves, report
from glasshouse.classification import average_precision, roc_auc, threshold_metrics

rng = np.random.default_rng(12)
N = 1500
EXPO = rng.uniform(0.2, 1.0, size=N)
RATE = rng.gamma(2.0, 0.1, size=N)
REGION = rng.choice(["n", "s", "e"], size=N)
AGE = rng.uniform(18, 80, size=N)
TIME = rng.integers(2015, 2024, size=N).astype(float)
COUNT = rng.poisson(RATE * EXPO).astype(float)
PROB = rng.beta(0.6, 3.0, size=N)
LABEL = (rng.uniform(size=N) < PROB).astype(float)


def _freq_report() -> report.Report:
    return report.build(
        "frequency",
        COUNT / EXPO,
        {"glm": RATE * rng.lognormal(0, 0.1, size=N), "mean": np.full(N, RATE.mean())},
        weight=EXPO,
        features={"region": REGION, "age": AGE},
        time=TIME,
        dataset="synthetic",
        describe="made up",
        split={"kind": "random", "method": "kfold", "k": 5, "seed": 0},
    )


def test_frequency_report_is_complete_and_valid() -> None:
    r = _freq_report()
    doc = r.to_dict()
    report.validate(doc)
    assert doc["task"]["family"] == "poisson" and doc["task"]["primary_metric"] == "deviance"
    assert doc["models"] == ["glm", "mean"]
    assert set(doc["scorecards"]) == {"glm", "mean"}
    assert doc["comparisons"][0]["a"] == "glm" and len(doc["comparisons"]) == 1
    kinds = sorted(c["kind"] for c in doc["curves"])
    assert kinds == [
        "calibration",
        "calibration",
        "double_lift",
        "lift",
        "lift",
        "lorenz",
        "lorenz",
    ]
    res = doc["residuals"]["glm"]
    assert [b["feature"] for b in res["by_feature"]] == ["region", "age"]
    assert res["over_time"]["feature"] == "time"
    assert len(res["scatter"]["fitted"]) == N  # under the sample cap: all rows
    assert doc["provenance"]["sample_rows"] == N and doc["provenance"]["n_rows"] == N


def test_binary_report_uses_roc_and_pr_and_the_prior() -> None:
    doc = report.build("binary", LABEL, {"m": PROB, "flat": np.full(N, 0.3)}).to_dict()
    report.validate(doc)
    grid = doc["thresholds"]["m"]
    assert len(grid["threshold"]) == 101 and grid["threshold"][50] == 0.5
    at_half = threshold_metrics(LABEL, PROB, threshold=0.5)
    assert grid["mcc"][50] == pytest.approx(at_half.mcc)
    assert grid["alerts"][50] == pytest.approx(at_half.tp + at_half.fp)
    assert grid["alerts_per_tp"][100] is None  # threshold 1.0 flags nothing: undefined, null
    assert "thresholds" not in report.build("regression", PROB, {"m": PROB}).to_dict()
    kinds = {c["kind"] for c in doc["curves"]}
    assert {"roc", "pr", "lift", "calibration"} <= kinds and "lorenz" not in kinds
    assert doc["task"]["threshold"] == 0.5 and doc["task"]["primary_metric"] == "average_precision"
    assert doc["naive"]["roc_auc"] == pytest.approx(0.5)
    roc = next(c for c in doc["curves"] if c["kind"] == "roc" and c["label"] == "m")
    assert roc["auc"] == pytest.approx(roc_auc(LABEL, PROB))
    # the curve is thinned to 1000 points; its area approximates the exact AUC held in `auc`
    assert np.trapezoid(roc["tpr"], roc["fpr"]) == pytest.approx(roc["auc"], abs=2e-3)
    prc = next(c for c in doc["curves"] if c["kind"] == "pr" and c["label"] == "m")
    assert prc["average_precision"] == pytest.approx(average_precision(LABEL, PROB))
    assert prc["positive_rate"] == pytest.approx(LABEL.mean())


@pytest.mark.parametrize("task", ["severity", "pure_premium", "regression"])
def test_other_tasks_build_and_validate(task: str) -> None:
    y = rng.gamma(2.0, 100.0, size=N) if task != "regression" else rng.normal(size=N)
    if task == "pure_premium":
        y = y * (rng.uniform(size=N) < 0.3)
    mu = (
        np.abs(y * rng.lognormal(0, 0.3, size=N)) + 1e-3
        if task != "regression"
        else y + rng.normal(scale=0.5, size=N)
    )
    power = 1.5 if task == "pure_premium" else None
    doc = report.build(task, y, {"a": mu}, power=power).to_dict()  # type: ignore[arg-type]
    report.validate(doc)
    assert doc["task"]["type"] == task


def test_sampling_caps_the_scatter_only() -> None:
    doc = report.build(
        "frequency", COUNT / EXPO, {"glm": RATE}, weight=EXPO, sample=100, seed=3
    ).to_dict()
    assert len(doc["residuals"]["glm"]["scatter"]["fitted"]) == 100
    assert doc["provenance"]["sample_rows"] == 100 and doc["provenance"]["n_rows"] == N
    assert doc["scorecards"]["glm"]["n_rows"] == N  # scores use every row


def test_fails_early() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        report.build("survival", COUNT, {"m": RATE})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one model"):
        report.build("frequency", COUNT, {})
    with pytest.raises(ValueError, match="predictions but y has"):
        report.build("frequency", COUNT, {"m": RATE[:10]})
    bad = _freq_report().to_dict()
    bad["schema"] = "glasshouse-report/0"
    with pytest.raises(ValueError, match="does not match"):
        report.validate(bad)


def test_roc_pr_curves_thin_and_keep_the_score() -> None:
    long_y = (rng.uniform(size=20_000) < 0.1).astype(float)
    long_s = rng.uniform(size=20_000)
    c = curves.roc(long_y, long_s, max_points=300)
    assert len(c.fpr) == 300 and c.auc == roc_auc(long_y, long_s)
    p = curves.pr(long_y, long_s, max_points=300)
    assert len(p.recall) == 300 and p.average_precision == average_precision(long_y, long_s)


def test_fixture_for_the_typescript_side(tmp_path: Path) -> None:
    """The small fixture the TS tests render: `uv run python tests/make_report_fixture.py`."""
    r = _freq_report()
    out = r.write(tmp_path / "report_small.json")
    doc = json.loads(out.read_text())
    report.validate(doc)
    fixture = Path("tests/fixtures/report_small.json")
    if fixture.exists():
        pinned = json.loads(fixture.read_text())
        assert pinned["schema"] == doc["schema"] and pinned["models"] == doc["models"]


def test_to_html_is_self_contained(tmp_path: Path) -> None:
    r = _freq_report()
    out = r.to_html(tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert html.count("GlasshouseReport") >= 2  # the viewer and the boot script
    assert 'type="application/json"' in html and '"schema": "glasshouse-report/1"' in html
    assert 'integrity="sha384-' in html and "plotly-2.35.2.min.js" in html
    assert "__GLASSHOUSE_" not in html  # every placeholder filled
    assert out.stat().st_size < 2_000_000  # the fixture is small; the viewer is ~25 KB
    # a </script> inside the data cannot break out of the JSON block
    doc = r.to_dict()
    doc["provenance"]["describe"] = "evil </script><script>alert(1)</script>"
    html2 = report.to_html(doc, tmp_path / "r2.html").read_text(encoding="utf-8")
    assert (
        "</script><script>alert"
        not in html2.split('id="glasshouse-report-data"')[1].split("</script>")[0]
    )
