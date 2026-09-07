"""Regenerate tests/fixtures/report_small.json — the document the TypeScript tests render.

Run: uv run python tests/make_report_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report import AGE, COUNT, EXPO, LABEL, PROB, RATE, REGION, TIME, N

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from glasshouse import GLM, bench, report, splits
    from glasshouse.bench import ModelSpec, TaskSpec
    from glasshouse.research import CANN

    fixtures = Path(__file__).parent / "fixtures"
    rng = np.random.default_rng(12)
    freq = report.build(
        "frequency",
        COUNT / EXPO,
        {
            "glm": RATE * rng.lognormal(0, 0.1, size=N),
            "mean": np.full(N, RATE.mean()),
            "kan": RATE * rng.lognormal(0, 0.15, size=N),
        },
        weight=EXPO,
        features={"region": REGION, "age": AGE},
        time=TIME,
        dataset="synthetic",
        describe="made up",
        split={"kind": "random", "method": "kfold", "k": 5, "seed": 0},
    )
    # the Model tab needs fitted models: a small bench on the same synthetic data supplies
    # the explain block, labelled with the report's model names so the viewer lines up
    frame = pd.DataFrame({"region": REGION, "age": AGE, "Exposure": EXPO, "ClaimNb": COUNT})
    run = bench.run(
        frame,
        TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        [
            ModelSpec(
                "glm",
                lambda: GLM(family="poisson", terms={"region": "onehot", "age": "smooth"}),
                ["region", "age"],
            ),
            ModelSpec(
                "mean",
                lambda: GLM(family="poisson", alpha="cv", l1_ratio=1.0, cv=3, alpha_rule="min"),
                ["age"],
            ),
            ModelSpec(
                "kan",
                lambda: CANN(
                    glm=lambda: GLM(family="poisson", terms={"region": "onehot"}),
                    network="kan",
                    hidden=(3,),
                    epochs=3,
                    encoding="raw",
                ),
                ["region", "age"],
            ),
        ],
        splits.kfold(N, k=3, seed=0),
        features=["region", "age"],
    )
    freq.doc["explain"] = run.doc["explain"]
    out = freq.write(fixtures / "report_small.json")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    binary = report.build(
        "binary",
        LABEL,
        {"model": PROB, "flat": np.full(N, 0.3)},
        dataset="synthetic-binary",
        describe="made up, for the viewer tests",
    )
    out = binary.write(fixtures / "report_binary_small.json")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
