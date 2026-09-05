"""Regenerate tests/fixtures/report_small.json — the document the TypeScript tests render.

Run: uv run python tests/make_report_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report import AGE, COUNT, EXPO, LABEL, PROB, REGION, N, _freq_report

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from glasshouse import GLM, bench, report, splits
    from glasshouse.bench import ModelSpec, TaskSpec

    fixtures = Path(__file__).parent / "fixtures"
    freq = _freq_report()
    # the Model tab needs fitted models: a small bench on the same synthetic data supplies
    # the explain block, labelled with the report's model names so the viewer lines up
    frame = pd.DataFrame({"region": REGION, "age": AGE, "Exposure": EXPO, "ClaimNb": COUNT})
    run = bench.run(
        frame,
        TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True),
        [
            ModelSpec(
                "glm", lambda: GLM(family="poisson", terms={"region": "onehot"}), ["region", "age"]
            ),
            ModelSpec("mean", lambda: GLM(family="poisson"), ["age"]),
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
