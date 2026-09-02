"""Regenerate tests/fixtures/report_small.json — the document the TypeScript tests render.

Run: uv run python tests/make_report_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report import LABEL, PROB, N, _freq_report

if __name__ == "__main__":
    import numpy as np

    from glasshouse import report

    fixtures = Path(__file__).parent / "fixtures"
    out = _freq_report().write(fixtures / "report_small.json")
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
