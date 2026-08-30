"""Regenerate tests/fixtures/report_small.json — the document the TypeScript tests render.

Run: uv run python tests/make_report_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_report import _freq_report

if __name__ == "__main__":
    out = _freq_report().write(Path(__file__).parent / "fixtures" / "report_small.json")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
