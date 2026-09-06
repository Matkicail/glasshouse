"""Every ``python`` block in ``docs/comparing-models.md`` runs, in order, in one namespace.

The page is the user's guide to the comparison report. A guide whose examples do not run
is worse than none, so this is the test that keeps it honest. The printed output in the
page was pasted from a run of this test, never typed.
"""

from __future__ import annotations

import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

try:  # example 1 fits LightGBM, which needs libomp at load time; skip with the fix named
    import lightgbm  # noqa: F401
except (ImportError, OSError) as _err:
    pytest.skip(
        f"lightgbm unavailable: {_err} (macOS: brew install libomp)", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "comparing-models.md"
README = ROOT / "README.md"


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def _run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    blocks = _python_blocks(page.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)  # the examples write reports/ into the working directory
    namespace: dict[str, object] = {}
    printed = io.StringIO()
    with redirect_stdout(printed):
        for block in blocks:
            exec(compile(block, str(page), "exec"), namespace)  # the docs' own code
    if os.environ.get("GLASSHOUSE_SHOW_DOC_OUTPUT"):
        print(printed.getvalue())
    return printed.getvalue()


def test_every_python_block_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(_python_blocks(DOC.read_text(encoding="utf-8"))) >= 5
    printed = _run(DOC, tmp_path, monkeypatch)
    assert os.path.exists("reports/synthetic_frequency/report.html")
    assert os.path.exists("reports/synthetic_binary.html")
    assert os.path.exists("reports/synthetic_regression.html")
    # the leaderboard the page shows is the one the code prints
    assert "| deviance |" in printed and "naive" in printed


def test_the_readme_example_runs_and_prints_what_the_readme_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = README.read_text(encoding="utf-8")
    printed = _run(README, tmp_path, monkeypatch)
    assert os.path.exists("reports/readme/report.html")
    shown = re.search(r"```text\n(.*?)```", text, flags=re.DOTALL)
    assert shown is not None
    # every scored line in the README's pasted output is what the code prints today
    for line in shown.group(1).splitlines():
        if line.startswith("| ") and "metric" not in line:
            assert line in printed, line
