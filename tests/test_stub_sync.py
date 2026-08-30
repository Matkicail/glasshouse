"""The hand-written ``_core.pyi`` must list exactly the functions the Rust module exports."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from glasshouse import _core


def _stub_functions() -> set[str]:
    stub = Path(inspect.getfile(_core)).with_name("_core.pyi")
    tree = ast.parse(stub.read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _extension_functions() -> set[str]:
    return {name for name, obj in inspect.getmembers(_core) if inspect.isbuiltin(obj)}


def test_stub_matches_extension() -> None:
    stub, ext = _stub_functions(), _extension_functions()
    assert stub == ext, (
        f"_core.pyi is out of step with the Rust module: "
        f"missing from stub {sorted(ext - stub)}, stale in stub {sorted(stub - ext)}"
    )
