#!/usr/bin/env bash
# The one gate. Lints, type-checks and tests both languages. Green before every push.
# Usage: ./check.sh            run everything
#        ./check.sh rust       only the Rust checks
#        ./check.sh py         only the Python checks (builds the extension first)
#        ./check.sh fix        auto-format Rust + Python, then run everything
# Each step prints what it is doing; the first failure stops the run (set -e).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

what="${1:-all}"

rust() {
  echo "==> cargo fmt --check        (formatting; run ./check.sh fix to apply)"
  cargo fmt --all -- --check
  echo "==> cargo clippy             (lints, warnings are errors)"
  cargo clippy --workspace --all-targets -- -D warnings
  echo "==> cargo test               (Rust unit tests)"
  cargo test --workspace --quiet
}

py() {
  echo "==> uv sync                  (creates .venv, builds the Rust extension in dev mode)"
  uv sync --quiet
  echo "==> ruff format --check      (formatting)"
  uv run ruff format --check python tests
  echo "==> ruff check               (lint, docstrings, complexity)"
  uv run ruff check python tests
  echo "==> mypy --strict            (types)"
  uv run mypy
  echo "==> pytest                   (golden + property tests)"
  uv run pytest
}

case "$what" in
  all) rust; py ;;
  rust) rust ;;
  py) py ;;
  fix)
    cargo fmt --all
    uv run ruff format python tests
    uv run ruff check --fix python tests
    rust; py ;;
  *) echo "unknown target: $what (all | rust | py | fix)"; exit 2 ;;
esac
echo "==> all green"
