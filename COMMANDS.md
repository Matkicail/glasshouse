# COMMANDS — what does what, how, and why

Written for a person without Claude (or a smaller model) to run this project start to finish.
Every command below is safe to run as-is from the repo root. If something here is wrong, fix the
doc in the same PR as the fix — this file is the recipe, and a stale recipe is worse than none.

## 0. One-time setup (a fresh machine)

| Command | What it does | Why |
|---|---|---|
| `curl -sSf https://sh.rustup.rs \| sh -s -- -y --profile minimal --component clippy,rustfmt` | Installs Rust (compiler, cargo, clippy, rustfmt) into `~/.cargo`, no sudo | The numerics are Rust; clippy and rustfmt are the gate |
| `. "$HOME/.cargo/env"` (once per shell, or add to `~/.bashrc`) | Puts `cargo` on your PATH | The installer doesn't touch your current shell |
| `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Installs `uv` (Python package manager) | Replaces pip/venv/pyenv; also builds the Rust extension |
| `uv python install 3.12` | Installs Python 3.12 managed by uv | Floor is 3.12 (3.10 is EOL Oct 2026); CI runs 3.12 + 3.13 |
| `uv sync` | Creates `.venv`, installs dev deps, compiles the Rust extension (`glasshouse._core`) | One command, whole environment. Re-run after changing `Cargo.toml`/`pyproject.toml` |

| `curl -sSfO https://nodejs.org/dist/latest-v22.x/node-v22.x.y-linux-x64.tar.xz` then untar into `~/.local/node` and add `~/.local/node/bin` to PATH (verify the SHA256 from `SHASUMS256.txt`) | Installs Node 22 user-locally | Only needed to work on the report viewer (`report/`); Python users never need Node |
| `cd report && npm ci` | Installs the pinned JS dev tools (TypeScript 5.9, vitest, jsdom) | Exact versions from `package-lock.json`; `npm audit` and OSV scan them in CI |

You need a C compiler for the Rust build (`gcc` on Linux/WSL, Xcode CLT on macOS, MSVC Build
Tools on Windows).

## 1. Every day

| Command | What it does | Why |
|---|---|---|
| `./check.sh` | fmt-check → clippy → cargo test → uv sync → ruff → mypy → pytest | **The gate.** Green before every push. First failure stops it |
| `./check.sh fix` | Auto-formats Rust and Python, applies safe ruff fixes, then runs the gate | Fixes the boring failures for you |
| `./check.sh rust` / `./check.sh py` | One language only | Faster loop while you're in one side |
| `uv run pytest -k name` | Runs matching tests | Same as pytest; `uv run` uses the project venv |
| `cargo test -p glasshouse-core name` | Runs matching Rust tests | Rust unit tests live next to the code |
| `uv run maturin develop` | Rebuilds only the extension into `.venv` | When you edit `crates/` and don't want a full `uv sync` |
| `uv run python -c "import glasshouse; print(glasshouse.__version__)"` | Smoke test the install | Proves the extension loaded |
| `uv run glasshouse list` | Lists the named benchmarks | Recipes anyone can rerun |
| `uv run glasshouse bench fremtpl2_glm` | Runs a benchmark, writes `benchmarks/<name>/report.{json,md,html}` — the html is the interactive suite | First run fetches the data from OpenML (~70 s) into `~/.cache/glasshouse`; after that ~25 s |
| `cd report && npm run check` | build → checked-in `python/glasshouse/_report/report.js` must be unchanged → vitest on the fixture → size budget → `npm audit` | The viewer's gate. If the built file differs, you forgot to rebuild after editing `src/` |
| `uv run python -c "from glasshouse import report; ..."; r.to_html('out.html')` | Writes one self-contained HTML report | Double-click to open; Plotly from a pinned CDN, tables if offline |
| `uv run python tests/make_report_fixture.py` | Regenerates `tests/fixtures/report_small.json` | The document the TypeScript report tests render; regenerate when `report.build` changes shape (and bump `python/glasshouse/_report/schema.json`) |
| `RAYON_NUM_THREADS=4 uv run glasshouse bench fremtpl2_glm` | Caps the Rust solver at 4 threads (default: every core) | Results are identical whatever the count; cap it on a shared machine or to compare timings |
| `GLASSHOUSE_NETWORK_TESTS=1 uv run pytest -k pinned` | Reruns the committed benchmark and checks the numbers match to 1e-6 | The regression test that stops numbers drifting silently; needs the cached data |

## 2. Where things live

- `crates/core/` — pure Rust numerics (families, deviances, metrics). No Python types. Tests inline.
- `crates/py/` — PyO3 bindings. **No logic** — converts arrays, calls core, maps errors to `ValueError`.
- `python/glasshouse/` — the Python API users import. Docstrings with runnable examples live here.
  `_core.pyi` is the type stub for the extension — update it when `crates/py/src/lib.rs` changes
  (`tests/test_stub_sync.py` fails if you forget).
- `tests/` — pytest: golden tests vs scikit-learn/statsmodels/glum/R fixtures, hypothesis properties.
- `docs/` — `methods.md` (formulas, citations, weights), `comparing-models.md` (the report
  end to end, every example run by `tests/test_docs_examples.py`), `report-suite.md` (plan). Tracked.
- `report/` — the TypeScript viewer source (`src/*.ts`, vitest); it compiles into `python/glasshouse/_report/report.js`, which sits next to `schema.json` (the JSON contract) and `template.html` so the wheel ships them
  (built, checked in), `template.html`, `test/`. Python `report.to_html` glues them.
- `check.sh` — the gate. `COMMANDS.md` — this file. `CLAUDE.md` — the rules.
- `*.md` at the top level other than the whitelisted ones (README, CLAUDE, COMMANDS, CHANGELOG)
  are **gitignored on purpose** — plans and session notes are internal.

## 3. Adding things (the recipe)

**A family (distribution).** One row in `crates/core/src/family.rs`: its name in `parse`, the
`y`/`mu` support rules (plain-English messages included), and its `unit_deviance`. Every metric
picks it up automatically. Golden test vs scikit-learn/statsmodels in `tests/test_metrics.py`.

**A metric.** (1) Formula + citation in `docs/methods.md`. (2) Rust: `crates/core/src/metrics.rs`
using the shared `validate` + weighted path; unit tests inline. (3) Binding: one `#[pyfunction]` in
`crates/py/src/lib.rs`, add to the module, add to `_core.pyi`. (4) Python facade in
`python/glasshouse/metrics.py` with a docstring: what it's for, when it's good, when it lies,
runnable example. (5) `tests/`: golden vs a named reference + hypothesis properties. (6) `./check.sh`.
Missing any of these → not done.

**A dependency.** Rust: `cargo add <crate> -p <crate-name>`. Python: `uv add <pkg>` (runtime) or
`uv add --group dev <pkg>`. Then justify it in one sentence in the PR. Core runtime deps stay
at numpy (+ optional pyarrow/polars).

## 4. Git

| Command | What it does |
|---|---|
| `git switch -c feat-thing` | New branch off `main` (never work on `main`) |
| `git add <files>` | Stage explicitly — never `git add -A` (that's how internal notes get committed) |
| `git commit -m "feat(metrics): gamma deviance"` | Plain English, `type(scope): what`, no AI attribution |
| `git push -u origin HEAD` | Push the branch (never `HEAD:main`) then open a PR |

CI runs the same `./check.sh` steps on Linux + macOS + Windows and both Python versions, plus
`cargo audit`, `pip-audit`, gitleaks.

### Cutting a release

| Step | Command | Why |
|---|---|---|
| 1 | Bump `version` in `pyproject.toml`, `Cargo.toml`, `python/glasshouse/__init__.py`, `report/package.json`, `report/src/main.ts`; turn `## Unreleased` in `CHANGELOG.md` into `## X.Y.Z — date` | One version everywhere; the release workflow refuses a tag that does not match `pyproject.toml` |
| 2 | Open the PR, get it green, merge | Releases are cut from `main` only |
| 3 | `git switch main && git pull && git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z` | The tag is the trigger. `.github/workflows/release.yml` builds wheels (Linux, macOS, Windows; 3.12 and 3.13) and the sdist, then publishes |
| rc | Tag `vX.Y.ZrcN` instead | Goes to TestPyPI; `pip install -i https://test.pypi.org/simple/ glasshouse==X.Y.ZrcN` to try it |

Publishing uses PyPI Trusted Publishing: no token in the repo, GitHub proves the job's identity
to PyPI. One-time setup by the project owner, on pypi.org (and test.pypi.org): add a pending
publisher for `Matkicail/glasshouse`, workflow `release.yml`, environment `pypi` (`testpypi`).
Then create the two environments under the repo's Settings → Environments.

### The docs site

| Command | What it does | Why |
|---|---|---|
| `uv run --group docs mkdocs serve` | Serves the docs at http://127.0.0.1:8000 with live reload | Preview `docs/` as the site reads |
| `uv run --group docs mkdocs build --strict` | Builds `site/`; a broken link fails the build | The docs check `.github/workflows/docs.yml` runs on every PR; a push to `main` deploys to GitHub Pages (enable Pages → Source: GitHub Actions once) |

## 5. When it breaks

- `import glasshouse` → `ImportError: _core` — the extension isn't built: `uv sync` (or
  `uv run maturin develop`).
- `error: linker 'cc' not found` — install a C compiler (see §0).
- `uv sync` rebuilds every time — expected when `crates/` changed (`cache-keys` in pyproject).
- `ValueError: y must be finite and inside the family's support, but 12 row(s) are not — …` —
  that's the library refusing bad data on purpose. The message names the fix.
