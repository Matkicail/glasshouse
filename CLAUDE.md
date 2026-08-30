# glasshouse — rules for anyone (or any model) working in this repo

Read fully before writing code. These override defaults. If a task conflicts with a rule, stop
and say so. The full plan and backlog live in `plan2-glasshouse.md` (untracked, on the dev's
machine) — read the newest `sessionNN-addendum.md` too if one exists.

## What this is
Interpretable, well-rounded ML: glass-box models (GLMs first), truthful weight-aware metrics, and
a scorecard that always shows whether you beat a naive baseline. Rust core + Python API. Actuarial
families (Poisson/gamma/Tweedie, offsets) are rows in the table, not the identity. Not trying to
be scikit-learn — does a few things and does them well.

## Stack (decided)
Rust core (`crates/core`, ndarray + rayon) · PyO3 bindings (`crates/py`, no logic) · Python
package (`python/glasshouse`) · maturin · uv · ruff · mypy --strict · pytest + hypothesis ·
proptest · mkdocs-material · optional PyTorch under `glasshouse[research]` · Python ≥ 3.12 ·
licence MIT OR Apache-2.0.

## Design rules
- All code is debt. Smallest PR that fully solves a stated need. No speculative abstraction.
- Interfaces only with two real implementations or at a committed boundary. Pre-approved:
  `Family`, `Link`, the data-in adapter, the `Model` protocol. Everything else is a function first.
- One factory: the string→implementation registry (`family="poisson"`). No ABCs "for later".
- Composition over inheritance; frozen result objects; fitting is a function.
- DRY at the numerics (one deviance, one spline basis, one weight path), not at the API.
- Rust owns numerics, Python owns ergonomics, bindings own nothing.
- Fails early and clearly, with reasons why: bad dtypes / NaNs / zero exposure / y outside the
  family's support are caught before a model runs, with row counts and the fix. Never coerce or
  drop silently.
- Every public thing has a runnable docstring example, a golden or property test, and a guessable
  name.

## Correctness rules
- A metric is a rumour until it matches the reference. Nothing merges without a golden test
  (statsmodels / glum / sklearn / R fixtures, with tolerances) plus property tests.
- Every metric takes `sample_weight`; offsets are first-class. Weight semantics documented once in
  `docs/methods`, linked, not repeated.
- The scorecard is a panel, never one number, and always includes the naive baseline row.
- Seeds and split indices are stored artifacts; benchmark reports are committed and drift-tested.

## Process rules
- `main` + feature branch + PR. Never push to `main`. `git push -u origin HEAD`.
- Stage explicitly; never `git add -A`. `*.md` is gitignored except the whitelisted docs.
- Commits: plain English, `type(scope): what`, no AI attribution, no session IDs.
- `./check.sh` green before push (fmt, clippy, cargo test, ruff, mypy, pytest). If it can't run
  locally, CI is the gate — say so in the PR.
- No pickle for save/load. Actions pinned by SHA. Trusted Publishing on tag.
- Docs and tests ship in the same PR as the code. "I'll come back to it" is not a plan.
- `COMMANDS.md` explains every command: what, how, why — good enough for a person without Claude.

## Working style
Plain English, explain the why. Execute when the path is clear; ask only at real forks. Give
honest cost/benefit and recommend stopping on marginal wins. Verify before asserting — run the
gate, show the output. Write a session addendum at the end of a working session.
