# glasshouse

Interpretable, well-rounded ML with a Rust core and a Python API.

Glass-box models (GLMs first) and metrics that tell the truth: weighted, exposure-aware, reported
as a panel rather than a single number, and always measured against a naive baseline — so you
know whether your model is actually helping.

Actuarial families (Poisson, gamma, Tweedie, offsets) are first-class, but they're rows in the
table, not the identity: rare-event classification, churn, forecasting and general regression/classification go
through the same machinery.

**Status:** pre-alpha. The first milestone (metrics → GLM → scorecard → benchmark report) is in
progress. Not on PyPI yet.

```python
from glasshouse.metrics import poisson_deviance

poisson_deviance(y, mu, sample_weight=exposure)
```

## Develop

```bash
uv sync        # builds the Rust extension into .venv
./check.sh     # the gate: fmt, clippy, cargo test, ruff, mypy, pytest
```

See `COMMANDS.md` for what every command does and why, `CLAUDE.md` for the rules, and
`docs/methods.md` for the formulas, references and the weights convention.

## Licence

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
