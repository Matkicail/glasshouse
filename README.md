# glasshouse

Interpretable, well-rounded ML with a Rust core and a Python API.

Glass-box models (GLMs first) and metrics that tell the truth: weighted, exposure-aware,
reported as a panel rather than one number, and always measured against a naive baseline, so
you know whether a model is actually helping. One self-contained HTML report compares any
models on one dataset: scorecard, tournament, curves, partial dependence, residuals by
segment.

Actuarial families (Poisson, gamma, Tweedie, offsets) are first-class, but they are rows in
the table, not the identity: rare-event classification, churn and general regression go
through the same machinery.

```bash
pip install "glasshouse[data]"     # numpy core; [data] adds pandas, pyarrow, scikit-learn
```

## Ten lines, versus naive

Fit a Poisson GLM with a one-hot region and a penalised smooth on age, score it held-out on
five folds against the mean-rate baseline, and get the scorecard.

```python
import numpy as np
import pandas as pd

from glasshouse import GLM, bench, splits
from glasshouse.bench import ModelSpec, TaskSpec

rng = np.random.default_rng(0)
n = 6000
df = pd.DataFrame({"region": rng.choice(["north", "south", "east"], n), "age": rng.uniform(18, 80, n)})
df["Exposure"] = rng.uniform(0.2, 1.0, n)
rate = np.exp(-2.5 + 0.0015 * (df.age - 45) ** 2 + df.region.map({"north": 0.0, "south": 0.3, "east": -0.2}))
df["ClaimNb"] = rng.poisson(rate * df.Exposure).astype(float)

task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
glm = ModelSpec("glm", lambda: GLM(family="poisson", terms={"region": "onehot", "age": "smooth"}), ["region", "age"])
folds = splits.stratified(df.ClaimNb, k=5, seed=0)
result = bench.run(df, task, [glm], folds, features=["region", "age"], dataset="synthetic")
print(result.to_markdown())
result.write("reports/readme")   # report.json, report.md and report.html
```

```text
# synthetic — poisson (ClaimNb)

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 6,000. Models: glm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm | naive |
|---|---|---|
| deviance | **0.71274 ± 0.017** | 0.77801 |
| d2 | **0.083577 ± 0.019** | 8.8818e-17 |
| gini | **0.34708 ± 0.033** | 0 |
| normalized_gini | **0.36984 ± 0.035** | 0 |
| balance | **0.99895 ± 0.026** | 1 |
| rmse | **0.53306 ± 0.02** | 0.54543 |
| mae | **0.28067 ± 0.0051** | 0.29619 |
| r2 | **0.044606 ± 0.013** | 0 |

Fit time (all folds): glm 0.9s.
```

Every number is held-out, every score takes the exposure as its weight, and the naive column
is the same folds scored with the weighted mean. `report.html` is the full suite; double-click
it.

If you already have predictions from any library, skip the fitting: `report.build(task, y,
{"glm": mu_glm, "gbm": mu_gbm}, weight=exposure, features=...)` and `to_html`.

## What the report shows

| tab | what it answers |
|---|---|
| Overview | every model on the task's panel with a tick against naive, the tournament (every risk to the cheapest model: who wins what, at what actual over expected), provenance |
| Data | the outcome and the weight before any model: distributions, and each feature's weight and outcome rate |
| Compare | two models: which wins each metric, their win sets, the double lift, both calibrations |
| Curves | Lorenz, lift, calibration, one-way actual vs predicted by feature; ROC and precision-recall for binary tasks |
| Model | permutation importance and partial dependence for every model; coefficients, relativities and "explain a row" (one bar per feature, adding up to the price) for the glass-box ones |
| Residuals | deviance and Pearson residuals, A/E by feature, and A/E on the grid of every pair of features with thin cells greyed: the interaction view |
| Threshold | binary only: the cost of a cut in alerts per catch |

`docs/comparing-models.md` walks through it with one worked example per task type; every code
block there runs as a test.

## Five real benchmarks

Each is one command, reproducible from a seeded recipe, with a committed summary and a drift
test that pins its numbers. Five shapes, so the scorecard is proven where models usually go
wrong: heavy-tailed claims, a rare event, a time-ordered series, and a churn book.

**French motor claim frequency** (freMTPL2, 678,013 policies, Poisson with exposure offset,
stratified 5-fold): `uv run glasshouse bench fremtpl2_challengers`.

| metric | glm_full | glm_splines | glm_smooth | glm_interaction | lightgbm | naive |
|---|---|---|---|---|---|---|
| deviance | 0.60493 ± 0.0025 | 0.59279 ± 0.0021 | 0.59198 ± 0.0021 | 0.59128 ± 0.0021 | **0.5724 ± 0.0026** | 0.62488 |
| d2 | 0.03192 ± 0.00033 | 0.051355 ± 0.0011 | 0.052654 ± 0.0011 | 0.053769 ± 0.0014 | **0.08399 ± 0.0023** | 0 |
| gini | 0.39515 ± 0.016 | 0.48879 ± 0.016 | 0.48938 ± 0.016 | 0.49098 ± 0.016 | **0.53505 ± 0.021** | 0 |
| balance | **1 ± 0.0049** | 1 ± 0.0037 | 1 ± 0.0033 | 0.99991 ± 0.0038 | 0.99912 ± 0.0037 | 1 |

The splined GLM closes most of the gap to LightGBM on ranking while staying a table of
relativities. The two-feature A/E grids in the report name the biggest interaction it
misses, age by bonus-malus, and adding that one term as a tensor-product spline buys a
little more; the research track's nets on top of the same GLM (`docs/research.md`) say the
rest of the gap is many small interactions, not one.

**French motor claim severity** (freMTPL2sev, 24,944 policies with claims, gamma, weight =
claim count): `uv run glasshouse bench fremtpl2_sev`.

| metric | glm_gamma | lightgbm | naive |
|---|---|---|---|
| deviance | **1.5641 ± 0.14** | 1.5785 ± 0.21 | 1.5635 |
| d2 | **-0.0047 ± 0.045** | -0.0083 ± 0.011 | 0 |
| gini | **0.057 ± 0.075** | 0.042 ± 0.058 | 0 |
| balance | **1.011 ± 0.12** | 1.081 ± 0.13 | 1 |

Neither model beats the mean claim on held-out deviance, and the panel says so. That is the
known result on this data: severity is close to unpredictable from the policy features, and
a report that hides the naive row would have you deploy a model that adds nothing.

**Hourly bike rentals** (17,379 hours over two years, Poisson, time-ordered folds that train
strictly before they test): `uv run glasshouse bench bike_sharing`.

| metric | glm_poisson | lightgbm | naive |
|---|---|---|---|
| deviance | 66.51 ± 19 | **43.02 ± 14** | 161.63 |
| d2 | 0.586 ± 0.096 | **0.733 ± 0.077** | 0 |
| balance | **1.401 ± 0.17** | 1.405 ± 0.15 | 1 |
| mae | 85.4 ± 22 | **72.2 ± 23** | 146.7 |

Both models rank the hours well and both are 40% under on the level: demand grew from 2011
to 2012 and a model trained on the past cannot see the growth. A random split would have
hidden that; the time split and the residuals-over-time view show it.

**Telco customer churn** (7,043 customers, 26.5% churn, logistic vs lasso logistic on
fifteen one-hot factors, stratified 5-fold): `uv run glasshouse bench telco_churn`.

| metric | logistic | lasso_logistic | group_lasso_logistic | naive |
|---|---|---|---|---|
| log_loss | 0.4186 ± 0.0097 | **0.4183 ± 0.01** | 0.4266 ± 0.012 | 0.5786 |
| roc_auc | 0.8433 ± 0.0088 | **0.8434 ± 0.0093** | 0.8373 ± 0.012 | 0.5 |
| average_precision | 0.6539 ± 0.016 | **0.6543 ± 0.018** | 0.6484 ± 0.014 | 0.2654 |
| mcc | **0.4663 ± 0.024** | 0.4642 ± 0.022 | 0.4400 ± 0.02 | 0 |

The lasso path chosen by cross-validation lands on the same model as the plain logistic,
which is the honest answer on a book this size with fifteen well-chosen factors. The group
lasso on the one-standard-error rule drops whole factors and pays a little log-loss for a
shorter rating table; the report shows the price of that choice rather than hiding it.

**Credit-card fraud** (284,807 transactions, 0.17% positives, logistic, stratified 5-fold):
`uv run glasshouse bench creditcard_glm`.

| metric | logistic | naive |
|---|---|---|
| log_loss | **0.0041 ± 0.00058** | 0.012715 |
| average_precision | **0.7598 ± 0.045** | 0.0017275 |
| roc_auc | **0.97429 ± 0.0077** | 0.5 |
| mcc | **0.73176 ± 0.035** | 0 |

Under that imbalance the ROC-AUC flatters; average precision against the prior, and the
Threshold tab's alerts per catch, are the numbers a fraud team can act on.

## What is in the box

- **Metrics** that all take `sample_weight`: deviance for five families, D², Gini and
  normalised Gini, calibration table and balance, log-loss, Brier, ROC-AUC, average
  precision, KS, MCC, F1, and the plain regression errors. Golden-tested against
  statsmodels, scikit-learn and glum; property-tested with hypothesis.
- **GLM** by IRLS in Rust: five families, identity/log/logit links, offsets, weights, robust
  standard errors, one-hot and target encoders that never let a row see its own y, B-spline,
  piecewise linear and penalised smooth terms with the penalty chosen by GCV, tensor-product
  interaction terms, monotone constraints, per-row attributions, and lasso,
  ridge, elastic-net and group lasso with a cross-validated path. Parallel row passes that give the same
  bits whatever the thread count.
- **Splits** that declare what the data is (random, stratified, grouped, time-ordered), so
  leakage is a property of the split, not of the transform.
- **The bench and the report**: fit anything with `fit` and `predict` on folds, score it,
  write one HTML file. Adapters for glum, scikit-learn and LightGBM are included.

- **A research fence** (`pip install "glasshouse[research]"`, torch): neural models that
  must beat the spline plus monotone GLM on held-out deviance and calibration before they
  leave it. So far: the CANN (the GLM frozen, a small net learning its residual), an
  additive net (one net per feature, corrections you can draw) and LocalGLMnet (the GLM's
  coefficients as functions of the row), one class and one training loop, each showing up on
  "explain a row" as extra columns next to the GLM's terms.

Not trying to be scikit-learn. It does a few things and does them well.

## Develop

```bash
uv sync        # builds the Rust extension into .venv
./check.sh     # the gate: fmt, clippy, cargo test, ruff, mypy, pytest
```

`COMMANDS.md` explains every command, `CLAUDE.md` the rules, `docs/methods.md` the formulas
and the weights convention. The report viewer is TypeScript under `report/` and compiles into
the package, so Python users never need Node.

## Licence

MIT OR Apache-2.0, at your option. See `LICENSE-MIT` and `LICENSE-APACHE`.
