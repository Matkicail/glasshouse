# The report suite — plan

Status: **design, not built.** This is the plan for Tier 4.5 and what it grows into. It exists
so the goal is understood before code is written, and so the TypeScript side is designed for
reliability from the start rather than bolted on.

## The goal, in one paragraph

Someone has fitted several models on one dataset — a GLM, a boosted tree, a net, whatever —
and wants to know which one to trust and *why*. They hand over the actuals, each model's
predictions, the weights or exposure, and say what kind of task it is. They get back a single
self-contained HTML file: every score on the panel for that task, every model side by side
against each other and against the naive baseline, the charts that show *where* the models
differ, and the residuals that show *what* each one gets wrong. It opens from a double-click,
needs no server, no install, no internet, and reads the same way for a pricing actuary, a
fraud analyst, and a forecaster — because it knows which of those it is talking to.

It is a suite, not a chart: the scorecard, the comparison, the four curves, residual
diagnostics, and the data's own provenance (citation, cleaning rules, split kind), in one
place, from one JSON.

## Who uses it, and what they need to see first

| Person | Task | The first thing they look for |
|---|---|---|
| Pricing actuary | frequency, severity, pure premium | Double lift between the incumbent and the challenger; A/E by decile; balance |
| Fraud / rare-event analyst | binary, very imbalanced | Precision–recall curve and average precision; MCC at the working threshold; how many alerts per true positive |
| Credit / risk scorer | binary, ranking matters | ROC, KS, the score distribution by class; calibration of the probability |
| Forecaster / general regression | continuous, often time-ordered | Residuals over time; MAE/RMSE vs naive; calibration of the level |
| Anyone comparing models | all of the above | "Which one wins, on what, by how much — and is it better than doing nothing?" |

The report's first screen answers the last row. Everything else is one click deeper.

## Task types — the report must know which one it is

The task is **declared**, never guessed (same rule as splits). It selects the panel, the
naive baseline, the default charts, and the residual definition.

| task | `y` | `pred` | weight | naive baseline | panel (from `scorecard`) | charts | residuals |
|---|---|---|---|---|---|---|---|
| `frequency` | counts or rates | rate | exposure | weighted mean rate | Poisson deviance, D², Gini, norm. Gini, balance, calibration | Lorenz, lift, double lift, calibration | deviance & Pearson residuals; A/E by feature |
| `severity` | positive amounts | amount | claim count | weighted mean | gamma / Tweedie deviance, D², Gini, balance, MAE, RMSE | lift, double lift, calibration, Lorenz | deviance residuals; A/E by feature; QQ against gamma |
| `pure_premium` | amount incl. zeros | amount | exposure | weighted mean | Tweedie(p) deviance, D², Gini, balance | Lorenz, lift, double lift, calibration | deviance residuals; zero-mass check |
| `binary` | 0/1 | probability | optional | class prior | log-loss, Brier, ROC-AUC, average precision, KS, MCC, F1, calibration | PR curve, ROC, reliability, score-by-class histogram, alerts-vs-captures | probability residual by bin; confusion at chosen threshold |
| `regression` | real | real | optional | weighted mean | gaussian deviance (MSE), R², MAE, RMSE, (MAPE/sMAPE if `y ≠ 0`), calibration | lift, calibration, residual-vs-fitted | raw residuals; QQ; residual over time if time given |
| `ranking` | non-negative | score | optional | constant | Gini, normalised Gini, KS, ROC-AUC if binary | Lorenz, CAP, cumulative gains | rank residuals (actual decile − predicted decile) |

`frequency`, `severity` and `pure_premium` are the actuarial trio; `binary`, `regression` and
`ranking` are the general ML ones. The same machinery, different panels — exactly the XGBoost
"wide scope, narrow excellence" shape.

## Inputs — the dataset spec

One dataset per report (several is a later feature). The user provides:

- `task`: one of the six above; for `pure_premium` also the Tweedie `power`.
- `y` (actuals), `weight` (exposure / count / none), optional `offset` already applied inside
  the predictions (the report works on the response scale).
- `models`: a mapping of label → predictions on the *same rows*. Two to ~ten; the UI stays
  readable at ten.
- optional `features`: a few columns to slice residuals and A/E by (`Region`, `DrivAge`,
  `Amount` band). Numeric features are binned by weighted decile; categoricals by level.
- optional `time`: for residuals-over-time and to state that the split was time-ordered.
- optional `groups`: to state that the split was grouped.
- `split`: the `Splits.to_dict()` that produced the held-out rows, so the report can say
  "scored on fold 3 of a stratified 5-fold, seed 0" — provenance, not decoration.
- `describe`: the `data.describe()` text, or the user's own sentence about the data.
- `sample`: "all rows" or a stated fraction/count — large data is downsampled *for the
  scatter charts only*; every score and every binned curve uses all rows. The report states
  which it did.

All of this is one JSON document, produced by Python (`glasshouse.report.build(...)`) from the
existing `scorecard.to_dict()` and `curves.to_dict()` plus the new residual tables. The
Python side computes everything; the TS side only draws.

## The screens (v1)

1. **Overview** — one row per model: the task's panel with a "better than naive?" tick per
   metric, and the winner per metric highlighted across models. Sorted by the task's
   primary metric (deviance for the actuarial trio, average precision for binary, MAE for
   regression, normalised Gini for ranking) — with a visible note that the primary is a
   convention, not a verdict. The provenance block (data, cleaning, split, sample) sits
   under it, copyable.
2. **Compare** — pick model A and model B: the `compare()` table, the double lift, and both
   calibration curves overlaid. This is the pricing-review screen.
3. **Curves** — the task's charts with model toggles: Lorenz/lift/calibration, or PR/ROC/
   reliability/score histograms, or residual-vs-fitted. One chart type at a time, all models
   on it.
4. **Residuals** — per model: the residual definition for the task; a residual-vs-fitted
   scatter (sampled); A/E or mean residual **by feature** (each provided feature, binned),
   which is where a modeller finds the segment a model gets wrong; residual over time if
   `time` was given; QQ where a distribution is assumed.
5. **Binary extras** (only when `task = binary`) — threshold slider driving the confusion
   counts, precision/recall/MCC/F1, and "alerts per true positive"; the cost of a threshold
   in words a fraud team uses.

Every number on screen is traceable to a field in the JSON, and the JSON is downloadable
from the page. Every chart has a one-line "how to read this" caption from the metrics guide
("Gini is blind to calibration — read it next to A/E").

## Residuals — defined per task, computed in Rust

- **Deviance residual** `sign(y − mu) · sqrt(w · d(y, mu))` — the family's own residual;
  for a well-specified model roughly symmetric with unit variance. Reuses `unit_deviance`.
- **Pearson residual** `(y − mu) · sqrt(w / V(mu))` — the same scale as the dispersion.
- **Raw residual** `y − mu` for `regression`; **probability residual** `y − p` for `binary`.
- **A/E by feature**: `sum(w y) / sum(w mu)` per bin or level, with the bin's weight — the
  calibration table sliced by a feature. This is the one that finds segments.
- **QQ**: sorted deviance residuals against normal quantiles (a straight line means the family
  is plausible; a curve means it is not).

These are new Rust functions (`residuals`, `ae_by_feature`) with golden tests against
statsmodels' `resid_deviance` / `resid_pearson`, exposed like every other metric, and used by
the Python report builder. Nothing is computed in TypeScript.

## The TypeScript side — lightweight, local, reliable

- **One folder, no framework, no bundler.** `report/src/*.ts` compiled by `tsc` (strict) to
  `report/dist/report.js`, which is **checked in** so Python users never need Node. Build
  is a one-line `npm run build` for contributors; CI verifies the checked-in output matches
  a fresh build (no drift).
- **One HTML file out.** Python templates `report.html` = the JS + the JSON (inline, in a
  `<script type="application/json">`) + a small CSS. Opens offline. Plotly.js is loaded from
  a pinned CDN URL with a **local fallback**: if Plotly is unreachable, every table and the
  provenance still render and each chart shows its data as a table. Optional `--inline-plotly`
  embeds the library (~3.5 MB) for air-gapped use.
- **One schema.** `report/schema.json` (JSON Schema) describes the document. Python validates
  what it writes (a test); TypeScript types are generated from the schema (`json-schema-to-
  typescript`) so a change to the contract fails both builds, not one.
- **Tested like code.** vitest renders the report from `tests/fixtures/report_small.json`
  (the same fixture the Python tests write) and asserts every screen builds; a size budget
  (JS + CSS < 200 KB, HTML without data < 300 KB); TS strict; ESLint minimal.
- **Accessible and readable.** Keyboard-operable selectors, colour-blind-safe palette, the
  same model always the same colour across every chart, numbers formatted by task (rates as
  `0.0523`, amounts with thousands separators, probabilities as percentages).
- **Not in v1:** editing, re-fitting, uploading a second dataset, saving state, PDF export.
  Plain browser print works.

## How it lands, in order

1. **Rust residuals** (`residuals`, `ae_by_feature`) + goldens vs statsmodels. Small.
2. **`glasshouse.report.build(...)`** producing the JSON per the schema, from scorecards,
   curves, residual tables and provenance; the `bench` command writes it. Python tests
   validate the schema and the fixture.
3. **TS v1**: Overview + Compare + Curves for the actuarial trio and `regression`.
4. **Binary screen** (PR/ROC/reliability/threshold) and `ranking` — the fraud and credit
   readings.
5. **Residuals screen** by feature and over time.
6. Docs page: "reading a report" — one worked example per task type, screenshots.

Go/no-go after step 3: if the single-file report is not obviously more useful than the
Plotly HTMLs Python already writes, stop there and keep the JSON contract only.

## Adopted from M's reference benchmarker (Downloads/examples, 2026-09-01)

A rustystats-based nine-model benchmark with a 7-tab report served as a reference. Already
aligned: deviance conventions, weighted Gini, A/E by decile, double lift, stable model
colours. Adopted into the viewer (the JSON already carried the data): the one-way analysis
(actual vs predicted by feature bin with exposure bars), fit seconds on the leaderboard, and
the residuals tab. Adopted into the backlog, in order of value:

1. **Profit matrix / tournament** — DONE (2026-09-05): `glasshouse.tournament`, the
   `tournament` block of the document (pairs + overall), the win-set table on Compare and
   the tournament on Overview. Priced tasks only; ties split equally.
2. **Two-feature A/P heatmap** — A/E on a bins-times-bins grid of two features, cells under a
   weight floor greyed out. Finds interaction segments one-way views miss.
3. **Data overview tab** — per-feature distributions with the outcome rate overlaid.
4. **SHAP for tree models** — waits for GBDT wrappers in the bench. Partial dependence
   and permutation importance landed first (2026-09-05, the Model tab): model-agnostic,
   on held-out rows, with the fold spread; SHAP would add per-row attributions on top.

## Out of scope, on purpose

Multi-dataset dashboards; live model serving; anything that recomputes a metric in the
browser; a design system; hosting. Each is a rabbit hole with a plausible name.
