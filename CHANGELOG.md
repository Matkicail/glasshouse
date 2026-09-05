# Changelog

All notable changes, newest first. Pre-1.0: minor versions may break the API; the entry says so.

## Unreleased

### Added
- `Smooth(monotone="increasing")` / `BSpline(monotone="decreasing")`: shape-constrained
  spline terms. The solver honours the constraint at every IRLS step (an active-set QP on
  the chain of coefficient differences), GCV and the balance property carry over, tied
  coefficients count once in the edf. Golden against the exact enumeration of the KKT
  system.

### Changed
- The GLM solver runs its row passes in parallel (rayon) with fixed-chunk partial sums, so a
  fit is identical whatever the thread count; a full fit on a 540k-row fold went from
  3.4 s to 0.4 s. The GCV search for smooths warm-starts each evaluation from the previous
  one and skips the inference it does not read: one smooth on that fold went from 65 s to
  8 s. `_core.glm_fit` gains `warm_start=` and `inference=`.
- `fremtpl2_challengers` gains a `glm_smooth` row (GCV smooths on the four numeric
  features), and all four benchmarks are re-pinned on the fixes below.

### Fixed
- `bench` scored rate tasks on `mu / exposure`, which breaks exact ties by rounding and let
  the Gini of a model with many identical rows move at the fourth decimal with the solver's
  last bits. It now scores the model's rate (`predict` with no offset); the pins are stable
  to 1e-6 across solver changes again.

### Added
- `docs/comparing-models.md`: the comparison report end to end, with a worked example per
  task type (frequency GLM vs GCV smooth vs LightGBM on folds; binary and regression from
  scikit-learn predictions) and how to read each tab. Every code block runs as a test.
- `encoders.Smooth` and GLM `terms={"age": "smooth"}`: penalised P-spline smooths whose
  wiggliness is chosen by GCV during `fit` (pin it with `Smooth(lam=...)`). The model gains
  `edf_`, `lambda_` and the searched `gcv_` grid; the intercept stays unpenalised so balance
  survives. Solver golden vs statsmodels `GLMGam` to machine precision.
- Family table (gaussian, poisson, gamma, tweedie with power, binomial) with one shared
  deviance path; `metrics.deviance`, `metrics.d2`, and named per-family functions.
- Ranking: `metrics.gini`, `metrics.normalized_gini` — exposure-weighted, ties grouped.
- Calibration: `metrics.calibration_table` (A/E by weighted decile), `metrics.balance`.
- Classification: `classification.threshold_metrics` (accuracy, balanced accuracy, precision,
  recall, F1, MCC), `roc_auc`, `average_precision`, `ks`, `log_loss`, `brier`.
- Regression: `regression.rmse`, `mae`, `mape`, `smape`, `msle`, `r2`.
- The panel: `scorecard.scorecard` (always with a naive baseline row) and `scorecard.compare`.
- `docs/methods.md`: formulas, references, and the weights convention.
- `GLM`: IRLS in Rust with step-halving, offsets, weights, all five families, identity/log/
  logit links; standard errors, null deviance, dispersion, per-row contributions, a printable
  fit trace, JSON round-trip, HC1 robust standard errors. Golden-tested against statsmodels.
- `report.to_html` and the `report/` TypeScript viewer (Overview, Compare, Curves; A/E by
  feature; tables fallback when Plotly is unavailable): one self-contained HTML per report.
- `report.build` / `report.validate` + `report/schema.json`: the whole comparison document for
  a declared task type (frequency, severity, pure_premium, binary, regression); `curves.roc` /
  `curves.pr`.
- `residuals.deviance` / `pearson` (golden vs statsmodels) and `residuals.ae_by_feature`: actual /
  expected sliced by a numeric (equal-weight bins) or categorical feature.
- `encoders.BSpline` and GLM `terms={"age": BSpline(df=8)}` (or `"spline"`): cubic B-spline
  terms with training-fold quantile knots, clamped extrapolation, Rust Cox–de Boor kernel
  golden against scipy.
- Binary reports carry a precomputed 101-point threshold grid and the viewer gains a
  Threshold tab (slider, workload table, precision/recall/MCC chart); NaN metrics now
  serialize as null so every report parses in the browser; `creditcard_glm` benchmark.
- `gbdt.LightGBM`: gradient-boosted trees behind the same bench protocol — GLM-family
  objectives, offsets via init_score, fold-safe early stopping, native categoricals; the
  `fremtpl2_challengers` benchmark pits it against the GLM.
- `foss.GlumPoisson` / `foss.SklearnPoisson` adapters and the `fremtpl2_vs_foss` benchmark:
  ours vs glum vs scikit-learn on identical designs and folds, deviance agreeing to 5 decimals.
- `glasshouse bench <name>` now writes the full suite: `report.json` (glasshouse-report/1 with
  a per-fold `bench` block), `report.md`, and the interactive `report.html`.
- `bench.run` + `glasshouse bench <name>`: models × folds on one dataset, scored the same way,
  written to `report.json` / `report.md`; `benchmarks/fremtpl2_glm` committed and pinned.
- `curves.lorenz` / `lift` / `double_lift` / `calibration` as data (the JSON contract), and
  `plots.*` Plotly renderers over them (`glasshouse[plots]`).
- `data.load` / `data.describe`: freMTPL2 frequency (documented Wüthrich–Merz cleaning) and the
  ULB credit-card set, fetched from OpenML once and cached; needs `glasshouse[data]`.
- `encoders.OneHot` / `TargetEncode` / `Standardize`, and `GLM(terms=...)` with `fit(fold=...)`:
  encoders fit on the training fold only; target encoding is out-of-fold, or past-only when
  the fold is time-ordered; no row ever sees its own outcome.
- `splits.kfold` / `splits.stratified` / `splits.time_ordered` / `splits.grouped`: folds that carry the declared
  kind of the data (random / time / group), stored as index arrays.
- `arrays.to_vector` / `arrays.to_matrix`: the one data door — lists, NumPy, pandas, Polars
  and Arrow in; clean float64 out, or a message naming the column, the count and the fix.

### Notes
- Every metric takes `sample_weight` (sample-weight semantics; see `docs/methods.md`).
- All metrics are golden-tested against scikit-learn or a named reference, and property-tested
  with hypothesis.
