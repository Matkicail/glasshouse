# Changelog

All notable changes, newest first. Pre-1.0: minor versions may break the API; the entry says so.

## Unreleased

### Added
- Three datasets and benchmarks, so the scorecard is proven on five shapes: `fremtpl2_sev`
  (gamma severity, claims summed per policy and joined to the frequency frame, weight =
  claim count; GLM with splines vs LightGBM), `bike_sharing` (Poisson hourly counts on a
  time-ordered split, residuals over time; GLM vs LightGBM) and `telco_churn` (logistic vs
  lasso logistic with the cross-validated path). Each has a documented cleaner tested on a
  raw-shaped sample, a committed `report.md` and `pinned.json`, and the drift test now
  covers every committed benchmark.
- `bench.run(time=...)` and `Benchmark.time`: name a column and the report shows residuals
  over it; set it whenever the split is time-ordered.
- `Dataset.requires`: a loader can join to another dataset's cleaned frame.

- `terms={"age": "piecewise"}` and `encoders.piecewise_linear`: a piecewise linear term,
  the piecewise linear encoding of Gorishniy et al. (2022) as a GLM term. It is a degree-1
  B-spline on quantile knots, one alias and no second implementation; a test checks the
  fitted curve equals the bin-fill encoding's. Takes `monotone=` like any spline.
- `GLM(group_lasso=True)`: the group lasso, where a one-hot factor or a spline term leaves
  the model whole or stays whole. Groups are the model's own terms, so nothing new to
  declare; a plain column is a group of one and gets the ordinary lasso update exactly. One
  majorisation step per group inside the same coordinate descent; the cross-validated path
  and `alpha_max` understand groups. Checked against the grouped KKT conditions.
  `telco_churn` gains a `group_lasso_logistic` row and is re-pinned.
- The Model tab draws the solver's choices: for a model fitted with `alpha="cv"`, the
  cross-validated deviance along the path with its standard error band, the one-standard-
  error line and the chosen alpha, next to every coefficient's path as the penalty relaxes
  (`AlphaPath` now carries the coefficients per alpha and serialises); for a model with
  smooth terms, each smooth's GCV trace against lambda with the chosen point and the
  effective degrees of freedom on hover. The alpha or lambda every fold chose is listed
  under the chart, so the spread is on the record.
- `GLM.term_contributions`: per-row contributions summed per input column, intercept first,
  adding up to the linear predictor. The bench keeps, for every glass-box model, thirty
  held-out rows worth explaining (the ten priced highest across folds, the ten lowest, ten at
  random) and the Model tab draws each as a bar per feature on the link scale, with the
  relativity on hover for a log link: "why this price", for one policy.

### Changed
- The design build is vectorised: the missing-value check on a categorical column, the
  one-hot construction and the out-of-fold target encoding looped over rows in Python. A
  full-design GLM fit on freMTPL2 went from 2.1 s to 1.2 s, of which the solver is 0.4 s.
- The GCV search for a smooth's penalty brackets the optimum on a one-point-a-decade grid
  and narrows it by golden section to a twentieth of a decade, instead of walking a 23-point
  grid and then a 9-point one; a second coordinate sweep starts from a bracket around the
  previous optimum. 22 evaluations a smooth instead of 32, and 10 instead of 32 on the
  second sweep. `fremtpl2_challengers` is re-pinned on the new `glm_smooth` numbers.

### Fixed
- The elastic-net coordinate descent crawled, or hit its sweep cap, on designs with an
  intercept next to frequent 0/1 columns (any one-hot factor): the intercept was swept like
  every other coordinate and is nearly collinear with such a column. It is now kept at its
  closed form after every update, which is the same as descending on centred columns and is
  what glmnet does. Its tolerance is 1e-8 on the coefficient step rather than 1e-10, still
  well inside the glum goldens. And a row pass on fewer than 16 384 rows runs on the calling
  thread, summing the same chunks in the same order, because rayon's scheduling cost more
  than the work on small data. A lasso fit on the 7 043-row Telco churn set went from 6.3 s
  to 0.5 s and its cross-validated path from over ten minutes to 23 s; results on the large
  benchmarks are bit-for-bit unchanged.

## 0.1.0 — 2026-09-06

The first release on PyPI. Everything below the second heading was unreleased until now.

### Added
- The wheel carries the report viewer (`glasshouse/_report/`: the schema, the HTML template
  and the compiled viewer), so `report.to_html` works from an installed package, not only
  from a checkout. Releases build wheels for Linux, macOS and Windows on a `vX.Y.Z` tag and
  publish through PyPI Trusted Publishing; `rc` tags go to TestPyPI. Docs site (mkdocs).
- `residuals.ae_by_two`: actual over expected on the grid of two features, each cut exactly
  as the one-way table cuts it, with a weight floor marking cells too thin to read. The
  report carries one grid per pair of `features` and the Residuals tab draws it as a
  heatmap with thin cells greyed out: the interaction view, and the honest case for an
  interaction term. In Rust the one-way and two-way tables now share one binning rule
  (`bin_index`) and one accumulator (`grid_table`).
- `glasshouse.profile`: `histogram` and `feature_profile`, the data before any model. The
  report gains a `data` block and a Data tab after the Overview: the outcome and the weight
  summarised (weighted mean, spread, quantiles, share of the weight on exact zeros) and
  drawn as even-width histograms, then one profile per feature with the weight in each bin
  or level and the mean outcome there.
- `glasshouse.explain`: `partial_dependence` and `permutation_importance` for any model
  through its predictions, and `coefficients` for glass-box ones. The bench computes them
  on held-out rows per fold and the report gains a Model tab: importance per model with the
  fold spread, a partial-dependence chart per feature with one line per model and a fold
  band, and each GLM's coefficient table with relativities.
- `GLM(alpha=..., l1_ratio=...)`: lasso, ridge and elastic-net GLMs by coordinate descent
  inside the same IRLS, in glmnet's and glum's convention (golden against glum for three
  families). `alpha="cv"` walks the path from `alpha_max` with warm starts and picks by
  k-fold cross-validation (`"1se"` or `"min"`); the path stays on the model as `path_`.
  Exact zeros; edf counts the active set; standard errors refused for an L1 fit.
- `tournament.win_sets` / `tournament.tournament` and the report's `tournament` block: every
  row goes to the cheapest model (ties split), and each model is judged on the business it
  won: share, predicted, actual, profit, A/E. The Compare tab shows the pair's win sets
  next to the double lift; the Overview shows the all-model tournament. Priced tasks only.
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
