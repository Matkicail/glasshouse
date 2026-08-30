# Changelog

All notable changes, newest first. Pre-1.0: minor versions may break the API; the entry says so.

## Unreleased

### Added
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
