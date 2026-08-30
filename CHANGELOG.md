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

### Notes
- Every metric takes `sample_weight` (sample-weight semantics; see `docs/methods.md`).
- All metrics are golden-tested against scikit-learn or a named reference, and property-tested
  with hypothesis.
