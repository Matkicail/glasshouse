# Methods — the formulas, the references, and how weights work

This is the one place the maths lives. Docstrings say what a metric is *for* and when it
lies; this page says what it *is*, and where it comes from. Notation throughout: observed
`y_i`, predicted mean `mu_i`, weight `w_i`, `n` rows, `W = sum(w_i)`.

## Weights: one paragraph, referenced everywhere

Every metric takes `sample_weight`, and it always means the same thing: **row `i` counts
`w_i` times as much as a unit row** in every average. So a mean is `sum(w_i * x_i) / W`, and
scaling every weight by a constant changes nothing. That is the *sample-weight* convention
(the one scikit-learn uses), and it is the right one for **exposure**: a policy observed for
half a year is half a row.

Two consequences worth knowing:

- When `sample_weight` is exposure, `y` and `mu` must both be **rates** (per unit of
  exposure). If you have totals (claim counts, amounts) and a prediction of totals, pass no
  weight — the exposure is already inside both numbers.
- glasshouse does not implement *frequency weights* ("this row appears `k` times"). For every
  metric here the two conventions give the same mean, so nothing is lost for scoring; the
  distinction only matters for standard errors and degrees of freedom, and will be made
  explicit when the GLM's inference lands. Until then, `w` is a sample weight.

Weights must be finite and non-negative, and must sum to more than zero. Metrics that rank
or bin (Gini, calibration) additionally refuse zero weights, because a row with no exposure
has no rate.

## Deviance and D²

The unit deviance is twice the log-likelihood gap between the saturated model (which
predicts `y_i` exactly) and the fitted mean, under the exponential-dispersion family:

| family | unit deviance `d(y, mu)` | support of `y` | `mu` |
|---|---|---|---|
| gaussian | `(y − mu)²` | any | any |
| poisson | `2 (y ln(y/mu) − (y − mu))`, with `y ln y → 0` | `y ≥ 0` | `> 0` |
| gamma | `2 (ln(mu/y) + (y − mu)/mu)` | `y > 0` | `> 0` |
| tweedie, power `p` | `2 ( y⁺^(2−p) / ((1−p)(2−p)) − y mu^(1−p)/(1−p) + mu^(2−p)/(2−p) )` | `p<0`: any; `1≤p<2`: `≥0`; `p≥2`: `>0` | `> 0` (any if `p = 0`) |
| binomial | `2 ( y ln(y/mu) + (1−y) ln((1−y)/(1−mu)) )` | `0 ≤ y ≤ 1` | `(0, 1)` |

`y⁺ = max(y, 0)` for `p < 0` (the "extreme stable" convention, matching scikit-learn).
Tweedie at `p = 0, 1, 2` delegates to gaussian, poisson, gamma bit-for-bit; `0 < p < 1` is
not a Tweedie distribution and is refused. Logs are taken as `ln y − ln mu`, never `ln(y/mu)`,
so subnormal values cannot underflow to `ln 0`.

- **Mean deviance** `= sum(w_i d(y_i, mu_i)) / W` (`metrics.deviance`).
- **D²** `= 1 − D(y, mu) / D(y, ȳ)` where `ȳ` is the weighted mean of `y` — the intercept-only
  (null) model (`metrics.d2`). Undefined, and refused, when `y` is constant.

The Tweedie deviance is a strictly consistent scoring function for the mean, which is why the
same function is both the GLM's loss and its honest evaluation metric.

References: McCullagh & Nelder, *Generalized Linear Models* (2nd ed., 1989), §2.3; Jørgensen,
*The Theory of Dispersion Models* (1997); scikit-learn `mean_tweedie_deviance`,
`d2_tweedie_score` (the golden references in `tests/test_metrics.py`).

## Gini and normalised Gini

Sort rows by `score` ascending, tie groups kept whole. The Lorenz curve accumulates weight on
the x-axis and `y` on the y-axis: `x_k = cum w / W`, `L_k = cum y / sum(y)`. With trapezoid
area `A = sum (x_k − x_{k−1})(L_k + L_{k−1}) / 2`:

- **Gini** `= 1 − 2A` (`metrics.gini`), in `[−1, 1]`; 0 for a constant score.
- **Normalised Gini** `= Gini(score) / Gini(y / w)` (`metrics.normalized_gini`) — the ratio to
  the perfect ranking. For 0/1 labels with unit weights it equals `2·AUC − 1` (Somers' D, the
  accuracy ratio); the raw Gini is that times `1 − prevalence`.

The ordered-Lorenz Gini of Frees, Meyers & Cummings (rank by `score / base premium`,
accumulate base premium) is the same computation with a different key and weight; it is
deferred until there is a base model to compare against.

References: Frees, Meyers & Cummings, "Summarizing Insurance Scores Using a Gini Index",
*JASA* 106 (2011); "Insurance Ratemaking and a Gini Index", *J. Risk & Insurance* 81 (2014);
Wüthrich, "Model selection with Gini indices under auto-calibration", *Eur. Actuarial J.* 13
(2023) — Gini is not strictly consistent unless predictors are auto-calibrated, which is why
the scorecard never shows it alone. Golden reference: the Kaggle "normalized Gini" function
(Allstate / Porto Seguro competitions) and scikit-learn `roc_auc_score`.

## Calibration and the balance property

`metrics.calibration_table`: sort by `mu`, cut into `n_bins` bins of equal total weight (tie
groups never split; a group goes to the bin its weight-midpoint falls in), and per bin report
`predicted = sum(w mu)/sum(w)`, `actual = sum(w y)/sum(w)`, and `actual / predicted`.

`metrics.balance` `= sum(w y) / sum(w mu)`; 1 means the model reproduces the total. A GLM with
the canonical link satisfies this on its training data by construction (the score equations
sum to zero); models trained by gradient descent generally do not.

References: Denuit, Charpentier & Trufin, "Autocalibration and Tweedie dominance for insurance
pricing with machine learning", *IME* 101 (2021), arXiv:2103.03635; Wüthrich & Ziegel,
"Isotonic recalibration under a low signal-to-noise ratio", *Scand. Actuarial J.* (2024),
arXiv:2301.02692. Golden reference: scikit-learn `calibration_curve(strategy="quantile")`.

## Binary classification

Labels must be exactly 0/1. With weighted confusion counts `TP, FP, FN, TN` at
`score ≥ threshold` (`classification.threshold_metrics`):

- accuracy `(TP+TN)/(TP+TN+FP+FN)`; balanced accuracy `(TPR + TNR)/2`;
  precision `TP/(TP+FP)`; recall `TP/(TP+FN)`; F1 `2TP/(2TP+FP+FN)`;
- **MCC** `(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`, in `[−1, 1]`.
  Undefined ratios (an empty marginal) return 0, scikit-learn's convention.

Ranking metrics walk the tie groups from the highest score down:

- **ROC-AUC**: trapezoid area under (FPR, TPR); ties are one step, so tied pairs count half.
- **Average precision**: `sum_k (R_k − R_{k−1}) P_k` — the step-wise area under the
  precision–recall curve, scikit-learn's definition (not the trapezoid, which is optimistic).
- **KS**: `max_k | F_pos(k) − F_neg(k) |`, the largest gap between the two classes' weighted
  score distributions.

Probability metrics reuse the deviances: **log-loss** `= binomial deviance / 2`;
**Brier** `= gaussian deviance` on labels vs probabilities. Both are proper scoring rules.

Bounded results are clamped to their range (`[0,1]`, `[−1,1]`) because platform summation
order can leave a rounding error of one ulp outside it (observed on macOS CI).

References: Chicco & Jurman, "The advantages of the Matthews correlation coefficient (MCC)
over F1 score and accuracy in binary classification evaluation", *BMC Genomics* 21 (2020);
Saito & Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When
Evaluating Binary Classifiers on Imbalanced Datasets", *PLoS ONE* 10 (2015); Gneiting &
Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", *JASA* 102 (2007).
Golden references: the scikit-learn functions of the same names, all with `sample_weight`.

## Regression errors

All weighted means over rows (`regression.*`):

| metric | per-row term | domain |
|---|---|---|
| RMSE | `sqrt(mean (y − mu)²)` | — |
| MAE | `|y − mu|` | — |
| MAPE | `|y − mu| / |y|` | `y ≠ 0` |
| sMAPE | `2 |y − mu| / (|y| + |mu|)`, in `[0, 2]` | not both zero |
| MSLE | `(ln(1+y) − ln(1+mu))²` | `y, mu > −1` |
| R² | `1 − mean (y − mu)² / mean (y − ȳ)²` | `y` not constant |

Golden references: scikit-learn `root_mean_squared_error`, `mean_absolute_error`,
`mean_absolute_percentage_error`, `mean_squared_log_error`, `r2_score`; sMAPE against its own
formula (scikit-learn has none).

## The scorecard

`scorecard.scorecard` runs the panel for a family and, always, the same panel for the naive
model — the weighted mean of `y` (the class prior for binomial). What the naive row must
score is pinned by tests: `D² = R² = Gini = 0`, `balance = 1`; for binomial `ROC-AUC = 0.5`,
`AP = prevalence`, `Brier = p(1−p)`. `scorecard.compare` judges each metric by its direction
(`HIGHER_IS_BETTER`); `balance` by distance from 1.

## The GLM (IRLS)

With design `X` (intercept column included), link `g`, offset `o` and prior weights `w`, each
iteration forms the working response `z = (eta − o) + (y − mu) / g'(mu)⁻¹` and working weights
`W = w · (dmu/deta)² / V(mu)`, solves `(XᵀWX) β = XᵀWz` by Cholesky, and accepts the step only
if the total deviance did not increase — otherwise the step toward the proposal is halved
(up to 20 times). Convergence is a relative deviance change below `tol` (default `1e-10`; R
uses `1e-8`). Starting means follow R's `mustart`. Every iteration is kept in the trace.

Covariance `= φ · (XᵀWX)⁻¹` at the final mean, with dispersion `φ` fixed at 1 for Poisson and
binomial and otherwise the Pearson estimate `Σ w (y − mu)² / V(mu) / (n − p)`. The null
deviance is a fresh intercept-only IRLS fit with the same offset and weights. A rank-deficient
`XᵀWX` is reported with the column index (a relative pivot tolerance of `1e-12`).

References: McCullagh & Nelder (1989) §2.5; Nelder & Wedderburn, "Generalized Linear Models",
*JRSS A* 135 (1972). Golden reference: `statsmodels.GLM` with `var_weights` and `offset`.

## Numerical notes

- Sums are plain `f64` accumulations in row order. Against scikit-learn on ~5k rows they agree
  to `1e-12`; on millions of rows the accumulated error grows and pairwise or compensated
  summation may be warranted — to be measured, not assumed, before it is changed.
- Ties are exact float equality, everywhere, by one shared helper. Property tests that
  transform scores must use exact transforms (`s * 0.5`, `−s`), never `s*a + b` or `1 − s`.
