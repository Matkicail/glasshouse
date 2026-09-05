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

Because tie groups are kept whole, the score must carry its ties exactly. For a rate model
that means scoring the model's own rate (`predict` with no offset), not the offset
prediction divided by exposure: the two agree in exact arithmetic, but the division splits
rows with identical features into distinct floats, and on freMTPL2 that rounding noise moves
the Gini at the fourth decimal and makes it depend on the solver's last bits. The bench
scores the rate directly for this reason.

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
if the total deviance did not increase by more than `tol`'s worth of noise — otherwise the
step toward the proposal is halved (up to 20 times). Convergence is a relative deviance
change below `tol` (default `1e-10`; R uses `1e-8`); a step that moves the deviance by less
than that, either way, is accepted and counts as converged, so a solver started at its
optimum stops after one iteration instead of halving twenty times. Starting means follow R's `mustart`, or the fit starts from supplied
coefficients (a *warm start*: judged from its first step like any other, and allowed to
converge at once). Every iteration is kept in the trace.

Every pass over the rows (`XᵀWX`, `XᵀWz`, the linear predictor, the deviance, the sandwich)
runs in parallel over fixed chunks of 4,096 rows. Each chunk forms its own partial sum and
the partials are added in chunk order, so a fit is bit-for-bit identical whatever the thread
count (`RAYON_NUM_THREADS` caps it). Reproducible first, fast second.

Covariance `= φ · (XᵀWX)⁻¹` at the final mean, with dispersion `φ` fixed at 1 for Poisson and
binomial and otherwise the Pearson estimate `Σ w (y − mu)² / V(mu) / (n − p)`. The null
deviance is a fresh intercept-only IRLS fit with the same offset and weights. A rank-deficient
`XᵀWX` is reported with the column index (a relative pivot tolerance of `1e-12`).

**Robust (HC1) covariance** `= (XᵀWX)⁻¹ (Σᵢ sᵢsᵢᵀ) (XᵀWX)⁻¹ · n/(n−p)`, with row score
`sᵢ = xᵢ · wᵢ (yᵢ − muᵢ) (dmu/deta) / V(muᵢ)`. The dispersion cancels, so these stand even when
the variance function is wrong (over-dispersed counts fitted as Poisson). The bread is the
*expected* information `XᵀWX` — the same matrix behind `se_`, and what R's `sandwich` uses for
`glm` objects; statsmodels uses the *observed* Hessian instead, which coincides only under a
canonical link. Two further statsmodels notes: its GLM reports identical numbers for `HC0` and
`HC1` (it omits the `n/(n−p)` factor). The golden tests therefore compare against statsmodels'
`HC0 · √(n/(n−p))` for the canonical-link families, and against the formula written out in
NumPy for all five. Reference: MacKinnon & White, "Some heteroskedasticity-consistent covariance matrix
estimators with improved finite sample properties", *J. Econometrics* 29 (1985).

AIC/BIC are deliberately not provided yet: they need the full per-family log-likelihood
(log-gamma terms; none in closed form for Tweedie), and the scorecard compares on deviance.

References: McCullagh & Nelder (1989) §2.5; Nelder & Wedderburn, "Generalized Linear Models",
*JRSS A* 135 (1972). Golden reference: `statsmodels.GLM` with `var_weights` and `offset`.

## Residuals

- **Deviance residual** `r_i = sign(y_i − mu_i) · sqrt(w_i · d(y_i, mu_i))`; `Σ r_i² = D`.
- **Pearson residual** `r_i = (y_i − mu_i) · sqrt(w_i / V(mu_i))`; `Σ r_i² = X²`, the Pearson
  chi-square behind the dispersion estimate.
- **A/E by feature**: the calibration table with the rows binned by a feature instead of the
  prediction — equal-weight bins (ties whole) for a numeric column, one row per level for a
  categorical — reporting weight, mean prediction, mean outcome and their ratio per bin.

Golden references: statsmodels `resid_deviance`, `resid_pearson` (with `var_weights`).

## B-spline terms

A spline term expands one numeric column into ``df`` cubic B-spline basis columns. Interior
knots sit at quantiles of the training rows (``df − degree`` of them, evenly spaced in
probability); boundary knots are the training min and max, repeated ``degree + 1`` times; the
first basis function is dropped so the expansion does not duplicate the intercept (R's
``bs()`` convention). At prediction time values outside the training range are evaluated at
the clamped boundary — a cubic tail extrapolated silently is how spline models go wrong
quietly. The basis is evaluated in Rust by the Cox–de Boor recursion.

Reference: de Boor, *A Practical Guide to Splines* (2001). Golden reference:
``scipy.interpolate.BSpline.design_matrix`` on identical knot vectors.

## Penalised smooths (P-splines) and GCV

A `"smooth"` term is a P-spline: a cubic B-spline basis on *evenly spaced* interior knots
over the training range, with a penalty on the squared second differences of its
coefficients, `β' D₂'D₂ β` — Eilers & Marx's construction, where even spacing is what makes
coefficient differences a fair measure of wiggliness. The first basis column is dropped so
the term does not fight the intercept; the penalty matrix is built on the full basis and
reduced the same way. IRLS then solves `(X'WX + S) β = X'Wz` with `S` the block-embedded
penalty times `λ`; the fixed point minimises the penalised deviance `D + β'Sβ`, and
step-halving and convergence run on that same objective.

`λ` is chosen by minimising GCV, `n·D / (n − edf)²`, with effective degrees of freedom
`edf = tr((X'WX + S)⁻¹ X'WX)` — mgcv's criterion with `γ = 1` — over a coarse log-spaced
grid plus one finer pass around the winner (two coordinate sweeps when several smooths are
free). Every `(λ, GCV, edf)` evaluated stays on the model in `gcv_`, so the choice can be
read, not re-run. Because only the smooth's block is penalised, the intercept's score
equation is untouched and the balance property survives penalisation exactly.

The search is exact but not expensive: every evaluation is a converged penalised IRLS fit at
that `λ`, warm-started from the previous evaluation's coefficients (the fine pass from the
coarse winner) and skipping the null model and the covariances, which only the final fit
needs. Neighbouring `λ` have neighbouring optima, so an evaluation typically takes one to
three iterations instead of seven.

The reported covariance is `φ (X'WX + S)⁻¹` (the Bayesian posterior covariance, mgcv's
convention) and dispersion divides by `n − edf`.

References: Eilers & Marx, "Flexible smoothing with B-splines and penalties", *Statistical
Science* 11 (1996); Wood, *Generalized Additive Models*, 2nd ed. (2017). Golden reference:
statsmodels `GLMGam` — it penalises the log-likelihood where we penalise the deviance
(−2·loglik), so `S = 2·α·cov_der2` must and does reproduce its coefficients and edf to
machine precision. mgcv itself needs R, which the test machines do not have.

## Elastic-net (lasso, ridge) GLM

`GLM(alpha=a, l1_ratio=r)` minimises, in glmnet's and glum's convention,

`D(β) / (2 Σw) + a · ( r · Σ_j |β_j| + (1 − r)/2 · Σ_j β_j² )`

over every column but the intercept, `D` being the total weighted deviance. `r = 1` is the
lasso, `r = 0` ridge. The same `alpha` therefore means the same model here, in glum and in
glmnet (with `standardize = FALSE`); the golden test checks that directly. Columns are
penalised on their own scale, so a column measured in thousands is penalised less than one
measured in units: standardise first (`terms={"x": "standardize"}`) when the features are
not comparable.

Each IRLS step solves its weighted least squares as the penalised quadratic problem by
cyclic coordinate descent with soft-thresholding (Friedman, Hastie & Tibshirani, *JSS* 33,
2010): coefficient `j` moves to `S(ρ_j, Σw·a·r) / (Σ_i W_i x_ij² + Σw·a·(1 − r))`, with
`ρ_j` the weighted partial-residual correlation and `S` the soft-threshold; full sweeps,
then sweeps over the active set until it settles, then a confirming full sweep. The outer
loop, its step-halving on the penalised deviance and the warm starts are the ones every
GLM here uses. Exact zeros are exact: a coefficient the threshold switches off is `0.0`,
not small.

The effective degrees of freedom of a lasso fit count the non-zero coefficients, with the
ridge part of an elastic-net shrinking that count as for any quadratic penalty (Zou,
Hastie & Tibshirani, "On the degrees of freedom of the lasso", *Ann. Statist.* 35, 2007).
Standard errors are not offered for an L1 fit: the selection is part of the estimator, and
a covariance that ignores it would be wrong (refit the selected columns unpenalised if you
need one, with the usual post-selection caveats). Ridge fits report the posterior
covariance as the smooths do.

`alpha="cv"` walks a path of `n_alphas` values, log-spaced from `alpha_max` (the largest
gradient of half the mean deviance at the fit of the unpenalised columns, divided by
`l1_ratio`; everything is zero from there up) down to `alpha_ratio · alpha_max`. On each
inner fold (random k-fold on the training rows; a time-ordered outer fold is refused, as
the inner folds would let the future score the past) the path is fitted with warm starts
and scored by held-out mean deviance. `"min"` takes the lowest mean; `"1se"` takes the most
penalised alpha within one standard error of it, the usual choice when the simpler model is
worth a decimal. The path, the fold means, the standard errors and the number of non-zero
coefficients are kept on the model (`path_`), so the choice can be read, not re-run.

Golden references: glum `GeneralizedLinearRegressor` at the same `alpha` / `l1_ratio` for
gaussian, poisson (with offset) and binomial, and the ridge case against the quadratic
penalty solver with `S = Σw·a·I`; the coordinate-descent step against its own KKT
conditions and the closed-form ridge.

## Partial dependence and permutation importance

Every model on a bench report gets the same two explanations, computed on a sample of each
fold's held-out rows and averaged over folds with the fold spread shown.

- **Partial dependence** (Friedman, "Greedy function approximation", *Ann. Statist.* 29,
  2001): for each grid value `v` of a feature, set the feature to `v` on every row, predict,
  and average: `PD(v) = mean_i f(x_i with feature = v)`. The grid is the feature's
  quantiles (evenly spaced in probability, so the curve is drawn where the data is) or its
  levels. For a GLM the curve is exactly the term's effect: on the log link the ratio of
  two grid points is `exp(β · Δv)`, which a test checks.
- **Permutation importance** (Breiman, "Random forests", *Machine Learning* 45, 2001;
  Fisher, Rudin & Dominici, "All models are wrong, but many are useful", *JMLR* 20, 2019):
  shuffle one feature across the held-out rows, re-score the mean deviance, and report the
  increase. One shuffle per feature per fold, seeded by the fold; the spread across folds
  says how stable the ranking is.

Both vary one feature with the others held as they are, so for tightly bound features the
model is being asked about rows that do not exist; the one-way and A/E views on the Curves
tab, which slice the actual data, are the complement. For a GLM the report also shows the
coefficients averaged over folds, with their spread and, on a log link, the relativities
`exp(β)`. That table is the glass box itself; the two pictures above are how the same
questions get answered for a model that has no such table.

## Win sets and the tournament

Several models price the same rows; every row goes to the model whose prediction is lowest,
an exact tie split equally among the tied models. For model `m` the win set `W_m` is then
summed with the row weights: `share = Σ_{W_m} w / Σ w`, `predicted = Σ_{W_m} w · p_m`,
`actual = Σ_{W_m} w · y`, `profit = predicted − actual`, and `A/E = actual / predicted`. The
two-model case is the pairwise win-set table on the report's Compare tab; all models at
once is the tournament on the Overview.

What it tells you: a model that wins business with `A/E > 1` is winning it by under-pricing
it, which is adverse selection; a model with a small share and `A/E < 1` is pricing
conservatively and losing the business it prices well. Because the prediction is what would
be charged, the table exists for frequency, severity, pure premium and regression tasks,
not for binary ones (a probability is not a price). It is a description of the models on
this data, not a forecast of a market: real conversion depends on much more than the
lowest price.

Reference: this is the "win set" and "profit matrix" view of pricing reviews (see Goldburd,
Khare & Tevet, *Generalized Linear Models for Insurance Rating*, CAS Monograph 5, 2nd ed.
2020, on model comparison); golden reference: the definition written out in NumPy
(`tests/test_tournament.py`).

## Monotone constraints on spline terms

`Smooth(monotone="increasing")` or `BSpline(monotone="decreasing")` fits the term under a
shape constraint. A cubic B-spline is non-decreasing wherever its coefficients are
non-decreasing along the knots (de Boor, *A Practical Guide to Splines*, ch. XI: the curve
follows its control polygon), so the constraint is the chain `0 ≤ β₂ ≤ β₃ ≤ …` on the
term's kept columns, the leading zero being the dropped first basis coefficient that the
intercept absorbs. The condition is sufficient, not necessary: a monotone curve with a
dipping coefficient near a boundary will have that dip ironed out.

Each IRLS step then solves its weighted least squares subject to the chain, as a quadratic
programme `min ½ β'Hβ − b'β  s.t. Aβ ≥ 0` with `H = X'WX + S` and `b = X'Wz`, by a primal
active-set method. An active constraint is a *tie*: two adjacent coefficients equal, or a
leading run held at zero. A tied run is one merged column of the design, so every
subproblem is an ordinary Cholesky solve on a reduced matrix; ties are added when a step
would cross a constraint and released when the KKT multiplier says the objective would
rather move apart. Step-halving stays inside the constraints because a convex combination
of two feasible points is feasible, and the fixed point satisfies the KKT conditions of the
constrained penalised deviance. The intercept is never constrained, so the balance property
survives.

Coefficients an active tie joins count as one in the effective degrees of freedom and share
a covariance entry; a run held at zero counts for nothing. GCV runs unchanged over the
constrained fits.

References: de Boor (2001); Pya & Wood, "Shape constrained additive models", *Statistics
and Computing* 25 (2015), which reaches the same fits by reparameterisation. Golden
reference: the constrained problem itself, solved exactly by enumerating every active set
of the KKT system on a gaussian problem (`tests/test_monotone.py`); isotonic regression
(pool-adjacent-violators) for the QP solver on its own. mgcv's `scam` needs R, which the
test machines do not have.

## Encoders and leakage

Leakage is a property of the split, not the transform. Every encoder is fitted on the
training rows only (the GLM's `fold=` guarantees it), and:

- **One-hot**: levels learned on train, sorted, first level dropped as the reference; an
  unseen level at prediction is refused by name (or encoded as the reference on request).
- **Target encoding**: `(Σ w y + m · prior) / (Σ w + m)` per level, with `m` the smoothing
  weight (in units of exposure) and `prior` the weighted mean of `y`. The training rows get
  *out-of-fold* values (inner k-fold), or, when the fold is time-ordered, *past-only* values
  with the running mean of the past as prior — a row with no past is encoded as 0, never the
  global mean (which is the future). New rows get the full-training-data table; unseen levels
  get the prior. Reference: Micci-Barreca, "A preprocessing scheme for high-cardinality
  categorical attributes", *SIGKDD Explorations* 3 (2001).
- **Standardize**: weighted mean and population standard deviation from the training rows.

## Numerical notes

- Metric sums are plain `f64` accumulations in row order. Against scikit-learn on ~5k rows
  they agree to `1e-12`; on millions of rows the accumulated error grows and pairwise or
  compensated summation may be warranted — to be measured, not assumed, before it is changed.
  The GLM solver's sums are chunked (see above), which is a fixed pairwise-by-chunk order.
- Ties are exact float equality, everywhere, by one shared helper. Property tests that
  transform scores must use exact transforms (`s * 0.5`, `−s`), never `s*a + b` or `1 − s`.
