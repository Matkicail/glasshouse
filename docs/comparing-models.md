# Comparing models: the report, end to end

You have several models on one dataset and want to know which one to trust, and why. This
page shows the two ways to get the comparison report, one worked example per task type, and
how to read what comes out. Every Python block on this page runs as a test
(`tests/test_docs_examples.py`), so the code is current and the printed numbers are real.

The report is one self-contained HTML file. Double-click it. No server, no notebook, no
install for the reader. Plotly loads from a pinned CDN; offline, every chart shows its
numbers as a table instead.

## Two ways in

**You already have predictions.** From any model, any library. Hand over the actuals, each
model's predictions on the same rows, the weight or exposure if there is one, and say what
kind of task it is. `report.build` scores everything and `to_html` writes the file.

**You want glasshouse to fit and score on folds.** Give `bench.run` the frame, a task, the
models to fit, and the folds. It fits each model on each training fold, scores the held-out
fold, pools the out-of-fold predictions for the curves, and builds the same report with the
per-fold numbers attached. Any object with `fit(X, y, sample_weight=, offset=, fold=)` and
`predict(X, offset=)` can be a model here; `GLM`, `LightGBM` and the glum and scikit-learn
adapters already are.

The task type is declared, never guessed. It picks the family, the metric panel, the naive
baseline, the curves, and the residual definition:

| task | family | scored on | panel | naive baseline |
|---|---|---|---|---|
| `frequency` | poisson | rate per unit exposure, exposure as weight | deviance, D², Gini, normalised Gini, balance, RMSE, MAE, R² | the weighted mean |
| `severity` | gamma | amount per claim, claim count as weight | same | the weighted mean |
| `pure_premium` | tweedie | amount per unit exposure | same | the weighted mean |
| `binary` | binomial | probability | log-loss, Brier, ROC-AUC, average precision, KS, MCC, F1, balance | the class prior |
| `regression` | gaussian | the raw target | same as frequency | the mean |

The naive row is always there. Every comparison reads "vs naive" first.

## Example 1: frequency, GLM vs GCV smooth vs LightGBM

The real version of this is one command and needs the freMTPL2 data (fetched from OpenML
once, then cached):

```bash
uv run glasshouse bench fremtpl2_challengers
# writes benchmarks/fremtpl2_challengers/report.{json,md,html}
```

The committed `benchmarks/fremtpl2_challengers/report.md` is its summary table. Below is
the same recipe on synthetic data so it runs in seconds. The outcome is a Poisson count with
an exposure, one categorical, and one numeric feature with a U-shaped effect that a linear
term cannot follow.

```python
import numpy as np
import pandas as pd

from glasshouse import GLM, bench, splits
from glasshouse.bench import ModelSpec, TaskSpec
from glasshouse.gbdt import LightGBM

rng = np.random.default_rng(0)
n = 6000
df = pd.DataFrame(
    {
        "region": rng.choice(["north", "south", "east"], size=n),
        "age": rng.uniform(18, 80, size=n),
        "Exposure": rng.uniform(0.2, 1.0, size=n),
    }
)
eta = -2.5 + 0.0015 * (df.age - 45) ** 2 + df.region.map({"north": 0.0, "south": 0.3, "east": -0.2})
df["ClaimNb"] = rng.poisson(np.exp(eta) * df.Exposure).astype(float)

task = TaskSpec(family="poisson", target="ClaimNb", exposure="Exposure", rate=True)
cols = ["region", "age"]
models = [
    ModelSpec("glm_linear", lambda: GLM(family="poisson", terms={"region": "onehot"}), cols),
    ModelSpec(
        "glm_smooth",
        lambda: GLM(family="poisson", terms={"region": "onehot", "age": "smooth"}),
        cols,
    ),
    ModelSpec("lightgbm", lambda: LightGBM(family="poisson", categorical=["region"]), cols),
]
folds = splits.kfold(n, k=3, seed=0)
result = bench.run(df, task, models, folds, dataset="synthetic frequency", features=["age"])
print(result.to_markdown())
```

```text
# synthetic frequency — poisson (ClaimNb)

Split: {'kind': 'random', 'method': 'kfold', 'k': 3, 'seed': 0}. Rows: 6,000. Models: glm_linear, glm_smooth, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_linear | glm_smooth | lightgbm | naive |
|---|---|---|---|---|
| deviance | 0.7436 ± 0.036 | **0.71431 ± 0.029** | 0.74902 ± 0.036 | 0.77719 |
| d2 | 0.043227 ± 0.00066 | **0.080579 ± 0.0067** | 0.036209 ± 0.002 | -3.7007e-17 |
| gini | 0.24842 ± 0.01 | **0.34338 ± 0.022** | 0.28301 ± 0.021 | 0 |
| normalized_gini | 0.26453 ± 0.0096 | **0.36581 ± 0.025** | 0.30134 ± 0.021 | 0 |
| balance | 1.0003 ± 0.086 | **0.99974 ± 0.086** | 0.84322 ± 0.063 | 1 |
| rmse | 0.53816 ± 0.024 | **0.53346 ± 0.022** | 0.54067 ± 0.022 | 0.54507 |
| mae | 0.28835 ± 0.0075 | **0.28106 ± 0.0066** | 0.3043 ± 0.011 | 0.29608 |
| r2 | 0.025201 ± 0.00071 | **0.041904 ± 0.0057** | 0.015785 ± 0.0068 | -7.4015e-17 |

Fit time (all folds): glm_linear 0.1s, glm_smooth 0.2s, lightgbm 0.4s.
```

The smooth found the U on its own (its `lambda_` is chosen by GCV on each training fold)
and wins every metric. LightGBM with library defaults on 4,000-row training folds is
under-fit and loses even to the linear GLM. More telling is its balance of 0.84: it predicts
16 % fewer claims than happened. Trees fitted by gradient descent have no balance property;
a GLM with the canonical link has it by construction. A single ranking score would have
hidden that. If a term must not dip by business rule, say a premium that cannot fall as
the bonus-malus rises, give it `Smooth(monotone="increasing")` and the fit honours it at
every step; the one-way chart on the Curves tab shows the constraint doing its work. Now
write the report and open it:

```python
out = result.write("reports/synthetic_frequency")
print(sorted(p.name for p in out.iterdir()))
```

```text
['pinned.json', 'report.html', 'report.json', 'report.md']
```

`report.html` is the file to send around. `report.json` is the same document as data, for
anything that wants to re-read the numbers. `pinned.json` holds the summary numbers, so a
test can check that a rerun reproduces them.

## Example 2: binary classification with scikit-learn models

Nothing here is a glasshouse model. Two scikit-learn classifiers produce probabilities on a
held-out set, and the report compares them. This is the `report.build` route: you bring the
predictions, glasshouse brings the panel, the curves and the naive baseline.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from glasshouse import report

X, y = make_classification(
    n_samples=8000, n_features=12, n_informative=5, weights=[0.95], random_state=0
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
logistic = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
boosted = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)

rep = report.build(
    "binary",
    y_te,
    {
        "logistic": logistic.predict_proba(X_te)[:, 1],
        "boosted": boosted.predict_proba(X_te)[:, 1],
    },
    features={"x0": X_te[:, 0]},
    dataset="synthetic binary",
    describe="make_classification, 5 % positives",
)
for label, card in rep.cards.items():
    print(f"{label:9s}", {k: round(v, 4) for k, v in card.metrics.items() if k != "balance"})
print("naive    ", {k: round(v, 4) for k, v in card.naive.items() if k != "balance"})
rep.to_html("reports/synthetic_binary.html")
```

```text
logistic  {'log_loss': 0.0798, 'brier': 0.0153, 'roc_auc': 0.932, 'average_precision': 0.8499, 'ks': 0.8392, 'mcc': 0.8194, 'f1': 0.821}
boosted   {'log_loss': 0.0692, 'brier': 0.0107, 'roc_auc': 0.9528, 'average_precision': 0.9026, 'ks': 0.8706, 'mcc': 0.8794, 'f1': 0.8793}
naive     {'log_loss': 0.2094, 'brier': 0.0509, 'roc_auc': 0.5, 'average_precision': 0.0537, 'ks': 0.0, 'mcc': 0.0, 'f1': 0.0}
```

For a binary task the report also carries a threshold grid: the viewer's Threshold tab has
a slider, and the table under it shows how many rows you would flag, how many of the
positives you catch, and the alerts per catch at that cut. The `threshold=` argument sets
the cut used for MCC and F1 in the panel (default 0.5).

## Example 3: plain regression

Same route, gaussian family, MAE as the primary metric. Any regressor's predictions work.

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

Xr, yr = make_regression(n_samples=5000, n_features=8, noise=20.0, random_state=1)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=1)
ridge = Ridge().fit(Xr_tr, yr_tr)
trees = HistGradientBoostingRegressor(random_state=1).fit(Xr_tr, yr_tr)

rep = report.build(
    "regression",
    yr_te,
    {"ridge": ridge.predict(Xr_te), "trees": trees.predict(Xr_te)},
    dataset="synthetic regression",
)
for label, card in rep.cards.items():
    print(f"{label:6s}", {k: round(card.metrics[k], 3) for k in ("mae", "rmse", "r2")})
print("naive ", {k: round(card.naive[k], 3) for k in ("mae", "rmse", "r2")})
rep.to_html("reports/synthetic_regression.html")
```

```text
ridge  {'mae': 15.922, 'rmse': 20.009, 'r2': 0.987}
trees  {'mae': 30.11, 'rmse': 39.364, 'r2': 0.95}
naive  {'mae': 140.117, 'rmse': 175.373, 'r2': 0.0}
```

A linear target, so ridge wins. The point is the shape of the workflow, not the result.

## Reading the report

The tabs, in the order a reader meets them:

- **Overview.** Where the data came from, how it was cleaned, and how it was split. Then
  the leaderboard: every model on the panel, held-out, with the naive row. Fit seconds per
  model. For priced tasks, the tournament: every row goes to the cheapest model, and each
  model is judged on the business it won. Nothing on this screen is computed in the
  browser; every number is in the JSON.
- **Compare.** Pick two models. Each metric says which one wins and by how much. The double
  lift chart is here: rows are ranked by the ratio of the two models' predictions and
  binned. In the bins where they disagree most (the ends), the line that sits on the actual
  line is the model to trust there. Between two models with similar deviance, this chart
  settles it. Above it, the pair's win sets: the share each model wins by pricing lower,
  what it charges there and what that business costs. An A/E above 1 on a win set is
  business won by under-pricing it.
- **Curves.** Lorenz (with the Gini), lift (predicted vs actual by predicted decile), and
  calibration (actual over expected by decile, with the balance). For a binary task, ROC and
  precision-recall as well. Every curve has one line per model, same colour on every chart.
- **Model.** Glass-box where possible, explained where not. Permutation importance per
  model: how much worse the held-out deviance gets when a feature is shuffled. Partial
  dependence per feature: what each model says the feature does, one line per model, with
  the spread across folds as a band. For a GLM, its coefficients averaged over folds with
  their spread and, on a log link, the relativities. This tab needs the fitted models, so
  it comes from `bench.run`; a report built from predictions alone does not have it.
- **Residuals.** Deviance and Pearson residuals, a one-way view of actual vs predicted by
  each feature you passed in `features=` (with exposure bars), and residuals over time when
  a `time=` column was given. This is where a model that scores well but misprices one
  segment gets caught.
- **Threshold** (binary only). The slider described above.

What to look at first depends on who you are:

| you are | look first at |
|---|---|
| pricing a frequency or severity model | double lift against the incumbent; calibration by decile; balance |
| screening fraud or another rare event | precision-recall and average precision; MCC at your threshold; alerts per catch |
| building a credit or risk score | ROC and KS; calibration of the probability |
| forecasting or general regression | MAE and RMSE vs naive; residuals over time |

The metric definitions, their references, and when each one lies are in
[methods.md](methods.md). The short version: deviance says whether the model fits the
distribution you declared; Gini says whether it ranks; calibration says whether you can
trust the number; the naive row says whether any of it was worth doing.

## Pieces on their own

Every part of the report is a plain function, if you only want one of them.

```python
from glasshouse import curves, plots
from glasshouse.classification import roc_auc, average_precision
from glasshouse.scorecard import scorecard, compare

p_log = logistic.predict_proba(X_te)[:, 1]
p_gbm = boosted.predict_proba(X_te)[:, 1]
print(round(roc_auc(y_te, p_gbm), 4), round(average_precision(y_te, p_gbm), 4))

card_a = scorecard(y_te, p_log, family="binomial", label="logistic")
card_b = scorecard(y_te, p_gbm, family="binomial", label="boosted")
print(compare(card_a, card_b))

lift_a = curves.lift(y_te, p_log, label="logistic")
lift_b = curves.lift(y_te, p_gbm, label="boosted")
figure = plots.lift(lift_a, lift_b)  # a Plotly figure; figure.show() in a notebook
dl = curves.double_lift(y_te, p_log, p_gbm, label_a="logistic", label_b="boosted")
print(len(dl.actual), "bins")
```

```text
0.9528 0.9026
metric                    logistic       boosted  better
log_loss                   0.07981      0.069152  boosted
brier                      0.01531      0.010679  boosted
roc_auc                    0.93198        0.9528  boosted
average_precision          0.84993       0.90258  boosted
ks                         0.83925        0.8706  boosted
mcc                        0.81941       0.87937  boosted
f1                         0.82096       0.87931  boosted
balance                     1.0306        1.2585  logistic
10 bins
```

Note the last row. The boosted classifier ranks better on every score and is the worse
calibrated: actual positives are 26 % more than its probabilities add up to. If the
probability is going to be used as a number (a price, a reserve, an expected count), that
row matters more than the AUC.

`curves.*` return the data (the JSON contract); `plots.*` draw it with Plotly
(`pip install glasshouse[plots]`). Weighted everywhere: pass `sample_weight=` and the
same rules apply as in the report.

## When it refuses

The report and the bench fail early, by design, with the row count and the fix in the
message: predictions of different lengths, a label outside 0/1 for a binary task, NaNs in
`y`, zero exposure on a rate task, a frame missing a column a model asked for. Read the
message; it names the column.
