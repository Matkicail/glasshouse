# The research track

A fenced set of neural models that have to earn their place on the same report as
everything else. Nothing here is imported by the core: `pip install "glasshouse[research]"`
brings torch (the CPU build is enough), and `from glasshouse.research import CANN` is the
door.

## The fence

Every model in this track is judged by one rule, fixed before any of it was written: it
enters the core only after it beats the spline plus monotone GLM on **held-out deviance and
on calibration** on freMTPL2, on the stored splits. Until then it is a challenger row on the
leaderboard, scored by the same Rust deviance, drawn on the same curves, explained by the
same importances and partial dependence, and nothing more. The order of the track is
smallest step first:

1. **CANN**, the GLM frozen as a skip connection and a small net on its residual. Built.
2. An additive net (one small net per feature), whose shapes are curves you can draw.
3. LocalGLMnet, where the coefficients themselves become functions of the row.
4. A KAN-style additive model, run with three numeric encodings side by side (raw,
   piecewise linear, periodic), so the feature-encoding trade-off is a picture, not an
   argument.

The go/no-go sits at each step. A model that fails it stays a notebook, and the report
shows why.

## CANN: the combined actuarial neural network

Wüthrich and Merz's idea (Schelldorfer & Wüthrich, "Nesting classical actuarial models into
neural networks", 2019): keep the GLM exactly as it is and let a small network learn only
what the GLM got wrong.

```
link(mu) = [GLM linear predictor, frozen] + [network output] + offset
```

The GLM is fitted first, on the training rows, then frozen. The network's last layer starts
at zero, so before training the CANN *is* the GLM to the last decimal. Training minimises the
family deviance, the same one every score in this library uses (`deviance_torch` is checked
against the Rust deviance to rounding), and the network moves away from zero only where that
lowers it.

What you get that a tree does not give: the network's output is one number per row on the
link scale, and for a log link its exponential is the multiplicative correction to the
GLM's price. `CANN.correction` returns it; `term_contributions` returns the GLM's terms and
that one extra column, so the report's "explain a row" shows a policy as the GLM's
relativities plus the network's factor.

What is given up, stated plainly:

- **The balance property.** A GLM adds up to the total by construction; a net does not. The
  CANN restores it after training by one shift of the correction on the training rows
  (a factor for a log link, a constant for identity; nothing for logit, where one constant
  cannot do it exactly). The balance row on the leaderboard shows what is left.
- **Standard errors.** None for the net part; the coefficient table on the Model tab is the
  GLM's, and the network is a column, not a coefficient.
- **A fit that can overfit.** Training stops early on a seeded slice of the training rows
  only, the way the LightGBM adapter does; `history_` and `best_epoch_` record the run.

```python
from glasshouse import GLM
from glasshouse.research import CANN

cann = CANN(
    family="poisson",
    glm=lambda: GLM(family="poisson", terms={"Region": "onehot", "DrivAge": "smooth"}),
    hidden=(20, 15, 10),
)
```

It takes a factory for the GLM, so the bench fits a fresh one per fold. The network's inputs
are the GLM's own encoded columns, standardised, so it sees what the GLM sees.

## The go/no-go run

`uv run glasshouse bench fremtpl2_cann` fits the smooth GLM (the bar), the CANN built on
that same GLM, and LightGBM on the challengers' stratified five-fold split of freMTPL2.
The committed `benchmarks/fremtpl2_cann/report.md` is its summary and `pinned.json` its
drift test.

Run on 2026-09-06 (held-out, mean ± std over five folds; best per metric in bold):

| metric | glm_smooth | cann | lightgbm | naive |
|---|---|---|---|---|
| deviance | 0.59198 ± 0.0021 | 0.58414 ± 0.0027 | **0.5724 ± 0.0026** | 0.62488 |
| d2 | 0.0527 ± 0.0011 | 0.0652 ± 0.0016 | **0.0840 ± 0.0023** | 0 |
| gini | 0.4894 ± 0.016 | 0.4799 ± 0.026 | **0.5351 ± 0.021** | 0 |
| balance | **1.0000 ± 0.0033** | 0.9997 ± 0.0039 | 0.9991 ± 0.0037 | 1 |

Fit time over all folds: glm_smooth 188 s, cann 286 s (it contains the GLM), lightgbm 47 s.

**Verdict: no-go, and the fence is doing its job.** The CANN beats the GLM on held-out
deviance, by 1.3 % and by more than three fold standard deviations, and closes about forty
percent of the gap to LightGBM. It does not beat the GLM on calibration: the balance is
within a tenth of a percent either way, which is a tie, not a win, and the Gini is lower
(within the fold spread, but lower). The rule asks for both, so the CANN stays in the
research fence as a challenger row. What it earned is a place on the report: its correction
column on "explain a row" and the two-feature A/E grids say where the 1.3 % lives, which is
the input to the next model in the track, not a reason to ship this one.


## Save and load

`to_dict` writes the GLM, the network's weights as plain lists, the scaling and the shift;
`from_dict` rebuilds it. No pickle, ever.
