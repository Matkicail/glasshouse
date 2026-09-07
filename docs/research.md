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
2. **An additive net** (one small net per feature), whose corrections are curves you can
   draw and which cannot learn an interaction. Built, as `AdditiveNet`.
3. **LocalGLMnet**, where the GLM's coefficients become functions of the row. Built, as
   `LocalGLMnet`.
4. **A two-layer KAN**, run with three numeric encodings side by side (raw, piecewise
   linear, periodic), so the feature-encoding trade-off is a picture, not an argument.
   Built, as `CANN(network="kan", encoding=...)`.

All four are one class and one training loop: `CANN(network=...)`, with `AdditiveNet` and
`LocalGLMnet` as named versions. Only the shape of the correction differs, so a difference
between their rows on the leaderboard is a difference in what the correction is allowed to
be, nothing else.

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

## The additive net and LocalGLMnet

Both keep the frozen GLM and change what the network is allowed to add.

**AdditiveNet** gives each input column its own small net (a one-hot or spline block gets a
linear map per column) and sums them: `link(mu) = GLM + Σ_j g_j(x_j)`. Every correction is a
function of one feature, so `term_contributions` returns one `<term> (net)` column per
feature and "explain a row" reads as the GLM's terms plus a bend per feature. What it
cannot do, by construction, is an interaction; on data where the GLM already has smooth
terms it should add close to nothing, and that is the honest test of whether the CANN's
gain came from shapes or from interactions.

**LocalGLMnet** (Richman & Wüthrich, "LocalGLMnet: interpretable deep learning for tabular
data", 2023) lets a net output one coefficient per design column *for each row*:
`link(mu) = GLM + Σ_j β_j(x) x_j`. A coefficient that stays near zero on every row is a
column the net does not use; one that moves with another feature is an interaction, and
`attention(X)` returns the β's so you can chart them. Its contributions are `β_j(x) x_j`
summed per term, so "explain a row" works the same way.

```python
from glasshouse.research import AdditiveNet, LocalGLMnet

add = AdditiveNet(family="poisson", glm=lambda: GLM(family="poisson", terms={...}))
local = LocalGLMnet(family="poisson", glm=lambda: GLM(family="poisson", terms={...}))
beta, names = local.fit(df, y, offset=offset).attention(df)
```

## Three ways to hand a network a number

The papers behind this are Gorishniy, Rubachev & Babenko, "On embeddings for numerical
features in tabular deep learning" (NeurIPS 2022) for the encodings and Liu et al., "KAN:
Kolmogorov-Arnold networks" (2024) for the network; the definitions below are theirs,
checked against the papers, with the one place ours differs stated.

`CANN(encoding=...)` decides what the network sees for each numeric input column;
categorical and interaction terms always keep the GLM's design columns.

- **`"design"`** (the default): the GLM's own encoded columns, so a smooth term's spline
  basis is what the net gets. Everything before this page was run this way.
- **`"raw"`**: the standardised value, one column.
- **`"piecewise"`** (their PLE, quantile variant): cut the column into `bins` quantile
  bins on the training rows; the value becomes, per bin, 1 if the bin is entirely below it,
  0 if entirely above, and the fill `(x − b_{t−1}) / (b_t − b_{t−1})` for the bin it sits
  in. A linear layer on that is a piecewise linear function bending at the bin edges. We
  build it as the degree-1 B-spline `piecewise_linear` term, which spans the same functions
  in a different basis (hat functions instead of fills); the network's first layer is
  linear, so the model class is identical. One difference: inside the training range the
  two agree, beyond it the paper's fills extrapolate linearly and our basis holds the
  boundary value. (Their target-aware "T" bins are not offered: they let `y` into the
  design.)
- **`"periodic"`**: `concat[sin(v), cos(v)]`, `v = [2πc₁x, …, 2πc_kx]`, with `frequencies`
  coefficients `c` trained and initialised from `N(0, sigma)`, as a front layer of the
  network. The paper says `sigma` is the hyperparameter that matters, and it is: on
  standardised inputs 0.3 trains well where 1.0 overfits, so 0.3 is the default and it is a
  knob.

Purpose, in the papers' terms: a plain MLP on a raw scalar is bad at sharp or local
effects; the encodings give it the position of the value in a bin, or its phase at several
frequencies, and with them a plain MLP matched attention models and competed with boosted
trees on their benchmarks. Whether that carries to a frequency model on top of a smooth
GLM is what the run below is for.

## The KAN

The Kolmogorov-Arnold theorem says any continuous function of `n` variables is
`f(x) = Σ_{q=1}^{2n+1} Φ_q(Σ_p φ_{q,p}(x_p))`: sums of one-dimensional functions of sums of
one-dimensional functions. A KAN is a network built that way: every edge carries a learnable
one-dimensional function instead of a weight, and nodes only add. Each edge is
`φ(x) = w_b·silu(x) + w_s·Σ_i c_i B_i(x)`, a cubic B-spline on a grid of `grid` intervals
plus a smooth residual basis. The paper's stated advantages are accuracy on functions with
compositional structure and a route to interpretability by pruning edges and reading the
surviving curves; its stated cost is training roughly ten times slower than an MLP.

Ours is two layers, `inputs → hidden[0] → 1`, on a fixed grid over the standardised range
(the paper's grid refinement is not implemented), with the output layer starting at zero so
the model starts as the GLM. A one-layer KAN would be a sum of curves per feature, which is
the additive net and which the smooth GLM already is; the second layer is what lets it
express an interaction while every piece stays a one-dimensional curve. `edge_curves`
returns the first layer's `φ_{q,p}` on a grid per input, and the Model tab draws them: the
KAN's whole claim to being readable, as a picture per input.

## The go/no-go run

`uv run glasshouse bench fremtpl2_cann` fits the smooth GLM (the bar), the three nets built
on that same GLM, and LightGBM on the challengers' stratified five-fold split of freMTPL2.
The committed `benchmarks/fremtpl2_cann/report.md` is its summary and `pinned.json` its
drift test.

Run on 2026-09-06 (held-out, mean ± std over five folds; best per metric in bold):

| metric | glm_smooth | glm_interaction | cann | additive | localglm | lightgbm | naive |
|---|---|---|---|---|---|---|---|
| deviance | 0.59198 ± 0.0021 | 0.59128 ± 0.0021 | 0.58414 ± 0.0027 | 0.59438 ± 0.0017 | 0.58869 ± 0.0023 | **0.5724 ± 0.0026** | 0.62488 |
| d2 | 0.0527 ± 0.0011 | 0.0538 ± 0.0014 | 0.0652 ± 0.0016 | 0.0488 ± 0.0023 | 0.0579 ± 0.0034 | **0.0840 ± 0.0023** | 0 |
| gini | 0.4894 ± 0.016 | 0.4910 ± 0.016 | 0.4799 ± 0.026 | 0.4785 ± 0.021 | 0.4771 ± 0.022 | **0.5351 ± 0.021** | 0 |
| balance | **1.0000 ± 0.0033** | 0.9999 ± 0.0038 | 0.9997 ± 0.0039 | 1.0000 ± 0.0028 | 1.0001 ± 0.0045 | 0.9991 ± 0.0037 | 1 |

Fit time over all folds: glm_smooth 163 s, glm_interaction 208 s, cann 279 s, additive
346 s, localglm 235 s, lightgbm 53 s (each net contains the GLM).

**Verdicts, all three no-go, and each one says something.**

- **CANN**: beats the GLM on held-out deviance by 1.3 %, more than three fold standard
  deviations, and closes about forty percent of the gap to LightGBM. Balance is a tie within
  a tenth of a percent and the Gini is a little lower. The rule asks for both, so it stays
  in the fence as a challenger row.
- **Additive net**: *worse* than the smooth GLM on deviance (0.5944 against 0.5920). The
  GLM's smooths already hold every marginal shape the data supports, and a second bend per
  feature only fits noise. This is the useful result of the run: the CANN's gain is not
  shapes, it is interactions, because the one model that cannot learn interactions gains
  nothing.
- **LocalGLMnet**: better than the GLM by 0.6 %, less than half the CANN's gain, on the same
  interactions expressed as row-dependent coefficients. Balance is a tie, Gini a little
  lower. It stays in the fence too, but its `attention` output is the most readable account
  of *which* interactions matter, which is what the next model should be built from.

What the three net rows together say is that on this data the route from a smooth GLM
towards LightGBM's deviance runs through interactions, not through more flexible marginals.

**The glass-box reply, `glm_interaction`.** The two-feature A/E grid on the Residuals tab
names DrivAge by BonusMalus as the biggest interaction the smooth GLM misses, so the GLM
got exactly that one, as a 4 x 4 tensor-product spline (`Interaction(df=4)`). It helps, and
it is honest about how much: deviance down 0.12 % and the Gini up a little, with the balance
kept. That is about a tenth of the CANN's gain. So the CANN's 1.3 % is not one big
interaction a rating table could add; it is many small ones, which is what a network is
for and a rating table is not. The trade is now stated in numbers: a table of relativities
plus one interaction reaches 0.5913; a network on top of that table reaches 0.5841; a
boosted tree reaches 0.5724 and cannot be read. Which of those to deploy is a business
decision the report makes plainly, not a modelling one it hides.

## The encodings and the KAN, side by side

`uv run glasshouse bench fremtpl2_kan` fits the smooth GLM, the CANN under the design, the
piecewise and the periodic encodings, the two-layer KAN under raw, piecewise and periodic,
and LightGBM, on the same splits. The committed `benchmarks/fremtpl2_kan/report.md` is its
summary and `pinned.json` its drift test.

Run on 2026-09-07 (held-out, mean ± std over five folds; best per metric in bold):

| metric | glm_smooth | cann | cann_piecewise | cann_periodic | kan_raw | kan_piecewise | kan_periodic | lightgbm |
|---|---|---|---|---|---|---|---|---|
| deviance | 0.59198 ± 0.0021 | 0.58414 ± 0.0027 | 0.58360 ± 0.0032 | 0.58251 ± 0.0019 | 0.58299 ± 0.0020 | 0.58196 ± 0.0019 | 0.59197 ± 0.0021 | **0.5724 ± 0.0026** |
| d2 | 0.0527 | 0.0652 | 0.0661 | 0.0678 | 0.0670 | 0.0687 | 0.0527 | **0.0840** |
| gini | 0.4894 ± 0.016 | 0.4799 ± 0.026 | 0.5009 ± 0.024 | 0.4881 ± 0.012 | 0.4863 ± 0.021 | 0.4903 ± 0.020 | 0.4894 ± 0.016 | **0.5351 ± 0.021** |
| balance | 1.0000 ± 0.0033 | 0.9997 ± 0.0039 | 0.9991 ± 0.0026 | 0.9995 ± 0.0033 | 1.0003 ± 0.0041 | 0.9997 ± 0.0037 | **1.0000 ± 0.0033** | 0.9991 ± 0.0037 |
| fit, all folds | 168 s | 316 s | 249 s | 388 s | 1 437 s | 2 372 s | 4 319 s | 53 s |

**Verdicts.**

- **The encoding matters more than the architecture.** Under the MLP, the piecewise and
  periodic encodings both beat the design columns on deviance, and piecewise lifts the
  Gini above the GLM's (0.501 against 0.489) where the design-column CANN had lowered it.
  Under the KAN, piecewise is the best net of the whole track on deviance, 1.7 % below the
  GLM, with the Gini held. The paper's claim carries: how a number reaches the network is
  a first-order choice.
- **KAN against MLP, same encoding:** a little better on deviance (0.5820 against 0.5836
  piecewise; 0.5830 against 0.5841 raw against design), at five to ten times the fit time,
  which is the cost the paper states. The KAN under the periodic encoding did not train at
  all: it sits on the GLM's numbers to the fourth decimal after seventy minutes, an
  honest failure rather than a result.
- **The fence rule, applied.** No row beats the GLM on both deviance and calibration:
  every balance is a tie within a tenth of a percent, none is better. So every model stays
  in the fence. The two nearest the gate are `kan_piecewise` (deviance, Gini held, balance
  a tie) and `cann_piecewise` (deviance, Gini up, balance a shade worse), and the honest
  reading of the balance row is that a re-balanced net ties the GLM on calibration by
  construction and will not beat it; if that rule is to decide anything, it should ask for
  calibration *by segment*, the A/E grids, which is where the difference would show.
- **What to look at.** On the Model tab, the KAN's edge functions per input (the curves the
  model is made of) and "explain a row" with the network column; on the Compare tab, the
  double lift between `kan_piecewise` and `glm_smooth`; on Residuals, the DrivAge by
  BonusMalus grid for each.


## Save and load

`to_dict` writes the GLM, the network's weights as plain lists, the scaling and the shift;
`from_dict` rebuilds it. No pickle, ever.
