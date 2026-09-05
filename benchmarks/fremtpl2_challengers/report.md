# fremtpl2_freq — poisson (ClaimNb)

fremtpl2_freq — poisson frequency (offset = log Exposure); target ClaimNb, exposure Exposure
source: French MTPL claim frequency, 678 013 policies (CASdatasets freMTPL2freq; OpenML 41214). Cleaning follows Noll, Schelldorfer & Wüthrich, 'Case Study: French Motor Third-Party Liability Claims' (SSRN 3164764, 2018) and Wüthrich & Merz, 'Statistical Foundations of Actuarial Learning and its Applications' (Springer 2023, §13.1).
cleaning:
  - ClaimNb capped at 4 (a few rows report 5-16 claims in one period)
  - Exposure capped at 1 (a few rows exceed one policy-year)
  - VehPower capped at 9, VehAge at 20, DrivAge at 90, BonusMalus at 150
  - LogDensity = log(Density) added; AreaCode = A..F -> 1..6 added
  - Frequency = ClaimNb / Exposure added (the rate; model ClaimNb with offset log Exposure)
  - quotes stripped from the string columns OpenML ships ('Diesel' -> Diesel)
  - NOT applied: the Wüthrich-Merz Appendix A.1 de-duplication of near-identical policies

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glm_full, glm_splines, glm_smooth, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_full | glm_splines | glm_smooth | lightgbm | naive |
|---|---|---|---|---|---|
| deviance | 0.60493 ± 0.0025 | 0.59279 ± 0.0021 | 0.59198 ± 0.0021 | **0.5724 ± 0.0026** | 0.62488 |
| d2 | 0.03192 ± 0.00033 | 0.051355 ± 0.0011 | 0.052654 ± 0.0011 | **0.08399 ± 0.0023** | 2.3381e-14 |
| gini | 0.39515 ± 0.016 | 0.48879 ± 0.016 | 0.48938 ± 0.016 | **0.53505 ± 0.021** | 0 |
| normalized_gini | 0.39913 ± 0.016 | 0.49371 ± 0.016 | 0.49431 ± 0.016 | **0.54044 ± 0.022** | 0 |
| balance | **1 ± 0.0049** | 1 ± 0.0037 | 1 ± 0.0033 | 0.99912 ± 0.0037 | 1 |
| rmse | 0.73496 ± 0.012 | 0.73367 ± 0.012 | 0.73362 ± 0.012 | **0.72948 ± 0.012** | 0.73654 |
| mae | 0.1868 ± 0.00047 | 0.18577 ± 0.00055 | 0.18562 ± 0.0006 | **0.18221 ± 0.00066** | 0.18894 |
| r2 | 0.0042698 ± 0.00023 | 0.0077833 ± 0.00052 | 0.0079093 ± 0.00054 | **0.019076 ± 0.00078** | 5.5955e-15 |

Fit time (all folds): glm_full 9.2s, glm_splines 13.4s, glm_smooth 301.6s, lightgbm 36.8s.
