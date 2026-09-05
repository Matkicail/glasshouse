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

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glm_simple, glm_full.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_simple | glm_full | naive |
|---|---|---|---|
| deviance | 0.6106 ± 0.0025 | **0.60493 ± 0.0025** | 0.62488 |
| d2 | 0.022849 ± 0.0003 | **0.03192 ± 0.00033** | 2.3381e-14 |
| gini | 0.19634 ± 0.02 | **0.39515 ± 0.016** | 0 |
| normalized_gini | 0.19833 ± 0.02 | **0.39913 ± 0.016** | 0 |
| balance | 1 ± 0.0054 | **1 ± 0.0049** | 1 |
| rmse | 0.73529 ± 0.012 | **0.73496 ± 0.012** | 0.73654 |
| mae | 0.18705 ± 0.00043 | **0.1868 ± 0.00047** | 0.18894 |
| r2 | 0.0033886 ± 0.00015 | **0.0042698 ± 0.00023** | 5.5955e-15 |

Fit time (all folds): glm_simple 3.9s, glm_full 9.0s.
