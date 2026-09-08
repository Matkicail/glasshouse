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

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glm_smooth, cann, cann_piecewise, cann_periodic, kan_raw, kan_piecewise, kan_periodic, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_smooth | cann | cann_piecewise | cann_periodic | kan_raw | kan_piecewise | kan_periodic | lightgbm | naive |
|---|---|---|---|---|---|---|---|---|---|
| deviance | 0.59198 ± 0.0021 | 0.58414 ± 0.0027 | 0.5836 ± 0.0032 | 0.58251 ± 0.0019 | 0.58299 ± 0.002 | 0.58196 ± 0.0019 | 0.59197 ± 0.0021 | **0.5724 ± 0.0026** | 0.62488 |
| deviance_per_row | 0.31289 ± 0.0011 | 0.30874 ± 0.0015 | 0.30846 ± 0.0017 | 0.30788 ± 0.0011 | 0.30814 ± 0.0012 | 0.30759 ± 0.00082 | 0.31288 ± 0.0011 | **0.30253 ± 0.0012** | 0.33028 |
| d2 | 0.052654 ± 0.0011 | 0.065198 ± 0.0016 | 0.06607 ± 0.0026 | 0.067807 ± 0.0024 | 0.067026 ± 0.0023 | 0.068685 ± 0.0016 | 0.052657 ± 0.0011 | **0.08399 ± 0.0023** | 2.3381e-14 |
| gini | 0.48938 ± 0.016 | 0.47987 ± 0.026 | 0.50091 ± 0.024 | 0.48809 ± 0.012 | 0.48632 ± 0.021 | 0.49029 ± 0.02 | 0.48941 ± 0.016 | **0.53505 ± 0.021** | 0 |
| normalized_gini | 0.49431 ± 0.016 | 0.4847 ± 0.026 | 0.50596 ± 0.024 | 0.49301 ± 0.012 | 0.49121 ± 0.021 | 0.49523 ± 0.02 | 0.49433 ± 0.016 | **0.54044 ± 0.022** | 0 |
| balance | 1 ± 0.0033 | 0.99968 ± 0.0039 | 0.99912 ± 0.0026 | 0.99951 ± 0.0033 | 1.0003 ± 0.0041 | 0.99967 ± 0.0037 | **1 ± 0.0033** | 0.99912 ± 0.0037 | 1 |
| rmse | 0.73362 ± 0.012 | 0.73176 ± 0.012 | 0.73155 ± 0.012 | 0.73158 ± 0.012 | 0.73134 ± 0.012 | 0.73124 ± 0.012 | 0.73362 ± 0.012 | **0.72948 ± 0.012** | 0.73654 |
| mae | 0.18562 ± 0.0006 | 0.18409 ± 0.00058 | 0.18376 ± 0.00058 | 0.18381 ± 0.00061 | 0.18352 ± 0.00069 | 0.18347 ± 0.00061 | 0.18562 ± 0.0006 | **0.18221 ± 0.00066** | 0.18894 |
| r2 | 0.0079095 ± 0.00054 | 0.012929 ± 0.00072 | 0.013503 ± 0.00069 | 0.013413 ± 0.00084 | 0.014057 ± 0.00088 | 0.014346 ± 0.00087 | 0.0079105 ± 0.00054 | **0.019076 ± 0.00078** | 5.5955e-15 |

Fit time (all folds): glm_smooth 152.8s, cann 254.8s, cann_piecewise 240.0s, cann_periodic 346.0s, kan_raw 1179.7s, kan_piecewise 2018.0s, kan_periodic 3669.9s, lightgbm 49.9s.
