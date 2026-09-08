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

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glm_smooth, glm_interaction, cann, additive, localglm, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_smooth | glm_interaction | cann | additive | localglm | lightgbm | naive |
|---|---|---|---|---|---|---|---|
| deviance | 0.59198 ± 0.0021 | 0.59128 ± 0.0021 | 0.58414 ± 0.0027 | 0.59438 ± 0.0017 | 0.58869 ± 0.0023 | **0.5724 ± 0.0026** | 0.62488 |
| d2 | 0.052654 ± 0.0011 | 0.053769 ± 0.0014 | 0.065198 ± 0.0016 | 0.048807 ± 0.0023 | 0.057903 ± 0.0034 | **0.08399 ± 0.0023** | 2.3381e-14 |
| gini | 0.48938 ± 0.016 | 0.49098 ± 0.016 | 0.47987 ± 0.026 | 0.4785 ± 0.021 | 0.47714 ± 0.022 | **0.53505 ± 0.021** | 0 |
| normalized_gini | 0.49431 ± 0.016 | 0.49592 ± 0.016 | 0.4847 ± 0.026 | 0.48332 ± 0.021 | 0.48194 ± 0.022 | **0.54044 ± 0.022** | 0 |
| balance | **1 ± 0.0033** | 0.99991 ± 0.0038 | 0.99968 ± 0.0039 | 0.99996 ± 0.0028 | 1.0001 ± 0.0045 | 0.99912 ± 0.0037 | 1 |
| rmse | 0.73362 ± 0.012 | 0.73352 ± 0.012 | 0.73176 ± 0.012 | 0.73384 ± 0.012 | 0.73314 ± 0.012 | **0.72948 ± 0.012** | 0.73654 |
| mae | 0.18562 ± 0.0006 | 0.1855 ± 0.00058 | 0.18409 ± 0.00058 | 0.18571 ± 0.00063 | 0.1841 ± 0.00079 | **0.18221 ± 0.00066** | 0.18894 |
| r2 | 0.0079095 ± 0.00054 | 0.0081742 ± 0.00062 | 0.012929 ± 0.00072 | 0.0073034 ± 0.00055 | 0.0091895 ± 0.003 | **0.019076 ± 0.00078** | 5.5955e-15 |

Fit time (all folds): glm_smooth 162.5s, glm_interaction 208.3s, cann 278.5s, additive 346.4s, localglm 234.7s, lightgbm 52.6s.
