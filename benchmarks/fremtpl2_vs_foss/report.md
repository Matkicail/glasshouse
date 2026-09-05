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

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glasshouse, glum, sklearn.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glasshouse | glum | sklearn | naive |
|---|---|---|---|---|
| deviance | **0.60484 ± 0.0025** | 0.60484 ± 0.0025 | 0.60484 ± 0.0025 | 0.62488 |
| d2 | **0.032069 ± 0.00051** | 0.032069 ± 0.00051 | 0.032065 ± 0.00052 | 2.3381e-14 |
| gini | **0.39456 ± 0.019** | 0.39456 ± 0.019 | 0.39444 ± 0.019 | 0 |
| normalized_gini | **0.39854 ± 0.019** | 0.39854 ± 0.019 | 0.39842 ± 0.019 | 0 |
| balance | **1 ± 0.0049** | 1 ± 0.0049 | 0.99997 ± 0.0049 | 1 |
| rmse | **0.73496 ± 0.012** | 0.73496 ± 0.012 | 0.73496 ± 0.012 | 0.73654 |
| mae | **0.18678 ± 0.00047** | 0.18678 ± 0.00047 | 0.18678 ± 0.00047 | 0.18894 |
| r2 | **0.0042855 ± 0.00027** | 0.0042855 ± 0.00027 | 0.0042837 ± 0.00027 | 5.5955e-15 |

Fit time (all folds): glasshouse 10.3s, glum 23.8s, sklearn 690.8s.
