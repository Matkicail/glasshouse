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

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 678,013. Models: glm_full, glm_splines, glm_piecewise, glm_smooth, glm_interaction, glm_interactions, lightgbm_additive, lightgbm_pairs, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.
Ladder: glm_full → glm_splines → glm_smooth → glm_interaction → glm_interactions → lightgbm_additive → lightgbm_pairs → lightgbm.

| metric | glm_full | glm_splines | glm_piecewise | glm_smooth | glm_interaction | glm_interactions | lightgbm_additive | lightgbm_pairs | lightgbm | naive |
|---|---|---|---|---|---|---|---|---|---|---|
| deviance | 0.60493 ± 0.0025 | 0.59279 ± 0.0021 | 0.59221 ± 0.0021 | 0.59198 ± 0.0021 | 0.59128 ± 0.0021 | 0.58628 ± 0.0021 | 0.58766 ± 0.0024 | 0.57751 ± 0.0025 | **0.5724 ± 0.0026** | 0.62488 |
| deviance_per_row | 0.31973 ± 0.0013 | 0.31331 ± 0.0011 | 0.31301 ± 0.0011 | 0.31289 ± 0.0011 | 0.31252 ± 0.0011 | 0.30988 ± 0.0012 | 0.31061 ± 0.0013 | 0.30524 ± 0.0013 | **0.30253 ± 0.0012** | 0.33028 |
| d2 | 0.03192 ± 0.00033 | 0.051355 ± 0.0011 | 0.052279 ± 0.0012 | 0.052654 ± 0.0011 | 0.053769 ± 0.0014 | 0.061765 ± 0.0015 | 0.059556 ± 0.0016 | 0.075801 ± 0.0025 | **0.08399 ± 0.0023** | 2.3381e-14 |
| gini | 0.39515 ± 0.016 | 0.48879 ± 0.016 | 0.48625 ± 0.018 | 0.48938 ± 0.016 | 0.49098 ± 0.016 | 0.48295 ± 0.022 | 0.49636 ± 0.016 | 0.51516 ± 0.023 | **0.53505 ± 0.021** | 0 |
| normalized_gini | 0.39913 ± 0.016 | 0.49371 ± 0.016 | 0.49114 ± 0.018 | 0.49431 ± 0.016 | 0.49592 ± 0.016 | 0.48781 ± 0.022 | 0.50135 ± 0.016 | 0.52035 ± 0.023 | **0.54044 ± 0.022** | 0 |
| balance | **1 ± 0.0049** | 1 ± 0.0037 | 1 ± 0.0036 | 1 ± 0.0033 | 0.99991 ± 0.0038 | 1 ± 0.0028 | 1.0011 ± 0.0031 | 1.0011 ± 0.0037 | 0.99912 ± 0.0037 | 1 |
| rmse | 0.73496 ± 0.012 | 0.73367 ± 0.012 | 0.73364 ± 0.012 | 0.73362 ± 0.012 | 0.73352 ± 0.012 | 0.73274 ± 0.012 | 0.7332 ± 0.012 | 0.731 ± 0.012 | **0.72948 ± 0.012** | 0.73654 |
| mae | 0.1868 ± 0.00047 | 0.18577 ± 0.00055 | 0.1857 ± 0.00055 | 0.18562 ± 0.0006 | 0.1855 ± 0.00058 | 0.18461 ± 0.00064 | 0.18485 ± 0.00064 | 0.18314 ± 0.00057 | **0.18221 ± 0.00066** | 0.18894 |
| r2 | 0.0042698 ± 0.00023 | 0.0077833 ± 0.00052 | 0.0078674 ± 0.00056 | 0.0079095 ± 0.00054 | 0.0081742 ± 0.00062 | 0.010291 ± 0.00075 | 0.0090341 ± 0.00068 | 0.014991 ± 0.0008 | **0.019076 ± 0.00078** | 5.5955e-15 |

Fit time (all folds): glm_full 8.2s, glm_splines 14.8s, glm_piecewise 15.6s, glm_smooth 168.2s, glm_interaction 218.3s, glm_interactions 963.2s, lightgbm_additive 192.8s, lightgbm_pairs 43.5s, lightgbm 57.6s.

Interaction candidates (deviance per unit weight a pair's A/E grid explains beyond the one-way views; the three biggest per model):
- glm_full: VehBrand × VehAge 0.00219, VehGas × VehAge 0.0015, DrivAge × BonusMalus 0.00128
- glm_splines: VehBrand × VehAge 0.00196, VehGas × VehAge 0.00153, Region × DrivAge 0.00121
- glm_piecewise: VehBrand × VehAge 0.00192, VehGas × VehAge 0.00153, Region × DrivAge 0.0012
- glm_smooth: VehBrand × VehAge 0.00194, VehGas × VehAge 0.00152, Region × DrivAge 0.0012
- glm_interaction: VehBrand × VehAge 0.00193, VehGas × VehAge 0.00151, Region × DrivAge 0.00116
- glm_interactions: VehBrand × BonusMalus 0.00128, VehAge × BonusMalus 0.00117, Region × LogDensity 0.00109
- lightgbm_additive: VehBrand × VehAge 0.00201, VehGas × VehAge 0.00138, Region × DrivAge 0.0012
- lightgbm_pairs: Region × DrivAge 0.000668, VehBrand × Region 0.000638, Region × LogDensity 0.000549
- lightgbm: Region × LogDensity 0.000563, Region × DrivAge 0.000496, VehBrand × Region 0.000462
