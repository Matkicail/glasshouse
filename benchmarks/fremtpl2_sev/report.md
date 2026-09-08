# fremtpl2_sev — gamma (Severity)

fremtpl2_sev — gamma severity (mean claim amount per claim, weight = ClaimCount); target Severity, exposure ClaimCount
source: French MTPL claim severity, 26 639 claims on 24 950 policies (CASdatasets freMTPL2sev; OpenML 41215), joined to the cleaned freMTPL2freq policies on IDpol. Same references as fremtpl2_freq (Noll, Schelldorfer & Wüthrich 2018; Wüthrich & Merz 2023, §13.1).
cleaning:
  - ClaimAmount per claim capped at 1 000 000 (three claims exceed it)
  - claims summed per policy: ClaimCount (the weight) and ClaimTotal; Severity = ClaimTotal / ClaimCount (the target)
  - inner join to the cleaned frequency frame on IDpol: 6 claims with no policy row dropped; policies with ClaimNb > 0 but no severity record (about 9 100) are not in this frame, a known quirk of the source
  - ClaimCount is the count in the severity file; it agrees with the capped ClaimNb on 99.9 % of policies

Split: {'kind': 'random', 'method': 'kfold', 'k': 5, 'seed': 0}. Rows: 24,944. Models: glm_gamma, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_gamma | lightgbm | naive |
|---|---|---|---|
| deviance | **1.5641 ± 0.14** | 1.5785 ± 0.21 | 1.5635 |
| deviance_per_row | **1.6581 ± 0.15** | 1.6734 ± 0.22 | 1.6575 |
| d2 | **-0.0046644 ± 0.045** | -0.0083372 ± 0.011 | -1.4211e-15 |
| gini | **0.057426 ± 0.075** | 0.041688 ± 0.058 | 0 |
| normalized_gini | **0.089989 ± 0.12** | 0.065486 ± 0.089 | 0 |
| balance | **1.0109 ± 0.12** | 1.0807 ± 0.13 | 1 |
| rmse | 12478 ± 5.1e+03 | **12469 ± 5.1e+03** | 12467 |
| mae | 2007.5 ± 1.7e+02 | **1894 ± 1.7e+02** | 2002.5 |
| r2 | -0.0035075 ± 0.0048 | **-0.00014549 ± 0.00065** | -1.7764e-16 |

Fit time (all folds): glm_gamma 2.6s, lightgbm 8.7s.
