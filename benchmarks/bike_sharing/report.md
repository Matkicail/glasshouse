# bike_sharing — poisson (count)

bike_sharing — poisson counts, time-ordered (hourly rentals; split on hour_index); target count
source: Bike Sharing Dataset, Capital Bikeshare, Washington DC (Fanaee-T & Gama, 'Event labeling combining ensemble detectors and background knowledge', Progress in AI, 2014; UCI). 17 379 hourly rows, 2011-01-01 to 2012-12-31, weather and calendar features. OpenML 42712 (which omits the casual/registered split of the count).
cleaning:
  - hour_index = row order added; the source is chronological (checked: year-month never decreases), so a time-ordered split cuts on it
  - year 0/1 -> 2011/2012; holiday and workingday -> 0/1 floats
  - season and weather kept as their level names; temperatures, humidity and windspeed are the source's normalised values, untouched
  - nothing dropped, nothing capped

Split: {'kind': 'time', 'method': 'time_ordered', 'n_folds': 5, 'min_train_fraction': 0.3}. Rows: 17,379. Models: glm_poisson, glm_interaction, lightgbm_additive, lightgbm_pairs, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.
Ladder: glm_poisson → glm_interaction → lightgbm_additive → lightgbm_pairs → lightgbm.

| metric | glm_poisson | glm_interaction | lightgbm_additive | lightgbm_pairs | lightgbm | naive |
|---|---|---|---|---|---|---|
| deviance | 66.508 ± 19 | 45.836 ± 16 | 65.093 ± 18 | **41.813 ± 14** | 43.015 ± 14 | 161.63 |
| d2 | 0.58618 ± 0.096 | 0.71568 ± 0.093 | 0.59359 ± 0.095 | **0.74127 ± 0.079** | 0.73281 ± 0.077 | 0 |
| gini | 0.39949 ± 0.021 | 0.4391 ± 0.017 | 0.40224 ± 0.022 | **0.44531 ± 0.016** | 0.44469 ± 0.016 | 0 |
| normalized_gini | 0.84323 ± 0.042 | 0.92697 ± 0.035 | 0.84897 ± 0.042 | **0.93995 ± 0.029** | 0.93867 ± 0.03 | 0 |
| balance | 1.4006 ± 0.17 | 1.4015 ± 0.17 | 1.4025 ± 0.16 | **1.3897 ± 0.16** | 1.405 ± 0.15 | 1 |
| rmse | 128.39 ± 33 | 108.41 ± 31 | 126.43 ± 32 | **103 ± 30** | 103.63 ± 29 | 182.25 |
| mae | 85.364 ± 22 | 73.269 ± 23 | 85.071 ± 22 | **71.479 ± 23** | 72.17 ± 23 | 146.7 |
| r2 | 0.49765 ± 0.11 | 0.63684 ± 0.12 | 0.51145 ± 0.11 | **0.67383 ± 0.1** | 0.6707 ± 0.095 | 0 |

Fit time (all folds): glm_poisson 1.6s, glm_interaction 2.4s, lightgbm_additive 17.1s, lightgbm_pairs 12.6s, lightgbm 13.3s.

Interaction candidates (deviance per unit weight a pair's A/E grid explains beyond the one-way views; the three biggest per model):
- glm_poisson: hour × workingday 19.6, workingday × temp 1.64, hour × temp 1.54
- glm_interaction: hour × temp 1.44, season × hour 1.12, workingday × temp 0.707
- lightgbm_additive: hour × workingday 19.7, hour × temp 1.62, workingday × temp 1.59
- lightgbm_pairs: hour × workingday 0.669, temp × humidity 0.442, season × temp 0.431
- lightgbm: season × temp 0.403, temp × humidity 0.392, hour × temp 0.315
