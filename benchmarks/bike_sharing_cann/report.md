# bike_sharing — poisson (count)

bike_sharing — poisson counts, time-ordered (hourly rentals; split on hour_index); target count
source: Bike Sharing Dataset, Capital Bikeshare, Washington DC (Fanaee-T & Gama, 'Event labeling combining ensemble detectors and background knowledge', Progress in AI, 2014; UCI). 17 379 hourly rows, 2011-01-01 to 2012-12-31, weather and calendar features. OpenML 42712 (which omits the casual/registered split of the count).
cleaning:
  - hour_index = row order added; the source is chronological (checked: year-month never decreases), so a time-ordered split cuts on it
  - year 0/1 -> 2011/2012; holiday and workingday -> 0/1 floats
  - season and weather kept as their level names; temperatures, humidity and windspeed are the source's normalised values, untouched
  - nothing dropped, nothing capped

Split: {'kind': 'time', 'method': 'time_ordered', 'n_folds': 5, 'min_train_fraction': 0.3}. Rows: 17,379. Models: glm_poisson, cann, cann_piecewise, kan_piecewise, lightgbm.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | glm_poisson | cann | cann_piecewise | kan_piecewise | lightgbm | naive |
|---|---|---|---|---|---|---|
| deviance | 66.508 ± 19 | 42.229 ± 14 | 43.038 ± 15 | **40.78 ± 14** | 43.015 ± 14 | 161.63 |
| d2 | 0.58618 ± 0.096 | 0.73726 ± 0.088 | 0.73315 ± 0.086 | **0.74771 ± 0.075** | 0.73281 ± 0.077 | 0 |
| gini | 0.39949 ± 0.021 | 0.44583 ± 0.015 | 0.44489 ± 0.015 | **0.44597 ± 0.015** | 0.44469 ± 0.016 | 0 |
| normalized_gini | 0.84323 ± 0.042 | 0.94116 ± 0.031 | 0.93912 ± 0.03 | **0.94131 ± 0.025** | 0.93867 ± 0.03 | 0 |
| balance | 1.4006 ± 0.17 | 1.399 ± 0.16 | 1.3974 ± 0.16 | **1.3818 ± 0.16** | 1.405 ± 0.15 | 1 |
| rmse | 128.39 ± 33 | 102.2 ± 30 | 103.26 ± 30 | **100.78 ± 29** | 103.63 ± 29 | 182.25 |
| mae | 85.364 ± 22 | 71.583 ± 23 | 71.754 ± 23 | **70.335 ± 23** | 72.17 ± 23 | 146.7 |
| r2 | 0.49765 ± 0.11 | 0.67787 ± 0.1 | 0.6707 ± 0.11 | **0.68934 ± 0.092** | 0.6707 ± 0.095 | 0 |

Fit time (all folds): glm_poisson 1.0s, cann 30.6s, cann_piecewise 7.8s, kan_piecewise 48.5s, lightgbm 13.3s.
