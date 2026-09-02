# creditcard — binomial (Class)

creditcard — binomial, rare event (0.17 % positives); target Class
source: Credit Card Fraud Detection, ULB Machine Learning Group (Dal Pozzolo et al., 'Calibrating probability with undersampling for unbalanced classification', IEEE CIDM 2015). 284 807 transactions over two days, 492 frauds; features V1-V28 are PCA components, plus Amount (the OpenML copy carries no Time column). OpenML 1597.
cleaning:
  - Class parsed to 0/1 float; nothing else touched

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 284,807. Models: logistic.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | logistic | naive |
|---|---|---|
| log_loss | **0.0040999 ± 0.00058** | 0.012715 |
| brier | **0.00069464 ± 8.6e-05** | 0.0017245 |
| roc_auc | **0.97429 ± 0.0077** | 0.5 |
| average_precision | **0.7598 ± 0.045** | 0.0017275 |
| ks | **0.88331 ± 0.022** | 0 |
| mcc | **0.73176 ± 0.035** | 0 |
| f1 | **0.72091 ± 0.036** | 0 |
| balance | **1.0043 ± 0.047** | 1 |

Fit time (all folds): logistic 5.2s.
