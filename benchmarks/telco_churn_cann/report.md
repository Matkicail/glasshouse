# telco_churn — binomial (Churn)

telco_churn — binomial churn (26.5 % positives); target Churn
source: Telco Customer Churn (IBM sample data set; Kaggle blastchar/telco-customer-churn; OpenML 42178). 7 043 customers, 19 features: contract, services, tenure, charges.
cleaning:
  - Churn Yes/No -> 1/0 float; customerID dropped
  - quotes stripped from the string columns OpenML ships ('One year' -> One year)
  - TotalCharges is blank for 11 customers with zero tenure; set to 0
  - 'No internet service' and 'No phone service' on the add-on columns -> No: they repeat InternetService = No / PhoneService = No exactly and would make the one-hot design rank-deficient
  - SeniorCitizen, tenure, MonthlyCharges as floats; nothing dropped

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 7,043. Models: logistic, cann, cann_piecewise, kan_piecewise.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | logistic | cann | cann_piecewise | kan_piecewise | naive |
|---|---|---|---|---|---|
| log_loss | 0.41859 ± 0.0097 | 0.41861 ± 0.0097 | 0.41425 ± 0.0092 | **0.41163 ± 0.0094** | 0.5786 |
| brier | 0.13599 ± 0.0033 | 0.136 ± 0.0033 | 0.13456 ± 0.0032 | **0.13374 ± 0.003** | 0.19495 |
| roc_auc | 0.84327 ± 0.0088 | 0.84326 ± 0.0088 | 0.84721 ± 0.0082 | **0.84914 ± 0.0086** | 0.5 |
| average_precision | 0.65388 ± 0.016 | 0.65377 ± 0.016 | 0.6595 ± 0.017 | **0.66485 ± 0.012** | 0.26537 |
| ks | 0.53961 ± 0.02 | 0.53979 ± 0.019 | **0.5507 ± 0.015** | 0.55018 ± 0.015 | 0 |
| mcc | 0.46628 ± 0.024 | 0.46668 ± 0.023 | 0.46508 ± 0.028 | **0.46926 ± 0.02** | 0 |
| f1 | **0.59201 ± 0.02** | 0.59185 ± 0.02 | 0.58515 ± 0.023 | 0.58759 ± 0.021 | 0 |
| balance | 1.0008 ± 0.025 | **0.99924 ± 0.025** | 1.0048 ± 0.031 | 1.0105 ± 0.031 | 1 |

Fit time (all folds): logistic 1.2s, cann 29.4s, cann_piecewise 5.4s, kan_piecewise 16.8s.
