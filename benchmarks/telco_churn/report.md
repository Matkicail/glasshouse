# telco_churn — binomial (Churn)

telco_churn — binomial churn (26.5 % positives); target Churn
source: Telco Customer Churn (IBM sample data set; Kaggle blastchar/telco-customer-churn; OpenML 42178). 7 043 customers, 19 features: contract, services, tenure, charges.
cleaning:
  - Churn Yes/No -> 1/0 float; customerID dropped
  - quotes stripped from the string columns OpenML ships ('One year' -> One year)
  - TotalCharges is blank for 11 customers with zero tenure; set to 0
  - 'No internet service' and 'No phone service' on the add-on columns -> No: they repeat InternetService = No / PhoneService = No exactly and would make the one-hot design rank-deficient
  - SeniorCitizen, tenure, MonthlyCharges as floats; nothing dropped

Split: {'kind': 'random', 'method': 'stratified', 'k': 5, 'seed': 0}. Rows: 7,043. Models: logistic, lasso_logistic.
Scores are held-out, mean ± std over folds. Best per metric in bold; `naive` is the weighted mean of y (class prior for binomial), same folds.

| metric | logistic | lasso_logistic | naive |
|---|---|---|---|
| log_loss | 0.41859 ± 0.0097 | **0.41828 ± 0.01** | 0.5786 |
| brier | 0.13599 ± 0.0033 | **0.1359 ± 0.0035** | 0.19495 |
| roc_auc | 0.84327 ± 0.0088 | **0.84343 ± 0.0093** | 0.5 |
| average_precision | 0.65388 ± 0.016 | **0.65426 ± 0.018** | 0.26537 |
| ks | 0.53961 ± 0.02 | **0.54182 ± 0.019** | 0 |
| mcc | **0.46628 ± 0.024** | 0.4642 ± 0.022 | 0 |
| f1 | **0.59201 ± 0.02** | 0.58891 ± 0.02 | 0 |
| balance | 1.0008 ± 0.025 | **1.0006 ± 0.025** | 1 |

Fit time (all folds): logistic 1.6s, lasso_logistic 96.6s.
