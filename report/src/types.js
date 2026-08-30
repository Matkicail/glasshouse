"use strict";
// The shape of a glasshouse-report/1 document. Hand-written to mirror report/schema.json;
// the vitest suite renders the Python-produced fixture, so a drift between the two fails a test.
// Nothing here is computed: the browser only draws what Python wrote.
// Direction of "better" per metric; mirrors glasshouse.scorecard.HIGHER_IS_BETTER.
const HIGHER_IS_BETTER = {
    deviance: false, d2: true, gini: true, normalized_gini: true, rmse: false, mae: false, r2: true,
    mcc: true, f1: true, roc_auc: true, average_precision: true, ks: true, log_loss: false, brier: false,
};
