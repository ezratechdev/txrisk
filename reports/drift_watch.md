# Drift watch

Days checked: 2023-03-03, 2026-09-05, 2026-09-07.

| day | addresses | fraud_rate | mean_score | pr_auc | ece |
|---|---|---|---|---|---|
| 2023-03-03 | 550584 | 0.018 | 0.019 | 0.381 | 0.001 |
| 2026-09-05 | 1124039 | 0.056 | 0.057 | 0.395 | 0.017 |
| 2026-09-07 | 1200191 | 0.076 | 0.032 | 0.740 | 0.044 |

**Recalibrate.** Calibration error reached 0.044, above the 0.02 the probabilities are trusted within. Ranking may still be fine, which is exactly why this is easy to miss.

## Why calibration rather than ranking

A model trained in 2023 and shown September 2026 still put real attackers at the top, at
0.94 precision, while its calibration error grew fourfold. Ranking is the last thing to
fail and the first thing people check, which is how a model spends months reporting
probabilities that no longer mean anything.

`fraud_rate` and `mean_score` are here for the same reason: if the rules stop firing, the
labels this check relies on quietly disappear, and a clean report would mean nothing.
