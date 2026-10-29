# FraudShield AI Model Card

## Summary

FraudShield AI is a portfolio research backend for binary transaction-fraud ranking and review prioritization. It is not validated for production banking decisions, regulatory compliance, or use as the sole basis for blocking customers.

## Data and evaluation

- Dataset: IEEE-CIS Fraud Detection, 590,540 labeled transactions.
- Rows are ordered by `TransactionDT`; the oldest 80% is used for training and the newest 20% is held out once for final evaluation.
- The training period contains 472,432 rows; the final holdout contains 118,108 rows with 3.44% fraud prevalence.
- Feature generation excludes labels and uses only earlier transactions for seven-day card and card/address history features.
- Feature selection and tuning use an initial training-period warm-up. The final holdout is not used for fraud-model fitting, calibration, threshold selection, or model selection.
- Three shared forward-in-time windows compare the stack and CatBoost. The stack had a slightly higher mean average precision across those windows (0.5459 vs. 0.5439) and better mean ROC-AUC; the holdout remains a one-period evaluation.

## Model and serving

The served model stacks XGBoost, LightGBM, CatBoost, Random Forest, and MLP probabilities with an XGBoost meta-model. An isotonic calibrator is fitted on an earlier out-of-time training segment; the decision threshold is selected on a later training segment using illustrative cost weights. SMOTE is limited to XGBoost and MLP; LightGBM, CatBoost, and Random Forest use native class weighting. Isolation Forest and an autoencoder are reported as separate research baselines and are not part of the served score.

Training and API scoring share the same feature pipeline. Prior-seven-day histories are maintained for card and card/address entities. API history is in process memory, so multi-worker or multi-replica serving needs a shared event store.

## Latest holdout results

| Candidate | ROC-AUC | Average precision | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Served stack | 0.9142 | 0.4883 | 0.0225 | 0.0902 |
| CatBoost standalone | 0.9145 | 0.5079 | 0.0723 | 0.2565 |

The served stack's 95% stratified-bootstrap intervals are **0.9090–0.9190** for ROC-AUC and **0.4735–0.5026** for average precision. CatBoost's corresponding intervals are **0.9103–0.9190** and **0.4933–0.5242**. The average-precision intervals overlap; these one-run results do not establish a meaningful ranking difference. CatBoost's raw probabilities are less calibrated, while the served stack has lower Brier score and log loss.

At the training-selected stack threshold of 0.059, holdout precision is 0.1995, recall is 0.7805, and F1 is 0.3178. This threshold reflects illustrative false-negative, false-positive, and review-cost weights; it is not a real institution's operating policy. Holdout average precision is about 14 times the 3.44% prevalence baseline.

Across four consecutive holdout segments, stack ROC-AUC ranges from 0.9066 to 0.9195. The train-versus-holdout adversarial classifier reaches AUC 0.7895 and remains flagged **CRITICAL** for feature shift. Its leading signals include `D15`, `D10`, and `D11`; this is evidence that future-period transfer remains uncertain. Named feature PSI values are recorded in `outputs/results/adversarial_validation.json`.

## Limits and intended use

Use for experimentation, coursework, and portfolio demonstration. The dataset represents historical US e-commerce activity, and results may not transfer to other products, regions, or current fraud patterns. Validate on additional later periods, monitor drift, review fairness, and set thresholds from real operating costs before any deployment decision.

The latest local run metadata is in `outputs/results/training_run.json`; holdout metrics and intervals are in `outputs/results/holdout_evaluation.json`; shared-window comparisons are in `outputs/results/temporal_cv_metrics.json`. Model binaries and raw data are excluded from Git, so a clean clone must train a fresh compatible bundle before serving.
