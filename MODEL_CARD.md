# FraudShield AI — Model Card

**Version:** 3.0 | **Last Updated:** April 2026

---

## Model Details

| Field | Value |
|-------|-------|
| **Architecture** | Dual Stacking Ensemble (8 Base Models) |
| **Meta-Learner** | XGBoost + Rank-Weighted Blend (auto-selects best) |
| **Training Data** | IEEE-CIS Fraud Detection (~590K transactions) |
| **Cross-Validation** | 7-Fold Stratified |
| **Hyperparameter Tuning** | 150 Optuna trials per model |
| **Uncertainty Quantification** | Conformal Prediction (95% coverage) |
| **Probability Calibration** | Isotonic Regression |

## Intended Use

- **Primary:** Real-time credit/debit card fraud detection for financial institutions
- **Secondary:** Transaction monitoring, suspicious activity reporting (SAR), regulatory compliance
- **Not Intended For:** Credit scoring, identity verification, or standalone legal decision-making

## Performance

| Metric | Value |
|--------|-------|
| AUC-ROC (XGBoost) | **0.9776** |
| AUC-ROC (Ensemble) | **0.9763** |
| Ensemble F1 | 0.765 |
| Ensemble Precision | 0.834 |
| Ensemble Recall | 0.707 |
| Cross-Validation | 7-Fold with Optuna HPO (150 trials) |
| Detection Layers | 25 engineered feature groups |
| Inference Latency | <50ms per transaction |
| Threshold | Cost-optimized at 0.015 (FN=₹850, FP=₹25) |
| Adversarial Validation | AUC 0.5004 (EXCELLENT — no data leakage) |

## Feature Groups (25 Layers)

1. Amount Analysis | 2. Temporal Patterns | 3. Card Fingerprinting | 4. Address Intelligence
5. Email Domain Risk | 6. Device Forensics | 7. Velocity Engine | 8. Merchant Risk
9. Behavioral DNA | 10. SIM Swap Detection | 11. Seasonal Baselines | 12. Adaptive Auth
13. Mule Network | 14. Dormant Account Hijack | 15. Round Amount Anomaly | 16. Category Mismatch
17. New Account Risk | 18. Graph Analysis | 19. UID Profiling | 20. Target Encoding
21. V-Feature PCA | 22. Frequency Encoding | 23. Peer-Group Deviation | 24. Transaction Entropy
25. Cross-Feature Fraud Rates

## Ethical Considerations

### Bias & Fairness
- **Fairness audit** is built into the dashboard to detect bias across card types, product categories, and demographics
- Model does not use protected attributes (race, gender, age) as direct features
- Peer-group deviation features are computed per-card, not per-demographic

### Transparency
- **SHAP** TreeExplainer provides global and local feature importance
- **LIME** generates plain-English explanations for each decision
- **Counterfactual analysis** explains what would need to change to flip a decision
- All decisions are auditable via scored transaction logs

### Right to Explanation (GDPR Art. 22)
- Every fraud decision includes a human-readable explanation
- Customers can request detailed reasoning for any flagged transaction
- Conformal prediction identifies "uncertain" cases for mandatory human review

### Limitations
- Trained on US e-commerce data; may need recalibration for other regions
- Performance may degrade on transaction patterns not represented in training data
- Adversarial attacks specifically designed to evade this model have not been fully tested
- Model assumes stationary fraud patterns; concept drift requires periodic retraining

## Monitoring & Maintenance

- **Drift detection** tracks feature distribution shifts and AUC degradation
- **Retraining triggers:** AUC drops below 0.97 → warning; below 0.95 → critical
- **Adversarial testing** included for noise injection robustness validation

## Regulatory Compliance

| Regulation | Status | Implementation |
|------------|--------|---------------|
| PCI-DSS | ✅ Compliant | No raw card data stored |
| RBI Framework | ✅ Aligned | 4-tier risk scoring |
| GDPR Art. 22 | ✅ Enabled | LIME explanations |
| EU AI Act | ✅ Ready | Conformal prediction |

---

*This model card follows the format proposed by Mitchell et al. (2019) — "Model Cards for Model Reporting."*
