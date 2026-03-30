# FraudShield AI — Summary Report
## FrostHack · March 2026 · Financial Services Track

---

## 1. Problem Statement

Financial fraud costs the global economy over $32.4 billion annually, with approximately 68% of fraudulent transactions going undetected by traditional rule-based systems. These legacy systems suffer from excessive false positives, zero explainability, and uniform responses that block legitimate customers alongside actual fraud.

FraudShield AI addresses these challenges with a multi-layered ML pipeline that detects fraud more accurately, explains each decision, and responds proportionally to the assessed risk level.

---

## 2. Dataset

**Source:** IEEE-CIS Fraud Detection Dataset (Kaggle)  
**Size:** 590,540 transactions with 434 raw features  
**Fraud Rate:** 3.5% (20,663 fraudulent transactions)  
**Split:** 80% train / 20% test — stratified

The dataset includes transaction details (amount, product code, time delta), card metadata, device/browser information, email domains, and 339 anonymous features from Vesta Corporation's payment processing system.

---

## 3. Methodology

### 3.1 Data Ingestion & Preprocessing
- Merged transaction data with identity data on TransactionID
- Handled missing values: median imputation for numerical, "MISSING" category for categorical
- Label-encoded categorical features with target encoding for high-cardinality columns

### 3.2 Feature Engineering (17 Layers)

| Layer | Name | Description |
|-------|------|-------------|
| L1 | Amount Intelligence | Z-scores, log transforms, percentile bins |
| L2 | Behavioral DNA | Per-card spending profiles (mean, std, frequency) |
| L3 | SIM Swap Detection | Device change frequency per card |
| L4 | Time Baselines | Hour-of-day, day-of-week, night transaction flags |
| L5 | Adaptive Auth Score | Composite pre-model risk score |
| L6 | Merchant Risk | Category-level fraud rate encoding |
| L7 | Mule Network | Shared device count across cards |
| L8 | Dormant Hijacking | Time since last transaction per card |
| L9 | Round Amount | Detection of suspiciously round dollar amounts |
| L10 | Category Mismatch | Card type vs product category inconsistency |
| L11 | New Account Risk | First-time card usage flags |
| L12 | Velocity Features | Transaction frequency in sliding time windows |
| L13 | Email Domain Risk | Disposable/risky email provider scoring |
| L14 | Graph-Lite Fraud Ring | Network features from shared devices/cards |
| L15 | Synthetic Device | Device fingerprint anomaly scoring |
| L16 | Graph Analysis | PageRank, betweenness centrality, Louvain communities |
| L17 | Target Encoding | Smoothed target encoding for high-cardinality categoricals |

### 3.3 Graph-Based Fraud Ring Detection
- Built a transaction relationship graph using NetworkX
- Nodes = users, edges = shared devices/addresses/email domains
- Applied Louvain community detection to identify fraud rings
- Extracted 11 graph features: PageRank, betweenness centrality, community fraud rate, fraud neighbor ratio, etc.

### 3.4 Class Imbalance Handling
- SMOTE oversampling of fraud class to approximately 15% ratio
- Model-level balancing: XGBoost `scale_pos_weight`, CatBoost `auto_class_weights`, LightGBM `is_unbalance`

### 3.5 Hyperparameter Optimization
- Optuna Bayesian optimization with 30 trials for XGBoost and LightGBM
- Objective: maximize AUC-ROC on stratified 3-fold cross-validation
- Best parameters saved and applied to final training

### 3.6 Model Training & Stacking

Seven models were trained to create a robust ensemble:

1. **XGBoost** — Gradient boosted trees, Optuna-tuned
2. **LightGBM** — Fast gradient boosting, Optuna-tuned
3. **CatBoost** (400 iterations, depth 6) — Native categorical handling
4. **Random Forest** (200 trees) — Bagging ensemble for stability
5. **MLP Neural Network** (128-64-32 architecture, ReLU) — Non-linear pattern detection
6. **Isolation Forest** (100 estimators, contamination=3.5%) — Unsupervised anomaly detection
7. **Deep Autoencoder** (64→16→64 bottleneck) — Reconstruction-error-based anomaly detection

**Stacking Ensemble:** Top 4 models (XGBoost, LightGBM, CatBoost, MLP) trained with 5-fold OOF predictions, stacked via Logistic Regression meta-learner. Weaker models (RF, IsoForest) excluded from stacking to avoid noise.

### 3.7 SHAP Explainability
- SHAP TreeExplainer applied to the XGBoost model
- Generated global feature importance (summary + bar plots) and local explanations (3 waterfall plots)
- Enables analysts to understand individual flagging decisions

### 3.8 Adversarial Robustness Testing
Five attack scenarios were simulated:
1. **Amount Splitting** — Can the model detect fraud when amounts are split?
2. **Time Evasion** — Does it work during business hours?
3. **Device Spoofing** — Can it catch fraud from normal-looking devices?
4. **Threshold Sensitivity** — Is performance stable across thresholds?
5. **Feature Perturbation** — Do small feature changes fool the model?

### 3.9 Risk Scoring & Adaptive Authentication
Each transaction receives:
- A continuous risk score (0-100)
- A 4-tier authentication response: GREEN (approve), YELLOW (PIN), ORANGE (biometric), RED (block)
- Feature-based explanations for each decision

---

## 4. Results

### Model Performance (Test Set: 118,108 transactions)

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.9479 | 0.6284 | 0.8661 | 0.4931 |
| LightGBM | 0.9481 | 0.5835 | 0.5091 | 0.6833 |
| CatBoost | 0.9517 | 0.6173 | 0.5631 | 0.6830 |
| MLP Neural Net | 0.9470 | 0.6416 | 0.8019 | 0.5347 |
| Random Forest | 0.9235 | 0.5073 | 0.4131 | 0.6571 |
| Isolation Forest | 0.7135 | 0.1272 | 0.1758 | 0.0997 |
| Deep Autoencoder | 0.6958 | — | — | — |
| **Stacking Ensemble** | **0.9558** | **0.6863** | **0.7631** | **0.6235** |

### Adversarial Robustness: 5/5 tests passed (100% score)
### Autoencoder: Fraud reconstruction error is 3.4x higher than normal transactions

---

## 5. Business Impact

### Projected Annual Savings (Mid-Size Bank, 50M transactions/year)

| Metric | Value |
|--------|-------|
| Fraud detection rate | 62.4% |
| Precision | 76.3% |
| Fraudulent transactions prevented | ~1.09M/year |
| Estimated fraud losses avoided | ~$927M |
| False positive reduction | 99.3% legitimate transactions unaffected |

---

## 6. Deliverables

| Deliverable | Description |
|-------------|-------------|
| `main.py` | Complete 11-step pipeline |
| `dashboard.py` | 9-page Streamlit dashboard |
| `api.py` | FastAPI real-time scoring endpoint |
| `outputs/FraudShield_AI_Report.pdf` | Auto-generated PDF report |
| `Summary_Report.md` | This document |
| `FraudShield_AI.rmzp` | RapidMiner process export |
| 16 visualizations | ROC, confusion matrix, SHAP, graphs, etc. |

---

## 7. Tools & Technologies

- **ML:** XGBoost, LightGBM, CatBoost, scikit-learn, Imbalanced-Learn
- **AutoML:** Optuna (Bayesian hyperparameter optimization)
- **Deep Learning:** Autoencoder (sklearn MLPRegressor)
- **Explainability:** SHAP (TreeExplainer)
- **Graph Analysis:** NetworkX, python-louvain
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Report:** fpdf2 (auto PDF generation)

---

*Jetash Jethi · FrostHack March 2026*
