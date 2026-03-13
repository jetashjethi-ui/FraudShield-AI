# FraudShield AI — Summary Report
## FrostHack · March 2026 · Financial Services Track

---

## 1. Problem Statement

Financial fraud costs the global economy over $32.4 billion annually (2024), with approximately 68% of fraudulent transactions going undetected by traditional rule-based systems. These legacy systems suffer from three fundamental flaws: excessive false positives (80%+), zero explainability for flagged transactions, and a one-size-fits-all response that blocks legitimate customers alongside actual fraud.

FraudShield AI addresses all three of these challenges by building a complete end-to-end machine learning pipeline that not only detects fraud more accurately, but also explains *why* each transaction was flagged and responds proportionally to the assessed risk level.

---

## 2. Dataset

**Source:** IEEE-CIS Fraud Detection Dataset (Kaggle)  
**Size:** 590,540 transactions with 434 raw features  
**Fraud Rate:** 3.5% (20,663 fraudulent transactions)  
**Split:** 80% train (472,432) / 20% test (118,108) — stratified  

The dataset includes transaction details (amount, product code, time delta), card metadata (card type, issuing bank), device/browser information, email domains, and 339 anonymous engineered features (V1-V339) from Vesta Corporation's real-world payment processing system.

---

## 3. Methodology

### 3.1 Data Ingestion & Preprocessing
- Merged transaction data with identity data on TransactionID (144,233 identity matches from 590,540 transactions)
- Handled missing values: median imputation for numerical, "MISSING" category for categorical
- Replaced infinite values with zeros
- Label-encoded remaining categorical features

### 3.2 Feature Engineering (15 Layers)

We engineered 15 specialized detection layers, each targeting a distinct fraud signal:

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

### 3.3 Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Over-sampling) to oversample fraud class to approximately 15%
- Combined with model-level balancing: XGBoost `scale_pos_weight` and CatBoost `auto_class_weights`

### 3.4 Model Training

Five diverse models were trained to create a robust ensemble:

1. **XGBoost** (500 trees, depth 8, lr=0.05) — Gradient boosted trees optimized for structured data
2. **CatBoost** (500 iterations, depth 8, lr=0.05) — Handles categorical features natively
3. **Random Forest** (200 trees, depth 10) — Bagging ensemble for stability
4. **MLP Neural Network** (128-64-32 architecture, ReLU, Adam) — Deep learning for non-linear patterns
5. **Isolation Forest** (100 estimators, contamination=3.5%) — Unsupervised anomaly detection

### 3.5 Ensemble & Optimal Threshold
- Weighted voting ensemble: XGBoost (30%) + CatBoost (25%) + RF (15%) + MLP (15%) + IsoForest (15%)
- Optimal threshold selection: algorithmically scanned 40 thresholds (0.30–0.70) to maximize F1-score

### 3.6 Risk Scoring & Adaptive Authentication
Each transaction receives:
- A continuous risk score (0-100)
- A 4-tier authentication response: GREEN (approve), YELLOW (PIN), ORANGE (biometric), RED (block)
- Human-readable XAI explanations derived from feature importance analysis

---

## 4. Results

### Model Performance (Test Set: 118,108 transactions)

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.9541 | 0.6665 | 0.6643 | 0.6688 |
| CatBoost | 0.9409 | 0.6225 | 0.5995 | 0.6475 |
| MLP Neural Net | 0.9076 | 0.5753 | 0.7311 | 0.4742 |
| Random Forest | 0.8815 | 0.3945 | 0.2932 | 0.6030 |
| Isolation Forest | 0.7023 | 0.1044 | 0.1744 | 0.0745 |
| **Ensemble (5-Model)** | **0.9374** | **0.6729** | **0.7588** | **0.6044** |

### Risk Distribution

| Tier | Count | Percentage |
|------|-------|------------|
| GREEN (Auto-Approve) | 85,802 | 72.6% |
| YELLOW (PIN Verify) | 28,629 | 24.2% |
| ORANGE (Biometric) | 2,957 | 2.5% |
| RED (Block) | 720 | 0.6% |

### Key Achievements
- **0.7% false alarm rate** — only 794 out of 113,975 legitimate transactions incorrectly flagged
- **60.4% fraud detection** — catches 2,498 out of 4,133 actual fraud cases in real-time
- **Optimal threshold tuning** improved recall by 7.1% compared to the default 0.5 cutoff
- **75.9% precision** — 3 out of every 4 flags are genuine fraud

---

## 5. Actionable Insights & Business Impact

### Projected Annual Impact (Mid-Size Bank, 10M transactions/year)
- **$2.1M savings** from reduced false positive manual review (140K fewer cases × $15/review)
- **$9.5M fraud prevented** (60.4% detection × $450 avg fraud × 35K annual attempts)
- **72.6% of customers** experience zero friction (auto-approved)

### Key Decision-Making Recommendations
1. Deploy the ensemble model as a real-time API behind the bank's payment gateway
2. Route RED-tier transactions to the fraud investigation team immediately
3. Use XAI explanations to train junior analysts on fraud pattern recognition
4. Monitor model drift monthly and retrain quarterly with fresh transaction data

---

## 6. Tools & Technologies

- **Platform:** Python 3.10+ with Altair RapidMiner AI Studio integration
- **ML Libraries:** XGBoost, CatBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Streamlit
- **Dashboard:** Streamlit (interactive real-time fraud monitoring)
- **Version Control:** Git, GitHub

---

## 7. Future Enhancements

- **Phase 1 (0-3 months):** Deploy as FastAPI microservice with Redis caching; add LightGBM as 6th model
- **Phase 2 (3-6 months):** Graph Neural Networks for deep fraud ring detection; stacking meta-learner
- **Phase 3 (6-12 months):** Online learning for hourly model updates; federated learning across banks

---

*Team: Jetash Jethi | FrostHack March 2026 | github.com/jetashjethi-ui/FraudShield-AI*
