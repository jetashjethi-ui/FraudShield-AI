# FraudShield AI - Summary Report
**FrostHack · March 2026 · Financial Services Track**

---

## Problem

Credit card fraud costs over $32 billion a year globally and most of it goes undetected. Traditional rule-based systems have two big problems: they flag too many legitimate transactions (high false positives) and they can't explain why something was flagged. There's no way for an analyst to review the decision.

I wanted to build something that detects fraud accurately, explains every decision, and responds differently based on how suspicious a transaction actually is.

---

## Dataset

IEEE-CIS Fraud Detection dataset from Kaggle. 590,540 transactions, 434 features, 3.5% fraud rate. I merged the transaction table with the identity table on TransactionID and split 80/20 stratified.

---

## Approach

### Feature Engineering

I built 17 feature layers, each targeting a different fraud signal:

- **Amount features**: z-scores, log transforms, percentile bins
- **Behavioral profiles**: per-card averages, standard deviations, spending frequency
- **SIM swap signals**: counting how many devices each card has been used on
- **Time features**: hour of day, night flag, weekend flag
- **Merchant risk**: fraud rate per product category
- **Mule detection**: counting cards sharing the same device
- **Dormant accounts**: time since last transaction per card
- **Round amounts**: flagging suspiciously round values ($1000, $5000)
- **Velocity**: transaction frequency in recent time windows
- **Email risk**: scoring disposable/anonymous email providers
- **Graph features**: built a NetworkX graph, ran Louvain community detection, extracted PageRank, betweenness centrality, community fraud rate (11 features total)
- **Target encoding**: smoothed encoding for high-cardinality categoricals like card1, addr1

### Handling Imbalance

Used SMOTE to oversample fraud to ~15% ratio. Also used model-level balancing (scale_pos_weight for XGBoost, auto_class_weights for CatBoost, is_unbalance for LightGBM).

### Hyperparameter Tuning

Optuna with 30 Bayesian trials for XGBoost and LightGBM. Optimized for AUC-ROC on 3-fold stratified CV. Saved best params and applied them to final training.

### Models

Trained 7 models:
1. XGBoost (Optuna-tuned)
2. LightGBM (Optuna-tuned)
3. CatBoost (400 iterations, depth 6)
4. Random Forest (200 trees)
5. MLP Neural Net (128-64-32 hidden layers, ReLU)
6. Isolation Forest (unsupervised, 3.5% contamination)
7. Deep Autoencoder (64→16→64 bottleneck, trained on normal transactions only)

### Stacking

Took the top 4 (XGBoost, LightGBM, CatBoost, MLP) and generated 5-fold out-of-fold predictions. Fed those as features to a logistic regression meta-learner. Excluded RF and Isolation Forest from stacking because their weaker predictions were adding noise.

### Autoencoder

Trained a bottleneck autoencoder only on legitimate transactions. The idea is that it learns what "normal" looks like, and when a fraud transaction comes through, the reconstruction error is much higher. Got a 3.4x error ratio (fraud vs normal). AUC is 0.696 on its own, but it complements the supervised models by catching patterns that weren't in the labeled data.

### Adversarial Testing

Ran 5 tests to check if the model can be fooled:
- Amount splitting (small amounts): 65.7% catch rate
- Time evasion (business hours fraud): 60.5%
- Device spoofing (normal devices): 64.7%
- Threshold sensitivity: 99.2% stable
- Feature perturbation (random noise): 95.7% confident

All 5 passed.

---

## Results

| Model | AUC | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| XGBoost | 0.9479 | 0.6284 | 0.8661 | 0.4931 |
| LightGBM | 0.9481 | 0.5835 | 0.5091 | 0.6833 |
| CatBoost | 0.9517 | 0.6173 | 0.5631 | 0.6830 |
| MLP | 0.9470 | 0.6416 | 0.8019 | 0.5347 |
| Random Forest | 0.9235 | 0.5073 | 0.4131 | 0.6571 |
| Isolation Forest | 0.7135 | 0.1272 | 0.1758 | 0.0997 |
| Autoencoder | 0.6958 | - | - | - |
| **Ensemble** | **0.9558** | **0.6863** | **0.7631** | **0.6235** |

The ensemble catches 62.4% of all fraud with 76.3% precision. 99.3% of legitimate transactions go through without any friction.

---

## Business Impact

For a mid-size bank doing 50M transactions/year:
- ~1.09M fraudulent transactions prevented
- ~$927M in avoided losses
- 99.3% of legit customers unaffected

---

## Deliverables

- `main.py` - full pipeline (run this to reproduce everything)
- `dashboard.py` - 9-page Streamlit dashboard
- `api.py` - FastAPI endpoint for real-time scoring
- `outputs/FraudShield_AI_Report.pdf` - auto-generated PDF report
- `FraudShield_AI.rmzp` - RapidMiner process export
- 16 visualizations (ROC, confusion matrices, SHAP, graphs, etc.)

## Tools Used

XGBoost, LightGBM, CatBoost, scikit-learn, Imbalanced-Learn, Optuna, SHAP, NetworkX, python-louvain, Streamlit, FastAPI, fpdf2

---
Jetash Jethi · FrostHack 2026
