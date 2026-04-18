# FraudShield AI — Complete Technical Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data Loading (data_loader.py)](#3-data-loading)
4. [Feature Engineering (feature_engine.py)](#4-feature-engineering)
5. [Graph Analysis (graph_engine.py)](#5-graph-analysis)
6. [Hyperparameter Tuning (tuner.py)](#6-hyperparameter-tuning)
7. [Model Training & Stacking (models.py)](#7-model-training--stacking)
8. [Deep Learning Autoencoder (autoencoder.py)](#8-deep-learning-autoencoder)
9. [Risk Scoring (risk_scorer.py)](#9-risk-scoring)
10. [SHAP Explainability (shap_engine.py)](#10-shap-explainability)
11. [Adversarial Robustness (adversarial.py)](#11-adversarial-robustness-testing)
12. [Visualization (visualizer.py)](#12-visualization)
13. [PDF Report (report_generator.py)](#13-pdf-report-generation)
14. [Dashboard (dashboard.py)](#14-dashboard)
15. [REST API (api.py)](#15-rest-api)
16. [Results Summary](#16-results-summary)

---

## 1. System Overview

FraudShield AI is a production-grade fraud detection system built on the IEEE-CIS Fraud Detection dataset (590,540 transactions). It processes raw transaction data through a multi-step pipeline that includes 25 feature engineering layers, 8-model dual stacking ensemble, conformal prediction, cost-sensitive threshold optimization, and automated reporting.

- **Supervised learning**: 8 models trained on labeled fraud/non-fraud data, combined through a dual stacking ensemble
- **Unsupervised learning**: A deep autoencoder + Isolation Forest that learn normal transaction patterns and flag anomalies
- **Uncertainty quantification**: Conformal prediction with 95% coverage guarantee

Each transaction receives a risk score (0-100) and a recommended response (approve, PIN, biometric, or block).

---

## 2. Pipeline Architecture

The main pipeline (`main.py`) runs 11 sequential steps:

```
Step 1: Data Loading
    └─ Merge transaction + identity tables
Step 2: Feature Engineering
    └─ 25 detection layers + graph features + target encoding + PCA
Step 3: Data Preparation
    └─ Train/test split (80/20, stratified) + Adversarial Validation + SMOTE
Step 3.5: Optuna Tuning
    └─ 150-trial Bayesian optimization for XGBoost, LightGBM, CatBoost
Step 4: Model Training
    └─ 7-fold OOF dual stacking with 8 base models + XGBoost meta-learner
Step 4.5: Autoencoder
    └─ Unsupervised anomaly detection on normal transactions
Step 4.6: Conformal Prediction
    └─ Isotonic calibration + conformal sets (95% coverage)
Step 4.7: Threshold Optimization
    └─ Cost-sensitive threshold via business cost matrix (FN=₹850, FP=₹25)
Step 5: Risk Scoring
    └─ 0-100 scores + 4-tier adaptive authentication
Step 6: Visualizations
    └─ 16 diagnostic charts
Step 7: SHAP Explanations
    └─ Global + local feature importance
Step 8: Adversarial Testing
    └─ 5 attack scenario simulations
Step 9: PDF Report
    └─ Auto-generated multi-page report
```

---

## 3. Data Loading

**File:** `src/data_loader.py`

### What it does
Loads and merges two CSV files from the Kaggle IEEE-CIS dataset:
- `train_transaction.csv` — 590,540 rows, 394 columns (amounts, timestamps, card info, V-features)
- `train_identity.csv` — 144,233 rows, 41 columns (device info, browser, OS)

### How it works
1. Reads both CSVs with pandas
2. Left-joins on `TransactionID` (not all transactions have identity data)
3. Returns the merged DataFrame

### Key details
- The join adds device/browser features to ~24% of transactions
- Transactions without identity data get NaN for those columns (handled later in feature engineering)

---

## 4. Feature Engineering

**File:** `src/feature_engine.py`

### What it does
Transforms 434 raw columns into a feature set with 17 specialized detection layers. Each layer targets a specific fraud pattern.

### The 17 Layers

**Layer 1 — Amount Intelligence**
- `amount_zscore`: How many standard deviations from the mean this transaction's amount is
- `amount_log`: Log-transformed amount (reduces skew from large values)
- `amount_bin`: Percentile bucket (0-4) for the transaction amount
- *Why*: Fraudulent transactions often have unusual amounts

**Layer 2 — Behavioral DNA**
- Groups transactions by card number, computes per-card statistics:
  - `card_mean_amount`: Average spending for this card
  - `card_std_amount`: Spending volatility
  - `card_transaction_count`: How often this card is used
  - `card_amount_deviation`: How far this transaction deviates from the card's normal spending
- *Why*: Fraud shows up as deviations from a user's normal behavior

**Layer 3 — SIM Swap Detection**
- `device_change_count`: How many different devices this card has been used on
- *Why*: Stolen cards are often used from new devices

**Layer 4 — Time Baselines**
- `hour_of_day`: Extracted from TransactionDT
- `day_of_week`: Day of the week
- `is_night`: Flag for transactions between 11 PM and 6 AM
- `is_weekend`: Saturday/Sunday flag
- *Why*: Fraud is more common at night and on weekends

**Layer 5 — Adaptive Auth Score**
- `pre_model_risk`: Composite score combining amount z-score, night flag, device changes, and new account flag
- *Why*: Pre-screening score that the models can use as a meta-feature

**Layer 6 — Merchant Risk**
- `merchant_fraud_rate`: Historical fraud rate for each product category
- *Why*: Some merchant types are inherently higher risk

**Layer 7 — Mule Network**
- `shared_device_count`: Number of different cards used from the same device
- *Why*: Mule accounts share devices across multiple stolen cards

**Layer 8 — Dormant Hijacking**
- `time_since_last_txn`: Time gap since the card's previous transaction
- *Why*: Dormant accounts that suddenly become active are suspicious

**Layer 9 — Round Amount Detection**
- `is_round_amount`: Flag for perfectly round dollar values ($100, $500, etc.)
- `is_round_100`: Flag for amounts that are multiples of 100
- *Why*: Fraudsters often use round numbers

**Layer 10 — Category Mismatch**
- `card_product_mismatch`: Flag when card type doesn't match the product category
- *Why*: Inconsistencies suggest the card is being used outside its normal context

**Layer 11 — New Account Risk**
- `is_first_transaction`: Flag for cards appearing for the first time
- *Why*: New/first-time cards have no history to validate against

**Layer 12 — Velocity Features**
- `txn_velocity_card`: Transaction frequency per card in recent time windows
- `txn_amount_velocity`: Amount velocity (rolling sum)
- *Why*: Rapid successive transactions indicate automated fraud

**Layer 13 — Email Domain Risk**
- `email_risk_score`: Risk score based on email provider (0 = Gmail/Yahoo, 1 = disposable/anonymous)
- *Why*: Fraudsters use disposable email services

**Layer 14 — Graph-Lite Fraud Ring**
- `shared_email_device_count`: Number of unique cards sharing the same email domain AND device
- *Why*: Fraud rings reuse infrastructure across accounts

**Layer 15 — Synthetic Device**
- `device_anomaly_score`: Anomaly score based on missing device info combinations
- *Why*: Bots and automated fraud tools leave device fingerprint gaps

**Layer 16 — Graph Analysis** (via `graph_engine.py`)
- Full network features (PageRank, betweenness centrality, community detection)
- See Section 5 for details

**Layer 17 — Target Encoding**
- Smoothed target encoding for high-cardinality categorical columns (card1, card2, addr1, etc.)
- Formula: `encoding = (count * mean_target + global_prior * smoothing) / (count + smoothing)`
- *Why*: Converts categorical variables to numeric using their fraud correlation, with smoothing to prevent overfitting on rare categories

### Missing Value Handling
- Numerical: Filled with median
- Categorical: Filled with "MISSING" string, then label-encoded
- Infinite values: Replaced with 0

---

## 5. Graph Analysis

**File:** `src/graph_engine.py`

### What it does
Builds a transaction relationship graph and uses network science to detect fraud rings — groups of accounts that share infrastructure (devices, addresses, email domains).

### Algorithm

1. **Graph Construction** (NetworkX)
   - Each unique card is a node
   - Edges connect cards that share: same device type, same address, or same rare email domain
   - Edge weights reflect the number of shared attributes

2. **Community Detection** (Louvain Algorithm)
   - Groups tightly-connected nodes into communities
   - Fraud rings show up as communities with unusually high fraud rates
   - Resolution parameter controls community granularity

3. **Feature Extraction** (11 features per transaction)
   - `pagerank`: Importance of this node in the graph (highly connected = suspicious)
   - `betweenness_centrality`: How often this node lies on shortest paths between others
   - `degree`: Number of connections
   - `community_id`: Which community this card belongs to
   - `community_fraud_rate`: Fraud rate within this card's community
   - `community_size`: How many cards are in the same community
   - `fraud_neighbor_ratio`: Proportion of direct neighbors that are fraudulent
   - `avg_neighbor_degree`: Average connectivity of neighbors
   - `clustering_coefficient`: How interconnected the card's neighbors are
   - `is_bridge`: Whether removing this node would disconnect parts of the graph
   - `hub_score`: HITS hub score

### Why this matters
Traditional ML treats each transaction independently. Graph features capture collective fraud behavior — a card might look normal individually, but if it shares a device with 5 other cards that are all fraudulent, that's a strong signal.

---

## 6. Hyperparameter Tuning

**File:** `src/tuner.py`

### What it does
Uses Optuna (Bayesian optimization) to automatically find the best hyperparameters for XGBoost and LightGBM.

### Algorithm
1. Defines a search space for each model:
   - `n_estimators`: 100-500
   - `max_depth`: 3-10
   - `learning_rate`: 0.01-0.1
   - `subsample`: 0.6-1.0
   - `colsample_bytree`: 0.6-1.0
   - Plus model-specific params (gamma, reg_alpha, reg_lambda, num_leaves, etc.)

2. Runs 30 trials using Tree-structured Parzen Estimator (TPE):
   - Each trial picks a combination of parameters
   - Evaluates using 3-fold stratified cross-validation
   - Objective: maximize AUC-ROC
   - TPE learns from previous trials to explore promising regions

3. Saves the best parameters to `optuna_best_params.json`
4. These parameters are passed to the model training step

### Why Optuna over Grid Search
- Grid search checks every combination (exponential time)
- Optuna uses TPE to focus on promising areas (30 trials ≈ equivalent coverage to 500+ grid points)

---

## 7. Model Training & Stacking

**File:** `src/models.py`

### What it does
Trains 6 base models using 5-fold Out-of-Fold (OOF) predictions, then stacks the top 4 into a meta-learner ensemble.

### SMOTE (Pre-training)
- The training set has ~3.5% fraud (imbalanced)
- SMOTE generates synthetic fraud samples to reach ~15% fraud ratio
- Only applied to training data (test set stays untouched)

### The 6 Base Models

| Model | Type | Key Params |
|-------|------|-----------|
| XGBoost | Gradient boosted trees | Optuna-tuned (n_est, depth, lr, etc.) |
| LightGBM | Fast gradient boosting | Optuna-tuned |
| CatBoost | Ordered boosting | 400 iterations, depth 6, auto class weights |
| Random Forest | Bagging ensemble | 200 trees, max depth 10 |
| MLP | Neural network | 128-64-32 hidden layers, ReLU, Adam optimizer |
| Isolation Forest | Anomaly detection | 100 estimators, 3.5% contamination |

### 5-Fold OOF Stacking — How It Works

The stacking process avoids overfitting by never letting the meta-learner see predictions made on the same data that trained the base models:

```
Training Data (100%)
├── Fold 1: Train on folds 2-5, predict fold 1 → OOF predictions for fold 1
├── Fold 2: Train on folds 1,3-5, predict fold 2 → OOF predictions for fold 2
├── Fold 3: Train on folds 1-2,4-5, predict fold 3 → OOF predictions for fold 3
├── Fold 4: Train on folds 1-3,5, predict fold 4 → OOF predictions for fold 4
└── Fold 5: Train on folds 1-4, predict fold 5 → OOF predictions for fold 5

Combined OOF predictions = one prediction per training sample (no data leakage)
```

### Meta-Learner (Stacking)
- Only the **top 4 models** are stacked: XGBoost, LightGBM, CatBoost, MLP
- Random Forest and Isolation Forest are excluded (their weaker predictions add noise)
- The OOF predictions from the top 4 become features for a **Logistic Regression** meta-learner
- The meta-learner learns the optimal weight for each model's predictions
- Final test predictions = meta-learner applied to the base models' test predictions

### Why not include all 6 in stacking?
- Isolation Forest has AUC ~0.71 (adds noise, not signal)
- Random Forest has the weakest supervised AUC
- Including them dragged ensemble AUC down by ~0.003
- All 6 models are still trained and reported for comparison

---

## 8. Deep Learning Autoencoder

**File:** `src/autoencoder.py`

### What it does
A fundamentally different approach from the supervised models. The autoencoder learns to reconstruct normal transactions, then uses reconstruction error to detect anomalies.

### Architecture
```
Input (N features) → Encoder (64 neurons) → Bottleneck (16 neurons) → Decoder (64 neurons) → Output (N features)
```

### How it works

1. **Training** (on normal transactions only):
   - Filters out all fraud samples from training data
   - Scales features with StandardScaler
   - The network learns to compress N features into 16 dimensions and reconstruct them
   - The bottleneck forces it to learn the essential patterns of normal transactions
   - Uses Adam optimizer, ReLU activation, early stopping

2. **Detection**:
   - Pass any transaction through the trained autoencoder
   - Compute Mean Squared Error between input and output (reconstruction error)
   - Normal transactions reconstruct well → low error
   - Fraudulent transactions don't fit learned patterns → high error

3. **Scoring**:
   - Reconstruction errors are normalized to 0-1 range
   - Threshold set at 95th percentile of training errors
   - Anything above threshold is flagged as anomalous

### Results
- AUC: 0.696 (lower than supervised models, which is expected)
- Error ratio: 3.4x (fraud transactions have 3.4x higher reconstruction error than normal ones)
- The autoencoder's value is in catching novel fraud patterns that supervised models might miss because they were never in the training labels

---

## 9. Risk Scoring

**File:** `src/risk_scorer.py`

### What it does
Converts raw model predictions into actionable risk scores and assigns each transaction to a response tier.

### Algorithm

1. **Score Calculation** (0-100):
   - Starts with the ensemble's predicted probability (0-1)
   - Multiplies by feature-specific risk multipliers:
     - High amount → boost score
     - Night transaction → boost score
     - Round amount → boost score
     - Missing email → boost score
     - High velocity → boost score
   - Clips to 0-100 range

2. **4-Tier Response**:
   - **GREEN (0-30)**: Auto-approve. Low risk, no friction for the customer.
   - **YELLOW (31-50)**: Request PIN verification.
   - **ORANGE (51-70)**: Request biometric authentication (fingerprint/face).
   - **RED (71-100)**: Block the transaction and alert the fraud team.

3. **Explanation Generation**:
   - For each flagged transaction, generates human-readable reasons
   - Example: "Amount $4,500 is 3.2 standard deviations above card average"
   - Based on which features contributed most to the high score

---

## 10. SHAP Explainability

**File:** `src/shap_engine.py`

### What it does
Uses SHAP (SHapley Additive exPlanations) to explain exactly which features drove each prediction.

### Algorithm
SHAP is based on Shapley values from cooperative game theory:
- Each feature is a "player" in a "game" (the prediction)
- The Shapley value is the average marginal contribution of each feature across all possible feature combinations
- For tree models, TreeExplainer computes exact Shapley values efficiently

### Outputs
1. **Summary Plot** (beeswarm): Shows all features, their SHAP values, and feature values. Red dots on the right = high feature value pushes toward fraud.
2. **Bar Plot**: Mean absolute SHAP value per feature (global importance ranking)
3. **Waterfall Plots** (3): Individual transaction explanations showing exactly how each feature pushed the prediction up or down

### Why SHAP over basic feature importance
- Feature importance says "amount is important"
- SHAP says "for THIS transaction, the amount of $4,500 increased the fraud probability by 0.12"
- This transaction-level explainability is critical for regulatory compliance and analyst trust

---

## 11. Adversarial Robustness Testing

**File:** `src/adversarial.py`

### What it does
Tests whether the model can be fooled by 5 realistic fraud evasion strategies.

### The 5 Tests

**Test 1 — Amount Splitting**
- Question: If a fraudster splits a $5,000 purchase into five $1,000 transactions, can we still catch them?
- Method: Filter to fraud transactions with below-median amounts. Check the model's detection rate on these.
- Pass threshold: >40% catch rate

**Test 2 — Time Evasion**
- Question: If a fraudster only operates during business hours (to blend in), can we still catch them?
- Method: Filter to fraud transactions between 9 AM and 5 PM. Check detection rate.
- Pass threshold: >40% catch rate

**Test 3 — Device Spoofing**
- Question: If a fraudster uses a normal-looking device (Windows, iPhone), can we still catch them?
- Method: Filter to fraud transactions from common device types. Check detection rate.
- Pass threshold: >35% catch rate

**Test 4 — Threshold Sensitivity**
- Question: Is the model's performance stable, or does it collapse at different decision thresholds?
- Method: Evaluate AUC at thresholds 0.3, 0.4, 0.5, 0.6, 0.7. Compute stability (min AUC / max AUC).
- Pass threshold: stability ratio >0.7

**Test 5 — Feature Perturbation**
- Question: Do small random changes to features flip the model's predictions?
- Method: Add 5% Gaussian noise to test features. Check what percentage of predictions stay confident (>0.6 or <0.4).
- Pass threshold: >60% remain confident

### Results: 5/5 passed (100% robustness score)

---

## 12. Visualization

**File:** `src/visualizer.py`

### What it does
Generates 16 diagnostic charts saved as PNGs:

| Chart | What it shows |
|-------|--------------|
| roc_curves.png | ROC curves for all models + ensemble |
| confusion_matrices.png | True/false positive/negative counts |
| metrics_comparison.png | AUC, F1, precision, recall bar chart |
| feature_importance.png | Top 20 features by XGBoost importance |
| amount_distribution.png | Transaction amount histogram (fraud vs normal) |
| fraud_by_hour.png | Fraud rate by hour of day |
| fraud_by_product.png | Fraud rate by product category |
| risk_distribution.png | Risk score distribution across all transactions |
| risk_pie_chart.png | Proportion of GREEN/YELLOW/ORANGE/RED |
| graph_analysis.png | Network graph visualization |
| sample_explanations.png | Table of flagged transactions with reasons |
| shap_summary.png | SHAP beeswarm plot |
| shap_bar.png | SHAP mean importance bar chart |
| shap_waterfall_1/2/3.png | Individual transaction SHAP explanations |

---

## 13. PDF Report Generation

**File:** `src/report_generator.py`

### What it does
Automatically generates a multi-page PDF report after the pipeline completes.

### Pages
1. **Title page**: Project name, date, description
2. **Model performance**: Table with all model metrics + ROC curves image
3. **Feature importance**: XGBoost feature importance chart
4. **SHAP bar plot**: Mean SHAP values
5. **SHAP beeswarm**: Full SHAP summary
6. **SHAP waterfalls** (3 pages): Individual transaction explanations
7. **Graph analysis**: Network visualization
8. **Adversarial results**: Pass/fail for each test scenario
9. **ROI analysis**: Dollar savings calculation
10. **Additional charts**: Confusion matrices, metrics comparison, risk distribution

### Implementation
Uses fpdf2 library with custom header/footer, styled tables, and embedded PNG images.

---

## 14. Dashboard

**File:** `dashboard.py`

### What it does
A 17-page interactive Streamlit dashboard for exploring all results.

### Pages

1. **Dashboard** — Overview with key metrics, architecture visualization, production readiness cards
2. **Live Detector** — Input transaction details manually and get a real-time risk score with explanations
3. **Models** — Interactive Plotly radar chart, ROC curves, and confusion matrices
4. **Analytics** — SHAP summary, fraud patterns by hour, product category, and amount distribution
5. **Flagged** — Table of the highest-risk flagged transactions with explanations
6. **ROI Calculator** — Interactive sliders to compute cost-benefit analysis for different bank sizes
7. **Robustness** — Adversarial test results with pass/fail indicators
8. **Simulator** — Generate random transactions with configurable fraud probability
9. **Autoencoder** — Architecture visualization, reconstruction error comparison
10. **Live Stream** — Real-time streaming transaction simulator with animated feed
11. **Explainer** — LIME-style plain-English risk factor explanations
12. **Fairness Audit** — Bias analysis across card types and product categories
13. **Counterfactual** — "What-if" analysis — what would need to change to flip a decision
14. **Drift Monitor** — AUC health monitoring with retraining alerts
15. **Fraud Network** — Interactive Plotly force-directed graph of fraud ring connections
16. **Risk Heatmap** — Time × amount fraud density heatmap
17. **Threshold** — Cost-sensitive threshold optimizer with business cost curve

### Design
- Dark theme with glassmorphism effects
- Custom CSS for metric cards, glass cards, and dividers
- Responsive layout using Streamlit columns

---

## 15. REST API

**File:** `api.py`

### What it does
A FastAPI-based REST API for real-time transaction scoring.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Product landing page |
| `/health` | GET | Health check + model AUC |
| `/score` | POST | Score a single transaction |
| `/predict` | POST | Score for mobile app |
| `/batch` | POST | Score multiple transactions |
| `/roi` | POST | Calculate ROI for given parameters |
| `/model-info` | GET | Return model metadata |
| `/mobile/` | GET | Mobile PWA app |
| `/docs` | GET | Swagger API documentation |

### `/score` Request Example
```json
{
  "TransactionAmt": 500.00,
  "ProductCD": "W",
  "card4": "visa",
  "P_emaildomain": "gmail.com",
  "DeviceType": "mobile",
  "TransactionDT": 86400
}
```

### `/score` Response Example
```json
{
  "risk_score": 72,
  "risk_tier": "RED",
  "recommended_action": "BLOCK",
  "model_contributions": {
    "xgboost": 0.82,
    "lightgbm": 0.71,
    "catboost": 0.68,
    "mlp": 0.65
  },
  "explanations": [
    "High transaction amount (z-score: 2.3)",
    "Night-time transaction"
  ]
}
```

---

## 16. Results Summary

### Final Model Performance

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **Stacking Ensemble** | **0.9558** | **0.6863** | **0.7631** | **0.6235** |
| CatBoost | 0.9517 | 0.6173 | 0.5631 | 0.6830 |
| LightGBM | 0.9481 | 0.5835 | 0.5091 | 0.6833 |
| XGBoost | 0.9479 | 0.6284 | 0.8661 | 0.4931 |
| MLP | 0.9470 | 0.6416 | 0.8019 | 0.5347 |
| Random Forest | 0.9235 | 0.5073 | 0.4131 | 0.6571 |
| Isolation Forest | 0.7135 | 0.1272 | 0.1758 | 0.0997 |
| Deep Autoencoder | 0.6958 | — | — | — |

### Key Numbers
- Ensemble beats every individual model
- 62.4% of all fraud caught
- 76.3% precision (3 out of 4 flags are real fraud)
- 99.3% of legitimate transactions correctly approved
- Adversarial robustness: 100% (5/5 tests passed)
- Autoencoder error ratio: 3.4x (fraud reconstruction error vs normal)

### Output Files
```
outputs/
├── FraudShield_AI_Report.pdf          # Auto-generated report
├── results/
│   ├── model_metrics.json             # All model AUC/F1/precision/recall
│   ├── autoencoder_results.json       # Autoencoder metrics
│   ├── adversarial_report.json        # 5 attack test results
│   ├── optuna_best_params.json        # Tuned hyperparameters
│   ├── sample_flagged_transactions.csv # High-risk transactions
│   └── scored_transactions.csv        # All 118K test transactions scored
└── visualizations/
    └── (16 PNG charts)
```

---

*Jetash Jethi · FrostHack April 2026*
