# FraudShield AI

Fraud detection system for the IEEE-CIS dataset, built for FrostHack (March 2026).

## Overview

FraudShield AI processes 590K+ financial transactions through a multi-stage pipeline:
- 17 feature engineering layers extract behavioral, temporal, and network signals
- 7 models (including a deep autoencoder) are trained and evaluated
- A 4-model stacking ensemble with logistic regression meta-learner produces final predictions
- Each transaction receives a risk score (0-100) and a 4-tier response (approve/PIN/biometric/block)

## Results

| Model | AUC |
|-------|-----|
| XGBoost | 0.948 |
| CatBoost | 0.952 |
| LightGBM | 0.948 |
| MLP Neural Net | 0.947 |
| Random Forest | 0.924 |
| Isolation Forest | 0.713 |
| Deep Autoencoder | 0.696 |
| **Stacking Ensemble** | **0.956** |

- Adversarial robustness: 5/5 tests passed (100%)
- Autoencoder fraud/normal error ratio: 3.4x

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Run the pipeline
```bash
python main.py
```
This trains all models, runs Optuna tuning (30 trials), generates SHAP explanations, runs adversarial tests, and produces a PDF report.

### Launch the dashboard
```bash
streamlit run dashboard.py
```
Opens a 9-page interactive dashboard at `http://localhost:8501`.

### Start the API
```bash
uvicorn api:app --reload
```
REST API at `http://localhost:8000` with endpoints: `/score`, `/batch`, `/health`, `/roi`.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle. Place `train_transaction.csv` and `train_identity.csv` in the `data/` folder.

## Project Structure

```
├── main.py                  # Full pipeline (Steps 1-9)
├── dashboard.py             # Streamlit dashboard (9 pages)
├── api.py                   # FastAPI real-time scoring
├── src/
│   ├── data_loader.py       # Dataset loading and merging
│   ├── feature_engine.py    # 17 feature engineering layers
│   ├── models.py            # Model training, 5-fold OOF stacking
│   ├── risk_scorer.py       # Risk scoring and tier assignment
│   ├── visualizer.py        # Chart generation (16 plots)
│   ├── shap_engine.py       # SHAP explainability
│   ├── graph_engine.py      # NetworkX graph analysis + Louvain
│   ├── tuner.py             # Optuna hyperparameter optimization
│   ├── adversarial.py       # Adversarial robustness testing
│   ├── autoencoder.py       # Deep learning anomaly detector
│   └── report_generator.py  # Auto PDF report generation
├── data/                    # Dataset files (not in repo)
├── outputs/
│   ├── results/             # Metrics, reports, scored transactions
│   └── visualizations/      # All generated plots
├── requirements.txt
└── Summary_Report.md
```

## Pipeline Steps

1. **Data Loading** — Merge transaction + identity tables
2. **Feature Engineering** — 17 detection layers + graph features + target encoding
3. **Data Preparation** — Train/test split, SMOTE oversampling
4. **Optuna Tuning** — 30-trial Bayesian optimization for XGBoost + LightGBM
5. **Model Training** — 5-fold OOF stacking with 4-model meta-learner
6. **Autoencoder** — Unsupervised anomaly detection (bottleneck architecture)
7. **Risk Scoring** — 0-100 score with 4-tier adaptive authentication
8. **Visualizations** — 16 diagnostic plots
9. **SHAP** — Global + local explainability (summary, bar, waterfall)
10. **Adversarial Testing** — 5 attack scenarios
11. **PDF Report** — Auto-generated multi-page report

## Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard | Overview metrics, risk distribution, key charts |
| Live Detector | Score individual transactions in real-time |
| Models | ROC curves, confusion matrices, per-model comparison |
| Analytics | Fraud patterns by hour, product, amount |
| Flagged | Table of high-risk flagged transactions |
| ROI Calculator | Cost-benefit analysis with adjustable parameters |
| Robustness | Adversarial test results |
| Simulator | Generate and score random transactions live |
| Autoencoder | Architecture visualization, reconstruction error analysis |

## Tech Stack

Python 3.10+, XGBoost, LightGBM, CatBoost, scikit-learn, SHAP, NetworkX, Optuna, Streamlit, FastAPI, fpdf2

---

*Jetash Jethi · FrostHack March 2026*
