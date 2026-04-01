# FraudShield AI

Fraud detection system built on the IEEE-CIS dataset for FrostHack (March 2026).

## What it does

Processes 590K+ transactions through a pipeline that:
- Engineers 17 feature layers (behavioral, temporal, network-based)
- Trains 7 ML models and stacks the best 4 into an ensemble
- Includes a deep autoencoder for unsupervised anomaly detection
- Assigns risk scores (0-100) with adaptive authentication tiers

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

Adversarial robustness: 5/5 tests passed. Autoencoder error ratio: 3.4x (fraud vs normal).

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This runs the full pipeline: feature engineering, Optuna tuning (30 trials), model training, autoencoder, SHAP, adversarial tests, and PDF report generation.

Dashboard:
```bash
streamlit run dashboard.py
```

API:
```bash
uvicorn api:app --reload
```

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle. Put `train_transaction.csv` and `train_identity.csv` in `data/`.

## Project structure

```
main.py               # Pipeline entry point
dashboard.py           # 9-page Streamlit dashboard
api.py                 # FastAPI scoring endpoint
src/
  data_loader.py       # Load and merge datasets
  feature_engine.py    # 17 feature layers
  models.py            # Training + 5-fold OOF stacking
  risk_scorer.py       # Risk scores and tier assignment
  visualizer.py        # Chart generation
  shap_engine.py       # SHAP explanations
  graph_engine.py      # NetworkX + Louvain
  tuner.py             # Optuna optimization
  adversarial.py       # Robustness tests
  autoencoder.py       # Anomaly detection
  report_generator.py  # PDF report
outputs/
  results/             # Metrics, flagged transactions
  visualizations/      # 16 plots
```

## Pipeline overview

1. Load and merge transaction + identity data
2. Engineer 17 feature layers (amount stats, behavioral profiles, velocity, graph features, target encoding, etc.)
3. Train/test split (80/20 stratified) + SMOTE
4. Optuna hyperparameter tuning (30 trials, XGBoost + LightGBM)
5. Train 7 models, stack top 4 with logistic regression meta-learner
6. Run autoencoder on normal transactions to detect anomalies
7. Score transactions 0-100, assign GREEN/YELLOW/ORANGE/RED tiers
8. Generate 16 visualizations + SHAP plots
9. Run 5 adversarial attack simulations
10. Generate PDF report

## Dashboard

9 pages: overview, live detector, model comparison, analytics, flagged transactions, ROI calculator, adversarial results, live simulator, autoencoder analysis.

## Tech

Python, XGBoost, LightGBM, CatBoost, scikit-learn, SHAP, NetworkX, Optuna, Streamlit, FastAPI, fpdf2

---
Jetash Jethi · FrostHack 2026
