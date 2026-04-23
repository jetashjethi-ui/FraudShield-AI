# FraudShield AI — Executive Summary

## System at a Glance

| Component | Details |
|-----------|---------|
| **Dataset** | IEEE-CIS Fraud Detection (590,540 transactions, 434 features) |
| **Feature Engineering** | 25 detection layers (velocity, graph, behavioral, entropy, etc.) |
| **Models** | 8-model dual stacking ensemble (XGBoost, LightGBM, CatBoost, RF, MLP, IsoForest, TabNet, Autoencoder) |
| **Best Model AUC** | **0.9776** (XGBoost) |
| **Ensemble AUC** | **0.9763** (4-Model Stacking) |
| **Cross-Validation** | 7-Fold Stratified with Optuna HPO (150 trials) |
| **Uncertainty** | Conformal Prediction with 95% coverage guarantee |
| **Calibration** | Isotonic probability calibration |
| **Threshold** | Cost-optimized at 0.015 (FN cost=850, FP cost=25) with 89% recall |
| **Explainability** | SHAP + LIME + Counterfactual analysis |
| **Dashboard** | 11-page interactive Streamlit dashboard (deployed live) |
| **Mobile App** | Progressive Web App with WebSocket live feed |
| **API** | FastAPI REST API with Swagger + WebSocket `/ws/feed` |
| **Deployment** | Docker + Streamlit Cloud (live) |

---

## Model Performance (Real Pipeline Results)

| Model | AUC | F1 | Precision | Recall |
|-------|:---:|:--:|:---------:|:------:|
| **XGBoost** | **0.9776** | 0.747 | 0.900 | 0.639 |
| **Stacking Ensemble (4-Model)** | **0.9763** | 0.765 | 0.834 | 0.707 |
| CatBoost | 0.9721 | 0.659 | 0.581 | 0.762 |
| LightGBM | 0.9662 | 0.592 | 0.484 | 0.761 |
| MLP Neural Net | 0.9641 | 0.694 | 0.824 | 0.600 |
| Random Forest | 0.9553 | 0.561 | 0.451 | 0.744 |
| Isolation Forest | 0.7270 | 0.095 | 0.170 | 0.066 |
| Autoencoder | 0.7110 | — | — | — |

---

## Production Validation

| Check | Result |
|-------|:------:|
| **Adversarial Validation** | AUC = 0.5004 — no train/test leakage |
| **Adversarial Robustness** | 5/5 attack scenarios passed, AUC stays above 0.92 |
| **Cost-Optimized Threshold** | 0.015 threshold with annual savings of 3.55 Crore |
| **Conformal Coverage** | 95% prediction coverage achieved |

---

## 4-Tier Risk Routing

| Tier | Score Range | Action | Distribution |
|------|:-----------:|--------|:------------:|
| GREEN | 0–30 | Auto-approve | 55.2% |
| YELLOW | 31–50 | PIN verification | 41.5% |
| ORANGE | 51–70 | Biometric re-verification | 1.97% |
| RED | 71–100 | Block + fraud team alert | 1.38% |

---

## 25 Detection Layers

Amount Analysis, Behavioral DNA, SIM Swap Detection, Seasonal Baselines, Adaptive Auth, Merchant Risk, Mule Network, Dormant Account Hijack, Round Amount Anomaly, Category Mismatch, New Account Risk, Velocity Engine, Target Encoding, Email Risk Scoring, Fraud Ring Detection, Graph Analysis (NetworkX + Louvain), UID Profiling, V-Feature PCA, Frequency Encoding, Time Windows, Peer Group Deviation, Entropy Features, Lag Pattern Analysis, Cross-Feature Rates, Feature Selection (Boruta + IV).

---

## Architecture Pipeline

```
Stage 1: INGEST    → Raw Transaction Data (590,540 txns, 434 features)
Stage 2: FEATURE   → 25 Detection Layers (velocity, graph, behavioral, entropy)
Stage 3: ENSEMBLE  → 8 Models + Meta-Learner (XGB, LGB, CatBoost, RF, MLP, IsoForest, TabNet, AE)
Stage 4: CALIBRATE → Conformal Prediction (95% coverage guarantee)
Stage 5: DECIDE    → 4-Tier Risk Routing (Approve, PIN, Biometric, Block)
                     └── SHAP + LIME explanations per decision
```

---

## Dashboard Pages (11)

1. **Command Center** — KPI overview, architecture pipeline, model leaderboard, detection layers grid
2. **Live Stream** — Real-time WebSocket transaction feed with color-coded risk
3. **Risk Analyzer** — Interactive transaction scoring with explainable risk factors (SHAP)
4. **Flagged Queue** — High-risk transaction review with risk tiers
5. **Model Lab** — Multi-metric radar chart, ROC curves, model comparison
6. **Analytics** — Risk score distribution, routing tier donut, score statistics
7. **Fraud Network** — Force-directed graph visualization with Louvain community detection
8. **Risk Heatmap** — Amount vs time-of-day risk density
9. **ROI Calculator** — Before/after deployment savings calculator
10. **Robustness** — 5 adversarial attack test results
11. **Threshold Tuner** — Cost-sensitive threshold optimization with configurable FN/FP costs

---

## Product Ecosystem

| Interface | URL | Description |
|-----------|-----|-------------|
| Dashboard | [fraudshield-ai.streamlit.app](https://fraudshield-ai.streamlit.app) | 11-page interactive analytics hub (live) |
| Landing Page | `localhost:8000` | Product landing page |
| Mobile App | `localhost:8000/mobile/` | Installable PWA with fraud alerts |
| API Docs | `localhost:8000/docs` | Swagger interactive documentation |
| WebSocket | `ws://localhost:8000/ws/feed` | Real-time transaction stream |
| GitHub | [github.com/jetashjethi-ui/FraudShield-AI](https://github.com/jetashjethi-ui/FraudShield-AI) | Source code |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML/AI | XGBoost, LightGBM, CatBoost, scikit-learn, PyTorch (Autoencoder), Optuna |
| Explainability | SHAP, LIME |
| Graph Analysis | NetworkX, Louvain community detection |
| Dashboard | Streamlit, Plotly, custom CSS |
| API | FastAPI, WebSocket |
| Deployment | Docker, Streamlit Cloud |
| Data | Pandas, NumPy, SciPy |

---

*FraudShield AI — Real-time fraud detection with explainable AI*
