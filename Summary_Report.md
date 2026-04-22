# FraudShield AI — Executive Summary

## System at a Glance

| Component | Details |
|-----------|---------|
| **Dataset** | IEEE-CIS Fraud Detection (590,540 transactions) |
| **Feature Engineering** | 25 detection layers (velocity, graph, behavioral, entropy, etc.) |
| **Models** | 8-model dual stacking ensemble (XGBoost, LightGBM, CatBoost, RF, MLP, IsoForest, TabNet, Autoencoder) |
| **Best Model AUC** | **0.9776** (XGBoost) |
| **Ensemble AUC** | **0.9763** (4-Model Stacking) |
| **Cross-Validation** | 7-Fold Stratified with Optuna HPO (150 trials) |
| **Uncertainty** | Conformal Prediction with 95% coverage guarantee |
| **Calibration** | Isotonic probability calibration |
| **Threshold** | Cost-optimized at 0.015 (FN=₹850, FP=₹25) → 89% recall |
| **Explainability** | SHAP + LIME + Counterfactual analysis |
| **Dashboard** | 17-page interactive Streamlit dashboard |
| **Mobile App** | Progressive Web App with WebSocket live feed |
| **API** | FastAPI REST API with Swagger + WebSocket `/ws/feed` |
| **Deployment** | Docker + Render.com + Streamlit Cloud |

---

## Model Performance (Real Pipeline Results)

| Model | AUC | F1 | Precision | Recall |
|-------|:---:|:--:|:---------:|:------:|
| **XGBoost** | **0.9776** | 0.747 | 0.900 | 0.639 |
| **Ensemble** | **0.9763** | 0.765 | 0.834 | 0.707 |
| CatBoost | 0.9721 | 0.659 | 0.581 | 0.762 |
| LightGBM | 0.9662 | 0.592 | 0.484 | 0.761 |
| MLP Neural Net | 0.9641 | 0.694 | 0.824 | 0.600 |
| Random Forest | 0.9553 | 0.561 | 0.451 | 0.744 |
| Autoencoder | 0.7110 | — | — | — |
| Isolation Forest | 0.7270 | 0.095 | 0.170 | 0.066 |

## Production Validation

| Check | Result |
|-------|:------:|
| **Adversarial Validation** | AUC = 0.5004 → **EXCELLENT** (no train/test leakage) |
| **Adversarial Robustness** | 5/5 attack scenarios passed |
| **Cost-Optimized Threshold** | 0.015 → ₹3.55 Cr annual savings |
| **Conformal Coverage** | 95% target achieved |

## Risk Scoring System

| Tier | Score Range | Action | Use Case |
|------|:-----------:|--------|----------|
| 🟢 GREEN | 0-30 | Auto-approve | Zero friction for legitimate transactions |
| 🟡 YELLOW | 31-50 | PIN verify | Minor suspicion, confirm identity |
| 🟠 ORANGE | 51-70 | Biometric | Elevated risk, require face/fingerprint |
| 🔴 RED | 71-100 | Block | Immediate freeze + fraud team alert |

## Architecture

```
Raw Data (590K) → 25 Feature Layers → 8-Model Dual Stack → Conformal → 4-Tier Risk Decision
                                                          ↓
                                              SHAP + LIME Explanations
```

## Product Ecosystem

| Interface | URL | Description |
|-----------|-----|-------------|
| 🌐 Landing Page | `localhost:8000` | Startup-quality product page |
| 📱 Mobile App | `localhost:8000/mobile/` | Installable PWA with fraud alerts |
| 📄 API Docs | `localhost:8000/docs` | Swagger interactive documentation |
| 🖥️ Dashboard | `localhost:8501` | 17-page analytics hub |
| ⚡ WebSocket | `ws://localhost:8000/ws/feed` | Real-time transaction feed |

---

*FraudShield AI*
