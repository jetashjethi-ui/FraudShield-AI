# FraudShield AI 🛡️

**Adaptive Real-Time Fraud Detection with Explainable Risk Intelligence**

> 11 Detection Layers | Multi-Model Ensemble | Explainable AI | Adaptive Authentication

## 🎯 Overview

FraudShield AI is an intelligent fraud detection system built for the **Financial Services Hackathon (March 2026)**. Unlike traditional black-box models, FraudShield AI employs **11 distinct detection layers** that don't just flag fraud — they **explain why**, **adapt to seasonal spending**, catch **SIM swaps before the fraud happens**, profile **risky merchants**, spot **mule networks**, detect **first-transaction abuse**, and recommend **proportional authentication** based on risk severity.

## 🏗️ Architecture

```
[Data Loading] → [11-Layer Feature Engineering] → [Multi-Model Ensemble]
                                                          │
                                                    ┌─────┼─────┐
                                                    │     │     │
                                                  XGBoost  RF  IsoForest
                                                    │     │     │
                                                    └──┬──┘     │
                                                       │        │
                                                  [Weighted Voting]
                                                       │
                                                [Risk Score Engine]
                                                   (0-100 score)
                                                       │
                                          ┌────────────┼────────────┐
                                          │            │            │
                                  [Explainability] [Risk Tiers] [Adaptive Auth]
                                  (8 reason checks) (4 colors)  (4 responses)
```

## 🔍 11 Detection Layers

| # | Layer | Description |
|---|-------|-------------|
| 1 | **Explainable AI (XAI)** | Human-readable reasons for every flagged transaction |
| 2 | **Behavioral Fingerprinting** | Spending DNA per customer (time, amount, merchant patterns) |
| 3 | **SIM Swap Early Warning** | Detects sudden device/SIM changes before fraud |
| 4 | **Seasonal Baselines** | Contextual spending norms (Diwali, holidays won't trigger false alarms) |
| 5 | **Adaptive Step-Up Auth** | GREEN/YELLOW/ORANGE/RED proportional authentication |
| 6 | **Merchant Risk Profiling** | Dynamic fraud rate scoring per merchant category |
| 7 | **Shared Device Detection** | Identifies mule networks via shared device fingerprints |
| 8 | **Dormant Account Hijack** | Flags inactive accounts with sudden high-value activity |
| 9 | **Round Amount Suspicion** | Scores round amounts higher (₹10,000 vs ₹9,437.50) |
| 10 | **Category Mismatch** | Detects unusual merchant category choices per user |
| 11 | **New Account First-Txn** | Flags high-value first transactions on new accounts |

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn
```

### Dataset
Download the IEEE-CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place CSV files in the `data/` directory.

### Run
```bash
python main.py
```

## 📊 Output

The pipeline generates:
- **Model Metrics**: AUC-ROC, F1, Precision, Recall for all models
- **Scored Transactions**: Every transaction with risk score (0-100), risk tier, and explanation
- **Visualizations**: 10 publication-ready charts in `outputs/visualizations/`
- **Sample Flagged Transactions**: Top fraudulent transactions with explanations

## 📁 Project Structure

```
FraudShield-AI/
├── main.py                    # Entry point — runs complete pipeline
├── src/
│   ├── data_loader.py         # Data loading, merging, cleaning
│   ├── feature_engine.py      # 11 detection layers as features
│   ├── models.py              # XGBoost + RF + IsoForest + Ensemble
│   ├── risk_scorer.py         # Risk scoring + explainability
│   └── visualizer.py          # 10 visualization charts
├── data/                      # IEEE-CIS dataset (CSV files)
├── outputs/
│   ├── results/               # Metrics, scored CSVs
│   └── visualizations/        # Generated charts
└── README.md
```

## 🏆 Hackathon Differentiators

1. **11 layers vs competitors' 1-2** — massive innovation gap
2. **Explainability** — judges consistently rank this #1
3. **India-specific** features (SIM swap, round amounts, UPI context)
4. **Adaptive auth** — product thinking, not just ML
5. **Seasonal awareness** — production-ready, not just lab accuracy

## 📜 License

Built for the Financial Services Hackathon, March 2026.
