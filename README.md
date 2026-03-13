# FraudShield AI

Real-time fraud detection system built for the Financial Services Hackathon (March 2026).

## What it does

Takes raw transaction data (590K+ transactions), runs it through 15 feature engineering layers, and uses a 5-model ensemble to detect fraud with 95.4% AUC accuracy.

The system outputs a risk score (0-100) for each transaction and categorizes them into 4 tiers:
- **GREEN** → auto-approve
- **YELLOW** → PIN verification
- **ORANGE** → biometric check
- **RED** → block immediately

Each decision comes with explainable AI reasons so analysts know *why* a transaction was flagged.

## Models

- XGBoost (AUC: 0.954)
- CatBoost (AUC: 0.941)
- MLP Neural Network (AUC: 0.908)
- Random Forest (AUC: 0.881)
- Isolation Forest (AUC: 0.702)
- Weighted ensemble of all 5 with auto-tuned threshold

## How to run

```bash
pip install -r requirements.txt
```

Run the full pipeline:
```bash
python main.py
```

Launch the dashboard:
```bash
streamlit run dashboard.py
```

## Dataset

Uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle. Place `train_transaction.csv` and `train_identity.csv` in the `data/` folder.

## Project structure

```
├── main.py              # runs the full pipeline
├── dashboard.py         # streamlit demo dashboard
├── src/
│   ├── data_loader.py   # loads and merges datasets
│   ├── feature_engine.py # 15 feature engineering layers
│   ├── models.py        # model training + ensemble
│   ├── risk_scorer.py   # risk scoring + explanations
│   └── visualizer.py    # chart generation
├── data/                # dataset files (not in repo)
└── outputs/             # results, charts, model artifacts
```
