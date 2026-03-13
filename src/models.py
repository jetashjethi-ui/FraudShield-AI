"""
FraudShield AI — Model Training Module
Trains XGBoost (GBT), Random Forest, Isolation Forest, MLP Neural Network, and Weighted Ensemble.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def prepare_features(df):
    """Prepare feature matrix X and target y from engineered dataframe."""
    print("\n[PREP] Preparing features for modeling...")

    # Separate target
    y = df['isFraud'].copy()

    # Columns to exclude from features
    exclude = {'TransactionID', 'isFraud', 'TransactionDT', 'DeviceInfo',
               'DeviceType', 'P_emaildomain', 'R_emaildomain'}
    
    # Get feature columns (numeric only for tree models)
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    # Encode remaining categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('MISSING')
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Fill remaining NaN with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    print(f"  → Features: {X.shape[1]} columns")
    print(f"  → Target: {y.sum():,} fraud / {(y == 0).sum():,} legit")

    return X, y, feature_cols, label_encoders


def apply_smote(X, y, fraud_ratio=0.15):
    """Oversample fraud class using SMOTE."""
    print(f"\n[SMOTE] Oversampling fraud to ~{fraud_ratio*100:.0f}%...")
    
    # Calculate sampling strategy
    n_legit = (y == 0).sum()
    target_fraud = int(n_legit * fraud_ratio / (1 - fraud_ratio))

    smote = SMOTE(
        sampling_strategy={1: target_fraud},
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X, y)
    
    print(f"  → Before: {len(X):,} samples ({y.mean()*100:.2f}% fraud)")
    print(f"  → After:  {len(X_res):,} samples ({y_res.mean()*100:.2f}% fraud)")
    
    return X_res, y_res


def train_models(X_train, y_train, X_test, y_test):
    """Train all models and return predictions + metrics."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    results = {}
    models = {}

    # ─── MODEL 1: XGBoost (Gradient Boosted Trees) ───
    print("\n  [Model 1] XGBoost (Gradient Boosted Trees)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=sum(y_train == 0) / max(sum(y_train == 1), 1),
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba > 0.5).astype(int)
    models['xgboost'] = xgb_model
    results['xgboost'] = evaluate_model("XGBoost", y_test, xgb_pred, xgb_proba)

    # ─── MODEL 2: Random Forest ───
    print("\n  [Model 2] Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)
    models['random_forest'] = rf_model
    results['random_forest'] = evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

    # ─── MODEL 3: Isolation Forest (Anomaly Detector) ───
    print("\n  [Model 3] Isolation Forest (Anomaly Detection)...")
    iso_model = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.035,  # ~3.5% fraud rate
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_train)
    iso_scores = -iso_model.score_samples(X_test)  # higher = more anomalous
    # Normalize to 0-1
    iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    iso_pred = (iso_scores_norm > 0.5).astype(int)
    models['isolation_forest'] = iso_model
    results['isolation_forest'] = evaluate_model("Isolation Forest", y_test, iso_pred, iso_scores_norm)

    # ─── MODEL 4: MLP Neural Network ───
    print("\n  [Model 4] MLP Neural Network (Deep Learning)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=50,
        batch_size=2048,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    mlp_model.fit(X_train_scaled, y_train)
    mlp_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
    mlp_pred = (mlp_proba > 0.5).astype(int)
    models['mlp'] = mlp_model
    models['scaler'] = scaler
    results['mlp'] = evaluate_model("MLP Neural Net", y_test, mlp_pred, mlp_proba)

    # ─── ENSEMBLE: Weighted Average (4 models) ───
    print("\n  [Ensemble] Weighted Voting (XGB:0.40 + RF:0.20 + MLP:0.20 + IsoF:0.20)...")
    ensemble_proba = 0.40 * xgb_proba + 0.20 * rf_proba + 0.20 * mlp_proba + 0.20 * iso_scores_norm
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    results['ensemble'] = evaluate_model("Ensemble (4-Model)", y_test, ensemble_pred, ensemble_proba)

    return models, results, {
        'xgb_proba': xgb_proba,
        'rf_proba': rf_proba,
        'mlp_proba': mlp_proba,
        'iso_scores': iso_scores_norm,
        'ensemble_proba': ensemble_proba,
        'ensemble_pred': ensemble_pred,
    }


def evaluate_model(name, y_true, y_pred, y_proba):
    """Compute and display all metrics for a model."""
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    print(f"    → AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"    → Confusion Matrix: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

    return {
        'name': name,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def get_feature_importance(models, feature_names):
    """Extract feature importances from tree models."""
    importances = {}
    
    if 'xgboost' in models:
        imp = models['xgboost'].feature_importances_
        importances['xgboost'] = dict(zip(feature_names, imp))
    
    if 'random_forest' in models:
        imp = models['random_forest'].feature_importances_
        importances['random_forest'] = dict(zip(feature_names, imp))
    
    # Average importance across models
    avg_imp = {}
    for feat in feature_names:
        scores = []
        for model_name in ['xgboost', 'random_forest']:
            if model_name in importances:
                scores.append(importances[model_name].get(feat, 0))
        avg_imp[feat] = np.mean(scores) if scores else 0
    
    importances['average'] = avg_imp
    return importances


def save_metrics(results, output_dir):
    """Save metrics to JSON file."""
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    
    save_data = {}
    for model_name, res in results.items():
        save_data[model_name] = {
            'name': res['name'],
            'auc': float(res['auc']),
            'f1': float(res['f1']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'confusion_matrix': res['confusion_matrix'].tolist()
        }
    
    with open(metrics_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n[SAVE] Metrics saved to {metrics_path}")
    return save_data


def save_model_artifacts(models, label_encoders, feature_names, output_dir):
    """Save trained models and encoders for the dashboard."""
    artifacts_dir = os.path.join(output_dir, "model_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save XGBoost model
    if 'xgboost' in models:
        models['xgboost'].save_model(os.path.join(artifacts_dir, 'xgboost_model.json'))
    
    # Save other models with pickle
    for name in ['random_forest', 'isolation_forest', 'mlp', 'scaler']:
        if name in models:
            with open(os.path.join(artifacts_dir, f'{name}.pkl'), 'wb') as f:
                pickle.dump(models[name], f)
    
    # Save label encoders
    with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature names
    with open(os.path.join(artifacts_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    print(f"[SAVE] Model artifacts saved to {artifacts_dir}")
