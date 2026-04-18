"""
FraudShield AI — Optuna Hyperparameter Tuning
Automated tuning for XGBoost and LightGBM using Optuna.
Saves best parameters for use in the main training pipeline.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def tune_xgboost(X, y, n_trials=20, n_folds=3):
    """Tune XGBoost hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda',
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', study_name='xgboost_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    -> Best XGBoost AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X, y, n_trials=20, n_folds=3):
    """Tune LightGBM hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(0)])
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', study_name='lightgbm_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    -> Best LightGBM AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def tune_catboost(X, y, n_trials=20, n_folds=3):
    """Tune CatBoost hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 500),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 64, 255),
            'auto_class_weights': 'Balanced',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 0,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', study_name='catboost_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"    -> Best CatBoost AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def run_tuning(X_train, y_train, output_dir, n_trials=15):
    """Run full Optuna tuning and save best parameters."""
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    # Sample for speed (tuning on full data is too slow)
    rng = np.random.RandomState(42)
    sample_size = min(150000, len(X_train))
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)
    if hasattr(X_train, 'iloc'):
        X_sample = X_train.iloc[sample_idx]
    else:
        X_sample = X_train[sample_idx]
    y_sample = y_train.iloc[sample_idx]
    print(f"  [TUNE] Using {sample_size:,} samples for tuning...")

    print(f"\n  [XGBoost] Tuning {n_trials} trials...")
    xgb_params = tune_xgboost(X_sample, y_sample, n_trials=n_trials)

    print(f"\n  [LightGBM] Tuning {n_trials} trials...")
    lgb_params = tune_lightgbm(X_sample, y_sample, n_trials=n_trials)

    cat_trials = max(n_trials // 2, 20)  # CatBoost is slower, use fewer trials
    print(f"\n  [CatBoost] Tuning {cat_trials} trials...")
    cat_params = tune_catboost(X_sample, y_sample, n_trials=cat_trials)

    # Save best params
    best_params = {
        'xgboost': xgb_params,
        'lightgbm': lgb_params,
        'catboost': cat_params,
    }

    params_path = os.path.join(output_dir, "results", "optuna_best_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2, default=str)

    print(f"\n  [SAVE] Best params saved to {params_path}")
    return best_params
