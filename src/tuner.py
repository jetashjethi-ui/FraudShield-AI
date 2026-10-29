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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def _trial_progress_callback(total_trials):
    """Print periodic progress so a long serial tuning run does not look stuck."""
    def report(study, trial):
        completed = trial.number + 1
        if completed == 1 or completed % 10 == 0 or completed == total_trials:
            print(
                f"    -> {completed}/{total_trials} trials; "
                f"best chronological CV ROC-AUC: {study.best_value:.4f}",
                flush=True,
            )
    return report


def tune_xgboost(X, y, n_trials=5, n_folds=2, use_gpu=False):
    """Tune XGBoost hyperparameters with Optuna."""

    def objective(trial):
        nonlocal use_gpu
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 120, 350),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': 4,
            'verbosity': 0,
            'tree_method': 'hist',
            'max_bin': 64,
        }
        if use_gpu:
            params['device'] = 'cuda'

        splitter = TimeSeriesSplit(n_splits=n_folds)
        aucs = []
        for train_idx, val_idx in splitter.split(X):
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except Exception as exc:
                message = str(exc).lower()
                if not use_gpu or not any(word in message for word in ('cuda', 'gpu', 'memory', 'device')):
                    raise
                print(f"    [TUNE] XGBoost GPU fit failed ({exc}); continuing tuning on CPU.")
                use_gpu = False
                params['device'] = 'cpu'
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', study_name='xgboost_tune')
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_trial_progress_callback(n_trials)],
        show_progress_bar=False,
    )

    print(f"    -> Best XGBoost AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X, y, n_trials=5, n_folds=2):
    """Tune LightGBM hyperparameters with Optuna."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 120, 350),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'is_unbalance': True,
            'random_state': 42,
            'n_jobs': 4,
            'verbosity': -1,
        }

        splitter = TimeSeriesSplit(n_splits=n_folds)
        aucs = []
        for train_idx, val_idx in splitter.split(X):
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
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_trial_progress_callback(n_trials)],
        show_progress_bar=False,
    )

    print(f"    -> Best LightGBM AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def tune_catboost(X, y, n_trials=5, n_folds=2, use_gpu=False):
    """Tune CatBoost hyperparameters with Optuna."""

    def objective(trial):
        nonlocal use_gpu
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
            'thread_count': 4,
            'allow_writing_files': False,
        }
        if use_gpu:
            params.update(task_type='GPU', devices='0', gpu_ram_part=0.25, boosting_type='Plain')

        splitter = TimeSeriesSplit(n_splits=n_folds)
        aucs = []
        for train_idx, val_idx in splitter.split(X):
            if hasattr(X, 'iloc'):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(**params)
            try:
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            except Exception as exc:
                message = str(exc).lower()
                if not use_gpu or not any(word in message for word in ('cuda', 'gpu', 'memory', 'device')):
                    raise
                print(f"    [TUNE] CatBoost GPU fit failed ({exc}); continuing tuning on CPU.")
                use_gpu = False
                for key in ('task_type', 'devices', 'gpu_ram_part', 'boosting_type'):
                    params.pop(key, None)
                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', study_name='catboost_tune')
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_trial_progress_callback(n_trials)],
        show_progress_bar=False,
    )

    print(f"    -> Best CatBoost AUC: {study.best_value:.4f}")
    print(f"    -> Best params: {study.best_params}")
    return study.best_params


def run_tuning(X_train, y_train, output_dir, n_trials=150, profile=None):
    """Run full Optuna tuning and save best parameters."""
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    # Thin the ordered training period without shuffling its chronology.
    sample_size = min(20000, len(X_train))
    sample_idx = np.linspace(0, len(X_train) - 1, sample_size, dtype=int)
    if hasattr(X_train, 'iloc'):
        X_sample = X_train.iloc[sample_idx]
    else:
        X_sample = X_train[sample_idx]
    y_sample = y_train.iloc[sample_idx]
    print(f"  [TUNE] Using {sample_size:,} samples for tuning...")

    print(f"\n  [XGBoost] Tuning {n_trials} trials...")
    if profile is None:
        from src.runtime_config import training_profile
        profile = training_profile()
    xgb_params = tune_xgboost(
        X_sample, y_sample, n_trials=n_trials, n_folds=2,
        use_gpu=bool(profile.get('xgb_gpu')),
    )

    print(f"\n  [LightGBM] Tuning {n_trials} trials...")
    lgb_params = tune_lightgbm(X_sample, y_sample, n_trials=n_trials, n_folds=2)

    cat_trials = n_trials
    print(f"\n  [CatBoost] Tuning {cat_trials} trials...")
    cat_params = tune_catboost(
        X_sample, y_sample, n_trials=cat_trials, n_folds=2,
        use_gpu=bool(profile.get('catboost_gpu')),
    )

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
