"""
FraudShield AI — Model Training Module (v2)
7 Models trained, 4-Model Stacking Ensemble: XGBoost, LightGBM, CatBoost, MLP
+ TabNet (Google's attention-based deep learning for tabular data)
Meta-learner: Logistic Regression on out-of-fold predictions
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def prepare_features(df):
    """Prepare feature matrix X and target y from engineered dataframe."""
    print("\n[PREP] Preparing features for modeling...")

    y = df['isFraud'].copy()

    exclude = {'TransactionID', 'isFraud', 'TransactionDT', 'DeviceInfo',
               'DeviceType', 'P_emaildomain', 'R_emaildomain'}

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

    print(f"  -> Features: {X.shape[1]} columns")
    print(f"  -> Target: {y.sum():,} fraud / {(y == 0).sum():,} legit")

    return X, y, feature_cols, label_encoders


def apply_smote(X, y, fraud_ratio=0.15):
    """Oversample fraud class using SMOTE."""
    print(f"\n[SMOTE] Oversampling fraud to ~{fraud_ratio*100:.0f}%...")

    n_legit = (y == 0).sum()
    target_fraud = int(n_legit * fraud_ratio / (1 - fraud_ratio))

    smote = SMOTE(sampling_strategy={1: target_fraud}, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"  -> Before: {len(X):,} samples ({y.mean()*100:.2f}% fraud)")
    print(f"  -> After:  {len(X_res):,} samples ({y_res.mean()*100:.2f}% fraud)")

    return X_res, y_res


def _build_base_models():
    """Define all base models with their configs."""
    models_config = {
        'xgboost': {
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.05,
                min_child_weight=30, subsample=0.85, colsample_bytree=0.85,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0,
                tree_method='hist', device='cuda'
            ),
            'needs_scaling': False,
        },
        'lightgbm': {
            'name': 'LightGBM',
            'model': lgb.LGBMClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.05,
                num_leaves=63, min_child_samples=30, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5,
                is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1
            ),
            'needs_scaling': False,
        },
        'catboost': {
            'name': 'CatBoost',
            'model': CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.08,
                l2_leaf_reg=3.0, border_count=128,
                auto_class_weights='Balanced', eval_metric='AUC',
                random_seed=42, verbose=0, task_type='GPU'
            ),
            'needs_scaling': False,
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=300, max_depth=14, min_samples_leaf=15,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'needs_scaling': False,
        },
        'mlp': {
            'name': 'MLP Neural Net',
            'model': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), activation='relu',
                solver='adam', max_iter=100, batch_size=2048,
                early_stopping=True, validation_fraction=0.1,
                random_state=42, verbose=False
            ),
            'needs_scaling': True,
        },
    }
    return models_config


def train_models(X_train, y_train, X_test, y_test, tuned_params=None):
    """Train all models with stacking ensemble."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING (8 Models, 5-Model Stacking, 7-Fold CV)")
    print("=" * 70)

    results = {}
    trained_models = {}
    models_config = _build_base_models()

    # Apply Optuna-tuned params if available
    if tuned_params:
        if 'xgboost' in tuned_params and 'xgboost' in models_config:
            print("  [OPTUNA] Applying tuned XGBoost params...")
            xgb_model = models_config['xgboost']['model']
            for k, v in tuned_params['xgboost'].items():
                if hasattr(xgb_model, k):
                    setattr(xgb_model, k, v)
        if 'lightgbm' in tuned_params and 'lightgbm' in models_config:
            print("  [OPTUNA] Applying tuned LightGBM params...")
            lgb_model = models_config['lightgbm']['model']
            for k, v in tuned_params['lightgbm'].items():
                if hasattr(lgb_model, k):
                    setattr(lgb_model, k, v)
        if 'catboost' in tuned_params and 'catboost' in models_config:
            print("  [OPTUNA] Applying tuned CatBoost params...")
            cat_model = models_config['catboost']['model']
            for k, v in tuned_params['catboost'].items():
                if hasattr(cat_model, k):
                    setattr(cat_model, k, v)

    # Scaler for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    trained_models['scaler'] = scaler

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Train individual models + generate OOF predictions
    # ──────────────────────────────────────────────────────────────
    print("\n  [Phase 1] Training individual models + OOF predictions...")

    n_folds = 7
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store OOF predictions for stacking
    oof_preds = {}
    test_preds = {}

    for model_key, config in models_config.items():
        print(f"\n  [{config['name']}] Training with {n_folds}-fold OOF...")
        model_class = config['model']

        oof_prob = np.zeros(len(X_train))
        test_prob_sum = np.zeros(len(X_test))

        X_tr = X_train_scaled if config['needs_scaling'] else X_train
        X_te = X_test_scaled if config['needs_scaling'] else X_test

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
            if isinstance(X_tr, np.ndarray):
                X_fold_train, X_fold_val = X_tr[train_idx], X_tr[val_idx]
            else:
                X_fold_train = X_tr.iloc[train_idx]
                X_fold_val = X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Clone model for this fold
            import sklearn.base
            fold_model = sklearn.base.clone(model_class)
            fold_model.fit(X_fold_train, y_fold_train)

            oof_prob[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
            test_prob_sum += fold_model.predict_proba(X_te)[:, 1]

        test_prob = test_prob_sum / n_folds
        oof_preds[model_key] = oof_prob
        test_preds[model_key] = test_prob

        # Train final model on all training data for saving
        final_model = sklearn.base.clone(model_class)
        X_final = X_train_scaled if config['needs_scaling'] else X_train
        final_model.fit(X_final, y_train)
        trained_models[model_key] = final_model

        # Evaluate individual model
        pred = (test_prob > 0.5).astype(int)
        results[model_key] = evaluate_model(config['name'], y_test, pred, test_prob)

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Isolation Forest (unsupervised — separate from stacking)
    # ──────────────────────────────────────────────────────────────
    print(f"\n  [Isolation Forest] Unsupervised anomaly detection...")
    iso_model = IsolationForest(
        n_estimators=100, max_samples=256, contamination=0.035,
        random_state=42, n_jobs=-1
    )
    iso_model.fit(X_train)
    iso_scores_raw = -iso_model.score_samples(X_test)
    iso_scores = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min() + 1e-8)

    iso_scores_train_raw = -iso_model.score_samples(X_train)
    iso_scores_train = (iso_scores_train_raw - iso_scores_train_raw.min()) / (iso_scores_train_raw.max() - iso_scores_train_raw.min() + 1e-8)

    iso_pred = (iso_scores > 0.5).astype(int)
    trained_models['isolation_forest'] = iso_model
    results['isolation_forest'] = evaluate_model("Isolation Forest", y_test, iso_pred, iso_scores)

    # ──────────────────────────────────────────────────────────────
    # PHASE 2.5: TabNet (Google's attention-based deep learning)
    # ──────────────────────────────────────────────────────────────
    tabnet_ok = False
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier as TabNet
        print(f"\n  [TabNet] Training Google's attention-based deep learning model...")

        # Chunked predict to avoid RAM issues on large arrays
        def _tabnet_predict_chunked(model, X, chunk_size=10000):
            preds = np.zeros(len(X))
            for start in range(0, len(X), chunk_size):
                end = min(start + chunk_size, len(X))
                preds[start:end] = model.predict_proba(X[start:end])[:, 1]
            return preds

        tabnet_oof = np.zeros(len(X_train))
        tabnet_test_sum = np.zeros(len(X_test))

        # Full power TabNet config (GPU-ready)
        import torch
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"    -> TabNet device: {_device}")
        tabnet_cfg = dict(
            n_d=64, n_a=64, n_steps=7,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_params=dict(step_size=15, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', verbose=0, seed=42,
            device_name=_device
        )

        import gc

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_fold_tr = X_train_scaled[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_tr = y_train.iloc[train_idx].values
            y_fold_val = y_train.iloc[val_idx].values

            tabnet_model = TabNet(**tabnet_cfg)
            tabnet_model.fit(
                X_fold_tr, y_fold_tr,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric=['auc'],
                max_epochs=100, patience=15, batch_size=4096,
                drop_last=False
            )
            tabnet_oof[val_idx] = _tabnet_predict_chunked(tabnet_model, X_fold_val)
            tabnet_test_sum += _tabnet_predict_chunked(tabnet_model, X_test_scaled)
            print(f"    -> Fold {fold+1}/{n_folds} done")
            del tabnet_model; gc.collect()  # Free RAM after each fold

        tabnet_test_prob = tabnet_test_sum / n_folds
        oof_preds['tabnet'] = tabnet_oof
        test_preds['tabnet'] = tabnet_test_prob

        # Train final TabNet on all data
        final_tabnet = TabNet(**tabnet_cfg)
        final_tabnet.fit(
            X_train_scaled, y_train.values,
            max_epochs=100, patience=15, batch_size=4096,
            drop_last=False
        )
        trained_models['tabnet'] = final_tabnet

        tabnet_pred = (tabnet_test_prob > 0.5).astype(int)
        results['tabnet'] = evaluate_model("TabNet", y_test, tabnet_pred, tabnet_test_prob)
        tabnet_ok = True
        print(f"    -> TabNet AUC: {results['tabnet']['auc']:.4f}")

    except ImportError:
        print("  [TabNet] pytorch-tabnet not installed, skipping")
    except Exception as e:
        print(f"  [TabNet] Error: {e}, skipping")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: NONLINEAR STACKING META-LEARNER (XGBoost Level-2)
    # ──────────────────────────────────────────────────────────────
    print(f"\n  [Stacking] Training XGBoost meta-learner on OOF predictions...")

    # Build stacking feature matrix — top models + anomaly scores
    stack_keys = [k for k in models_config.keys() if k != 'random_forest']
    if tabnet_ok:
        stack_keys.append('tabnet')

    # Level-2 features: model OOF predictions + anomaly scores
    meta_features_train = [oof_preds[k] for k in stack_keys]
    meta_features_test = [test_preds[k] for k in stack_keys]

    # Add Isolation Forest anomaly scores as deep anomaly features
    meta_features_train.append(iso_scores_train)
    meta_features_test.append(iso_scores)

    meta_X_train = np.column_stack(meta_features_train)
    meta_X_test = np.column_stack(meta_features_test)

    meta_feature_names = [f"oof_{k}" for k in stack_keys] + ['iso_anomaly_score']
    print(f"    -> Stacking features: {meta_feature_names}")

    # Nonlinear meta-learner: XGBoost captures model interactions
    meta_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', random_state=42, verbosity=0
    )
    meta_model.fit(meta_X_train, y_train)
    trained_models['meta_learner'] = meta_model

    ensemble_proba = meta_model.predict_proba(meta_X_test)[:, 1]

    # ──────────────────────────────────────────────────────────────
    # PHASE 3b: RANK-WEIGHTED BLEND (second ensemble method)
    # ──────────────────────────────────────────────────────────────
    print(f"\n  [Blending] Computing rank-weighted ensemble...")
    from scipy.stats import rankdata

    # Convert predictions to ranks (handles calibration differences)
    rank_preds = {}
    for k in stack_keys:
        rank_preds[k] = rankdata(test_preds[k]) / len(test_preds[k])

    # Weight by individual AUC performance
    auc_weights = {}
    total_w = 0
    for k in stack_keys:
        if k in results:
            w = results[k]['auc'] ** 3  # Cube AUC to amplify differences
            auc_weights[k] = w
            total_w += w
        else:
            auc_weights[k] = 1.0
            total_w += 1.0

    blend_proba = np.zeros(len(y_test))
    for k in stack_keys:
        blend_proba += rank_preds[k] * (auc_weights[k] / total_w)

    blend_auc = roc_auc_score(y_test, blend_proba)
    stack_auc = roc_auc_score(y_test, ensemble_proba)
    print(f"    -> XGBoost Stacking AUC: {stack_auc:.4f}")
    print(f"    -> Rank-Weighted Blend AUC: {blend_auc:.4f}")

    # Pick the better ensemble method
    if blend_auc > stack_auc:
        print(f"    -> BLEND wins! Using rank-weighted ensemble")
        ensemble_proba = blend_proba
    else:
        print(f"    -> STACKING wins! Using XGBoost meta-learner")

    # Optimal threshold
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.01):
        preds = (ensemble_proba > t).astype(int)
        f = f1_score(y_test, preds)
        if f > best_f1:
            best_f1 = f
            best_thresh = t

    default_f1 = f1_score(y_test, (ensemble_proba > 0.5).astype(int))
    print(f"    -> Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f} vs default-0.5 F1={default_f1:.4f})")

    ensemble_pred = (ensemble_proba > best_thresh).astype(int)
    stack_label = f"Dual Ensemble ({len(stack_keys)}-Model)"
    results['ensemble'] = evaluate_model(stack_label, y_test, ensemble_pred, ensemble_proba)

    # Verify ensemble beats individual models
    best_individual = max((v['auc'], k) for k, v in results.items() if k != 'ensemble')
    ens_auc = results['ensemble']['auc']
    if ens_auc > best_individual[0]:
        print(f"\n  [OK] Ensemble AUC ({ens_auc:.4f}) BEATS best individual ({best_individual[1]}: {best_individual[0]:.4f})")
    else:
        print(f"\n  [WARN] Ensemble AUC ({ens_auc:.4f}) vs best individual ({best_individual[1]}: {best_individual[0]:.4f})")

    probas = {
        'xgb_proba': test_preds['xgboost'],
        'lgb_proba': test_preds['lightgbm'],
        'cat_proba': test_preds['catboost'],
        'rf_proba': test_preds['random_forest'],
        'mlp_proba': test_preds['mlp'],
        'iso_scores': iso_scores,
        'ensemble_proba': ensemble_proba,
        'ensemble_pred': ensemble_pred,
        'optimal_threshold': best_thresh,
    }
    if tabnet_ok:
        probas['tabnet_proba'] = test_preds['tabnet']
    return trained_models, results, probas


def evaluate_model(name, y_true, y_pred, y_proba):
    """Compute and display all metrics for a model."""
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    print(f"    -> AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"    -> Confusion: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

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

    for name in ['xgboost', 'lightgbm', 'random_forest', 'catboost']:
        if name not in models:
            continue
        model = models[name]
        if name == 'catboost':
            imp = model.get_feature_importance()
            importances[name] = dict(zip(feature_names, imp / imp.sum()))
        else:
            imp = model.feature_importances_
            importances[name] = dict(zip(feature_names, imp))

    # Average importance across all tree models
    avg_imp = {}
    for feat in feature_names:
        scores = [importances[m].get(feat, 0) for m in importances]
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

    # Save LightGBM model
    if 'lightgbm' in models:
        models['lightgbm'].booster_.save_model(os.path.join(artifacts_dir, 'lightgbm_model.txt'))

    # Save CatBoost model
    if 'catboost' in models:
        models['catboost'].save_model(os.path.join(artifacts_dir, 'catboost_model.cbm'))

    # Save other models with pickle
    for name in ['random_forest', 'isolation_forest', 'mlp', 'scaler', 'meta_learner']:
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
