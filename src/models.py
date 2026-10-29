"""Temporal model training, calibration, stacking, and artifact export."""

import pandas as pd
import numpy as np
import os
import json
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.runtime_config import training_profile, available_ram_gb
from src.evaluation import ranking_metrics


def _chronological_splits(n_samples, n_splits, transaction_times=None):
    """Build expanding-window folds, preferring equal-duration time windows."""
    if transaction_times is None:
        return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))
    times = np.asarray(transaction_times, dtype=np.float64).reshape(-1)
    if len(times) != n_samples or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("transaction_times must be finite, chronological, and match the rows.")
    time_span = float(times[-1] - times[0])
    if time_span <= 0:
        return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))

    folds = []
    for fold in range(1, n_splits + 1):
        train_end_time = times[0] + time_span * fold / (n_splits + 1)
        validation_end_time = times[0] + time_span * (fold + 1) / (n_splits + 1)
        train_end = int(np.searchsorted(times, train_end_time, side="right"))
        validation_end = int(np.searchsorted(times, validation_end_time, side="right"))
        if train_end <= 0 or validation_end <= train_end:
            return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples)))
        folds.append((np.arange(train_end), np.arange(train_end, validation_end)))
    return folds


def prepare_features(df):
    """Compatibility wrapper around the safe reusable feature contract."""
    print("\n[PREP] Preparing features for modeling...")
    from src.inference_pipeline import FraudFeaturePipeline
    if 'isFraud' not in df:
        raise ValueError("Training data must contain the isFraud target column.")
    y = df['isFraud'].astype(int).copy()
    pipeline = FraudFeaturePipeline().fit(df)
    X = pipeline.transform(df)
    feature_cols = pipeline.get_feature_names_out()

    print(f"  -> Features: {X.shape[1]} columns")
    print(f"  -> Target: {y.sum():,} fraud / {(y == 0).sum():,} legit")

    return X, y, feature_cols, {'feature_pipeline': pipeline}


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


def _build_base_models(profile=None):
    """Define all base models with their configs."""
    profile = profile or training_profile()
    n_jobs = int(profile["n_jobs"])
    models_config = {
        'xgboost': {
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.05,
                min_child_weight=30, subsample=0.85, colsample_bytree=0.85,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                eval_metric='auc', random_state=42, n_jobs=n_jobs, verbosity=0,
                tree_method='hist', max_bin=64,
                **({"device": "cuda"} if profile.get("xgb_gpu") else {})
            ),
            'needs_scaling': False,
            'gpu_enabled': bool(profile.get("xgb_gpu")),
        },
        'lightgbm': {
            'name': 'LightGBM',
            'model': lgb.LGBMClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.05,
                num_leaves=63, min_child_samples=30, subsample=0.85,
                colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5,
                is_unbalance=True, random_state=42, n_jobs=n_jobs, verbosity=-1
            ),
            'needs_scaling': False,
        },
        'catboost': {
            'name': 'CatBoost',
            'model': CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.08,
                l2_leaf_reg=3.0, border_count=128,
                auto_class_weights='Balanced', eval_metric='AUC',
                random_seed=42, verbose=0,
                thread_count=n_jobs, allow_writing_files=False,
                task_type='GPU' if profile.get("catboost_gpu") else 'CPU',
                **({"devices": "0", "gpu_ram_part": 0.30, "boosting_type": "Plain"}
                   if profile.get("catboost_gpu") else {})
            ),
            'needs_scaling': False,
            'gpu_enabled': bool(profile.get("catboost_gpu")),
        },
        'random_forest': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=180, max_depth=12, min_samples_leaf=20,
                class_weight='balanced', random_state=42, n_jobs=n_jobs,
                max_samples=0.8
            ),
            'needs_scaling': False,
            'gpu_enabled': False,
        },
        'mlp': {
            'name': 'MLP Neural Net',
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu',
                solver='adam', max_iter=40, batch_size=2048,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=6, random_state=42, verbose=False
            ),
            'needs_scaling': True,
            'gpu_enabled': False,
        },
    }
    return models_config


def _fit_with_fallback(model, X, y, config):
    """Retry a GPU fit on CPU only when the failure is device/resource related."""
    try:
        model.fit(X, y)
        return model
    except Exception as exc:
        message = str(exc).lower()
        gpu_failure = any(token in message for token in (
            "cuda", "gpu", "out of memory", "device", "no visible",
        ))
        if not config.get("gpu_enabled") or not gpu_failure:
            raise
        print(f"    [GPU] Device fit failed ({exc}); retrying this model on CPU.")
        base_model = config["model"]
        if isinstance(base_model, xgb.XGBClassifier):
            base_model.set_params(device="cpu", tree_method="hist")
        elif isinstance(base_model, CatBoostClassifier):
            base_model.set_params(task_type="CPU")
        else:
            raise
        config["gpu_enabled"] = False
        gc.collect()
        from sklearn.base import clone
        fallback = clone(base_model)
        fallback.fit(X, y)
        return fallback


def train_models(X_train, y_train, X_test, y_test, tuned_params=None, transaction_times=None):
    """Train a time-aware OOF stack and evaluate only on a later holdout."""
    from imblearn.over_sampling import SMOTE
    from sklearn.base import clone
    from src.threshold_optimizer import optimize_threshold

    print("\n" + "=" * 70)
    print("MODEL TRAINING (temporal OOF, API-compatible stack)")
    print("=" * 70)
    results, trained_models = {}, {}
    profile = training_profile()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(profile["n_jobs"]))
    if profile["available_ram_gb"] is not None and profile["available_ram_gb"] < 1.5:
        raise MemoryError(
            f"Only {profile['available_ram_gb']:.1f} GiB RAM is free before model fitting; "
            "close other applications and retry."
        )
    models_config = _build_base_models(profile)
    print(
        f"  Runtime profile: {profile['available_ram_gb']} GiB free RAM, "
        f"XGBoost GPU={profile['xgb_gpu']}, CatBoost GPU={profile['catboost_gpu']}"
    )

    if tuned_params:
        for name in ("xgboost", "lightgbm", "catboost"):
            if name in tuned_params and name in models_config:
                model = models_config[name]["model"]
                for key, value in tuned_params[name].items():
                    if hasattr(model, key):
                        setattr(model, key, value)

    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else pd.DataFrame(X_train)
    X_test = X_test.reset_index(drop=True) if hasattr(X_test, "reset_index") else pd.DataFrame(X_test)
    time_values = None if transaction_times is None else np.asarray(transaction_times, dtype=np.float64).reshape(-1)
    if time_values is not None and len(time_values) != len(X_train):
        raise ValueError("transaction_times length must match X_train.")

    # Expanding windows avoid training an OOF model on transactions later than
    # the row it predicts. The initial warm-up period is deliberately excluded
    # from meta-model fitting and threshold selection.
    folds = _chronological_splits(len(X_train), n_splits=4, transaction_times=time_values)
    covered = np.zeros(len(X_train), dtype=bool)
    for _, validation_idx in folds:
        covered[validation_idx] = True
    oof_preds, test_preds = {}, {}
    temporal_cv_metrics = {"base_models": {}, "stack_vs_catboost": []}

    for model_key, config in models_config.items():
        print(f"\n  [{config['name']}] expanding-window OOF...")
        model_class = config["model"]
        oof_prob = np.full(len(X_train), np.nan, dtype=np.float32)
        model_fold_metrics = []

        for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
            x_fold_train = X_train.iloc[train_idx]
            x_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]

            # Keep native class weighting for LightGBM, CatBoost, and RF.
            # SMOTE is reserved for XGBoost and MLP, which have no equivalent
            # balanced objective in this configuration.
            if model_key in {"xgboost", "mlp"}:
                try:
                    sampler = SMOTE(sampling_strategy=0.15, random_state=42)
                    x_fit, y_fit = sampler.fit_resample(x_fold_train, y_fold_train)
                except ValueError:
                    x_fit, y_fit = x_fold_train, y_fold_train
            else:
                x_fit, y_fit = x_fold_train, y_fold_train

            fold_scaler = None
            if config["needs_scaling"]:
                fold_scaler = StandardScaler()
                x_fit = fold_scaler.fit_transform(x_fit)
                x_val = fold_scaler.transform(x_fold_val)
            else:
                x_val = x_fold_val

            fold_model = clone(config["model"])
            fold_model = _fit_with_fallback(fold_model, x_fit, y_fit, config)
            fold_scores = fold_model.predict_proba(x_val)[:, 1]
            oof_prob[val_idx] = fold_scores
            fold_report = ranking_metrics(y_train.iloc[val_idx], fold_scores)
            fold_report.update({
                "fold": int(fold_no),
                "train_rows": int(len(train_idx)),
                "validation_start_row": int(val_idx[0]),
                "validation_end_row_exclusive": int(val_idx[-1] + 1),
            })
            if time_values is not None:
                fold_report.update({
                    "train_time_end": float(time_values[train_idx[-1]]),
                    "validation_time_start": float(time_values[val_idx[0]]),
                    "validation_time_end": float(time_values[val_idx[-1]]),
                })
            model_fold_metrics.append(fold_report)
            print(f"    -> fold {fold_no}/{len(folds)}")
            del x_fold_train, x_fold_val, y_fold_train, x_fit, y_fit, x_val, fold_model
            gc.collect()

        oof_preds[model_key] = oof_prob
        temporal_cv_metrics["base_models"][model_key] = model_fold_metrics

        # Persist a final model fit on the complete training period.
        final_scaler = None
        x_final = X_train
        y_final = y_train
        if model_key in {"xgboost", "mlp"}:
            try:
                sampler = SMOTE(sampling_strategy=0.15, random_state=42)
                x_final, y_final = sampler.fit_resample(X_train, y_train)
            except ValueError:
                pass
        if config["needs_scaling"]:
            final_scaler = StandardScaler()
            x_final = final_scaler.fit_transform(x_final)
            trained_models["scaler"] = final_scaler
        final_model = clone(config["model"])
        final_model = _fit_with_fallback(final_model, x_final, y_final, config)
        trained_models[model_key] = final_model
        x_test_final = final_scaler.transform(X_test) if config["needs_scaling"] else X_test
        test_prob = final_model.predict_proba(x_test_final)[:, 1]
        test_preds[model_key] = test_prob
        results[model_key] = evaluate_model(
            config["name"], y_test, (test_prob >= 0.5).astype(int), test_prob
        )
        if x_final is not X_train:
            del x_final, y_final
        del x_test_final
        gc.collect()

    # Isolation Forest remains an independently reported baseline. Its scores
    # are not batch-normalized and are intentionally not part of the stack.
    iso_model = IsolationForest(
        n_estimators=100, max_samples=256, contamination=0.035,
        random_state=42, n_jobs=int(profile["n_jobs"]),
    ).fit(X_train)
    train_raw = -iso_model.score_samples(X_train)
    test_raw = -iso_model.score_samples(X_test)
    iso_min, iso_max = float(train_raw.min()), float(train_raw.max())
    iso_scores = np.clip((test_raw - iso_min) / max(iso_max - iso_min, 1e-8), 0, 1)
    trained_models["isolation_forest"] = iso_model
    trained_models["iso_score_min"] = iso_min
    trained_models["iso_score_max"] = iso_max
    results["isolation_forest"] = evaluate_model(
        "Isolation Forest", y_test, (iso_scores >= 0.5).astype(int), iso_scores
    )

    # Only base models that are persisted and served participate in stacking.
    stack_keys = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
    meta_X_all = np.column_stack([oof_preds[key] for key in stack_keys])
    meta_X_test = np.column_stack([test_preds[key] for key in stack_keys])
    meta_X_train = meta_X_all[covered]
    meta_y_train = y_train.to_numpy()[covered]
    covered_indices = np.flatnonzero(covered)
    meta_times = time_values[covered] if time_values is not None else None

    # Generate strictly out-of-time meta probabilities to choose the decision
    # threshold without consulting the final test labels.
    meta_oof = np.full(len(meta_y_train), np.nan)
    meta_folds = _chronological_splits(
        len(meta_X_train), n_splits=3, transaction_times=meta_times
    )
    for fold_no, (train_idx, val_idx) in enumerate(meta_folds, start=1):
        fold_meta = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            random_state=42, n_jobs=int(profile["n_jobs"]), verbosity=0,
            tree_method="hist", max_bin=64,
            **({"device": "cuda"} if profile.get("xgb_gpu") else {})
        )
        fold_meta_config = {"model": fold_meta, "gpu_enabled": bool(profile.get("xgb_gpu"))}
        fold_meta = _fit_with_fallback(
            fold_meta, meta_X_train[train_idx], meta_y_train[train_idx], fold_meta_config
        )
        fold_scores = fold_meta.predict_proba(meta_X_train[val_idx])[:, 1]
        meta_oof[val_idx] = fold_scores
        fold_labels = meta_y_train[val_idx]
        catboost_scores = oof_preds["catboost"][covered_indices[val_idx]]
        comparison = {
            "fold": int(fold_no),
            "train_rows": int(len(train_idx)),
            "validation_start_row": int(covered_indices[val_idx[0]]),
            "validation_end_row_exclusive": int(covered_indices[val_idx[-1]] + 1),
            "models": {
                "ensemble": ranking_metrics(fold_labels, fold_scores),
                "catboost": ranking_metrics(fold_labels, catboost_scores),
            },
        }
        if time_values is not None:
            comparison.update({
                "train_time_end": float(time_values[covered_indices[train_idx[-1]]]),
                "validation_time_start": float(time_values[covered_indices[val_idx[0]]]),
                "validation_time_end": float(time_values[covered_indices[val_idx[-1]]]),
            })
        temporal_cv_metrics["stack_vs_catboost"].append(comparison)
    meta_valid = np.isfinite(meta_oof)
    if meta_valid.sum() < 100:
        raise ValueError("Insufficient training data for time-aware meta cross-validation.")
    valid_scores = meta_oof[meta_valid]
    valid_labels = meta_y_train[meta_valid]
    calibration_cut = max(50, int(len(valid_scores) * 0.70))
    calibration_cut = min(calibration_cut, len(valid_scores) - 50)
    probability_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    probability_calibrator.fit(valid_scores[:calibration_cut], valid_labels[:calibration_cut])
    threshold_scores = probability_calibrator.predict(valid_scores[calibration_cut:])
    threshold_labels = valid_labels[calibration_cut:]
    threshold_result = optimize_threshold(threshold_labels, threshold_scores)
    best_thresh = float(threshold_result["optimal_threshold"])
    trained_models["decision_threshold"] = best_thresh
    trained_models["probability_calibrator"] = probability_calibrator

    meta_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        random_state=42, n_jobs=int(profile["n_jobs"]), verbosity=0,
        tree_method="hist", max_bin=64,
        **({"device": "cuda"} if profile.get("xgb_gpu") else {})
    )
    meta_config = {"model": meta_model, "gpu_enabled": bool(profile.get("xgb_gpu"))}
    meta_model = _fit_with_fallback(meta_model, meta_X_train, meta_y_train, meta_config)
    trained_models["meta_learner"] = meta_model
    ensemble_proba_raw = meta_model.predict_proba(meta_X_test)[:, 1]
    ensemble_proba = probability_calibrator.predict(ensemble_proba_raw)
    ensemble_pred = (ensemble_proba >= best_thresh).astype(int)
    results["ensemble"] = evaluate_model(
        "Temporal OOF Stacking Ensemble", y_test, ensemble_pred, ensemble_proba
    )

    probas = {
        "xgb_proba": test_preds["xgboost"],
        "lgb_proba": test_preds["lightgbm"],
        "cat_proba": test_preds["catboost"],
        "rf_proba": test_preds["random_forest"],
        "mlp_proba": test_preds["mlp"],
        "iso_scores": iso_scores,
        "ensemble_proba": ensemble_proba,
        "ensemble_proba_raw": ensemble_proba_raw,
        "ensemble_pred": ensemble_pred,
        "optimal_threshold": best_thresh,
        "oof_ensemble_proba": threshold_scores,
        "oof_ensemble_y": threshold_labels,
        "calibration_oof_ensemble_proba": valid_scores,
        "calibration_oof_ensemble_y": valid_labels,
        "threshold_optimization": threshold_result,
        "temporal_cv_metrics": temporal_cv_metrics,
    }
    return trained_models, results, probas


def evaluate_model(name, y_true, y_pred, y_proba):
    """Compute and display all metrics for a model."""
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ranking = ranking_metrics(y_true, y_proba)

    print(
        f"    -> ROC-AUC: {auc:.4f} | AP: {ranking['average_precision']:.4f} | "
        f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
    )
    print(f"    -> Confusion: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")

    return {
        'name': name,
        'auc': auc,
        'average_precision': ranking['average_precision'],
        'brier_score': ranking['brier_score'],
        'log_loss': ranking['log_loss'],
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
            'average_precision': float(res['average_precision']),
            'brier_score': float(res['brier_score']),
            'log_loss': float(res['log_loss']),
            'f1': float(res['f1']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'confusion_matrix': res['confusion_matrix'].tolist()
        }

    with open(metrics_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[SAVE] Metrics saved to {metrics_path}")
    return save_data


def save_model_artifacts(models, label_encoders, feature_names, output_dir, feature_pipeline=None):
    """Save the complete versioned model bundle used by the scoring API."""
    artifacts_dir = os.path.join(output_dir, "model_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    manifest_path = os.path.join(artifacts_dir, "bundle.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)  # Invalidate the previous bundle before replacing files.

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
    for name in ['random_forest', 'isolation_forest', 'mlp', 'scaler', 'meta_learner', 'probability_calibrator']:
        if name in models:
            with open(os.path.join(artifacts_dir, f'{name}.pkl'), 'wb') as f:
                pickle.dump(models[name], f)

    # Save label encoders
    with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)

    # Save feature names
    with open(os.path.join(artifacts_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    if feature_pipeline is None:
        raise ValueError("A fitted feature_pipeline is required to create a serving bundle.")
    with open(os.path.join(artifacts_dir, "feature_pipeline.pkl"), "wb") as f:
        pickle.dump(feature_pipeline, f)

    # Stacking input order is part of the serving contract.
    with open(os.path.join(artifacts_dir, 'stacking_config.json'), 'w') as f:
        json.dump({
            'base_models': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'mlp'],
            'threshold_source': 'time_ordered_oof',
            'score_method': 'meta_learner',
            'decision_threshold': float(models.get('decision_threshold', 0.5)),
        }, f, indent=2)

    # Written last: the API treats this manifest as proof that the bundle was
    # fully exported by this pipeline version.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 4,
            "base_models": ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"],
            "feature_count": len(feature_names),
        }, f, indent=2)

    print(f"[SAVE] Model artifacts saved to {artifacts_dir}")
