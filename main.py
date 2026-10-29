"""Train, evaluate, and export the FraudShield chronological fraud model."""

import sys
import os
import time
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import warnings
warnings.filterwarnings('ignore')

# Fix Windows cp1252 encoding crash for Unicode characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.runtime_config import require_memory_headroom

if __name__ == "__main__":
    # Check before importing the dataframe/boosting stack, whose runtime
    # libraries consume several hundred MB of resident memory.
    require_memory_headroom(4.0, "loading training libraries")

import pandas as pd
import numpy as np

from src.data_loader import load_data, high_missing_columns
from src.inference_pipeline import FraudFeaturePipeline
from src.models import train_models, get_feature_importance, save_metrics, save_model_artifacts
from src.risk_scorer import compute_risk_scores, generate_explanations, build_output_table
from src.visualizer import generate_all_visualizations
from src.shap_engine import compute_shap_explanations
from src.tuner import run_tuning
from src.report_generator import generate_pdf_report
from src.autoencoder import run_autoencoder_detection
from src.feature_selector import select_features
from src.conformal import run_conformal_pipeline
from src.threshold_optimizer import run_threshold_optimization
from src.adversarial_validation import run_adversarial_validation
from src.runtime_config import training_profile
from src.evaluation import (
    chronological_segments,
    decision_metrics,
    stratified_bootstrap_intervals,
)
import gc


def main():
    start_time = time.time()
    
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

    print("=" * 70)
    print("  FraudShield AI -- Complete Pipeline")
    print("  Chronological holdout | shared train/serve features | calibrated stack")
    print("=" * 70)
    require_memory_headroom(3.0, "loading the IEEE-CIS training data")
    profile = training_profile()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(profile["n_jobs"]))
    print(
        f"  Runtime: {profile['available_ram_gb']} GiB free RAM | "
        f"{profile['gpu_free_gb']} GiB free VRAM | "
        f"GPU={'enabled' if profile['use_gpu'] else 'disabled'} | "
        f"CPU workers={profile['n_jobs']}"
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 70)
    df = load_data(DATA_DIR)
    step1_time = time.time()
    print(f"  ⏱ Step 1 completed in {step1_time - start_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: CHRONOLOGICAL HOLDOUT + TRAIN-FIT PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: CHRONOLOGICAL SPLIT & LEAKAGE-SAFE FEATURES")
    print("=" * 70)
    if "TransactionDT" not in df.columns:
        raise ValueError("TransactionDT is required for a chronological holdout.")
    df.sort_values("TransactionDT", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)
    split_at = int(len(df) * 0.8)
    train_raw = df.iloc[:split_at]
    test_raw = df.iloc[split_at:]
    dropped_columns = high_missing_columns(train_raw, threshold=0.70)
    if dropped_columns:
        df.drop(columns=dropped_columns, inplace=True)
        train_raw = df.iloc[:split_at]
        test_raw = df.iloc[split_at:]
        print(f"  Dropped {len(dropped_columns)} columns using training-period missingness only")
    y_train = train_raw["isFraud"].astype(int).reset_index(drop=True)
    y_test = test_raw["isFraud"].astype(int).reset_index(drop=True)
    train_times = pd.to_numeric(train_raw["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
    test_times = pd.to_numeric(test_raw["TransactionDT"], errors="coerce").to_numpy(dtype=np.float64)
    test_meta = test_raw[[col for col in ("TransactionID", "TransactionAmt", "isFraud", "ProductCD") if col in test_raw]].reset_index(drop=True)

    feature_pipeline = FraudFeaturePipeline()
    X_train = feature_pipeline.fit_transform(train_raw)
    # Holdout rows receive a local copy of the training history; their history
    # must not leak into the serialized serving state.
    X_test = feature_pipeline.transform(test_raw, update_state=False)
    feature_names = feature_pipeline.get_feature_names_out()
    del train_raw, test_raw, df
    gc.collect()
    print(f"  Train period: {len(X_train):,} transactions")
    print(f"  Later holdout: {len(X_test):,} transactions")
    print(f"  Features fit from training period only: {len(feature_names)}")
    step2_time = time.time()

    # Selection and tuning happen only on the initial warm-up window. Later
    # temporal OOF folds therefore cannot influence feature choices or their
    # hyperparameters before they are scored.
    first_fold_boundary = train_times[0] + (train_times[-1] - train_times[0]) / 5
    warmup_rows = int(np.searchsorted(train_times, first_fold_boundary, side="right"))
    warmup_rows = max(200, warmup_rows)
    warmup_rows = min(warmup_rows, len(X_train))
    X_warmup = X_train.iloc[:warmup_rows]
    y_warmup = y_train.iloc[:warmup_rows]
    keep_cols, drop_cols = select_features(X_warmup, y_warmup)
    if drop_cols:
        X_train.drop(columns=drop_cols, inplace=True)
        X_test.drop(columns=drop_cols, inplace=True)
        feature_names = keep_cols
        print(f"  [SELECT] Reduced features: {len(keep_cols)}")
    X_warmup = X_train.iloc[:warmup_rows]

    # Adversarial Validation — verify train/test match
    print("\n  [Adversarial Validation] Checking train/test distribution...")
    try:
        adv_result = run_adversarial_validation(X_train, X_test, OUTPUT_DIR)
    except Exception as e:
        print(f"  [Adversarial Validation] Skipped: {e}")

    step3_time = time.time()
    print(f"  Step 3 completed in {step3_time - step2_time:.1f}s")

    # ===============================================================
    # STEP 3.5: OPTUNA HYPERPARAMETER TUNING
    # ===============================================================
    print("\n" + "=" * 70)
    print("STEP 3.5: OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)
    require_memory_headroom(2.0, "model tuning")
    tuned_params = run_tuning(X_warmup, y_warmup, OUTPUT_DIR, n_trials=150, profile=profile)
    del X_warmup, y_warmup
    gc.collect()

    step3b_time = time.time()
    print(f"  Step 3.5 completed in {step3b_time - step3_time:.1f}s")

    # ===============================================================
    # STEP 4: MODEL TRAINING
    # ===============================================================
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    models, results, predictions = train_models(
        X_train, y_train, X_test, y_test,
        tuned_params=tuned_params,
        transaction_times=train_times,
    )
    
    # Feature importance
    importances = get_feature_importance(models, feature_names)
    
    # Save metrics and model artifacts
    save_metrics(results, RESULTS_DIR)
    save_model_artifacts(models, {}, feature_names, OUTPUT_DIR, feature_pipeline)

    temporal_cv = predictions.get("temporal_cv_metrics", {})
    with open(os.path.join(RESULTS_DIR, "temporal_cv_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(temporal_cv, handle, indent=2, allow_nan=False)

    holdout_evaluation = {
        "evaluation_split": "final chronological 20% holdout; not used for fitting, tuning, calibration, or threshold selection",
        "rows": int(len(y_test)),
        "fraud_rate": float(y_test.mean()),
        "models": {},
        "chronological_segments": {},
    }
    for model_key in ("catboost", "ensemble"):
        score_key = {"catboost": "cat_proba", "ensemble": "ensemble_proba"}[model_key]
        scores = np.asarray(predictions[score_key], dtype=float)
        threshold = float(predictions["optimal_threshold"]) if model_key == "ensemble" else 0.5
        model_evaluation = decision_metrics(y_test, scores, threshold)
        model_evaluation["confidence_intervals"] = stratified_bootstrap_intervals(
            y_test, scores, n_bootstrap=200, seed=42 if model_key == "ensemble" else 43
        )
        holdout_evaluation["models"][model_key] = model_evaluation
        holdout_evaluation["chronological_segments"][model_key] = chronological_segments(
            y_test, scores, test_times, n_segments=4
        )
    with open(os.path.join(RESULTS_DIR, "holdout_evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(holdout_evaluation, handle, indent=2, allow_nan=False)

    step4_time = time.time()
    print(f"  ⏱ Step 4 completed in {step4_time - step3_time:.1f}s")

    # ===============================================================
    # STEP 4.5: DEEP LEARNING AUTOENCODER
    # ===============================================================
    ae_model, ae_results = run_autoencoder_detection(
        X_train, y_train, X_test, y_test, OUTPUT_DIR
    )

    step4b_time = time.time()
    print(f"  Step 4.5 completed in {step4b_time - step4_time:.1f}s")

    # ===============================================================
    # STEP 4.6: CONFORMAL PREDICTION + CALIBRATION
    # ===============================================================
    # Use training OOF ensemble probabilities as calibration set
    oof_proba = predictions.get('oof_ensemble_proba', None)
    if oof_proba is not None:
        calibrated_proba, is_uncertain, confidence, conformal_results = run_conformal_pipeline(
            predictions['oof_ensemble_y'], y_test, oof_proba,
            predictions['ensemble_proba'], OUTPUT_DIR
        )
        predictions['calibrated_proba'] = calibrated_proba
        predictions['is_uncertain'] = is_uncertain
        predictions['confidence'] = confidence

    step4c_time = time.time()
    print(f"  Step 4.6 completed in {step4c_time - step4b_time:.1f}s")

    # ===============================================================
    # STEP 4.7: SAVE OOF PREDICTIONS + THRESHOLD OPTIMIZATION
    # ===============================================================
    if oof_proba is not None:
        oof_data = {
            'y_true': np.asarray(predictions['oof_ensemble_y']).tolist(),
            'y_probs': np.asarray(oof_proba).tolist(),
        }
        oof_path = os.path.join(RESULTS_DIR, 'oof_predictions.json')
        with open(oof_path, 'w') as f:
            json.dump(oof_data, f)
        print(f"[SAVE] OOF predictions saved to {oof_path}")

    cost_weights = {
        "cost_fn": float(os.getenv("FRAUD_COST_FALSE_NEGATIVE", "850")),
        "cost_fp": float(os.getenv("FRAUD_COST_FALSE_POSITIVE", "25")),
        "cost_review": float(os.getenv("FRAUD_COST_MANUAL_REVIEW", "15")),
    }
    if any(value < 0 or not np.isfinite(value) for value in cost_weights.values()):
        raise ValueError("Fraud cost weights must be finite, non-negative numbers.")
    threshold_result = run_threshold_optimization(RESULTS_DIR, **cost_weights)
    step4d_time = time.time()
    print(f"  Step 4.7 completed in {step4d_time - step4c_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: RISK SCORING & EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: RISK SCORING & EXPLAINABILITY ENGINE")
    print("=" * 70)
    
    # Use test set for scoring output
    # Attach output metadata to the existing holdout frame instead of copying
    # its full feature matrix a second time.
    df_test = X_test
    for col in ("TransactionID", "TransactionAmt", "isFraud", "ProductCD"):
        if col in test_meta.columns:
            df_test[col] = test_meta[col].to_numpy(copy=False)
    
    risk_score, risk_category, auth_recommendation = compute_risk_scores(df_test, predictions)
    
    print("\n[XAI] Generating explanations (sampling for speed)...")
    # Generate explanations for a sample (full set would be too slow)
    sample_size = min(5000, len(df_test))
    sample_idx = np.random.RandomState(42).choice(len(df_test), sample_size, replace=False)
    sample_df = df_test.iloc[sample_idx].reset_index(drop=True)
    sample_explanations = generate_explanations(sample_df, risk_score[sample_idx])
    
    # For the full output, use a placeholder explanation based on risk category
    full_explanations = []
    sampled_explanations = dict(zip(sample_idx.tolist(), sample_explanations))
    for i in range(len(df_test)):
        if i in sampled_explanations:
            full_explanations.append(sampled_explanations[i])
        else:
            full_explanations.append(f"Risk Score: {risk_score[i]:.0f} | Category: {risk_category[i]}")
    
    # Build output table
    output_df = build_output_table(
        df_test, risk_score, risk_category, auth_recommendation,
        full_explanations, predictions
    )
    
    # Save scored output
    output_path = os.path.join(RESULTS_DIR, "scored_transactions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"[SAVE] Scored transactions saved to {output_path}")
    
    # Save a sample for presentation
    sample_output = output_df[output_df['risk_category'].isin(['RED_BLOCK', 'ORANGE_BIOMETRIC'])].head(50)
    sample_path = os.path.join(RESULTS_DIR, "sample_flagged_transactions.csv")
    sample_output.to_csv(sample_path, index=False)
    print(f"[SAVE] Sample flagged transactions saved to {sample_path}")

    step5_time = time.time()
    print(f"  ⏱ Step 5 completed in {step5_time - step4_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)
    generate_all_visualizations(results, importances, feature_names, output_df, df_test, OUTPUT_DIR)

    step6_time = time.time()
    print(f"  Step 6 completed in {step6_time - step5_time:.1f}s")

    # ===============================================================
    # STEP 7: SHAP EXPLAINABILITY
    # ===============================================================
    print("\n" + "=" * 70)
    print("STEP 7: SHAP EXPLAINABILITY")
    print("=" * 70)

    # Use best tree model for SHAP (LightGBM or XGBoost)
    shap_model = models.get('lightgbm', models.get('xgboost'))
    if shap_model is not None:
        shap_values, shap_explanations = compute_shap_explanations(
            shap_model, X_test[feature_names], feature_names, OUTPUT_DIR, n_samples=500
        )
    else:
        print("  [SHAP] No tree model found, skipping...")

    step7_time = time.time()
    print(f"  Step 7 completed in {step7_time - step6_time:.1f}s")

    # ===============================================================
    # STEP 8: AUTO-GENERATED PDF REPORT
    # ===============================================================
    pdf_path = generate_pdf_report(OUTPUT_DIR)

    step8_time = time.time()
    print(f"  Step 8 completed in {step8_time - step7_time:.1f}s")

    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "imbalanced-learn", "optuna"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    run_metadata = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - start_time, 2),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "dataset": {
            "name": "IEEE-CIS Fraud Detection",
            "total_rows": int(len(y_train) + len(y_test)),
            "train_rows": int(len(y_train)),
            "chronological_holdout_rows": int(len(y_test)),
            "train_fraud_rate": float(y_train.mean()),
            "holdout_fraud_rate": float(y_test.mean()),
            "split_fraction": 0.8,
        },
        "features": {
            "count": len(feature_names),
            "history_features": len(feature_pipeline.history_feature_names_),
            "initial_warmup_rows_for_selection_and_tuning": warmup_rows,
            "high_missing_columns_dropped": len(dropped_columns),
            "low_importance_features_dropped": len(drop_cols),
        },
        "imbalance_handling": {
            "smote_models": ["xgboost", "mlp"],
            "native_class_weight_models": ["lightgbm", "catboost", "random_forest"],
        },
        "runtime": profile,
        "decision_policy_cost_weights": cost_weights,
        "model_metrics_file": "model_metrics.json",
        "model_bundle_dir": "../model_artifacts",
    }
    metadata_path = os.path.join(RESULTS_DIR, "training_run.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)
    print(f"[SAVE] Training run metadata saved to {metadata_path}")

    # ===============================================================
    # FINAL SUMMARY
    # ===============================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  FRAUDSHIELD AI -- PIPELINE COMPLETE!")
    print("=" * 70)
    
    print(f"\n  Total execution time: {total_time/60:.1f} minutes")
    
    print(f"\n  MODEL PERFORMANCE:")
    for model_name, res in results.items():
        print(f"    {res['name']:20s} | ROC-AUC: {res['auc']:.4f} | AP: {res['average_precision']:.4f} | F1: {res['f1']:.4f} | "
              f"Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f}")
    
    print(f"\n  RISK SCORING:")
    for cat in ['GREEN_APPROVE', 'YELLOW_PIN_VERIFY', 'ORANGE_BIOMETRIC', 'RED_BLOCK']:
        count = (output_df['risk_category'] == cat).sum()
        pct = count / len(output_df) * 100
        print(f"    {cat:20s}: {count:>8,} ({pct:5.1f}%)")
    
    print(f"\n  OUTPUT FILES:")
    print(f"    outputs/results/model_metrics.json")
    print(f"    outputs/results/scored_transactions.csv")
    print(f"    outputs/results/sample_flagged_transactions.csv")
    print(f"    outputs/visualizations/ (charts)")

    cv_comparison = temporal_cv.get("stack_vs_catboost", [])
    if cv_comparison:
        aggregate = {}
        for candidate in ("catboost", "ensemble"):
            values = [fold["models"][candidate]["average_precision"] for fold in cv_comparison]
            aggregate[candidate] = float(np.mean(values))
        preferred = max(aggregate, key=aggregate.get)
        print(
            f"\n  TRAINING-PERIOD CV LEADER (mean average precision): "
            f"{preferred} ({aggregate[preferred]:.4f}); final holdout was not used to select it."
        )
    
    # Business impact
    ensemble = results['ensemble']
    cm = ensemble['confusion_matrix']
    caught_pct = ensemble['recall'] * 100
    fp_per_1000 = cm[0][1] / (cm[0][0] + cm[0][1]) * 1000
    print(f"\n  BUSINESS IMPACT (for your pitch):")
    print(f"    \"Catches {caught_pct:.0f}% of all fraud\"")
    print(f"    \"Only inconveniences {fp_per_1000:.1f} out of every 1,000 legitimate customers\"")
    
    print(f"\n  Pipeline finished successfully.")


if __name__ == "__main__":
    main()
