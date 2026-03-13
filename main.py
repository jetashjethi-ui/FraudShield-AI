"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FraudShield AI — Adaptive Real-Time Fraud Detection                    ║
║  with Explainable Risk Intelligence                                     ║
║                                                                          ║
║  15 Detection Layers | 4-Model Ensemble | Explainable AI              ║
║  Financial Services Hackathon — March 13-14, 2026                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.feature_engine import build_all_features
from src.models import prepare_features, apply_smote, train_models, get_feature_importance, save_metrics, save_model_artifacts
from src.risk_scorer import compute_risk_scores, generate_explanations, build_output_table
from src.visualizer import generate_all_visualizations


def main():
    start_time = time.time()
    
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

    print("╔" + "═" * 68 + "╗")
    print("║  FraudShield AI — Complete Pipeline                                ║")
    print("║  15 Detection Layers | 4-Model Ensemble | Explainable AI            ║")
    print("╚" + "═" * 68 + "╝")

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
    # STEP 2: FEATURE ENGINEERING (11 LAYERS)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING (15 DETECTION LAYERS)")
    print("=" * 70)
    df = build_all_features(df)
    step2_time = time.time()
    print(f"  ⏱ Step 2 completed in {step2_time - step1_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: PREPARE & SPLIT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: PREPARE FEATURES & TRAIN/TEST SPLIT")
    print("=" * 70)
    X, y, feature_names, label_encoders = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    # Apply SMOTE on training set only
    X_train_res, y_train_res = apply_smote(X_train, y_train, fraud_ratio=0.15)

    step3_time = time.time()
    print(f"  ⏱ Step 3 completed in {step3_time - step2_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    models, results, predictions = train_models(X_train_res, y_train_res, X_test, y_test)
    
    # Feature importance
    importances = get_feature_importance(models, feature_names)
    
    # Save metrics and model artifacts
    save_metrics(results, RESULTS_DIR)
    save_model_artifacts(models, label_encoders, feature_names, OUTPUT_DIR)

    step4_time = time.time()
    print(f"  ⏱ Step 4 completed in {step4_time - step3_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: RISK SCORING & EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: RISK SCORING & EXPLAINABILITY ENGINE")
    print("=" * 70)
    
    # Use test set for scoring output
    df_test = df.iloc[X_test.index].reset_index(drop=True)
    
    risk_score, risk_category, auth_recommendation = compute_risk_scores(df_test, predictions)
    
    print("\n[XAI] Generating explanations (sampling for speed)...")
    # Generate explanations for a sample (full set would be too slow)
    sample_size = min(5000, len(df_test))
    sample_idx = np.random.RandomState(42).choice(len(df_test), sample_size, replace=False)
    sample_df = df_test.iloc[sample_idx].reset_index(drop=True)
    sample_explanations = generate_explanations(sample_df, risk_score[sample_idx])
    
    # For the full output, use a placeholder explanation based on risk category
    full_explanations = []
    for i in range(len(df_test)):
        if i in set(sample_idx):
            pos = list(sample_idx).index(i)
            full_explanations.append(sample_explanations[pos])
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
    print(f"  ⏱ Step 6 completed in {step6_time - step5_time:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  FRAUDSHIELD AI — PIPELINE COMPLETE!                              ║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\n  Total execution time: {total_time/60:.1f} minutes")
    
    print(f"\n  MODEL PERFORMANCE:")
    for model_name, res in results.items():
        print(f"    {res['name']:20s} | AUC: {res['auc']:.4f} | F1: {res['f1']:.4f} | "
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
    print(f"    outputs/visualizations/ (10 charts)")
    
    best = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n  BEST MODEL: {best[1]['name']} (AUC: {best[1]['auc']:.4f})")
    
    # Business impact
    ensemble = results['ensemble']
    cm = ensemble['confusion_matrix']
    caught_pct = ensemble['recall'] * 100
    fp_per_1000 = cm[0][1] / (cm[0][0] + cm[0][1]) * 1000
    print(f"\n  BUSINESS IMPACT (for your pitch):")
    print(f"    \"Catches {caught_pct:.0f}% of all fraud\"")
    print(f"    \"Only inconveniences {fp_per_1000:.1f} out of every 1,000 legitimate customers\"")
    
    print(f"\n  GO WIN THIS HACKATHON! 🏆")


if __name__ == "__main__":
    main()
