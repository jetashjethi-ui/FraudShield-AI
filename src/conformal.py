"""
Conformal Prediction + Probability Calibration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cutting-edge uncertainty quantification for fraud detection.
Instead of just "fraud / not fraud", provides:
  - Calibrated probability (actually matches real-world fraud rate)
  - Confidence interval for each prediction
  - Uncertainty flag for borderline cases → route to human review

This is what production-grade ML looks like.
"""

import numpy as np
import json
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def calibrate_probabilities(y_cal, proba_cal, proba_test):
    """Isotonic Regression calibration.
    
    Ensures that when the model says '80% fraud chance',
    it actually IS fraud 80% of the time.
    
    Banks REQUIRE calibrated probabilities for risk management.
    """
    print("\n  [Calibration] Isotonic Regression probability calibration...")
    
    # Fit isotonic regression on calibration set
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(proba_cal, y_cal)
    
    # Calibrate test probabilities
    calibrated_proba = iso.predict(proba_test)
    
    # Measure improvement
    brier_before = brier_score_loss(y_cal, proba_cal)
    proba_cal_calibrated = iso.predict(proba_cal)
    brier_after = brier_score_loss(y_cal, proba_cal_calibrated)
    
    print(f"    -> Brier Score (before): {brier_before:.4f}")
    print(f"    -> Brier Score (after):  {brier_after:.4f}")
    print(f"    -> Calibration improvement: {((brier_before - brier_after) / brier_before * 100):.1f}%")
    
    return calibrated_proba, iso


def conformal_prediction(y_cal, proba_cal, proba_test, alpha=0.05):
    """Split Conformal Prediction for uncertainty quantification.
    
    Provides a guaranteed coverage: at least (1-alpha)% of predictions
    contain the true label. Default alpha=0.05 → 95% coverage.
    
    For each test sample, outputs:
    - prediction_set: which classes are possible (fraud, not-fraud, or BOTH)
    - is_uncertain: True if both classes are possible → needs human review
    - confidence: how confident we are (1 = certain, 0 = uncertain)
    """
    print(f"\n  [Conformal] Split Conformal Prediction (α={alpha}, target coverage={1-alpha:.0%})...")
    
    n_cal = len(y_cal)
    
    # Compute nonconformity scores on calibration set
    # Score = 1 - probability of the TRUE class
    scores = np.where(
        y_cal == 1,
        1 - proba_cal,      # For fraud: 1 - P(fraud)
        proba_cal            # For not-fraud: P(fraud) = 1 - P(not-fraud)
    )
    
    # Compute quantile threshold (with finite-sample correction)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = np.quantile(scores, q_level)
    
    print(f"    -> Calibration samples: {n_cal:,}")
    print(f"    -> Nonconformity threshold: {threshold:.4f}")
    
    # Build prediction sets for test data
    # Include class 1 (fraud) if: 1 - P(fraud) <= threshold
    # Include class 0 (not-fraud) if: P(fraud) <= threshold
    include_fraud = (1 - proba_test) <= threshold
    include_legit = proba_test <= threshold
    
    # Classify prediction sets
    n_test = len(proba_test)
    prediction_sets = []
    is_uncertain = np.zeros(n_test, dtype=bool)
    confidence = np.zeros(n_test)
    
    for i in range(n_test):
        if include_fraud[i] and include_legit[i]:
            # Both classes possible → UNCERTAIN
            prediction_sets.append({0, 1})
            is_uncertain[i] = True
            confidence[i] = max(proba_test[i], 1 - proba_test[i])
        elif include_fraud[i]:
            prediction_sets.append({1})
            confidence[i] = proba_test[i]
        elif include_legit[i]:
            prediction_sets.append({0})
            confidence[i] = 1 - proba_test[i]
        else:
            # Empty set (rare) → default to higher probability class
            prediction_sets.append({1} if proba_test[i] > 0.5 else {0})
            confidence[i] = max(proba_test[i], 1 - proba_test[i])
    
    # Statistics
    n_certain_fraud = sum(1 for s in prediction_sets if s == {1})
    n_certain_legit = sum(1 for s in prediction_sets if s == {0})
    n_uncertain = is_uncertain.sum()
    
    print(f"    -> Certain FRAUD:     {n_certain_fraud:>8,} ({n_certain_fraud/n_test*100:5.1f}%)")
    print(f"    -> Certain LEGIT:     {n_certain_legit:>8,} ({n_certain_legit/n_test*100:5.1f}%)")
    print(f"    -> UNCERTAIN (human): {n_uncertain:>8,} ({n_uncertain/n_test*100:5.1f}%)")
    print(f"    -> Average confidence: {confidence.mean():.4f}")
    
    return prediction_sets, is_uncertain, confidence, threshold


def run_conformal_pipeline(y_train, y_test, ensemble_proba_train, ensemble_proba_test, output_dir):
    """Full conformal prediction + calibration pipeline.
    
    Uses the training OOF predictions as the calibration set.
    """
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION + PROBABILITY CALIBRATION")
    print("=" * 70)
    
    # Step 1: Calibrate probabilities
    calibrated_proba, calibrator = calibrate_probabilities(
        y_train, ensemble_proba_train, ensemble_proba_test
    )
    
    # Step 2: Conformal prediction with calibrated probabilities
    prediction_sets, is_uncertain, confidence, threshold = conformal_prediction(
        y_train, ensemble_proba_train, calibrated_proba, alpha=0.05
    )
    
    # Step 3: Evaluate calibrated AUC
    auc_raw = roc_auc_score(y_test, ensemble_proba_test)
    auc_calibrated = roc_auc_score(y_test, calibrated_proba)
    print(f"\n    -> Raw AUC:        {auc_raw:.4f}")
    print(f"    -> Calibrated AUC: {auc_calibrated:.4f}")
    
    # Step 4: Save results
    results = {
        'calibration': {
            'brier_before': float(brier_score_loss(y_test, ensemble_proba_test)),
            'brier_after': float(brier_score_loss(y_test, calibrated_proba)),
            'auc_raw': float(auc_raw),
            'auc_calibrated': float(auc_calibrated),
        },
        'conformal': {
            'alpha': 0.05,
            'coverage_target': 0.95,
            'threshold': float(threshold),
            'certain_fraud_pct': float(sum(1 for s in prediction_sets if s == {1}) / len(y_test)),
            'certain_legit_pct': float(sum(1 for s in prediction_sets if s == {0}) / len(y_test)),
            'uncertain_pct': float(is_uncertain.sum() / len(y_test)),
            'avg_confidence': float(confidence.mean()),
        }
    }
    
    results_path = os.path.join(output_dir, "results", "conformal_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  [SAVE] Conformal results saved to {results_path}")
    
    return calibrated_proba, is_uncertain, confidence, results
