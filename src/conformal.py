"""Probability calibration and split-conformal prediction utilities.

Coverage claims rely on exchangeability. A time-ordered fraud dataset may
violate that assumption, so measured holdout coverage must be reported and
the nominal level must not be presented as a guarantee under drift.
"""

import numpy as np
import json
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def calibrate_probabilities(y_cal, proba_cal, proba_test):
    """Isotonic Regression calibration.
    
    Fits an isotonic mapping on a dedicated calibration segment. This does not
    guarantee calibration under population shift or future drift.
    """
    print("\n  [Calibration] Isotonic Regression probability calibration...")
    
    # Fit isotonic regression on calibration set
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(proba_cal, y_cal)
    
    # Calibrate both the independent calibration and test predictions.
    calibrated_cal = iso.predict(proba_cal)
    calibrated_proba = iso.predict(proba_test)
    
    # Measure improvement
    brier_before = brier_score_loss(y_cal, proba_cal)
    brier_after = brier_score_loss(y_cal, calibrated_cal)
    
    print(f"    -> Brier Score (before): {brier_before:.4f}")
    print(f"    -> Brier Score (after):  {brier_after:.4f}")
    improvement = 0.0 if brier_before <= 0 else (brier_before - brier_after) / brier_before * 100
    print(f"    -> Calibration Brier change: {improvement:.1f}%")
    
    return calibrated_cal, calibrated_proba, iso


def conformal_prediction(y_cal, proba_cal, proba_test, alpha=0.05):
    """Split Conformal Prediction for uncertainty quantification.
    
    Under exchangeability, split conformal targets marginal coverage
    1-alpha. Temporal drift can invalidate that assumption.
    
    For each test sample, outputs:
    - prediction_set: which classes are possible (fraud, not-fraud, or BOTH)
    - is_uncertain: True if both classes are possible → needs human review
    - confidence: how confident we are (1 = certain, 0 = uncertain)
    """
    print(f"\n  [Conformal] Split Conformal Prediction (α={alpha}, target coverage={1-alpha:.0%})...")

    y_cal = np.asarray(y_cal, dtype=int)
    proba_cal = np.asarray(proba_cal, dtype=float)
    proba_test = np.asarray(proba_test, dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(y_cal) == 0 or len(proba_test) == 0 or len(y_cal) != len(proba_cal):
        raise ValueError("Calibration labels/probabilities and test probabilities must be non-empty and aligned")
    if not (np.isfinite(proba_cal).all() and np.isfinite(proba_test).all()):
        raise ValueError("Conformal prediction probabilities must be finite")
    if ((proba_cal < 0) | (proba_cal > 1)).any() or ((proba_test < 0) | (proba_test > 1)).any():
        raise ValueError("Conformal prediction probabilities must be in [0, 1]")
    
    n_cal = len(y_cal)
    
    # Compute nonconformity scores on calibration set
    # Score = 1 - probability of the TRUE class
    scores = np.where(
        y_cal == 1,
        1 - proba_cal,      # For fraud: 1 - P(fraud)
        proba_cal            # For not-fraud: P(fraud) = 1 - P(not-fraud)
    )
    
    # Compute quantile threshold (with finite-sample correction)
    order = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    threshold = float(np.partition(scores, order)[order])
    
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
    """Evaluate conformal sets using independent, already-calibrated OOF data.

    The probabilities passed here must be transformed by a calibrator fitted
    on a different, earlier segment of the training period.
    """
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION + PROBABILITY CALIBRATION")
    print("=" * 70)
    
    calibrated_cal = np.asarray(ensemble_proba_train, dtype=float)
    calibrated_proba = np.asarray(ensemble_proba_test, dtype=float)

    # Step 1: conformal prediction on a later OOF calibration window.
    prediction_sets, is_uncertain, confidence, threshold = conformal_prediction(
        y_train, calibrated_cal, calibrated_proba, alpha=0.05
    )
    
    # Evaluate the calibrated holdout probabilities.
    auc_calibrated = roc_auc_score(y_test, calibrated_proba)
    print(f"\n    -> Calibrated holdout AUC: {auc_calibrated:.4f}")
    
    # Step 4: Save results
    empirical_coverage = float(np.mean([
        int(label) in prediction_set for label, prediction_set in zip(y_test, prediction_sets)
    ]))
    results = {
        'calibration': {
            'brier_after': float(brier_score_loss(y_test, calibrated_proba)),
            'auc_calibrated': float(auc_calibrated),
        },
        'conformal': {
            'alpha': 0.05,
            'coverage_target': 0.95,
            'empirical_test_coverage': empirical_coverage,
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
