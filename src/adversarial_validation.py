"""
FraudShield AI — Adversarial Validation
Tests whether train and test sets come from the same distribution.

If a classifier can easily distinguish train from test (AUC >> 0.5), 
the model may be overfitting to train-specific patterns.
AUC ~0.5 = good (distributions match, model generalizes).
"""

import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def run_adversarial_validation(X_train, X_test, output_dir, n_splits=3):
    """
    Train a LightGBM classifier to distinguish train from test.
    AUC close to 0.5 means no data leakage / distribution mismatch.
    """
    print("\n[Adversarial Validation] Checking train/test distribution match...")

    n_train = len(X_train)
    n_test = len(X_test)

    # Convert to numpy if needed
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
    else:
        X_train_np = np.array(X_train)

    if hasattr(X_test, 'values'):
        X_test_np = X_test.values
    else:
        X_test_np = np.array(X_test)

    # Create adversarial labels: 0 = train, 1 = test
    X_all = np.vstack([X_train_np, X_test_np])
    y_adv = np.concatenate([np.zeros(n_train), np.ones(n_test)])

    # Shuffle
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_all))
    X_all = X_all[idx]
    y_adv = y_adv[idx]

    # Replace inf/nan
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

    # Use LightGBM if available, else RandomForest
    try:
        import lightgbm as lgb
        use_lgb = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        use_lgb = False

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_adv)):
        Xt, Xv = X_all[train_idx], X_all[val_idx]
        yt, yv = y_adv[train_idx], y_adv[val_idx]

        if use_lgb:
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.7, colsample_bytree=0.7,
                verbose=-1, n_jobs=-1
            )
            model.fit(Xt, yt, eval_set=[(Xv, yv)])
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            model.fit(Xt, yt)

        preds = model.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(yv, preds)
        auc_scores.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")

    mean_auc = np.mean(auc_scores)
    
    # Interpretation
    if mean_auc < 0.55:
        status = "EXCELLENT"
        interpretation = "Train and test distributions are nearly identical. No data leakage detected."
    elif mean_auc < 0.65:
        status = "GOOD"
        interpretation = "Minor distribution differences exist but are acceptable."
    elif mean_auc < 0.75:
        status = "WARNING"
        interpretation = "Moderate distribution mismatch. Review feature engineering for leakage."
    else:
        status = "CRITICAL"
        interpretation = "Severe distribution mismatch. High risk of overfitting."

    passed = mean_auc < 0.65

    result = {
        'adversarial_auc': round(float(mean_auc), 4),
        'fold_aucs': [round(float(a), 4) for a in auc_scores],
        'status': status,
        'passed': passed,
        'interpretation': interpretation,
        'n_train': n_train,
        'n_test': n_test,
    }

    # Feature importance (which features leak most)
    if use_lgb and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_leak = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]
        result['top_leaking_features'] = [
            {'index': int(i), 'importance': int(importances[i])} for i in top_leak
        ]

    # Save
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "adversarial_validation.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[Adversarial Validation] Mean AUC: {mean_auc:.4f} — {status}")
    print(f"  → {interpretation}")
    print(f"[SAVE] Results saved to {out_path}")

    return result
