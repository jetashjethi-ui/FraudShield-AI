"""
FraudShield AI — Adversarial Validation
Tests whether train and test sets come from the same distribution.

If a classifier can distinguish train from the later holdout (AUC >> 0.5),
feature distributions have shifted. This is not a test for label leakage.
"""

import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def run_adversarial_validation(X_train, X_test, output_dir, n_splits=3, sample_per_period=20000):
    """
    Train a LightGBM classifier to distinguish train from test.
    AUC near 0.5 means the feature distributions are hard to distinguish.
    It cannot establish that labels or features are leakage-free.
    """
    print("\n[Adversarial Validation] Checking train/test distribution match...")

    n_train = len(X_train)
    n_test = len(X_test)
    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(X_train.shape[1])]
    original_n_train, original_n_test = n_train, n_test

    # Sample each period before concatenation. Building a full train+holdout
    # matrix duplicated more than a GB on the full IEEE-CIS feature table.
    rng = np.random.RandomState(42)
    n_train_sample = min(n_train, sample_per_period)
    n_test_sample = min(n_test, sample_per_period)
    train_idx = np.linspace(0, n_train - 1, n_train_sample, dtype=int)
    test_idx = np.linspace(0, n_test - 1, n_test_sample, dtype=int)
    X_train_np = X_train.iloc[train_idx].to_numpy(dtype=np.float32, copy=False) if hasattr(X_train, 'iloc') else np.asarray(X_train, dtype=np.float32)[train_idx]
    X_test_np = X_test.iloc[test_idx].to_numpy(dtype=np.float32, copy=False) if hasattr(X_test, 'iloc') else np.asarray(X_test, dtype=np.float32)[test_idx]

    # Per-feature population stability index complements the global
    # adversarial AUC and makes the drift report actionable by field name.
    psi_by_feature = {}
    quantiles = np.linspace(0.0, 1.0, 11)
    for column_index, feature_name in enumerate(feature_names):
        train_values = X_train_np[:, column_index]
        test_values = X_test_np[:, column_index]
        train_values = train_values[np.isfinite(train_values)]
        test_values = test_values[np.isfinite(test_values)]
        if not len(train_values) or not len(test_values):
            psi_by_feature[feature_name] = 0.0
            continue
        edges = np.unique(np.quantile(train_values, quantiles))
        if len(edges) < 2:
            baseline = train_values[0]
            train_counts = np.array([np.sum(train_values == baseline), np.sum(train_values != baseline)], dtype=float)
            test_counts = np.array([np.sum(test_values == baseline), np.sum(test_values != baseline)], dtype=float)
        else:
            internal_edges = edges[1:-1]
            if len(internal_edges) == 0:
                internal_edges = np.array([(edges[0] + edges[-1]) / 2.0])
            bins = np.concatenate(([-np.inf], internal_edges, [np.inf]))
            train_counts = np.histogram(train_values, bins=bins)[0].astype(float)
            test_counts = np.histogram(test_values, bins=bins)[0].astype(float)
        epsilon = 1.0 / max(len(train_values), len(test_values), 1)
        train_share = np.clip(train_counts / max(train_counts.sum(), 1.0), epsilon, None)
        test_share = np.clip(test_counts / max(test_counts.sum(), 1.0), epsilon, None)
        psi_by_feature[feature_name] = float(np.sum((test_share - train_share) * np.log(test_share / train_share)))

    # Create adversarial labels: 0 = train, 1 = test
    n_train, n_test = n_train_sample, n_test_sample
    X_all = np.vstack([X_train_np, X_test_np]).astype(np.float32, copy=False)
    del X_train_np, X_test_np
    y_adv = np.concatenate([np.zeros(n_train, dtype=np.int8), np.ones(n_test, dtype=np.int8)])

    # Shuffle
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
                verbose=-1, n_jobs=4
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
        interpretation = "No strong train/holdout feature shift detected. This does not test label leakage."
    elif mean_auc < 0.65:
        status = "GOOD"
        interpretation = "Minor train/holdout feature shift detected; inspect affected features."
    elif mean_auc < 0.75:
        status = "WARNING"
        interpretation = "Moderate train/holdout feature shift detected; inspect drift and split design."
    else:
        status = "CRITICAL"
        interpretation = "Severe train/holdout feature shift detected; reported metrics may not transfer."

    passed = bool(mean_auc < 0.65)

    result = {
        'adversarial_auc': round(float(mean_auc), 4),
        'fold_aucs': [round(float(a), 4) for a in auc_scores],
        'status': status,
        'passed': passed,
        'interpretation': interpretation,
        'n_train': original_n_train,
        'n_test': original_n_test,
        'sampled_n_train': n_train,
        'sampled_n_test': n_test,
        'feature_psi': psi_by_feature,
    }

    # Feature importance (which features most distinguish the split)
    if use_lgb and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_leak = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]
        result['top_shift_features'] = [
            {
                'index': int(i),
                'name': feature_names[i],
                'importance': int(importances[i]),
                'psi': float(psi_by_feature.get(feature_names[i], 0.0)),
            }
            for i in top_leak
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
