"""
FraudShield AI — Feature Selection Module
Uses LightGBM importance to prune noisy features that hurt ensemble performance.
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def select_features(X_train, y_train, min_importance_pct=0.001, n_folds=3):
    """Drop features with near-zero importance using a quick LightGBM scan.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels
        min_importance_pct: Features below this % of total importance get dropped
        n_folds: CV folds for importance estimation
    
    Returns:
        keep_cols: list of column names to keep
        drop_cols: list of column names that were dropped
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION (LightGBM Importance)")
    print("=" * 70)

    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f"f{i}" for i in range(X_train.shape[1])]
    n_features = len(feature_names)
    print(f"  [SELECT] Scanning {n_features} features...")

    # Quick LightGBM to estimate feature importance
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, random_state=42, n_jobs=-1, verbosity=-1
    )

    # Use sample for speed
    rng = np.random.RandomState(42)
    sample_size = min(80000, len(X_train))
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)
    if hasattr(X_train, 'iloc'):
        X_sample = X_train.iloc[sample_idx]
    else:
        X_sample = X_train[sample_idx]
    y_sample = y_train.iloc[sample_idx]

    # Aggregate importance across folds
    importance_sum = np.zeros(n_features)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        if hasattr(X_sample, 'iloc'):
            X_tr, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
        else:
            X_tr, X_val = X_sample[train_idx], X_sample[val_idx]
        y_tr, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0)])
        importance_sum += model.feature_importances_

    avg_importance = importance_sum / n_folds
    total_importance = avg_importance.sum()

    # Compute relative importance
    relative_importance = avg_importance / total_importance

    # Find features to drop (below threshold)
    drop_mask = relative_importance < min_importance_pct
    drop_cols = [feature_names[i] for i in range(n_features) if drop_mask[i]]
    keep_cols = [feature_names[i] for i in range(n_features) if not drop_mask[i]]

    # Show top and bottom features
    sorted_idx = np.argsort(relative_importance)[::-1]
    print(f"\n  Top 10 most important features:")
    for i in sorted_idx[:10]:
        print(f"    {feature_names[i]:35s} {relative_importance[i]*100:.3f}%")

    if drop_cols:
        print(f"\n  Bottom features (dropped):")
        for i in sorted_idx[-min(5, len(drop_cols)):]:
            if drop_mask[i]:
                print(f"    {feature_names[i]:35s} {relative_importance[i]*100:.5f}%")

    print(f"\n  [SELECT] Keeping {len(keep_cols)}/{n_features} features (dropped {len(drop_cols)} noisy features)")
    return keep_cols, drop_cols
