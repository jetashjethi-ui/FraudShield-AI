"""
FraudShield AI — SHAP Explainability Module
Real SHAP (SHapley Additive exPlanations) for scientifically rigorous XAI.
Generates waterfall plots and feature-level explanations per transaction.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_shap_explanations(model, X_test, feature_names, output_dir, n_samples=500):
    """
    Compute SHAP values for the best model (XGBoost/LightGBM) and generate:
    1. SHAP summary plot (global feature importance)
    2. SHAP waterfall plots for sample flagged transactions
    3. Per-transaction feature-level explanations
    """
    if not HAS_SHAP:
        print("  [SHAP] shap library not installed, skipping...")
        return None, None

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY ENGINE")
    print("=" * 70)

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Sample for speed
    rng = np.random.RandomState(42)
    if len(X_test) > n_samples:
        sample_idx = rng.choice(len(X_test), n_samples, replace=False)
        if isinstance(X_test, pd.DataFrame):
            X_sample = X_test.iloc[sample_idx]
        else:
            X_sample = X_test[sample_idx]
    else:
        X_sample = X_test
        sample_idx = np.arange(len(X_test))

    # Use TreeExplainer for tree models (fast)
    print(f"  [SHAP] Computing SHAP values for {len(X_sample):,} samples...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classifiers, shap_values might be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use fraud class
    except Exception as e:
        print(f"  [SHAP] TreeExplainer failed ({e}), trying KernelExplainer...")
        try:
            # Fallback: use a small background dataset
            bg = shap.sample(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_sample[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as e2:
            print(f"  [SHAP] KernelExplainer also failed: {e2}")
            return None, None

    # Ensure feature_names match
    if isinstance(X_sample, pd.DataFrame):
        feat_names = list(X_sample.columns)
    else:
        feat_names = feature_names[:shap_values.shape[1]] if feature_names else \
            [f"f{i}" for i in range(shap_values.shape[1])]

    print(f"  [SHAP] SHAP values computed: shape {shap_values.shape}")

    # ── 1. Global SHAP Summary (Beeswarm) ────────────────────────
    print("  [SHAP] Generating global summary plot...")
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                          max_display=20, show=False, plot_size=None)
        plt.title("FraudShield AI -- SHAP Feature Importance (Global)", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'shap_summary.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print("    -> shap_summary.png saved")
    except Exception as e:
        print(f"    -> Summary plot failed: {e}")

    # ── 2. SHAP Bar Plot (Mean absolute SHAP) ────────────────────
    print("  [SHAP] Generating mean importance bar plot...")
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                          plot_type="bar", max_display=20, show=False, plot_size=None)
        plt.title("FraudShield AI -- Mean |SHAP| Feature Importance", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'shap_bar.png'), bbox_inches='tight', dpi=150)
        plt.close()
        print("    -> shap_bar.png saved")
    except Exception as e:
        print(f"    -> Bar plot failed: {e}")

    # ── 3. Waterfall Plots for High-Risk Transactions ─────────────
    print("  [SHAP] Generating waterfall plots for flagged transactions...")
    try:
        # Find the top fraud-probability transactions
        mean_abs_shap = np.abs(shap_values).mean(axis=1)
        top_indices = np.argsort(mean_abs_shap)[-3:][::-1]  # Top 3 most explained

        for rank, idx in enumerate(top_indices):
            plt.figure(figsize=(10, 6))
            # Create shap.Explanation object
            explanation = shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
                data=X_sample.iloc[idx].values if isinstance(X_sample, pd.DataFrame)
                    else X_sample[idx],
                feature_names=feat_names
            )
            shap.waterfall_plot(explanation, max_display=12, show=False)
            plt.title(f"Transaction #{rank+1} -- SHAP Waterfall", fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'shap_waterfall_{rank+1}.png'),
                        bbox_inches='tight', dpi=150)
            plt.close()
        print(f"    -> {len(top_indices)} waterfall plots saved")
    except Exception as e:
        print(f"    -> Waterfall plots failed: {e}")

    # ── 4. Build per-transaction SHAP explanations ────────────────
    print("  [SHAP] Building per-transaction explanations...")
    shap_explanations = []
    for i in range(len(shap_values)):
        sv = shap_values[i]
        # Get top 5 features by absolute SHAP value
        top_feat_idx = np.argsort(np.abs(sv))[-5:][::-1]
        reasons = []
        for fi in top_feat_idx:
            fname = feat_names[fi] if fi < len(feat_names) else f"feature_{fi}"
            direction = "increases" if sv[fi] > 0 else "decreases"
            reasons.append(f"{fname} {direction} fraud risk (SHAP={sv[fi]:.3f})")
        shap_explanations.append(" | ".join(reasons))

    print(f"  [SHAP] Generated explanations for {len(shap_explanations)} transactions")

    return shap_values, shap_explanations
