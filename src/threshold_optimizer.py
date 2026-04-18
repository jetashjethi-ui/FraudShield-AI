"""
Cost-Sensitive Threshold Optimizer
Finds the optimal decision threshold using a business cost matrix instead of a naive 0.5 cutoff.
"""

import numpy as np
import json
import os


def optimize_threshold(y_true, y_probs, 
                       cost_fn=850,   # Cost of missed fraud (false negative)
                       cost_fp=25,    # Cost of false positive (customer friction)
                       cost_review=15,# Cost of manual review
                       n_thresholds=1000):
    """
    Find the threshold that minimizes total business cost.
    
    Returns dict with optimal threshold, costs, and comparison metrics.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    best_threshold = 0.5
    best_cost = float('inf')
    results = []
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        # Business cost
        total_cost = (fn * cost_fn) + (fp * cost_fp) + ((tp + fp) * cost_review)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        results.append({
            'threshold': round(float(t), 4),
            'cost': round(float(total_cost), 2),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t
    
    # Get metrics at default 0.5 and optimal threshold
    default_idx = min(range(len(results)), key=lambda i: abs(results[i]['threshold'] - 0.5))
    optimal_idx = min(range(len(results)), key=lambda i: abs(results[i]['threshold'] - best_threshold))
    
    # Annual projection (50M transactions)
    annual_txns = 50_000_000
    scale = annual_txns / max(len(y_true), 1)
    
    default_annual = results[default_idx]['cost'] * scale
    optimal_annual = results[optimal_idx]['cost'] * scale
    annual_savings = default_annual - optimal_annual
    
    output = {
        'optimal_threshold': round(float(best_threshold), 4),
        'default_threshold': 0.5,
        'cost_at_optimal': round(float(best_cost), 2),
        'cost_at_default': round(float(results[default_idx]['cost']), 2),
        'savings_vs_default': round(float(results[default_idx]['cost'] - best_cost), 2),
        'annual_savings_projected': f"₹{annual_savings:,.0f}",
        'optimal_metrics': results[optimal_idx],
        'default_metrics': results[default_idx],
        'cost_matrix': {
            'false_negative_cost': cost_fn,
            'false_positive_cost': cost_fp,
            'review_cost': cost_review,
        },
        'threshold_curve': results[::max(1, len(results)//100)],  # Sample 100 points for plotting
    }
    
    return output


def run_threshold_optimization(results_dir):
    """Load OOF predictions and run optimization."""
    
    oof_path = os.path.join(results_dir, 'oof_predictions.json')
    if not os.path.exists(oof_path):
        print("[Threshold] No OOF predictions found, skipping optimization")
        return None
    
    with open(oof_path, 'r') as f:
        oof_data = json.load(f)
    
    y_true = oof_data.get('y_true', [])
    y_probs = oof_data.get('y_probs', [])
    
    if not y_true or not y_probs:
        print("[Threshold] Empty OOF data, skipping")
        return None
    
    print(f"[Threshold] Running cost-sensitive optimization on {len(y_true)} predictions...")
    result = optimize_threshold(y_true, y_probs)
    
    # Save results
    out_path = os.path.join(results_dir, 'threshold_optimization.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[Threshold] Optimal threshold: {result['optimal_threshold']:.4f} (vs default 0.5)")
    print(f"[Threshold] Cost savings: {result['savings_vs_default']:.2f}")
    print(f"[Threshold] Projected annual savings: {result['annual_savings_projected']}")
    
    return result
