"""Imbalance-aware evaluation helpers for chronological fraud scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_metrics(y_true, scores) -> dict[str, float | int]:
    """Return threshold-free metrics plus probability calibration losses."""
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != p.size or y.size == 0:
        raise ValueError("Labels and prediction scores must be non-empty and have equal length.")
    if not np.isfinite(p).all():
        raise ValueError("Prediction scores contain non-finite values.")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    if np.unique(y).size != 2:
        return {
            "rows": int(y.size),
            "fraud_rate": float(y.mean()),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
    return {
        "rows": int(y.size),
        "fraud_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def decision_metrics(y_true, scores, threshold: float) -> dict[str, float | int | list[list[int]]]:
    """Add threshold-specific outcomes and fixed-capacity review metrics."""
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    pred = p >= float(threshold)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    positive_count = max(int(np.sum(y == 1)), 1)
    ranking = ranking_metrics(y, p)
    ranking.update({
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "review_budgets": {},
    })

    order = np.argsort(-p, kind="stable")
    for fraction in (0.01, 0.02, 0.05):
        review_count = min(len(y), max(1, int(np.ceil(len(y) * fraction))))
        caught = int(np.sum(y[order[:review_count]] == 1))
        ranking["review_budgets"][f"top_{int(fraction * 100)}pct"] = {
            "reviewed": review_count,
            "fraud_found": caught,
            "precision_at_capacity": float(caught / review_count),
            "recall_at_capacity": float(caught / positive_count),
        }
    return ranking


def stratified_bootstrap_intervals(
    y_true,
    scores,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile intervals that preserve the observed class counts."""
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Bootstrap intervals require both classes in the holdout.")

    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "average_precision": [], "brier_score": []}
    for _ in range(int(n_bootstrap)):
        idx = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        rng.shuffle(idx)
        metrics = ranking_metrics(y[idx], p[idx])
        for name in values:
            values[name].append(float(metrics[name]))

    intervals = {}
    for name, samples in values.items():
        low, high = np.quantile(samples, [0.025, 0.975])
        intervals[name] = {"lower_95": float(low), "upper_95": float(high)}
    return {
        "method": "stratified percentile bootstrap",
        "replicates": int(n_bootstrap),
        "intervals": intervals,
    }


def chronological_segments(y_true, scores, times, n_segments: int = 4) -> list[dict]:
    """Report drift-sensitive metrics over consecutive holdout time segments."""
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if not (len(y) == len(p) == len(t)):
        raise ValueError("Labels, scores, and timestamps must have equal lengths.")
    order = np.argsort(t, kind="stable")
    segments = []
    for segment_no, indices in enumerate(np.array_split(order, n_segments), start=1):
        if len(indices) == 0:
            continue
        metrics = ranking_metrics(y[indices], p[indices])
        metrics.update({
            "segment": segment_no,
            "transaction_time_start": float(np.min(t[indices])),
            "transaction_time_end": float(np.max(t[indices])),
        })
        segments.append(metrics)
    return segments
