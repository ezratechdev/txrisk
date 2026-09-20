"""Metrics for rare-event classifiers.

Accuracy is left out on purpose: when 10% of cases are fraud, calling everything
legitimate scores 90%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Highest share of positives caught while flagging at most max_fpr of the negatives."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Gap between predicted probability and observed positive rate, averaged over bins."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    return float(
        sum(
            (bins == b).mean() * abs(y_true[bins == b].mean() - prob[bins == b].mean())
            for b in np.unique(bins)
        )
    )


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = prob >= threshold
    metrics = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, prob, 0.01),
        "brier": brier_score_loss(y_true, prob),
        "ece": expected_calibration_error(y_true, prob),
    }
    return {name: float(value) for name, value in metrics.items()}


def f1_by_group(
    y_true: np.ndarray, prob: np.ndarray, groups: np.ndarray, threshold: float = 0.5
) -> pd.Series:
    """Positive-class F1 per group (e.g. per time step); NaN where a group has no positives."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(prob) >= threshold})
    return frame.groupby(np.asarray(groups))[["y", "pred"]].apply(
        lambda g: f1_score(g["y"], g["pred"], zero_division=0) if g["y"].any() else np.nan
    )
