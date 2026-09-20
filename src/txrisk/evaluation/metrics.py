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
    precision_recall_curve,
    precision_recall_fscore_support,
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


UNKNOWN = "unknown"


def threshold_for_best_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Decision threshold with the best F1 on validation data.

    A 0.5 cut-off only makes sense when the classes are somewhat balanced. At 0.5% fraud a
    calibrated model rarely exceeds 0.5, so an assumed cut-off flags nothing and the model
    looks broken when it is the threshold that is wrong.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(f1[:-1].argmax())])


def threshold_for_coverage(max_prob: np.ndarray, coverage: float) -> float:
    """Confidence cut-off that gives an answer for `coverage` of cases, rather than "unknown".

    Picked on validation data from the training period, never on the test set.
    """
    return float(np.quantile(np.asarray(max_prob), 1.0 - coverage))


def predict_with_rejection(
    proba: np.ndarray, classes: np.ndarray, threshold: float, unknown_label: str = UNKNOWN
) -> np.ndarray:
    """Name the most likely class, or answer `unknown_label` when no class is convincing."""
    classes = np.asarray(classes)
    return np.where(
        proba.max(axis=1) >= threshold, classes[proba.argmax(axis=1)], unknown_label
    )


def evaluate_open_set(
    y_true: np.ndarray,
    predicted: np.ndarray,
    known_classes: list[str],
    unknown_label: str = UNKNOWN,
) -> dict[str, float]:
    """Score a fraud-type classifier that is allowed to answer "unknown".

    `y_true` holds the true type, or `unknown_label` for types the model was never trained
    on. The two are reported separately on purpose: a model that confidently names the
    wrong fraud type for a new kind of fraud is worse than one that admits it doesn't know.
    """
    y_true, predicted = np.asarray(y_true), np.asarray(predicted)
    known = np.isin(y_true, np.asarray(known_classes))
    unseen = ~known

    def share(rows: np.ndarray, hits: np.ndarray) -> float:
        return float(hits[rows].mean()) if rows.any() else float("nan")

    # Average only over classes that actually occur in the test period. Including classes
    # with no test rows would score the model on families it never had a chance to predict.
    scored = [c for c in known_classes if c in set(y_true[known])]
    macro_f1 = float("nan")
    if scored:
        macro_f1 = f1_score(
            y_true[known], predicted[known], labels=scored, average="macro", zero_division=0
        )
    return {
        "known_accuracy": share(known, predicted == y_true),
        "known_macro_f1": float(macro_f1),
        "known_answered": share(known, predicted != unknown_label),
        "unseen_rejected": share(unseen, predicted == unknown_label),
        "n_known": int(known.sum()),
        "n_unseen": int(unseen.sum()),
        "n_classes_scored": len(scored),
    }


def per_class_scores(y_true: np.ndarray, predicted: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support for each class, so rare types stay visible."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )


def f1_by_group(
    y_true: np.ndarray, prob: np.ndarray, groups: np.ndarray, threshold: float = 0.5
) -> pd.Series:
    """Positive-class F1 per group (e.g. per time step); NaN where a group has no positives."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(prob) >= threshold})
    return frame.groupby(np.asarray(groups))[["y", "pred"]].apply(
        lambda g: f1_score(g["y"], g["pred"], zero_division=0) if g["y"].any() else np.nan
    )
