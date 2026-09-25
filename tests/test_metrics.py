import numpy as np
import pytest

from txrisk.evaluation.metrics import (
    evaluate,
    expected_calibration_error,
    f1_by_group,
    precision_at_k,
    recall_at_fpr,
)


def test_recall_at_fpr_with_perfect_ranking():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    assert recall_at_fpr(y, scores, max_fpr=0.0) == 1.0


def test_recall_at_fpr_respects_the_false_positive_budget():
    # The top score is a negative, so catching any positive costs 1 of 4 negatives (25% FPR).
    y = np.array([0, 1, 0, 0, 0, 1])
    scores = np.array([0.99, 0.9, 0.1, 0.2, 0.3, 0.8])
    assert recall_at_fpr(y, scores, max_fpr=0.2) == 0.0
    assert recall_at_fpr(y, scores, max_fpr=0.25) == 1.0


def test_ece_is_zero_when_calibrated():
    y = np.array([1, 0, 0, 0] * 25)
    assert expected_calibration_error(y, np.full(100, 0.25)) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    y = np.array([1, 0] * 50)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_evaluate_reports_threshold_and_ranking_metrics():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5
    assert metrics["roc_auc"] == 0.75


def test_f1_by_group_is_nan_for_groups_without_positives():
    f1 = f1_by_group(
        np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.8, 0.2]), groups=np.array([1, 1, 2, 2])
    )
    assert f1[1] == 1.0
    assert np.isnan(f1[2])


def test_precision_at_k_reads_only_the_top_of_the_list():
    """An investigator works down from the top and stops; the tail is never seen."""
    y_true = np.array([1, 1, 0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    assert precision_at_k(y_true, scores, k=2) == 1.0
    assert precision_at_k(y_true, scores, k=4) == 0.5


def test_precision_at_k_breaks_ties_the_way_the_list_is_read():
    # Saturated scores tie; the measure must reflect the order actually shown.
    y_true = np.array([0, 1, 1])
    assert precision_at_k(y_true, np.array([1.0, 1.0, 1.0]), k=1) == 0.0


def test_precision_at_k_handles_a_shorter_list_than_asked_for():
    assert precision_at_k(np.array([1, 0]), np.array([0.9, 0.1]), k=100) == 0.5
