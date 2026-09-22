import numpy as np
import pytest

from txrisk.evaluation.metrics import (
    UNKNOWN,
    evaluate_open_set,
    per_class_scores,
    predict_with_rejection,
    threshold_for_coverage,
)

CLASSES = np.array(["cerber", "locky"])


def test_threshold_for_coverage_answers_the_requested_share():
    max_prob = np.linspace(0.0, 1.0, 101)
    threshold = threshold_for_coverage(max_prob, coverage=0.9)
    assert (max_prob >= threshold).mean() == pytest.approx(0.9, abs=0.01)


def test_rejection_keeps_confident_answers_and_drops_the_rest():
    proba = np.array([[0.9, 0.1], [0.55, 0.45]])
    predicted = predict_with_rejection(proba, CLASSES, threshold=0.6)
    assert predicted.tolist() == ["cerber", UNKNOWN]


def test_open_set_scores_known_and_unseen_types_separately():
    y_true = np.array(["cerber", "cerber", "locky", UNKNOWN, UNKNOWN])
    predicted = np.array(["cerber", "locky", "locky", UNKNOWN, "cerber"])
    scores = evaluate_open_set(y_true, predicted, known_classes=list(CLASSES))

    assert scores["n_known"] == 3 and scores["n_unseen"] == 2
    assert scores["known_accuracy"] == 2 / 3
    # One unseen row was called "cerber" instead of being rejected.
    assert scores["unseen_rejected"] == 0.5
    assert scores["known_answered"] == 1.0


def test_abstaining_beats_guessing_on_unseen_types():
    y_true = np.array([UNKNOWN] * 4)
    abstains = evaluate_open_set(y_true, np.array([UNKNOWN] * 4), list(CLASSES))
    guesses = evaluate_open_set(y_true, np.array(["cerber"] * 4), list(CLASSES))
    assert abstains["unseen_rejected"] == 1.0
    assert guesses["unseen_rejected"] == 0.0


def test_per_class_scores_keeps_rare_classes_visible():
    y_true = np.array(["cerber", "cerber", "locky"])
    predicted = np.array(["cerber", "cerber", UNKNOWN])
    scores = per_class_scores(y_true, predicted, labels=list(CLASSES)).set_index("class")
    assert scores.loc["cerber", "f1"] == 1.0
    assert scores.loc["locky", "recall"] == 0.0
    assert scores.loc["locky", "support"] == 1


def test_each_class_gets_its_own_confidence_bar():
    """A single bar calibrated on an imbalanced slice silences the rare class.

    Here poisoning is predicted with near-certainty and phishing with less. One global bar
    would sit at poisoning's confidence and reject every phishing row, including correct ones.
    """
    from txrisk.evaluation.metrics import thresholds_per_class

    classes = np.array(["phishing", "poisoning"])
    calibration = np.array([[0.02, 0.98]] * 90 + [[0.80, 0.20]] * 10)

    bars = thresholds_per_class(calibration, classes, coverage=0.9)
    assert bars["poisoning"] > bars["phishing"]

    scored = np.array([[0.80, 0.20], [0.98, 0.02]])
    answers = predict_with_rejection(scored, classes, bars)
    assert answers.tolist() == ["phishing", "phishing"]


def test_a_single_bar_still_works_for_one_class_problems():
    classes = np.array(["phishing", "poisoning"])
    answers = predict_with_rejection(np.array([[0.9, 0.1], [0.55, 0.45]]), classes, 0.6)
    assert answers.tolist() == ["phishing", UNKNOWN]
