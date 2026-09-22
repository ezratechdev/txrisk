"""The training path itself, on data small enough to run in a test.

Every earlier failure here was invisible: an exception inside a long run, hidden behind a
pipe, looking exactly like an out-of-memory kill. Exercising fit_and_score on tiny data
catches that class of breakage in seconds instead of half an hour.
"""

import numpy as np
import pandas as pd

from txrisk.evaluation.splits import Split
from txrisk.experiments.risk_model import RULE_SIGNALS, explain, fit_and_score, split_on_days

FEATURES = ["token_in_count", "distinct_tokens", "fan_out"]


def toy_data(rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Half innocent, half obviously not, so a working model separates them easily."""
    rng = np.random.default_rng(seed)
    innocent = pd.DataFrame({
        "token_in_count": rng.normal(3, 1, rows),
        "distinct_tokens": rng.normal(1, 0.3, rows),
        "fan_out": rng.normal(0.2, 0.05, rows),
        "label": 0,
    })
    guilty = pd.DataFrame({
        "token_in_count": rng.normal(40, 5, rows // 4),
        "distinct_tokens": rng.normal(9, 1, rows // 4),
        "fan_out": rng.normal(0.9, 0.05, rows // 4),
        "label": 1,
    })
    data = pd.concat([innocent, guilty], ignore_index=True)
    # Shuffle before handing out days. A real day holds fraud and ordinary accounts mixed
    # together; leaving all the fraud at the end gives the calibrator a slice of one class.
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    data["address"] = [f"0x{i:040x}" for i in range(len(data))]
    data["day"] = np.where(rng.random(len(data)) < 0.5, "day1", "day2")
    return data.sort_values("day", kind="stable").reset_index(drop=True)


def test_training_produces_calibrated_scores_and_a_usable_model():
    data = toy_data()
    half = len(data) // 2
    split = Split("toy", np.arange(len(data)) < half, np.arange(len(data)) >= half)

    scores, model, metrics = fit_and_score(data, split, FEATURES, seed=0)

    assert len(scores) == split.test.sum()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert metrics["roc_auc"] > 0.9
    assert model.feature_importances_.shape == (len(FEATURES),)


def test_explanations_name_features_that_exist():
    data = toy_data()
    half = len(data) // 2
    split = Split("toy", np.arange(len(data)) < half, np.arange(len(data)) >= half)
    _, model, _ = fit_and_score(data, split, FEATURES, seed=0)

    reason = explain(model, data.iloc[-1], FEATURES)
    assert any(name in reason for name in FEATURES)


def test_scoring_a_named_window_trains_on_everything_else():
    data = toy_data()
    split = split_on_days(data, ["day2"])
    assert set(data.loc[split.test, "day"]) == {"day2"}
    assert set(data.loc[split.train, "day"]) == {"day1"}


def test_rule_signals_are_named_consistently():
    # Guards the circularity test: these names must match the feature builder's output.
    assert all(signal.endswith(("zero_value", "zero_value_share")) for signal in RULE_SIGNALS)
