import numpy as np
import pandas as pd
import pytest

from txrisk import serve
from txrisk.monitor import RECALIBRATE_ABOVE, verdict
from txrisk.serve import Scorer, band, score_transactions

FEATURES = ["token_in_count", "distinct_tokens"]
RISKY = "0x" + "d" * 40
ORDINARY = "0x" + "1" * 40
OTHER = "0x" + "2" * 40


class FakeModel:
    """Scores an address by its first feature, so expectations stay readable."""

    def predict_proba(self, X):
        risk = np.clip(X[:, 0] / 10.0, 0, 1)
        return np.column_stack([1 - risk, risk])


def test_bands_read_the_way_people_speak():
    assert band(0.95) == "high"
    assert band(0.5) == "medium"
    assert band(0.01) == "low"


def test_a_transaction_takes_the_risk_of_its_worse_party(monkeypatch):
    """One dangerous party is enough; averaging lets a busy innocent one hide it."""
    scored = pd.DataFrame({
        "address": [RISKY, ORDINARY],
        "token_in_count": [9.0, 0.0],
        "distinct_tokens": [5.0, 1.0],
        "risk": [0.9, 0.0],
    })
    monkeypatch.setattr(serve, "address_scores", lambda day, scorer: scored)
    monkeypatch.setattr(
        serve, "rule_reasons", lambda day: {RISKY: ["token_drain: swept 9 wallets"]}
    )
    monkeypatch.setattr(serve, "load_day", lambda *a, **k: pd.DataFrame({
        "hash": ["0xpayment"], "from_address": [ORDINARY], "to_address": [RISKY], "value": [1.0],
    }))

    result = score_transactions("2023-03-03", Scorer(FakeModel(), FEATURES), top=5)

    assert result[0]["risk_score"] == 0.9
    assert result[0]["risk_band"] == "high"
    assert result[0]["riskier_party"] == {"address": RISKY, "side": "receiver"}
    assert "token_drain" in result[0]["reasons"][0]


def test_a_score_with_no_rule_behind_it_says_so(monkeypatch):
    scored = pd.DataFrame({
        "address": [OTHER], "token_in_count": [7.0], "distinct_tokens": [2.0], "risk": [0.7],
    })
    monkeypatch.setattr(serve, "address_scores", lambda day, scorer: scored)
    monkeypatch.setattr(serve, "rule_reasons", lambda day: {})
    monkeypatch.setattr(serve, "load_day", lambda *a, **k: pd.DataFrame({
        "hash": ["0xa"], "from_address": [OTHER], "to_address": [ORDINARY], "value": [1.0],
    }))

    result = score_transactions("2023-03-03", Scorer(FakeModel(), FEATURES), top=1)
    assert "model's judgement" in result[0]["reasons"][0]


def test_a_scorer_refuses_data_missing_the_features_it_was_trained_on():
    scorer = Scorer(FakeModel(), FEATURES)
    with pytest.raises(ValueError, match="distinct_tokens"):
        scorer.score(pd.DataFrame({"token_in_count": [1.0]}))


def test_drift_watch_calls_for_recalibration_when_probabilities_slip():
    slipped = pd.DataFrame({"day": ["d1", "d2"], "ece": [0.005, RECALIBRATE_ABOVE + 0.01]})
    assert "Recalibrate" in verdict(slipped)

    steady = pd.DataFrame({"day": ["d1"], "ece": [0.004]})
    assert "Recalibrate" not in verdict(steady)


def test_drift_watch_says_nothing_confident_without_labels():
    assert "nothing can be said" in verdict(pd.DataFrame({"day": ["d1"], "addresses": [10]}))
