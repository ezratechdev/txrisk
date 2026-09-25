import pandas as pd

from txrisk import mcp_server

DAY = "2023-03-01"
ADDRESS = "0x" + "a" * 40


def test_asking_about_an_unextracted_day_says_how_to_fix_it(monkeypatch):
    """Declining with the remedy beats guessing from data that is not there."""
    monkeypatch.setattr(mcp_server, "extracted_days", lambda: [DAY])
    answer = mcp_server.score_address_tool(ADDRESS, "1999-01-01")

    assert "error" in answer
    assert answer["extracted_days"] == [DAY]
    assert "txrisk.data.ethereum" in answer["how_to_fix"]


def test_an_address_that_did_nothing_is_told_so(monkeypatch):
    monkeypatch.setattr(mcp_server, "extracted_days", lambda: [DAY])
    monkeypatch.setattr(mcp_server, "load_scorer", lambda: object())
    monkeypatch.setattr(mcp_server, "address_scores",
                        lambda day, scorer: pd.DataFrame(columns=["address", "risk"]))

    answer = mcp_server.score_address_tool(ADDRESS, DAY)
    assert answer["active"] is False
    assert "nothing to score" in answer["note"]


def test_a_score_arrives_with_its_reasons_and_a_caveat(monkeypatch):
    scored = pd.DataFrame({
        "address": [ADDRESS], "risk": [0.91], "token_in_count": [12.0],
        "token_out_count": [0.0], "distinct_tokens": [7.0], "token_in_counterparties": [12.0],
    })
    monkeypatch.setattr(mcp_server, "extracted_days", lambda: [DAY])
    monkeypatch.setattr(mcp_server, "load_scorer", lambda: object())
    monkeypatch.setattr(mcp_server, "address_scores", lambda day, scorer: scored)
    monkeypatch.setattr(mcp_server, "rule_reasons",
                        lambda day: {ADDRESS: ["address_poisoning (confirmed): imitates ..."]})

    answer = mcp_server.score_address_tool(ADDRESS.upper(), DAY)

    assert answer["risk_score"] == 0.91
    assert answer["risk_band"] == "high"
    assert "address_poisoning" in answer["reasons"][0]
    assert answer["behaviour"]["distinct_tokens"] == 7
    assert "not proof" in answer["caveat"]


def test_rules_can_be_asked_without_a_model_in_the_answer(monkeypatch):
    monkeypatch.setattr(mcp_server, "rule_reasons", lambda day: {ADDRESS: ["token_drain: ..."]})
    answer = mcp_server.explain_rules_tool(ADDRESS, DAY)

    assert answer["rule_hits"] == ["token_drain: ..."]
    assert "risk_score" not in answer


def test_available_days_is_empty_rather_than_exploding_when_nothing_is_extracted(monkeypatch):
    def missing():
        raise FileNotFoundError("no data")
    monkeypatch.setattr(mcp_server, "extracted_days", missing)
    assert mcp_server.available_days() == []
