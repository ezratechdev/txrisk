import json

import pandas as pd

from txrisk.data import activity as activity_module
from txrisk.data.activity import build_activity, busiest_days, fetch_activity, read_activity

A = "0x" + "a" * 40
B = "0x" + "b" * 40


def fake_api(records_by_action, calls=None):
    def _request(address, action, timeout=30.0):
        if calls is not None:
            calls.append((address, action))
        return records_by_action.get(action, [])
    return _request


def test_turns_timestamps_into_dates(monkeypatch):
    # 1 January 2026 00:00 UTC and a few hours later the same day.
    monkeypatch.setattr(activity_module, "_request", fake_api({
        "txlist": [{"timeStamp": "1767225600"}, {"timeStamp": "1767250000"}],
        "tokentx": [{"timeStamp": "1767312000"}],
    }))
    monkeypatch.setattr(activity_module.time, "sleep", lambda _: None)

    assert fetch_activity(A) == {"2026-01-01", "2026-01-02"}


def test_records_without_a_timestamp_are_skipped(monkeypatch):
    monkeypatch.setattr(activity_module, "_request", fake_api({"txlist": [{"blockNumber": "1"}]}))
    monkeypatch.setattr(activity_module.time, "sleep", lambda _: None)
    assert fetch_activity(A) == set()


def test_a_stopped_run_resumes_instead_of_asking_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(activity_module, "_request",
                        fake_api({"txlist": [{"timeStamp": "1767225600"}]}, calls))
    monkeypatch.setattr(activity_module.time, "sleep", lambda _: None)
    cache = tmp_path / "blockscout.jsonl"
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"address": A, "dates": ["2026-01-01"]}) + "\n", encoding="utf-8")

    build_activity([A, B], cache)

    # Someone else's server: the address already looked up is not asked for again.
    assert {address for address, _ in calls} == {B}
    assert set(read_activity(cache)["address"]) == {A, B}


def test_busiest_days_ranks_by_how_many_reported_addresses_were_active():
    activity = pd.DataFrame({
        "address": [A, B, A, A],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-01"],
    })
    ranked = busiest_days(activity)
    assert ranked.iloc[0]["date"] == "2026-01-01"
    assert ranked.iloc[0]["reported_addresses"] == 2


def test_no_activity_gives_an_empty_ranking_not_an_error():
    assert busiest_days(pd.DataFrame(columns=["address", "date"])).empty
