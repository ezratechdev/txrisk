import json

import pandas as pd
import pytest

from txrisk.data import activity as activity_module
from txrisk.data.activity import (
    Refused,
    build_activity,
    busiest_days,
    fetch_activity,
    read_activity,
)

A = "0x" + "a" * 40
B = "0x" + "b" * 40


def fake_feeds(items_by_feed, calls=None):
    def _request(address, feed, timeout=30.0):
        if calls is not None:
            calls.append((address, feed))
        value = items_by_feed.get(feed, [])
        if isinstance(value, Exception):
            raise value
        return value
    return _request


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(activity_module.time, "sleep", lambda _: None)


def test_keeps_the_date_part_of_each_timestamp(monkeypatch):
    monkeypatch.setattr(activity_module, "_request", fake_feeds({
        "transactions": [{"timestamp": "2026-01-01T04:54:11.000000Z"},
                         {"timestamp": "2026-01-01T22:10:00.000000Z"}],
        "token-transfers": [{"timestamp": "2026-01-02T00:00:01.000000Z"}],
    }))
    assert fetch_activity(A) == {"2026-01-01", "2026-01-02"}


def test_a_refusal_is_raised_not_recorded_as_silence(monkeypatch):
    """A rate-limited request and an inactive address both return nothing useful.

    Treating the first as the second marked 93% of addresses inactive and would have
    picked the wrong days to extract.
    """
    monkeypatch.setattr(activity_module, "_request",
                        fake_feeds({"transactions": Refused("Too many requests")}))
    with pytest.raises(Refused):
        fetch_activity(A, attempts=2)


def test_an_address_with_genuinely_no_activity_gives_an_empty_set(monkeypatch):
    monkeypatch.setattr(activity_module, "_request",
                        fake_feeds({"transactions": [], "token-transfers": []}))
    assert fetch_activity(A) == set()


def test_refused_addresses_are_left_for_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_module, "_request",
                        fake_feeds({"transactions": Refused("rate limited")}))
    cache = tmp_path / "blockscout.jsonl"

    build_activity([A], cache, workers=1)

    # Nothing written, so the next run asks again instead of trusting a refusal.
    assert not cache.exists() or not cache.read_text(encoding="utf-8").strip()


def test_a_stopped_run_resumes_instead_of_asking_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(activity_module, "_request", fake_feeds(
        {"transactions": [{"timestamp": "2026-01-01T00:00:00Z"}], "token-transfers": []}, calls
    ))
    cache = tmp_path / "blockscout.jsonl"
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"address": A, "dates": ["2026-01-01"]}) + "\n", encoding="utf-8")

    build_activity([A, B], cache, workers=1)

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
