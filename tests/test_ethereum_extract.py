from datetime import date, datetime

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from txrisk.data import ethereum
from txrisk.data.ethereum import (
    APPROVAL_FOR_ALL_TOPIC,
    APPROVAL_TOPIC,
    TABLES,
    TRANSFER_TOPIC,
    _keep_approvals,
    _keep_eth_movements,
    day_prefix,
    extract_day,
    output_path,
)

DAY = date(2026, 9, 1)


def test_keeps_only_approval_events():
    logs = pa.table({
        "topics": [[APPROVAL_TOPIC, "0xowner"], [TRANSFER_TOPIC], [APPROVAL_FOR_ALL_TOPIC]],
        "address": ["token", "token", "nft"],
    })
    assert _keep_approvals(logs).column("address").to_pylist() == ["token", "nft"]


def test_logs_without_topics_do_not_break_the_filter():
    # Anonymous events carry no topics at all; indexing into them would raise.
    logs = pa.table({"topics": [[], [APPROVAL_TOPIC]], "address": ["anon", "token"]})
    assert _keep_approvals(logs).column("address").to_pylist() == ["token"]


def test_keeps_successful_eth_movements_only():
    traces = pa.table({
        "value": [1.0, 0.0, 5.0],
        "status": [1, 1, 0],
        "transaction_hash": ["moved", "no value", "reverted"],
    })
    kept = _keep_eth_movements(traces).column("transaction_hash").to_pylist()
    assert kept == ["moved"]


def test_filtered_specs_request_the_columns_their_filters_read():
    assert "topics" in TABLES["approvals"].columns
    assert {"value", "status"} <= set(TABLES["eth_transfers"].columns)


def test_day_prefix_points_at_the_source_table_not_the_local_name():
    expected = "aws-public-blockchain/v1.0/eth/logs/date=2026-09-01"
    assert day_prefix(TABLES["approvals"], DAY) == expected


def test_output_is_partitioned_by_day(tmp_path):
    path = output_path("transactions", DAY, tmp_path)
    assert path.parent.name == "date=2026-09-01"
    assert path.parent.parent.name == "transactions"


def test_a_failed_table_does_not_abandon_the_rest_of_the_range(tmp_path, monkeypatch):
    """Ranges take hours; one dropped connection must not discard the days that worked."""
    def flaky(table, day, root, filesystem=None, attempts=8, **kwargs):
        if table == "approvals":
            raise OSError("network reset")
        return ethereum.DayResult(table, day, rows_read=10, rows_kept=5,
                                  bytes_written=100, seconds=0.1)

    monkeypatch.setattr(ethereum, "extract_day", flaky)
    monkeypatch.setattr(ethereum, "open_bucket", lambda: None)

    results, failures = ethereum.extract_range(
        ["transactions", "approvals"], [DAY], root=tmp_path, workers=2
    )
    assert [r.table for r in results] == ["transactions"]
    assert [(table, day) for table, day, _ in failures] == [("approvals", DAY)]


def test_finished_days_are_not_fetched_again(tmp_path, monkeypatch):
    done = output_path("transactions", DAY, tmp_path)
    done.parent.mkdir(parents=True)
    done.write_bytes(b"already here")
    monkeypatch.setattr(ethereum, "open_bucket", lambda: None)
    monkeypatch.setattr(ethereum, "extract_day", lambda *a, **k: pytest.fail("refetched"))

    results, failures = ethereum.extract_range(["transactions"], [DAY], tmp_path)
    assert not results and not failures


def test_open_table_reads_every_extracted_day_as_one_dataset(tmp_path):
    for day, addresses in [("2026-09-01", ["a"]), ("2026-09-02", ["b", "c"])]:
        partition = tmp_path / "transactions" / f"date={day}"
        partition.mkdir(parents=True)
        pq.write_table(pa.table({"from_address": addresses}), partition / "part.parquet")

    table = ethereum.open_table("transactions", tmp_path).to_table()
    assert sorted(table.column("from_address").to_pylist()) == ["a", "b", "c"]
    assert sorted(set(table.column("date").to_pylist())) == ["2026-09-01", "2026-09-02"]


def test_load_day_canonicalises_addresses_so_callers_cannot_forget(tmp_path):
    """The padded/plain mismatch has to be impossible to reintroduce downstream."""
    partition = tmp_path / "token_transfers" / f"date={DAY:%Y-%m-%d}"
    partition.mkdir(parents=True)
    padded = "0x" + "0" * 24 + "a" * 40
    pq.write_table(pa.table({"from_address": [padded], "value": [1.0]}), partition / "part.parquet")

    frame = ethereum.load_day("token_transfers", "2026-09-01", ["from_address", "value"], tmp_path)
    assert frame["from_address"].iat[0] == "0x" + "a" * 40


def test_extracted_days_lists_what_is_on_disk(tmp_path):
    for day in ["2026-09-01", "2026-09-02"]:
        partition = tmp_path / "transactions" / f"date={day}"
        partition.mkdir(parents=True)
        pq.write_table(pa.table({"hash": ["a"]}), partition / "part.parquet")
    assert ethereum.extracted_days(tmp_path) == ["2026-09-01", "2026-09-02"]


def test_open_table_says_what_to_run_when_nothing_is_extracted(tmp_path):
    with pytest.raises(FileNotFoundError, match="txrisk.data.ethereum"):
        ethereum.open_table("transactions", tmp_path)


def test_extract_day_keeps_wanted_rows_and_drops_the_rest(tmp_path, monkeypatch):
    """End to end against a local stand-in for the bucket, so CI needs no network."""
    bucket = tmp_path / "bucket"
    source = bucket / "logs" / f"date={DAY:%Y-%m-%d}"
    source.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "address": ["token", "token", "nft"],
            "topics": [[APPROVAL_TOPIC], [TRANSFER_TOPIC], [APPROVAL_FOR_ALL_TOPIC]],
            "data": ["0x1", "0x2", "0x3"],
            "transaction_hash": ["a", "b", "c"],
            "log_index": [0, 1, 2],
            "block_number": [1, 1, 1],
            "block_timestamp": [datetime(2026, 9, 1)] * 3,
            "unwanted": ["drop me"] * 3,
        }),
        source / "part.parquet",
    )
    monkeypatch.setattr(ethereum, "BUCKET", bucket.as_posix())

    out = tmp_path / "out"
    result = extract_day("approvals", DAY, out, pafs.LocalFileSystem())

    assert (result.rows_read, result.rows_kept) == (3, 2)
    written = pq.read_table(output_path("approvals", DAY, out))
    assert written.column("address").to_pylist() == ["token", "nft"]
    assert "unwanted" not in written.column_names


def test_scattered_days_can_be_extracted_without_the_gap_between_them(tmp_path, monkeypatch):
    """Days where known-bad addresses were active are years apart, not adjacent."""
    asked = []
    monkeypatch.setattr(ethereum, "open_bucket", lambda: None)
    monkeypatch.setattr(ethereum, "extract_day",
                        lambda table, day, root, filesystem=None, attempts=8, **k:
                        (asked.append(day) or ethereum.DayResult(table, day, 1, 1, 1, 0.1)))

    wanted = [date(2022, 10, 26), date(2023, 3, 20)]
    ethereum.extract_range(["transactions"], wanted, tmp_path, workers=1)
    assert sorted(asked) == wanted
