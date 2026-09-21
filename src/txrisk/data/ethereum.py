"""Extract the Ethereum data the models need, from the AWS public blockchain dataset.

The bucket stores one Parquet file per table per day with every column. A full day is
about 5.5 GB, most of it raw call data that this project never reads. This module streams
each day in batches, keeps only the columns and rows that carry signal, and writes the
result locally, so disk use stays flat however long the date range is.

No AWS account, key or cost: the bucket is public open data under an MIT-0 licence.

    python -m txrisk.data.ethereum --start 2026-09-01 --days 7

Caveat on amounts: this dataset stores `value` as a float, so wei amounts above about
2^53 lose their last digits. That is fine for size bands and ratios, which is all the
features use, but it is not exact accounting.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from txrisk.data.addresses import canonicalise_addresses
from txrisk.paths import RAW_DIR

BUCKET = "aws-public-blockchain/v1.0/eth"
REGION = "us-east-2"
BATCH_ROWS = 250_000

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_TOPIC = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
APPROVAL_FOR_ALL_TOPIC = "0x17307eab39ab6107e8899845ad3d59bd9653f200f220920489ca2b5937696c31"


def _keep_approvals(table: pa.Table) -> pa.Table:
    """Approval and ApprovalForAll events: the ones a victim signs before a wallet drain."""
    topics = table.column("topics").combine_chunks()
    table = table.filter(pc.greater(pc.list_value_length(topics), 0))
    if table.num_rows == 0:
        return table
    first_topic = pc.list_element(table.column("topics").combine_chunks(), 0)
    wanted = pa.array([APPROVAL_TOPIC, APPROVAL_FOR_ALL_TOPIC])
    return table.filter(pc.is_in(first_topic, value_set=wanted))


def _keep_eth_movements(table: pa.Table) -> pa.Table:
    """Internal calls that actually moved ETH and did not revert."""
    moved = pc.greater(table.column("value"), 0)
    succeeded = pc.equal(table.column("status"), 1)
    return table.filter(pc.and_(moved, succeeded))


@dataclass(frozen=True)
class TableSpec:
    """One local table: where it comes from, what we keep, and why."""

    source: str
    columns: tuple[str, ...]
    why: str
    keep: Callable[[pa.Table], pa.Table] | None = None


TABLES: dict[str, TableSpec] = {
    "transactions": TableSpec(
        source="transactions",
        columns=(
            "hash", "from_address", "to_address", "value", "gas", "receipt_gas_used",
            "receipt_effective_gas_price", "nonce", "receipt_status",
            "receipt_contract_address", "transaction_type", "transaction_index",
            "block_number", "block_timestamp",
        ),
        why="the backbone: who paid whom, when, and whether it succeeded",
    ),
    "token_transfers": TableSpec(
        source="token_transfers",
        columns=(
            "token_address", "from_address", "to_address", "value", "transaction_hash",
            "log_index", "block_number", "block_timestamp",
        ),
        why="most theft moves tokens, not ETH",
    ),
    "approvals": TableSpec(
        source="logs",
        columns=(
            "address", "topics", "data", "transaction_hash", "log_index",
            "block_number", "block_timestamp",
        ),
        why="approval phishing: the victim grants a spender, the drain follows",
        keep=_keep_approvals,
    ),
    "eth_transfers": TableSpec(
        source="traces",
        columns=(
            "from_address", "to_address", "value", "transaction_hash", "trace_address",
            "call_type", "status", "block_number", "block_timestamp",
        ),
        why="ETH moved inside contract calls, which the transactions table cannot show",
        keep=_keep_eth_movements,
    ),
    "contracts": TableSpec(
        source="contracts",
        columns=("address", "bytecode", "block_number", "block_timestamp"),
        why="newly deployed code, for scam-token and honeypot checks",
    ),
}


@dataclass
class DayResult:
    table: str
    day: date
    rows_read: int
    rows_kept: int
    bytes_written: int
    seconds: float

    @property
    def kept_share(self) -> float:
        return self.rows_kept / self.rows_read if self.rows_read else 0.0


def day_prefix(spec: TableSpec, day: date) -> str:
    return f"{BUCKET}/{spec.source}/date={day:%Y-%m-%d}"


def output_path(table: str, day: date, root: Path) -> Path:
    return root / table / f"date={day:%Y-%m-%d}" / "part.parquet"


def open_bucket() -> pafs.S3FileSystem:
    """Anonymous, read-only access to the public bucket, with retries and timeouts.

    The timeouts matter as much as the retries. Without them, a socket that dies while the
    machine sleeps leaves the reader blocked for ever: the extraction looks like it is
    still running, and no retry is ever attempted. With them the read fails and is retried.
    """
    return pafs.S3FileSystem(
        anonymous=True,
        region=REGION,
        retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=5),
        connect_timeout=10.0,
        request_timeout=120.0,
    )


def _stream_day(spec: TableSpec, day: date, destination: Path, filesystem) -> tuple[int, int]:
    sources = [
        info.path
        for info in filesystem.get_file_info(pafs.FileSelector(day_prefix(spec, day)))
        if info.path.endswith(".parquet")
    ]
    if not sources:
        raise FileNotFoundError(f"no data in the bucket for {spec.source} on {day:%Y-%m-%d}")

    rows_read = rows_kept = 0
    writer = None
    try:
        for path in sources:
            reader = pq.ParquetFile(path, filesystem=filesystem)
            for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=list(spec.columns)):
                rows_read += batch.num_rows
                kept = pa.Table.from_batches([batch])
                if spec.keep is not None:
                    kept = spec.keep(kept)
                if kept.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(destination, kept.schema, compression="zstd")
                writer.write_table(kept)
                rows_kept += kept.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows_read, rows_kept


def extract_day(
    table: str,
    day: date,
    root: Path,
    filesystem: pafs.FileSystem | None = None,
    attempts: int = 8,
    max_pause: float = 60.0,
) -> DayResult:
    """Stream one day of one table from the bucket into a local Parquet file.

    A failed day is retried from the start and only becomes the real file once it is
    complete. A half-written day is worse than no day: it would silently train a model on
    a fraction of the traffic. Backoff runs to minutes rather than seconds, because a
    dropped connection takes far longer to come back than a rate-limited request does.
    """
    spec = TABLES[table]
    filesystem = filesystem or open_bucket()
    started = time.perf_counter()
    out = output_path(table, day, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".part")

    for attempt in range(1, attempts + 1):
        try:
            rows_read, rows_kept = _stream_day(spec, day, partial, filesystem)
            break
        except OSError as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            pause = min(2**attempt, max_pause)
            print(f"  {day:%Y-%m-%d} {table:15} {error}; "
                  f"retry {attempt}/{attempts - 1} in {pause:.0f}s")
            time.sleep(pause)

    bytes_written = 0
    if partial.exists():
        partial.replace(out)
        bytes_written = out.stat().st_size
    return DayResult(
        table=table,
        day=day,
        rows_read=rows_read,
        rows_kept=rows_kept,
        bytes_written=bytes_written,
        seconds=time.perf_counter() - started,
    )


def extract_range(
    tables: list[str],
    days: list[date],
    root: Path,
    skip_existing: bool = True,
    workers: int = 3,
    attempts: int = 8,
) -> tuple[list[DayResult], list[tuple[str, date, str]]]:
    """Extract every (table, day) pair, and keep going when one of them fails.

    A range can take hours, so one dead connection must not throw away the days that
    already worked. Failures are collected and reported at the end; re-running the same
    command retries only what is missing.
    """
    jobs = []
    for day in days:
        for table in tables:
            if skip_existing and output_path(table, day, root).exists():
                print(f"  {day:%Y-%m-%d} {table:15} already extracted, skipping")
                continue
            jobs.append((table, day))

    filesystem = open_bucket()
    results: list[DayResult] = []
    failures: list[tuple[str, date, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {
            pool.submit(extract_day, table, day, root, filesystem, attempts): (table, day)
            for table, day in jobs
        }
        for future in as_completed(running):
            table, day = running[future]
            try:
                result = future.result()
            except OSError as error:
                failures.append((table, day, str(error)))
                print(f"  {day:%Y-%m-%d} {table:15} gave up: {error}")
                continue
            results.append(result)
            print(
                f"  {day:%Y-%m-%d} {table:15} {result.rows_kept:>10,} rows kept "
                f"({result.kept_share:5.1%} of {result.rows_read:>10,}) "
                f"{result.bytes_written / 1e6:7.1f} MB in {result.seconds:5.0f}s"
            )
    return results, failures


def open_table(table: str, root: Path = RAW_DIR / "ethereum") -> ds.Dataset:
    """Every extracted day of one table as a single dataset, read lazily.

    Filters and column choices push down to the per-day files, so a query over a month
    never loads a month into memory.
    """
    path = root / table
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m txrisk.data.ethereum --start <date> --days <n>"
        )
    return ds.dataset(path, format="parquet", partitioning="hive")


def extracted_days(root: Path = RAW_DIR / "ethereum", table: str = "transactions") -> list[str]:
    days = open_table(table, root).to_table(columns=["date"]).column("date").to_pylist()
    return sorted({str(day) for day in days})


def load_day(
    table: str, day: str, columns: list[str], root: Path = RAW_DIR / "ethereum"
) -> pd.DataFrame:
    """One day of one table, with every address in the same format.

    Canonicalising belongs here rather than in each caller: the raw files keep the source's
    own spellings, and nothing downstream has to remember that one table pads its
    addresses to 32 bytes while the others do not.
    """
    rows = open_table(table, root).to_table(columns=columns, filter=pc.field("date") == day)
    return canonicalise_addresses(rows.to_pandas())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract Ethereum data, one day at a time.")
    parser.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="how many days from --start")
    parser.add_argument(
        "--dates", nargs="*", type=date.fromisoformat,
        help="specific days, which need not be adjacent: the days where known-bad "
             "addresses were active are scattered across years",
    )
    parser.add_argument("--tables", default=",".join(TABLES), help=f"any of: {', '.join(TABLES)}")
    parser.add_argument("--root", type=Path, default=RAW_DIR / "ethereum")
    parser.add_argument("--redo", action="store_true", help="re-extract days already on disk")
    parser.add_argument("--workers", type=int, default=3, help="day-tables fetched at once")
    parser.add_argument("--attempts", type=int, default=8, help="tries per day before giving up")
    args = parser.parse_args(argv)

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = set(tables) - set(TABLES)
    if unknown:
        parser.error(f"unknown tables: {sorted(unknown)}. Choose from {sorted(TABLES)}")
    if not args.dates and not args.start:
        parser.error("give either --start (with --days) or --dates")

    days = args.dates or [args.start + timedelta(days=n) for n in range(args.days)]
    results, failures = extract_range(
        tables, sorted(set(days)), args.root,
        skip_existing=not args.redo, workers=args.workers, attempts=args.attempts,
    )
    total_mb = sum(r.bytes_written for r in results) / 1e6
    print(f"\n{len(results)} day-tables, {total_mb:,.0f} MB written to {args.root}")
    if failures:
        print(f"{len(failures)} failed: " + ", ".join(f"{t} {d:%Y-%m-%d}" for t, d, _ in failures))
        print("Run the same command again to retry only those; finished days are skipped.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
