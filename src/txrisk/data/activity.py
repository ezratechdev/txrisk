"""When were the reported addresses actually busy?

Extracting a window and hoping fraud happened inside it only works for address poisoning,
which never stops. Across three days of September, four of the 2,530 reported phishing
addresses appeared at all, and none did anything recognisable. Phishing is episodic.

So the labels choose the days instead. This asks a public block explorer when each
reported address was active, and the answer says which days are worth extracting.

Blockscout's public instance needs no account or key, and it rate-limits, which matters
more than it sounds: a refused request and an address that did nothing both look like an
empty list. Recording the first as the second would quietly mark thousands of addresses
as inactive and choose the wrong days to extract. A refusal is therefore retried, never
written down, and an address with no answer is left for the next run.

    python -m txrisk.data.activity
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from txrisk.data.labels import load_labels
from txrisk.paths import PROCESSED_DIR, RAW_DIR

API = "https://eth.blockscout.com/api/v2"
FEEDS = ("transactions", "token-transfers")
"""What the address sent or received, and the token movements it took part in."""

PAUSE_SECONDS = 0.2
WORKERS = 4


class Refused(RuntimeError):
    """The explorer declined to answer: rate limit, outage, or anything unrecognised."""


def _request(address: str, feed: str, timeout: float = 30.0) -> list[dict]:
    url = f"{API}/addresses/{address}/{feed}"
    request = urllib.request.Request(url, headers={"User-Agent": "txrisk (open source)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []  # the explorer knows nothing about this address
        raise Refused(f"HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        raise Refused(str(error)) from error

    if not isinstance(payload, dict) or "items" not in payload:
        raise Refused(str(payload)[:120])
    return payload["items"]


def fetch_activity(address: str, attempts: int = 5) -> set[str]:
    """The dates this address was active. Raises Refused rather than guessing silence."""
    dates: set[str] = set()
    for feed in FEEDS:
        for attempt in range(1, attempts + 1):
            try:
                for item in _request(address, feed):
                    stamp = item.get("timestamp")
                    if stamp:
                        dates.add(str(stamp)[:10])
                break
            except Refused:
                if attempt == attempts:
                    raise
                time.sleep(2**attempt)
        time.sleep(PAUSE_SECONDS)
    return dates


def build_activity(
    addresses: Iterable[str], cache_path: Path, workers: int = WORKERS
) -> pd.DataFrame:
    """Look each address up once, appending as it goes so a stopped run resumes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            done = {json.loads(line)["address"] for line in handle if line.strip()}

    remaining = [a for a in addresses if a not in done]
    print(f"{len(done):,} already looked up, {len(remaining):,} to go")
    if not remaining:
        return read_activity(cache_path)

    lock = threading.Lock()
    refused = 0
    with cache_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_activity, address): address for address in remaining}
        for index, future in enumerate(as_completed(futures), start=1):
            address = futures[future]
            try:
                dates = future.result()
            except Refused as error:
                refused += 1
                if refused <= 3:
                    print(f"    {address}: {error}; left for the next run", file=sys.stderr)
                continue
            with lock:
                handle.write(json.dumps({"address": address, "dates": sorted(dates)}) + "\n")
                handle.flush()
            if index % 100 == 0 or index == len(remaining):
                print(f"  {index:,}/{len(remaining):,} done, {refused:,} refused")
    return read_activity(cache_path)


def read_activity(cache_path: Path) -> pd.DataFrame:
    """One row per (address, date) the explorer reported."""
    rows = []
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows.extend({"address": record["address"], "date": d} for d in record["dates"])
    return pd.DataFrame(rows, columns=["address", "date"])


def busiest_days(activity: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    """The days worth extracting: where the most reported addresses were active."""
    if activity.empty:
        return pd.DataFrame(columns=["date", "reported_addresses"])
    counts = activity.groupby("date")["address"].nunique().sort_values(ascending=False)
    return counts.head(count).rename_axis("date").reset_index(name="reported_addresses")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraud-type", default="phishing", help="which labels to look up")
    parser.add_argument("--limit", type=int, help="stop after this many addresses")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)

    labels = load_labels()
    wanted = labels[(labels["chain"] == "ethereum") & (labels["fraud_type"] == args.fraud_type)]
    addresses = sorted(set(wanted["address"]))[: args.limit]
    print(f"looking up {len(addresses):,} {args.fraud_type} addresses on Blockscout")

    cache_path = RAW_DIR / "activity" / "blockscout.jsonl"
    activity = build_activity(addresses, cache_path, workers=args.workers)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    activity.to_parquet(PROCESSED_DIR / "label_activity.parquet", index=False)

    found = activity["address"].nunique()
    print(f"\n{len(activity):,} address-days for {found:,} addresses")
    print("\nbusiest days (extract these):")
    print(busiest_days(activity).to_string(index=False))


if __name__ == "__main__":
    main()
