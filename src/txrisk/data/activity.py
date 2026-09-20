"""When were the reported addresses actually busy?

Extracting a window and hoping fraud happened inside it does not work for anything but
address poisoning. Across three days of September, four of the 2,530 reported phishing
addresses appeared at all, and none of them did anything recognisable. Phishing is
episodic; poisoning is constant.

So the labels choose the days instead. This asks a public block explorer when each
reported address was active, and the answer says which days are worth extracting: the
ones where many known-bad addresses were doing something.

Blockscout's public instance is used because it needs no account and no key. It is
someone else's server, so requests are paced, and only the dates are kept - never a copy
of their database. As everywhere else here, the repository ships the fetcher, not the data.

    python -m txrisk.data.activity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from txrisk.data.labels import load_labels
from txrisk.paths import PROCESSED_DIR, RAW_DIR

API = "https://eth.blockscout.com/api"
ACTIONS = ("txlist", "tokentx")
"""Transactions the address sent or received, and token movements it took part in."""

PAUSE_SECONDS = 0.4
PAGE_SIZE = 200


def _request(address: str, action: str, timeout: float = 30.0) -> list[dict]:
    query = (
        f"{API}?module=account&action={action}&address={address}"
        f"&page=1&offset={PAGE_SIZE}&sort=desc"
    )
    request = urllib.request.Request(query, headers={"User-Agent": "txrisk (open source)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    result = payload.get("result")
    return result if isinstance(result, list) else []


def fetch_activity(address: str, attempts: int = 4) -> set[str]:
    """The dates on which this address did anything, as far as the explorer can see."""
    dates: set[str] = set()
    for action in ACTIONS:
        for attempt in range(1, attempts + 1):
            try:
                for record in _request(address, action):
                    stamp = record.get("timeStamp") or record.get("timestamp")
                    if stamp:
                        moment = datetime.fromtimestamp(int(stamp), tz=UTC)
                        dates.add(moment.strftime("%Y-%m-%d"))
                break
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
                if attempt == attempts:
                    print(f"    {address} {action}: giving up ({error})", file=sys.stderr)
                    break
                time.sleep(2**attempt)
        time.sleep(PAUSE_SECONDS)
    return dates


def build_activity(
    addresses: Iterable[str], cache_path: Path, pause: float = PAUSE_SECONDS
) -> pd.DataFrame:
    """Look up each address once, appending as it goes so a stopped run resumes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            done = {json.loads(line)["address"] for line in handle if line.strip()}

    remaining = [a for a in addresses if a not in done]
    print(f"{len(done):,} addresses already looked up, {len(remaining):,} to go")
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, address in enumerate(remaining, start=1):
            dates = fetch_activity(address)
            handle.write(json.dumps({"address": address, "dates": sorted(dates)}) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(remaining):
                print(f"  {index:,}/{len(remaining):,} looked up")
            time.sleep(pause)
    return read_activity(cache_path)


def read_activity(cache_path: Path) -> pd.DataFrame:
    """One row per (address, date) the explorer reported."""
    rows = []
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
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
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS)
    args = parser.parse_args(argv)

    labels = load_labels()
    wanted = labels[(labels["chain"] == "ethereum") & (labels["fraud_type"] == args.fraud_type)]
    addresses = sorted(set(wanted["address"]))[: args.limit]
    print(f"looking up {len(addresses):,} {args.fraud_type} addresses on Blockscout")

    cache_path = RAW_DIR / "activity" / "blockscout.jsonl"
    activity = build_activity(addresses, cache_path, pause=args.pause)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    activity.to_parquet(PROCESSED_DIR / "label_activity.parquet", index=False)

    print(f"\n{len(activity):,} address-days for {activity['address'].nunique():,} addresses")
    print("\nbusiest days (extract these):")
    print(busiest_days(activity).to_string(index=False))


if __name__ == "__main__":
    main()
