"""One table of known-bad addresses, whatever the source called them.

Every source has its own vocabulary: "Phish/Hack", "SDN", "scam token", "drainer". Models
and rules need one taxonomy, one address format and one record of where each claim came
from, because a label is only as good as its source and its date.

Licensing shapes what can live here. This module ships *fetchers*, never label data: each
user downloads under the source's own terms. Sources are added only when their terms allow
automated access:

* `ofac_sdn` - US Treasury sanctions list. A US government work, so public domain.
* Etherscan's "Phish/Hack" tags are deliberately absent: their terms restrict scraping and
  redistribution. Chainabuse needs a per-user API key. Both can be added locally by anyone
  who accepts those terms.

    python -m txrisk.data.labels
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from txrisk.data.download import fetch
from txrisk.paths import PROCESSED_DIR, RAW_DIR

FRAUD_TYPES = (
    "sanctions",
    "phishing",
    "scam_token",
    "ponzi",
    "ransomware",
    "laundering",
    "hack",
    "other",
)
"""The taxonomy every source is mapped onto. Keep it small: a type nobody can detect from
chain data is not worth a class."""

COLUMNS = ["address", "chain", "fraud_type", "source", "reference", "collected_on"]

OFAC_SDN_URL = "https://sanctionslistservice.ofac.treas.gov/api/publicationpreview/exports/sdn.csv"
_DIGITAL_CURRENCY = re.compile(r"Digital Currency Address - ([A-Z0-9]+) ([a-zA-Z0-9]+)")
_ETHEREUM_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")


def detect_chain(asset: str, address: str) -> str | None:
    """Which chain an entry belongs to, or None for chains this project does not cover.

    The asset code alone is not enough: USDT exists on Ethereum and on Tron, and a Tron
    address recorded as Bitcoin would be a silent data error. Ethereum addresses are
    recognisable by shape, so shape decides first and the asset code only breaks ties.
    """
    if _ETHEREUM_ADDRESS.match(address):
        return "ethereum"
    if asset == "XBT":
        return "bitcoin"
    return None


def normalise_address(address: str, chain: str) -> str:
    """Ethereum addresses are case-insensitive, Bitcoin's are not."""
    return address.lower() if chain == "ethereum" else address


def parse_ofac(text: str, collected_on: date) -> pd.DataFrame:
    """Pull digital currency addresses out of the SDN list.

    The SDN file has no header and keeps crypto addresses inside the free-text remarks
    field, several per entity, so the addresses are matched by pattern rather than column.
    """
    rows = []
    for record in csv.reader(io.StringIO(text)):
        if not record:
            continue
        entity = record[1].strip().strip('"') if len(record) > 1 else ""
        for asset, address in _DIGITAL_CURRENCY.findall(" ".join(record)):
            chain = detect_chain(asset, address)
            if chain is None:
                continue  # Tron, Monero, Solana and friends: out of scope for now
            rows.append({
                "address": normalise_address(address, chain),
                "chain": chain,
                "fraud_type": "sanctions",
                "source": "ofac_sdn",
                "reference": entity,
                "collected_on": collected_on,
            })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.drop_duplicates(subset=["address", "chain", "source"]).reset_index(drop=True)


def fetch_ofac(raw_dir: Path) -> pd.DataFrame:
    destination = raw_dir / "ofac" / "sdn.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch(OFAC_SDN_URL, destination)
    text = destination.read_text(encoding="utf-8", errors="replace")
    return parse_ofac(text, collected_on=date.today())


SOURCES: dict[str, Callable[[Path], pd.DataFrame]] = {"ofac_sdn": fetch_ofac}


def build_labels(
    sources: list[str] | None = None,
    raw_dir: Path = RAW_DIR / "labels",
    cache_dir: Path | None = PROCESSED_DIR,
) -> pd.DataFrame:
    """Fetch each source and combine them into one table of claims about addresses.

    One address can appear several times, once per source that reported it. That is on
    purpose: agreement between independent sources is evidence, and collapsing it away
    would throw that evidence out.
    """
    frames = [SOURCES[name](raw_dir) for name in (sources or list(SOURCES))]
    labels = pd.concat(frames, ignore_index=True)[COLUMNS]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(cache_dir / "labels.parquet", index=False)
    return labels


def load_labels(cache_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = cache_dir / "labels.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m txrisk.data.labels")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the known-bad address table.")
    parser.add_argument(
        "--sources", default=",".join(SOURCES), help=f"any of: {', '.join(SOURCES)}"
    )
    args = parser.parse_args(argv)

    labels = build_labels([s.strip() for s in args.sources.split(",") if s.strip()])
    print(f"\n{len(labels):,} labels, {labels['address'].nunique():,} distinct addresses")
    summary = (
        labels.groupby(["source", "chain", "fraud_type"]).size().reset_index(name="labels")
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
