"""Loader for the BitcoinHeist ransomware dataset (Akcora et al., 2019).

2.9M daily observations of Bitcoin addresses from 2009-2018. Each row is labelled with
the ransomware family paid to that address, or "white" for everything else. Two
properties shape how the data can be used:

* "white" means "never reported as ransomware", not "verified legitimate". The negatives
  are unlabelled, so measured precision is a lower bound.
* Labels come from three research groups (Montreal, Padua, Princeton) that name the same
  family differently, so names have to be normalised before classes can be compared.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from txrisk.paths import PROCESSED_DIR, RAW_DIR

FEATURES = ["length", "weight", "count", "looped", "neighbors", "income"]
"""Address-level graph features supplied with the dataset. Year and day are deliberately
left out: a model must not be able to key on when the test period starts."""

WHITE = "white"
_SOURCE_PREFIX = re.compile(r"^(montreal|padua|princeton)")
_VERSION_SUFFIX = re.compile(r"v\d+(\.\d+)?$")


def normalise_family(label: pd.Series) -> pd.Series:
    """princetonCerber -> cerber, montrealDMALockerv3 -> dmalocker, white -> white."""
    family = label.str.replace(_SOURCE_PREFIX, "", regex=True)
    return family.str.replace(_VERSION_SUFFIX, "", regex=True).str.lower()


def load_bitcoinheist(
    raw_dir: Path = RAW_DIR / "bitcoinheist", cache_dir: Path | None = PROCESSED_DIR
) -> pd.DataFrame:
    """One row per address-day: address, year, day, features, family, is_ransomware."""
    cache = cache_dir / "bitcoinheist.parquet" if cache_dir else None
    if cache and cache.exists():
        return pd.read_parquet(cache)

    csv = raw_dir / "BitcoinHeistData.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run: python -m txrisk.data.download bitcoinheist"
        )
    frame = pd.read_csv(csv, engine="pyarrow")
    missing = {"address", "year", "day", "label", *FEATURES} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv.name} is missing expected columns: {sorted(missing)}")

    frame["family"] = normalise_family(frame["label"])
    frame["is_ransomware"] = frame["family"] != WHITE
    frame[FEATURES] = frame[FEATURES].astype("float32")
    frame = frame.drop(columns=["label"])

    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
    return frame
