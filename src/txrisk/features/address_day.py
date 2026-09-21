"""Per-address behaviour for one day, the input every model works from.

A transaction on its own says almost nothing: the same transfer is innocent or damning
depending on who sent it and what else they did. So the unit here is an address-day, and
a transaction is scored later from the behaviour of the two accounts involved.

Every feature is a count, a share or an amount that can be read aloud in an explanation
("this address sent 412 transfers, all of them worth nothing, to 412 different accounts").
Features nobody can interpret make hits nobody will act on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from txrisk.data.ethereum import load_day
from txrisk.paths import PROCESSED_DIR, RAW_DIR

# Amounts arrive as floats in the source data, so they are magnitudes, never exact wei.


def _side(frame: pd.DataFrame, own: str, other: str, prefix: str) -> pd.DataFrame:
    """Count activity and distinct counterparties for one side of a transfer table."""
    usable = frame.dropna(subset=[own])
    if usable.empty:
        return pd.DataFrame()
    grouped = usable.groupby(own)
    out = pd.DataFrame({
        f"{prefix}_count": grouped.size(),
        f"{prefix}_counterparties": grouped[other].nunique(),
    })
    if "value" in usable.columns:
        out[f"{prefix}_value"] = grouped["value"].sum()
        out[f"{prefix}_zero_value"] = grouped["value"].apply(lambda v: int((v == 0).sum()))
    out.index.name = "address"
    return out


def features_for_day(
    day: str, root: Path = RAW_DIR / "ethereum", cache_dir: Path | None = PROCESSED_DIR
) -> pd.DataFrame:
    """Build a day's features once and keep them, because rebuilding costs minutes.

    Each day is written on its own so a model can read the days it needs without ever
    holding the whole window in memory. On a machine with a gigabyte or two to spare that
    is the difference between training and swapping.
    """
    if cache_dir is None:
        return build_address_features(day, root)
    path = cache_dir / "features" / f"date={day}" / "part.parquet"
    if path.exists():
        return pd.read_parquet(path)
    features = build_address_features(day, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features


def build_address_features(day: str, root: Path = RAW_DIR / "ethereum") -> pd.DataFrame:
    """One row per address that did anything on this day."""
    transactions = load_day(
        day=day, table="transactions", root=root,
        columns=["from_address", "to_address", "value", "receipt_status",
                 "receipt_effective_gas_price"],
    )
    transfers = load_day(
        day=day, table="token_transfers", root=root,
        columns=["from_address", "to_address", "value", "token_address"],
    )
    approvals = load_day(day=day, table="approvals", root=root, columns=["address", "topics"])
    contracts = load_day(day=day, table="contracts", root=root, columns=["address"])

    parts = [
        _side(transactions, "from_address", "to_address", "tx_out"),
        _side(transactions, "to_address", "from_address", "tx_in"),
        _side(transfers, "from_address", "to_address", "token_out"),
        _side(transfers, "to_address", "from_address", "token_in"),
    ]
    features = pd.concat([p for p in parts if not p.empty], axis=1).fillna(0.0)

    # Distinct tokens touched: a drainer sweeps whatever it finds, a user holds a few.
    if not transfers.empty:
        sent = transfers[["from_address", "token_address"]].rename(
            columns={"from_address": "address"}
        )
        received = transfers[["to_address", "token_address"]].rename(
            columns={"to_address": "address"}
        )
        tokens = pd.concat([sent, received]).dropna()
        features["distinct_tokens"] = tokens.groupby("address")["token_address"].nunique()

    if not transactions.empty:
        failed = transactions[transactions["receipt_status"] == 0]
        features["failed_tx_out"] = failed.groupby("from_address").size()
        features["gas_price_mean"] = transactions.groupby("from_address")[
            "receipt_effective_gas_price"
        ].mean()

    features["approvals_given"] = approvals.groupby("address").size() if not approvals.empty else 0
    features["deployed_contract"] = (
        features.index.isin(set(contracts["address"].dropna())).astype(int)
    )

    features = features.fillna(0.0)
    # Amounts span eighty orders of magnitude: a spam token can "transfer" 2^256 units,
    # which overflows a 32-bit float to infinity and drowns every real amount. Their
    # magnitude is what carries signal, so keep the logarithm rather than the number.
    amounts = [c for c in features.columns if c.endswith("_value") and "zero" not in c]
    for column in amounts:
        features[column] = np.log1p(features[column].clip(lower=0))
    features = features.rename(columns={c: f"{c}_log" for c in amounts})

    features["zero_value_share"] = _share(
        features.get("token_out_zero_value", 0), features.get("token_out_count", 0)
    )
    features["failed_share"] = _share(features["failed_tx_out"], features.get("tx_out_count", 0))
    features["fan_out"] = _share(
        features.get("token_out_counterparties", 0), features.get("token_out_count", 0)
    )
    features["day"] = day
    # Counts and shares fit in 32 bits, and halving the frame matters more than the
    # last digit of a transfer count.
    numeric = features.select_dtypes("number").columns
    features[numeric] = features[numeric].astype("float32")
    return features.reset_index().rename(columns={"index": "address"})


def _share(numerator, denominator) -> pd.Series:
    """Ratios where a zero denominator means "no activity", not an error."""
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = (
        pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    )
    return (numerator / denominator.replace(0, pd.NA)).fillna(0.0)
