"""Who an address deals with, which is often more telling than what it does.

A fresh address that has done nothing unusual, paying an account that drained wallets
yesterday, is not an ordinary address. Behaviour alone cannot see that; the graph can.

The hard rule here is that "known bad" must mean *known by then*. Rule hits from the day
being scored describe that very day's fraud, so using them would hand the model its own
answer and read as a triumph. Only reports and hits from earlier days are allowed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from txrisk.data.ethereum import load_day
from txrisk.paths import PROCESSED_DIR, RAW_DIR

GRAPH_COLUMNS = [
    "degree",
    "bad_neighbours",
    "bad_neighbour_share",
    "bad_two_hops_away",
]


def edges_for_day(day: str, root: Path = RAW_DIR / "ethereum") -> pd.DataFrame:
    """Every pair of addresses that moved value to one another on this day."""
    transfers = load_day(
        "token_transfers", day, ["from_address", "to_address", "value"], root
    )
    payments = load_day("transactions", day, ["from_address", "to_address", "value"], root)
    both = pd.concat([transfers, payments], ignore_index=True)
    both = both[both["value"] > 0].dropna(subset=["from_address", "to_address"])
    return both[["from_address", "to_address"]].drop_duplicates()


def build_graph_features(
    day: str, known_bad: set[str], root: Path = RAW_DIR / "ethereum"
) -> pd.DataFrame:
    """Per address: how many counterparties it has, and how many were already suspect.

    `known_bad` must contain only what was known before this day.
    """
    edges = edges_for_day(day, root)
    if edges.empty:
        return pd.DataFrame(columns=["address", *GRAPH_COLUMNS])

    # Treat the graph as undirected: paying a drainer and being paid by one both matter.
    undirected = pd.concat([
        edges.rename(columns={"from_address": "address", "to_address": "neighbour"}),
        edges.rename(columns={"to_address": "address", "from_address": "neighbour"}),
    ], ignore_index=True).drop_duplicates()

    undirected["neighbour_is_bad"] = undirected["neighbour"].isin(known_bad)
    grouped = undirected.groupby("address")
    features = pd.DataFrame({
        "degree": grouped.size(),
        "bad_neighbours": grouped["neighbour_is_bad"].sum(),
    })
    features["bad_neighbour_share"] = features["bad_neighbours"] / features["degree"]

    # Two hops: a counterparty of a counterparty. Laundering puts a hop between itself and
    # the theft, so the address that matters is often one step further out.
    touches_bad = set(features.index[features["bad_neighbours"] > 0])
    undirected["neighbour_touches_bad"] = undirected["neighbour"].isin(touches_bad)
    features["bad_two_hops_away"] = grouped["neighbour_touches_bad"].sum()

    features.index.name = "address"
    return features.reset_index()[["address", *GRAPH_COLUMNS]]


def graph_features_for_day(
    day: str, known_bad: set[str], root: Path = RAW_DIR / "ethereum",
    cache_dir: Path | None = PROCESSED_DIR,
) -> pd.DataFrame:
    """Build a day's graph features once and keep them; the joins are not cheap."""
    if cache_dir is None:
        return build_graph_features(day, known_bad, root)
    path = cache_dir / "graph" / f"date={day}" / "part.parquet"
    if path.exists():
        return pd.read_parquet(path)
    features = build_graph_features(day, known_bad, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features


def known_bad_before(day: str, hits: pd.DataFrame, reported: set[str]) -> set[str]:
    """What could honestly have been known before this day started."""
    earlier = hits[(hits["day"] < day) & (hits["confidence"] == "confirmed")]
    culprits = earlier[earlier["role"].isin(["attacker", "collector", "spender", "receiver"])]
    return set(culprits["address"]) | reported
