"""Loader for the Elliptic Bitcoin dataset (Weber et al., 2019).

203,769 transactions over 49 time steps about two weeks apart: 2% labelled illicit,
21% licit, the rest unknown. The features are anonymised, so a model trained here
cannot score live transactions. The dataset is for comparing methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from txrisk.paths import PROCESSED_DIR, RAW_DIR

# The paper counts 94 local features, one of which is the time step. We keep the time
# step out of the model inputs: test time steps never appear in training, so all a model
# could learn from it is "later is different", which is leakage rather than signal.
LOCAL_FEATURES = [f"local_{i}" for i in range(93)]
AGG_FEATURES = [f"agg_{i}" for i in range(72)]
FEATURE_SETS = {"local": LOCAL_FEATURES, "all": LOCAL_FEATURES + AGG_FEATURES}

_CLASS_TO_LABEL = {"1": 1.0, "2": 0.0}  # 1 = illicit, 2 = licit; "unknown" becomes NaN


@dataclass(frozen=True)
class Elliptic:
    txs: pd.DataFrame
    """One row per transaction: tx_id, time_step, label (1 illicit, 0 licit, NaN), features."""
    edges: pd.DataFrame
    """Payment flows between transactions: src, dst."""

    @property
    def labeled(self) -> pd.DataFrame:
        return self.txs[self.txs["label"].notna()].reset_index(drop=True)


def load_elliptic(
    raw_dir: Path = RAW_DIR / "elliptic", cache_dir: Path | None = PROCESSED_DIR
) -> Elliptic:
    """Parse the raw CSVs, caching the result as Parquet (pass cache_dir=None to skip)."""
    tx_cache = cache_dir / "elliptic_txs.parquet" if cache_dir else None
    edge_cache = cache_dir / "elliptic_edges.parquet" if cache_dir else None
    if tx_cache and tx_cache.exists() and edge_cache.exists():
        return Elliptic(pd.read_parquet(tx_cache), pd.read_parquet(edge_cache))

    features_csv = raw_dir / "elliptic_txs_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(
            f"{features_csv} not found. Run: python -m txrisk.data.download elliptic"
        )
    features = pd.read_csv(features_csv, header=None, engine="pyarrow")
    feature_columns = LOCAL_FEATURES + AGG_FEATURES
    if features.shape[1] != 2 + len(feature_columns):
        raise ValueError(
            f"expected {2 + len(feature_columns)} columns in {features_csv.name}, "
            f"found {features.shape[1]}"
        )
    features.columns = ["tx_id", "time_step", *feature_columns]
    features[feature_columns] = features[feature_columns].astype("float32")

    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv", dtype={"class": str})
    classes = classes.rename(columns={"txId": "tx_id"})
    classes["label"] = classes["class"].map(_CLASS_TO_LABEL)
    txs = features.merge(classes[["tx_id", "label"]], on="tx_id", how="left", validate="1:1")

    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    edges = edges.rename(columns={"txId1": "src", "txId2": "dst"})

    if tx_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        txs.to_parquet(tx_cache, index=False)
        edges.to_parquet(edge_cache, index=False)
    return Elliptic(txs, edges)
