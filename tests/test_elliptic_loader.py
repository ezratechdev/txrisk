import numpy as np
import pandas as pd
import pytest

from txrisk.data.elliptic import AGG_FEATURES, LOCAL_FEATURES, load_elliptic

N_FEATURES = len(LOCAL_FEATURES) + len(AGG_FEATURES)


def write_raw_files(raw, n_features=N_FEATURES):
    """A three-transaction stand-in for the real CSVs, in the same layout."""
    raw.mkdir()
    rows = [[tx_id, t, *np.linspace(0, 1, n_features)] for tx_id, t in [(10, 1), (11, 1), (12, 2)]]
    pd.DataFrame(rows).to_csv(raw / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": [10, 11, 12], "class": ["1", "2", "unknown"]}).to_csv(
        raw / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame({"txId1": [10], "txId2": [11]}).to_csv(
        raw / "elliptic_txs_edgelist.csv", index=False
    )


def test_parses_labels_features_and_edges(tmp_path):
    write_raw_files(tmp_path / "raw")
    data = load_elliptic(tmp_path / "raw", cache_dir=None)
    assert list(data.txs.columns[:2]) == ["tx_id", "time_step"]
    assert data.txs.shape == (3, 2 + N_FEATURES + 1)
    assert data.txs["label"].iloc[0] == 1.0 and data.txs["label"].iloc[1] == 0.0
    assert np.isnan(data.txs["label"].iloc[2])
    assert data.labeled["tx_id"].tolist() == [10, 11]
    assert data.edges.to_dict("records") == [{"src": 10, "dst": 11}]


def test_rejects_unexpected_column_count(tmp_path):
    write_raw_files(tmp_path / "raw", n_features=10)
    with pytest.raises(ValueError, match="columns"):
        load_elliptic(tmp_path / "raw", cache_dir=None)


def test_cache_returns_the_same_data(tmp_path):
    write_raw_files(tmp_path / "raw")
    first = load_elliptic(tmp_path / "raw", cache_dir=tmp_path / "cache")
    second = load_elliptic(tmp_path / "raw", cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(first.txs, second.txs)
    pd.testing.assert_frame_equal(first.edges, second.edges)
