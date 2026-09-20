import numpy as np
import pandas as pd
import pytest

from txrisk.data.bitcoinheist import FEATURES, load_bitcoinheist, normalise_family
from txrisk.evaluation.splits import drop_overlapping_groups, temporal_split


def write_raw_files(raw):
    raw.mkdir()
    rows = pd.DataFrame({
        "address": ["a", "b", "c"],
        "year": [2016, 2017, 2017],
        "day": [1, 2, 3],
        **{name: [1.0, 2.0, 3.0] for name in FEATURES},
        "label": ["princetonCerber", "white", "montrealDMALockerv3"],
    })
    rows.to_csv(raw / "BitcoinHeistData.csv", index=False)


def test_normalise_family_merges_labelling_sources_and_versions():
    labels = pd.Series(["princetonCerber", "montrealCryptoLocker", "paduaJigsaw",
                        "montrealJigSaw", "montrealDMALockerv3", "white"])
    assert normalise_family(labels).tolist() == [
        "cerber", "cryptolocker", "jigsaw", "jigsaw", "dmalocker", "white"
    ]


def test_loader_marks_ransomware_rows(tmp_path):
    write_raw_files(tmp_path / "raw")
    frame = load_bitcoinheist(tmp_path / "raw", cache_dir=None)
    assert frame["family"].tolist() == ["cerber", "white", "dmalocker"]
    assert frame["is_ransomware"].tolist() == [True, False, True]
    assert "label" not in frame.columns


def test_loader_rejects_a_file_with_missing_columns(tmp_path):
    raw = tmp_path / "raw"
    write_raw_files(raw)
    truncated = pd.read_csv(raw / "BitcoinHeistData.csv").drop(columns=["income"])
    truncated.to_csv(raw / "BitcoinHeistData.csv", index=False)
    with pytest.raises(ValueError, match="income"):
        load_bitcoinheist(raw, cache_dir=None)


def test_dropping_shared_addresses_removes_them_from_the_test_side():
    year = np.array([2016, 2016, 2017, 2017])
    address = np.array(["a", "b", "a", "c"])
    split = drop_overlapping_groups(temporal_split(year, 2016), address)
    # "a" was in training, so only the row for the unseen address "c" is scored.
    assert address[split.test].tolist() == ["c"]
    assert address[split.train].tolist() == ["a", "b"]
