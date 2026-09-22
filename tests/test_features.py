import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from txrisk.experiments.risk_model import RULE_SIGNALS
from txrisk.features.address_day import build_address_features

DAY = "2026-09-01"
A = "0x" + "a" * 40
B = "0x" + "b" * 40
TOKEN = "0x" + "c" * 40


def write_day(root, tables):
    for table, frame in tables.items():
        partition = root / table / f"date={DAY}"
        partition.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                       partition / "part.parquet")


def build(root, transfers=None, transactions=None, approvals=None, contracts=None):
    write_day(root, {
        "transactions": transactions if transactions is not None else pd.DataFrame({
            "from_address": [A], "to_address": [B], "value": [1.0],
            "receipt_status": [1], "receipt_effective_gas_price": [100.0],
        }),
        "token_transfers": transfers if transfers is not None else pd.DataFrame({
            "from_address": [A], "to_address": [B], "value": [5.0], "token_address": [TOKEN],
        }),
        "approvals": approvals if approvals is not None else pd.DataFrame({
            "address": pd.Series([], dtype="object"), "topics": pd.Series([], dtype="object"),
        }),
        "contracts": contracts if contracts is not None else pd.DataFrame(
            {"address": pd.Series([], dtype="object")}
        ),
    })
    return build_address_features(DAY, root).set_index("address")


def test_counts_both_sides_of_a_transfer(tmp_path):
    features = build(tmp_path)
    assert features.loc[A, "token_out_count"] == 1
    assert features.loc[B, "token_in_count"] == 1
    assert features.loc[A, "tx_out_counterparties"] == 1


def test_huge_token_amounts_do_not_overflow(tmp_path):
    # Spam tokens "transfer" up to 2^256, which becomes infinity in 32-bit float.
    huge = pd.DataFrame({
        "from_address": [A], "to_address": [B], "value": [1.16e77], "token_address": [TOKEN],
    })
    features = build(tmp_path, transfers=huge)
    value = features.loc[A, "token_out_value_log"]
    assert np.isfinite(value)
    assert np.isfinite(np.float32(value))


def test_zero_value_counts_stay_counts(tmp_path):
    spam = pd.DataFrame({
        "from_address": [A] * 3, "to_address": [B] * 3, "value": [0.0] * 3,
        "token_address": [TOKEN] * 3,
    })
    features = build(tmp_path, transfers=spam)
    assert features.loc[A, "token_out_zero_value"] == 3
    assert features.loc[A, "zero_value_share"] == 1.0


def test_every_zero_value_feature_is_excluded_by_the_circularity_test(tmp_path):
    """The label is "a zero-value transfer arrived", so any such feature gives it away.

    Missing the incoming side once made the test report a model that had simply been
    handed the answer.
    """
    features = build(tmp_path)
    zero_value_features = {c for c in features.columns if "zero_value" in c}
    assert zero_value_features
    assert zero_value_features <= set(RULE_SIGNALS)


def test_features_are_built_once_and_reused(tmp_path, monkeypatch):
    """Rebuilding a day costs minutes; on a machine with little spare memory it also
    means holding a whole window at once."""
    from txrisk.features import address_day

    calls = []
    monkeypatch.setattr(address_day, "build_address_features",
                        lambda day, root=None: calls.append(day) or pd.DataFrame(
                            {"address": [A], "tx_out_count": [1.0], "day": [day]}))

    first = address_day.features_for_day(DAY, tmp_path, cache_dir=tmp_path / "cache")
    second = address_day.features_for_day(DAY, tmp_path, cache_dir=tmp_path / "cache")

    assert calls == [DAY]
    pd.testing.assert_frame_equal(first, second)


def test_features_are_stored_as_32_bit(tmp_path):
    features = build(tmp_path)
    assert all(str(features[c].dtype) == "float32" for c in features.select_dtypes("number"))


def test_training_days_can_be_thinned_but_scored_days_never_are(tmp_path, monkeypatch):
    """Thinning the day being scored would flatter the result; positives never go."""
    import pandas as pd

    from txrisk.experiments import risk_model

    def fake_features(day, root=None, cache_dir=None):
        return pd.DataFrame({
            "address": [f"0x{i:040x}" for i in range(100)],
            "tx_out_count": [1.0] * 100,
            "day": [day] * 100,
        })

    hits = pd.DataFrame({
        "rule": ["address_poisoning"] * 4, "day": ["train", "train", "score", "score"],
        "address": [f"0x{i:040x}" for i in range(4)], "role": ["attacker"] * 4,
        "confidence": ["confirmed"] * 4, "transaction_hash": ["t"] * 4, "evidence": ["e"] * 4,
    })
    cache = tmp_path / "rule_hits.parquet"
    hits.to_parquet(cache, index=False)
    monkeypatch.setattr(risk_model, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(risk_model, "features_for_day", fake_features)

    data = risk_model.load_dataset(["train", "score"], negative_rate=0.2, scored_days=["score"])
    per_day = data.groupby("day").size()

    assert per_day["score"] == 100
    assert per_day["train"] < 100
    assert int(data["label"].sum()) == 4
