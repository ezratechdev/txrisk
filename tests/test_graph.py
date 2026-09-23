import pandas as pd

from txrisk.features.graph import build_graph_features, known_bad_before

DAY = "2026-09-01"
GOOD = "0x" + "1" * 40
BAD = "0x" + "b" * 40
NEIGHBOUR_OF_BAD = "0x" + "2" * 40


def write_day(root, transfers, payments=None):
    for table, frame in [("token_transfers", transfers),
                         ("transactions", payments if payments is not None else transfers)]:
        partition = root / table / f"date={DAY}"
        partition.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(partition / "part.parquet", index=False)


def edges(rows):
    return pd.DataFrame(rows, columns=["from_address", "to_address", "value"])


def test_counts_counterparties_already_known_to_be_bad(tmp_path):
    write_day(tmp_path, edges([(GOOD, BAD, 1.0), (GOOD, NEIGHBOUR_OF_BAD, 1.0)]))
    features = build_graph_features(DAY, {BAD}, tmp_path).set_index("address")

    assert features.loc[GOOD, "degree"] == 2
    assert features.loc[GOOD, "bad_neighbours"] == 1
    assert features.loc[GOOD, "bad_neighbour_share"] == 0.5


def test_sees_an_address_one_step_further_out(tmp_path):
    # GOOD -> NEIGHBOUR_OF_BAD -> BAD: laundering puts a hop between itself and the theft.
    write_day(tmp_path, edges([(GOOD, NEIGHBOUR_OF_BAD, 1.0), (NEIGHBOUR_OF_BAD, BAD, 1.0)]))
    features = build_graph_features(DAY, {BAD}, tmp_path).set_index("address")

    assert features.loc[GOOD, "bad_neighbours"] == 0
    assert features.loc[GOOD, "bad_two_hops_away"] == 1


def test_transfers_worth_nothing_do_not_make_a_relationship(tmp_path):
    # Spam arrives unbidden; being sent it says nothing about who you deal with.
    write_day(tmp_path, edges([(BAD, GOOD, 0.0)]))
    features = build_graph_features(DAY, {BAD}, tmp_path)
    assert features.empty


def test_only_earlier_days_count_as_known(tmp_path):
    """Using the scored day's own hits would hand the model its answer."""
    hits = pd.DataFrame({
        "day": ["2026-08-31", "2026-09-01"],
        "address": [BAD, GOOD],
        "role": ["attacker", "attacker"],
        "confidence": ["confirmed", "confirmed"],
    })
    known = known_bad_before("2026-09-01", hits, reported=set())
    assert known == {BAD}


def test_reported_addresses_count_whatever_the_day(tmp_path):
    hits = pd.DataFrame(columns=["day", "address", "role", "confidence"])
    assert known_bad_before("2026-09-01", hits, reported={BAD}) == {BAD}
