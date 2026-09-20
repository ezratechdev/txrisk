import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED, HIT_COLUMNS
from txrisk.rules.approvals import decode_approvals, find_approval_phishing
from txrisk.rules.exposure import find_known_bad_exposure
from txrisk.rules.poisoning import find_address_poisoning, lookalike_key

DAY = "2026-09-01"
VICTIM = "0x" + "1" * 40
REAL = "0xabcd" + "0" * 32 + "beef"
FAKE = "0xabcd" + "9" * 32 + "beef"  # same ends as REAL, different middle


def transfers(rows):
    return pd.DataFrame(rows, columns=["from_address", "to_address", "value", "transaction_hash"])


def test_lookalike_key_uses_both_ends_of_the_address():
    keys = lookalike_key(pd.Series([REAL, FAKE]))
    assert keys.iat[0] == keys.iat[1] == "abcd_beef"


def test_finds_a_planted_address_that_imitates_a_real_counterparty():
    hits = find_address_poisoning(transfers([
        (VICTIM, REAL, 5.0, "genuine"),
        (VICTIM, FAKE, 0.0, "planted"),
    ]), DAY)

    assert list(hits.columns) == HIT_COLUMNS
    assert set(hits["role"]) == {"attacker", "victim"}
    assert hits.loc[hits["role"] == "attacker", "address"].iat[0] == FAKE
    assert hits.loc[hits["role"] == "victim", "address"].iat[0] == VICTIM
    assert "imitates" in hits["evidence"].iat[0]
    # Both ends matching a real counterparty is not chance, so this one is not a guess.
    assert set(hits["confidence"]) == {CONFIRMED}


def test_a_zero_value_transfer_alone_is_not_poisoning():
    # Zero-value transfers are common; without a look-alike there is nothing to report.
    hits = find_address_poisoning(transfers([
        (VICTIM, REAL, 5.0, "genuine"),
        (VICTIM, "0x" + "7" * 40, 0.0, "unrelated"),
    ]), DAY)
    assert hits.empty


def test_paying_an_address_you_already_use_is_not_an_imitation_of_it():
    hits = find_address_poisoning(transfers([
        (VICTIM, REAL, 5.0, "genuine"),
        (VICTIM, REAL, 0.0, "zero value to the same address"),
    ]), DAY)
    assert hits.empty


def approval_logs(pairs):
    return pd.DataFrame([
        {
            "address": "0xtoken",
            "topics": ["0xtopic0", "0x" + "0" * 24 + owner[2:], "0x" + "0" * 24 + spender[2:]],
            "transaction_hash": f"tx{index}",
        }
        for index, (owner, spender) in enumerate(pairs)
    ])


def test_decodes_owner_and_spender_from_padded_topics():
    decoded = decode_approvals(approval_logs([(VICTIM, FAKE)]))
    assert decoded["owner"].iat[0] == VICTIM
    assert decoded["spender"].iat[0] == FAKE.lower()


def test_logs_without_both_addresses_are_skipped():
    incomplete = pd.DataFrame([
        {"address": "0xtoken", "topics": ["0xtopic0"], "transaction_hash": "tx"}
    ])
    assert decode_approvals(incomplete).empty


def test_a_new_contract_many_wallets_approved_is_a_candidate_until_tokens_move():
    drainer = "0x" + "d" * 40
    owners = [f"0x{index:040x}" for index in range(25)]
    contracts = pd.DataFrame({"address": [drainer]})

    hits = find_approval_phishing(approval_logs([(o, drainer) for o in owners]), contracts, DAY)
    assert hits["address"].tolist() == [drainer]
    assert hits["confidence"].iat[0] == CANDIDATE
    assert "25 distinct wallets" in hits["evidence"].iat[0]


def test_the_hit_is_confirmed_once_the_approvers_tokens_move_to_the_spender():
    drainer = "0x" + "d" * 40
    owners = [f"0x{index:040x}" for index in range(25)]
    swept = transfers([(owners[0], drainer, 500.0, "drain1"), (owners[1], drainer, 10.0, "drain2")])

    hits = find_approval_phishing(
        approval_logs([(o, drainer) for o in owners]),
        pd.DataFrame({"address": [drainer]}),
        DAY,
        token_transfers=swept,
    )
    assert hits["confidence"].iat[0] == CONFIRMED
    assert "tokens then moved from 2 of them" in hits["evidence"].iat[0]


def test_an_established_router_is_not_flagged_however_many_approvals_it_takes():
    router = "0x" + "e" * 40
    owners = [f"0x{index:040x}" for index in range(500)]
    deployed_this_window = pd.DataFrame({"address": ["0x" + "f" * 40]})

    hits = find_approval_phishing(
        approval_logs([(o, router) for o in owners]), deployed_this_window, DAY
    )
    assert hits.empty


def test_a_new_contract_with_few_approvals_is_not_flagged():
    fresh = "0x" + "d" * 40
    owners = [f"0x{index:040x}" for index in range(3)]
    hits = find_approval_phishing(
        approval_logs([(o, fresh) for o in owners]), pd.DataFrame({"address": [fresh]}), DAY
    )
    assert hits.empty


def test_exposure_names_the_list_that_reported_the_address():
    listed = "0x" + "b" * 40
    hits = find_known_bad_exposure(
        transfers([(VICTIM, listed, 1.0, "tx1"), (VICTIM, REAL, 1.0, "tx2")]),
        {listed: "ofac_sdn (sanctions, LAZARUS GROUP)"},
        DAY,
    )
    assert hits["address"].tolist() == [listed]
    assert hits["role"].tolist() == ["receiver"]
    assert "LAZARUS GROUP" in hits["evidence"].iat[0]


def test_exposure_reports_nothing_when_no_listed_address_is_active():
    hits = find_known_bad_exposure(
        transfers([(VICTIM, REAL, 1.0, "tx1")]), {"0x" + "b" * 40: "listed"}, DAY
    )
    assert hits.empty
