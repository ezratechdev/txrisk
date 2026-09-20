import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED, HIT_COLUMNS
from txrisk.rules.approvals import decode_approvals, find_approval_phishing
from txrisk.rules.drains import approval_pairs, find_token_drains
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


def drain_tables(rows, initiators):
    """rows: (owner, recipient, value, tx); initiators: {tx: who sent the transaction}."""
    transfers = transfers_frame(rows)
    transactions = pd.DataFrame(
        [{"transaction_hash": tx, "from_address": who} for tx, who in initiators.items()]
    )
    return transfers, transactions


def transfers_frame(rows):
    return pd.DataFrame(
        rows, columns=["from_address", "to_address", "value", "transaction_hash"]
    )


def test_flags_an_account_that_empties_wallets_it_does_not_own():
    collector = "0x" + "c" * 40
    victims = [f"0x{i:040x}" for i in range(6)]
    rows = [(v, collector, 100.0, f"tx{i}") for i, v in enumerate(victims)]
    transfers, transactions = drain_tables(rows, {f"tx{i}": collector for i in range(6)})

    approved = {(v, collector) for v in victims}
    hits = find_token_drains(transfers, transactions, DAY, approved, corroborating={collector})
    collectors = hits[hits["role"] == "collector"]
    assert collectors["address"].tolist() == [collector]
    assert "6 wallets that received nothing back" in collectors["evidence"].iat[0]
    assert collectors["confidence"].iat[0] == CONFIRMED
    assert set(hits[hits["role"] == "victim"]["address"]) == set(victims)


def test_a_trade_is_not_a_drain_because_the_owner_gets_something_back():
    router = "0x" + "e" * 40
    traders = [f"0x{i:040x}" for i in range(6)]
    rows = []
    for index, trader in enumerate(traders):
        rows.append((trader, router, 100.0, f"tx{index}"))       # token in
        rows.append((router, trader, 99.0, f"tx{index}"))        # token back out
    transfers, transactions = drain_tables(rows, {f"tx{i}": router for i in range(6)})

    assert find_token_drains(transfers, transactions, DAY, {(t, router) for t in traders}).empty


def test_eth_returned_in_the_same_transaction_also_counts_as_a_trade():
    router = "0x" + "e" * 40
    traders = [f"0x{i:040x}" for i in range(6)]
    rows = [(t, router, 100.0, f"tx{i}") for i, t in enumerate(traders)]
    transfers, transactions = drain_tables(rows, {f"tx{i}": router for i in range(6)})
    eth_back = pd.DataFrame([
        {"transaction_hash": f"tx{i}", "to_address": t, "value": 1e18}
        for i, t in enumerate(traders)
    ])

    approved = {(t, router) for t in traders}
    assert find_token_drains(
        transfers, transactions, DAY, approved, eth_transfers=eth_back
    ).empty


def test_moving_your_own_tokens_is_never_a_drain():
    owner = "0x" + "a" * 40
    rows = [(owner, f"0x{i:040x}", 10.0, f"tx{i}") for i in range(6)]
    transfers, transactions = drain_tables(rows, {f"tx{i}": owner for i in range(6)})
    assert find_token_drains(transfers, transactions, DAY, {(owner, owner)}).empty


def test_an_established_collector_stays_a_candidate():
    """A payroll or subscription contract also pulls approved tokens and returns nothing."""
    collector = "0x" + "c" * 40
    payees = [f"0x{i:040x}" for i in range(6)]
    rows = [(p, collector, 100.0, f"tx{i}") for i, p in enumerate(payees)]
    transfers, transactions = drain_tables(rows, {f"tx{i}": collector for i in range(6)})

    hits = find_token_drains(transfers, transactions, DAY, {(p, collector) for p in payees})
    assert hits[hits["role"] == "collector"]["confidence"].iat[0] == CANDIDATE


def test_a_sweep_without_an_approval_is_not_reported():
    """Exchanges emptying their own deposit addresses look identical without this."""
    sweeper = "0x" + "c" * 40
    deposits = [f"0x{i:040x}" for i in range(50)]
    rows = [(d, sweeper, 100.0, f"tx{i}") for i, d in enumerate(deposits)]
    transfers, transactions = drain_tables(rows, {f"tx{i}": sweeper for i in range(50)})

    assert find_token_drains(transfers, transactions, DAY, approved=set()).empty
    assert not find_token_drains(
        transfers, transactions, DAY, {(d, sweeper) for d in deposits}
    ).empty


def test_approval_pairs_come_from_decoded_approvals():
    decoded = decode_approvals(approval_logs([(VICTIM, FAKE)]))
    assert approval_pairs(decoded) == {(VICTIM, FAKE.lower())}
