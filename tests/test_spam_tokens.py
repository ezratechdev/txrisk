import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED
from txrisk.rules.spam_tokens import find_spam_airdrops

DAY = "2023-03-01"
DISTRIBUTOR = "0x" + "d" * 40
TOKEN = "0x" + "c" * 40


def drop(recipients, sender=DISTRIBUTOR, token=TOKEN):
    return pd.DataFrame({
        "from_address": [sender] * len(recipients),
        "to_address": recipients,
        "token_address": [token] * len(recipients),
        "value": [1.0] * len(recipients),
        "transaction_hash": [f"tx{i}" for i in range(len(recipients))],
    })


def wallets(count, start=0):
    return [f"0x{i:040x}" for i in range(start, start + count)]


def test_a_small_distribution_is_not_a_campaign():
    assert find_spam_airdrops(drop(wallets(50)), DAY).empty


def test_a_large_drop_is_a_candidate_until_the_next_days_are_known():
    """On the day itself a spam drop and a real airdrop look exactly alike."""
    hits = find_spam_airdrops(drop(wallets(300)), DAY)
    assert hits["address"].tolist() == [DISTRIBUTOR]
    assert hits["confidence"].iat[0] == CANDIDATE
    assert "300 wallets" in hits["evidence"].iat[0]


def test_a_token_nobody_touches_again_is_confirmed():
    recipients = wallets(300)
    # Only two of the three hundred ever move it.
    later = pd.DataFrame({
        "from_address": recipients[:2],
        "to_address": ["0x" + "e" * 40] * 2,
        "token_address": [TOKEN] * 2,
        "value": [1.0] * 2,
        "transaction_hash": ["later1", "later2"],
    })
    hits = find_spam_airdrops(drop(recipients), DAY, later_transfers=later)

    assert hits["confidence"].iat[0] == CONFIRMED
    assert "99% of them never touched it again" in hits["evidence"].iat[0]


def test_an_airdrop_people_actually_use_stays_a_candidate():
    # Half the recipients move the token on: they wanted it.
    recipients = wallets(300)
    claimed = recipients[:150]
    later = pd.DataFrame({
        "from_address": claimed,
        "to_address": ["0x" + "e" * 40] * len(claimed),
        "token_address": [TOKEN] * len(claimed),
        "value": [1.0] * len(claimed),
        "transaction_hash": [f"use{i}" for i in range(len(claimed))],
    })
    hits = find_spam_airdrops(drop(recipients), DAY, later_transfers=later)
    assert hits["confidence"].iat[0] == CANDIDATE


def test_a_freshly_minted_token_is_worth_saying_so():
    hits = find_spam_airdrops(drop(wallets(300)), DAY, new_tokens={TOKEN})
    assert "minted in this window" in hits["evidence"].iat[0]


def test_two_senders_of_the_same_token_are_counted_apart():
    other = "0x" + "9" * 40
    both = pd.concat([drop(wallets(300)), drop(wallets(300, start=1000), sender=other)])
    hits = find_spam_airdrops(both, DAY)
    assert set(hits["address"]) == {DISTRIBUTOR, other}
