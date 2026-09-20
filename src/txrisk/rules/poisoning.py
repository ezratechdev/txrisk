"""Address poisoning: a look-alike address planted in someone's transaction history.

The attacker sends a zero-value token transfer that appears to come *from* the victim and
go *to* an address that looks like one the victim really uses - same first and last few
characters, different middle. Wallets abbreviate addresses as 0x1234...abcd, so the fake
is indistinguishable at a glance. Later the victim copies the address from their history
and pays the attacker.

The pattern is unusually clean: a real transfer never has to be worth nothing, and two
addresses sharing both ends is not chance. That makes this one of the few frauds a rule
can call with confidence.
"""

from __future__ import annotations

import pandas as pd

from txrisk.rules import build_hits, empty_hits

RULE = "address_poisoning"
PREFIX = 4
SUFFIX = 4


def lookalike_key(address: pd.Series, prefix: int = PREFIX, suffix: int = SUFFIX) -> pd.Series:
    """What an abbreviated address looks like: its first and last characters."""
    stripped = address.str.removeprefix("0x")
    return stripped.str[:prefix] + "_" + stripped.str[-suffix:]


def find_address_poisoning(
    token_transfers: pd.DataFrame, day: str, prefix: int = PREFIX, suffix: int = SUFFIX
) -> pd.DataFrame:
    """Zero-value transfers to an address that imitates one the sender really pays.

    The sender of the zero-value transfer is the victim whose history is being poisoned,
    so they are recorded with the role "victim" and the look-alike with "attacker".
    """
    if token_transfers.empty:
        return empty_hits()

    genuine = token_transfers[token_transfers["value"] > 0]
    planted = token_transfers[token_transfers["value"] == 0]
    if genuine.empty or planted.empty:
        return empty_hits()

    genuine = genuine[["from_address", "to_address"]].dropna().drop_duplicates()
    planted = planted.dropna(subset=["from_address", "to_address"])
    if genuine.empty or planted.empty:
        return empty_hits()

    genuine = genuine.assign(key=lookalike_key(genuine["to_address"], prefix, suffix))
    planted = planted.assign(key=lookalike_key(planted["to_address"], prefix, suffix))

    paired = planted.merge(
        genuine.rename(columns={"to_address": "imitated_address"}),
        on=["from_address", "key"],
        how="inner",
    )
    # A transfer to an address the sender genuinely uses is not an imitation of it.
    paired = paired[paired["to_address"] != paired["imitated_address"]]
    # One planted address can resemble several real counterparties; it is still one hit.
    paired = paired.drop_duplicates(subset=["from_address", "to_address"])
    if paired.empty:
        return empty_hits()

    evidence = (
        "zero-value transfer to " + paired["to_address"]
        + ", which imitates " + paired["imitated_address"]
        + " that this address really pays"
    )
    attacker = build_hits(
        RULE, day, paired, "to_address", "attacker", "transaction_hash", evidence
    )
    victim = build_hits(
        RULE, day, paired, "from_address", "victim", "transaction_hash", evidence
    )
    return pd.concat([attacker, victim], ignore_index=True)
