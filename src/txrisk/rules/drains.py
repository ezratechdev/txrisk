"""The drain itself: tokens leaving a wallet in a transaction its owner did not send.

An approval lets someone else move your tokens. That is ordinary - every trade works that
way - so the approval is not the fraud and cannot be detected as such. The theft is the
moment the spender uses it, and that moment has a signature: the tokens leave the owner's
wallet in a transaction the owner did not initiate, **and the owner receives nothing in
return**. A trade returns something in the same transaction. A drain does not.

That "nothing in return" test is what separates this from ordinary trading, so both sides
are checked: tokens coming back, and ETH coming back.

That test alone is not enough, and the chain says so loudly: on one day it produced 1,207
"collectors", the largest of which emptied 79,636 wallets. Exchanges sweeping their own
deposit addresses do exactly this, and they are not thieves.

What a drainer cannot avoid is the approval. It can only move tokens the owner approved
**to it**, so the drain must be preceded by an approval from that owner to that spender.
Exchange sweeps of their own addresses do not need one. Requiring the link is what turns
a description of ordinary plumbing into evidence of theft.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED, build_hits, empty_hits

RULE = "token_drain"
MIN_VICTIMS = 5


def _received_in_transaction(transfers: pd.DataFrame, eth_transfers: pd.DataFrame) -> set:
    """Every (transaction, address) pair where that address received something."""
    pairs = set(zip(transfers["transaction_hash"], transfers["to_address"], strict=False))
    if eth_transfers is not None and not eth_transfers.empty:
        returned = eth_transfers[eth_transfers["value"] > 0]
        pairs |= set(zip(returned["transaction_hash"], returned["to_address"], strict=False))
    return pairs


def approval_pairs(decoded_approvals: pd.DataFrame) -> set[tuple[str, str]]:
    """(owner, spender) pairs, the permission a drain has to have been granted."""
    if decoded_approvals is None or decoded_approvals.empty:
        return set()
    return set(
        zip(decoded_approvals["owner"], decoded_approvals["spender"], strict=False)
    )


def find_token_drains(
    token_transfers: pd.DataFrame,
    transactions: pd.DataFrame,
    day: str,
    approved: set[tuple[str, str]],
    eth_transfers: pd.DataFrame | None = None,
    corroborating: set[str] | None = None,
    min_victims: int = MIN_VICTIMS,
) -> pd.DataFrame:
    """Accounts that used approvals to empty wallets that got nothing back.

    `approved` holds the (owner, spender) pairs seen granting permission. Only movements
    covered by one of them count: without that link this rule describes exchange
    plumbing rather than theft.

    `corroborating` holds addresses that are newly deployed or already reported. A hit is
    only `confirmed` when the collector is one of them, because a subscription or payroll
    contract pulls approved tokens and returns nothing on-chain too.
    """
    if token_transfers.empty or transactions.empty or not approved:
        return empty_hits()

    initiators = transactions[["transaction_hash", "from_address"]].rename(
        columns={"from_address": "initiator"}
    )
    moves = token_transfers[token_transfers["value"] > 0].merge(
        initiators, on="transaction_hash", how="inner"
    )
    moves = moves.dropna(subset=["from_address", "initiator"])
    # Somebody else moved these tokens. Ordinary so far: this is how every trade works.
    third_party = moves[moves["from_address"] != moves["initiator"]]
    if third_party.empty:
        return empty_hits()

    received = _received_in_transaction(token_transfers, eth_transfers)
    got_nothing_back = [
        (tx, owner) not in received
        for tx, owner in zip(
            third_party["transaction_hash"], third_party["from_address"], strict=False
        )
    ]
    drained = third_party[np.array(got_nothing_back)]
    if drained.empty:
        return empty_hits()

    # The permission the theft depended on: owner approved this very spender.
    linked = [
        (owner, spender) in approved
        for owner, spender in zip(
            drained["from_address"], drained["initiator"], strict=False
        )
    ]
    drained = drained[np.array(linked)]
    if drained.empty:
        return empty_hits()

    per_collector = drained.groupby("initiator").agg(
        victims=("from_address", "nunique"),
        transfers=("from_address", "size"),
        transaction_hash=("transaction_hash", "first"),
    )
    per_collector = per_collector[per_collector["victims"] >= min_victims].reset_index()
    if per_collector.empty:
        return empty_hits()

    corroborated = per_collector["initiator"].isin(corroborating or set())
    evidence = (
        "used approvals to move tokens out of " + per_collector["victims"].astype(str)
        + " wallets that received nothing back ("
        + per_collector["transfers"].astype(str) + " transfers)"
    )
    evidence = evidence.where(
        ~corroborated, evidence + ", from an address that is newly deployed or already reported"
    )
    # A subscription or payroll contract also pulls approved tokens and returns nothing
    # on-chain. Without separate corroboration - the collector being brand new, or already
    # reported - this stays a candidate rather than an accusation.
    confidence = pd.Series(
        np.where(corroborated, CONFIRMED, CANDIDATE), index=per_collector.index
    )

    collector = build_hits(
        RULE, day, per_collector, "initiator", "collector", "transaction_hash",
        evidence, confidence,
    )
    victims = drained[drained["initiator"].isin(set(per_collector["initiator"]))]
    victims = victims.drop_duplicates(subset=["from_address"])
    victim_hits = build_hits(
        RULE, day, victims, "from_address", "victim", "transaction_hash",
        "tokens were moved out of this wallet by " + victims["initiator"],
        CANDIDATE,
    )
    return pd.concat([collector, victim_hits], ignore_index=True)
