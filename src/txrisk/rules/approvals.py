"""Approval phishing: many wallets granting spending rights to a brand-new contract.

A drainer works by persuading people to sign an approval. The approval itself is an
ordinary transaction that a wallet shows as harmless, and the theft happens later when
the spender moves the tokens.

Approval counts alone cannot find this: the busiest spender in one day of Ethereum took
21,251 approvals and was a trading router that everyone uses. What separates a drainer is
age. A contract deployed hours ago that hundreds of strangers have already approved is
not infrastructure, whereas a router has been there for years. This rule therefore only
considers spenders whose deployment is inside the window being scanned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED, build_hits, empty_hits

RULE = "approval_phishing"
MIN_DISTINCT_OWNERS = 20


def address_from_topic(topic: pd.Series) -> pd.Series:
    """An address in a log topic is padded to 32 bytes; the address is the last 20."""
    return "0x" + topic.str.removeprefix("0x").str[-40:].str.lower()


def decode_approvals(approvals: pd.DataFrame) -> pd.DataFrame:
    """Turn raw Approval/ApprovalForAll logs into owner, spender and token columns."""
    if approvals.empty:
        return pd.DataFrame(columns=["owner", "spender", "token_address", "transaction_hash"])
    topics = approvals["topics"]
    usable = approvals[topics.str.len() >= 3].copy()
    if usable.empty:
        return pd.DataFrame(columns=["owner", "spender", "token_address", "transaction_hash"])
    usable["owner"] = address_from_topic(usable["topics"].str[1])
    usable["spender"] = address_from_topic(usable["topics"].str[2])
    return usable.rename(columns={"address": "token_address"})[
        ["owner", "spender", "token_address", "transaction_hash"]
    ]


def count_sweeps(spender: str, owners: set[str], token_transfers: pd.DataFrame) -> int:
    """How many of those owners then sent tokens to the spender itself."""
    if token_transfers.empty:
        return 0
    moved = token_transfers[
        (token_transfers["to_address"] == spender)
        & (token_transfers["value"] > 0)
        & (token_transfers["from_address"].isin(owners))
    ]
    return int(moved["from_address"].nunique())


def find_approval_phishing(
    approvals: pd.DataFrame,
    contracts: pd.DataFrame,
    day: str,
    token_transfers: pd.DataFrame | None = None,
    min_owners: int = MIN_DISTINCT_OWNERS,
) -> pd.DataFrame:
    """Freshly deployed spenders approved by many distinct wallets.

    Only contracts deployed within the scanned window are considered, which is what keeps
    established routers and exchanges out: on one day of Ethereum, 218 spenders were
    approved by 20 or more wallets and only 2 of them had been deployed that day.

    Being new and popular is not yet fraud, though. A protocol that launched this morning
    looks the same. The hit is only `confirmed` once tokens are seen moving from the
    approving wallets into the spender, which is the drain itself.
    """
    decoded = decode_approvals(approvals)
    if decoded.empty or contracts.empty:
        return empty_hits()

    new_contracts = set(contracts["address"].dropna())
    suspect = decoded[decoded["spender"].isin(new_contracts)]
    if suspect.empty:
        return empty_hits()

    per_spender = suspect.groupby("spender").agg(
        victims=("owner", "nunique"),
        approvals=("owner", "size"),
        transaction_hash=("transaction_hash", "first"),
    )
    per_spender = per_spender[per_spender["victims"] >= min_owners].reset_index()
    if per_spender.empty:
        return empty_hits()

    transfers = pd.DataFrame() if token_transfers is None else token_transfers
    per_spender["swept"] = [
        count_sweeps(spender, set(suspect.loc[suspect["spender"] == spender, "owner"]), transfers)
        for spender in per_spender["spender"]
    ]

    drained = per_spender["swept"] > 0
    evidence = (
        per_spender["victims"].astype(str) + " distinct wallets approved this spender ("
        + per_spender["approvals"].astype(str) + " approvals) on the day it was deployed"
    )
    evidence = evidence.where(
        ~drained,
        evidence + "; tokens then moved from " + per_spender["swept"].astype(str) + " of them",
    )
    confidence = pd.Series(np.where(drained, CONFIRMED, CANDIDATE), index=per_spender.index)
    return build_hits(
        RULE, day, per_spender, "spender", "spender", "transaction_hash", evidence, confidence
    )
