"""Deterministic detectors for fraud patterns that leave an unambiguous trace.

Rules come before models on purpose. They are precise, they explain themselves, and a
person can check whether a hit is right. They also produce the labels the models need:
the reported-scam lists are far too small to train on, as one day of Ethereum showed by
containing no sanctioned address at all.

Each rule takes a day's tables and returns hits in one shape, so the engine can run them
all and write a single table:

    rule, day, address, role, confidence, transaction_hash, evidence

`address` is the account the hit is about and `role` says how it was involved, because
"the victim" and "the attacker" appear in the same row of chain data and must not be
confused when these hits become training labels.

`confidence` separates what a rule proves from what it merely suspects. A zero-value
transfer to a look-alike address is fraud by construction, so it is `confirmed`. A new
contract that many wallets approved might equally be a protocol that launched that
morning, so it is a `candidate` until the tokens are seen to move. Training on candidates
as though they were confirmed would teach a model that every popular new contract is a
drainer.
"""

from __future__ import annotations

import pandas as pd

HIT_COLUMNS = ["rule", "day", "address", "role", "confidence", "transaction_hash", "evidence"]

CONFIRMED = "confirmed"
CANDIDATE = "candidate"


def empty_hits() -> pd.DataFrame:
    return pd.DataFrame(columns=HIT_COLUMNS)


def build_hits(
    rule: str, day: str, frame: pd.DataFrame, address: str, role: str,
    transaction_hash: str, evidence: pd.Series, confidence: str | pd.Series = CONFIRMED,
) -> pd.DataFrame:
    """Assemble hits from a rule's working frame, naming the columns it used."""
    if frame.empty:
        return empty_hits()
    return pd.DataFrame({
        "rule": rule,
        "day": day,
        "address": frame[address].to_numpy(),
        "role": role,
        "confidence": confidence.to_numpy() if isinstance(confidence, pd.Series) else confidence,
        "transaction_hash": frame[transaction_hash].to_numpy(),
        "evidence": evidence.to_numpy(),
    })[HIT_COLUMNS]
