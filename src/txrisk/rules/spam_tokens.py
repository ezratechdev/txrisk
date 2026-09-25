"""Spam token airdrops: a worthless token pushed at thousands of wallets at once.

The distributor mints something, sends it to every address it can find, and waits for
people to look the token up, land on a site the token's name advertises, and sign
something they should not have. The transfer itself costs the victim nothing, which is why
it is worth doing at scale.

Two things separate it from a real airdrop, and neither is the size of the distribution:

* **Nobody does anything with it.** People move tokens they wanted - to an exchange, to a
  pool, to another wallet. A token nobody touches afterwards is a token nobody asked for.
* **It arrives from nowhere.** A genuine airdrop rewards people who used something; a spam
  drop goes to addresses picked off the chain, and the token itself was minted days ago.

The first is the test that matters, and it is the one that needs more than a single day to
answer honestly: a real airdrop's recipients need time to react.
"""

from __future__ import annotations

import pandas as pd

from txrisk.rules import CANDIDATE, CONFIRMED, build_hits, empty_hits

RULE = "spam_token_airdrop"
MIN_RECIPIENTS = 200
"""Below this it is a distribution, not a campaign."""
IGNORED_SHARE = 0.95
"""When this share of recipients never touch the token again, nobody asked for it."""


def find_spam_airdrops(
    token_transfers: pd.DataFrame,
    day: str,
    later_transfers: pd.DataFrame | None = None,
    new_tokens: set[str] | None = None,
    min_recipients: int = MIN_RECIPIENTS,
) -> pd.DataFrame:
    """Senders pushing one token to thousands of wallets that never use it.

    `later_transfers` covers the days after this one, and is what decides whether the
    recipients wanted the token. Without it every hit stays a candidate, because on the day
    itself a spam drop and a real airdrop look exactly alike.
    """
    if token_transfers.empty:
        return empty_hits()

    moved = token_transfers.dropna(subset=["from_address", "to_address", "token_address"])
    campaigns = moved.groupby(["from_address", "token_address"]).agg(
        recipients=("to_address", "nunique"),
        transfers=("to_address", "size"),
        transaction_hash=("transaction_hash", "first"),
    )
    campaigns = campaigns[campaigns["recipients"] >= min_recipients].reset_index()
    if campaigns.empty:
        return empty_hits()

    rows = []
    for campaign in campaigns.itertuples():
        recipients = set(
            moved.loc[
                (moved["from_address"] == campaign.from_address)
                & (moved["token_address"] == campaign.token_address),
                "to_address",
            ]
        )
        ignored_share, confidence = _how_unwanted(
            campaign.token_address, recipients, later_transfers
        )
        minted_recently = campaign.token_address in (new_tokens or set())
        evidence = (
            f"sent one token to {campaign.recipients:,} wallets in a day"
            + (", from a token minted in this window" if minted_recently else "")
        )
        if ignored_share is not None:
            evidence += f"; {ignored_share:.0%} of them never touched it again"
        rows.append({
            "from_address": campaign.from_address,
            "transaction_hash": campaign.transaction_hash,
            "evidence": evidence,
            "confidence": confidence,
        })

    frame = pd.DataFrame(rows)
    return build_hits(
        RULE, day, frame, "from_address", "distributor", "transaction_hash",
        frame["evidence"], frame["confidence"],
    )


def _how_unwanted(
    token: str, recipients: set[str], later_transfers: pd.DataFrame | None
) -> tuple[float | None, str]:
    """What share of recipients never moved the token on, once they had the chance."""
    if later_transfers is None or later_transfers.empty or not recipients:
        return None, CANDIDATE
    used = later_transfers[
        (later_transfers["token_address"] == token)
        & (later_transfers["from_address"].isin(recipients))
    ]
    ignored = 1.0 - (used["from_address"].nunique() / len(recipients))
    return ignored, CONFIRMED if ignored >= IGNORED_SHARE else CANDIDATE
