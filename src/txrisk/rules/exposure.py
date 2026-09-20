"""Direct exposure to an address someone has already reported.

The plainest signal there is, and the baseline every model has to beat: did this
transaction touch a known-bad address? It is also the most limited, because it only
knows what has already been reported. One day of Ethereum contained no sanctioned
address at all, so this rule finds nothing on its own most days. It earns its place by
being certain when it does fire, and by seeding the labels other work depends on.
"""

from __future__ import annotations

import pandas as pd

from txrisk.rules import build_hits, empty_hits

RULE = "known_bad_exposure"


def find_known_bad_exposure(
    transfers: pd.DataFrame,
    known_bad: dict[str, str],
    day: str,
    sender_column: str = "from_address",
    receiver_column: str = "to_address",
) -> pd.DataFrame:
    """Transfers where either side is on a known-bad list.

    `known_bad` maps address to why it is listed, so the hit can say which list named it
    rather than only that something matched.
    """
    if transfers.empty or not known_bad:
        return empty_hits()

    hits = []
    for column, role in ((sender_column, "sender"), (receiver_column, "receiver")):
        side = transfers[transfers[column].isin(known_bad)]
        if side.empty:
            continue
        evidence = side[column].map(known_bad).radd(f"{role} is listed: ")
        hits.append(build_hits(RULE, day, side, column, role, "transaction_hash", evidence))
    return pd.concat(hits, ignore_index=True) if hits else empty_hits()
