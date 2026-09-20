"""One address format everywhere.

The source data is not consistent, and the inconsistencies are silent rather than loud:

* `token_transfers` stores addresses padded to 32 bytes (66 characters), while every other
  table uses the usual 20 bytes (42). Joining the two on the raw strings matches nothing,
  and nothing is exactly what a broken join looks like from the outside.
* Contract-creation rows carry the *string* `"None"` where the recipient would be, so a
  null check does not catch them and "None" becomes a very popular counterparty.
* Case is not meaningful in an Ethereum address, but it is meaningful to a join.

Everything is reduced to `0x` plus 40 lowercase hex characters on read, and anything that
is not an address becomes missing.
"""

from __future__ import annotations

import pandas as pd

ADDRESS_LENGTH = 40


def canonical_address(values: pd.Series) -> pd.Series:
    """Reduce any address spelling to 0x + 40 lowercase hex; non-addresses become NA."""
    text = values.astype("string").str.lower().str.removeprefix("0x")
    usable = text.str.fullmatch(f"[0-9a-f]{{{ADDRESS_LENGTH},}}").fillna(False)
    return ("0x" + text.str[-ADDRESS_LENGTH:]).where(usable, pd.NA)


def canonicalise_addresses(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply it to every address column of a frame, leaving other columns alone."""
    columns = [c for c in frame.columns if c == "address" or c.endswith("_address")]
    if not columns:
        return frame
    return frame.assign(**{column: canonical_address(frame[column]) for column in columns})
