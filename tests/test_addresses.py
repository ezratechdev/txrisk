import pandas as pd

from txrisk.data.addresses import canonical_address, canonicalise_addresses

PLAIN = "0x" + "a" * 40
PADDED = "0x" + "0" * 24 + "a" * 40


def test_padded_and_plain_spellings_become_the_same_address():
    # token_transfers pads to 32 bytes, every other table does not; a join across the two
    # silently matches nothing unless they are reduced to one form.
    values = canonical_address(pd.Series([PLAIN, PADDED]))
    assert values.iat[0] == values.iat[1] == PLAIN


def test_case_is_ignored():
    assert canonical_address(pd.Series(["0x" + "A" * 40])).iat[0] == PLAIN


def test_contract_creation_placeholders_become_missing():
    # Contract creations store the string "None", which a null check would not catch.
    values = canonical_address(pd.Series(["None", "", None, "0x00"]))
    assert values.isna().all()


def test_non_hex_values_are_rejected_rather_than_truncated():
    assert canonical_address(pd.Series(["not an address at all!!"])).isna().all()


def test_only_address_columns_are_touched():
    frame = pd.DataFrame({"from_address": [PADDED], "value": [1.0], "topics": [["0xabc"]]})
    out = canonicalise_addresses(frame)
    assert out["from_address"].iat[0] == PLAIN
    assert out["value"].iat[0] == 1.0
    assert out["topics"].iat[0] == ["0xabc"]
