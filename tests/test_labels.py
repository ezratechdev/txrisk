from datetime import date

import pytest

from txrisk.data.labels import (
    COLUMNS,
    detect_chain,
    normalise_address,
    parse_ofac,
    parse_scamsniffer,
)

COLLECTED = date(2026, 9, 20)

ETH_ADDRESS = "0x" + "A" * 40
BTC_ADDRESS = "1Ai5226mm9XvUAK4ZuFJnrgNbaAPPXJDbF"
TOKEN_ADDRESS = "0x" + "B" * 40

SDN_SAMPLE = "\n".join([
    f'123,"LAZARUS GROUP","-0- ","DPRK3","-0- ",,,,,,,'
    f'"Digital Currency Address - ETH {ETH_ADDRESS}; '
    f'Digital Currency Address - XBT {BTC_ADDRESS}"',
    '456,"SOME VESSEL","vessel","IRAN",,,,,,,,"No digital currency here"',
    '999,"TRON ENTITY","-0- ","CYBER2",,,,,,,,'
    '"Digital Currency Address - TRX TNiq9AXBp9EjUqhDhrwrfvAA8U3GUQZH81"',
    f'789,"MIXER LLC","-0- ","CYBER2",,,,,,,,'
    f'"Digital Currency Address - USDT {TOKEN_ADDRESS}"',
]) + "\n"


def test_detects_chain_from_address_shape():
    assert detect_chain("ETH", "0x" + "a" * 40) == "ethereum"
    assert detect_chain("XBT", BTC_ADDRESS) == "bitcoin"


def test_tokens_on_other_chains_are_not_mistaken_for_bitcoin():
    # USDT exists on Tron too; a Tron address filed under "bitcoin" would be a silent error.
    assert detect_chain("USDT", "TNiq9AXBp9EjUqhDhrwrfvAA8U3GUQZH81") is None
    assert detect_chain("XMR", "4AdUndXHHZ6cfufTMvppY6JwXNouMBzSkb") is None
    assert detect_chain("USDT", TOKEN_ADDRESS) == "ethereum"


def test_normalises_ethereum_case_but_leaves_bitcoin_alone():
    assert normalise_address("0xAbC" + "d" * 37, "ethereum") == "0xabc" + "d" * 37
    assert normalise_address("1AiBcD", "bitcoin") == "1AiBcD"


def test_parses_several_addresses_from_one_sanctions_entry():
    labels = parse_ofac(SDN_SAMPLE, COLLECTED)
    assert list(labels.columns) == COLUMNS
    assert len(labels) == 3
    lazarus = labels[labels["reference"] == "LAZARUS GROUP"]
    assert set(lazarus["chain"]) == {"ethereum", "bitcoin"}
    assert lazarus[lazarus["chain"] == "ethereum"]["address"].iat[0] == ETH_ADDRESS.lower()


def test_labels_carry_their_source_type_and_date():
    labels = parse_ofac(SDN_SAMPLE, COLLECTED)
    assert set(labels["source"]) == {"ofac_sdn"}
    assert set(labels["fraud_type"]) == {"sanctions"}
    assert set(labels["collected_on"]) == {COLLECTED}


def test_rows_without_addresses_are_ignored():
    assert parse_ofac('1,"NOBODY","-0-","PROG",,,,,,,,"nothing to see"\n', COLLECTED).empty


def test_token_entries_keep_their_ethereum_address():
    labels = parse_ofac(SDN_SAMPLE, COLLECTED)
    usdt = labels[labels["reference"] == "MIXER LLC"]
    assert usdt["chain"].iat[0] == "ethereum"


@pytest.mark.parametrize("duplicate_rows", [2, 3])
def test_the_same_address_from_one_source_is_recorded_once(duplicate_rows):
    text = SDN_SAMPLE.splitlines()[0] + "\n"
    assert len(parse_ofac(text * duplicate_rows, COLLECTED)) == 2


def test_parses_the_phishing_blacklist_and_keeps_only_addresses():
    text = f'["{ETH_ADDRESS}", "{TOKEN_ADDRESS.lower()}", "not-an-address", ""]'
    labels = parse_scamsniffer(text, COLLECTED)
    assert labels["address"].tolist() == [ETH_ADDRESS.lower(), TOKEN_ADDRESS.lower()]
    assert set(labels["fraud_type"]) == {"phishing"}
    assert set(labels["source"]) == {"scamsniffer"}


def test_phishing_and_sanctions_labels_share_one_schema():
    # The two sources describe different things; the models need one table.
    sanctions = parse_ofac(SDN_SAMPLE, COLLECTED)
    phishing = parse_scamsniffer(f'["{ETH_ADDRESS}"]', COLLECTED)
    assert list(sanctions.columns) == list(phishing.columns) == COLUMNS
