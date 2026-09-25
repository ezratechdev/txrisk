"""An MCP server, so an assistant can ask this model about an address or a transaction.

The models sit behind command-line experiments, which suits research and suits nobody
else. This exposes the three questions worth asking, over a protocol an assistant already
speaks:

* `score_address` - how risky is this address, and why
* `score_transaction` - how risky is this transaction, and which party makes it so
* `explain_rules` - what the deterministic detectors found, with no model involved

Every answer carries its evidence, and the tools decline rather than guess when a day has
not been extracted or the model has not been trained. An assistant that receives a number
with no provenance will repeat it as fact.

Run it with:

    uv run python -m txrisk.mcp_server

and point an MCP client at the command. See docs/mcp.md.
"""

from __future__ import annotations

import json
from typing import Any

from txrisk.data.ethereum import extracted_days
from txrisk.serve import address_scores, band, load_scorer, rule_reasons, score_transactions

try:  # pragma: no cover - exercised only when the optional dependency is installed
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None

SERVER_NAME = "txrisk"


def available_days() -> list[str]:
    try:
        return extracted_days()
    except FileNotFoundError:
        return []


def score_address_tool(address: str, day: str) -> dict[str, Any]:
    """Risk for one address on one day, with the rules that fired and the numbers behind it."""
    address = address.strip().lower()
    days = available_days()
    if day not in days:
        return {
            "error": f"{day} has not been extracted",
            "extracted_days": days,
            "how_to_fix": f"uv run python -m txrisk.data.ethereum --dates {day}",
        }

    scored = address_scores(day, load_scorer())
    row = scored[scored["address"] == address]
    if row.empty:
        return {"address": address, "day": day, "active": False,
                "note": "this address did nothing on this day, so there is nothing to score"}

    risk = float(row["risk"].iat[0])
    reasons = rule_reasons(day).get(address, [])
    return {
        "address": address,
        "day": day,
        "active": True,
        "risk_score": round(risk, 4),
        "risk_band": band(risk),
        "reasons": reasons or ["no rule fired; this is the model's judgement alone"],
        "behaviour": {
            "transfers_in": int(row["token_in_count"].iat[0]),
            "transfers_out": int(row["token_out_count"].iat[0]),
            "distinct_tokens": int(row["distinct_tokens"].iat[0]),
            "counterparties_in": int(row["token_in_counterparties"].iat[0]),
        },
        "caveat": "a risk indicator, not proof; victims and attackers appear in the same data",
    }


def score_transaction_tool(day: str, top: int = 10) -> dict[str, Any]:
    """The riskiest transactions of a day, each with the party that makes it risky."""
    days = available_days()
    if day not in days:
        return {"error": f"{day} has not been extracted", "extracted_days": days}
    return {"day": day, "transactions": score_transactions(day, load_scorer(), top=top)}


def explain_rules_tool(address: str, day: str) -> dict[str, Any]:
    """What the deterministic detectors found for an address. No model, only evidence."""
    address = address.strip().lower()
    reasons = rule_reasons(day).get(address, [])
    return {
        "address": address,
        "day": day,
        "rule_hits": reasons,
        "note": "rules are checkable by hand against the chain; a model score is not",
    }


def build_server():  # pragma: no cover - needs the optional dependency
    if FastMCP is None:
        raise ImportError(
            "the MCP server needs the 'mcp' package: uv add mcp, or uv sync --extra mcp"
        )
    server = FastMCP(SERVER_NAME)
    server.tool()(score_address_tool)
    server.tool()(score_transaction_tool)
    server.tool()(explain_rules_tool)
    return server


def main() -> None:  # pragma: no cover
    if FastMCP is None:
        print(json.dumps({
            "error": "the 'mcp' package is not installed",
            "install": "uv sync --extra mcp",
        }, indent=2))
        raise SystemExit(1)
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
