# Reaching the model from an assistant

The models live behind command-line experiments, which suits research and suits nobody
else. The MCP server puts them behind three questions an assistant can ask directly.

## Setup

```sh
uv sync --extra mcp

# The server answers from extracted days and a trained model, so both have to exist.
uv run python -m txrisk.data.ethereum --dates 2023-03-01 2023-03-02
uv run python -m txrisk.experiments.rule_scan --days 2023-03-01 2023-03-02
uv run python -m txrisk.serve train --days 2023-03-01
```

Point an MCP client at the command. For Claude Code:

```sh
claude mcp add txrisk -- uv run --directory /path/to/txrisk python -m txrisk.mcp_server
```

Or, in a client that reads a JSON config:

```json
{
  "mcpServers": {
    "txrisk": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/txrisk", "python", "-m", "txrisk.mcp_server"]
    }
  }
}
```

## The tools

| Tool | Question | Answers with |
|---|---|---|
| `score_address_tool` | How risky is this address on this day? | Score, band, rules that fired, the behaviour behind it |
| `score_transaction_tool` | Which transactions on this day look worst? | Ranked transactions, each naming the riskier party |
| `explain_rules_tool` | What did the detectors find? | Rule hits only — no model in the answer |

`explain_rules_tool` exists because a rule hit can be checked by hand against the chain
and a model score cannot. When the two disagree, the rule is the one to trust.

## What it will not do

The tools decline rather than guess. Asking about a day that has not been extracted
returns the list of days that have, and the command that would fetch the missing one.
Asking about an address that did nothing that day says so, instead of returning a
confident zero.

Every answer carries its reasons and a caveat, because an assistant handed a bare number
will repeat it as fact. These scores are risk indicators, not findings of wrongdoing, and
a victim and an attacker appear in the same row of chain data.

## Scope

Ethereum only. The model knows address poisoning, phishing, and spam-token airdrops; for
anything else its answer is `other`, which means "not one of the frauds I know" rather
than "harmless". A day must be extracted before it can be asked about — this reads local
Parquet files, not a live node.
