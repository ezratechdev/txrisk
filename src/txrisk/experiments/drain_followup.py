"""Do the drain candidates behave like thieves or like payment processors?

A single day cannot tell them apart: both pull approved tokens out of wallets and return
nothing on-chain. Several days can, because the two behave differently over time.

* **Repeat victims.** A subscription is paid again next week by the same wallet. Nobody
  volunteers to be drained twice.
* **Which tokens.** A payroll contract pulls the one token it deals in. A drainer takes
  whatever the wallet happens to hold.
* **What happens next.** A processor keeps its balance or spends it on its business. A
  drainer forwards everything onward and goes quiet.

None of these is proof on its own, so the report shows the evidence per collector rather
than a verdict, and says how many fall on each side.

    python -m txrisk.experiments.drain_followup
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from txrisk.data.ethereum import extracted_days, load_day
from txrisk.data.labels import load_labels
from txrisk.paths import PROCESSED_DIR
from txrisk.report import markdown_table, save_report

MANY_TOKENS = 3
"""Taking this many different tokens out of wallets looks like emptying them."""
REPEAT_SHARE = 0.1
"""Above this share of returning payers, the wallets are customers rather than victims."""


def collector_behaviour(collectors: set[str], days: list[str]) -> pd.DataFrame:
    """Per collector: who paid it, in what, how often, and where it went afterwards."""
    inflow, outflow = [], []
    for day in days:
        transfers = load_day(
            "token_transfers", day,
            ["from_address", "to_address", "value", "token_address", "transaction_hash"],
        )
        received = transfers[transfers["to_address"].isin(collectors)]
        inflow.append(received.assign(day=day))
        sent = transfers[transfers["from_address"].isin(collectors)]
        outflow.append(sent.assign(day=day))
        print(f"  {day}: {len(received):,} transfers in, {len(sent):,} out")

    received = pd.concat(inflow, ignore_index=True)
    sent = pd.concat(outflow, ignore_index=True)
    if received.empty:
        return pd.DataFrame()

    per_victim_days = received.groupby(["to_address", "from_address"])["day"].nunique()
    repeat_share = per_victim_days.gt(1).groupby(level=0).mean()

    summary = pd.DataFrame({
        "victims": received.groupby("to_address")["from_address"].nunique(),
        "tokens_taken": received.groupby("to_address")["token_address"].nunique(),
        "days_active": received.groupby("to_address")["day"].nunique(),
        "transfers_in": received.groupby("to_address").size(),
        "transfers_out": sent.groupby("from_address").size(),
        "repeat_victim_share": repeat_share,
    }).fillna(0.0)
    summary["forwarded_ratio"] = summary["transfers_out"] / summary["transfers_in"].clip(lower=1)
    summary.index.name = "collector"
    return summary.reset_index()


def reported_addresses() -> set[str]:
    """Addresses someone else has already reported for phishing, if the labels are built."""
    try:
        labels = load_labels()
    except FileNotFoundError:
        return set()
    return set(labels.loc[labels["fraud_type"] == "phishing", "address"])


def classify(summary: pd.DataFrame) -> pd.Series:
    """A reading of the evidence, not a verdict: which way each collector leans."""
    empties_wallets = summary["tokens_taken"] >= MANY_TOKENS
    nobody_returns = summary["repeat_victim_share"] < REPEAT_SHARE
    return pd.Series(
        [
            "drain-like" if empty and alone else "processor-like"
            for empty, alone in zip(empties_wallets, nobody_returns, strict=False)
        ],
        index=summary.index,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", nargs="*", help="days to follow (default: all extracted)")
    args = parser.parse_args(argv)

    days = args.days or extracted_days()
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    # Both rules point at the same kind of address: something that pulls other people's
    # tokens. Neither can judge it from one day, which is what this step is for.
    watched = hits[
        ((hits["rule"] == "token_drain") & (hits["role"] == "collector"))
        | ((hits["rule"] == "approval_phishing") & (hits["role"] == "spender"))
    ]
    collectors = set(watched["address"])
    print(f"following {len(collectors)} candidates over {len(days)} day(s)")

    summary = collector_behaviour(collectors, days)
    if summary.empty:
        print("no activity found for these addresses")
        return
    summary["leaning"] = classify(summary)
    reported = reported_addresses()
    summary["on_blacklist"] = summary["collector"].isin(reported)
    counts = summary["leaning"].value_counts()

    # What a later phase can train on: behaviour over the window, or an independent report.
    promoted = summary[(summary["leaning"] == "drain-like") | summary["on_blacklist"]]
    labels = pd.DataFrame({
        "address": promoted["collector"],
        "fraud_type": "phishing",
        "basis": np.where(promoted["on_blacklist"], "reported", "behaviour over the window"),
        "tokens_taken": promoted["tokens_taken"],
        "victims": promoted["victims"],
    })
    labels.to_parquet(PROCESSED_DIR / "promoted_labels.parquet", index=False)
    print(f"\n{len(labels)} addresses promoted to phishing labels "
          f"({int(promoted['on_blacklist'].sum())} of them independently reported)")

    ranked = summary.nlargest(15, "victims")[
        ["collector", "victims", "tokens_taken", "days_active",
         "repeat_victim_share", "forwarded_ratio", "leaning"]
    ]
    report = f"""# Drain candidates over time

Generated by `python -m txrisk.experiments.drain_followup` over {len(days)} day(s):
{days[0]} to {days[-1]}, following {len(collectors)} collectors the drain rule flagged.

{markdown_table(counts.rename_axis("leaning").reset_index(name="collectors"))}

"Drain-like" means the collector took {MANY_TOKENS} or more different tokens out of wallets
and almost none of those wallets ever paid it again. "Processor-like" means the opposite:
few token types, or payers who come back. Neither is proof, and {len(days)} days is a short
window to judge "never came back" on.

## The busiest candidates

{markdown_table(ranked)}

## How to read this

- `repeat_victim_share` is the share of payers that paid on more than one day. Customers
  return; drained wallets do not.
- `tokens_taken` is how many different tokens the collector pulled. A payroll contract
  deals in one. Emptying a wallet takes whatever is in it.
- `forwarded_ratio` is outgoing transfers per incoming one. A collector that immediately
  forwards everything is moving stolen funds on; a business holds and spends its balance.
"""
    print(f"\n{counts.to_dict()}")
    print(f"Report written to {save_report('drain_followup', report, counts.to_dict())}")


if __name__ == "__main__":
    main()
