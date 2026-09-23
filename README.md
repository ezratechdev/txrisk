# txrisk

Open-source models that estimate how likely a blockchain transaction is to be fraudulent
and, where the on-chain evidence allows, what kind of fraud it is.

**Status: working end to end on Ethereum, for two kinds of fraud.** Address poisoning is
detected reliably; phishing is detected where the evidence exists. Everything here is
evidence-first: every score carries the reasons behind it, and the reports in
[reports/](reports/) record what does not work as carefully as what does.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync

# Known-bad addresses, and when they were active (no account needed for either)
uv run python -m txrisk.data.labels
uv run python -m txrisk.data.activity

# A window of Ethereum. Pick days where fraud actually happened, not just any week.
uv run python -m txrisk.data.ethereum --dates 2023-02-24 2023-02-25 2023-02-26

# Deterministic detectors, then the models
uv run python -m txrisk.experiments.rule_scan
uv run python -m txrisk.experiments.risk_model
uv run python -m txrisk.experiments.fraud_type --with-graph

# Score real transactions, and watch the model age
uv run python -m txrisk.serve train --days 2023-02-24 2023-02-25 2023-02-26
uv run python -m txrisk.serve score --day 2023-02-27 --top 5
uv run python -m txrisk.monitor --days 2023-02-27
```

Data lands in `data/`, which git ignores. Set `TXRISK_DATA_DIR` to keep it on another drive.

## What it finds

| Rule | What it looks for | Precision against an independent blacklist |
|---|---|---|
| `address_poisoning` | Zero-value transfers to an address imitating one the victim really pays | fraud by construction |
| `token_drain` | Approval-backed sweeps where the owner receives nothing back | 33% (a 1,616× enrichment) |
| `approval_phishing` | New contracts many wallets approve, then emptied of several tokens | candidates for the window step to judge |
| `known_bad_exposure` | Either party on a reported list | exact |

| Model | Measured on |
|---|---|
| Risk score: 0.75 precision, 14.5× lift, calibration error 0.007 | later days, addresses never seen in training |
| Fraud type: poisoning F1 0.990, phishing F1 0.936 | same, with the rules' own definitions removed from the inputs |

## How this is evaluated

These rules are why the numbers above are believable, and each was learned the hard way.

- **Split by time and by actor.** Test rows for an address seen in training are dropped.
  A third of one day's attackers were active earlier; without this the model scores well
  for recognising them rather than for generalising.
- **Remove the features that spell out the rule.** Labels come from rules, so a model can
  re-learn the rule and look excellent. Poisoning is *defined* by a zero-value transfer;
  leaving that feature in turns a hard problem into a lookup.
- **Compare with lift, not PR-AUC.** Fraud is several times more common in some windows
  than others, which moves PR-AUC on its own.
- **Let the labels choose the days.** A quiet week contained one reported phishing address
  a day; a targeted window contained 135–160. No modelling survives the wrong window.
- **Separate what a rule proves from what it suspects.** A staking contract pulls approved
  tokens and returns nothing, exactly like a drain. Confirmation needs corroboration.
- **Watch calibration, not ranking.** A model three and a half years stale ranked better
  than ever on one day while its probabilities were four times worse.

## Datasets

| Name | Contents | Source |
|---|---|---|
| `ethereum` | Transactions, token transfers, approvals, ETH movements, new contracts | [AWS public blockchain dataset](https://registry.opendata.aws/aws-public-blockchain/) (MIT-0, no account) |
| `labels` | Sanctioned and reported phishing addresses | OFAC SDN (public domain), ScamSniffer (GPL-3.0) |
| `activity` | When reported addresses were active | Blockscout public API |
| `elliptic`, `bitcoinheist` | Bitcoin benchmarks used to develop the method | Weber et al. 2019; Akcora et al. 2019 |

This repository ships fetchers, never label data: each source is downloaded under its own
terms. About 8% of each day in the bucket is kept — raw call data, non-approval events and
traces that moved no ETH are dropped — and days stream one at a time, so disk use stays
flat over any range.

## Responsible use

Scores are risk indicators, not proof of wrongdoing, and the model's most confident
outputs are not always its most reliable: the top-ranked transaction on one day was an
ordinary address with four incoming transfers. Don't publish lists of addresses flagged by
these models. False positives point at real people, and a victim and an attacker appear in
the same row of chain data.
