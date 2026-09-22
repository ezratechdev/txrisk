"""Phase 5: which kind of fraud, and the honesty to say "unknown".

Two types have evidence behind them now: address poisoning, confirmed by construction, and
phishing, from addresses other people reported plus collectors the window step promoted.
They arrive in wildly different quantities - tens of thousands against hundreds - so the
first thing this has to avoid is a classifier that answers "poisoning" forever and scores
well doing it. Every class is therefore weighted by its rarity and judged on macro-averaged
metrics, where a class of 600 counts as much as a class of 50,000.

The second thing it has to avoid is confidence about fraud it has never seen. Dozens of
types exist on chain and two are labelled here, so an unfamiliar one is the normal case,
not the exception. The classifier may answer "unknown", and is tested on whether it does:
drain victims, who are neither poisoning attackers nor phishing collectors, are put in
front of it precisely to see whether it invents a label for them.

    python -m txrisk.experiments.fraud_type
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from txrisk.data.ethereum import extracted_days
from txrisk.evaluation.metrics import (
    UNKNOWN,
    evaluate_open_set,
    per_class_scores,
    predict_with_rejection,
    thresholds_per_class,
)
from txrisk.evaluation.splits import drop_overlapping_groups, temporal_split
from txrisk.experiments.risk_model import IDENTIFIERS, RULE_SIGNALS
from txrisk.features.address_day import features_for_day
from txrisk.paths import PROCESSED_DIR
from txrisk.report import markdown_table, save_report

POISONING = "poisoning"
PHISHING = "phishing"
TARGET_COVERAGE = 0.9
"""Answer, rather than say "unknown", for this share of validation cases."""


def labelled_addresses(days: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Per day: the poisoning attackers, and the addresses tied to phishing."""
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    attackers = hits[
        (hits["rule"] == "address_poisoning")
        & (hits["role"] == "attacker")
        & (hits["confidence"] == "confirmed")
    ]
    poisoning = {day: set(group["address"]) for day, group in attackers.groupby("day")}

    # Phishing has to be something the address *did* that day, not something it once was.
    # Labelling every reported address on every day it drew breath put 155 rows of pure
    # inactivity into the class - median token transfers in and out both zero - and no
    # model can separate "was on a list once" from an ordinary quiet day.
    doing_phishing = hits[
        ((hits["rule"] == "token_drain") & (hits["role"] == "collector"))
        | ((hits["rule"] == "approval_phishing") & (hits["role"] == "spender"))
        | ((hits["rule"] == "known_bad_exposure") & (hits["role"] == "receiver"))
    ]
    phishing = {day: set(group["address"]) for day, group in doing_phishing.groupby("day")}
    return {d: poisoning.get(d, set()) for d in days}, {d: phishing.get(d, set()) for d in days}


def load_labelled(days: list[str]) -> pd.DataFrame:
    """Only the addresses with a known type, plus the drain victims used as a test."""
    poisoning, phishing = labelled_addresses(days)
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    victims = hits[(hits["rule"] == "token_drain") & (hits["role"] == "victim")]
    victims_by_day = {day: set(group["address"]) for day, group in victims.groupby("day")}

    frames = []
    for day in days:
        features = features_for_day(day)
        kind = pd.Series("", index=features.index, dtype=object)
        kind[features["address"].isin(poisoning[day])] = POISONING
        kind[features["address"].isin(phishing[day])] = PHISHING
        kind[
            (kind == "") & features["address"].isin(victims_by_day.get(day, set()))
        ] = "drain victim"
        keep = features[kind != ""].copy()
        keep["kind"] = kind[kind != ""].to_numpy()
        frames.append(keep)
        counts = keep["kind"].value_counts().to_dict()
        print(f"  {day}: {counts}")
    return pd.concat(frames, ignore_index=True)


def hold_out_each_type(
    known: pd.DataFrame, split, features: list[str], seed: int, coverage: float = TARGET_COVERAGE
) -> str:
    """Train without one type, then show it that type. Does it abstain, or guess?

    Drain victims were the wrong probe for this: a victim is the other role in the same
    fraud, so calling them phishing is not the mistake it looks like. Withholding a whole
    type is the real question, and the one a new kind of fraud will ask.
    """
    lines = []
    for held in sorted(set(known["kind"])):
        train = split.train & (known["kind"] != held).to_numpy()
        probe = split.test & (known["kind"] == held).to_numpy()
        if not train.any() or not probe.any() or known.loc[train, "kind"].nunique() < 1:
            continue
        X = known[features].to_numpy(dtype=np.float32)
        y = known["kind"].to_numpy()

        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced",
            random_state=seed, verbose=-1,
        ).fit(X[train], y[train])
        if len(model.classes_) < 2:
            # Only one type left to learn, so every probability is 1.0 and no bar can be
            # set from the data. Confidence cannot express doubt it was never shown.
            lines.append(
                f"- **{held} withheld**: only one type remained in training, so the model "
                f"had no way to express doubt and named it for all {int(probe.sum()):,} rows."
            )
            continue
        bars = thresholds_per_class(model.predict_proba(X[train]), model.classes_, coverage)
        answers = predict_with_rejection(model.predict_proba(X[probe]), model.classes_, bars)
        rejected = float((answers == UNKNOWN).mean())
        named = pd.Series(answers[answers != UNKNOWN]).value_counts().to_dict()
        lines.append(
            f"- **{held} withheld**: {rejected:.1%} of {int(probe.sum()):,} rows answered "
            f"\"unknown\", the rest called {named}."
        )
    return "\n".join(lines) if lines else "not enough types to withhold one"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", nargs="*", help="days to use (default: all extracted)")
    parser.add_argument("--test-days", nargs="*", help="days to score")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    days = sorted(args.days or extracted_days())
    data = load_labelled(days)
    test_days = args.test_days or days[-1:]

    known = data[data["kind"].isin([POISONING, PHISHING])].reset_index(drop=True)
    split = temporal_split(known["day"].isin(test_days).to_numpy().astype(int), train_until=0)
    split = drop_overlapping_groups(split, known["address"].to_numpy())

    every_feature = [c for c in known.columns if c not in IDENTIFIERS + ["kind"]]
    # The rules key on transfers worth nothing, and poisoning is defined by receiving one.
    # Leaving those features in lets the classifier recover the rule instead of the fraud.
    features = [c for c in every_feature if c not in RULE_SIGNALS]
    X = known[features].to_numpy(dtype=np.float32)
    y = known["kind"].to_numpy()

    train_rows = np.flatnonzero(split.train)
    cut = int(len(train_rows) * 0.8)
    fitting, calibrating = train_rows[:cut], train_rows[cut:]

    # Weighted by rarity: without it the answer is "poisoning" every time, and the score
    # for saying so is high.
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
        random_state=args.seed, verbose=-1,
    )
    model.fit(X[fitting], y[fitting])

    # One bar per class: calibrated on a slice that is almost entirely poisoning, a
    # single bar sits at poisoning's confidence and answers "unknown" for every phishing
    # row, including the ones the model got right.
    threshold = thresholds_per_class(
        model.predict_proba(X[calibrating]), model.classes_, TARGET_COVERAGE
    )
    predicted = predict_with_rejection(
        model.predict_proba(X[split.test]), model.classes_, threshold
    )
    truth = y[split.test]
    scores = evaluate_open_set(truth, predicted, [POISONING, PHISHING])

    per_class = per_class_scores(truth, predicted, [POISONING, PHISHING])
    counts = pd.Series(truth).value_counts().rename_axis("type").reset_index(name="test rows")

    bars = ", ".join(f"{name} {bar:.4f}" for name, bar in sorted(threshold.items()))
    unknown_line = hold_out_each_type(known, split, features, args.seed)

    report = f"""# Fraud type

Generated by `python -m txrisk.experiments.fraud_type --seed {args.seed}` over
{len(days)} day(s), scoring {', '.join(test_days)}.

{markdown_table(counts)}

Trained on the earlier days, scored on the later ones, with any address seen in training
dropped from the test set. The features that spell out the rules are removed: poisoning is
*defined* as receiving a transfer worth nothing, so leaving those in would let the
classifier recover the rule rather than learn what the fraud looks like.

Classes are weighted by rarity, and `known_macro_f1` averages over the types so the
smaller one counts as much as the larger.

## Scores

{markdown_table(pd.DataFrame([scores]))}

## Per type

{markdown_table(per_class)}

Confidence bars, one per class, set on held-out training data: {bars}.

## Fraud it was never taught

Each type withheld from training in turn, then shown to the model:

{unknown_line}

A classifier that names a trained type for an unfamiliar fraud is worse than one that
abstains, because the name travels further than the doubt. With two types labelled out of
the dozens that exist, meeting an unfamiliar one is the ordinary case, not an edge case.

## Limits

- Phishing labels come from reports and from the window step, both of which lag reality.
  The class is small, and its members were active in one eight-day window in 2023.
- Poisoning labels come from a rule, so this classifier inherits whatever that rule
  misses.
- The confidence cut-off is set for a {TARGET_COVERAGE:.0%} answer rate on held-out
  training data, not tuned on the test days.
"""
    print(f"\n{scores}")
    print(f"{unknown_line}")
    print(f"Report written to {save_report('fraud_type', report, scores)}")


if __name__ == "__main__":
    main()
