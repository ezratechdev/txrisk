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
from txrisk.data.labels import load_labels
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
from txrisk.features.graph import GRAPH_COLUMNS, graph_features_for_day, known_bad_before
from txrisk.paths import PROCESSED_DIR
from txrisk.report import markdown_table, save_report
from txrisk.serve import load_scorer

POISONING = "poisoning"
PHISHING = "phishing"
SPAM = "spam_token_airdrop"
OTHER = "other"
"""Neither fraud this model knows. An answer it can give, not merely a low score."""

FRAUD_TYPES = [POISONING, PHISHING, SPAM]
TARGET_COVERAGE = 0.9
"""Answer, rather than say "unknown", for this share of validation cases."""


def labelled_addresses(days: list[str]) -> dict[str, dict[str, set[str]]]:
    """Per fraud type, per day: the addresses doing it."""
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

    # Spam drops are only confirmed once the days after show nobody wanted the token.
    spam_hits = hits[(hits["rule"] == "spam_token_airdrop") & (hits["confidence"] == "confirmed")]
    spam = {day: set(group["address"]) for day, group in spam_hits.groupby("day")}

    return {
        POISONING: {d: poisoning.get(d, set()) for d in days},
        PHISHING: {d: phishing.get(d, set()) for d in days},
        SPAM: {d: spam.get(d, set()) for d in days},
    }


def load_labelled(
    days: list[str],
    with_graph: bool = False,
    other_per_day: int = 4000,
    seed: int = 0,
    scorer: object | None = None,
    risk_floor: float = 0.5,
) -> pd.DataFrame:
    """Addresses with a known type, plus a sample of everything else as `other`.

    `other` is what makes "unknown" sayable. Left to a confidence bar alone, the model can
    only express doubt by scoring low, and a classifier trained on two types has nothing to
    be uncertain between: withhold one and a single class remains, so it names that one for
    everything. Give it a third class of ordinary addresses and "neither of the frauds I
    know" becomes an answer it can give outright.

    Which addresses fill `other` decides what the scores mean. Drawn at random from the
    chain, it is mostly accounts no one would ever ask about, and the classifier is scored
    on a question it will never be asked. In use, the type head only ever sees addresses
    the risk model has already flagged, so `other` is drawn from those: risky, but neither
    of the frauds this model knows. That is the population it will meet.

    Either way the sample excludes any address a rule touched that day, in any role, so
    `other` means "not one of these frauds" rather than "fraud nobody happened to label".
    """
    by_type = labelled_addresses(days)
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    flagged_by_day = {day: set(group["address"]) for day, group in hits.groupby("day")}

    reported = set(load_labels().query("chain == 'ethereum'")["address"])
    rng = np.random.default_rng(seed)
    frames = []
    for day in days:
        features = features_for_day(day)
        if with_graph:
            graph = graph_features_for_day(day, known_bad_before(day, hits, reported))
            features = features.merge(graph, on="address", how="left")
            features[GRAPH_COLUMNS] = features[GRAPH_COLUMNS].fillna(0.0)

        kind = pd.Series("", index=features.index, dtype=object)
        for fraud_type in FRAUD_TYPES:
            kind[features["address"].isin(by_type[fraud_type][day])] = fraud_type

        untouched = (kind == "") & ~features["address"].isin(flagged_by_day.get(day, set()))
        if scorer is not None:
            risky = pd.Series(scorer.score(features) >= risk_floor, index=features.index)
            untouched &= risky
        candidates = np.flatnonzero(untouched.to_numpy())
        if len(candidates):
            chosen = rng.choice(candidates, size=min(other_per_day, len(candidates)), replace=False)
            kind.iloc[chosen] = OTHER

        keep = features[kind != ""].copy()
        keep["kind"] = kind[kind != ""].to_numpy()
        frames.append(keep)
        print(f"  {day}: {keep['kind'].value_counts().to_dict()}")
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
    for held in FRAUD_TYPES:
        train = split.train & (known["kind"] != held).to_numpy()
        probe = split.test & (known["kind"] == held).to_numpy()
        if not train.any() or not probe.any():
            continue
        X = known[features].to_numpy(dtype=np.float32)
        y = known["kind"].to_numpy()

        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced",
            random_state=seed, verbose=-1,
        ).fit(X[train], y[train])
        bars = thresholds_per_class(model.predict_proba(X[train]), model.classes_, coverage)
        answers = predict_with_rejection(model.predict_proba(X[probe]), model.classes_, bars)

        # Not naming the surviving fraud is the win: either "other" or an outright
        # abstention means the model did not misattribute one fraud to another.
        withheld_judgement = float(np.isin(answers, [OTHER, UNKNOWN]).mean())
        wrong_type = pd.Series(
            answers[~np.isin(answers, [OTHER, UNKNOWN])]
        ).value_counts().to_dict()
        lines.append(
            f"- **{held} withheld**: {withheld_judgement:.1%} of {int(probe.sum()):,} rows "
            f"answered \"{OTHER}\" or \"{UNKNOWN}\" rather than naming the fraud it did know"
            + (f"; {wrong_type} were misattributed." if wrong_type else ".")
        )
    return "\n".join(lines) if lines else "not enough types to withhold one"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", nargs="*", help="days to use (default: all extracted)")
    parser.add_argument("--test-days", nargs="*", help="days to score")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-graph", action="store_true", help="add who each address deals with"
    )
    parser.add_argument(
        "--other-from", choices=["risky", "anyone"], default="risky",
        help="where the 'other' class comes from: addresses the risk model flagged "
             "(what this meets in use), or any address at all",
    )
    parser.add_argument("--name", default="fraud_type", help="report name")
    args = parser.parse_args(argv)

    days = sorted(args.days or extracted_days())
    scorer = None
    if args.other_from == "risky":
        try:
            scorer = load_scorer()
        except FileNotFoundError:
            print("  no saved risk model; drawing 'other' from any address instead")
    data = load_labelled(days, with_graph=args.with_graph, scorer=scorer, seed=args.seed)
    test_days = args.test_days or days[-1:]

    known = data.reset_index(drop=True)
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
    present = [c for c in [*FRAUD_TYPES, OTHER] if c in set(truth)]
    scores = evaluate_open_set(truth, predicted, present)

    per_class = per_class_scores(truth, predicted, present)
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
    print(f"Report written to {save_report(args.name, report, scores)}")


if __name__ == "__main__":
    main()
