"""Phase 4: a calibrated risk score per address, with reasons.

The honest starting point is that there is almost nothing to learn from. Of 2,530
reported phishing addresses, one was active on the day extracted here, among 1.2 million
active addresses. The rules, by contrast, confirm about 60,000 poisoning attackers a day.

So this model is trained on what the rules confirm, and that creates a circularity: a
model taught by a rule can simply re-learn the rule. The experiment therefore trains
twice, once with every feature and once with the rule's own signals removed. The second
score is the honest one. If a model that cannot see "sent transfers worth nothing" still
finds these addresses, it has learned how the campaigns behave rather than how the rule
is written.

    python -m txrisk.experiments.risk_model
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from txrisk.data.ethereum import extracted_days
from txrisk.data.labels import load_labels
from txrisk.evaluation.metrics import evaluate
from txrisk.evaluation.splits import (
    Split,
    drop_overlapping_groups,
    random_split,
    temporal_split,
)
from txrisk.features.address_day import features_for_day
from txrisk.paths import PROCESSED_DIR
from txrisk.report import markdown_table, save_report

RULE_SIGNALS = [
    "token_out_zero_value",
    "token_in_zero_value",
    "tx_out_zero_value",
    "tx_in_zero_value",
    "zero_value_share",
]
"""Every feature that counts transfers worth nothing, on both sides.

Incoming matters as much as outgoing, and missing that made the first version of this
test meaningless. A poisoning attacker is labelled because a zero-value transfer arrived
at their address, so `token_in_zero_value` alone almost gives the label away. Leaving it
in let the "honest" run score exactly like the other one.
"""

IDENTIFIERS = ["address", "day"]


def load_dataset(
    days: list[str],
    negative_rate: float = 1.0,
    scored_days: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Address-day behaviour, labelled by what the rules confirmed that day.

    Fifteen days of every address on Ethereum does not fit in the memory this machine has
    spare, and most of it is the same innocent account doing the same innocent thing. Days
    that will only be trained on can therefore keep a sample of their negatives, while days
    that will be scored keep every row: thinning the test set would flatter the result.
    Positives are never dropped.
    """
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    attackers = hits[(hits["role"] == "attacker") & (hits["confidence"] == "confirmed")]
    attackers_by_day = {day: set(group["address"]) for day, group in attackers.groupby("day")}
    scored = set(scored_days or [])
    rng = np.random.default_rng(seed)

    frames = []
    for day in days:
        features = features_for_day(day)
        features["label"] = features["address"].isin(attackers_by_day.get(day, set())).astype(
            "int8"
        )
        kept = len(features)
        if negative_rate < 1.0 and day not in scored:
            negatives = features["label"] == 0
            sampled = negatives & (rng.random(len(features)) < negative_rate)
            features = features[~negatives | sampled]
        frames.append(features)
        print(f"  {day}: {len(features):,} of {kept:,} addresses kept, "
              f"{int(features['label'].sum()):,} positive")
    return pd.concat(frames, ignore_index=True)


def split_on_days(data: pd.DataFrame, test_days: list[str]) -> Split:
    """Train on everything outside the named days and test on them.

    Used to measure decay: train on one period, score a much later one. Addresses seen in
    training are still dropped, so the question stays "does this generalise" rather than
    "does it remember".
    """
    test = data["day"].isin(test_days).to_numpy()
    split = Split(f"train elsewhere, test on {', '.join(test_days)}", ~test, test)
    return drop_overlapping_groups(split, data["address"].to_numpy())


def make_split(data: pd.DataFrame, seed: int) -> Split:
    """Train on earlier days, test on the last one, and never on the same address twice.

    Addresses active on both sides are dropped from the test set. Without that the model
    is rewarded for recognising an address it has already seen rather than for
    recognising the behaviour, which is the whole question.
    """
    days = sorted(data["day"].unique())
    if len(days) == 1:
        return random_split(len(data), test_fraction=0.3, seed=seed)
    order = {day: index for index, day in enumerate(days)}
    split = temporal_split(data["day"].map(order).to_numpy(), train_until=len(days) - 2)
    return drop_overlapping_groups(split, data["address"].to_numpy())


def fit_and_score(
    data: pd.DataFrame, split: Split, features: list[str], seed: int
) -> tuple[np.ndarray, LGBMClassifier, dict[str, float]]:
    X = data[features].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=int)

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )
    # Calibrate on a slice of training data the model never saw, rather than on
    # cross-validation folds. It costs one extra fit instead of three copies of the data,
    # and the held-out slice is the latest part of the training period, which is closer to
    # the conditions the scores will meet.
    train_rows = np.flatnonzero(split.train)
    cut = int(len(train_rows) * 0.8)
    fitting, calibrating = train_rows[:cut], train_rows[cut:]

    if len(np.unique(y[fitting])) < 2 or len(np.unique(y[calibrating])) < 2:
        raise ValueError(
            "the training period must contain both fraud and ordinary addresses "
            f"(fitting on {len(fitting):,} rows, calibrating on {len(calibrating):,}); "
            "widen the window or check that the rule hits cover these days"
        )

    model.fit(X[fitting], y[fitting])
    # FrozenEstimator keeps the fitted model as it is while the calibrator learns from the
    # held-out slice; scikit-learn dropped the old cv="prefit" spelling for this.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X[calibrating], y[calibrating])
    probabilities = calibrated.predict_proba(X[split.test])[:, 1]
    scores = evaluate(y[split.test], probabilities)
    # PR-AUC moves with how common fraud is, and these windows differ several-fold, so
    # comparing them without dividing by the base rate compares the windows, not the models.
    base_rate = float(y[split.test].mean())
    scores["base_rate"] = base_rate
    scores["pr_auc_lift"] = scores["pr_auc"] / base_rate if base_rate else float("nan")
    return probabilities, model, scores


def top_features(model: LGBMClassifier, features: list[str], count: int = 8) -> pd.DataFrame:
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    share = importance / importance.sum()
    return pd.DataFrame({
        "feature": importance.head(count).index,
        "importance share": share.head(count).to_numpy(),
    })


def explain(model: LGBMClassifier, row: pd.Series, features: list[str], count: int = 3) -> str:
    """The features that pushed one address's score up, straight from the trees."""
    contributions = model.predict(
        row[features].to_numpy(dtype=np.float32).reshape(1, -1), pred_contrib=True
    )[0][:-1]
    ranked = pd.Series(contributions, index=features).sort_values(ascending=False).head(count)
    return "; ".join(f"{name}={row[name]:,.2f}" for name in ranked.index)


def external_check(data: pd.DataFrame, scores: np.ndarray, split: Split) -> pd.DataFrame:
    """How the few externally reported addresses scored, if any are in the test set."""
    labels = load_labels()
    reported = set(labels.loc[labels["chain"] == "ethereum", "address"])
    tested = data[split.test].reset_index(drop=True)
    found = tested.index[tested["address"].isin(reported)]
    if len(found) == 0:
        return pd.DataFrame(columns=["address", "score", "percentile"])
    order = pd.Series(scores).rank(pct=True)
    return pd.DataFrame({
        "address": tested.loc[found, "address"].to_numpy(),
        "score": scores[found],
        "percentile": order[found].to_numpy(),
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", nargs="*", help="days to use (default: all extracted)")
    parser.add_argument("--test-days", nargs="*", help="days to score, for measuring decay")
    parser.add_argument("--name", default="risk_model", help="report name")
    parser.add_argument(
        "--negative-rate", type=float, default=1.0,
        help="share of innocent addresses to keep on training days (test days keep all)",
    )
    args = parser.parse_args(argv)

    days = args.days or extracted_days()
    scored = args.test_days or days[-1:]
    # The calibration slice is the latest training day, and it has to carry the real
    # proportion of fraud: calibrating on thinned data would inflate every probability.
    training_days = [d for d in sorted(days) if d not in set(scored)]
    untouched = scored + training_days[-1:]
    data = load_dataset(days, args.negative_rate, scored_days=untouched, seed=args.seed)
    split = (
        split_on_days(data, args.test_days) if args.test_days else make_split(data, args.seed)
    )
    numeric = [c for c in data.columns if c not in IDENTIFIERS + ["label"]]

    runs, models, scored = [], {}, {}
    for name, features in [
        ("every feature", numeric),
        ("without the rule's own signals", [c for c in numeric if c not in RULE_SIGNALS]),
    ]:
        probabilities, model, metrics = fit_and_score(data, split, features, args.seed)
        runs.append({"features": name, "count": len(features), **metrics})
        models[name] = (model, features)
        scored[name] = probabilities
        print(f"  {name:32} PR-AUC={metrics['pr_auc']:.3f} "
              f"recall@1%FPR={metrics['recall_at_1pct_fpr']:.3f}")

    honest = "without the rule's own signals"
    model, features = models[honest]
    tested = data[split.test].reset_index(drop=True)
    ranked = tested.assign(score=scored[honest]).nlargest(5, "score")
    examples = pd.DataFrame({
        "address": ranked["address"].to_numpy(),
        "score": ranked["score"].to_numpy(),
        "confirmed by rules": ranked["label"].map({1: "yes", 0: "no"}).to_numpy(),
        "what drove it": [explain(model, row, features) for _, row in ranked.iterrows()],
    })

    report = f"""# Risk model

Generated by `python -m txrisk.experiments.risk_model --seed {args.seed}` over
{len(days)} day(s), {len(data):,} address-days, {int(data['label'].sum()):,} of them
confirmed by the rules ({data['label'].mean():.3%}).

{"Trained on earlier days and tested on the last one." if len(days) > 1 else
 "**One day only**, so the split is random rather than by time. Treat these numbers as a "
 "smoke test: a model tested on the same day it trained on has seen the same campaigns."}

## Scores

{markdown_table(pd.DataFrame(runs))}

The second row is the one to trust: it cannot see whether an address sent or received
transfers worth nothing, which is how the rule defines a hit, so it has to recognise the
behaviour instead. Where the two rows agree, the rule's own definition was adding nothing
the rest of the behaviour did not already show.

Read `pr_auc_lift` rather than `pr_auc` when comparing windows. Fraud is several times
more common in one of these periods than the other, which moves PR-AUC on its own; lift
divides that out, and 1.0 means no better than guessing.

## Highest-scoring addresses in the test set

{markdown_table(examples)}

## What drove the model

{markdown_table(top_features(*models[honest]))}

## Reported addresses in the test set

{markdown_table(external_check(data, scored[honest], split))}

## Limits

- Positives come from one rule, so this is a poisoning detector, not a fraud detector.
  Other fraud types are absent from the labels rather than absent from the chain.
- Unlabelled is not innocent. Anything the rules did not catch counts as negative here,
  so precision is a lower bound and the model is punished for finding real fraud nobody
  has confirmed.
- Scores are calibrated against that same noisy definition of fraud.
"""
    print(f"\nReport written to {save_report(args.name, report, {'runs': runs})}")


if __name__ == "__main__":
    main()
