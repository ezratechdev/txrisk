"""Phase 1b: can a model name the *type* of fraud, and admit when it can't?

Two stages on the BitcoinHeist dataset, both trained on 2009-2016 and tested on 2017-2018:

* detection  - ransomware payment address or not (binary, ~0.5% positives in the test period);
* typology   - which ransomware family, with "unknown" allowed as an answer.

The test period contains families that do not occur in training at all (WannaCry appeared
in May 2017), which is the situation that matters in production: a fraud type nobody has
labelled yet. Guessing a trained family for those cases is a failure, answering "unknown"
is success.

    python -m txrisk.experiments.bitcoinheist_typology
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from txrisk.data.bitcoinheist import FEATURES, load_bitcoinheist
from txrisk.evaluation.metrics import (
    UNKNOWN,
    evaluate,
    evaluate_open_set,
    per_class_scores,
    predict_with_rejection,
    threshold_for_best_f1,
    threshold_for_coverage,
)
from txrisk.evaluation.splits import (
    Split,
    drop_overlapping_groups,
    random_split,
    temporal_split,
)
from txrisk.models.baselines import baseline_models
from txrisk.report import markdown_table, save_report

TRAIN_UNTIL = 2016
MIN_TRAIN_ROWS = 50
"""A family needs this many training rows to become a class of its own. Rarer families are
left out of training, so they arrive at test time as fraud types the model has never seen."""
TARGET_COVERAGE = 0.9
"""Answer (rather than say "unknown") for 90% of validation cases; sets the confidence cut-off."""


def split_by_address(frame: pd.DataFrame, train_until: int) -> Split:
    """Split by year, then drop test rows for addresses already seen in training."""
    split = temporal_split(frame["year"].to_numpy(), train_until)
    return drop_overlapping_groups(split, frame["address"].to_numpy())


def holdout_addresses(
    address: np.ndarray, train: np.ndarray, seed: int, fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Carve whole addresses out of the training rows, for choosing thresholds.

    Thresholds are part of the model, so they have to be set on data the final scoring
    never sees. Holding out whole addresses stops the same actor deciding its own cut-off.
    """
    rng = np.random.default_rng(seed)
    train_addresses = pd.unique(address[train])
    held_out = rng.choice(train_addresses, size=max(1, len(train_addresses) // 5), replace=False)
    in_validation = pd.Series(address).isin(set(held_out)).to_numpy()
    return train & ~in_validation, train & in_validation


def run_detection(frame: pd.DataFrame, split: Split, seed: int) -> pd.DataFrame:
    """Stage A: ransomware payment address or not, with the threshold learned, not assumed."""
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    y = frame["is_ransomware"].to_numpy(dtype=int)
    fit_rows, validation_rows = holdout_addresses(frame["address"].to_numpy(), split.train, seed)
    rows = []
    for name, make_model in baseline_models(seed).items():
        start = time.perf_counter()
        model = make_model().fit(X[fit_rows], y[fit_rows])
        threshold = threshold_for_best_f1(
            y[validation_rows], model.predict_proba(X[validation_rows])[:, 1]
        )
        prob = model.predict_proba(X[split.test])[:, 1]
        scores = evaluate(y[split.test], prob, threshold=threshold)
        # PR-AUC depends on how common fraud is in the test set, so the two splits cannot
        # be compared without it. "lift" is PR-AUC over the base rate: 1.0 is random.
        base_rate = float(y[split.test].mean())
        rows.append({"model": name, "threshold": threshold, **scores,
                     "base_rate": base_rate, "pr_auc_lift": scores["pr_auc"] / base_rate,
                     "n_test": int(split.test.sum()),
                     "fit_seconds": time.perf_counter() - start})
        print(f"  detection/{split.name.split()[0]:9} {name:20} "
              f"PR-AUC={rows[-1]['pr_auc']:.3f} F1={rows[-1]['f1']:.3f}")
    return pd.DataFrame(rows)


def run_typology(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict, list[str]]:
    """Stage B: which ransomware family, with "unknown" allowed."""
    ransomware = frame[frame["is_ransomware"]].reset_index(drop=True)
    split = split_by_address(ransomware, TRAIN_UNTIL)

    train = ransomware[split.train]
    counts = train["family"].value_counts()
    known_classes = sorted(counts[counts >= MIN_TRAIN_ROWS].index)
    trainable = split.train & ransomware["family"].isin(known_classes).to_numpy()

    fit_mask, validation_mask = holdout_addresses(
        ransomware["address"].to_numpy(), trainable, seed
    )

    X = ransomware[FEATURES].to_numpy(dtype=np.float32)
    family = ransomware["family"].to_numpy()
    y_test = np.where(np.isin(family[split.test], known_classes), family[split.test], UNKNOWN)

    rows, answers = [], {}
    for name, make_model in baseline_models(seed).items():
        start = time.perf_counter()
        model = make_model().fit(X[fit_mask], family[fit_mask])
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        threshold = threshold_for_coverage(
            model.predict_proba(X[validation_mask]).max(axis=1), TARGET_COVERAGE
        )
        predicted = predict_with_rejection(model.predict_proba(X[split.test]), classes, threshold)
        rows.append({"model": name, "confidence_cutoff": threshold,
                     **evaluate_open_set(y_test, predicted, known_classes),
                     "fit_seconds": time.perf_counter() - start})
        answers[name] = (predicted, classes)
        print(f"  typology  {name:20} macro-F1={rows[-1]['known_macro_f1']:.3f} "
              f"unseen rejected={rows[-1]['unseen_rejected']:.3f}")

    results = pd.DataFrame(rows)
    best = results.loc[results["known_macro_f1"].idxmax(), "model"]
    predicted, _ = answers[best]
    per_class = per_class_scores(y_test, predicted, known_classes)
    detail = {
        "best_model": best,
        "per_class": per_class[per_class["support"] > 0].reset_index(drop=True),
        "absent_classes": per_class.loc[per_class["support"] == 0, "class"].tolist(),
        "unseen": _unseen_answers(family[split.test], y_test, predicted),
        "n_train_rows": int(fit_mask.sum()),
        "n_test_rows": int(split.test.sum()),
    }
    return results, detail, known_classes


def _unseen_answers(family: np.ndarray, y_test: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """For each never-trained family, how often the model abstained and what it said instead."""
    unseen = y_test == UNKNOWN
    rows = []
    for name in sorted(set(family[unseen])):
        answers = pd.Series(predicted[unseen & (family == name)])
        wrong = answers[answers != UNKNOWN]
        rows.append({
            "unseen family": name,
            "rows": len(answers),
            "answered unknown": float((answers == UNKNOWN).mean()),
            "most common wrong answer": wrong.mode().iat[0] if not wrong.empty else "–",
        })
    return pd.DataFrame(rows)


def build_report(frame: pd.DataFrame, detection: pd.DataFrame, detection_random: pd.DataFrame,
                 typology: pd.DataFrame, detail: dict, known_classes: list[str],
                 seed: int) -> str:
    n_ransomware = int(frame["is_ransomware"].sum())
    absent = detail["absent_classes"]
    return f"""# BitcoinHeist: detection and fraud type

Generated by `python -m txrisk.experiments.bitcoinheist_typology --seed {seed}`.

Data: {len(frame):,} address-days, of which {n_ransomware:,} ({n_ransomware / len(frame):.2%})
are ransomware payments covering {frame["family"].nunique() - 1} families, after normalising
the three labelling sources. Trained on {TRAIN_UNTIL} and earlier, tested on {TRAIN_UNTIL + 1}
onwards, with test rows for addresses seen in training removed.

## Stage A: is this a ransomware payment address?

Each model's decision threshold is chosen on held-out training addresses, because at this
prevalence a 0.5 cut-off flags almost nothing.

{markdown_table(detection)}

The same models under a random split, which is how this dataset is usually scored:

{markdown_table(detection_random)}

Read those two tables through `pr_auc_lift`, not `pr_auc`. The random split's test set has a
higher share of ransomware in it, which raises PR-AUC on its own; lift divides that out. Once
it is divided out the two splits rank the models the same way, so here — unlike the Elliptic
dataset, where a random split inflated every metric — the split is not what makes detection
hard. These six address-level features simply carry weak signal for it.

## Stage B: which ransomware family?

{len(known_classes)} families had at least {MIN_TRAIN_ROWS} training rows and became classes:
{", ".join(known_classes)}. Every other family is treated as a fraud type the model has
never seen. The confidence cut-off is set on held-out training addresses to answer
{TARGET_COVERAGE:.0%} of them, never on the test data.

- `known_macro_f1` averages F1 over the trained families, so rare ones still count.
- `unseen_rejected` is the share of never-trained families correctly answered "unknown".
- `known_answered` is how often the model committed to a family when it did know it.

{markdown_table(typology)}

### Per-family results ({detail["best_model"]}, {detail["n_test_rows"]:,} test rows)

{markdown_table(detail["per_class"])}

{len(absent)} trained families have no test rows at all and are left out of the average
above: {", ".join(absent) if absent else "none"}. Those campaigns had stopped by the test
period, which is itself the point: fraud types come and go.

### Fraud types the model was never trained on

{markdown_table(detail["unseen"])}

## What this means for the design

- **Detection needs more than these features.** At the learned threshold precision is a few
  percent. That supports the planned order on real data: known-bad list lookups and
  deterministic rules first, model second, rather than a model alone.
- **Naming the type is the easier half, when the type is trained and still active.** The
  dominant family in the test period is recognised with high precision.
- **Everything else is the hard part.** Families that had stopped by the test period, and
  families that never appear in training, are where the model breaks down.
- **Rejection has to be tuned deliberately.** At a {TARGET_COVERAGE:.0%} answer rate the best
  model still names a trained family for most fraud types it has never seen. A higher
  cut-off would reject more of them, at the cost of answering fewer known ones.

## Caveats

- "white" means "never reported as ransomware", not "verified legitimate", so precision
  here is a lower bound: some flagged addresses may be genuine ransomware nobody reported.
- Labels come from reports, so the data is biased towards families researchers tracked.
- Ransomware labels thin out after 2017, and 2018 has almost none. The test period is
  dominated by one family (Cerber), which is why macro-F1 is reported next to accuracy.
- Features are the dataset's own address-level graph statistics, not raw transactions.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    frame = load_bitcoinheist()
    strict = split_by_address(frame, TRAIN_UNTIL)
    detection = run_detection(frame, strict, args.seed)
    detection_random = run_detection(
        frame, random_split(len(frame), float(strict.test.mean()), seed=args.seed), args.seed
    )
    typology, detail, known_classes = run_typology(frame, args.seed)
    markdown = build_report(
        frame, detection, detection_random, typology, detail, known_classes, args.seed
    )
    raw = {
        "detection": detection.to_dict("records"),
        "detection_random_split": detection_random.to_dict("records"),
        "typology": typology.to_dict("records"),
        "per_class": detail["per_class"].to_dict("records"),
        "unseen": detail["unseen"].to_dict("records"),
        "known_classes": known_classes,
    }
    print(f"Report written to {save_report('bitcoinheist_typology', markdown, raw)}")


if __name__ == "__main__":
    main()
