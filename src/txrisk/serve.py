"""Scoring a transaction, which is what all of this was for.

A transaction is scored from the two accounts it involves, because that is where the
evidence lives. Sending to an address that drained wallets yesterday is the risk; the
transfer itself looks like any other.

Every score carries its reasons. A number nobody can question is a number nobody should
act on, and these scores can be wrong in ways only a person looking at the evidence will
notice. The first transaction this ranked highest scored exactly 1.0 and turned out to be
an ordinary address with four incoming transfers; the reasons field said, correctly, that
no rule had fired. `tied_at_top` says how many share that top score, because a calibrated
model that saturates cannot tell them apart and should not pretend otherwise.

    python -m txrisk.serve train --days 2023-02-24 ... 2023-03-01
    python -m txrisk.serve score --day 2023-03-03 --top 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from txrisk.data.ethereum import load_day
from txrisk.data.labels import load_labels
from txrisk.experiments.risk_model import IDENTIFIERS, RULE_SIGNALS, load_dataset
from txrisk.features.address_day import features_for_day
from txrisk.features.graph import GRAPH_COLUMNS, graph_features_for_day, known_bad_before
from txrisk.paths import PROCESSED_DIR

MODEL_DIR = PROCESSED_DIR / "model"
BANDS = [(0.8, "high"), (0.4, "medium"), (0.0, "low")]


@dataclass
class Scorer:
    """A trained risk model plus the feature names it expects, in that order."""

    model: object
    features: list[str]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [name for name in self.features if name not in frame]
        if missing:
            raise ValueError(f"these features are missing from the data: {missing}")
        return self.model.predict_proba(frame[self.features].to_numpy(dtype=np.float32))[:, 1]


def band(score: float) -> str:
    return next(name for floor, name in BANDS if score >= floor)


def train(days: list[str], seed: int = 0, negative_rate: float = 0.4) -> Scorer:
    """Fit on the given days and keep the result, so scoring never retrains."""
    data = load_dataset(days, negative_rate, scored_days=days[-1:], seed=seed)
    features = [c for c in data.columns if c not in IDENTIFIERS + ["label", *RULE_SIGNALS]]
    X = data[features].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=int)

    cut = int(len(data) * 0.8)
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    ).fit(X[:cut], y[:cut])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X[cut:], y[cut:])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(calibrated, features)
    # Store the pieces, not the wrapper. Pickling the dataclass records where it was
    # defined, and training runs as __main__, so anything else loading it later fails.
    joblib.dump({"model": calibrated, "features": features}, MODEL_DIR / "risk.joblib")
    (MODEL_DIR / "trained_on.json").write_text(
        json.dumps({"days": days, "features": features, "rows": len(data)}, indent=2),
        encoding="utf-8",
    )
    return scorer


def load_scorer() -> Scorer:
    path = MODEL_DIR / "risk.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m txrisk.serve train --days ...")
    saved = joblib.load(path)
    return Scorer(saved["model"], saved["features"])


def address_scores(day: str, scorer: Scorer) -> pd.DataFrame:
    """Risk for every address active on a day, with the numbers behind it."""
    features = features_for_day(day)
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    reported = set(load_labels().query("chain == 'ethereum'")["address"])
    graph = graph_features_for_day(day, known_bad_before(day, hits, reported))
    features = features.merge(graph, on="address", how="left")
    features[GRAPH_COLUMNS] = features[GRAPH_COLUMNS].fillna(0.0)
    features["risk"] = scorer.score(features)
    return features


def rule_reasons(day: str) -> dict[str, list[str]]:
    """What the rules said about each address, which needs no model to explain."""
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    today = hits[(hits["day"] == day) & (hits["role"] != "victim")]
    reasons: dict[str, list[str]] = {}
    for row in today.itertuples():
        reasons.setdefault(row.address, []).append(f"{row.rule} ({row.confidence}): {row.evidence}")
    return reasons


def score_transactions(day: str, scorer: Scorer, top: int = 10) -> list[dict]:
    """The riskiest transactions of a day, each with the case against it.

    A transaction takes the higher of its two addresses' scores: one dangerous party is
    enough, and averaging would let a busy innocent counterparty hide it.
    """
    scores = address_scores(day, scorer)
    risk = dict(zip(scores["address"], scores["risk"], strict=False))
    reasons = rule_reasons(day)

    transactions = load_day(
        "transactions", day, ["hash", "from_address", "to_address", "value"]
    ).dropna(subset=["from_address"])
    sender = transactions["from_address"].map(risk).fillna(0.0)
    receiver = transactions["to_address"].map(risk).fillna(0.0)
    transactions["risk"] = np.maximum(sender, receiver)
    transactions["riskier_side"] = np.where(sender >= receiver, "sender", "receiver")

    # Isotonic calibration maps a whole range of scores to exactly 1.0, so hundreds of
    # addresses tie at the top and "the riskiest transactions" becomes row order. Evidence
    # breaks the tie: a transaction a rule can explain outranks one that only scores high.
    transactions["party"] = np.where(
        transactions["riskier_side"] == "sender",
        transactions["from_address"], transactions["to_address"],
    )
    transactions["has_evidence"] = transactions["party"].isin(reasons).astype(int)
    ranked = transactions.sort_values(["risk", "has_evidence"], ascending=False).head(top)

    tied = int((transactions["risk"] >= transactions["risk"].max() - 1e-9).sum())
    scored = []
    for row in ranked.itertuples():
        scored.append({
            "transaction": row.hash,
            "risk_score": round(float(row.risk), 4),
            "risk_band": band(float(row.risk)),
            "riskier_party": {"address": row.party, "side": row.riskier_side},
            "reasons": reasons.get(
                row.party, ["no rule fired; this is the model's judgement alone"]
            ),
            "tied_at_top": tied if row.risk >= transactions["risk"].max() - 1e-9 else None,
        })
    return scored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("train", help="fit the scorer and keep it")
    fit.add_argument("--days", nargs="+", required=True)
    fit.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("score", help="score a day's transactions")
    run.add_argument("--day", required=True)
    run.add_argument("--top", type=int, default=10)
    run.add_argument("--out", type=Path, help="write the scored transactions here as JSON")

    args = parser.parse_args(argv)
    if args.command == "train":
        scorer = train(args.days, seed=args.seed)
        print(f"trained on {len(args.days)} day(s), {len(scorer.features)} features")
        print(f"saved to {MODEL_DIR / 'risk.joblib'}")
        return

    scored = score_transactions(args.day, load_scorer(), top=args.top)
    text = json.dumps(scored, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"{len(scored)} transactions written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
