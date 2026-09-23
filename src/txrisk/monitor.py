"""Watching a model age.

Trained on 2023 and pointed at 2026, the detector still ranked attackers above everyone
else - precision stayed at 0.94 - while its calibration error went from 0.007 to 0.028.
The ordering survived three and a half years; the probabilities did not.

So the thing to watch is calibration, not ranking. Waiting for ranking to collapse means
waiting years while every probability the model reports is quietly wrong, and a score of
0.9 that means 0.6 is worse than no score at all. Recalibrating on recent days is cheap;
retraining from scratch is not.

    python -m txrisk.monitor --days 2026-09-05 2026-09-06 2026-09-07
"""

from __future__ import annotations

import argparse

import pandas as pd

from txrisk.evaluation.metrics import evaluate
from txrisk.paths import PROCESSED_DIR
from txrisk.report import markdown_table, save_report
from txrisk.serve import address_scores, load_scorer

RECALIBRATE_ABOVE = 0.02
"""Calibration error at which the probabilities stop deserving to be believed."""


def check_day(day: str, scorer) -> dict[str, float]:
    """Score a day and compare the scores with what the rules found there."""
    scored = address_scores(day, scorer)
    hits = pd.read_parquet(PROCESSED_DIR / "rule_hits.parquet")
    attackers = set(
        hits[(hits["day"] == day) & (hits["role"] == "attacker")
             & (hits["confidence"] == "confirmed")]["address"]
    )
    truth = scored["address"].isin(attackers).to_numpy().astype(int)
    if truth.sum() == 0:
        return {"day": day, "addresses": len(scored), "note": "no confirmed fraud that day"}

    measured = evaluate(truth, scored["risk"].to_numpy())
    return {
        "day": day,
        "addresses": len(scored),
        "fraud_rate": float(truth.mean()),
        "mean_score": float(scored["risk"].mean()),
        "pr_auc": measured["pr_auc"],
        "ece": measured["ece"],
    }


def verdict(checks: pd.DataFrame) -> str:
    if "ece" not in checks:
        return "No day had confirmed fraud to check against, so nothing can be said yet."
    worst = checks["ece"].max()
    if worst > RECALIBRATE_ABOVE:
        return (
            f"**Recalibrate.** Calibration error reached {worst:.3f}, above the "
            f"{RECALIBRATE_ABOVE:.2f} the probabilities are trusted within. Ranking may "
            "still be fine, which is exactly why this is easy to miss."
        )
    return f"Calibration error stayed at {worst:.3f}; the probabilities can still be believed."


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", nargs="+", required=True)
    args = parser.parse_args(argv)

    scorer = load_scorer()
    checks = pd.DataFrame([check_day(day, scorer) for day in args.days])
    print(checks.to_string(index=False))

    summary = verdict(checks)
    report = f"""# Drift watch

Days checked: {', '.join(args.days)}.

{markdown_table(checks)}

{summary}

## Why calibration rather than ranking

A model trained in 2023 and shown September 2026 still put real attackers at the top, at
0.94 precision, while its calibration error grew fourfold. Ranking is the last thing to
fail and the first thing people check, which is how a model spends months reporting
probabilities that no longer mean anything.

`fraud_rate` and `mean_score` are here for the same reason: if the rules stop firing, the
labels this check relies on quietly disappear, and a clean report would mean nothing.
"""
    print(f"\n{summary}")
    print(f"Report written to {save_report('drift_watch', report, checks.to_dict('records'))}")


if __name__ == "__main__":
    main()
