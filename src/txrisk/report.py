"""Helpers for writing experiment reports, so every experiment reports the same way."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from txrisk.paths import REPORTS_DIR


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "–" if np.isnan(value) else f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def save_report(stem: str, markdown: str, raw: dict) -> Path:
    """Write <stem>.md for people and <stem>.json for later comparison."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{stem}.md"
    path.write_text(markdown, encoding="utf-8")
    (REPORTS_DIR / f"{stem}.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path
