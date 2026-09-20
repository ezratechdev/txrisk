"""Train/test splits.

Split by time by default. A random split lets a model train on the same weeks, and the
same actors, that it is tested on, which inflates every metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    """Boolean mask over rows."""
    test: np.ndarray


def temporal_split(time_step: np.ndarray, train_until: int) -> Split:
    """Train on time steps <= train_until and test on everything later."""
    t = np.asarray(time_step)
    name = f"temporal (train t<={train_until}, test t>{train_until})"
    return Split(name, t <= train_until, t > train_until)


def random_split(n_rows: int, test_fraction: float, seed: int = 0) -> Split:
    """Uniform random split. Kept only to show how much it overstates performance."""
    test = np.zeros(n_rows, dtype=bool)
    rng = np.random.default_rng(seed)
    test[rng.choice(n_rows, size=round(n_rows * test_fraction), replace=False)] = True
    return Split(f"random ({test_fraction:.0%} test)", ~test, test)
