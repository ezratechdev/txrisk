"""Baseline classifiers. Anything more complex (graph models, fraud-type heads) must beat these."""

from __future__ import annotations

from collections.abc import Callable

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def baseline_models(seed: int = 0) -> dict[str, Callable[[], ClassifierMixin]]:
    """Factories, so every (model, feature set) run starts from an untrained estimator."""
    return {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        # Same settings as Weber et al. (2019): 50 trees, 50 candidate features per split.
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=50, max_features=50, n_jobs=-1, random_state=seed
        ),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            random_state=seed,
            verbose=-1,
        ),
    }
