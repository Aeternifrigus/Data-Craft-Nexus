"""Messy data, made on purpose.

PMLB's datasets are clean: 38 of the first 40 had no missing values and none
had text in a numeric column. The files people upload are not, and the quality
axis (A62 missing, A64 noisy) had no evidence behind it. This damages clean
datasets in known ways and at known rates, so the benchmark can say what each
kind of damage does to each model, and whether the profiler notices it.

  missing_10   10% of feature cells blanked, completely at random
  missing_30   30% of feature cells blanked
  dirty_5      5% of the cells in numeric columns replaced by the junk that
               spreadsheets and exports produce ("?", "n/a", "#VALUE!" ...)
  labels_10    10% of the training labels replaced by another row's label.
               Only while fitting: every model is still scored on the true
               labels, which is what label noise costs a user in practice

Everything is seeded, so a condition damages a dataset the same way every
time and for every model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

SEED = 0
JUNK = ["?", "n/a", "-", "unknown", "#VALUE!", "NA ", "missing"]

CONDITIONS = {
    "missing_10": "10% of feature cells missing, at random",
    "missing_30": "30% of feature cells missing, at random",
    "dirty_5": "5% of numeric cells replaced by text junk",
    "labels_10": "10% of training labels replaced by another row's",
}


def damage(frame: pd.DataFrame, condition: str, numeric: list[str], seed: int = SEED) -> pd.DataFrame:
    """The dataset with its features damaged. Label noise happens at fit time instead (NoisyLabels)."""
    if condition in ("clean", "labels_10"):
        return frame
    rng = np.random.default_rng(seed)
    out = frame.copy()
    features = [c for c in out.columns if c != "target"]
    if condition.startswith("missing_"):
        rate = int(condition.split("_")[1]) / 100
        for col in features:
            mask = rng.random(len(out)) < rate
            out[col] = out[col].astype(object)
            out.loc[mask, col] = np.nan
        return out
    if condition == "dirty_5":
        for col in [c for c in features if c in numeric]:
            mask = rng.random(len(out)) < 0.05
            out[col] = out[col].astype(object)
            out.loc[mask, col] = rng.choice(JUNK, size=int(mask.sum()))
        return out
    raise ValueError(f"unknown condition {condition}")


def coerce(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """What any pipeline has to do with messy columns before a model sees them.

    A numeric column's unparseable cells become missing, and a categorical
    column is made uniformly text, keeping its blanks, so the encoder does not
    trip over a mix of numbers and strings.
    """
    X = X.copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical:
        X[col] = X[col].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return X


class NoisyLabels(BaseEstimator):
    """Fit the wrapped estimator on labels with a fraction replaced, predict as it does.

    Cross-validation calls fit on the training folds and scores predictions
    against the untouched test labels, so the noise is exactly where a real
    labelling mistake would be: in what the model learns from.
    """

    def __init__(self, estimator=None, rate: float = 0.1, seed: int = SEED):
        self.estimator = estimator
        self.rate = rate
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y).copy()
        rng = np.random.default_rng(self.seed + len(y))
        chosen = rng.random(len(y)) < self.rate
        y[chosen] = y[rng.integers(0, len(y), size=int(chosen.sum()))]
        self.estimator_ = clone(self.estimator).fit(X, y)
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)
