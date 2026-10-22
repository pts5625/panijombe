"""The waterlogging model: a blend of two learners.

1. Gradient boosting with monotonic physical constraints (captures thresholds
   and interactions, e.g. a 3-hour burst on already-wet ground).
2. Logistic regression (smooth, hard to overfit with only ~30 events).

Each learner's output is converted to a percentile of its own training
scores, and the two percentiles are averaged. This rank blend is robust when
the two models produce probabilities on different scales.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import MONOTONIC

HGB_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=150,
                  l2_regularization=5.0, random_state=7)
GRID = np.linspace(0, 1, 1001)


class BlendModel:
    def __init__(self, features):
        self.features = list(features)
        self.hgb = HistGradientBoostingClassifier(
            **HGB_PARAMS, monotonic_cst=[MONOTONIC[f] for f in self.features])
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))

    def fit(self, X, y, sample_weight):
        X = X[self.features]
        self.hgb.fit(X, y, sample_weight=sample_weight)
        self.lr.fit(X, y, logisticregression__sample_weight=sample_weight)
        self.q_hgb = np.quantile(self.hgb.predict_proba(X)[:, 1], GRID)
        self.q_lr = np.quantile(self.lr.predict_proba(X)[:, 1], GRID)
        return self

    def score(self, X):
        X = X[self.features]
        a = np.interp(self.hgb.predict_proba(X)[:, 1], self.q_hgb, GRID)
        b = np.interp(self.lr.predict_proba(X)[:, 1], self.q_lr, GRID)
        return 0.5 * (a + b)

    def predict_proba(self, X):
        s = self.score(X)
        return np.column_stack([1 - s, s])
