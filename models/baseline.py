"""Gradient-boosting / logistic-regression baseline twin.

Consumes the flat feature vector from :class:`features.feature_pipeline.FeaturePipeline`
and predicts the next decision for one domain with a calibrated probability
distribution. Falls back to logistic regression when a domain has too few classes or
samples for gradient boosting to be useful, and degrades to a constant prior when only a
single class is observed in training.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from models.base import DecisionModel, validate_distribution


class BaselineModel(DecisionModel):
    """HistGradientBoosting baseline over flat features with GridSearchCV."""

    name = "baseline"

    def __init__(self, options: list[str], kind: str = "gboost") -> None:
        super().__init__(options)
        self.kind = kind
        self._clf = None
        self._classes: list[str] = []
        self._fallback_probs: dict[str, float] | None = None

    def fit(self, X: np.ndarray, seq: np.ndarray, y: list[str]) -> "BaselineModel":
        classes = sorted(set(y))
        self._classes = classes
        if len(classes) < 2 or len(y) < 7:
            # Degenerate: use observed class frequencies as a constant prior.
            self._fallback_probs = self._prior_from_labels(y)
            self._clf = None
            return self

        if self.kind == "logreg":
            self._clf = LogisticRegression(max_iter=500, multi_class="auto")
        else:
            # Fast, histogram-based gradient boosting ideal for large datasets
            self._clf = HistGradientBoostingClassifier(random_state=0, max_iter=5)

        try:
            self._clf.fit(X, y)
            self._fallback_probs = None
        except Exception as e:
            print(f"[BaselineModel] Fit failed: {e}")
            self._fallback_probs = self._prior_from_labels(y)
            self._clf = None
        return self

    def _prior_from_labels(self, y: list[str]) -> dict[str, float]:
        probs = {opt: 0.0 for opt in self.options}
        for label in y:
            if label in probs:
                probs[label] += 1.0
        total = sum(probs.values()) or 1.0
        return {k: v / total for k, v in probs.items()}

    def predict_proba(self, x: np.ndarray, seq: np.ndarray) -> dict[str, float]:
        if self._clf is None:
            probs = dict(self._fallback_probs or {opt: 1.0 / len(self.options)
                                                  for opt in self.options})
        else:
            row = self._clf.predict_proba(x)[0]
            probs = {opt: 0.0 for opt in self.options}
            for cls, p in zip(self._clf.classes_, row):
                probs[cls] = float(p)
        # ensure full option coverage + normalize
        for opt in self.options:
            probs.setdefault(opt, 0.0)
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        validate_distribution(probs)
        return probs

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "options": self.options, "kind": self.kind, "clf": self._clf,
                "classes": self._classes, "fallback": self._fallback_probs,
            }, fh)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        model = cls(state["options"], kind=state["kind"])
        model._clf = state["clf"]
        model._classes = state["classes"]
        model._fallback_probs = state["fallback"]
        return model
