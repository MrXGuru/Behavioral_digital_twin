"""Concept-drift detection from rolling prediction accuracy.

The :class:`DriftDetector` keeps a rolling window of recent *labeled* predictions and flags
drift when, with at least ``window`` observations, the rolling accuracy falls below a
configured threshold.

A "labeled" prediction is one whose ``actual`` outcome is known (not ``None``). Only labeled
predictions occupy the rolling window, so ``status()`` always reflects the most recent
``window`` labeled predictions regardless of how many unlabeled predictions were recorded in
between.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class DriftStatus:
    """Result of a drift evaluation over the rolling window.

    Attributes:
        drift: True iff at least ``window`` labeled predictions exist and the rolling
            accuracy is below the configured threshold.
        score: Drift magnitude, defined as ``1.0 - window_acc`` (0.0 when no observations).
            Higher means more divergence between predictions and observed behavior.
        window_acc: Proportion of correct predictions over the most recent ``window``
            labeled predictions, or ``None`` when no labeled predictions have been observed.
    """

    drift: bool
    score: float
    window_acc: float | None

    def as_dict(self) -> dict:
        return {
            "drift": self.drift,
            "score": round(self.score, 4),
            "window_acc": (
                round(self.window_acc, 4) if self.window_acc is not None else None
            ),
        }


class DriftDetector:
    """Rolling-accuracy concept-drift detector."""

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.window = window
        self.threshold = threshold
        # Rolling window over labeled predictions only; oldest entries drop off as new
        # labeled predictions arrive. Each entry is (predicted, actual).
        self._recent_labeled: deque[tuple[str, str]] = deque(maxlen=window)

    def record(self, predicted: str, actual: str | None, confidence: float) -> None:
        """Update the rolling record of recent labeled predictions.

        Only labeled predictions (``actual is not None``) are tracked in the rolling
        window. Unlabeled predictions are ignored so the window always holds the most
        recent ``window`` labeled observations. The ``confidence`` argument is part of the
        interface but does not affect the accuracy-based drift signal.
        """
        if actual is not None:
            self._recent_labeled.append((predicted, actual))

    def status(self) -> DriftStatus:
        """Return current drift status from the rolling window of labeled predictions."""
        observed = len(self._recent_labeled)
        if observed == 0:
            return DriftStatus(drift=False, score=0.0, window_acc=None)
        correct = sum(1 for predicted, actual in self._recent_labeled if predicted == actual)
        window_acc = correct / observed
        drift = observed >= self.window and window_acc < self.threshold
        return DriftStatus(drift=drift, score=1.0 - window_acc, window_acc=window_acc)
