"""Schedule and drift-threshold retrain trigger."""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """Fires when the scheduled interval has elapsed or drift exceeds a threshold.

    :param drift_threshold: ``should_retrain`` returns True when the latest recorded
        drift score for the domain exceeds this value.
    :param schedule_hours: Minimum interval between retrains.  A retrain fires if no
        retrain has occurred within this many hours regardless of drift score.
    """

    def __init__(
        self,
        schedule_hours: float = 24.0,
        drift_threshold: float = 0.4,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.schedule_hours = schedule_hours
        self._last_retrain: dict[str, datetime] = {}
        self._latest_drift: dict[str, float] = {}

    def record_drift(self, domain: str, drift_score: float) -> None:
        """Record the most recent drift score for *domain*."""
        self._latest_drift[domain] = drift_score

    def record_retrain(self, domain: str) -> None:
        """Mark that a retrain just completed for *domain*."""
        self._last_retrain[domain] = datetime.now(timezone.utc)

    def should_retrain(self, domain: str) -> bool:
        """Return True if a retrain is warranted for *domain*.

        Fires when:
        * No retrain has been recorded (first run), or
        * The scheduled interval has elapsed, or
        * The latest drift score exceeds ``drift_threshold``.
        """
        now = datetime.now(timezone.utc)
        last = self._last_retrain.get(domain)

        # Schedule-based: never trained or interval elapsed
        if last is None or (now - last) >= timedelta(hours=self.schedule_hours):
            return True

        # Drift-threshold based
        score = self._latest_drift.get(domain, 0.0)
        return score > self.drift_threshold
