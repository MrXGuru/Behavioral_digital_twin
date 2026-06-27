"""Append-only production drift monitoring series."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class DriftPoint:
    user_id: str
    domain: str
    window_acc: float
    drift: bool
    timestamp: datetime


class DriftMonitor:
    """Append-only, timestamp-ordered drift series.

    Never mutates or drops prior points — every call to :meth:`append` adds a new
    :class:`DriftPoint` to the end of the internal list.
    """

    def __init__(self) -> None:
        self._series: list[DriftPoint] = []

    def append(
        self,
        user_id: str,
        domain: str,
        window_acc: float,
        drift: bool,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a new drift observation.  Never modifies existing entries."""
        point = DriftPoint(
            user_id=user_id,
            domain=domain,
            window_acc=window_acc,
            drift=drift,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self._series.append(point)

    def series(self, user_id: str, domain: str) -> list[dict]:
        """Return all drift points for *(user_id, domain)* in insertion order."""
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "window_acc": p.window_acc,
                "drift": p.drift,
                "domain": p.domain,
            }
            for p in self._series
            if p.user_id == user_id and p.domain == domain
        ]
