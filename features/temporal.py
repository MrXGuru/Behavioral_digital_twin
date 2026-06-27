"""Temporal feature helpers for the feature pipeline.

Provides cyclical hour encoding, day-of-week one-hot, time-of-day one-hot, and an
incremental rolling-frequency tracker. The rolling tracker is O(1) per record and, by
construction, only ever reflects decisions strictly earlier than the current record
(the caller updates it *after* reading features), guaranteeing no future leakage.
"""

from __future__ import annotations

import math
from datetime import datetime

from data.schema import Domain, TimeOfDay, options

#: Stable ordering for day-of-week one-hot (Mon..Sun -> 0..6).
_DOW = list(range(7))

#: Stable ordering for time-of-day one-hot.
_TOD_ORDER: tuple[str, ...] = tuple(t.value for t in TimeOfDay)


def hour_cyclical(ts: datetime) -> list[float]:
    """Return ``[sin, cos]`` cyclical encoding of the timestamp hour."""
    angle = 2.0 * math.pi * (ts.hour / 24.0)
    return [math.sin(angle), math.cos(angle)]


def dow_onehot(ts: datetime) -> list[float]:
    """Return a 7-dim one-hot of the timestamp weekday (Mon..Sun)."""
    wd = ts.weekday()
    return [1.0 if wd == d else 0.0 for d in _DOW]


def tod_onehot(time_of_day_value: str) -> list[float]:
    """Return a one-hot of the time-of-day bucket in stable order."""
    return [1.0 if time_of_day_value == t else 0.0 for t in _TOD_ORDER]


class RollingFrequencyTracker:
    """Track per-option decision frequencies for one domain incrementally.

    ``features()`` returns the normalized distribution of decisions observed *so far*
    (strictly before the current record, provided the caller updates after reading).
    Before any observation it returns a uniform distribution.
    """

    def __init__(self, domain: str | Domain) -> None:
        self.options = list(options(domain))
        self._counts = {opt: 0 for opt in self.options}
        self._total = 0

    def features(self) -> list[float]:
        """Return the normalized frequency vector over the domain's options."""
        if self._total == 0:
            n = len(self.options)
            return [1.0 / n] * n
        return [self._counts[opt] / self._total for opt in self.options]

    def update(self, decision: str) -> None:
        """Record one observed decision."""
        if decision in self._counts:
            self._counts[decision] += 1
            self._total += 1

    @property
    def width(self) -> int:
        return len(self.options)


def temporal_width() -> int:
    """Total width contributed by the standalone temporal encoders (excl. rolling)."""
    return 2 + len(_DOW) + len(_TOD_ORDER)
