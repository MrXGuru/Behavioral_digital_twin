"""Property-based test for temporal split integrity.

Property 6: Temporal split integrity
    *For any* record set and valid ``val_fraction``, ``time_based_split(records,
    val_fraction) -> (train, val)`` yields a partition of the input where:
      * **Partition**: ``train`` and ``val`` together contain exactly the input records
        as a multiset -- none lost, none duplicated.
      * **Temporal ordering**: when both partitions are non-empty, every train timestamp
        is <= every val timestamp (i.e. ``max(train ts) <= min(val ts)``).

**Validates: Requirements 6.1, 6.2**
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data.schema import (
    DOMAIN_OPTIONS,
    DecisionRecord,
    Domain,
    day_type,
    time_of_day,
    validate,
)
from models.evaluate import time_based_split

# Categorical context vocabularies (kept small; not the focus of this property).
_LOCATIONS = ("home", "work", "gym", "store", "transit")
_WEATHER = ("clear", "cloudy", "rain", "snow")
_OUTCOMES = ("good", "neutral", "bad")

# A deliberately small pool of timestamps so that many records collide on the same
# timestamp. This exercises the train/val boundary where equal timestamps may straddle
# the split -- the case most likely to break either partition or ordering guarantees.
_TS_POOL = [
    datetime(2024, 1, 1, 8, 0, 0),  # weekday morning
    datetime(2024, 1, 1, 8, 0, 0),  # exact duplicate of the above
    datetime(2024, 1, 3, 14, 30, 0),  # weekday afternoon
    datetime(2024, 1, 6, 19, 15, 0),  # weekend evening
    datetime(2024, 1, 6, 19, 15, 0),  # exact duplicate
    datetime(2024, 1, 9, 23, 45, 0),  # weekday night
    datetime(2024, 2, 1, 5, 0, 0),  # weekday morning boundary
]


@st.composite
def decision_records(draw: st.DrawFn) -> DecisionRecord:
    """Draw a single schema-valid :class:`DecisionRecord`.

    ``day_type`` and ``time_of_day`` are derived from the timestamp (so the record always
    passes :func:`validate`), and ``decision_made`` is sampled from the drawn domain's
    option set. Timestamps are drawn from a small pool so duplicates are common.
    """
    ts = draw(st.sampled_from(_TS_POOL))
    domain = draw(st.sampled_from(list(Domain)))
    decision = draw(st.sampled_from(DOMAIN_OPTIONS[domain]))
    record = DecisionRecord(stress_level="low", 
        user_id=draw(st.sampled_from(("u1", "u2"))),
        timestamp=ts,
        domain=domain.value,
        location=draw(st.sampled_from(_LOCATIONS)),
        weather=draw(st.sampled_from(_WEATHER)),
        day_type=day_type(ts).value,
        time_of_day=time_of_day(ts).value,
        mood_energy=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        decision_made=decision,
        outcome=draw(st.sampled_from(_OUTCOMES)),
    )
    # Guard the generator itself: every drawn record must be schema-valid.
    validate(record)
    return record


def _key(record: DecisionRecord) -> tuple:
    """A hashable projection over all fields, for multiset comparison."""
    return (
        record.user_id,
        record.timestamp,
        record.domain,
        record.location,
        record.weather,
        record.day_type,
        record.time_of_day,
        record.mood_energy,
        record.decision_made,
        record.outcome,
    )


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(
    records=st.lists(decision_records(), min_size=2, max_size=40),
    val_fraction=st.floats(
        min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False
    ),
)
def test_temporal_split_integrity(
    records: list[DecisionRecord], val_fraction: float
) -> None:
    """``time_based_split`` partitions the input and keeps train ts <= val ts."""
    train, val = time_based_split(records, val_fraction)

    # Partition (count): no records lost or invented.
    assert len(train) + len(val) == len(records)

    # Partition (multiset): train + val is exactly the input multiset, regardless of
    # the order the split returns them in.
    assert Counter(_key(r) for r in train + val) == Counter(_key(r) for r in records)

    # Temporal ordering: when both sides are non-empty, the whole train tail precedes
    # the whole val head in time (equal boundary timestamps are allowed).
    if train and val:
        assert max(r.timestamp for r in train) <= min(r.timestamp for r in val)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_invalid_val_fraction_raises(bad_fraction: float) -> None:
    """``val_fraction`` outside the open interval (0, 1) raises ``ValueError``."""
    records = [
        DecisionRecord(stress_level="low", 
            user_id="u1",
            timestamp=_TS_POOL[0],
            domain=Domain.FOCUS.value,
            location="home",
            weather="clear",
            day_type=day_type(_TS_POOL[0]).value,
            time_of_day=time_of_day(_TS_POOL[0]).value,
            mood_energy=0.5,
            decision_made="pomodoro",
            outcome="good",
        )
    ]
    with pytest.raises(ValueError):
        time_based_split(records, bad_fraction)
