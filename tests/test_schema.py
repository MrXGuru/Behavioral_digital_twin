"""Unit tests for the shared decision-record schema validation.

Covers Requirements 2.1-2.5: a valid record passes, and each individually
violated rule (bad domain, decision not in option set, out-of-range mood,
mismatched time_of_day, mismatched day_type) is rejected with a
``SchemaValidationError``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schema import (
    DayType,
    Domain,
    DecisionRecord,
    SchemaValidationError,
    TimeOfDay,
    day_type,
    is_valid,
    options,
    time_of_day,
    validate,
)


# A Monday (weekday=0) at 09:00 -> weekday / morning. Used as a known-good base.
WEEKDAY_MORNING = datetime(2024, 1, 1, 9, 0)  # 2024-01-01 is a Monday
# A Saturday (weekday=5) at 22:00 -> weekend / night.
WEEKEND_NIGHT = datetime(2024, 1, 6, 22, 0)  # 2024-01-06 is a Saturday


def make_record(
    *,
    timestamp: datetime = WEEKDAY_MORNING,
    domain: str = Domain.FOCUS.value,
    decision_made: str = "pomodoro",
    mood_energy: float = 0.5,
    time_of_day_value: str | None = None,
    day_type_value: str | None = None,
) -> DecisionRecord:
    """Build a DecisionRecord that is valid by default.

    ``time_of_day``/``day_type`` are derived from ``timestamp`` unless explicitly
    overridden, so individual tests can violate exactly one rule at a time.
    """
    return DecisionRecord(stress_level="low", 
        user_id="u1",
        timestamp=timestamp,
        domain=domain,
        location="home",
        weather="clear",
        day_type=day_type_value
        if day_type_value is not None
        else day_type(timestamp).value,
        time_of_day=time_of_day_value
        if time_of_day_value is not None
        else time_of_day(timestamp).value,
        mood_energy=mood_energy,
        decision_made=decision_made,
        outcome="ok",
    )


# ---------------------------------------------------------------------------
# Timestamp-derived helpers (support Rules 4 & 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hour, expected",
    [
        (5, TimeOfDay.MORNING),
        (11, TimeOfDay.MORNING),
        (12, TimeOfDay.AFTERNOON),
        (16, TimeOfDay.AFTERNOON),
        (17, TimeOfDay.EVENING),
        (20, TimeOfDay.EVENING),
        (21, TimeOfDay.NIGHT),
        (4, TimeOfDay.NIGHT),
        (0, TimeOfDay.NIGHT),
    ],
)
def test_time_of_day_boundaries(hour: int, expected: TimeOfDay) -> None:
    assert time_of_day(datetime(2024, 1, 1, hour, 0)) == expected


@pytest.mark.parametrize(
    "day, expected",
    [
        (1, DayType.WEEKDAY),  # Mon
        (5, DayType.WEEKDAY),  # Fri
        (6, DayType.WEEKEND),  # Sat
        (7, DayType.WEEKEND),  # Sun
    ],
)
def test_day_type_weekday_vs_weekend(day: int, expected: DayType) -> None:
    assert day_type(datetime(2024, 1, day)) == expected


def test_options_returns_domain_option_set() -> None:
    assert options("focus") == ("pomodoro", "flow_state", "light_work", "admin")
    assert options(Domain.TASK) == ("deep_work", "email", "meeting", "break")


def test_options_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError):
        options("teleport")


# ---------------------------------------------------------------------------
# Valid records pass
# ---------------------------------------------------------------------------


def test_valid_record_passes_and_returns_same_record() -> None:
    record = make_record()
    assert validate(record) is record
    assert is_valid(record) is True


@pytest.mark.parametrize("domain", [d.value for d in Domain])
def test_valid_record_for_each_domain(domain: str) -> None:
    decision = options(domain)[0]
    record = make_record(domain=domain, decision_made=decision)
    assert is_valid(record) is True


@pytest.mark.parametrize("mood", [0.0, 0.5, 1.0])
def test_mood_energy_boundaries_are_valid(mood: float) -> None:
    assert is_valid(make_record(mood_energy=mood)) is True


def test_weekend_night_record_is_valid() -> None:
    assert is_valid(make_record(timestamp=WEEKEND_NIGHT)) is True


# ---------------------------------------------------------------------------
# Rule 1: invalid domain rejected (Requirement 2.1)
# ---------------------------------------------------------------------------


def test_invalid_domain_rejected() -> None:
    record = make_record(domain="teleport", decision_made="pomodoro")
    with pytest.raises(SchemaValidationError, match="invalid domain"):
        validate(record)
    assert is_valid(record) is False


# ---------------------------------------------------------------------------
# Rule 2: decision_made not in option set rejected (Requirement 2.2)
# ---------------------------------------------------------------------------


def test_decision_not_in_option_set_rejected() -> None:
    record = make_record(domain=Domain.FOCUS.value, decision_made="focus_Z")
    with pytest.raises(SchemaValidationError, match="invalid decision"):
        validate(record)
    assert is_valid(record) is False


def test_decision_from_other_domain_rejected() -> None:
    # "coffee" is a valid purchase option but not a focus option.
    record = make_record(domain=Domain.FOCUS.value, decision_made="coffee")
    with pytest.raises(SchemaValidationError, match="invalid decision"):
        validate(record)


# ---------------------------------------------------------------------------
# Rule 3: mood_energy out of range rejected (Requirement 2.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mood", [-0.01, 1.01, -1.0, 2.0])
def test_mood_energy_out_of_range_rejected(mood: float) -> None:
    record = make_record(mood_energy=mood)
    with pytest.raises(SchemaValidationError, match="out of range"):
        validate(record)
    assert is_valid(record) is False


def test_mood_energy_non_numeric_rejected() -> None:
    record = make_record()
    record.mood_energy = "high"  # type: ignore[assignment]
    with pytest.raises(SchemaValidationError, match="must be a number"):
        validate(record)


def test_mood_energy_bool_rejected() -> None:
    record = make_record()
    record.mood_energy = True  # type: ignore[assignment]
    with pytest.raises(SchemaValidationError, match="must be a number"):
        validate(record)


# ---------------------------------------------------------------------------
# Rule 4: time_of_day inconsistent with timestamp rejected (Requirement 2.4)
# ---------------------------------------------------------------------------


def test_mismatched_time_of_day_rejected() -> None:
    # WEEKDAY_MORNING is 09:00 -> morning; claim it is "night".
    record = make_record(
        timestamp=WEEKDAY_MORNING, time_of_day_value=TimeOfDay.NIGHT.value
    )
    with pytest.raises(SchemaValidationError, match="time_of_day"):
        validate(record)
    assert is_valid(record) is False


def test_non_datetime_timestamp_rejected() -> None:
    record = make_record()
    record.timestamp = "2024-01-01T09:00"  # type: ignore[assignment]
    with pytest.raises(SchemaValidationError, match="timestamp must be a datetime"):
        validate(record)


# ---------------------------------------------------------------------------
# Rule 5: day_type inconsistent with timestamp rejected (Requirement 2.5)
# ---------------------------------------------------------------------------


def test_mismatched_day_type_rejected() -> None:
    # WEEKDAY_MORNING is a Monday -> weekday; claim it is "weekend".
    record = make_record(
        timestamp=WEEKDAY_MORNING, day_type_value=DayType.WEEKEND.value
    )
    with pytest.raises(SchemaValidationError, match="day_type"):
        validate(record)
    assert is_valid(record) is False
