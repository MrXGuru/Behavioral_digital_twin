"""Focused unit tests for schema validation error reporting (Requirements 2.1-2.5).

Broad rule coverage (valid records, every violated rule) lives in
``tests/test_schema.py``. This module complements it by asserting that each
rejection raises :class:`SchemaValidationError` with a *descriptive* message that
names the offending field (and, where applicable, the offending value). Clear
diagnostics matter because validation runs at the boundary between the synthetic
data source and the downstream feature/model layers.

Covered scenarios:
    1. A fully valid record passes ``validate()`` and ``is_valid()`` is ``True``.
    2. Invalid ``domain`` rejected with a descriptive error. (Req 2.1)
    3. ``decision_made`` not in the domain option set rejected. (Req 2.2)
    4. ``mood_energy`` out of ``[0, 1]`` (-0.1 and 1.1) rejected. (Req 2.3)
    5. ``time_of_day`` inconsistent with timestamp hour rejected. (Req 2.4)
    6. ``day_type`` inconsistent with timestamp weekday rejected. (Req 2.5)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from data.schema import (
    DayType,
    DecisionRecord,
    Domain,
    SchemaValidationError,
    TimeOfDay,
    day_type,
    is_valid,
    time_of_day,
    validate,
)

# 2024-01-01 is a Monday -> weekday / 09:00 morning. Known-good baseline.
WEEKDAY_MORNING = datetime(2024, 1, 1, 9, 0)


def make_record(
    *,
    timestamp: datetime = WEEKDAY_MORNING,
    domain: str = Domain.FOCUS.value,
    decision_made: str = "pomodoro",
    mood_energy: float = 0.5,
    time_of_day_value: str | None = None,
    day_type_value: str | None = None,
) -> DecisionRecord:
    """Build a record valid by default; derive temporal fields from ``timestamp``.

    Overriding exactly one of ``time_of_day_value`` / ``day_type_value`` (or any
    single argument) lets each test violate exactly one rule at a time.
    """
    return DecisionRecord(stress_level="low", 
        user_id="u1",
        timestamp=timestamp,
        domain=domain,
        location="home",
        weather="clear",
        day_type=(
            day_type_value if day_type_value is not None else day_type(timestamp).value
        ),
        time_of_day=(
            time_of_day_value
            if time_of_day_value is not None
            else time_of_day(timestamp).value
        ),
        mood_energy=mood_energy,
        decision_made=decision_made,
        outcome="ok",
    )


# ---------------------------------------------------------------------------
# 1. Valid record passes
# ---------------------------------------------------------------------------


def test_valid_record_passes_validate_and_is_valid() -> None:
    record = make_record()
    assert validate(record) is record
    assert is_valid(record) is True


# ---------------------------------------------------------------------------
# 2. Invalid domain rejected with a descriptive error (Req 2.1)
# ---------------------------------------------------------------------------


def test_invalid_domain_error_is_descriptive() -> None:
    record = make_record(domain="teleport")
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(record)

    message = str(exc_info.value)
    assert "domain" in message  # names the offending field
    assert "teleport" in message  # names the offending value
    assert is_valid(record) is False


# ---------------------------------------------------------------------------
# 3. decision_made not in the domain option set rejected (Req 2.2)
# ---------------------------------------------------------------------------


def test_decision_not_in_option_set_error_is_descriptive() -> None:
    record = make_record(domain=Domain.FOCUS.value, decision_made="focus_Z")
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(record)

    message = str(exc_info.value)
    assert "invalid decision" in message  # names the offending field
    assert "focus_Z" in message  # names the offending value
    assert is_valid(record) is False


# ---------------------------------------------------------------------------
# 4. mood_energy out of [0, 1] rejected (Req 2.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mood", [-0.1, 1.1])
def test_mood_energy_out_of_range_error_is_descriptive(mood: float) -> None:
    record = make_record(mood_energy=mood)
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(record)

    message = str(exc_info.value)
    assert "mood_energy" in message  # names the offending field
    assert str(mood) in message  # names the offending value
    assert is_valid(record) is False


# ---------------------------------------------------------------------------
# 5. time_of_day inconsistent with timestamp hour rejected (Req 2.4)
# ---------------------------------------------------------------------------


def test_mismatched_time_of_day_error_is_descriptive() -> None:
    # 09:00 -> morning; claim it is "night".
    record = make_record(
        timestamp=WEEKDAY_MORNING, time_of_day_value=TimeOfDay.NIGHT.value
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(record)

    message = str(exc_info.value)
    assert "time_of_day" in message  # names the offending field
    assert "morning" in message  # names the expected value
    assert is_valid(record) is False


# ---------------------------------------------------------------------------
# 6. day_type inconsistent with timestamp weekday rejected (Req 2.5)
# ---------------------------------------------------------------------------


def test_mismatched_day_type_error_is_descriptive() -> None:
    # Monday -> weekday; claim it is "weekend".
    record = make_record(
        timestamp=WEEKDAY_MORNING, day_type_value=DayType.WEEKEND.value
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        validate(record)

    message = str(exc_info.value)
    assert "day_type" in message  # names the offending field
    assert "weekday" in message  # names the expected value
    assert is_valid(record) is False
