"""Shared, versioned data schema for the Behavioral Digital Twin.

This module is the single source of truth for the decision record data model that is
shared across the data, feature, and model layers. It defines:

* the domain enum and per-domain option sets (``focus``, ``task``, ``purchase``),
* the ``time_of_day`` and ``day_type`` enums plus timestamp-derived helpers,
* the :class:`DecisionRecord` dataclass,
* :func:`validate` enforcing all record validation rules,
* the reserved ``UNK`` / ``PAD`` sentinels, the schema version, and the canonical
  column ordering used by the store and feature pipeline.

Keeping this contract in one place lets the synthetic data source be swapped for real
data later without touching downstream layers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Schema versioning + reserved sentinels
# ---------------------------------------------------------------------------

#: Bump whenever the column ordering or field semantics change in a breaking way.
SCHEMA_VERSION: str = "1.1"

#: Allowed values for a record's ``source_mode``
SOURCE_MODES: tuple[str, ...] = ("synthetic", "real", "manual", "github", "calendar", "slack", "browser_extension")

#: Explicit bucket for categorical values unseen during feature-pipeline fit.
UNK: str = "<UNK>"

#: Reserved identifier used to left-pad short history windows to length K.
PAD: str = "<PAD>"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Supported decision domains."""

    FOCUS = "focus"
    TASK = "task"


class DayType(str, Enum):
    """Whether a timestamp falls on a weekday or a weekend."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"


class TimeOfDay(str, Enum):
    """Coarse time-of-day bucket derived from a timestamp's hour."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


# ---------------------------------------------------------------------------
# Per-domain option sets (the valid ``decision_made`` values per domain)
# ---------------------------------------------------------------------------

#: Mapping from domain to its ordered tuple of valid decision options.
DOMAIN_OPTIONS: dict[Domain, tuple[str, ...]] = {
    Domain.FOCUS: ("pomodoro", "flow_state", "light_work", "admin"),
    Domain.TASK: ("deep_work", "email", "meeting", "break"),
}


def options(domain: str | Domain) -> tuple[str, ...]:
    """Return the option set (valid ``decision_made`` values) for ``domain``.

    Raises:
        ValueError: if ``domain`` is not a supported domain.
    """
    dom = _coerce_domain(domain)
    return DOMAIN_OPTIONS[dom]


# ---------------------------------------------------------------------------
# Timestamp-derived helpers
# ---------------------------------------------------------------------------


def time_of_day(ts: datetime) -> TimeOfDay:
    """Derive the :class:`TimeOfDay` bucket from a timestamp's hour.

    Boundaries (24h clock):
        * morning   : ``05:00`` <= hour < ``12:00``
        * afternoon : ``12:00`` <= hour < ``17:00``
        * evening   : ``17:00`` <= hour < ``21:00``
        * night     : ``21:00`` <= hour or hour < ``05:00``
    """
    hour = ts.hour
    if 5 <= hour < 12:
        return TimeOfDay.MORNING
    if 12 <= hour < 17:
        return TimeOfDay.AFTERNOON
    if 17 <= hour < 21:
        return TimeOfDay.EVENING
    return TimeOfDay.NIGHT


def day_type(ts: datetime) -> DayType:
    """Derive the :class:`DayType` from a timestamp's weekday.

    ``datetime.weekday()`` returns 0 (Monday) .. 6 (Sunday); values 5 and 6
    (Saturday/Sunday) are weekend, everything else is a weekday.
    """
    return DayType.WEEKEND if ts.weekday() >= 5 else DayType.WEEKDAY


# ---------------------------------------------------------------------------
# DecisionRecord data model
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """A single observed decision with its context, choice, and outcome.

    Attributes:
        user_id: Identifier of the user the decision belongs to.
        timestamp: When the decision was made.
        domain: One of ``focus`` / ``task`` / ``purchase``.
        location: Categorical context (e.g. ``home``, ``work``).
        weather: Categorical context (e.g. ``clear``, ``rain``).
        day_type: ``weekday`` / ``weekend`` (consistent with ``timestamp``).
        time_of_day: ``morning`` / ``afternoon`` / ``evening`` / ``night``
            (consistent with ``timestamp``).
        mood_energy: Proxy for mood/energy in the range ``[0, 1]``.
        stress_level: Proxy for stress in ``low`` / ``medium`` / ``high``.
        decision_made: The chosen option; must belong to ``domain``'s option set.
        outcome: Realized outcome label for the decision.
        source_mode: Provenance of the record ("manual", "github", "calendar", etc).
        domain_category: Used for browser records (focus, communication, distraction, neutral).
        duration_seconds: Time spent on this record, used for passive data (browser, meetings).
    """

    user_id: str
    timestamp: datetime
    domain: str
    location: str
    weather: str
    day_type: str
    time_of_day: str
    mood_energy: float
    stress_level: str
    decision_made: str
    outcome: str
    source_mode: str = "manual"
    domain_category: str | None = None
    duration_seconds: int | None = None


#: Canonical column ordering used by the store and feature pipeline. Derived from
#: the dataclass field order so the model and persisted layout never drift apart.
COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(DecisionRecord))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class SchemaValidationError(ValueError):
    """Raised when a :class:`DecisionRecord` violates a schema validation rule."""


def _coerce_domain(domain: str | Domain) -> Domain:
    """Return the :class:`Domain` for ``domain`` or raise ``ValueError``."""
    if isinstance(domain, Domain):
        return domain
    try:
        return Domain(domain)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Domain)
        raise ValueError(
            f"invalid domain {domain!r}; must be one of: {valid}"
        ) from exc


def validate(record: DecisionRecord) -> DecisionRecord:
    """Validate a :class:`DecisionRecord` against all schema rules.

    Enforces:
        1. ``domain`` is one of the supported domains.
        2. ``decision_made`` belongs to the domain's option set.
        3. ``mood_energy`` is within ``[0, 1]``.
        4. ``time_of_day`` is consistent with ``timestamp`` hour.
        5. ``day_type`` is consistent with ``timestamp`` weekday.
        6. ``source_mode`` is one of the allowed values (``synthetic`` / ``real``).
        7. ``stress_level`` is one of (``low``, ``medium``, ``high``).

    Returns:
        The same ``record`` (so callers can chain) when every rule holds.

    Raises:
        SchemaValidationError: with a descriptive message on the first violation.
    """
    # Rule 1: valid domain.
    try:
        domain = Domain(record.domain)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Domain)
        raise SchemaValidationError(
            f"invalid domain {record.domain!r}; must be one of: {valid}"
        ) from exc

    # Rule 2: decision_made in the domain option set (Relaxed to allow manual input)
    option_set = DOMAIN_OPTIONS[domain]
    if record.decision_made not in option_set:
        raise SchemaValidationError(
            f"invalid decision {record.decision_made!r} for domain {domain.value!r}; "
            f"must be one of: {', '.join(option_set)}"
        )

    # Rule 3: mood_energy in [0, 1].
    if not isinstance(record.mood_energy, (int, float)) or isinstance(
        record.mood_energy, bool
    ):
        raise SchemaValidationError(
            f"mood_energy must be a number, got {type(record.mood_energy).__name__}"
        )
    if not 0.0 <= float(record.mood_energy) <= 1.0:
        raise SchemaValidationError(
            f"mood_energy {record.mood_energy!r} is out of range; must be in [0, 1]"
        )

    # Rules 4 & 5: time_of_day / day_type consistent with the timestamp.
    if not isinstance(record.timestamp, datetime):
        raise SchemaValidationError(
            f"timestamp must be a datetime, got {type(record.timestamp).__name__}"
        )

    expected_tod = time_of_day(record.timestamp)
    if record.time_of_day != expected_tod.value:
        raise SchemaValidationError(
            f"time_of_day {record.time_of_day!r} is inconsistent with timestamp "
            f"hour {record.timestamp.hour}; expected {expected_tod.value!r}"
        )

    expected_day_type = day_type(record.timestamp)
    if record.day_type != expected_day_type.value:
        raise SchemaValidationError(
            f"day_type {record.day_type!r} is inconsistent with timestamp "
            f"weekday {record.timestamp.weekday()}; expected {expected_day_type.value!r}"
        )

    # Rule 6: source_mode is one of the allowed values.
    if record.source_mode not in SOURCE_MODES:
        raise SchemaValidationError(
            f"source_mode {record.source_mode!r} is not valid; must be one of: "
            f"{', '.join(SOURCE_MODES)}"
        )

    # Rule 7: stress_level
    if getattr(record, "stress_level", "medium") not in ("low", "medium", "high"):
        raise SchemaValidationError(
            f"stress_level {getattr(record, 'stress_level')!r} is not valid; must be low, medium, or high"
        )

    return record


def is_valid(record: DecisionRecord) -> bool:
    """Return ``True`` if ``record`` passes :func:`validate`, ``False`` otherwise."""
    try:
        validate(record)
    except SchemaValidationError:
        return False
    return True
