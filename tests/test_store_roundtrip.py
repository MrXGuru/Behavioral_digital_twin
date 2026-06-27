"""Unit tests for :class:`~data.decision_store.DecisionStore` round-trip and filtering.

Covers Requirements 3.1, 3.2, 3.4, 3.5:

* 3.1/3.2: appending then loading returns the same records field-by-field
  (including a tz-aware UTC :class:`datetime` ``timestamp`` and a ``float``
  ``mood_energy``), and both the CSV and SQLite backends produce identical
  loaded results for the same input.
* 3.4: ``load`` supports ``user_id`` and ``since`` filtering and returns records
  sorted ascending by timestamp; ``append`` is additive.
* 3.5: ``count`` (overall and per ``user_id``) returns the correct totals.

All timestamps are UTC-aware so they round-trip cleanly without the naive/aware
TypeError that arises when mixing the two kinds.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.decision_store import CSV_BACKEND, SQLITE_BACKEND, DecisionStore
from data.schema import (
    COLUMNS,
    DOMAIN_OPTIONS,
    DecisionRecord,
    Domain,
    day_type,
    time_of_day,
)

BACKENDS = [CSV_BACKEND, SQLITE_BACKEND]

# Convenience alias so all test timestamps are UTC-aware.
_UTC = timezone.utc


def _store_path(backend: str, tmp_path) -> str:
    suffix = "csv" if backend == CSV_BACKEND else "db"
    return str(tmp_path / f"decisions.{suffix}")


def make_record(
    *,
    user_id: str = "u1",
    timestamp: datetime = datetime(2024, 1, 1, 9, 0, tzinfo=_UTC),
    domain: Domain = Domain.FOCUS,
    location: str = "home",
    weather: str = "clear",
    mood_energy: float = 0.5,
    decision_index: int = 0,
    outcome: str = "ok",
) -> DecisionRecord:
    """Build a schema-consistent :class:`DecisionRecord` with a UTC timestamp."""
    opts = DOMAIN_OPTIONS[domain]
    return DecisionRecord(stress_level="low", 
        user_id=user_id,
        timestamp=timestamp,
        domain=domain.value,
        location=location,
        weather=weather,
        day_type=day_type(timestamp).value,
        time_of_day=time_of_day(timestamp).value,
        mood_energy=mood_energy,
        decision_made=opts[decision_index % len(opts)],
        outcome=outcome,
    )


def sample_records() -> list[DecisionRecord]:
    """A small, deterministic mixed-user, mixed-timestamp record set (all UTC)."""
    return [
        make_record(user_id="u1", timestamp=datetime(2024, 1, 3, 14, 30, tzinfo=_UTC),
                    domain=Domain.TASK, mood_energy=0.25, decision_index=1),
        make_record(user_id="u1", timestamp=datetime(2024, 1, 1, 9, 0, tzinfo=_UTC),
                    domain=Domain.FOCUS, mood_energy=0.9, decision_index=2),
        DecisionRecord(user_id="alice", timestamp=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
                    domain=Domain.FOCUS, location="home", weather="clear", day_type="weekday", time_of_day="morning",
                    decision_made="admin", outcome="good",
                    mood_energy=0.0),
        make_record(user_id="u2", timestamp=datetime(2024, 1, 6, 22, 0, tzinfo=_UTC),
                    domain=Domain.FOCUS, mood_energy=1.0, decision_index=0),
    ]


def assert_records_equal(actual: DecisionRecord, expected: DecisionRecord) -> None:
    """Assert two records are equal field-by-field.

    Timestamps are compared as moments in time (UTC-normalised) so that a naive
    datetime stored and a UTC-aware datetime loaded from the store are considered
    equal when they represent the same moment.
    """
    for col in COLUMNS:
        a = getattr(actual, col)
        e = getattr(expected, col)
        if col == "timestamp":
            # Normalize both to UTC for comparison
            a_utc = a.replace(tzinfo=_UTC) if a.tzinfo is None else a.astimezone(_UTC)
            e_utc = e.replace(tzinfo=_UTC) if e.tzinfo is None else e.astimezone(_UTC)
            assert a_utc == e_utc, f"timestamp mismatch: {a!r} != {e!r}"
        else:
            assert type(a) is type(e), f"type mismatch for {col!r}: {type(a)} != {type(e)}"
            assert a == e, f"value mismatch for {col!r}: {a!r} != {e!r}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_append_then_load_round_trips(backend, tmp_path) -> None:
    """append then load returns equal records, sorted ascending by timestamp."""
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    records = sample_records()
    store.append(records)

    loaded = store.load()
    expected = sorted(records, key=lambda r: r.timestamp)

    assert len(loaded) == len(expected)
    for got, exp in zip(loaded, expected):
        assert_records_equal(got, exp)

    for rec in loaded:
        assert isinstance(rec.timestamp, datetime)
        assert rec.timestamp.tzinfo is not None  # must be tz-aware
        assert isinstance(rec.mood_energy, float)


def test_csv_and_sqlite_produce_identical_results(tmp_path) -> None:
    """The CSV and SQLite backends load identical results for the same input."""
    records = sample_records()
    csv_store = DecisionStore(CSV_BACKEND, _store_path(CSV_BACKEND, tmp_path))
    sqlite_store = DecisionStore(SQLITE_BACKEND, _store_path(SQLITE_BACKEND, tmp_path))
    csv_store.append(records)
    sqlite_store.append(records)

    csv_loaded = csv_store.load()
    sqlite_loaded = sqlite_store.load()

    assert len(csv_loaded) == len(sqlite_loaded)
    for csv_rec, sqlite_rec in zip(csv_loaded, sqlite_loaded):
        assert_records_equal(csv_rec, sqlite_rec)


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_filters_by_user_id(backend, tmp_path) -> None:
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    store.append(sample_records())

    u1 = store.load(user_id="u1")
    assert len(u1) == 2
    assert all(r.user_id == "u1" for r in u1)

    assert store.load(user_id="missing") == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_filters_by_since(backend, tmp_path) -> None:
    """load(since=...) returns only records with timestamp >= since (UTC-aware)."""
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    store.append(sample_records())

    since = datetime(2024, 1, 2, 19, 15, tzinfo=_UTC)
    loaded = store.load(since=since)

    assert all(r.timestamp >= since for r in loaded)
    # Three of the four sample timestamps are >= 2024-01-02 19:15 (boundary inclusive).
    assert len(loaded) == 3
    assert loaded == sorted(loaded, key=lambda r: r.timestamp)


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_combined_user_and_since_filter(backend, tmp_path) -> None:
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    store.append(sample_records())

    since = datetime(2024, 1, 5, 0, 0, tzinfo=_UTC)
    loaded = store.load(user_id="u2", since=since)

    assert len(loaded) == 1
    assert loaded[0].user_id == "u2"
    assert loaded[0].timestamp == datetime(2024, 1, 6, 22, 0, tzinfo=_UTC)


@pytest.mark.parametrize("backend", BACKENDS)
def test_count_overall_and_per_user(backend, tmp_path) -> None:
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    store.append(sample_records())

    assert store.count() == 4
    assert store.count(user_id="u1") == 2
    assert store.count(user_id="u2") == 2
    assert store.count(user_id="missing") == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_append_is_additive(backend, tmp_path) -> None:
    store = DecisionStore(backend, _store_path(backend, tmp_path))
    first = sample_records()[:2]
    second = sample_records()[2:]

    store.append(first)
    assert store.count() == 2

    store.append(second)
    assert store.count() == 4

    loaded = store.load()
    expected = sorted(sample_records(), key=lambda r: r.timestamp)
    assert len(loaded) == len(expected)
    for got, exp in zip(loaded, expected):
        assert_records_equal(got, exp)
