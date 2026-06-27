"""Property-based test for temporal ordering of stored/generated records.

Property 3: Temporal ordering
    *For any* set of decision records appended to a ``DecisionStore`` in any order,
    ``load()`` returns them sorted ascending by timestamp. The same ordering guarantee
    holds for ``SyntheticDataGenerator.generate()``.

**Validates: Requirements 1.2, 3.3**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pytest

from data.decision_store import CSV_BACKEND, SQLITE_BACKEND, DecisionStore
from data.schema import (
    DOMAIN_OPTIONS,
    DecisionRecord,
    Domain,
    day_type,
    time_of_day,
)
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator

_UTC = timezone.utc

# A fixed calendar anchor; timestamps are sampled as offsets from this instant so the
# generated records span a realistic, varied window without depending on wall-clock time.
_ANCHOR = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)

# All supported domain string values.
_DOMAIN_VALUES = [d.value for d in Domain]


@st.composite
def decision_records(draw: st.DrawFn) -> DecisionRecord:
    """Draw a single schema-valid :class:`DecisionRecord`.

    ``time_of_day`` and ``day_type`` are derived from the sampled timestamp via the
    schema helpers so the record always satisfies the consistency rules, and
    ``decision_made`` is drawn from the domain's option set. This keeps generated
    records inside the valid input space while letting timestamps vary widely.
    """
    # Sample a timestamp as a second-offset within a ~1.5 year window. Seconds give
    # fine-grained ordering while the wide window produces frequent distinct values.
    offset_seconds = draw(st.integers(min_value=0, max_value=46_000_000))
    timestamp = _ANCHOR + timedelta(seconds=offset_seconds)

    domain = draw(st.sampled_from(_DOMAIN_VALUES))
    decision_made = draw(st.sampled_from(DOMAIN_OPTIONS[Domain(domain)]))

    return DecisionRecord(stress_level="low", 
        user_id=draw(st.sampled_from(["u1", "u2", "alice"])),
        timestamp=timestamp,
        domain=domain,
        location=draw(st.sampled_from(["home", "work", "gym", "store", "transit"])),
        weather=draw(st.sampled_from(["clear", "cloudy", "rain", "snow"])),
        day_type=day_type(timestamp).value,
        time_of_day=time_of_day(timestamp).value,
        mood_energy=draw(st.floats(min_value=0.0, max_value=1.0)),
        decision_made=decision_made,
        outcome=draw(st.sampled_from(["good", "neutral", "bad"])),
    )


def _is_ascending(records: list[DecisionRecord]) -> bool:
    """Return True if ``records`` are non-decreasing by timestamp."""
    return all(
        records[i].timestamp <= records[i + 1].timestamp
        for i in range(len(records) - 1)
    )


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(
    records=st.lists(decision_records(), max_size=40),
    shuffle_seed=st.randoms(use_true_random=False),
)
@pytest.mark.parametrize("backend", [CSV_BACKEND, SQLITE_BACKEND])
def test_load_returns_records_ascending_by_timestamp(
    backend: str,
    records: list[DecisionRecord],
    shuffle_seed,
) -> None:
    """load() returns appended records sorted ascending by timestamp (Property 3)."""
    # Append in a shuffled order to prove ordering is enforced on load, not insertion.
    shuffled = list(records)
    shuffle_seed.shuffle(shuffled)

    suffix = "csv" if backend == CSV_BACKEND else "db"
    # A fresh temp dir per generated example so stores never leak across inputs
    # (function-scoped fixtures are not reset between @given examples).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"store.{suffix}"
        store = DecisionStore(backend=backend, path=str(path))
        store.append(shuffled)

        loaded = store.load()

    # Same multiset of records (nothing lost or duplicated) ...
    assert len(loaded) == len(records)
    # ... and the result is ascending by timestamp.
    assert _is_ascending(loaded)


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(
    records=st.lists(decision_records(), min_size=1, max_size=40),
    shuffle_seed=st.randoms(use_true_random=False),
)
@pytest.mark.parametrize("backend", [CSV_BACKEND, SQLITE_BACKEND])
def test_load_with_filters_preserves_ascending_order(
    backend: str,
    records: list[DecisionRecord],
    shuffle_seed,
) -> None:
    """Filtered loads (user_id / since) also return ascending-by-timestamp results."""
    shuffled = list(records)
    shuffle_seed.shuffle(shuffled)

    suffix = "csv" if backend == CSV_BACKEND else "db"
    # Filter by an existing user_id and a midpoint `since` cutoff.
    user_id = records[0].user_id
    cutoff = _ANCHOR + timedelta(seconds=23_000_000)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"store.{suffix}"
        store = DecisionStore(backend=backend, path=str(path))
        store.append(shuffled)

        by_user = store.load(user_id=user_id)
        since_filtered = store.load(since=cutoff)

    assert _is_ascending(by_user)
    assert all(r.user_id == user_id for r in by_user)

    assert _is_ascending(since_filtered)
    cutoff_aware = cutoff if cutoff.tzinfo is not None else cutoff.replace(tzinfo=_UTC)
    assert all(r.timestamp >= cutoff_aware for r in since_filtered)


@st.composite
def generator_configs(draw: st.DrawFn) -> GeneratorConfig:
    """Draw a valid ``GeneratorConfig`` within the generator's precondition space."""
    lo = draw(st.integers(min_value=0, max_value=4))
    hi = draw(st.integers(min_value=lo, max_value=lo + 4))
    domains = draw(
        st.lists(st.sampled_from(_DOMAIN_VALUES), min_size=1, unique=True)
    )
    return GeneratorConfig(
        n_days=draw(st.integers(min_value=1, max_value=20)),
        decisions_per_day=(lo, hi),
        domains=domains,
        weekend_shift=draw(st.floats(min_value=0.0, max_value=1.0)),
        drift_rate=draw(st.floats(min_value=0.0, max_value=1.0)),
        noise=draw(st.floats(min_value=0.0, max_value=1.0)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        user_id=draw(st.sampled_from(["u1", "u2", "alice"])),
    )


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(config=generator_configs())
def test_generator_output_is_ascending_by_timestamp(config: GeneratorConfig) -> None:
    """SyntheticDataGenerator.generate() returns records ascending by timestamp."""
    records = SyntheticDataGenerator(config).generate()
    assert _is_ascending(records)
