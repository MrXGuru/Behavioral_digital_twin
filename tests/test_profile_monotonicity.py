"""Property-based test for online profile-update monotonicity.

Property 7: Profile monotonicity
    *For any* ``UserProfile`` and any sequence of update batches applied via
    ``ProfileUpdater.update``:

    * each ``decision_counts[option]`` is non-decreasing across updates and increases by
      exactly the number of occurrences of that option in the new records (Req 11.2);
    * ``last_updated`` is non-decreasing across updates and, after an update with a
      non-empty batch, equals ``max(prior last_updated, max timestamp among new records)``
      (Req 11.4).

The test also asserts ``ProfileUpdater.update`` is side-effect free (the input profile is
never mutated).

**Validates: Requirements 11.2, 11.4**
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data.schema import DOMAIN_OPTIONS, DecisionRecord, Domain, day_type, time_of_day
from personalization.profile_store import UserProfile
from personalization.updater import ProfileUpdater

_USER_ID = "u-monotonic"

# Timestamps are timezone-aware (UTC) so they are comparable with the cold-start
# ``last_updated`` (the UTC Unix epoch) and always strictly after it.
_MIN_TS = datetime(2000, 1, 1)
_MAX_TS = datetime(2030, 1, 1)


@st.composite
def decision_records(draw: st.DrawFn) -> DecisionRecord:
    """Draw a single schema-valid ``DecisionRecord`` for the fixed test user.

    ``day_type`` / ``time_of_day`` are derived from the drawn timestamp and
    ``decision_made`` is drawn from the chosen domain's option set, so every record is
    schema-valid. Only ``timestamp`` and ``decision_made`` matter for Property 7, but a
    fully valid record keeps the input realistic.
    """
    ts = draw(
        st.datetimes(
            min_value=_MIN_TS,
            max_value=_MAX_TS,
            timezones=st.just(timezone.utc),
        )
    )
    domain = draw(st.sampled_from(list(Domain)))
    decision = draw(st.sampled_from(DOMAIN_OPTIONS[domain]))
    mood = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return DecisionRecord(stress_level="low", 
        user_id=_USER_ID,
        timestamp=ts,
        domain=domain.value,
        location=draw(st.sampled_from(["home", "work", "gym", "cafe"])),
        weather=draw(st.sampled_from(["clear", "rain", "cloudy", "snow"])),
        day_type=day_type(ts).value,
        time_of_day=time_of_day(ts).value,
        mood_energy=mood,
        decision_made=decision,
        outcome=draw(st.sampled_from(["ok", "bad", "neutral"])),
    )


@st.composite
def update_batches(draw: st.DrawFn) -> list[list[DecisionRecord]]:
    """Draw a sequence of update batches (each a possibly-empty list of records)."""
    return draw(
        st.lists(
            st.lists(decision_records(), min_size=0, max_size=5),
            min_size=1,
            max_size=6,
        )
    )


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(batches=update_batches())
def test_profile_monotonicity(batches: list[list[DecisionRecord]]) -> None:
    """Counts and ``last_updated`` evolve monotonically across update batches."""
    updater = ProfileUpdater(learning_rate=0.3)
    profile = UserProfile.cold_start(_USER_ID)

    for batch in batches:
        # Snapshot the prior state to check monotonicity and non-mutation.
        prior_counts = dict(profile.decision_counts)
        prior_last_updated = profile.last_updated
        prior_embedding = list(profile.embedding)
        input_profile = profile

        updated = updater.update(profile, batch)

        # Side-effect free: the input profile is never mutated.
        assert input_profile.decision_counts == prior_counts
        assert input_profile.last_updated == prior_last_updated
        assert input_profile.embedding == prior_embedding

        # user_id is preserved across updates.
        assert updated.user_id == _USER_ID

        # Req 11.2: counts are monotonic non-decreasing, and each option's count
        # increases by exactly its number of occurrences in the new records.
        batch_counts = Counter(record.decision_made for record in batch)
        all_options = set(prior_counts) | set(updated.decision_counts)
        for option in all_options:
            prior_value = prior_counts.get(option, 0)
            new_value = updated.decision_counts.get(option, 0)
            assert new_value >= prior_value  # never drops below its prior value
            assert new_value == prior_value + batch_counts.get(option, 0)

        # Req 11.4: last_updated is non-decreasing and equals
        # max(prior, max timestamp among new records) for a non-empty batch.
        assert updated.last_updated >= prior_last_updated
        if batch:
            expected = max(
                prior_last_updated, max(record.timestamp for record in batch)
            )
            assert updated.last_updated == expected
        else:
            assert updated.last_updated == prior_last_updated

        profile = updated


def test_empty_batch_leaves_profile_unchanged() -> None:
    """An empty update batch returns the profile unchanged (example case)."""
    updater = ProfileUpdater(learning_rate=0.3)
    profile = UserProfile.cold_start(_USER_ID)

    updated = updater.update(profile, [])

    assert updated.decision_counts == profile.decision_counts
    assert updated.last_updated == profile.last_updated
    assert updated.embedding == profile.embedding


def test_counts_increment_by_occurrences_example() -> None:
    """Counts increment by per-option occurrences across two sequential batches."""
    updater = ProfileUpdater(learning_rate=0.3)
    profile = UserProfile.cold_start(_USER_ID)

    ts1 = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)

    def focus_record(ts: datetime, decision: str) -> DecisionRecord:
        return DecisionRecord(stress_level="low", 
            user_id=_USER_ID,
            timestamp=ts,
            domain=Domain.FOCUS.value,
            location="home",
            weather="clear",
            day_type=day_type(ts).value,
            time_of_day=time_of_day(ts).value,
            mood_energy=0.5,
            decision_made=decision,
            outcome="ok",
        )

    batch1 = [focus_record(ts1, "pomodoro"), focus_record(ts1, "pomodoro")]
    profile = updater.update(profile, batch1)
    assert profile.decision_counts == {"pomodoro": 2}
    assert profile.last_updated == ts1

    batch2 = [focus_record(ts2, "pomodoro"), focus_record(ts2, "flow_state")]
    profile = updater.update(profile, batch2)
    assert profile.decision_counts == {"pomodoro": 3, "flow_state": 1}
    assert profile.last_updated == ts2
