"""Unit tests for the per-user profile store.

Covers Requirement 11.1: the Profile_Store supplies a current per-user profile
(embedding + statistics) for any requested user, returning a deterministic
cold-start default for unknown users and never raising.
"""

from __future__ import annotations

from datetime import datetime, timezone

from personalization.profile_store import (
    COLD_START_TIMESTAMP,
    EMBED_DIM,
    EMBEDDING_DIM,
    UserProfile,
    UserProfileStore,
)


def test_cold_start_default_for_unknown_user():
    store = UserProfileStore()

    profile = store.get("never-seen")

    assert profile.user_id == "never-seen"
    assert profile.embedding == [0.0] * EMBEDDING_DIM
    assert profile.decision_counts == {}
    assert profile.last_updated == COLD_START_TIMESTAMP


def test_get_does_not_persist_cold_start_profile():
    store = UserProfileStore()

    store.get("u1")

    assert store.has("u1") is False


def test_upsert_then_get_returns_stored_profile():
    store = UserProfileStore()
    profile = UserProfile(
        user_id="u1",
        embedding=[0.1] * EMBEDDING_DIM,
        decision_counts={"pomodoro": 3},
        last_updated=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    store.upsert(profile)

    fetched = store.get("u1")
    assert fetched is profile
    assert store.has("u1") is True


def test_upsert_replaces_existing_profile():
    store = UserProfileStore()
    store.upsert(UserProfile(user_id="u1", decision_counts={"pomodoro": 1}))
    store.upsert(UserProfile(user_id="u1", decision_counts={"flow_state": 9}))

    assert store.get("u1").decision_counts == {"flow_state": 9}


def test_delete_removes_profile_and_is_idempotent():
    store = UserProfileStore()
    store.upsert(UserProfile(user_id="u1"))

    store.delete("u1")
    store.delete("u1")  # no-op, must not raise

    assert store.has("u1") is False


def test_cold_start_instances_are_independent():
    a = UserProfile.cold_start("a")
    b = UserProfile.cold_start("b")

    a.embedding[0] = 1.0
    a.decision_counts["pomodoro"] = 1

    assert b.embedding == [0.0] * EMBEDDING_DIM
    assert b.decision_counts == {}


def test_embedding_dim_alias_matches():
    assert EMBED_DIM == EMBEDDING_DIM


def test_embedding_summary_shape():
    summary = UserProfile.cold_start("u1").embedding_summary()

    assert summary == {"dim": EMBEDDING_DIM, "norm": 0.0}
