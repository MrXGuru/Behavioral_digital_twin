"""Integration tests for personalization and drift surfacing (Task 10.7).

Exercises the live FastAPI app end-to-end via ``TestClient`` to confirm three
behaviors the dashboard depends on:

1. ``drift_status`` is present in the ``POST /predict_next_decision`` response and
   carries the expected ``DriftStatus`` shape (Requirements 7.1, 10.1).
2. ``GET /user_profile/{user_id}`` returns the documented profile summary shape:
   ``user_id``, ``decision_counts``, ``embedding_summary`` (``dim`` + ``norm``),
   and ``last_updated`` (Requirements 7.4, 11.1).
3. A user's profile is refreshed/populated by ``POST /retrain`` — counts become
   non-empty and ``last_updated`` advances past the cold-start epoch (Req 11.1).

Isolation: each test runs against a :class:`TwinService` backed by a temporary
SQLite store (monkeypatched onto ``api.main.service``) so the real
``data/generated/decisions.db`` is never touched. Distinct user_ids and a distinct
module name avoid any collision with ``tests/test_api_integration.py`` (Task 9.3).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from api.service import TwinService
from data.decision_store import DecisionStore
from personalization.profile_store import COLD_START_TIMESTAMP


@pytest.fixture()
def client_and_service(tmp_path, monkeypatch):
    """A TestClient wired to a TwinService backed by an isolated temp SQLite store."""
    store = DecisionStore("sqlite", str(tmp_path / "t.db"))
    service = TwinService(store)
    monkeypatch.setattr(api_main, "service", service)
    with TestClient(app) as client:
        yield client, service


def _parse_iso_z(value: str) -> datetime:
    """Parse an ISO-8601 ``...Z`` timestamp into a tz-aware UTC datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_drift_status_present_in_predict(client_and_service):
    """drift_status is present and well-formed in the predict response (Req 7.1, 10.1)."""
    client, service = client_and_service
    user_id = "pd_drift_user"

    # Seed decision data for the fresh user, then retrain so a model artifact exists
    # (POST /predict_next_decision does not auto-train; retrain only trains once the
    # store has enough records for a valid temporal split).
    service.ensure_user(user_id)
    retrain = client.post("/retrain", json={"user_id": user_id})
    assert retrain.status_code == 200
    assert retrain.json()["status"] == "retrained"

    resp = client.post(
        "/predict_next_decision",
        json={
            "user_id": user_id,
            "context": {
                "domain": "focus",
                "location": "home",
                "weather": "clear",
                "day_type": "weekday",
                "time_of_day": "morning",
                "mood_energy": 0.5,
            },
            "recent_decisions": ["pomodoro", "flow_state"],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "drift_status" in body
    drift = body["drift_status"]
    assert set(("drift", "score", "window_acc")).issubset(drift.keys())
    assert isinstance(drift["drift"], bool)
    assert isinstance(drift["score"], (int, float))
    assert drift["window_acc"] is None or isinstance(drift["window_acc"], (int, float))
    if drift["window_acc"] is not None:
        assert 0.0 <= drift["window_acc"] <= 1.0


def test_user_profile_shape(client_and_service):
    """GET /user_profile/{id} returns the documented profile summary shape (Req 7.4)."""
    client, _ = client_and_service
    user_id = "pd_profile_user"

    resp = client.get(f"/user_profile/{user_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["user_id"] == user_id
    assert isinstance(body["decision_counts"], dict)

    summary = body["embedding_summary"]
    assert set(("dim", "norm")).issubset(summary.keys())
    assert isinstance(summary["dim"], int)
    assert summary["dim"] >= 0
    assert isinstance(summary["norm"], (int, float))
    assert summary["norm"] >= 0.0

    # last_updated is part of the contract (may be None for a pure cold start).
    assert "last_updated" in body


def test_profile_refreshed_on_retrain(client_and_service):
    """POST /retrain refreshes/populates the user's profile (Req 11.1)."""
    client, service = client_and_service
    user_id = "pd_refresh_user"

    # Seed decision data WITHOUT training (ensure_user only populates the store; it
    # does not refresh the profile). The profile therefore remains a cold-start
    # default until the retrain below folds the records in.
    service.ensure_user(user_id)

    # Before any training the profile is a cold-start default: empty counts at epoch.
    before = service.profiles.get(user_id)
    assert before.decision_counts == {}
    assert before.last_updated == COLD_START_TIMESTAMP

    retrain = client.post("/retrain", json={"user_id": user_id})
    assert retrain.status_code == 200
    assert retrain.json()["status"] == "retrained"

    # After retrain the persisted profile is populated and advanced past the epoch.
    after = service.profiles.get(user_id)
    assert sum(after.decision_counts.values()) > 0
    assert after.last_updated > COLD_START_TIMESTAMP

    # The advance is also visible through the public profile endpoint.
    resp = client.get(f"/user_profile/{user_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sum(body["decision_counts"].values()) > 0
    assert _parse_iso_z(body["last_updated"]) > COLD_START_TIMESTAMP
