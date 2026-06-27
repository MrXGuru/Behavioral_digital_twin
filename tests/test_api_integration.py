"""Integration tests for the Prediction Engine API via FastAPI's ``TestClient``.

Exercises the Requirement 7 POST endpoints end-to-end through the real ASGI app:

* ``POST /predict_next_decision`` happy-path response shape and invariants
  (Requirements 7.1, 7.3),
* ``POST /retrain`` happy-path status + per-domain evaluation metrics (Requirement 7.5),
* ``POST /retrain`` insufficient-data skip returning HTTP 200 (Requirement 7.7),
* ``POST /predict_next_decision`` 409 for an untrained domain (Requirement 7.6).

Test isolation: the module-level ``api.main.service`` writes to a real SQLite DB and
auto-seeds synthetic data on the training path. Each test (or module-scoped trained
fixture) swaps in a fresh :class:`TwinService` backed by a temporary SQLite file so the
real DB is never touched and scenarios are deterministic.

**Validates: Requirements 7.1, 7.3, 7.5, 7.6, 7.7**
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app
from api.service import TwinService
from data.decision_store import DecisionStore
from data.schema import options

# Raw option set for the ``focus`` domain (pomodoro / flow_state / light_work / admin).
FOCUS_OPTIONS = set(options("focus"))

# A well-formed prediction context for the ``focus`` domain.
FOCUS_CONTEXT = {
    "domain": "focus",
    "location": "home",
    "weather": "clear",
    "day_type": "weekday",
    "time_of_day": "morning",
    "mood_energy": 0.5,
}


def _isolated_service(tmp_dir) -> TwinService:
    """Build a TwinService backed by an empty temporary SQLite store."""
    store = DecisionStore("sqlite", str(tmp_dir / "t.db"))
    return TwinService(store)


@pytest.fixture
def fresh_client(tmp_path, monkeypatch):
    """A TestClient wired to a fresh, empty isolated service (auto-restored)."""
    svc = _isolated_service(tmp_path)
    monkeypatch.setattr(main_module, "service", svc)
    return TestClient(app)


@pytest.fixture(scope="module")
def trained_ctx(tmp_path_factory):
    """Train once on a temp store, yielding the client + retrain response to reuse.

    Module-scoped so the (relatively expensive) seed-and-train flow runs a single time
    for both the retrain happy-path and predict happy-path assertions.
    """
    tmp_dir = tmp_path_factory.mktemp("trained")
    svc = _isolated_service(tmp_dir)
    original = main_module.service
    main_module.service = svc
    try:
        client = TestClient(app)
        user_id = "twin_happy_user"
        # Populate the isolated store with synthetic history for this user so the
        # Requirement 7 POST /retrain path (which does not auto-seed) has enough
        # records for a valid temporal split and actually trains.
        svc.ensure_user(user_id)
        retrain = client.post("/retrain", json={"user_id": user_id})
        yield {"client": client, "user_id": user_id, "retrain": retrain}
    finally:
        main_module.service = original


# ---------------------------------------------------------------------------
# Requirement 7.6: missing-artifact -> HTTP 409
# ---------------------------------------------------------------------------


def test_predict_missing_artifact_returns_409(fresh_client):
    """Predicting before any retrain yields 409 directing the client to /retrain."""
    resp = fresh_client.post(
        "/predict_next_decision",
        json={
            "user_id": "never_trained_user",
            "context": FOCUS_CONTEXT,
            "recent_decisions": [],
        },
    )

    assert resp.status_code == 409
    assert "retrain" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Requirement 7.5: retrain happy path -> status + per-domain metrics
# ---------------------------------------------------------------------------


def test_retrain_happy_path_returns_metrics(trained_ctx):
    """Retrain returns a success status and per-domain accuracy/macro_f1/brier."""
    resp = trained_ctx["retrain"]

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"retrained", "completed"}

    metrics = body["metrics"]
    assert metrics, "expected non-empty per-domain metrics on a successful retrain"

    for domain, domain_metrics in metrics.items():
        assert domain in {"focus", "task", "purchase"}
        for key in ("baseline", "sequence", "winner"):
            assert key in domain_metrics, f"missing {key} for domain {domain}"


# ---------------------------------------------------------------------------
# Requirements 7.1 / 7.3: predict happy path -> valid response shape + invariants
# ---------------------------------------------------------------------------


def test_predict_valid_response_shape(trained_ctx):
    """After retrain, predict returns a valid distribution and consistent confidence."""
    client = trained_ctx["client"]
    resp = client.post(
        "/predict_next_decision",
        json={
            "user_id": trained_ctx["user_id"],
            "context": FOCUS_CONTEXT,
            "recent_decisions": ["pomodoro", "flow_state", "pomodoro"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()

    # Req 7.1: required fields present.
    for key in ("predicted_decision", "confidence", "class_probs", "drift_status"):
        assert key in body

    # Req 7.3: predicted_decision belongs to the requested domain option set.
    assert body["predicted_decision"] in FOCUS_OPTIONS

    # Req 7.2/7.3: class_probs is a valid, non-negative distribution summing to ~1.
    class_probs = body["class_probs"]
    assert set(class_probs) == FOCUS_OPTIONS
    assert all(p >= 0.0 for p in class_probs.values())
    assert math.isclose(sum(class_probs.values()), 1.0, abs_tol=1e-3)

    # Req 7.3: confidence is in [0, 1] and equals max(class_probs).
    confidence = body["confidence"]
    assert 0.0 <= confidence <= 1.0
    assert math.isclose(confidence, max(class_probs.values()), abs_tol=1e-3)

    # Req 7.1: drift_status is present and well-formed.
    drift_status = body["drift_status"]
    assert "drift" in drift_status
    assert isinstance(drift_status["drift"], bool)


# ---------------------------------------------------------------------------
# Requirement 7.7: insufficient data -> HTTP 200 skipped / insufficient_data
# ---------------------------------------------------------------------------


def test_retrain_insufficient_data_is_skipped(fresh_client):
    """Retrain on an empty store skips with HTTP 200 and reason insufficient_data."""
    resp = fresh_client.post("/retrain", json={"user_id": "user_with_no_records"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "skipped"
    assert body["reason"] == "insufficient_data"
