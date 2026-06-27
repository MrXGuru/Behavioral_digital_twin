"""Integration tests for the Phase 3 exact-shape ``{user_id}`` endpoints.

Exercises the four frontend-facing endpoints plus the consolidated dashboard
through :class:`fastapi.testclient.TestClient`, validating each response against
the frozen Model 5 pydantic schemas and the documented edge-case behavior:

* Dashboard schema conformance — ``GET /twin/{user_id}`` matches
  :class:`~api.schemas.DashboardResponse`, every ``hit == (predicted == actual)``,
  and ``accuracy`` is within ``[0, 1]`` (Requirement 14.1).
* Empty-state arrays — ``timeline`` / ``decisions`` / ``driftEvents`` are returned
  as ``[]`` (never ``null``) when there is nothing to model (Requirement 14.2).
* Untrained-domain 409 — ``GET /predict_next_decision/{user_id}`` for a user with no
  trained model responds 409 and leaves stored state unchanged (Requirement 14.4).
* Insufficient-data skip — ``POST /retrain/{user_id}`` with too few records responds
  200 / ``status == "skipped"`` / ``reason == "insufficient_data"`` (Requirement 14.8).
* History / drift-event ordering — ``GET /history/{user_id}`` and
  ``GET /drift_events/{user_id}`` return schema-valid rows ordered ascending
  (Requirements 14.5 / 14.6).

Isolation: ``api.main`` holds a module-level ``service`` (a ``TwinService``) and a
module-level ``adapter`` whose ``_adapter()`` helper rebinds ``adapter.service`` to the
current module-level ``service`` on every call. The endpoints reference the module
global ``service`` at call time, so swapping ``api.main.service`` for a ``TwinService``
backed by a temp SQLite store fully isolates these tests from the real DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.schemas import DashboardResponse, DecisionRow, DriftEvent
from data.decision_store import DecisionStore
from data.schema import DecisionRecord, day_type, time_of_day
from api.service import TwinService

TRAINED_USER = "trained_user"
EMPTY_USER = "empty_user"
NO_DATA_USER = "fresh_user"


def _make_store(tmp_path) -> DecisionStore:
    """Return a DecisionStore backed by an isolated temp SQLite file."""
    return DecisionStore("sqlite", str(tmp_path / "decisions.db"))


def _untrainable_records(user_id: str, n: int = 60) -> list[DecisionRecord]:
    """Build ``n`` schema-shaped records in a domain the service does not model.

    The records satisfy ``ensure_user``'s ``min_records`` floor (so no synthetic data
    is seeded) yet contribute zero trainable rows for any modeled domain, producing a
    genuine empty-state response (no models trained -> empty timeline/decisions/drift).
    """
    base = datetime(2024, 1, 1, 9, 0)
    records: list[DecisionRecord] = []
    for i in range(n):
        ts = base + timedelta(hours=i)
        records.append(DecisionRecord(stress_level="low", 
            user_id=user_id, timestamp=ts, domain="misc",
            location="home", weather="clear",
            day_type=day_type(ts).value, time_of_day=time_of_day(ts).value,
            mood_energy=0.5, decision_made="noop", outcome="ok",
        ))
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_client(monkeypatch, tmp_path):
    """TestClient over an isolated service whose store starts empty."""
    service = TwinService(store=_make_store(tmp_path))
    monkeypatch.setattr(main_module, "service", service)
    client = TestClient(main_module.app)
    client.service = service  # type: ignore[attr-defined]
    return client


@pytest.fixture(scope="module")
def trained_client(tmp_path_factory):
    """Module-scoped TestClient over an isolated, trained service.

    Warms the cache once via ``GET /twin`` (which seeds synthetic data and trains the
    per-domain winning models) so the trained-user reads below are fast and share one
    trained service. The module-level ``api.main.service`` is restored afterwards.
    """
    tmp_path = tmp_path_factory.mktemp("trained")
    service = TwinService(store=_make_store(tmp_path))
    original = main_module.service
    main_module.service = service
    try:
        client = TestClient(main_module.app)
        client.service = service  # type: ignore[attr-defined]
        service.ensure_user(TRAINED_USER)
        warmup = client.get(f"/twin/{TRAINED_USER}")
        assert warmup.status_code == 200, warmup.text
        yield client
    finally:
        main_module.service = original


# ---------------------------------------------------------------------------
# 1. Dashboard schema conformance (Requirement 14.1)
# ---------------------------------------------------------------------------


def test_twin_dashboard_matches_schema(trained_client):
    resp = trained_client.get(f"/twin/{TRAINED_USER}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Validates against the frozen Model 5 shape (raises if any field/bound is wrong).
    dashboard = DashboardResponse(**body)

    assert 0.0 <= dashboard.accuracy <= 1.0
    assert isinstance(dashboard.timeline, list)
    assert isinstance(dashboard.decisions, list)
    assert isinstance(dashboard.driftEvents, list)
    # A seeded/trained user produces at least one held-out decision row.
    assert dashboard.decisions, "expected a trained user to have decision rows"

    for d in dashboard.decisions:
        assert d.hit == (d.predicted == d.actual)
        assert 0.0 <= d.confidence <= 1.0
    for t in dashboard.timeline:
        assert 0.0 <= t.confidence <= 1.0


# ---------------------------------------------------------------------------
# 2. Empty-state arrays (Requirement 14.2)
# ---------------------------------------------------------------------------


def test_empty_state_returns_empty_arrays_not_null(empty_client):
    # Seed records the service cannot model so no synthetic data is generated and no
    # domain is trained -> the consolidated response must expose empty arrays.
    empty_client.service.store.append(_untrainable_records(EMPTY_USER))

    twin = empty_client.get(f"/twin/{EMPTY_USER}")
    assert twin.status_code == 200, twin.text
    dashboard = DashboardResponse(**twin.json())
    assert dashboard.timeline == []
    assert len(dashboard.decisions) == 60
    assert dashboard.decisions[0].predicted == "Learning..."
    assert dashboard.driftEvents == []

    history = empty_client.get(f"/history/{EMPTY_USER}")
    assert history.status_code == 200, history.text
    assert len(history.json()) == 60

    drift = empty_client.get(f"/drift_events/{EMPTY_USER}")
    assert drift.status_code == 200, drift.text
    assert drift.json() == []  # list, never null


# ---------------------------------------------------------------------------
# 3. Untrained-domain -> 409 without mutating state (Requirement 14.4)
# ---------------------------------------------------------------------------


def test_predict_untrained_domain_returns_409_without_mutation(empty_client):
    store = empty_client.service.store
    count_before = store.count(user_id=NO_DATA_USER)

    resp = empty_client.get(f"/predict_next_decision/{NO_DATA_USER}", params={"domain": "focus"})

    assert resp.status_code == 409, resp.text
    assert "retrain" in resp.json()["detail"].lower()
    # Non-mutating predict path: stored state is unchanged (no seeding/training).
    assert store.count(user_id=NO_DATA_USER) == count_before == 0


# ---------------------------------------------------------------------------
# 4. Insufficient-data retrain skip (Requirement 14.8)
# ---------------------------------------------------------------------------


def test_retrain_insufficient_data_is_skipped(empty_client):
    store = empty_client.service.store
    # An untouched temp store has zero records for this user -> below the split floor.
    assert store.count(user_id=NO_DATA_USER) == 0

    resp = empty_client.post(f"/retrain/{NO_DATA_USER}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "skipped"
    assert body["reason"] == "insufficient_data"
    # Skipping must not seed/train new data.
    assert store.count(user_id=NO_DATA_USER) == 0


# ---------------------------------------------------------------------------
# 5. History / drift-event shape + ordering after training (Req 14.5 / 14.6)
# ---------------------------------------------------------------------------


def test_history_rows_conform_and_sorted(trained_client):
    resp = trained_client.get(f"/history/{TRAINED_USER}")
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert isinstance(rows, list)
    assert rows, "expected a trained user to have history rows"

    parsed = [DecisionRow(**row) for row in rows]
    for d in parsed:
        assert d.hit == (d.predicted == d.actual)
        assert 0.0 <= d.confidence <= 1.0

    timestamps = [d.timestamp for d in parsed]
    assert timestamps == sorted(timestamps), "decisions must be ascending by timestamp"


def test_drift_events_conform_and_sorted(trained_client):
    resp = trained_client.get(f"/drift_events/{TRAINED_USER}")
    assert resp.status_code == 200, resp.text
    events = resp.json()
    assert isinstance(events, list)

    parsed = [DriftEvent(**e) for e in events]
    dates = [e.date for e in parsed]
    assert dates == sorted(dates), "drift events must be ascending by date"
