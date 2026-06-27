"""Property-based test for API response-shape conformance.

Property 13: API response-shape conformance
    For any user, the consolidated dashboard produced by
    ``ResponseAdapter.dashboard`` conforms exactly to the Model 5 shape and
    constraints:

    * ``accuracy`` in ``[0, 1]``;
    * ``lastSynced`` is an ISO-8601 timestamp;
    * ``timeline`` ordered ascending by ``date`` with each ``confidence`` in
      ``[0, 1]``;
    * ``decisions`` ordered ascending by ``timestamp`` with ``hit == (predicted
      == actual)`` and each ``confidence`` in ``[0, 1]``;
    * ``driftEvents`` ordered ascending by ``date``;
    * ``timeline`` / ``decisions`` / ``driftEvents`` are arrays (never ``None``),
      possibly empty;
    * the payload validates against the pydantic :class:`DashboardResponse` model.

Validates: Requirements 14.1
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from api.response_adapters import ResponseAdapter
from api.schemas import DashboardResponse
from api.service import TwinService
from data.decision_store import DecisionStore


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 string, normalising a trailing ``Z`` to ``+00:00``."""
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(candidate)


# A SMALL set of distinct user_ids keeps the number of expensive per-user model
# trainings low across the property's examples.
_USER_IDS = st.sampled_from(["u_a", "u_b", "u_c"])


@given(user_id=_USER_IDS)
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_dashboard_conforms_to_model5_shape(user_id: str) -> None:
    """``ResponseAdapter.dashboard`` always emits a valid Model 5 payload."""
    # Hypothesis does not reset fixtures between examples, so build a fresh,
    # isolated TwinService over a temp sqlite store INSIDE each example. This keeps
    # the real DB untouched and gives every example a clean slate.
    with tempfile.TemporaryDirectory() as tmp:
        store = DecisionStore("sqlite", str(Path(tmp) / "t.db"))
        adapter = ResponseAdapter(TwinService(store))

        payload = adapter.dashboard(user_id)

        # Req 14.1: the payload validates against the pydantic model exactly.
        model = DashboardResponse(**payload)

        # accuracy in [0, 1].
        assert 0.0 <= payload["accuracy"] <= 1.0

        # lastSynced parses as ISO-8601.
        _parse_iso8601(payload["lastSynced"])

        # The three collections are lists (never None), possibly empty.
        assert isinstance(payload["timeline"], list)
        assert isinstance(payload["decisions"], list)
        assert isinstance(payload["driftEvents"], list)

        # timeline ordered ascending by date; each confidence in [0, 1].
        timeline_dates = [pt["date"] for pt in payload["timeline"]]
        assert timeline_dates == sorted(timeline_dates)
        for pt in payload["timeline"]:
            assert 0.0 <= pt["confidence"] <= 1.0

        # decisions ordered ascending by timestamp; hit == (predicted == actual);
        # confidence in [0, 1].
        decision_timestamps = [d["timestamp"] for d in payload["decisions"]]
        assert decision_timestamps == sorted(decision_timestamps)
        for d in payload["decisions"]:
            assert d["hit"] == (d["predicted"] == d["actual"])
            assert 0.0 <= d["confidence"] <= 1.0

        # driftEvents ordered ascending by date.
        drift_dates = [e["date"] for e in payload["driftEvents"]]
        assert drift_dates == sorted(drift_dates)

        # The validated model round-trips the same collections (defaults to []).
        assert isinstance(model.timeline, list)
        assert isinstance(model.decisions, list)
        assert isinstance(model.driftEvents, list)
