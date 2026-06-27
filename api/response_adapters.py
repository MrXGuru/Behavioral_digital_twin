"""ResponseAdapter: reconcile internal twin results into the exact Phase 3 shapes.

The :class:`~api.service.TwinService` already assembles the consolidated Model 5
response (``{accuracy, lastSynced, timeline, decisions, driftEvents}``), orders its
arrays ascending, and computes each decision's ``hit`` flag. This adapter is a thin
shape-conformance layer (Requirement 14.1, 14.2): it delegates to the service and then
*re-asserts* the frontend contract so the React dashboard can consume the data without
defensive parsing.

Specifically, for every response it guarantees:

* ``timeline`` is ordered ascending by ``date``.
* ``decisions`` is ordered ascending by ``timestamp``.
* ``driftEvents`` is ordered ascending by ``date``.
* every ``decisions[i].hit`` equals ``predicted == actual``.
* ``timeline``/``decisions``/``driftEvents`` are ``[]`` (never ``None``) when empty.

The adapter never mutates stored records and never changes the internal
predict/retrain/drift logic — it only normalizes shapes.
"""

from __future__ import annotations

from api.service import TwinService


class ResponseAdapter:
    """Map internal twin results onto the exact frontend-facing response shapes.

    Wraps a single :class:`~api.service.TwinService` instance whose predict/retrain/
    drift logic is reused unchanged; this class only adapts and normalizes the output
    shapes to the frozen Model 5 contract.
    """

    def __init__(self, service: TwinService | None = None) -> None:
        """Create an adapter over ``service`` (a fresh :class:`TwinService` by default)."""
        self.service = service or TwinService()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_decisions(decisions: list[dict] | None) -> list[dict]:
        """Return decisions ascending by ``timestamp`` with ``hit`` recomputed.

        Empty/``None`` input yields ``[]``. ``hit`` is always re-derived as
        ``predicted == actual`` so the frontend invariant holds regardless of how the
        upstream value was produced.
        """
        items = list(decisions or [])
        items.sort(key=lambda d: d["timestamp"])
        for d in items:
            d["hit"] = bool(d["predicted"] == d["actual"])
        return items

    @staticmethod
    def _normalize_timeline(timeline: list[dict] | None) -> list[dict]:
        """Return timeline points ordered ascending by ``date`` (``[]`` when empty)."""
        items = list(timeline or [])
        items.sort(key=lambda t: t["date"])
        return items

    @staticmethod
    def _normalize_drift_events(drift_events: list[dict] | None) -> list[dict]:
        """Return drift events ordered ascending by ``date`` (``[]`` when empty)."""
        items = list(drift_events or [])
        items.sort(key=lambda e: e["date"])
        return items

    # ------------------------------------------------------------------
    def dashboard(self, user_id: str) -> dict:
        """Return the exact, frozen Model 5 consolidated dashboard response.

        Shape: ``{accuracy, lastSynced, timeline, decisions, driftEvents}`` with
        ``timeline`` ascending by ``date``, ``decisions`` ascending by ``timestamp``,
        ``driftEvents`` ascending by ``date``, every decision's ``hit`` equal to
        ``predicted == actual``, and the three arrays guaranteed to be ``[]`` (never
        ``None``) when there is no data (Requirement 14.1, 14.2).
        """
        full = self.service.dashboard(user_id)
        return {
            "accuracy": full["accuracy"],
            "lastSynced": full["lastSynced"],
            "timeline": self._normalize_timeline(full.get("timeline")),
            "decisions": self._normalize_decisions(full.get("decisions")),
            "driftEvents": self._normalize_drift_events(full.get("driftEvents")),
        }

    def history(self, user_id: str) -> dict:
        """Return ``{accuracy, lastSynced, timeline, decisions}`` (no ``driftEvents``).

        ``decisions`` is ordered ascending by ``timestamp`` with ``hit`` recomputed, and
        both arrays are ``[]`` (never ``None``) when the user has no data
        (Requirement 14.2, 14.5).
        """
        hist = self.service.history(user_id)
        return {
            "accuracy": hist["accuracy"],
            "lastSynced": hist["lastSynced"],
            "timeline": self._normalize_timeline(hist.get("timeline")),
            "decisions": self._normalize_decisions(hist.get("decisions")),
        }

    def drift_events(self, user_id: str) -> list[dict]:
        """Return the ``driftEvents`` array ascending by ``date`` (``[]`` when none).

        Conforms to the Model 5 drift-event shape ``{date, domain, note}``
        (Requirement 14.6).
        """
        return self._normalize_drift_events(self.service.drift_events(user_id))

    def predict_next(self, user_id: str, domain: str = "focus") -> dict:
        """Return ``{predicted, confidence}`` for the user's next decision.

        Routes through the non-mutating :meth:`TwinService.predict_next_existing`, so an
        untrained ``(user_id, domain)`` raises :class:`~api.service.ModelNotTrainedError`
        (translated to HTTP 409 by the endpoint) *without* seeding data or training a
        model (Requirement 14.4). ``predicted`` is the RAW option value from the domain
        Option_Set and ``confidence`` is passed through as a float in ``[0, 1]``
        (Requirement 14.3).
        """
        result = self.service.predict_next_existing(user_id, domain)
        return {
            "predicted": result["predicted_decision"],
            "confidence": float(result["confidence"]),
        }

    def retrain(self, user_id: str) -> dict:
        """Return the Phase 3 retrain result ``{status, metrics, reason?}``.

        Reuses the existing retrain logic (:meth:`TwinService.retrain_models`) and maps
        its outcome onto the frozen Phase 3 contract: a successful retrain becomes
        ``status == "completed"`` with a per-domain ``metrics`` object carrying Accuracy,
        macro-F1, and Brier (Requirement 14.7); too few records for a valid temporal
        split yields ``status == "skipped"`` / ``reason == "insufficient_data"`` while the
        previously trained artifact is left untouched (Requirement 14.8).
        """
        result = self.service.retrain_models(user_id)
        status = "completed" if result.get("status") == "retrained" else result["status"]
        out: dict = {"status": status}
        if result.get("metrics") is not None:
            out["metrics"] = result["metrics"]
        if result.get("reason") is not None:
            out["reason"] = result["reason"]
        return out
