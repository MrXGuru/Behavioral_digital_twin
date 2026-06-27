"""Pydantic models for the Behavioral Digital Twin HTTP API.

This module holds two related but distinct families of models:

1. **Requirement 7 prediction-engine models** — request/response shapes for the
   internal prediction API (``POST /predict_next_decision``, ``GET /user_profile/{id}``,
   ``POST /retrain``). These mirror the internal data models ``Context`` (Model 2),
   ``PredictionResult`` (Model 4), and ``DriftStatus`` (Component 6) and enforce the
   validation rules from Requirement 7 (``confidence`` in ``[0, 1]``, non-negative
   ``class_probs`` that sum to ~1, ``confidence == max(class_probs)``).
2. **Phase 3 frontend response models** — the exact consolidated JSON shapes the React
   dashboard consumes (``/twin``, ``/history``, ``/drift_events`` and the GET-based
   predict/retrain endpoints). These are retained unchanged in behavior.

Models are written for Pydantic v2.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Requirement 7: Prediction Engine API models
# ---------------------------------------------------------------------------

#: Tolerance used when checking that a probability distribution sums to ~1.
_PROB_SUM_TOL: float = 1e-3


class ContextModel(BaseModel):
    """Prediction-time context (mirrors internal Model 2: ``Context``).

    Describes the situation in which the next decision is made. ``domain`` selects
    which model/label space is used; the categorical fields are encoded by the
    feature pipeline (unseen values map to the explicit ``"<UNK>"`` bucket).

    Attributes:
        domain: One of ``focus`` / ``task`` / ``purchase``; selects the model.
        location: Categorical context (e.g. ``home``, ``work``).
        weather: Categorical context (e.g. ``clear``, ``rain``).
        day_type: ``weekday`` / ``weekend``.
        time_of_day: ``morning`` / ``afternoon`` / ``evening`` / ``night``.
        mood_energy: Proxy for mood/energy in the range ``[0, 1]``.
        stress_level: Proxy for stress in ``low`` / ``medium`` / ``high``.
        timestamp: Optional decision time; defaults to "now" downstream when omitted.
    """

    domain: str
    location: str
    weather: str
    day_type: str
    time_of_day: str
    mood_energy: float = Field(..., ge=0.0, le=1.0)
    stress_level: str = "medium"
    timestamp: datetime | None = None


class PredictRequest(BaseModel):
    """Request body for ``POST /predict_next_decision`` (Requirement 7.1).

    Attributes:
        user_id: Identifier of the user the prediction is for.
        context: Prediction-time :class:`ContextModel`.
        recent_decisions: Time-ordered recent decision labels (may be empty for a
            cold start; the sequence is left-padded with ``PAD`` downstream).
    """

    user_id: str
    context: ContextModel
    recent_decisions: list[str] = Field(default_factory=list)


class DriftStatusModel(BaseModel):
    """Drift status mirroring :class:`api.drift.DriftStatus` (Component 6).

    Attributes:
        drift: True iff at least ``window`` labeled predictions exist and the rolling
            accuracy is below the configured threshold.
        score: Drift magnitude, ``1.0 - window_acc`` (``0.0`` when no observations).
        window_acc: Rolling accuracy over the most recent ``window`` labeled
            predictions, or ``None`` when no labeled predictions exist.
    """

    drift: bool
    score: float = Field(..., ge=0.0)
    window_acc: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response for ``POST /predict_next_decision`` (mirrors Model 4 ``PredictionResult``).

    Validation rules (Requirement 7.2 / 7.3):
        * ``confidence`` is within ``[0, 1]``.
        * every value in ``class_probs`` is non-negative.
        * ``class_probs`` sum to approximately ``1.0``.
        * ``confidence`` equals the maximum value in ``class_probs``.

    Attributes:
        predicted_decision: The chosen option; belongs to the domain's option set.
        confidence: Maximum class probability (calibrated), in ``[0, 1]``.
        class_probs: Mapping of option label to probability (a valid distribution).
        drift_status: Current :class:`DriftStatusModel`.
        model_name: Winning model family (``"baseline"`` | ``"sequence"``); optional
            to keep the response usable when the engine does not surface it.
    """

    predicted_decision: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probs: dict[str, float]
    drift_status: DriftStatusModel
    model_name: str | None = None

    @field_validator("class_probs")
    @classmethod
    def _probs_valid_distribution(cls, value: dict[str, float]) -> dict[str, float]:
        """Ensure ``class_probs`` are non-negative and sum to approximately 1."""
        if any(p < 0.0 for p in value.values()):
            raise ValueError("class_probs must all be non-negative")
        if value:
            total = math.fsum(value.values())
            if not math.isclose(total, 1.0, abs_tol=_PROB_SUM_TOL):
                raise ValueError(
                    f"class_probs must sum to ~1.0 (got {total:.6f})"
                )
        return value

    @model_validator(mode="after")
    def _confidence_is_max_prob(self) -> "PredictionResponse":
        """Ensure ``confidence`` equals the maximum value in ``class_probs``."""
        if self.class_probs:
            max_prob = max(self.class_probs.values())
            if not math.isclose(self.confidence, max_prob, abs_tol=_PROB_SUM_TOL):
                raise ValueError(
                    "confidence must equal max(class_probs.values()) "
                    f"(confidence={self.confidence}, max={max_prob})"
                )
        return self


class EmbeddingSummaryModel(BaseModel):
    """Compact JSON-friendly summary of a user embedding.

    Mirrors :meth:`personalization.profile_store.UserProfile.embedding_summary`.

    Attributes:
        dim: Length of the embedding vector.
        norm: Euclidean norm of the embedding vector.
    """

    dim: int = Field(..., ge=0)
    norm: float = Field(..., ge=0.0)


class UserProfileResponse(BaseModel):
    """Response for ``GET /user_profile/{id}`` (Requirement 7.4).

    Attributes:
        user_id: Identifier of the user.
        decision_counts: Per-option decision frequencies (option -> count).
        embedding_summary: Compact embedding summary.
        last_updated: Timestamp of the most recent record folded into the profile,
            or ``None`` if the profile has never been updated.
    """

    user_id: str
    decision_counts: dict[str, int] = Field(default_factory=dict)
    embedding_summary: EmbeddingSummaryModel
    last_updated: datetime | None = None


class RetrainRequest(BaseModel):
    """Request body for ``POST /retrain`` (Requirement 7.5).

    Attributes:
        user_id: Optional user to retrain; ``None`` retrains across the default scope.
        since: Optional lower bound on record timestamps used for retraining.
    """

    user_id: str | None = None
    since: datetime | None = None


class LogDecisionRequest(BaseModel):
    """Request body for ``POST /decisions/{user_id}`` — log a real decision.

    Only ``domain`` and ``decision_made`` are required. Context fields default
    to sensible values so quick logging is frictionless; ``timestamp``,
    ``time_of_day``, and ``day_type`` are auto-derived server-side from the
    current time.

    Attributes:
        domain: One of ``focus`` / ``task`` / ``purchase``.
        decision_made: The option the user actually chose (must belong to the
            domain's option set).
        location: Categorical context (default ``"home"``).
        weather: Categorical context (default ``"clear"``).
        mood_energy: Mood/energy proxy in ``[0, 1]`` (default ``0.5``).
        stress_level: Stress proxy in ``low`` / ``medium`` / ``high`` (default ``medium``).
    """

    domain: str
    decision_made: str
    location: str = "home"
    weather: str = "clear"
    mood_energy: float = Field(default=0.5, ge=0.0, le=1.0)
    stress_level: str = "medium"


class MetricsModel(BaseModel):
    """Evaluation metrics (mirrors :class:`models.evaluate.Metrics`).

    Attributes:
        accuracy: Held-out accuracy.
        macro_f1: Macro-averaged F1 score.
        brier: Multiclass Brier score (lower is better calibrated).
        n: Number of validation observations the metrics were computed over.
    """

    accuracy: float = Field(..., ge=0.0, le=1.0)
    macro_f1: float = Field(..., ge=0.0, le=1.0)
    brier: float = Field(..., ge=0.0)
    n: int | None = Field(default=None, ge=0)


class RetrainResponse(BaseModel):
    """Response for ``POST /retrain`` (Requirement 7.5 / 7.7).

    On a successful retrain ``status`` is ``"retrained"`` and ``metrics`` carries the
    per-domain evaluation metrics. When there are too few records for a valid temporal
    split (Requirement 7.7) ``status`` is ``"skipped"`` and ``reason`` is
    ``"insufficient_data"``.

    Attributes:
        status: Outcome of the retrain (e.g. ``"retrained"`` | ``"skipped"``).
        metrics: Per-domain evaluation metrics, present on a successful retrain.
        reason: Explanation when a retrain is skipped (e.g. ``"insufficient_data"``).
    """

    status: str
    metrics: dict[str, Any] | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Phase 3: frontend-facing consolidated response models (React contract)
#
# Retained from the existing implementation. ``NextDecisionResponse`` and
# ``RetrainSummaryResponse`` are the former ``PredictionResponse`` / ``RetrainResponse``
# (renamed so the Requirement 7 names above can carry the PredictionResult shape).
#
# Requirement 14.1 — the consolidated dashboard payload (internal "Model 5") has the
# exact shape::
#
#     {
#       "accuracy": <float in [0, 1]>,
#       "lastSynced": <ISO-8601 string>,
#       "timeline": [{"date", "actual", "predicted", "confidence" in [0, 1]}],
#       "decisions": [{"id", "timestamp", "domain", "predicted", "actual",
#                      "hit", "confidence" in [0, 1]}],
#       "driftEvents": [{"date", "domain", "note"}]
#     }
#
# The models below enforce that shape and the documented numeric bounds. Collections
# default to empty lists so the empty/cold-start state (Requirement 14.2) validates
# cleanly when the adapter emits no rows.
# ---------------------------------------------------------------------------


def _validate_iso8601(value: str) -> str:
    """Validate that ``value`` parses as an ISO-8601 timestamp.

    Accepts a trailing ``Z`` (UTC designator) by normalising it to ``+00:00`` before
    delegating to :meth:`datetime.fromisoformat`. Kept intentionally light: it only
    asserts parseability and returns the original string unchanged.
    """
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:  # pragma: no cover - exercised via model validation
        raise ValueError(f"lastSynced must be an ISO-8601 timestamp (got {value!r})") from exc
    return value


class TimelinePoint(BaseModel):
    """One point on the accuracy/decision timeline (Requirement 14.1).

    Attributes:
        date: Calendar/period label for the point (string).
        actual: Observed numeric value for the period (e.g. realized accuracy/count).
        predicted: Predicted numeric value for the period.
        confidence: Aggregate confidence for the period, in ``[0, 1]``.
    """

    date: str
    actual: float
    predicted: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class HeatmapPoint(BaseModel):
    """Daily count of positive habits for the contribution graph."""
    date: str
    count: int



class DecisionRow(BaseModel):
    """One row in the recent-decisions table (Requirement 14.1).

    Attributes:
        id: Stable identifier of the decision record.
        timestamp: Decision time as a string (ISO-8601 from the adapter).
        domain: Decision domain (``focus`` / ``task`` / ``purchase``).
        predicted: The model's predicted option label.
        actual: The realized option label.
        hit: True iff ``predicted == actual``.
        confidence: Prediction confidence, in ``[0, 1]``.
    """

    id: int
    timestamp: str
    domain: str
    predicted: str
    actual: str
    hit: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class DriftEvent(BaseModel):
    """One drift event surfaced to the dashboard (Requirement 14.1).

    Attributes:
        date: When the drift event was observed (string).
        domain: Domain the drift event applies to.
        note: Human-readable description of the drift event.
    """

    date: str
    domain: str
    note: str


class DataMaturity(BaseModel):
    """Data maturity state for a user (Phase 2 requirement).

    Below ``threshold`` decisions logged the twin shows "still learning"
    rather than potentially unreliable polished predictions.

    Attributes:
        count: Number of decisions logged for this user.
        threshold: Minimum for reliable predictions (15 per the spec).
        status: ``"learning"`` if count < threshold, else ``"ready"``.
        message: Human-readable explanation.
    """

    count: int = Field(..., ge=0)
    threshold: int = Field(..., ge=1)
    status: str  # "learning" | "ready"
    message: str


class DashboardResponse(BaseModel):
    """Consolidated dashboard payload — internal "Model 5" (Requirement 14.1).

    Attributes:
        accuracy: Overall accuracy, in ``[0, 1]``.
        lastSynced: ISO-8601 timestamp of the last data sync.
        timeline: Per-period :class:`TimelinePoint` entries (defaults to ``[]``).
        decisions: Recent :class:`DecisionRow` entries (defaults to ``[]``).
        driftEvents: Surfaced :class:`DriftEvent` entries (defaults to ``[]``).
        data_maturity: Maturity state (count, threshold, status, message).
        heatmap: Daily counts of positive habits.
    """

    accuracy: float = Field(..., ge=0.0, le=1.0)
    lastSynced: str
    timeline: list[TimelinePoint] = Field(default_factory=list)
    decisions: list[DecisionRow] = Field(default_factory=list)
    driftEvents: list[DriftEvent] = Field(default_factory=list)
    data_maturity: DataMaturity | None = None
    heatmap: list[HeatmapPoint] = Field(default_factory=list)

    @field_validator("lastSynced")
    @classmethod
    def _last_synced_is_iso8601(cls, value: str) -> str:
        return _validate_iso8601(value)


class HistoryResponse(BaseModel):
    """History payload for the dashboard (Requirement 14.1).

    Same shape as :class:`DashboardResponse` minus ``driftEvents``.

    Attributes:
        accuracy: Overall accuracy, in ``[0, 1]``.
        lastSynced: ISO-8601 timestamp of the last data sync.
        timeline: Per-period :class:`TimelinePoint` entries (defaults to ``[]``).
        decisions: Recent :class:`DecisionRow` entries (defaults to ``[]``).
    """

    accuracy: float = Field(..., ge=0.0, le=1.0)
    lastSynced: str
    timeline: list[TimelinePoint] = Field(default_factory=list)
    decisions: list[DecisionRow] = Field(default_factory=list)

    @field_validator("lastSynced")
    @classmethod
    def _last_synced_is_iso8601(cls, value: str) -> str:
        return _validate_iso8601(value)


class NextDecisionResponse(BaseModel):
    """Frontend GET ``/predict_next_decision/{user_id}`` shape."""

    user_id: str
    domain: str
    predicted_decision: str
    confidence: float
    class_probs: dict[str, float]
    model_name: str


class PredictNextResponse(BaseModel):
    """Phase 3 ``GET /predict_next_decision/{user_id}`` response (Requirement 14.3).

    The exact, minimal shape the React dashboard consumes, derived from the internal
    prediction result.

    Attributes:
        predicted: The predicted option; a RAW value in the requested domain Option_Set.
        confidence: Prediction confidence, in ``[0, 1]``.
    """

    predicted: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RetrainResultResponse(BaseModel):
    """Phase 3 ``POST /retrain/{user_id}`` response (Requirement 14.7 / 14.8).

    On a successful retrain ``status`` is ``"completed"`` and ``metrics`` carries the
    full per-domain comparison report (baseline metrics, sequence metrics, winner, rationale).
    When there are too few records for a valid temporal split, ``status`` is ``"skipped"`` and
    ``reason`` is ``"insufficient_data"`` (the previously trained artifact is retained).

    Attributes:
        status: Outcome of the retrain (``"completed"`` | ``"skipped"``).
        metrics: Per-domain full comparison report, present on a successful retrain.
        reason: Explanation when a retrain is skipped (e.g. ``"insufficient_data"``).
    """

    model_config = {"arbitrary_types_allowed": True}

    status: str
    metrics: dict[str, Any] | None = None
    reason: str | None = None


class RetrainSummaryResponse(BaseModel):
    """Frontend POST ``/retrain/{user_id}`` shape."""

    status: str
    reason: str | None = None
    metrics: list[dict] | None = None
    lastSynced: str


# ---------------------------------------------------------------------------
# Requirement 16: Digital Twin Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for ``POST /chat/{user_id}`` (Requirement 16.1).

    Attributes:
        question: The natural-language question to ask about the user's twin.
    """

    question: str


class CSVImportResponse(BaseModel):
    """Response for ``POST /decisions/{user_id}/import-csv``.

    Attributes:
        imported: Number of records successfully imported.
        skipped: Number of rows skipped (parse errors, invalid schema, duplicates).
        errors: List of row-level error messages (capped at 20 for readability).
    """

    imported: int = Field(..., ge=0)
    skipped: int = Field(..., ge=0)
    errors: list[str] = Field(default_factory=list)
