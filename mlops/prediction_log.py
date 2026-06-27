"""Per-prediction logging for the Behavioral Digital Twin MLOps platform."""
from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class PredictionLogEntry:
    log_id: str
    user_id: str
    domain: str
    prediction: str
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: datetime
    actual: str | None = None


class PredictionLogger:
    """Logs one entry per served prediction; allows attaching actual outcomes."""

    def __init__(self) -> None:
        self._entries: dict[str, PredictionLogEntry] = {}

    def log(
        self,
        *,
        user_id: str,
        domain: str,
        prediction: str,
        confidence: float,
        model_version: str,
        latency_ms: float,
        timestamp: datetime | None = None,
    ) -> str:
        """Record one prediction. Returns the log_id for later outcome attachment."""
        log_id = str(uuid.uuid4())
        entry = PredictionLogEntry(
            log_id=log_id,
            user_id=user_id,
            domain=domain,
            prediction=prediction,
            confidence=confidence,
            model_version=model_version,
            latency_ms=latency_ms,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self._entries[log_id] = entry
        try:
            logger.info(
                "prediction_logged",
                extra={
                    "log_id": log_id,
                    "domain": domain,
                    "prediction": prediction,
                    "confidence": confidence,
                },
            )
        except Exception:
            pass  # logging failures must not block the response

        return log_id

    def attach_actual(self, log_id: str, actual: str) -> None:
        """Attach the actual outcome once known (at most once per log_id)."""
        if log_id in self._entries and self._entries[log_id].actual is None:
            self._entries[log_id].actual = actual

    def entries(self) -> list[PredictionLogEntry]:
        return list(self._entries.values())
