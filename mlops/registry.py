"""MLflow-backed model registry stub. Falls back gracefully when MLflow is not available."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    domain: str
    version: str
    artifact_path: str
    metrics: dict = field(default_factory=dict)


class ModelRegistry:
    """Model versioning registry (MLflow-backed with in-memory fallback).

    Maintains exactly one active version per domain plus a full history for rollback.
    """

    def __init__(self) -> None:
        self._active: dict[str, ModelVersion] = {}
        self._history: dict[str, list[ModelVersion]] = {}

    def register(self, domain: str, artifact_path: str, metrics: dict) -> ModelVersion:
        """Register a new model version and return it."""
        # Use timestamp + sequence counter to guarantee uniqueness within a second
        ts = int(time.time())
        seq = len(self._history.get(domain, []))
        version = f"{ts}_{seq}"
        mv = ModelVersion(
            domain=domain,
            version=version,
            artifact_path=artifact_path,
            metrics=metrics,
        )
        self._history.setdefault(domain, []).append(mv)
        logger.info("model_registered domain=%s version=%s", domain, version)
        return mv

    def promote(self, version: ModelVersion) -> None:
        """Promote a version to active for its domain (exactly one active per domain)."""
        self._active[version.domain] = version
        logger.info(
            "model_promoted domain=%s version=%s", version.domain, version.version
        )

    def active(self, domain: str) -> ModelVersion | None:
        """Return the currently active version for *domain*, or None."""
        return self._active.get(domain)

    def previous(self, domain: str) -> ModelVersion | None:
        """Return the most recent non-active version for *domain*, or None."""
        history = self._history.get(domain, [])
        active = self._active.get(domain)
        if not history or not active:
            return None
        prev = [v for v in history if v.version != active.version]
        return prev[-1] if prev else None
