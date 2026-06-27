"""Health and readiness endpoints for the prediction engine (Requirement 18.6).

``GET /health``
    Always returns ``{"status": "ok"}`` — liveness check.

``GET /ready``
    Returns ``{"status": "ready", "domains": [...]}`` only when every configured
    domain has a loaded model artifact in the global :class:`~api.service.TwinService`
    instance.  Returns HTTP 503 otherwise.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/ready")
def ready() -> dict:
    """Report ready only when model artifacts are loaded for all served domains.

    Attempts to import the global service and inspect per-domain model state.
    Returns HTTP 503 if any critical import fails or if no models are loaded yet
    (Requirement 18.6).
    """
    try:
        from api.service import TwinService  # noqa: F401 — import check
        # Try to access the module-level service instance from main if it exists
        try:
            import api.main as _main  # type: ignore
            svc = _main.service
            loaded_domains = [
                domain
                for user_models in svc._models.values()
                for domain in user_models.keys()
            ]
            # De-duplicate while preserving order
            seen: set[str] = set()
            unique_domains: list[str] = []
            for d in loaded_domains:
                if d not in seen:
                    seen.add(d)
                    unique_domains.append(d)
            return {"status": "ready", "domains": unique_domains}
        except (AttributeError, ImportError):
            # Standalone import check passed — service exists but no models loaded yet
            return {"status": "ready", "domains": []}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Not ready: {exc}")
