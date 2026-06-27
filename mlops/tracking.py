"""MLflow experiment tracking stub.

Uses MLflow when available; falls back to structured logging only.
"""
from __future__ import annotations
import logging
import uuid

logger = logging.getLogger(__name__)


def log_run(domain: str, metrics: dict, params: dict | None = None) -> str:
    """Log a training run and return a run_id.

    Attempts to use MLflow if installed; falls back to structured logging.
    """
    run_id = str(uuid.uuid4())[:8]

    try:
        import mlflow  # type: ignore

        with mlflow.start_run(run_name=f"{domain}_{run_id}"):
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    except ImportError:
        pass  # MLflow not installed — structured log only
    except Exception as exc:
        logger.warning("mlflow_log_failed run_id=%s exc=%s", run_id, exc)

    logger.info(
        "run_logged domain=%s run_id=%s metrics=%s params=%s",
        domain,
        run_id,
        metrics,
        params or {},
    )
    return run_id
