"""Prometheus-style metrics exporters.

Uses ``prometheus_client`` when available; otherwise all metric operations are no-ops
so the service runs identically with or without the optional dependency.

Public API
----------
``record_prediction(latency_ms, domain, success=True)``
    Increment throughput and error counters; observe latency.

``record_error(endpoint)``
    Increment the error counter for *endpoint*.

``time_prediction(domain)``
    Context-manager that measures wall-clock time and calls :func:`record_prediction`.

``get_metrics_text()``
    Return Prometheus text-format metrics (or a JSON stub when prometheus_client
    is not installed).
"""
from __future__ import annotations
import json
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST  # type: ignore

    _HAS_PROMETHEUS = True

    PREDICTION_COUNT = Counter(
        "bdt_predictions_total",
        "Total predictions served",
        ["domain"],
    )
    PREDICTION_LATENCY = Histogram(
        "bdt_prediction_latency_seconds",
        "Prediction latency in seconds",
        ["domain"],
    )
    ERROR_COUNT = Counter(
        "bdt_errors_total",
        "Total errors",
        ["endpoint"],
    )
    THROUGHPUT_COUNT = Counter(
        "bdt_throughput_total",
        "Total successful predictions (throughput)",
        ["domain"],
    )
    ACTIVE_MODELS = Gauge(
        "bdt_active_models",
        "Number of active trained models",
    )

except ImportError:
    _HAS_PROMETHEUS = False

# In-memory stubs used when prometheus_client is absent
_stub_counts: dict[str, int] = {}
_stub_errors: dict[str, int] = {}
_stub_latencies: list[float] = []


def record_prediction(latency_ms: float, domain: str, success: bool = True) -> None:
    """Increment the prediction counter and observe latency for *domain*.

    :param latency_ms: Elapsed time in milliseconds.
    :param domain: Decision domain (e.g. ``"route"``).
    :param success: When False the error counter is incremented instead.
    """
    if _HAS_PROMETHEUS:
        try:
            PREDICTION_COUNT.labels(domain=domain).inc()
            PREDICTION_LATENCY.labels(domain=domain).observe(latency_ms / 1000)
            if success:
                THROUGHPUT_COUNT.labels(domain=domain).inc()
        except Exception:
            pass
    else:
        # Stub accounting
        key = f"predictions:{domain}"
        _stub_counts[key] = _stub_counts.get(key, 0) + 1
        _stub_latencies.append(latency_ms)
        if success:
            tkey = f"throughput:{domain}"
            _stub_counts[tkey] = _stub_counts.get(tkey, 0) + 1


def record_error(endpoint: str) -> None:
    """Increment the error counter for *endpoint*."""
    if _HAS_PROMETHEUS:
        try:
            ERROR_COUNT.labels(endpoint=endpoint).inc()
        except Exception:
            pass
    else:
        _stub_errors[endpoint] = _stub_errors.get(endpoint, 0) + 1


@contextmanager
def time_prediction(domain: str):
    """Context manager that measures wall-clock time and calls :func:`record_prediction`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_prediction(elapsed_ms, domain, success=True)


def get_metrics_text() -> str:
    """Return metrics in Prometheus text format, or a JSON stub if the library is absent."""
    if _HAS_PROMETHEUS:
        try:
            return generate_latest().decode("utf-8")
        except Exception as exc:
            logger.warning("prometheus_generate_failed exc=%s", exc)
    # JSON fallback — same shape, machine-readable
    return json.dumps({
        "predictions": _stub_counts,
        "errors": _stub_errors,
        "latencies_ms": _stub_latencies[-100:],  # last 100 for brevity
    }, indent=2)
