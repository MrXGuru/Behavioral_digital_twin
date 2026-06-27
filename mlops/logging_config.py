"""Structured logging configuration for the Behavioral Digital Twin backend.

Usage::

    from mlops.logging_config import setup_logging, get_logger

    setup_logging()          # call once at startup (idempotent)
    log = get_logger(__name__)
    log.info("prediction served", extra={"domain": "route"})
"""
from __future__ import annotations
import logging
import json
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        # Include any extra key/value pairs passed via the `extra` kwarg
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in (
                "msg", "args", "exc_info", "exc_text", "stack_info",
                "lineno", "filename", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            ):
                log_obj[key] = value
        return json.dumps(log_obj)

class _PrettyTerminalFormatter(logging.Formatter):
    """Emit logs in a human-readable, beautifully formatted string for local dev."""
    
    # Terminal color codes
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m',
        'DIM': '\033[2m',
        'BOLD': '\033[1m',
        'CYAN': '\033[96m'
    }

    def format(self, record: logging.LogRecord) -> str:
        # Format the core message
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        dim = self.COLORS['DIM']
        bold = self.COLORS['BOLD']
        cyan = self.COLORS['CYAN']
        
        # Build the extras string
        extras = []
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in (
                "msg", "args", "exc_info", "exc_text", "stack_info",
                "lineno", "filename", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            ):
                extras.append(f"{key}={value}")
        
        extra_str = f" {dim}[{', '.join(extras)}]{reset}" if extras else ""
        
        # Format standard ML prediction logs explicitly for best readability
        if record.msg == "prediction_logged" and "domain" in record.__dict__ and "prediction" in record.__dict__:
            domain = record.__dict__.get("domain")
            pred = record.__dict__.get("prediction")
            conf = record.__dict__.get("confidence", 0) * 100
            msg = f"✨ Predicted {bold}{cyan}{domain}{reset} ➜ {bold}{pred}{reset} ({conf:.1f}% confidence)"
            return f"{dim}[{ts}]{reset} {color}{record.levelname:<8}{reset} {msg}"
            
        return f"{dim}[{ts}]{reset} {color}{record.levelname:<8}{reset} {dim}{record.name:<15}{reset} {record.getMessage()}{extra_str}"

def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured JSON logging for the backend.

    Idempotent: calling this more than once has no effect.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured — do nothing
    handler = logging.StreamHandler(sys.stdout)
    
    # Use pretty formatter if running locally in a terminal, otherwise fallback to JSON
    if sys.stdout.isatty() or os.environ.get("PRETTY_LOGS"):
        handler.setFormatter(_PrettyTerminalFormatter())
    else:
        handler.setFormatter(_JsonFormatter())
        
    root.addHandler(handler)
    root.setLevel(level)

import os
# Alias so callers that already import configure_logging continue to work.
configure_logging = setup_logging


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger for *name*.

    Equivalent to ``logging.getLogger(name)`` but documents that JSON
    formatting has been applied by :func:`setup_logging`.
    """
    return logging.getLogger(name)
