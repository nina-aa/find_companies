"""Structured JSON logging on the stdlib (no dependency).

``get_logger(run_id)`` returns a lightweight adapter whose calls emit one JSON
object per line, every line carrying ``run_id``. The same data also lives on the
``RunTrace`` in the response — this is just the streaming surface (visible in the
HF Spaces console; rendered human-readably by ``app.cli run --verbose``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

_CONFIGURED = False
LOGGER_NAME = "agentsearch"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in getattr(record, "context", {}).items():
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel((level or os.environ.get("LOG_LEVEL", "INFO")).upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


class RunLogger:
    """Adapter that stamps ``run_id`` (and any bound context) onto every record."""

    def __init__(self, run_id: str, **bound):
        configure()
        self._logger = logging.getLogger(LOGGER_NAME)
        self._context = {"run_id": run_id, **bound}

    def _emit(self, level: int, event: str, **fields) -> None:
        self._logger.log(level, event, extra={"context": {**self._context, **fields}})

    def info(self, event: str, **fields) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields) -> None:
        self._emit(logging.ERROR, event, **fields)


def get_logger(run_id: str, **bound) -> RunLogger:
    return RunLogger(run_id, **bound)
