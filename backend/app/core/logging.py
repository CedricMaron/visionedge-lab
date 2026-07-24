"""Structured JSON logging via structlog.

Every log line can carry: timestamp, session_id, client_device, execution_location,
model_name, model_version, runtime, precision, input_resolution, selected_classes,
frame_id, stage timings, detections, errors, fallback decisions. Callers pass those
as keyword args; the processors render them as JSON.
"""
from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(json_output: bool = True, level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level, logging.INFO))

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "visionedge"):
    return structlog.get_logger(name)
