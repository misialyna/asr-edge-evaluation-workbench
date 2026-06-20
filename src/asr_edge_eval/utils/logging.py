"""Structured logging configuration using ``structlog``.

Two renderers are supported:

* ``console`` — colored, human-friendly output for development.
* ``json`` — line-delimited JSON for CI log aggregation.

The configured logger is returned by :func:`get_logger`. We do not
configure the root logger at import time so that consumers (e.g. a
Streamlit app) keep their own log handlers.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

__all__ = ["configure_logging", "get_logger"]

LogFormat = Literal["console", "json"]

_configured = False


def configure_logging(level: str = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog + stdlib logging.

    Safe to call multiple times; subsequent calls replace the
    configuration. Setting ``fmt="json"`` is recommended for CI and
    Jetson runs where log lines will be ingested programmatically.
    """
    global _configured
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib loggers (matplotlib, faster-whisper, urllib3, etc.)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured logger.

    The first call lazily calls :func:`configure_logging` with defaults
    so that a stray ``get_logger("x").info("y")`` does not fail.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
