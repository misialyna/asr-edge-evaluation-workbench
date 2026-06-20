"""Telemetry: tegrastats parser, psutil fallback, collector lifecycle."""

from .collector import TelemetryCollector, TelemetrySample
from .psutil_fallback import PsutilPoller
from .tegrastats import TegrastatsLine, parse_line, summarize

__all__ = [
    "PsutilPoller",
    "TelemetryCollector",
    "TelemetrySample",
    "TegrastatsLine",
    "parse_line",
    "summarize",
]
