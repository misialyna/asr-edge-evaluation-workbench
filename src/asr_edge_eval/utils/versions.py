"""Capture software versions relevant to a benchmark run.

Captured at run start and stamped into ``HardwareMetadata``:

* Python version.
* faster-whisper version (if importable).
* jiwer version.
* platform string (``platform.platform()``).
* the optional ASR-edge package version itself.

A failed import never raises — the corresponding value is ``None``.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

__all__ = ["software_versions", "python_version"]


def python_version() -> str:
    """Return the Python interpreter version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _safe_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover — defensive
        return None


def software_versions() -> dict[str, Any]:
    """Return a dict of relevant software versions for the run header."""
    return {
        "python": python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "faster_whisper": _safe_version("faster-whisper"),
        "jiwer": _safe_version("jiwer"),
        "pydantic": _safe_version("pydantic"),
        "asr_edge_evaluation_workbench": _safe_version("asr-edge-evaluation-workbench"),
    }
