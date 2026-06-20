"""ASR runtime adapters.

The orchestrator interacts with ASR systems exclusively through the
:func:`AsrRuntime` protocol defined here. Two implementations ship
in v1:

* :class:`FakeAdapter` — deterministic, parameterised; used by tests
  and by demo mode so the project runs end-to-end on any developer
  machine without downloading multi-GB model weights.
* :class:`FasterWhisperAdapter` — real inference using the
  ``faster-whisper`` package. Lazily imported so the project
  imports cleanly on systems without ``faster-whisper`` installed.
"""

from __future__ import annotations

from .base import AsrRuntime, TranscriptionResult
from .fake_adapter import FakeAdapter, FakeConfig

__all__ = [
    "AsrRuntime",
    "FakeAdapter",
    "FakeConfig",
    "TranscriptionResult",
]
