"""Runtime protocol and result types.

The orchestrator only ever calls three methods:

* ``load()`` — once, before the first ``transcribe`` call; may load
  model weights, warm up caches, etc.
* ``transcribe(path)`` — once per inference; returns a
  :class:`TranscriptionResult`.
* ``unload()`` — once, at the end; frees resources.

The protocol is intentionally minimal so that alternative runtimes
(WhisperTRT in v2, MPS in v2, etc.) can drop in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["AsrRuntime", "TranscriptionResult"]


@dataclass(frozen=True)
class TranscriptionResult:
    """The output of a single inference call.

    ``inference_duration_s`` is measured *by the runtime*, not by the
    orchestrator, so it captures the actual model call and excludes
    only the file-read I/O that happens inside the adapter.
    """

    text: str
    detected_language: str | None
    inference_duration_s: float
    audio_duration_s: float
    extra: dict[str, str] | None = None


@runtime_checkable
class AsrRuntime(Protocol):
    """Structural type for an ASR runtime.

    A class satisfies this protocol simply by implementing the three
    methods. No inheritance is required.
    """

    def load(self) -> None: ...

    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...

    def unload(self) -> None: ...
