"""Deterministic fake ASR runtime.

Used by:

* Unit tests (deterministic transcripts, configurable latency).
* Integration tests (full pipeline without model weights).
* Demo mode (run end-to-end on a MacBook without GPUs).

The fake reads the real audio file's duration from the WAV header so
that ``audio_duration_s`` and the resulting RTF / RTFx are realistic.
It does not, however, look at the audio content: the transcript is
whatever the caller configured.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from .base import TranscriptionResult

__all__ = ["FakeAdapter", "FakeConfig"]


@dataclass
class FakeConfig:
    """Configuration for the deterministic fake adapter."""

    transcript: str = "To jest fałszywa transkrypcja."
    simulated_latency_s: float = 0.25
    detected_language: str = "pl"
    extra: dict[str, str] = field(default_factory=dict)
    # If True, simulated_latency_s is ignored and the function returns
    # immediately. Used in unit tests where determinism matters more
    # than realism.
    zero_latency: bool = False


class FakeAdapter:
    """A deterministic, GPU-free ASR runtime for tests and demos."""

    def __init__(self, config: FakeConfig | None = None) -> None:
        self._cfg = config or FakeConfig()
        self._loaded = False

    @property
    def config(self) -> FakeConfig:
        """Return the active configuration (mainly for assertions in tests)."""
        return self._cfg

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not self._loaded:
            raise RuntimeError("FakeAdapter used before load()")
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        audio_duration_s = _read_wav_duration_s(audio_path)
        t0 = time.perf_counter()
        if not self._cfg.zero_latency and self._cfg.simulated_latency_s > 0:
            time.sleep(self._cfg.simulated_latency_s)
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=self._cfg.transcript,
            detected_language=self._cfg.detected_language,
            inference_duration_s=elapsed,
            audio_duration_s=audio_duration_s,
            extra=dict(self._cfg.extra),
        )


def _read_wav_duration_s(audio_path: Path) -> float:
    """Read the duration of a WAV file from its header.

    Falls back to 0.0 if the file cannot be parsed; the orchestrator
    treats a zero duration as a configuration error, so a
    non-WAV-input in demo mode is loud rather than silent.
    """
    try:
        with wave.open(str(audio_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return 0.0
