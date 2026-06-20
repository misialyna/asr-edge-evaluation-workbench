"""Faster-Whisper runtime adapter.

This module imports :mod:`faster_whisper` lazily: the package is
optional (declared in the ``[asr]`` extra in ``pyproject.toml``)
and may be absent on developer machines. The adapter raises a
helpful :class:`RuntimeError` on instantiation if the package is
not installed.
"""

from __future__ import annotations

import importlib
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..schemas import ComputeType, ModelSize, RuntimeConfig
from .base import TranscriptionResult

__all__ = ["FasterWhisperAdapter", "faster_whisper_available"]


def faster_whisper_available() -> bool:
    """Return True if the ``faster_whisper`` package can be imported."""
    try:
        importlib.import_module("faster_whisper")
    except Exception:
        return False
    return True


if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # noqa: F401 — type check only


def _read_wav_duration_s(audio_path: Path) -> float:
    """Read duration of a WAV file. Returns 0.0 for non-WAV inputs."""
    try:
        with wave.open(str(audio_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return 0.0


class FasterWhisperAdapter:
    """Real ``faster-whisper`` inference.

    Parameters
    ----------
    config:
        A validated :class:`RuntimeConfig`. The ``runtime`` field
        must be ``faster-whisper``.

    The class is deliberately small: it does not own caching, device
    selection, or any of the dozen of options faster-whisper exposes.
    Those belong in the experiment config or in future extensions.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        if config.runtime.value != "faster-whisper":
            raise ValueError(
                f"FasterWhisperAdapter requires runtime=faster-whisper, got {config.runtime!r}"
            )
        self._config = config
        self._model: Any = None
        self._model_size_value: str = config.model_size.value

    @property
    def model_size(self) -> ModelSize:
        """Return the configured model size enum."""
        return self._config.model_size

    @property
    def compute_type(self) -> ComputeType:
        """Return the configured compute type enum."""
        return self._config.compute_type

    def load(self) -> None:
        """Load the model weights.

        Downloads (if not cached) happen here. Pass ``download_root``
        via environment if you need to control the cache location.
        """
        try:
            module = importlib.import_module("faster_whisper")
        except Exception as exc:  # pragma: no cover — depends on env
            raise RuntimeError(
                "faster-whisper is not installed. Install with: "
                "pip install 'asr-edge-evaluation-workbench[asr]'"
            ) from exc

        whisper_model_cls = module.WhisperModel
        self._model = whisper_model_cls(
            self._model_size_value,
            device="auto",
            compute_type=self._config.compute_type.value,
        )

    def unload(self) -> None:
        """Free the model reference; ``faster-whisper`` releases memory on GC."""
        self._model = None

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("FasterWhisperAdapter used before load()")
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        audio_duration_s = _read_wav_duration_s(audio_path)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            str(audio_path),
            language=self._config.language,
            beam_size=self._config.beam_size,
            vad_filter=self._config.vad_filter,
            **self._config.decode_options,
        )
        # ``segments`` is a generator; materialise it so the
        # wall-clock measurement covers the full decoding.
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=text,
            detected_language=getattr(info, "language", None),
            inference_duration_s=elapsed,
            audio_duration_s=audio_duration_s,
        )
