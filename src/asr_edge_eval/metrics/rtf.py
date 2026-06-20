"""Real-Time Factor (RTF) and inverse Real-Time Factor (RTFx).

We store *both* in the result row because the two communities that
read these numbers disagree on the convention:

* RTF = inference_time / audio_duration. Lower is better.
  RTF < 1 means "faster than real time" in this convention.
* RTFx = audio_duration / inference_time. Higher is better.
  RTFx > 1 means "faster than real time" in this convention,
  matching the Open ASR Leaderboard reporting.

Both are mathematically equivalent; the report uses ``rtfx`` in
its headline tables and ``rtf`` in the methodology section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RtfResult", "compute_rtf"]


@dataclass(frozen=True)
class RtfResult:
    """Result of a single RTF / RTFx computation."""

    rtf: float
    rtfx: float

    def __post_init__(self) -> None:
        if math.isnan(self.rtf) or math.isnan(self.rtfx):
            raise ValueError("RTF / RTFx must be finite numbers")


def compute_rtf(inference_s: float, audio_s: float) -> RtfResult:
    """Compute RTF and RTFx.

    Parameters
    ----------
    inference_s:
        Wall-clock inference duration in seconds. Must be > 0.
    audio_s:
        Duration of the audio input in seconds. Must be > 0.

    Raises
    ------
    ValueError
        If either input is not strictly positive.
    """
    if inference_s <= 0:
        raise ValueError(f"inference_s must be > 0, got {inference_s!r}")
    if audio_s <= 0:
        raise ValueError(f"audio_s must be > 0, got {audio_s!r}")
    rtf = inference_s / audio_s
    rtfx = audio_s / inference_s
    return RtfResult(rtf=rtf, rtfx=rtfx)
