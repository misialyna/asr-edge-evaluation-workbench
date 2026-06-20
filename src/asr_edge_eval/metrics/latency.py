"""Latency aggregation across the N measured runs of one audio file.

We treat the per-run inference durations as a sample distribution
and report:

* ``avg`` (mean) — robust for symmetric distributions.
* ``p50`` — typical latency.
* ``p95`` — tail latency, what user experience really feels.
* ``p99`` — long tail, ``None`` if N < 20 (insufficient sample).

The function uses a linear-interpolation percentile (``numpy``'s
default ``linear`` method) to match the convention used by most
APM tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["summarize_latencies", "LatencySummary"]


@dataclass(frozen=True)
class LatencySummary:
    """Summary of N measured inference durations."""

    avg_s: float
    p50_s: float
    p95_s: float
    p99_s: float | None
    n: int
    min_s: float
    max_s: float
    std_s: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if any(
            math.isnan(x)
            for x in (self.avg_s, self.p50_s, self.p95_s, self.min_s, self.max_s, self.std_s)
        ):
            raise ValueError("numeric fields must be finite")
        if self.p99_s is not None and math.isnan(self.p99_s):
            raise ValueError("p99_s must be finite or None")


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="linear"))


def summarize_latencies(durations_s: list[float] | tuple[float, ...]) -> LatencySummary:
    """Summarize a sequence of inference durations.

    Parameters
    ----------
    durations_s:
        The inference durations in seconds. Must contain at least one
        strictly positive value.

    Returns
    -------
    LatencySummary

    Raises
    ------
    ValueError
        If the input is empty or contains non-positive values.
    """
    if not durations_s:
        raise ValueError("durations_s must be non-empty")
    if any(d <= 0 for d in durations_s):
        raise ValueError("all durations must be > 0")

    arr = np.asarray(durations_s, dtype=np.float64)
    n = int(arr.size)
    avg = float(arr.mean())
    p50 = _percentile(arr, 50)
    p95 = _percentile(arr, 95)
    # Only emit p99 if we have enough samples for the statistic to be
    # meaningful.
    p99: float | None = _percentile(arr, 99) if n >= 20 else None
    return LatencySummary(
        avg_s=avg,
        p50_s=p50,
        p95_s=p95,
        p99_s=p99,
        n=n,
        min_s=float(arr.min()),
        max_s=float(arr.max()),
        std_s=float(arr.std(ddof=0)),
    )
