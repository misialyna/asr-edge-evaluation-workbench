"""Cross-platform telemetry fallback using ``psutil``.

Used in demo mode on any developer machine where tegrastats is not
available. ``psutil`` is installed as a core dependency (it is
small and ubiquitous) so this module can always be imported.

The :class:`PsutilPoller` exposes the same ``sample()`` method
shape as our tegrastats parser output, so the
:class:`TelemetryCollector` can treat the two interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil

__all__ = ["PsutilSample", "PsutilPoller"]


@dataclass(frozen=True)
class PsutilSample:
    """One point-in-time sample from psutil.

    GPU and temperature are best-effort: ``psutil`` does not expose
    them on every platform. On Linux they may be available via
    ``psutil.sensors_temperatures()`` and ``psutil.sensors_battery()``
    (battery power is rarely useful on edge devices).
    """

    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_percent: float | None = None
    gpu_mem_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "gpu_percent": self.gpu_percent,
            "gpu_mem_mb": self.gpu_mem_mb,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
        }


def _read_temperature_c() -> float | None:
    """Read the first available CPU temperature, if any."""
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    if not sensors:
        return None
    for entries in sensors.values():
        if not entries:
            continue
        current = entries[0].current
        if current is not None:
            return float(current)
    return None


def _read_gpu() -> tuple[float | None, float | None]:
    """Best-effort GPU read. Returns (percent, mem_mb).

    On most non-NVIDIA systems this will be ``(None, None)``; the
    result is then recorded as missing data and the report reflects
    that explicitly.
    """
    # pynvml is the canonical way; we don't add a hard dependency on
    # it. Try to import lazily.
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return None, None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(util.gpu), float(mem.used) / (1024.0 * 1024.0)
    except Exception:
        return None, None


class PsutilPoller:
    """A sampler that yields :class:`PsutilSample` values."""

    def __init__(self) -> None:
        # The first call to ``psutil.cpu_percent()`` is intentionally
        # called here so that subsequent calls return a non-zero
        # value (psutil computes utilization between two samples).
        psutil.cpu_percent(interval=None)

    def sample(self) -> PsutilSample:
        """Return one sample.

        Note: ``psutil.cpu_percent(interval=None)`` returns the value
        since the last call. Calling it here, immediately after the
        constructor's priming call, is the standard pattern.
        """
        cpu = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        gpu_pct, gpu_mem = _read_gpu()
        return PsutilSample(
            cpu_percent=cpu,
            ram_used_mb=float(vm.used) / (1024.0 * 1024.0),
            ram_total_mb=float(vm.total) / (1024.0 * 1024.0),
            gpu_percent=gpu_pct,
            gpu_mem_mb=gpu_mem,
            temperature_c=_read_temperature_c(),
        )
