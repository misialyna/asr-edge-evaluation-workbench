"""Robust, tolerant parser for ``tegrastats`` output.

The :command:`tegrastats` utility ships on every Jetson board and
emits one line per sampling interval. The line format drifts
between JetPack releases and even between board models, so the
parser is **explicitly tolerant**: any field that is not present in
a line is returned as ``None`` rather than raising.

Two test fixtures exercise both the JetPack 5 and the JetPack 6
output styles. A minimal fixture exercises the no-power / no-GPU
case so the parser does not depend on every field being present.

The parser is pure-Python and stateless: it converts one line into
a :class:`TegrastatsLine`. Aggregation is done by
:func:`summarize`, which reduces N samples into a
:class:`~asr_edge_eval.schemas.ResourceMetrics` value.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from ..schemas import ResourceMetrics, TelemetryBackend

__all__ = ["TegrastatsLine", "parse_line", "summarize", "validate_format"]


# ---------------------------------------------------------------------------
# Per-field patterns
# ---------------------------------------------------------------------------

# RAM 1234/5678MB  or  RAM 1234/5678MB (lfm)
_RAM_RE = re.compile(
    r"\bRAM\s+(?P<used>\d+)/(?P<total>\d+)\s*MB",
    re.IGNORECASE,
)

# SWAP 0/0MB
_SWAP_RE = re.compile(
    r"\bSWAP\s+(?P<used>\d+)/(?P<total>\d+)\s*MB",
    re.IGNORECASE,
)

# CPU [1%@1024,2%@1024,3%@1024,4%@1024,5%@1024,6%@1024,7%@1024,8%@1024]
_CPU_RE = re.compile(
    r"\bCPU\s*\[(?P<per_core>[^\]]*)\]",
    re.IGNORECASE,
)

# GPU 1234MiB or GPU 15% (some JetPack versions emit one or the other)
_GPU_RE = re.compile(
    r"\bGPU\s+(?P<percent>\d+)%",
    re.IGNORECASE,
)
_GPU_MIB_RE = re.compile(
    r"\bGPU\s+(?P<mib>\d+)\s*MiB",
    re.IGNORECASE,
)

# TEMP cpu@45.5C or TEMP 45C
_TEMP_RE = re.compile(
    r"\bTEMP\s+(?:[A-Za-z0-9_]+@)?(?P<celsius>-?\d+(?:\.\d+)?)C",
    re.IGNORECASE,
)

# POWER 1234/5678 (mW usually, no unit; JetPack 5 sometimes adds W)
_POWER_RE = re.compile(
    r"\bPOWER\s+(?P<used>\d+)/(?P<total>\d+)(?P<unit>m?W)?",
    re.IGNORECASE,
)

# VDD_IN 1234/5678 (sometimes reported as a separate field)
_VDD_RE = re.compile(
    r"\bVDD_IN\s+(?P<used>\d+)/(?P<total>\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TegrastatsLine:
    """Parsed form of a single tegrastats line.

    Every field is optional. ``None`` means "not present in this
    line", which is common when a Jetson module does not report a
    particular sensor (e.g. a 2GB Nano with no INA231 power monitor).
    """

    raw: str
    timestamp_ms: int | None = None  # we don't actually parse a timestamp; reserved
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None
    swap_used_mb: int | None = None
    swap_total_mb: int | None = None
    cpu_per_core_percent: tuple[int, ...] | None = None
    cpu_avg_percent: float | None = None
    gpu_percent: int | None = None
    gpu_mem_mib: int | None = None
    temperature_c: float | None = None
    power_used_mw: int | None = None
    power_total_mw: int | None = None
    vdd_in_used_mw: int | None = None
    vdd_in_total_mw: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (tuples → lists)."""
        d = asdict(self)
        if d["cpu_per_core_percent"] is not None:
            d["cpu_per_core_percent"] = list(d["cpu_per_core_percent"])
        d["raw"] = self.raw  # kept for debugging
        return d


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_cpu_block(block: str) -> tuple[int, ...]:
    """Parse a ``[1%@1024,2%@1024,...]`` block into a tuple of percentages."""
    out: list[int] = []
    for part in block.split(","):
        part = part.strip()
        if not part:
            continue
        # Each entry looks like ``3%@1024`` (JetPack 5) or
        # sometimes just ``3`` (no percent sign). Tolerate both.
        match = re.search(r"(-?\d+)\s*%?", part)
        if match:
            out.append(int(match.group(1)))
    return tuple(out)


def parse_line(line: str) -> TegrastatsLine:
    """Parse one tegrastats line. Never raises.

    Unknown or malformed lines yield a :class:`TegrastatsLine` with
    ``raw`` set and all numeric fields ``None``.
    """
    if line is None:
        return TegrastatsLine(raw="")
    s = line.strip()
    if not s:
        return TegrastatsLine(raw="")

    ram = _RAM_RE.search(s)
    swap = _SWAP_RE.search(s)
    cpu = _CPU_RE.search(s)
    gpu_pct = _GPU_RE.search(s)
    gpu_mib = _GPU_MIB_RE.search(s)
    temp = _TEMP_RE.search(s)
    power = _POWER_RE.search(s)
    vdd = _VDD_RE.search(s)

    cpu_tuple: tuple[int, ...] | None = None
    cpu_avg: float | None = None
    if cpu is not None:
        cpu_tuple = _parse_cpu_block(cpu.group("per_core"))
        if cpu_tuple:
            cpu_avg = float(sum(cpu_tuple)) / float(len(cpu_tuple))

    # GPU: percent and MiB come from different regexes. If only MiB
    # is present we leave percent as None; if only percent is
    # present we leave MiB as None. Both being None is valid.
    gpu_percent_val: int | None = int(gpu_pct.group("percent")) if gpu_pct else None
    gpu_mib_val: int | None = int(gpu_mib.group("mib")) if gpu_mib else None

    power_used: int | None = None
    power_total: int | None = None
    if power is not None:
        power_used = int(power.group("used"))
        power_total = int(power.group("total"))
        # The unit is irrelevant for our purposes; we always store mW
        # and treat unlabelled values as already in mW (which is the
        # tegrastats default).

    vdd_used: int | None = None
    vdd_total: int | None = None
    if vdd is not None:
        vdd_used = int(vdd.group("used"))
        vdd_total = int(vdd.group("total"))

    return TegrastatsLine(
        raw=s,
        ram_used_mb=int(ram.group("used")) if ram else None,
        ram_total_mb=int(ram.group("total")) if ram else None,
        swap_used_mb=int(swap.group("used")) if swap else None,
        swap_total_mb=int(swap.group("total")) if swap else None,
        cpu_per_core_percent=cpu_tuple,
        cpu_avg_percent=cpu_avg,
        gpu_percent=gpu_percent_val,
        gpu_mem_mib=gpu_mib_val,
        temperature_c=float(temp.group("celsius")) if temp else None,
        power_used_mw=power_used,
        power_total_mw=power_total,
        vdd_in_used_mw=vdd_used,
        vdd_in_total_mw=vdd_total,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def summarize(
    samples: list[TegrastatsLine],
    *,
    interval_ms: int | None = None,
    backend: TelemetryBackend = TelemetryBackend.TEGRASTATS,
) -> ResourceMetrics:
    """Aggregate a list of parsed tegrastats lines into ResourceMetrics.

    ``cpu_avg_percent`` is the *average across time* of the per-line
    CPU averages. ``cpu_peak_percent`` is the *max across time* of
    the per-line averages. The same convention is used for RAM, GPU
    and temperature.
    """
    if not samples:
        return ResourceMetrics(
            telemetry_backend=backend,
            telemetry_interval_ms=interval_ms,
            telemetry_samples=0,
        )

    cpu_avgs = [s.cpu_avg_percent for s in samples if s.cpu_avg_percent is not None]
    cpu_peaks = [max(s.cpu_per_core_percent) for s in samples if s.cpu_per_core_percent] or []

    ram_used = [float(s.ram_used_mb) for s in samples if s.ram_used_mb is not None]
    # RAM "peak" is the highest *used* value across the run, which is
    # what a process would actually see under memory pressure.

    gpu_pcts = [float(s.gpu_percent) for s in samples if s.gpu_percent is not None]
    gpu_mems = [float(s.gpu_mem_mib) for s in samples if s.gpu_mem_mib is not None]

    temps = [float(s.temperature_c) for s in samples if s.temperature_c is not None]
    power_used = [float(s.power_used_mw) for s in samples if s.power_used_mw is not None]

    # GPU memory peak: take max of the used-MiB samples; convert to MB
    # (1 MiB = 1.048576 MB). The schema field is MB.
    gpu_mem_peak_mb = _safe_max([m * 1.048576 for m in gpu_mems])
    gpu_mem_avg_mb = _safe_mean([m * 1.048576 for m in gpu_mems])

    # Power: stored in mW in the parser; schema wants watts. Convert
    # if present.
    power_avg_w_value: float | None = (
        sum(power_used) / (1000.0 * len(power_used)) if power_used else None
    )
    power_peak_w_value: float | None = (max(power_used) / 1000.0) if power_used else None

    return ResourceMetrics(
        cpu_avg_percent=_safe_mean(cpu_avgs),
        cpu_peak_percent=_safe_max(cpu_peaks) if cpu_peaks else _safe_max(cpu_avgs),
        ram_avg_mb=_safe_mean(ram_used),
        ram_peak_mb=_safe_max(ram_used),
        gpu_avg_percent=_safe_mean(gpu_pcts),
        gpu_peak_percent=_safe_max(gpu_pcts),
        gpu_mem_avg_mb=gpu_mem_avg_mb,
        gpu_mem_peak_mb=gpu_mem_peak_mb,
        temperature_avg_c=_safe_mean(temps),
        temperature_peak_c=_safe_max(temps),
        power_avg_w=power_avg_w_value,
        power_peak_w=power_peak_w_value,
        telemetry_backend=backend,
        telemetry_interval_ms=interval_ms,
        telemetry_samples=len(samples),
    )


# ---------------------------------------------------------------------------
# Format validator (for the ``validate-tegrastats-format`` CLI)
# ---------------------------------------------------------------------------


def validate_format(path: str) -> dict[str, bool]:
    """Return which fields are present in the given tegrastats log file.

    Useful as a pre-flight check before a Jetson run.
    """
    detected: dict[str, bool] = {
        "ram": False,
        "swap": False,
        "cpu": False,
        "gpu_percent": False,
        "gpu_mib": False,
        "temperature": False,
        "power": False,
        "vdd_in": False,
    }
    from pathlib import Path  # local import to keep top-level lean

    p = Path(path)
    if not p.is_file():
        return detected
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if _RAM_RE.search(line):
            detected["ram"] = True
        if _SWAP_RE.search(line):
            detected["swap"] = True
        if _CPU_RE.search(line):
            detected["cpu"] = True
        if _GPU_RE.search(line):
            detected["gpu_percent"] = True
        if _GPU_MIB_RE.search(line):
            detected["gpu_mib"] = True
        if _TEMP_RE.search(line):
            detected["temperature"] = True
        if _POWER_RE.search(line):
            detected["power"] = True
        if _VDD_RE.search(line):
            detected["vdd_in"] = True
    return detected
