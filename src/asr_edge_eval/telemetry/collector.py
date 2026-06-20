"""Telemetry collector: lifecycle around a single inference run.

The collector is responsible for:

1. starting telemetry capture (either spawning ``tegrastats`` or
   polling ``psutil``);
2. keeping it running for the duration of the inference;
3. stopping it reliably — even if the inference raises;
4. aggregating the captured samples into a
   :class:`~asr_edge_eval.schemas.ResourceMetrics`.

It is a context manager so that ``try/finally`` is automatic.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..schemas import ResourceMetrics, TelemetryBackend
from .psutil_fallback import PsutilPoller, PsutilSample
from .tegrastats import (
    TegrastatsLine,
    parse_line as parse_tegrastats_line,
    summarize as summarize_tegrastats,
)

__all__ = ["TelemetryCollector", "TelemetrySample", "build_collector"]


@dataclass(frozen=True)
class TelemetrySample:
    """A single internal sample used by the collector.

    The collector may receive samples from either tegrastats
    (``tegrastats_line`` populated) or psutil (``psutil_sample``
    populated), or both. Exactly one is set in practice.
    """

    tegrastats_line: TegrastatsLine | None = None
    psutil_sample: PsutilSample | None = None
    wall_time_s: float = 0.0


def _spawn_tegrastats(path: str, interval_ms: int, log_path: Path) -> subprocess.Popen[str]:
    """Spawn tegrastats, redirecting stdout to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — handle lives in _stop_tegrastats
    # tegrastats writes to stdout; we capture to a file so it can be
    # re-parsed by a tool after the run, in addition to the live
    # samples the collector reads.
    proc = subprocess.Popen(
        [path, "--interval", str(interval_ms)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        # New process group so we can SIGTERM cleanly without
        # affecting the parent.
        start_new_session=True,
    )
    return proc


def _stop_tegrastats(proc: subprocess.Popen[str] | None) -> None:
    """Stop a tegrastats subprocess reliably."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    except (OSError, ProcessLookupError):
        pass


def _kill_process_group(pgid: int) -> None:
    """Best-effort kill of the whole process group of a tegrastats child."""
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    # Give the OS a moment, then SIGKILL if still alive.
    time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return


def _read_new_lines(path: Path, since_byte: int) -> tuple[list[str], int]:
    """Return lines newly appended to ``path`` since ``since_byte``."""
    if not path.is_file():
        return [], since_byte
    with open(path, encoding="utf-8", errors="replace") as f:
        f.seek(since_byte)
        new = f.read()
    return new.splitlines(), since_byte + len(new.encode("utf-8", errors="replace"))


class TelemetryCollector:
    """Capture telemetry around a single inference run.

    Usage
    -----

    .. code-block:: python

        with TelemetryCollector(backend=..., ...) as tc:
            runtime.transcribe(audio_path)
        metrics = tc.summary()

    The collector is robust to:

    * a missing tegrastats binary (falls back to no telemetry, not
      to a crash);
    * an inference that raises (the ``__exit__`` runs ``finally``
      semantics);
    * a tegrastats process that dies mid-run (the next sample simply
      doesn't appear).
    """

    def __init__(
        self,
        *,
        backend: TelemetryBackend,
        interval_ms: int = 500,
        tegrastats_path: str = "/usr/bin/tegrastats",
        raw_log_dir: Path | None = None,
        run_id: str = "telemetry",
    ) -> None:
        self._backend = backend
        self._interval_ms = interval_ms
        self._tegrastats_path = tegrastats_path
        self._raw_log_dir = raw_log_dir
        self._run_id = run_id
        self._samples: list[TelemetrySample] = []
        self._proc: subprocess.Popen[str] | None = None
        self._raw_log_path: Path | None = None
        self._raw_offset: int = 0
        self._stopped = True
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._psutil: PsutilPoller | None = None

    # ----- context manager -----

    def __enter__(self) -> TelemetryCollector:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ----- lifecycle -----

    def start(self) -> None:
        if not self._stopped:
            return
        self._stopped = False
        self._samples = []
        if self._backend == TelemetryBackend.TEGRASTATS:
            self._start_tegrastats()
        elif self._backend == TelemetryBackend.PSUTIL:
            self._psutil = PsutilPoller()
            self._start_psutil_poller()
        else:
            # TelemetryBackend.NONE: do nothing.
            pass

    def _start_tegrastats(self) -> None:
        if shutil.which(self._tegrastats_path) is None and not Path(self._tegrastats_path).exists():
            # Binary not found; record a warning and degrade to psutil
            # for this run so the experiment can still complete.
            from ..utils.logging import get_logger

            get_logger(__name__).warning(
                "tegrastats binary not found at",
                path=self._tegrastats_path,
                fallback="psutil",
            )
            self._backend = TelemetryBackend.PSUTIL
            self._psutil = PsutilPoller()
            self._start_psutil_poller()
            return

        if self._raw_log_dir is not None:
            self._raw_log_path = (self._raw_log_dir / f"{self._run_id}.log").resolve()
        else:
            self._raw_log_path = None

        log_path = self._raw_log_path or Path(f"/tmp/{self._run_id}-tegrastats.log")
        self._proc = _spawn_tegrastats(self._tegrastats_path, self._interval_ms, log_path)
        self._raw_log_path = log_path
        # Spawn a reader thread that pulls lines from the log file as
        # they are written. The thread is daemon so the program can
        # exit even if a tegrastats child is stuck.
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._tegrastats_reader_loop, name="tegrastats-reader", daemon=True
        )
        self._reader_thread.start()

    def _start_psutil_poller(self) -> None:
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._psutil_poller_loop, name="psutil-poller", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Tell the reader thread to exit.
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        # Stop the tegrastats subprocess.
        if self._proc is not None:
            pgid = 0
            try:
                pgid = os.getpgid(self._proc.pid)
            except (OSError, ProcessLookupError):
                pgid = 0
            _stop_tegrastats(self._proc)
            if pgid > 0:
                _kill_process_group(pgid)
            self._proc = None

    # ----- reader loops -----

    def _tegrastats_reader_loop(self) -> None:
        assert self._raw_log_path is not None
        while not self._reader_stop.is_set():
            lines, self._raw_offset = _read_new_lines(self._raw_log_path, self._raw_offset)
            now = time.monotonic()
            for line in lines:
                parsed = parse_tegrastats_line(line)
                self._samples.append(TelemetrySample(tegrastats_line=parsed, wall_time_s=now))
            # Sleep a fraction of the interval to be responsive
            # without burning CPU.
            time.sleep(max(self._interval_ms / 1000.0 / 4.0, 0.05))

    def _psutil_poller_loop(self) -> None:
        assert self._psutil is not None
        while not self._reader_stop.is_set():
            try:
                sample = self._psutil.sample()
            except Exception:  # pragma: no cover — defensive
                sample = None
            if sample is not None:
                self._samples.append(
                    TelemetrySample(psutil_sample=sample, wall_time_s=time.monotonic())
                )
            time.sleep(max(self._interval_ms / 1000.0, 0.05))

    # ----- public API -----

    def samples(self) -> Sequence[TelemetrySample]:
        """Return a copy of the collected samples (for debugging)."""
        return list(self._samples)

    def summary(self) -> ResourceMetrics:
        """Aggregate the captured samples into a ResourceMetrics value."""
        if self._backend == TelemetryBackend.TEGRASTATS:
            lines = [s.tegrastats_line for s in self._samples if s.tegrastats_line is not None]
            return summarize_tegrastats(
                lines,
                interval_ms=self._interval_ms,
                backend=TelemetryBackend.TEGRASTATS,
            )
        if self._backend == TelemetryBackend.PSUTIL:
            return _summarize_psutil(
                [s.psutil_sample for s in self._samples if s.psutil_sample is not None],
                interval_ms=self._interval_ms,
            )
        return ResourceMetrics(
            telemetry_backend=TelemetryBackend.NONE,
            telemetry_interval_ms=self._interval_ms,
            telemetry_samples=0,
        )

    @property
    def backend(self) -> TelemetryBackend:
        return self._backend


def _summarize_psutil(
    samples: list[PsutilSample],
    *,
    interval_ms: int,
) -> ResourceMetrics:
    if not samples:
        return ResourceMetrics(
            telemetry_backend=TelemetryBackend.PSUTIL,
            telemetry_interval_ms=interval_ms,
        )
    cpus = [s.cpu_percent for s in samples]
    rams = [s.ram_used_mb for s in samples]
    gpu_pcts = [s.gpu_percent for s in samples if s.gpu_percent is not None]
    gpu_mems = [s.gpu_mem_mb for s in samples if s.gpu_mem_mb is not None]
    temps = [s.temperature_c for s in samples if s.temperature_c is not None]

    def _mean(xs: list[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    def _max(xs: list[float]) -> float | None:
        return float(max(xs)) if xs else None

    return ResourceMetrics(
        cpu_avg_percent=_mean(cpus),
        cpu_peak_percent=_max(cpus),
        ram_avg_mb=_mean(rams),
        ram_peak_mb=_max(rams),
        gpu_avg_percent=_mean(gpu_pcts),
        gpu_peak_percent=_max(gpu_pcts),
        gpu_mem_avg_mb=_mean(gpu_mems),
        gpu_mem_peak_mb=_max(gpu_mems),
        temperature_avg_c=_mean(temps),
        temperature_peak_c=_max(temps),
        power_avg_w=None,
        power_peak_w=None,
        telemetry_backend=TelemetryBackend.PSUTIL,
        telemetry_interval_ms=interval_ms,
        telemetry_samples=len(samples),
    )


def build_collector(
    *,
    backend: TelemetryBackend,
    interval_ms: int,
    tegrastats_path: str,
    raw_log_dir: Path | None,
    run_id: str,
) -> TelemetryCollector:
    """Factory kept for symmetry; prefer constructing the class directly."""
    return TelemetryCollector(
        backend=backend,
        interval_ms=interval_ms,
        tegrastats_path=tegrastats_path,
        raw_log_dir=raw_log_dir,
        run_id=run_id,
    )


def _ensure_utf8_stdio() -> None:  # pragma: no cover — Windows-only safety
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
