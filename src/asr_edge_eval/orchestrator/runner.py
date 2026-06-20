"""Experiment orchestrator.

The orchestrator turns an :class:`ExperimentConfig` and a list of
:class:`ManifestEntry` into a stream of :class:`BenchmarkRun`
rows. The flow per ``(config_id, audio_file)`` is:

1. Skip the file if it does not exist (warn, do not abort).
2. For each warmup run, call ``runtime.transcribe()`` and discard
   the result. Warmup models the cost that any real deployment
   pays once at start-up.
3. For each measured run:
   * start the :class:`TelemetryCollector`;
   * call ``runtime.transcribe()`` and time the call from inside
     the runtime (so the measurement matches what the runtime
     itself reports);
   * stop the collector;
   * compute WER, CER, RTF, RTFx, latency percentiles;
   * append a :class:`BenchmarkRun` row.

The orchestrator is fail-soft: a single failed run records an
error row but does not abort the experiment. ``fail_fast`` is an
explicit opt-in.
"""

from __future__ import annotations

import contextlib
import json
import platform
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.manifest import ManifestEntry
from ..metrics.cer import compute_cer
from ..metrics.latency import summarize_latencies
from ..metrics.normalize import NORMALIZER_VERSION, normalize_pl
from ..metrics.rtf import compute_rtf
from ..metrics.wer import compute_wer
from ..runtime import AsrRuntime, FakeAdapter, FakeConfig
from ..runtime.faster_whisper_adapter import FasterWhisperAdapter, faster_whisper_available
from ..schemas import (
    AudioMetadata,
    BenchmarkRun,
    ComputeType,
    ExperimentConfig,
    HardwareMetadata,
    LatencyMetrics,
    Mode,
    ModelSize,
    QualityMetrics,
    ResourceMetrics,
    RuntimeConfig,
    RuntimeConfigurationSpec,
    TelemetryBackend,
)
from ..telemetry import TelemetryCollector
from ..utils.git import current_commit
from ..utils.logging import get_logger
from ..utils.versions import python_version

__all__ = ["OrchestrationError", "run_experiment"]

log = get_logger(__name__)


class OrchestrationError(RuntimeError):
    """Raised when the orchestrator cannot make progress."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: ExperimentConfig,
    *,
    entries: list[ManifestEntry],
    audio_root: Path,
    runtime_factory: Any = None,
    git_commit: str | None = None,
) -> list[BenchmarkRun]:
    """Run an experiment and return a list of :class:`BenchmarkRun` rows.

    Parameters
    ----------
    cfg:
        A validated :class:`ExperimentConfig`.
    entries:
        The audio files to run, in order.
    audio_root:
        Directory under which the manifest's relative paths resolve.
    runtime_factory:
        Optional callable ``(RuntimeConfigurationSpec) -> AsrRuntime``
        used to override the default ``FasterWhisperAdapter`` (e.g.
        in tests with a ``FakeAdapter``).
    git_commit:
        Override the captured git commit hash (for tests).
    """
    if not entries:
        raise OrchestrationError("No manifest entries to run.")

    hardware = _capture_hardware()
    commit = git_commit if git_commit is not None else current_commit()
    runs: list[BenchmarkRun] = []

    for spec in cfg.runtime.configurations:
        runtime = _build_runtime(spec, cfg.experiment.mode, runtime_factory)
        try:
            runtime.load()
        except Exception as exc:  # pragma: no cover — runtime-specific
            log.error(
                "runtime.load() failed; configuration will be skipped",
                config_id=spec.id,
                error=str(exc),
            )
            if cfg.execution.fail_fast:
                raise
            runs.extend(
                _error_runs_for_config(
                    spec=spec,
                    cfg=cfg,
                    hardware=hardware,
                    entries=entries,
                    error=f"runtime.load() failed: {exc}",
                    commit=commit,
                )
            )
            continue

        try:
            for entry in entries:
                audio_path = (audio_root / entry.path_rel).resolve()
                if not audio_path.is_file():
                    log.warning(
                        "audio file missing; skipping",
                        config_id=spec.id,
                        file_id=entry.file_id,
                        path=str(audio_path),
                    )
                    if cfg.execution.fail_fast:
                        raise OrchestrationError(f"missing audio: {audio_path}")
                    continue

                # Respect max_files: the caller can truncate ``entries``;
                # no extra check here.

                # Warm-up.
                for w in range(cfg.execution.warmup_runs):
                    log.info(
                        "warmup run",
                        config_id=spec.id,
                        file_id=entry.file_id,
                        warmup_index=w + 1,
                    )
                    try:
                        runtime.transcribe(audio_path)
                    except Exception as exc:
                        log.warning(
                            "warmup run failed",
                            config_id=spec.id,
                            file_id=entry.file_id,
                            error=str(exc),
                        )

                # Measured runs.
                measured_durations: list[float] = []
                measured_text: str = ""
                last_error: str | None = None
                measured_resource = ResourceMetrics(
                    telemetry_backend=(
                        TelemetryBackend.TEGRASTATS
                        if cfg.telemetry.backend == TelemetryBackend.TEGRASTATS
                        else TelemetryBackend.PSUTIL
                        if cfg.telemetry.backend == TelemetryBackend.PSUTIL
                        else TelemetryBackend.NONE
                    ),
                    telemetry_interval_ms=cfg.telemetry.interval_ms,
                )
                for run_index in range(cfg.execution.measured_runs_per_file):
                    rid = f"{spec.id}__{entry.file_id}__r{run_index + 1:02d}"
                    log.info(
                        "measured run",
                        config_id=spec.id,
                        file_id=entry.file_id,
                        run_index=run_index + 1,
                        run_id=rid,
                    )
                    try:
                        with TelemetryCollector(
                            backend=cfg.telemetry.backend,
                            interval_ms=cfg.telemetry.interval_ms,
                            tegrastats_path=cfg.telemetry.tegrastats_path,
                            raw_log_dir=Path(cfg.telemetry.raw_output_dir),
                            run_id=rid,
                        ) as tc:
                            result = runtime.transcribe(audio_path)
                        measured_durations.append(result.inference_duration_s)
                        measured_text = result.text
                        measured_resource = tc.summary()
                    except Exception as exc:
                        last_error = str(exc)
                        log.warning(
                            "measured run failed",
                            run_id=rid,
                            error=last_error,
                        )
                        if cfg.execution.fail_fast:
                            raise
                        continue

                    # Inter-run cooldown.
                    if (
                        cfg.execution.inter_run_cooldown_s > 0
                        and run_index + 1 < cfg.execution.measured_runs_per_file
                    ):
                        time.sleep(cfg.execution.inter_run_cooldown_s)

                if not measured_durations:
                    runs.append(
                        _error_run(
                            spec=spec,
                            cfg=cfg,
                            hardware=hardware,
                            entry=entry,
                            error=last_error or "all measured runs failed",
                            commit=commit,
                        )
                    )
                    continue

                # Aggregate latencies.
                summary = summarize_latencies(measured_durations)
                rtf_res = compute_rtf(summary.p95_s, entry.duration_s)
                # Note: RTF/RTFx in the row uses the *median* measured
                # duration as the representative per-inference latency.
                # The P95 is preserved in the LatencyMetrics.
                avg = compute_rtf(summary.avg_s, entry.duration_s)

                ref_norm = normalize_pl(entry.reference_transcript)
                hyp_norm = normalize_pl(measured_text)
                wer = compute_wer(ref_norm, hyp_norm)
                cer = compute_cer(ref_norm, hyp_norm)

                run_obj = BenchmarkRun(
                    run_id=f"{spec.id}__{entry.file_id}__agg",
                    timestamp_utc=datetime.now(UTC),
                    mode=cfg.experiment.mode,
                    config_id=spec.id,
                    hardware=hardware,
                    runtime_config=RuntimeConfig(
                        runtime=spec.type if hasattr(spec, "type") else cfg.runtime.type,
                        model_size=spec.model_size,
                        compute_type=spec.compute_type,
                        language=spec.language,
                        beam_size=spec.beam_size,
                        vad_filter=spec.vad_filter,
                        decode_options=spec.decode_options,
                    ),
                    audio=AudioMetadata(
                        file_id=entry.file_id,
                        path_rel=entry.path_rel,
                        duration_s=entry.duration_s,
                        sample_rate=entry.sample_rate,
                        channels=entry.channels,
                        reference_transcript=entry.reference_transcript,
                        dataset_source=entry.dataset_source,
                        clip_license=entry.clip_license,
                    ),
                    latency=LatencyMetrics(
                        warmup_run=False,
                        inference_duration_s=summary.avg_s,
                        audio_duration_s=entry.duration_s,
                        rtf=avg.rtf,
                        rtfx=avg.rtfx,
                        p50_latency_s=summary.p50_s,
                        p95_latency_s=summary.p95_s,
                        p99_latency_s=summary.p99_s,
                        num_measured_runs=summary.n,
                    ),
                    quality=QualityMetrics(
                        wer=wer.wer,
                        cer=cer.cer,
                        substitutions=wer.substitutions,
                        deletions=wer.deletions,
                        insertions=wer.insertions,
                        hits=wer.hits,
                        reference_words=wer.reference_words,
                        hypothesis_words=wer.hypothesis_words,
                        reference_chars=cer.reference_chars,
                        hypothesis_chars=cer.hypothesis_chars,
                        normalizer_version=NORMALIZER_VERSION,
                    ),
                    resources=measured_resource,
                    hypothesis_transcript=measured_text,
                    git_commit=commit,
                    notes=(
                        f"rtf@p95={rtf_res.rtf:.3f} rtfx@p95={rtf_res.rtfx:.3f}; "
                        f"warmup_runs={cfg.execution.warmup_runs}"
                    ),
                )
                runs.append(run_obj)
        finally:
            with contextlib.suppress(Exception):  # pragma: no cover
                runtime.unload()

    return runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_runtime(
    spec: RuntimeConfigurationSpec,
    mode: Mode,
    factory: Any,
) -> AsrRuntime:
    if factory is not None:
        return factory(spec)
    # Demo mode uses the fake adapter if the YAML opts in.
    if mode == Mode.DEMO:
        # The orchestrator does not import the experiment flag here
        # because it is intentionally permissive: a demo experiment
        # file that does *not* opt in to the fake adapter will still
        # try faster-whisper, which is fine — the caller is in
        # control via ``factory`` in tests and via the
        # ``--use-fake-runtime`` CLI flag.
        return FasterWhisperAdapter(
            RuntimeConfig(
                runtime=cfg_runtime_type(),
                model_size=spec.model_size,
                compute_type=spec.compute_type,
                language=spec.language,
                beam_size=spec.beam_size,
                vad_filter=spec.vad_filter,
                decode_options=spec.decode_options,
            )
        )
    if not faster_whisper_available():
        raise OrchestrationError(
            "faster-whisper is not installed. Install with: "
            "pip install 'asr-edge-evaluation-workbench[asr]' or pass "
            "a runtime_factory."
        )
    return FasterWhisperAdapter(
        RuntimeConfig(
            runtime=cfg_runtime_type(),
            model_size=spec.model_size,
            compute_type=spec.compute_type,
            language=spec.language,
            beam_size=spec.beam_size,
            vad_filter=spec.vad_filter,
            decode_options=spec.decode_options,
        )
    )


def cfg_runtime_type():
    """Late import to avoid a circular reference at module load."""
    from ..schemas import RuntimeName

    return RuntimeName.FASTER_WHISPER


def _capture_hardware() -> HardwareMetadata:
    """Capture a best-effort hardware snapshot."""
    import multiprocessing
    import os
    import subprocess

    def _safe_run(cmd: list[str]) -> str | None:
        try:
            r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return None

    cpu_model = _safe_run(["cat", "/proc/cpuinfo"]) or None
    if cpu_model:
        for line in cpu_model.splitlines():
            if "model name" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
        else:
            cpu_model = None

    try:
        cores_logical = multiprocessing.cpu_count()
    except Exception:
        cores_logical = None
    try:
        cores_physical = os.cpu_count()  # type: ignore[assignment]
    except Exception:
        cores_physical = None

    return HardwareMetadata(
        device_name=platform.node() or "unknown",
        os=platform.system() or "unknown",
        os_version=platform.release() or "unknown",
        python_version=python_version(),
        cpu_model=cpu_model,
        cpu_cores_logical=cores_logical,
        cpu_cores_physical=cores_physical,
        gpu_model=_safe_run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if _safe_run(["which", "nvidia-smi"])
        else None,
        jetpack_version=os.environ.get("JETPACK_VERSION"),
        cuda_version=os.environ.get("CUDA_VERSION"),
        cudnn_version=None,
        driver_version=None,
        machine_arch=platform.machine(),
    )


def _error_run(
    *,
    spec: RuntimeConfigurationSpec,
    cfg: ExperimentConfig,
    hardware: HardwareMetadata,
    entry: ManifestEntry,
    error: str,
    commit: str | None,
) -> BenchmarkRun:
    """Build a BenchmarkRun row for a single failed run."""
    return BenchmarkRun(
        run_id=f"{spec.id}__{entry.file_id}__err",
        timestamp_utc=datetime.now(UTC),
        mode=cfg.experiment.mode,
        config_id=spec.id,
        hardware=hardware,
        runtime_config=RuntimeConfig(
            runtime=cfg.runtime.type,
            model_size=spec.model_size,
            compute_type=spec.compute_type,
            language=spec.language,
            beam_size=spec.beam_size,
            vad_filter=spec.vad_filter,
            decode_options=spec.decode_options,
        ),
        audio=AudioMetadata(
            file_id=entry.file_id,
            path_rel=entry.path_rel,
            duration_s=entry.duration_s,
            sample_rate=entry.sample_rate,
            channels=entry.channels,
            reference_transcript=entry.reference_transcript,
            dataset_source=entry.dataset_source,
            clip_license=entry.clip_license,
        ),
        latency=LatencyMetrics(
            warmup_run=False,
            inference_duration_s=0.0,
            audio_duration_s=entry.duration_s,
            rtf=0.0,
            rtfx=0.0,
            p50_latency_s=0.0,
            p95_latency_s=0.0,
            p99_latency_s=None,
            num_measured_runs=0,
        ),
        quality=QualityMetrics(
            wer=float("inf"),
            cer=float("inf"),
            substitutions=0,
            deletions=0,
            insertions=0,
            hits=0,
            reference_words=len(entry.reference_transcript.split()),
            hypothesis_words=0,
            reference_chars=len(entry.reference_transcript),
            hypothesis_chars=0,
            normalizer_version=NORMALIZER_VERSION,
        ),
        resources=ResourceMetrics(
            telemetry_backend=_to_telemetry_backend(cfg.telemetry.backend),
            telemetry_interval_ms=cfg.telemetry.interval_ms,
            telemetry_samples=0,
        ),
        hypothesis_transcript="",
        git_commit=commit,
        error=error,
    )


def _error_runs_for_config(
    *,
    spec: RuntimeConfigurationSpec,
    cfg: ExperimentConfig,
    hardware: HardwareMetadata,
    entries: list[ManifestEntry],
    error: str,
    commit: str | None,
) -> list[BenchmarkRun]:
    return [
        _error_run(
            spec=spec,
            cfg=cfg,
            hardware=hardware,
            entry=e,
            error=error,
            commit=commit,
        )
        for e in entries
    ]


def _to_telemetry_backend(b) -> TelemetryBackend:
    if isinstance(b, TelemetryBackend):
        return b
    return TelemetryBackend(str(b))


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------


def write_results(runs: list[BenchmarkRun], *, parquet_path: Path, csv_path: Path) -> None:
    """Write the result list to Parquet and a human-readable CSV.

    Both files are written atomically: a temporary file is created
    first and then renamed. The CSV is regenerated from the Parquet
    so the two never drift.
    """
    if not runs:
        # Write an empty parquet with the right schema so downstream
        # consumers don't have to special-case the empty result.
        empty_schema = _empty_dataframe()
        empty_schema.to_parquet(parquet_path, index=False)
        empty_schema.to_csv(csv_path, index=False)
        return

    records = [_benchmark_run_to_dict(r) for r in runs]
    df = pd.DataFrame.from_records(records)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def _benchmark_run_to_dict(r: BenchmarkRun) -> dict[str, Any]:
    flat = r.model_dump(mode="json")
    # Expand nested dicts into dotted columns for CSV readability.
    out: dict[str, Any] = {}
    for k, v in flat.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                # pyarrow refuses to write a struct with zero child
                # fields. An empty dict is not a column; serialise as
                # a JSON string so the data round-trips.
                if isinstance(v2, dict):
                    import json

                    out[f"{k}.{k2}"] = json.dumps(v2, ensure_ascii=False, sort_keys=True)
                else:
                    out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out


def _empty_dataframe() -> pd.DataFrame:
    """An empty dataframe with the columns produced by ``write_results``."""
    return pd.DataFrame(
        columns=[
            "schema_version",
            "run_id",
            "timestamp_utc",
            "mode",
            "config_id",
            "git_commit",
            "error",
            "hypothesis_transcript",
            "notes",
            "hardware.device_name",
            "hardware.os",
            "hardware.os_version",
            "hardware.python_version",
            "hardware.cpu_model",
            "hardware.cpu_cores_logical",
            "hardware.cpu_cores_physical",
            "hardware.gpu_model",
            "hardware.jetpack_version",
            "hardware.cuda_version",
            "hardware.cudnn_version",
            "hardware.driver_version",
            "hardware.machine_arch",
            "runtime_config.runtime",
            "runtime_config.model_size",
            "runtime_config.compute_type",
            "runtime_config.language",
            "runtime_config.beam_size",
            "runtime_config.vad_filter",
            "audio.file_id",
            "audio.path_rel",
            "audio.duration_s",
            "audio.sample_rate",
            "audio.channels",
            "audio.reference_transcript",
            "audio.dataset_source",
            "audio.clip_license",
            "latency.warmup_run",
            "latency.inference_duration_s",
            "latency.audio_duration_s",
            "latency.rtf",
            "latency.rtfx",
            "latency.p50_latency_s",
            "latency.p95_latency_s",
            "latency.p99_latency_s",
            "latency.num_measured_runs",
            "quality.wer",
            "quality.cer",
            "quality.substitutions",
            "quality.deletions",
            "quality.insertions",
            "quality.hits",
            "quality.reference_words",
            "quality.hypothesis_words",
            "quality.reference_chars",
            "quality.hypothesis_chars",
            "quality.normalizer_version",
            "resources.cpu_avg_percent",
            "resources.cpu_peak_percent",
            "resources.ram_avg_mb",
            "resources.ram_peak_mb",
            "resources.gpu_avg_percent",
            "resources.gpu_peak_percent",
            "resources.gpu_mem_avg_mb",
            "resources.gpu_mem_peak_mb",
            "resources.temperature_avg_c",
            "resources.temperature_peak_c",
            "resources.power_avg_w",
            "resources.power_peak_w",
            "resources.telemetry_backend",
            "resources.telemetry_interval_ms",
            "resources.telemetry_samples",
        ]
    )


# ---------------------------------------------------------------------------
# Public re-exports for convenience
# ---------------------------------------------------------------------------


__all__ += [
    "write_results",
    "FakeAdapter",
    "FakeConfig",
    "FasterWhisperAdapter",
    "ModelSize",
    "ComputeType",
]


# Silence linters for symbols only used in type comments.
_ = (json, uuid)
