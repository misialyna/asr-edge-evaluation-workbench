"""Pydantic v2 schemas for the ASR Edge Evaluation Workbench.

These models are the single source of truth for:

* the shape of an experiment configuration (loaded from YAML);
* the shape of every benchmark run that lands in the result store;
* the validation rules applied at every system boundary.

Design choices:

* ``extra="forbid"`` on every model so typos in YAML or Parquet
  surface as ``ValidationError`` rather than silent drift.
* All optional fields default to ``None`` — telemetry fields are
  never invented; if a tegrastats field is missing the value is
  ``None``, not a zero, not a guess.
* ``schema_version`` is stamped into every ``BenchmarkRun`` so that
  downstream consumers can reject rows from incompatible runs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"
NORMALIZER_VERSION: str = "pl-v1.0"

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Mode(str, Enum):
    """Operating mode of a benchmark run."""

    JETSON = "jetson"
    DEMO = "demo"


class RuntimeName(str, Enum):
    """The ASR runtime backing a configuration."""

    FASTER_WHISPER = "faster-whisper"


class ComputeType(str, Enum):
    """Weight quantization / precision mode of a CTranslate2 model."""

    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class ModelSize(str, Enum):
    """Whisper model sizes (faster-whisper naming)."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"
    DISTIL_LARGE_V3 = "distil-large-v3"


class TelemetryBackend(str, Enum):
    """Which telemetry backend produced the resource measurements."""

    TEGRASTATS = "tegrastats"
    PSUTIL = "psutil"
    NONE = "none"


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


class HardwareMetadata(BaseModel):
    """Snapshot of the hardware and OS at the time of a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    device_name: str = Field(min_length=1)
    os: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    cpu_model: str | None = None
    cpu_cores_logical: int | None = Field(default=None, ge=1)
    cpu_cores_physical: int | None = Field(default=None, ge=1)
    gpu_model: str | None = None
    jetpack_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    driver_version: str | None = None
    machine_arch: str | None = None  # e.g. "aarch64", "x86_64"


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Configuration of a single ASR inference path."""

    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeName
    model_size: ModelSize
    compute_type: ComputeType
    language: str = Field(default="pl", min_length=1, max_length=8)
    beam_size: int = Field(default=1, ge=1, le=10)
    vad_filter: bool = False
    decode_options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AudioMetadata(BaseModel):
    """Information about the audio file used in a single inference run."""

    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    path_rel: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1)
    reference_transcript: str
    dataset_source: str = Field(min_length=1)
    clip_license: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class LatencyMetrics(BaseModel):
    """Per-run latency summary.

    Definitions:

    * ``inference_duration_s`` — wall-clock time of a single inference
      call, including the model's own pre/post-processing.
    * ``rtf = inference_duration_s / audio_duration_s`` — lower is
      better. ``rtf < 1`` means the inference was faster than the audio
      played in real time.
    * ``rtfx = audio_duration_s / inference_duration_s`` — higher is
      better. ``rtfx > 1`` is the standard "faster than real time"
      definition used by the Open ASR Leaderboard.
    * ``p50`` / ``p95`` / ``p99_latency_s`` — the corresponding
      percentiles of the N measured runs aggregated into this row.
    """

    model_config = ConfigDict(extra="forbid")

    warmup_run: bool
    inference_duration_s: float = Field(ge=0)
    audio_duration_s: float = Field(gt=0)
    rtf: float = Field(ge=0)
    rtfx: float = Field(ge=0)
    p50_latency_s: float = Field(ge=0)
    p95_latency_s: float = Field(ge=0)
    p99_latency_s: float | None = Field(default=None, ge=0)
    num_measured_runs: int = Field(ge=1)


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class QualityMetrics(BaseModel):
    """Word- and character-level error metrics for a single inference."""

    model_config = ConfigDict(extra="forbid")

    wer: float = Field(ge=0)
    cer: float = Field(ge=0)
    substitutions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    insertions: int = Field(ge=0)
    hits: int = Field(ge=0)
    reference_words: int = Field(ge=0)
    hypothesis_words: int = Field(ge=0)
    reference_chars: int = Field(ge=0)
    hypothesis_chars: int = Field(ge=0)
    normalizer_version: str = Field(default=NORMALIZER_VERSION, min_length=1)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class ResourceMetrics(BaseModel):
    """Per-run resource and thermal summary.

    Every numeric field is optional and may be ``None`` if the chosen
    telemetry backend did not report it. The presence of a value, not
    a non-zero value, is the only signal that the metric was
    available.
    """

    model_config = ConfigDict(extra="forbid")

    cpu_avg_percent: float | None = Field(default=None, ge=0, le=100)
    cpu_peak_percent: float | None = Field(default=None, ge=0, le=100)
    ram_avg_mb: float | None = Field(default=None, ge=0)
    ram_peak_mb: float | None = Field(default=None, ge=0)
    gpu_avg_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_peak_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_mem_avg_mb: float | None = Field(default=None, ge=0)
    gpu_mem_peak_mb: float | None = Field(default=None, ge=0)
    temperature_avg_c: float | None = None
    temperature_peak_c: float | None = None
    power_avg_w: float | None = Field(default=None, ge=0)
    power_peak_w: float | None = Field(default=None, ge=0)
    telemetry_backend: TelemetryBackend
    telemetry_interval_ms: int | None = Field(default=None, ge=1)
    telemetry_samples: int | None = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


class BenchmarkRun(BaseModel):
    """One row of the result store.

    One ``BenchmarkRun`` corresponds to a single inference call
    (or to a single set of N aggregated runs against the same audio
    file with the same configuration, in which case the latency and
    quality fields are aggregated).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    timestamp_utc: datetime
    mode: Mode
    config_id: str = Field(min_length=1)
    hardware: HardwareMetadata
    runtime_config: RuntimeConfig
    audio: AudioMetadata
    latency: LatencyMetrics
    quality: QualityMetrics
    resources: ResourceMetrics
    hypothesis_transcript: str = ""
    git_commit: str | None = None
    error: str | None = None
    notes: str | None = None

    @field_validator("git_commit")
    @classmethod
    def _shorten_commit(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip()[:40]


# ---------------------------------------------------------------------------
# Experiment configuration (YAML)
# ---------------------------------------------------------------------------


class RuntimeConfigurationSpec(BaseModel):
    """A single configuration entry inside an experiment config."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    model_size: ModelSize
    compute_type: ComputeType
    language: str = Field(default="pl", min_length=1, max_length=8)
    beam_size: int = Field(default=1, ge=1, le=10)
    vad_filter: bool = False
    decode_options: dict[str, Any] = Field(default_factory=dict)


class RuntimeSpec(BaseModel):
    """Runtime section of an experiment config."""

    model_config = ConfigDict(extra="forbid")

    type: RuntimeName
    configurations: list[RuntimeConfigurationSpec] = Field(min_length=1)

    @field_validator("configurations")
    @classmethod
    def _unique_ids(cls, v: list[RuntimeConfigurationSpec]) -> list[RuntimeConfigurationSpec]:
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate configuration ids: {duplicates}")
        return v


class DatasetSpec(BaseModel):
    """Dataset section of an experiment config."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: str = Field(min_length=1)
    audio_root: str = Field(min_length=1)
    max_files: int | None = Field(default=None, ge=1)
    shuffle: bool = False
    seed: int = 42


class ExecutionSpec(BaseModel):
    """Execution section of an experiment config."""

    model_config = ConfigDict(extra="forbid")

    warmup_runs: int = Field(default=1, ge=0)
    measured_runs_per_file: int = Field(default=3, ge=1)
    inter_run_cooldown_s: float = Field(default=0.0, ge=0)
    timeout_per_file_s: float = Field(default=600.0, gt=0)
    fail_fast: bool = False
    use_fake_runtime_if_demo: bool = True


class TelemetrySpec(BaseModel):
    """Telemetry section of an experiment config.

    The ``backend`` is tied to the experiment mode: ``tegrastats`` for
    real Jetson runs, ``psutil`` for demo runs. The validator below
    enforces that.
    """

    model_config = ConfigDict(extra="forbid")

    backend: TelemetryBackend
    interval_ms: int = Field(default=500, ge=50, le=60_000)
    tegrastats_path: str = Field(default="/usr/bin/tegrastats", min_length=1)
    raw_output_dir: str = Field(default="results/runs/telemetry", min_length=1)


class OutputSpec(BaseModel):
    """Output section of an experiment config."""

    model_config = ConfigDict(extra="forbid")

    results_parquet: str = Field(default="results/runs/results.parquet", min_length=1)
    results_csv: str = Field(default="results/runs/results.csv", min_length=1)
    report_md: str = Field(default="results/reports/benchmark_report.md", min_length=1)
    figures_dir: str = Field(default="results/figures", min_length=1)


class ExperimentMetadata(BaseModel):
    """Top-level metadata for an experiment config."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    mode: Mode
    language: str = Field(default="pl", min_length=1, max_length=8)
    schema_version: Literal["1.0"] = SCHEMA_VERSION


class ExperimentConfig(BaseModel):
    """The complete shape of an experiment YAML file."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentMetadata
    dataset: DatasetSpec
    runtime: RuntimeSpec
    execution: ExecutionSpec
    telemetry: TelemetrySpec
    output: OutputSpec

    @field_validator("telemetry")
    @classmethod
    def _telemetry_matches_mode(cls, v: TelemetrySpec, info: Any) -> TelemetrySpec:
        # Cross-field validation against ``experiment.mode``.
        mode: Mode | None = None
        try:
            exp = info.data.get("experiment") if isinstance(info.data, dict) else None
            if exp is not None:
                mode = getattr(exp, "mode", None)
        except Exception:
            mode = None
        if mode == Mode.JETSON and v.backend not in (TelemetryBackend.TEGRASTATS,):
            raise ValueError(
                f"telemetry.backend must be 'tegrastats' when mode=jetson, got {v.backend!r}"
            )
        if mode == Mode.DEMO and v.backend not in (TelemetryBackend.PSUTIL, TelemetryBackend.NONE):
            raise ValueError(
                f"telemetry.backend must be 'psutil' or 'none' when mode=demo, got {v.backend!r}"
            )
        return v


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class ConfigSummary(BaseModel):
    """Per-configuration aggregate used by the recommender."""

    model_config = ConfigDict(extra="forbid")

    config_id: str
    wer_mean: float
    wer_std: float
    rtfx_mean: float
    rtfx_std: float
    ram_peak_mb: float | None
    gpu_mem_peak_mb: float | None
    temperature_peak_c: float | None
    power_peak_w: float | None
    num_runs: int = Field(ge=1)


class RecommendationWeights(BaseModel):
    """Weights for the rule-based recommendation scorer."""

    model_config = ConfigDict(extra="forbid")

    wer: float = Field(default=0.5, ge=0, le=1)
    rtfx: float = Field(default=0.3, ge=0, le=1)
    ram: float = Field(default=0.2, ge=0, le=1)

    @field_validator("wer", "rtfx", "ram")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        # Field already enforces ge=0 in Pydantic; this is a guardrail
        # so the object can never carry a NaN.
        if v != v:
            raise ValueError("weight must not be NaN")
        return v


class Recommendations(BaseModel):
    """Scenario-based recommendations produced by the analysis layer."""

    model_config = ConfigDict(extra="forbid")

    real_time_config_id: str
    balanced_config_id: str
    quality_first_config_id: str
    real_time_score: float
    balanced_score: float
    quality_first_score: float
    weights: RecommendationWeights
    summaries: list[ConfigSummary]
    rationale: str = ""
