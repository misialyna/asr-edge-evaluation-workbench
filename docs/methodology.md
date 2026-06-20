# Methodology

This document describes the exact experimental protocol used
by the orchestrator. Every number in the report is reproducible
from the inputs listed here.

## 1. Dataset

* **Source**: Mozilla Common Voice Polish v11.0 (CC0).
* **Selection**: 20 short clips (2–10 s) drawn from the
  `validated` split, no demographic filtering, no noise
  augmentation.
* **License**: CC0-1.0. The manifest records `dataset_source`
  and `clip_license` per row.
* **Format**: 16 kHz mono PCM-16 WAV.
* **Manifest**: JSONL at `data/manifests/benchmark_manifest.jsonl`,
  with a SHA-256 checksum in
  `data/manifests/manifest_checksum.txt` and a version line in
  `data/manifests/manifest_version.txt`.

See `docs/decisions/0002-dataset-choice.md` for the rationale.

## 2. Configurations

* `small_int8`: Faster-Whisper `small`, `compute_type=int8`.
* `small_fp16`: Faster-Whisper `small`, `compute_type=float16`.
* `medium_fp16`: Faster-Whisper `medium`, `compute_type=float16`.

All three use `beam_size=1`, `vad_filter=false`, and the
default `decode_options={}`. Greedy decoding is used on
purpose: it makes latency comparisons fairer across
configurations and keeps the runs deterministic for a
given hardware and software stack.

## 3. Run protocol

For each (configuration, audio file):

1. **Warmup**: 1 unrecorded inference to absorb first-call
   overhead (model load, CTranslate2 workspace allocation,
   CUDA kernel JIT if any).
2. **Measured runs**: 3 inferences. The orchestrator records
   per-run latency from inside the runtime, and telemetry
   (CPU/RAM/GPU/temperature/power) from the
   `TelemetryCollector` covering each call.
3. **Aggregation**: per-run durations are aggregated into
   `avg`, `p50`, `p95`, `p99` (when N ≥ 20), min, max, std.
   The row stored in the result store carries the
   aggregated metrics, **not** the individual samples.
4. **Cooldown**: an optional `inter_run_cooldown_s` (default
   0) between measured runs. Useful on Jetson to let
   thermals settle.

## 4. Telemetry

* **Jetson mode**: `tegrastats` with a 500 ms sampling
  interval. The collector spawns it as a child process
  group and terminates it cleanly after the run. Raw
  per-run logs are written to
  `results/runs/telemetry/{run_id}.log`.
* **Demo mode**: `psutil` polling for CPU/RAM, no GPU,
  no power. Missing fields are reported as `n/a` in the
  output, never as a fabricated zero.

## 5. Metric definitions

* **WER** (`quality.wer`): word-level error rate from
  `jiwer`, computed on the *normalized* reference and
  hypothesis. Normalization preserves diacritics.
* **CER** (`quality.cer`): character-level error rate,
  same pipeline.
* **RTF** (`latency.rtf`): `inference_duration_s / audio_duration_s`.
  Lower is better; `<1` means faster than real time.
* **RTFx** (`latency.rtfx`): `audio_duration_s / inference_duration_s`.
  Higher is better; `>1` means faster than real time.
* **P95 inference duration** (`latency.p95_latency_s`):
  95th percentile of the N measured inference durations
  for one (config, audio) row.
* **Resource metrics**: mean and peak across the run for
  CPU, RAM, GPU%, GPU memory, temperature, and power. See
  `docs/metrics.md` for the exact field list.

## 6. Recommendation

* **Real-time priority**: configuration with
  `mean(RTFx) ≥ 1.5` and the lowest WER. If none qualify,
  fall back to the fastest available and surface that in
  the rationale.
* **Balanced**: minimize
  `w_wer * norm(wer) + w_rtfx * penalty(rtfx) + w_ram * norm(ram_peak_mb)`,
  weights from `configs/recommend_weights.yaml` (defaults
  `wer=0.5, rtfx=0.3, ram=0.2`).
* **Quality-first**: configuration with the lowest WER.

The recommendation is **never** a black-box model. The
weights are visible in the report, the rationale is
generated, and the scorer is unit-tested.

## 7. Reproducing a run

```bash
uv sync --extra dev --extra dashboard --extra asr --extra hf
python -m data.scripts.prepare_polish_eval --max-clips 20
python -m scripts.run_benchmark --config configs/experiments/jetson_orin_default.yaml
python -m scripts.build_report --results results/runs/results.parquet
```

The git commit hash and the manifest checksum are recorded
in the result file; a future run with the same hash,
manifest, and software versions is expected to be within
the noise of the original.
