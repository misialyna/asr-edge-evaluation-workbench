# ASR Edge Evaluation Workbench — Benchmark Report

_Generated: 2026-07-14T05:57:47+00:00_

## 1. Run metadata

| Field | Value |
|---|---|
| Project | asr-edge-evaluation-workbench |
| Schema version | 1.0 |
| Run timestamp (UTC) | 2026-07-14T05:56:55+00:00 |
| Mode | demo |
| Hardware | MacBook-Pro-Michalina.local (arm64) |
| OS | Darwin 25.4.0 |
| Python | 3.13.13 |
| CPU | n/a (15 logical cores) |
| GPU | n/a |
| JetPack | n/a |
| CUDA | n/a |
| Git commit | 074511ccc2c6 |

## 2. Recommendation

small_int8 qualifies for real-time (mean RTFx=6.20 ≥ 1.5) with the lowest WER (1.000). medium_fp16 minimises the weighted score (wer=0.50, rtfx=0.30, ram=0.20). small_int8 has the lowest WER (1.000), suitable for batch / quality-first use.

| Scenario | Recommended config | WER (mean) | RTFx (mean) | Peak RAM (MB) | Peak temp (°C) |
|---|---|---|---|---|---|
| Real-time priority | `small_int8` | 1.000 | 6.196 | 23374.641 | n/a |
| Balanced | `medium_fp16` | 1.000 | 6.203 | 23290.406 | n/a |
| Quality-first | `small_int8` | 1.000 | 6.196 | 23374.641 | n/a |

Recommendation weights: WER=0.5 · RTFx=0.3 · RAM=0.2.

## 3. Per-configuration summary

| Config | Runs | WER (mean) | WER (std) | RTFx (mean) | RTFx (std) | Peak RAM (MB) | Peak GPU (MB) | Peak T (°C) | Peak P (W) |
|---|---|---|---|---|---|---|---|---|---|
| `small_int8` | 10 | 1.000 | 0.000 | 6.196 | 1.519 | 23374.641 | n/a | n/a | n/a |
| `small_fp16` | 10 | 1.000 | 0.000 | 6.187 | 1.503 | 23277.391 | n/a | n/a | n/a |
| `medium_fp16` | 10 | 1.000 | 0.000 | 6.203 | 1.499 | 23290.406 | n/a | n/a | n/a |


## 4. Quality vs real-time performance

![WER vs RTFx](wer_vs_rtfx.png)

The dashed line at RTFx=1.0 is the "faster than real time" boundary.
Configurations above the line are candidates for real-time use;
configurations below are suitable for batch processing.

## 5. Latency distribution

![Latency box plot](latency_box.png)

## 6. Resource usage

![Resource bars](resource_bars.png)

## 7. Limitations

- This run is based on **3 configurations** of one runtime
  (Faster-Whisper). WhisperTRT and other runtimes are not
  included in v1.
- The Polish evaluation set is a frozen subset of Mozilla Common
  Voice Polish; broader Polish coverage (e.g. PolEval) is planned
  for v2.
- Recommendation logic is **rule-based and transparent**; the
  weights are published above and editable in
  `configs/recommend_weights.yaml`. There is no learned ranker.
- Results are not directly comparable across different JetPack or
  CUDA versions — re-run on the target device.

## 8. Reproducing this report

```bash
uv sync --extra dev --extra dashboard
uv run python -m scripts.run_benchmark --config configs/experiments/demo_mode.yaml
uv run python -m scripts.build_report --results results/runs/results.parquet --out results/reports/benchmark_report.md
```

## 9. Credits and licenses

- Faster-Whisper: MIT
- jiwer: Apache-2.0 / MIT
- Mozilla Common Voice Polish: CC0-1.0
- Streamlit: Apache-2.0
- Pydantic: MIT
- See `docs/decisions/` for the project's own architecture decision records.