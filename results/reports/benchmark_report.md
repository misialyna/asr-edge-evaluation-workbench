# ASR Edge Evaluation Workbench — Benchmark Report

_Generated: 2026-06-20T14:07:41+00:00_

## 1. Run metadata

| Field | Value |
|---|---|
| Project | asr-edge-evaluation-workbench |
| Schema version | 1.0 |
| Run timestamp (UTC) | 2026-06-20T14:06:49+00:00 |
| Mode | demo |
| Hardware | cloud-compute-tjfpt (x86_64) |
| OS | Linux 5.10.134-18.0.11.lifsea8.x86_64 |
| Python | 3.11.2 |
| CPU | Intel(R) Xeon(R) Platinum (1 logical cores) |
| GPU | n/a |
| JetPack | n/a |
| CUDA | n/a |
| Git commit | n/a |

## 2. Recommendation

small_int8 qualifies for real-time (mean RTFx=6.27 ≥ 1.5) with the lowest WER (1.000). small_fp16 minimises the weighted score (wer=0.50, rtfx=0.30, ram=0.20). small_int8 has the lowest WER (1.000), suitable for batch / quality-first use.

| Scenario | Recommended config | WER (mean) | RTFx (mean) | Peak RAM (MB) | Peak temp (°C) |
|---|---|---|---|---|---|
| Real-time priority | `small_int8` | 1.000 | 6.275 | 163.582 | n/a |
| Balanced | `small_fp16` | 1.000 | 6.278 | 163.711 | n/a |
| Quality-first | `small_int8` | 1.000 | 6.275 | 163.582 | n/a |

Recommendation weights: WER=0.5 · RTFx=0.3 · RAM=0.2.

## 3. Per-configuration summary

| Config | Runs | WER (mean) | WER (std) | RTFx (mean) | RTFx (std) | Peak RAM (MB) | Peak GPU (MB) | Peak T (°C) | Peak P (W) |
|---|---|---|---|---|---|---|---|---|---|
| `small_int8` | 10 | 1.000 | 0.000 | 6.275 | 1.530 | 163.582 | n/a | n/a | n/a |
| `small_fp16` | 10 | 1.000 | 0.000 | 6.278 | 1.529 | 163.711 | n/a | n/a | n/a |
| `medium_fp16` | 10 | 1.000 | 0.000 | 6.278 | 1.529 | 169.254 | n/a | n/a | n/a |


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