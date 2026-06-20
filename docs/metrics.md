# Metrics

Every metric in the project has a single, written definition.
This document is the canonical reference. If a number in the
report disagrees with the table below, **this document is
right** and the report is a bug.

## Quality

| Field | Definition | Range |
|---|---|---|
| `quality.wer` | Word Error Rate from `jiwer.process_words`, computed on the **normalized** reference and hypothesis. | [0, +∞) |
| `quality.cer` | Character Error Rate from `jiwer.process_characters`, same pipeline. | [0, +∞) |
| `quality.substitutions` | Number of word substitutions in the alignment. | N₀ |
| `quality.deletions` | Number of words in the reference with no match in the hypothesis. | N₀ |
| `quality.insertions` | Number of words in the hypothesis with no match in the reference. | N₀ |
| `quality.hits` | Number of correctly aligned words. | N₀ |
| `quality.reference_words` | Word count of the normalized reference. | N₀ |
| `quality.hypothesis_words` | Word count of the normalized hypothesis. | N₀ |
| `quality.reference_chars` | Character count of the normalized reference (whitespace included). | N₀ |
| `quality.hypothesis_chars` | Character count of the normalized hypothesis. | N₀ |
| `quality.normalizer_version` | The version of `normalize_pl` used. Bumped on every pipeline change. | string |

### Normalizer

`normalize_pl` (version `pl-v1.0`) applies, in order:

1. Unicode NFC normalization.
2. Whitespace collapse + trim.
3. Lowercase.
4. Strip non-word punctuation (commas, periods, quotes, etc.).
5. Keep intra-word hyphens and apostrophes.
6. Keep Polish diacritics (ąćęłńóśźż).

A change to the normalizer is a **breaking** change to WER
comparability. Bump the version, document the diff, and treat
old numbers as not directly comparable.

## Latency

| Field | Definition | Range |
|---|---|---|
| `latency.warmup_run` | Always `False` in the row; reserved for per-run granularity. | bool |
| `latency.inference_duration_s` | Mean of the N measured inference durations. | [0, +∞) |
| `latency.audio_duration_s` | Duration of the audio input, read from the WAV header. | (0, +∞) |
| `latency.rtf` | `inference_duration_s / audio_duration_s`. | [0, +∞) |
| `latency.rtfx` | `audio_duration_s / inference_duration_s`. | [0, +∞) |
| `latency.p50_latency_s` | 50th percentile of N measured inference durations. | [0, +∞) |
| `latency.p95_latency_s` | 95th percentile of N measured inference durations. | [0, +∞) |
| `latency.p99_latency_s` | 99th percentile, `None` if N < 20. | [0, +∞) ∪ {None} |
| `latency.num_measured_runs` | The N above. | N₀ |

We use the convention that `RTFx > 1` means "faster than real
time" (matching the Open ASR Leaderboard). The conventional
`RTF < 1` is also stored for readers used to that framing.

## Resources

Every field is `None` if the chosen backend did not report it.
The presence of a value, not a non-zero value, is the only
signal that the metric is available.

| Field | Definition | Source | Unit |
|---|---|---|---|
| `resources.cpu_avg_percent` | Mean across the run of the per-line CPU averages. | tegrastats, psutil | % |
| `resources.cpu_peak_percent` | Max of per-line per-core CPU peaks. | tegrastats, psutil | % |
| `resources.ram_avg_mb` | Mean used RAM across the run. | tegrastats, psutil | MB |
| `resources.ram_peak_mb` | Max used RAM across the run. | tegrastats, psutil | MB |
| `resources.gpu_avg_percent` | Mean GPU utilisation. | tegrastats, nvidia-smi (debug) | % |
| `resources.gpu_peak_percent` | Peak GPU utilisation. | tegrastats, nvidia-smi (debug) | % |
| `resources.gpu_mem_avg_mb` | Mean used GPU memory. | tegrastats, nvidia-smi (debug) | MB |
| `resources.gpu_mem_peak_mb` | Peak used GPU memory. | tegrastats, nvidia-smi (debug) | MB |
| `resources.temperature_avg_c` | Mean board temperature. | tegrastats | °C |
| `resources.temperature_peak_c` | Peak board temperature. | tegrastats | °C |
| `resources.power_avg_w` | Mean total board power. | tegrastats (Orin) | W |
| `resources.power_peak_w` | Peak total board power. | tegrastats (Orin) | W |
| `resources.telemetry_backend` | Which backend produced the data. | enum | string |
| `resources.telemetry_interval_ms` | Sampling interval. | config | ms |
| `resources.telemetry_samples` | Number of samples aggregated into the row. | collector | N₀ |

## Recommendation

| Field | Definition |
|---|---|
| `real_time_config_id` | Config with `mean(RTFx) ≥ 1.5` and lowest WER; fallback to fastest available if none qualify. |
| `balanced_config_id` | Config minimizing the weighted score. |
| `quality_first_config_id` | Config with the lowest WER. |
| `weights` | The three weights used by the balanced scorer. |
| `rationale` | Human-readable explanation. |

The weights default to `wer=0.5, rtfx=0.3, ram=0.2` and are
loaded from `configs/recommend_weights.yaml`. Any positive
weights that sum (after normalization) to 1 are valid.
