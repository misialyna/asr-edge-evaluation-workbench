# Results template (filled in after the real run)

This document is a placeholder for the actual numbers. After
running the benchmark on a real Jetson AGX Orin, copy this
file to `docs/results.md` and fill in the tables. The README
will then link to it.

## Per-configuration summary

| Config | Runs | WER (mean) | WER (std) | RTFx (mean) | RTFx (std) | Peak RAM (MB) | Peak GPU (MB) | Peak T (°C) | Peak P (W) |
|---|---|---|---|---|---|---|---|---|---|
| `small_int8` | … | … | … | … | … | … | … | … | … |
| `small_fp16` | … | … | … | … | … | … | … | … | … |
| `medium_fp16` | … | … | … | … | … | … | … | … | … |

## Recommendation

| Scenario | Recommended config | Reason |
|---|---|---|
| Real-time priority | … | … |
| Balanced | … | … |
| Quality-first | … | … |

## Hardware / software

| Field | Value |
|---|---|
| Device | Jetson AGX Orin |
| JetPack | … |
| CUDA | … |
| Python | … |
| faster-whisper | … |
| jiwer | … |
| Git commit | … |
| Manifest checksum | … |
