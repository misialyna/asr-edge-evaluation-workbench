# Architecture

The project is intentionally layered so that each component can
be replaced or extended without touching the others. The
end-to-end flow is:

```
YAML experiment config
   │
   ▼
[config] load + validate (Pydantic v2)
   │
   ▼
[data.manifest] load frozen JSONL
   │
   ▼
┌─ for each (config_id, audio_file) ─────────────────────────────┐
│                                                              │
│  [runtime].load()                                            │
│  │                                                           │
│  ├─ warmup × N (discarded)                                  │
│  │                                                           │
│  ├─ measured × N:                                            │
│  │   [telemetry.collector].start()                           │
│  │   [runtime].transcribe(audio)                             │
│  │   [telemetry.collector].stop()                            │
│  │   [metrics].normalize → wer/cer/rtf/latency              │
│  │   append [schemas.BenchmarkRun]                           │
│  │                                                           │
│  └─ [runtime].unload()                                       │
│                                                              │
│  [result_store] parquet + CSV + JSON sidecar                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
[analysis.recommend] rule-based scoring (weights from YAML)
   │
   ▼
[reporting] Markdown via Jinja2 + matplotlib PNGs
   │
   ▼
[app] Streamlit dashboard
```

## Component responsibilities

| Component | Responsibility |
|---|---|
| `config` | Load and validate the experiment YAML against the Pydantic schema. |
| `schemas` | Single source of truth for the run row, the experiment config, and the recommendations. |
| `runtime` | `AsrRuntime` Protocol with two implementations: `FakeAdapter` and `FasterWhisperAdapter`. |
| `telemetry` | `tegrastats` parser, `psutil` poller, and a `TelemetryCollector` lifecycle. |
| `data` | Manifest loader and Polish dataset prep script. |
| `metrics` | Polish text normalizer, WER, CER, RTF, RTFx, latency aggregator. |
| `orchestrator` | The main loop: warm-up + measured runs, fail-soft, structured result rows. |
| `analysis` | Rule-based recommender with explicit weights. |
| `reporting` | Jinja2-rendered Markdown report and matplotlib figures. |
| `app` | Streamlit dashboard with four views. |

## Why this shape

- **Single source of truth for shape**: every system boundary
  (YAML, Parquet, Markdown template, dashboard table) reads
  from a Pydantic model. There is no "second definition of a
  result row" anywhere.
- **Tolerance in the parser, strictness at the schema**:
  `tegrastats` output drifts; the parser tolerates that, but
  once a row enters the result store it has to match the
  schema exactly.
- **Reproducibility is a row-level property**, not a
  project-level slogan: `git_commit`, `schema_version`,
  `normalizer_version`, and `manifest_checksum` (to be added
  to the manifest header in v1.1) are stamped into every
  result row.
