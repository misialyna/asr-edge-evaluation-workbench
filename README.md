# ASR Edge Evaluation Workbench

Narzędzie do benchmarkowania modeli ASR (Automatic Speech Recognition) na edge hardware — zaprojektowane dla **NVIDIA Jetson Orin** z fallbackiem na zwykły sprzęt via `psutil`. Mierzy WER, CER, RTF, latencję i zużycie zasobów (RAM, GPU, temperatura, moc), po czym generuje raport Markdown i dashboard Streamlit z automatyczną rekomendacją konfiguracji.

Projekt powiązany z robotem [AIWAT (WATUS)](https://github.com/misialyna/watus_project) — workbench służył do wyznaczenia optymalnej konfiguracji Whisper dla systemu voice.

---

## Architektura

```
YAML experiment config
       │
       ▼
[config] Pydantic v2 load + validate
       │
       ▼
[data.manifest] frozen JSONL manifest
       │
       ▼
  dla każdego (config_id, audio_file):
    [runtime].load()
    ├─ warmup × N  (odrzucone)
    └─ measured × N:
        [telemetry].start()
        [runtime].transcribe(audio)
        [telemetry].stop()
        [metrics] normalize → WER/CER/RTF/latency
        append BenchmarkRun row
    [runtime].unload()
    zapis: Parquet + CSV + JSON sidecar
       │
       ▼
[analysis.recommend]  rule-based scoring (wagi z YAML)
       │
       ▼
[reporting]  Markdown (Jinja2) + matplotlib PNG
       │
       ▼
[app]  Streamlit dashboard
```

## Struktura projektu

```
├── app/
│   └── streamlit_app.py           # Dashboard (4 widoki)
├── configs/
│   ├── benchmark/                 # Konfiguracje modeli (small_int8, small_fp16, medium_fp16)
│   ├── dataset/                   # Konfiguracja datasetu (Polish eval v1)
│   ├── experiments/               # Eksperymenty (demo, jetson_orin_default)
│   └── recommend_weights.yaml     # Wagi rekomendacji (WER·0.5, RTFx·0.3, RAM·0.2)
├── data/
│   ├── manifests/                 # Zamrożony JSONL manifest z checksumą
│   └── scripts/                   # Skrypty przygotowania datasetu
├── docs/
│   ├── decisions/                 # ADR 0001–0005 (Architecture Decision Records)
│   ├── architecture.md
│   ├── methodology.md
│   ├── metrics.md
│   └── limitations.md
├── notebooks/
│   ├── 01_explore_results.ipynb
│   ├── 02_quality_vs_latency.ipynb
│   └── 03_resource_tradeoffs.ipynb
├── results/
│   ├── figures/                   # WER vs RTFx, latency box, resource bars
│   └── reports/benchmark_report.md
├── scripts/
│   ├── run_benchmark.py           # Główny skrypt benchmarkowy
│   ├── build_report.py
│   ├── collect_tegrastats.py
│   ├── parse_tegrastats.py
│   └── launch_dashboard.py
├── src/asr_edge_eval/
│   ├── config.py                  # Ładowanie i walidacja YAML
│   ├── schemas.py                 # Pydantic — single source of truth
│   ├── analysis/recommend.py      # Rule-based recommender
│   ├── data/                      # Manifest loader, dataset helpers
│   ├── metrics/                   # WER, CER, RTF, RTFx, latency, normalizer
│   ├── orchestrator/runner.py     # Główna pętla: warmup + measured runs
│   ├── reporting/                 # Jinja2 Markdown + matplotlib
│   ├── runtime/                   # AsrRuntime Protocol, FasterWhisperAdapter, FakeAdapter
│   ├── telemetry/                 # tegrastats parser, psutil fallback, TelemetryCollector
│   └── utils/                     # git, logging, versions
└── tests/
    └── fixtures/tegrastats_samples/  # Przykładowe logi tegrastats (JetPack 5 & 6)
```

## Wyniki benchmarku (demo mode, CPU)

| Konfiguracja | WER (mean) | RTFx (mean) | Peak RAM |
|---|---|---|---|
| `small_int8` | 1.000 | 6.28× | 164 MB |
| `small_fp16` | 1.000 | 6.28× | 164 MB |
| `medium_fp16` | 1.000 | 6.28× | 169 MB |

Rekomendacja wag: WER=0.5 · RTFx=0.3 · RAM=0.2.  
Pełny raport: [`results/reports/benchmark_report.md`](results/reports/benchmark_report.md)

## Uruchomienie

```bash
# Instalacja (uv)
uv sync --extra dev --extra dashboard

# Demo (bez Jetsona)
uv run python -m scripts.run_benchmark \
  --config configs/experiments/demo_mode.yaml

# Pełny benchmark na Jetsonie
uv run python -m scripts.run_benchmark \
  --config configs/experiments/jetson_orin_default.yaml

# Raport
uv run python -m scripts.build_report \
  --results results/runs/results.parquet \
  --out results/reports/benchmark_report.md

# Dashboard
uv run python -m scripts.launch_dashboard
```

## Metryki

| Metryka | Opis |
|---|---|
| **WER** | Word Error Rate — główna miara jakości transkrypcji |
| **CER** | Character Error Rate |
| **RTF** | Real-Time Factor — czas przetwarzania / długość audio |
| **RTFx** | 1/RTF — ile razy szybciej niż real-time (>1.0 = real-time capable) |
| **Latency** | Czas od początku do końca jednej transkrypcji |

## Telemetria

- **Jetson Orin** — `tegrastats` (RAM, GPU, temperatura, moc)
- **Inne platformy** — `psutil` fallback (RAM CPU, brak GPU/temp)

Parser `tegrastats` toleruje różnice między JetPack 5 i JetPack 6.

## Dataset

Polski zestaw ewaluacyjny oparty na Mozilla Common Voice Polish (CC0-1.0), zamrożony jako JSONL manifest z checksumą. Skrypt `data/scripts/prepare_polish_eval.py` pozwala odtworzyć podzbiór.

## ADR (Architecture Decision Records)

| Nr | Decyzja |
|---|---|
| [0001](docs/decisions/0001-faster-whisper-as-baseline.md) | Faster-Whisper jako baseline runtime |
| [0002](docs/decisions/0002-dataset-choice.md) | Wybór datasetu |
| [0003](docs/decisions/0003-three-config-comparison.md) | Porównanie 3 konfiguracji |
| [0004](docs/decisions/0004-tegrastats-over-nvidia-smi.md) | tegrastats zamiast nvidia-smi |
| [0005](docs/decisions/0005-recommendation-rule-based.md) | Rule-based recommender |

## Zależności

`faster-whisper` · `pydantic>=2` · `jinja2` · `matplotlib` · `streamlit` · `pandas` · `pyarrow` · `jiwer` · `psutil`
