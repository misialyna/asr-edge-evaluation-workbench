"""Render the Markdown benchmark report from result rows.

The report is a single Markdown document with these sections:

1. Header (project, run, hardware, software versions, git commit).
2. Recommendation table (one row per scenario).
3. Per-configuration summary table.
4. Quality vs RTFx scatter.
5. Latency box plot.
6. Resource bar chart.
7. Limitations and honest caveats.
8. Reproduction instructions.

The Markdown is produced by a small Jinja2 template. Numeric
values are formatted with a small helper that converts ``inf`` and
``nan`` to ``n/a`` so the report never contains ``nan`` literals.
"""

from __future__ import annotations

import contextlib
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined, select_autoescape

from ..analysis.recommend import Recommendations, recommend
from ..schemas import SCHEMA_VERSION, BenchmarkRun
from .plots import plot_latency_box, plot_resource_bars, plot_wer_vs_rtfx

__all__ = ["render_markdown_report"]


_REPORT_TEMPLATE = """# ASR Edge Evaluation Workbench — Benchmark Report

_Generated: {{ generated_at }}_

## 1. Run metadata

| Field | Value |
|---|---|
| Project | asr-edge-evaluation-workbench |
| Schema version | {{ schema_version }} |
| Run timestamp (UTC) | {{ timestamp_utc }} |
| Mode | {{ mode }} |
| Hardware | {{ hardware.device_name }} ({{ hardware.machine_arch or 'unknown arch' }}) |
| OS | {{ hardware.os }} {{ hardware.os_version }} |
| Python | {{ hardware.python_version }} |
| CPU | {{ hardware.cpu_model or 'n/a' }} ({{ hardware.cpu_cores_logical or '?' }} logical cores) |
| GPU | {{ hardware.gpu_model or 'n/a' }} |
| JetPack | {{ hardware.jetpack_version or 'n/a' }} |
| CUDA | {{ hardware.cuda_version or 'n/a' }} |
| Git commit | {{ git_commit or 'n/a' }} |

## 2. Recommendation

{{ recommendations.rationale }}

| Scenario | Recommended config | WER (mean) | RTFx (mean) | Peak RAM (MB) | Peak temp (°C) |
|---|---|---|---|---|---|
| Real-time priority | `{{ recommendations.real_time_config_id }}` | {{ fmt(recommendations_summary['rt'].wer_mean) }} | {{ fmt(recommendations_summary['rt'].rtfx_mean) }} | {{ fmt(recommendations_summary['rt'].ram_peak_mb) }} | {{ fmt(recommendations_summary['rt'].temperature_peak_c) }} |
| Balanced | `{{ recommendations.balanced_config_id }}` | {{ fmt(recommendations_summary['bal'].wer_mean) }} | {{ fmt(recommendations_summary['bal'].rtfx_mean) }} | {{ fmt(recommendations_summary['bal'].ram_peak_mb) }} | {{ fmt(recommendations_summary['bal'].temperature_peak_c) }} |
| Quality-first | `{{ recommendations.quality_first_config_id }}` | {{ fmt(recommendations_summary['q'].wer_mean) }} | {{ fmt(recommendations_summary['q'].rtfx_mean) }} | {{ fmt(recommendations_summary['q'].ram_peak_mb) }} | {{ fmt(recommendations_summary['q'].temperature_peak_c) }} |

Recommendation weights: WER={{ recommendations.weights.wer }} · RTFx={{ recommendations.weights.rtfx }} · RAM={{ recommendations.weights.ram }}.

## 3. Per-configuration summary

| Config | Runs | WER (mean) | WER (std) | RTFx (mean) | RTFx (std) | Peak RAM (MB) | Peak GPU (MB) | Peak T (°C) | Peak P (W) |
|---|---|---|---|---|---|---|---|---|---|
{% for s in recommendations.summaries -%}
| `{{ s.config_id }}` | {{ s.num_runs }} | {{ fmt(s.wer_mean) }} | {{ fmt(s.wer_std) }} | {{ fmt(s.rtfx_mean) }} | {{ fmt(s.rtfx_std) }} | {{ fmt(s.ram_peak_mb) }} | {{ fmt(s.gpu_mem_peak_mb) }} | {{ fmt(s.temperature_peak_c) }} | {{ fmt(s.power_peak_w) }} |
{% endfor %}

## 4. Quality vs real-time performance

![WER vs RTFx]({{ figures.wer_vs_rtfx }})

The dashed line at RTFx=1.0 is the "faster than real time" boundary.
Configurations above the line are candidates for real-time use;
configurations below are suitable for batch processing.

## 5. Latency distribution

![Latency box plot]({{ figures.latency }})

## 6. Resource usage

![Resource bars]({{ figures.resources }})

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
"""


def _fmt(value: Any) -> str:
    """Format a numeric value for the report.

    ``inf``, ``nan`` and ``None`` all become ``n/a``. Floats are
    rendered with 3 decimals; ints as integers.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _summary_lookup(recs: Recommendations) -> dict[str, Any]:
    by_id = {s.config_id: s for s in recs.summaries}
    return {
        "rt": by_id[recs.real_time_config_id],
        "bal": by_id[recs.balanced_config_id],
        "q": by_id[recs.quality_first_config_id],
    }


def render_markdown_report(
    runs: list[BenchmarkRun],
    *,
    out_path: Path,
    figures_dir: Path,
    weights_path: Path | None = None,
    recommendations: Recommendations | None = None,
) -> Path:
    """Render the Markdown report and write it to ``out_path``.

    Parameters
    ----------
    runs:
        The result rows from one experiment.
    out_path:
        Where the Markdown file will be written.
    figures_dir:
        Where PNG figures will be written. A relative path is
        embedded in the report.
    weights_path:
        Optional path to a YAML file with recommendation weights.
    """
    if recommendations is None:
        recommendations = recommend(runs, weights_path=weights_path)
    summaries = _summary_lookup(recommendations)

    # Figures (best-effort: failures degrade to a missing figure).
    figures_rel = {
        "wer_vs_rtfx": "wer_vs_rtfx.png",
        "latency": "latency_box.png",
        "resources": "resource_bars.png",
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        plot_wer_vs_rtfx(runs, figures_dir / figures_rel["wer_vs_rtfx"])
    with contextlib.suppress(Exception):
        plot_latency_box(runs, figures_dir / figures_rel["latency"])
    with contextlib.suppress(Exception):
        plot_resource_bars(runs, figures_dir / figures_rel["resources"])

    # Pull the most-common hardware snapshot from the runs.
    hw = runs[0].hardware if runs else None
    mode = runs[0].mode.value if runs else "unknown"
    git_commit = runs[0].git_commit if runs else None

    env = Environment(autoescape=select_autoescape(["html", "xml"]), undefined=StrictUndefined)
    template = env.from_string(_REPORT_TEMPLATE)
    md = template.render(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        schema_version=SCHEMA_VERSION,
        timestamp_utc=runs[0].timestamp_utc.isoformat(timespec="seconds") if runs else "n/a",
        mode=mode,
        hardware=hw,
        git_commit=git_commit,
        recommendations=recommendations,
        recommendations_summary=summaries,
        figures=figures_rel,
        fmt=_fmt,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def runs_from_parquet(path: Path) -> list[BenchmarkRun]:
    """Load runs from a Parquet file written by the orchestrator.

    Used by the build_report CLI and the Streamlit dashboard.

    Order of preference:

    1. ``path.with_suffix(".json")`` sidecar (preserves nested
       objects exactly).
    2. The Parquet file itself, with the flattened schema
       reconstructed into ``BenchmarkRun`` rows.
    """
    import json

    json_sidecar = path.with_suffix(".json")
    if json_sidecar.is_file():
        raw = json.loads(json_sidecar.read_text(encoding="utf-8"))
        return [BenchmarkRun.model_validate(r) for r in raw]
    df = pd.read_parquet(path)
    return _runs_from_flat_df(df)


def _runs_from_flat_df(df: pd.DataFrame) -> list[BenchmarkRun]:
    """Best-effort reconstruction from the flattened Parquet/CSV columns.

    The Parquet and CSV are denormalized for human readability; we
    re-nest them to feed :class:`BenchmarkRun` validation.
    """
    out: list[BenchmarkRun] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {
            "schema_version": row.get("schema_version", SCHEMA_VERSION),
            "run_id": row["run_id"],
            "timestamp_utc": row["timestamp_utc"],
            "mode": row["mode"],
            "config_id": row["config_id"],
            "hypothesis_transcript": row.get("hypothesis_transcript", ""),
            "git_commit": row.get("git_commit") if pd.notna(row.get("git_commit")) else None,
            "error": row.get("error") if pd.notna(row.get("error")) else None,
            "notes": row.get("notes") if pd.notna(row.get("notes")) else None,
            "hardware": {
                k.replace("hardware.", ""): row.get(k)
                for k in df.columns
                if k.startswith("hardware.")
            },
            "runtime_config": {
                k.replace("runtime_config.", ""): row.get(k)
                for k in df.columns
                if k.startswith("runtime_config.")
            },
            "audio": {
                k.replace("audio.", ""): row.get(k) for k in df.columns if k.startswith("audio.")
            },
            "latency": {
                k.replace("latency.", ""): row.get(k)
                for k in df.columns
                if k.startswith("latency.")
            },
            "quality": {
                k.replace("quality.", ""): row.get(k)
                for k in df.columns
                if k.startswith("quality.")
            },
            "resources": {
                k.replace("resources.", ""): row.get(k)
                for k in df.columns
                if k.startswith("resources.")
            },
        }
        # Replace NaN with None recursively.
        record = _nan_to_none(record)
        out.append(BenchmarkRun.model_validate(record))
    return out


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if obj is None:
        return None
    try:
        import math as _math

        if isinstance(obj, float) and _math.isnan(obj):
            return None
    except Exception:
        pass
    return obj


def write_runs_json(runs: list[BenchmarkRun], path: Path) -> None:
    """Dump runs as a single JSON array (for the dashboard and sidecar)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump(mode="json") for r in runs]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
