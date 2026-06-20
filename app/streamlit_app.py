"""Streamlit dashboard for exploring benchmark results.

The dashboard reads the latest ``results/runs/results.parquet``
(or a path supplied by the user) and provides four views:

1. **Overview** — recommendation table + per-config headline numbers.
2. **Quality** — WER distribution and WER vs RTFx scatter.
3. **Latency & resources** — latency percentiles, peak RAM/GPU/temp.
4. **Per-file** — drill-down into a single audio file.

The app is intentionally read-only: it does not run benchmarks
itself. Use ``make demo`` or ``scripts/run_benchmark.py`` to
produce data first.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover — importable check
    st = None  # type: ignore[assignment]

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]

from asr_edge_eval.analysis.recommend import recommend
from asr_edge_eval.reporting.report import (
    _runs_from_flat_df,
    runs_from_parquet,
)
from asr_edge_eval.schemas import BenchmarkRun

DEFAULT_RESULTS = Path("results/runs/results.parquet")


def _load_runs(path: Path) -> list[BenchmarkRun]:
    if not path.is_file():
        return []
    if path.suffix == ".parquet":
        # Prefer the JSON sidecar if present — it preserves nested
        # objects exactly. Fall back to the flattened CSV/Parquet
        # reconstruction otherwise.
        try:
            return runs_from_parquet(path)
        except Exception:
            df = pd.read_parquet(path)
            return _runs_from_flat_df(df)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return _runs_from_flat_df(df)
    return []


def _to_dataframe(runs: list[BenchmarkRun]) -> pd.DataFrame:
    """Flatten runs into a DataFrame for the dashboard's tables."""
    rows: list[dict[str, Any]] = []
    for r in runs:
        rows.append(
            {
                "config_id": r.config_id,
                "file_id": r.audio.file_id,
                "wer": r.quality.wer,
                "cer": r.quality.cer,
                "rtf": r.latency.rtf,
                "rtfx": r.latency.rtfx,
                "p50_s": r.latency.p50_latency_s,
                "p95_s": r.latency.p95_latency_s,
                "ram_peak_mb": r.resources.ram_peak_mb,
                "gpu_mem_peak_mb": r.resources.gpu_mem_peak_mb,
                "temp_peak_c": r.resources.temperature_peak_c,
                "power_peak_w": r.resources.power_peak_w,
                "audio_duration_s": r.latency.audio_duration_s,
                "error": r.error,
            }
        )
    return pd.DataFrame(rows)


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _render_overview(runs: list[BenchmarkRun], df: pd.DataFrame) -> None:
    st.header("Overview")
    if not runs:
        st.info("No runs loaded. Run `make demo` first to produce results.")
        return
    try:
        recs = recommend(runs)
    except ValueError as exc:
        st.warning(f"Cannot compute recommendations: {exc}")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Real-time", f"`{recs.real_time_config_id}`")
    col2.metric("Balanced", f"`{recs.balanced_config_id}`")
    col3.metric("Quality-first", f"`{recs.quality_first_config_id}`")
    st.write(recs.rationale)
    st.subheader("Per-configuration summary")
    summary_rows = []
    for s in recs.summaries:
        summary_rows.append(
            {
                "config": s.config_id,
                "runs": s.num_runs,
                "WER mean": _safe(s.wer_mean),
                "WER std": _safe(s.wer_std),
                "RTFx mean": _safe(s.rtfx_mean),
                "RTFx std": _safe(s.rtfx_std),
                "Peak RAM (MB)": _safe(s.ram_peak_mb),
                "Peak GPU (MB)": _safe(s.gpu_mem_peak_mb),
                "Peak T (°C)": _safe(s.temperature_peak_c),
                "Peak P (W)": _safe(s.power_peak_w),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def _render_quality(df: pd.DataFrame) -> None:
    st.header("Quality")
    if df.empty:
        st.info("No data to plot.")
        return
    df_ok = df[df["wer"].notna() & df["wer"].apply(lambda v: v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))]
    if df_ok.empty:
        st.info("No quality data (all runs errored?).")
        return
    if px is not None:
        fig = px.box(df_ok, x="config_id", y="wer", points="all", title="WER by configuration")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(
            df_ok,
            x="rtfx",
            y="wer",
            color="config_id",
            title="WER vs RTFx",
            hover_data=["file_id"],
        )
        fig2.add_vline(x=1.0, line_dash="dash", line_color="grey", annotation_text="RTFx=1")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.line_chart(df_ok.pivot_table(index="file_id", columns="config_id", values="wer"))


def _render_latency_resources(df: pd.DataFrame) -> None:
    st.header("Latency & resources")
    if df.empty:
        st.info("No data to plot.")
        return
    if px is not None:
        melted = df.melt(
            id_vars=["config_id"],
            value_vars=["p50_s", "p95_s"],
            var_name="percentile",
            value_name="latency_s",
        )
        fig = px.box(melted, x="config_id", y="latency_s", color="percentile", title="Latency percentiles")
        st.plotly_chart(fig, use_container_width=True)
        if df["ram_peak_mb"].notna().any() or df["gpu_mem_peak_mb"].notna().any():
            ram_df = df.groupby("config_id", as_index=False).agg(
                ram_peak_mb=("ram_peak_mb", "max"),
                gpu_mem_peak_mb=("gpu_mem_peak_mb", "max"),
            )
            fig2 = px.bar(
                ram_df.melt(id_vars=["config_id"], var_name="metric", value_name="value"),
                x="config_id",
                y="value",
                color="metric",
                barmode="group",
                title="Peak memory by configuration (MB)",
            )
            st.plotly_chart(fig2, use_container_width=True)
        if df["temp_peak_c"].notna().any():
            temp_df = df.dropna(subset=["temp_peak_c"]).groupby("config_id", as_index=False)["temp_peak_c"].max()
            fig3 = px.bar(temp_df, x="config_id", y="temp_peak_c", title="Peak temperature (°C)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.bar_chart(df.groupby("config_id")["p95_s"].max())


def _render_per_file(runs: list[BenchmarkRun], df: pd.DataFrame) -> None:
    st.header("Per-file drill-down")
    if df.empty or "file_id" not in df.columns:
        st.info("No data to drill into.")
        return
    files = sorted(df["file_id"].dropna().unique().tolist())
    if not files:
        st.info("No file IDs in the result set.")
        return
    chosen = st.selectbox("File ID", files)
    sub = df[df["file_id"] == chosen]
    st.dataframe(sub, use_container_width=True)
    matching = [r for r in runs if r.audio.file_id == chosen]
    if matching:
        st.subheader("Hypothesis transcript(s)")
        for r in matching:
            st.write(
                f"**{r.config_id}** (WER={r.quality.wer:.3f}, RTFx={r.latency.rtfx:.2f}): "
                f"{r.hypothesis_transcript or '(empty)'}"
            )
        st.subheader("Reference transcript")
        st.write(matching[0].audio.reference_transcript)


def main() -> None:  # pragma: no cover — streamlit runtime
    if st is None:
        raise SystemExit(
            "Streamlit is not installed. Install with: "
            "pip install 'asr-edge-evaluation-workbench[dashboard]'"
        )
    st.set_page_config(page_title="ASR Edge Evaluation Workbench", layout="wide")
    st.title("ASR Edge Evaluation Workbench — Dashboard")
    st.caption("Polish ASR benchmarking on Jetson-class hardware. v1: Faster-Whisper only.")

    path_str = st.sidebar.text_input("Results path", str(DEFAULT_RESULTS))
    path = Path(path_str).expanduser().resolve()
    runs = _load_runs(path)
    df = _to_dataframe(runs)

    tab_overview, tab_quality, tab_latency, tab_per_file = st.tabs(
        ["Overview", "Quality", "Latency & resources", "Per-file"]
    )
    with tab_overview:
        _render_overview(runs, df)
    with tab_quality:
        _render_quality(df)
    with tab_latency:
        _render_latency_resources(df)
    with tab_per_file:
        _render_per_file(runs, df)


if __name__ == "__main__":  # pragma: no cover
    main()
