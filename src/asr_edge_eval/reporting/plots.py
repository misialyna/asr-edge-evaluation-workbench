"""Matplotlib helpers for the report.

Three plots are produced by the default report:

* :func:`plot_wer_vs_rtfx` — scatter of WER vs RTFx per
  configuration, with the RTFx=1.0 reference line.
* :func:`plot_latency_box` — box plot of inference latency per
  configuration, with one box per config.
* :func:`plot_resource_bars` — grouped bar chart of peak RAM and
  peak GPU memory per configuration.

All functions save the figure to disk and return the path. They
never display a window (use ``matplotlib``'s ``Agg`` backend) so
they work on headless servers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..schemas import BenchmarkRun

__all__ = ["plot_wer_vs_rtfx", "plot_latency_box", "plot_resource_bars"]


def _per_config(runs: Iterable[BenchmarkRun]) -> dict[str, list[BenchmarkRun]]:
    grouped: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for r in runs:
        if r.error is not None:
            continue
        grouped[r.config_id].append(r)
    return grouped


def plot_wer_vs_rtfx(runs: Iterable[BenchmarkRun], out_path: Path) -> Path:
    """Scatter WER vs RTFx, one point per (config, audio file) measurement."""
    grouped = _per_config(runs)
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for cid, rs in sorted(grouped.items()):
        xs = [r.latency.rtfx for r in rs]
        ys = [r.quality.wer for r in rs if math.isfinite(r.quality.wer)]
        n = min(len(xs), len(ys))
        if n == 0:
            continue
        ax.scatter(xs[:n], ys[:n], label=cid, alpha=0.8, s=50)
        plotted += n
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, label="real-time (RTFx=1)")
    ax.set_xlabel("RTFx (higher = faster than real time)")
    ax.set_ylabel("WER (lower = better)")
    if plotted > 0:
        ax.legend(loc="best")
    ax.set_title("Quality vs Real-Time Performance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_latency_box(runs: Iterable[BenchmarkRun], out_path: Path) -> Path:
    """Box plot of average inference latency per configuration."""
    grouped = _per_config(runs)
    labels: list[str] = []
    data: list[list[float]] = []
    for cid in sorted(grouped):
        rs = grouped[cid]
        vals = [r.latency.inference_duration_s for r in rs if r.latency.inference_duration_s > 0]
        if not vals:
            continue
        labels.append(cid)
        data.append(vals)
    fig, ax = plt.subplots(figsize=(7, 5))
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Inference latency (s)")
    ax.set_title("Per-Run Inference Latency by Configuration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_resource_bars(runs: Iterable[BenchmarkRun], out_path: Path) -> Path:
    """Grouped bar chart of peak RAM and peak GPU memory per configuration."""
    grouped = _per_config(runs)
    labels: list[str] = []
    rams: list[float] = []
    gpus: list[float] = []
    for cid in sorted(grouped):
        rs = grouped[cid]
        ram_peaks = [r.resources.ram_peak_mb for r in rs if r.resources.ram_peak_mb is not None]
        gpu_peaks = [
            r.resources.gpu_mem_peak_mb for r in rs if r.resources.gpu_mem_peak_mb is not None
        ]
        labels.append(cid)
        rams.append(max(ram_peaks) if ram_peaks else 0.0)
        gpus.append(max(gpu_peaks) if gpu_peaks else 0.0)
    if not labels:
        # No data — write a placeholder.
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No resource data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rams, width, label="Peak RAM (MB)")
    ax.bar(x + width / 2, gpus, width, label="Peak GPU mem (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Peak Memory by Configuration")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
