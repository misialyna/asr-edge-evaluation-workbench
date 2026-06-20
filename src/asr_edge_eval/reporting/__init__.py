"""Reporting layer: Markdown report and matplotlib plots."""

from .plots import plot_latency_box, plot_resource_bars, plot_wer_vs_rtfx
from .report import render_markdown_report

__all__ = [
    "plot_latency_box",
    "plot_resource_bars",
    "plot_wer_vs_rtfx",
    "render_markdown_report",
]
