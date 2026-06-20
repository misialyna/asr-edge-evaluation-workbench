"""ASR Edge Evaluation Workbench.

A reproducible benchmarking and decision-support tool that measures
accuracy, latency, real-time factor, and resource use of speech-to-text
configurations on a frozen evaluation set, on either NVIDIA Jetson
hardware (real measurements) or a developer machine (deterministic demo
mode).

Public surface is intentionally small: load an experiment, run it,
inspect the resulting ``BenchmarkRun`` rows, render a report.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Michalina Moszyńska"

__all__ = [
    "__version__",
    "__author__",
]
