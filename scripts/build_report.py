"""Regenerate the Markdown report from a saved Parquet results file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr_edge_eval.reporting.report import render_markdown_report, runs_from_parquet
from asr_edge_eval.utils.logging import configure_logging, get_logger


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_report",
        description="Render the Markdown benchmark report from a saved Parquet result file.",
    )
    p.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to a Parquet results file (with .json sidecar if available).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/reports/benchmark_report.md"),
        help="Where to write the Markdown report.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Where to write the PNG figures.",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("configs/recommend_weights.yaml"),
        help="Path to the recommendation weights YAML (optional).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging()
    log = get_logger("build_report")
    if not args.results.is_file():
        log.error("results file not found", path=str(args.results))
        return 1
    runs = runs_from_parquet(args.results)
    if not runs:
        log.error("no runs found in results file")
        return 2
    weights_path = args.weights if args.weights.is_file() else None
    out = render_markdown_report(
        runs,
        out_path=args.out,
        figures_dir=args.figures_dir,
        weights_path=weights_path,
    )
    log.info("report written", path=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
