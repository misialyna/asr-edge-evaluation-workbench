"""Run a benchmark experiment end-to-end.

Usage
-----

::

    python -m scripts.run_benchmark --config configs/experiments/demo_mode.yaml
    python -m scripts.run_benchmark --config configs/experiments/demo_mode.yaml --use-fake-runtime
    python -m scripts.run_benchmark --config configs/experiments/jetson_orin_default.yaml

The script:

1. loads the experiment YAML via :func:`asr_edge_eval.config.load_experiment`;
2. loads the frozen manifest;
3. for each (configuration, audio file) executes the warmup +
   measured runs via :func:`asr_edge_eval.orchestrator.run_experiment`;
4. writes Parquet + CSV + JSON sidecar to ``output.results_parquet`` /
   ``output.results_csv``;
5. (optionally) triggers the Markdown report generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr_edge_eval.config import ConfigError, load_experiment
from asr_edge_eval.data.manifest import load_manifest
from asr_edge_eval.orchestrator.runner import run_experiment, write_results
from asr_edge_eval.reporting.report import write_runs_json
from asr_edge_eval.runtime import FakeAdapter, FakeConfig
from asr_edge_eval.schemas import RuntimeConfigurationSpec
from asr_edge_eval.utils.logging import configure_logging, get_logger


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="Run an ASR Edge benchmark experiment.",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML config.",
    )
    p.add_argument(
        "--use-fake-runtime",
        action="store_true",
        help=(
            "Force the deterministic FakeAdapter (no model weights, no GPU). "
            "Useful for CI and demo mode on a developer machine."
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the manifest path (otherwise read from the YAML).",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Override the audio root directory.",
    )
    p.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help="Override the output Parquet path.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override the output CSV path.",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the Markdown report after the run.",
    )
    p.add_argument(
        "--log-format",
        choices=["console", "json"],
        default="console",
        help="Log renderer (default: console).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging(level=args.log_level, fmt=args.log_format)
    log = get_logger("run_benchmark")

    try:
        cfg = load_experiment(args.config)
    except ConfigError as exc:
        log.error("config error", error=str(exc))
        return 2

    manifest_path = args.manifest or Path(cfg.dataset.manifest_path)
    audio_root = (args.audio_root or Path(cfg.dataset.audio_root)).expanduser().resolve()
    output_parquet = (args.output_parquet or Path(cfg.output.results_parquet)).expanduser().resolve()
    output_csv = (args.output_csv or Path(cfg.output.results_csv)).expanduser().resolve()

    try:
        entries = load_manifest(manifest_path)
    except Exception as exc:
        log.error("manifest load failed", path=str(manifest_path), error=str(exc))
        return 3

    # Apply max_files (deterministic) and shuffle.
    if cfg.dataset.shuffle:
        import random

        rng = random.Random(cfg.dataset.seed)
        rng.shuffle(entries)
    if cfg.dataset.max_files is not None:
        entries = entries[: cfg.dataset.max_files]
    if not entries:
        log.error("no manifest entries after filtering")
        return 4

    factory = None
    if args.use_fake_runtime or cfg.execution.use_fake_runtime_if_demo:
        def _factory(spec: RuntimeConfigurationSpec) -> FakeAdapter:
            return FakeAdapter(FakeConfig(transcript="DETERMINISTIC FAKE TRANSCRIPT"))

        factory = _factory
        log.info("using FakeAdapter (deterministic, no model weights)")

    log.info(
        "starting experiment",
        name=cfg.experiment.name,
        mode=cfg.experiment.mode.value,
        n_entries=len(entries),
        configurations=[c.id for c in cfg.runtime.configurations],
    )

    runs = run_experiment(
        cfg,
        entries=entries,
        audio_root=audio_root,
        runtime_factory=factory,
    )
    log.info("runs completed", n=len(runs))

    write_results(runs, parquet_path=output_parquet, csv_path=output_csv)
    # JSON sidecar used by the dashboard to reload nested objects.
    write_runs_json(runs, output_parquet.with_suffix(".json"))
    log.info("wrote results", parquet=str(output_parquet), csv=str(output_csv))

    if not args.no_report:
        try:
            from asr_edge_eval.reporting.report import render_markdown_report

            report_path = Path(cfg.output.report_md).expanduser().resolve()
            figures_dir = Path(cfg.output.figures_dir).expanduser().resolve()
            weights_path = Path("configs/recommend_weights.yaml")
            if not weights_path.is_file():
                weights_path = None
            render_markdown_report(
                runs,
                out_path=report_path,
                figures_dir=figures_dir,
                weights_path=weights_path,
            )
            log.info("wrote report", path=str(report_path))
        except Exception as exc:  # pragma: no cover — non-fatal
            log.warning("report generation failed", error=str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
