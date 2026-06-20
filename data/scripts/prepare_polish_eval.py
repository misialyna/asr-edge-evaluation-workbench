"""Prepare a frozen Polish evaluation subset from Common Voice Polish.

Usage::

    python -m data.scripts.prepare_polish_eval \\
        --output-audio data/prepared/audio \\
        --output-manifest data/manifests/benchmark_manifest.jsonl \\
        --max-clips 20

This script downloads Mozilla Common Voice Polish, filters it to
short clean clips, normalizes the audio to 16 kHz mono WAV, and
writes a frozen JSONL manifest.

The actual download + filtering logic lives in
:mod:`asr_edge_eval.data.dataset`; this script is the CLI wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr_edge_eval.data.dataset import PreparationConfig, prepare_subset
from asr_edge_eval.data.manifest import manifest_checksum
from asr_edge_eval.utils.logging import configure_logging, get_logger


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_polish_eval",
        description="Download and freeze a Polish ASR evaluation subset from Common Voice.",
    )
    p.add_argument(
        "--output-audio",
        type=Path,
        default=Path("data/prepared/audio"),
        help="Where to write normalized 16kHz mono WAV clips.",
    )
    p.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/manifests/benchmark_manifest.jsonl"),
        help="Where to write the frozen JSONL manifest.",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=20,
        help="Number of clips to select.",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum clip duration in seconds.",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum clip duration in seconds.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap on the number of input records to read (debug).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_logging()
    log = get_logger("prepare_polish_eval")

    cfg = PreparationConfig(
        output_audio_dir=args.output_audio.expanduser().resolve(),
        output_manifest=args.output_manifest.expanduser().resolve(),
        max_clips=args.max_clips,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
        seed=args.seed,
    )

    try:
        entries = prepare_subset(cfg, max_records=args.max_records)
    except Exception as exc:
        log.error("preparation failed", error=str(exc))
        return 1

    sha = manifest_checksum(cfg.output_manifest)
    version_path = cfg.output_manifest.parent / "manifest_version.txt"
    version_path.write_text(
        f"polish_eval_v1, {len(entries)} clips, sha256={sha}\n",
        encoding="utf-8",
    )
    checksum_path = cfg.output_manifest.parent / "manifest_checksum.txt"
    checksum_path.write_text(f"{sha}  {cfg.output_manifest.name}\n", encoding="utf-8")
    log.info("manifest ready", n=len(entries), sha256=sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
