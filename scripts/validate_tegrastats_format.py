"""Inspect a tegrastats log and print which fields are present.

Useful as a pre-flight check on a Jetson before a benchmark run:

::

    python -m scripts.validate_tegrastats_format results/runs/telemetry/run-001.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr_edge_eval.telemetry.tegrastats import validate_format
from asr_edge_eval.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_tegrastats_format",
        description="Print which fields are present in a tegrastats log.",
    )
    parser.add_argument("log", type=Path, help="Path to a tegrastats log file.")
    args = parser.parse_args(argv)
    configure_logging()
    log = get_logger("validate_tegrastats_format")
    if not args.log.is_file():
        log.error("file not found", path=str(args.log))
        return 1
    detected = validate_format(args.log)
    for field, present in detected.items():
        marker = "OK" if present else "--"
        print(f"[{marker}] {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
