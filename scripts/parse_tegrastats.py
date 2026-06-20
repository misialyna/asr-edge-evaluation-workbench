"""Parse a tegrastats log to a per-line JSON array on stdout.

Usage::

    python -m scripts.parse_tegrastats results/runs/telemetry/run-001.log > parsed.json

Useful for ad-hoc inspection. The full pipeline uses
:mod:`asr_edge_eval.telemetry.tegrastats` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asr_edge_eval.telemetry.tegrastats import parse_line
from asr_edge_eval.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parse_tegrastats",
        description="Parse a tegrastats log to a per-line JSON array on stdout.",
    )
    parser.add_argument("log", type=Path, help="Path to a tegrastats log file.")
    args = parser.parse_args(argv)
    configure_logging()
    if not args.log.is_file():
        print(f"file not found: {args.log}", file=sys.stderr)
        return 1
    parsed = [parse_line(line).as_dict() for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines()]
    json.dump(parsed, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
