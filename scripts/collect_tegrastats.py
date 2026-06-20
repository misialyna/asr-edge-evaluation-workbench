"""Standalone tegrastats collector (for ad-hoc logging)."""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from asr_edge_eval.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collect_tegrastats",
        description="Run tegrastats for a fixed duration and write a log file.",
    )
    parser.add_argument(
        "--path",
        default="/usr/bin/tegrastats",
        help="Path to the tegrastats binary.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output log file path.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration in seconds. If omitted, run until SIGINT.",
    )
    args = parser.parse_args(argv)
    configure_logging()
    log = get_logger("collect_tegrastats")

    if shutil.which(args.path) is None and not Path(args.path).exists():
        log.error("tegrastats not found", path=args.path)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log.info("starting tegrastats", path=args.path, out=str(args.out), interval_ms=args.interval_ms)

    with open(args.out, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(  # noqa: S603
            [args.path, "--interval", str(args.interval_ms)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            if args.duration_s is not None:
                time.sleep(args.duration_s)
            else:
                proc.wait()
        except KeyboardInterrupt:
            log.info("interrupted; stopping tegrastats")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                import os

                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

    log.info("done", out=str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
