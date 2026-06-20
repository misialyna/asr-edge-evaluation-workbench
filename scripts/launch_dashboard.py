"""Convenience wrapper to launch the Streamlit dashboard."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from asr_edge_eval.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    log = get_logger("launch_dashboard")
    if shutil.which("streamlit") is None:
        log.error("streamlit not installed; pip install 'asr-edge-evaluation-workbench[dashboard]'")
        return 1
    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    cmd = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    log.info("launching", cmd=" ".join(cmd))
    return subprocess.call(cmd)  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
