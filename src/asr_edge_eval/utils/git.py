"""Capture the current git commit hash for reproducibility.

We intentionally swallow all exceptions: a missing git binary, a
non-git directory, or a corrupted repository must not break a
benchmark run. The result is ``None`` in those cases and is recorded
as such in the result row.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["current_commit", "is_git_repo"]


def is_git_repo(path: str | Path | None = None) -> bool:
    """Return ``True`` if ``path`` (or ``cwd``) is inside a git working tree."""
    if shutil.which("git") is None:
        return False
    cwd = str(path) if path else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def current_commit(path: str | Path | None = None, short: bool = True) -> str | None:
    """Return the current git commit hash, or ``None`` if not in a repo.

    Parameters
    ----------
    path:
        Directory to inspect. ``None`` means the current working
        directory.
    short:
        If ``True``, return the 12-character short hash; otherwise the
        full SHA-1.
    """
    if shutil.which("git") is None:
        return None
    args = ["git", "rev-parse", "--short=12" if short else "rev-parse", "HEAD"]
    cwd = str(path) if path else None
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
