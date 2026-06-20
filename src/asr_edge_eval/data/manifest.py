"""Frozen JSONL manifest loader for the Polish evaluation set.

The manifest is a line-delimited JSON file. Each line is one
:class:`ManifestEntry`. The file is **append-only in practice**:
once a run has been executed against a manifest version, that
version is frozen and the result rows reference it.

The loader validates every line and refuses the whole file on the
first malformed line — partial loads are an error class we do not
want to deal with.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["ManifestEntry", "ManifestError", "load_manifest", "write_manifest", "manifest_checksum"]


class ManifestError(ValueError):
    """Raised when a manifest file is malformed or cannot be loaded."""


@dataclass(frozen=True)
class ManifestEntry:
    """One row of the benchmark manifest.

    The fields match the columns written to disk; loading uses the
    same names so a round-trip is lossless.
    """

    file_id: str
    path_rel: str
    duration_s: float
    sample_rate: int
    channels: int
    reference_transcript: str
    dataset_source: str
    clip_license: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> ManifestEntry:
        try:
            return cls(
                file_id=str(obj["file_id"]),
                path_rel=str(obj["path_rel"]),
                duration_s=float(obj["duration_s"]),
                sample_rate=int(obj["sample_rate"]),
                channels=int(obj["channels"]),
                reference_transcript=str(obj["reference_transcript"]),
                dataset_source=str(obj["dataset_source"]),
                clip_license=str(obj["clip_license"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Invalid manifest entry: {obj!r}: {exc}") from exc


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load a JSONL manifest. Raises :class:`ManifestError` on any malformed line."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ManifestError(f"Manifest not found: {p}")
    out: list[ManifestEntry] = []
    with open(p, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Invalid JSON in manifest {p} line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ManifestError(f"Manifest {p} line {lineno} is not a JSON object")
            out.append(ManifestEntry.from_json(obj))
    return out


def write_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    """Write a JSONL manifest, creating parent directories as needed."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_json(), ensure_ascii=False))
            f.write("\n")


def iter_manifest(path: str | Path) -> Iterator[ManifestEntry]:
    """Iterate a manifest without materializing the full list."""
    yield from load_manifest(path)


def manifest_checksum(path: str | Path) -> str:
    """Return the SHA-256 checksum of a manifest file."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ManifestError(f"Manifest not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
