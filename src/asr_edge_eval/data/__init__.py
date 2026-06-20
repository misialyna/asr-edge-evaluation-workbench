"""Data layer: frozen manifest loading and dataset preparation."""

from .manifest import ManifestEntry, ManifestError, load_manifest, write_manifest

__all__ = [
    "ManifestEntry",
    "ManifestError",
    "load_manifest",
    "write_manifest",
]
