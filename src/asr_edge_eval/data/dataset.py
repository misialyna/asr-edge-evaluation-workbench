"""Polish dataset preparation.

The default source is **Mozilla Common Voice Polish** (CC0). This
module provides the *preparation* logic — downloading, filtering
to a small representative subset, normalizing the audio to 16 kHz
mono WAV, and writing a frozen manifest.

If the ``datasets`` package is not installed (declared in the
``[hf]`` extra), the function fails fast with a helpful error
message rather than falling back to a network-requiring default.

The actual download logic is in :file:`data/scripts/prepare_polish_eval.py`
so that the heavy ``datasets`` import is only paid when the user
explicitly asks to prepare a dataset, not at the top of every
module.
"""

from __future__ import annotations

import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ManifestEntry, write_manifest

__all__ = ["prepare_subset", "PreparationConfig", "PreparationError"]


class PreparationError(RuntimeError):
    """Raised when the Polish subset cannot be prepared."""


@dataclass(frozen=True)
class PreparationConfig:
    """Knobs for the subset preparer."""

    output_audio_dir: Path
    output_manifest: Path
    max_clips: int = 20
    min_duration_s: float = 2.0
    max_duration_s: float = 10.0
    target_sample_rate: int = 16_000
    target_channels: int = 1
    seed: int = 42


def _wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return 0.0


def _normalize_wav_in_place(
    src: Path,
    dst: Path,
    *,
    target_sr: int,
    target_channels: int,
) -> float:
    """Resample + downmix to ``(target_sr, target_channels)``.

    Returns the new duration in seconds. Uses ``soundfile`` (already
    a dependency) for I/O and a simple linear resampler for
    portability. Production code can swap in ``librosa`` or
    ``scipy.signal.resample`` without touching the call site.
    """
    import soundfile as sf

    data, sr = sf.read(str(src), always_2d=True)
    # Downmix to mono if requested.
    if target_channels == 1 and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    # Truncate to first channel if the file has more channels than
    # requested.
    if data.shape[1] > target_channels:
        data = data[:, :target_channels]
    # Resample if needed. The simple linear approach is adequate
    # for the 16 kHz target since faster-whisper downsamples
    # internally anyway.
    if sr != target_sr:
        n_out = int(round(data.shape[0] * target_sr / float(sr)))
        # Per-channel linear interpolation.
        old_x = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=True)
        new_x = np.linspace(0.0, 1.0, num=n_out, endpoint=True)
        out = np.empty((n_out, data.shape[1]), dtype=data.dtype)
        for ch in range(data.shape[1]):
            out[:, ch] = np.interp(new_x, old_x, data[:, ch])
        data = out
        sr = target_sr
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return data.shape[0] / float(sr)


def _iter_common_voice_polish(split: str = "validated") -> Iterable[dict[str, Any]]:
    """Yield records from Common Voice Polish ``split``.

    Imported lazily so that ``asr_edge_eval`` does not require the
    ``datasets`` package as a hard dependency.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover — env-specific
        raise PreparationError(
            "datasets package is required for dataset preparation. "
            "Install with: pip install 'asr-edge-evaluation-workbench[hf]'"
        ) from exc
    ds = load_dataset(
        "mozilla-foundation/common_voice",
        "pl",
        split=split,
        trust_remote_code=True,
    )
    yield from ds


def prepare_subset(
    cfg: PreparationConfig,
    *,
    source: str = "common_voice_pl",
    license_str: str = "CC0-1.0",
    source_version: str = "11.0",
    max_records: int | None = None,
) -> list[ManifestEntry]:
    """Download a Polish subset and write a frozen manifest.

    Parameters
    ----------
    cfg:
        Preparation configuration.
    source, license_str, source_version:
        Recorded in the manifest for citation. Defaults match
        Common Voice Polish v11.0.
    max_records:
        Optional hard cap on the number of *input* records to read
        before filtering. Useful for fast smoke tests.

    Returns
    -------
    list[ManifestEntry]
        The manifest that was written to ``cfg.output_manifest``.
    """
    cfg.output_audio_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    selected: list[ManifestEntry] = []

    count = 0
    for row in _iter_common_voice_polish():
        if max_records is not None and count >= max_records:
            break
        count += 1
        audio = row.get("audio") or {}
        path_obj = audio.get("path")
        if not path_obj:
            continue
        src = Path(str(path_obj))
        if not src.is_file():
            continue
        sentence = (row.get("sentence") or "").strip()
        if not sentence:
            continue
        # Quick pre-filter: trust the dataset's reported duration.
        # If unavailable, fall back to a wav header read.
        reported = row.get("duration") if "duration" in row else None
        try:
            duration_s = float(reported) if reported is not None else _wav_duration_s(src)
        except (TypeError, ValueError):
            continue
        if duration_s < cfg.min_duration_s or duration_s > cfg.max_duration_s:
            continue

        file_id = f"cvpl_{count:05d}"
        dst = cfg.output_audio_dir / f"{file_id}.wav"
        try:
            new_dur = _normalize_wav_in_place(
                src,
                dst,
                target_sr=cfg.target_sample_rate,
                target_channels=cfg.target_channels,
            )
        except Exception as exc:
            raise PreparationError(f"Failed to normalize {src}: {exc}") from exc
        rel = str(dst.relative_to(cfg.output_manifest.parent.parent))
        entry = ManifestEntry(
            file_id=file_id,
            path_rel=rel,
            duration_s=new_dur,
            sample_rate=cfg.target_sample_rate,
            channels=cfg.target_channels,
            reference_transcript=sentence,
            dataset_source=f"{source}:{source_version}",
            clip_license=license_str,
        )
        selected.append(entry)
        if len(selected) >= cfg.max_clips:
            break

    if not selected:
        raise PreparationError("No clips selected. Check the dataset name, split and filters.")

    # Deterministic ordering for reproducibility.
    selected.sort(key=lambda e: e.file_id)
    write_manifest(selected, cfg.output_manifest)
    return selected


def select_short_subset(
    entries: list[ManifestEntry], n: int, *, seed: int = 42
) -> list[ManifestEntry]:
    """Pick ``n`` clips deterministically from ``entries``."""
    if n <= 0:
        return []
    if n >= len(entries):
        return list(entries)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(entries), size=n, replace=False).tolist())
    return [entries[i] for i in indices]
