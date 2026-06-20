"""Generate a tiny synthetic Polish manifest for the demo mode.

This is **not** a substitute for the real Common Voice Polish subset
prepared by :file:`data/scripts/prepare_polish_eval.py`. It exists
so that ``make demo`` works on a developer machine without
downloading any data.

The transcripts here are short, clean Polish sentences — enough
to exercise WER, CER, RTF, RTFx, and the recommendation logic
end-to-end.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from asr_edge_eval.data.manifest import write_manifest
from asr_edge_eval.data.manifest import ManifestEntry


def _write_wav_sine(path: Path, duration_s: float, freq_hz: float, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sample_rate)
    amplitude = 0.1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            value = int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames += struct.pack("<h", value)
        w.writeframes(bytes(frames))


SYNTHETIC_CLIPS = [
    ("clip_001", 1.0, 440.0, "Dzień dobry, jak się masz?"),
    ("clip_002", 1.5, 523.0, "Warszawa jest stolicą Polski."),
    ("clip_003", 2.0, 587.0, "Kraków to piękne miasto."),
    ("clip_004", 1.2, 659.0, "Wrocław leży nad Odrą."),
    ("clip_005", 1.8, 698.0, "Gdańsk znany jest z portu."),
    ("clip_006", 1.4, 783.0, "Poznań to miasto krótkie."),
    ("clip_007", 2.2, 880.0, "Łódź ma wiele zabytków."),
    ("clip_008", 1.6, 987.0, "Katowice są stolicą Śląska."),
    ("clip_009", 1.1, 1000.0, "Lublin to miasto wschodu."),
    ("clip_010", 1.9, 1100.0, "Białystok leży na Podlasiu."),
]


def main() -> int:
    audio_dir = Path("data/prepared/audio")
    manifest_path = Path("data/manifests/benchmark_manifest.jsonl")

    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    for clip_id, duration, freq, transcript in SYNTHETIC_CLIPS:
        wav_path = audio_dir / f"{clip_id}.wav"
        _write_wav_sine(wav_path, duration_s=duration, freq_hz=freq, sample_rate=16000)
        entries.append(
            ManifestEntry(
                file_id=clip_id,
                path_rel=f"prepared/audio/{clip_id}.wav",
                duration_s=duration,
                sample_rate=16000,
                channels=1,
                reference_transcript=transcript,
                dataset_source="synthetic:demo",
                clip_license="CC0-1.0",
            )
        )
    write_manifest(entries, manifest_path)
    print(f"wrote {len(entries)} synthetic clips to {audio_dir}")
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
