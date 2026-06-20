"""Character Error Rate via ``jiwer``.

CER is computed the same way as WER but at the character level. We
do *not* remove whitespace before computing CER, because removing
it would inflate the denominator for models that correctly produce
spaces; instead we treat whitespace as a regular character.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

__all__ = ["CerResult", "compute_cer"]


@dataclass(frozen=True)
class CerResult:
    """Result of a single CER computation."""

    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_chars: int
    hypothesis_chars: int


def compute_cer(reference: str, hypothesis: str) -> CerResult:
    """Compute CER between two already-normalized strings.

    An empty reference with an empty hypothesis yields CER 0.0 and
    zero counts. An empty reference with a non-empty hypothesis
    yields ``float("inf")`` (clamped from jiwer's division-by-zero).
    """
    ref_chars = list(reference) if reference else []
    hyp_chars = list(hypothesis) if hypothesis else []
    if not ref_chars and not hyp_chars:
        return CerResult(0.0, 0, 0, 0, 0, 0, 0)
    if not ref_chars:
        return CerResult(
            cer=float("inf"),
            substitutions=0,
            deletions=0,
            insertions=len(hyp_chars),
            hits=0,
            reference_chars=0,
            hypothesis_chars=len(hyp_chars),
        )
    if not hyp_chars:
        return CerResult(
            cer=1.0,
            substitutions=0,
            deletions=len(ref_chars),
            insertions=0,
            hits=0,
            reference_chars=len(ref_chars),
            hypothesis_chars=0,
        )

    out = jiwer.process_characters(reference, hypothesis)
    return CerResult(
        cer=float(out.cer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        reference_chars=len(ref_chars),
        hypothesis_chars=len(hyp_chars),
    )
