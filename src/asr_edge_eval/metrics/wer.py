"""Word Error Rate via ``jiwer``.

We compute WER on the *already-normalized* reference and hypothesis,
which is the standard pattern used by the Open ASR Leaderboard and
which keeps the analysis layer free of hidden normalization rules.

The result object carries the same counts that jiwer returns (S, D,
I, H) so that downstream code can derive its own metrics (e.g. MER,
WIL) without re-running the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

__all__ = ["WerResult", "compute_wer"]


@dataclass(frozen=True)
class WerResult:
    """Result of a single WER computation.

    All counts are non-negative integers. ``wer == (S + D + I) / N``,
    which ``jiwer`` guarantees and which is asserted in tests.
    """

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    hypothesis_words: int


def compute_wer(reference: str, hypothesis: str) -> WerResult:
    """Compute WER between two already-normalized strings.

    An empty reference with an empty hypothesis yields WER 0.0 and
    zero counts. An empty reference with a non-empty hypothesis
    yields infinite WER; we clamp it to ``float("inf")`` so the
    caller can detect it.
    """
    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []
    if not ref_words and not hyp_words:
        return WerResult(0.0, 0, 0, 0, 0, 0, 0)
    if not ref_words:
        # Empty reference + non-empty hypothesis: jiwer >= 3.0
        # refuses the input, but the conventional answer is "every
        # hypothesis word is an insertion, giving infinite WER".
        return WerResult(
            wer=float("inf"),
            substitutions=0,
            deletions=0,
            insertions=len(hyp_words),
            hits=0,
            reference_words=0,
            hypothesis_words=len(hyp_words),
        )
    if not hyp_words:
        # Non-empty reference + empty hypothesis: every reference
        # word is a deletion, giving WER = 1.0.
        return WerResult(
            wer=1.0,
            substitutions=0,
            deletions=len(ref_words),
            insertions=0,
            hits=0,
            reference_words=len(ref_words),
            hypothesis_words=0,
        )

    # jiwer >= 3.0 exposes the Alignment object via process_words.
    out = jiwer.process_words(reference, hypothesis)
    return WerResult(
        wer=float(out.wer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        reference_words=len(ref_words),
        hypothesis_words=len(hyp_words),
    )
