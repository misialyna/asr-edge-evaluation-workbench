"""Polish text normalization used before computing WER/CER.

The pipeline is deliberately conservative:

1. Unicode NFC normalization so the same logical character always
   compares equal.
2. Strip leading and trailing whitespace, collapse internal runs.
3. Lowercase.
4. Remove punctuation that does not appear inside words, while
   preserving intra-word hyphens and apostrophes.
5. **Do not strip diacritics** — they are a real signal for Polish
   WER and removing them would systematically under-report errors.

The pipeline is versioned (``NORMALIZER_VERSION``) and that version
is stamped into every ``QualityMetrics`` row. Any change to the
pipeline must bump the version, so that old and new WER numbers
remain comparable to a documented delta.

The pipeline is implemented in pure Python, not via ``jiwer``, so we
can unit-test it independently and so that the CER calculation can
use exactly the same character stream.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

__all__ = ["NORMALIZER_VERSION", "normalize_pl"]

NORMALIZER_VERSION: Final[str] = "pl-v1.0"

# Polish diacritics — kept intact on purpose.
_POLISH_DIACRITICS: Final[str] = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

# Characters allowed inside words even if surrounded by other text.
_INTRA_WORD_PUNCT: Final[str] = "-'`"

# A "word" is a run of letters or digits, optionally with intra-word
# punctuation. We deliberately do NOT include pure-digit tokens in
# CER scoring later (those are normalized separately if needed).
_WORD_RE: Final[re.Pattern[str]] = re.compile(
    rf"[{_POLISH_DIACRITICS}A-Za-z0-9{re.escape(_INTRA_WORD_PUNCT)}]+",
    flags=re.UNICODE,
)

# Junk we want collapsed to a single space.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _strip_punctuation_outside_words(text: str) -> str:
    """Drop punctuation that is not part of a word, keep word chars.

    The implementation walks the string and emits either the original
    word characters or a single space, never punctuation. This is
    intentionally explicit (rather than a regex ``sub``) so the
    behaviour is easy to reason about and to test.
    """
    out: list[str] = []
    last_was_space = True
    for match in _WORD_RE.finditer(text):
        if match.start() > 0 and not last_was_space:
            out.append(" ")
        out.append(match.group(0))
        last_was_space = False
    return "".join(out).strip()


def normalize_pl(text: str) -> str:
    """Return the normalized form of a Polish text.

    The function is total: it accepts empty strings, ``None``-like
    inputs are not expected (callers must pass ``str``), and it
    never raises. The output is always ``str``.
    """
    if not text:
        return ""
    # Step 1: Unicode NFC.
    text = unicodedata.normalize("NFC", text)
    # Step 2: collapse whitespace and trim.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    # Step 3: lowercase (Polish has no ß-style exceptions to worry
    # about — Polish ł / Ł are well-behaved in ``str.lower()``).
    text = text.lower()
    # Step 4: strip non-word punctuation.
    text = _strip_punctuation_outside_words(text)
    # Step 5: collapse whitespace introduced by punctuation removal.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
