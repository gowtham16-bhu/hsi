"""Text normalization and tokenization shared by matching and BM25."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[0-9a-z]+(?:['’][0-9a-z]+)?")



def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Used for title matching."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower().replace("’", "'")
    return " ".join(t.split())


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))

