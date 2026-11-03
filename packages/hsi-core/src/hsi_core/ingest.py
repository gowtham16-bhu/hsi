"""C2 ingestion (structure part): source documents become passages.

1. Structure first: every section is split on its own; passages never cross sections.
2. Inside long sections, boundaries are content-defined: a boundary falls after word i when the
   rolling hash of the last `window` words satisfies h mod M == 0, bounded by min/max size.
   An edit then changes only the chunks around it, not every chunk after it.
3. Passage ids depend on (document, content hash, duplicate index), never on position, so
   unchanged chunks keep their ids across edits (needed by C10 incremental updates).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from hsi_core.model import Passage, SourceDocument, content_hash


@dataclass(frozen=True)
class ChunkConfig:
    min_words: int = 48
    max_words: int = 240
    modulus: int = 64        # M: average extra length beyond min_words
    window: int = 8          # words in the rolling window

    def __post_init__(self) -> None:
        if not 1 <= self.min_words <= self.max_words or self.modulus < 1 or self.window < 1:
            raise ValueError("invalid chunk config")


def _word_hash(w: str) -> int:
    return int.from_bytes(hashlib.blake2b(w.lower().encode(), digest_size=8).digest(), "little")


def split_words(words: list[str], cfg: ChunkConfig) -> list[list[str]]:
    if len(words) <= cfg.max_words:
        return [words] if words else []
    hashes = [_word_hash(w) for w in words]
    chunks: list[list[str]] = []
    start = 0
    for i in range(len(words)):
        size = i - start + 1
        if size < cfg.min_words:
            continue
        window = hashes[max(start, i - cfg.window + 1): i + 1]
        h = 0
        for x in window:
            h = (h * 1_000_003 + x) & 0xFFFFFFFFFFFFFFFF
        if size >= cfg.max_words or h % cfg.modulus == 0:
            chunks.append(words[start: i + 1])
            start = i + 1
    if start < len(words):
        tail = words[start:]
        if chunks and len(tail) < cfg.min_words // 2:
            chunks[-1] = chunks[-1] + tail
        else:
            chunks.append(tail)
    return chunks


def ingest_document(doc: SourceDocument, cfg: ChunkConfig | None = None) -> list[Passage]:
    cfg = cfg or ChunkConfig()
    passages: list[Passage] = []
    dup: Counter[str] = Counter()
    for sec in doc.sections:
        words = sec.text.split()
        path = (sec.heading,) if sec.heading else ()
        for chunk in split_words(words, cfg):
            text = " ".join(chunk)
            h = content_hash(f"{doc.title or ''}\n{text}")
            passages.append(Passage.build(doc_id=doc.id, text=text, title=doc.title, acl=doc.acl,
                                          section_path=path, ordinal=dup[h]))
            dup[h] += 1
    return passages
