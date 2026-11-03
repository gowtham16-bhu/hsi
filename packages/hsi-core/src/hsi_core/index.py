"""C7 (Gate 0 form): one vector index over passages and shortcuts, with unpacking.

I = { (v_p, p) : p ∈ P } ∪ { (k_c, c) : c ∈ S }
R_k(q) = first k passages of unpack(topK_I(q))
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hsi_core.encoding import chain_text, l2_normalize, passage_text
from hsi_core.interfaces import Encoder, VectorStore
from hsi_core.model import Chain, KeyMode, Passage, Shortcut


@dataclass(frozen=True)
class Hit:
    entry: int
    score: float
    passages: tuple[str, ...]


@dataclass(frozen=True)
class Retrieved:
    passages: list[str]
    hits: list[Hit]
    latency_ms: float
    scores: list[float] = field(default_factory=list)   # per returned passage, when the retriever has them


def unpack(hits: Sequence[Hit], k: int) -> list[str]:
    """Passages in hit order, each chain in its own order, without duplicates, up to k."""
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        for pid in h.passages:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
                if len(out) == k:
                    return out
    return out


class ShortcutIndex:
    """Unified index. With no shortcuts it is the plain dense passage index."""

    def __init__(self, encoder: Encoder, store_factory: Callable[[], VectorStore],
                 batch_size: int = 256) -> None:
        self.encoder = encoder
        self.store = store_factory()
        self.batch_size = batch_size
        self.entries: list[tuple[str, ...]] = []
        self.levels: list[int] = []
        self.passage_vecs: dict[str, np.ndarray] = {}

    def _encode_batched(self, texts: list[str]) -> np.ndarray:
        parts = [self.encoder.encode(texts[i:i + self.batch_size], "passage")
                 for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(parts) if parts else np.zeros((0, self.encoder.dim), dtype=np.float32)

    def build(self, passages: Sequence[Passage], shortcuts: Sequence[Shortcut] = (),
              vector_transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None
              ) -> ShortcutIndex:
        """`vector_transform` rewrites passage vectors before indexing (e.g. graph smoothing)."""
        by_id = {p.id: p for p in passages}
        raw = self._encode_batched([passage_text(p) for p in passages])
        self.raw_vecs = {p.id: v for p, v in zip(passages, raw, strict=True)}
        self.passage_vecs = vector_transform(self.raw_vecs) if vector_transform else dict(self.raw_vecs)
        pvecs = np.stack([self.passage_vecs[p.id] for p in passages]) if passages else raw
        self._add([(p.id,) for p in passages], pvecs, [0] * len(passages))

        unsupported = {s.key_mode for s in shortcuts} - {KeyMode.TEXT, KeyMode.MEAN}
        if unsupported:
            raise NotImplementedError(f"key modes {sorted(m.value for m in unsupported)} arrive with the C5 composer")
        mean_keyed = [s for s in shortcuts if s.key_mode is KeyMode.MEAN]
        if mean_keyed:
            kvecs = l2_normalize(np.stack([np.sum([self.passage_vecs[p] for p in s.chain.passages], axis=0)
                                           for s in mean_keyed]))
            self._add([s.chain.passages for s in mean_keyed], kvecs, [s.chain.level for s in mean_keyed])
        text_keyed = [s for s in shortcuts if s.key_mode is KeyMode.TEXT]
        if text_keyed:
            texts = [chain_text([by_id[pid] for pid in s.chain.passages]) for s in text_keyed]
            kvecs = self._encode_batched(texts)
            self._add([s.chain.passages for s in text_keyed], kvecs, [s.chain.level for s in text_keyed])
        return self

    def build_from_vectors(self, ids: Sequence[str], vecs: np.ndarray) -> ShortcutIndex:
        """A passage-only index from stored vectors (row i is passage ids[i]); no encoding (A5 §2)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        if len(ids) != len(vecs):
            raise ValueError("ids and vectors differ in length")
        self.raw_vecs = {pid: v for pid, v in zip(ids, vecs, strict=True)}
        self.passage_vecs = dict(self.raw_vecs)
        self._add([(pid,) for pid in ids], vecs, [0] * len(ids))
        return self

    def _add(self, entries: list[tuple[str, ...]], vecs: np.ndarray, levels: list[int]) -> None:
        start = len(self.entries)
        ids = np.arange(start, start + len(entries), dtype=np.int64)
        if len(entries):
            self.store.add(ids, np.asarray(vecs, dtype=np.float32))
        self.entries.extend(entries)
        self.levels.extend(levels)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Hit]:
        scores, ids = self.store.search(query_vec.reshape(1, -1).astype(np.float32), top_k)
        return [Hit(entry=int(i), score=float(s), passages=self.entries[int(i)])
                for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def retrieve(self, query: str, k: int, top_k: int) -> Retrieved:
        return self.retrieve_vec(self.encoder.encode([query], "query")[0], k, top_k)

    def retrieve_vec(self, query_vec: np.ndarray, k: int, top_k: int) -> Retrieved:
        """Retrieval from an already-encoded query; latency excludes query encoding."""
        t0 = time.perf_counter()
        hits = self.search(query_vec, top_k)
        out = unpack(hits, k)
        return Retrieved(passages=out, hits=hits, latency_ms=(time.perf_counter() - t0) * 1000)

    def size_by_level(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for lvl in self.levels:
            counts[lvl] = counts.get(lvl, 0) + 1
        return counts


def shortcuts_from_chains(chains: Sequence[Chain], key_mode: KeyMode = KeyMode.TEXT) -> list[Shortcut]:
    return [Shortcut(chain=c, key_mode=key_mode) for c in chains]
