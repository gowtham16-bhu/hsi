"""Iterative LLM-free hop retrieval over any unified index (decision 0003).

One algorithm, two index choices, so the comparison isolates the effect of shortcuts:
  over the passage index           -> each round adds one passage
  over passages + 2-passage shortcuts -> each round can add a whole 2-passage chain

Round 1 searches with the question vector. Each later round updates the query from the path so far
and searches again; paths that would repeat a passage are dropped; the best `beam` paths are kept.

Query update (from config, no learned parameters yet):
  vec  : q' = normalize(q + λ · normalize(Σ v_p over the path))
  text : q' = enc_query(question ⊕ text(path))         (MDR-style re-encoding, no LLM)

Paths are ranked by the sum of their round scores, then unpacked into passages.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hsi_core.encoding import chain_text, l2_normalize
from hsi_core.index import Hit, Retrieved, ShortcutIndex, unpack
from hsi_core.interfaces import Encoder
from hsi_core.model import Passage


@dataclass(frozen=True)
class HopConfig:
    rounds: int
    beam: int
    per_round: int                  # hits fetched per path per round
    update: Literal["vec", "text"]
    vec_weight: float               # λ, used by the vec update

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.beam < 1 or self.per_round < 1 or self.vec_weight < 0:
            raise ValueError("invalid hop config")


@dataclass
class _Path:
    passages: tuple[str, ...]
    score: float


class HopRetriever:
    def __init__(self, index: ShortcutIndex, passages: Mapping[str, Passage], cfg: HopConfig,
                 query_encoder: Encoder | None = None) -> None:
        """`query_encoder` re-encodes updated queries in the text mode; pass an uncached encoder so
        latency reflects real query-time work."""
        self.index, self.passages, self.cfg = index, passages, cfg
        self.query_encoder = query_encoder or index.encoder
        self.ann_calls = 0

    def _next_query(self, question: str, q: np.ndarray, path: _Path) -> np.ndarray:
        if self.cfg.update == "vec":
            ctx = l2_normalize(np.sum([self.index.passage_vecs[p] for p in path.passages], axis=0))
            return l2_normalize(q + self.cfg.vec_weight * ctx)
        text = chain_text([self.passages[p] for p in path.passages])
        return self.query_encoder.encode([f"{question} {text}"], "query")[0]

    def retrieve_vec(self, q: np.ndarray, k: int, top_k: int, question: str) -> Retrieved:
        t0 = time.perf_counter()
        first = self.index.search(q, max(self.cfg.per_round, self.cfg.beam))
        self.ann_calls += 1
        frontier = [_Path(h.passages, h.score) for h in first[: self.cfg.beam]]
        found: dict[tuple[str, ...], float] = {h.passages: h.score for h in first}
        for _ in range(self.cfg.rounds - 1):
            nxt: dict[tuple[str, ...], float] = {}
            for path in frontier:
                hits = self.index.search(self._next_query(question, q, path), self.cfg.per_round)
                self.ann_calls += 1
                for h in hits:
                    if set(h.passages) & set(path.passages):
                        continue
                    key = path.passages + h.passages
                    if len(key) > 4:
                        continue
                    nxt[key] = max(nxt.get(key, float("-inf")), path.score + h.score)
            if not nxt:
                break
            ranked = sorted(nxt.items(), key=lambda kv: (-kv[1], kv[0]))
            frontier = [_Path(p, s) for p, s in ranked[: self.cfg.beam]]
            for p, s in ranked:
                found[p] = max(found.get(p, float("-inf")), s)
        ordered = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
        hits_out = [Hit(entry=-1, score=s, passages=p) for p, s in ordered]
        return Retrieved(passages=unpack(hits_out, k), hits=hits_out[:top_k],
                         latency_ms=(time.perf_counter() - t0) * 1000)
