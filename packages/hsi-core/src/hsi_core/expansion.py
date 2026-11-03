"""Online graph expansion: the baseline the thesis must beat (RQ1).

Take the top seed passages from dense search, expand them along the same capped graph that
builds shortcuts, score every chain, and unpack the best. Two scoring variants:

  vec  : key(c) = normalize(Σ v_p)            cheap, comparable latency to a lookup
  text : key(c) = enc(text(c)) at query time   strongest, but encodes every candidate online
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hsi_core.encoding import chain_text, l2_normalize
from hsi_core.graph import PassageGraph, _prior
from hsi_core.index import Hit, Retrieved, ShortcutIndex, unpack
from hsi_core.interfaces import Encoder


@dataclass(frozen=True)
class ExpansionConfig:
    seeds: int = 5
    beam: int = 3                 # extensions kept per chain per step
    max_passages: int = 3
    variant: Literal["vec", "text"] = "vec"

    def __post_init__(self) -> None:
        if self.seeds < 1 or self.beam < 1 or not 2 <= self.max_passages <= 4:
            raise ValueError("invalid expansion config")


class OnlineExpansion:
    def __init__(self, dense: ShortcutIndex, graph: PassageGraph, cfg: ExpansionConfig,
                 online_encoder: Encoder | None = None) -> None:
        """`online_encoder` encodes candidate chains in the text variant. Pass an uncached
        encoder so measured latency reflects real query-time work."""
        if any(lvl > 0 for lvl in dense.levels):
            raise ValueError("online expansion must start from a passage-only index")
        self.dense, self.graph, self.cfg = dense, graph, cfg
        self.online_encoder = online_encoder or dense.encoder

    def _score(self, q: np.ndarray, chains: list[tuple[str, ...]]) -> np.ndarray:
        if self.cfg.variant == "vec":
            keys = l2_normalize(np.stack([np.sum([self.dense.passage_vecs[p] for p in c], axis=0)
                                          for c in chains]))
        else:
            texts = [chain_text([self.graph.passages[p] for p in c]) for c in chains]
            keys = self.online_encoder.encode(texts, "passage")
        return keys @ q

    def retrieve(self, query: str, k: int, top_k: int) -> Retrieved:
        return self.retrieve_vec(self.dense.encoder.encode([query], "query")[0], k, top_k)

    def retrieve_vec(self, q: np.ndarray, k: int, top_k: int) -> Retrieved:
        """Retrieval from an already-encoded query; latency excludes query encoding."""
        t0 = time.perf_counter()
        seed_hits = self.dense.search(q, max(top_k, self.cfg.seeds))
        scored: dict[tuple[str, ...], float] = {h.passages: h.score for h in seed_hits}
        frontier = [h.passages for h in seed_hits[: self.cfg.seeds]]
        for _ in range(self.cfg.max_passages - 1):
            cands: list[tuple[str, ...]] = []
            for c in frontier:
                nxt = [e for e in self.graph.neighbors(c[-1]) if e.dst not in c]
                nxt.sort(key=lambda e: (-_prior(e), e.dst))
                cands.extend((*c, e.dst) for e in nxt)
            cands = [c for c in dict.fromkeys(cands) if c not in scored]
            if not cands:
                break
            s = self._score(q, cands)
            for c, v in zip(cands, s, strict=True):
                scored[c] = float(v)
            # keep the best `beam` children per parent as the next frontier
            by_parent: dict[tuple[str, ...], list[tuple[float, tuple[str, ...]]]] = {}
            for c, v in zip(cands, s, strict=True):
                by_parent.setdefault(c[:-1], []).append((float(v), c))
            frontier = [c for kids in by_parent.values()
                        for _, c in sorted(kids, key=lambda x: (-x[0], x[1]))[: self.cfg.beam]]
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        hits = [Hit(entry=-1, score=v, passages=c) for c, v in ranked]
        return Retrieved(passages=unpack(hits, k), hits=hits[:top_k],
                         latency_ms=(time.perf_counter() - t0) * 1000)
