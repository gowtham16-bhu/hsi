"""Personalized PageRank from dense seeds over the passage graph (decisions 0003, 0004).

    r = α·s + (1 − α)·P_qᵀ r
    s        ∝ dense scores of the top `seeds` passages
    P_q(u→v) ∝ w(u,v) · exp(β · cos(q, v_v))          question-aware transitions ("intelligent
                                                      surfer", Richardson & Domingos 2002)

β = 0 is plain PPR.

Ranking (decision 0006):
    total    rank by r
    arrived  rank by r − α·s, the mass that reached a passage through edges; the top `keep_seeds`
             dense seeds are placed first, the rest of the budget is filled by arrived mass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hsi_core.csr import CompactGraph
from hsi_core.graph import PassageGraph
from hsi_core.index import Retrieved, ShortcutIndex
from hsi_core.transitions import EdgeTable, LinearTransition


@dataclass(frozen=True)
class PPRConfig:
    seeds: int
    restart: float          # α
    iterations: int
    tolerance: float
    beta: float             # question awareness of transitions; 0 = plain PPR
    rank_by: Literal["total", "arrived"]
    keep_seeds: int         # with rank_by=arrived: dense seeds placed first

    def __post_init__(self) -> None:
        if self.seeds < 1 or not 0.0 < self.restart < 1.0 or self.iterations < 1 or self.tolerance <= 0 \
                or self.beta < 0 or not 0 <= self.keep_seeds <= self.seeds:
            raise ValueError("invalid PPR config")


class PPRRetriever:
    def __init__(self, dense: ShortcutIndex, graph: PassageGraph | CompactGraph, cfg: PPRConfig,
                 transition_model: LinearTransition | None = None) -> None:
        """With `transition_model`, P_q(u→v) ∝ exp(θ·z(φ)) (decision 0007) replaces the β rule.
        `graph` may be given already compacted (decision 0012); positions must follow the dense index."""
        if any(lvl > 0 for lvl in dense.levels):
            raise ValueError("PPR seeds must come from a passage-only index")
        self.dense, self.cfg = dense, cfg
        self.ids = [e[0] for e in dense.entries]
        self.vecs = np.stack([dense.passage_vecs[pid] for pid in self.ids]).astype(np.float32) \
            if self.ids else np.zeros((0, 1), dtype=np.float32)
        self.table = EdgeTable(graph, self.ids, self.vecs)
        self.src, self.dst = self.table.src, self.table.dst
        self.w_raw = self.table.csr.weight.astype(np.float64)
        self.model = transition_model

    def _transitions(self, q: np.ndarray, question: str) -> tuple[np.ndarray, np.ndarray]:
        w = self.w_raw
        if self.model is not None:
            s = self.model.scores(self.table.features(q, question))
            w = np.exp(s - s.max())
        elif self.cfg.beta > 0:
            rel = self.vecs @ q.astype(np.float32)
            w = w * np.exp(self.cfg.beta * (rel[self.dst] - rel.max()))   # shift for numerical stability
        out_sum = np.bincount(self.src, weights=w, minlength=len(self.ids))
        return w / np.maximum(out_sum[self.src], 1e-300), out_sum == 0

    def retrieve_vec(self, q: np.ndarray, k: int, top_k: int, question: str = "") -> Retrieved:
        t0 = time.perf_counter()
        hits = self.dense.search(q, self.cfg.seeds)
        s = np.zeros(len(self.ids))
        for h in hits:
            s[h.entry] = max(h.score, 0.0)
        if s.sum() == 0:
            return Retrieved(passages=[], hits=hits, latency_ms=(time.perf_counter() - t0) * 1000)
        s /= s.sum()
        if self.model is not None and not question:
            raise ValueError("learned transitions need the question text")
        trans, dangling = self._transitions(q, question)
        r = s.copy()
        a = self.cfg.restart
        for _ in range(self.cfg.iterations):
            nxt = np.bincount(self.dst, weights=trans * r[self.src], minlength=len(r))
            nxt += r[dangling].sum() * s          # dangling mass returns to the seeds
            nxt = a * s + (1 - a) * nxt
            done = np.abs(nxt - r).sum() < self.cfg.tolerance
            r = nxt
            if done:
                break
        if self.cfg.rank_by == "total":
            order = np.lexsort((np.arange(len(r)), -r))[:k]
            out = [self.ids[i] for i in order if r[i] > 0]
        else:
            first = [h.passages[0] for h in hits[: self.cfg.keep_seeds]]
            arrived = r - a * s
            order = np.lexsort((np.arange(len(arrived)), -arrived))
            rest = [self.ids[i] for i in order[: k + len(first)] if arrived[i] > 0 and self.ids[i] not in first]
            out = (first + rest)[:k]
        return Retrieved(passages=out, hits=hits[:top_k],
                         latency_ms=(time.perf_counter() - t0) * 1000)
