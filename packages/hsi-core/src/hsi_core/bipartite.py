"""Question-aware walk over the passage–entity graph (decision 0004, candidate 2).

Nodes are passages and entities. A passage links to every entity it mentions (or is about):

    p → e  ∝ salience(p, e) · spec(e)                  spec(e) = 1 / df(e)^γ
    e → p  ∝ salience(p, e) · exp(β · cos(q, v_p))     question-aware arrival

No fan-out cap and no hub cut: frequent entities are down-weighted by spec(e). γ = 0 treats every
entity alike; γ = 1 matches HippoRAG's node specificity.

    r = α·s + (1 − α)·P_qᵀ r        s ∝ dense scores of the top `seeds` passages

Passages are ranked by r restricted to passage nodes. Two walk steps (passage → entity → passage)
correspond to one hop of the passage graph.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hsi_core.graph import PassageGraph
from hsi_core.index import Retrieved, ShortcutIndex


@dataclass(frozen=True)
class BipartiteConfig:
    seeds: int
    restart: float          # α
    iterations: int
    tolerance: float
    beta: float             # question awareness of e → p arrivals
    gamma: float            # entity specificity exponent

    def __post_init__(self) -> None:
        if self.seeds < 1 or not 0.0 < self.restart < 1.0 or self.iterations < 1 or self.tolerance <= 0 \
                or self.beta < 0 or self.gamma < 0:
            raise ValueError("invalid bipartite walk config")


class BipartiteWalkRetriever:
    def __init__(self, dense: ShortcutIndex, graph: PassageGraph, cfg: BipartiteConfig) -> None:
        if any(lvl > 0 for lvl in dense.levels):
            raise ValueError("seeds must come from a passage-only index")
        self.dense, self.cfg = dense, cfg
        self.pids = [e[0] for e in dense.entries]
        ppos = {pid: i for i, pid in enumerate(self.pids)}
        ents = sorted({e for m in graph.mention_salience.values() for e in m})
        epos = {e: len(self.pids) + i for i, e in enumerate(ents)}
        self.n = len(self.pids) + len(ents)
        p_idx, e_idx, sal = [], [], []
        for pid, m in graph.mention_salience.items():
            if pid not in ppos:
                continue
            for ent, s in m.items():
                p_idx.append(ppos[pid])
                e_idx.append(epos[ent])
                sal.append(max(s, 1e-6))
        self.p_idx = np.asarray(p_idx, dtype=np.int64)
        self.e_idx = np.asarray(e_idx, dtype=np.int64)
        self.sal = np.asarray(sal, dtype=np.float64)
        df = np.bincount(self.e_idx, minlength=self.n).astype(np.float64)
        spec = 1.0 / np.maximum(df, 1.0) ** cfg.gamma
        # p -> e transitions do not depend on the question: normalize once
        w_pe = self.sal * spec[self.e_idx]
        out_p = np.bincount(self.p_idx, weights=w_pe, minlength=self.n)
        self.t_pe = w_pe / np.maximum(out_p[self.p_idx], 1e-300)
        self.p_has_out = out_p > 0
        self.vecs = np.stack([dense.passage_vecs[pid] for pid in self.pids]).astype(np.float32) \
            if self.pids else np.zeros((0, 1), dtype=np.float32)

    def _e_to_p(self, q: np.ndarray) -> np.ndarray:
        w = self.sal
        if self.cfg.beta > 0:
            rel = self.vecs @ q.astype(np.float32)
            w = w * np.exp(self.cfg.beta * (rel[self.p_idx] - rel.max()))
        out_e = np.bincount(self.e_idx, weights=w, minlength=self.n)
        return w / np.maximum(out_e[self.e_idx], 1e-300)

    def retrieve_vec(self, q: np.ndarray, k: int, top_k: int) -> Retrieved:
        t0 = time.perf_counter()
        hits = self.dense.search(q, self.cfg.seeds)
        s = np.zeros(self.n)
        for h in hits:
            s[h.entry] = max(h.score, 0.0)
        if s.sum() == 0:
            return Retrieved(passages=[], hits=hits, latency_ms=(time.perf_counter() - t0) * 1000)
        s /= s.sum()
        t_ep = self._e_to_p(q)
        a = self.cfg.restart
        r = s.copy()
        for _ in range(self.cfg.iterations):
            nxt = np.bincount(self.e_idx, weights=self.t_pe * r[self.p_idx], minlength=self.n)  # p -> e
            nxt += np.bincount(self.p_idx, weights=t_ep * r[self.e_idx], minlength=self.n)    # e -> p
            stuck = r[: len(self.pids)][~self.p_has_out[: len(self.pids)]].sum()
            nxt += stuck * s                                             # passages with no entities restart
            nxt = a * s + (1 - a) * nxt
            done = np.abs(nxt - r).sum() < self.cfg.tolerance
            r = nxt
            if done:
                break
        rp = r[: len(self.pids)]
        order = np.lexsort((np.arange(len(rp)), -rp))[:k]
        return Retrieved(passages=[self.pids[i] for i in order if rp[i] > 0], hits=hits[:top_k],
                         latency_ms=(time.perf_counter() - t0) * 1000)
