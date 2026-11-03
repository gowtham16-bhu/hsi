"""Local question-aware PPR by forward push (Andersen, Chung & Lang 2006). Decisions 0005, 0012.

Global PPR touches every edge of the graph on every query, so its cost grows with the corpus.
Forward push keeps an estimate p and a residual r, starting from r = s (the seed distribution):

    while some node u has r[u] > ε:
        p[u] += α · r[u]
        for each out-edge u → v:  r[v] += (1 − α) · r[u] · P_q(u → v)
        r[u] = 0

Each push settles at least α·ε of probability mass, so the number of pushes is at most 1/(α·ε),
independent of corpus size N. Transitions are computed only for the passages actually pushed:

    rule     P_q(u → v) ∝ w(u, v) · exp(β · cos(q, v_v))
    learned  P_q(u → v) ∝ exp(θ · z(φ(q, u, v)))         (decision 0007, same φ as global PPR)

Only the vectors of pushed passages' neighbours are read. Passages are ranked by p.
"""
from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hsi_core.csr import CompactGraph
from hsi_core.graph import PassageGraph
from hsi_core.index import Retrieved, ShortcutIndex
from hsi_core.transitions import LinearTransition, edge_features


@dataclass(frozen=True)
class PushConfig:
    seeds: int
    restart: float          # α
    epsilon: float          # residual threshold ε (smaller = more accurate, more work)
    beta: float             # rule-based question awareness; 0 = plain PPR; unused with a learned model
    max_pushes: int         # hard safety bound on work per query

    def __post_init__(self) -> None:
        if self.seeds < 1 or not 0.0 < self.restart < 1.0 or self.epsilon <= 0 or self.beta < 0 \
                or self.max_pushes < 1:
            raise ValueError("invalid push config")


class LocalPushRetriever:
    def __init__(self, dense: ShortcutIndex, graph: PassageGraph | CompactGraph, cfg: PushConfig,
                 transition_model: LinearTransition | None = None,
                 vectors_at: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        """`vectors_at(positions)` returns passage vectors by graph position; by default they come from
        the dense index's memory. At scale it is backed by the vector store, so only the vectors of
        visited passages are fetched."""
        if any(lvl > 0 for lvl in dense.levels):
            raise ValueError("seeds must come from a passage-only index")
        self.dense, self.cfg, self.model = dense, cfg, transition_model
        ids = [e[0] for e in dense.entries]
        if isinstance(graph, CompactGraph):
            if graph.ids != ids:
                raise ValueError("compact graph positions differ from the dense index")
            self.g = graph
        else:
            vecs = np.stack([dense.passage_vecs[pid] for pid in ids]).astype(np.float32)
            self.g = CompactGraph.from_graph(graph, ids, vecs)
        if vectors_at is None:
            mat = np.stack([dense.passage_vecs[pid] for pid in ids]).astype(np.float32)
            vectors_at = mat.__getitem__
        self.vectors_at = vectors_at
        self.pos = {pid: i for i, pid in enumerate(ids)}
        self.last_pushes = 0
        self.total_pushes = 0
        self.last_vectors_read = 0

    def _rel(self, nodes: np.ndarray, q: np.ndarray, cache: dict[int, float]) -> np.ndarray:
        missing = [int(v) for v in nodes if int(v) not in cache]
        if missing:
            sims = self.vectors_at(np.asarray(missing, dtype=np.int64)) @ q
            for v, x in zip(missing, sims, strict=True):
                cache[v] = float(x)
            self.last_vectors_read += len(missing)
        return np.fromiter((cache[int(v)] for v in nodes), dtype=np.float64, count=len(nodes))

    def _transitions(self, u: int, q: np.ndarray, rel: dict[int, float],
                     in_q: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        sl = self.g.edges_of(u)
        dst = self.g.dst[sl]
        if len(dst) == 0:
            return dst, np.zeros(0)
        if self.model is not None:
            phi = edge_features(self.g, sl, self._rel(dst, q, rel), self._rel(np.array([u]), q, rel)[0], in_q)
            s = self.model.scores(phi)
            w = np.exp(s - s.max())
        else:
            w = self.g.weight[sl].astype(np.float64)
            if self.cfg.beta > 0:
                sims = self._rel(dst, q, rel)
                w = w * np.exp(self.cfg.beta * (sims - sims.max()))
        return dst, w / w.sum()

    def retrieve_vec(self, q: np.ndarray, k: int, top_k: int, question: str = "") -> Retrieved:
        t0 = time.perf_counter()
        hits = self.dense.search(q, self.cfg.seeds)
        seeds = {self.pos[h.passages[0]]: max(h.score, 0.0) for h in hits}
        total = sum(seeds.values())
        if total == 0:
            return Retrieved(passages=[], hits=hits, latency_ms=(time.perf_counter() - t0) * 1000)
        if self.model is not None and not question:
            raise ValueError("learned transitions need the question text")
        in_q = self.g.bridges_in(question) if self.model is not None else None
        q = np.asarray(q, dtype=np.float32)
        a, eps = self.cfg.restart, self.cfg.epsilon
        r = {u: v / total for u, v in seeds.items()}
        p: dict[int, float] = {}
        rel: dict[int, float] = {}
        self.last_vectors_read = 0
        heap = [(-v, u) for u, v in r.items() if v > eps]
        heapq.heapify(heap)
        pushes = 0
        while heap and pushes < self.cfg.max_pushes:
            _, u = heapq.heappop(heap)
            ru = r.get(u, 0.0)
            if ru <= eps:
                continue                       # stale heap entry
            pushes += 1
            p[u] = p.get(u, 0.0) + a * ru
            r[u] = 0.0
            dst, prob = self._transitions(u, q, rel, in_q)
            spread = (1 - a) * ru
            if len(dst) == 0:                  # dangling: mass returns to the seeds
                targets = [(v, w / total) for v, w in seeds.items()]
            else:
                targets = zip(dst.tolist(), prob.tolist(), strict=True)
            for v, pr in targets:
                nv = r.get(v, 0.0) + spread * pr
                r[v] = nv
                if nv > eps:
                    heapq.heappush(heap, (-nv, v))
        self.last_pushes = pushes
        self.total_pushes += pushes
        ranked = sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return Retrieved(passages=[self.g.ids[u] for u, _ in ranked], hits=hits[:top_k],
                         latency_ms=(time.perf_counter() - t0) * 1000)
