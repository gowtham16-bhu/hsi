"""Compact (CSR) passage graph for walks (decision 0012).

Edges of passage u are dst[indptr[u]:indptr[u+1]]. Every per-edge array is aligned with `dst`.
Memory per edge: dst int32 + bridge int32 + four float32 features = 24 bytes, versus several hundred
bytes for a Python `Edge` object.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hsi_core.graph import PassageGraph
from hsi_core.model import EdgeType
from hsi_core.text import tokens
from hsi_core.titles import TokenAhoCorasick


def cos_uv(vecs: np.ndarray, src: np.ndarray, dst: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """cos(u, v) per edge, in chunks so memory stays O(chunk · d), not O(edges · d)."""
    vecs = np.asarray(vecs, dtype=np.float32)
    out = np.empty(len(dst), dtype=np.float32)
    for a in range(0, len(dst), chunk):
        b = min(a + chunk, len(dst))
        out[a:b] = np.einsum("ij,ij->i", vecs[src[a:b]], vecs[dst[a:b]])
    return out


@dataclass
class CompactGraph:
    ids: list[str]                 # passage position -> passage id
    indptr: np.ndarray             # int64 (n + 1)
    dst: np.ndarray                # int32 (e)
    weight: np.ndarray             # float32 (e)
    salience: np.ndarray           # float32 (e)
    co_mention: np.ndarray         # float32 (e), 1.0 for co-mention edges
    bridge: np.ndarray             # int32 (e), index into `bridges`
    cos_uv: np.ndarray             # float32 (e), cos(u, v) of the passage vectors
    bridges: list[str]             # entity keys
    bridge_log_df: np.ndarray      # float64 (b), log(1 + participants)

    def __post_init__(self) -> None:
        self._matcher: TokenAhoCorasick | None = None

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def edge_count(self) -> int:
        return len(self.dst)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.indptr, self.dst, self.weight, self.salience, self.co_mention,
                                      self.bridge, self.cos_uv, self.bridge_log_df))

    def src(self) -> np.ndarray:
        """Source position of every edge (materialized on demand; global walks need it)."""
        return np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))

    def edges_of(self, u: int) -> slice:
        return slice(int(self.indptr[u]), int(self.indptr[u + 1]))

    def bridges_in(self, question: str) -> np.ndarray:
        """Boolean mask over bridges whose surface form occurs in the question text."""
        if self._matcher is None:
            m = TokenAhoCorasick()
            for i, b in enumerate(self.bridges):
                toks = tokens(b.split(":", 1)[1])
                if toks:
                    m.add(toks, str(i))
            m.build()
            self._matcher = m
        mask = np.zeros(len(self.bridges), dtype=bool)
        for hit in self._matcher.find(tokens(question)):
            mask[int(hit.key)] = True
        return mask

    @classmethod
    def from_graph(cls, graph: PassageGraph, ids: list[str], vecs: np.ndarray) -> CompactGraph:
        """`ids` and `vecs` give passage positions and their vectors (row i is passage ids[i])."""
        pos = {pid: i for i, pid in enumerate(ids)}
        bridges = sorted({e.bridge for es in graph.out_edges.values() for e in es})
        bpos = {b: i for i, b in enumerate(bridges)}
        counts = np.zeros(len(ids) + 1, dtype=np.int64)
        rows: list[list] = [[] for _ in ids]
        for s, edges in graph.out_edges.items():
            if s not in pos:
                continue
            row = rows[pos[s]]
            for e in edges:
                if e.dst in pos:
                    row.append((pos[e.dst], e.weight, e.salience,
                                1.0 if e.type is EdgeType.CO_MENTION else 0.0, bpos[e.bridge]))
        for i, row in enumerate(rows):
            counts[i + 1] = len(row)
        indptr = np.cumsum(counts)
        flat = [x for row in rows for x in row]
        dst = np.fromiter((x[0] for x in flat), dtype=np.int32, count=len(flat))
        src = np.repeat(np.arange(len(ids), dtype=np.int64), np.diff(indptr))
        df = np.asarray([graph.participants.get(b, 1) for b in bridges], dtype=np.float64)
        return cls(ids=list(ids), indptr=indptr, dst=dst,
                   weight=np.fromiter((x[1] for x in flat), dtype=np.float32, count=len(flat)),
                   salience=np.fromiter((x[2] for x in flat), dtype=np.float32, count=len(flat)),
                   co_mention=np.fromiter((x[3] for x in flat), dtype=np.float32, count=len(flat)),
                   bridge=np.fromiter((x[4] for x in flat), dtype=np.int32, count=len(flat)),
                   cos_uv=cos_uv(vecs, src, dst), bridges=bridges, bridge_log_df=np.log1p(df))

    def save(self, path: str | Path) -> None:
        np.savez(path, ids=np.asarray(self.ids, dtype=str), indptr=self.indptr, dst=self.dst,
                 weight=self.weight, salience=self.salience, co_mention=self.co_mention, bridge=self.bridge,
                 cos_uv=self.cos_uv, bridges=np.asarray(self.bridges, dtype=str), bridge_log_df=self.bridge_log_df)

    @classmethod
    def load(cls, path: str | Path) -> CompactGraph:
        with np.load(path, allow_pickle=False) as d:
            return cls(ids=d["ids"].tolist(), indptr=d["indptr"], dst=d["dst"], weight=d["weight"],
                       salience=d["salience"], co_mention=d["co_mention"], bridge=d["bridge"],
                       cos_uv=d["cos_uv"], bridges=d["bridges"].tolist(), bridge_log_df=d["bridge_log_df"])
