"""Graph-smoothed passage vectors (decision 0008). Index time only; no LLM, no re-encoding.

    v'_p = normalize((1 − γ)·v_p + γ · Σ_{n ∈ N(p)} w(p, n)·v_n / Σ_n w(p, n))

repeated `steps` times over the capped graph. Passages without neighbours keep their vector.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from hsi_core.encoding import l2_normalize
from hsi_core.graph import PassageGraph


@dataclass(frozen=True)
class SmoothingConfig:
    gamma: float
    steps: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0 or self.steps < 1:
            raise ValueError("invalid smoothing config")


def smooth_vectors(vecs: Mapping[str, np.ndarray], graph: PassageGraph, cfg: SmoothingConfig) -> dict[str, np.ndarray]:
    ids = list(vecs)
    pos = {pid: i for i, pid in enumerate(ids)}
    src, dst, w = [], [], []
    for s, edges in graph.out_edges.items():
        if s not in pos:
            continue
        for e in edges:
            if e.dst in pos:
                src.append(pos[s])
                dst.append(pos[e.dst])
                w.append(e.weight)
    v = np.stack([vecs[pid] for pid in ids]).astype(np.float32)
    if not src or cfg.gamma == 0.0:
        return {pid: v[i] for i, pid in enumerate(ids)}
    src_a, dst_a = np.asarray(src), np.asarray(dst)
    w_a = np.asarray(w, dtype=np.float32)
    order = np.argsort(src_a, kind="stable")
    src_a, dst_a, w_a = src_a[order], dst_a[order], w_a[order]
    starts = np.flatnonzero(np.r_[True, src_a[1:] != src_a[:-1]])
    owners = src_a[starts]
    wsum = np.add.reduceat(w_a, starts)
    for _ in range(cfg.steps):
        agg = np.add.reduceat(v[dst_a] * w_a[:, None], starts, axis=0) / wsum[:, None]
        new = v.copy()
        new[owners] = (1 - cfg.gamma) * v[owners] + cfg.gamma * l2_normalize(agg)
        v = l2_normalize(new)
    return {pid: v[i] for i, pid in enumerate(ids)}
