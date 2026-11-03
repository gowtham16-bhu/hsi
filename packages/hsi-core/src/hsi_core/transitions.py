"""Edge features and learned log-linear transitions for graph walks (decision 0007).

One function computes the features for both training and inference, so they cannot drift apart.

    P(u → v | q) ∝ exp(θ · (φ(q, u, v) − μ) / σ)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hsi_core.csr import CompactGraph
from hsi_core.graph import PassageGraph
from hsi_core.model import EdgeType

FEATURES = ("cos_q_v", "cos_u_v", "cos_q_u", "edge_weight", "log_df", "bridge_in_question",
            "co_mention", "salience")
# question-independent subset, for index-time edge selection (decision 0010)
EDGE_FEATURES = ("cos_u_v", "edge_weight", "log_df", "co_mention", "salience")


def candidate_features(src_vec: np.ndarray, cands: list[tuple], participants, vector_of) -> np.ndarray:
    """EDGE_FEATURES for one passage's candidate edges (dst, weight, bridge, salience, type)."""
    return np.array([[float(vector_of(d) @ src_vec), w, np.log1p(participants.get(b, 1)),
                      1.0 if t is EdgeType.CO_MENTION else 0.0, s] for d, w, b, s, t in cands], dtype=np.float64)


def make_edge_scorer(model: LinearTransition, vector_of):
    """An index-time EdgeScorer from a model trained on EDGE_FEATURES."""
    if model.features != EDGE_FEATURES:
        raise ValueError("edge scorer needs a model trained on EDGE_FEATURES")

    def score(src: str, cands: list[tuple], participants) -> np.ndarray:
        return model.scores(candidate_features(vector_of(src), cands, participants, vector_of))
    return score


def stack_features(rel_dst: np.ndarray, cos_uv: np.ndarray, rel_src: np.ndarray, weight: np.ndarray,
                   log_df: np.ndarray, in_question: np.ndarray, co_mention: np.ndarray,
                   salience: np.ndarray) -> np.ndarray:
    """φ in FEATURES order for a set of edges. Every walk and the trainer call this one function."""
    return np.stack([rel_dst, cos_uv, rel_src, weight, log_df, in_question, co_mention, salience],
                    axis=1).astype(np.float64)


def edge_features(g: CompactGraph, edges: slice | np.ndarray, rel_dst: np.ndarray, rel_src: np.ndarray | float,
                  in_q: np.ndarray) -> np.ndarray:
    """φ for the given edges of a compact graph. `rel_*` are cos(q, ·); `in_q` is the question's
    bridge mask (CompactGraph.bridges_in)."""
    b = g.bridge[edges]
    n = len(b)
    return stack_features(rel_dst, g.cos_uv[edges], np.broadcast_to(rel_src, (n,)), g.weight[edges],
                          g.bridge_log_df[b], in_q[b].astype(np.float64), g.co_mention[edges], g.salience[edges])


class EdgeTable:
    """All edges of a compact graph with per-query features (global walks and training)."""

    def __init__(self, graph: PassageGraph | CompactGraph, ids: list[str], vecs: np.ndarray) -> None:
        self.csr = graph if isinstance(graph, CompactGraph) else CompactGraph.from_graph(graph, ids, vecs)
        if self.csr.ids != list(ids):
            raise ValueError("compact graph positions differ from `ids`")
        self.n = self.csr.n
        self.src = self.csr.src()
        self.dst = self.csr.dst.astype(np.int64)
        self.vecs = np.asarray(vecs, dtype=np.float32)

    def bridges_in(self, question: str) -> np.ndarray:
        return self.csr.bridges_in(question)

    def features(self, q: np.ndarray, question: str) -> np.ndarray:
        rel = (self.vecs @ q.astype(np.float32)).astype(np.float64)
        return edge_features(self.csr, slice(None), rel[self.dst], rel[self.src], self.bridges_in(question))


@dataclass(frozen=True)
class LinearTransition:
    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    theta: np.ndarray

    def __post_init__(self) -> None:
        if self.features not in (FEATURES, EDGE_FEATURES):
            raise ValueError(f"model features {self.features} match neither {FEATURES} nor {EDGE_FEATURES}")

    def scores(self, phi: np.ndarray) -> np.ndarray:
        return ((phi - self.mean) / self.std) @ self.theta

    def save(self, path: str | Path, meta: dict) -> None:
        Path(path).write_text(json.dumps({"features": list(self.features), "mean": self.mean.tolist(),
                                          "std": self.std.tolist(), "theta": self.theta.tolist(),
                                          "meta": meta}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LinearTransition:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["features"]), np.asarray(d["mean"]), np.asarray(d["std"]), np.asarray(d["theta"]))


def fit_listwise(groups: list[tuple[np.ndarray, int]], l2: float, lr: float, epochs: int,
                 seed: int, features: tuple[str, ...] = FEATURES) -> LinearTransition:
    """Fit θ by listwise softmax cross-entropy: each group is (features of u's out-edges, gold index)."""
    allphi = np.concatenate([g for g, _ in groups])
    mean, std = allphi.mean(0), np.maximum(allphi.std(0), 1e-6)
    zs = [((g - mean) / std, y) for g, y in groups]
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.01, size=allphi.shape[1])
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    for t in range(1, epochs + 1):
        grad = l2 * theta
        for z, y in zs:
            s = z @ theta
            p = np.exp(s - s.max())
            p /= p.sum()
            grad += (p @ z - z[y]) / len(zs)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return LinearTransition(features, mean, std, theta)


def listwise_loss(model: LinearTransition, groups: list[tuple[np.ndarray, int]]) -> tuple[float, float]:
    """Mean cross-entropy and top-1 accuracy of the gold next hop."""
    ce, acc = 0.0, 0
    for g, y in groups:
        s = model.scores(g)
        s = s - s.max()
        ce += -(s[y] - np.log(np.exp(s).sum()))
        acc += int(np.argmax(s) == y)
    return ce / len(groups), acc / len(groups)
