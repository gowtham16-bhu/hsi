"""The retrieval service: loads one index and answers permission-aware retrieval requests (A5 §3–5)."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.config import PluginSpec
from hsi_core.csr import CompactGraph
from hsi_core.index import ShortcutIndex
from hsi_core.push import LocalPushRetriever, PushConfig
from hsi_core.transitions import LinearTransition


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServeConfig(_Strict):
    index_dir: str
    encoder: PluginSpec                  # must be the encoder the index was built with
    vector_store: PluginSpec
    metadata_store: PluginSpec
    push: dict[str, Any]                 # PushConfig options
    max_k: int
    mcp_principals: list[str]            # identity used for MCP requests (A5 §3)
    audit: bool


class RetrievalService:
    def __init__(self, cfg: ServeConfig) -> None:
        self.cfg = cfg
        root = Path(cfg.index_dir)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.index_id: str = self.manifest["index_id"]
        self.encoder = registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options)
        if self.encoder.name != self.manifest["encoder"]["name"]:
            raise ValueError(f"encoder {self.encoder.name!r} differs from the index's "
                             f"{self.manifest['encoder']['name']!r}; queries and passages must share one model")
        ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
        vecs = np.load(root / "vectors.npy")
        dense = ShortcutIndex(self.encoder, lambda: registry.load(
            "vector_store", cfg.vector_store.plugin, **cfg.vector_store.options)).build_from_vectors(ids, vecs)
        graph = CompactGraph.load(root / "graph.npz")
        self.push = LocalPushRetriever(dense, graph, PushConfig(**cfg.push),
                                       transition_model=LinearTransition.load(root / "transitions.json"))
        self.store = registry.load("metadata_store", cfg.metadata_store.plugin, **cfg.metadata_store.options)
        self._lock = threading.Lock()
        acls = self.store.read_acls(self.index_id)
        missing = [pid for pid in ids if pid not in acls]
        if missing:
            raise ValueError(f"{len(missing)} passages of index {self.index_id!r} have no ACL row in the store")
        self.public = np.array([bool(acls[pid].get("public")) for pid in ids])
        members: dict[str, list[int]] = defaultdict(list)
        for i, pid in enumerate(ids):
            for pr in acls[pid].get("principals", []):
                members[pr].append(i)
        self._members = {pr: np.asarray(v, dtype=np.int64) for pr, v in members.items()}
        self.n = len(ids)

    def visible(self, principals: list[str]) -> np.ndarray:
        """A passage is visible when it is public or shares a principal with the request (A5 §4)."""
        mask = self.public.copy()
        for pr in principals:
            pos = self._members.get(pr)
            if pos is not None:
                mask[pos] = True
        return mask

    def retrieve(self, question: str, k: int, principals: list[str]) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("empty question")
        if not 1 <= k <= self.cfg.max_k:
            raise ValueError(f"k must be between 1 and {self.cfg.max_k}")
        t0 = time.perf_counter()
        qv = self.encoder.encode([question], "query")[0]
        r = self.push.retrieve_vec(qv, k, k, question, visible=self.visible(principals))
        with self._lock:
            rows = self.store.read_passages(self.index_id, r.passages)
            if self.cfg.audit:
                self.store.append_audit(self.index_id, principals, question, r.passages)
        passages = [{"id": pid, "doc_id": rows[pid]["doc_id"], "title": rows[pid]["title"],
                     "text": rows[pid]["text"], "score": s} for pid, s in zip(r.passages, r.scores, strict=True)]
        return {"index_id": self.index_id, "passages": passages, "latency_ms": (time.perf_counter() - t0) * 1000}

    def health(self) -> dict[str, Any]:
        return {"index_id": self.index_id, "passages": self.n, "edges": self.push.g.edge_count,
                "encoder": self.manifest["encoder"]["name"]}

    def close(self) -> None:
        self.store.close()
