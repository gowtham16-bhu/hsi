"""`hsi-index build`: connectors → passages → vectors → streamed graph → a versioned index (A5 §1–2).

No LLM is called. The only connections made are to the configured connectors and to PostgreSQL.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.build import GraphSection, build_compact
from hsi_core.config import PluginSpec, config_hash
from hsi_core.connectors import CursorCheckpoint
from hsi_core.encoding import CachedEncoder
from hsi_core.index import ShortcutIndex
from hsi_core.ingest import ChunkConfig, ingest_document
from hsi_core.model import Passage, SourceDocument, Tombstone
from hsi_core.transitions import LinearTransition


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkSection(_Strict):
    min_words: int
    max_words: int
    modulus: int
    window: int


class IndexConfig(_Strict):
    index_id: str
    out_dir: str
    sources: list[PluginSpec]            # connector plugins
    chunking: ChunkSection
    encoder: PluginSpec
    cache_dir: str | None                # embedding cache; None = no cache
    vector_store: PluginSpec
    metadata_store: PluginSpec
    graph: GraphSection
    transition_model: str


def _documents(cfg: IndexConfig) -> tuple[list[SourceDocument], dict[str, dict[str, str]]]:
    docs: dict[str, SourceDocument] = {}
    cursors: dict[str, dict[str, str]] = {}
    for src in cfg.sources:
        conn = registry.load("connector", src.plugin, **src.options)
        for stream in conn.streams():
            for rec in conn.sync(stream, None):
                if isinstance(rec, SourceDocument):
                    docs[rec.id] = rec
                elif isinstance(rec, Tombstone):
                    docs.pop(rec.record_id, None)
                elif isinstance(rec, CursorCheckpoint):
                    cursors.setdefault(conn.name, {})[rec.stream] = rec.cursor
    return [docs[k] for k in sorted(docs)], cursors


def build(cfg: IndexConfig) -> dict[str, Any]:
    t_start = time.perf_counter()
    docs, cursors = _documents(cfg)
    chunk = ChunkConfig(**cfg.chunking.model_dump())
    passages: dict[str, Passage] = {}
    for d in docs:
        for p in ingest_document(d, chunk):
            passages.setdefault(p.id, p)
    plist = list(passages.values())
    if not plist:
        raise ValueError("no passages: the sources returned no documents")

    raw = registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options)
    enc = CachedEncoder(raw, cfg.cache_dir) if cfg.cache_dir else raw
    t0 = time.perf_counter()
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", cfg.vector_store.plugin,
                                                     **cfg.vector_store.options)).build(plist)
    t_encode = time.perf_counter() - t0
    ids = [e[0] for e in dense.entries]
    vecs = np.stack([dense.passage_vecs[p] for p in ids]).astype(np.float32)
    t0 = time.perf_counter()
    graph, stats = build_compact(cfg.graph, plist, ids, vecs, vector_of=dense.raw_vecs.__getitem__)
    t_graph = time.perf_counter() - t0
    LinearTransition.load(cfg.transition_model)          # fail early on a bad model file

    out = Path(cfg.out_dir) / cfg.index_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs)
    (out / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    graph.save(out / "graph.npz")
    shutil.copyfile(cfg.transition_model, out / "transitions.json")

    store = registry.load("metadata_store", cfg.metadata_store.plugin, **cfg.metadata_store.options)
    try:
        store.migrate()
        store.delete_corpus(cfg.index_id)
        store.write_passages(cfg.index_id, plist)
    finally:
        store.close()

    manifest = {"index_id": cfg.index_id, "config": cfg.model_dump(), "config_hash": config_hash(cfg),
                "encoder": {"name": raw.name, "dim": int(vecs.shape[1])}, "documents": len(docs),
                "passages": len(ids), "graph": stats, "cursors": cursors, "llm_calls": 0,
                "build_seconds": {"encode": t_encode, "graph": t_graph, "total": time.perf_counter() - t_start}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
