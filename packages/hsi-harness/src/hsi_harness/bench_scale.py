"""Scaling benchmark (decisions 0005, 0012): does per-query walk cost stay flat as the corpus grows?

For each corpus size N, the corpus is the benchmark questions' own passages plus real passages from
the pooled datasets (seeded shuffle), so quality is measured on the same questions against more and
more distractors. Reported per N: graph build time, edges, compact-graph bytes per edge, pushes and
vectors read per query, walk latency (seed search timed separately) and gold-passage recall@k for
dense, global learned PPR and learned push.

Usage: `hsi-bench-scale --config configs/bench/scale.yaml`. No LLM is used.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.config import PluginSpec, config_hash, load_yaml
from hsi_core.csr import CompactGraph
from hsi_core.encoding import CachedEncoder
from hsi_core.index import ShortcutIndex
from hsi_core.metrics import percentile
from hsi_core.ppr import PPRConfig, PPRRetriever
from hsi_core.push import LocalPushRetriever, PushConfig
from hsi_core.transitions import LinearTransition
from hsi_harness.datasets import load
from hsi_harness.gate0 import GraphSection, SampleRef, build_graph


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PoolSource(_Strict):
    dataset: str
    data_file: str


class BenchConfig(_Strict):
    questions: PoolSource             # benchmark questions come from this source
    question_count: int
    seed: int
    exclude_samples: list[SampleRef]  # earlier samples of the same source, each drawn excluding the ones before it
    pool: list[PoolSource]            # passages added as distractors, in this order of preference
    sizes: list[int]
    encoder: PluginSpec
    vector_store: PluginSpec
    cache_dir: str
    out_dir: str
    graph: GraphSection
    transition_model: str
    ppr: dict[str, Any]               # PPRConfig options for the global reference walk
    push: dict[str, Any]              # PushConfig options
    global_ppr_max_size: int          # global PPR is skipped above this size (it is the slow reference)
    k: int


def _recall(got: list[str], gold: tuple[str, ...], k: int) -> float:
    return len(set(gold) & set(got[:k])) / len(gold)


def run(cfg: BenchConfig) -> dict[str, Any]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    excluded: set[str] = set()
    for ref in cfg.exclude_samples:
        excluded |= {q.id for q in load(cfg.questions.dataset, cfg.questions.data_file, ref.size, ref.seed,
                                        exclude=excluded or None).questions}
    qds = load(cfg.questions.dataset, cfg.questions.data_file, cfg.question_count, cfg.seed, exclude=excluded)
    own = {p.id: p for p in qds.passages}
    pool: dict[str, Any] = {}
    for src in cfg.pool:
        for p in load(src.dataset, src.data_file, None, cfg.seed).passages:
            if p.id not in own:
                pool.setdefault(p.id, p)
    order = sorted(pool)
    random.Random(cfg.seed).shuffle(order)
    if max(cfg.sizes) > len(own) + len(order):
        raise ValueError(f"largest size {max(cfg.sizes)} exceeds the {len(own) + len(order)} real passages pooled")

    enc = CachedEncoder(registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options), cfg.cache_dir)
    model = LinearTransition.load(cfg.transition_model)
    qvecs = enc.encode([q.question for q in qds.questions], "query")
    rows = []
    for n in sorted(cfg.sizes):
        passages = list(own.values()) + [pool[i] for i in order[: n - len(own)]]
        t0 = time.perf_counter()
        dense = ShortcutIndex(enc, lambda: registry.load("vector_store", cfg.vector_store.plugin,
                                                         **cfg.vector_store.options)).build(passages)
        t_encode_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        graph = build_graph(cfg.graph, passages, vector_of=dense.raw_vecs.__getitem__)
        t_graph = time.perf_counter() - t0
        ids = [e[0] for e in dense.entries]
        vecs = np.stack([dense.passage_vecs[pid] for pid in ids]).astype(np.float32)
        t0 = time.perf_counter()
        csr = CompactGraph.from_graph(graph, ids, vecs)
        t_csr = time.perf_counter() - t0
        del graph

        push = LocalPushRetriever(dense, csr, PushConfig(**cfg.push), transition_model=model)
        glob = PPRRetriever(dense, csr, PPRConfig(**cfg.ppr), transition_model=model) \
            if n <= cfg.global_ppr_max_size else None
        seeds = int(cfg.push["seeds"])
        lat: dict[str, list[float]] = {"seed_search": [], "push": [], "global": []}
        rec: dict[str, list[float]] = {"dense": [], "push": [], "global": []}
        pushes, reads = [], []
        for q, qv in zip(qds.questions, qvecs, strict=True):
            t0 = time.perf_counter()
            hits = dense.search(qv, max(seeds, cfg.k))
            lat["seed_search"].append((time.perf_counter() - t0) * 1000)
            rec["dense"].append(_recall([h.passages[0] for h in hits], q.gold, cfg.k))
            r = push.retrieve_vec(qv, cfg.k, cfg.k, q.question)
            lat["push"].append(r.latency_ms)
            rec["push"].append(_recall(r.passages, q.gold, cfg.k))
            pushes.append(push.last_pushes)
            reads.append(push.last_vectors_read)
            if glob is not None:
                g = glob.retrieve_vec(qv, cfg.k, cfg.k, q.question)
                lat["global"].append(g.latency_ms)
                rec["global"].append(_recall(g.passages, q.gold, cfg.k))
        row = {
            "passages": len(passages), "edges": csr.edge_count,
            "csr_bytes_per_edge": csr.nbytes / max(1, csr.edge_count),
            "build_s": {"encode_and_index": t_encode_index, "graph": t_graph, "compact": t_csr},
            "pushes_per_query": {"p50": percentile(pushes, 50), "p95": percentile(pushes, 95), "max": max(pushes)},
            "vectors_read_per_query": {"p50": percentile(reads, 50), "p95": percentile(reads, 95)},
            "latency_ms": {k: ({"p50": percentile(v, 50), "p95": percentile(v, 95)} if v else None)
                           for k, v in lat.items()},
            f"passage_recall@{cfg.k}": {k: (float(np.mean(v)) if v else None) for k, v in rec.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {"kind": "bench-scale", "config": cfg.model_dump(), "config_hash": config_hash(cfg.model_dump()),
              "questions": len(qds.questions), "machine": platform.platform(), "llm_calls": 0, "rows": rows}
    path = out / f"scale-{report['config_hash']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [f"# Scaling benchmark `{report['config_hash']}`", "",
          f"{len(qds.questions)} {cfg.questions.dataset} questions; encoder `{enc.name}`; no LLM calls.", "",
          f"| Passages | Edges | Bytes/edge | Graph build s | Pushes/query p50 (max) | Vectors read p50 | "
          f"Seed search p50 ms | Push p50 ms | Global p50 ms | GP@{cfg.k} dense | push | global |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["latency_ms"]["global"]
        pr = r[f"passage_recall@{cfg.k}"]
        pp = r["pushes_per_query"]
        g_ms = "" if g is None else f"{g['p50']:.1f}"
        g_rec = "" if pr["global"] is None else f"{pr['global'] * 100:.1f}"
        md.append(f"| {r['passages']:,} | {r['edges']:,} | {r['csr_bytes_per_edge']:.0f} | "
                  f"{r['build_s']['graph']:.0f} | {pp['p50']:.0f} ({pp['max']}) | "
                  f"{r['vectors_read_per_query']['p50']:.0f} | {r['latency_ms']['seed_search']['p50']:.1f} | "
                  f"{r['latency_ms']['push']['p50']:.1f} | {g_ms} | "
                  f"{pr['dense'] * 100:.1f} | {pr['push'] * 100:.1f} | {g_rec} |")
    (out / f"scale-{report['config_hash']}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report["paths"] = {"json": str(path)}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hsi-bench-scale", description="Walk cost and quality vs corpus size.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    rep = run(BenchConfig(**load_yaml(args.config)))
    print(json.dumps(rep["paths"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
