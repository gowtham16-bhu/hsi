"""Retrieval experiment runner for Gate 0 / Gate 0′ (specs/04, decisions 0002 and 0003).

Everything about an experiment comes from its config: the systems and their settings, the shortcut
chains, and the pass rule. The runner contains no thresholds of its own (constitution P13).
No LLM is used anywhere in this runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hsi_core import registry
from hsi_core.bipartite import BipartiteConfig, BipartiteWalkRetriever
from hsi_core.bm25 import BM25Index
from hsi_core.chains import ChainConfig, enumerate_chains, select_by_prior
from hsi_core.config import PluginSpec, config_hash, load_yaml
from hsi_core.csr import CompactGraph
from hsi_core.encoding import CachedEncoder, passage_text
from hsi_core.expansion import ExpansionConfig, OnlineExpansion
from hsi_core.graph import GraphConfig, build_title_graph, entity_key
from hsi_core.graph_stream import build_compact_graph
from hsi_core.hops import HopConfig, HopRetriever
from hsi_core.index import ShortcutIndex, shortcuts_from_chains
from hsi_core.metrics import bootstrap_mean, full_chain_recall, paired_difference, percentile
from hsi_core.model import KeyMode
from hsi_core.ppr import PPRConfig, PPRRetriever
from hsi_core.push import LocalPushRetriever, PushConfig
from hsi_core.smoothing import SmoothingConfig, smooth_vectors
from hsi_core.transitions import LinearTransition, make_edge_scorer
from hsi_core.waterfall import classify, summarize
from hsi_harness.datasets import Dataset, load


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphSection(_Strict):
    fan_out: int
    hub_degree_cap: int
    bidirectional: bool
    co_mention: bool
    weight_multi_token: float
    weight_single_token: float
    reverse_factor: float
    co_mention_multi_token: float
    co_mention_single_token: float
    hub_mode: Literal["drop", "facets"] = "drop"      # decision 0014
    facet_size: int | None = None
    facet_iterations: int | None = None
    mention_extractor: PluginSpec | None = None
    edge_scorer: str | None = None          # learned index-time edge selection model (decision 0010)

    def to_core(self) -> GraphConfig:
        return GraphConfig(**self.model_dump(exclude={"mention_extractor", "edge_scorer"}))


class ChainSection(_Strict):
    max_passages: int = Field(ge=2, le=4)
    beam_level2: int
    beam_level3: int
    budget_ratio: float | None
    key_mode: Literal["text", "mean"]


class SmoothingSection(_Strict):
    gamma: float
    steps: int


class SystemSpec(_Strict):
    name: str
    kind: Literal["bm25", "dense", "expand", "ppr", "push", "bipartite", "hop"]
    index: str = "passages"            # "passages", "shortcuts", or a name from `passage_indexes`
    options: dict[str, Any] = {}


class Comparison(_Strict):
    candidate: str
    baseline: str
    slice: str                       # "all", "3", or hops joined by "+", e.g. "3+4"
    k: int
    metric: Literal["full_chain", "passage_recall"] = "full_chain"
    min_difference: float | None = None
    ci_low_above: float | None = None
    ci_low_at_least: float | None = None
    max_latency_ratio: float | None = None


class DecisionRule(_Strict):
    name: str
    all_of: list[Comparison]


class SampleRef(_Strict):
    seed: int
    size: int


class Gate0Config(_Strict):
    dataset: Literal["musique", "2wiki", "hotpotqa"]
    data_file: str
    sample_size: int | None
    seed: int
    exclude_sample: SampleRef | None = None
    encoder: PluginSpec
    vector_store: PluginSpec
    metadata_store: PluginSpec | None = None
    cache_dir: str
    out_dir: str
    graph: GraphSection
    chains: ChainSection
    passage_indexes: dict[str, SmoothingSection] = {}   # extra passage indexes with smoothed vectors
    systems: list[SystemSpec]
    top_k_hits: int
    passage_budgets: list[int]
    bootstrap: int
    decision: DecisionRule | None = None


def _code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True, cwd=Path(__file__).parent)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slice_hops(name: str, hops: list[int]) -> set[int]:
    return set(hops) if name == "all" else {int(x) for x in name.split("+")}


def graph_inputs(section: GraphSection, passages: list, vector_of=None) -> tuple[Any, Any]:
    """Mentions and the optional learned edge scorer, shared by both graph builders."""
    scorer = None
    if section.edge_scorer is not None:
        if vector_of is None:
            raise ValueError("a learned edge scorer needs passage vectors")
        scorer = make_edge_scorer(LinearTransition.load(section.edge_scorer), vector_of)
    mentions = None
    if section.mention_extractor is not None:
        extractor = registry.load("mention_extractor", section.mention_extractor.plugin,
                                  **section.mention_extractor.options)
        extractor.fit(p.text for p in passages)
        mentions = {p.id: {entity_key(s): sal for s, sal in extractor.extract(p.text)} for p in passages}
    return mentions, scorer


def build_graph(section: GraphSection, passages: list, vector_of=None, keep_candidates: bool = False) -> Any:
    """Reference graph construction (PassageGraph), shared by the experiment runner and the trainers."""
    mentions, scorer = graph_inputs(section, passages, vector_of)
    return build_title_graph(passages, section.to_core(), mentions=mentions, edge_scorer=scorer,
                             keep_candidates=keep_candidates)


def build_compact(section: GraphSection, passages: list, ids: list[str], vecs: Any, vector_of=None) -> Any:
    """Streaming construction straight to CSR (decision 0013); same edges as `build_graph`."""
    mentions, scorer = graph_inputs(section, passages, vector_of)
    return build_compact_graph(passages, ids, vecs, section.to_core(), mentions=mentions, edge_scorer=scorer)


def evaluate_rule(rule: DecisionRule, report: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for c in rule.all_of:
        d = report["comparisons"][f"{c.candidate}|{c.baseline}|{c.slice}|{c.k}|{c.metric}"]
        ok = True
        if c.min_difference is not None:
            ok &= d["mean"] >= c.min_difference
        if c.ci_low_above is not None:
            ok &= d["low"] > c.ci_low_above
        if c.ci_low_at_least is not None:
            ok &= d["low"] >= c.ci_low_at_least
        if c.max_latency_ratio is not None:
            lat = report["latency_ms"]
            ok &= lat[c.candidate]["p50"] <= c.max_latency_ratio * lat[c.baseline]["p50"]
        checks.append(c.model_dump() | {"difference": d, "pass": bool(ok)})
    return {"rule": rule.name, "status": "pass" if all(x["pass"] for x in checks) else "fail", "checks": checks}


def run(cfg: Gate0Config) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_hash = _file_hash(cfg.data_file)
    exclude = None
    if cfg.exclude_sample is not None:
        prior = load(cfg.dataset, cfg.data_file, cfg.exclude_sample.size, cfg.exclude_sample.seed)
        exclude = {q.id for q in prior.questions}
    ds: Dataset = load(cfg.dataset, cfg.data_file, cfg.sample_size, cfg.seed, exclude=exclude)
    passages = ds.passages
    by_id = {p.id: p for p in passages}

    # encoders and the passage index (needed first: learned edge selection uses passage vectors)
    raw_encoder = registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options)
    encoder = CachedEncoder(raw_encoder, cfg.cache_dir)

    def store_factory():
        return registry.load("vector_store", cfg.vector_store.plugin, **cfg.vector_store.options)

    t0 = time.perf_counter()
    dense = ShortcutIndex(encoder, store_factory).build(passages)
    t_dense = time.perf_counter() - t0

    # offline: graph and chains (no LLM) --------------------------------------------------
    t0 = time.perf_counter()
    gcfg = cfg.graph.to_core()
    graph = build_graph(cfg.graph, passages, vector_of=dense.raw_vecs.__getitem__)
    needs_shortcuts = any(s.index == "shortcuts" for s in cfg.systems)
    levels, stored = {}, []
    if needs_shortcuts:
        levels = enumerate_chains(graph, ChainConfig(max_passages=cfg.chains.max_passages,
                                                     beam_level2=cfg.chains.beam_level2,
                                                     beam_level3=cfg.chains.beam_level3))
        budget = None if cfg.chains.budget_ratio is None else int(cfg.chains.budget_ratio * len(passages))
        stored = select_by_prior(levels, budget)
    t_graph = time.perf_counter() - t0

    # indexes ------------------------------------------------------------------------------
    key_mode = KeyMode(cfg.chains.key_mode)
    shortcut_index = None
    t_hsi = 0.0
    if needs_shortcuts:
        t0 = time.perf_counter()
        shortcut_index = ShortcutIndex(encoder, store_factory).build(passages, shortcuts_from_chains(stored, key_mode))
        t_hsi = time.perf_counter() - t0
    indexes = {"passages": dense, "shortcuts": shortcut_index}
    for name, sm in cfg.passage_indexes.items():
        scfg = SmoothingConfig(gamma=sm.gamma, steps=sm.steps)
        indexes[name] = ShortcutIndex(encoder, store_factory).build(
            passages, vector_transform=lambda v, scfg=scfg: smooth_vectors(v, graph, scfg))
    unknown = {s.index for s in cfg.systems} - set(indexes)
    if unknown:
        raise ValueError(f"systems refer to unknown indexes: {sorted(unknown)}")

    def seed_index(s: SystemSpec) -> ShortcutIndex:
        if s.index == "shortcuts":
            raise ValueError(f"{s.name}: walks need a passage index for their seeds")
        return indexes[s.index]

    compact_by_index: dict[str, CompactGraph] = {}

    def compact(s: SystemSpec) -> CompactGraph:
        """One compact graph per seed index (cos(u, v) uses that index's vectors), decision 0012."""
        if s.index not in compact_by_index:
            idx = seed_index(s)
            ids = [e[0] for e in idx.entries]
            compact_by_index[s.index] = CompactGraph.from_graph(
                graph, ids, np.stack([idx.passage_vecs[pid] for pid in ids]).astype(np.float32))
        return compact_by_index[s.index]

    retrievers: dict[str, Any] = {}
    for s in cfg.systems:
        o = s.options
        if s.kind == "bm25":
            retrievers[s.name] = BM25Index(o["k1"], o["b"], frozenset(o.get("stopwords", []))).build(
                [p.id for p in passages], [passage_text(p) for p in passages])
        elif s.kind == "dense":
            retrievers[s.name] = indexes[s.index]
        elif s.kind == "expand":
            retrievers[s.name] = OnlineExpansion(seed_index(s), graph, ExpansionConfig(**o),
                                                 online_encoder=raw_encoder)
        elif s.kind in ("ppr", "push"):
            o = dict(o)
            model_path = o.pop("transition_model", None)
            model = LinearTransition.load(model_path) if model_path else None
            g = compact(s)
            retrievers[s.name] = PPRRetriever(seed_index(s), g, PPRConfig(**o), transition_model=model) \
                if s.kind == "ppr" else LocalPushRetriever(seed_index(s), g, PushConfig(**o), transition_model=model)
        elif s.kind == "bipartite":
            retrievers[s.name] = BipartiteWalkRetriever(seed_index(s), graph, BipartiteConfig(**o))
        elif s.kind == "hop":
            retrievers[s.name] = HopRetriever(indexes[s.index], by_id, HopConfig(**o), query_encoder=raw_encoder)

    # online: retrieve for every question ---------------------------------------------------
    kmax = max(cfg.passage_budgets)
    qvecs = encoder.encode([q.question for q in ds.questions], "query")
    names = [s.name for s in cfg.systems]
    results: dict[str, list[list[str]]] = {n: [] for n in names}
    latency: dict[str, list[float]] = {n: [] for n in names}
    for q, qv in zip(ds.questions, qvecs, strict=True):
        for s in cfg.systems:
            r = retrievers[s.name]
            if s.kind == "bm25":
                t0 = time.perf_counter()
                results[s.name].append([pid for pid, _ in r.search(q.question, kmax)])
                latency[s.name].append((time.perf_counter() - t0) * 1000)
                continue
            out = r.retrieve_vec(qv, kmax, cfg.top_k_hits, q.question) if s.kind in ("hop", "ppr", "push") \
                else r.retrieve_vec(qv, kmax, cfg.top_k_hits)
            results[s.name].append(out.passages)
            latency[s.name].append(out.latency_ms)

    # metrics ------------------------------------------------------------------------------
    hops = sorted({q.hops for q in ds.questions})
    slices = ["all", *[str(h) for h in hops]] + (["+".join(str(h) for h in hops if h >= 3)]
                                                 if sum(h >= 3 for h in hops) > 1 else [])
    values: dict[tuple[str, str, int], list[float]] = {}
    recall: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    passage_recall: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for n in names:
        for sl in slices:
            want = _slice_hops(sl, hops)
            idx = [i for i, q in enumerate(ds.questions) if q.hops in want]
            row = {}
            for k in cfg.passage_budgets:
                vals = [float(full_chain_recall(results[n][i], ds.questions[i].gold, k)) for i in idx]
                values[(n, sl, k)] = vals
                row[f"r@{k}"] = bootstrap_mean(vals, cfg.bootstrap, cfg.seed).as_dict()
            recall[n][sl] = row
            passage_recall[n][sl] = {f"r@{k}": bootstrap_mean(
                [len(set(ds.questions[i].gold) & set(results[n][i][:k])) / len(ds.questions[i].gold) for i in idx],
                cfg.bootstrap, cfg.seed).as_dict() for k in cfg.passage_budgets}

    def per_question(n: str, i: int, k: int, metric: str) -> float:
        gold = ds.questions[i].gold
        if metric == "full_chain":
            return float(full_chain_recall(results[n][i], gold, k))
        return len(set(gold) & set(results[n][i][:k])) / len(gold)

    def diff(a: str, b: str, sl: str, k: int, metric: str) -> dict[str, Any]:
        idx = [i for i, q in enumerate(ds.questions) if q.hops in _slice_hops(sl, hops)]
        va = [per_question(a, i, k, metric) for i in idx]
        vb = [per_question(b, i, k, metric) for i in idx]
        return paired_difference(va, vb, cfg.bootstrap, cfg.seed).as_dict()

    comparisons: dict[str, dict[str, Any]] = {}
    if cfg.decision is not None:
        for c in cfg.decision.all_of:
            comparisons[f"{c.candidate}|{c.baseline}|{c.slice}|{c.k}|{c.metric}"] = diff(
                c.candidate, c.baseline, c.slice, c.k, c.metric)

    waterfall: dict[str, dict[str, dict[str, float]]] = {}
    if needs_shortcuts:
        candidates = {c.passages for lvl in levels.values() for c in lvl}
        stored_set = {c.passages for c in stored}
        for s in cfg.systems:
            if s.index != "shortcuts":
                continue
            waterfall[s.name] = {}
            for h in hops:
                for k in cfg.passage_budgets:
                    stages = [classify(q.gold, q.ordered, graph, candidates, stored_set, results[s.name][i], k)
                              for i, q in enumerate(ds.questions) if q.hops == h]
                    waterfall[s.name][f"{h}hop@{k}"] = summarize(stages)

    report: dict[str, Any] = {
        "kind": "gate0",
        "config": cfg.model_dump(mode="json"),
        "config_hash": config_hash(cfg),
        "code_version": _code_version(),
        "data_hash": data_hash,
        "encoder": raw_encoder.name,
        "dataset": {"name": ds.name, "questions": len(ds.questions), "passages": len(passages),
                    "by_hop": {str(h): sum(q.hops == h for q in ds.questions) for h in hops},
                    "excluded_questions": len(exclude or ())},
        "graph": graph.stats() | {"config": asdict(gcfg)},
        "chains": {"candidates_by_level": {str(k): len(v) for k, v in levels.items()}, "stored": len(stored)},
        "index_entries": {"passages": len(dense.entries),
                          "shortcuts": len(shortcut_index.entries) if shortcut_index else 0},
        "build_seconds": {"graph_and_chains": t_graph, "passage_index": t_dense, "shortcut_index": t_hsi},
        "llm_calls": 0,
        "recall": dict(recall),
        "passage_recall": dict(passage_recall),
        "comparisons": comparisons,
        "latency_ms": {n: {"p50": percentile(latency[n], 50), "p95": percentile(latency[n], 95)} for n in names},
        "ann_calls_per_question": {n: retrievers[n].ann_calls / len(ds.questions)
                                   for n in names if hasattr(retrievers[n], "ann_calls")},
        "pushes_per_question": {n: retrievers[n].total_pushes / len(ds.questions)
                                for n in names if hasattr(retrievers[n], "total_pushes")},
        "waterfall": waterfall,
        "encoder_cache": {"hits": encoder.hits, "misses": encoder.misses},
    }
    report["decision"] = evaluate_rule(cfg.decision, report) if cfg.decision else None
    report["total_seconds"] = time.perf_counter() - t_start

    if cfg.metadata_store is not None:
        corpus_id = f"{ds.name}-n{cfg.sample_size}-s{cfg.seed}-{data_hash[:8]}"
        store = registry.load("metadata_store", cfg.metadata_store.plugin, **cfg.metadata_store.options)
        try:
            store.migrate()
            store.write_passages(corpus_id, passages)
            store.write_edges(corpus_id, (e for es in graph.out_edges.values() for e in es))
            if stored:
                store.write_shortcuts(corpus_id, shortcuts_from_chains(stored, key_mode))
            report["run_id"] = store.record_run(report | {"corpus_id": corpus_id})
            report["corpus_id"] = corpus_id
        finally:
            store.close()

    stem = f"{ds.name}-{report['config_hash']}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    report["paths"] = {"json": str(out_dir / f"{stem}.json"), "markdown": str(out_dir / f"{stem}.md")}
    return report


def _fmt(e: dict[str, Any]) -> str:
    return f"{100 * e['mean']:.1f} [{100 * e['low']:.1f}, {100 * e['high']:.1f}]"


def render_markdown(r: dict[str, Any]) -> str:
    budgets = r["config"]["passage_budgets"]
    slices = list(next(iter(r["recall"].values())).keys())
    lines = [
        f"# Retrieval report: {r['dataset']['name']}",
        "",
        f"- Questions: {r['dataset']['questions']} (by hop: {r['dataset']['by_hop']}; "
        f"excluded from an earlier sample: {r['dataset']['excluded_questions']}), passages: {r['dataset']['passages']}",
        f"- Encoder: `{r['encoder']}` · config `{r['config_hash']}` · code `{r['code_version']}` "
        f"· data `{r['data_hash'][:12]}`",
        f"- LLM calls: {r['llm_calls']}",
        f"- Graph: {r['graph']['raw_edges']} raw edges, {r['graph']['capped_edges']} capped, {r['graph']['hubs']} hubs",
        f"- Chains: candidates {r['chains']['candidates_by_level']}, stored {r['chains']['stored']}",
        f"- Index entries: {r['index_entries']}",
        "",
        "## Full-chain recall (%, 95% CI)",
        "",
        "| System | Slice | " + " | ".join(f"R@{k}" for k in budgets) + " |",
        "|---|---|" + "---|" * len(budgets),
    ]
    for s, rows in r["recall"].items():
        for sl in slices:
            lines.append(f"| {s} | {sl} | " + " | ".join(_fmt(rows[sl][f"r@{k}"]) for k in budgets) + " |")
    lines += ["", "## Gold-passage recall (% of gold passages retrieved, 95% CI)", "",
              "| System | Slice | " + " | ".join(f"R@{k}" for k in budgets) + " |",
              "|---|---|" + "---|" * len(budgets)]
    for s, rows in r.get("passage_recall", {}).items():
        for sl in slices:
            lines.append(f"| {s} | {sl} | " + " | ".join(_fmt(rows[sl][f"r@{k}"]) for k in budgets) + " |")
    lines += ["", "## Latency (ms, query encoding excluded) and lookups", "",
              "| System | p50 | p95 | ANN calls / question | pushes / question |", "|---|---|---|---|---|"]
    for s, v in r["latency_ms"].items():
        calls = r["ann_calls_per_question"].get(s)
        pushes = r.get("pushes_per_question", {}).get(s)
        lines.append(f"| {s} | {v['p50']:.2f} | {v['p95']:.2f} | {'' if calls is None else f'{calls:.1f}'} | "
                     f"{'' if pushes is None else f'{pushes:.0f}'} |")
    for name, wf in r["waterfall"].items():
        lines += ["", f"## Failure waterfall: {name} (share of questions)", "",
                  "| Slice | link missed | pruned | not stored | not retrieved | ok | n |",
                  "|---|---|---|---|---|---|---|"]
        for sl, w in wf.items():
            lines.append(f"| {sl} | {w['link_missed']:.2f} | {w['pruned']:.2f} | {w['not_stored']:.2f} | "
                         f"{w['not_retrieved']:.2f} | {w['ok']:.2f} | {w['n']} |")
    d = r.get("decision")
    if d:
        lines += ["", f"## Decision rule `{d['rule']}`: **{d['status']}**", "",
                  "| Candidate − baseline | Slice | k | Difference (points, 95% CI) | Condition | Pass |",
                  "|---|---|---|---|---|---|"]
        for c in d["checks"]:
            cond = ", ".join(f"{k}={c[k]}" for k in ("metric", "min_difference", "ci_low_above", "ci_low_at_least",
                                                     "max_latency_ratio") if c[k] is not None)
            lines.append(f"| {c['candidate']} − {c['baseline']} | {c['slice']} | {c['k']} | {_fmt(c['difference'])} "
                         f"| {cond} | {'yes' if c['pass'] else 'no'} |")
        lines += ["", "The binding decision is written by a person in `specs/decisions/`."]
    return "\n".join(lines) + "\n"


def load_config(path: str) -> Gate0Config:
    return Gate0Config.model_validate(load_yaml(path))


def env_dsn_available() -> bool:
    return bool(os.environ.get("HSI_PG_DSN"))
