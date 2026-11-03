"""C4 (Gate 0 form): passage graph from anchors, mentions and co-mentions, with no LLM.

Entities are keyed by normalized surface form ("ent:<text>"), so a title and a mention of the
same name are the same entity.

Edges (specs/03 C4, decision 0001):
  anchor      A → B when A mentions e and B is an anchor of e (B's title is e); optional reverse
  co-mention  A ↔ B when both mention e, for 2 ≤ |participants(e)| ≤ D_max

Hub entities (more than D_max participants) give no edges here. Hub facets replace this rule in
Gate 1. `linked(a, b)` answers "do a and b share any entity at all", before any cap, which is what
the waterfall needs to separate "never linked" from "linked but pruned".
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from hsi_core.model import Edge, EdgeType, Passage
from hsi_core.text import normalize, tokens
from hsi_core.titles import TokenAhoCorasick, longest_non_overlapping


@dataclass(frozen=True)
class GraphConfig:
    fan_out: int = 8                     # m: kept out-edges per passage
    hub_degree_cap: int = 50             # D_max
    min_title_chars: int = 3
    bidirectional: bool = True
    weight_multi_token: float = 0.9      # anchor edge, multi-token entity
    weight_single_token: float = 0.6     # anchor edge, single-token entity
    reverse_factor: float = 0.85
    co_mention: bool = True
    co_mention_multi_token: float = 0.7  # w_c for multi-token entities
    co_mention_single_token: float = 0.45
    hub_mode: str = "drop"               # "drop": hubs give no edges; "facets": decision 0014
    facet_size: int | None = None        # facets mode: max passages per facet
    facet_iterations: int | None = None  # facets mode: power-iteration steps per split

    def __post_init__(self) -> None:
        if self.fan_out < 1 or self.hub_degree_cap < 2:
            raise ValueError("fan_out must be >= 1 and hub_degree_cap >= 2")
        if self.hub_mode not in ("drop", "facets"):
            raise ValueError("hub_mode must be 'drop' or 'facets'")
        if self.hub_mode == "facets" and (self.facet_size is None or not 2 <= self.facet_size <= self.hub_degree_cap
                                          or self.facet_iterations is None or self.facet_iterations < 1):
            raise ValueError("facets need 2 <= facet_size <= hub_degree_cap and facet_iterations >= 1")


def entity_key(surface: str) -> str:
    return "ent:" + normalize(surface)


entity_key_for_title = entity_key


@dataclass
class PassageGraph:
    passages: dict[str, Passage]
    anchors: dict[str, list[str]]                   # entity -> anchor passage ids
    out_edges: dict[str, list[Edge]]                # capped, best first
    entities_of: dict[str, frozenset[str]] = field(default_factory=dict)
    mention_salience: dict[str, dict[str, float]] = field(default_factory=dict)   # passage -> {entity: salience}
    candidates: dict[str, dict[str, tuple]] | None = None   # all edges before the cap (when requested)
    hubs: set[str] = field(default_factory=set)
    participants: dict[str, int] = field(default_factory=dict)
    raw_edge_count: int = 0

    def neighbors(self, pid: str) -> list[Edge]:
        return self.out_edges.get(pid, [])

    def linked(self, a: str, b: str) -> bool:
        return bool(self.entities_of.get(a, frozenset()) & self.entities_of.get(b, frozenset()))

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.out_edges.values())

    def stats(self) -> dict[str, int]:
        by_type: dict[str, int] = defaultdict(int)
        for es in self.out_edges.values():
            for e in es:
                by_type[e.type.value] += 1
        return {
            "passages": len(self.passages),
            "entities": len(self.participants),
            "entities_with_anchor": len(self.anchors),
            "raw_edges": self.raw_edge_count,
            "capped_edges": self.edge_count,
            "hubs": len(self.hubs),
        } | {f"capped_{k}": v for k, v in sorted(by_type.items())}


def _prior(edge: Edge) -> float:
    return edge.weight * (0.5 + 0.5 * edge.salience)


# a candidate edge before capping: (dst, weight, bridge, salience, type)
Candidate = tuple[str, float, str, float, EdgeType]
EdgeScorer = Callable[[str, list[Candidate], Mapping[str, int]], np.ndarray]


@dataclass
class Prepared:
    """Everything both graph builders share, computed by one code path (decision 0013)."""
    by_id: dict[str, Passage]
    order: dict[str, int]                         # passage id -> position in the input sequence
    own: dict[str, str]                           # passage id -> its own title entity
    anchors: dict[str, list[str]]                 # entity -> anchor passage ids
    n_tokens: dict[str, int]
    ment: dict[str, dict[str, float]]             # passage -> {mentioned entity: salience}, in found order
    members: dict[str, dict[str, float]]          # entity -> {participant passage: salience}
    participants: dict[str, int]
    hubs: set[str]

    def ntok(self, ent: str) -> int:
        return self.n_tokens.get(ent) or len(tokens(ent.split(":", 1)[1]))


def prepare(passages: Sequence[Passage], cfg: GraphConfig,
            extra_anchors: Mapping[str, Iterable[str]] | None = None,
            mentions: Mapping[str, Mapping[str, float]] | None = None) -> Prepared:
    by_id = {p.id: p for p in passages}
    own: dict[str, str] = {}
    anchors: dict[str, list[str]] = defaultdict(list)
    for p in passages:
        if p.title and len(normalize(p.title)) >= cfg.min_title_chars:
            e = entity_key(p.title)
            anchors[e].append(p.id)
            own[p.id] = e
    for ent, pids in (extra_anchors or {}).items():
        anchors[ent].extend(pid for pid in pids if pid in by_id)

    matcher = TokenAhoCorasick()
    n_tokens: dict[str, int] = {}
    for ent in anchors:
        toks = tokens(ent.split(":", 1)[1])
        if toks:
            matcher.add(toks, ent)
            n_tokens[ent] = len(toks)
    matcher.build()

    ment: dict[str, dict[str, float]] = {}
    for p in passages:
        toks = tokens(p.text)
        found: dict[str, float] = {}
        for m in longest_non_overlapping(matcher.find(toks)):
            if m.key != own.get(p.id) and m.key not in found:
                found[m.key] = 1.0 - m.start / max(1, len(toks))
        for ent, sal in (mentions or {}).get(p.id, {}).items():
            if ent != own.get(p.id):
                found[ent] = max(found.get(ent, 0.0), sal)
        ment[p.id] = found

    members: dict[str, dict[str, float]] = defaultdict(dict)
    for pid, found in ment.items():
        for ent, sal in found.items():
            members[ent][pid] = sal
    for ent, pids in anchors.items():
        for pid in pids:
            members[ent][pid] = 1.0
    participants = {e: len(m) for e, m in members.items()}
    hubs = {e for e, c in participants.items() if c > cfg.hub_degree_cap}
    return Prepared(by_id=by_id, order={p.id: i for i, p in enumerate(passages)}, own=own, anchors=dict(anchors),
                    n_tokens=n_tokens, ment=ment, members=dict(members), participants=participants, hubs=hubs)


def build_title_graph(passages: Sequence[Passage], cfg: GraphConfig | None = None,
                      extra_anchors: Mapping[str, Iterable[str]] | None = None,
                      mentions: Mapping[str, Mapping[str, float]] | None = None,
                      edge_scorer: EdgeScorer | None = None, keep_candidates: bool = False) -> PassageGraph:
    """Build the graph. `mentions` maps passage id -> {entity key: salience} from a mention
    extractor plugin; title mentions are always found here with an Aho–Corasick matcher.

    The fan-out cap keeps each passage's top edges by `edge_scorer` (decision 0010) when given,
    otherwise by the prior weight · (0.5 + 0.5 · salience).

    Reference implementation: it holds every candidate pair in memory. At scale use
    `graph_stream.build_compact_graph` (decision 0013), which yields the same edges."""
    cfg = cfg or GraphConfig()
    if cfg.hub_mode != "drop":
        raise ValueError("the reference builder supports hub_mode='drop' only; use graph_stream.build_compact_graph")
    prep = prepare(passages, cfg, extra_anchors, mentions)
    by_id, own, anchors, ment, members = prep.by_id, prep.own, prep.anchors, prep.ment, prep.members
    participants, hubs, ntok = prep.participants, prep.hubs, prep.ntok
    entities_of = {pid: frozenset(ment.get(pid, {}).keys() | ({own[pid]} if pid in own else set()))
                   for pid in by_id}

    # best (prior, weight, bridge, salience, type) per (src, dst)
    best: dict[str, dict[str, tuple[float, float, str, float, EdgeType]]] = defaultdict(dict)
    raw = 0

    def add(src: str, dst: str, w: float, bridge: str, sal: float, et: EdgeType) -> None:
        nonlocal raw
        raw += 1
        prior = w * (0.5 + 0.5 * sal)
        cur = best[src].get(dst)
        if cur is None or prior > cur[0]:
            best[src][dst] = (prior, w, bridge, sal, et)

    for src, found in ment.items():
        for ent, sal in found.items():
            targets = [t for t in anchors.get(ent, ()) if t != src]
            if not targets or ent in hubs:
                continue
            base = cfg.weight_multi_token if ntok(ent) > 1 else cfg.weight_single_token
            w = base / math.sqrt(len(targets))
            for dst in targets:
                add(src, dst, w, ent, sal, EdgeType.MENTION)
                if cfg.bidirectional:
                    add(dst, src, w * cfg.reverse_factor, ent, 1.0, EdgeType.MENTION)

    if cfg.co_mention:
        for ent, mem in members.items():
            k = len(mem)
            if k < 2 or ent in hubs:
                continue
            base = cfg.co_mention_multi_token if ntok(ent) > 1 else cfg.co_mention_single_token
            w = base / math.sqrt(k - 1)
            pids = sorted(mem)
            for i, a in enumerate(pids):
                for b in pids[i + 1:]:
                    add(a, b, w, ent, mem[a], EdgeType.CO_MENTION)
                    add(b, a, w, ent, mem[b], EdgeType.CO_MENTION)

    out_edges: dict[str, list[Edge]] = {}
    for src, dsts in best.items():
        if edge_scorer is not None and len(dsts) > cfg.fan_out:
            items = sorted(dsts.items())
            scores = edge_scorer(src, [(d, w, b, s, t) for d, (_, w, b, s, t) in items], participants)
            ranked = [items[i] for i in np.lexsort((np.arange(len(items)), -scores))[: cfg.fan_out]]
        else:
            ranked = sorted(dsts.items(), key=lambda kv: (-kv[1][0], kv[0]))[: cfg.fan_out]
        out_edges[src] = [Edge(src=src, dst=dst, type=et, weight=min(1.0, w), bridge=bridge, salience=sal)
                          for dst, (_, w, bridge, sal, et) in ranked]

    salience = {pid: dict(found) for pid, found in ment.items()}
    for ent, pids in anchors.items():
        for pid in pids:
            salience.setdefault(pid, {})[ent] = 1.0
    return PassageGraph(passages=by_id, anchors=dict(anchors), out_edges=out_edges,
                        entities_of=entities_of, mention_salience=salience, hubs=hubs,
                        participants=participants, raw_edge_count=raw,
                        candidates={s: dict(d) for s, d in best.items()} if keep_candidates else None)
