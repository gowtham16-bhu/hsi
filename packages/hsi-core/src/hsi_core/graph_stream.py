"""Streaming graph builder: one source passage at a time, straight to CSR (decision 0013).

The reference `graph.build_title_graph` holds every candidate (src, dst) pair before the fan-out cap,
which grows with Σ_e k_e² over entities. Here each source u generates only its own candidates:

    forward anchor   u mentions e, v is an anchor of e          u → v   (w)
    reverse anchor   x mentions e, u is an anchor of e          u → x   (w · reverse_factor)
    co-mention       u and v both participate in e (non-hub)    u → v   (w_c)

keeps the best candidate per destination, caps to the top m, and appends to the CSR arrays.
Contributions are applied in the same order as the reference builder, so ties resolve identically and
the resulting arrays are equal (test `test_streaming_builder_equals_reference`).

Hub facets (decision 0014, `hub_mode="facets"`): a hub entity keeps its forward anchor edges, and its
participants are split by vector into facets of at most `facet_size`; co-mention and reverse anchor edges
are formed within a facet only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from hsi_core.csr import CompactGraph, cos_uv
from hsi_core.graph import EdgeScorer, GraphConfig, prepare
from hsi_core.model import EdgeType, Passage


def hub_facets(x: np.ndarray, max_size: int, iterations: int) -> list[np.ndarray]:
    """Split rows of x into groups of at most `max_size` by recursive median bisection along the principal
    direction (power iteration from the row farthest from the mean). Deterministic and balanced."""
    out: list[np.ndarray] = []
    stack = [np.arange(len(x))]
    while stack:
        g = stack.pop()
        if len(g) <= max_size:
            out.append(g)
            continue
        xc = x[g] - x[g].mean(axis=0)
        v = xc[int(np.argmax(np.einsum("ij,ij->i", xc, xc)))].astype(np.float64)
        for _ in range(iterations):
            nv = xc.T @ (xc @ v)
            norm = float(np.linalg.norm(nv))
            if norm == 0.0:
                break
            v = nv / norm
        order = np.argsort(xc @ v, kind="stable")
        half = len(g) // 2
        stack.append(g[order[half:]])
        stack.append(g[order[:half]])
    return out


def build_compact_graph(passages: Sequence[Passage], ids: Sequence[str], vecs: np.ndarray,
                        cfg: GraphConfig | None = None,
                        extra_anchors: Mapping[str, Iterable[str]] | None = None,
                        mentions: Mapping[str, Mapping[str, float]] | None = None,
                        edge_scorer: EdgeScorer | None = None,
                        progress: Callable[[int], None] | None = None) -> tuple[CompactGraph, dict[str, int]]:
    """Return the capped graph as a CompactGraph over positions `ids` (row i ↔ vecs[i]), plus stats."""
    cfg = cfg or GraphConfig()
    prep = prepare(passages, cfg, extra_anchors, mentions)
    anchors, ment, members, hubs, order = prep.anchors, prep.ment, prep.members, prep.hubs, prep.order

    # inverted indexes (sizes ~ number of mentions, not pairs)
    mentioners: dict[str, list[tuple[str, int]]] = defaultdict(list)   # entity -> [(x, position in x's found)]
    for x, found in ment.items():
        for pos, ent in enumerate(found):
            mentioners[ent].append((x, pos))
    anchored_by: dict[str, list[str]] = defaultdict(list)              # passage -> entities it anchors
    for ent, pids in anchors.items():
        for pid in dict.fromkeys(pids):
            anchored_by[pid].append(ent)
    ent_rank = {e: i for i, e in enumerate(members)}                   # reference co-mention loop order
    member_of: dict[str, list[str]] = defaultdict(list)
    for ent, mem in members.items():
        for pid in mem:
            member_of[pid].append(ent)
    sorted_members: dict[str, list[str]] = {}
    pos_of = {pid: i for i, pid in enumerate(ids)}
    facets = cfg.hub_mode == "facets"
    facet_of: dict[tuple[str, str], int] = {}          # (hub entity, passage) -> facet index
    facet_members: dict[tuple[str, int], list[str]] = {}
    if facets:
        vecs32 = np.asarray(vecs, dtype=np.float32)
        for ent in sorted(hubs):
            pids = sorted(members[ent])
            rows = vecs32[[pos_of[p] for p in pids]]
            for j, grp in enumerate(hub_facets(rows, cfg.facet_size, cfg.facet_iterations)):
                facet_members[(ent, j)] = sorted(pids[g] for g in grp.tolist())
                for g in grp.tolist():
                    facet_of[(ent, pids[g])] = j

    def anchor_w(ent: str, n_targets: int) -> float:
        base = cfg.weight_multi_token if prep.ntok(ent) > 1 else cfg.weight_single_token
        return base / math.sqrt(n_targets)

    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    out_dst: list[int] = []
    out_w: list[float] = []
    out_sal: list[float] = []
    out_co: list[float] = []
    out_bridge: list[str] = []
    raw = 0

    for i, u in enumerate(ids):
        best: dict[str, tuple[float, float, str, float, EdgeType]] = {}

        def add(dst: str, w: float, bridge: str, sal: float, et: EdgeType, best=best) -> None:
            prior = w * (0.5 + 0.5 * sal)
            cur = best.get(dst)
            if cur is None or prior > cur[0]:
                best[dst] = (prior, w, bridge, sal, et)

        # phase 1: anchor edges, in the reference order (mentioner x, entity position, target index, dir)
        phase1: list[tuple[tuple[int, int, int, int], str, float, str, float]] = []
        for pos, (ent, sal) in enumerate(ment.get(u, {}).items()):
            al = anchors.get(ent, ())
            targets = [t for t in al if t != u]
            if not targets or (ent in hubs and not facets):
                continue
            w = anchor_w(ent, len(targets))
            for t, dst in enumerate(targets):
                phase1.append(((order[u], pos, t, 0), dst, w, ent, sal))
        if cfg.bidirectional:
            for ent in anchored_by.get(u, ()):
                if ent in hubs and not facets:
                    continue
                al = anchors[ent]
                mine = facet_of.get((ent, u)) if ent in hubs else None
                for x, pos in mentioners.get(ent, ()):
                    if mine is not None and facet_of[(ent, x)] != mine:
                        continue
                    targets = [t for t in al if t != x]
                    if not targets:
                        continue
                    w = anchor_w(ent, len(targets)) * cfg.reverse_factor
                    for t, dst in enumerate(targets):
                        if dst == u:
                            phase1.append(((order[x], pos, t, 1), x, w, ent, 1.0))
        raw += len(phase1)
        phase1.sort(key=lambda c: c[0])
        for _, dst, w, ent, sal in phase1:
            add(dst, w, ent, sal, EdgeType.MENTION)

        # phase 2: co-mention edges, entities in the reference order, partners in sorted order
        if cfg.co_mention:
            for ent in sorted(member_of.get(u, ()), key=ent_rank.__getitem__):
                mem = members[ent]
                if ent in hubs:
                    if not facets:
                        continue
                    pids = facet_members[(ent, facet_of[(ent, u)])]
                else:
                    pids = sorted_members.get(ent)
                    if pids is None:
                        pids = sorted_members[ent] = sorted(mem)
                k = len(pids)
                if k < 2:
                    continue
                base = cfg.co_mention_multi_token if prep.ntok(ent) > 1 else cfg.co_mention_single_token
                w = base / math.sqrt(k - 1)
                sal = mem[u]
                for b in pids:
                    if b != u:
                        add(b, w, ent, sal, EdgeType.CO_MENTION)
                raw += k - 1

        # cap, then append to CSR
        if edge_scorer is not None and len(best) > cfg.fan_out:
            items = sorted(best.items())
            scores = edge_scorer(u, [(d, w, b, s, t) for d, (_, w, b, s, t) in items], prep.participants)
            ranked = [items[j] for j in np.lexsort((np.arange(len(items)), -scores))[: cfg.fan_out]]
        else:
            ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[: cfg.fan_out]
        n_row = 0
        for dst, (_, w, bridge, sal, et) in ranked:
            j = pos_of.get(dst)
            if j is None:
                continue
            out_dst.append(j)
            out_w.append(min(1.0, w))
            out_sal.append(sal)
            out_co.append(1.0 if et is EdgeType.CO_MENTION else 0.0)
            out_bridge.append(bridge)
            n_row += 1
        indptr[i + 1] = indptr[i] + n_row
        if progress is not None:
            progress(i + 1)

    bridges = sorted(set(out_bridge))
    bpos = {b: j for j, b in enumerate(bridges)}
    dst = np.asarray(out_dst, dtype=np.int32)
    src = np.repeat(np.arange(len(ids), dtype=np.int64), np.diff(indptr))
    df = np.asarray([prep.participants.get(b, 1) for b in bridges], dtype=np.float64)
    g = CompactGraph(ids=list(ids), indptr=indptr, dst=dst,
                     weight=np.asarray(out_w, dtype=np.float32), salience=np.asarray(out_sal, dtype=np.float32),
                     co_mention=np.asarray(out_co, dtype=np.float32),
                     bridge=np.fromiter((bpos[b] for b in out_bridge), dtype=np.int32, count=len(out_bridge)),
                     cos_uv=cos_uv(vecs, src, dst), bridges=bridges, bridge_log_df=np.log1p(df))
    stats = {"passages": len(ids), "entities": len(prep.participants), "entities_with_anchor": len(anchors),
             "raw_edges": raw, "capped_edges": g.edge_count, "hubs": len(hubs),
             "hub_facets": len(facet_members)}
    return g, stats
