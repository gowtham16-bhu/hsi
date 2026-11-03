"""C5 (enumeration): candidate chains under fan-out caps and beams (specs/03 C5).

level 1: every capped edge A → B
level 2: each kept level-1 chain extended by its top b2 next hops
level 3: each kept level-2 chain extended by its top b3 next hops
log P(c) = Σ log weight(edge)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hsi_core.graph import PassageGraph, _prior
from hsi_core.model import Chain


@dataclass(frozen=True)
class ChainConfig:
    max_passages: int = 3            # 2..4 (Gate 0 uses 3: 2- and 3-passage chains)
    beam_level2: int = 3             # b2
    beam_level3: int = 2             # b3

    def __post_init__(self) -> None:
        if not 2 <= self.max_passages <= 4:
            raise ValueError("max_passages must be 2, 3 or 4")


def enumerate_chains(graph: PassageGraph, cfg: ChainConfig | None = None) -> dict[int, list[Chain]]:
    """Returns {level: chains}. Deterministic: same graph and config give the same output."""
    cfg = cfg or ChainConfig()
    levels: dict[int, list[Chain]] = {}
    lvl1: list[Chain] = []
    for src in sorted(graph.out_edges):
        for e in graph.out_edges[src]:
            lvl1.append(Chain(passages=(src, e.dst), bridges=(e.bridge,), log_prob=math.log(e.weight)))
    levels[1] = lvl1

    beams = {2: cfg.beam_level2, 3: cfg.beam_level3}
    prev = lvl1
    for level in range(2, cfg.max_passages):
        seen: dict[tuple[str, ...], Chain] = {}
        for c in prev:
            last = c.passages[-1]
            nxt = [e for e in graph.neighbors(last) if e.dst not in c.passages]
            nxt.sort(key=lambda e: (-_prior(e), e.dst))
            for e in nxt[: beams[level]]:
                cand = Chain(passages=(*c.passages, e.dst), bridges=(*c.bridges, e.bridge),
                             log_prob=c.log_prob + math.log(e.weight))
                cur = seen.get(cand.passages)
                if cur is None or cand.log_prob > cur.log_prob:
                    seen[cand.passages] = cand
        levels[level] = [seen[k] for k in sorted(seen)]
        prev = levels[level]
    return levels


def select_by_prior(levels: dict[int, list[Chain]], budget: int | None) -> list[Chain]:
    """Gate 0 selection: keep all chains, or the top `budget` by log P(c), ties by chain id.

    Budgeted facility-location selection (C6) replaces this in Gate 1.
    """
    every = [c for lvl in sorted(levels) for c in levels[lvl]]
    if budget is None or budget >= len(every):
        return every
    if budget < 0:
        raise ValueError("budget must be >= 0")
    return sorted(every, key=lambda c: (-c.log_prob, c.id))[:budget]
