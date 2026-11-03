"""Failure waterfall (specs/04): the first stage at which each question's gold chain is lost.

Gate 0 stages (no reader yet):
  link_missed    some hop of the gold chain has no edge in the uncapped graph, in any ordering
  pruned         linked, but no ordering of the gold chain is among the candidate chains
  not_stored     a candidate, but not stored as a shortcut (budget)
  not_retrieved  stored, but the gold passages are not all in R_k(q)
  ok             full chain retrieved
Gate 1 adds lost_in_assembly and reader_wrong.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import pairwise, permutations

from hsi_core.graph import PassageGraph


class Stage(StrEnum):
    LINK_MISSED = "link_missed"
    PRUNED = "pruned"
    NOT_STORED = "not_stored"
    NOT_RETRIEVED = "not_retrieved"
    OK = "ok"


ORDER = [Stage.LINK_MISSED, Stage.PRUNED, Stage.NOT_STORED, Stage.NOT_RETRIEVED, Stage.OK]


def _orderings(gold: Sequence[str], ordered: bool) -> Iterable[tuple[str, ...]]:
    if ordered:
        yield tuple(gold)
        yield tuple(reversed(gold))
    else:
        yield from permutations(gold)


def classify(gold: Sequence[str], ordered: bool, graph: PassageGraph,
             candidates: set[tuple[str, ...]], stored: set[tuple[str, ...]],
             retrieved: Sequence[str], k: int) -> Stage:
    if len(gold) < 2:
        raise ValueError("waterfall needs a gold chain of at least 2 passages")
    if set(gold) <= set(retrieved[:k]):
        return Stage.OK
    orders = list(_orderings(gold, ordered))
    if not any(all(graph.linked(a, b) for a, b in pairwise(o)) for o in orders):
        return Stage.LINK_MISSED
    if not any(o in candidates for o in orders):
        return Stage.PRUNED
    if not any(o in stored for o in orders):
        return Stage.NOT_STORED
    return Stage.NOT_RETRIEVED


def summarize(stages: Iterable[Stage]) -> dict[str, float]:
    c = Counter(stages)
    n = sum(c.values())
    return {s.value: (c.get(s, 0) / n if n else 0.0) for s in ORDER} | {"n": n}
