# Decision 0001: add co-mention edges to the Gate 0 graph

**Date:** 2026-11-02 · **Status:** accepted · **Affects:** C4, T0.8

## Context
The first Gate 0 dry run built the graph from title mentions only (passage A links to passage B when A
mentions B's title). Before measuring retrieval, the waterfall showed that most gold chains never
enter the graph (MuSiQue, 1,000 dev questions, seed 0):

| Hops | Questions | Stage ① link missed |
|---|---|---|
| 2 | 510 | 343 (67%) |
| 3 | 315 | 299 (95%) |
| 4 | 175 | 173 (99%) |

A diagnostic over 1,660 consecutive gold passage pairs:

| Link type | Pairs covered |
|---|---|
| One passage mentions the other's title | 24% |
| The bridge answer string appears in both passages | 83% |
| Both passages share a capitalized name span | 78% |

The bridge entity is usually *mentioned by* both passages, not the *title* of either.

## Decision
Add `co-mention` edges (spec C4): passages that mention the same entity are linked through it, with
weight decreasing in the entity's document frequency and the existing hub cap. Mentions come from a
mention-extractor plugin. Gate 0 uses a deterministic capitalized-span extractor (no model, no LLM).

## Consequences
- Running Gate 0 on the title-only graph would have measured the graph builder, not RQ1.
- More edges per passage make the fan-out cap matter more; stage ② (pruned) is reported to show it.
- The thesis claim is unchanged; this is a supporting-module fix (tier: supporting).
