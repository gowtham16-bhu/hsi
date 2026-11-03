# Decision 0009: wider graph, learned choice at query time

**Date:** 2026-11-02 · **Status:** accepted for development

## Evidence (MuSiQue train development sample, 1,182 gold hops)
| Graph | Gold hops sharing an entity | Gold hops present as an edge |
|---|---|---|
| fan-out 8, hub cap 50 (current) | 87.3% | 38.7% |
| fan-out 32, hub cap 50 | 87.3% | 56.7% |
| fan-out 32, hub cap 200 | 87.3% | 61.8% |
| fan-out 64, hub cap 500 | 87.3% | 69.9% |

The index-time fan-out cap, ranked by a question-independent prior, deletes most correct hops before
any question arrives. Learned transitions choose the correct next hop more often than the rule
(52% vs 43% top-1, validation) but can only choose among existing edges.

## Decision
Development graphs use fan-out 32 and hub cap 200. Transitions are retrained on that graph. Per-query
cost stays bounded by local push (decision 0005). Edges must move to compact array storage before
large corpora (the uncapped development graph exhausted 15 GB as Python objects).
