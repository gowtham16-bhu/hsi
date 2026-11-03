# Decision 0010: learned index-time edge selection

**Date:** 2026-11-02 · **Status:** accepted for development

## Evidence
Fan-out 32 + learned transitions (run `820636249af67b59`, MuSiQue train development): gold-passage
R@5 62.8 (dense 56.0), 2-hop full-chain R@10 63.4 (dense 43.2), but 3–4 hop full-chain stays at ~18%.
Only 55% of gold hops survive the index-time cap, so all hops of a 3-hop chain survive in roughly a
third of cases. Wider caps keep more (fan-out 64 / hub cap 500: 70%) at twice the edges.

## Decision
Replace the hand-made prior that ranks edges for the cap with a learned, question-independent
edge scorer, trained like the transitions (listwise softmax over each gold passage's candidate
edges in a wide candidate graph, positive = gold next passage). Features: edge weight, log df of the
bridge, co-mention flag, mention salience, cos(u, v). Keep the top `fan_out` edges per passage by
this score. Same index size, more gold hops kept. Training data: MuSiQue train, disjoint from the
development sample. Metric: share of gold hops kept at fan-out 32, then retrieval on development.

## Outcome (2026-11-02): **failed, not adopted**
Validation gold hops kept at fan-out 32: prior 64.1%, learned 57.7% (`runs/models/edges-musique-v1.json`).
The listwise objective rewards ranking the gold edge first among ~211 candidates (top-1 15%), while
only membership in the top 32 matters. Development retrieval did not change (`577226e1cb8ebfe4`).
A top-k objective could be retried later; wider caps (decision 0009) come first.
