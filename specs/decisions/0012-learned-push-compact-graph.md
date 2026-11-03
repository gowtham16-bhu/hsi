# Decision 0012: learned transitions in local push, over a compact graph

**Date:** 2026-11-03 · **Status:** pre-registered · **Affects:** C4 storage, C7 walks, NFR-1, NFR-7

## Context
The final test (decision 0011) used **global** PPR: every query touches every edge 50 times, so the
cost grows with the corpus (117 ms at 11.6K passages). Decision 0005 introduced local forward push,
whose work per query is bounded by 1/(α·ε) independent of N, but push only knows the rule-based β
transitions, not the learned transitions that produced the result. Separately, the graph is held as
Python `Edge` objects; an uncapped diagnostic graph ran out of memory at 15 GB.

## Decision
1. **Compact graph (CSR).** `hsi_core.csr.CompactGraph` stores the capped graph as arrays: `indptr`,
   `dst` (int32 passage positions) and per-edge static features (weight, salience, co-mention flag,
   bridge id, cos(u, v)), plus bridge document frequencies and bridge surface tokens. It is built from
   a `PassageGraph`, saved to and loaded from one `.npz` file, and is what walks read.
2. **One feature function.** `transitions.stack_features` assembles φ for any slice of edges. Global
   PPR (`EdgeTable`) and push both call it, so training and both walks cannot drift apart.
3. **Learned push.** `LocalPushRetriever` accepts a `transition_model`. For a pushed passage u it
   computes φ only for u's out-edges, with cos(q, v) from the vectors of those neighbours only.
   The question's bridge mask is computed once per query from the question text.
4. **Runner.** System kind `push` with options `seeds, restart, epsilon, max_pushes, beta` and optional
   `transition_model`, like `ppr`.

## Pre-registered acceptance (development split only; the test sets stay closed)
Run `configs/dev/musique-train-push.yaml`: MuSiQue **train** sample (seed 0, 1,000 questions,
bge-small, fan-out 32, transitions `transitions-musique-v2-fan32.json`), the same setting as the
`learned-s3-total` development system.

- **A1 — quality:** push (ε = 1e-4) gold-passage recall@5 minus global learned PPR: 95% CI lower
  bound ≥ −1.0 point.
- **A2 — speed:** push median latency below global learned PPR's median latency.
- **A3 — exactness:** at ε = 1e-7 on the test fixtures, push top-k equals global PPR top-k with the
  same learned model (unit test).

## Scaling benchmark (decision 0005 acceptance, now with learned transitions)
`hsi-bench-scale` builds nested real corpora (MuSiQue train passages, then adding 2Wiki dev passages)
at the sizes listed in `configs/bench/scale.yaml`, and reports per size: graph build time, edges, CSR
bytes per edge, pushes per query, walk latency (seed search excluded and reported separately), and
global-PPR walk latency for comparison. Pass: median pushes per query at the largest size within
1.5× of the smallest size, while global PPR latency grows with N.

Exact (flat) seed search still scans every vector; an approximate index is a vector-store plugin
concern (decision 0005 item 2) and is measured separately.

## Outcome (2026-11-03): **pass** (A1, A2, A3)
Run `runs/dev/musique-a04aeaa235e94bed.json` (MuSiQue train, 1,000 questions, bge-small, fan-out 32).

| System | GP@5 | GP@10 | Full chain @10 | p50 ms | p95 ms | Pushes / question |
|---|---|---|---|---|---|---|
| dense | 56.0 | 62.8 | 33.0 | 1.8 | 10.5 | — |
| global learned PPR | 62.1 | 70.8 | 51.1 | 83.2 | 261.1 | — |
| push learned, ε = 1e-3 | 62.1 | 70.6 | 50.9 | **12.5** | 38.5 | 120 |
| push learned, ε = 1e-4 | 62.1 | 70.7 | 51.2 | 46.8 | 79.7 | 489 |
| push learned, ε = 1e-5 | 62.1 | 70.7 | 51.1 | 169.4 | 316.2 | 2,164 |

- **A1:** push ε = 1e-4 − global, GP@5: −0.1 [−0.1, 0.0] points (bound −1.0): pass.
- **A2:** 46.8 ms < 83.2 ms: pass. ε = 1e-3 keeps the same recall at 12.5 ms (6.7× faster than global).
- **A3:** unit test `test_learned_push_matches_learned_global_ppr`: pass.
- **No regression from the refactor:** global learned PPR on the compact graph reproduces the earlier
  run `musique-820636249af67b59` exactly (GP@5 0.62117, full chain @10 0.511); its p50 latency fell from
  155 ms to 83 ms because the edge arrays are now built once and shared.

## Scaling benchmark (2026-11-03): **incomplete**: the 140K size ran out of memory while building the graph
Config `configs/bench/scale.yaml`, 200 unseen MuSiQue train questions, bge-small, log `runs/bench/scale.log`.

| Passages | Edges | Bytes/edge | Encode s | Graph build s | Pushes p50 (max) | Push p50 ms | Global p50 ms | GP@5 dense | push | global |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 272,339 | 24.5 | 163 | 12 | 498 (843) | 44.4 | 68.1 | 55.9 | 64.8 | 64.8 |
| 40,000 | 1,067,120 | 24.5 | 561 | 52 | 537 (862) | 54.4 | 215.8 | 48.5 | 55.4 | 55.2 |
| 140,000 | — | — | — | killed (13.8 GB) | — | — | — | — | — | — |

- From 10K to 40K, pushes per query rose 8% (498 → 537) while global PPR latency tripled (68 → 216 ms).
  This matches the pass condition so far, but the pre-registered largest size did not run, so the
  benchmark is **not passed**.
- **Finding:** the graph *build* does not scale. `build_title_graph` keeps every candidate pair in Python
  dicts before the fan-out cap, plus Python `Edge` objects. Next: build the graph as a streaming array
  pipeline (per-entity candidate generation → per-source top-m selection → CSR), never materializing
  all candidates as Python objects.
- Encoding ran at about 70 passages/s on this 4-CPU container (bge-small). It dominates ingestion and must be
  parallelized (GPU or many workers); that is independent of the algorithm.
