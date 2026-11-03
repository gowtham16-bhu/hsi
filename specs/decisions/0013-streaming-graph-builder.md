# Decision 0013: streaming graph builder (per-source, straight to CSR)

**Date:** 2026-11-03 · **Status:** pre-registered · **Affects:** C4 build, NFR-1 (ingestion scale)

## Context
The scaling benchmark (decision 0012) ran out of memory (13.8 GB) building the graph for 140K passages.
`build_title_graph` keeps every candidate (src, dst) pair in a global Python dict before the fan-out
cap, then creates a Python `Edge` object per kept edge. Candidate pairs grow with Σ_e k_e² over
entities (k_e ≤ D_max participants), far faster than the N · m edges actually kept.

## Decision
`hsi_core.graph_stream.build_compact_graph` builds the same graph one **source passage at a time**:

1. Shared preparation (identical code path to `build_title_graph`, factored into `graph.prepare`):
   anchors, title matching, mentions, entity members, participants, hubs.
2. For each source u, generate only u's candidates: forward anchor edges from u's mentions, reverse
   anchor edges where u is an anchor, co-mention edges for u's non-hub entities. Keep the best
   candidate per destination, in the same contribution order as the reference builder, so ties
   resolve identically.
3. Apply the fan-out cap (prior order, or the learned `edge_scorer`) and append u's top-m edges directly
   to the CSR arrays. Peak extra memory is one source's candidates, not all pairs.

`build_title_graph` stays as the reference implementation (as global PPR is for push), for the small
corpora that also need `PassageGraph` (waterfall, smoothing, chain enumeration).

## Pre-registered acceptance
- **B1 (exactness):** on the test fixtures and on the MuSiQue train development corpus (14,223 passages,
  fan-out 32, hub cap 200), the streamed CSR equals `CompactGraph.from_graph(build_title_graph(...))`
  array for array, with and without a learned edge scorer.
- **B2 (memory):** peak RSS of the scaling benchmark at 140,000 passages stays below 8 GB (half the
  container), and the run completes.
- **B3 (the deferred decision 0012 benchmark):** median pushes per query at 140K within 1.5× of 10K.

## Outcome B1 (2026-11-03): **pass**
- Unit test `test_streaming_builder_equals_reference`: five variants (with and without learned mentions,
  no reverse edges, duplicate extra anchors, tiny hub cap, a tie-heavy edge scorer), all arrays equal.
- MuSiQue train development corpus (14,223 passages, `configs/dev/musique-train-learned-fan32.yaml`):
  403,167 edges, 2,797,714 raw candidates, **identical** CSR with no edge scorer and with
  `edges-musique-v1.json`. Streaming 10.0 s vs reference 18.5 s (18.8 s vs 32.1 s with the scorer).

## Outcome B2, B3 (2026-11-03): **pass**, with a new quality finding
Run `runs/bench/scale-14ea5d075d7f9769.json` (`configs/bench/scale.yaml`, 200 unseen MuSiQue train questions).

| Passages | Edges | Graph build s | Peak RSS | Pushes p50 (max) | Push p50 ms | Global p50 ms | GP@5 dense | push | global |
|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 272,339 | 7 | 1.2 GB | 498 (843) | 52.8 | 72.8 | 55.9 | 64.8 | 64.8 |
| 40,000 | 1,067,120 | 38 | 1.6 GB | 537 (862) | 60.6 | 271.2 | 48.5 | 55.4 | 55.2 |
| 140,000 | 3,724,902 | 128 | 3.2 GB | 515 (900) | 57.5 | 3,058.5 | 44.8 | 45.0 | 45.0 |

- **B2:** 140K completes at 3.2 GB peak (bound 8 GB; the reference builder was killed at 13.8 GB).
- **B3:** median pushes at 140K = 1.03× those at 10K (bound 1.5×). Push latency is flat (53–61 ms) while
  global PPR grows 42×. Decision 0012's deferred benchmark therefore also passes.
- **Finding (not a pre-registered criterion):** the walk's recall gain over dense shrinks as distractors grow:
  +8.9 (10K), +6.9 (40K), +0.2 (140K). Push and global agree, so this is the graph or the model, not push.
  Diagnosis follows in decision 0014.
