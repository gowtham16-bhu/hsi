# Decision 0015: let the question choose among more stored candidates

**Date:** 2026-11-03 · **Status:** pre-registered (diagnostic experiment) · **Affects:** C4 fan-out, C7

## Context
Decision 0014's diagnosis: at 140K passages, 112 of 469 gold passages were missed although a gold seed
existed, because the seed's link to them was pruned at build time. The fan-out cap keeps m = 32 edges
per passage by a question-independent prior, while the candidate set grows with N. The learned walk
already scores edges per question, but it can only choose among the edges that were kept.

## Experiment
Keep a larger stored candidate pool per passage, m ∈ {64, 128}; the learned softmax picks among them per
question. Same hop model, facets 50, same 200 questions and corpora
(`configs/bench/scale-fan64.yaml`, `configs/bench/scale-fan128.yaml`, sizes 10K and 140K).

## Pre-registered reading
- **Q1:** at 140K, push GP@5 − dense ≥ +5.0 points for some m → static pruning is the main cause, and
  the next design makes the pool question-aware without storing m · N edges.
- **Q2:** at 10K, push GP@5 ≥ 63.8.
- **Q3:** push p50 latency at 140K ≤ 300 ms (NFR-1) and peak RSS < 8 GB.
- If Q1 fails for both m, pruning is not the main cause; seeds (encoder) are next.
