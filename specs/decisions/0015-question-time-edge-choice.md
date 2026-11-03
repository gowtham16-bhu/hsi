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

## Outcome (2026-11-03): **Q1 failed**; pruning is a partial cause, not the main one
Runs `runs/bench/scale-fan64.log`, `runs/bench/scale-fan128.log`.

| m | Passages | Edges | Push GP@5 | Gain over dense (44.8 / 55.9) | Pushes p50 | Push p50 / p95 ms | Peak RSS |
|---|---|---|---|---|---|---|---|
| 32 (0014) | 140K | 4.4M | 45.1 | +0.2 | 555 | 65 / — | 3.1 GB |
| 64 | 140K | 8.7M | 46.3 | +1.5 | 682 | 97 / 165 | 3.2 GB |
| 128 | 140K | 16.5M | 47.5 | +2.6 | 819 | 165 / 252 | 3.7 GB |
| 64 | 10K | 0.58M | 65.8 | +9.9 | 653 | 74 / 129 | 1.2 GB |
| 128 | 10K | 1.02M | 65.4 | +9.5 | 835 | 140 / 268 | 1.2 GB |

- Q1 (gain ≥ +5 at 140K): fail for both m. Q2 (10K ≥ 63.8): pass. Q3 (p50 ≤ 300 ms, RSS < 8 GB): pass.
- Restoring pruned links recovers part of the gain (+0.2 → +2.6), with diminishing returns and 2.5× latency.
- Per the pre-registered reading, pruning is not the main cause. Next (decision 0016): the hop model was trained
  on 14K-passage graphs with m = 32; at 140K its candidates, degrees and bridge frequencies differ. Retrain it
  at scale before changing the encoder, because retraining costs minutes and the encoder costs hours of CPU.
