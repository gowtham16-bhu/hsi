# Decision 0016: train the hop model on a graph as large as the one it serves

**Date:** 2026-11-03 · **Status:** pre-registered · **Affects:** C7 (learned transitions)

## Context
Decisions 0014 and 0015: at 140K passages the walk's gain over dense shrinks to +0.2 (m = 32) or +2.6
(m = 128). When a gold seed has an edge to the missed passage, the walk still ranks it into the top 5 only
26% of the time (57% at 10K). The hop model θ was trained on a 23K-passage graph with m = 32, where
bridge frequencies, degrees and the number of competing neighbours are all smaller than at 140K.

## Decision
`hsi-train-transitions` accepts `distractors` and `corpus_size`, and builds its graph with the streaming
builder. With no distractors it reproduces the earlier model (max |Δθ| = 9e-9, float32 storage).

Train `transitions-musique-v5-140k-fan128` (`configs/train/transitions-musique-v5-140k-fan128.yaml`): the same
2,000 training questions (seed 1, excluding the development sample), their passages plus real distractors
up to 140,000 passages (MuSiQue train, 2Wiki dev), m = 128, facets 50. The benchmark questions are disjoint
from the training questions.

## Pre-registered acceptance (`configs/bench/scale-fan128-v5.yaml`)
- **T1:** at 140K, push GP@5 − dense ≥ +5.0 points.
- **T2:** at 10K, push GP@5 ≥ 63.8.
- **T3:** push p50 ≤ 300 ms at 140K.
If T1 fails, the remaining lever is the seeds (encoder), decision 0017.
