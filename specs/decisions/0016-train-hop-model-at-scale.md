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

## Outcome (2026-11-03): **T1 failed** (T2, T3 pass). Retraining at scale changes nothing measurable.
Model `runs/models/transitions-musique-v5-140k-fan128.json` (140,000-passage graph, 974 training groups;
validation top-1 36.6% vs the rule's 15.5%). Benchmark `runs/bench/scale-fan128-v5.log`:

| Passages | Dense GP@5 | Push, v2 model (0015) | Push, v5 model (trained at 140K) | Pushes p50 | Push p50 ms |
|---|---|---|---|---|---|
| 10K | 55.9 | 65.4 (+9.5) | 64.9 (+9.0) | 768 | 132 |
| 140K | 44.8 | 47.5 (+2.6) | 47.5 (+2.6) | 782 | 188 |

The retrained model weights rare bridges and anchor edges more strongly (θ log_df −0.41 → −1.11,
co_mention −0.36 → −0.67) but ranks the same passages into the top 5. The hop model is not the cause.

**Conclusion of the scale track (decisions 0013–0016).** Cost scales as designed (flat pushes, flat
latency, 3.7 GB at 140K). Quality gain over dense falls from about +9 at 10K to +2.6 at 140K, and neither
graph fixes (facets, larger pools) nor retraining recover it. The diagnosed remaining cause is the seeds:
misses with no gold passage among the seeds nearly double (57 → 104 of 469). Testing a stronger encoder at
140K needs hours of CPU or a GPU; it is deferred. Per the owner's plan, algorithm work stops here and the
product track starts (A5). This negative scale result is reported as a finding, not hidden.
