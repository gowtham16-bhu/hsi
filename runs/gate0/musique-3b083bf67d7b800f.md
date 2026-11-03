# Gate 0 report: musique

- Questions: 1000 (by hop: {'2': 510, '3': 315, '4': 175}), passages: 11800
- Encoder: `onnx:BAAI/bge-small-en-v1.5:512:cls` · config `3b083bf67d7b800f` · code `unversioned` · data `15fa63794d18`
- LLM calls: 0
- Graph: 602408 raw edges, 81520 capped, 200 hubs
- Chains: candidates {'1': 81520, '2': 241814}, stored 323334
- Index entries: dense 11800, hsi 335134

## Full-chain recall (%, 95% CI)

| System | Hop | R@5 | R@10 |
|---|---|---|---|
| bm25 | all | 9.2 [7.5, 11.1] | 14.3 [12.0, 16.5] |
| bm25 | 2 | 17.5 [14.1, 20.8] | 24.7 [21.0, 28.4] |
| bm25 | 3 | 1.0 [0.0, 2.2] | 5.4 [3.2, 7.9] |
| bm25 | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| dense | all | 15.4 [13.1, 17.7] | 22.2 [19.6, 24.8] |
| dense | 2 | 28.4 [24.5, 32.2] | 35.9 [31.8, 39.8] |
| dense | 3 | 2.9 [1.3, 4.8] | 11.7 [8.3, 15.6] |
| dense | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |
| expand-vec | all | 24.2 [21.7, 26.8] | 34.1 [31.0, 37.1] |
| expand-vec | 2 | 44.1 [40.0, 48.4] | 59.2 [55.1, 63.5] |
| expand-vec | 3 | 5.4 [2.9, 7.9] | 12.1 [8.6, 15.6] |
| expand-vec | 4 | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.7] |
| hsi | all | 24.7 [22.1, 27.4] | 35.5 [32.6, 38.7] |
| hsi | 2 | 45.3 [41.2, 49.2] | 61.0 [56.5, 65.1] |
| hsi | 3 | 5.1 [2.5, 7.6] | 13.3 [9.8, 16.8] |
| hsi | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |

## Paired difference: HSI minus baseline (points, 95% CI)

| Comparison | Hop | R@5 | R@10 |
|---|---|---|---|
| hsi-vs-bm25 | 2 | 27.8 [22.9, 32.5] | 36.3 [31.6, 41.4] |
| hsi-vs-bm25 | 3 | 4.1 [1.3, 7.0] | 7.9 [3.8, 12.1] |
| hsi-vs-bm25 | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |
| hsi-vs-dense | 2 | 16.9 [12.4, 21.8] | 25.1 [20.4, 30.2] |
| hsi-vs-dense | 3 | 2.2 [-0.6, 5.4] | 1.6 [-3.8, 6.3] |
| hsi-vs-dense | 4 | 0.0 [0.0, 0.0] | 0.0 [-2.3, 2.3] |
| hsi-vs-expand-vec | 2 | 1.2 [-1.4, 3.7] | 1.8 [-1.2, 4.5] |
| hsi-vs-expand-vec | 3 | -0.3 [-1.9, 1.0] | 1.3 [-1.6, 4.1] |
| hsi-vs-expand-vec | 4 | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.7] |

## Retrieval latency (ms, query encoding excluded)

| System | p50 | p95 |
|---|---|---|
| bm25 | 1.34 | 2.05 |
| dense | 1.84 | 2.65 |
| expand-vec | 4.76 | 6.82 |
| hsi | 39.05 | 52.57 |

## HSI failure waterfall (share of questions)

| Slice | link missed | pruned | not stored | not retrieved | ok | n |
|---|---|---|---|---|---|---|
| 2hop@5 | 0.09 | 0.17 | 0.00 | 0.29 | 0.45 | 510 |
| 2hop@10 | 0.08 | 0.15 | 0.00 | 0.16 | 0.61 | 510 |
| 3hop@5 | 0.16 | 0.75 | 0.00 | 0.04 | 0.05 | 315 |
| 3hop@10 | 0.16 | 0.69 | 0.00 | 0.02 | 0.13 | 315 |
| 4hop@5 | 0.29 | 0.71 | 0.00 | 0.00 | 0.00 | 175 |
| 4hop@10 | 0.29 | 0.70 | 0.00 | 0.00 | 0.01 | 175 |

## Gate 0 decision (computed)

- Status: **fail**
- Tested: 3-hop R@10, HSI vs expand-vec: 1.3 [-1.6, 4.1] points
- Latency p50: {'hsi': 39.04971549991387, 'expand-vec': 4.761208999752853}
- Rule: mean diff >= 0.05 and CI low > 0 and hsi p50 <= 1.5x baseline p50

The binding decision is written by a person in `specs/decisions/gate0.md`.
