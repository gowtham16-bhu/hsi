# Gate 0 report: musique

- Questions: 1000 (by hop: {'2': 510, '3': 315, '4': 175}), passages: 11800
- Encoder: `onnx:BAAI/bge-small-en-v1.5:512:cls` · config `38440f4d3eb9e179` · code `unversioned` · data `15fa63794d18`
- LLM calls: 0
- Graph: 580550 raw edges, 80392 capped, 145 hubs
- Chains: candidates {'1': 80392, '2': 238068}, stored 318460
- Index entries: dense 11800, hsi 330260

## Full-chain recall (%, 95% CI)

| System | Hop | R@5 | R@10 |
|---|---|---|---|
| bm25 | all | 10.4 [8.5, 12.4] | 13.8 [11.7, 16.1] |
| bm25 | 2 | 19.8 [16.5, 23.1] | 24.5 [21.0, 28.0] |
| bm25 | 3 | 1.0 [0.0, 2.2] | 4.1 [2.2, 6.3] |
| bm25 | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| dense | all | 15.4 [13.1, 17.7] | 22.2 [19.6, 24.8] |
| dense | 2 | 28.4 [24.5, 32.2] | 35.9 [31.8, 39.8] |
| dense | 3 | 2.9 [1.3, 4.8] | 11.7 [8.3, 15.6] |
| dense | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |
| expand-vec | all | 24.8 [22.1, 27.4] | 34.3 [31.2, 37.2] |
| expand-vec | 2 | 45.5 [41.2, 49.8] | 58.6 [54.5, 62.9] |
| expand-vec | 3 | 5.1 [2.9, 7.6] | 13.0 [9.2, 16.8] |
| expand-vec | 4 | 0.0 [0.0, 0.0] | 1.7 [0.0, 4.0] |
| hsi | all | 25.8 [23.1, 28.4] | 35.6 [32.4, 38.6] |
| hsi | 2 | 47.5 [43.1, 51.8] | 61.6 [57.3, 65.7] |
| hsi | 3 | 5.1 [2.9, 7.6] | 12.7 [9.2, 16.2] |
| hsi | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |

## Paired difference: HSI minus baseline (points, 95% CI)

| Comparison | Hop | R@5 | R@10 |
|---|---|---|---|
| hsi-vs-bm25 | 2 | 27.6 [23.1, 32.4] | 37.1 [32.4, 42.0] |
| hsi-vs-bm25 | 3 | 4.1 [1.3, 7.0] | 8.6 [4.4, 12.4] |
| hsi-vs-bm25 | 4 | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.9] |
| hsi-vs-dense | 2 | 19.0 [14.7, 23.5] | 25.7 [21.0, 30.4] |
| hsi-vs-dense | 3 | 2.2 [-0.6, 5.4] | 1.0 [-4.1, 6.0] |
| hsi-vs-dense | 4 | 0.0 [0.0, 0.0] | 0.0 [-1.7, 2.3] |
| hsi-vs-expand-vec | 2 | 2.0 [-0.4, 4.5] | 2.9 [0.4, 5.7] |
| hsi-vs-expand-vec | 3 | 0.0 [-1.6, 1.6] | -0.3 [-3.5, 2.9] |
| hsi-vs-expand-vec | 4 | 0.0 [0.0, 0.0] | -0.6 [-2.9, 1.1] |

## Retrieval latency (ms, query encoding excluded)

| System | p50 | p95 |
|---|---|---|
| bm25 | 0.62 | 0.95 |
| dense | 1.90 | 2.80 |
| expand-vec | 5.03 | 6.99 |
| hsi | 39.88 | 52.33 |

## HSI failure waterfall (share of questions)

| Slice | link missed | pruned | not stored | not retrieved | ok | n |
|---|---|---|---|---|---|---|
| 2hop@5 | 0.13 | 0.14 | 0.00 | 0.26 | 0.47 | 510 |
| 2hop@10 | 0.12 | 0.11 | 0.00 | 0.15 | 0.62 | 510 |
| 3hop@5 | 0.27 | 0.65 | 0.00 | 0.03 | 0.05 | 315 |
| 3hop@10 | 0.25 | 0.61 | 0.00 | 0.02 | 0.13 | 315 |
| 4hop@5 | 0.33 | 0.67 | 0.00 | 0.00 | 0.00 | 175 |
| 4hop@10 | 0.33 | 0.66 | 0.00 | 0.00 | 0.01 | 175 |

## Gate 0 decision (computed)

- Status: **fail**
- Tested: 3-hop R@10, HSI vs expand-vec: -0.3 [-3.5, 2.9] points
- Latency p50: {'hsi': 39.87826300044617, 'expand-vec': 5.033055000239983}
- Rule: mean diff >= 0.05 and CI low > 0 and hsi p50 <= 1.5x baseline p50

The binding decision is written by a person in `specs/decisions/gate0.md`.
