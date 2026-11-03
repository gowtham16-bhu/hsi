# Retrieval report: musique

- Questions: 1000 (by hop: {'2': 734, '3': 212, '4': 54}; excluded from an earlier sample: 0), passages: 14223
- Encoder: `onnx:BAAI/bge-small-en-v1.5:512:cls` · config `a04aeaa235e94bed` · code `unversioned` · data `83a75b1e11e4`
- LLM calls: 0
- Graph: 2797714 raw edges, 403167 capped, 43 hubs
- Chains: candidates {}, stored 0
- Index entries: {'passages': 14223, 'shortcuts': 0}

## Full-chain recall (%, 95% CI)

| System | Slice | R@5 | R@10 |
|---|---|---|---|
| dense | all | 24.3 [21.8, 27.0] | 33.0 [30.2, 35.9] |
| dense | 2 | 32.0 [28.6, 35.4] | 43.2 [39.6, 46.7] |
| dense | 3 | 3.3 [0.9, 5.7] | 5.7 [2.8, 9.0] |
| dense | 4 | 1.9 [0.0, 5.6] | 1.9 [0.0, 5.6] |
| dense | 3+4 | 3.0 [1.1, 5.3] | 4.9 [2.6, 7.5] |
| global-learned-s3 | all | 37.3 [34.3, 40.2] | 51.1 [48.1, 54.0] |
| global-learned-s3 | 2 | 47.3 [43.6, 50.8] | 63.4 [59.7, 66.9] |
| global-learned-s3 | 3 | 10.8 [7.1, 15.6] | 18.9 [13.7, 24.5] |
| global-learned-s3 | 4 | 5.6 [0.0, 13.0] | 11.1 [3.7, 20.4] |
| global-learned-s3 | 3+4 | 9.8 [6.4, 13.5] | 17.3 [12.8, 21.8] |
| push-learned-s3-e1e-3 | all | 37.3 [34.2, 40.1] | 50.9 [48.0, 53.9] |
| push-learned-s3-e1e-3 | 2 | 47.4 [43.6, 50.8] | 63.2 [59.5, 66.8] |
| push-learned-s3-e1e-3 | 3 | 10.4 [6.6, 14.6] | 18.4 [13.2, 23.6] |
| push-learned-s3-e1e-3 | 4 | 5.6 [0.0, 13.0] | 11.1 [3.7, 20.4] |
| push-learned-s3-e1e-3 | 3+4 | 9.4 [6.0, 13.2] | 16.9 [12.4, 21.4] |
| push-learned-s3-e1e-4 | all | 37.2 [34.2, 40.1] | 51.2 [48.2, 54.1] |
| push-learned-s3-e1e-4 | 2 | 47.1 [43.3, 50.5] | 63.5 [59.8, 67.0] |
| push-learned-s3-e1e-4 | 3 | 10.8 [7.1, 15.6] | 18.9 [13.7, 24.5] |
| push-learned-s3-e1e-4 | 4 | 5.6 [0.0, 13.0] | 11.1 [3.7, 20.4] |
| push-learned-s3-e1e-4 | 3+4 | 9.8 [6.4, 13.5] | 17.3 [12.8, 21.8] |
| push-learned-s3-e1e-5 | all | 37.3 [34.3, 40.2] | 51.1 [48.1, 54.0] |
| push-learned-s3-e1e-5 | 2 | 47.3 [43.6, 50.8] | 63.4 [59.7, 66.9] |
| push-learned-s3-e1e-5 | 3 | 10.8 [7.1, 15.6] | 18.9 [13.7, 24.5] |
| push-learned-s3-e1e-5 | 4 | 5.6 [0.0, 13.0] | 11.1 [3.7, 20.4] |
| push-learned-s3-e1e-5 | 3+4 | 9.8 [6.4, 13.5] | 17.3 [12.8, 21.8] |

## Gold-passage recall (% of gold passages retrieved, 95% CI)

| System | Slice | R@5 | R@10 |
|---|---|---|---|
| dense | all | 56.0 [54.1, 58.0] | 62.8 [60.9, 64.7] |
| dense | 2 | 62.0 [59.9, 64.0] | 68.8 [66.6, 70.9] |
| dense | 3 | 41.8 [38.7, 45.0] | 48.0 [44.8, 51.3] |
| dense | 4 | 30.6 [25.0, 37.5] | 39.8 [33.3, 46.8] |
| dense | 3+4 | 39.5 [36.3, 42.5] | 46.3 [43.2, 49.4] |
| global-learned-s3 | all | 62.1 [60.1, 64.2] | 70.8 [68.7, 72.7] |
| global-learned-s3 | 2 | 68.5 [66.0, 71.0] | 77.5 [75.1, 79.8] |
| global-learned-s3 | 3 | 46.5 [42.8, 50.5] | 54.7 [50.6, 59.0] |
| global-learned-s3 | 4 | 36.1 [29.2, 44.4] | 41.7 [33.8, 50.5] |
| global-learned-s3 | 3+4 | 44.4 [40.8, 48.0] | 52.1 [48.3, 55.7] |
| push-learned-s3-e1e-3 | all | 62.1 [60.0, 64.1] | 70.6 [68.6, 72.7] |
| push-learned-s3-e1e-3 | 2 | 68.5 [66.0, 71.0] | 77.5 [75.0, 79.7] |
| push-learned-s3-e1e-3 | 3 | 46.2 [42.5, 50.0] | 54.4 [50.3, 58.6] |
| push-learned-s3-e1e-3 | 4 | 36.1 [29.2, 44.4] | 41.7 [33.8, 50.5] |
| push-learned-s3-e1e-3 | 3+4 | 44.2 [40.6, 47.7] | 51.8 [47.9, 55.5] |
| push-learned-s3-e1e-4 | all | 62.1 [60.0, 64.1] | 70.7 [68.7, 72.8] |
| push-learned-s3-e1e-4 | 2 | 68.5 [66.0, 70.9] | 77.6 [75.1, 79.9] |
| push-learned-s3-e1e-4 | 3 | 46.5 [42.8, 50.5] | 54.4 [50.2, 58.7] |
| push-learned-s3-e1e-4 | 4 | 36.1 [29.2, 44.4] | 41.7 [33.8, 50.5] |
| push-learned-s3-e1e-4 | 3+4 | 44.4 [40.8, 48.0] | 51.8 [48.2, 55.6] |
| push-learned-s3-e1e-5 | all | 62.1 [60.1, 64.2] | 70.7 [68.7, 72.7] |
| push-learned-s3-e1e-5 | 2 | 68.5 [66.0, 71.0] | 77.5 [75.1, 79.8] |
| push-learned-s3-e1e-5 | 3 | 46.5 [42.8, 50.5] | 54.6 [50.5, 58.8] |
| push-learned-s3-e1e-5 | 4 | 36.1 [29.2, 44.4] | 41.7 [33.8, 50.5] |
| push-learned-s3-e1e-5 | 3+4 | 44.4 [40.8, 48.0] | 51.9 [48.2, 55.6] |

## Latency (ms, query encoding excluded) and lookups

| System | p50 | p95 | ANN calls / question | pushes / question |
|---|---|---|---|---|
| dense | 1.80 | 10.49 |  |  |
| global-learned-s3 | 83.20 | 261.07 |  |  |
| push-learned-s3-e1e-3 | 12.49 | 38.52 |  | 120 |
| push-learned-s3-e1e-4 | 46.79 | 79.70 |  | 489 |
| push-learned-s3-e1e-5 | 169.39 | 316.23 |  | 2164 |

## Decision rule `decision-0012`: **pass**

| Candidate − baseline | Slice | k | Difference (points, 95% CI) | Condition | Pass |
|---|---|---|---|---|---|
| push-learned-s3-e1e-4 − global-learned-s3 | all | 5 | -0.1 [-0.1, 0.0] | metric=passage_recall, ci_low_at_least=-0.01, max_latency_ratio=1.0 | yes |

The binding decision is written by a person in `specs/decisions/`.
