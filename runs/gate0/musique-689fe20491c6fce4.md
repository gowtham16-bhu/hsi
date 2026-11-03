# Retrieval report: musique

- Questions: 1000 (by hop: {'2': 532, '3': 312, '4': 156}; excluded from an earlier sample: 1000), passages: 11566
- Encoder: `onnx:BAAI/bge-small-en-v1.5:512:cls` · config `689fe20491c6fce4` · code `unversioned` · data `15fa63794d18`
- LLM calls: 0
- Graph: 587052 raw edges, 80462 capped, 201 hubs
- Chains: candidates {'1': 80462}, stored 80462
- Index entries: {'passages': 11566, 'shortcuts': 92028}

## Full-chain recall (%, 95% CI)

| System | Slice | R@5 | R@10 |
|---|---|---|---|
| bm25 | all | 10.0 [8.1, 11.8] | 14.5 [12.5, 16.7] |
| bm25 | 2 | 18.0 [14.8, 21.2] | 24.8 [21.4, 28.6] |
| bm25 | 3 | 1.3 [0.3, 2.6] | 3.8 [1.9, 6.1] |
| bm25 | 4 | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.9] |
| bm25 | 3+4 | 0.9 [0.2, 1.7] | 2.8 [1.5, 4.3] |
| dense | all | 15.3 [13.0, 17.5] | 22.4 [19.7, 25.1] |
| dense | 2 | 26.7 [23.1, 30.6] | 34.8 [31.0, 38.9] |
| dense | 3 | 3.5 [1.6, 5.8] | 12.5 [9.0, 16.3] |
| dense | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| dense | 3+4 | 2.4 [1.1, 3.8] | 8.3 [6.0, 10.9] |
| expand-vec | all | 22.2 [19.6, 24.7] | 32.4 [29.6, 35.3] |
| expand-vec | 2 | 38.7 [34.2, 42.7] | 53.6 [49.2, 57.7] |
| expand-vec | 3 | 4.5 [2.2, 7.1] | 11.9 [8.3, 15.4] |
| expand-vec | 4 | 1.3 [0.0, 3.2] | 1.3 [0.0, 3.2] |
| expand-vec | 3+4 | 3.4 [1.9, 5.1] | 8.3 [6.0, 11.1] |
| ppr | all | 15.4 [13.1, 17.7] | 34.8 [31.8, 37.7] |
| ppr | 2 | 26.9 [23.3, 30.6] | 57.1 [53.2, 61.1] |
| ppr | 3 | 3.5 [1.6, 5.8] | 14.1 [10.3, 18.3] |
| ppr | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| ppr | 3+4 | 2.4 [1.1, 3.8] | 9.4 [6.8, 12.2] |
| hop-passages-2 | all | 13.8 [11.6, 16.0] | 24.3 [21.7, 26.9] |
| hop-passages-2 | 2 | 25.6 [21.8, 29.3] | 41.4 [37.4, 45.5] |
| hop-passages-2 | 3 | 0.6 [0.0, 1.6] | 7.4 [4.5, 10.3] |
| hop-passages-2 | 4 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| hop-passages-2 | 3+4 | 0.4 [0.0, 1.1] | 4.9 [3.2, 7.1] |
| hop-passages-4 | all | 10.2 [8.4, 12.2] | 18.2 [15.8, 20.6] |
| hop-passages-4 | 2 | 19.2 [15.8, 22.6] | 31.0 [27.1, 35.0] |
| hop-passages-4 | 3 | 0.0 [0.0, 0.0] | 5.1 [2.9, 7.7] |
| hop-passages-4 | 4 | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.9] |
| hop-passages-4 | 3+4 | 0.0 [0.0, 0.0] | 3.6 [2.1, 5.6] |
| hop-shortcuts-2 | all | 22.4 [20.0, 25.1] | 30.9 [28.1, 33.9] |
| hop-shortcuts-2 | 2 | 40.2 [35.9, 44.2] | 53.0 [48.9, 57.0] |
| hop-shortcuts-2 | 3 | 2.9 [1.3, 4.8] | 8.3 [5.4, 11.5] |
| hop-shortcuts-2 | 4 | 0.6 [0.0, 1.9] | 0.6 [0.0, 1.9] |
| hop-shortcuts-2 | 3+4 | 2.1 [0.9, 3.6] | 5.8 [3.6, 7.9] |

## Latency (ms, query encoding excluded) and lookups

| System | p50 | p95 | ANN calls / question |
|---|---|---|---|
| bm25 | 1.29 | 2.14 |  |
| dense | 1.85 | 2.61 |  |
| expand-vec | 9.72 | 15.66 |  |
| ppr | 12.73 | 20.15 |  |
| hop-passages-2 | 10.60 | 15.77 | 6.0 |
| hop-passages-4 | 24.12 | 36.39 | 16.0 |
| hop-shortcuts-2 | 68.05 | 85.41 | 6.0 |

## Failure waterfall: hop-shortcuts-2 (share of questions)

| Slice | link missed | pruned | not stored | not retrieved | ok | n |
|---|---|---|---|---|---|---|
| 2hop@5 | 0.08 | 0.21 | 0.00 | 0.30 | 0.40 | 532 |
| 2hop@10 | 0.08 | 0.20 | 0.00 | 0.19 | 0.53 | 532 |
| 3hop@5 | 0.13 | 0.84 | 0.00 | 0.00 | 0.03 | 312 |
| 3hop@10 | 0.13 | 0.79 | 0.00 | 0.00 | 0.08 | 312 |
| 4hop@5 | 0.27 | 0.72 | 0.00 | 0.00 | 0.01 | 156 |
| 4hop@10 | 0.27 | 0.72 | 0.00 | 0.00 | 0.01 | 156 |

## Decision rule `gate0-prime-rq1-prime`: **fail**

| Candidate − baseline | Slice | k | Difference (points, 95% CI) | Condition | Pass |
|---|---|---|---|---|---|
| hop-shortcuts-2 − hop-passages-2 | 3+4 | 10 | 0.9 [-1.5, 3.2] | min_difference=0.05, ci_low_above=0.0 | no |
| hop-shortcuts-2 − hop-passages-4 | 3+4 | 10 | 2.1 [-0.4, 4.5] | ci_low_at_least=-0.02 | yes |
| hop-shortcuts-2 − ppr | 3+4 | 10 | -3.6 [-6.6, -0.9] | ci_low_at_least=0.0 | no |
| hop-shortcuts-2 − expand-vec | 3+4 | 10 | -2.6 [-5.1, 0.0] | ci_low_at_least=0.0 | no |

The binding decision is written by a person in `specs/decisions/`.
