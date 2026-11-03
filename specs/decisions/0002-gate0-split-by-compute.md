# Decision 0002: split Gate 0 into 0a (mean keys, full) and 0b (text keys, subsample)

**Date:** 2026-11-02 · **Status:** accepted · **Affects:** specs/04 Gate 0, C5 key modes

## Context
Measured on the cloud dev machine (4 vCPU, no GPU), bge-small-en-v1.5 through ONNX Runtime:
21.6 passages/s, 7.5 three-passage chain texts/s. The MuSiQue Gate 0 graph (decision 0001,
m = 8, b2 = 3) has ~313K candidate chains.

| Work | Items | Time |
|---|---|---|
| Text keys for all chains | ~313K | ~11.6 h |
| expand-text on 1,000 questions (~160 chains encoded per question) | ~160K | ~6 h |

## Decision
- **Gate 0a** (1,000 questions, MuSiQue and 2Wiki): shortcut keys `k_c = normalize(Σ v_p)` (`KeyMode.MEAN`),
  baselines BM25, dense, expand-vec. Mean keys use exactly the scoring of expand-vec, so the comparison
  isolates the RQ1 question: does a global, precomputed, query-agnostic chain index beat expanding
  from the top seed passages at query time?
- **Gate 0b** (200 questions, same protocol): text keys and expand-text, run when compute allows.
- The Gate 0 pass rule is applied to 0a. 0b is reported alongside and cannot overturn a 0a failure.

## Consequences
- Text-key indexing costs ~27× passage encoding per chain. This is itself a result for RQ2/RQ4: it
  motivates composed keys (C5), which cost vector arithmetic only.
