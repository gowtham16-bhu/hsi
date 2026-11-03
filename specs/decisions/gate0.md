# Gate 0 decision (RQ1)

**Status:** DECIDED by the project owner — Gate 0a **failed**; the original RQ1 claim is dropped (kill rule). Work continues with RQ1′ (decision 0003). · **Date:** 2026-11-02

## Runs
| Dataset | Config | Questions | Encoder | Key mode | LLM calls |
|---|---|---|---|---|---|
| MuSiQue | `3b083bf67d7b800f` (`configs/gate0/musique-0a.yaml`) | 1,000 (510 / 315 / 175 at 2 / 3 / 4 hops) | bge-small-en-v1.5 (ONNX) | mean (decision 0002) | 0 |
| 2WikiMultiHopQA | `a8156c312297ac9c` (`configs/gate0/2wiki-0a.yaml`) | 1,000 (779 / 221 at 2 / 4 hops) | same | mean | 0 |

Graph: anchor + co-mention edges (decision 0001), corpus-learned mention extractor, m = 8, b2 = 3.

## Result against the pre-registered rule
Rule: HSI beats online expansion (expand-vec) on 3-hop full-chain R@10 by ≥ 5 points, 95% CI above 0,
at p50 latency ≤ 1.5× the baseline.

| Dataset | Tested slice | HSI − expand-vec (points, 95% CI) | Latency p50 HSI / expand-vec | Rule |
|---|---|---|---|---|
| MuSiQue | 3-hop R@10 | +1.3 [−1.6, +4.1] | 39.0 / 4.8 ms | **fail** |
| 2Wiki | 2-hop R@10 (no 3-hop questions; provisional) | +3.6 [+1.0, +6.3] | 20.5 / 4.3 ms | **fail** |

Full-chain R@10 (%):

| System | MuSiQue 2 / 3 / 4 hops | 2Wiki 2 / 4 hops |
|---|---|---|
| BM25 | 24.7 / 5.4 / 0.0 | 49.2 / 0.5 |
| Dense | 35.9 / 11.7 / 1.1 | 61.6 / 0.0 |
| Online expansion (vec) | 59.2 / 12.1 / 0.6 | 72.1 / 4.5 |
| HSI (mean keys) | 61.0 / 13.3 / 1.1 | 75.7 / 7.7 |

## What the waterfall says
MuSiQue 3-hop @10: link missed 16%, **pruned 69%**, not stored 0%, not retrieved 2%, ok 13%.

- When a gold 3-chain is stored, a single lookup retrieves it in about 13 of 15 cases. The lookup works.
- Most gold chains are never generated: the query-independent beam (3 extensions per 2-chain) does not
  pick them. The failure is in candidate generation (C5 enumeration), not in retrieval.
- Latency is high because the index uses exact flat search over 335K entries; an ANN index would fix
  it, but the rule fails on recall regardless.

## Options for the owner
1. **Apply the kill rule as written.** Drop the hierarchy claim; narrow to LLM-free 2-hop shortcuts
   (gains are small: +1.8 MuSiQue, +3.6 2Wiki at 2 hops).
2. **One pre-registered diagnostic, then decide.** New hypothesis H1′: "the RQ1 failure is caused by
   enumeration coverage". Test it once on *fresh* questions (different seed, no overlap with these
   1,000) with exhaustive level-2 extension (b2 = m). This does not overturn the 0a failure; it decides
   whether the thesis is worth continuing in a revised form (question-aware selection, C6).
3. **Stop the thesis direction** and reuse the built engine for another research question.

Recommendation: option 2, because the waterfall localizes the failure precisely and the test costs no
LLM calls and under an hour of compute. Whatever the outcome, the paper must report this Gate 0a failure.
