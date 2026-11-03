# Decision 0011: one-time test on the HippoRAG 2 evaluation sets (pre-registered)

**Date:** 2026-11-02 · **Status:** pre-registered, written before the test run

## Final configuration (chosen on the MuSiQue train development split only)
- Encoder: bge-large-en-v1.5 (ONNX), query instruction from the model card.
- Graph: anchor + co-mention edges, corpus-learned mention extractor, fan-out 32, hub cap 200.
- Walk: PPR from dense seeds with learned transitions `runs/models/transitions-musique-v2-fan32.json`
  (trained on MuSiQue train with bge-small vectors; no retraining for the test).
- **System A** (top-5 precision): 3 seeds, restart 0.5, rank by total mass.
- **System B** (recall): 5 seeds, restart 0.5, rank by total mass, graph-smoothed vectors γ = 0.2.
- Baseline: dense retrieval with the same encoder. Rule walk (β = 2, 5 seeds) reported for reference.
Development evidence: runs `2629eefbbb069a31` (fan-out 32) and `242bb2c91e2d399d` (fan-out 64).

## Test data
HippoRAG 2's released sets (`osunlp/HippoRAG_2`): `musique.json` (1,000 questions) and
`2wikimultihopqa.json` (1,000 questions), corpora built from their paragraphs. Run once each.

## Pass rule (our method against its own dense baseline, same encoder)
On MuSiQue, gold-passage recall@5 over all questions:
`System A − dense ≥ 3 points` with the 95% bootstrap CI low end > 0.
2Wiki is reported with the same comparison (the transitions were trained on MuSiQue only, so 2Wiki
also measures transfer across datasets).

## External reference (reported, not part of the rule)
HippoRAG 2's published recall@5 on the same sets, with NV-Embed-v2 (7B) and an LLM-built graph. The
encoders differ (335M vs 7B), so this is context, not a like-for-like claim.

## Disclosure
Some of these MuSiQue questions were in the Gate 0 / Gate 0′ samples (seeds 0 and 1 of the MuSiQue
dev set) evaluated before decision 0004. Early supporting choices (co-mention edges, the learned
extractor) were made while looking at those results. All tuning since decision 0004 used the train split.

## Outcome (2026-11-02, single run each): **pass on both sets**
| Set | Run | Dense R@5 | System A R@5 | A − dense (95% CI) | System B R@10 | 4/3+4-hop full chain R@10 (best) |
|---|---|---|---|---|---|---|
| MuSiQue | `089d967e55b5b477` | 56.5 | **63.4** | **+6.9 [+5.3, +8.3]** | 73.3 | 22.0 (System B) |
| 2Wiki | `598a64a2f177e137` | 72.7 | **92.1** | **+19.4 [+17.7, +20.9]** | 96.9 | 85.5 (System A, 4-hop) |

Corpora match HippoRAG 2's released corpora (MuSiQue 11,655 unique passages both; 2Wiki 6,120 vs 6,119).

External reference, HippoRAG 2 paper (arXiv 2502.14802; figures from a secondary summary, to be checked
against the paper's table): Recall@5 MuSiQue NV-Embed-v2 69.7 → HippoRAG 2 74.7; 2Wiki 76.5 → 90.4.
- 2Wiki: System A **92.1 vs 90.4**, with a 335M encoder and no LLM at index time.
- MuSiQue: System A **63.4 vs 74.7**, 11.3 points below; the gap is mostly the encoder (dense 56.5 vs 69.7).

Caveats: one run per set; the HippoRAG 2 numbers are theirs, not re-run here; 2Wiki dev questions
overlapping this set appeared in Gate 0/0′ samples before decision 0004; the transitions were trained on
MuSiQue only, so 2Wiki also shows cross-dataset transfer. Latency p50: 117 ms (MuSiQue), 45 ms (2Wiki),
with global PPR; local push (decision 0005) is not yet wired to learned transitions.
