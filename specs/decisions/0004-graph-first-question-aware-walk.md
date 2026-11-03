# Decision 0004: graph-first retrieval with a question-aware walk

**Date:** 2026-11-02 · **Status:** accepted (direction); test rule to be fixed before the test run

## Evidence (analysis of the Gate 0′ MuSiQue sample; no claims made on it)
3–4 hop questions (468): a gold passage is among the 5 dense seeds in 85.9%; all hops are linked in
the graph in 70.7%; all gold passages are in PPR's top 200 in only 35.0%. Median PPR rank of gold
hop 1 / 2 / 3: 3 / 10 / 46. Seeds and links are mostly fine; **PPR dilutes** — each step spreads score
evenly over ~8 neighbours regardless of the question.

## Direction
Core retrieval becomes a walk over the LLM-free graph, seeded by dense search. Candidate algorithms,
developed in this order:
1. **Question-aware transitions** ("intelligent surfer", Richardson & Domingos 2002):
   `P_q(u→v) ∝ w(u,v) · exp(β · cos(q, v_v))`; β = 0 is plain PPR.
2. Entity–passage bipartite walk with entity specificity weights (no fan-out cap, no hub cut).
3. Precomputed PPR vectors (linearity of PPR, Jeh & Widom 2003) so a query is a sparse sum of
   index-time vectors: the "shortcut" idea in its correct form.

## Protocol
- **Development:** MuSiQue **train** split, 1,000 questions (seed 0), same corpus protocol. All tuning
  happens here.
- **Test:** the 417 MuSiQue dev questions never used before (excluding seeds 0 and 1), and 2Wiki dev
  with seed 2 excluding seeds 0 and 1. Run once, with a rule written into this file before running.
- New metric alongside full-chain recall: **gold-passage recall@k** (share of gold passages retrieved),
  so results compare with published numbers.
