# 00 — Thesis

This is the first document of the project. Every other spec exists to serve it. When a design choice is unclear, the question is: *does it help prove or disprove this thesis?*

## Revision 2 (2026-11-02): current supported claim
After two failed gates (gate0.md, gate0-prime.md) the work moved to graph-first retrieval (decisions
0004–0010). The pre-registered final test (decision 0011, HippoRAG 2 evaluation sets, one run each) passed:

> **LLM-free multi-hop retrieval.** A passage graph built without any LLM (titles, co-mentions, a
> corpus-learned mention extractor), walked by PPR with a small learned transition model, raises
> gold-passage recall@5 over dense retrieval with the same encoder by +6.9 points on MuSiQue and +19.4
> on 2Wiki. On 2Wiki it exceeds HippoRAG 2's published recall@5 (92.1 vs 90.4) with a 335M encoder and
> no LLM; on MuSiQue it remains 11.3 points below, mostly because of the encoder.

Claims about precomputed multi-hop chains (original RQ1) and shortcut hops (RQ1′) are withdrawn.

## Revision 1 (2026-11-02)
Gate 0a failed the original RQ1 (decision `gate0.md`): a question-independent index cannot cover the
3–4 hop chains questions need. The thesis is revised to **shortcut hops** (decision 0003): precompute
2-passage shortcuts with no LLM, and hop over them at query time, so deep questions need about half the
lookup rounds. RQ1′ in decision 0003 replaces RQ1 below; the original text is kept as the record.

## 1. Title
**Hierarchical Shortcut Index: compiling multi-hop evidence chains at index time for single-lookup retrieval**

## 2. Thesis statement
> The cost and depth problems of multi-hop retrieval can be moved from query time to index time **without any LLM**. A budgeted hierarchy of precomputed passage-chain shortcuts, keyed by a learned composer over passage vectors, lets **one vector lookup** retrieve complete 2–4 hop evidence chains. It matches or beats iterative and LLM-indexed methods at a fraction of their cost, with the largest gains at 3–4 hops.

## 3. Motivation
The numbers below come from search excerpts of the cited papers and must be checked against the PDFs before they are quoted.

- **The depth cliff.** On MuSiQue, F1 falls sharply with hop count: IRCoT about 43 → 32 → 13, and HippoRAG 2 about 59 → 42 → 31, at 2 → 3 → 4 hops (as reported in the T²RAG preprint, Table 4).
- **Deep questions are common.** About 48% of the MuSiQue dev set needs 3 or 4 hops.
- **Current fixes are expensive.**
  - *Query-time iteration* (IRCoT, agents): about 3.5–8 LLM calls and several seconds per question.
  - *Index-time LLM extraction* (GraphRAG, LightRAG, HippoRAG 2, IndexRAG): about 12M–150M LLM tokens to index an 11.7K-passage corpus.
- **The missing bridge.** Single-shot retrieval fails because later hops share almost no words or meaning with the question. Only the first hop is easy to find.

## 4. Formal problem

**Corpus and graph.** A corpus `P = {p₁, …, p_N}` of passages, and a directed graph `G = (P ∪ E, A)` over passages and entities, built without LLM calls. A **chain** `c = (p₁, …, p_ℓ)` is a path in `G` through bridge entities `e₁, …, e_{ℓ−1}`, with `ℓ ∈ {1, 2, 3, 4}`.

**Question.** A question `q` has a gold chain `c* = (p*₁, …, p*_ℓ)`: the passages needed to answer it.

**Index.** A candidate set `C` of chains generated from `G`, and a stored subset `S ⊆ C` with a key `k_c` for each `c ∈ S`. The index is
```
I = { (v_p, p) : p ∈ P }  ∪  { (k_c, c) : c ∈ S },     |S| ≤ βN
```

**Single-lookup retrieval.** One nearest-neighbour search over `I`, then **unpacking** each hit into its passages, in order:
```
R_k(q) = first k passages of unpack( topK_I(q) )
```

**Objective.**
```
maximize   E_q [ 1( c* ⊆ R_k(q) ) ]                    full-chain recall at passage budget k
subject to |S| ≤ βN
           zero LLM calls while building G, C, S and the keys
           one nearest-neighbour search per question
```

The three design questions of the thesis are exactly the three free choices in this problem:

| Free choice | Thesis component |
|---|---|
| Which chain shapes enter `C` | **Hierarchy** (levels 1–3, caps, beams) |
| How each chain gets its key `k_c` | **Composer** with Chain-ICT training |
| Which `S ⊆ C` to keep under `βN` | **Budgeted selection** |

**Extended setting (QUALITY mode).** After the single search, a bounded completion step may extend chains along graph edges, scored by the composer, with no extra search and no LLM. It costs `O(b · m · L)` vector operations. RQ1 is tested **without** this step, so the core claim stays clean.

**Answering.** Assemble `R_k(q)` in hop order, make one reader LLM call, and verify the reader's bridges against `G` before answering (Section 6, RQ5).

## 5. Method in one view
Full details, parameters and acceptance criteria are in `03-core-engine.md`.

```
 OFFLINE (no LLM)
   graph G  ──▶  candidate chains C (levels 1–3)  ──▶  keys k_c = f(v_p₁ … v_p_ℓ)  ──▶  S = select(C, βN)
                                                                                     │
 ONLINE                                                                              ▼
   q  ──▶  one search over I = P ∪ S  ──▶  unpack  ──▶  [completion]  ──▶  rerank  ──▶  1 reader call  ──▶  verify
```

- **Hierarchy (C5).** Level 1 stores 2-passage chains, level 2 stores 3-passage chains, level 3 stores 4-passage chains, each built from the kept chains of the level below. Every entry stores its passage IDs, so a hit unpacks into ordered evidence.
- **Composer (C5).** A small transformer maps a chain's frozen passage vectors to one key. It is trained with InfoNCE on questions, with **Chain-ICT** pseudo-questions made from the corpus itself, and with an alignment term toward the frozen text encoder.
- **Selection (C6).** Facility-location coverage of useful chains, solved by lazy greedy per level, with a `(1 − 1/e)` guarantee per level under uniform cost.
- **Supporting machinery.** Soft alias edges and relation-faceted hubs make `G` robust (C3, C4). Graph-checked reading makes answers trustworthy (C9).

## 6. Research questions and hypotheses
Each hypothesis states what would **disprove** it.

| RQ | Question | Hypothesis | Disproved if | Experiment |
|---|---|---|---|---|
| **RQ1 Depth** | Can single-lookup chains recover deep evidence? | Full-chain recall@10 at 3 hops beats online graph expansion by ≥ 5 points (95% CI > 0), and beats iterative retrieval per unit of cost | The 95% CI includes 0, or the gain disappears at matched latency | E1, **Gate 0** |
| **RQ2 Keys** | Can chain keys be built from passage vectors alone? | Composed keys reach ≥ 90% of text-encoded keys' recall at 2 hops; Chain-ICT keeps ≥ 80% of the gain on unseen corpora | Below 80% of text keys, or no cross-corpus gain | E3, ablations |
| **RQ3 Budget** | How should a fixed index size be spent? | Greedy coverage selection beats random, degree-only and IndexRAG-style rules at every β | Any heuristic matches it at two or more β values | E2 |
| **RQ4 Cost** | Where does HSI sit on accuracy vs cost? | HSI is Pareto-better than LLM-indexed and iterative methods on MuSiQue | A baseline gives higher F1 at equal or lower cost | E4 |
| **RQ5 Trust** | Does checking the reader against the graph reduce wrong answers? | At equal coverage, error among answered questions is lower than with the unchecked reader | No difference across the risk–coverage curve | risk–coverage |

**Kill rule.** If RQ1 fails at Gate 0, the hierarchy claim is dropped, and the work narrows to what the data supports (for example, LLM-free budgeted 2-hop shortcuts).

## 7. Contributions (claim ledger)
A contribution may appear in the paper, blog or any public material only when its evidence column is filled.

| # | Contribution | Tier | Evidence required | Status |
|---|---|---|---|---|
| 1 | Hierarchical shortcut index: multi-level 2–4 hop chain entries in a flat vector index, unpacked into ordered evidence | **Core** | RQ1 | unproven |
| 2 | LLM-free composed chain keys with Chain-ICT self-supervision | **Core** | RQ2 | unproven |
| 3 | Budgeted chain selection with a per-level guarantee, plus verified-use promotion | **Core** | RQ3 | unproven |
| 4 | Soft alias edges in chain probability; relation-faceted hubs | Supporting | ablations | unproven |
| 5 | Graph-checked single-call reading with calibrated abstention | Supporting | RQ5 | unproven |
| 6 | Per-hop failure waterfall as an evaluation method | Method | used throughout | to build |

**Novelty status.** Searches up to November 2026 found no prior work combining contributions 1–3. Closest work: IndexRAG (2-hop LLM-written bridging facts), AAR (learned pair associations for reranking), HippoRAG 2 (query-time graph walk), TreeHop (query-time vector composition). The search is repeated before submission.

## 8. Positioning

| Method | When multi-hop work happens | LLM at index time | Searches per question | Deep (3–4 hop) units | Budget control |
|---|---|---|---|---|---|
| IRCoT, agents | query time | no | many | no | no |
| MDR, Beam retrieval | query time | no | one per hop | no | no |
| HippoRAG 2 | query time (graph walk) | yes | 1 + walk | no | no |
| HopRAG | index + query (walk) | yes | 1 + walk | no | no |
| IndexRAG | index time | yes | 1 | 2-hop facts | fixed caps |
| AAR | query time (rerank) | no | 1 + rerank | no | no |
| **HSI** | **index time** | **no** | **1** | **yes, 2–4 hops** | **optimized** |

## 9. Scope

| In the thesis | Supporting the thesis | Additions (not thesis claims) | Out of scope |
|---|---|---|---|
| Hierarchy, composer, selection, single-lookup retrieval and unpacking | Ingestion without titles, entity resolution, hub facets, graph-checked reading | Connectors, permissions, incremental updates, multimodal ingestion, copilot, domain applications | Training new LLMs; answer generation quality beyond verification; non-English |

Additions are specified in `specs/additions/`. They may use the core but never change its semantics (constitution P1).

## 10. Threats to validity

| Threat | Mitigation |
|---|---|
| Benchmark artifacts (shortcut-solvable questions) | Headline results on MuSiQue, which was built to reduce them; report per-hop results |
| Train–dev overlap in passages and chains | Leakage report; results on the unseen-chain subset |
| Weak baselines | Re-run every baseline with the same reader, corpora and passage budget; publish configs |
| Reader variance | Fixed reader across methods; 3 seeds; bootstrap confidence intervals |
| Unfair passage budgets | Compare at equal passages in context (5 and 10), never at equal hits |
| Over-claiming | Claim ledger (Section 7); limitations section next to each claim |

## 11. Deliverables and timeline
Mapped to the gates in `04-evaluation-and-gates.md`.

| Gate | Answers | Deliverable |
|---|---|---|
| Gate 0 (weeks 1–3) | RQ1, without completion or LLM | Decision record + numbers |
| Gate 1 (weeks 4–8) | RQ2, RQ3, RQ4 | Full engine results |
| Gate 2 (weeks 9–12) | RQ5, cross-corpus, ablations | Preprint, code release, blog post, venue submission |

**Paper outline:** introduction; related work; problem formulation (Section 4); method; experiments (RQ1–RQ5); analysis (waterfall, failure cases, ablations); extensions (one short section on additions); limitations; conclusion.
