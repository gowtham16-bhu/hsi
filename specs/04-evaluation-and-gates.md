# 04 — Evaluation and Gates

Every experiment here answers a research question from `00-thesis.md`.

| RQ | Experiments | Gate |
|---|---|---|
| RQ1 Depth | Gate 0 suite, E1 | 0 |
| RQ2 Keys | E3, key-mode and Chain-ICT ablations | 1–2 |
| RQ3 Budget | E2 | 1 |
| RQ4 Cost | E4, E5 | 1 |
| RQ5 Trust | risk–coverage, verification ablation | 2 |
| Additions | E6 (domain harness), security suites | 2 |

## 1. Datasets

| Set | Use | Why |
|---|---|---|
| MuSiQue (CC BY 4.0) | Main research benchmark, 2–4 hops with per-hop gold paragraphs | Hardest standard multi-hop set |
| 2WikiMultiHopQA (Apache-2.0) | Second benchmark; has hyperlinks and evidence triples | Comparison questions, generalization |
| HotpotQA (CC BY-SA) | Sanity check only | Nearly saturated for dense retrieval |
| **OSV security harness** (built here) | Real-world showcase and second contribution | Aliases, hubs, company-style text, daily changes, structured ground truth |
| Natural question set (200, human-checked) | Main result for the harness | Template questions are easier than real ones |
| Resolution set | C3 evaluation | Built from OSV alias lists and package naming |

Corpora follow the HippoRAG 2 protocol (per-dataset passage pools, about 1,000 dev questions) so results compare with published work. All baselines are re-run here, never copied from papers.

## 2. Metrics

| Metric | Definition |
|---|---|
| Full-chain recall@k | Share of questions whose **every** gold passage is among the first k passages in context |
| Waterfall | Per question, the first failing stage: ① link missed, ② pruned, ③ not selected, ④ not retrieved, ⑤ lost in assembly, ⑥ reader wrong. Reported per hop depth |
| Answer EM / F1 | Standard; plus an LLM-judge accuracy reported separately |
| Risk–coverage | Error among answered questions vs share answered, across thresholds |
| Resolution | Pairwise F1 and B³ F1 |
| Cost | LLM calls, tokens, latency p50 / p95, index build time and LLM tokens (must be 0) |
| Freshness | Time from source change to answerable |
| Equivalence | Overlap of top-10 results between incremental updates and a full rebuild |

**Fairness rules:** same reader model, same corpora, same passage budget (5 and 10) for every method. 3 seeds, 95% bootstrap confidence intervals, paired significance tests. A leakage report shows the share of dev gold chains seen in training, with results on the unseen subset.

## 3. Baselines
BM25 · dense (same encoder) · hybrid · **online graph expansion from top passages (no shortcuts)** · cross-encoder on dense results · HippoRAG 2 · IndexRAG (if code is available) · AAR · RAPTOR · IRCoT · IRCoT with HSI as its retriever.

## 4. Experiments
- **E1** per-hop full-chain recall and F1 on MuSiQue (headline table).
- **E2** recall against budget β for each selection rule (headline plot).
- **E3** cross-corpus: train on 2Wiki + HotpotQA, test on MuSiQue, and the reverse.
- **E4** cost: accuracy-vs-cost Pareto plot.
- **E5** IRCoT with HSI underneath: agent turns saved.
- **E6** OSV harness: all of the above on real security data, plus freshness and permission tests.
- **Ablations:** online expansion only · reranker only · key modes · λ sweep · per-level vs shared composer · hub faceting on/off · soft aliases on/off · Chain-ICT on/off · hard negatives on/off.

## 5. Gates

### Gate 0: does precomputing chains help? (no LLM needed)
- **Split by compute (decision 0002):** 0a uses mean-of-vectors keys on 1,000 questions; 0b uses text keys and expand-text on 200 questions.
- **Run:** dense, BM25, online expansion, and 2- and 3-hop shortcuts with `text` keys, on MuSiQue and 2Wiki.
- **Pass:** shortcuts beat online expansion on 3-hop full-chain recall@10 by ≥ 5 points, with a 95% CI excluding 0, at a comparable or lower latency.
- **Fail:** stop and narrow the claim before any spending on LLM calls.

### Gate 1: does the full engine work?
- **Run:** composer, entity resolution, hub facets, selection, completion, reranker, reader and verifier. Experiments E1, E2, E4.
- **Pass:** 3-hop and 4-hop F1 above HippoRAG 2 (re-run) with p < 0.05 in QUALITY mode, with one LLM call; error among answered ≤ 5%.
- **Kill or pivot:** if levels 2–3 don't beat 2-hop shortcuts plus online completion, narrow the paper to LLM-free budgeted 2-hop shortcuts.

### Gate 2: is it ready to publish and pilot?
- **Run:** E3, E5, E6, all ablations, security test suites.
- **Pass:** cross-corpus keeps ≥ 80% of in-corpus gain; all security tests pass; results reproducible from a clean checkout.
- **Then:** open-source release, arXiv preprint, blog post, LinkedIn post, venue submission, internal pilot.

## 6. Continuous integration
- Every pull request touching `hsi-core` runs the Gate 0 suite on a fixed 200-question slice.
- Regression tolerance: −1.0 point on any per-hop recall, or any new permission or injection failure, fails the build.
