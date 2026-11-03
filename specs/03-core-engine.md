# 03 — Core Engine

Ten modules. Each has: purpose, interface, algorithm, parameters (with defaults), and acceptance criteria. Defaults are starting values to be tuned at the gates. Acceptance numbers marked *(target)* are set or adjusted after Gate 0.

Notation: `q` question vector, `v_p` passage vector, `p_ℓ` last passage of a chain, `ℓ` chain length in passages, `N` number of passages, `m` fan-out cap, `β` shortcut budget ratio.


## Tiers
The modules are not equally important to the thesis (`00-thesis.md`).

| Tier | Modules | Role |
|---|---|---|
| **Thesis core** | C5 chains + composer, C6 selection, C7 index + single-lookup retrieval and unpacking | The claims tested by RQ1–RQ4 |
| **Supporting** | C2 ingestion, C3 entity resolution, C4 graph + hub facets, C9 verification | Robustness on real data; RQ5; ablations |
| **Foundation** | C1 canonical model + interfaces | Types every module speaks |
| **Addition** | C8 permissions, C10 incremental update | Product requirements; in the engine, but not thesis claims |

Gate 0 tests the thesis core with the simplest supporting pieces (titles as anchors, title matching, `text` keys) so that RQ1 is answered before anything else is built.

---

## C1. Canonical model and plugin interfaces · *foundation*
**Purpose.** The types in `02-architecture.md` §3, and the plugin protocols in `additions/A2`.
**Acceptance.**
- All types are immutable, validated and serializable to JSON.
- The `hsi-core` package imports no network, LLM or vendor library (enforced by an import-linter rule in CI).

---

## C2. Ingestion · *supporting*
**Purpose.** Turn source documents into passages, and find what each passage is about, without relying on titles.

**Interface.**
```python
def ingest(doc: SourceDocument, parser: Parser, chunker: Chunker, ner: MentionExtractor) -> IngestResult
# IngestResult = passages[], mentions[], topic_scores[(passage_id, entity_candidate, score)]
```

**Algorithm.**
1. Parser plugin extracts text with structure (sections, headings, tables, lists).
2. Chunking is structure-first (split at section boundaries), then **content-defined** inside long sections: a boundary is placed where `rollhash(last 48 bytes) mod M = 0`, bounded by `min_tokens` and `max_tokens`. A local edit then changes O(1) chunks.
3. Mentions come from the NER plugin, plus regex families from config (IDs such as CVE, ticket keys, purl), plus defined-term patterns ("hereinafter", "referred to as").
4. Topic score:
```
topic(p, e) = 0.4·heading(p,e) + 0.3·subject(p,e) + 0.2·tf(e,p)/max_e′ tf(e′,p) + 0.1·early(p,e)
Anchors(e)  = { p : topic(p,e) ≥ τ_topic }
```
5. A document title, when present, counts as a heading at the document level.

**Parameters.** `min_tokens=64`, `max_tokens=320`, `M=2^7`, `τ_topic=0.5`.

**Acceptance.**
- Editing one sentence in a 50-page document changes ≤ 3 passage hashes.
- On the evaluation corpora, ≥ 95% of gold passages have at least one anchor entity *(target)*.

---

## C3. Entity resolution (aliases and same-name splitting) · *supporting*
**Purpose.** Group mentions and structured refs into entities: merge aliases, split different entities that share a name.

**Interface.**
```python
def resolve(mentions: list[Mention], facts: list[StructuredFact], model: PairModel) -> Resolution
# Resolution = entities[], mention→entity map, soft alias edges[]
```

**Algorithm.**
1. **Hard joins first.** Mentions or facts with the same `EntityRef` are one entity (confidence 1.0). Facts with predicate `alias_of` and confidence ≥ 0.95 are hard joins.
2. **Blocking.** Candidate pairs share a normalized token, initials, acronym or ref prefix.
3. **Pair probability.**
```
p_ij = σ(wᵀ φ(m_i, m_j))
φ = [ jaro_winkler, token_jaccard, acronym_match, alias_pattern,
      cos(c_i, c_j), adamic_adar(N_i, N_j), same_doc_coref, type_agree, source_trust ]
type mismatch → p_ij = 0
```
4. **Correlation clustering** inside each block:
```
minimize  Σ_{i<j} x_ij·(1 − p_ij) + (1 − x_ij)·p_ij
solver: randomized pivot (KwikCluster), best of R runs
```
5. **Soft aliases.** Pairs with `τ_low ≤ p_ij < τ_high` in different clusters become soft alias edges with a Beta prior: `α₀ = κ·p_ij`, `β₀ = κ·(1 − p_ij)`.
6. **Feedback.** When a verified answer uses a soft alias, `α += 1`. When the verifier rejects a bridge because of it, `β += 1`.
7. **Trust.** Alias evidence from sources with `trust < τ_trust` can only create soft edges, never hard merges.

**Training.** `w` is fit by logistic regression on weak labels: positives from alias patterns, shared refs and public alias data (such as advisory alias lists); negatives from type mismatches and same-string pairs with conflicting contexts.

**Parameters.** `τ_low=0.35`, `τ_high=0.80`, `κ=4`, `R=5`, `τ_trust=0.5`.

**Acceptance.**
- Pairwise F1 and B³ F1 reported on a labeled resolution set. B³ F1 ≥ 0.90 on the OSV alias set *(target)*.
- Two same-name entities with disjoint contexts and no shared neighbors are never merged (test fixture).
- Deterministic for a fixed seed.

---

## C4. Graph and hub facets · *supporting*
**Purpose.** Build the typed passage–entity graph, and split hub entities so chains through them stay meaningful.

**Algorithm.**
1. Edges:
   - `mention`: passage → entity, weight `P(link)`.
   - `anchor`: entity → passage, weight `topic(p, e)`.
   - `structured`: entity → entity from facts, weight `confidence`.
   - `reference`, `thread`, `section`: passage → passage from source metadata, weights from config.
   - `co-mention`: passage ↔ passage when both mention the same entity `e`, with no anchor required.
     Weight `w_c / sqrt(df(e) − 1)`, where `df(e)` is the number of passages mentioning `e`; only for
     `2 ≤ df(e) ≤ D_max`. Added after Gate 0 measured that on MuSiQue only 24% of consecutive gold hops
     are linked by a title mention, while the bridge entity appears in both passages in 83%
     (decision record `specs/decisions/0001-co-mention-edges.md`).
2. A **hub** is an entity with in-degree `> D_max`.
3. **Facets.** For each hub mention, build a relation-context vector `r_m` from the words between the mention and its governing verb or noun. Cluster `r_m` (k-means, `k` chosen by silhouette in `[2, F_max]`). Each cluster becomes a facet entity `facet_of = hub`.
4. Facets still above `D_max` are split again once. What remains is left to online completion only.
```
pairs before  = k(k−1)/2
pairs after   = Σ_f k_f(k_f−1)/2,   Σ_f k_f = k
```

**Parameters.** `D_max=50`, `F_max=32`.

**Acceptance.**
- Shortcut candidates through hubs drop by ≥ 10× versus unfaceted hubs on the evaluation corpora *(target)*.
- Waterfall stage ② (pruned) rate reported before and after faceting.

---

## C5. Candidate chains and composer · *thesis core*
**Purpose.** Enumerate useful 2–4 hop chains under caps, and compute one key vector per chain.

**Chain enumeration.**
```
level 1: all A → B with edge weight ≥ τ_edge, at most m per A, ranked by weight × mention salience
level 2: extend each kept level-1 chain by its top b₂ next hops (no repeated passages)
level 3: extend each kept level-2 chain by its top b₃ next hops
log P(c) = Σ_t log weight(edge_t) + Σ_{soft aliases a} log π_a
```

**Composer.** A small transformer over frozen passage vectors.
```
input   [LEN_ℓ] v_1+pos_1 … v_ℓ+pos_ℓ
model   2 layers, d = encoder dim, 4 heads; mean-pool → linear → L2-normalize
output  k_c
```

**Training.**
```
L = InfoNCE(q, k_c ; in-batch + hard negatives)                 labeled questions, when available
  + InfoNCE(q̃, k_c ; same-start wrong-branch negatives)          Chain-ICT pseudo-questions
  + λ · (1 − cos(k_c, enc(text(c))))                            alignment to the frozen encoder

Chain-ICT:  q̃ = mask(s₁, e₁) ⊕ mask(s_ℓ, subject(s_ℓ)), with all bridge names removed
hard negatives: same p₁ with a wrong next hop · correct passages in the wrong order · chains through the same hub
```

**Key modes.** `composed` (vectors only), `text` (encode joined chain text), `hybrid` (composer selects, chosen chains get text keys).

**Drift check.** `d = 1 − mean_{c ∈ sample} cos(f(c), enc(text(c)))`. If `d > δ`, that corpus switches to `hybrid` and Chain-ICT retraining is scheduled.

**Parameters.** `τ_edge=0.3`, `m=8`, `b₂=3`, `b₃=2`, `λ ∈ {0, 0.1, 0.3, 0.5}` (swept), `δ=0.15`, hard negatives = 4 per example.

**Acceptance.**
- Composed keys reach ≥ 90% of `text`-key full-chain recall@10 at 2 hops *(target)*.
- Cross-corpus: keeps ≥ 80% of the in-corpus gain (experiment E3).

---

## C6. Budgeted selection · *thesis core*
**Purpose.** Choose which candidate chains become stored shortcuts, within `|S| ≤ βN`.

**Algorithm.**
```
u(c)  = σ(θᵀ ψ(c))     ψ = [bridge-entity degree (IDF-like), mention position, log P(c),
                            question-likeness of k_c, chain length]      corpus-independent features
F(S)  = Σ_{c ∈ candidates} u(c) · max_{s ∈ S} sim(c, s)
sim   = Jaccard(passages(c), passages(s))
solver: lazy greedy, per level, with a level budget split
```
With uniform cost per shortcut, greedy is within (1 − 1/e) of optimal **per level**. Interactions between levels are heuristic, and the paper must say so.

**Verified-use promotion.** A chain built by online completion that serves verified answers gets `benefit = uses × time saved`. It replaces the lowest-benefit stored shortcut of the same level when its benefit is higher. `|S| ≤ βN` always holds.

**Parameters.** `β ∈ {0.5, 1, 2, 4}`, level split `50 / 30 / 20`.

**Acceptance.**
- At every tested `β`, greedy selection gives higher 3–4 hop full-chain recall than random, degree-only and the IndexRAG-style rule (experiment E2) *(target)*.

---

## C7. Index and retrieval · *thesis core*
**Purpose.** Store passages and shortcuts in one vector index, retrieve chains, complete short ones, and rerank.

**Algorithm.**
1. Index rows: passages (`level 0`) and shortcuts (`level ≥ 1`), with a side table `shortcut → passages, bridges`.
2. Retrieve top-K from the dense index and top-K from BM25, then fuse with reciprocal rank fusion.
3. Unpack shortcut hits into ordered chains. Single passages become chains of length 1.
4. **C8 permission filter** (always, before anything else sees the chains).
5. **Completion:** for each chain ending at `p_ℓ` with `ℓ < 4`, score neighbors:
```
g(q, c ⊕ p) = cos(q, f(v_c, v_p)) + w₄ · log P(p_ℓ → p)        beam b, up to 4 passages
```
6. **Chain score and rerank:**
```
S(q, c) = w₁·cos(q, k_c) + w₂·cos(q, v_{p₁}) + w₃·max_{s ∈ p_ℓ} cos(q, v_s) + w₄·log P(c) − w₅·(ℓ − 1)
R(q, c) = S(q, c) + w₆·CE(q, text(c))           cross-encoder on the top 50 only
```

**Modes.** FAST: steps 1–4 + S. QUALITY: all steps. MAX: QUALITY + one retry in C9.

**Parameters.** `K=20`, `b=5`, rerank pool 50, `w` learned on the training split.

**Acceptance.**
- Retrieval latency meets NFR-1 *(target)*.
- Full-chain recall@10 passages at 3–4 hops ≥ dense + 10 points (experiment E1) *(target)*.

---

## C8. Permissions · *addition*
**Purpose.** Guarantee that no user sees evidence they cannot see in the source.

**Rules.**
```
visible(p, u)   ⇔  acl(p).public  or  acl(p).principals ∩ principals(u) ≠ ∅
visible(c, u)   ⇔  ∀ p ∈ c : visible(p, u)
visible(fact,u) ⇔  same rule on the fact's acl
```
- Filtering happens in C7 step 4, before completion, reranking or the reader.
- Completion only expands into visible passages.
- **Default policy `hide`:** hidden hops are not mentioned at all, so their existence isn't revealed. Policy `redacted` ("some evidence is not accessible to you") is opt-in.
- **Strict mode:** shortcuts are built only from passages that share the same ACL, so even ranking is unaffected by hidden content.

**Acceptance.**
- Property-based test: for random corpora, ACLs and users, no returned chain, passage, snippet, count or confidence depends on a passage the user can't see (in strict mode), and no returned chain contains one (in all modes).
- ACL changes at the source take effect within NFR-2.

---

## C9. Reader contract and verification · *supporting*
**Purpose.** One LLM call that returns structured reasoning, then checks that reasoning against the graph.

**Assembly.** Top chains in hop order, deduplicated, within the passage budget. Two separate slots for comparison questions. Passages are wrapped as quoted data with source IDs (P9).

**Reader output schema (JSON).**
```json
{ "chains": [ { "chain_id": 1, "bridges": ["..."], "answer": "...", "quote": "..." } ], "final": "..." }
```

**Verification.**
```
bridge check   B(c,y) = (1/(ℓ−1)) · Σ_t 1[ y.bridge_t ∈ aliases(e_t) ]
quote check    Q(c,y) = 1[ y.quote ⊂ text(p_ℓ) ] · 1[ y.answer ⊂ y.quote ]
type check     T(q,y) = 1[ type(y.answer) = expected_type(q) ]
entailment     E(c,y) = P_NLI(entails | p_ℓ, answer statement)
chain vote     V(a)   = Σ_c R(q,c)·1[y_c.answer = a]·B(c,y) / Σ_c R(q,c)
final          a*     = argmax_a V(a)
confidence     conf   = isotonic(σ(θ₀ + θ₁ log P(c*) + θ₂E + θ₃B + θ₄T + θ₅Q + θ₆V(a*)))
decision       conf ≥ τ → answered      conf < τ → partial (verified hops only)
τ chosen on dev so that error among answered ≤ 5%   (NFR-5)
```
MAX mode: if `conf < τ`, one retry with the next-best chains.

**Acceptance.**
- Risk–coverage curve reported. At the chosen `τ`, error among answered ≤ 5% on dev *(target)*.
- A reader output whose bridges contradict the graph is never returned as `answered` (test fixture).
- Instructions planted inside documents never change the output format or trigger actions (injection test suite in `additions/A3`).

---

## C10. Incremental update · *addition*
**Purpose.** Apply source changes in seconds, without full rebuilds.

**Algorithm.**
```
Δ   = passages whose hash changed (added, edited, removed)
E_Δ = entities mentioned or anchored in Δ, before ∪ after
I   = { stored c : c ∩ Δ ≠ ∅ } ∪ { stored c : a bridge of c ∈ E_Δ and its edge changed }
rebuild C3–C6 inside the L-hop radius of Δ, write new rows, then swap atomically and tombstone I
work = O(|Δ| · m^(L−1)), independent of N
```
Conflicting facts are kept with `valid_from`, and readers see the newest first.

**Acceptance.**
- After any sequence of updates, retrieval results equal a full rebuild on the same final data, up to selection ties *(target: ≥ 99% identical top-10)*.
- One-document update completes in ≤ 5 s at 1M passages *(target)*.

---

## Cross-cutting: evaluation hooks
Every module exposes counters for the failure waterfall (`04`): link missed (C2–C3), pruned (C4–C5), not selected (C6), not retrieved (C7), lost in assembly (C9), reader wrong (C9).
