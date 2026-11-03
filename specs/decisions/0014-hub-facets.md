# Decision 0014: hub facets instead of dropping common names

**Date:** 2026-11-03 · **Status:** pre-registered · **Affects:** C4 (graph build), C7 (walk inputs)

## Context
The scaling benchmark (decision 0013) showed the walk's recall gain over dense falling from +8.9 points
(10K passages) to +0.2 (140K). Diagnosis on the 269 gold hop pairs of the 200 benchmark questions:

| Passages | Gold pairs linked | Shared name is a hub (> 200 participants) | Pruned by fan-out | No shared name |
|---|---|---|---|---|
| 10K | 68% | 4% | 13% | 15% |
| 40K | 54% | 12% | 19% | 15% |
| 140K | 35% | 33% | 18% | 14% |

The hub cap is an absolute count (D_max = 200). As the corpus grows, ordinary bridge names cross it
(median bridge participants 20 at 10K, 252 at 140K) and lose **all** their edges, including the forward
anchor edge "u mentions e → the passage about e". At larger N almost every name would become a hub.

## Decision
New graph option `hub_mode: facets` (the default stays `drop`, so earlier runs reproduce exactly):

1. **Forward anchor edges are kept for hub entities.** Each mentioning passage produces one candidate per
   anchor, so the cost is linear in participants and needs no cap. How common the name is stays visible to
   the walk through the `log_df` feature (df = participants of the whole entity).
2. **Facets.** The participants of a hub entity are split by their passage vectors into facets of at most
   `facet_size` passages: recursive bisection at the median projection onto the principal direction
   (power iteration, `facet_iterations` steps, deterministic start). It is balanced by construction,
   deterministic, and costs O(k · d · log(k / facet_size)) for k participants.
3. **Within a facet,** the normal rules apply as if the facet were its own entity: co-mention edges with
   k = facet size, and reverse anchor edges only to mentioners in the anchor's own facet.
4. Implemented in the streaming builder only (`graph_stream.build_compact_graph`). The reference builder
   refuses `facets`.

## Pre-registered acceptance (`configs/bench/scale-facets-50.yaml`, same 200 questions and corpora)
Development data only (MuSiQue train and 2Wiki dev passages); the HippoRAG 2 test sets stay closed.
Two facet sizes are tried, 50 and 200; the better one at 140K is chosen and both are reported.

- **H1 — quality at scale:** at 140K, push GP@5 minus dense GP@5 ≥ +5.0 points (now +0.2).
- **H2 — no loss at small scale:** at 10K, push GP@5 ≥ 63.8 (now 64.8, minus 1 point).
- **H3 — cost:** median pushes per query at 140K ≤ 1.5× at 10K; peak RSS at 140K < 8 GB.
- **Diagnostic:** share of gold hop pairs linked at 140K (now 35%).

200 questions give a 95% interval of roughly ±4 points on GP@5; the decision is read with that in mind.

## Outcome (2026-11-03): **H1 failed** for both facet sizes (H2, H3 pass). Not adopted as a fix on its own.
Runs `runs/bench/scale-facets-50.log`, `runs/bench/scale-facets-200.log` (results JSON in `runs/bench/`).

| Passages | Dense GP@5 | Push, drop (0013) | Push, facets 50 | Push, facets 200 | Pushes p50, facets 50 | Peak RSS |
|---|---|---|---|---|---|---|
| 10K | 55.9 | 64.8 (+8.9) | 64.8 (+9.0) | 64.6 (+8.7) | 521 | 1.2 GB |
| 40K | 48.5 | 55.4 (+6.9) | 55.3 (+6.9) | 55.3 (+6.9) | 559 | 1.6 GB |
| 140K | 44.8 | 45.0 (+0.2) | 45.1 (+0.2) | 44.8 (−0.0) | 555 | 3.1 GB |

**Diagnosis** (`facets 50`, all 469 gold passages of the 200 questions; "missed" = not in dense top 5):

| | 10K | 140K |
|---|---|---|
| Found by dense top 5 | 251 | 199 |
| Missed, and no gold passage among the 3 seeds | 57 | 104 |
| Missed, a gold seed exists but has no edge to it | 64 | 112 |
| Missed, an edge from a gold seed exists | 97 | 54 |
| ... of which the walk brings it into the top 5 | 55 (57%) | 14 (26%) |

Three causes grow together with N, and facets address none of them enough:
1. **Seeds:** misses with no gold seed nearly double (57 → 104). The encoder (bge-small) loses the first hop
   among more distractors.
2. **Static pruning:** a seed's link to the next gold passage is missing more often (64 → 112). Every passage
   keeps 32 edges chosen at build time **without the question**, while its candidate set grows with N.
3. **Ranking competition:** when the edge exists, the walk ranks it into the top 5 less often (57% → 26%).

Facets keep the forward anchor edges and cost nothing extra; they stay available as an option.
The next step targets cause 2 first (decision 0015).
