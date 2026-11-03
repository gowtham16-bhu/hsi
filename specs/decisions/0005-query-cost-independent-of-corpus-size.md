# Decision 0005: query-time cost must not grow with corpus size

**Date:** 2026-11-02 · **Status:** accepted · **Affects:** C7, NFR-1, NFR-7

## Context
The owner asked whether the system can ingest and serve billions of passages. Two prototype
components touch the whole corpus on every query:
- global PPR iterates over every edge (≈ N · m) up to 50 times per question;
- exact flat vector search scans every vector.

## Decision
1. **Walks use local forward push** (Andersen, Chung & Lang 2006), `hsi_core.push`. Pushes per query
   ≤ 1/(α·ε), independent of N. Question-aware weights are computed only for visited passages; their
   vectors are fetched by id, so at scale only visited vectors are read.
2. **Vector search uses an approximate index** at scale (HNSW or IVF-PQ through the vector-store plugin).
   Exact search remains the default for small evaluation corpora.
3. Global PPR stays as the reference implementation for correctness tests.

## Ingestion cost model (for planning; to be measured)
- Encoding: linear and parallel; the same cost as any vector-RAG system. Dominates ingestion.
- Mention statistics: one map-reduce pass for corpus counts, then per-passage extraction (parallel).
- Graph: per-entity co-mention pairs bounded by D_max; edges ≤ N · m.
- No LLM calls at any stage.

## Acceptance
- Push results match global PPR on test corpora (top results identical at ε = 1e-7).
- Pushes per query stay flat as the corpus grows (benchmark on 10K → 100K → 1M passage graphs).
