# 05 — Gate 0 Tasks

Goal: answer RQ1 from `00-thesis.md` ("do precomputed chains beat online expansion at 3–4 hops?") with real numbers, at no LLM cost. Each task has one acceptance check. Tasks are ordered by dependency.

## Track A: engine foundation

| ID | Task | Depends on | Acceptance |
|---|---|---|---|
| T0.1 | Repository scaffold: `hsi-core`, `hsi-harness`, `hsi-plugins-*`; lint, type check, tests and import-linter in CI; `docker-compose.dev.yml` with PostgreSQL | none | CI green; `hsi-core` cannot import network or LLM libraries; integration tests run against real PostgreSQL |
| T0.2 | C1 canonical model and plugin protocols | T0.1 | Round-trip JSON tests for every type; protocols type-check |
| T0.3 | Plugin registry (entry points) and config loader with validation and hashing | T0.2 | Unknown plugin or invalid config fails at startup with a clear message |
| T0.4 | Encoder plugin (bge-small) with on-disk vector cache | T0.3 | Same text → same vector; cache hit on re-run |
| T0.5 | FAISS vector-store plugin (add, search, delete, persist) | T0.3 | Exact-search results match a brute-force reference |
| T0.6 | Dataset loaders: MuSiQue, 2Wiki, HotpotQA → canonical documents + gold chains | T0.2 | Counts match the dataset cards; every gold passage is found in the corpus |
| T0.7 | C2 ingestion (Wikipedia-style path: titles as anchors) | T0.6 | ≥ 95% of gold passages have an anchor entity |
| T0.8 | C4 graph: anchor edges (title matching with Aho–Corasick), co-mention edges from a mention-extractor plugin, hyperlinks where present | T0.7 | Edge counts reported; gold-chain coverage in graph reported per hop |
| T0.9 | C5 chain enumeration with caps and beams (no composer yet) | T0.8 | Candidate counts per level reported; gold-chain coverage per hop reported |
| T0.10 | Shortcut index with `text` keys, unified with passages, plus unpacking | T0.4, T0.5, T0.9 | Shortcut hits unpack to the correct ordered passages |

## Track B: baselines and evaluation

| ID | Task | Depends on | Acceptance |
|---|---|---|---|
| T0.11 | BM25 baseline | T0.6 | Recall on HotpotQA within a few points of published BM25 numbers |
| T0.12 | Dense baseline | T0.4, T0.5 | Same check against published dense numbers |
| T0.13 | Online graph expansion baseline (top passages + graph neighbors, scored by the encoder) | T0.8, T0.12 | Runs at fixed fan-out and depth; latency recorded |
| T0.14 | Waterfall evaluator and full-chain recall@k, per hop depth | T0.6 | Unit-tested on hand-built cases; bootstrap CIs |
| T0.15 | Gate 0 runner and report (table + plot, config and data hashes recorded) | T0.10–T0.14 | One command reproduces the report from a clean checkout |

## Track C: security harness (in parallel)

| ID | Task | Depends on | Acceptance |
|---|---|---|---|
| T0.16 | OSV connector (file dump mode) emitting documents and alias facts | T0.2 | Passes the connector conformance kit |
| T0.17 | Connector conformance kit, first version | T0.2 | Fails a deliberately broken test connector; passes the OSV connector |
| T0.18 | Harness question generator: 2–3 hop questions from OSV aliases, affected packages and dependency links, with gold chains | T0.16 | 500 questions generated; a 50-question sample checked by hand |

## Definition of done for Gate 0
- `T0.15` report exists, with per-hop full-chain recall@5 and @10 for dense, BM25, online expansion and shortcuts on MuSiQue and 2Wiki, with 95% CIs.
- The Gate 0 decision (pass or fail) is written down in `specs/decisions/gate0.md`, with the numbers.
