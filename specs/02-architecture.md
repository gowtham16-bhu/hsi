# 02 — Architecture

## 1. Layers

```
 ┌────────────────────────────── SURFACES (plugins) ──────────────────────────────┐
 │  Web UI · Slack / Teams · REST API · IDE plugin · Watch notifier               │
 └──────────────────────────────────────┬─────────────────────────────────────────┘
 ┌──────────────────────────────── SERVER ───────────────────────────────────────┐
 │  API (FastAPI) · auth (SSO plugin) · RBAC · audit log · job scheduler          │
 └──────────────────────────────────────┬─────────────────────────────────────────┘
 ┌──────────────────────────── CORE ENGINE (fixed) ───────────────────────────────┐
 │  C1 model · C2 ingestion · C3 entity resolution · C4 graph + hub facets ·      │
 │  C5 chains + composer · C6 selection · C7 index + retrieval · C8 permissions · │
 │  C9 reader contract + verification · C10 incremental update · waterfall eval   │
 └──────┬───────────────┬────────────────┬───────────────┬────────────────────────┘
        │               │                │               │     stable interfaces
 ┌──────▼─────┐ ┌───────▼──────┐ ┌───────▼──────┐ ┌──────▼────────┐
 │ Connectors │ │ Encoders,    │ │ Vector and   │ │ LLM readers,  │   PLUGINS
 │ (any       │ │ rerankers,   │ │ metadata     │ │ NLI, redactor,│
 │  source)   │ │ NER          │ │ stores       │ │ chunk parsers │
 └────────────┘ └──────────────┘ └──────────────┘ └───────────────┘
```

**Rule:** arrows only point down into plugins through the interfaces in `A2`. The core imports no plugin package.

## 2. Packages

| Package | Contents | Depends on |
|---|---|---|
| `hsi-core` | Canonical model, algorithms C1–C10, plugin interfaces, evaluation | numpy, pydantic; no network or LLM clients |
| `hsi-server` | API, auth, RBAC, audit, scheduler, Watch | `hsi-core`, FastAPI |
| `hsi-plugins-*` | One package per plugin family, e.g. `hsi-plugins-faiss`, `hsi-plugins-connectors-rest` | `hsi-core` |
| `hsi-harness` | Benchmark loaders, question generators, runners, reports | `hsi-core` |

Plugins are discovered through Python entry points (group `hsi.plugins`) and selected in config.

## 3. Data model (canonical)

```python
EntityRef      = (scheme: str, id: str)        # e.g. ("purl", "pkg:maven/org.example/lib@1.2"),
                                               # ("cve", "CVE-2021-44228"), ("cwe", "CWE-79"),
                                               # ("service", "payment-api"), ("src:<connector>", "<native id>")
Acl            = { principals: set[str], public: bool }      # users and groups, mapped to SSO identities

SourceDocument = id, connector, uri, title?, sections[], body, acl, updated_at, metadata, trust
StructuredFact = subject: EntityRef, predicate: str, object: EntityRef | literal,
                 confidence ∈ [0,1], provenance (connector, uri, updated_at), acl
Tombstone      = id of a document or fact that was deleted at the source

Passage        = id, doc_id, section_path, text, hash, acl, valid_from
Mention        = passage_id, span, surface, type, entity_id?, prob, context_vector
Entity         = id, canonical_name, type, refs: set[EntityRef], facet_of?, vector
AliasEdge      = entity_a, entity_b, alpha, beta, hard: bool
Edge           = src (passage | entity), dst (passage | entity), type, weight, origin (text | structured)
Chain          = passages[ℓ], bridges[ℓ−1], log_prob, level
Shortcut       = id, chain, key_vector, benefit, version
Answer         = text?, chains[], confidence, status (answered | partial), audit_id
```

Canonical identifiers (`EntityRef`) are how sources join each other. When two sources name the same thing with the same ref, they join with confidence 1.0. When they don't, entity resolution (C3) joins them with a probability.

## 4. Storage

| Store | Default | Alternatives (plugins) |
|---|---|---|
| Metadata, graph, ACLs, aliases, audit | PostgreSQL in **every** environment, including development (Docker) | none |
| Vectors (passages + shortcuts) | FAISS in-process | pgvector, Qdrant, Milvus, OpenSearch |
| Passage text and versions | PostgreSQL | object storage for large blobs |
| Job queue | PostgreSQL table with `SKIP LOCKED` | none needed |

No graph database is required. Edges are relational tables with indexes on `src` and `dst`.

**Environment parity.** Development uses the same PostgreSQL major version as production, started with `docker compose -f docker-compose.dev.yml up`. There is no SQLite or in-memory substitute. Any other middleware (for example Redis for caching, or Kafka for push connectors) is added only when a spec requires it, and then it runs as a real service in development too. The core never knows which middleware exists: it talks only to the interfaces in `A2`.

## 5. Data flows

**Ingest (pull or push)**
```
connector.sync(cursor) → records → redactor (secrets) → C2 ingestion → C3 resolution
→ C4 graph + facets → C5 candidate chains + keys → C6 selection → C7 index write → cursor saved
```

**Query**
```
user + question → principals(user) → C7 retrieve (dense + BM25) → unpack → C8 permission filter
→ C7 completion → rerank → C9 assemble → LLM reader (1 call) → C9 verify → answer | partial → audit
```

**Update**
```
changed records → C10 diff → affected radius → local rebuild of C3–C7 → atomic swap → Watch evaluation
```

## 6. Deployment

| Mode | How | Notes |
|---|---|---|
| Development | `pip install` + PostgreSQL in Docker + FAISS | same database engine as production |
| Pilot | Docker Compose: server, worker, Postgres | one machine, optional GPU |
| Production | Helm chart: server, workers, Postgres, optional vector DB | single-tenant, inside the company network |
| Air-gapped | Production + on-prem LLM + internal advisory mirror | egress fully blocked |

GPU is optional. Encoding and the composer run on CPU at lower speed.

## 7. Configuration
One YAML file selects plugins and sets core parameters. Example: `configs/hsi.example.yaml`. Core parameters have defaults defined in `03-core-engine.md`. Config is validated at startup, and its hash is recorded with every index and every evaluation run.
