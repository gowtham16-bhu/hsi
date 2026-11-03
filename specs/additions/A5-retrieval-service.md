# A5 — Addition: Retrieval service and MCP server (first usable product)

An application on top of the core engine. It makes HSI usable from any agent before any UI exists.

## 1. What it is
Two commands, both in the new package `hsi-service`:

| Command | Does |
|---|---|
| `hsi-index build --config <index.yaml>` | Pulls records from the configured connectors, chunks them, embeds passages, builds the graph with the streaming builder, and writes a versioned index |
| `hsi-serve --config <serve.yaml>` | Loads one index and answers retrieval requests over REST and MCP |

No LLM is called by either command. Neither opens any outbound connection except to the configured
connectors (at build) and to PostgreSQL.

## 2. Index layout
One directory per index version, plus rows in PostgreSQL (rule: PostgreSQL in every environment).

| Item | Where | Contents |
|---|---|---|
| `manifest.json` | index dir | index id, build config (and its hash), encoder name and dimension, counts, build times, code version |
| `vectors.npy`, `ids.json` | index dir | passage vectors (float32, row i = position i) and passage ids in position order |
| `graph.npz` | index dir | the `CompactGraph` |
| `transitions.json` | index dir | a copy of the hop model used |
| `hsi.passages` | PostgreSQL | text, title, document id and ACL per passage, keyed by (index id, passage id) |

The vector store plugin is rebuilt from `vectors.npy` at load (exact search for now; an ANN plugin later).

## 3. Requests
**REST** `POST /v1/retrieve`

```json
{"question": "...", "k": 5, "principals": ["user:alice", "group:appsec"]}
```

Response: `{"passages": [{"id", "doc_id", "title", "text", "score"}], "latency_ms", "index_id"}`.
`GET /v1/health` returns the index id and counts.

**MCP** (stdio and streamable HTTP): one tool, `retrieve(question, k)`, returning the same passages as text
blocks, each labeled as quoted source data. With MCP the principals come from the server's config
(the identity of the person or service running it), never from the tool arguments.

## 4. Permissions (C8; A3 S2)
`principals` must be set by a trusted caller: the company's auth gateway in front of the REST service, or
the server config for MCP. The service does not authenticate users itself.

A passage is **visible** when its ACL is public or shares a principal with the request. For each request:
1. Seeds are the top-`seeds` **visible** passages by vector score.
2. The walk never pushes into, or through, an invisible passage. Its out-edges are renormalized over
   visible neighbours only. A passage whose neighbours are all invisible is treated as dangling.
3. Only visible passages are returned.

**Guarantee (testable):** no invisible passage is returned, and no score depends on the vector or text of
an invisible passage at query time. **Stated limit:** graph structure is built from all documents, so
which edges exist and their weights (e.g. how common a name is) can reflect documents the requester cannot
see. Where that is unacceptable (HR), use one index per access group (strict mode).

## 5. Audit (A3 S8)
Every request is appended to `hsi.audit` in PostgreSQL: time, principals, question, returned passage ids,
index id, and a hash chained to the previous row. A tamper check recomputes the chain.

## 6. Acceptance
- **R1:** an end-to-end test builds an index from a JSONL connector, serves it, and retrieves over REST and
  MCP (in-process clients), with PostgreSQL.
- **R2:** permission property test: for random ACLs and principals, no returned passage is invisible, and
  results equal a run where invisible passages' vectors are replaced by random vectors.
- **R3:** the audit chain verifies, and a modified row is detected.
- **R4:** the service and index builder make no network connection other than PostgreSQL and the
  configured connectors (checked by running the tests with a socket guard).

## 7. Connector for this step
`connector:jsonl` (options `path`, `source_name`): documents from JSON Lines files (`id`, `title`, `text`, `acl`), so any system can be
exported and indexed without new code. Further connectors (MCP, Black Duck, Fortify) follow A2.

## 8. Status (2026-11-03): R1–R4 pass (tests/test_service.py)
- **R1:** JSONL → `hsi-index build` → service; REST (FastAPI test client) and MCP (in-memory client session)
  both return passages; the MCP tool list is exactly `retrieve`.
- **R2:** for five principal sets, no invisible passage is returned or used as a seed, and results (ids and
  scores) are identical when every invisible passage's vector is replaced by random noise.
- **R3:** the audit chain verifies; editing one row is detected at that row; restoring it verifies again.
- **R4:** the end-to-end test runs under a Python-level socket guard. **Limit:** PostgreSQL connects through
  libpq, outside that guard; a container-level egress test (A3 S1) is still to do.
- The JSONL connector passes the conformance kit (C0–C7) and rejects rows without a valid ACL.
