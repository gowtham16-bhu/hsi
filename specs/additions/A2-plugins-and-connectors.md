# A2 — Addition: Plugins and Connectors

How the core is attached to any source and any model. Plugins never change core semantics (constitution P1).

## 1. Plugin families

| Family | Interface | First implementation | Later options |
|---|---|---|---|
| Connector | `Connector` (§2) | Declarative REST, file system, SBOM (CycloneDX / SPDX), SARIF, OSV | anything that fits the contract |
| Parser | `Parser.parse(bytes, mime) -> StructuredText` | Markdown, HTML, PDF, DOCX, email | OCR, slides, spreadsheets |
| Mention extractor | `MentionExtractor.extract(text) -> list[Mention]` | spaCy NER + configured regex families | domain NER models |
| Encoder | `Encoder.encode(texts, kind) -> ndarray` | bge-small-en-v1.5 | any embedding model |
| Reranker | `Reranker.score(q, texts) -> list[float]` | bge-reranker | any cross-encoder |
| NLI | `NLI.entail(premise, hypothesis) -> float` | small open NLI model | any |
| Vector store | `VectorStore.add / search / delete` | FAISS in-process | pgvector, Qdrant, Milvus, OpenSearch |
| Reader (LLM) | `Reader.complete(prompt, schema) -> json` | on-prem open model server; company-approved API endpoint | any |
| Redactor | `Redactor.scan(text) -> list[Span]` | pattern + entropy secret detection | DLP integrations |
| Identity | `Identity.principals(user) -> set[str]` | OIDC / SAML groups | LDAP |
| Surface | `Surface.serve(app)` | web UI, REST API | Slack, Teams, IDE |

Plugins are registered through the `hsi.plugins` entry-point group and selected by name in config. Every plugin declares a version and a capabilities object. The core validates both at startup.

## 2. The connector contract
A connector can wrap **anything**: a product API, a database, a file share, a chat tool, an event stream, a mailbox, or a folder of exported reports. The core sees only canonical records.

```python
class Connector(Protocol):
    name: str
    version: str

    def capabilities(self) -> ConnectorCaps
        # incremental: bool, push: bool, emits: {documents, facts, acls, tombstones},
        # identity_mapping: bool, rate_limit: RateLimit

    def streams(self) -> list[StreamSpec]
        # what this source contains, e.g. "findings", "work_items", "pages"

    def sync(self, stream: StreamSpec, cursor: Cursor | None) -> Iterator[Record]
        # Record = SourceDocument | StructuredFact | AclUpdate | Tombstone | CursorCheckpoint

    def principals(self) -> Iterator[PrincipalMapping]
        # source user or group → SSO identity, when the source has its own users

    def on_event(self, payload: bytes) -> Iterator[Record]
        # push mode: webhooks or stream messages, optional
```

### Rules every connector must follow
1. **Read-only.** Uses read-only credentials. No write calls.
2. **Access rules are explicit.** Every record carries an ACL, or is explicitly marked `public`. Records without either are rejected.
3. **Identifiers.** Emits `EntityRef`s with standard schemes where they exist (`purl`, `cve`, `ghsa`, `cwe`, `url`, `email`), and `src:<connector>` for native IDs.
4. **Idempotent and resumable.** Re-running from a cursor produces no duplicates. Deletions are emitted as tombstones.
5. **Polite.** Respects the source's rate limits and backs off on errors.
6. **No vendor logic leaks into the core** (P12). Mapping from vendor fields to canonical records happens inside the connector.

### Conformance kit
Every connector must pass `hsi-connector-kit` before it can be enabled:
- idempotent re-sync, resume from cursor, tombstones applied;
- every record has an ACL or `public`;
- rate-limit compliance under a simulated throttling source;
- no write calls (recorded against a strict HTTP proxy);
- schema validation of all records.

## 3. Ways to build a connector

### A. Declarative (no code)
For REST, GraphQL and SQL sources, which covers most tools. One YAML file:
```yaml
connector: rest
name: example-findings
base_url: https://tool.internal.example/api
auth: { type: bearer, token_env: TOOL_TOKEN }
rate_limit: { requests_per_second: 5 }
streams:
  - name: findings
    request: { path: /findings, params: { updated_since: "{{cursor}}" } }
    pagination: { type: cursor, next: "$.next" }
    records: "$.items[*]"
    cursor_field: "$.updated_at"
    document:
      id: "$.id"
      title: "$.title"
      body: ["$.description", "$.recommendation", "$.comments[*].text"]
      acl: { principals: "$.project.groups[*]" }
    facts:
      - subject: { scheme: "src:example", id: "$.project.id" }
        predicate: has_finding
        object:  { scheme: "src:example", id: "$.id" }
      - subject: { scheme: "src:example", id: "$.id" }
        predicate: weakness
        object:  { scheme: cwe, id: "$.cwe" }
```

### B. Standard formats (one adapter, many tools)
Many tools already export the same formats, so one adapter covers all of them:

| Format | Covers | Becomes |
|---|---|---|
| CycloneDX, SPDX (SBOM) | composition-analysis tools, build tools | facts: application uses component (purl) |
| SARIF | static analysis tools | documents (finding text) + facts: finding in file, weakness (CWE) |
| CSAF, OpenVEX | vendor advisories, exploitability statements | documents + facts: affects, not affected, fixed in |
| OSV | open-source advisories | documents + alias facts (CVE ↔ GHSA ↔ others) |
| CODEOWNERS, service catalog files | ownership | facts: path or service owned by team |
| Markdown, HTML, PDF, DOCX, email | documentation, runbooks, postmortems | documents |

### C. Code connectors (Python SDK)
For complex sources: custom authentication, unusual pagination, binary formats, or permission models that need several API calls. They implement the `Connector` protocol directly and must pass the same conformance kit.

### D. MCP connectors (any Model Context Protocol server)
One generic adapter, `hsi-connector-mcp`, turns any MCP server into a source, local (stdio) or remote (HTTP).
A company that already exposes Jira, Confluence, GitHub, a database or an in-house tool through MCP needs
no new connector code. Each instance is configured declaratively:

| MCP feature | Becomes | Config says |
|---|---|---|
| `resources/list` + `resources/read` | documents (one per resource URI) | which URI patterns to ingest; title and section fields |
| a read-only tool with arguments (e.g. "list findings since T") | documents and/or facts | tool name, argument template, how result fields map to documents, entities and facts |
| `resources/subscribe` notifications | incremental updates (push) | whether to subscribe or poll |

- **Access rules:** MCP has no permission model of its own, so the config must declare where each record's
  ACL comes from: a field in the returned data, a fixed group, or `public`. Records without one are rejected
  (rule 2 above).
- **Incremental sync:** the cursor is the set of (URI, content hash) pairs; changed hashes are re-ingested, and
  URIs that disappear become tombstones.
- **Read-only:** the adapter calls only the tools named in its config, and never tools that write.
- It must pass the same conformance kit as every other connector.

The reverse direction is a product surface, not a connector: HSI itself can be **served as an MCP server**
(a `retrieve` tool returning passages and their evidence chain), so any agent that speaks MCP can use it.

### E. Public and global data
Public sources (OSV, NVD, GitHub advisories, vendor advisories, Wikipedia, standards documents, public
documentation) are ordinary connectors whose records are marked `public`. They enter the same graph as
private data, so a public CVE page becomes the anchor that links a company's own scanner findings, tickets
and services.

- **Direction of data:** public data flows *in*; connectors fetch whole feeds or pages on a schedule, never
  per-question lookups, so no question text or private data is sent out.
- **Air-gapped sites:** feeds are mirrored to an internal location and ingested from there.
- **Provenance:** every public record keeps its source URL and fetch time, shown in the evidence chain.

## 4. Examples (not special cases)
These show how real tools map to the contract. None of them gets special treatment in the core. Exact API details must be confirmed against each company's own instance and documentation.

| Example source | Likely build route | Documents | Facts | ACL source |
|---|---|---|---|---|
| Composition-analysis tool (e.g. Black Duck) | declarative REST or CycloneDX export | vulnerability and remediation text | application version uses component; component has vulnerability; vulnerability aliases | project permissions |
| Static-analysis platform (e.g. OpenText Fortify) | declarative REST or SARIF export | issue descriptions, audit comments | application version has issue; issue in file; issue weakness (CWE) | application permissions |
| Value-stream / ALM platform (e.g. OpenText ValueEdge) | declarative REST | work item and defect text, comments | defect linked to work item; work item in release; release ships application | workspace permissions |
| Ticketing (Jira, ServiceNow) | declarative REST | ticket text and comments | ticket references entity IDs found in text | project permissions |
| Wiki (Confluence, SharePoint) | code connector (permission model) | pages | page links | space or site permissions |
| Git hosting | code connector | READMEs, release notes, PR text | repository declares component; CODEOWNERS ownership | repository permissions |
| Service catalog / CMDB | declarative REST or files | service descriptions | service deploys application; service owned by team | catalog permissions |
| Chat (Slack, Teams) | code connector | incident channel threads | thread references entity IDs | channel membership |
| Event streams (Kafka) | push mode | event payloads | depends on topic schema | topic ACLs |
| Databases | declarative SQL | rows rendered as documents | rows as facts | database grants mapped to groups |

## 5. Versioning and compatibility
- Plugin interfaces are versioned with semantic versioning. A breaking change bumps the major version and is announced in the changelog.
- The core supports the current and previous major version of each interface.
- Records include the connector name and version, so any index can be traced to the connectors that built it.
