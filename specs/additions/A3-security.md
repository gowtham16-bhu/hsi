# A3 — Addition: Security

Required for any deployment of the core. Not part of the thesis claims.

## 1. Assets
Source documents and facts, access rules, connector credentials, the index (vectors can leak content), audit logs, and user questions.

## 2. Threat model and required controls

| ID | Threat | Required controls | Test |
|---|---|---|---|
| S1 | Data sent outside the network | Egress off by default with an allowlist; zero LLM calls at index time; only retrieved passages go to an external reader; every outbound call audited | Full suite passes with egress blocked; an egress proxy logs zero unexpected destinations |
| S2 | Permission bypass through chains or shortcuts | C8 rules enforced in core; completion only into visible passages; strict mode for ACL-homogeneous shortcuts | Property-based tests (C8 acceptance) |
| S3 | Prompt injection inside documents | Passages wrapped and labeled as quoted data; reader output must match the JSON schema; no actions from reader output without user confirmation; verification ignores instructions in text | Injection suite: planted instructions in documents never change the format, the answer source, or trigger an action |
| S4 | Secrets inside documents | Redactor plugin at ingest (patterns + entropy); detected secrets are masked before indexing and never stored in vectors | Seeded-secret corpus: zero secrets retrievable |
| S5 | Connector credential theft | Read-only tokens; secrets only from environment or a vault; never logged | Log scan for credential patterns |
| S6 | Graph poisoning (a planted document creates false aliases or links) | Source trust weights; low-trust sources create only soft aliases; alias provenance visible in answers | Poisoning fixture: a low-trust document cannot cause a hard merge |
| S7 | Inference from vectors | Vector store encrypted at rest; same access controls as text; strict mode where required | Configuration review |
| S8 | Audit tampering | Append-only audit log with hash chaining | Tamper test detects a modified entry |
| S9 | Over-reliance on wrong answers | Confidence shown; partial results instead of guesses; error rate among answered questions measured (NFR-5) | Risk–coverage report on every release |

## 3. Identity and access
- Users authenticate through the company SSO (OIDC or SAML) using an identity plugin.
- `principals(user)` = the user ID + SSO groups + source-specific identities mapped by connectors.
- Admin roles: connector management, config, audit access. These are separate from user roles.

## 4. Audit
Every question, answer status, cited sources, outbound LLM call (destination and size, never content unless configured), write action and admin change is recorded with a timestamp, user and config hash.

## 5. Data lifecycle
- Deletions at the source arrive as tombstones and remove passages, vectors, facts and dependent shortcuts within NFR-2.
- Retention periods are configured per connector.
- A full purge of one connector's data is one admin command and is verified by a test.
