# A1 — Addition: Product (HSI Copilot)

An application built on the core engine. Not part of the thesis claims (`00-thesis.md` §9).

## 1. Problem
Engineers and security teams answer connected questions every day, and the answers are spread across many tools:

> "A scanner flagged this vulnerable component. Which of our services ship it, who owns them, did we already accept the risk, and is there a known workaround?"

Each tool holds one hop of the answer. Today a person follows the hops by hand across five or more tools. That takes hours, and it's error-prone during incidents.

Existing AI assistants either stay inside one tool's data, or need every document sent to a cloud service for indexing. Many companies forbid the second.

## 2. Product
**HSI Copilot**: a self-hosted copilot that connects to any company source, builds a multi-hop evidence index **without any LLM calls**, and answers connected questions with **verified, hop-by-hop evidence**, entirely inside the company network.

## 3. Users

| Persona | Typical question | What they need most |
|---|---|---|
| **AppSec engineer** | "Are we exposed to this CVE, and where?" | Complete, verified blast radius |
| **SRE / on-call** | "Which team owns the service that calls the failing API, and what fixed it last time?" | Speed during incidents |
| **Engineering / release manager** | "Which open defects block the release that ships this component?" | Links between work items, code and findings |
| **Developer** | "Why was this finding marked 'not an issue' before?" | History and decisions in context |
| **CISO / compliance** | "Show evidence that critical findings were handled within policy." | Audit trail and reports |

## 4. User stories
Each story has an acceptance test in the evaluation suite.

- **US-1 Exposure.** As an AppSec engineer, I ask whether a vulnerability affects us and get every affected service, with a verified chain for each. *Accept:* each chain links finding → component → repository or application → service → owner, every hop clickable.
- **US-2 Ownership.** As an SRE, I ask who owns a failing component and get the owning team and on-call contact with sources. *Accept:* answer cites the ownership source (service catalog, CODEOWNERS or equivalent).
- **US-3 History.** As a developer, I ask what was decided about a finding and get the ticket, comment or audit decision. *Accept:* the cited decision is the latest one by date.
- **US-4 Partial truth.** When evidence is incomplete, I see what was verified and what is missing. *Accept:* no unverified hop is presented as fact.
- **US-5 Watch.** I subscribe to a condition ("new critical finding on anything my team owns") and get alerts with chains. *Accept:* alert within the freshness target after the source changes.
- **US-6 Permissions.** I never see evidence from documents I can't open in the source tool. *Accept:* property tests in `additions/A3-security.md` pass.
- **US-7 New source.** As a platform engineer, I add a new source by writing a declarative connector file, with no core changes. *Accept:* the connector passes the conformance kit.
- **US-8 Air-gapped.** As an admin, I run the whole system with no internet access. *Accept:* full test suite passes with egress blocked.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Ingest any source through the connector contract (`A2`): documents, structured facts, access rules, deletions |
| FR-2 | Build the multi-hop index with zero LLM calls (`03`, C2–C7) |
| FR-3 | Resolve entity aliases and split same-name entities (`03`, C3) |
| FR-4 | Answer with one retrieval pass and at most one LLM call in FAST and QUALITY modes, at most two in MAX |
| FR-5 | Return every answer with a verified chain, or a partial result (`03`, C9) |
| FR-6 | Enforce source permissions on every chain (`03`, C8; `A3`) |
| FR-7 | Apply source changes incrementally (`03`, C10) |
| FR-8 | Watch subscriptions with alerts on new chains |
| FR-9 | Surfaces: web UI, chat (Slack or Teams through plugins), REST API, IDE plugin later |
| FR-10 | Write actions only with explicit user confirmation, audited |
| FR-11 | Failure-waterfall reporting on any labeled question set |

## 6. Non-functional requirements
All numbers are **targets** until measured at the gates.

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Retrieval latency, excluding the LLM call | p95 ≤ 300 ms at 1M passages, QUALITY mode |
| NFR-2 | Freshness from source change to answerable | ≤ 60 s for push sources, ≤ one sync interval for pull sources |
| NFR-3 | Index-time LLM calls | exactly 0 |
| NFR-4 | Default network egress | none |
| NFR-5 | Error rate among answered questions | ≤ 5% at the chosen confidence threshold, measured on the dev set |
| NFR-6 | Deployment | single server with Docker Compose; Kubernetes with Helm; air-gapped |
| NFR-7 | Scale | 10M passages on one node with an external vector store |
| NFR-8 | Reproducibility | identical results for identical input, config and seed |

## 7. Out of scope (for now)
- Replacing scanners or matching dependency versions. Existing tools do that; HSI connects their results.
- Multi-tenant SaaS hosting.
- Autonomous write actions without user confirmation.
- Languages other than English (later).

## 8. Success measures
- **Research:** the gate criteria in `04-evaluation-and-gates.md`.
- **Product pilot:** on one real team, median time to answer an exposure question drops from hours to minutes, with zero permission violations.
