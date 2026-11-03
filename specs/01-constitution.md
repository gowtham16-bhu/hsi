# 01 — Constitution

These principles are non-negotiable. Every spec, task and pull request is checked against them. A change that breaks one is rejected, however useful it looks.

## P0. Thesis first
The core engine exists to test the thesis in `00-thesis.md`. A change to the core must serve one of its research questions, or be required to measure them. Additions (connectors, permissions, multimodal ingestion, the copilot, domain applications) build on the core and never change its semantics.

## P1. The core is the product
The algorithm (entity resolution, hub facets, chain building, composer, selection, chain scoring, completion, verification, incremental update) is defined **once**, in the core package. Plugins decide *how* data is fetched, stored or sent to a model. They never change *what* the algorithm does. Swapping a plugin may change quality only through that plugin's own output, never through a different code path in the core.

## P2. No data leaves the network
- Building the index makes **zero LLM calls**. The core package has no dependency on any LLM client, and a test enforces this.
- Network egress is **off by default**. Every outbound destination is an explicit allowlist entry in config.
- When an external LLM endpoint is configured, only the passages retrieved for that one question are sent, and every call is audited.

## P3. Permission-aware by construction
Every passage carries the access rules of its source. **A chain is visible to a user only if every passage in it is visible to that user.** This is enforced inside the core retrieval step, never in a plugin, and covered by property-based tests.

## P4. Evidence or silence
Every answer either comes with a verified evidence chain, where each hop links to its source, or is returned as a partial result that says what was and wasn't found. The system never states a claim it cannot support with retrieved evidence.

## P5. Measured, not assumed
Every change to the core reports the failure waterfall, split by 2, 3 and 4 hops. Continuous integration fails on a regression beyond the tolerance set in `04-evaluation-and-gates.md`.

## P6. Deterministic and reproducible
The same input documents, config and seed produce the same index and the same results. Every run records its config hash, code version, model versions and data hashes.

## P7. Production-grade, no mock features
No demo-only code paths, fake data in production code, or placeholder features. Test doubles are allowed only inside unit tests. Integration tests use real components.

## P8. Read-only by default
Connectors only read. Any action that writes to an external system (creating a ticket, commenting) requires explicit confirmation by the user, every time, and is audited.

## P9. Documents are data, never instructions
Text from any source is treated as evidence only. Instructions found inside documents are never followed. The reader prompt, the verifier and the action layer are built on this assumption.

## P10. Honest claims
Performance and novelty claims in docs, papers, blog posts and marketing require evidence: a benchmark run, a citation, or a reproducible script. Known limitations are documented next to the claim they limit.

## P11. Spec first
Every module has a spec with measurable acceptance criteria before code is written. When the implementation needs to differ, the spec is updated first, in the same pull request.

## P12. Source-agnostic core
The core knows only canonical records (documents, structured facts, access rules, deletions). It never contains code specific to a product, vendor or file format. Anything vendor-specific lives in a connector.

## P13. No hard-coded knowledge in code
Code contains algorithms only. Knowledge about a language, domain, dataset, vendor or model
(word lists, stop words, name patterns, ID prefixes, prompt or instruction strings, file layouts,
thresholds) is either **learned from the data** (for example casing statistics, collocation counts)
or **read from configuration or the model's own files**. Every tunable value appears in the run's
config file and is recorded in its report. A reviewer must be able to find every such value
without reading the code.
