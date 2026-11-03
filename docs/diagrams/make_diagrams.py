"""Generate the HSI architecture diagrams as standalone SVG files.

Run:  python docs/diagrams/make_diagrams.py
Output: one .svg per diagram in docs/diagrams/
"""
from __future__ import annotations

import html
import os
import textwrap

OUT = os.path.dirname(os.path.abspath(__file__))

# fill, stroke, title colour
KIND = {
    "core":   ("#FBF0D9", "#C98A12", "#6E4A05"),
    "plugin": ("#E4EDF7", "#2C5C8F", "#1E4470"),
    "store":  ("#EEF0F3", "#6B7684", "#2B3440"),
    "user":   ("#E2F2EA", "#2D7A55", "#1D553B"),
    "source": ("#FFFFFF", "#9AA4B1", "#16202B"),
    "warn":   ("#F7E4E1", "#A9433A", "#7A2C25"),
}
INK, MUTED, LINE = "#16202B", "#566272", "#7D8796"
SANS = "'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def wrap(text: str, width: float, size: float) -> list[str]:
    chars = max(8, int((width - 24) / (size * 0.56)))
    return textwrap.wrap(text, chars) or [""]


class Canvas:
    def __init__(self, w: int, h: int, title: str, subtitle: str = ""):
        self.w, self.h = w, h
        self.items: list[str] = []
        self.text(30, 42, title, 24, INK, bold=True)
        if subtitle:
            self.text(30, 68, subtitle, 14, MUTED)

    # primitives -----------------------------------------------------------
    def text(self, x, y, s, size=13, fill=INK, bold=False, mono=False, anchor="start"):
        fam = MONO if mono else SANS
        weight = "700" if bold else "400"
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" xml:space="preserve" style="white-space:pre">{esc(s)}</text>'
        )

    def rect(self, x, y, w, h, fill, stroke, rx=10, dash=False, sw=1.6):
        d = ' stroke-dasharray="7 5"' if dash else ""
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def arrow(self, pts, color=LINE, dash=False, label=None, label_at=None):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        da = ' stroke-dasharray="5 4"' if dash else ""
        self.items.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{da} marker-end="url(#ah)"/>'
        )
        if label:
            lx, ly = label_at or pts[0]
            self.text(lx, ly, label, 11.5, MUTED, mono=True)

    def box(self, x, y, w, h, title, lines=(), kind="core", tag=None, size=12.5):
        fill, stroke, tcol = KIND[kind]
        self.rect(x, y, w, h, fill, stroke)
        ty = y + 26
        tlines = wrap(title, w, 14.5)
        for t in tlines:
            self.text(x + 12, ty, t, 14.5, tcol, bold=True)
            ty += 18
        ty += 2
        for line in lines:
            for part in wrap(line, w, size):
                self.text(x + 12, ty, part, size, MUTED)
                ty += size + 4.5
        if tag:
            self.text(x + w - 10, y + 18, tag, 11, stroke, mono=True, anchor="end")

    @staticmethod
    def box_height(w, title, lines, size=12.5):
        n_t = len(wrap(title, w, 14.5))
        n_l = sum(len(wrap(l, w, size)) for l in lines)
        return 26 + 18 * (n_t - 1) + 8 + n_l * (size + 4.5) + 10

    def lane(self, x, y, label):
        self.text(x, y, label.upper(), 11.5, MUTED, mono=True)

    def formula(self, x, y, w, lines):
        h = 22 + 20 * len(lines)
        self.rect(x, y, w, h, "#FFFFFF", "#C98A12", rx=6)
        self.rect(x, y, 5, h, "#C98A12", "#C98A12", rx=2)
        ty = y + 26
        for line in lines:
            self.text(x + 18, ty, line, 13, INK, mono=True)
            ty += 20
        return h

    def legend(self, x, y, kinds):
        names = {"core": "core engine (fixed)", "plugin": "plugin (swappable, domain-owned)",
                 "store": "storage / middleware", "user": "output / user", "source": "input / source",
                 "warn": "fallback / partial"}
        for k in kinds:
            fill, stroke, _ = KIND[k]
            self.rect(x, y - 12, 16, 16, fill, stroke, rx=3)
            self.text(x + 24, y + 1, names[k], 12.5, MUTED)
            x += 30 + len(names[k]) * 7.2

    def row(self, x0, y, total_w, items, gap=34):
        n = len(items)
        w = (total_w - gap * (n - 1)) / n
        h = max(self.box_height(w, it["t"], it.get("l", ())) for it in items)
        geo = []
        for i, it in enumerate(items):
            x = x0 + i * (w + gap)
            self.box(x, y, w, h, it["t"], it.get("l", ()), it.get("k", "core"), it.get("tag"))
            geo.append((x, y, w, h))
            if i:
                px, _py, pw, _ph = geo[i - 1]
                self.arrow([(px + pw + 2, y + h / 2), (x - 4, y + h / 2)])
        return geo

    def snake(self, a, b):
        """Connect the last box of one row to the first box of the next row."""
        ax, ay, aw, ah = a
        bx, by, bw, _bh = b
        mid = (ay + ah + by) / 2
        self.arrow([(ax + aw / 2, ay + ah + 2), (ax + aw / 2, mid), (bx + bw / 2, mid), (bx + bw / 2, by - 4)])

    def save(self, name):
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{LINE}"/></marker></defs>'
            f'<rect width="100%" height="100%" fill="#F8F9FB"/>'
            + "".join(self.items)
            + "</svg>"
        )
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)


# --------------------------------------------------------------------------
# 1. Overall picture
# --------------------------------------------------------------------------
def overall():
    c = Canvas(1500, 880, "HSI: overall picture",
               "Any source in, verified multi-hop answers out. Everything runs inside the company network.")
    c.rect(15, 92, 1470, 760, "none", "#2D7A55", rx=14, dash=True)
    c.text(32, 116, "COMPANY NETWORK · NO DATA LEAVES · ZERO LLM CALLS AT INDEX TIME", 12, "#2D7A55", mono=True)

    c.lane(40, 146, "Any source")
    sources = [
        ("Security tools", ["SCA, SAST, DAST, secrets"]),
        ("ALM and tickets", ["work items, defects, incidents"]),
        ("Wikis and runbooks", ["pages, postmortems"]),
        ("Code hosting", ["READMEs, SBOMs, CODEOWNERS"]),
        ("Service catalog / CMDB", ["services, owners, deployments"]),
        ("Chat, email, streams", ["incident threads, events"]),
        ("Public feeds (mirrored)", ["OSV, NVD, vendor advisories"]),
    ]
    y = 160
    src_mid = []
    for t, l in sources:
        c.box(40, y, 230, 72, t, l, "source")
        src_mid.append(y + 36)
        y += 86

    c.box(320, 160, 240, 588, "Connector plugins",
          ["One contract for any source.", "",
           "YAML: REST, GraphQL, SQL", "",
           "Standard formats: CycloneDX, SPDX, SARIF, CSAF, OpenVEX, OSV", "",
           "Python SDK for complex sources", "",
           "Read-only credentials", "Every record carries an ACL", "Resumable cursors, tombstones",
           "", "Must pass the conformance kit"], "plugin")
    for ym in src_mid:
        c.arrow([(272, ym), (316, ym)])

    c.box(620, 160, 460, 500, "HSI core engine", ["fixed and domain-agnostic"], "core")
    mods = ["C1  Canonical model + interfaces", "C2  Ingestion", "C3  Entity resolution",
            "C4  Graph + hub facets", "C5  Chains + composer", "C6  Budgeted selection",
            "C7  Index + retrieval", "C8  Permissions", "C9  Reader contract + verification",
            "C10 Incremental update"]
    my = 222
    for m in mods:
        c.rect(640, my, 420, 34, "#FFFFFF", "#C98A12", rx=6, sw=1.2)
        c.text(654, my + 22, m, 13.5, INK, mono=True)
        my += 42
    c.arrow([(562, 410), (616, 410)])
    c.text(566, 400, "records", 11, MUTED, mono=True)

    stores = [("PostgreSQL", ["metadata, graph, ACLs, aliases, audit, job queue"]),
              ("Vector store", ["FAISS in-process, or pgvector, Qdrant, Milvus"]),
              ("Middleware", ["added only when a spec needs it, e.g. Kafka for push"])]
    sx = 620
    for t, l in stores:
        c.box(sx, 700, 145, 120, t, l, "store")
        c.arrow([(sx + 72, 662), (sx + 72, 696)])
        sx += 157

    c.box(1130, 160, 320, 130, "Model plugins",
          ["encoder, reranker, NLI, NER, parsers, redactor"], "plugin")
    c.box(1130, 310, 320, 120, "Reader LLM plugin",
          ["on-prem model server, or the company-approved endpoint", "1 call per question"], "plugin")
    c.box(1130, 450, 320, 120, "Surfaces",
          ["web UI, REST API, Slack / Teams, IDE, Watch alerts"], "plugin")
    c.box(1130, 600, 320, 90, "Users (SSO)",
          ["see only what they can open in the source"], "user")
    c.arrow([(1082, 225), (1126, 225)])
    c.arrow([(1082, 370), (1126, 370)])
    c.arrow([(1082, 510), (1126, 510)])
    c.arrow([(1290, 572), (1290, 596)])
    c.legend(40, 872 - 30, ["core", "plugin", "store", "source", "user"])
    c.save("01-overall.svg")


# --------------------------------------------------------------------------
# 2. Core engine
# --------------------------------------------------------------------------
def core_engine():
    c = Canvas(1500, 900, "HSI core engine",
               "Ten modules. They speak only canonical types (C1) and never know which domain, source or vendor they serve.")
    c.box(40, 92, 1420, 50, "Failure-waterfall counters on every module:  ① link missed  ② pruned  ③ not selected  ④ not retrieved  ⑤ lost in assembly  ⑥ reader wrong", (), "store")

    c.lane(40, 176, "Index time · no LLM")
    c.row(40, 188, 1420, [
        {"t": "C2 Ingestion", "l": ["structure-first chunks", "content-defined splits", "topic score instead of titles"]},
        {"t": "C3 Entity resolution", "l": ["merge aliases", "split same names", "soft alias edges"]},
        {"t": "C4 Graph + hub facets", "l": ["typed edges", "hubs split by relation"]},
        {"t": "C5 Chains + composer", "l": ["2-, 3-, 4-hop candidates", "one key per chain"]},
        {"t": "C6 Budgeted selection", "l": ["facility location", "≤ βN shortcuts", "verified-use promotion"]},
        {"t": "C7 Index write", "l": ["passages + shortcuts", "side table: chain → passages"]},
    ])

    c.lane(40, 356, "Query time · 1 LLM call")
    c.row(40, 368, 1420, [
        {"t": "C7 Retrieve", "l": ["dense + BM25", "fusion, unpack chains"]},
        {"t": "C8 Permissions", "l": ["chain visible only if every passage is visible"]},
        {"t": "C7 Complete + rerank", "l": ["composer-scored beam", "chain score S, cross-encoder"]},
        {"t": "C9 Assemble + read", "l": ["hop-ordered context", "reader plugin, JSON"]},
        {"t": "C9 Verify", "l": ["bridges vs graph", "quote, type, NLI, vote"]},
        {"t": "Answer or partial", "l": ["evidence chain", "calibrated confidence"], "k": "user"},
    ])

    c.lane(40, 536, "Change time · seconds")
    c.row(40, 548, 1420, [
        {"t": "C10 Diff", "l": ["content-defined re-chunk", "hash diff Δ"]},
        {"t": "Affected radius", "l": ["entities touched", "chains through Δ"]},
        {"t": "Local rebuild", "l": ["C3–C6 inside L hops of Δ"]},
        {"t": "Atomic swap", "l": ["new rows in", "old rows tombstoned"]},
        {"t": "Watch", "l": ["new chains → alerts"], "k": "user"},
    ])

    c.box(40, 700, 1420, 64, "C1 Canonical model + plugin interfaces",
          ["Every module above speaks only these types. No vendor code. No network or LLM imports."], "core")
    c.lane(40, 798, "Reached only through C1 interfaces")
    chips = ["Connectors", "Parsers", "NER", "Encoder", "Vector store", "Reranker", "NLI", "Reader LLM", "Identity"]
    x = 40
    for ch in chips:
        w = 30 + len(ch) * 8.4
        c.rect(x, 810, w, 32, KIND["plugin"][0], KIND["plugin"][1], rx=16)
        c.text(x + w / 2, 831, ch, 13, KIND["plugin"][2], anchor="middle")
        x += w + 12
    c.legend(40, 880, ["core", "plugin", "store", "user"])
    c.save("02-core-engine.svg")


# --------------------------------------------------------------------------
# 3. Module diagrams C1..C10
# --------------------------------------------------------------------------
def module(fname, title, subtitle, rows, formula, height=None):
    W = 1400
    est = 100
    c0 = Canvas(W, 10, "", "")
    for r in rows:
        n = len(r)
        w = (1320 - 34 * (n - 1)) / n
        est += max(c0.box_height(w, it["t"], it.get("l", ())) for it in r) + 60
    est += 22 + 20 * len(formula) + 90
    c = Canvas(W, height or int(est), title, subtitle)
    y = 100
    prev = None
    for r in rows:
        geo = c.row(40, y, 1320, r)
        if prev:
            c.snake(prev[-1], geo[0])
        prev = geo
        y = geo[0][1] + geo[0][3] + 60
    fh = c.formula(40, y - 20, 1320, formula)
    c.legend(40, y - 20 + fh + 40, ["source", "core", "plugin", "user", "warn"])
    c.save(fname)


def modules():
    module("c01-canonical-model.svg", "C1 · Canonical model and plugin interfaces",
           "The only language the core speaks. Sources and plugins are translated into it at the edge.",
           [[{"t": "Connector records", "l": ["SourceDocument", "StructuredFact", "AclUpdate", "Tombstone"], "k": "source"},
             {"t": "Validation", "l": ["schema check", "ACL or public required", "EntityRef schemes"]},
             {"t": "Internal types", "l": ["Passage, Mention", "Entity, AliasEdge, Edge"]},
             {"t": "Retrieval types", "l": ["Chain, Shortcut", "Answer"], "k": "user"}],
            [{"t": "Connector", "l": ["protocol"], "k": "plugin"}, {"t": "Parser", "l": ["protocol"], "k": "plugin"},
             {"t": "NER", "l": ["protocol"], "k": "plugin"}, {"t": "Encoder", "l": ["protocol"], "k": "plugin"},
             {"t": "Vector store", "l": ["protocol"], "k": "plugin"}, {"t": "Reranker, NLI", "l": ["protocol"], "k": "plugin"},
             {"t": "Reader", "l": ["protocol"], "k": "plugin"}, {"t": "Identity", "l": ["protocol"], "k": "plugin"}]],
           ["EntityRef = (scheme, id)      e.g. (purl, pkg:maven/org.example/lib@1.2), (cve, CVE-2021-44228)",
            "Acl       = { principals, public }",
            "Edge      = (src, dst, type, weight, origin: text | structured)",
            "Rule      : hsi-core imports no network, LLM or vendor library (checked in CI)"])

    module("c02-ingestion.svg", "C2 · Ingestion",
           "Turns any document into passages and finds what each passage is about, without needing titles.",
           [[{"t": "Source document", "l": ["from any connector"], "k": "source"},
             {"t": "Parser plugin", "l": ["sections, headings, tables, lists"], "k": "plugin"},
             {"t": "Structure-first chunks", "l": ["split at section boundaries"]},
             {"t": "Content-defined split", "l": ["long sections only", "edits change O(1) chunks"]}],
            [{"t": "Mentions", "l": ["NER plugin", "regex ID families", "defined terms"]},
             {"t": "Topic score", "l": ["what each passage is about"]},
             {"t": "Anchors", "l": ["passages that are about entity e"]},
             {"t": "Passages + mentions", "l": ["to C3 and C4"], "k": "user"}]],
           ["boundary   : rollhash(last 48 bytes) mod M = 0,  min_tokens ≤ size ≤ max_tokens",
            "topic(p,e) = 0.4·heading + 0.3·subject + 0.2·tf(e,p)/max tf + 0.1·early",
            "Anchors(e) = { p : topic(p,e) ≥ τ_topic }"])

    module("c03-entity-resolution.svg", "C3 · Entity resolution",
           "Aliases (merge) and same names (split) are one problem, solved with one objective.",
           [[{"t": "Mentions + facts", "l": ["from C2 and connectors"], "k": "source"},
             {"t": "Hard joins", "l": ["same EntityRef", "alias_of ≥ 0.95"]},
             {"t": "Blocking", "l": ["shared token, initials, acronym, ref prefix"]},
             {"t": "Pair probability", "l": ["string, acronym, alias pattern, context, neighbors, coref, type, trust"]}],
            [{"t": "Correlation clustering", "l": ["randomized pivot", "best of R runs"]},
             {"t": "Soft aliases", "l": ["uncertain pairs kept as weighted edges"]},
             {"t": "Feedback", "l": ["verified use: α + 1", "rejected bridge: β + 1"]},
             {"t": "Entities + alias edges", "l": ["to C4"], "k": "user"}]],
           ["p_ij  = σ( wᵀ φ(m_i, m_j) )            type mismatch → p_ij = 0",
            "min   Σ_{i<j} x_ij·(1 − p_ij) + (1 − x_ij)·p_ij",
            "soft  τ_low ≤ p_ij < τ_high   →   π_a ~ Beta(α, β),  π_a = α / (α + β)"])

    module("c04-graph-hub-facets.svg", "C4 · Graph and hub facets",
           "Builds the typed graph and splits hub entities by relation so chains through them stay meaningful.",
           [[{"t": "Passages, entities, facts", "l": ["from C2, C3, connectors"], "k": "source"},
             {"t": "Typed edges", "l": ["mention, anchor", "structured fact", "reference, thread, section"]},
             {"t": "Hub check", "l": ["in-degree > D_max"]},
             {"t": "Relation contexts", "l": ["\"president of ___\"", "\"born in ___\""]},
             {"t": "Facets", "l": ["k-means on contexts", "k by silhouette"]},
             {"t": "Typed graph", "l": ["to C5"], "k": "user"}]],
           ["pairs before : k(k − 1) / 2",
            "pairs after  : Σ_f k_f(k_f − 1) / 2     with Σ_f k_f = k",
            "example      : k = 10,000 → ~5·10⁷ pairs;  20 facets of 500 → ~2.5·10⁶ pairs"])

    module("c05-chains-composer.svg", "C5 · Candidate chains and composer",
           "Enumerates 2–4 hop chains under caps, then gives each chain one key vector, with no LLM.",
           [[{"t": "Typed graph", "l": ["from C4"], "k": "source"},
             {"t": "Level 1", "l": ["A → B", "≤ m per passage"]},
             {"t": "Level 2", "l": ["extend by beam b₂"]},
             {"t": "Level 3", "l": ["extend by beam b₃"]},
             {"t": "Chain probability", "l": ["edges × soft aliases"]}],
            [{"t": "Composer f", "l": ["2-layer transformer over frozen passage vectors"]},
             {"t": "Training", "l": ["real questions", "Chain-ICT pseudo-questions", "alignment to encoder"]},
             {"t": "Drift check", "l": ["d > δ → hybrid keys, retrain"], "k": "warn"},
             {"t": "Chains + keys", "l": ["to C6"], "k": "user"}]],
           ["log P(c) = Σ_t log w(edge_t) + Σ_a log π_a",
            "L        = InfoNCE(q, k_c) + InfoNCE(q̃, k_c) + λ·(1 − cos(k_c, enc(text(c))))",
            "drift    : d = 1 − mean cos(f(c), enc(text(c)))"])

    module("c06-budgeted-selection.svg", "C6 · Budgeted selection",
           "Chooses which chains become stored shortcuts, and lets the index learn from verified use.",
           [[{"t": "Chains + keys", "l": ["from C5"], "k": "source"},
             {"t": "Usefulness u(c)", "l": ["corpus-independent features"]},
             {"t": "Facility location", "l": ["cover useful chains, avoid near-duplicates"]},
             {"t": "Lazy greedy", "l": ["per level", "50 / 30 / 20 split"]},
             {"t": "Stored shortcuts", "l": ["≤ βN"], "k": "user"}],
            [{"t": "Completed chain", "l": ["built at query time by C7"], "k": "source"},
             {"t": "Benefit", "l": ["verified uses × time saved"]},
             {"t": "Promotion", "l": ["replace the lowest-benefit shortcut of its level"]},
             {"t": "Budget holds", "l": ["|S| ≤ βN always"], "k": "user"}]],
           ["F(S)  = Σ_c u(c) · max_{s ∈ S} Jaccard(passages(c), passages(s))",
            "greedy ≥ (1 − 1/e) · OPT  per level, with uniform cost per shortcut"])

    module("c07-index-retrieval.svg", "C7 · Index and retrieval",
           "One vector index holds passages and shortcuts. Retrieval returns whole ordered chains.",
           [[{"t": "Question", "l": ["from a surface"], "k": "source"},
             {"t": "Encode", "l": ["encoder plugin"], "k": "plugin"},
             {"t": "Dense + BM25", "l": ["vector store plugin", "keyword index"]},
             {"t": "Fusion", "l": ["reciprocal rank fusion"]},
             {"t": "Unpack", "l": ["shortcut → ordered passages"]},
             {"t": "C8 filter", "l": ["before anything else"]}],
            [{"t": "Completion", "l": ["composer-scored beam", "up to 4 passages"]},
             {"t": "Chain score S", "l": ["key, start, answer sentence, probability, length"]},
             {"t": "Cross-encoder", "l": ["reranker plugin, top 50"], "k": "plugin"},
             {"t": "Top chains", "l": ["to C9"], "k": "user"}]],
           ["S(q,c) = w₁·cos(q,k_c) + w₂·cos(q,v_p₁) + w₃·max_s cos(q,v_s) + w₄·log P(c) − w₅·(ℓ − 1)",
            "R(q,c) = S(q,c) + w₆·CE(q, text(c))",
            "modes  : FAST = S only · QUALITY = all steps · MAX = QUALITY + one retry in C9"])

    module("c08-permissions.svg", "C8 · Permissions",
           "Nobody sees evidence they cannot open in the source tool. Enforced in the core, never in a plugin.",
           [[{"t": "User", "l": ["signed in"], "k": "source"},
             {"t": "Identity plugin", "l": ["SSO user + groups"], "k": "plugin"},
             {"t": "Mapped identities", "l": ["source users and groups from connectors"]},
             {"t": "principals(u)", "l": ["one set per request"]}],
            [{"t": "Passage ACL", "l": ["from the source record"], "k": "source"},
             {"t": "visible(p, u)", "l": ["per passage"]},
             {"t": "visible(c, u)", "l": ["every passage visible"]},
             {"t": "Policy", "l": ["hide (default) or redacted"]},
             {"t": "Strict mode", "l": ["ACL-homogeneous shortcuts"], "k": "user"}]],
           ["visible(p,u) ⇔ public(p) ∨ principals(p) ∩ principals(u) ≠ ∅",
            "visible(c,u) ⇔ ∀ p ∈ c : visible(p,u)",
            "completion expands only into visible passages"])

    module("c09-reader-verification.svg", "C9 · Reader contract and verification",
           "One LLM call returns structured reasoning; the graph checks it before anything reaches the user.",
           [[{"t": "Top chains", "l": ["from C7"], "k": "source"},
             {"t": "Assemble", "l": ["hop order", "comparison slots", "passages quoted as data"]},
             {"t": "Reader plugin", "l": ["1 LLM call", "JSON per chain"], "k": "plugin"},
             {"t": "Checks", "l": ["bridges vs graph", "quote in passage", "answer type, NLI"]}],
            [{"t": "Chain vote", "l": ["weighted by chain score and bridge check"]},
             {"t": "Confidence", "l": ["isotonic calibration"]},
             {"t": "conf ≥ τ: answer", "l": ["with evidence chain"], "k": "user"},
             {"t": "conf < τ: partial", "l": ["verified hops only", "MAX: one retry"], "k": "warn"}]],
           ["B(c,y) = mean_t 1[ y.bridge_t ∈ aliases(e_t) ]       Q = quote ⊂ p_ℓ and answer ⊂ quote",
            "V(a)   = Σ_c R(q,c)·1[y_c = a]·B(c,y) / Σ_c R(q,c)",
            "conf   = isotonic( σ(θ·[log P, E, B, T, Q, V]) )      τ: error among answered ≤ 5%"])

    module("c10-incremental-update.svg", "C10 · Incremental update",
           "A changed document is answerable in seconds. Work depends on the change, not on corpus size.",
           [[{"t": "Changed records", "l": ["pull sync or push event"], "k": "source"},
             {"t": "Re-chunk", "l": ["content-defined boundaries"]},
             {"t": "Hash diff Δ", "l": ["added, edited, removed passages"]},
             {"t": "Entities touched", "l": ["E_Δ before ∪ after"]},
             {"t": "Invalid set I", "l": ["shortcuts through Δ or changed bridges"]}],
            [{"t": "Local rebuild", "l": ["C3–C6 inside L hops of Δ"]},
             {"t": "Write new rows", "l": ["PostgreSQL + vector store"], "k": "store"},
             {"t": "Atomic swap", "l": ["tombstone I"]},
             {"t": "Watch", "l": ["evaluate rules on new chains"], "k": "user"}]],
           ["work = O(|Δ| · m^(L−1))   independent of N",
            "check: incremental result = full rebuild result (≥ 99% identical top-10)"])


# --------------------------------------------------------------------------
# 4. Flows
# --------------------------------------------------------------------------
def ingestion_flow():
    c = Canvas(1500, 620, "Data ingestion flow",
               "Each domain brings its own sources and parsing (blue). The core path (amber) is identical for every domain.")
    c.lane(40, 118, "Domain-owned · plugins")
    a = c.row(40, 130, 1420, [
        {"t": "Domain sources", "l": ["tools, files, databases, chat, streams"], "k": "source"},
        {"t": "Connectors", "l": ["YAML, standard formats, or Python SDK"], "k": "plugin"},
        {"t": "Parsers", "l": ["PDF, HTML, DOCX, email, exports"], "k": "plugin"},
        {"t": "Entity patterns", "l": ["NER model, ID regexes, defined terms"], "k": "plugin"},
        {"t": "Redactor", "l": ["secrets masked before indexing"], "k": "plugin"},
        {"t": "Canonical records", "l": ["documents, facts, ACLs, deletions"], "k": "store"},
    ])
    c.lane(40, 318, "Core · fixed")
    b = c.row(40, 330, 1420, [
        {"t": "C2 Ingest", "l": ["passages, mentions, topic score"]},
        {"t": "C3 Resolve", "l": ["aliases, same names"]},
        {"t": "C4 Graph", "l": ["typed edges, hub facets"]},
        {"t": "C5 Chains", "l": ["2–4 hops, composer keys"]},
        {"t": "C6 Select", "l": ["≤ βN shortcuts"]},
        {"t": "C7 Index", "l": ["PostgreSQL + vector store"], "k": "store"},
    ])
    c.snake(a[-1], b[0])
    c.box(40, 470, 1420, 74, "A domain pack = connectors + parsers + entity patterns + trust weights",
          ["Switching from security to IT operations, legal or support changes only the blue boxes. Zero LLM calls anywhere on this page."], "user")
    c.legend(40, 590, ["source", "plugin", "store", "core"])
    c.save("10-ingestion-flow.svg")


def retrieval_flow():
    c = Canvas(1500, 640, "Retrieval and augmentation flow",
               "Domains own how questions come in and how answers look (blue). The core owns retrieval, permissions and verification (amber).")
    a = c.row(40, 110, 1420, [
        {"t": "User question", "l": ["any surface"], "k": "source"},
        {"t": "Surface plugin", "l": ["web, Slack / Teams, REST, IDE"], "k": "plugin"},
        {"t": "Identity plugin", "l": ["principals(u) from SSO"], "k": "plugin"},
        {"t": "C7 Retrieve", "l": ["dense + BM25, unpack chains"]},
        {"t": "C8 Permissions", "l": ["hide what u cannot see"]},
        {"t": "C7 Complete + rerank", "l": ["beam, chain score, cross-encoder"]},
    ])
    b = c.row(40, 320, 1420, [
        {"t": "C9 Assemble", "l": ["hop-ordered chains, quoted as data"]},
        {"t": "Domain prompt template", "l": ["question types, wording, output fields"], "k": "plugin"},
        {"t": "Reader LLM plugin", "l": ["1 call, JSON per chain"], "k": "plugin"},
        {"t": "C9 Verify", "l": ["bridges, quote, type, NLI, vote, confidence"]},
        {"t": "Domain answer formatter", "l": ["links back to source tools"], "k": "plugin"},
        {"t": "Answer or partial", "l": ["evidence chain + confidence"], "k": "user"},
    ])
    c.snake(a[-1], b[0])
    c.box(40, 470, 1420, 74, "The augmentation is checked, not trusted",
          ["The reader's bridges are compared with the graph; weak answers become verified partial results. Write actions need user confirmation."], "warn")
    c.legend(40, 610, ["source", "plugin", "core", "user", "warn"])
    c.save("11-retrieval-augmentation-flow.svg")


def update_flow():
    c = Canvas(1500, 560, "Update flow",
               "Source changes reach answers in seconds, with no LLM calls and no full rebuild.")
    a = c.row(40, 110, 1420, [
        {"t": "Source change", "l": ["edit, new record, deletion, permission change"], "k": "source"},
        {"t": "Connector", "l": ["pull sync with cursor, or push webhook / stream"], "k": "plugin"},
        {"t": "Job queue", "l": ["PostgreSQL table, SKIP LOCKED workers"], "k": "store"},
        {"t": "C10 Diff", "l": ["content-defined re-chunk, hash diff Δ"]},
        {"t": "Affected radius", "l": ["entities touched, chains through Δ"]},
    ])
    b = c.row(40, 310, 1420, [
        {"t": "Local rebuild", "l": ["C3–C6 inside L hops"]},
        {"t": "Atomic swap", "l": ["new rows in, old rows tombstoned"], "k": "store"},
        {"t": "Watch rules", "l": ["domain-defined conditions"], "k": "plugin"},
        {"t": "Notify", "l": ["surface plugin, with evidence chain"], "k": "plugin"},
        {"t": "User", "l": ["alert they are allowed to see"], "k": "user"},
    ])
    c.snake(a[-1], b[0])
    c.legend(40, 520, ["source", "plugin", "store", "core", "user"])
    c.save("12-update-flow.svg")


def harness():
    c = Canvas(1500, 700, "Evaluation harness",
               "Every system gets the same data, the same reader and the same passage budget. Results decide the gates.")
    a = c.row(40, 110, 1420, [
        {"t": "Datasets", "l": ["MuSiQue, 2Wiki, HotpotQA", "OSV security harness", "200 human-checked questions"], "k": "source"},
        {"t": "Loaders", "l": ["canonical documents", "gold chains per hop"], "k": "plugin"},
        {"t": "Question generator", "l": ["2–4 hop questions from structured facts"], "k": "plugin"},
        {"t": "Systems under test", "l": ["HSI FAST / QUALITY / MAX", "dense, BM25, online expansion", "HippoRAG 2, IndexRAG, AAR, IRCoT"], "k": "core"},
    ])
    b = c.row(40, 340, 1420, [
        {"t": "Fixed conditions", "l": ["same reader model", "same passage budget", "3 seeds"], "k": "store"},
        {"t": "Evaluator", "l": ["waterfall per hop", "full-chain recall@k", "F1, risk–coverage", "cost, freshness"]},
        {"t": "Security suites", "l": ["permissions, injection, secrets, poisoning"], "k": "warn"},
        {"t": "Reports", "l": ["tables, plots, confidence intervals, hashes"], "k": "user"},
        {"t": "Gate decision", "l": ["pass, narrow or stop", "written to specs/decisions"], "k": "user"},
    ])
    c.snake(a[-1], b[0])
    c.box(40, 540, 1420, 74, "Continuous integration",
          ["Every core change runs a fixed 200-question slice. A drop of more than 1 point on any per-hop recall, or any new security failure, fails the build."], "store")
    c.legend(40, 670, ["source", "plugin", "core", "store", "warn", "user"])
    c.save("20-harness.svg")


def domains():
    c = Canvas(1500, 900, "Attaching HSI to any domain",
               "A domain pack brings its own ingestion and its own retrieval and augmentation. The core engine is the same for all of them.")
    cx, cy, cw, ch = 560, 330, 380, 230
    c.box(cx, cy, cw, ch, "HSI core engine", ["C1–C10, unchanged", "", "ingest API", "query API", "update API", "", "PostgreSQL + vector store"], "core")
    packs = [
        (40, 120, "Software supply-chain security",
         ["Ingest: SCA and SAST exports, SBOMs, advisories, ALM items",
          "Augment: exposure questions, owners, risk decisions, Watch on new CVEs"]),
        (40, 560, "IT operations and incidents",
         ["Ingest: tickets, runbooks, postmortems, CMDB, chat threads",
          "Augment: incident chains, ownership, the fix used last time"]),
        (1080, 120, "Legal and compliance",
         ["Ingest: contracts, policies, registries, audit evidence",
          "Augment: obligation chains, defined terms, evidence for controls"]),
        (1080, 560, "Customer support",
         ["Ingest: tickets, product docs, release notes, known issues",
          "Augment: root-cause chains, known-fix answers"]),
        (560, 680, "Your next domain",
         ["Ingest: write connectors (YAML, formats, SDK)",
          "Augment: prompt template, answer format, Watch rules"]),
    ]
    for x, y, t, l in packs:
        w = 380
        h = c.box_height(w, t, l)
        c.box(x, y, w, h, t, l, "plugin")
        if x < cx:
            c.arrow([(x + w + 2, y + h / 2), (cx - 4, cy + ch / 2 + (-40 if y < cy else 40))])
        elif x > cx:
            c.arrow([(x - 2, y + h / 2), (cx + cw + 4, cy + ch / 2 + (-40 if y < cy else 40))])
        else:
            c.arrow([(x + w / 2, y - 2), (cx + cw / 2, cy + ch + 4)])
    c.legend(40, 870, ["core", "plugin"])
    c.save("30-domain-attachment.svg")


def thesis():
    c = Canvas(1500, 760, "HSI thesis",
               "The core engine is the thesis. Supporting modules make it robust. Additions apply it. Each layer depends only on the one below.")
    # layers
    c.box(40, 100, 1420, 70, "Additions · applications, not thesis claims",
          ["Connectors for any source (A2) · permissions and security (A3) · multimodal ingestion (A4) · copilot (A1) · incremental updates (C10) · domain packs"], "plugin")
    c.box(40, 200, 1420, 70, "Supporting · robustness on real data",
          ["C2 ingestion without titles · C3 entity resolution with soft aliases · C4 relation-faceted hubs · C9 graph-checked reading (RQ5)"], "store")
    c.box(40, 300, 1420, 130, "Thesis core · HSI",
          ["Hierarchy: 2-, 3- and 4-passage chain entries in one flat index, unpacked into ordered evidence (C5, C7)",
           "Composer: one key per chain from frozen passage vectors, trained with Chain-ICT, no LLM (C5)",
           "Budgeted selection: |S| ≤ βN, facility location, (1 − 1/e) per level (C6)",
           "Single lookup: one nearest-neighbour search over I = P ∪ S (C7)"], "core")
    c.arrow([(750, 172), (750, 196)])
    c.arrow([(750, 272), (750, 296)])
    c.formula(40, 460, 1420, [
        "maximize   E_q [ 1( c* ⊆ R_k(q) ) ]            R_k(q) = first k passages of unpack( topK over I = P ∪ S )",
        "subject to |S| ≤ βN · zero LLM calls while building the index · one nearest-neighbour search per question",
    ])
    c.lane(40, 566, "Research questions → gates")
    c.row(40, 578, 1420, [
        {"t": "RQ1 Depth", "l": ["single-lookup chains beat online expansion at 3–4 hops", "Gate 0"], "tag": "kill rule"},
        {"t": "RQ2 Keys", "l": ["composed keys ≈ text keys, transfer across corpora", "Gate 1–2"]},
        {"t": "RQ3 Budget", "l": ["greedy coverage beats heuristics at every β", "Gate 1"]},
        {"t": "RQ4 Cost", "l": ["Pareto-better than LLM-indexed and iterative methods", "Gate 1"]},
        {"t": "RQ5 Trust", "l": ["graph-checked reading lowers error at equal coverage", "Gate 2"], "k": "store"},
    ])
    c.legend(40, 730, ["core", "store", "plugin"])
    c.save("00-thesis.svg")


if __name__ == "__main__":
    thesis()
    overall()
    core_engine()
    modules()
    ingestion_flow()
    retrieval_flow()
    update_flow()
    harness()
    domains()
    print("\n".join(sorted(f for f in os.listdir(OUT) if f.endswith(".svg"))))
