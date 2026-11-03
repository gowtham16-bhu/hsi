# HSI: retrieval that follows shared names between documents

**HSI finds the documents a question needs even when some of them don't look like the question.**
It links documents that mention the same names, then, for each question, walks those links from the
best search results, steered by a small learned model. The index is built without any LLM calls, on an
ordinary CPU.

HSI is an open-source research project. It works on text today; it is not a finished product.

![HSI overview: the problem, the core components, and the results](docs/images/hsi-overview.png)

---

## Contents
1. [The problem](#1-the-problem)
2. [How HSI works](#2-how-hsi-works)
3. [Results](#3-results)
4. [Where HSI does not help (yet)](#4-where-hsi-does-not-help-yet)
5. [Quickstart](#5-quickstart)
6. [Index your own documents](#6-index-your-own-documents)
7. [Reproduce the experiments](#7-reproduce-the-experiments)
8. [Repository layout](#8-repository-layout)
9. [How this project is run](#9-how-this-project-is-run)
10. [Related work: what is new and what is not](#10-related-work-what-is-new-and-what-is-not)
11. [Roadmap](#11-roadmap)
12. [License](#12-license)

---

## 1. The problem

Some questions need facts from two or more documents, and only the first one looks like the question.
A real example from the 2WikiMultiHopQA benchmark:

> **"Where was the composer of film Billy Elliot born?"**

| | Document | What it says | Plain vector search |
|---|---|---|---|
| Hop 1 | *Billy Elliot* | "…directed by Stephen Daldry with music composed by **Stephen Warbeck**." | found (rank 1) |
| Hop 2 | *Stephen Warbeck* | "**Stephen Warbeck** … is an English composer … born in **Southampton, Hampshire**." | **missed** (not in the top 5) |

The second document never mentions *Billy Elliot*, so it doesn't look like the question. The only thing
that connects the two is the **shared name** "Stephen Warbeck". HSI follows that link and returns the
second document at rank 4 of 5.

The usual fix, as in GraphRAG or HippoRAG, is to send every document to an LLM to extract facts and
build a knowledge graph. That costs money per document and sends the data to an LLM. HSI builds its
links with statistics instead.

## 2. How HSI works

HSI has two phases.

**Offline, once per collection (no LLM calls)**

| Step | Component | What it does |
|---|---|---|
| 1 | Chunker | Splits documents into passages |
| 2 | Encoder (plugin) | Turns each passage into a vector, e.g. with `BAAI/bge-small-en-v1.5` or `bge-large-en-v1.5` |
| 3 | Name finder | Finds the names in each passage: document titles matched across all text, plus names learned from the text itself (today, from how consistently a word is capitalized) |
| 4 | Link builder | Links two passages when they share a name. Names that appear in too many passages are skipped, and each passage keeps its 32 strongest links |
| 5 | Compact graph | Stores the links as flat arrays (about 24 bytes per link), built one passage at a time |

**Online, for each question (about 12 ms on a CPU)**

| Step | Component | What it does |
|---|---|---|
| 1 | Encoder | Encodes the question with the same model |
| 2 | Seed search | Finds the 3 passages closest in meaning |
| 3 | Steered walk | Spreads probability from those passages along the links; the **hop model** decides how much goes to each link |
| 4 | Ranking | Returns the top-k passages, with their sources |

### The hop model

From each passage, the hop model scores every outgoing link with 8 signals and turns the scores into
probabilities:

```
P(u → v | question) = softmax over u's links of  θ · z(φ(question, u, v))
```

| Signal φ | Learned weight θ | What the model learned |
|---|---|---|
| next passage matches the question | +1.98 | strongest signal |
| the shared name appears in the question | −1.24 | that passage was probably found already |
| next passage repeats the current one | −0.70 | prefer new information |
| link strength | +0.42 | stronger links are better |
| how common the shared name is | −0.41 | rare names are better bridges |
| link type is "both mention it" | −0.36 | "page about it" links are better |
| how prominent the name is in the current passage | +0.33 | follow what the passage is about |
| current passage matches the question | 0.00 | the same for every link, so it cancels |

It was trained on 2,000 MuSiQue *training* questions whose correct chains are known, by ranking the
true next passage first (listwise softmax cross-entropy). It is 8 numbers, trains in seconds on a CPU,
and picks the right next passage first **43.9%** of the time on held-out examples, against 33.1% for
the best hand-written rule and about 3.4% at random.

### The walk, and why it stays fast

The walk is personalized PageRank with restart probability α = 0.5. HSI computes it with **local
forward push**, which only touches passages near the seeds. The work per question is bounded by
1 / (α · ε), a limit that does not depend on the size of the collection. Full derivations, worked
examples and design decisions are in [`specs/`](specs/).

## 3. Results

All numbers are measured, pre-registered before the run, and recorded in
[`specs/decisions/`](specs/decisions/) and [`runs/`](runs/). Failures are recorded the same way.

**Final test** (decision 0011): HippoRAG 2's released evaluation sets, 1,000 questions each,
`bge-large-en-v1.5` for both methods, zero LLM calls. Metric: share of the needed passages in the top 5.

| Dataset | Vector search | HSI | Gain (95% CI) |
|---|---|---|---|
| 2WikiMultiHopQA | 72.7% | **92.1%** | +19.4 [+17.7, +20.9] |
| MuSiQue | 56.5% | **63.4%** | +6.9 [+5.3, +8.3] |

On 4-hop 2Wiki questions, the whole evidence chain is in HSI's top 10 for 85.5% of questions, against
2.1% for vector search. For reference, HippoRAG 2 reports 90.4 (2Wiki) and 74.7 (MuSiQue) with a 7B
encoder and an LLM-built graph; those figures come from a summary of their paper and are still to be
checked against the paper's own table.

**Speed** (decision 0012, MuSiQue training split, 1,000 questions, `bge-small`):

| Walk | Recall@5 | Latency (p50) | Work per question |
|---|---|---|---|
| Full walk over every link | 62.1% | 83 ms | all links, 50 passes |
| Local push, ε = 10⁻³ | 62.1% | 12.5 ms | 120 pushes |

**Scale** (decisions 0013–0016, 200 unseen questions, real distractor passages):

| Passages | Links kept per passage | Peak memory | Pushes per question | Gain over vector search |
|---|---|---|---|---|
| 10,000 | 32 | 1.2 GB | 498 | +8.9 points |
| 40,000 | 32 | 1.6 GB | 537 | +6.9 points |
| 140,000 | 32 | 3.2 GB | 515 | **+0.2 points** |
| 140,000 | 128 | 3.7 GB | 782 | **+2.6 points** |

Cost stays flat as the collection grows, but the quality gain shrinks. Splitting common names into
facets, storing more links (the last row) and retraining the hop model at scale did not recover it.
The most likely remaining cause is the encoder, which finds the right first passage less often among
many distractors.

## 4. Where HSI does not help (yet)

| Situation | Why |
|---|---|
| Large collections (≥ 140K passages) | The gain over vector search shrinks to about 2.6 points (see above) |
| Questions one document can answer | There is nothing to link; the cost or benefit there has not been measured yet |
| Documents connected by ideas, not names | It links names and titles, not concepts such as "photosynthesis" |
| Lowercase names (`payments-api`, `log4j`, code identifiers) | Today's name finder relies on capitalization; only titles would link. A case-independent version is planned |
| Languages without capital letters | Same reason; only titles would link |
| Collections where almost every document shares the same few names | Those names are skipped as too common |
| Images, audio, video | Text only for now |

## 5. Quickstart

Requirements: Python 3.11+, and PostgreSQL 16 (through Docker, or local binaries).

```bash
git clone https://github.com/gowtham16-bhu/hsi.git
```

```bash
cd hsi && python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install -e packages/hsi-core -e packages/hsi-plugins-basic -e packages/hsi-plugins-faiss \
            -e packages/hsi-plugins-postgres -e packages/hsi-plugins-onnx -e packages/hsi-harness \
            -e packages/hsi-connector-kit -e packages/hsi-plugins-osv -e packages/hsi-service \
            pytest ruff import-linter
```

Start PostgreSQL. The script uses Docker when available, otherwise local binaries, and prints the
connection string:

```bash
eval "$(scripts/dev-postgres.sh)"
```

Run the checks:

```bash
ruff check . && lint-imports && pytest
```

### Try it on real data: the 2Wiki demo

Download the datasets (about 300 MB) and convert the 2Wiki corpus to JSONL:

```bash
scripts/get-datasets.sh
```

```bash
python scripts/make-demo-jsonl.py data/hipporag2/2wikimultihopqa_corpus.json data/demo/2wiki-corpus.jsonl
```

Build the index (about 2 minutes on 4 CPUs; the encoder downloads on first use):

```bash
hsi-index build --config configs/demo/2wiki-index.yaml
```

Serve it over REST:

```bash
hsi-serve --config configs/demo/2wiki-serve.yaml --transport rest --host 127.0.0.1 --port 8765
```

Ask a question from another terminal:

```bash
curl -s -X POST http://127.0.0.1:8765/v1/retrieve -H 'content-type: application/json' \
  -d '{"question": "Where was the composer of film Billy Elliot born?", "k": 5, "principals": ["user:demo"]}'
```

The response lists the passages with their titles, text and scores. `principals` is the identity of the
person asking: only passages they are allowed to see are used and returned.

### Use it from an AI agent (MCP)

`hsi-serve --transport mcp-stdio` exposes one tool, `retrieve(question, k)`. For an MCP client that
launches servers from a JSON config, use absolute paths:

```json
{
  "mcpServers": {
    "hsi": {
      "command": "/path/to/hsi/.venv/bin/hsi-serve",
      "args": ["--config", "/path/to/hsi/configs/demo/2wiki-serve.yaml",
               "--transport", "mcp-stdio", "--host", "127.0.0.1", "--port", "0"],
      "cwd": "/path/to/hsi",
      "env": {"HSI_PG_DSN": "postgresql://hsi:hsi_dev_only@127.0.0.1:5432/hsi"}
    }
  }
}
```

With MCP, the identity comes from `mcp_principals` in the serve config, never from the tool arguments.
Passages are returned labeled as quoted source data, not instructions.

## 6. Index your own documents

Export your documents as JSON Lines, one document per line. Every document needs an access rule:

```json
{"id": "doc-1", "title": "Payments API", "text": "payments-api is owned by Team Checkout ...", "acl": {"principals": ["group:engineering"]}}
{"id": "doc-2", "title": "Holiday policy", "text": "...", "acl": {"public": true}}
```

Then copy [`configs/demo/2wiki-index.yaml`](configs/demo/2wiki-index.yaml), point
`sources[0].options.path` at your file, change `index_id`, and build. Lines without a valid `acl` are
rejected. A new kind of source is added as a connector plugin; see
[`specs/additions/A2-plugins-and-connectors.md`](specs/additions/A2-plugins-and-connectors.md).

| Config section | What it sets |
|---|---|
| `sources` | Connector plugins and their options |
| `chunking` | Passage size |
| `encoder` | Embedding model plugin; the serve config must use the same one |
| `graph` | Link weights, fan-out (links kept per passage), the cap for common names, the name finder |
| `transition_model` | The hop model file |
| `push` (serve config) | Seeds, restart probability α, precision ε |

Every value lives in the config. The code holds no word lists, thresholds or model names.

## 7. Reproduce the experiments

Download the data first (`scripts/get-datasets.sh`).

| Experiment | Command |
|---|---|
| Final test, 2Wiki | `hsi-gate0 --config configs/test/2wiki-hipporag2-final.yaml` |
| Final test, MuSiQue | `hsi-gate0 --config configs/test/musique-hipporag2-final.yaml` |
| Train the hop model | `hsi-train-transitions --config configs/train/transitions-musique-v2-fan32.yaml` |
| Fast walk vs full walk | `hsi-gate0 --config configs/dev/musique-train-push.yaml` |
| Scaling benchmark | `hsi-bench-scale --config configs/bench/scale.yaml` |

Each run writes a JSON and a Markdown report to `runs/`, keyed by the hash of its config. The same config
gives the same results.

## 8. Repository layout

| Path | Contents |
|---|---|
| `packages/hsi-core` | The core engine: name finding, graph and compact graph, streaming builder, walks, hop model, metrics. Imports only numpy, pydantic and yaml |
| `packages/hsi-plugins-onnx`, `hsi-plugins-st` | Encoders: ONNX Runtime, sentence-transformers |
| `packages/hsi-plugins-faiss` | Vector store |
| `packages/hsi-plugins-postgres` | Metadata store, audit log, SQL migrations |
| `packages/hsi-plugins-basic` | Name finder, JSONL connector, a hashing encoder for tests only |
| `packages/hsi-plugins-osv` | Connector for OSV vulnerability advisories |
| `packages/hsi-connector-kit` | Conformance checks every connector must pass |
| `packages/hsi-service` | `hsi-index` and `hsi-serve` (REST and MCP), permission-aware retrieval |
| `packages/hsi-harness` | Dataset loaders, experiment runner, trainers, scaling benchmark |
| `specs/` | Thesis, constitution, architecture, core engine, evaluation plan, product additions |
| `specs/decisions/` | One record per decision, with its pre-registered rule and outcome |
| `configs/` | Every experiment and demo config |
| `runs/` | Reports of every run |
| `tests/` | Unit and end-to-end tests |

## 9. How this project is run

- **Spec first.** A spec or decision record changes before the code does.
- **Pre-registered tests.** Each experiment states its pass rule before it runs; tuning uses training
  splits only, and final tests run once.
- **Failures are kept.** The first two ideas failed, and so did three attempts to fix the scale problem.
  Each is recorded with its numbers in `specs/decisions/`.
- **No hard-coded knowledge.** No word lists, stop words, language rules or magic thresholds in code.
- **The core stays independent.** Plugins and applications never change the core engine (checked by
  import-linter and a test).

How the design got here:

| # | Idea | Result | Decision |
|---|---|---|---|
| 1 | Pre-compute chains of linked passages as extra index entries | Failed: 69% of 3-hop chains were cut before they were stored | `gate0.md` |
| 2 | Search again at every hop | Failed: long chains were lost and queries drifted | `gate0-prime.md` |
| 3 | Walk a graph of shared names, steered by a learned hop model | Worked | 0004–0011 |
| 4 | Local push instead of a full walk | Worked: same quality, 6.7× faster | 0012 |
| 5 | Scale to 140K passages | Cost fine, gain shrinks; open | 0013–0016 |

## 10. Related work: what is new and what is not

The general idea, an LLM-free graph plus personalized PageRank for multi-hop retrieval, is already
published: SPRIG (2026), LinearRAG (ICLR 2026) and NoLLMRAG. HippoRAG and HippoRAG 2 use personalized
PageRank over LLM-built graphs. Learning random-walk weights from features goes back to *Supervised
Random Walks* (Backstrom and Leskovec, 2011), and local push to Andersen, Chung and Lang (2006).

What HSI adds, pending a full head-to-head comparison: a hop model learned listwise from gold chains;
that model inside local push, so cost stays flat; a name finder learned from the corpus with no
name-recognition model; and a scale study that reports where the gain fades. See the novelty audit in
[`specs/00-thesis.md`](specs/00-thesis.md).

> The name HSI comes from the first idea, a *Hierarchical Shortcut Index* of pre-computed chains. That
> idea failed and was dropped; the name stayed.

## 11. Roadmap

- Find why the gain shrinks on large collections, starting with stronger embedding models
- Case-independent name finding (lowercase names, identifiers, languages without capital letters)
- Measure single-document questions and end-to-end answer quality with an LLM reader
- Code plugins: identifiers from syntax trees, one passage per function
- More connectors (MCP servers, standard export formats); Docker image and a network-blocked test
- Images, audio and video through transcripts, captions and text extraction

## 12. License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
