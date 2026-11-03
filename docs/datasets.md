# Datasets for Gate 0

Place files under `data/` (ignored by git). Licenses: MuSiQue CC BY 4.0, 2WikiMultiHopQA Apache-2.0, HotpotQA CC BY-SA 4.0.

| Dataset | File the loader expects | Source |
|---|---|---|
| MuSiQue | `data/musique/musique_ans_v1.0_dev.jsonl` | github.com/StonyBrookNLP/musique (download script) |
| 2WikiMultiHopQA | `data/2wiki/dev.json` | github.com/Alab-NII/2wikimultihop (data link in README) |
| HotpotQA | `data/hotpotqa/hotpot_dev_distractor_v1.json` | hotpotqa.github.io |

The encoder weights (`BAAI/bge-small-en-v1.5`) download from Hugging Face on first use. In an air-gapped
setup, copy the model directory to the machine and set `encoder.options.model` to its local path.

## Protocol
- 1,000 questions sampled with seed 0 (answerable questions only for MuSiQue).
- Corpus = all paragraphs of the sampled questions, deduplicated by (title, text), as in HippoRAG 2.
- Gold chain = supporting paragraphs; for MuSiQue, in decomposition order.
