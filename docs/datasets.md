# Datasets

Run `scripts/get-datasets.sh` to download everything below into `data/` (needs the `hf` command).

Place files under `data/` (ignored by git). Licenses: MuSiQue CC BY 4.0, 2WikiMultiHopQA Apache-2.0, HotpotQA CC BY-SA 4.0.

| Dataset | Files | Source used here | Original |
|---|---|---|---|
| MuSiQue | `data/musique/musique_ans_v1.0_dev.jsonl`, `musique_ans_v1.0_train.jsonl` | Hugging Face `dgslibisey/MuSiQue` | github.com/StonyBrookNLP/musique |
| 2WikiMultiHopQA | `data/2wiki/dev.json` | Hugging Face `voidful/2WikiMultiHopQA` | github.com/Alab-NII/2wikimultihop |
| HippoRAG 2 evaluation sets (final test) | `data/hipporag2/{musique,2wikimultihopqa}{,_corpus}.json` | Hugging Face `osunlp/HippoRAG_2` | the HippoRAG 2 release |
| HotpotQA (not used yet) | `data/hotpotqa/hotpot_dev_distractor_v1.json` | — | hotpotqa.github.io |

The encoder weights (`BAAI/bge-small-en-v1.5`, `bge-large-en-v1.5`) download from Hugging Face on first use. In an air-gapped
setup, copy the model directory to the machine and set `encoder.options.model` to its local path.

## Protocol
- 1,000 questions sampled with seed 0 (answerable questions only for MuSiQue).
- Corpus = all paragraphs of the sampled questions, deduplicated by (title, text), as in HippoRAG 2.
- Gold chain = supporting paragraphs; for MuSiQue, in decomposition order.
