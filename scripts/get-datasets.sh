#!/usr/bin/env bash
# Download the public datasets used in the experiments into data/ (ignored by git).
# Needs the `hf` command (installed with huggingface_hub). Licenses: MuSiQue CC BY 4.0,
# 2WikiMultiHopQA Apache-2.0; HippoRAG 2's evaluation sets are redistributions of both.
set -euo pipefail
cd "$(dirname "$0")/.."

hf download dgslibisey/MuSiQue musique_ans_v1.0_dev.jsonl musique_ans_v1.0_train.jsonl \
  --repo-type dataset --local-dir data/musique
hf download voidful/2WikiMultiHopQA dev.json --repo-type dataset --local-dir data/2wiki
hf download osunlp/HippoRAG_2 musique.json musique_corpus.json 2wikimultihopqa.json 2wikimultihopqa_corpus.json \
  --repo-type dataset --local-dir data/hipporag2
echo "datasets are in data/"
