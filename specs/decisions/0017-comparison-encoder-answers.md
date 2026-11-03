# Decision 0017: head-to-head with LinearRAG, the encoder at scale, and answer quality

**Date:** 2026-11-03 · **Status:** pre-registered · **Affects:** claims in the thesis, README and posts

No HSI parameter, model or config changes are allowed in this decision. Each experiment uses the
configurations already fixed by decisions 0011–0016.

## E1 — Retrieval head-to-head with LinearRAG
LinearRAG (ICLR 2026) is the closest published LLM-free graph retriever with public code
(github.com/DEEP-PolyU/LinearRAG, GPL-3.0; run as an external program, none of its code enters HSI).

- **Data:** HippoRAG 2's MuSiQue and 2Wiki sets, the same 1,000 questions and corpora as decision 0011.
- **Systems:** LinearRAG retrieval with its published per-dataset settings (`scripts/run.sh`: spaCy
  `en_core_web_trf`, `all-mpnet-base-v2`); LinearRAG with `bge-large-en-v1.5`; dense retrieval with each
  encoder; HSI System A from decision 0011 (bge-large; its numbers are already measured, not re-tuned).
- **Metric:** gold-passage recall@5 and @10 (the same function as decision 0011), plus indexing time,
  query latency and LLM calls.
- **Reading (no pass/fail):** report the numbers as measured, with 95% bootstrap intervals for the paired
  HSI − LinearRAG difference at the same encoder.
- **Not compared:** Microsoft LazyGraphRAG calls an LLM at query time; no LLM is available here.

## E2 — Does a stronger encoder recover the gain at scale?
Decision 0014's diagnosis: misses with no gold seed nearly double from 10K to 140K passages.

- **Setup:** `configs/bench/scale-large.yaml` = `configs/bench/scale-facets-50.yaml` with
  `bge-large-en-v1.5` instead of `bge-small`; sizes 10K and 40K (140K needs about 5 hours of CPU
  encoding and is run only if E2 at 40K is promising). Same 200 questions and corpora.
- **Pre-registered reading:** the encoder is confirmed as the main cause if, at 40K, HSI's gain over
  dense with bge-large is at least 2 points larger than with bge-small (+6.9), **or** the share of
  questions with a gold seed rises by at least 5 points.

## E3 — Do better passages give better answers?
- **Reader:** a local instruction model, `Qwen/Qwen2.5-1.5B-Instruct`, run on CPU with greedy
  decoding (temperature 0), the same prompt for every system: the question, the
  top-5 passages, and an instruction to answer in a few words.
- **Data:** the first 200 questions of each HippoRAG 2 test set (MuSiQue, 2Wiki).
- **Systems:** dense top-5 vs HSI System A top-5, bge-large.
- **Metrics:** exact match and token F1 against the gold answer and its aliases (standard SQuAD
  normalization).
- **Pre-registered reading:** better passages give better answers if HSI − dense token F1 has a 95%
  bootstrap interval above 0 on 2Wiki. MuSiQue is reported either way.
