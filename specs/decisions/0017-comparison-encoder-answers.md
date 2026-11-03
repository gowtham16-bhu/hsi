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

## Amendment (2026-11-03, before any E2, E3 or E1 result): where the experiments run
- **Speed measured in the cloud container (4 CPUs):** the E3 reader took 889 s for 3 questions × 2 systems
  while E2 shared the CPU, so 200 questions per dataset would take well over 10 hours. Those 3 smoke-test
  answers were seen (HSI's passages gave 2 correct answers, dense 0); they are a pipeline check and are not
  part of any result.
- **Change:** E2 and E3 may run on the owner's Apple Silicon Mac. The device is part of each config
  (`encoder.options.device: mps` for the sentence-transformers encoder, `reader.device` and `reader.dtype`
  for the reader), so it is recorded in every report. Nothing else changes: same questions, systems,
  prompt, metrics and readings. Results from a different device are reported as such.
- **If the run stays on CPU,** E3 uses the first N questions per dataset, N chosen from the measured
  uncontended speed to fit about 3 hours, and stated in the report.

## Outcome E2 (2026-11-03, cloud CPU): reading met on its second condition only; interpretation: not mainly the encoder
Run `runs/bench/scale-8554c223c1cd44d8.json` (`configs/bench/scale-large.yaml`); bge-small reference from
`runs/bench/` facets-50 run (decision 0014). Seed shares computed on the same corpora (cached vectors).

| Encoder | Passages | Questions with a gold seed | Dense GP@5 | HSI GP@5 | Gain |
|---|---|---|---|---|---|
| bge-small | 10K | 85.5% | 55.9 | 64.8 | +9.0 |
| bge-small | 40K | 82.5% | 48.5 | 55.3 | +6.8 |
| bge-large | 10K | 91.0% | 63.0 | 71.0 | +7.9 |
| bge-large | 40K | 89.0% | 57.6 | 62.2 | +4.6 |

- **Condition 1 (gain at 40K ≥ +8.8 with bge-large):** not met; the gain is +4.6, smaller than bge-small's +6.8.
- **Condition 2 (gold-seed share at 40K up ≥ 5 points):** met; +6.5 points (82.5% → 89.0%).
- **Interpretation.** The rule said "either condition", so it is formally met. The substance points the other
  way: the stronger encoder fixes the seeds and raises both systems (HSI 55.3 → 62.2), but the walk's added
  value shrinks (+6.8 → +4.6), and it shrinks faster from 10K to 40K (−3.3 vs −2.2). The scale problem is
  therefore **not mainly the encoder**. The "either" wording was too lenient; future rules state the
  condition that the claim rests on. Next candidates: the walk spreads probability over more competing
  neighbours as N grows (decision 0014's third cause), and the hop model's features do not separate the
  right neighbour from many similar ones.
- Absolute result worth keeping: HSI with bge-large reaches 71.0% (10K) and 62.2% (40K).
