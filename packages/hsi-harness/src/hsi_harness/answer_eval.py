"""Decision 0017 E3: do better passages give better answers?

A local reader model answers each question from each system's top-k passages; answers are scored with
exact match and token F1 against the gold answer and its aliases (SQuAD-style normalization, rules from
config). The prompt comes from config, never from code.

    hsi-answer-eval --config configs/compare/answers-2wiki.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from hsi_core.config import config_hash, load_yaml
from hsi_core.metrics import bootstrap_mean, paired_difference


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReaderSection(_Strict):
    model: str                       # Hugging Face id or local path of an instruction model
    max_new_tokens: int
    threads: int | None = None


class NormalizationSection(_Strict):
    lowercase: bool
    strip_punctuation: bool
    remove_words: list[str]          # e.g. the articles removed by SQuAD v1.1's official script


class AnswerEvalConfig(_Strict):
    per_question_file: str           # from hsi-gate0 with save_per_question: true
    data_file: str                   # original dataset file, for answer aliases
    id_field: str
    aliases_field: str | None
    systems: list[str]
    question_count: int
    k: int
    system_prompt: str
    user_template: str               # with {passages} and {question}
    passage_template: str            # with {title} and {text}
    reader: ReaderSection
    normalization: NormalizationSection
    bootstrap: int
    seed: int
    out_dir: str


def normalize(s: str, n: NormalizationSection) -> str:
    if n.lowercase:
        s = s.lower()
    if n.strip_punctuation:
        s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    words = [w for w in s.split() if w not in set(n.remove_words)]
    return " ".join(words)


def f1(pred: str, gold: str, n: NormalizationSection) -> float:
    p, g = normalize(pred, n).split(), normalize(gold, n).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return float(p == g)
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def run(cfg: AnswerEvalConfig) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.reader.threads:
        torch.set_num_threads(cfg.reader.threads)
    pq = json.loads(Path(cfg.per_question_file).read_text(encoding="utf-8"))
    raw = json.loads(Path(cfg.data_file).read_text(encoding="utf-8"))
    aliases = {str(r[cfg.id_field]): list(r.get(cfg.aliases_field) or []) if cfg.aliases_field else [] for r in raw}
    questions = pq["questions"][: cfg.question_count]
    tok = AutoTokenizer.from_pretrained(cfg.reader.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.reader.model, torch_dtype=torch.float32)
    model.eval()

    per_system: dict[str, dict[str, list]] = {s: {"em": [], "f1": [], "answers": []} for s in cfg.systems}
    t0 = time.perf_counter()
    for q in questions:
        golds = [q["answer"], *aliases.get(str(q["id"]), [])]
        for s in cfg.systems:
            ids = q["results"][s][: cfg.k]
            passages = "\n\n".join(cfg.passage_template.format(title=pq["passages"][p]["title"] or "",
                                                               text=pq["passages"][p]["text"]) for p in ids)
            messages = [{"role": "system", "content": cfg.system_prompt},
                        {"role": "user", "content": cfg.user_template.format(passages=passages,
                                                                             question=q["question"])}]
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=cfg.reader.max_new_tokens, do_sample=False)
            answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            answer = re.split(r"\n", answer, maxsplit=1)[0].strip()
            per_system[s]["answers"].append(answer)
            per_system[s]["em"].append(float(max(normalize(answer, cfg.normalization) == normalize(g, cfg.normalization)
                                                 for g in golds)))
            per_system[s]["f1"].append(max(f1(answer, g, cfg.normalization) for g in golds))
    seconds = time.perf_counter() - t0

    summary = {s: {m: bootstrap_mean(v[m], cfg.bootstrap, cfg.seed).as_dict() for m in ("em", "f1")}
               for s, v in per_system.items()}
    base, cand = cfg.systems[0], cfg.systems[-1]
    diffs = {m: paired_difference(per_system[cand][m], per_system[base][m], cfg.bootstrap, cfg.seed).as_dict()
             for m in ("em", "f1")}
    report = {"kind": "answer-eval", "config": cfg.model_dump(), "config_hash": config_hash(cfg),
              "questions": len(questions), "seconds": seconds, "summary": summary,
              "difference": {"candidate": cand, "baseline": base, **diffs},
              "answers": [{"id": q["id"], "question": q["question"], "gold": q["answer"],
                           **{s: per_system[s]["answers"][i] for s in cfg.systems}} for i, q in enumerate(questions)]}
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"answers-{report['config_hash']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hsi-answer-eval", description="Answer quality from retrieved passages.")
    ap.add_argument("--config", required=True)
    rep = run(AnswerEvalConfig.model_validate(load_yaml(ap.parse_args(argv).config)))
    print(json.dumps({"summary": rep["summary"], "difference": rep["difference"], "path": rep["path"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
