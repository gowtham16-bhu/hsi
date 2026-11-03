"""Loaders for MuSiQue, 2WikiMultiHopQA and HotpotQA into canonical passages + gold chains.

Corpus protocol (as in HippoRAG 2): sample N questions, the corpus is the union of all their
paragraphs (gold and distractors), deduplicated by (title, text).

Expected files (download instructions in docs/datasets.md):
  musique   musique_ans_v1.0_dev.jsonl    one JSON object per line
  2wiki     dev.json                      JSON list
  hotpotqa  hotpot_dev_distractor_v1.json JSON list
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

from hsi_core.model import PUBLIC, Passage


@dataclass(frozen=True)
class QAItem:
    id: str
    dataset: str
    question: str
    answer: str
    hops: int
    gold: tuple[str, ...]      # passage ids, in hop order when `ordered`
    ordered: bool


@dataclass(frozen=True)
class Dataset:
    name: str
    passages: list[Passage]
    questions: list[QAItem]


class _Corpus:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.by_key: dict[tuple[str, str], Passage] = {}

    def add(self, title: str, text: str) -> str:
        title, text = title.strip(), " ".join(text.split())
        key = (title, text)
        p = self.by_key.get(key)
        if p is None:
            doc_id = hashlib.sha256(f"{self.dataset}\x1f{title}\x1f{text}".encode()).hexdigest()[:16]
            p = Passage.build(doc_id=doc_id, text=text, title=title or None, acl=PUBLIC)
            self.by_key[key] = p
        return p.id


def _sample(records: list[dict], n: int | None, seed: int, key: str,
            exclude: set[str] | None = None) -> list[dict]:
    records = sorted((r for r in records if r[key] not in (exclude or ())), key=lambda r: r[key])
    if n is None or n >= len(records):
        return records
    return sorted(random.Random(seed).sample(records, n), key=lambda r: r[key])


def _read_json_or_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


_MUSIQUE_HOPS = re.compile(r"^(\d)hop")


def load_musique(path: str, n: int | None, seed: int, exclude: set[str] | None = None) -> Dataset:
    rows = [r for r in _read_json_or_jsonl(path) if r.get("answerable", True)]
    rows = _sample(rows, n, seed, "id", exclude)
    corpus = _Corpus("musique")
    items = []
    for r in rows:
        idx_to_pid = {p["idx"]: corpus.add(p["title"], p["paragraph_text"]) for p in r["paragraphs"]}
        order = [d["paragraph_support_idx"] for d in r["question_decomposition"]]
        if any(i is None for i in order):
            continue
        gold = tuple(dict.fromkeys(idx_to_pid[i] for i in order))
        m = _MUSIQUE_HOPS.match(r["id"])
        hops = int(m.group(1)) if m else len(gold)
        # Linear compositions (2hop, 3hop1, 4hop1) are paths in hop order; the others are trees.
        linear = hops == 2 or r["id"].startswith(f"{hops}hop1")
        items.append(QAItem(r["id"], "musique", r["question"], r["answer"], hops, gold, linear))
    return Dataset("musique", list(corpus.by_key.values()), items)


def _context_dataset(name: str, rows: Iterable[dict]) -> Dataset:
    corpus = _Corpus(name)
    items = []
    for r in rows:
        title_to_pid = {}
        for title, sents in r["context"]:
            title_to_pid[title] = corpus.add(title, " ".join(s.strip() for s in sents))
        sup = list(dict.fromkeys(t for t, _ in r["supporting_facts"]))
        gold = tuple(title_to_pid[t] for t in sup if t in title_to_pid)
        if len(gold) < 2:
            continue
        items.append(QAItem(r["_id"], name, r["question"], str(r["answer"]), len(gold), gold, False))
    return Dataset(name, list(corpus.by_key.values()), items)


def load_2wiki(path: str, n: int | None, seed: int, exclude: set[str] | None = None) -> Dataset:
    return _context_dataset("2wiki", _sample(_read_json_or_jsonl(path), n, seed, "_id", exclude))


def load_hotpotqa(path: str, n: int | None, seed: int, exclude: set[str] | None = None) -> Dataset:
    return _context_dataset("hotpotqa", _sample(_read_json_or_jsonl(path), n, seed, "_id", exclude))


LOADERS = {"musique": load_musique, "2wiki": load_2wiki, "hotpotqa": load_hotpotqa}


def load(name: str, path: str, n: int | None, seed: int, exclude: set[str] | None = None) -> Dataset:
    try:
        loader = LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(LOADERS)}") from None
    return loader(path, n, seed, exclude)
