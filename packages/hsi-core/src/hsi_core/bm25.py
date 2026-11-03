"""Okapi BM25 over passages (keyword baseline and the lexical half of hybrid retrieval)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from hsi_core.text import tokens


class BM25Index:
    def __init__(self, k1: float, b: float, stopwords: frozenset[str] = frozenset()) -> None:
        """`stopwords` come from configuration; IDF already down-weights common terms."""
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.stopwords = frozenset(stopwords)
        self._ids: list[str] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._doc_len = np.zeros(0)
        self._avgdl = 0.0

    def _terms(self, text: str) -> list[str]:
        return [t for t in tokens(text) if t not in self.stopwords]

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self._ids = list(ids)
        tmp: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lens = np.zeros(len(texts), dtype=np.float64)
        for i, t in enumerate(texts):
            toks = self._terms(t)
            lens[i] = len(toks)
            for term, tf in Counter(toks).items():
                tmp[term][0].append(i)
                tmp[term][1].append(tf)
        self._postings = {t: (np.asarray(d, dtype=np.int64), np.asarray(f, dtype=np.float64))
                          for t, (d, f) in tmp.items()}
        self._doc_len = lens
        self._avgdl = float(lens.mean()) if len(lens) else 0.0
        return self

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        n = len(self._ids)
        if n == 0 or k <= 0:
            return []
        scores = np.zeros(n, dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self._doc_len / max(self._avgdl, 1e-9))
        for term in set(self._terms(query)):
            post = self._postings.get(term)
            if post is None:
                continue
            docs, tf = post
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += idf * tf * (self.k1 + 1) / (tf + norm[docs])
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]
