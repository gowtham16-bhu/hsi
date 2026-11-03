"""Capitalized-span mention extractor whose language knowledge is learned from the corpus.

No word lists. `fit(texts)` measures, for every word, how often it is written capitalized:

    cap_ratio(w) = capitalized occurrences of w / all occurrences of w

A token is part of a name when it starts with an uppercase letter (Unicode-aware) and
cap_ratio ≥ `min_cap_ratio`. Function words that are capitalized only at sentence starts get a
low ratio and drop out by themselves, in any language that uses letter case.

Two name runs A, B separated by 1..`max_join_tokens` lower-case tokens are merged when the merged
span occurs at least `min_join_count` times and accounts for at least `min_join_share` of the
occurrences of the rarer of A and B (a compound like "University of Oxford" is rarely split; two
names joined by a conjunction usually also occur apart).

A token with a punctuation mark within its last `max_clitic_chars` + 1 characters (a possessive or
clitic) is reduced to the part before the mark when that part is seen as a standalone name token
at least `min_join_count` times.

Every threshold comes from configuration; none has a built-in default.
"""
from __future__ import annotations

import itertools
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _clean(token: str) -> str:
    """Strip leading and trailing punctuation (Unicode categories P*)."""
    start, end = 0, len(token)
    while start < end and _is_punct(token[start]):
        start += 1
    while end > start and _is_punct(token[end - 1]):
        end -= 1
    return token[start:end]


class Word(NamedTuple):
    text: str            # token with leading/trailing punctuation removed
    offset: int          # character offset in the source text
    open_before: bool    # punctuation directly before the token (a boundary)
    close_after: bool    # punctuation directly after the token (a boundary)


def _words(text: str) -> list[Word]:
    """Whitespace tokens; punctuation at a token's edge marks a boundary (Unicode categories P*)."""
    out: list[Word] = []
    pos = 0
    for raw in text.split():
        i = text.index(raw, pos)
        pos = i + len(raw)
        w = _clean(raw)
        if w:
            out.append(Word(w, i, _is_punct(raw[0]), _is_punct(raw[-1])))
    return out


class CapitalizedMentionExtractor:
    def __init__(self, *, min_cap_ratio: float, min_join_count: int, min_join_share: float, max_join_tokens: int,
                 max_words: int, min_chars: int, max_clitic_chars: int) -> None:
        if not 0.0 < min_cap_ratio <= 1.0 or min_join_count < 1 or not 0.0 < min_join_share <= 1.0 \
                or max_join_tokens < 0 \
                or max_words < 1 or min_chars < 1 or max_clitic_chars < 0:
            raise ValueError("invalid mention extractor parameters")
        self.min_cap_ratio = min_cap_ratio
        self.min_join_count = min_join_count
        self.min_join_share = min_join_share
        self.max_join_tokens = max_join_tokens
        self.max_words = max_words
        self.min_chars = min_chars
        self.max_clitic_chars = max_clitic_chars
        self.name = (f"capitalized:r{min_cap_ratio}:j{min_join_count}s{min_join_share}x{max_join_tokens}"
                     f":w{max_words}:c{min_chars}:k{max_clitic_chars}")
        self._cap: Counter[str] = Counter()
        self._all: Counter[str] = Counter()
        self._standalone: Counter[str] = Counter()
        self._joined: Counter[str] = Counter()
        self._run_counts: Counter[str] = Counter()
        self._fitted = False

    # learning -----------------------------------------------------------------------------
    def fit(self, texts: Iterable[str]) -> CapitalizedMentionExtractor:
        texts = list(texts)
        for t in texts:
            for w in _words(t):
                self._all[w.text.lower()] += 1
                if w.text[0].isupper():
                    self._cap[w.text.lower()] += 1
        self._fitted = True
        for t in texts:
            words = _words(t)
            for w in words:
                if self._namey(w.text):
                    self._standalone[w.text.lower()] += 1
            runs = self._runs(words)
            for a, b in runs:
                self._run_counts[self._span(words, a, b).lower()] += 1
            for span in self._join_candidates(words, runs):
                self._joined[span] += 1
        return self

    def cap_ratio(self, word: str) -> float:
        n = self._all.get(word.lower(), 0)
        return self._cap.get(word.lower(), 0) / n if n else 0.0

    # extraction ---------------------------------------------------------------------------
    def _namey(self, w: str) -> bool:
        return w[0].isupper() and self.cap_ratio(w) >= self.min_cap_ratio

    def _head(self, w: str) -> str:
        for i in range(len(w) - 1, max(0, len(w) - 2 - self.max_clitic_chars), -1):
            if _is_punct(w[i]):
                head = w[:i]
                if head and self._standalone.get(head.lower(), 0) >= self.min_join_count:
                    return head
        return w

    def _runs(self, words: list[Word]) -> list[tuple[int, int]]:
        """Maximal [start, end) word-index ranges of name tokens, never crossing punctuation."""
        runs: list[tuple[int, int]] = []
        i = 0
        while i < len(words):
            if self._namey(words[i].text):
                j = i + 1
                while j < len(words) and self._namey(words[j].text) \
                        and not words[j - 1].close_after and not words[j].open_before:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        return runs

    def _gap_ok(self, words: list[Word], a_end: int, b_start: int) -> bool:
        gap = words[a_end:b_start]
        if not 0 < len(gap) <= self.max_join_tokens or words[a_end - 1].close_after or words[b_start].open_before:
            return False
        return all(not g.text[0].isupper() and not g.open_before and not g.close_after for g in gap)

    def _span(self, words: list[Word], start: int, end: int) -> str:
        return " ".join(self._head(w.text) for w in words[start:end])

    def _join_candidates(self, words: list[Word], runs: list[tuple[int, int]]) -> list[str]:
        return [self._span(words, a[0], b[1]).lower()
                for a, b in itertools.pairwise(runs) if self._gap_ok(words, a[1], b[0])]

    def _associated(self, words: list[Word], a: tuple[int, int], b: tuple[int, int]) -> bool:
        joined = self._joined.get(self._span(words, a[0], b[1]).lower(), 0)
        if joined < self.min_join_count:
            return False
        rarer = min(self._run_counts.get(self._span(words, *a).lower(), 0),
                    self._run_counts.get(self._span(words, *b).lower(), 0))
        return rarer > 0 and joined / rarer >= self.min_join_share

    def extract(self, text: str) -> list[tuple[str, float]]:
        if not self._fitted:
            raise RuntimeError("call fit(corpus texts) before extract()")
        words = _words(text)
        runs = self._runs(words)
        merged: list[tuple[int, int]] = []
        for run in runs:
            if merged and self._gap_ok(words, merged[-1][1], run[0]) and self._associated(words, merged[-1], run):
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        n = max(1, len(text))
        out: dict[str, tuple[str, float]] = {}
        for start, end in merged:
            surface = self._span(words, start, end)
            if len(surface) < self.min_chars or end - start > self.max_words:
                continue
            out.setdefault(surface.lower(), (surface, 1.0 - words[start].offset / n))
        return list(out.values())
