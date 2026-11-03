"""Aho–Corasick matcher over token sequences, used to find entity-title mentions in passages (C4).

Matching works on normalized tokens, so "Mark Zuckerberg's" matches the title "Mark Zuckerberg"
and partial words never match.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from hsi_core.text import tokens


@dataclass(frozen=True, slots=True)
class Match:
    key: str            # the pattern key (e.g. normalized title)
    start: int          # token index of the first matched token
    end: int            # token index after the last matched token


class TokenAhoCorasick:
    def __init__(self) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[tuple[str, int]]] = [[]]   # (key, length in tokens)
        self._built = False

    def add(self, pattern_tokens: Iterable[str], key: str) -> None:
        if self._built:
            raise RuntimeError("cannot add patterns after build()")
        node = 0
        toks = list(pattern_tokens)
        if not toks:
            return
        for t in toks:
            nxt = self._goto[node].get(t)
            if nxt is None:
                nxt = len(self._goto)
                self._goto[node][t] = nxt
                self._goto.append({})
                self._fail.append(0)
                self._out.append([])
            node = nxt
        self._out[node].append((key, len(toks)))

    def build(self) -> TokenAhoCorasick:
        q: deque[int] = deque()
        for nxt in self._goto[0].values():
            self._fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for t, s in self._goto[r].items():
                q.append(s)
                f = self._fail[r]
                while f and t not in self._goto[f]:
                    f = self._fail[f]
                cand = self._goto[f].get(t, 0)
                self._fail[s] = cand if cand != s else 0
                self._out[s] = self._out[s] + self._out[self._fail[s]]
        self._built = True
        return self

    def find(self, toks: list[str]) -> list[Match]:
        if not self._built:
            raise RuntimeError("call build() first")
        node = 0
        found: list[Match] = []
        for i, t in enumerate(toks):
            while node and t not in self._goto[node]:
                node = self._fail[node]
            node = self._goto[node].get(t, 0)
            for key, length in self._out[node]:
                found.append(Match(key=key, start=i - length + 1, end=i + 1))
        return found


def longest_non_overlapping(matches: list[Match]) -> list[Match]:
    """Keep the longest matches, left to right, so "New York Times" wins over "New York"."""
    chosen: list[Match] = []
    taken: set[int] = set()
    for m in sorted(matches, key=lambda m: (-(m.end - m.start), m.start)):
        span = set(range(m.start, m.end))
        if span & taken:
            continue
        chosen.append(m)
        taken |= span
    return sorted(chosen, key=lambda m: m.start)


def title_tokens(title: str) -> list[str]:
    return tokens(title)
