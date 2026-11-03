"""Feature-hashing encoder: a real, deterministic lexical encoder that needs no model weights.

Each token and adjacent token pair is hashed into `dim` buckets with a sign, weighted
by sublinear term frequency, then L2-normalized. It is a weak semantic encoder, so results
with it are never reported as thesis evidence; it exists for pipeline tests and offline smoke runs.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np

from hsi_core.text import tokens


class HashingEncoder:
    def __init__(self, dim: int = 1024, bigrams: bool = True) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim
        self.bigrams = bigrams
        self.name = f"hashing:{dim}:{int(bigrams)}"

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def encode(self, texts: Sequence[str], kind: Literal["query", "passage"]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokens(t)
            feats = Counter(toks)
            if self.bigrams:
                feats.update(f"{a}_{b}" for a, b in itertools.pairwise(toks))
            for f, tf in feats.items():
                j, s = self._bucket(f)
                out[i, j] += s * (1.0 + np.log(tf))
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out
