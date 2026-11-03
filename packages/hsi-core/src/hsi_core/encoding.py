"""Encoder helpers: passage text rendering and a persistent on-disk cache around any Encoder."""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Literal

import numpy as np

from hsi_core.interfaces import Encoder
from hsi_core.model import Passage


def passage_text(p: Passage) -> str:
    return f"{p.title}: {p.text}" if p.title else p.text


def chain_text(passages: Sequence[Passage]) -> str:
    return "\n".join(passage_text(p) for p in passages)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class CachedEncoder:
    """Wraps an Encoder; vectors are stored in npz shards keyed by sha256(model, kind, text)."""

    def __init__(self, inner: Encoder, cache_dir: str) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", inner.name)
        self._dir = os.path.join(cache_dir, slug)
        os.makedirs(self._dir, exist_ok=True)
        self._mem: dict[str, np.ndarray] = {}
        for fn in sorted(os.listdir(self._dir)):
            if fn.endswith(".npz"):
                with np.load(os.path.join(self._dir, fn)) as z:
                    for k, v in zip(z["keys"], z["vecs"], strict=True):
                        self._mem[str(k)] = v
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha256(f"{self.name}\x1f{kind}\x1f{text}".encode()).hexdigest()

    def encode(self, texts: Sequence[str], kind: Literal["query", "passage"]) -> np.ndarray:
        keys = [self._key(t, kind) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in self._mem}.items())
        if missing:
            vecs = self.inner.encode([t for _, t in missing], kind)
            new_keys = np.asarray([k for k, _ in missing])
            for k, v in zip(new_keys, vecs, strict=True):
                self._mem[str(k)] = v
            shard = os.path.join(self._dir, f"shard-{len(os.listdir(self._dir)):06d}.npz")
            np.savez(shard, keys=new_keys, vecs=np.asarray(vecs, dtype=np.float32))
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._mem[k] for k in keys]).astype(np.float32)
