"""FAISS vector store: exact inner-product search with stable integer ids."""
from __future__ import annotations

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim
        self._index: faiss.IndexIDMap2 | None = None
        if dim is not None:
            self._create(dim)

    def _create(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if vectors.ndim != 2 or len(ids) != len(vectors):
            raise ValueError("vectors must be (n, d) and match ids")
        if self._index is None:
            self._create(vectors.shape[1])
        if vectors.shape[1] != self._dim:
            raise ValueError(f"dimension {vectors.shape[1]} != index dimension {self._dim}")
        self._index.add_with_ids(vectors, ids)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self._index is None or self._index.ntotal == 0:
            n = len(queries)
            return np.zeros((n, k), dtype=np.float32), -np.ones((n, k), dtype=np.int64)
        return self._index.search(queries, k)

    def delete(self, ids: np.ndarray) -> int:
        if self._index is None:
            return 0
        return int(self._index.remove_ids(np.ascontiguousarray(ids, dtype=np.int64)))

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def save(self, path: str) -> None:
        if self._index is None:
            raise RuntimeError("nothing to save")
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        idx = faiss.read_index(path)
        self._index = idx
        self._dim = idx.d
