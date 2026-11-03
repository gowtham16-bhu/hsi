"""Plugin protocols. The core reaches every plugin only through these (constitution P1, P12)."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from hsi_core.model import Chain, Edge, Passage, Shortcut


@runtime_checkable
class Encoder(Protocol):
    """Turns text into L2-normalized float32 vectors."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str], kind: Literal["query", "passage"]) -> np.ndarray:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Exact or approximate inner-product search over integer ids."""

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, ids), each of shape (n_queries, k); missing results have id -1."""
        ...

    def delete(self, ids: np.ndarray) -> int: ...

    def __len__(self) -> int: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Durable storage for passages, graph, shortcuts and run records."""

    def migrate(self) -> None: ...

    def write_passages(self, corpus_id: str, passages: Iterable[Passage]) -> int: ...

    def write_edges(self, corpus_id: str, edges: Iterable[Edge]) -> int: ...

    def write_shortcuts(self, corpus_id: str, shortcuts: Iterable[Shortcut]) -> int: ...

    def record_run(self, run: dict[str, Any]) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class MentionExtractor(Protocol):
    """Finds entity mentions in text. Returns (surface form, salience in [0, 1]) pairs.
    `fit` lets an extractor learn its statistics from the corpus before extracting."""

    name: str

    def fit(self, texts: Iterable[str]) -> MentionExtractor: ...

    def extract(self, text: str) -> list[tuple[str, float]]: ...


class ChainScorer(Protocol):
    """Scores chains for a query; used by online expansion."""

    def __call__(self, query_vec: np.ndarray, chains: Sequence[Chain]) -> np.ndarray: ...
