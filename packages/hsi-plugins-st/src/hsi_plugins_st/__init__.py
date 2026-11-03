"""Sentence-transformers encoder. Weights are loaded from a local path or the Hugging Face cache.

Set HF_HUB_OFFLINE=1 in air-gapped deployments and point `model` at a local directory.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np


class SentenceTransformerEncoder:
    def __init__(self, *, model: str, device: str, batch_size: int, max_seq_length: int,
                 query_instruction: str) -> None:
        """`query_instruction` is the prefix the model card prescribes for queries ("" if none)."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self._model.max_seq_length = max_seq_length
        self._batch = batch_size
        self._qi = query_instruction
        self.name = f"st:{model}:{max_seq_length}"
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: Sequence[str], kind: Literal["query", "passage"]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        items = [self._qi + t for t in texts] if kind == "query" else list(texts)
        vecs = self._model.encode(items, batch_size=self._batch, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
