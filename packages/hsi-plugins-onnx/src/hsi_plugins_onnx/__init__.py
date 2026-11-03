"""ONNX Runtime encoder for BERT-style embedding models.

Loads the ONNX graph (`onnx_file`, from config) and `tokenizer.json` from a Hugging Face repo id or a
local directory. Pooling and the padding token are read from the model's own files; the query
instruction comes from config. Vectors are L2-normalized. In air-gapped setups pass a local directory.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def _resolve(model: str, filename: str) -> str:
    if os.path.isdir(model):
        return os.path.join(model, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, filename)


class OnnxEncoder:
    def __init__(self, *, model: str, onnx_file: str, max_seq_length: int, batch_size: int, query_instruction: str,
                 threads: int | None = None, pooling: Literal["cls", "mean"] | None = None,
                 providers: list[str] | None = None) -> None:
        """`query_instruction`: the prefix the model card prescribes for queries ("" if none).
        `pooling`: read from the model's 1_Pooling/config.json unless given here.
        `providers`: ONNX Runtime execution providers in priority order (e.g. CoreMLExecutionProvider on a
        Mac); CPU when not given. Providers the installed runtime lacks are refused, not silently skipped."""
        self._tok = Tokenizer.from_file(_resolve(model, "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_seq_length)
        with open(_resolve(model, "tokenizer_config.json"), encoding="utf-8") as f:
            pad_token = json.load(f).get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        pad_id = self._tok.token_to_id(pad_token) if pad_token else None
        if pad_id is None:
            raise ValueError(f"{model}: tokenizer_config.json defines no usable pad_token")
        self._tok.enable_padding(pad_id=pad_id, pad_token=pad_token)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        chosen = providers or ["CPUExecutionProvider"]
        missing = [p for p in chosen if p not in ort.get_available_providers()]
        if missing:
            raise ValueError(f"ONNX Runtime here lacks providers {missing}; available: {ort.get_available_providers()}")
        self._sess = ort.InferenceSession(_resolve(model, onnx_file), opts, providers=chosen)
        self._inputs = {i.name for i in self._sess.get_inputs()}
        if pooling is None:
            with open(_resolve(model, "1_Pooling/config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("pooling_mode_cls_token"):
                pooling = "cls"
            elif cfg.get("pooling_mode_mean_tokens"):
                pooling = "mean"
            else:
                raise ValueError(f"{model}: unsupported pooling in 1_Pooling/config.json; pass pooling=")
        self._pooling = pooling
        self._qi = query_instruction
        self._batch = batch_size
        self.name = f"onnx:{model}:{max_seq_length}:{pooling}"
        self.dim = int(self._embed(["dimension probe"]).shape[1])

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self._tok.encode_batch(texts)
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._sess.run(None, feed)[0]
        if self._pooling == "cls":
            pooled = hidden[:, 0]
        else:
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        pooled = pooled.astype(np.float32)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    def encode(self, texts: Sequence[str], kind: Literal["query", "passage"]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        items = [self._qi + t for t in texts] if kind == "query" else list(texts)
        # sort by length so batches pad less, then restore the original order
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for s in range(0, len(order), self._batch):
            idx = order[s:s + self._batch]
            out[idx] = self._embed([items[i] for i in idx])
        return out
