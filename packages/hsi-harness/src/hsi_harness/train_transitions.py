"""Train learned walk transitions from gold hop chains (decision 0007). No LLM.

    hsi-train-transitions --config configs/train/transitions-musique.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.build import GraphSection, build_compact
from hsi_core.config import PluginSpec, config_hash, load_yaml
from hsi_core.encoding import CachedEncoder
from hsi_core.index import ShortcutIndex
from hsi_core.transitions import FEATURES, edge_features, fit_listwise, listwise_loss
from hsi_harness.bench_scale import PoolSource
from hsi_harness.datasets import load
from hsi_harness.gate0 import SampleRef


class TrainSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    l2: float
    lr: float
    epochs: int
    seed: int
    validation_fraction: float
    baseline_beta: float            # rule-based walk to compare against: log w + β·cos(q, v)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    data_file: str
    sample_size: int
    seed: int
    exclude_sample: SampleRef | None = None
    encoder: PluginSpec
    vector_store: PluginSpec
    cache_dir: str
    graph: GraphSection
    training: TrainSection
    output: str
    distractors: list[PoolSource] = []   # decision 0016: train on a graph as large as the one served
    corpus_size: int | None = None       # with distractors: total passages (questions' own + seeded sample)


def run(cfg: TrainConfig) -> dict:
    exclude = None
    if cfg.exclude_sample:
        exclude = {q.id for q in load(cfg.dataset, cfg.data_file, cfg.exclude_sample.size,
                                      cfg.exclude_sample.seed).questions}
    ds = load(cfg.dataset, cfg.data_file, cfg.sample_size, cfg.seed, exclude=exclude)
    passages = list(ds.passages)
    if cfg.distractors:
        if cfg.corpus_size is None:
            raise ValueError("distractors need corpus_size")
        own = {p.id for p in passages}
        pool: dict[str, Any] = {}
        for src in cfg.distractors:
            for p in load(src.dataset, src.data_file, None, cfg.seed).passages:
                if p.id not in own:
                    pool.setdefault(p.id, p)
        order = sorted(pool)
        random.Random(cfg.seed).shuffle(order)
        if cfg.corpus_size - len(passages) > len(order):
            raise ValueError("corpus_size exceeds the passages available")
        passages += [pool[i] for i in order[: cfg.corpus_size - len(passages)]]
    enc = CachedEncoder(registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options), cfg.cache_dir)
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", cfg.vector_store.plugin,
                                                     **cfg.vector_store.options)).build(passages)
    ids = [e[0] for e in dense.entries]
    pos = {pid: i for i, pid in enumerate(ids)}
    vecs = np.stack([dense.passage_vecs[p] for p in ids]).astype(np.float32)
    csr, _stats = build_compact(cfg.graph, passages, ids, vecs, vector_of=dense.raw_vecs.__getitem__)
    qvecs = enc.encode([q.question for q in ds.questions], "query")
    groups: list[tuple[np.ndarray, int, int]] = []        # (features, gold index, question index)
    reachable = transitions = 0
    for qi, (q, qv) in enumerate(zip(ds.questions, qvecs, strict=True)):
        if not q.ordered:
            continue
        in_q = None
        q32 = np.asarray(qv, dtype=np.float32)
        for u, v in zip(q.gold, q.gold[1:], strict=False):
            transitions += 1
            sl = csr.edges_of(pos[u])
            dst = csr.dst[sl]
            hit = np.nonzero(dst == pos[v])[0]
            if len(hit) == 0 or len(dst) < 2:
                continue
            reachable += 1
            if in_q is None:
                in_q = csr.bridges_in(q.question)
            # features of u's out-edges only: the same function the walks use (decision 0007)
            phi = edge_features(csr, sl, (vecs[dst] @ q32).astype(np.float64), float(vecs[pos[u]] @ q32), in_q)
            groups.append((phi, int(hit[0]), qi))
    rng = random.Random(cfg.training.seed)
    qids = sorted({g[2] for g in groups})
    rng.shuffle(qids)
    n_val = int(len(qids) * cfg.training.validation_fraction)
    val_q = set(qids[:n_val])
    train = [(g, y) for g, y, qi in groups if qi not in val_q]
    val = [(g, y) for g, y, qi in groups if qi in val_q]
    model = fit_listwise(train, cfg.training.l2, cfg.training.lr, cfg.training.epochs, cfg.training.seed)
    # rule-based comparison on the same validation groups: score = log w + β·cos(q, v)
    i_w, i_qv = FEATURES.index("edge_weight"), FEATURES.index("cos_q_v")
    rule_acc = float(np.mean([np.argmax(np.log(g[:, i_w]) + cfg.training.baseline_beta * g[:, i_qv]) == y
                              for g, y in val])) if val else float("nan")
    tr_ce, tr_acc = listwise_loss(model, train)
    va_ce, va_acc = listwise_loss(model, val) if val else (float("nan"), float("nan"))
    meta = {"config_hash": config_hash(cfg), "questions": len(ds.questions), "corpus_passages": len(passages),
            "gold_transitions": transitions,
            "reachable_in_capped_graph": reachable, "train_groups": len(train), "validation_groups": len(val),
            "train": {"cross_entropy": tr_ce, "top1": tr_acc},
            "validation": {"cross_entropy": va_ce, "top1": va_acc, "rule_based_top1": rule_acc},
            "theta": dict(zip(FEATURES, model.theta.round(4).tolist(), strict=True)),
            "mean_out_degree": float(np.mean([len(g) for g, _ in train])) if train else 0.0}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.output, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hsi-train-transitions", description="Train learned walk transitions.")
    ap.add_argument("--config", required=True)
    cfg = TrainConfig.model_validate(load_yaml(ap.parse_args(argv).config))
    print(json.dumps(run(cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
