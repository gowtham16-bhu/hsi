"""Train learned index-time edge selection (decision 0010). No LLM.

For each gold hop u → v whose target is among u's candidate edges (before the fan-out cap), learn a
question-independent score so that v ranks high. Reports the share of gold hops kept at the
deployment fan-out, learned vs prior, on held-out questions.

    hsi-train-edges --config configs/train/edges-musique-v1.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from hsi_core import registry
from hsi_core.config import PluginSpec, config_hash, load_yaml
from hsi_core.encoding import CachedEncoder
from hsi_core.index import ShortcutIndex
from hsi_core.transitions import EDGE_FEATURES, candidate_features, fit_listwise, listwise_loss
from hsi_harness.datasets import load
from hsi_harness.gate0 import GraphSection, SampleRef, build_graph


class EdgeTrainSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    l2: float
    lr: float
    epochs: int
    seed: int
    validation_fraction: float
    deploy_fan_out: int


class EdgeTrainConfig(BaseModel):
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
    training: EdgeTrainSection
    output: str


def run(cfg: EdgeTrainConfig) -> dict:
    exclude = None
    if cfg.exclude_sample:
        exclude = {q.id for q in load(cfg.dataset, cfg.data_file, cfg.exclude_sample.size,
                                      cfg.exclude_sample.seed).questions}
    ds = load(cfg.dataset, cfg.data_file, cfg.sample_size, cfg.seed, exclude=exclude)
    enc = CachedEncoder(registry.load("encoder", cfg.encoder.plugin, **cfg.encoder.options), cfg.cache_dir)
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", cfg.vector_store.plugin,
                                                     **cfg.vector_store.options)).build(ds.passages)
    vec = dense.raw_vecs.__getitem__
    graph = build_graph(cfg.graph, ds.passages, keep_candidates=True)
    cand = graph.candidates or {}
    groups, prior_rank = [], []
    hops = 0
    for qi, q in enumerate(ds.questions):
        if not q.ordered:
            continue
        for u, v in zip(q.gold, q.gold[1:], strict=False):
            hops += 1
            items = sorted(cand.get(u, {}).items())
            dsts = [d for d, _ in items]
            if v not in dsts or len(items) < 2:
                groups.append((None, -1, qi))
                prior_rank.append(10**9)
                continue
            c = [(d, w, b, s, t) for d, (_, w, b, s, t) in items]
            phi = candidate_features(vec(u), c, graph.participants, vec)
            y = dsts.index(v)
            prior = np.array([p for _, (p, *_rest) in items])
            prior_rank.append(int((prior > prior[y]).sum()))
            groups.append((phi, y, qi))
    rng = random.Random(cfg.training.seed)
    qids = sorted({qi for _, _, qi in groups})
    rng.shuffle(qids)
    val_q = set(qids[: int(len(qids) * cfg.training.validation_fraction)])
    train = [(g, y) for g, y, qi in groups if g is not None and qi not in val_q]
    model = fit_listwise(train, cfg.training.l2, cfg.training.lr, cfg.training.epochs, cfg.training.seed,
                         features=EDGE_FEATURES)
    m = cfg.training.deploy_fan_out
    val_idx = [i for i, (_, _, qi) in enumerate(groups) if qi in val_q]
    kept_prior = sum(prior_rank[i] < m for i in val_idx)
    kept_learned = 0
    for i in val_idx:
        g, y, _ = groups[i]
        if g is not None:
            s = model.scores(g)
            kept_learned += int((s > s[y]).sum() < m)
    val_groups = [(g, y) for i in val_idx for g, y, _ in [groups[i]] if g is not None]
    ce, top1 = listwise_loss(model, val_groups) if val_groups else (float("nan"), float("nan"))
    meta = {"config_hash": config_hash(cfg), "gold_hops": hops,
            "hops_with_candidate_edge": sum(g is not None for g, _, _ in groups),
            "validation_hops": len(val_idx), "deploy_fan_out": m,
            "validation_kept_at_fan_out": {"prior": kept_prior / max(1, len(val_idx)),
                                           "learned": kept_learned / max(1, len(val_idx))},
            "validation_cross_entropy": ce, "validation_top1": top1,
            "theta": dict(zip(EDGE_FEATURES, model.theta.round(4).tolist(), strict=True)),
            "mean_candidates": float(np.mean([len(g) for g, _ in train])) if train else 0.0}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.output, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hsi-train-edges", description="Train learned index-time edge selection.")
    ap.add_argument("--config", required=True)
    print(json.dumps(run(EdgeTrainConfig.model_validate(load_yaml(ap.parse_args(argv).config))), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
