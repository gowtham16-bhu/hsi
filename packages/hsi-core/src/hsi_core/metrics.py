"""Evaluation metrics (specs/04): full-chain recall@k, bootstrap intervals, paired differences."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def full_chain_recall(retrieved: Sequence[str], gold: Sequence[str], k: int) -> bool:
    """True when every gold passage is among the first k retrieved passages."""
    if not gold:
        raise ValueError("empty gold chain")
    return set(gold) <= set(retrieved[:k])


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "low": self.low, "high": self.high, "n": self.n}


def bootstrap_mean(values: Sequence[float], n_boot: int = 1000, seed: int = 0,
                   alpha: float = 0.05) -> Estimate:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return Estimate(float(x.mean()), float(np.quantile(means, alpha / 2)),
                    float(np.quantile(means, 1 - alpha / 2)), int(x.size))


def paired_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> Estimate:
    """Bootstrap interval of mean(a − b) over the same questions."""
    xa, xb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if xa.shape != xb.shape:
        raise ValueError("paired samples must have the same length")
    return bootstrap_mean(xa - xb, n_boot=n_boot, seed=seed, alpha=alpha)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")
