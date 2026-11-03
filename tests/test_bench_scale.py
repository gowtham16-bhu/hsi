import json
from pathlib import Path

import numpy as np

from hsi_harness.bench_scale import BenchConfig, run


def test_bench_scale_runs_and_reports_each_size(fixtures, tmp_path):
    """Wiring only (hashing encoder, fixture data): not evidence for any claim (CLAUDE.md rule 5)."""
    from hsi_core.transitions import FEATURES, LinearTransition
    rng = np.random.default_rng(0)
    LinearTransition(FEATURES, np.zeros(len(FEATURES)), np.ones(len(FEATURES)),
                     rng.normal(size=len(FEATURES))).save(tmp_path / "t.json", {"note": "test"})
    cfg = BenchConfig(
        questions={"dataset": "musique", "data_file": str(fixtures / "musique_mini.jsonl")},
        question_count=3, seed=0, exclude_samples=[],
        pool=[{"dataset": "2wiki", "data_file": str(fixtures / "2wiki_mini.json")}], sizes=[11, 14],
        encoder={"plugin": "hashing", "options": {"dim": 256}}, vector_store={"plugin": "faiss"},
        cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "out"),
        graph=dict(fan_out=8, hub_degree_cap=50, bidirectional=True, co_mention=True, weight_multi_token=0.9,
                   weight_single_token=0.6, reverse_factor=0.85, co_mention_multi_token=0.7,
                   co_mention_single_token=0.45),
        transition_model=str(tmp_path / "t.json"),
        ppr={"seeds": 2, "restart": 0.5, "iterations": 50, "tolerance": 1e-9, "beta": 0.0, "rank_by": "total",
             "keep_seeds": 0},
        push={"seeds": 2, "restart": 0.5, "epsilon": 1e-6, "beta": 0.0, "max_pushes": 10_000},
        global_ppr_max_size=11, k=5)
    rep = run(cfg)
    assert [r["passages"] for r in rep["rows"]] == [11, 14] and rep["llm_calls"] == 0
    assert rep["rows"][0]["latency_ms"]["global"] is not None and rep["rows"][1]["latency_ms"]["global"] is None
    assert json.loads(Path(rep["paths"]["json"]).read_text())["config_hash"] == rep["config_hash"]
