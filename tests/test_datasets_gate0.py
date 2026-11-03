import json
from pathlib import Path

import pytest

from hsi_harness.datasets import load
from hsi_harness.gate0 import Gate0Config, run


def test_musique_loader(mini_musique):
    assert len(mini_musique.questions) == 3            # unanswerable one skipped
    q3 = next(q for q in mini_musique.questions if q.hops == 3)
    titles = {p.id: p.title for p in mini_musique.passages}
    assert [titles[g] for g in q3.gold] == ["Instagram", "Facebook", "Mark Zuckerberg"]
    assert q3.ordered
    assert len({(p.title, p.text) for p in mini_musique.passages}) == len(mini_musique.passages)


@pytest.mark.parametrize("name,file", [("2wiki", "2wiki_mini.json"), ("hotpotqa", "hotpot_mini.json")])
def test_context_loaders(fixtures, name, file):
    ds = load(name, str(fixtures / file), None, 0)
    assert len(ds.questions) == 1 and ds.questions[0].hops == 2 and not ds.questions[0].ordered


GRAPH = dict(fan_out=8, hub_degree_cap=50, bidirectional=True, co_mention=True, weight_multi_token=0.9,
             weight_single_token=0.6, reverse_factor=0.85, co_mention_multi_token=0.7, co_mention_single_token=0.45)
SYSTEMS = [
    {"name": "bm25", "kind": "bm25", "options": {"k1": 1.2, "b": 0.75}},
    {"name": "dense", "kind": "dense"},
    {"name": "expand-vec", "kind": "expand", "options": {"seeds": 3, "beam": 2, "max_passages": 3, "variant": "vec"}},
    {"name": "expand-text", "kind": "expand", "options": {"seeds": 3, "beam": 2, "max_passages": 3, "variant": "text"}},
    {"name": "ppr", "kind": "ppr", "options": {"seeds": 3, "restart": 0.5, "iterations": 30, "tolerance": 1e-6,
                                               "beta": 5.0, "rank_by": "arrived", "keep_seeds": 2}},
    {"name": "hop-passages-2", "kind": "hop", "options": {"rounds": 2, "beam": 3, "per_round": 5, "update": "vec",
                                                          "vec_weight": 1.0}},
    {"name": "hop-shortcuts-2", "kind": "hop", "index": "shortcuts",
     "options": {"rounds": 2, "beam": 3, "per_round": 5, "update": "text", "vec_weight": 1.0}},
    {"name": "hsi", "kind": "dense", "index": "shortcuts"},
]
DECISION = {"name": "smoke", "all_of": [
    {"candidate": "hop-shortcuts-2", "baseline": "hop-passages-2", "slice": "all", "k": 10, "ci_low_at_least": -1.0}]}


def _cfg(fixtures, tmp_path, metadata_store=None):
    return Gate0Config(
        dataset="musique", data_file=str(fixtures / "musique_mini.jsonl"), sample_size=None, seed=0,
        encoder={"plugin": "hashing", "options": {"dim": 512}}, vector_store={"plugin": "faiss"},
        metadata_store=metadata_store, cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "out"),
        graph=GRAPH, chains={"max_passages": 3, "beam_level2": 3, "beam_level3": 2, "budget_ratio": None,
                             "key_mode": "mean"},
        systems=SYSTEMS, top_k_hits=20, passage_budgets=[5, 10], bootstrap=200, decision=DECISION)


def test_gate0_runner_offline(fixtures, tmp_path):
    rep = run(_cfg(fixtures, tmp_path))
    assert rep["llm_calls"] == 0
    assert set(rep["recall"]) == {s["name"] for s in SYSTEMS}
    assert rep["decision"]["status"] == "pass" and rep["ann_calls_per_question"]["hop-passages-2"] > 1
    saved = json.loads(Path(rep["paths"]["json"]).read_text())
    assert saved["config_hash"] == rep["config_hash"]
    assert "Retrieval report" in Path(rep["paths"]["markdown"]).read_text()
    # same config, same results (constitution P6)
    rep2 = run(_cfg(fixtures, tmp_path))
    assert rep2["recall"] == rep["recall"] and rep2["waterfall"] == rep["waterfall"]


@pytest.mark.postgres
def test_gate0_runner_records_to_postgres(fixtures, tmp_path, pg_dsn):
    from hsi_plugins_postgres import PostgresStore
    rep = run(_cfg(fixtures, tmp_path, metadata_store={"plugin": "postgres", "options": {"dsn": pg_dsn}}))
    store = PostgresStore(pg_dsn)
    try:
        assert store.schema_version() >= 1
        assert store.count("passages", rep["corpus_id"]) == rep["dataset"]["passages"]
        assert store.count("shortcuts", rep["corpus_id"]) == rep["chains"]["stored"]
        assert rep["run_id"]
    finally:
        store.delete_corpus(rep["corpus_id"])
        store.close()
