import pytest

from hsi_core.bm25 import BM25Index
from hsi_core.chains import ChainConfig, enumerate_chains
from hsi_core.graph import build_title_graph
from hsi_core.metrics import bootstrap_mean, full_chain_recall, paired_difference
from hsi_core.waterfall import Stage, classify, summarize


def test_bm25_ranks_relevant_first():
    idx = BM25Index(1.2, 0.75).build(["a", "b", "c"], ["the seine river flows through paris",
                                              "lyon sits on the rhone", "pixel phones by google"])
    res = idx.search("Which river flows through Paris?", 2)
    assert res[0][0] == "a"
    assert idx.search("zzz", 3) == []


def test_full_chain_recall():
    assert full_chain_recall(["a", "b", "c"], ["a", "c"], 3)
    assert not full_chain_recall(["a", "b", "c"], ["a", "c"], 2)
    with pytest.raises(ValueError):
        full_chain_recall(["a"], [], 1)


def test_bootstrap_is_deterministic_and_brackets_mean():
    vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    a, b = bootstrap_mean(vals, 500, 7), bootstrap_mean(vals, 500, 7)
    assert a == b and a.low <= a.mean <= a.high and a.mean == pytest.approx(0.7)
    d = paired_difference([1, 1, 1, 0], [0, 1, 0, 0], 500, 0)
    assert d.mean == pytest.approx(0.5)


def test_waterfall_stages(mini_musique):
    g = build_title_graph(mini_musique.passages)
    levels = enumerate_chains(g, ChainConfig(max_passages=3))
    cands = {c.passages for lvl in levels.values() for c in lvl}
    q = next(q for q in mini_musique.questions if q.hops == 3)
    assert classify(q.gold, q.ordered, g, cands, cands, list(q.gold), 3) is Stage.OK
    assert classify(q.gold, q.ordered, g, cands, cands, [], 3) is Stage.NOT_RETRIEVED
    assert classify(q.gold, q.ordered, g, cands, set(), [], 3) is Stage.NOT_STORED
    assert classify(q.gold, q.ordered, g, set(), set(), [], 3) is Stage.PRUNED
    other = next(p.id for p in mini_musique.passages if p.title == "Snapchat")
    assert classify((q.gold[0], other), True, g, cands, cands, [], 3) is Stage.LINK_MISSED
    s = summarize([Stage.OK, Stage.PRUNED, Stage.OK, Stage.LINK_MISSED])
    assert s["ok"] == 0.5 and s["n"] == 4
