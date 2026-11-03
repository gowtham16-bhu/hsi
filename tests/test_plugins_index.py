import numpy as np
import pytest

from hsi_core import registry
from hsi_core.chains import ChainConfig, enumerate_chains, select_by_prior
from hsi_core.encoding import CachedEncoder
from hsi_core.expansion import ExpansionConfig, OnlineExpansion
from hsi_core.graph import build_title_graph
from hsi_core.index import Hit, ShortcutIndex, shortcuts_from_chains, unpack
from hsi_core.interfaces import Encoder, VectorStore


def test_registry_lists_and_loads_installed_plugins():
    av = registry.available()
    assert "faiss" in av["vector_store"] and "hashing" in av["encoder"]
    assert isinstance(registry.load("vector_store", "faiss"), VectorStore)
    assert isinstance(registry.load("encoder", "hashing", dim=64), Encoder)
    with pytest.raises(registry.PluginError, match="no plugin"):
        registry.load("vector_store", "does-not-exist")
    with pytest.raises(registry.PluginError, match="unknown plugin family"):
        registry.load("teleporter", "x")


def test_faiss_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[:7] + 0.01
    store = registry.load("vector_store", "faiss")
    store.add(np.arange(500), x)
    _scores, ids = store.search(q, 10)
    ref = np.argsort(-(q @ x.T), axis=1)[:, :10]
    assert (ids == ref).all()
    assert store.delete(np.array([0, 1])) == 2 and len(store) == 498


def test_unpack_order_and_dedup():
    hits = [Hit(0, 0.9, ("a", "b")), Hit(1, 0.8, ("b", "c")), Hit(2, 0.7, ("d",))]
    assert unpack(hits, 10) == ["a", "b", "c", "d"]
    assert unpack(hits, 2) == ["a", "b"]


def test_cached_encoder_persists(tmp_path):
    enc = registry.load("encoder", "hashing", dim=64)
    c1 = CachedEncoder(enc, str(tmp_path))
    v1 = c1.encode(["alpha beta", "gamma"], "passage")
    assert c1.misses == 2
    c2 = CachedEncoder(enc, str(tmp_path))
    v2 = c2.encode(["gamma", "alpha beta"], "passage")
    assert c2.misses == 0 and np.allclose(v1[::-1], v2)


def test_shortcut_index_and_expansion_end_to_end(mini_musique):
    enc = registry.load("encoder", "hashing", dim=512)
    g = build_title_graph(mini_musique.passages)
    stored = select_by_prior(enumerate_chains(g, ChainConfig(max_passages=3)), None)

    def factory():
        return registry.load("vector_store", "faiss")

    dense = ShortcutIndex(enc, factory).build(mini_musique.passages)
    hsi = ShortcutIndex(enc, factory).build(mini_musique.passages, shortcuts_from_chains(stored))
    assert len(hsi.entries) == len(mini_musique.passages) + len(stored)
    assert hsi.size_by_level()[0] == len(mini_musique.passages)

    q = next(q for q in mini_musique.questions if q.hops == 3)
    r = hsi.retrieve(q.question, k=5, top_k=20)
    assert len(r.passages) == len(set(r.passages)) <= 5
    assert all(set(h.passages) <= set(g.passages) for h in r.hits)

    for variant in ("vec", "text"):
        exp = OnlineExpansion(dense, g, ExpansionConfig(seeds=3, beam=2, variant=variant))
        r2 = exp.retrieve(q.question, k=5, top_k=20)
        assert len(r2.passages) == len(set(r2.passages)) <= 5
    with pytest.raises(ValueError):
        OnlineExpansion(hsi, g, ExpansionConfig())


def test_ppr_and_hops(mini_musique):
    from hsi_core.hops import HopConfig, HopRetriever
    from hsi_core.ppr import PPRConfig, PPRRetriever
    enc = registry.load("encoder", "hashing", dim=512)
    g = build_title_graph(mini_musique.passages)

    def factory():
        return registry.load("vector_store", "faiss")

    dense = ShortcutIndex(enc, factory).build(mini_musique.passages)
    q = next(q for q in mini_musique.questions if q.hops == 3)
    qv = enc.encode([q.question], "query")[0]
    ppr = PPRRetriever(dense, g, PPRConfig(seeds=2, restart=0.5, iterations=50, tolerance=1e-9, beta=0.0,
                                            rank_by="total", keep_seeds=0))
    r = ppr.retrieve_vec(qv, 5, 10)
    assert len(r.passages) == len(set(r.passages)) <= 5
    # mass spreads along edges: a graph neighbour of a seed gets a score even if it is not a seed
    assert len(r.passages) > 2
    aware = PPRRetriever(dense, g, PPRConfig(seeds=2, restart=0.5, iterations=50, tolerance=1e-9, beta=10.0,
                                              rank_by="arrived", keep_seeds=1))
    assert set(aware.retrieve_vec(qv, 5, 10).passages) <= set(g.passages)
    by_id = {p.id: p for p in mini_musique.passages}
    hop = HopRetriever(dense, by_id, HopConfig(rounds=3, beam=2, per_round=3, update="vec", vec_weight=1.0))
    r2 = hop.retrieve_vec(qv, 5, 10, q.question)
    assert hop.ann_calls == 1 + 2 * 2 and len(r2.passages) == len(set(r2.passages)) <= 5
    assert all(len(h.passages) == len(set(h.passages)) for h in r2.hits)
    with pytest.raises(ValueError):
        PPRConfig(seeds=1, restart=1.0, iterations=1, tolerance=1e-6, beta=0.0, rank_by="total", keep_seeds=0)


def test_bipartite_walk(mini_musique):
    from hsi_core.bipartite import BipartiteConfig, BipartiteWalkRetriever
    enc = registry.load("encoder", "hashing", dim=512)
    g = build_title_graph(mini_musique.passages)
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", "faiss")).build(mini_musique.passages)
    q = next(q for q in mini_musique.questions if q.hops == 3)
    qv = enc.encode([q.question], "query")[0]
    for beta, gamma in [(0.0, 0.0), (10.0, 1.0)]:
        walk = BipartiteWalkRetriever(dense, g, BipartiteConfig(seeds=2, restart=0.5, iterations=60,
                                                                tolerance=1e-9, beta=beta, gamma=gamma))
        r = walk.retrieve_vec(qv, 5, 10)
        assert len(r.passages) == len(set(r.passages)) <= 5 and set(r.passages) <= set(g.passages)
    # probability mass is conserved: transition rows sum to 1
    sums = np.zeros(walk.n)
    np.add.at(sums, walk.p_idx, walk.t_pe)
    assert np.allclose(sums[walk.p_has_out], 1.0)


def test_local_push_matches_global_ppr_and_bounds_work(mini_musique):
    """Push PPR approximates global PPR, and its work is bounded by 1/(α·ε), not by corpus size."""
    from hsi_core.ppr import PPRConfig, PPRRetriever
    from hsi_core.push import LocalPushRetriever, PushConfig
    enc = registry.load("encoder", "hashing", dim=512)
    g = build_title_graph(mini_musique.passages)
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", "faiss")).build(mini_musique.passages)
    for q in mini_musique.questions:
        qv = enc.encode([q.question], "query")[0]
        for beta in (0.0, 10.0):
            exact = PPRRetriever(dense, g, PPRConfig(seeds=2, restart=0.5, iterations=200, tolerance=1e-12,
                                                     beta=beta, rank_by="total", keep_seeds=0)
                                 ).retrieve_vec(qv, 3, 10).passages
            push = LocalPushRetriever(dense, g, PushConfig(seeds=2, restart=0.5, epsilon=1e-7, beta=beta,
                                                           max_pushes=100_000))
            approx = push.retrieve_vec(qv, 3, 10).passages
            assert approx[0] == exact[0] and set(approx) == set(exact)
            assert push.last_pushes <= 1 / (0.5 * 1e-7)


def test_listwise_fit_learns_signal_and_round_trips(tmp_path):
    from hsi_core.transitions import FEATURES, LinearTransition, fit_listwise, listwise_loss
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(300):
        g = rng.normal(size=(6, len(FEATURES)))
        y = int(np.argmax(2.0 * g[:, 1] - 1.0 * g[:, 4]))   # gold depends on two features
        groups.append((g, y))
    model = fit_listwise(groups, l2=1e-3, lr=0.05, epochs=200, seed=0)
    _, acc = listwise_loss(model, groups)
    assert acc > 0.9 and model.theta[1] > 0 > model.theta[4]
    model.save(tmp_path / "m.json", {"note": "test"})
    again = LinearTransition.load(tmp_path / "m.json")
    assert np.allclose(again.theta, model.theta)


def test_smoothing_moves_vectors_toward_neighbours(mini_musique):
    from hsi_core.smoothing import SmoothingConfig, smooth_vectors
    enc = registry.load("encoder", "hashing", dim=256)
    g = build_title_graph(mini_musique.passages)
    raw = {p.id: v for p, v in zip(mini_musique.passages, enc.encode([p.text for p in mini_musique.passages],
                                                                       "passage"), strict=True)}
    same = smooth_vectors(raw, g, SmoothingConfig(gamma=0.0, steps=1))
    assert all(np.allclose(same[k], raw[k]) for k in raw)
    sm = smooth_vectors(raw, g, SmoothingConfig(gamma=0.5, steps=1))
    src = next(s for s, es in g.out_edges.items() if es)
    nb = g.out_edges[src][0].dst
    assert float(sm[src] @ raw[nb]) > float(raw[src] @ raw[nb])
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in sm.values())


def test_compact_graph_round_trip_matches_edges(mini_musique, tmp_path):
    from hsi_core.csr import CompactGraph
    enc = registry.load("encoder", "hashing", dim=128)
    g = build_title_graph(mini_musique.passages)
    ids = [p.id for p in mini_musique.passages]
    vecs = enc.encode([p.text for p in mini_musique.passages], "passage")
    c = CompactGraph.from_graph(g, ids, vecs)
    assert c.edge_count == g.edge_count
    for i, pid in enumerate(ids):
        assert sorted(ids[d] for d in c.dst[c.edges_of(i)]) == sorted(e.dst for e in g.neighbors(pid))
    c.save(tmp_path / "g.npz")
    back = CompactGraph.load(tmp_path / "g.npz")
    assert back.ids == c.ids and back.bridges == c.bridges
    assert all(np.array_equal(getattr(back, f), getattr(c, f))
               for f in ("indptr", "dst", "weight", "salience", "co_mention", "bridge", "cos_uv", "bridge_log_df"))


def test_learned_push_matches_learned_global_ppr(mini_musique):
    """Decision 0012 A3: with the same learned model, push top-k equals global PPR top-k at small ε."""
    from hsi_core.ppr import PPRConfig, PPRRetriever
    from hsi_core.push import LocalPushRetriever, PushConfig
    from hsi_core.transitions import FEATURES, LinearTransition
    enc = registry.load("encoder", "hashing", dim=512)
    g = build_title_graph(mini_musique.passages)
    dense = ShortcutIndex(enc, lambda: registry.load("vector_store", "faiss")).build(mini_musique.passages)
    rng = np.random.default_rng(3)
    model = LinearTransition(FEATURES, rng.normal(size=len(FEATURES)), rng.uniform(0.5, 2, size=len(FEATURES)),
                             rng.normal(size=len(FEATURES)))
    glob = PPRRetriever(dense, g, PPRConfig(seeds=2, restart=0.5, iterations=200, tolerance=1e-12, beta=0.0,
                                            rank_by="total", keep_seeds=0), transition_model=model)
    push = LocalPushRetriever(dense, glob.table.csr, PushConfig(seeds=2, restart=0.5, epsilon=1e-9, beta=0.0,
                                                                max_pushes=1_000_000), transition_model=model)
    for q in mini_musique.questions:
        qv = enc.encode([q.question], "query")[0]
        exact = glob.retrieve_vec(qv, 4, 10, q.question).passages
        approx = push.retrieve_vec(qv, 4, 10, q.question).passages
        assert approx == exact
        assert push.last_vectors_read <= len(mini_musique.passages)
    with pytest.raises(ValueError):
        push.retrieve_vec(qv, 4, 10)
