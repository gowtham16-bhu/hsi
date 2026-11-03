import itertools

from hsi_core.chains import ChainConfig, enumerate_chains, select_by_prior
from hsi_core.graph import GraphConfig, build_title_graph, entity_key_for_title
from hsi_core.model import PUBLIC, Passage
from hsi_core.text import tokens
from hsi_core.titles import TokenAhoCorasick, longest_non_overlapping


def test_aho_corasick_token_matching():
    ac = TokenAhoCorasick()
    ac.add(tokens("New York"), "ny")
    ac.add(tokens("New York Times"), "nyt")
    ac.add(tokens("York"), "york")
    ac.build()
    found = longest_non_overlapping(ac.find(tokens("She read the New York Times in York.")))
    assert [m.key for m in found] == ["nyt", "york"]
    # no partial-word matches
    assert ac.find(tokens("Yorkshire")) == []


def _p(title, text):
    return Passage.build(doc_id=title, text=text, title=title, acl=PUBLIC)


def test_graph_links_mentions_to_anchors(mini_musique):
    g = build_title_graph(mini_musique.passages)
    by_title = {p.title: p.id for p in mini_musique.passages}
    assert g.linked(by_title["Instagram"], by_title["Facebook"])
    assert g.linked(by_title["Facebook"], by_title["Mark Zuckerberg"])
    edges = {(e.src, e.dst) for e in g.neighbors(by_title["Instagram"])}
    assert (by_title["Instagram"], by_title["Facebook"]) in edges
    e = next(e for e in g.neighbors(by_title["Instagram"]) if e.dst == by_title["Facebook"])
    assert e.bridge == entity_key_for_title("Facebook")


def test_fan_out_cap_and_hubs():
    hub = _p("Hub", "The hub entity.")
    others = [_p(f"P{i}", f"Passage {i} mentions Hub and Target{i % 3}.") for i in range(10)]
    targets = [_p(f"Target{i}", f"Target{i} is a thing.") for i in range(3)]
    g = build_title_graph([hub, *others, *targets], GraphConfig(fan_out=2, hub_degree_cap=5))
    assert entity_key_for_title("Hub") in g.hubs
    assert all(len(v) <= 2 for v in g.out_edges.values())
    # hub edges are not used for chains, but the pair still counts as linked
    assert all(e.bridge != entity_key_for_title("Hub") for v in g.out_edges.values() for e in v)
    assert g.linked(others[0].id, hub.id)


def test_chains_contain_gold_and_are_deterministic(mini_musique):
    g = build_title_graph(mini_musique.passages)
    levels = enumerate_chains(g, ChainConfig(max_passages=3))
    q = next(q for q in mini_musique.questions if q.hops == 3)
    assert q.gold in {c.passages for c in levels[2]}
    assert levels == enumerate_chains(g, ChainConfig(max_passages=3))
    every = select_by_prior(levels, None)
    assert len(every) == sum(len(v) for v in levels.values())
    top = select_by_prior(levels, 3)
    assert len(top) == 3 and all(a.log_prob >= b.log_prob for a, b in itertools.pairwise(top))


EXTRACTOR = dict(min_cap_ratio=0.5, min_join_count=2, min_join_share=0.8, max_join_tokens=2, max_words=6, min_chars=3,
                 max_clitic_chars=2)

# A small corpus with realistic casing: function words appear mostly in lower case.
CORPUS = [
    "The University of Oxford is old. Students at the University of Oxford study hard.",
    "However, the film was a success. He said the cast was strong and that the cast was praised.",
    "Critics said he was right, however the studio disagreed and he left the project.",
    "Markus Zusak wrote it. Markus Zusak won awards, and Markus Zusak's novel sold well.",
    "Lowndes County is in Alabama. The county seat of Lowndes County is Hayneville.",
    "Julie Andrews starred. The film stars Julie Andrews and Christopher Plummer.",
    "Julie Andrews plays the queen of Genovia. There is nothing in common here, however.",
    "In the end the film and the novel were both popular in the region.",
    "Christopher Plummer later played a captain, and Christopher Plummer won an award.",
]


def test_capitalized_mentions_are_learned_from_corpus():
    from hsi_core import registry
    ext = registry.load("mention_extractor", "capitalized", **EXTRACTOR).fit(CORPUS)
    got = {s for s, _ in ext.extract("However, the University of Oxford hosted Markus Zusak's talk "
                                     "in Lowndes County.")}
    assert {"University of Oxford", "Markus Zusak", "Lowndes County"} <= got
    # function words drop out because the corpus writes them in lower case most of the time
    assert not {"However", "The", "He"} & got
    assert ext.cap_ratio("the") < 0.5 < ext.cap_ratio("Oxford")
    names = {s for s, _ in ext.extract("The film stars Julie Andrews and Christopher Plummer.")}
    assert {"Julie Andrews", "Christopher Plummer"} <= names      # "and" never joins: not frequent as one span


def test_extractor_requires_fit_and_valid_parameters():
    import pytest as _pytest

    from hsi_core import registry
    with _pytest.raises(RuntimeError):
        registry.load("mention_extractor", "capitalized", **EXTRACTOR).extract("Oxford")
    with _pytest.raises(ValueError):
        registry.load("mention_extractor", "capitalized", **(EXTRACTOR | {"min_cap_ratio": 0}))


def test_co_mention_edges():
    from hsi_core import registry
    from hsi_core.graph import entity_key
    a = _p("On Golden Pond (2001 film)", "The film stars Julie Andrews and Christopher Plummer.")
    b = _p("The Princess Diaries (film)", "Julie Andrews plays the queen of Genovia.")
    c = _p("Unrelated", "Nothing in common here.")
    ps = [a, b, c]
    ext = registry.load("mention_extractor", "capitalized", **EXTRACTOR).fit([*CORPUS, *(p.text for p in ps)])
    mentions = {p.id: {entity_key(s): sal for s, sal in ext.extract(p.text)} for p in ps}
    g = build_title_graph(ps, mentions=mentions)
    assert g.linked(a.id, b.id) and not g.linked(a.id, c.id)
    e = next(e for e in g.neighbors(a.id) if e.dst == b.id)
    assert e.type.value == "co_mention" and e.bridge == entity_key("Julie Andrews")
    no_co = build_title_graph(ps, GraphConfig(co_mention=False), mentions=mentions)
    assert not any(e.dst == b.id for e in no_co.neighbors(a.id))


def test_edge_scorer_changes_which_edges_survive_the_cap(mini_musique):
    import numpy as np
    kept_default = build_title_graph(mini_musique.passages, GraphConfig(fan_out=1), keep_candidates=True)
    assert kept_default.candidates is not None
    src = next(s for s, d in kept_default.candidates.items() if len(d) > 1)
    worst = min(kept_default.candidates[src].items(), key=lambda kv: (kv[1][0], kv[0]))[0]

    def prefer_worst(s, cands, participants):
        return np.array([1.0 if c[0] == worst else 0.0 for c in cands])
    g = build_title_graph(mini_musique.passages, GraphConfig(fan_out=1), edge_scorer=prefer_worst)
    assert [e.dst for e in g.neighbors(src)] == [worst]
