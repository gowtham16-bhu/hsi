import pytest
from pydantic import ValidationError

from hsi_core.model import PUBLIC, Acl, Chain, Edge, EdgeType, EntityRef, Passage


def test_acl_requires_someone():
    with pytest.raises(ValidationError):
        Acl()
    assert Acl(principals=frozenset({"group:sre"})).allows(frozenset({"group:sre", "u:1"}))
    assert not Acl(principals=frozenset({"group:sre"})).allows(frozenset({"u:1"}))
    assert PUBLIC.allows(frozenset())


def test_passage_ids_are_stable_and_content_addressed():
    a = Passage.build(doc_id="d1", text="Paris lies on the Seine.", title="Paris", acl=PUBLIC)
    b = Passage.build(doc_id="d1", text="Paris  lies on the   Seine.", title="Paris", acl=PUBLIC)
    c = Passage.build(doc_id="d1", text="Paris lies on the Loire.", title="Paris", acl=PUBLIC)
    assert a.id == b.id and a.hash == b.hash
    assert a.id != c.id


def test_chain_validation():
    c = Chain(passages=("a", "b", "c"), bridges=("e1", "e2"), log_prob=-0.5)
    assert c.level == 2 and len(c.id) == 24
    with pytest.raises(ValidationError):
        Chain(passages=("a", "a"), bridges=("e",))
    with pytest.raises(ValidationError):
        Chain(passages=("a", "b"), bridges=())
    with pytest.raises(ValidationError):
        Chain(passages=("a", "b", "c", "d", "e"), bridges=("1", "2", "3", "4"))


def test_edge_rejects_self_loop_and_bad_weight():
    with pytest.raises(ValidationError):
        Edge(src="a", dst="a", type=EdgeType.MENTION, weight=0.5, bridge="x")
    with pytest.raises(ValidationError):
        Edge(src="a", dst="b", type=EdgeType.MENTION, weight=0.0, bridge="x")


def test_json_round_trip():
    for obj in [EntityRef(scheme="cve", id="CVE-2021-44228"),
                Passage.build(doc_id="d", text="t", acl=PUBLIC),
                Chain(passages=("a", "b"), bridges=("x",), log_prob=-1.0)]:
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj
