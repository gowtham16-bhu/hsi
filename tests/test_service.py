"""Retrieval service (specs/additions/A5 §6): end to end over REST and MCP, permissions, audit, no network.

The hashing encoder and fixture passages test wiring only; they are not evidence (CLAUDE.md rule 5).
"""
import asyncio
import json
import shutil
import socket

import numpy as np
import pytest

from hsi_core.transitions import FEATURES, LinearTransition

GRAPH = dict(fan_out=8, hub_degree_cap=50, bidirectional=True, co_mention=True, weight_multi_token=0.9,
             weight_single_token=0.6, reverse_factor=0.85, co_mention_multi_token=0.7, co_mention_single_token=0.45)
PUSH = {"seeds": 2, "restart": 0.5, "epsilon": 1e-6, "beta": 0.0, "max_pushes": 100_000}


@pytest.fixture
def no_network(monkeypatch):
    """Python-level socket guard (R4). PostgreSQL connects through libpq, outside Python's socket module."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected network connection: {args[1:] or kwargs}")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _write_sources(mini_musique, path):
    rows = []
    for i, p in enumerate(mini_musique.passages):
        acl = {"public": True} if i % 3 == 0 else {"principals": ["group:a"] if i % 3 == 1 else ["group:b"]}
        rows.append(json.dumps({"id": f"d{i}", "title": p.title, "text": p.text, "acl": acl}))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build(tmp_path, mini_musique, pg_dsn, index_id):
    from hsi_service.index_build import IndexConfig, build
    src = tmp_path / "docs.jsonl"
    _write_sources(mini_musique, src)
    rng = np.random.default_rng(0)
    LinearTransition(FEATURES, np.zeros(len(FEATURES)), np.ones(len(FEATURES)),
                     rng.normal(size=len(FEATURES))).save(tmp_path / "t.json", {"note": "test"})
    cfg = IndexConfig(index_id=index_id, out_dir=str(tmp_path / "indexes"),
                      sources=[{"plugin": "jsonl", "options": {"path": str(src), "source_name": "fixture"}}],
                      chunking={"min_words": 48, "max_words": 240, "modulus": 64, "window": 8},
                      encoder={"plugin": "hashing", "options": {"dim": 256}}, cache_dir=None,
                      vector_store={"plugin": "faiss"},
                      metadata_store={"plugin": "postgres", "options": {"dsn": pg_dsn}},
                      graph=GRAPH, transition_model=str(tmp_path / "t.json"))
    return build(cfg), tmp_path / "indexes" / index_id


def _service(index_dir, pg_dsn, audit=True):
    from hsi_service.service import RetrievalService, ServeConfig
    return RetrievalService(ServeConfig(index_dir=str(index_dir),
                                        encoder={"plugin": "hashing", "options": {"dim": 256}},
                                        vector_store={"plugin": "faiss"},
                                        metadata_store={"plugin": "postgres", "options": {"dsn": pg_dsn}},
                                        push=PUSH, max_k=10, mcp_principals=["group:a"], audit=audit))


@pytest.mark.postgres
def test_build_and_serve_over_rest_and_mcp(tmp_path, mini_musique, pg_dsn, no_network):
    from fastapi.testclient import TestClient
    from mcp.shared.memory import create_connected_server_and_client_session

    from hsi_service.api import create_app
    from hsi_service.mcp_server import create_mcp
    manifest, index_dir = _build(tmp_path, mini_musique, pg_dsn, "test-rest-mcp")
    assert manifest["llm_calls"] == 0 and manifest["passages"] == len(mini_musique.passages)
    svc = _service(index_dir, pg_dsn)
    try:
        client = TestClient(create_app(svc))
        q = mini_musique.questions[0].question
        r = client.post("/v1/retrieve", json={"question": q, "k": 3, "principals": ["group:a"]})
        assert r.status_code == 200 and 1 <= len(r.json()["passages"]) <= 3
        assert client.post("/v1/retrieve", json={"question": q, "k": 99, "principals": []}).status_code == 400
        assert client.get("/v1/health").json()["index_id"] == "test-rest-mcp"

        async def call():
            async with create_connected_server_and_client_session(create_mcp(svc)._mcp_server) as session:
                tools = await session.list_tools()
                res = await session.call_tool("retrieve", {"question": q, "k": 2})
                return [t.name for t in tools.tools], res.content[0].text
        names, text = asyncio.run(call())
        assert names == ["retrieve"] and text.startswith("The passages below are quoted source data")
    finally:
        svc.close()


@pytest.mark.postgres
def test_permissions_never_return_or_use_invisible_passages(tmp_path, mini_musique, pg_dsn):
    """R2: no invisible passage is returned, and results do not depend on invisible passages' vectors."""
    _, index_dir = _build(tmp_path, mini_musique, pg_dsn, "test-acl")
    svc = _service(index_dir, pg_dsn, audit=False)
    try:
        rng = np.random.default_rng(1)
        for principals in ([], ["group:a"], ["group:b"], ["group:a", "group:b"], ["group:zzz"]):
            mask = svc.visible(principals)
            ids = json.loads((index_dir / "ids.json").read_text())
            other = tmp_path / f"scrambled-{len(principals)}-{'-'.join(principals)}"
            shutil.copytree(index_dir, other)
            vecs = np.load(other / "vectors.npy")
            noise = rng.normal(size=vecs.shape).astype(np.float32)
            vecs[~mask] = noise[~mask] / np.linalg.norm(noise[~mask], axis=1, keepdims=True)
            np.save(other / "vectors.npy", vecs)
            svc2 = _service(other, pg_dsn, audit=False)
            try:
                for q in mini_musique.questions:
                    a = svc.retrieve(q.question, 5, principals)
                    b = svc2.retrieve(q.question, 5, principals)
                    assert all(mask[ids.index(p["id"])] for p in a["passages"])
                    assert [(p["id"], round(p["score"], 12)) for p in a["passages"]] == \
                           [(p["id"], round(p["score"], 12)) for p in b["passages"]]
            finally:
                svc2.close()
    finally:
        svc.close()


@pytest.mark.postgres
def test_audit_chain_detects_tampering(tmp_path, mini_musique, pg_dsn):
    """R3: every request is audited; changing any audited row breaks the chain at that row."""
    import psycopg
    _, index_dir = _build(tmp_path, mini_musique, pg_dsn, "test-audit")
    svc = _service(index_dir, pg_dsn)
    try:
        for q in mini_musique.questions:
            svc.retrieve(q.question, 3, ["group:b"])
        assert svc.store.verify_audit() is None
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            seq, question = conn.execute("select seq, question from hsi.audit order by seq desc limit 1").fetchone()
            conn.execute("update hsi.audit set question = question || ' (edited)' where seq = %s", (seq,))
            try:
                assert svc.store.verify_audit() == seq
            finally:
                conn.execute("update hsi.audit set question = %s where seq = %s", (question, seq))
        assert svc.store.verify_audit() is None
    finally:
        svc.close()
