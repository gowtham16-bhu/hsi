"""Connector contract, conformance kit, OSV connector, and C2 document ingestion.

OSV fixtures use the real OSV schema with clearly fictional ids (TEST-*), so no test asserts
facts about real vulnerabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsi_connector_kit import run_kit
from hsi_core import registry
from hsi_core.connectors import Connector, CursorCheckpoint
from hsi_core.ingest import ChunkConfig, ingest_document
from hsi_core.model import PUBLIC, Section, SourceDocument, StructuredFact, Tombstone


def _osv(vid: str, n: int, modified: str = "2026-01-01T00:00:00Z", **extra) -> dict:
    rec = {
        "schema_version": "1.6.0", "id": vid, "modified": modified, "published": "2025-12-01T00:00:00Z",
        "aliases": [f"CVE-2099-{1000 + n}", f"GHSA-test-{n:04d}-xxxx"],
        "summary": f"Test advisory {n}: remote code execution in examplelib-{n}",
        "details": f"A crafted input to examplelib-{n} lets an attacker run code. Upgrade to 2.{n}.1.",
        "affected": [{"package": {"ecosystem": "PyPI", "name": f"examplelib-{n}",
                                  "purl": f"pkg:pypi/examplelib-{n}"},
                      "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": f"2.{n}.1"}]}]}],
    }
    rec.update(extra)
    return rec


class OsvDirDriver:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _write(self, rec: dict) -> None:
        (self.root / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")

    def seed(self, n: int) -> list[str]:
        for i in range(n):
            self._write(_osv(f"TEST-{i:04d}", i))
        return [f"osv:TEST-{i:04d}" for i in range(n)]

    def modify(self, record_id: str) -> None:
        vid = record_id.removeprefix("osv:")
        i = int(vid.split("-")[1])
        self._write(_osv(vid, i, modified="2026-02-01T00:00:00Z"))

    def delete(self, record_id: str) -> None:
        (self.root / f"{record_id.removeprefix('osv:')}.json").unlink()


def test_osv_connector_passes_conformance_kit(tmp_path):
    res = run_kit(lambda: registry.load("connector", "osv", path=str(tmp_path)), OsvDirDriver(tmp_path))
    assert res.ok, res.failed
    assert {"C1 records", "C2 checkpoint", "C3 deterministic", "C4 quiet resume", "C5 change", "C6 delete"} \
        <= set(res.passed)


def test_kit_catches_a_broken_connector(tmp_path):
    class NoCheckpoint:
        name, version = "broken", "0"

        def __init__(self) -> None:
            self.inner = registry.load("connector", "osv", path=str(tmp_path))

        def capabilities(self):
            return self.inner.capabilities()

        def streams(self):
            return self.inner.streams()

        def sync(self, stream, cursor):
            return (r for r in self.inner.sync(stream, cursor) if not isinstance(r, CursorCheckpoint))

    res = run_kit(NoCheckpoint, OsvDirDriver(tmp_path))
    assert not res.ok and "C2 checkpoint" in res.failed


def test_osv_records_facts_and_withdrawn(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_osv("TEST-0001", 1)))
    (tmp_path / "b.json").write_text(json.dumps(_osv("TEST-0002", 2, withdrawn="2026-01-02T00:00:00Z")))
    (tmp_path / "bad.json").write_text("{not json")
    conn = registry.load("connector", "osv", path=str(tmp_path))
    assert isinstance(conn, Connector)
    recs = list(conn.sync(conn.streams()[0], None))
    docs = [r for r in recs if isinstance(r, SourceDocument)]
    facts = [r for r in recs if isinstance(r, StructuredFact)]
    assert [d.id for d in docs] == ["osv:TEST-0001"]
    assert {f.predicate for f in facts} == {"alias_of", "affects", "fixed_in"}
    assert any(f.object.key == "cve:CVE-2099-1001" for f in facts if f.predicate == "alias_of")
    assert any(isinstance(r, Tombstone) and r.record_id == "osv:TEST-0002" for r in recs)
    assert len(conn.errors) == 1 and conn.errors[0][0].endswith("bad.json")


def _doc(text: str) -> SourceDocument:
    return SourceDocument(id="doc1", connector="test", title="Runbook", acl=PUBLIC,
                          sections=(Section(heading="Intro", text="Short intro."), Section(heading="Body", text=text)))


def test_ingest_structure_first_and_local_edits():
    words = [f"w{i % 97}x{i}" for i in range(3000)]
    base = ingest_document(_doc(" ".join(words)))
    assert base[0].text == "Short intro." and base[0].section_path == ("Intro",)
    cfg = ChunkConfig()
    assert all(len(p.text.split()) <= cfg.max_words + cfg.min_words // 2 for p in base)
    edited = words.copy()
    edited[1500] = "CHANGED"
    after = ingest_document(_doc(" ".join(edited)))
    changed = {p.hash for p in after} - {p.hash for p in base}
    assert 1 <= len(changed) <= 3
    # unchanged chunks keep their ids
    assert len({p.id for p in after} & {p.id for p in base}) >= len(base) - 3


@pytest.mark.parametrize("bad", [dict(min_words=0), dict(min_words=10, max_words=5), dict(modulus=0)])
def test_chunk_config_validation(bad):
    with pytest.raises(ValueError):
        ChunkConfig(**bad)
