"""JSON Lines connector (specs/additions/A5 §7): any system exported as one JSON object per line.

Each line: {"id": str, "text": str, "acl": {"public": true} | {"principals": [str, ...]},
            "title": str?, "uri": str?, "sections": [{"heading": str?, "text": str}]?}

`text` or `sections` is required. Lines without a valid ACL are rejected and reported in `errors`
(A2 rule 2). The cursor is the content hash of every record, so a resumed sync yields only changed
records, plus tombstones for records that disappeared.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hsi_core.connectors import ConnectorCaps, CursorCheckpoint, Record, StreamSpec
from hsi_core.model import Acl, Section, SourceDocument, Tombstone

STREAM = StreamSpec(name="documents", description="documents from JSON Lines files")


class JsonlConnector:
    version = "0.1.0"

    def __init__(self, path: str, source_name: str) -> None:
        """`path`: a .jsonl file or a directory of them; `source_name`: this source's connector name."""
        self.root = Path(path)
        if not self.root.exists():
            raise ValueError(f"JSONL path not found: {path}")
        self.name = source_name
        self.errors: list[tuple[str, str]] = []

    def capabilities(self) -> ConnectorCaps:
        return ConnectorCaps(incremental=True, emits_documents=True, emits_facts=False, emits_tombstones=True)

    def streams(self) -> list[StreamSpec]:
        return [STREAM]

    def _files(self) -> list[Path]:
        return [self.root] if self.root.is_file() else sorted(self.root.rglob("*.jsonl"))

    def _doc(self, rec: dict[str, Any]) -> SourceDocument:
        if "sections" in rec:
            sections = tuple(Section(heading=s.get("heading"), text=s["text"]) for s in rec["sections"])
        else:
            sections = (Section(text=rec["text"]),)
        acl = rec["acl"]
        return SourceDocument(id=f"{self.name}:{rec['id']}", connector=self.name, uri=rec.get("uri"),
                              title=rec.get("title"), sections=sections,
                              acl=Acl(public=bool(acl.get("public", False)),
                                      principals=frozenset(acl.get("principals", ()))))

    def _scan(self) -> dict[str, tuple[str, SourceDocument]]:
        found: dict[str, tuple[str, SourceDocument]] = {}
        self.errors = []
        for f in self._files():
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    doc = self._doc(rec)
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    self.errors.append((f"{f}:{n}", f"{type(e).__name__}: {e}"))
                    continue
                found[doc.id] = (hashlib.sha256(line.encode("utf-8")).hexdigest(), doc)
        return found

    def sync(self, stream: StreamSpec, cursor: str | None) -> Iterator[Record]:
        if stream.name != STREAM.name:
            raise ValueError(f"unknown stream {stream.name!r}")
        before: dict[str, str] = json.loads(cursor) if cursor else {}
        now = self._scan()
        for rid in sorted(now):
            h, doc = now[rid]
            if before.get(rid) != h:
                yield doc
        for rid in sorted(set(before) - set(now)):
            yield Tombstone(record_id=rid, connector=self.name)
        yield CursorCheckpoint(stream=STREAM.name,
                               cursor=json.dumps({rid: h for rid, (h, _) in now.items()}, sort_keys=True))
