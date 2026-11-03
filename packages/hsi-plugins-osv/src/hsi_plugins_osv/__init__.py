"""OSV connector, file-dump mode: reads OSV-schema JSON files from a local directory.

Records per advisory:
  SourceDocument  osv:<id>, sections Summary, Details, Affected, Aliases (public)
  StructuredFact  (osv, id) alias_of (cve|ghsa|osv, alias)          confidence 1.0
  StructuredFact  (osv, id) affects  (purl, …) or (pkg, eco:name)   confidence 1.0
  StructuredFact  (osv, id) fixed_in "eco:name@version"             confidence 1.0
  Tombstone       when a file disappears or the advisory is withdrawn

The cursor is a JSON manifest {id: modified}; resuming emits only new, changed or removed ids.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from hsi_core.connectors import ConnectorCaps, CursorCheckpoint, Record, StreamSpec
from hsi_core.model import PUBLIC, EntityRef, Section, SourceDocument, StructuredFact, Tombstone

NAME = "osv"
STREAM = StreamSpec(name="advisories", description="OSV vulnerability records")


def _scheme(alias: str) -> str:
    """OSV ids have the form <DATABASE>-<id> (OSV schema); the database prefix is the scheme."""
    db, sep, _ = alias.partition("-")
    return db.lower() if sep and db else "osv"


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class OsvConnector:
    name = NAME
    version = "0.1.0"

    def __init__(self, path: str, trust: float = 0.8) -> None:
        self.root = Path(path)
        if not self.root.is_dir():
            raise ValueError(f"OSV dump directory not found: {path}")
        self.trust = trust
        self.errors: list[tuple[str, str]] = []

    def capabilities(self) -> ConnectorCaps:
        return ConnectorCaps(incremental=True, emits_documents=True, emits_facts=True, emits_tombstones=True)

    def streams(self) -> list[StreamSpec]:
        return [STREAM]

    # ---------------------------------------------------------------------------------------
    def _scan(self) -> dict[str, tuple[str, dict[str, Any]]]:
        """id -> (modified, record). Invalid files are recorded in self.errors and skipped."""
        found: dict[str, tuple[str, dict[str, Any]]] = {}
        self.errors = []
        for f in sorted(self.root.rglob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                vid, modified = rec["id"], rec["modified"]
            except (OSError, ValueError, KeyError, TypeError) as e:
                self.errors.append((str(f), f"{type(e).__name__}: {e}"))
                continue
            found[vid] = (modified, rec)
        return found

    def sync(self, stream: StreamSpec, cursor: str | None) -> Iterator[Record]:
        if stream.name != STREAM.name:
            raise ValueError(f"unknown stream {stream.name!r}")
        before: dict[str, str] = json.loads(cursor) if cursor else {}
        now = self._scan()
        manifest: dict[str, str] = {}
        for vid in sorted(now):
            modified, rec = now[vid]
            manifest[vid] = modified
            if before.get(vid) == modified:
                continue
            if rec.get("withdrawn"):
                yield Tombstone(record_id=f"osv:{vid}", connector=NAME)
                continue
            yield from self._records(rec)
        for vid in sorted(set(before) - set(now)):
            yield Tombstone(record_id=f"osv:{vid}", connector=NAME)
        yield CursorCheckpoint(stream=STREAM.name, cursor=json.dumps(manifest, sort_keys=True))

    def _records(self, rec: dict[str, Any]) -> Iterator[Record]:
        vid = rec["id"]
        subj = EntityRef(scheme="osv", id=vid)
        aliases = sorted(set(rec.get("aliases", [])))
        affected_lines: list[str] = []
        facts: list[StructuredFact] = []
        for a in aliases:
            facts.append(StructuredFact(subject=subj, predicate="alias_of",
                                        object=EntityRef(scheme=_scheme(a), id=a), connector=NAME, acl=PUBLIC))
        for aff in rec.get("affected", []):
            pkg = aff.get("package", {})
            eco, pname, purl = pkg.get("ecosystem", "?"), pkg.get("name", "?"), pkg.get("purl")
            obj = EntityRef(scheme="purl", id=purl) if purl else EntityRef(scheme="pkg", id=f"{eco}:{pname}")
            facts.append(StructuredFact(subject=subj, predicate="affects", object=obj, connector=NAME, acl=PUBLIC))
            for rng in aff.get("ranges", []):
                intro = [e["introduced"] for e in rng.get("events", []) if "introduced" in e]
                fixed = [e["fixed"] for e in rng.get("events", []) if "fixed" in e]
                for fx in fixed:
                    facts.append(StructuredFact(subject=subj, predicate="fixed_in", object=f"{eco}:{pname}@{fx}",
                                                connector=NAME, acl=PUBLIC))
                affected_lines.append(f"{eco} package {pname}: introduced {', '.join(intro) or 'unknown'}; "
                                      f"fixed {', '.join(fixed) or 'not fixed'}.")
        sections = [Section(heading="Summary", text=rec.get("summary") or vid)]
        if rec.get("details"):
            sections.append(Section(heading="Details", text=rec["details"]))
        if affected_lines:
            sections.append(Section(heading="Affected", text=" ".join(affected_lines)))
        if aliases:
            sections.append(Section(heading="Aliases", text=f"{vid} is also known as {', '.join(aliases)}."))
        yield SourceDocument(id=f"osv:{vid}", connector=NAME, uri=f"https://osv.dev/vulnerability/{vid}",
                             title=vid, sections=tuple(sections), acl=PUBLIC, trust=self.trust,
                             updated_at=_parse_time(rec.get("modified")))
        yield from facts
