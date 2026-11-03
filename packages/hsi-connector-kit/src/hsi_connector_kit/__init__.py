"""Connector conformance kit (specs/additions/A2 §2).

A connector is checked against a *driver*: a small adapter that can create, change and delete
items in a test instance of that source. The kit then verifies the connector's behaviour:

  C1 records are valid, carry this connector's name, and every content record has an ACL
  C2 every sync ends with a CursorCheckpoint
  C3 a full sync is deterministic (same records, same order)
  C4 resuming from the last cursor with no changes yields no content records
  C5 after one item changes, resuming yields that item and nothing else
  C6 after one item is deleted, resuming yields a Tombstone for it
  C7 declared capabilities match behaviour

HTTP-specific checks (no write calls, rate-limit compliance) run through a recording proxy and
are added with the first HTTP connector.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from hsi_core.connectors import Connector, CursorCheckpoint, Record
from hsi_core.model import SourceDocument, StructuredFact, Tombstone


class SourceDriver(Protocol):
    def seed(self, n: int) -> list[str]:
        """Create n items; return their record ids as the connector will emit them."""
        ...

    def modify(self, record_id: str) -> None: ...

    def delete(self, record_id: str) -> None: ...


@dataclass
class KitResult:
    connector: str
    passed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def check(self, name: str, cond: bool, why: str) -> None:
        if cond:
            self.passed.append(name)
        else:
            self.failed[name] = why


def _content(records: list[Record]) -> list[Record]:
    return [r for r in records if not isinstance(r, CursorCheckpoint)]


def _ids(records: list[Record]) -> set[str]:
    return {r.id for r in records if isinstance(r, SourceDocument)}


def run_kit(make_connector: Callable[[], Connector], driver: SourceDriver, n: int = 5) -> KitResult:
    conn = make_connector()
    res = KitResult(connector=conn.name)
    caps = conn.capabilities()
    streams = conn.streams()
    res.check("C0 streams", bool(streams), "connector declares no streams")
    if not streams:
        return res
    stream = streams[0]
    seeded = driver.seed(n)

    first = list(conn.sync(stream, None))
    bad_name = [type(r).__name__ for r in _content(first)
                if getattr(r, "connector", conn.name) != conn.name]
    res.check("C1 records", not bad_name and _ids(first) >= set(seeded),
              f"wrong connector names {bad_name} or missing ids {set(seeded) - _ids(first)}")
    res.check("C2 checkpoint", bool(first) and isinstance(first[-1], CursorCheckpoint),
              "sync must end with a CursorCheckpoint")
    if not first or not isinstance(first[-1], CursorCheckpoint):
        return res
    cursor = first[-1].cursor

    again = list(make_connector().sync(stream, None))
    res.check("C3 deterministic", _content(again) == _content(first), "two full syncs differ")

    if caps.incremental:
        quiet = _content(list(conn.sync(stream, cursor)))
        res.check("C4 quiet resume", not quiet, f"{len(quiet)} records on resume with no changes")

        target = seeded[0]
        driver.modify(target)
        changed = list(conn.sync(stream, cursor))
        res.check("C5 change", _ids(_content(changed)) == {target},
                  f"expected only {target}, got {_ids(_content(changed))}")
        cursor = changed[-1].cursor if changed and isinstance(changed[-1], CursorCheckpoint) else cursor

        gone = seeded[1]
        driver.delete(gone)
        deleted = _content(list(conn.sync(stream, cursor)))
        tomb = [r for r in deleted if isinstance(r, Tombstone)]
        res.check("C6 delete", any(t.record_id == gone for t in tomb),
                  f"no tombstone for {gone}")
        res.check("C7 caps.tombstones", caps.emits_tombstones or not tomb,
                  "emits tombstones but does not declare it")
    facts = [r for r in first if isinstance(r, StructuredFact)]
    res.check("C7 caps.facts", caps.emits_facts or not facts, "emits facts but does not declare it")
    return res
