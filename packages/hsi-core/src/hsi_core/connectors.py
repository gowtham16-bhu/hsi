"""The generic connector contract (specs/additions/A2 §2).

A connector can wrap anything: a product API, a database, files, chat, an event stream.
The core sees only these records.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from hsi_core.model import Acl, SourceDocument, StructuredFact, Tombstone


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StreamSpec(_Frozen):
    """A named stream of records inside a source, e.g. "advisories", "work_items", "pages"."""

    name: str = Field(min_length=1)
    description: str = ""


class RateLimit(_Frozen):
    requests_per_second: float | None = Field(default=None, gt=0)


class ConnectorCaps(_Frozen):
    incremental: bool
    push: bool = False
    emits_documents: bool = True
    emits_facts: bool = False
    emits_tombstones: bool = True
    identity_mapping: bool = False
    rate_limit: RateLimit = RateLimit()


class AclUpdate(_Frozen):
    record_id: str = Field(min_length=1)
    connector: str = Field(min_length=1)
    acl: Acl


class CursorCheckpoint(_Frozen):
    """Opaque resume point. A consumer stores the last checkpoint it has fully applied."""

    stream: str = Field(min_length=1)
    cursor: str


Record = SourceDocument | StructuredFact | AclUpdate | Tombstone | CursorCheckpoint


@runtime_checkable
class Connector(Protocol):
    name: str
    version: str

    def capabilities(self) -> ConnectorCaps: ...

    def streams(self) -> list[StreamSpec]: ...

    def sync(self, stream: StreamSpec, cursor: str | None) -> Iterator[Record]:
        """Yield records changed since `cursor` (everything when None), ending with a CursorCheckpoint."""
        ...
