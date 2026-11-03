"""C1: canonical model. Every core module speaks only these types (specs/02-architecture.md §3)."""
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def content_hash(text: str) -> str:
    """Stable hash of whitespace-normalized text."""
    norm = " ".join(text.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class EntityRef(_Frozen):
    """A canonical identifier, e.g. ("purl", "pkg:pypi/requests@2.31.0") or ("title", "mark zuckerberg")."""

    scheme: str = Field(min_length=1)
    id: str = Field(min_length=1)

    @property
    def key(self) -> str:
        return f"{self.scheme}:{self.id}"


class Acl(_Frozen):
    """Who may see a record. A record is either public or names at least one principal."""

    principals: frozenset[str] = frozenset()
    public: bool = False

    @model_validator(mode="after")
    def _must_grant_someone(self) -> Acl:
        if not self.public and not self.principals:
            raise ValueError("Acl must be public or name at least one principal")
        return self

    def allows(self, principals: frozenset[str]) -> bool:
        return self.public or bool(self.principals & principals)


PUBLIC = Acl(public=True)


class Section(_Frozen):
    heading: str | None = None
    text: str


class SourceDocument(_Frozen):
    id: str = Field(min_length=1)
    connector: str = Field(min_length=1)
    uri: str | None = None
    title: str | None = None
    sections: tuple[Section, ...] = Field(min_length=1)
    acl: Acl
    updated_at: datetime | None = None
    trust: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: tuple[tuple[str, str], ...] = ()


class StructuredFact(_Frozen):
    subject: EntityRef
    predicate: str = Field(min_length=1)
    object: EntityRef | str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    connector: str = Field(min_length=1)
    acl: Acl


class Tombstone(_Frozen):
    record_id: str = Field(min_length=1)
    connector: str = Field(min_length=1)


class Passage(_Frozen):
    id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    title: str | None = None
    section_path: tuple[str, ...] = ()
    text: str = Field(min_length=1)
    hash: str = Field(min_length=64, max_length=64)
    acl: Acl

    @classmethod
    def build(cls, *, doc_id: str, text: str, acl: Acl, title: str | None = None,
              section_path: tuple[str, ...] = (), ordinal: int = 0) -> Passage:
        h = content_hash(f"{title or ''}\n{text}")
        pid = hashlib.sha256(f"{doc_id}\x1f{ordinal}\x1f{h}".encode()).hexdigest()[:24]
        return cls(id=pid, doc_id=doc_id, title=title, section_path=section_path, text=text, hash=h, acl=acl)


class EdgeType(StrEnum):
    MENTION = "mention"          # passage mentions an entity whose anchor is another passage
    CO_MENTION = "co_mention"    # both passages mention the same entity (decision 0001)
    STRUCTURED = "structured"    # from a structured fact
    REFERENCE = "reference"      # explicit reference / hyperlink in source metadata
    THREAD = "thread"
    SECTION = "section"


class Edge(_Frozen):
    """Directed passage-to-passage edge through a bridge entity."""

    src: str
    dst: str
    type: EdgeType
    weight: float = Field(gt=0.0, le=1.0)
    bridge: str = Field(min_length=1)       # entity key the hop goes through
    salience: float = Field(default=1.0, ge=0.0, le=1.0)  # how prominent the mention is in src

    @model_validator(mode="after")
    def _no_self_loop(self) -> Edge:
        if self.src == self.dst:
            raise ValueError("self-loop edge")
        return self


class Chain(_Frozen):
    """An ordered evidence chain of 1–4 passages joined by bridge entities."""

    passages: tuple[str, ...] = Field(min_length=1, max_length=4)
    bridges: tuple[str, ...] = ()
    log_prob: float = Field(default=0.0, le=0.0)

    @field_validator("passages")
    @classmethod
    def _unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("chain repeats a passage")
        return v

    @model_validator(mode="after")
    def _bridges_match(self) -> Chain:
        if len(self.bridges) != len(self.passages) - 1:
            raise ValueError("a chain of n passages needs n-1 bridges")
        return self

    @property
    def level(self) -> int:
        """0 = single passage, 1 = 2-passage chain, 2 = 3-passage chain, 3 = 4-passage chain."""
        return len(self.passages) - 1

    @property
    def id(self) -> str:
        return hashlib.sha256("\x1f".join(self.passages).encode()).hexdigest()[:24]


class KeyMode(StrEnum):
    TEXT = "text"           # encode the joined chain text (Gate 0b)
    MEAN = "mean"           # normalize(sum of passage vectors): no training, no text encoding (Gate 0a)
    COMPOSED = "composed"   # composer over passage vectors (Gate 1)
    HYBRID = "hybrid"


class Shortcut(_Frozen):
    chain: Chain
    key_mode: KeyMode
    benefit: float = 0.0
    version: int = 1

    @property
    def id(self) -> str:
        return self.chain.id
