"""PostgreSQL metadata store. Schema migrations are plain SQL, applied in order and recorded."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any

import psycopg

from hsi_core.model import Edge, Passage, Shortcut

SCHEMA = "hsi"


def _migrations() -> list[tuple[int, str]]:
    root = files(__package__).joinpath("migrations")
    out = []
    for entry in root.iterdir():
        name = entry.name
        if name.endswith(".sql"):
            out.append((int(name.split("_", 1)[0]), entry.read_text(encoding="utf-8")))
    return sorted(out)


class PostgresStore:
    def __init__(self, dsn: str | None = None, dsn_env: str = "HSI_PG_DSN") -> None:
        dsn = dsn or os.environ.get(dsn_env)
        if not dsn:
            raise ValueError(f"no PostgreSQL DSN: pass dsn= or set {dsn_env}")
        self._conn = psycopg.connect(dsn, autocommit=False)

    # schema ---------------------------------------------------------------
    def migrate(self) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"create schema if not exists {SCHEMA}")
            cur.execute(f"create table if not exists {SCHEMA}.schema_version("
                        "version int primary key, applied_at timestamptz not null default now())")
            cur.execute(f"select version from {SCHEMA}.schema_version")
            done = {r[0] for r in cur.fetchall()}
            for version, sql in _migrations():
                if version not in done:
                    cur.execute(sql)
                    cur.execute(f"insert into {SCHEMA}.schema_version(version) values (%s)", (version,))

    def schema_version(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"select coalesce(max(version), 0) from {SCHEMA}.schema_version")
            return int(cur.fetchone()[0])

    # writes (upsert semantics through a temporary staging table) -----------
    def _upsert(self, table: str, cols: list[str], key: list[str], rows: Iterable[tuple]) -> int:
        stage = f"stage_{table}"
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in key) or f"{cols[0]} = excluded.{cols[0]}"
        n = 0
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"create temp table {stage} (like {SCHEMA}.{table} including defaults) on commit drop")
            with cur.copy(f"copy {stage} ({', '.join(cols)}) from stdin") as cp:
                for r in rows:
                    cp.write_row(r)
                    n += 1
            cur.execute(f"insert into {SCHEMA}.{table} ({', '.join(cols)}) "
                        f"select {', '.join(cols)} from {stage} "
                        f"on conflict ({', '.join(key)}) do update set {updates}")
        return n

    def write_passages(self, corpus_id: str, passages: Iterable[Passage]) -> int:
        rows = ((corpus_id, p.id, p.doc_id, p.title, p.text, p.hash,
                 json.dumps({"public": p.acl.public, "principals": sorted(p.acl.principals)}))
                for p in passages)
        return self._upsert("passages", ["corpus_id", "id", "doc_id", "title", "text", "hash", "acl"],
                            ["corpus_id", "id"], rows)

    def write_edges(self, corpus_id: str, edges: Iterable[Edge]) -> int:
        rows = ((corpus_id, e.src, e.dst, e.type.value, e.weight, e.bridge, e.salience) for e in edges)
        return self._upsert("edges", ["corpus_id", "src", "dst", "type", "weight", "bridge", "salience"],
                            ["corpus_id", "src", "dst", "bridge"], rows)

    def write_shortcuts(self, corpus_id: str, shortcuts: Iterable[Shortcut]) -> int:
        rows = ((corpus_id, s.id, list(s.chain.passages), list(s.chain.bridges), s.chain.log_prob,
                 s.chain.level, s.key_mode.value, s.benefit, s.version) for s in shortcuts)
        return self._upsert("shortcuts", ["corpus_id", "id", "passages", "bridges", "log_prob", "level",
                                          "key_mode", "benefit", "version"], ["corpus_id", "id"], rows)

    def record_run(self, run: dict[str, Any]) -> str:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(f"insert into {SCHEMA}.runs(kind, config_hash, code_version, data_hash, payload) "
                        "values (%s, %s, %s, %s, %s) returning id",
                        (run.get("kind", "unknown"), run.get("config_hash"), run.get("code_version"),
                         run.get("data_hash"), json.dumps(run, default=str)))
            return str(cur.fetchone()[0])

    # reads ------------------------------------------------------------------
    def count(self, table: str, corpus_id: str) -> int:
        if table not in {"passages", "edges", "shortcuts"}:
            raise ValueError("unknown table")
        with self._conn.cursor() as cur:
            cur.execute(f"select count(*) from {SCHEMA}.{table} where corpus_id = %s", (corpus_id,))
            return int(cur.fetchone()[0])

    def read_passages(self, corpus_id: str, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Passage rows by id: doc_id, title, text, acl."""
        ids = list(ids)
        if not ids:
            return {}
        with self._conn.cursor() as cur:
            cur.execute(f"select id, doc_id, title, text, acl from {SCHEMA}.passages "
                        "where corpus_id = %s and id = any(%s)", (corpus_id, ids))
            rows = {r[0]: {"doc_id": r[1], "title": r[2], "text": r[3], "acl": r[4]} for r in cur.fetchall()}
        self._conn.commit()
        return rows

    def read_acls(self, corpus_id: str) -> dict[str, dict[str, Any]]:
        """ACL of every passage of a corpus (loaded once by the retrieval service)."""
        with self._conn.cursor() as cur:
            cur.execute(f"select id, acl from {SCHEMA}.passages where corpus_id = %s", (corpus_id,))
            out = {r[0]: r[1] for r in cur.fetchall()}
        self._conn.commit()
        return out

    # audit (append-only, hash-chained) -----------------------------------------
    @staticmethod
    def _audit_hash(prev: str, entry: dict[str, Any]) -> str:
        blob = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + blob).encode("utf-8")).hexdigest()

    def append_audit(self, index_id: str, principals: Iterable[str], question: str, passages: Iterable[str]) -> str:
        entry = {"at": datetime.now(UTC).isoformat(), "index_id": index_id,
                 "principals": sorted(principals), "question": question, "passages": list(passages)}
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("select pg_advisory_xact_lock(hashtext('hsi.audit'))")
            cur.execute(f"select hash from {SCHEMA}.audit order by seq desc limit 1")
            row = cur.fetchone()
            prev = row[0] if row else "0" * 64
            h = self._audit_hash(prev, entry)
            cur.execute(f"insert into {SCHEMA}.audit(at, index_id, principals, question, passages, prev_hash, hash) "
                        "values (%s, %s, %s, %s, %s, %s, %s)",
                        (entry["at"], index_id, json.dumps(entry["principals"]), question,
                         json.dumps(entry["passages"]), prev, h))
        return h

    def verify_audit(self) -> int | None:
        """Recompute the chain; return the seq of the first bad row, or None when intact."""
        prev = "0" * 64
        with self._conn.cursor() as cur:
            cur.execute(f"select seq, at, index_id, principals, question, passages, prev_hash, hash "
                        f"from {SCHEMA}.audit order by seq")
            for seq, at, index_id, principals, question, passages, prev_hash, h in cur.fetchall():
                entry = {"at": at.astimezone(UTC).isoformat(), "index_id": index_id, "principals": principals,
                         "question": question, "passages": passages}
                if prev_hash != prev or self._audit_hash(prev, entry) != h:
                    self._conn.commit()
                    return int(seq)
                prev = h
        self._conn.commit()
        return None

    def delete_corpus(self, corpus_id: str) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            for t in ("shortcuts", "edges", "passages"):
                cur.execute(f"delete from {SCHEMA}.{t} where corpus_id = %s", (corpus_id,))

    def close(self) -> None:
        self._conn.close()
