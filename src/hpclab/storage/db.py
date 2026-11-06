"""SQLite ledger access.

* WAL mode on a *local* disk (never NFS/scratch: WAL needs a single host's shared memory).
* One connection per thread; ``Database.tx()`` opens ``BEGIN IMMEDIATE`` so writers
  serialize and readers never see half a command.
* Helpers here are deliberately small: services own their SQL. What lives here is what
  every service must do the same way — events, outbox, idempotency, versioned updates.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, NamedTuple

from hpclab import SCHEMA_VERSION
from hpclab.domain.clock import Clock, SystemClock, iso
from hpclab.domain.errors import IdempotencyConflict, NotFound, VersionConflict
from hpclab.domain.hashing import canonical_json
from hpclab.domain.ids import new_id

MIGRATIONS: list[str] = ["schema.sql", "schema_002_v1_extras.sql"]
Row = dict[str, Any]


def _dict_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> Row:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def jdump(value: Any) -> str:
    return canonical_json(value)


def jload(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


class Tx:
    """A write transaction handle. All helpers run on the same connection."""

    def __init__(self, conn: sqlite3.Connection, clock: Clock, source: str, command_id: str | None) -> None:
        self.conn = conn
        self.clock = clock
        self.source = source
        self.command_id = command_id
        self._now = iso(clock.now())

    @property
    def now(self) -> str:
        assert self._now is not None
        return self._now

    # -- raw ---------------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Row | None:
        row: Row | None = self.conn.execute(sql, params).fetchone()
        return row

    def all(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        cols = ", ".join(row.keys())
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))

    def update(
        self,
        table: str,
        key_col: str,
        key: str,
        values: Mapping[str, Any],
        *,
        expected_version: int | None = None,
        bump_version: bool = True,
    ) -> int:
        """Update one row; optionally compare-and-set on ``version``. Returns the new version."""
        sets = dict(values)
        if "updated_at" not in sets and table not in ("artifacts",):
            sets["updated_at"] = self.now
        assignments = ", ".join(f"{c} = ?" for c in sets)
        params: list[Any] = list(sets.values())
        if bump_version:
            assignments += ", version = version + 1"
        sql = f"UPDATE {table} SET {assignments} WHERE {key_col} = ?"
        params.append(key)
        if expected_version is not None:
            sql += " AND version = ?"
            params.append(expected_version)
        cur = self.conn.execute(sql, params)
        if cur.rowcount != 1:
            exists = self.one(f"SELECT version FROM {table} WHERE {key_col} = ?", (key,))
            if exists is None:
                raise NotFound(f"{table} {key} not found")
            raise VersionConflict(
                f"{table} {key} is at version {exists['version']}, expected {expected_version}",
                details={"current_version": exists["version"], "expected_version": expected_version},
            )
        if not bump_version:
            return -1
        row = self.one(f"SELECT version FROM {table} WHERE {key_col} = ?", (key,))
        assert row is not None
        return int(row["version"])

    # -- events -------------------------------------------------------------
    def event(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        entity_version: int | None = None,
        causation_id: str | None = None,
        observed_at: str | None = None,
        source: str | None = None,
    ) -> str:
        event_id = new_id("event")
        self.insert(
            "events",
            {
                "event_id": event_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_version": entity_version,
                "event_type": event_type,
                "payload_json": jdump(dict(payload or {})),
                "command_id": self.command_id,
                "causation_id": causation_id,
                "source": source or self.source,
                "observed_at": observed_at,
                "recorded_at": self.now,
                "schema_version": SCHEMA_VERSION,
            },
        )
        return event_id

    # -- outbox -------------------------------------------------------------
    def enqueue(
        self,
        kind: str,
        *,
        operation_id: str | None = None,
        target_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        delay_s: float = 0.0,
        dedupe_key: str | None = None,
    ) -> int | None:
        """Add a durable work item. With ``dedupe_key`` a second enqueue is a no-op
        while the first is still pending/leased (returns None)."""
        not_before = iso(self.clock.now() + timedelta(seconds=delay_s))
        if dedupe_key is not None:
            existing = self.one("SELECT outbox_id, state FROM outbox WHERE dedupe_key = ?", (dedupe_key,))
            if existing is not None:
                if existing["state"] in ("pending", "leased"):
                    return None
                # finished item with same key: free the key for the new logical step
                self.execute(
                    "UPDATE outbox SET dedupe_key = NULL WHERE outbox_id = ?", (existing["outbox_id"],)
                )
        cur = self.execute(
            "INSERT INTO outbox (kind, operation_id, target_id, payload_json, state, not_before, dedupe_key,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (
                kind,
                operation_id,
                target_id,
                jdump(dict(payload or {})),
                not_before,
                dedupe_key,
                self.now,
                self.now,
            ),
        )
        return cur.lastrowid

    # -- operations ---------------------------------------------------------
    def create_operation(
        self,
        kind: str,
        *,
        target_type: str | None,
        target_id: str | None,
        stage: str | None = None,
        state: str = "queued",
        poll_after_ms: int = 2000,
        operation_id: str | None = None,
    ) -> str:
        op_id = operation_id or new_id("operation")
        self.insert(
            "operations",
            {
                "operation_id": op_id,
                "kind": kind,
                "state": state,
                "stage": stage,
                "target_type": target_type,
                "target_id": target_id,
                "progress_json": "{}",
                "poll_after_ms": poll_after_ms,
                "version": 1,
                "created_at": self.now,
                "updated_at": self.now,
            },
        )
        self.event(
            "operation", op_id, "operation.created", {"kind": kind, "target_id": target_id, "state": state}
        )
        return op_id

    def set_operation(
        self,
        operation_id: str,
        *,
        state: str | None = None,
        stage: str | None = None,
        result: Any = None,
        error: Mapping[str, Any] | None = None,
        progress: Mapping[str, Any] | None = None,
        poll_after_ms: int | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if state is not None:
            values["state"] = state
            if state in ("succeeded", "failed"):
                values["finished_at"] = self.now
        if stage is not None:
            values["stage"] = stage
        if result is not None:
            values["result_json"] = jdump(result)
        if error is not None:
            values["error_json"] = jdump(dict(error))
        if progress is not None:
            values["progress_json"] = jdump(dict(progress))
        if poll_after_ms is not None:
            values["poll_after_ms"] = poll_after_ms
        if not values:
            return
        version = self.update("operations", "operation_id", operation_id, values)
        self.event(
            "operation",
            operation_id,
            "operation.updated",
            {k: v for k, v in values.items() if k in ("state", "stage")},
            entity_version=version,
        )

    # -- idempotency ----------------------------------------------------------
    def idem_lookup(self, principal: str, tool: str, key: str, payload_hash: str) -> Any | None:
        """Return the stored response for (principal, tool, key) if payload matches.

        Raises ``IdempotencyConflict`` when the key was used with a different payload.
        """
        row = self.one(
            "SELECT payload_hash, response_json FROM idempotency WHERE principal=? AND tool=? AND idem_key=?",
            (principal, tool, key),
        )
        if row is None:
            return None
        if row["payload_hash"] != payload_hash:
            raise IdempotencyConflict(
                f"idempotency_key {key!r} was already used for a different {tool} request",
                details={"tool": tool, "idempotency_key": key},
            )
        return jload(row["response_json"])

    def idem_store(self, principal: str, tool: str, key: str, payload_hash: str, response: Any) -> None:
        self.insert(
            "idempotency",
            {
                "principal": principal,
                "tool": tool,
                "idem_key": key,
                "payload_hash": payload_hash,
                "response_json": jdump(response),
                "created_at": self.now,
            },
        )


class _OwnedConn(NamedTuple):
    """A per-thread connection and who opened it (see ``Database.conn``)."""

    conn: sqlite3.Connection
    thread: threading.Thread
    pid: int


class Database:
    def __init__(self, path: str | Path, clock: Clock | None = None) -> None:
        self.path = str(path)
        self.clock: Clock = clock or SystemClock()
        self._local = threading.local()
        self._lock = threading.Lock()
        self._conns_lock = threading.Lock()  # separate from _lock: migrate() holds _lock while connecting
        self._all_conns: list[_OwnedConn] = []
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    # connections ---------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = _dict_factory
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """The calling thread's connection, opened on first use.

        The controller runs every request, periodic task run and socket connection on a fresh
        thread, so connections are tracked with their owning thread and those of threads that
        have ended are closed whenever a new one is opened (``_reap_finished``). Without that,
        every short-lived thread would leak a connection (two descriptors: the database and its
        WAL) and a long-running controller would hit EMFILE."""
        c: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if c is None:
            self._reap_finished()
            c = self._connect()
            self._local.conn = c
            with self._conns_lock:
                self._all_conns.append(_OwnedConn(c, threading.current_thread(), os.getpid()))
        return c

    def _reap_finished(self) -> None:
        """Close the connections of threads that have ended.

        A finished thread's thread-local is gone, so nothing can use its connection any more.
        Connections opened by another process (inherited through ``fork()`` without exec, where
        every thread but the forking one looks finished) are left alone: an SQLite connection
        must never be used — closing included — across ``fork()``."""
        pid = os.getpid()
        keep: list[_OwnedConn] = []
        finished: list[_OwnedConn] = []
        with self._conns_lock:
            for o in self._all_conns:
                (finished if o.pid == pid and not o.thread.is_alive() else keep).append(o)
            if not finished:
                return
            self._all_conns = keep
        for o in finished:
            with contextlib.suppress(sqlite3.Error):
                o.conn.close()

    def close(self) -> None:
        """Close the calling thread's connection."""
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None
            with self._conns_lock:
                self._all_conns = [o for o in self._all_conns if o.conn is not c]

    def close_all(self) -> None:
        """Close every connection opened by any thread (shutdown). Threads that use the
        database afterwards get a fresh connection."""
        with self._conns_lock:
            owned, self._all_conns = self._all_conns, []
        for o in owned:
            with contextlib.suppress(sqlite3.Error):
                o.conn.close()
        self._local = threading.local()

    # reads -----------------------------------------------------------------------
    def one(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> Row | None:
        row: Row | None = self.conn.execute(sql, params).fetchone()
        return row

    def all(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def get(self, table: str, key_col: str, key: str) -> Row:
        row = self.one(f"SELECT * FROM {table} WHERE {key_col} = ?", (key,))
        if row is None:
            raise NotFound(f"{table.rstrip('s')} {key} not found", details={"table": table, "id": key})
        return row

    # writes ------------------------------------------------------------------------
    @contextmanager
    def tx(self, source: str = "api", command_id: str | None = None) -> Iterator[Tx]:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield Tx(conn, self.clock, source, command_id)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # migrations ----------------------------------------------------------------------
    def migrate(self) -> None:
        with self._lock:
            conn = self.conn
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM meta WHERE key='migration'").fetchone()
            applied = int(row["value"]) if row else 0
            for idx, name in enumerate(MIGRATIONS, start=1):
                if idx <= applied:
                    continue
                sql = resources.files("hpclab.storage").joinpath(name).read_text()
                # The version bump is part of the same transaction: ALTER TABLE is not
                # idempotent, so a crash must never leave a migration applied but unrecorded.
                bump = (
                    f"INSERT INTO meta(key, value) VALUES('migration', '{idx}') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value;"
                )
                conn.executescript("BEGIN IMMEDIATE;\n" + sql + "\n" + bump + "\nCOMMIT;")

    def now(self) -> datetime:
        return self.clock.now()
