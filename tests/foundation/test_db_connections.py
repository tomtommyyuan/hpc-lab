"""Per-thread SQLite connections must not outlive their threads.

The controller runs every request, periodic task run and socket connection on a fresh
thread. ``Database.conn`` opens one connection per thread; if the database keeps a strong
reference to each of them forever, every short-lived thread leaks a connection (two file
descriptors: the database and its WAL) and a long-running controller eventually hits
EMFILE ("unable to open database file")."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from hpclab.config import Paths
from hpclab.storage.db import Database

pytestmark = pytest.mark.timeout(60)


def _open_fds() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        if Path(d).is_dir():
            return len(os.listdir(d))
    pytest.skip("cannot count open file descriptors on this platform")


def _in_thread(db: Database) -> sqlite3.Connection:
    """Use the database on a fresh thread that then ends; return that thread's connection."""
    got: list[sqlite3.Connection] = []

    def body() -> None:
        assert db.one("SELECT 1 AS x") == {"x": 1}
        got.append(db.conn)

    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert got, "worker thread failed"
    return got[0]


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_connections_of_finished_threads_are_closed_and_dropped(paths: Paths) -> None:
    db = Database(paths.db)
    try:
        conns = [_in_thread(db) for _ in range(50)]
        # Each new thread's connect reaps those of the threads that already ended: only the
        # main thread's and the most recent worker's connection can still be tracked.
        assert len(db._all_conns) <= 3
        assert all(_is_closed(c) for c in conns[:-1])
        assert db.one("SELECT 1 AS x") == {"x": 1}  # the main thread's connection is untouched
    finally:
        db.close_all()


def test_open_fds_stay_bounded_across_many_short_lived_threads(paths: Paths) -> None:
    db = Database(paths.db)
    try:
        _in_thread(db)  # warm up (WAL/shm files exist from here on)
        before = _open_fds()
        for _ in range(200):
            _in_thread(db)
        after = _open_fds()
        # Leaking one connection per thread would add ~400 descriptors here.
        assert after - before <= 10, (before, after)
        assert len(db._all_conns) <= 3
    finally:
        db.close_all()


def test_live_threads_keep_their_connection(paths: Paths) -> None:
    db = Database(paths.db)
    release = threading.Event()
    ready = threading.Event()
    held: list[sqlite3.Connection] = []
    results: list[object] = []

    def long_lived() -> None:
        held.append(db.conn)
        ready.set()
        release.wait(30)
        # still usable after many other threads came and went
        results.append(db.one("SELECT 2 AS x"))
        results.append(db.conn is held[0])

    t = threading.Thread(target=long_lived)
    t.start()
    try:
        assert ready.wait(10)
        for _ in range(20):
            _in_thread(db)
        assert not _is_closed(held[0])
    finally:
        release.set()
        t.join()
    assert results == [{"x": 2}, True]
    _in_thread(db)  # the long-lived thread has ended now: its connection is reaped
    assert _is_closed(held[0])
    db.close_all()


def test_close_all_still_closes_live_threads_connections(paths: Paths) -> None:
    db = Database(paths.db)
    main = db.conn
    worker_conn = _in_thread(db)
    db.close_all()
    assert not db._all_conns
    assert _is_closed(main) and _is_closed(worker_conn)
    assert db.one("SELECT 1 AS x") == {"x": 1}  # reconnects transparently
    db.close_all()


def test_close_untracks_only_the_calling_threads_connection(paths: Paths) -> None:
    db = Database(paths.db)
    release = threading.Event()
    ready = threading.Event()
    held: list[sqlite3.Connection] = []

    def other() -> None:
        held.append(db.conn)
        ready.set()
        release.wait(30)

    t = threading.Thread(target=other)
    t.start()
    try:
        assert ready.wait(10)
        main = db.conn
        db.close()
        assert _is_closed(main) and not _is_closed(held[0])
        assert len(db._all_conns) == 1
    finally:
        release.set()
        t.join()
    db.close_all()


def test_connections_created_before_fork_are_not_closed_in_the_child(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forked child must never close (and thereby touch) connections its parent opened:
    SQLite connections must not be used across ``fork()``. The pid check is exercised by
    pretending the current process is a different one."""
    db = Database(paths.db)
    try:
        worker_conn = _in_thread(db)  # its thread has ended; normally reaped on next connect
        real_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
        _in_thread(db)
        assert not _is_closed(worker_conn)
    finally:
        monkeypatch.undo()
        db.close_all()


def test_many_threads_concurrently(paths: Paths) -> None:
    """Concurrent connects and reaps never close a connection that is still in use."""
    db = Database(paths.db)
    errors: list[BaseException] = []

    def body() -> None:
        try:
            for _ in range(5):
                with db.tx() as tx:
                    tx.execute("CREATE TABLE IF NOT EXISTS t_conc (x INTEGER)")
                    tx.execute("INSERT INTO t_conc(x) VALUES (1)")
                assert db.one("SELECT COUNT(*) AS n FROM t_conc")["n"] >= 1  # type: ignore[index]
        except BaseException as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    try:
        for _ in range(5):
            threads = [threading.Thread(target=body) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert not errors, errors
        assert db.one("SELECT COUNT(*) AS n FROM t_conc") == {"n": 200}
        assert len(db._all_conns) <= 10
    finally:
        db.close_all()
