from __future__ import annotations

import threading

from hpclab.config import Paths, get_paths
from hpclab.domain.errors import AuthRequired, Unavailable
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxItem, OutboxRunner
from hpclab.storage.db import Database


def _ctx(paths: Paths) -> ServiceContext:
    return ServiceContext.create(paths)


def test_auth_required_never_kills_item(paths: Paths) -> None:
    ctx = _ctx(paths)
    runner = OutboxRunner(ctx, max_tries=2, auth_retry_s=0.0)
    calls: list[int] = []

    def handler(c: ServiceContext, item: OutboxItem) -> None:
        calls.append(item.tries)
        if len(calls) < 6:
            raise AuthRequired("run hpclab auth sim")

    runner.register("k", handler)
    with ctx.db.tx() as tx:
        tx.enqueue("k")
    runner.drain()
    assert len(calls) == 6
    assert ctx.db.one("SELECT state FROM outbox")["state"] == "done"


def test_retryable_errors_are_retried_beyond_max_tries(paths: Paths) -> None:
    ctx = _ctx(paths)
    runner = OutboxRunner(ctx, max_tries=2, max_backoff_s=0.0)
    n = {"c": 0}

    def handler(c: ServiceContext, item: OutboxItem) -> None:
        n["c"] += 1
        if n["c"] < 5:
            raise Unavailable("cluster unreachable")

    runner.register("k", handler)
    with ctx.db.tx() as tx:
        tx.enqueue("k")
    for _ in range(10):
        runner.run_once()
    assert n["c"] == 5 and ctx.db.one("SELECT state FROM outbox")["state"] == "done"


def test_close_all_closes_other_threads(paths: Paths) -> None:
    db = Database(paths.db)
    t = threading.Thread(target=lambda: db.one("SELECT 1"))
    t.start()
    t.join()
    db.one("SELECT 1")
    assert len(db._all_conns) == 2
    db.close_all()
    assert db._all_conns == []
    assert db.one("SELECT 1 AS x")["x"] == 1  # reconnects transparently


def test_socket_path_bytes_and_no_eager_tmp_dir(tmp_path) -> None:
    home = tmp_path / ("研究" * 12)
    p = get_paths(home)
    assert len(str(p.home / "run" / "controller.sock")) < 200
    sock = p.socket
    assert len(str(sock).encode()) <= 100
    p.ensure()
    if not str(sock).startswith(str(p.home)):
        assert not sock.parent.exists()
