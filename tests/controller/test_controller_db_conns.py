"""A long-running controller must not accumulate SQLite connections.

Each app request (and each periodic task run and socket connection) runs on a fresh thread
and touches the ledger through ``Database.conn``, which opens one connection per thread. The
connections of finished threads must be released, or the controller runs out of file
descriptors ("unable to open database file") after a few hundred requests under a 256 fd
limit (macOS default) and stops working until restarted by hand."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hpclab.config import Paths
from hpclab.controller.client import HpcLabClient
from hpclab.controller.daemon import Controller, MinimalApp, default_app_factory

pytestmark = pytest.mark.timeout(60)

StartCtl = Callable[..., Controller]


def _open_fds() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        if Path(d).is_dir():
            return len(os.listdir(d))
    pytest.skip("cannot count open file descriptors on this platform")


def test_requests_do_not_accumulate_db_connections(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home, app_factory=default_app_factory)
    assert not isinstance(ctl.app, MinimalApp), getattr(ctl.app, "load_error", None)
    with HpcLabClient(home, autostart=False, timeout=10.0) as c:
        for _ in range(10):  # warm up: lazily opened files, periodic threads, caches
            env: dict[str, Any] = c.call("clusters_list", {})
            assert env["ok"], env
        conns_before, fds_before = len(ctl.ctx.db._all_conns), _open_fds()
        n = 150
        for _ in range(n):
            env = c.call("clusters_list", {})
            assert env["ok"], env
        conns_after, fds_after = len(ctl.ctx.db._all_conns), _open_fds()
    # Leaking would add one connection (two descriptors) per request: +150 / +300 here.
    # Only live threads (a handful of workers, request and periodic threads) may hold one.
    assert conns_after - conns_before <= 20, (conns_before, conns_after)
    assert fds_after - fds_before <= 40, (fds_before, fds_after)
