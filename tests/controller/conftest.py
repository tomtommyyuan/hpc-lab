"""Fixtures for controller tests: fresh homes, in-thread controllers, subprocess cleanup."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from hpclab_ctl_toy import ToyApp, spawned_pids

from hpclab.config import Paths, get_paths
from hpclab.controller import client as client_mod
from hpclab.controller.client import stop_controller
from hpclab.controller.daemon import Controller
from hpclab.controller.files import lock_is_held, pid_alive, socket_path


@pytest.fixture
def make_home(tmp_path: Path) -> Iterator[Callable[..., Paths]]:
    created: list[Paths] = []

    def make(name: str = "home") -> Paths:
        p = get_paths(tmp_path / name)
        created.append(p)
        return p

    yield make

    for p in created:
        if lock_is_held(p.lock):
            with contextlib.suppress(Exception):
                stop_controller(p, timeout=5.0, force=True)
        for pid in spawned_pids(p):
            if pid != os.getpid() and pid_alive(pid):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        sock_dir = socket_path(p).parent
        if sock_dir != p.home / "run":
            shutil.rmtree(sock_dir, ignore_errors=True)
    # reap controllers spawned by clients during the test
    with client_mod._spawned_lock:
        procs = list(client_mod._spawned)
    ours = {pid for p in created for pid in spawned_pids(p)}
    for proc in procs:
        if proc.poll() is None:
            if proc.pid not in ours:
                continue
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)


@pytest.fixture
def home(make_home: Callable[..., Paths]) -> Paths:
    return make_home("home")


@pytest.fixture
def start_ctl(make_home: Callable[..., Paths]) -> Iterator[Callable[..., Controller]]:
    """Start an in-thread controller (toy app by default); all are stopped at teardown."""
    started: list[Controller] = []

    def start(paths: Paths | None = None, app_factory: Any = None, **kw: Any) -> Controller:
        kw.setdefault("tick_interval_s", 0.05)
        kw.setdefault("lock_wait_s", 0.0)
        kw.setdefault("shutdown_timeout_s", 5.0)
        ctl = Controller(paths or make_home(), app_factory or ToyApp, **kw)
        started.append(ctl)
        return ctl.start_in_thread()

    yield start

    for ctl in reversed(started):
        ctl.stop(timeout=5.0)
