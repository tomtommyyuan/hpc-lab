"""Client autostart, concurrency, lifecycle helpers and crash recovery — with real processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from hpclab_ctl_toy import ToyApp, spawned_pids, toy_spawn_cmd, wait_for

from hpclab.config import Paths
from hpclab.controller import protocol as proto
from hpclab.controller.client import HpcLabClient, controller_status, start_controller, stop_controller
from hpclab.controller.daemon import Controller
from hpclab.controller.files import lock_is_held, pid_alive, read_pid
from hpclab.domain.errors import ControllerUnavailable, HpcLabError
from hpclab.storage.db import Database

pytestmark = pytest.mark.timeout(45)


def toy_client(paths: Paths, *extra: str, **kw: Any) -> HpcLabClient:
    return HpcLabClient(paths, spawn_cmd=toy_spawn_cmd(paths, *extra), timeout=10.0, **kw)


def db_rows(paths: Paths, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    db = Database(paths.db)
    try:
        return db.all(sql, params)
    finally:
        db.close()


def test_autostart_then_stop(home: Paths) -> None:
    with toy_client(home) as c:
        env = c.call("toy.pid")
        pid = env["data"]["pid"]
        assert env["ok"] and pid != os.getpid() and pid_alive(pid)
        assert c.controller_info is not None and c.controller_info["pid"] == pid
        assert spawned_pids(home) == [pid]
        assert read_pid(home.pidfile) == pid and lock_is_held(home.lock)
        # later calls reuse the running controller (no new spawn)
        assert c.call("toy.pid")["data"]["pid"] == pid

    status = controller_status(home)
    assert status["ok"] and status["data"]["running"] and status["data"]["pid"] == pid

    result = stop_controller(home, timeout=10)
    assert result == {"was_running": True, "stopped": True, "pid": pid, "method": "rpc", "forced": False}
    assert not lock_is_held(home.lock) and not home.pidfile.exists() and not home.socket.exists()
    assert wait_for(lambda: not pid_alive(pid), 5)
    stopped = controller_status(home)
    assert stopped["ok"] and stopped["data"]["running"] is False and stopped["data"]["state"] == "stopped"
    assert stop_controller(home)["was_running"] is False


def test_concurrent_autostart_spawns_one_controller(home: Paths) -> None:
    n = 4
    barrier = threading.Barrier(n)
    pids: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with toy_client(home) as c:
                barrier.wait()
                pids.append(c.call("toy.pid")["data"]["pid"])
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    assert len(pids) == n and len(set(pids)) == 1
    assert spawned_pids(home) == [pids[0]]  # the spawn lock prevented any second spawn


def test_sigterm_shuts_down_gracefully(home: Paths) -> None:
    with toy_client(home) as c:
        pid = c.call("toy.pid")["data"]["pid"]
    os.kill(pid, signal.SIGTERM)
    assert wait_for(lambda: not lock_is_held(home.lock), 10)
    assert not home.socket.exists() and not home.pidfile.exists()
    assert "controller stopping (signal SIGTERM)" in home.log.read_text()


def test_killed_controller_work_resumes_after_restart(
    home: Paths, start_ctl: Callable[..., Controller]
) -> None:
    with toy_client(home, "--block-forever") as c:
        oid = c.call("toy.enqueue", {"kind": "toy.block"})["data"]["outbox_id"]
        pid = c.call("toy.pid")["data"]["pid"]
    started = "SELECT * FROM events WHERE event_type='toy.block.started'"
    assert wait_for(lambda: db_rows(home, started), 10)
    os.kill(pid, signal.SIGKILL)  # crash in the middle of a handler
    assert wait_for(lambda: not lock_is_held(home.lock), 5)
    row = db_rows(home, "SELECT * FROM outbox WHERE outbox_id=?", (oid,))[0]
    assert row["state"] == "leased" and row["lease_owner"].startswith(f"controller:{pid}:")
    assert home.socket.exists() and read_pid(home.pidfile) == pid  # stale leftovers

    t0 = time.monotonic()
    start_ctl(home, app_factory=ToyApp)  # the lease (300 s) is released at startup
    done = "SELECT * FROM outbox WHERE outbox_id=? AND state='done'"
    assert wait_for(lambda: db_rows(home, done, (oid,)), 5)
    assert time.monotonic() - t0 < 5
    handled = db_rows(home, "SELECT payload_json FROM events WHERE event_type='toy.handled'")
    assert len(handled) == 1 and f'"pid":{os.getpid()}' in handled[0]["payload_json"]
    assert db_rows(home, done, (oid,))[0]["tries"] == 2


def test_client_reconnects_after_controller_restart(home: Paths) -> None:
    with toy_client(home) as c:
        first = c.call("toy.pid")["data"]["pid"]  # leaves a pooled connection to controller #1
        stop_controller(home)
        second = c.call("toy.pid")["data"]["pid"]  # stale connection → reconnect → autostart #2
    assert first != second and spawned_pids(home) == [first, second]


def test_startup_failure_is_reported_with_spawn_log(home: Paths) -> None:
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('toy startup failure: no module xyz\\n'); sys.exit(1)",
    ]
    c = HpcLabClient(home, spawn_cmd=cmd, startup_timeout=10)
    t0 = time.monotonic()
    with pytest.raises(ControllerUnavailable) as ei:
        c.call("toy.pid")
    assert time.monotonic() - t0 < 5  # fails fast instead of waiting out the timeout
    details = ei.value.details
    assert "exited with code 1" in ei.value.message
    assert any("toy startup failure" in line for line in details["spawn_log_tail"])
    assert details["log"] == str(home.log) and details["hint"]
    assert ei.value.retryable


def test_not_running_without_autostart(home: Paths) -> None:
    with pytest.raises(ControllerUnavailable) as ei:
        HpcLabClient(home, autostart=False).call("controller.ping")
    assert "not running" in ei.value.message
    assert ei.value.details["socket"] == str(home.socket) and "--foreground" in ei.value.details["hint"]
    assert not home.home.exists()  # a client without autostart never creates the home


def test_stop_controller_refuses_to_signal_its_own_process(
    home: Paths, start_ctl: Callable[..., Controller]
) -> None:
    start_ctl(home)
    with pytest.raises(HpcLabError, match="inside this process"):
        stop_controller(home)


def test_stale_pidfile_is_cleaned_without_signalling(home: Paths) -> None:
    home.ensure()
    home.pidfile.write_text(f"{os.getpid()}\n")  # would be fatal if trusted blindly
    assert stop_controller(home)["was_running"] is False
    assert not home.pidfile.exists()


def test_module_entrypoint_autostart_and_already_running(home: Paths) -> None:
    """The real `python -m hpclab.controller` (whatever service layer is installed)."""
    info = start_controller(home)
    assert info["protocol"] == proto.PROTOCOL_VERSION and info["pid"] != os.getpid()
    status = controller_status(home)
    assert status["ok"] and status["data"]["running"]
    # a second instance for the same home exits with the documented code
    proc = subprocess.run(
        [sys.executable, "-m", "hpclab.controller", "--home", str(home.home)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == proto.EXIT_ALREADY_RUNNING
    assert f"already running (pid {info['pid']})" in proc.stderr
    assert proc.stdout == ""  # never writes to stdout
    assert stop_controller(home, timeout=10)["stopped"]


def test_autostart_recovers_from_a_crashed_controller(home: Paths) -> None:
    with toy_client(home) as c:
        first = c.call("toy.pid")["data"]["pid"]
        os.kill(first, signal.SIGKILL)
        assert wait_for(lambda: not lock_is_held(home.lock), 5)
        assert home.socket.exists()  # stale socket file: connections are refused, not "missing"
        second = c.call("toy.pid")["data"]["pid"]  # dead pooled conn → refused → spawn → cleanup
    assert second != first and spawned_pids(home) == [first, second]
    assert read_pid(home.pidfile) == second
    assert "removing stale controller socket" in home.log.read_text()
