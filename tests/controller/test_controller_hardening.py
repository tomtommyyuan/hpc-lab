"""Adversarial regression tests: each one reproduces a failure found in review (it failed
before the fix), plus coverage for paths the first version only claimed.

Protocol: replies json cannot encode (must be answered, never re-executed by a client retry);
pathological request JSON (deep nesting, huge ints); oversized requests (answered, stream
resynchronized) and oversized replies (ControllerUnavailable, not a raw exception).
Ledger writers: wall-clock jumps while a handler runs (laptop sleep); threads outliving the
shutdown timeout keep the instance lock; handlers raising ``SystemExit``; forked children
inheriting the lock. Lifecycle: hanging ``app.close()``; the pidfile gap in ``stop_controller``;
``controller.shutdown`` never autostarting and answered while stopping; a deleted socket file;
a second signal; SIGTERM during startup; autostart races between real processes; non-ASCII
homes vs the AF_UNIX byte limit; protocol re-check after reconnect. Status: hung periodic tasks
and long handlers surface in ``attention``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hpclab_ctl_toy import RawConn, ToyApp, spawned_pids, toy_spawn_cmd, wait_for

from hpclab.config import Paths
from hpclab.controller import protocol as proto
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.controller.daemon import Controller, ControllerAlreadyRunning, _attention
from hpclab.controller.files import InstanceLock, lock_is_held
from hpclab.domain.clock import SystemClock
from hpclab.domain.errors import ControllerUnavailable
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxItem
from hpclab.storage.db import Database

pytestmark = pytest.mark.timeout(40)

StartCtl = Callable[..., Controller]


def client(paths: Paths, **kw: Any) -> HpcLabClient:
    kw.setdefault("autostart", False)
    kw.setdefault("timeout", 10.0)
    return HpcLabClient(paths, **kw)


def outbox_row(paths: Paths, outbox_id: int) -> dict[str, Any]:
    db = Database(paths.db)
    try:
        row = db.one("SELECT * FROM outbox WHERE outbox_id=?", (outbox_id,))
        assert row is not None
        return row
    finally:
        db.close()


def calls_of(ctl: Controller, method: str) -> int:
    app = ctl.app
    assert isinstance(app, ToyApp)
    return sum(1 for m, _, _ in app.calls if m == method)


class JumpClock(SystemClock):
    """Real time plus an adjustable offset for ``now()`` (wall clock); ``monotonic()`` stays real,
    exactly like a laptop waking from sleep."""

    def __init__(self) -> None:
        self.offset = timedelta(0)

    def now(self) -> datetime:
        return super().now() + self.offset


class GateApp(ToyApp):
    """ToyApp with a ``toy.gate`` outbox kind that blocks until ``gate`` is set and counts entries."""

    gate: threading.Event
    entries: list[int]

    def outbox_handlers(self) -> dict[str, Any]:
        handlers = super().outbox_handlers()

        def gated(ctx: ServiceContext, item: OutboxItem) -> None:
            self.entries.append(item.outbox_id)
            self.gate.wait(20)

        handlers["toy.gate"] = gated
        return handlers


def gate_app_factory(gate: threading.Event, entries: list[int]) -> Callable[[ServiceContext], GateApp]:
    def make(ctx: ServiceContext) -> GateApp:
        app = GateApp(ctx, periodic=False)
        app.gate, app.entries = gate, entries
        return app

    return make


# ---------------------------------------------------------------------------
# protocol robustness
# ---------------------------------------------------------------------------


def test_unencodable_result_is_answered_and_never_reexecuted(start_ctl: StartCtl, home: Paths) -> None:
    """A result that json cannot encode (RecursionError, not TypeError/ValueError) used to kill
    the connection without a reply; the client then re-sent the request — executing it twice."""
    ctl = start_ctl(home)
    with client(home) as c:
        env = c.call("toy.deep", {"depth": 20_000})
        assert env["ok"] is False and env["error"]["code"] == "internal"
        assert "serialized" in env["error"]["message"]
        assert calls_of(ctl, "toy.deep") == 1  # answered once, not retried by the client
        assert c.call("toy.echo", {"after": True})["ok"]


def test_pathological_request_json_gets_a_parse_error(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home)
    raw = RawConn(home)
    try:
        raw.send(b"[" * 200_000 + b"]" * 200_000 + b"\n")  # RecursionError inside json
        resp = raw.recv()
        assert resp is not None and resp["error"]["code"] == proto.PARSE_ERROR
        big_int = b'{"jsonrpc":"2.0","id":7,"method":"toy.echo","params":{"n":' + b"1" * 5000 + b"}}\n"
        raw.send(big_int)  # ValueError (int digit limit), not JSONDecodeError
        resp = raw.recv()
        assert resp is not None and resp["error"]["code"] == proto.PARSE_ERROR
        assert raw.call("toy.echo", {"ok": 1}, rid=8)["result"]["ok"]  # connection still usable
    finally:
        raw.close()


@pytest.mark.parametrize(
    "line", [b"[" * 200_000 + b"]" * 200_000, b'{"n":' + b"9" * 5000 + b"}"], ids=["deep", "bigint"]
)
def test_decode_message_maps_every_json_failure_to_protocol_error(line: bytes) -> None:
    with pytest.raises(proto.ProtocolError) as ei:
        proto.decode_message(line)
    assert ei.value.code == proto.PARSE_ERROR


def test_reply_above_the_client_limit_is_controller_unavailable(start_ctl: StartCtl, home: Paths) -> None:
    """A reply the client cannot frame must surface as ControllerUnavailable (the documented
    failure type), not as a raw protocol exception with a leaked connection."""
    start_ctl(home, max_message_bytes=32 * 1024 * 1024)
    with client(home, timeout=20.0) as c:
        with pytest.raises(ControllerUnavailable, match="too large"):
            c.call("toy.huge", {"n": proto.MAX_MESSAGE_BYTES + 64 * 1024})
        assert c.call("toy.echo", {"x": 1})["ok"]


# ---------------------------------------------------------------------------
# outbox leases and ledger writers
# ---------------------------------------------------------------------------


def test_wall_clock_jump_does_not_rerun_an_item_still_running(start_ctl: StartCtl, home: Paths) -> None:
    """Laptop sleep: the wall clock jumps past lease_until while a handler is still running. A
    sibling worker must not treat that lease as expired and run the same item concurrently."""
    clock = JumpClock()
    gate, entries = threading.Event(), list[int]()
    ctl = start_ctl(
        home,
        app_factory=gate_app_factory(gate, entries),
        context_factory=lambda p: ServiceContext.create(p, clock=clock),
        lease_s=30.0,
    )
    try:
        with client(home) as c:
            oid = c.call("toy.enqueue", {"kind": "toy.gate"})["data"]["outbox_id"]
        assert wait_for(lambda: entries == [oid], 3)
        clock.offset = timedelta(hours=2)  # resume after two hours asleep
        ctl._kick_workers()
        time.sleep(0.6)  # > 10 worker ticks
        assert entries == [oid], "the item was started twice"
        row = outbox_row(home, oid)
        assert row["state"] == "leased" and row["tries"] == 1
    finally:
        gate.set()
    assert wait_for(lambda: outbox_row(home, oid)["state"] == "done", 3)
    assert entries == [oid]


def test_lingering_handler_keeps_the_instance_lock(start_ctl: StartCtl, home: Paths) -> None:
    """A handler still running after the shutdown timeout may still write the ledger, so the
    instance lock (and pidfile) must stay held until it finishes. Otherwise a successor starts,
    releases the 'orphaned' lease and runs the same item concurrently: two writers."""
    gate, entries = threading.Event(), list[int]()
    ctl = start_ctl(home, app_factory=gate_app_factory(gate, entries), shutdown_timeout_s=0.5)
    app = ctl.app
    assert isinstance(app, GateApp)
    try:
        with client(home) as c:
            oid = c.call("toy.enqueue", {"kind": "toy.gate"})["data"]["outbox_id"]
        assert wait_for(lambda: entries == [oid], 3)
        t0 = time.monotonic()
        ctl.stop(timeout=0.5)
        assert time.monotonic() - t0 < 3
        assert ctl.state == "stopped" and not home.socket.exists()
        assert lock_is_held(home.lock), "lock released while a handler still runs"
        assert not app.closed  # the app is closed only after its last thread finished
        with pytest.raises(ControllerAlreadyRunning):
            Controller(home, ToyApp, lock_wait_s=0.0).start()
    finally:
        gate.set()
    assert wait_for(lambda: not lock_is_held(home.lock), 5)
    assert app.closed and not home.pidfile.exists()
    row = outbox_row(home, oid)
    assert row["state"] == "done" and row["tries"] == 1
    start_ctl(home, app_factory=gate_app_factory(gate, entries))  # successor starts cleanly
    time.sleep(0.2)
    assert entries == [oid]


def test_handler_raising_system_exit_does_not_kill_workers(start_ctl: StartCtl, home: Paths) -> None:
    class ExitApp(ToyApp):
        def outbox_handlers(self) -> dict[str, Any]:
            def exiting(ctx: ServiceContext, item: OutboxItem) -> None:
                sys.exit(3)

            return {"toy.exit": exiting, **super().outbox_handlers()}

    ctl = start_ctl(home, app_factory=lambda ctx: ExitApp(ctx, periodic=False))
    with client(home) as c:
        oid = c.call("toy.enqueue", {"kind": "toy.exit"})["data"]["outbox_id"]

    def retried() -> bool:
        row = outbox_row(home, oid)
        return row["state"] == "pending" and "SystemExit" in (row["last_error"] or "")

    assert wait_for(retried, 3), outbox_row(home, oid)
    workers = ctl.status()["workers"]
    assert workers["alive"] == workers["expected"] == 2
    with client(home) as c:  # and the workers still process other work
        other = c.call("toy.enqueue", {"kind": "toy.work"})["data"]["outbox_id"]
    assert wait_for(lambda: outbox_row(home, other)["state"] == "done", 3)


# ---------------------------------------------------------------------------
# shutdown and lifecycle helpers
# ---------------------------------------------------------------------------


def test_hanging_app_close_does_not_block_shutdown(start_ctl: StartCtl, home: Paths) -> None:
    """app.close() doing network I/O (e.g. closing ssh masters on a dead VPN) must not keep the
    controller — and its lock — alive forever."""
    release = threading.Event()

    class HangCloseApp(ToyApp):
        def close(self) -> None:
            release.wait(30)

    ctl = start_ctl(home, app_factory=lambda ctx: HangCloseApp(ctx, periodic=False), shutdown_timeout_s=1.0)
    stopper = threading.Thread(target=ctl.stop, kwargs={"timeout": 1.0}, daemon=True)
    try:
        stopper.start()
        stopper.join(8)
        assert not stopper.is_alive(), "stop() blocked on app.close()"
        assert not lock_is_held(home.lock) and not home.pidfile.exists()
    finally:
        release.set()


def test_stop_controller_waits_through_the_pidfile_gap(home: Paths) -> None:
    """Between 'pidfile removed' and 'lock released' (and between 'lock taken' and 'pidfile
    written') the pid is unknown; stop_controller must wait instead of failing."""
    home.ensure()
    holder = InstanceLock(home.lock)
    assert holder.acquire()
    timer = threading.Timer(0.3, holder.release)
    timer.start()
    try:
        result = stop_controller(home, timeout=5.0)
    finally:
        timer.join()
        holder.release()
    assert result["stopped"] is True and result["was_running"] is True
    assert not lock_is_held(home.lock)


def test_shutdown_call_never_autostarts_a_controller(home: Paths) -> None:
    c = HpcLabClient(home, spawn_cmd=toy_spawn_cmd(home), timeout=5.0)
    with pytest.raises(ControllerUnavailable, match="not running"):
        c.call("controller.shutdown")
    time.sleep(0.2)
    assert spawned_pids(home) == []  # starting a controller only to stop it would be absurd
    c.close()


def test_shutdown_request_while_already_stopping_is_ok(home: Paths) -> None:
    """A second stop request (another CLI, a retry) racing a shutdown gets its answer instead of a
    "shutting down" error that would make the client retry against a successor."""
    ctl = Controller(home, ToyApp, tick_interval_s=0.05, lock_wait_s=0.0).start_in_thread()
    try:
        with client(home) as c:
            assert c.call("controller.ping")["ok"]  # connected before the shutdown began
            ctl._stop.set()  # the window between "stop requested" and "socket closed"
            env = c.call("controller.shutdown")
            assert env["ok"] and env["data"]["stopping"] is True
            with pytest.raises(ControllerUnavailable):  # other methods are refused, not executed
                c.call("toy.echo", timeout=2.0)
        assert calls_of(ctl, "toy.echo") == 0
    finally:
        ctl.stop()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_does_not_keep_the_instance_lock(home: Paths) -> None:
    """fork() without exec (multiprocessing 'fork', a library daemonizing) duplicates the lock
    descriptor. When the controller then dies, the child's copy must not keep the lock held —
    otherwise no controller can start and stop_controller would signal a dead (reused) pid."""
    home.ensure()
    lock = InstanceLock(home.lock)
    assert lock.acquire()
    r, w = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # "process is multi-threaded"
        pid = os.fork()
    if pid == 0:  # child: announce, then linger holding whatever it inherited
        os.close(r)
        os.write(w, b"x")
        time.sleep(10)
        os._exit(0)
    os.close(w)
    try:
        assert os.read(r, 1) == b"x"
        # the owner dies: the kernel closes its descriptor without an explicit LOCK_UN
        fd, lock._fd = lock._fd, None
        assert fd is not None
        os.close(fd)
        assert not lock_is_held(home.lock), "a forked child kept the controller lock alive"
    finally:
        os.close(r)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


def test_attention_flags_hung_background_work() -> None:
    status: dict[str, Any] = {
        "app": {"fallback": False},
        "workers": {
            "alive": 1,
            "expected": 1,
            "threads": [
                {
                    "name": "worker-0",
                    "current": {"outbox_id": 7, "kind": "submit.stage", "running_for_s": 7300.0},
                }
            ],
        },
        "scheduler": {
            "alive": True,
            "tasks": {
                "observe.poll_all": {
                    "interval_s": 30.0,
                    "running": True,
                    "running_for_s": 4000.0,
                    "consecutive_failures": 0,
                },
                "notify.flush": {
                    "interval_s": 60.0,
                    "running": True,
                    "running_for_s": 5.0,
                    "consecutive_failures": 0,
                },
            },
        },
        "clusters": [],
        "outbox": {"by_state": {"dead": 0}},
    }
    notes = _attention(status)
    assert any("observe.poll_all" in n and "4000" in n for n in notes), notes
    assert any("submit.stage" in n and "7300" in n for n in notes), notes
    assert not any("notify.flush" in n for n in notes)


def test_client_setup_failures_are_controller_unavailable(
    home: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """call() promises an Envelope or ControllerUnavailable; an OSError while preparing the home
    (e.g. another user owns the /tmp socket directory) must not escape raw into the MCP server."""

    def denied(self: Paths) -> Paths:
        raise PermissionError(13, "Operation not permitted", "/tmp/hpclab-x")

    monkeypatch.setattr(Paths, "ensure", denied)
    c = HpcLabClient(home, spawn_cmd=toy_spawn_cmd(home), timeout=5.0)
    with pytest.raises(ControllerUnavailable) as ei:
        c.call("toy.pid")
    assert "Operation not permitted" in ei.value.message and ei.value.details["hint"]
    assert spawned_pids(home) == []


def test_controller_stops_when_its_socket_file_disappears(start_ctl: StartCtl, home: Paths) -> None:
    """/tmp cleaners (macOS periodic, systemd-tmpfiles) or a user may delete the socket file. The
    controller would then hold the lock while no client can reach it — and autostart refuses to
    spawn while the lock is held. It must notice and stop so the next client starts a fresh one."""
    ctl = start_ctl(home)
    assert "controller.socket_watch" in ctl.scheduler.stats()
    os.unlink(home.socket)
    ctl.scheduler.force_due(["controller.socket_watch"])
    ctl._sched_wake.set()
    assert ctl.wait(5), "controller kept running without a reachable socket"
    assert not lock_is_held(home.lock)
    assert "socket" in home.log.read_text().split("controller stopping")[-1]


def test_request_above_the_server_limit_is_a_limit_exceeded_envelope(
    start_ctl: StartCtl, home: Paths
) -> None:
    """The controller answers an oversized request with an id-less JSON-RPC error (it could not
    read the id) and closes the connection. With one request per connection at a time, that
    error belongs to the pending call: surface it as an Envelope, and do not pool the dead socket."""
    ctl = start_ctl(home, max_message_bytes=64 * 1024)
    with client(home) as c:
        env = c.call("toy.echo", {"blob": "x" * (100 * 1024)})
        assert env["ok"] is False and env["error"]["code"] == "limit_exceeded"
        assert env["error"]["details"]["jsonrpc_code"] == proto.MESSAGE_TOO_LARGE
        assert calls_of(ctl, "toy.echo") == 0
        assert c.call("toy.echo", {"small": 1})["ok"]


def test_ctrl_c_during_shutdown_exits_immediately_but_sigterm_stays_graceful(home: Paths) -> None:
    """Ctrl-C while a shutdown waits for a hung handler must not leave the operator waiting out
    the budget: a SIGINT during shutdown exits at once (the kernel frees the lock; the next
    controller releases the leases). A repeated SIGTERM — overlapping ``hpclab controller stop``
    runs, launchd/systemd — must stay graceful."""
    proc = subprocess.Popen(
        [*toy_spawn_cmd(home, "--block-forever")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert wait_for(lambda: home.socket.exists() and lock_is_held(home.lock), 10)
        with client(home) as c:
            c.call("toy.enqueue", {"kind": "toy.block"})
        db = Database(home.db)
        try:
            assert wait_for(lambda: db.all("SELECT 1 FROM events WHERE event_type='toy.block.started'"), 5)
        finally:
            db.close()
        proc.send_signal(signal.SIGTERM)
        assert wait_for(lambda: not home.socket.exists(), 5)  # shutdown began; the handler hangs
        proc.send_signal(signal.SIGTERM)  # a second SIGTERM: still graceful
        time.sleep(1.0)
        assert proc.poll() is None
        t0 = time.monotonic()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(5) == 128 + signal.SIGINT
        assert time.monotonic() - t0 < 3
        assert wait_for(lambda: not lock_is_held(home.lock), 2)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


_RACER = """
import json, sys, time
sys.path.insert(0, sys.argv[3])
from hpclab.config import get_paths
from hpclab.controller.client import HpcLabClient
from hpclab_ctl_toy import toy_spawn_cmd
paths = get_paths(sys.argv[1])
go = float(sys.argv[2])
while time.time() < go:
    time.sleep(0.001)
with HpcLabClient(paths, spawn_cmd=toy_spawn_cmd(paths), timeout=10.0, startup_timeout=15.0) as c:
    print(json.dumps(c.call("toy.pid")["data"]))
"""


def test_concurrent_autostart_from_separate_processes_spawns_one_controller(home: Paths) -> None:
    """The real race: several MCP servers / CLI invocations (separate processes, separate flock
    open-file descriptions) autostarting at the same instant."""
    n = 5
    go = time.time() + 1.5  # all racers start their first call at the same wall-clock instant
    here = str(Path(__file__).parent)
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(home.home), str(go), here],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(n)
    ]
    outs = []
    for p in racers:
        out, err = p.communicate(timeout=30)
        assert p.returncode == 0, err
        outs.append(json.loads(out.strip().splitlines()[-1]))
    pids = {o["pid"] for o in outs}
    assert len(pids) == 1
    assert spawned_pids(home) == list(pids)
    stop_controller(home, timeout=10)


def test_non_ascii_home_gets_a_bindable_socket_path(
    start_ctl: StartCtl, make_home: Callable[..., Paths]
) -> None:
    """AF_UNIX limits are in *bytes* (104 macOS, 108 Linux); a home with CJK directory names can
    be short in characters yet too long in bytes. The controller must still start and be reachable."""
    from hpclab.controller.files import socket_path

    home = make_home("实验" * 12)
    sock = socket_path(home)
    assert len(os.fsencode(str(sock))) <= 100
    ctl = start_ctl(home)
    with client(home) as c:
        assert c.call("controller.ping")["data"]["socket"] == str(sock)
    ctl.stop()
    assert not sock.exists() and not sock.parent.exists()


def test_sigterm_during_startup_ends_in_a_clean_shutdown(home: Paths) -> None:
    proc = subprocess.Popen(
        [*toy_spawn_cmd(home, "--slow-start", "1.5")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert wait_for(lambda: (home.home / "building-app").exists(), 10)  # lock held, app building
        assert lock_is_held(home.lock)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(10) == 0
        assert not lock_is_held(home.lock) and not home.pidfile.exists() and not home.socket.exists()
        assert "controller stopping (signal SIGTERM)" in home.log.read_text()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_s": float("inf")},
        {"interval_s": float("nan")},
        {"interval_s": 1.0, "jitter_s": float("inf")},
    ],
    ids=["inf-interval", "nan-interval", "inf-jitter"],
)
def test_periodic_task_rejects_non_finite_timing(kwargs: dict[str, float]) -> None:
    """An infinite interval or jitter makes next_due infinite: the task would silently never run
    again, which is exactly the kind of quiet failure the scheduler exists to prevent."""
    from hpclab.services.periodic import PeriodicTask

    with pytest.raises(ValueError):
        PeriodicTask("x", kwargs["interval_s"], lambda ctx: None, jitter_s=kwargs.get("jitter_s", 0.0))


def test_client_rechecks_the_protocol_after_reconnecting(
    start_ctl: StartCtl, home: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the controller was replaced (restart after an upgrade), a long-lived client (an MCP
    server) must re-verify the protocol before re-sending anything to the new one."""
    first = start_ctl(home)
    c = client(home)
    try:
        assert c.call("toy.echo", {"n": 1})["ok"]  # verified protocol + pooled connection
        first.stop()
        second = start_ctl(home)
        real = second.ping_data
        monkeypatch.setattr(second, "ping_data", lambda: {**real(), "protocol": proto.PROTOCOL_VERSION + 1})
        with pytest.raises(ControllerUnavailable, match="speaks protocol"):
            c.call("toy.echo", {"n": 2})
        assert calls_of(second, "toy.echo") == 0
    finally:
        c.close()


def test_controller_rejects_a_spinning_tick(home: Paths) -> None:
    with pytest.raises(ValueError, match="tick_interval_s"):
        Controller(home, ToyApp, tick_interval_s=0.0)
