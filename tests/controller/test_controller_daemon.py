"""Controller daemon behaviour with an in-thread controller and a toy App."""

from __future__ import annotations

import os
import socket
import stat
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hpclab_ctl_toy import RawConn, ToyApp, wait_for

from hpclab.config import Paths
from hpclab.controller import protocol as proto
from hpclab.controller.client import HpcLabClient
from hpclab.controller.daemon import Controller, ControllerAlreadyRunning, MinimalApp, default_app_factory
from hpclab.controller.files import lock_is_held, read_pid
from hpclab.domain.errors import ControllerUnavailable
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxRunner
from hpclab.storage.db import Database

pytestmark = pytest.mark.timeout(30)

StartCtl = Callable[..., Controller]


def client(paths: Paths, **kw: Any) -> HpcLabClient:
    kw.setdefault("autostart", False)
    kw.setdefault("timeout", 10.0)
    return HpcLabClient(paths, **kw)


def toy(ctl: Controller) -> ToyApp:
    app = ctl.app
    assert isinstance(app, ToyApp)
    return app


def events(paths: Paths, event_type: str) -> list[dict[str, Any]]:
    db = Database(paths.db)
    try:
        return db.all("SELECT * FROM events WHERE event_type=? ORDER BY seq", (event_type,))
    finally:
        db.close()


def outbox_row(paths: Paths, outbox_id: int) -> dict[str, Any]:
    db = Database(paths.db)
    try:
        row = db.one("SELECT * FROM outbox WHERE outbox_id=?", (outbox_id,))
        assert row is not None
        return row
    finally:
        db.close()


# ---------------------------------------------------------------------------
# request / response
# ---------------------------------------------------------------------------


def test_round_trip_and_builtins(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home)
    with client(home) as c:
        env = c.call("toy.echo", {"x": [1, "é"]})
        assert env["ok"] and env["data"]["echo"] == {"x": [1, "é"]}
        assert env["request_id"].startswith("req_")
        assert toy(ctl).calls[-1] == ("toy.echo", {"x": [1, "é"]}, env["request_id"])

        ping = c.call("controller.ping")["data"]
        assert ping["pid"] == os.getpid() and ping["protocol"] == proto.PROTOCOL_VERSION
        assert ping["app"] == "ToyApp" and ping["app_fallback"] is False

        status = c.call("controller.status")
        data = status["data"]
        assert status["ok"] and "outbox pending=0" in status["summary"]
        assert data["workers"]["alive"] == data["workers"]["expected"] == 2
        assert data["scheduler"]["alive"] and "toy.tick" in data["scheduler"]["tasks"]
        assert data["outbox"]["by_state"] == {"pending": 0, "leased": 0, "done": 0, "dead": 0}
        assert data["app"]["outbox_kinds"] == ["toy.block", "toy.work"]
        assert data["home"] == str(home.home) and data["log"] == str(home.log)
        assert data["clusters"] == []


def test_status_lists_what_needs_attention(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home, app_factory=lambda ctx: ToyApp(ctx, periodic=False))
    assert ctl.status()["attention"] == []
    with ctl.ctx.db.tx() as tx:
        oid = tx.enqueue("toy.unknown_kind")
        tx.execute("UPDATE outbox SET state='dead', last_error='boom' WHERE outbox_id=?", (oid,))
        tx.execute(
            "INSERT INTO poll_state (cluster_id, source, last_polled_at, last_ok_at, consecutive_failures, "
            "last_error) VALUES ('hai', 'squeue', ?, NULL, 3, 'auth_required: run hpclab auth hai')",
            (tx.now,),
        )
    with client(home) as c:
        env = c.call("controller.status")
    attention = env["data"]["attention"]
    assert any("cluster hai: squeue failing (3x, last ok never): auth_required" in a for a in attention)
    assert any("1 background step(s) gave up" in a and "toy.unknown_kind" in a for a in attention)
    assert "2 item(s) need attention" in env["summary"]
    assert env["data"]["clusters"] == [
        {
            "cluster_id": "hai",
            "poll": {
                "squeue": {
                    "last_polled_at": env["data"]["clusters"][0]["poll"]["squeue"]["last_polled_at"],
                    "last_ok_at": None,
                    "consecutive_failures": 3,
                    "backoff_until": None,
                    "last_error": "auth_required: run hpclab auth hai",
                }
            },
        }
    ]


def test_many_requests_on_one_connection_and_protocol_errors(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home)
    raw = RawConn(home)
    try:
        for i in range(5):
            resp = raw.call("toy.echo", {"i": i}, rid=f"r{i}")
            assert resp["id"] == f"r{i}" and resp["result"]["data"]["echo"] == {"i": i}
        raw.send(b"{not json\n")
        bad = raw.recv()
        assert bad is not None and bad["id"] is None and bad["error"]["code"] == proto.PARSE_ERROR
        raw.send({"jsonrpc": "2.0", "id": 9, "method": "toy.echo", "params": [1]})
        bad = raw.recv()
        assert bad is not None and bad["id"] == 9 and bad["error"]["code"] == proto.INVALID_PARAMS
        raw.send([{"jsonrpc": "2.0", "id": 1, "method": "toy.echo"}])
        bad = raw.recv()
        assert bad is not None and bad["error"]["code"] == proto.INVALID_REQUEST
        # a notification gets no reply: the next reply belongs to the next request
        raw.send({"jsonrpc": "2.0", "method": "toy.echo", "params": {"note": True}})
        assert raw.call("controller.ping", rid=10)["id"] == 10
        # the connection survived every protocol error
        assert raw.call("toy.echo", {"after": 1}, rid=11)["result"]["ok"]
    finally:
        raw.close()


def test_unknown_methods_are_envelope_errors(start_ctl: StartCtl, home: Paths) -> None:
    """Documented contract: unknown methods (app or controller.*) answer with an Envelope whose
    error.code is invalid_argument — never a JSON-RPC method-not-found error."""
    start_ctl(home)
    raw = RawConn(home)
    try:
        for method in ("nope.method", "controller.nope"):
            resp = raw.call(method, rid=1)
            assert "error" not in resp
            env = resp["result"]
            assert env["ok"] is False and env["error"]["code"] == "invalid_argument"
            assert method in env["error"]["message"] and env["error"]["details"]["known"]
    finally:
        raw.close()
    with client(home) as c:
        assert c.call("nope.method")["error"]["code"] == "invalid_argument"


def test_exceptions_become_envelopes_and_server_keeps_serving(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home, max_message_bytes=64 * 1024)
    with client(home) as c:
        boom = c.call("toy.boom")
        assert boom["ok"] is False and boom["error"]["code"] == "internal"
        assert boom["error"]["details"]["request_id"] == boom["request_id"]
        assert str(home.log) in boom["error"]["message"]

        nf = c.call("toy.not_found", {"id": "x1"})
        assert nf["error"]["code"] == "not_found" and nf["error"]["details"] == {"id": "x1"}

        assert c.call("toy.bad_return")["error"]["code"] == "internal"
        unser = c.call("toy.unserializable")
        assert unser["error"]["code"] == "internal" and "serialized" in unser["error"]["message"]

        huge = c.call("toy.huge", {"n": 100_000})
        assert huge["error"]["code"] == "limit_exceeded" and huge["error"]["details"]["size"] > 64 * 1024

        assert c.call("toy.echo", {"still": "alive"})["data"]["echo"] == {"still": "alive"}
        stats = c.call("controller.status")["data"]["requests"]
        assert stats["internal_errors"] >= 3 and stats["errors"] >= 5
    log_text = home.log.read_text()
    assert "kaboom from toy.boom" in log_text and "Traceback" in log_text


def test_oversized_request_gets_error_and_connection_resyncs(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home, max_message_bytes=1024)
    raw = RawConn(home)
    try:
        raw.send(b'{"jsonrpc":"2.0","id":1,"method":"toy.echo","params":{"x":"' + b"a" * 4096 + b'"}}\n')
        resp = raw.recv()
        assert resp is not None and resp["error"]["code"] == proto.MESSAGE_TOO_LARGE
        assert resp["error"]["data"]["max_bytes"] == 1024
        # the rest of the oversized line was discarded: the same connection keeps working
        assert raw.call("toy.echo", {"after": 1}, rid=2)["result"]["data"]["echo"] == {"after": 1}
        assert toy(ctl).calls[-1][0] == "toy.echo" and len(toy(ctl).calls) == 1
    finally:
        raw.close()
    with client(home) as c:
        assert c.call("controller.ping")["ok"]


def test_concurrent_clients_get_their_own_results(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home)
    shared = client(home)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        try:
            own = client(home) if n % 2 else shared
            barrier.wait()
            for i in range(15):
                payload = {"thread": n, "i": i, "blob": "z" * (n * 100 + i)}
                env = own.call("toy.echo", payload)
                assert env["ok"] and env["data"]["echo"] == payload
            if own is not shared:
                own.close()
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    shared.close()
    assert not errors, errors


def test_deadline_exceeded_is_a_retryable_envelope(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home)
    with client(home) as c:
        env = c.call("toy.sleep", {"s": 1.5}, timeout=1.0)  # client sends deadline_s=0.5
        assert env["ok"] is False and env["error"]["code"] == "unavailable"
        assert env["error"]["retryable"] is True and env["error"]["details"]["started"] is True
        assert c.call("controller.ping", timeout=1.0)["ok"]  # server stayed responsive


def test_busy_controller_rejects_without_starting(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home, max_concurrent_requests=1)
    slow = threading.Thread(target=lambda: client(home).call("toy.sleep", {"s": 1.0}, timeout=5.0))
    slow.start()
    assert wait_for(lambda: any(m == "toy.sleep" for m, _, _ in toy(ctl).calls), 3)
    with client(home) as c:
        env = c.call("toy.echo", {"x": 1}, timeout=0.6)  # deadline 0.1s < slot wait
        assert env["error"]["code"] == "unavailable" and env["error"]["details"]["started"] is False
        assert not any(m == "toy.echo" for m, _, _ in toy(ctl).calls)
        assert c.call("controller.ping")["ok"]  # built-ins bypass the request slots
    slow.join(5)


# ---------------------------------------------------------------------------
# single instance, files, shutdown
# ---------------------------------------------------------------------------


def test_single_instance_per_home(start_ctl: StartCtl, make_home: Callable[..., Paths]) -> None:
    home = make_home("a")
    start_ctl(home)
    second = Controller(home, ToyApp, lock_wait_s=0.0)
    with pytest.raises(ControllerAlreadyRunning, match=rf"already running \(pid {os.getpid()}\)"):
        second.start()
    assert second.state == "stopped"
    # the failed start did not disturb the running instance
    assert read_pid(home.pidfile) == os.getpid() and home.socket.exists()
    with client(home) as c:
        assert c.call("controller.ping")["ok"]
    other = make_home("b")
    start_ctl(other)
    with client(other) as c:
        assert c.call("controller.ping")["data"]["home"] == str(other.home)


def test_socket_and_directory_permissions(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home)
    assert stat.S_IMODE(os.stat(home.socket).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(home.socket.parent).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(home.home).st_mode) == 0o700
    assert read_pid(home.pidfile) == os.getpid()
    assert lock_is_held(home.lock)


def test_stale_socket_and_pidfile_are_cleaned_up(start_ctl: StartCtl, home: Paths) -> None:
    home.ensure()
    home.socket.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # /tmp fallback dir is created lazily
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(home.socket))  # a socket file nobody listens on (crashed controller)
    dead.close()
    home.pidfile.write_text("999999\n")
    with pytest.raises(ControllerUnavailable):
        client(home).call("controller.ping")
    start_ctl(home)
    assert read_pid(home.pidfile) == os.getpid()
    with client(home) as c:
        assert c.call("controller.ping")["ok"]


def test_graceful_shutdown_releases_everything(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home)
    c = client(home)
    assert c.call("toy.echo", {"a": 1})["ok"]  # leaves a pooled connection open
    ctl.stop()
    assert ctl.state == "stopped" and toy(ctl).closed
    assert not home.socket.exists() and not home.pidfile.exists()
    assert not lock_is_held(home.lock)
    with pytest.raises(ControllerUnavailable) as ei:
        c.call("toy.echo", {"a": 2})
    assert ei.value.details["log"] == str(home.log) and ei.value.details["hint"]
    # a successor can start in the same home and the old client reconnects to it
    successor = start_ctl(home)
    assert c.call("toy.echo", {"a": 3})["ok"]
    assert successor.status()["requests"]["total"] >= 1
    c.close()


def test_shutdown_request_over_the_socket(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home)
    with client(home) as c:
        env = c.call("controller.shutdown")
        assert env["ok"] and env["data"] == {"stopping": True, "pid": os.getpid()}
    assert ctl.wait(10)
    assert not home.socket.exists() and not lock_is_held(home.lock)


def test_requests_during_shutdown_are_refused_not_executed(home: Paths) -> None:
    ctl = Controller(home, ToyApp, tick_interval_s=0.05, lock_wait_s=0.0).start_in_thread()
    try:
        raw = RawConn(home)
        assert raw.call("controller.ping")["result"]["ok"]
        ctl._stop.set()  # simulate the window between "stop requested" and "socket closed"
        resp = raw.call("toy.echo", {"x": 1}, rid=5)
        assert resp["id"] == 5 and resp["error"]["code"] == proto.SHUTTING_DOWN
        assert not any(m == "toy.echo" for m, _, _ in toy(ctl).calls)
        raw.close()
    finally:
        ctl.stop()


def test_long_home_uses_short_socket_path(start_ctl: StartCtl, make_home: Callable[..., Paths]) -> None:
    home = make_home("very-long-directory-name-for-socket-fallback-" * 3)
    assert len(str(home.home / "run" / "controller.sock")) > 100
    assert home.socket.parent != home.home / "run" and len(str(home.socket)) <= 100
    ctl = start_ctl(home)
    with client(home) as c:
        assert c.call("controller.ping")["data"]["socket"] == str(home.socket)
    assert stat.S_IMODE(os.stat(home.socket.parent).st_mode) == 0o700
    ctl.stop()
    assert not home.socket.exists() and not home.socket.parent.exists()  # fallback dir removed too
    start_ctl(home)  # and recreated by the next controller
    with client(home) as c:
        assert c.call("controller.ping")["ok"]


# ---------------------------------------------------------------------------
# outbox workers and periodic tasks
# ---------------------------------------------------------------------------


def test_outbox_items_are_processed_promptly(start_ctl: StartCtl, home: Paths) -> None:
    start_ctl(home, tick_interval_s=5.0)  # a long tick: only the post-request kick can explain speed
    with client(home) as c:
        oid = c.call("toy.enqueue", {"kind": "toy.work", "payload": {"n": 1}})["data"]["outbox_id"]
        assert wait_for(lambda: outbox_row(home, oid)["state"] == "done", 3)
        status = c.call("controller.status")["data"]
    assert events(home, "toy.handled")[0]["entity_id"] == str(oid)
    assert status["outbox"]["by_state"]["done"] == 1
    assert sum(w["processed"] for w in status["workers"]["threads"]) == 1


def test_orphaned_leases_are_released_at_startup(home: Paths, start_ctl: StartCtl) -> None:
    home.ensure()
    db = Database(home.db)
    with db.tx() as tx:
        oid = tx.enqueue("toy.work", payload={"orphan": True})
    assert oid is not None
    ctx = ServiceContext.create(home, db=db)
    runner = OutboxRunner(ctx, handlers={"toy.work": lambda c, i: None}, worker_id="controller:1:dead:w0")
    assert [i.outbox_id for i in runner._lease(1)] == [oid]  # leased for 300s by a "crashed" controller
    db.close()
    start_ctl(home)
    assert wait_for(lambda: outbox_row(home, oid)["state"] == "done", 3)
    assert outbox_row(home, oid)["tries"] == 2


def test_foreign_leases_are_retried_after_lease_expiry(home: Paths, start_ctl: StartCtl) -> None:
    home.ensure()
    db = Database(home.db)
    with db.tx() as tx:
        oid = tx.enqueue("toy.work", payload={"foreign": True})
    assert oid is not None
    ctx = ServiceContext.create(home, db=db)
    runner = OutboxRunner(ctx, handlers={"toy.work": lambda c, i: None}, worker_id="cli-drain", lease_s=0.6)
    assert [i.outbox_id for i in runner._lease(1)] == [oid]
    db.close()
    start_ctl(home)
    # not someone we may pre-empt: it waits for the lease to expire, then runs
    assert outbox_row(home, oid)["state"] == "leased"
    assert wait_for(lambda: outbox_row(home, oid)["state"] == "done", 5)


def test_periodic_errors_do_not_stop_the_scheduler(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home)
    app = toy(ctl)
    assert wait_for(lambda: app.flaky_runs >= 3 and app.ticks >= 5, 5)
    ticks = app.ticks
    assert wait_for(lambda: app.ticks > ticks + 2, 3)  # still ticking after repeated failures
    with client(home) as c:
        sched = c.call("controller.status")["data"]["scheduler"]
    assert sched["alive"]
    flaky = sched["tasks"]["toy.flaky"]
    assert flaky["failures"] >= 3 and "toy.flaky always fails" in flaky["last_error"]
    assert sched["tasks"]["toy.tick"]["failures"] == 0
    assert "toy.flaky always fails" in home.log.read_text()
    assert any(a.startswith("periodic task toy.flaky failed") for a in ctl.status()["attention"])


def test_wake_runs_named_tasks_now(start_ctl: StartCtl, home: Paths) -> None:
    ctl = start_ctl(home, tick_interval_s=0.5)
    app = toy(ctl)
    assert app.hourly_runs == 0
    with client(home) as c:
        env = c.call("controller.wake", {"tasks": ["toy.hourly"]})
        assert env["ok"] and env["data"]["forced"] == ["toy.hourly"]
        assert wait_for(lambda: app.hourly_runs == 1, 2)
        bad = c.call("controller.wake", {"tasks": ["nope"]})
        assert bad["error"]["code"] == "invalid_argument" and "toy.hourly" in bad["error"]["details"]["known"]


def test_leases_of_running_items_are_renewed(start_ctl: StartCtl, home: Paths) -> None:
    gate = threading.Event()

    class SlowApp(ToyApp):
        def outbox_handlers(self) -> dict[str, Any]:
            def slow(ctx: ServiceContext, item: Any) -> None:
                gate.wait(10)

            return {"toy.slow": slow}

    ctl = start_ctl(home, app_factory=SlowApp, lease_s=0.6)
    with client(home) as c:
        oid = c.call("toy.enqueue", {"kind": "toy.slow"})["data"]["outbox_id"]
        assert wait_for(lambda: outbox_row(home, oid)["state"] == "leased", 3)
        first_until = outbox_row(home, oid)["lease_until"]
        assert wait_for(lambda: outbox_row(home, oid)["lease_until"] > first_until, 3)
        current = [w["current"] for w in c.call("controller.status")["data"]["workers"]["threads"]]
    assert any(cur and cur["outbox_id"] == oid and cur["kind"] == "toy.slow" for cur in current)
    assert outbox_row(home, oid)["tries"] == 1  # never picked up twice while its handler runs
    gate.set()
    assert wait_for(lambda: outbox_row(home, oid)["state"] == "done", 3)
    ctl.stop()


# ---------------------------------------------------------------------------
# app factory fallback
# ---------------------------------------------------------------------------


def test_default_app_factory_falls_back_to_minimal_app(
    monkeypatch: pytest.MonkeyPatch, start_ctl: StartCtl, home: Paths
) -> None:
    def broken(ctx: ServiceContext) -> Any:
        raise RuntimeError("services exploded")

    monkeypatch.setitem(sys.modules, "hpclab.services.app", types.SimpleNamespace(create_app=broken))
    ctl = start_ctl(home, app_factory=default_app_factory)
    assert isinstance(ctl.app, MinimalApp)
    with client(home) as c:
        env = c.call("runs.list")
        assert env["error"]["code"] == "invalid_argument"
        assert "services exploded" in env["error"]["message"]
        status = c.call("controller.status")
        assert status["data"]["app"]["fallback"] is True and "NOT loaded" in status["summary"]
        assert "services exploded" in status["data"]["attention"][0]


def test_missing_service_module_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "hpclab.services.app", None)  # import raises ModuleNotFoundError
    app = default_app_factory(None)  # type: ignore[arg-type]
    assert isinstance(app, MinimalApp) and app.load_error


def test_stop_requested_during_startup_still_shuts_down(home: Paths) -> None:
    holder: dict[str, Controller] = {}

    def factory(ctx: ServiceContext) -> ToyApp:
        holder["ctl"].request_stop("stop while starting")  # e.g. controller.shutdown racing startup
        return ToyApp(ctx)

    ctl = Controller(home, factory, lock_wait_s=0.0, tick_interval_s=0.05)
    holder["ctl"] = ctl
    ctl.start_in_thread()
    assert ctl.wait(10) and ctl.state == "stopped"
    assert not lock_is_held(home.lock) and not home.socket.exists()


def test_client_refuses_a_controller_speaking_another_protocol(
    start_ctl: StartCtl, home: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctl = start_ctl(home)
    real = ctl.ping_data
    monkeypatch.setattr(ctl, "ping_data", lambda: {**real(), "protocol": proto.PROTOCOL_VERSION + 1})
    with pytest.raises(ControllerUnavailable, match="speaks protocol") as ei:
        client(home).call("toy.echo")
    assert "restart" in ei.value.details["hint"]
