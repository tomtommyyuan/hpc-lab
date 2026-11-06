"""Toy App for controller tests — importable from tests and runnable as a controller process.

As a script (what the subprocess tests put in ``spawn_cmd``)::

    python hpclab_ctl_toy.py --home H [--block-forever] [--lease-s S] [--slow-start S]

Every start appends ``<pid>`` to ``<home>/spawned.txt`` *before* taking the instance lock, so
tests can count how many controller processes were spawned (including losers of a race).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hpclab.config import Paths, get_paths
from hpclab.controller import protocol as proto
from hpclab.controller.daemon import Controller, ControllerAlreadyRunning
from hpclab.controller.files import socket_path
from hpclab.domain.errors import NotFound
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.services.periodic import PeriodicTask

METHODS = [
    "toy.echo",
    "toy.sleep",
    "toy.boom",
    "toy.not_found",
    "toy.bad_return",
    "toy.unserializable",
    "toy.huge",
    "toy.enqueue",
    "toy.pid",
    "toy.deep",
]


class ToyApp:
    def __init__(
        self,
        ctx: ServiceContext,
        *,
        block_forever: bool = False,
        periodic: bool = True,
        tick_s: float = 0.05,
    ) -> None:
        self.ctx = ctx
        self.block_forever = block_forever
        self.periodic = periodic
        self.tick_s = tick_s
        self.calls: list[tuple[str, dict[str, Any], str]] = []
        self.lock = threading.Lock()
        self.ticks = 0
        self.flaky_runs = 0
        self.hourly_runs = 0
        self.closed = False

    # -- App protocol ----------------------------------------------------------
    def dispatch(self, method: str, params: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        with self.lock:
            self.calls.append((method, dict(params), request_id))
        if method == "toy.echo":
            return proto.ok_envelope(
                request_id, {"echo": params, "thread": threading.current_thread().name}, summary="echo"
            )
        if method == "toy.sleep":
            time.sleep(float(params.get("s", 0.1)))
            return proto.ok_envelope(request_id, {"slept": params.get("s")})
        if method == "toy.boom":
            raise RuntimeError("kaboom from toy.boom")
        if method == "toy.not_found":
            raise NotFound("no such toy", details={"id": params.get("id")})
        if method == "toy.bad_return":
            return ["not", "an", "envelope"]  # type: ignore[return-value]
        if method == "toy.unserializable":
            return {"ok": True, "request_id": request_id, "data": object()}
        if method == "toy.huge":
            return proto.ok_envelope(request_id, {"blob": "x" * int(params["n"])})
        if method == "toy.enqueue":
            with self.ctx.db.tx(source="api") as tx:
                oid = tx.enqueue(
                    params["kind"], target_id=params.get("target_id"), payload=params.get("payload") or {}
                )
            return proto.ok_envelope(request_id, {"outbox_id": oid})
        if method == "toy.pid":
            return proto.ok_envelope(request_id, {"pid": os.getpid()})
        if method == "toy.deep":
            deep: list[Any] = []
            cur = deep
            for _ in range(int(params.get("depth", 20_000))):
                nxt: list[Any] = []
                cur.append(nxt)
                cur = nxt
            return {"ok": True, "request_id": request_id, "data": deep}
        raise proto.MethodNotFound(method, known=METHODS)

    def outbox_handlers(self) -> dict[str, Handler]:
        return {"toy.work": self._work, "toy.block": self._block}

    def periodic_tasks(self) -> list[PeriodicTask]:
        if not self.periodic:
            return []
        return [
            PeriodicTask("toy.tick", self.tick_s, self._tick),
            PeriodicTask("toy.flaky", self.tick_s, self._flaky),
            PeriodicTask("toy.hourly", 3600.0, self._hourly, run_at_start=False),
        ]

    def close(self) -> None:
        self.closed = True

    # -- handlers / tasks ----------------------------------------------------------
    def _record(self, ctx: ServiceContext, item: OutboxItem, event_type: str) -> None:
        with ctx.db.tx(source="worker") as tx:
            tx.event(
                "toy",
                str(item.outbox_id),
                event_type,
                {"pid": os.getpid(), "kind": item.kind, "tries": item.tries, "payload": item.payload},
            )

    def _work(self, ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
        self._record(ctx, item, "toy.handled")
        return HandlerResult.done()

    def _block(self, ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
        if self.block_forever:
            self._record(ctx, item, "toy.block.started")
            time.sleep(3600)
        self._record(ctx, item, "toy.handled")
        return HandlerResult.done()

    def _tick(self, ctx: ServiceContext) -> None:
        self.ticks += 1

    def _flaky(self, ctx: ServiceContext) -> None:
        self.flaky_runs += 1
        raise ValueError("toy.flaky always fails")

    def _hourly(self, ctx: ServiceContext) -> None:
        self.hourly_runs += 1


# -- test helpers -------------------------------------------------------------

TOY_SCRIPT = Path(__file__).with_name("hpclab_ctl_toy.py")


def toy_spawn_cmd(paths: Paths, *extra: str) -> list[str]:
    return [sys.executable, str(TOY_SCRIPT), "--home", str(paths.home), *extra]


def wait_for(pred: Callable[[], Any], timeout: float = 5.0, interval: float = 0.02) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def spawned_pids(paths: Paths) -> list[int]:
    f = paths.home / "spawned.txt"
    if not f.exists():
        return []
    return [int(x) for x in f.read_text().split()]


class RawConn:
    """A bare socket speaking the line protocol, for protocol-level tests."""

    def __init__(self, paths: Paths, timeout: float = 5.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(str(socket_path(paths)))
        self.rfile = self.sock.makefile("rb")

    def send(self, msg: Any) -> None:
        data = msg if isinstance(msg, bytes) else (json.dumps(msg) + "\n").encode()
        self.sock.sendall(data)

    def recv(self) -> dict[str, Any] | None:
        line = self.rfile.readline()
        return json.loads(line) if line else None

    def call(self, method: str, params: dict[str, Any] | None = None, rid: Any = 1) -> dict[str, Any]:
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        resp = self.recv()
        assert resp is not None
        return resp

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.rfile.close()
        with contextlib.suppress(OSError):
            self.sock.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--block-forever", action="store_true")
    ap.add_argument("--lease-s", type=float, default=300.0)
    ap.add_argument("--slow-start", type=float, default=0.0, help="sleep this long while building the app")
    args = ap.parse_args(argv)
    paths = get_paths(args.home)
    paths.home.mkdir(parents=True, exist_ok=True)
    with open(paths.home / "spawned.txt", "a", encoding="ascii") as fh:
        fh.write(f"{os.getpid()}\n")

    def factory(ctx: ServiceContext) -> ToyApp:
        if args.slow_start:
            (paths.home / "building-app").touch()
            time.sleep(args.slow_start)
        return ToyApp(ctx, block_forever=args.block_forever)

    controller = Controller(paths, factory, lease_s=args.lease_s)
    try:
        controller.serve_forever()
    except ControllerAlreadyRunning as exc:
        sys.stderr.write(f"toy controller: {exc.message}\n")
        return proto.EXIT_ALREADY_RUNNING
    return proto.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
