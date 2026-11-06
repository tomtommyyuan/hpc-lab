"""Client for the controller socket, used by the CLI, the MCP server and Python callers.

``HpcLabClient.call(method, params)`` returns the controller's Envelope dict — for successes,
domain errors and unknown methods alike (protocol-level JSON-RPC errors are converted to
Envelope errors too) — and raises ``ControllerUnavailable`` only when no answer could be
obtained. Its ``details`` always name the home, socket and log files and carry a ``hint``.

Behaviour
---------
* Thread-safe: every call checks out its own connection from a small idle pool, so concurrent
  calls from one client (an MCP server serving parallel tool calls) do not serialize.
* Autostart: if the socket is missing or refuses connections, the client takes a spawn lock
  (``run/spawn.lock``, ``flock``) so concurrent clients never start two controllers, re-checks,
  and spawns ``python -m hpclab.controller --home <home>`` detached (new session, stdin from
  /dev/null, output to ``logs/controller-spawn.log``), then waits up to ``startup_timeout`` for
  ``controller.ping``. A controller that exits during startup is reported with the tail of the
  spawn log (import errors, bad config) instead of a bare timeout.
* Reconnect once: a pooled connection that turns out dead (controller restarted), a connection
  lost mid-call, or a "shutting down" reply is retried once on a fresh connection (autostarting
  if allowed, and re-verifying the protocol of whatever controller now answers). Re-sending is
  safe because every mutation carries an idempotency key, a "shutting down" reply guarantees
  the request did not run, and the controller never hangs up on a request it may have run
  without answering (only a crash does that). A *timeout* is never retried: the request may
  still be running.
* A JSON-RPC error with ``id: null`` (the controller could not read the id, e.g. the request
  was too large) answers the request in flight and is returned as an Envelope; if the
  controller hung up while we were still sending, its already-sent answer is still read.
* Every request carries ``deadline_s`` slightly below the client timeout, so a slow method yields
  a structured retryable Envelope from the controller rather than a blind client timeout.
* ``controller.shutdown`` never autostarts a controller (starting one only to stop it is absurd);
  it fails with ``ControllerUnavailable`` ("not running") instead.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import hpclab
from hpclab.config import Paths, check_home_filesystem, get_paths, network_home_hint
from hpclab.controller import protocol as proto
from hpclab.controller.files import (
    InstanceLock,
    PidRecord,
    check_private_dir,
    home_problem,
    lock_is_held,
    pid_alive,
    pid_record_problem,
    probe_lock,
    read_pid,
    read_pid_record,
    recorded_elsewhere,
    remove_pidfile,
    socket_path,
    spawn_log_path,
    tail_lines,
    unusable_home_hint,
    wait_lock_free,
)
from hpclab.domain.clock import SystemClock, iso
from hpclab.domain.errors import ControllerUnavailable, ErrorCode, HpcLabError
from hpclab.domain.ids import new_id

log = logging.getLogger("hpclab.client")

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_STARTUP_TIMEOUT_S = 10.0
_PING_TIMEOUT_S = 2.0
_MAX_SPAWNS = 3
_SPAWN_LOG_MAX_BYTES = 256 * 1024
_NOT_RUNNING = (FileNotFoundError, ConnectionRefusedError)
_NEVER_AUTOSTART = frozenset({"controller.shutdown"})

_spawned: list[subprocess.Popen[bytes]] = []
_spawned_lock = threading.Lock()


def _track_spawned(proc: subprocess.Popen[bytes]) -> None:
    """Keep a reference to spawned controllers and reap finished ones, so long-lived clients
    neither leak zombies nor trigger 'subprocess still running' warnings."""
    with _spawned_lock:
        _spawned[:] = [p for p in _spawned if p.poll() is None]
        _spawned.append(proc)


def _reap_spawned(pid: int | None, timeout: float) -> None:
    """Wait for (reap) a controller this process spawned, so it does not linger as a zombie."""
    with _spawned_lock:
        procs = [p for p in _spawned if p.pid == pid]
    for proc in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=timeout)
    with _spawned_lock:
        _spawned[:] = [p for p in _spawned if p.poll() is None]


def _deadline_for(timeout: float) -> float:
    return max(0.05, min(timeout - 0.5, timeout * 0.9))


class _Connection:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.rfile = sock.makefile("rb")

    def request(self, payload: bytes, timeout: float) -> bytes:
        self.sock.settimeout(timeout)
        try:
            self.sock.sendall(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The controller may have rejected the request (e.g. too large) and hung up while we
            # were still sending: its answer is then already waiting in our receive buffer.
            answer = self._pending_answer()
            if answer is None:
                raise
            return answer
        # replies can be up to the limit plus the JSON-RPC wrapper
        line = proto.read_message(self.rfile, max_bytes=proto.MAX_MESSAGE_BYTES + 4096)
        if line is None:
            raise ConnectionResetError("the controller closed the connection")
        return line

    def _pending_answer(self) -> bytes | None:
        try:
            self.sock.settimeout(0.5)
            return proto.read_message(self.rfile, max_bytes=proto.MAX_MESSAGE_BYTES + 4096)
        except (OSError, proto.ProtocolError):
            return None

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.rfile.close()
        with contextlib.suppress(OSError):
            self.sock.close()


class HpcLabClient:
    """Reconnectable, autostarting client for one hpclab home (see module docstring)."""

    def __init__(
        self,
        paths: Paths | None = None,
        *,
        autostart: bool = True,
        timeout: float = DEFAULT_TIMEOUT_S,
        spawn_cmd: list[str] | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
        connect_timeout: float = 2.0,
        max_idle_connections: int = 4,
    ) -> None:
        """``spawn_cmd`` replaces the default ``[python, -m, hpclab.controller, --home, <home>]``
        argv used for autostart; the child always gets ``HPCLAB_HOME=<home>`` in its environment."""
        self.paths = paths or get_paths()
        self.autostart = autostart
        self.timeout = timeout
        self.spawn_cmd = list(spawn_cmd) if spawn_cmd else None
        self.startup_timeout = startup_timeout
        self.connect_timeout = connect_timeout
        self.max_idle_connections = max_idle_connections
        self.controller_info: dict[str, Any] | None = None
        self._ids = itertools.count(1)
        self._ids_lock = threading.Lock()
        self._idle: list[_Connection] = []
        self._pool_lock = threading.Lock()
        self._ensure_lock = threading.Lock()
        self._warned_version = False
        self._closed = False

    # -- public API ---------------------------------------------------------------------
    def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Invoke ``method`` and return its Envelope dict.

        Raises ``ControllerUnavailable`` when the controller cannot be reached/started (including
        an unusable home: its ``details.hint`` then says why), the connection was lost twice, or
        no reply arrived within ``timeout`` (the request may then still complete: retry with the
        same idempotency key). Never raises ``OSError``. Params JSON cannot represent are
        answered with an ``invalid_argument`` Envelope, like oversized requests."""
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dict or None")
        t = self.timeout if timeout is None else timeout
        with self._ids_lock:
            rpc_id = next(self._ids)
        req = proto.Request(method=method, params=params or {}, id=rpc_id, deadline_s=_deadline_for(t))
        try:
            payload = proto.encode_message(req.to_dict())
        except proto.MessageTooLarge as exc:
            return proto.error_envelope(
                new_id("request"),
                ErrorCode.LIMIT_EXCEEDED,
                f"request to {method} is too large to send: {exc.message}",
                details={"size": exc.size, "max_bytes": exc.max_bytes},
            )
        except (TypeError, ValueError) as exc:
            return proto.error_envelope(
                new_id("request"),
                ErrorCode.INVALID_ARGUMENT,
                f"params of {method} cannot be encoded as JSON: {exc}",
                details={"method": method},
            )
        autostart = self.autostart and method not in _NEVER_AUTOSTART
        last_error: str = ""
        for _attempt in range(2):  # the original try + one reconnect
            if self.controller_info is None:
                # first call, or the connection was lost (the controller may have been replaced):
                # autostart if allowed, and verify the protocol before sending anything
                self._ensure(autostart)
            conn = self._checkout(autostart)
            try:
                line = conn.request(payload, t)
            except TimeoutError:
                conn.close()
                raise ControllerUnavailable(
                    f"the controller did not answer {method} within {t:g}s",
                    details=self._diagnostics(
                        hint="the request may still complete in the controller; retry with the same "
                        "idempotency_key, or check `hpclab controller status`",
                        may_have_executed=True,
                    ),
                ) from None
            except proto.ProtocolError as exc:  # reply above the frame limit: stream out of sync
                conn.close()
                raise ControllerUnavailable(
                    f"the controller's reply to {method} is too large to read: {exc.message}",
                    details=self._diagnostics(
                        hint="request a smaller page (limit / max_bytes); client and controller "
                        "message limits may differ — restart the controller after upgrading",
                        may_have_executed=True,
                    ),
                ) from None
            except OSError as exc:
                conn.close()
                self._drop_idle()  # siblings of a dead connection are almost certainly dead too
                self.controller_info = None
                last_error = f"{type(exc).__name__}: {exc}"
                log.info("controller connection lost during %s (%s); reconnecting", method, last_error)
                continue
            try:
                resp = proto.parse_response(proto.decode_message(line))
            except proto.ProtocolError as exc:
                conn.close()
                raise ControllerUnavailable(
                    f"malformed reply from the controller: {exc.message}",
                    details=self._diagnostics(
                        hint="client and controller versions may differ; restart the controller"
                    ),
                ) from None
            # An id-less error answers the request in flight: a connection carries one request at
            # a time, and the controller could not read the id (e.g. the request was too large).
            if resp.id != rpc_id and not (resp.id is None and resp.error is not None):
                conn.close()
                raise ControllerUnavailable(
                    f"reply id {resp.id!r} does not match request id {rpc_id}",
                    details=self._diagnostics(hint="restart the controller"),
                )
            if resp.error is not None:
                if resp.error.code == proto.SHUTTING_DOWN:
                    # not executed: safe to re-send to the successor (autostarted if allowed)
                    conn.close()
                    self._drop_idle()
                    self.controller_info = None
                    last_error = resp.error.message
                    log.info("controller is shutting down; retrying %s on its successor", method)
                    time.sleep(0.1)
                    continue
                if resp.id is None:
                    conn.close()  # after an id-less error the connection may be closed or out of sync
                else:
                    self._checkin(conn)
                return proto.envelope_from_rpc_error(resp.error, new_id("request"))
            self._checkin(conn)
            assert resp.result is not None
            return resp.result
        raise ControllerUnavailable(
            f"could not complete {method}: the controller went away twice ({last_error})",
            details=self._diagnostics(hint="check `hpclab controller status` and the controller log"),
        )

    def ensure_running(self) -> dict[str, Any]:
        """Make sure a controller answers on this home's socket (spawning one if allowed) and
        return its ping data (pid, version, protocol, ...)."""
        return self._ensure(self.autostart)

    def _ensure(self, autostart: bool) -> dict[str, Any]:
        with self._ensure_lock:
            info = self._try_ping()
            if info is not None:
                return self._accept(info)
            problem = home_problem(self.paths.home)
            if problem is not None:  # no controller can run here, and none can be started
                raise ControllerUnavailable(
                    f"cannot use the hpclab home {self.paths.home}: {problem}",
                    details=self._diagnostics(hint=unusable_home_hint(self.paths.home, problem)),
                )
            if not autostart:
                raise ControllerUnavailable(
                    "the hpclab controller is not running", details=self._diagnostics(hint=self._start_hint())
                )
            try:  # the controller refuses a network home too; say so instead of spawning it
                check_home_filesystem(self.paths.home)
            except HpcLabError as exc:
                raise ControllerUnavailable(
                    exc.message,
                    retryable=False,
                    details=self._diagnostics(
                        hint=network_home_hint(),
                        fstype=exc.details.get("fstype"),
                        mount_point=exc.details.get("mount_point"),
                    ),
                ) from None
            try:
                self.paths.ensure()
            except OSError as exc:
                raise ControllerUnavailable(
                    f"cannot prepare the hpclab home for the controller: {exc}",
                    details=self._diagnostics(
                        hint="check ownership/permissions of the home and socket directories (both must "
                        "belong to you); remove a socket directory created by another user"
                    ),
                ) from None
            spawn_lock = InstanceLock(self.paths.home / "run" / "spawn.lock")
            try:
                acquired = spawn_lock.acquire(wait_s=self.startup_timeout)
            except OSError as exc:
                raise ControllerUnavailable(
                    f"cannot take the controller spawn lock {spawn_lock.path}: {exc}",
                    details=self._diagnostics(
                        hint="check ownership/permissions of the home and its run/ directory (both must "
                        "belong to you)"
                    ),
                ) from None
            if not acquired:
                raise ControllerUnavailable(
                    "timed out waiting for another client that is starting the controller",
                    details=self._diagnostics(hint=self._start_hint()),
                )
            try:
                info = self._wait_or_spawn()
            finally:
                spawn_lock.release()
            return self._accept(info)

    def ping(self, timeout: float = _PING_TIMEOUT_S) -> dict[str, Any] | None:
        """Ping data if a controller answers right now, else None. Never spawns."""
        return self._try_ping(timeout)

    def close(self) -> None:
        with self._pool_lock:
            idle, self._idle = self._idle, []
            self._closed = True
        for c in idle:
            c.close()

    def __enter__(self) -> HpcLabClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- connections -----------------------------------------------------------------------
    def _connect(self) -> _Connection:
        path = socket_path(self.paths)
        if path.parent.exists():
            try:
                check_private_dir(path.parent)
            except HpcLabError as exc:
                raise ControllerUnavailable(
                    exc.message, details=self._diagnostics(hint=exc.message)
                ) from None
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(self.connect_timeout)
            s.connect(str(path))
        except BaseException:
            s.close()
            raise
        return _Connection(s)

    def _checkout(self, autostart: bool) -> _Connection:
        with self._pool_lock:
            if self._idle:
                return self._idle.pop()
        try:
            return self._connect()
        except _NOT_RUNNING:
            if not autostart:
                raise ControllerUnavailable(
                    "the hpclab controller is not running", details=self._diagnostics(hint=self._start_hint())
                ) from None
        except OSError as exc:
            raise ControllerUnavailable(
                f"cannot connect to the controller socket {socket_path(self.paths)}: {exc}",
                details=self._diagnostics(hint=self._start_hint()),
            ) from None
        self._ensure(autostart)
        try:
            return self._connect()
        except OSError as exc:
            raise ControllerUnavailable(
                f"the controller started but its socket refused the connection: {exc}",
                details=self._diagnostics(hint="check the controller log"),
            ) from None

    def _drop_idle(self) -> None:
        with self._pool_lock:
            idle, self._idle = self._idle, []
        for c in idle:
            c.close()

    def _checkin(self, conn: _Connection) -> None:
        with self._pool_lock:
            if not self._closed and len(self._idle) < self.max_idle_connections:
                self._idle.append(conn)
                return
        conn.close()

    # -- startup ------------------------------------------------------------------------------
    def _try_ping(self, timeout: float = _PING_TIMEOUT_S) -> dict[str, Any] | None:
        try:
            conn = self._connect()
        except (OSError, ControllerUnavailable):
            return None
        try:
            req = proto.Request(method="controller.ping", id=0, deadline_s=_deadline_for(timeout))
            line = conn.request(proto.encode_message(req.to_dict()), timeout)
            resp = proto.parse_response(proto.decode_message(line))
        except (OSError, proto.ProtocolError):
            return None
        finally:
            conn.close()
        if resp.result is None or not resp.result.get("ok"):
            return None
        data = resp.result.get("data")
        return data if isinstance(data, dict) else None

    def _accept(self, info: dict[str, Any]) -> dict[str, Any]:
        if info.get("protocol") != proto.PROTOCOL_VERSION:
            raise ControllerUnavailable(
                f"the running controller (pid {info.get('pid')}, version {info.get('version')}) speaks "
                f"protocol {info.get('protocol')}, this client needs {proto.PROTOCOL_VERSION}",
                details=self._diagnostics(hint="restart it: `hpclab controller restart`"),
            )
        if info.get("version") != hpclab.__version__ and not self._warned_version:
            self._warned_version = True
            log.warning(
                "controller pid %s runs hpclab %s but this client is %s; restart the controller to upgrade",
                info.get("pid"),
                info.get("version"),
                hpclab.__version__,
            )
        self.controller_info = info
        return info

    def _wait_or_spawn(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        proc: subprocess.Popen[bytes] | None = None
        spawns = 0
        while True:
            info = self._try_ping()
            if info is not None:
                return info
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                if rc != proto.EXIT_ALREADY_RUNNING:
                    raise ControllerUnavailable(
                        f"the controller exited with code {rc} during startup",
                        details=self._diagnostics(
                            hint="see spawn_log_tail / the controller log", with_spawn_tail=True
                        ),
                    )
                proc = None  # another controller owns the lock; wait for it to answer
            if proc is None and not lock_is_held(self.paths.lock):
                if spawns >= _MAX_SPAWNS:
                    raise ControllerUnavailable(
                        f"the controller failed to start after {spawns} attempts",
                        details=self._diagnostics(
                            hint="see spawn_log_tail / the controller log", with_spawn_tail=True
                        ),
                    )
                proc = self._spawn()
                spawns += 1
            if time.monotonic() >= deadline:
                if proc is not None:
                    hint = (
                        f"controller pid {proc.pid} is still starting (slow disk or imports?); retry in a "
                        "few seconds, or see spawn_log_tail / the controller log"
                    )
                elif lock_is_held(self.paths.lock):
                    hint = _held_lock_hint(self.paths)
                else:
                    hint = "see spawn_log_tail / the controller log"
                raise ControllerUnavailable(
                    f"the controller did not become ready within {self.startup_timeout:g}s",
                    details=self._diagnostics(hint=hint, with_spawn_tail=True),
                )
            time.sleep(0.05)

    def _spawn(self) -> subprocess.Popen[bytes]:
        cmd = self.spawn_cmd or [sys.executable, "-m", "hpclab.controller", "--home", str(self.paths.home)]
        env = dict(os.environ)
        env["HPCLAB_HOME"] = str(self.paths.home)
        log_path = spawn_log_path(self.paths)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                mode = "ab" if log_path.stat().st_size < _SPAWN_LOG_MAX_BYTES else "wb"
            except FileNotFoundError:
                mode = "ab"
            with open(log_path, mode) as out:
                out.write(
                    f"--- {iso(SystemClock().now())} spawned by pid {os.getpid()}: {shlex.join(cmd)}\n".encode()
                )
                out.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=out,
                    start_new_session=True,
                    close_fds=True,
                    cwd=str(self.paths.home),
                    env=env,
                )
        except OSError as exc:
            raise ControllerUnavailable(
                f"cannot start the controller ({shlex.join(cmd)}): {exc}",
                details=self._diagnostics(hint="check that hpclab is installed for this Python"),
            ) from None
        _track_spawned(proc)
        log.info("spawned hpclab controller pid %s for %s", proc.pid, self.paths.home)
        return proc

    # -- diagnostics -----------------------------------------------------------------------------
    def _start_hint(self) -> str:
        return (
            f"start it with `hpclab controller start` or `{Path(sys.executable).name} -m hpclab.controller "
            f"--home {self.paths.home} --foreground` (the latter shows startup errors)"
        )

    def _diagnostics(self, *, hint: str, with_spawn_tail: bool = False, **extra: Any) -> dict[str, Any]:
        return _diagnostics(self.paths, hint=hint, with_spawn_tail=with_spawn_tail, **extra)


def _diagnostics(paths: Paths, *, hint: str, with_spawn_tail: bool = False, **extra: Any) -> dict[str, Any]:
    """Details of every ``ControllerUnavailable``. Never raises: it runs on error paths, where the
    home itself may be the problem (then ``home_problem`` says why, ``lock_held`` is None =
    unknown, and the hint is about the home)."""
    problem = home_problem(paths.home)
    out: dict[str, Any] = {
        "home": str(paths.home),
        "socket": str(socket_path(paths)),
        "log": str(paths.log),
        "spawn_log": str(spawn_log_path(paths)),
        "pidfile_pid": read_pid(paths.pidfile),
        "lock_held": probe_lock(paths.lock),
        "hint": unusable_home_hint(paths.home, problem) if problem else hint,
        **extra,
    }
    if problem:
        out["home_problem"] = problem
    if with_spawn_tail:
        out["spawn_log_tail"] = tail_lines(spawn_log_path(paths))
        out["log_tail"] = tail_lines(paths.log, max_lines=10)
    return out


# ---------------------------------------------------------------------------
# Lifecycle helpers (CLI: hpclab controller start|stop|status)
# ---------------------------------------------------------------------------


def start_controller(
    paths: Paths | None = None,
    *,
    spawn_cmd: list[str] | None = None,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> dict[str, Any]:
    """Start the controller for ``paths`` if needed; return its ping data."""
    with HpcLabClient(paths, spawn_cmd=spawn_cmd, startup_timeout=startup_timeout) as client:
        return client.ensure_running()


def controller_status(paths: Paths | None = None, *, timeout: float = 5.0) -> dict[str, Any]:
    """Envelope dict describing the controller. When it is not running this is still ``ok=True``
    with ``data.running = False`` and ``data.state`` = ``stopped`` | ``unresponsive``.

    Raises ``ControllerUnavailable`` (never ``OSError``) when the home is not usable (unreadable,
    a file in its place, ...) or its lock file cannot be read: whether a controller runs there
    cannot be told, so it is neither reported as stopped nor as running; ``details.hint`` says
    what to fix."""
    p = paths or get_paths()
    with HpcLabClient(p, autostart=False, timeout=timeout) as client:
        try:
            return client.call("controller.status", timeout=timeout)
        except ControllerUnavailable as exc:
            if exc.details.get("home_problem"):
                raise
            held = probe_lock(p.lock)
            if held is None:
                raise _lock_unknown(p) from None
            rec = read_pid_record(p.pidfile) if held else None
            data = {
                "running": False,
                "state": "unresponsive" if held else "stopped",
                "pid": rec.pid if rec else None,
                "host": rec.host if rec else None,
                "home": str(p.home),
                "socket": str(socket_path(p)),
                "log": str(p.log),
                "reason": exc.message,
            }
            if rec is not None and recorded_elsewhere(rec):
                summary = (
                    f"the lock is held by a controller of host {rec.host or '?'} (pid {rec.pid}), another "
                    "machine sharing this hpclab home; it cannot be reached from here"
                )
            elif held:
                summary = f"controller holds the lock (pid {data['pid']}) but does not answer"
            else:
                summary = "controller is not running"
            return proto.ok_envelope(new_id("request"), data, summary=summary)


def stop_controller(
    paths: Paths | None = None, *, timeout: float = 20.0, force: bool = False
) -> dict[str, Any]:
    """Stop the controller for ``paths`` and wait until it released its lock.

    Asks politely over the socket (``controller.shutdown``) and falls back to SIGTERM of the
    pid in the pidfile — only while the instance lock is held, so a stale pidfile never gets an
    unrelated process signalled, and only when this machine recorded that pid and it still names
    the process that did (a home shared over NFS shares the pidfile and, where locks propagate,
    the lock: another machine's pid means nothing here; see ``_check_signallable``). With
    ``force`` a controller that does not exit within ``timeout`` is SIGKILLed (its leased outbox
    items are released at the next start). The default ``timeout`` exceeds the controller's own
    shutdown budget (10 s + bounded closes).

    Returns ``{was_running, stopped, pid, method, forced}``. Raises ``ControllerUnavailable``
    (never ``OSError``) when the home or its lock file is not usable, so whether a controller
    runs cannot be told, and when the controller's pid may not be signalled (another user's,
    another machine's, or a reused one)."""
    p = paths or get_paths()
    result: dict[str, Any] = {
        "was_running": False,
        "stopped": True,
        "pid": None,
        "method": None,
        "forced": False,
    }
    held = probe_lock(p.lock)
    if held is None:
        raise _lock_unknown(p)
    if not held:
        _cleanup_stale(p)
        return result
    result["was_running"] = True
    with HpcLabClient(p, autostart=False, timeout=5.0) as client:
        info = client.ping()
        pid: int | None = info.get("pid") if info else None
        if pid == os.getpid():
            raise HpcLabError(
                "the controller runs inside this process; call Controller.stop() instead",
                code=ErrorCode.PRECONDITION_FAILED,
            )
        if info is not None:
            with contextlib.suppress(ControllerUnavailable):
                env = client.call("controller.shutdown", timeout=5.0)
                if env.get("ok"):
                    result["method"] = "rpc"
    rec: PidRecord | None = None  # set when the pid comes from the pidfile, not from the controller
    if result["method"] is None:
        if pid is None:
            rec = read_pid_record(p.pidfile)
            pid = rec.pid if rec else None
        # The pid is briefly unknown right after a controller took the lock (pidfile not yet
        # written) and right before it releases it (pidfile already removed): wait that out.
        gap_end = time.monotonic() + min(timeout, 3.0)
        while pid is None and lock_is_held(p.lock) and time.monotonic() < gap_end:
            time.sleep(0.05)
            rec = read_pid_record(p.pidfile)
            pid = rec.pid if rec else None
        if pid is None and not lock_is_held(p.lock):
            _cleanup_stale(p)
            return result  # it finished stopping on its own
        if pid is None:
            raise ControllerUnavailable(
                "the controller lock is held but its pid is unknown",
                details=_diagnostics(p, hint="find the process holding the lock file (lsof) and stop it"),
            )
        if pid == os.getpid():
            raise HpcLabError(
                "the controller runs inside this process; call Controller.stop() instead",
                code=ErrorCode.PRECONDITION_FAILED,
            )
        _check_signallable(p, rec)
        _signal(p, pid, signal.SIGTERM)
        result["method"] = "sigterm"
    result["pid"] = pid
    if wait_lock_free(p.lock, timeout):
        _reap_spawned(pid, 2.0)
        return result
    if force and pid is not None and pid_alive(pid):
        _check_signallable(p, rec)  # the pid may have been reused while we waited
        log.warning("controller pid %s did not stop within %.0fs; sending SIGKILL", pid, timeout)
        _signal(p, pid, signal.SIGKILL)
        result["forced"] = True
        if wait_lock_free(p.lock, 5.0):
            _reap_spawned(pid, 2.0)
            _cleanup_stale(p)
            return result
    result["stopped"] = False
    result["hint"] = (
        "the controller is still shutting down (a background step may be finishing); retry or use force"
    )
    return result


def _signal(p: Paths, pid: int, sig: signal.Signals) -> None:
    """Send ``sig`` to the controller pid; a process that is already gone is fine. Any other
    failure (EPERM: the pid belongs to another user — a stale pidfile, or a controller started
    with sudo) is ``ControllerUnavailable``, never a raw ``OSError``."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ControllerUnavailable(
            f"cannot send {sig.name} to controller pid {pid}: {exc.strerror or exc}",
            details=_diagnostics(
                p,
                hint=f"pid {pid} (from {p.pidfile}) is not a process you may signal: if it is a controller "
                "started with sudo, stop it the same way; otherwise find the process holding "
                f"{p.lock} (lsof) and stop it",
                pid=pid,
            ),
        ) from None


def _check_signallable(p: Paths, rec: PidRecord | None) -> None:
    """Raise ``ControllerUnavailable`` (not retryable) when the pidfile's pid must not be
    signalled from this machine: another machine sharing this home (NFS) recorded it, so here it
    may name the user's editor or notebook kernel, or it was reused by a process started after
    the controller that recorded it. ``rec`` None (the pid came from the controller itself,
    over this machine's socket) passes."""
    problem = pid_record_problem(rec) if rec is not None else None
    if rec is None or problem is None:
        return
    if recorded_elsewhere(rec):
        where = f"host {rec.host}" if rec.host else "the machine that started it"
        hint = (
            f"stop it on {where} (`hpclab controller stop` there); nothing was signalled here. Machines "
            "must not share an hpclab home: give each machine its own HPCLAB_HOME (or --home) on its "
            "local disk"
        )
    else:
        hint = (
            f"nothing was signalled. Find the process holding {p.lock} (`lsof {p.lock}`) and stop it; "
            "then retry"
        )
    raise ControllerUnavailable(
        f"refusing to signal the controller: {problem}",
        retryable=False,
        details=_diagnostics(p, hint=hint, pid=rec.pid, pidfile_host=rec.host),
    )


def _held_lock_hint(p: Paths) -> str:
    """Hint for a controller lock that is held while the socket does not answer."""
    rec = read_pid_record(p.pidfile)
    if rec is not None and recorded_elsewhere(rec):
        return (
            f"the lock is held by the controller of host {rec.host or '?'} (pid {rec.pid}), another machine "
            "sharing this hpclab home (a network home); it cannot be reached or stopped from here. Stop it "
            "there, and give each machine its own HPCLAB_HOME on its local disk"
        )
    return "a controller holds the lock but does not answer; `hpclab controller stop --force` then retry"


def _lock_unknown(p: Paths) -> HpcLabError:
    """The lock cannot be probed (unusable home, or a lock file we cannot read — e.g. left
    root-owned by a controller started with sudo): whether a controller runs cannot be told.
    Returns the ``ControllerUnavailable`` to raise."""
    problem = home_problem(p.home)
    return ControllerUnavailable(
        f"cannot use the hpclab home {p.home}: {problem} (so whether a controller runs cannot be told)"
        if problem
        else f"cannot tell whether a controller runs: its lock file {p.lock} cannot be read",
        details=_diagnostics(
            p,
            hint=f"check ownership/permissions of {p.lock} (it must belong to you; a controller "
            "started with sudo leaves root-owned files behind)",
        ),
    )


def _cleanup_stale(p: Paths) -> None:
    """Remove a dead controller's pidfile and socket, only while holding its lock (best effort:
    never raises ``OSError``)."""
    try:
        if not p.lock.parent.is_dir():
            return  # nothing was ever started here; never create a home as a side effect
        lock = InstanceLock(p.lock)
        if not lock.acquire(wait_s=0):
            return
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            remove_pidfile(p.pidfile)
        with contextlib.suppress(OSError):
            sock = socket_path(p)
            if sock.is_socket():
                sock.unlink()
    finally:
        lock.release()
