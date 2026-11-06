"""The hpclab controller: one long-lived process per ``HPCLAB_HOME``.

It is the single writer of the ledger. Frontends (CLI, MCP servers, Python) are reconnectable
clients on a Unix socket; closing an agent session never stops background work.

Threads
-------
* ``controller-server`` accepts connections; each connection gets a handler thread and may send
  any number of requests (``protocol`` framing). Built-in ``controller.*`` methods run inline.
  App methods run on a fresh daemon thread per request, bounded by ``max_concurrent_requests``
  and a per-request deadline: past the deadline the caller gets a retryable ``unavailable``
  Envelope (``details.started`` says whether the work may still complete) and the server stays
  responsive. A hung call can never block shutdown because every thread is a daemon thread.
* ``worker-N`` (``worker_threads``) loop ``OutboxRunner.run_once(limit=1)``; SQLite
  ``BEGIN IMMEDIATE`` serializes leasing. Workers are woken right after every app request and
  periodic run (work was probably enqueued) and otherwise poll every ``tick_interval_s``.
* ``controller-scheduler`` runs ``PeriodicScheduler``; each due task runs on its own daemon
  thread so one slow task (a hung ssh poll) cannot starve the others, and never overlaps itself.
  A forward wall-clock jump (laptop sleep) makes every task due at once (catch-up after resume).
  Built-in tasks: ``controller.renew_leases``, ``controller.housekeeping`` and
  ``controller.socket_watch`` (stops the controller if its socket file was deleted, e.g. by a
  ``/tmp`` cleaner — otherwise it would hold the lock while no client could reach it).
* A reply is always sent for a request that may have run (never a silent hang-up, which would
  make clients re-send it); see ``protocol`` for which failures are Envelopes and which are
  JSON-RPC errors.

Crash safety
------------
* Single instance: exclusive ``flock`` on ``paths.lock`` for the process lifetime + pidfile. A
  stale socket file is removed only while holding the lock.
* The lock is held for as long as *any* thread of the controller that may write the ledger
  (outbox workers, periodic tasks, app requests) is alive. If some are still running when the
  shutdown timeout expires, the controller closes its socket and reports "stopped", but keeps
  the lock and pidfile until the last one finishes; ``python -m hpclab.controller`` then exits
  the process, which kills them and releases the lock in the same instant. So "lock free"
  really means "no previous controller can still write".
* Outbox leases held by a previous controller (``lease_owner`` ``controller:*``) are therefore
  released at startup — holding the lock proves that controller is gone — so work resumes
  immediately instead of after lease expiry. Leases of items still running are extended every
  ``lease_s/3``, and an idle worker extends a busy sibling's lease *before* leasing whenever it
  looks stale (the wall clock jumps forward when a laptop wakes up), so a long handler is never
  picked up twice while its controller is alive.
* Built-in periodic housekeeping prunes ``done`` outbox rows older than ``outbox_retention_s``
  (``dead`` rows are kept for diagnosis; events remain the audit log).

Logging goes to ``paths.log`` (rotating 5 MB × 3) through the ``hpclab`` logger — never stdout.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import logging.handlers
import os
import secrets
import signal
import socket
import socketserver
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import FrameType
from typing import Any, Protocol

import hpclab
from hpclab.config import Paths, check_home_filesystem
from hpclab.controller import protocol as proto
from hpclab.controller.files import (
    InstanceLock,
    check_private_dir,
    read_pid,
    remove_pidfile,
    socket_path,
    write_pidfile,
)
from hpclab.domain.clock import SystemClock, iso
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument
from hpclab.domain.ids import new_id
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem, OutboxRunner
from hpclab.services.periodic import PeriodicScheduler, PeriodicTask

log = logging.getLogger("hpclab.controller")

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s [%(threadName)s] %(message)s"


def log_formatter() -> logging.Formatter:
    """UTC ISO timestamps, matching the ledger's, so log lines and events correlate."""
    fmt = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = time.gmtime
    return fmt


BUILTIN_PREFIX = "controller."
BUILTIN_METHODS = ("controller.ping", "controller.status", "controller.wake", "controller.shutdown")
MAX_REQUEST_DEADLINE_S = 3600.0
SLOW_REQUEST_S = 5.0
_SLEEP_GAP_S = 30.0
_HOUSEKEEPING_INTERVAL_S = 3600.0
_HOUSEKEEPING_BATCH = 5000
_SOCKET_WATCH_INTERVAL_S = 30.0
_MAX_SKIP_BYTES = 256 * 1024 * 1024  # most of an oversized request we read through to resync
# controller.status "attention" thresholds for background work that may be hung
_HUNG_TASK_MIN_S = 600.0  # a periodic task running longer than max(this, N x its interval)
_HUNG_TASK_INTERVALS = 10.0
_LONG_HANDLER_S = 1800.0  # an outbox handler running longer than this


# ---------------------------------------------------------------------------
# App contract
# ---------------------------------------------------------------------------


class App(Protocol):
    """What the controller serves. The service facade implements this; tests use toy apps.

    * ``dispatch`` returns an Envelope dict for every call, including domain failures. It may
      raise ``protocol.MethodNotFound`` (or any ``HpcLabError``) — the controller converts those
      to Envelope errors — and any other exception becomes an ``internal`` Envelope error.
    * ``outbox_handlers`` / ``periodic_tasks`` are read once at startup.
    * ``close`` is called once at shutdown, after the last worker, periodic task and request
      thread finished (bounded: a ``close`` that hangs is abandoned after a few seconds). When
      the shutdown timeout leaves threads running, ``python -m hpclab.controller`` exits without
      calling it.
    """

    def dispatch(self, method: str, params: dict[str, Any], *, request_id: str) -> dict[str, Any]: ...

    def outbox_handlers(self) -> dict[str, Handler]: ...

    def periodic_tasks(self) -> list[PeriodicTask]: ...

    def close(self) -> None: ...


AppFactory = Callable[[ServiceContext], App]
ContextFactory = Callable[[Paths], ServiceContext]


class ControllerAlreadyRunning(HpcLabError):
    code = ErrorCode.PRECONDITION_FAILED

    def __init__(self, pid: int | None, paths: Paths) -> None:
        who = f"pid {pid}" if pid else "pid unknown"
        super().__init__(
            f"hpclab controller already running ({who}) for home {paths.home}",
            details={"pid": pid, "home": str(paths.home), "socket": str(socket_path(paths))},
        )
        self.pid = pid


class MinimalApp:
    """Serves nothing but the built-in ``controller.*`` methods.

    Used when the service layer cannot be loaded, so clients still reach a controller that
    explains *why* their method is unavailable instead of failing to connect at all."""

    def __init__(self, ctx: ServiceContext | None = None, *, load_error: str | None = None) -> None:
        self.ctx = ctx
        self.load_error = load_error

    def dispatch(self, method: str, params: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        if self.load_error:
            reason = (
                f"the hpclab service layer failed to load ({self.load_error}); only controller.* "
                "methods are available — see the controller log, fix the error and restart the controller"
            )
        else:
            reason = "this controller only serves controller.* methods"
        raise proto.MethodNotFound(method, known=list(BUILTIN_METHODS), reason=reason)

    def outbox_handlers(self) -> dict[str, Handler]:
        return {}

    def periodic_tasks(self) -> list[PeriodicTask]:
        return []

    def close(self) -> None:
        return None


def default_app_factory(ctx: ServiceContext) -> App:
    """``hpclab.services.app:create_app``, imported lazily; falls back to ``MinimalApp`` (with a
    logged reason) when the service layer is missing or fails to import/build."""
    try:
        module = importlib.import_module("hpclab.services.app")
        create_app = module.create_app
    except ModuleNotFoundError as exc:
        if exc.name == "hpclab.services.app":
            log.warning("hpclab.services.app is not installed; serving controller.* methods only")
            return MinimalApp(ctx, load_error="hpclab.services.app is not installed")
        log.exception("importing hpclab.services.app failed")
        return MinimalApp(ctx, load_error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        log.exception("importing hpclab.services.app failed")
        return MinimalApp(ctx, load_error=f"{type(exc).__name__}: {exc}")
    try:
        app: App = create_app(ctx)
    except Exception as exc:
        log.exception("hpclab.services.app.create_app failed")
        return MinimalApp(ctx, load_error=f"create_app failed: {type(exc).__name__}: {exc}")
    return app


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _WorkerState:
    name: str
    worker_id: str
    thread: threading.Thread | None = None
    processed: int = 0
    errors: int = 0
    current: OutboxItem | None = None
    current_since: float | None = None
    # ctx.clock time the current item's lease was last extended (see Controller._renew_busy_leases)
    lease_renewed_at: datetime | None = None
    last_item_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        cur, since = self.current, self.current_since
        return {
            "name": self.name,
            "worker_id": self.worker_id,
            "alive": bool(self.thread and self.thread.is_alive()),
            "processed": self.processed,
            "errors": self.errors,
            "last_item_at": self.last_item_at,
            "current": None
            if cur is None
            else {
                "outbox_id": cur.outbox_id,
                "kind": cur.kind,
                "target_id": cur.target_id,
                "tries": cur.tries,
                "running_for_s": round(time.monotonic() - since, 3) if since is not None else None,
            },
        }


@dataclass
class _RequestStats:
    total: int = 0
    errors: int = 0
    internal_errors: int = 0
    protocol_errors: int = 0
    deadline_exceeded: int = 0
    busy_rejected: int = 0
    in_flight: int = 0
    app_calls_running: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "total": self.total,
                "errors": self.errors,
                "internal_errors": self.internal_errors,
                "protocol_errors": self.protocol_errors,
                "deadline_exceeded": self.deadline_exceeded,
                "busy_rejected": self.busy_rejected,
                "in_flight": self.in_flight,
                "app_calls_running": self.app_calls_running,
            }


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64

    def __init__(self, address: str, controller: Controller) -> None:
        super().__init__(address, _Handler, bind_and_activate=False)
        self.controller = controller

    def handle_error(self, request: Any, client_address: Any) -> None:
        # the default implementation prints to stderr; keep everything in the log
        log.exception("connection handler crashed")


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def handle(self) -> None:
        ctl = self.server.controller
        conn: socket.socket = self.connection
        ctl._register_conn(conn)
        try:
            while True:
                try:
                    line = proto.read_message(self.rfile, max_bytes=ctl.max_message_bytes)
                except proto.MessageTooLarge as exc:
                    ctl._stats.add("protocol_errors")
                    log.warning("rejecting request: %s", exc.message)
                    if not self._send(ctl._encode_rpc_error(None, exc.to_rpc_error())):
                        return
                    # Discard the rest of that line: the sender is probably still writing it and
                    # would otherwise hit EPIPE before it ever reads our answer. Bounded, so a
                    # peer streaming garbage without newlines is disconnected.
                    try:
                        if not proto.skip_line(self.rfile, max_skip=_MAX_SKIP_BYTES):
                            return
                    except OSError:
                        return
                    continue
                except OSError:
                    return
                if line is None:
                    return
                # in_flight spans the reply write, so shutdown never hangs up before replying
                ctl._stats.add("in_flight")
                try:
                    try:
                        out = ctl.handle_line(line)
                    except Exception:
                        # handle_line never raises by contract; if it ever does, still answer:
                        # hanging up without a reply makes clients re-send a request that may
                        # already have run.
                        log.exception("unexpected failure while handling a request")
                        out = ctl._encode_rpc_error(
                            None,
                            proto.RpcError(
                                proto.INTERNAL_ERROR,
                                f"internal controller error; see {ctl.paths.log}",
                                data={"log": str(ctl.paths.log)},
                            ),
                        )
                    sent = out is None or self._send(out)
                finally:
                    ctl._stats.add("in_flight", -1)
                if not sent:
                    return
        finally:
            ctl._unregister_conn(conn)

    def _send(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
        except OSError:
            return False
        return True


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class Controller:
    """Single-instance controller for one hpclab home.

    Lifecycle: ``start()`` (or ``start_in_thread()``) → serving → ``stop()``. ``serve_forever()``
    starts, blocks until SIGTERM/SIGINT/SIGHUP or ``controller.shutdown``, then shuts down; Ctrl-C
    (SIGINT) during that shutdown exits the process immediately. A Controller object is
    single-use.

    ``stop()`` returns within the shutdown budget. If handlers/requests/periodic tasks are still
    running then, the socket is gone and ``state`` is ``"stopped"``, but ``finalize_pending`` is
    True: the app, pidfile and instance lock are released only when the last of them returns, so
    a successor can never run concurrently with them.
    """

    def __init__(
        self,
        paths: Paths,
        app_factory: AppFactory,
        *,
        context_factory: ContextFactory | None = None,
        worker_threads: int | None = None,
        tick_interval_s: float | None = None,
        lease_s: float = 300.0,
        request_deadline_s: float = 55.0,
        max_concurrent_requests: int = 16,
        max_message_bytes: int = proto.MAX_MESSAGE_BYTES,
        shutdown_timeout_s: float = 10.0,
        lock_wait_s: float = 0.5,
        outbox_retention_s: float = 7 * 86400.0,
        log_to_file: bool = True,
        log_level: int = logging.INFO,
    ) -> None:
        if lease_s <= 0 or request_deadline_s <= 0 or max_concurrent_requests < 1:
            raise ValueError("lease_s, request_deadline_s and max_concurrent_requests must be positive")
        if tick_interval_s is not None and not tick_interval_s > 0:
            raise ValueError("tick_interval_s must be > 0 (0 would make every worker spin)")
        if worker_threads is not None and worker_threads < 0:
            raise ValueError("worker_threads must be >= 0")
        self.paths = paths
        self.sock_path = socket_path(paths)  # byte-length-safe variant of paths.socket
        self.app_factory = app_factory
        self.context_factory = context_factory
        self._worker_threads_arg = worker_threads
        self._tick_arg = tick_interval_s
        self.worker_threads = worker_threads if worker_threads is not None else 2
        self.tick_interval_s = tick_interval_s if tick_interval_s is not None else 1.0
        self.lease_s = lease_s
        self.request_deadline_s = request_deadline_s
        self.max_concurrent_requests = max_concurrent_requests
        self.max_message_bytes = max_message_bytes
        self.shutdown_timeout_s = shutdown_timeout_s
        self.lock_wait_s = lock_wait_s
        self.outbox_retention_s = outbox_retention_s
        self.log_to_file = log_to_file
        self.log_level = log_level

        self.pid = os.getpid()
        self.instance_token = secrets.token_hex(4)
        self.started_at: str | None = None
        self._started_mono: float | None = None

        self._state = "new"  # new | starting | running | stopping | stopped
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._stop_reason = ""
        self._signal_received: int | None = None
        self._owner_waits = False
        self._shutdown_thread: threading.Thread | None = None

        self._wake_cond = threading.Condition()
        self._wake_gen = 0
        self._sched_wake = threading.Event()

        self._lock = InstanceLock(paths.lock)
        self._pidfile_written = False
        self._socket_id: tuple[int, int] | None = None  # (st_dev, st_ino) of the socket we bound
        self._server: _Server | None = None
        self._server_thread: threading.Thread | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._workers: list[_WorkerState] = []
        self._periodic_threads: set[threading.Thread] = set()
        self._periodic_lock = threading.Lock()
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self._stats = _RequestStats()
        # Threads that may write the ledger or run outbox handlers (workers, periodic tasks, app
        # requests). The instance is finalized (app closed, pidfile removed, lock released) only
        # when none is left — see _shutdown / _leave_live.
        self._live_lock = threading.Lock()
        self._live = 0
        self._finalize_pending = False
        self._finalizing = False
        self._close_timeout_s = max(1.0, min(5.0, shutdown_timeout_s))
        self._log_handler: logging.Handler | None = None
        self._saved_log_level: int | None = None

        self._ctx: ServiceContext | None = None
        self._app: App | None = None
        self._handlers: dict[str, Handler] = {}
        self._scheduler: PeriodicScheduler | None = None

    # -- accessors ------------------------------------------------------------------
    @property
    def ctx(self) -> ServiceContext:
        if self._ctx is None:
            raise RuntimeError("controller not started")
        return self._ctx

    @property
    def app(self) -> App:
        if self._app is None:
            raise RuntimeError("controller not started")
        return self._app

    @property
    def scheduler(self) -> PeriodicScheduler:
        if self._scheduler is None:
            raise RuntimeError("controller not started")
        return self._scheduler

    @property
    def socket_path(self) -> str:
        return str(self.sock_path)

    @property
    def state(self) -> str:
        return self._state

    @property
    def running(self) -> bool:
        return self._state == "running"

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> Controller:
        """Acquire the instance lock, build the app and start serving (non-blocking).

        Raises ``ControllerAlreadyRunning`` if another controller owns this home."""
        with self._state_lock:
            if self._state != "new":
                raise RuntimeError(
                    f"Controller.start() called in state {self._state!r}; objects are single-use"
                )
            self._state = "starting"
        try:
            self._start()
        except BaseException:
            self._abort_start()
            raise
        with self._state_lock:
            self._state = "running"
        if self._stop.is_set() and not self._owner_waits:
            # a stop (e.g. controller.shutdown) arrived while we were still starting
            self.request_stop(self._stop_reason or "stop requested during startup")
        return self

    def start_in_thread(self) -> Controller:
        """Start serving on background threads and return immediately (tests, embedding).
        Stop with ``stop()`` or use the controller as a context manager."""
        return self.start()

    def serve_forever(self, *, install_signal_handlers: bool = True) -> None:
        """Start (if needed) and block until asked to stop, then shut down cleanly.

        SIGTERM/SIGINT/SIGHUP handlers are installed when called from the main thread."""
        self._owner_waits = True
        previous: dict[int, Any] = {}
        # installed before startup so a SIGTERM during startup still ends in a clean shutdown
        if install_signal_handlers and threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[sig] = signal.signal(sig, self._on_signal)
        try:
            if self._state == "new":
                self.start()
            try:
                while not self._stop.is_set():
                    if self._signal_received is not None:
                        self.request_stop(f"signal {signal.Signals(self._signal_received).name}")
                        break
                    time.sleep(0.1)
            finally:
                self._shutdown()
        finally:
            for signum, handler in previous.items():
                with contextlib.suppress(ValueError, TypeError):
                    signal.signal(signum, handler)

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the controller to stop (thread-safe; idempotent)."""
        if not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()
        self._sched_wake.set()
        self._kick_workers()
        if self._owner_waits:
            return  # serve_forever's thread performs the shutdown
        with self._state_lock:
            if self._shutdown_thread is None and self._state == "running":
                self._shutdown_thread = threading.Thread(
                    target=self._shutdown, name="controller-shutdown", daemon=True
                )
                self._shutdown_thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop and wait for the shutdown to finish (bounded by ``timeout``)."""
        self.request_stop("stop() called")
        self._shutdown(timeout)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the controller has fully stopped."""
        return self._stopped.wait(timeout)

    def __enter__(self) -> Controller:
        if self._state == "new":
            self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        # Only set a plain attribute: taking locks (Event.set) inside a signal handler can
        # deadlock with the interrupted main thread. serve_forever polls this flag.
        if signum == signal.SIGINT and self._signal_received is not None and self._stop.is_set():
            # Ctrl-C while already stopping: the operator does not want to wait for a hung
            # handler. Exit now; the kernel releases the lock and the next controller releases
            # this one's outbox leases (handlers are crash-safe). Only SIGINT: a repeated SIGTERM
            # (overlapping `hpclab controller stop`, launchd/systemd) stays graceful.
            os._exit(128 + signum)
        if self._signal_received is None or not self._stop.is_set():
            self._signal_received = signum

    # -- startup ------------------------------------------------------------------------
    def _start(self) -> None:
        # before the lock: a home shared over NFS makes both lock and pidfile meaningless
        network_home_warning = check_home_filesystem(self.paths.home)
        paths = self.paths.ensure()
        if not self._lock.acquire(wait_s=self.lock_wait_s):
            raise ControllerAlreadyRunning(read_pid(paths.pidfile), paths)
        write_pidfile(paths.pidfile, self.pid)
        self._pidfile_written = True
        self._setup_logging()
        if network_home_warning:
            log.warning("%s", network_home_warning)
        self.started_at = iso(SystemClock().now())
        self._started_mono = time.monotonic()
        log.info(
            "controller starting: pid=%s version=%s home=%s socket=%s",
            self.pid,
            hpclab.__version__,
            paths.home,
            self.sock_path,
        )

        ctx = self.context_factory(paths) if self.context_factory else ServiceContext.create(paths)
        self._ctx = ctx
        if self._worker_threads_arg is None:
            self.worker_threads = max(1, int(ctx.settings.worker_threads))
        if self._tick_arg is None:
            self.tick_interval_s = max(0.01, float(ctx.settings.tick_interval_s))

        app = self.app_factory(ctx)
        self._app = app
        self._handlers = self._collect_handlers(app)
        tasks = self._builtin_tasks() + self._collect_tasks(app)
        self._scheduler = PeriodicScheduler(tasks, SystemClock())

        self._reclaim_orphaned_leases()
        self._bind_server()
        self._start_threads()
        log.info(
            "controller ready: app=%s workers=%d outbox_kinds=%s periodic=%s",
            type(app).__qualname__,
            self.worker_threads,
            sorted(self._handlers),
            [t.name for t in self._scheduler.tasks],
        )

    def _collect_handlers(self, app: App) -> dict[str, Handler]:
        getter = getattr(app, "outbox_handlers", None)
        handlers = dict(getter() or {}) if getter else {}
        for kind, fn in handlers.items():
            if not callable(fn):
                raise TypeError(f"outbox handler for {kind!r} is not callable")
        return handlers

    def _collect_tasks(self, app: App) -> list[PeriodicTask]:
        getter = getattr(app, "periodic_tasks", None)
        tasks = [PeriodicTask.coerce(t) for t in (getter() or [])] if getter else []
        for t in tasks:
            if t.name.startswith(BUILTIN_PREFIX):
                raise ValueError(f"periodic task name {t.name!r} uses the reserved prefix {BUILTIN_PREFIX!r}")
        return tasks

    def _builtin_tasks(self) -> list[PeriodicTask]:
        return [
            PeriodicTask(
                "controller.renew_leases",
                max(0.2, self.lease_s / 3.0),
                self._renew_leases,
                run_at_start=False,
            ),
            PeriodicTask("controller.housekeeping", _HOUSEKEEPING_INTERVAL_S, self._housekeeping),
            PeriodicTask(
                "controller.socket_watch", _SOCKET_WATCH_INTERVAL_S, self._socket_watch, run_at_start=False
            ),
        ]

    def _reclaim_orphaned_leases(self) -> None:
        with self.ctx.db.tx(source="worker") as tx:
            cur = tx.execute(
                "UPDATE outbox SET state='pending', lease_owner=NULL, lease_until=NULL, updated_at=? "
                "WHERE state='leased' AND lease_owner LIKE 'controller:%'",
                (tx.now,),
            )
            n = cur.rowcount
        if n:
            log.warning(
                "released %d outbox item(s) leased by a previous controller that did not shut down cleanly; "
                "they will be re-run (handlers are crash-safe)",
                n,
            )

    def _bind_server(self) -> None:
        sock_path = self.sock_path
        # (Re)create the socket directory under the lock: a predecessor removes a /tmp fallback
        # directory when it stops, possibly after our paths.ensure().
        with contextlib.suppress(FileExistsError):
            os.mkdir(sock_path.parent, 0o700)
        check_private_dir(sock_path.parent)
        try:
            st = os.lstat(sock_path)
        except FileNotFoundError:
            st = None
        if st is not None:
            if stat.S_ISDIR(st.st_mode):
                raise HpcLabError(
                    f"{sock_path} is a directory; remove it so the controller can bind its socket",
                    code=ErrorCode.PRECONDITION_FAILED,
                )
            # We hold the instance lock, so no live controller can own this file.
            log.warning("removing stale controller socket %s", sock_path)
            sock_path.unlink()
        server = _Server(str(sock_path), self)
        try:
            old_umask = os.umask(0o177)
            try:
                server.server_bind()
            finally:
                os.umask(old_umask)
            os.chmod(sock_path, 0o600)
            st_sock = os.lstat(sock_path)
            self._socket_id = (st_sock.st_dev, st_sock.st_ino)
            server.server_activate()
        except BaseException:
            server.server_close()
            with contextlib.suppress(FileNotFoundError):
                sock_path.unlink()
            raise
        self._server = server

    def _start_threads(self) -> None:
        assert self._server is not None
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="controller-server",
            daemon=True,
        )
        for i in range(self.worker_threads):
            ws = _WorkerState(
                name=f"worker-{i}", worker_id=f"controller:{self.pid}:{self.instance_token}:w{i}"
            )
            runner = OutboxRunner(
                self.ctx,
                handlers=self._wrap_handlers(ws),
                worker_id=ws.worker_id,
                lease_s=self.lease_s,
                # long transfers never occupy every worker: cancels and submissions get one
                max_bulk_workers=self.worker_threads - 1 if self.worker_threads > 1 else None,
            )
            ws.thread = threading.Thread(
                target=self._worker_loop, args=(ws, runner), name=ws.name, daemon=True
            )
            self._workers.append(ws)
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="controller-scheduler", daemon=True
        )
        self._server_thread.start()
        for ws in self._workers:
            assert ws.thread is not None
            self._enter_live()
            try:
                ws.thread.start()
            except BaseException:
                self._leave_live()
                raise
        self._scheduler_thread.start()

    def _abort_start(self) -> None:
        self._stop.set()
        self._sched_wake.set()
        self._kick_workers()
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
            self._server.server_close()
            self._unlink_socket()
        t_end = time.monotonic() + 2.0
        for t in self._background_threads():
            t.join(max(0.0, t_end - time.monotonic()))
        self._finalize_or_defer()
        with self._state_lock:
            self._state = "stopped"
        self._stopped.set()

    # -- logging -------------------------------------------------------------------------
    def _setup_logging(self) -> None:
        if not self.log_to_file:
            return
        self.paths.log.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            self.paths.log, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
        )
        handler.setFormatter(log_formatter())
        handler.setLevel(self.log_level)
        with contextlib.suppress(OSError):
            os.chmod(self.paths.log, 0o600)
        root = logging.getLogger("hpclab")
        root.addHandler(handler)
        self._saved_log_level = root.level
        if root.level == logging.NOTSET or root.level > self.log_level:
            root.setLevel(self.log_level)
        self._log_handler = handler

    def _teardown_logging(self) -> None:
        handler, self._log_handler = self._log_handler, None
        if handler is None:
            return
        root = logging.getLogger("hpclab")
        root.removeHandler(handler)
        handler.close()
        if self._saved_log_level is not None:
            root.setLevel(self._saved_log_level)

    # -- request handling -------------------------------------------------------------------
    def handle_line(self, line: bytes) -> bytes | None:
        """Process one framed request line; returns the encoded response (None for notifications).

        Never raises: once a request may have run, the peer must get an answer — a connection
        dropped without one makes clients re-send (and re-run) the request."""
        try:
            req = proto.parse_request(proto.decode_message(line))
        except proto.ProtocolError as exc:
            self._stats.add("protocol_errors")
            log.info("protocol error: %s", exc.message)
            return self._encode_rpc_error(exc.request_id, exc.to_rpc_error())
        # controller.shutdown is still answered (idempotently) while stopping: its goal is met
        if self._stop.is_set() and req.method != "controller.shutdown":
            if req.notification:
                return None
            return self._encode_rpc_error(
                req.id,
                proto.RpcError(
                    proto.SHUTTING_DOWN,
                    "controller is shutting down; the request was not executed",
                    data={"pid": self.pid},
                ),
            )
        request_id = new_id("request")
        try:
            envelope = self.execute(req, request_id)
            if req.notification:
                return None
            return self._encode_result(req, request_id, envelope)
        except Exception:  # defensive: both helpers already handle every failure they expect
            log.exception("request %s (%s): failed to produce a reply", request_id, req.method)
            if req.notification:
                return None
            return self._encode_rpc_error(
                req.id,
                proto.RpcError(
                    proto.INTERNAL_ERROR,
                    f"internal controller error while answering {req.method}; see {self.paths.log}",
                    data={"request_id": request_id, "log": str(self.paths.log)},
                ),
            )

    def execute(self, req: proto.Request, request_id: str) -> dict[str, Any]:
        """Run one request and return its Envelope dict. Never raises."""
        t0 = time.monotonic()
        env: dict[str, Any]
        try:
            if req.method.startswith(BUILTIN_PREFIX):
                env = self._builtin(req, request_id)
            else:
                env = self._dispatch_app(req, request_id)
        except Exception:  # defensive: the helpers above already catch everything
            log.exception("request %s (%s) failed", request_id, req.method)
            env = self._internal_envelope(request_id, req.method)
        self._stats.add("total")
        elapsed = time.monotonic() - t0
        ok = bool(env.get("ok"))
        if not ok:
            self._stats.add("errors")
            err = env.get("error") or {}
            code = err.get("code") if isinstance(err, dict) else None
            if code == ErrorCode.INTERNAL:
                self._stats.add("internal_errors")
            log.info("request %s %s -> %s (%.3fs)", request_id, req.method, code, elapsed)
        elif elapsed >= SLOW_REQUEST_S:
            log.info("slow request %s %s took %.1fs", request_id, req.method, elapsed)
        else:
            log.debug("request %s %s ok (%.3fs)", request_id, req.method, elapsed)
        return env

    def _dispatch_app(self, req: proto.Request, request_id: str) -> dict[str, Any]:
        deadline = min(req.deadline_s or self.request_deadline_s, MAX_REQUEST_DEADLINE_S)
        t_end = time.monotonic() + deadline
        if not self._request_slots.acquire(timeout=deadline):
            self._stats.add("busy_rejected")
            return proto.error_envelope(
                request_id,
                ErrorCode.UNAVAILABLE,
                f"controller busy: {self.max_concurrent_requests} requests already running; "
                f"{req.method} was not started — retry shortly",
                retryable=True,
                details={"started": False, "request_id": request_id, "deadline_s": deadline},
            )
        if self._stop.is_set() or not self._enter_live():
            self._request_slots.release()
            return proto.error_envelope(
                request_id,
                ErrorCode.UNAVAILABLE,
                f"controller is shutting down; {req.method} was not started — retry against its successor",
                retryable=True,
                details={"started": False, "request_id": request_id},
            )
        box: dict[str, dict[str, Any]] = {}
        done = threading.Event()

        def run() -> None:
            self._stats.add("app_calls_running")
            try:
                box["env"] = self._call_app(req.method, req.params, request_id)
            finally:
                self._stats.add("app_calls_running", -1)
                self._request_slots.release()
                done.set()
                self._kick_workers()
                self._leave_live()

        try:
            threading.Thread(target=run, name=f"request:{req.method}", daemon=True).start()
        except RuntimeError:
            self._request_slots.release()
            self._leave_live()
            log.exception("cannot start a request thread")
            return proto.error_envelope(
                request_id,
                ErrorCode.UNAVAILABLE,
                "controller cannot start a request thread (resource exhaustion); retry shortly",
                retryable=True,
                details={"started": False, "request_id": request_id},
            )
        if not done.wait(max(0.0, t_end - time.monotonic())):
            self._stats.add("deadline_exceeded")
            log.warning("request %s %s exceeded its %.1fs deadline", request_id, req.method, deadline)
            return proto.error_envelope(
                request_id,
                ErrorCode.UNAVAILABLE,
                f"{req.method} did not finish within {deadline:g}s; it is still running in the controller "
                "and may complete — retry (with the same idempotency_key for mutations) to get its result",
                retryable=True,
                details={
                    "started": True,
                    "request_id": request_id,
                    "deadline_s": deadline,
                    "method": req.method,
                },
            )
        return box.get("env") or self._internal_envelope(request_id, req.method)

    def _call_app(self, method: str, params: dict[str, Any], request_id: str) -> dict[str, Any]:
        try:
            env = self.app.dispatch(method, params, request_id=request_id)
        except HpcLabError as exc:  # includes protocol.MethodNotFound
            return proto.envelope_from_exception(exc, request_id)
        except Exception:
            log.exception("request %s (%s) raised", request_id, method)
            return self._internal_envelope(request_id, method)
        except BaseException as exc:  # SystemExit & co. from service code: answer, and log it
            log.error("request %s (%s) raised %s", request_id, method, type(exc).__name__, exc_info=True)
            return self._internal_envelope(request_id, method, reason=f"{type(exc).__name__} raised")
        if not isinstance(env, dict):
            log.error("dispatch(%s) returned %s instead of an Envelope dict", method, type(env).__name__)
            return self._internal_envelope(
                request_id, method, reason="the method returned a malformed result"
            )
        return env

    def _internal_envelope(
        self, request_id: str, method: str, *, reason: str | None = None
    ) -> dict[str, Any]:
        what = reason or "unexpected error"
        return proto.error_envelope(
            request_id,
            ErrorCode.INTERNAL,
            f"internal error in {method} ({what}); details are in the controller log {self.paths.log} "
            f"(request {request_id})",
            retryable=False,
            details={"request_id": request_id, "method": method, "log": str(self.paths.log)},
        )

    def _encode_result(self, req: proto.Request, request_id: str, env: dict[str, Any]) -> bytes:
        """Encode the reply; an unsendable result is replaced by an Envelope error (and counted
        as one, since ``execute`` only saw the app's own verdict)."""
        was_ok = bool(env.get("ok"))
        try:
            return proto.encode_message(
                proto.Response(id=req.id, result=env).to_dict(), max_bytes=self.max_message_bytes
            )
        except proto.MessageTooLarge as exc:
            log.error("result of %s (%s) is %d bytes; refusing to send", req.method, request_id, exc.size)
            env = proto.error_envelope(
                request_id,
                ErrorCode.LIMIT_EXCEEDED,
                f"the result of {req.method} is {exc.size} bytes, above the {exc.max_bytes}-byte protocol "
                "limit; request a smaller page (limit / max_bytes)",
                details={"size": exc.size, "max_bytes": exc.max_bytes, "request_id": request_id},
            )
        except Exception:  # TypeError / ValueError (incl. nesting too deep) / MemoryError ...
            log.exception("result of %s (%s) is not JSON-serializable", req.method, request_id)
            env = self._internal_envelope(request_id, req.method, reason="the result could not be serialized")
            self._stats.add("internal_errors")
        if was_ok:
            self._stats.add("errors")
        return proto.encode_message(proto.Response(id=req.id, result=env).to_dict())

    def _encode_rpc_error(self, rpc_id: proto.JsonRpcId, err: proto.RpcError) -> bytes:
        return proto.encode_message(proto.Response(id=rpc_id, error=err).to_dict())

    # -- built-in methods ------------------------------------------------------------------------
    def _builtin(self, req: proto.Request, request_id: str) -> dict[str, Any]:
        try:
            if req.method == "controller.ping":
                return proto.ok_envelope(request_id, self.ping_data(), summary="pong")
            if req.method == "controller.status":
                data = self.status()
                return proto.ok_envelope(request_id, data, summary=_status_summary(data))
            if req.method == "controller.wake":
                return proto.ok_envelope(request_id, self._wake(req.params), summary="woke background work")
            if req.method == "controller.shutdown":
                self.request_stop(f"controller.shutdown request {request_id}")
                return proto.ok_envelope(
                    request_id, {"stopping": True, "pid": self.pid}, summary=f"controller {self.pid} stopping"
                )
            raise proto.MethodNotFound(req.method, known=list(BUILTIN_METHODS))
        except HpcLabError as exc:
            return proto.envelope_from_exception(exc, request_id)
        except Exception:
            log.exception("built-in %s failed", req.method)
            return self._internal_envelope(request_id, req.method)

    def ping_data(self) -> dict[str, Any]:
        app = self._app
        return {
            "pong": True,
            "pid": self.pid,
            "version": hpclab.__version__,
            "protocol": proto.PROTOCOL_VERSION,
            "home": str(self.paths.home),
            "socket": str(self.sock_path),
            "started_at": self.started_at,
            "uptime_s": self._uptime(),
            "app": type(app).__qualname__ if app is not None else None,
            "app_fallback": isinstance(app, MinimalApp),
        }

    def _wake(self, params: dict[str, Any]) -> dict[str, Any]:
        tasks = params.get("tasks")
        if tasks is not None and (not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks)):
            raise InvalidArgument("tasks must be a list of periodic task names")
        try:
            forced = self.scheduler.force_due(tasks)
        except KeyError as exc:
            raise InvalidArgument(
                str(exc.args[0]) if exc.args else "unknown periodic task",
                details={"known": [t.name for t in self.scheduler.tasks]},
            ) from None
        self._sched_wake.set()
        self._kick_workers()
        return {"forced": forced}

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot; each ledger section degrades to ``{"error": ...}`` rather than
        failing the whole call."""
        app = self._app
        out: dict[str, Any] = {
            "running": True,
            "state": self._state,
            "pid": self.pid,
            "version": hpclab.__version__,
            "protocol": proto.PROTOCOL_VERSION,
            "started_at": self.started_at,
            "uptime_s": self._uptime(),
            "home": str(self.paths.home),
            "socket": str(self.sock_path),
            "log": str(self.paths.log),
            "db": str(self.paths.db),
            "app": {
                "class": type(app).__qualname__ if app is not None else None,
                "fallback": isinstance(app, MinimalApp),
                "load_error": getattr(app, "load_error", None),
                "outbox_kinds": sorted(self._handlers),
            },
            "requests": self._stats.snapshot(),
            "connections": len(self._conns),
            "workers": {
                "expected": self.worker_threads,
                "alive": sum(1 for w in self._workers if w.thread and w.thread.is_alive()),
                "threads": [w.snapshot() for w in self._workers],
            },
            "scheduler": {
                "alive": bool(self._scheduler_thread and self._scheduler_thread.is_alive()),
                "tasks": self._scheduler.stats() if self._scheduler else {},
            },
        }
        out["outbox"] = self._safe(self._outbox_status)
        out["clusters"] = self._safe(self._cluster_status)
        out["attention"] = _attention(out)
        return out

    def _safe(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a diagnostic call must never fail as a whole
            log.warning("status section %s failed: %s", fn.__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _outbox_status(self) -> dict[str, Any]:
        db = self.ctx.db
        now = iso(self.ctx.clock.now())
        by_state = {
            r["state"]: r["n"] for r in db.all("SELECT state, COUNT(*) AS n FROM outbox GROUP BY state")
        }
        due = db.one(
            "SELECT COUNT(*) AS n, MIN(not_before) AS oldest FROM outbox WHERE state='pending' AND not_before <= ?",
            (now,),
        ) or {"n": 0, "oldest": None}
        active = db.all(
            "SELECT kind, state, COUNT(*) AS n FROM outbox WHERE state IN ('pending','leased') "
            "GROUP BY kind, state ORDER BY kind, state"
        )
        dead = db.all(
            "SELECT outbox_id, kind, target_id, tries, last_error, updated_at FROM outbox "
            "WHERE state='dead' ORDER BY updated_at DESC, outbox_id DESC LIMIT 5"
        )
        return {
            "by_state": {s: int(by_state.get(s, 0)) for s in ("pending", "leased", "done", "dead")},
            "due_now": int(due["n"] or 0),
            "oldest_due_since": due["oldest"],
            "active_by_kind": [{"kind": r["kind"], "state": r["state"], "count": r["n"]} for r in active],
            "recent_dead": dead,
        }

    def _cluster_status(self) -> list[dict[str, Any]]:
        db = self.ctx.db
        clusters = [r["cluster_id"] for r in db.all("SELECT cluster_id FROM clusters ORDER BY cluster_id")]
        polls: dict[str, dict[str, Any]] = {c: {} for c in clusters}
        for r in db.all("SELECT * FROM poll_state ORDER BY cluster_id, source"):
            polls.setdefault(r["cluster_id"], {})[r["source"]] = {
                "last_polled_at": r["last_polled_at"],
                "last_ok_at": r["last_ok_at"],
                "consecutive_failures": r["consecutive_failures"],
                "backoff_until": r["backoff_until"],
                "last_error": r["last_error"],
            }
        return [{"cluster_id": c, "poll": polls[c]} for c in sorted(polls)]

    def _uptime(self) -> float | None:
        if self._started_mono is None:
            return None
        return round(time.monotonic() - self._started_mono, 3)

    # -- outbox workers --------------------------------------------------------------------------
    def _wrap_handlers(self, ws: _WorkerState) -> dict[str, Handler]:
        def wrap(kind: str, handler: Handler) -> Handler:
            def run(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
                ws.current, ws.current_since = item, time.monotonic()
                ws.lease_renewed_at = ctx.clock.now()  # OutboxRunner._lease just set lease_until
                try:
                    return handler(ctx, item)
                except Exception:
                    raise
                except BaseException as exc:
                    # SystemExit & co. would silently end the worker thread (and leave the item
                    # leased); turn them into an ordinary failure the runner retries / marks dead.
                    raise RuntimeError(f"outbox handler {kind} raised {type(exc).__name__}: {exc}") from exc
                finally:
                    ws.current = ws.current_since = ws.lease_renewed_at = None
                    ws.last_item_at = iso(ctx.clock.now())

            return run

        return {kind: wrap(kind, h) for kind, h in self._handlers.items()}

    def _worker_loop(self, ws: _WorkerState, runner: OutboxRunner) -> None:
        try:
            failures = 0
            while not self._stop.is_set():
                gen = self._wake_gen
                try:
                    # A sibling's lease looks expired after the wall clock jumped (laptop sleep):
                    # extend it *before* leasing, or run_once would hand us that same item again.
                    self._renew_busy_leases(force=False)
                    n = runner.run_once(limit=1)
                except Exception:
                    failures += 1
                    ws.errors += 1
                    log.exception("%s: outbox poll failed (%d in a row); backing off", ws.name, failures)
                    self._stop.wait(min(30.0, 0.25 * 2 ** min(failures, 7)))
                    continue
                failures = 0
                if n:
                    ws.processed += n
                    continue
                with self._wake_cond:
                    if self._wake_gen == gen and not self._stop.is_set():
                        self._wake_cond.wait(self.tick_interval_s)
        finally:
            self._leave_live()

    def _kick_workers(self) -> None:
        with self._wake_cond:
            self._wake_gen += 1
            self._wake_cond.notify_all()

    def _renew_leases(self, ctx: ServiceContext) -> None:
        """Built-in periodic task: extend the leases of every item being executed right now."""
        self._renew_busy_leases(force=True)

    def _renew_busy_leases(self, *, force: bool) -> int:
        """Extend (to now + lease_s) the leases of items this controller's workers are running.

        Without ``force`` only leases that look stale are touched: last extended ``lease_s/3``
        or more ago by ``ctx.clock`` — or "in the future" (the clock went back). That is cheap
        enough to run before every lease attempt, which is what closes the sleep/resume hole:
        after a long suspend every lease_until is in the past, and whichever worker wakes first
        renews its busy siblings before it could lease one of their items as "expired"."""
        ctx = self._ctx
        if ctx is None:
            return 0
        now = ctx.clock.now()
        stale_after = timedelta(seconds=self.lease_s / 3.0)
        busy: list[tuple[_WorkerState, int]] = []
        for ws in self._workers:
            item, renewed = ws.current, ws.lease_renewed_at
            if item is None:
                continue
            if force or renewed is None or now - renewed >= stale_after or now < renewed:
                busy.append((ws, item.outbox_id))
        if not busy:
            return 0
        until = iso(now + timedelta(seconds=self.lease_s))
        with ctx.db.tx(source="worker") as tx:
            for ws, outbox_id in busy:
                tx.execute(
                    "UPDATE outbox SET lease_until=? WHERE outbox_id=? AND state='leased' AND lease_owner=?",
                    (until, outbox_id, ws.worker_id),
                )
        for ws, outbox_id in busy:
            item = ws.current
            if item is not None and item.outbox_id == outbox_id:
                ws.lease_renewed_at = now
        return len(busy)

    def _socket_watch(self, ctx: ServiceContext) -> None:
        """Built-in periodic task. If our socket file was deleted or replaced (age-based /tmp
        cleaners on macOS and systemd, a user tidying up), no client can reach us while we still
        hold the lock — and autostart never spawns while the lock is held. Stop cleanly so the
        next client autostarts a fresh controller. Otherwise refresh the file's timestamps so
        age-based cleaners leave it alone."""
        path = self.sock_path
        try:
            st = os.lstat(path)
            ident: tuple[int, int] | None = (st.st_dev, st.st_ino)
        except FileNotFoundError:
            ident = None
        if ident is None or ident != self._socket_id:
            log.error(
                "controller socket %s was %s; stopping so the next client starts a reachable controller",
                path,
                "removed" if ident is None else "replaced",
            )
            self.request_stop(f"socket file {path} {'removed' if ident is None else 'replaced'}")
            return
        with contextlib.suppress(OSError):
            os.utime(path)

    def _housekeeping(self, ctx: ServiceContext) -> None:
        cutoff = iso(ctx.clock.now() - timedelta(seconds=self.outbox_retention_s))
        with ctx.db.tx(source="worker") as tx:
            cur = tx.execute(
                "DELETE FROM outbox WHERE outbox_id IN (SELECT outbox_id FROM outbox "
                "WHERE state='done' AND updated_at < ? LIMIT ?)",
                (cutoff, _HOUSEKEEPING_BATCH),
            )
            n = cur.rowcount
        if n:
            log.info("housekeeping: pruned %d finished outbox item(s) older than %s", n, cutoff)

    # -- periodic scheduler ------------------------------------------------------------------------
    def _scheduler_loop(self) -> None:
        sched = self.scheduler
        last_wall, last_mono = time.time(), time.monotonic()
        while not self._stop.is_set():
            wall, mono = time.time(), time.monotonic()
            gap = (wall - last_wall) - (mono - last_mono)
            last_wall, last_mono = wall, mono
            if gap > max(_SLEEP_GAP_S, 5 * self.tick_interval_s):
                log.info("wall clock jumped %.0fs ahead (sleep/resume?); running periodic tasks now", gap)
                sched.force_due()
            delay: float | None
            try:
                now = sched.clock.monotonic()
                for task in sched.due(now):
                    if self._stop.is_set():
                        break
                    sched.mark_started(task.name, now)
                    self._launch_periodic(task)
                delay = sched.next_due_in()
            except Exception:
                log.exception("periodic scheduler iteration failed")
                delay = None
            wait = self.tick_interval_s if delay is None else min(max(delay, 0.01), self.tick_interval_s)
            if self._sched_wake.wait(wait):
                self._sched_wake.clear()

    def _launch_periodic(self, task: PeriodicTask) -> None:
        if not self._enter_live():
            self.scheduler.mark_ran(task.name, ok=False, error="controller is shutting down")
            return
        t = threading.Thread(
            target=self._run_periodic, args=(task,), name=f"periodic:{task.name}", daemon=True
        )
        with self._periodic_lock:
            self._periodic_threads.add(t)
        try:
            t.start()
        except RuntimeError as exc:
            with self._periodic_lock:
                self._periodic_threads.discard(t)
            self._leave_live()
            log.error("cannot start periodic task %s: %s", task.name, exc)
            self.scheduler.mark_ran(task.name, ok=False, error=f"thread start failed: {exc}")

    def _run_periodic(self, task: PeriodicTask) -> None:
        sched = self.scheduler
        t0 = time.monotonic()
        ok = False
        error: str | None = None
        try:
            task.fn(self.ctx)
            ok = True
        except BaseException as exc:  # incl. SystemExit from service code: log it, keep scheduling
            error = f"{type(exc).__name__}: {exc}"
            if sched.last_error(task.name) != error:
                log.exception("periodic task %s failed", task.name)
            else:
                log.warning("periodic task %s failed again: %s", task.name, error)
        finally:
            sched.mark_ran(task.name, ok=ok, error=error, duration_s=time.monotonic() - t0)
            with self._periodic_lock:
                self._periodic_threads.discard(threading.current_thread())
            self._kick_workers()
            self._leave_live()

    # -- connections ---------------------------------------------------------------------------------
    def _register_conn(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.add(conn)

    def _unregister_conn(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.discard(conn)

    def _shutdown_conns(self, how: int) -> None:
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            with contextlib.suppress(OSError):
                c.shutdown(how)

    # -- shutdown ---------------------------------------------------------------------------------------
    def _shutdown(self, timeout: float | None = None) -> None:
        budget = self.shutdown_timeout_s if timeout is None else timeout
        with self._state_lock:
            state = self._state
            if state == "new":
                self._state = "stopped"
                self._stopped.set()
                return
            if state in ("stopping", "stopped", "starting"):
                wait_for_other = True
            else:
                self._state = "stopping"
                wait_for_other = False
        if wait_for_other:
            self._stopped.wait(budget + 2 * self._close_timeout_s + 5.0)
            return

        t_end = time.monotonic() + budget
        log.info("controller stopping (%s)", self._stop_reason or "requested")
        self._stop.set()
        self._sched_wake.set()
        self._kick_workers()

        # 1. stop accepting connections and remove the socket file
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
            self._server.server_close()
            self._unlink_socket()
        # 2. stop reading new requests; let running ones finish (bounded); then hang up
        self._shutdown_conns(socket.SHUT_RD)
        _wait_until(
            lambda: self._stats.in_flight == 0 and self._stats.app_calls_running == 0,
            min(t_end, time.monotonic() + 5.0),
        )
        self._shutdown_conns(socket.SHUT_RDWR)
        # 3. background threads
        for t in self._background_threads():
            if t is not threading.current_thread():
                t.join(max(0.0, t_end - time.monotonic()))
        busy = [
            w.snapshot()["current"] for w in self._workers if w.thread is not None and w.thread.is_alive()
        ]
        if busy:
            log.warning("shutdown timeout: outbox work still running %s", busy)
        # 4. close the app and context, remove the pidfile, release the lock — now, or when the
        #    last thread that may still write the ledger finishes
        self._finalize_or_defer()
        with self._state_lock:
            self._state = "stopped"
        self._stopped.set()

    def _background_threads(self) -> list[threading.Thread]:
        """Started worker, scheduler and periodic threads (never-started ones cannot be joined)."""
        threads = [w.thread for w in self._workers if w.thread is not None]
        if self._scheduler_thread is not None:
            threads.append(self._scheduler_thread)
        with self._periodic_lock:
            threads.extend(self._periodic_threads)
        return [t for t in threads if t.ident is not None]

    @property
    def finalize_pending(self) -> bool:
        """True after a stop whose shutdown timeout left threads running: the instance lock and
        pidfile stay held (no successor may start) until the last of them finishes."""
        with self._live_lock:
            return self._finalize_pending

    def _enter_live(self) -> bool:
        """Register a thread that may write the ledger. False once finalization has begun."""
        with self._live_lock:
            if self._finalizing:
                return False
            self._live += 1
            return True

    def _leave_live(self) -> None:
        run = False
        with self._live_lock:
            self._live -= 1
            if self._live == 0 and self._finalize_pending:
                self._finalize_pending = False
                self._finalizing = True
                run = True
        if run:
            log.info("the last lingering controller thread finished; releasing the instance")
            self._finalize()

    def _finalize_or_defer(self) -> None:
        with self._live_lock:
            if self._finalizing or self._finalize_pending:
                return  # already finalized, or already waiting for the last thread
            lingering = self._live
            if lingering == 0:
                self._finalizing = True
            else:
                self._finalize_pending = True
        if lingering == 0:
            self._finalize()
            return
        log.warning(
            "%d controller thread(s) still running after the shutdown timeout; keeping the instance lock "
            "and pidfile until they finish so no successor can run the same work concurrently",
            lingering,
        )

    def _finalize(self) -> None:
        """Close the app and service context (each bounded: a hung close must not keep the lock
        forever), remove the pidfile and release the instance lock — last."""
        if self._app is not None:
            _run_bounded(self._close_app, self._close_timeout_s, "app.close()")
        if self._ctx is not None:
            _run_bounded(self._close_clusters, self._close_timeout_s, "closing cluster transports")
            with contextlib.suppress(Exception):
                self._ctx.db.close()  # this thread's connection; others close with their threads
        if self._pidfile_written or self._lock.held:
            log.info("controller stopped (pid %s)", self.pid)
        self._release_instance()
        self._teardown_logging()

    def _close_clusters(self) -> None:
        if self._ctx is None:
            return
        try:
            self._ctx.clusters.close()
        except Exception:
            log.exception("closing cluster transports failed")

    def _close_app(self) -> None:
        close = getattr(self._app, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            log.exception("app.close() failed")

    def _unlink_socket(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            st = os.lstat(self.sock_path)
            if self._socket_id is None or (st.st_dev, st.st_ino) == self._socket_id:
                os.unlink(self.sock_path)  # never remove a file that is not our socket
        parent = self.sock_path.parent
        if parent != self.paths.home / "run":
            with contextlib.suppress(OSError):  # the per-home /tmp fallback dir; only if empty
                os.rmdir(parent)

    def _release_instance(self) -> None:
        if self._pidfile_written:
            remove_pidfile(self.paths.pidfile, self.pid)
            self._pidfile_written = False
        self._lock.release()


def _run_bounded(fn: Callable[[], None], timeout: float, what: str) -> bool:
    """Run ``fn`` on a daemon thread and wait at most ``timeout`` seconds for it."""
    done = threading.Event()

    def target() -> None:
        try:
            fn()
        finally:
            done.set()

    threading.Thread(target=target, name=f"controller-close:{what}", daemon=True).start()
    if done.wait(timeout):
        return True
    log.error("%s did not finish within %.1fs; continuing shutdown without waiting for it", what, timeout)
    return False


def _wait_until(pred: Callable[[], bool], t_end: float, poll_s: float = 0.02) -> bool:
    while not pred():
        if time.monotonic() >= t_end:
            return False
        time.sleep(poll_s)
    return True


def _attention(status: dict[str, Any]) -> list[str]:
    """Human-readable problems that need a person (or an agent) to act, most severe first."""
    notes: list[str] = []
    app = status.get("app") or {}
    if app.get("fallback"):
        notes.append(
            f"service layer not loaded ({app.get('load_error') or 'unknown error'}); only controller.* "
            f"methods work — fix it and restart the controller (log: {status.get('log')})"
        )
    workers = status.get("workers") or {}
    if workers.get("alive", 0) < workers.get("expected", 0):
        notes.append(
            f"only {workers.get('alive')}/{workers.get('expected')} outbox workers alive; restart the controller"
        )
    if not (status.get("scheduler") or {}).get("alive", True):
        notes.append("periodic scheduler thread is not running; restart the controller")
    for name, st in ((status.get("scheduler") or {}).get("tasks") or {}).items():
        if st.get("consecutive_failures", 0) >= 3:
            notes.append(
                f"periodic task {name} failed {st['consecutive_failures']} times in a row: {st.get('last_error')}"
            )
        running_for = st.get("running_for_s")
        interval = float(st.get("interval_s") or 0.0)
        if (
            st.get("running")
            and running_for is not None
            and running_for > max(_HUNG_TASK_MIN_S, _HUNG_TASK_INTERVALS * interval)
        ):
            notes.append(
                f"periodic task {name} has been running for {running_for:.0f}s (interval {interval:g}s) and "
                "may be hung (e.g. a remote call without a timeout); it will not run again until it returns "
                "— restart the controller if it does not"
            )
    for w in workers.get("threads") or []:
        cur = w.get("current") if isinstance(w, dict) else None
        running = (cur or {}).get("running_for_s")
        if cur and running is not None and running > _LONG_HANDLER_S:
            notes.append(
                f"{w.get('name')} has been running background step {cur.get('kind')} (outbox "
                f"{cur.get('outbox_id')}) for {running:.0f}s; check the controller log"
            )
    clusters = status.get("clusters")
    if isinstance(clusters, list):
        for c in clusters:
            for source, poll in (c.get("poll") or {}).items():
                if poll.get("consecutive_failures"):
                    notes.append(
                        f"cluster {c['cluster_id']}: {source} failing ({poll['consecutive_failures']}x, last ok "
                        f"{poll.get('last_ok_at') or 'never'}): {poll.get('last_error') or 'unknown error'}"
                    )
    outbox = status.get("outbox")
    if isinstance(outbox, dict) and "by_state" in outbox:
        dead = outbox["by_state"].get("dead", 0)
        if dead:
            kinds = sorted({str(r.get("kind")) for r in outbox.get("recent_dead") or []})
            notes.append(f"{dead} background step(s) gave up (dead), recent kinds: {', '.join(kinds)}")
    for section in ("outbox", "clusters"):
        value = status.get(section)
        if isinstance(value, dict) and "error" in value:
            notes.append(f"could not read {section} state: {value['error']}")
    return notes


def _status_summary(data: dict[str, Any]) -> str:
    outbox = data.get("outbox") or {}
    by_state = outbox.get("by_state") if isinstance(outbox, dict) else None
    parts = [f"controller pid {data.get('pid')} up {data.get('uptime_s') or 0:.0f}s"]
    if isinstance(by_state, dict):
        parts.append(f"outbox pending={by_state.get('pending', 0)} leased={by_state.get('leased', 0)}")
        if by_state.get("dead"):
            parts.append(f"dead={by_state['dead']}")
    workers = data.get("workers") or {}
    parts.append(f"workers {workers.get('alive', 0)}/{workers.get('expected', 0)}")
    if (data.get("app") or {}).get("fallback"):
        parts.append("service layer NOT loaded")
    if data.get("attention"):
        parts.append(f"{len(data['attention'])} item(s) need attention")
    return "; ".join(parts)
