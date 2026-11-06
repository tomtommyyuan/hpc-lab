"""Durable outbox runner.

Work items are leased (``lease_until``); a crashed worker's lease expires and the item is
picked up again. Handlers therefore must be *crash-safe*: before any non-idempotent
external side effect (sbatch, scancel) a handler commits an intent state, and on re-entry
it inspects that state instead of repeating the effect (e.g. an attempt found in
``submission_in_flight`` is reconciled, never re-submitted).

Handler contract::

    def handler(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None

Return ``None``/``HandlerResult.done()`` when finished, ``HandlerResult.retry(delay)`` to run
again later (bounded by ``max_tries``), or raise. ``TransportError`` with
``retryable=True`` is retried with exponential backoff; other exceptions are retried a few
times and then the item goes ``dead`` and its operation (if any) is marked failed.

Lanes. Due items are leased oldest first, except that ``URGENT_KINDS`` (scancel, sbatch,
reconcile, workflow cancel and dispatch) go before everything else that is due. A runner with
``max_bulk_workers`` set (the controller's workers) leaves ``BULK_KINDS`` - transfers of up to
10 minutes, login-node hashing, output capture - to at most that many workers at a time, so a
cancel or a submission never waits behind a queue of long downloads.

Auth gate. SSH that needs a human again (expired ControlMaster, Duo) must not be retried by
every waiting item on its own: that is one failed login per item every ``auth_retry_s``
against a shared login node, for as long as the user is away. Handlers see ``ctx.clusters``
through a gate kept per cluster in ``poll_state`` (source ``outbox``). The first
``AuthRequiredError`` of a background call opens it; while it is open, background calls to
that cluster fail locally with ``AuthRequiredError`` (``details.gated``) without starting ssh,
and one caller per ``auth_retry_s`` is let through as the probe. A call that gets through
closes the gate and makes the items parked on ``auth_required`` due at once; ``auth_resume``
(``hpclab auth``) clears it as well. An item refused locally is parked until the gate reopens,
read in the transaction that parks it: if the gate closed while the item was still leased (the
wake-up cannot see leased items), it runs again at once. Requests made by the user are never
gated.
"""

from __future__ import annotations

import copy
import functools
import logging
import random
import threading
import traceback
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, cast

from hpclab.adapters.base import AuthRequiredError, CommandFailed
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import ErrorCode, HpcLabError
from hpclab.services.clusters import _item_clusters
from hpclab.services.context import ClusterRegistry, ServiceContext
from hpclab.storage.db import Row, Tx, jload

log = logging.getLogger("hpclab.outbox")

# Latency-critical steps: leased before anything else that is due. ``workflow.dispatch`` is a
# single ledger transaction (deduplicated per workflow) that admits the next cells into freed
# slots; it must not wait behind the finalize backlog of the wave that just ended.
URGENT_KINDS: frozenset[str] = frozenset(
    {"cancel.execute", "workflow.cancel", "submit.dispatch", "submit.reconcile", "workflow.dispatch"}
)
# Long-running, throughput-oriented steps: limited to ``max_bulk_workers`` concurrent workers.
BULK_KINDS: frozenset[str] = frozenset(
    {
        "artifact.download",
        "submit.stage",
        "artifacts.index",
        "vault.capture",
        "prepare.snapshot",
        "evaluation.collect",
        "evaluation.auto",
    }
)

GATE_SOURCE = "outbox"  # poll_state.source of the background SSH auth gate
_AUTH_PREFIX = "auth_required"  # last_error prefix of items waiting for a login (see auth_resume)
_HOST_KEY_MARK = "(host key changed)"
# Adapter methods that never talk to the cluster: not gated.
_LOCAL_METHODS = frozenset(
    {
        "login_command",
        "check_master",
        "stop_master",
        "close",
        "control_path",
        "remote_python_available",
        "adopt_probe",
    }
)
# Methods that may answer from a cache: gated, but their success proves nothing about the login.
_NO_EVIDENCE = frozenset({"remote_user", "probe_capabilities"})


@dataclass(frozen=True)
class OutboxItem:
    outbox_id: int
    kind: str
    operation_id: str | None
    target_id: str | None
    payload: dict[str, Any]
    tries: int


@dataclass(frozen=True)
class HandlerResult:
    action: str  # done | retry
    delay_s: float = 0.0
    note: str | None = None

    @staticmethod
    def done(note: str | None = None) -> HandlerResult:
        return HandlerResult("done", note=note)

    @staticmethod
    def retry(delay_s: float, note: str | None = None) -> HandlerResult:
        return HandlerResult("retry", delay_s=delay_s, note=note)


Handler = Callable[[ServiceContext, OutboxItem], "HandlerResult | None"]


def _gate_open(row: Row | None) -> bool:
    return row is not None and int(row["consecutive_failures"] or 0) > 0 and row["backoff_until"] is not None


@dataclass
class OutboxRunner:
    ctx: ServiceContext
    handlers: dict[str, Handler] = field(default_factory=dict)
    worker_id: str = "worker"
    lease_s: float = 300.0
    max_tries: int = 8  # unexpected crashes / non-retryable budget
    max_retryable_tries: int = 300  # retryable errors (cluster unreachable): ~a day at the backoff cap
    max_backoff_s: float = 300.0
    # auth_required: wait for the human, never give up; also the per-cluster probe interval
    auth_retry_s: float = 120.0
    # workers (distinct lease owners) that may run BULK_KINDS at once; None = no limit
    max_bulk_workers: int | None = None

    def register(self, kind: str, handler: Handler) -> None:
        if kind in self.handlers:
            raise ValueError(f"duplicate outbox handler for {kind!r}")
        self.handlers[kind] = handler

    def register_all(self, handlers: dict[str, Handler]) -> None:
        for k, h in handlers.items():
            self.register(k, h)

    # -- leasing -------------------------------------------------------------
    def _lease(self, limit: int) -> list[OutboxItem]:
        db = self.ctx.db
        now = iso(self.ctx.clock.now())
        until = iso(self.ctx.clock.now() + timedelta(seconds=self.lease_s))
        items: list[OutboxItem] = []
        with db.tx(source="worker") as tx:
            # expired leases go back to pending (crash recovery)
            tx.execute(
                "UPDATE outbox SET state='pending', lease_owner=NULL, lease_until=NULL "
                "WHERE state='leased' AND lease_until < ?",
                (now,),
            )
            kinds = self._leasable_kinds(tx, now)
            if not kinds:
                return []
            marks = ",".join("?" for _ in kinds)
            urgent = [k for k in kinds if k in URGENT_KINDS]
            order = "not_before, outbox_id"
            params: list[Any] = [now, *kinds]
            if urgent:
                order = f"CASE WHEN kind IN ({','.join('?' for _ in urgent)}) THEN 0 ELSE 1 END, {order}"
                params += urgent
            rows = tx.all(
                f"SELECT * FROM outbox WHERE state='pending' AND not_before <= ? AND kind IN ({marks}) "
                f"ORDER BY {order} LIMIT ?",
                (*params, limit),
            )
            for r in rows:
                tx.execute(
                    "UPDATE outbox SET state='leased', lease_owner=?, lease_until=?, tries=tries+1, updated_at=? "
                    "WHERE outbox_id=?",
                    (self.worker_id, until, tx.now, r["outbox_id"]),
                )
                items.append(
                    OutboxItem(
                        outbox_id=r["outbox_id"],
                        kind=r["kind"],
                        operation_id=r["operation_id"],
                        target_id=r["target_id"],
                        payload=jload(r["payload_json"], {}),
                        tries=r["tries"] + 1,
                    )
                )
        return items

    def _leasable_kinds(self, tx: Tx, now: str | None) -> list[str]:
        """Registered kinds, without the bulk ones while other workers already fill the bulk lane."""
        kinds = list(self.handlers)
        bulk = [k for k in kinds if k in BULK_KINDS]
        if self.max_bulk_workers is None or not bulk:
            return kinds
        busy = tx.one(
            "SELECT COUNT(DISTINCT lease_owner) AS n FROM outbox WHERE state='leased' AND lease_until >= ? "
            f"AND lease_owner <> ? AND kind IN ({','.join('?' for _ in bulk)})",
            (now, self.worker_id, *bulk),
        )
        if busy is not None and int(busy["n"] or 0) >= self.max_bulk_workers:
            return [k for k in kinds if k not in BULK_KINDS]
        return kinds

    def _finish(
        self,
        item: OutboxItem,
        state: str,
        *,
        delay_s: float = 0.0,
        error: str | None = None,
        gated: Collection[str] = (),
    ) -> None:
        """``gated``: clusters whose auth gate refused this run locally; the item is parked
        until they reopen (see ``_gated_delay``)."""
        with self.ctx.db.tx(source="worker") as tx:
            if state == "pending":
                if gated:
                    delay_s = self._gated_delay(tx, gated, delay_s)
                nb = iso(self.ctx.clock.now() + timedelta(seconds=delay_s))
                tx.execute(
                    "UPDATE outbox SET state='pending', lease_owner=NULL, lease_until=NULL, not_before=?, "
                    "last_error=?, updated_at=? WHERE outbox_id=?",
                    (nb, error, tx.now, item.outbox_id),
                )
            else:
                tx.execute(
                    "UPDATE outbox SET state=?, lease_owner=NULL, lease_until=NULL, last_error=?, updated_at=? "
                    "WHERE outbox_id=?",
                    (state, error, tx.now, item.outbox_id),
                )
                if state == "dead" and item.operation_id:
                    op = tx.one("SELECT state FROM operations WHERE operation_id=?", (item.operation_id,))
                    if op and op["state"] not in ("succeeded", "failed", "blocked"):
                        tx.set_operation(
                            item.operation_id,
                            state="failed",
                            error={
                                "code": "internal",
                                "message": f"background step {item.kind} failed: {error}",
                                "retryable": False,
                            },
                        )

    def _backoff(self, tries: int) -> float:
        base = min(self.max_backoff_s, 2.0 ** min(tries, 10))
        return base * (0.75 + random.random() * 0.5)  # noqa: S311 - jitter only

    # -- auth gate (see module docstring) ---------------------------------------
    def _gate_row(self, reader: Any, cluster_id: str) -> Row | None:
        row: Row | None = reader.one(
            "SELECT consecutive_failures, backoff_until, last_error FROM poll_state "
            "WHERE cluster_id=? AND source=?",
            (cluster_id, GATE_SOURCE),
        )
        return row

    def _gated_delay(self, tx: Tx, clusters: Collection[str], delay_s: float) -> float:
        """How long an item the gate refused locally waits, decided in the transaction that
        parks it. If one of the gates closed since the refusal (a probe got through, or
        ``auth_resume``) the item runs again at once: the wake-up only sees *pending* items and
        this one was still leased. Otherwise it waits at least until the gates reopen, as they
        stand now (a probe that failed meanwhile pushed that out)."""
        until: datetime | None = None
        for cluster_id in clusters:
            row = self._gate_row(tx, cluster_id)
            if row is None or not _gate_open(row):
                return 0.0
            reopens = parse_iso(row["backoff_until"])
            if reopens is not None and (until is None or reopens > until):
                until = reopens
        if until is None:
            return delay_s
        return max(delay_s, 1.0, (until - self.ctx.clock.now()).total_seconds())

    def _gate_admit(
        self, cluster_id: str
    ) -> tuple[Literal["pass", "probe", "wait"], datetime | None, str | None]:
        """``pass`` (gate closed), ``probe`` (this caller tests the login; the next probe is
        pushed ``auth_retry_s`` out) or ``wait`` (refuse locally until the returned time)."""
        if not _gate_open(self._gate_row(self.ctx.db, cluster_id)):
            return "pass", None, None  # the common case: a plain read, no write lock
        with self.ctx.db.tx(source="worker") as tx:
            row = self._gate_row(tx, cluster_id)
            if row is None or not _gate_open(row):
                return "pass", None, None
            now = self.ctx.clock.now()
            reopens = parse_iso(row["backoff_until"])
            if reopens is not None and reopens > now:
                return "wait", reopens, row["last_error"]
            tx.execute(
                "UPDATE poll_state SET backoff_until=?, last_polled_at=? WHERE cluster_id=? AND source=?",
                (iso(now + timedelta(seconds=self.auth_retry_s)), tx.now, cluster_id, GATE_SOURCE),
            )
        log.info("%s: probing whether background SSH works again", cluster_id)
        return "probe", None, None

    def _gate_trip(self, cluster_id: str, exc: HpcLabError) -> None:
        """A background call needed a login: (re)open the gate for ``auth_retry_s``."""
        now = self.ctx.clock.now()
        mark = f" {_HOST_KEY_MARK}" if exc.details.get("reason") == "host_key" else ""
        message = f"{exc.code}{mark}: {exc.message}"[:1000]
        with self.ctx.db.tx(source="worker") as tx:
            row = self._gate_row(tx, cluster_id)
            failures = int(row["consecutive_failures"] or 0) + 1 if row is not None else 1
            tx.execute(
                "INSERT INTO poll_state (cluster_id, source, last_polled_at, consecutive_failures, backoff_until,"
                " last_error) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(cluster_id, source) DO UPDATE SET"
                " last_polled_at=excluded.last_polled_at, consecutive_failures=excluded.consecutive_failures,"
                " backoff_until=excluded.backoff_until, last_error=excluded.last_error",
                (
                    cluster_id,
                    GATE_SOURCE,
                    tx.now,
                    failures,
                    iso(now + timedelta(seconds=self.auth_retry_s)),
                    message,
                ),
            )
            if failures == 1:
                tx.event(
                    "cluster",
                    cluster_id,
                    "cluster.background_auth_paused",
                    {
                        "reason": exc.details.get("reason"),
                        "message": exc.message[:300],
                        "probe_every_s": self.auth_retry_s,
                        "remediation": f"hpclab auth {cluster_id}",
                    },
                    source="worker",
                )
        if failures == 1:
            log.warning(
                "%s: SSH needs a login again; background work for it waits (one probe every %.0fs) "
                "until `hpclab auth %s`",
                cluster_id,
                self.auth_retry_s,
                cluster_id,
            )

    def _gate_close(self, cluster_id: str) -> None:
        """A background call got through: close the gate and wake the items parked on it."""
        with self.ctx.db.tx(source="worker") as tx:
            row = self._gate_row(tx, cluster_id)
            if row is None or not int(row["consecutive_failures"] or 0):
                return  # already cleared (auth_resume, or another worker's probe)
            tx.execute(
                "UPDATE poll_state SET consecutive_failures=0, backoff_until=NULL, last_error=NULL, last_ok_at=? "
                "WHERE cluster_id=? AND source=?",
                (tx.now, cluster_id, GATE_SOURCE),
            )
            woken = self._wake_parked(tx, cluster_id)
            tx.event(
                "cluster",
                cluster_id,
                "cluster.background_auth_resumed",
                {"after_failures": int(row["consecutive_failures"]), "woken": woken},
                source="worker",
            )
        log.info("%s: background SSH works again; %d waiting item(s) resumed", cluster_id, woken)

    def _wake_parked(self, tx: Tx, cluster_id: str) -> int:
        """Items parked on ``auth_required`` for this cluster (or for no known cluster) run now."""
        rows = tx.all(
            "SELECT outbox_id, kind, operation_id, target_id, payload_json FROM outbox "
            "WHERE state='pending' AND substr(last_error, 1, ?) = ? AND not_before > ? ORDER BY outbox_id",
            (len(_AUTH_PREFIX), _AUTH_PREFIX, tx.now),
        )
        woken = 0
        for r in rows:
            clusters = _item_clusters(tx, r)
            if clusters and cluster_id not in clusters:
                continue
            tx.execute(
                "UPDATE outbox SET not_before=?, updated_at=? WHERE outbox_id=? AND state='pending'",
                (tx.now, tx.now, r["outbox_id"]),
            )
            woken += 1
        return woken

    # -- run -------------------------------------------------------------------
    def run_once(self, limit: int = 10) -> int:
        """Process up to ``limit`` due items. Returns how many were processed."""
        items = self._lease(limit)
        for item in items:
            self._run_item(item)
        return len(items)

    def drain(self, max_rounds: int = 100) -> int:
        """Run until nothing is due (tests / CLI one-shot mode)."""
        total = 0
        for _ in range(max_rounds):
            n = self.run_once()
            total += n
            if n == 0:
                break
        return total

    def _run_item(self, item: OutboxItem) -> None:
        handler = self.handlers.get(item.kind)
        if handler is None:
            self._finish(item, "dead", error=f"no handler for {item.kind}")
            return
        gate = _AuthGate(self)
        try:
            res = handler(gate.context(self.ctx), item)
        except HpcLabError as exc:
            if exc.code == ErrorCode.AUTH_REQUIRED:
                # MFA/Duo can be away for hours; the work must survive until the user re-auths.
                gate.saw_error(exc)
                log.info("outbox %s %s waiting for auth: %s", item.kind, item.target_id, exc.message)
                refused = gate.refused_only()
                self._finish(
                    item,
                    "pending",
                    # own login failed: the next try is one probe interval out, like the gate's
                    delay_s=0.0 if refused else self.auth_retry_s,
                    error=f"{_AUTH_PREFIX}: {exc.message}",
                    gated=refused,
                )
            elif exc.retryable and item.tries < self.max_retryable_tries:
                log.info("outbox %s %s retry: %s", item.kind, item.target_id, exc.message)
                self._finish(item, "pending", delay_s=self._backoff(item.tries), error=exc.message)
            else:
                log.warning("outbox %s %s dead: %s", item.kind, item.target_id, exc.message)
                self._finish(item, "dead", error=f"{exc.code}: {exc.message}")
            return
        except Exception as exc:  # noqa: BLE001 - worker must survive handler bugs
            log.error("outbox %s %s crashed:\n%s", item.kind, item.target_id, traceback.format_exc())
            if item.tries < min(3, self.max_tries):
                self._finish(item, "pending", delay_s=self._backoff(item.tries), error=repr(exc))
            else:
                self._finish(item, "dead", error=repr(exc))
            return
        if res is None or res.action == "done":
            self._finish(item, "done", error=None)
        elif item.tries >= 5000:
            # retry() is used for polling-style steps (e.g. reconcile windows); guard runaway loops
            self._finish(item, "dead", error="too many retries")
        else:
            note = res.note
            if gate.hit and not (note or "").startswith(_AUTH_PREFIX):
                # e.g. index/capture record an unreadable cluster as coverage and ask to retry:
                # still waiting for a login, so auth_resume and a successful probe wake it
                note = f"{_AUTH_PREFIX}: {note}" if note else _AUTH_PREFIX
            self._finish(item, "pending", delay_s=res.delay_s, error=note, gated=gate.refused_only())


class _AuthGate:
    """The auth gate as one handler run sees it (not shared between runs or workers)."""

    def __init__(self, runner: OutboxRunner) -> None:
        self.runner = runner
        self.tripped: dict[str, HpcLabError] = {}  # clusters whose login failed in this run
        self.blocked: dict[str, datetime] = {}  # clusters refused locally -> when the gate reopens
        self.probing: set[str] = set()  # clusters this run tests the login of
        self.passed: set[str] = set()  # clusters found open: no further checks in this run
        self._lock = threading.Lock()  # a handler may call adapters from several threads

    @property
    def hit(self) -> bool:
        return bool(self.tripped or self.blocked)

    def context(self, ctx: ServiceContext) -> ServiceContext:
        """A shallow copy of ``ctx`` whose cluster registry hands out gated adapters."""
        registry = getattr(ctx, "clusters", None)
        if registry is None:
            return ctx
        try:
            gated = copy.copy(ctx)
            gated.clusters = cast(ClusterRegistry, _GatedClusters(registry, self))
        except (AttributeError, TypeError):  # frozen / slotted test doubles: run ungated
            return ctx
        return gated

    def refused_only(self) -> tuple[str, ...]:
        """Clusters this run was refused on locally, unless a login attempt of its own failed
        (then it waits its own ``auth_retry_s`` like the gate does)."""
        with self._lock:
            return () if self.tripped else tuple(self.blocked)

    def before(self, cluster_id: str) -> None:
        with self._lock:
            first = self.tripped.get(cluster_id)
            if first is not None:
                raise _refusal(cluster_id, first.message, first.details)
            if cluster_id in self.passed or cluster_id in self.probing:
                return
        verdict, reopens, last_error = self.runner._gate_admit(cluster_id)
        with self._lock:
            if verdict == "pass":
                self.passed.add(cluster_id)
                return
            if verdict == "probe":
                self.probing.add(cluster_id)
                return
            at = reopens or self.runner.ctx.clock.now() + timedelta(seconds=self.runner.auth_retry_s)
            self.blocked[cluster_id] = at
        reason = "host_key" if last_error and _HOST_KEY_MARK in last_error else "auth"
        raise _refusal(
            cluster_id,
            f"{cluster_id}: SSH needs a login again; background work for {cluster_id} waits "
            f"(next automatic check at {iso(at)}): run `hpclab auth {cluster_id}`",
            {"reason": reason, "retry_at": iso(at), "last_error": last_error},
        )

    def succeeded(self, cluster_id: str, method: str) -> None:
        if method in _NO_EVIDENCE:
            return
        with self._lock:
            if cluster_id not in self.probing:
                return
            self.probing.discard(cluster_id)
            self.passed.add(cluster_id)
        self.runner._gate_close(cluster_id)

    def failed(self, cluster_id: str, exc: HpcLabError) -> None:
        with self._lock:
            if cluster_id in self.tripped:
                return
            self.tripped[cluster_id] = exc
            self.probing.discard(cluster_id)
            self.passed.discard(cluster_id)
        self.runner._gate_trip(cluster_id, exc)

    def saw_error(self, exc: HpcLabError) -> None:
        """An auth error a handler raised: an SSH login that failed outside the gated
        adapters (e.g. a transport built by hand) opens the gate too."""
        if not isinstance(exc, AuthRequiredError) or exc.details.get("gated"):
            return
        cluster_id = exc.details.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id and cluster_id not in self.blocked:
            self.failed(cluster_id, exc)


def _refusal(cluster_id: str, message: str, details: dict[str, Any]) -> AuthRequiredError:
    return AuthRequiredError(
        message,
        details={
            **details,
            "cluster_id": cluster_id,
            "gated": True,
            "maybe_executed": False,
            "hint": f"run `hpclab auth {cluster_id}` in a terminal and complete the login prompt",
        },
    )


class _GatedClusters:
    """``ctx.clusters`` for one background handler run: adapters go through the auth gate."""

    def __init__(self, inner: ClusterRegistry, gate: _AuthGate) -> None:
        self._inner = inner
        self._gate = gate

    def transport(self, cluster_id: str) -> Any:
        return _GatedAdapter(self._inner.transport(cluster_id), cluster_id, self._gate)

    def scheduler(self, cluster_id: str) -> Any:
        return _GatedAdapter(self._inner.scheduler(cluster_id), cluster_id, self._gate)

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:  # during copy/pickle, before __init__ ran: no recursion
            raise AttributeError(name)
        return getattr(inner, name)


class _GatedAdapter:
    """Transport / scheduler proxy: every call that may reach the cluster asks the gate first,
    and reports back whether the login worked."""

    def __init__(self, inner: Any, cluster_id: str, gate: _AuthGate) -> None:
        self._inner = inner
        self._cluster_id = cluster_id
        self._gate = gate

    def __repr__(self) -> str:
        return f"<gated {self.__dict__.get('_inner')!r}>"

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:  # during copy/pickle, before __init__ ran: no recursion
            raise AttributeError(name)
        attr = getattr(inner, name)
        if name.startswith("_") or name in _LOCAL_METHODS or not callable(attr) or isinstance(attr, type):
            return attr
        gate: _AuthGate = self.__dict__["_gate"]
        cluster_id: str = self.__dict__["_cluster_id"]

        @functools.wraps(attr)
        def call(*args: Any, **kwargs: Any) -> Any:
            gate.before(cluster_id)
            try:
                out = attr(*args, **kwargs)
            except AuthRequiredError as exc:
                gate.failed(cluster_id, exc)
                raise
            except (FileNotFoundError, CommandFailed):
                gate.succeeded(cluster_id, name)  # the remote side answered: the login works
                raise
            gate.succeeded(cluster_id, name)
            return out

        return call
