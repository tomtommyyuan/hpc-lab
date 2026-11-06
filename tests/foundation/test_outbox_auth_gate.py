"""Background SSH auth gate: an expired login pauses background work per cluster.

When SSH needs interactive auth again (ControlMaster gone, Duo session expired) every outbox
item that talks to that cluster used to retry its own login every ``auth_retry_s`` - N items
meant N failed pre-auth connections every two minutes against a shared login node, for as long
as the user was away. The outbox now keeps one gate per cluster (``poll_state`` source
``outbox``): the first auth failure opens it, background calls to that cluster are refused
locally (no ssh) and exactly one probe is let through per ``auth_retry_s``. A probe that gets
through closes the gate and wakes the parked items; ``auth_resume`` (``hpclab auth``) clears it.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from hpclab.adapters.base import AuthRequiredError, CommandResult, QueryResult
from hpclab.config import Paths
from hpclab.domain.clock import FakeClock, iso
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, UserOverlay
from hpclab.services.clusters import auth_resume
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import HandlerResult, OutboxItem, OutboxRunner
from hpclab.testing.factories import make_experiment

AUTH_RETRY_S = 120.0


class FlakyAuthTransport:
    """Counts every would-be ssh connection; raises like BatchMode against an MFA login node."""

    def __init__(self, cluster_id: str) -> None:
        self.cluster_id = cluster_id
        self.auth_broken = False
        self.calls: list[list[str]] = []

    def _connect(self) -> None:
        if self.auth_broken:
            raise AuthRequiredError(
                f"{self.cluster_id}: SSH to login.{self.cluster_id} needs interactive authentication",
                details={
                    "cluster_id": self.cluster_id,
                    "reason": "auth",
                    "login_command": ["ssh", f"login.{self.cluster_id}"],
                },
            )

    def remote_user(self) -> str:
        return "alice"

    def login_command(self) -> list[str]:
        return ["ssh", f"login.{self.cluster_id}"]

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        self.calls.append(list(argv))
        self._connect()
        return CommandResult(list(argv), 0, "ok\n", "", 0.01)

    def close(self) -> None:
        pass


class FlakyAuthScheduler:
    def __init__(self, transport: FlakyAuthTransport) -> None:
        self.transport = transport
        self.cluster_id = transport.cluster_id

    def query_accounting(self, *, job_ids: list[str] | None = None, **_: Any) -> QueryResult:
        self.transport.run(["sacct", *(job_ids or [])])
        return QueryResult(source="sacct", observed_at="2026-01-01T00:00:00Z", jobs=[])


def _profile(cluster_id: str) -> ClusterProfile:
    return ClusterProfile(
        cluster_id=cluster_id,
        display_name=cluster_id,
        transport="local",
        partitions={"gpu": PartitionRule(max_time="1-00:00:00", default=True)},
        default_partition="gpu",
        policy=SitePolicy(),
    )


class Env:
    def __init__(self, paths: Paths, clusters: tuple[str, ...] = ("c1", "c2")) -> None:
        self.clock = FakeClock()
        self.transports = {c: FlakyAuthTransport(c) for c in clusters}
        self.schedulers = {c: FlakyAuthScheduler(t) for c, t in self.transports.items()}
        self.ctx = ServiceContext.create(
            paths,
            clock=self.clock,
            transport_factory=lambda rc, p: self.transports[rc.cluster_id],  # type: ignore[arg-type,return-value]
            scheduler_factory=lambda t, rc, k: self.schedulers[rc.cluster_id],  # type: ignore[arg-type,return-value]
        )
        self.experiments: dict[str, str] = {}
        for c in clusters:
            self.ctx.clusters.register(_profile(c), UserOverlay(cluster_id=c, user="alice", remote_root="/r"))
            with self.ctx.db.tx() as tx:
                self.experiments[c] = make_experiment(tx, cluster_id=c)
        self.runner = OutboxRunner(self.ctx, auth_retry_s=AUTH_RETRY_S)
        self.handled: list[int] = []
        self.runner.register("remote.step", self._remote_step)
        self.runner.register("sched.step", self._sched_step)
        self.runner.register("swallow.step", self._swallowing_step)
        self.runner.register("local.step", self._local_step)

    # -- handlers ------------------------------------------------------------------
    def _remote_step(self, ctx: ServiceContext, item: OutboxItem) -> None:
        cid = str(item.payload["cluster"])
        ctx.clusters.transport(cid).run(["cat", "status.json"])
        ctx.clusters.transport(cid).run(["cat", "env.json"])
        self.handled.append(item.outbox_id)

    def _sched_step(self, ctx: ServiceContext, item: OutboxItem) -> None:
        cid = str(item.payload["cluster"])
        ctx.clusters.scheduler(cid).query_accounting(job_ids=[str(item.outbox_id)])
        self.handled.append(item.outbox_id)

    def _swallowing_step(self, ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
        """Like artifacts.index / vault.capture: an unreadable cluster becomes coverage, not an error."""
        cid = str(item.payload["cluster"])
        try:
            ctx.clusters.transport(cid).run(["sha256sum", "out.bin"])
        except AuthRequiredError as exc:
            return HandlerResult.retry(AUTH_RETRY_S, f"cluster unreachable: {exc.message}")
        self.handled.append(item.outbox_id)
        return None

    def _local_step(self, ctx: ServiceContext, item: OutboxItem) -> None:
        self.handled.append(item.outbox_id)

    # -- helpers -------------------------------------------------------------------
    def enqueue(self, kind: str, cluster: str, n: int = 1) -> list[int]:
        ids: list[int] = []
        with self.ctx.db.tx() as tx:
            for _ in range(n):
                oid = tx.enqueue(kind, target_id=self.experiments[cluster], payload={"cluster": cluster})
                assert oid is not None
                ids.append(oid)
        return ids

    def run_for(self, seconds: float, tick: float = 5.0) -> None:
        """Drive the outbox like the controller would, with a fake clock."""
        for _ in range(int(seconds / tick)):
            while self.runner.run_once(limit=1000):
                pass
            self.clock.advance(tick)

    def drain_now(self) -> None:
        while self.runner.run_once(limit=1000):
            pass

    def item(self, outbox_id: int) -> dict[str, Any]:
        row = self.ctx.db.one("SELECT * FROM outbox WHERE outbox_id=?", (outbox_id,))
        assert row is not None
        return dict(row)

    def states(self, ids: list[int]) -> set[str]:
        return {self.item(i)["state"] for i in ids}

    def gate(self, cluster: str) -> dict[str, Any] | None:
        return self.ctx.db.one("SELECT * FROM poll_state WHERE cluster_id=? AND source='outbox'", (cluster,))


def test_expired_auth_does_not_multiply_ssh_attempts_by_backlog(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("remote.step", "c1", 40)
    env.transports["c1"].auth_broken = True

    env.run_for(600)  # ten minutes away from the keyboard

    attempts = len(env.transports["c1"].calls)
    # one failed login per auth_retry_s for the whole cluster, not one per item (40 x 5 = 200)
    assert attempts <= 600 / AUTH_RETRY_S + 1, attempts
    assert env.states(ids) == {"pending"}  # nothing is given up on
    errors = {r["last_error"] for r in env.ctx.db.all("SELECT last_error FROM outbox")}
    assert all(e and e.startswith("auth_required") for e in errors), errors
    gate = env.gate("c1")
    assert gate is not None and gate["consecutive_failures"] >= 1 and gate["backoff_until"]
    paused = env.ctx.db.all(
        "SELECT * FROM events WHERE entity_id='c1' AND event_type='cluster.background_auth_paused'"
    )
    assert len(paused) == 1  # on the transition only, not on every probe


def test_scheduler_calls_share_the_gate(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("sched.step", "c1", 30)
    env.transports["c1"].auth_broken = True
    env.run_for(600)
    assert len(env.transports["c1"].calls) <= 600 / AUTH_RETRY_S + 1
    assert env.states(ids) == {"pending"}


def test_handlers_that_swallow_auth_errors_still_close_the_tap(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("swallow.step", "c1", 30)
    env.transports["c1"].auth_broken = True
    env.run_for(600)
    assert len(env.transports["c1"].calls) <= 600 / AUTH_RETRY_S + 1
    assert env.states(ids) == {"pending"}
    # parked as waiting-for-login, so `hpclab auth` (auth_resume) wakes them
    notes = {r["last_error"] for r in env.ctx.db.all("SELECT last_error FROM outbox")}
    assert all(n and n.startswith("auth_required") for n in notes), notes


def test_successful_probe_resumes_every_parked_item_at_once(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("remote.step", "c1", 20)
    env.transports["c1"].auth_broken = True
    env.run_for(300)
    env.transports["c1"].auth_broken = False  # the user logged in again in some terminal

    env.run_for(AUTH_RETRY_S + 10)  # at most one probe interval later ...

    assert env.states(ids) == {"done"}  # ... all of them ran, not one per interval
    gate = env.gate("c1")
    assert gate is not None and gate["consecutive_failures"] == 0 and gate["backoff_until"] is None
    assert env.ctx.db.all(
        "SELECT * FROM events WHERE entity_id='c1' AND event_type='cluster.background_auth_resumed'"
    )


def test_auth_resume_clears_the_gate(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("remote.step", "c1", 10)
    env.transports["c1"].auth_broken = True
    env.run_for(60)
    assert env.states(ids) == {"pending"}
    env.transports["c1"].auth_broken = False

    data = auth_resume(env.ctx, cluster_id="c1")  # `hpclab auth c1` after a successful login
    env.drain_now()  # no clock advance: runs right away

    assert data.woken == len(ids)
    assert env.states(ids) == {"done"}


def test_gate_is_per_cluster_and_ignores_local_work(paths: Paths) -> None:
    env = Env(paths)
    broken = env.enqueue("remote.step", "c1", 5)
    env.transports["c1"].auth_broken = True
    env.drain_now()
    assert env.states(broken) == {"pending"}

    other = env.enqueue("remote.step", "c2", 5)
    local = env.enqueue("local.step", "c1", 5)  # belongs to c1 but never talks to it
    env.drain_now()

    assert env.states(other) == {"done"}
    assert env.states(local) == {"done"}
    assert len(env.transports["c2"].calls) == 2 * 5


def _race_refusal_against_probe(env: Env, handler_kind: str) -> tuple[list[int], int]:
    """Two workers when the gate reopens: w0 takes the probe (its ssh call is in flight) and w1
    is refused locally; the probe succeeds and closes the gate *while w1's item is still
    leased*, so the wake-up of the parked items cannot see it. Returns (ids, raced item)."""
    ids = env.enqueue(handler_kind, "c1", 3)
    transport = env.transports["c1"]
    transport.auth_broken = True
    env.drain_now()  # the first item trips the gate, the others are refused and parked
    assert env.states(ids) == {"pending"}
    transport.auth_broken = False  # the user logged in again
    env.clock.advance(AUTH_RETRY_S + 1)  # the gate reopens for one probe

    in_flight, release = threading.Event(), threading.Event()
    real_run = transport.run

    def blocking_run(argv: list[str], **kw: Any) -> CommandResult:
        if not in_flight.is_set():
            in_flight.set()
            assert release.wait(10)
        return real_run(argv, **kw)

    transport.run = blocking_run  # type: ignore[method-assign]
    w0 = OutboxRunner(env.ctx, worker_id="w0", auth_retry_s=AUTH_RETRY_S, handlers=dict(env.runner.handlers))
    probe = threading.Thread(target=w0.run_once, kwargs={"limit": 1})
    raced: list[int] = []

    def refused_step(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
        raced.append(item.outbox_id)
        try:
            ctx.clusters.transport("c1").run(["cat", "status.json"])
        except AuthRequiredError as exc:
            assert exc.details.get("gated")  # refused locally: w0 holds the probe
            release.set()
            probe.join(10)  # w0's probe got through and closed the gate meanwhile
            if handler_kind == "swallow.step":
                return HandlerResult.retry(AUTH_RETRY_S, f"cluster unreachable: {exc.message}")
            raise
        raise AssertionError("the gate should have refused this call")

    w1 = OutboxRunner(env.ctx, worker_id="w1", auth_retry_s=AUTH_RETRY_S)
    w1.register(handler_kind, refused_step)
    probe.start()
    assert in_flight.wait(10)
    assert w1.run_once(limit=1) == 1
    probe.join(10)
    assert not probe.is_alive()
    transport.run = real_run  # type: ignore[method-assign]
    gate = env.gate("c1")
    assert gate is not None and gate["consecutive_failures"] == 0  # the probe closed it
    return ids, raced[0]


@pytest.mark.parametrize("kind", ["remote.step", "swallow.step"])
def test_refusal_that_races_a_successful_probe_is_not_parked(paths: Paths, kind: str) -> None:
    env = Env(paths)
    ids, raced = _race_refusal_against_probe(env, kind)
    row = env.item(raced)
    assert row["state"] == "pending" and row["last_error"].startswith("auth_required")
    # parked until "the next probe" although the gate is already closed: 120 s for nothing
    assert row["not_before"] <= iso(env.clock.now()), row["not_before"]

    env.drain_now()  # no clock advance

    assert env.states(ids) == {"done"}


def test_auth_resume_racing_a_refusal_does_not_leave_it_parked(paths: Paths) -> None:
    env = Env(paths)
    ids = env.enqueue("remote.step", "c1", 2)
    env.transports["c1"].auth_broken = True
    env.drain_now()
    env.transports["c1"].auth_broken = False
    env.clock.advance(10)  # the gate is still closed to probes

    def resume_mid_run(ctx: ServiceContext, item: OutboxItem) -> None:
        try:
            ctx.clusters.transport("c1").run(["cat", "status.json"])
        except AuthRequiredError:
            auth_resume(env.ctx, cluster_id="c1")  # `hpclab auth c1` lands right now
            raise

    runner = OutboxRunner(env.ctx, worker_id="w1", auth_retry_s=AUTH_RETRY_S)
    runner.register("remote.step", resume_mid_run)
    with env.ctx.db.tx() as tx:  # make one parked item due, as if leased at the same moment
        tx.execute("UPDATE outbox SET not_before=? WHERE outbox_id=?", (iso(env.clock.now()), ids[1]))
    assert runner.run_once(limit=1) == 1

    env.drain_now()

    assert env.states(ids) == {"done"}


def test_refusal_waits_for_a_failed_probes_new_reopening(paths: Paths) -> None:
    """The park time is read when the item is parked: a probe that failed meanwhile pushed it."""
    env = Env(paths)
    ids = env.enqueue("remote.step", "c1", 2)
    env.transports["c1"].auth_broken = True
    env.drain_now()
    env.clock.advance(30)

    def trip_mid_run(ctx: ServiceContext, item: OutboxItem) -> None:
        try:
            ctx.clusters.transport("c1").run(["cat", "status.json"])
        except AuthRequiredError:
            env.clock.advance(AUTH_RETRY_S)  # meanwhile the gate reopened, and a probe failed
            env.runner._gate_trip("c1", AuthRequiredError("probe failed", details={"cluster_id": "c1"}))
            raise

    runner = OutboxRunner(env.ctx, worker_id="w1", auth_retry_s=AUTH_RETRY_S)
    runner.register("remote.step", trip_mid_run)
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE outbox SET not_before=? WHERE outbox_id=?", (iso(env.clock.now()), ids[1]))
    assert runner.run_once(limit=1) == 1

    row = env.item(ids[1])
    gate = env.gate("c1")
    assert gate is not None and row["not_before"] >= gate["backoff_until"]


def test_a_single_item_still_retries_on_its_own_schedule(paths: Paths) -> None:
    """With one item the gate adds nothing: one attempt per auth_retry_s, as before."""
    env = Env(paths)
    (oid,) = env.enqueue("remote.step", "c1")
    env.transports["c1"].auth_broken = True
    env.run_for(600)
    assert 4 <= len(env.transports["c1"].calls) <= 6
    env.transports["c1"].auth_broken = False
    env.run_for(AUTH_RETRY_S + 10)
    assert env.states([oid]) == {"done"}
