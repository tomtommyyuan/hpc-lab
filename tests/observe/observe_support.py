"""Fakes and ledger helpers for observe/runs unit tests (imported by name; pytest prepend mode).

``make_env`` builds a real ``ServiceContext`` (SQLite ledger, FakeClock) whose cluster registry
hands out a scripted ``FakeScheduler`` and an in-memory ``FakeTransport``. Ledger rows come from
``hpclab.testing.factories``.
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpclab.adapters.base import (
    CancelResponse,
    GpuSample,
    QueryResult,
    RangeRead,
    SchedulerCapabilities,
    StatInfo,
)
from hpclab.config import get_paths
from hpclab.domain.clock import FakeClock, iso
from hpclab.domain.errors import HpcLabError
from hpclab.domain.models import JobObservation
from hpclab.domain.states import SchedulerState
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, UserOverlay
from hpclab.services.context import ServiceContext, Settings
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_attempt, make_exec_plan, make_experiment, make_plan, make_run

CLUSTER = "c1"
USER = "alice"


class FakeTransport:
    """In-memory remote filesystem with inode generations and injectable failures."""

    def __init__(self) -> None:
        self.cluster_id = CLUSTER
        self.files: dict[str, bytes] = {}
        self.inodes: dict[str, int] = {}
        self.mtimes: dict[str, float] = {}
        self.fail: dict[str, Exception] = {}  # method name -> exception to raise
        self.calls: list[tuple[str, str]] = []
        self._next_inode = 100

    # -- test helpers ---------------------------------------------------------
    def put(
        self, path: str, data: bytes | str, *, mtime: float | None = None, new_inode: bool = False
    ) -> None:
        raw = data.encode() if isinstance(data, str) else data
        if path not in self.inodes or new_inode:
            self._next_inode += 1
            self.inodes[path] = self._next_inode
        self.files[path] = raw
        self.mtimes[path] = mtime if mtime is not None else 1_788_000_000.0

    def append(self, path: str, data: bytes | str) -> None:
        raw = data.encode() if isinstance(data, str) else data
        self.put(path, self.files.get(path, b"") + raw, mtime=self.mtimes.get(path))

    def _check(self, method: str, path: str = "") -> None:
        self.calls.append((method, path))
        exc = self.fail.get(method)
        if exc is not None:
            raise exc

    # -- Transport protocol ---------------------------------------------------
    def remote_user(self) -> str:
        return USER

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._check("read_range", remote_path)
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        data = self.files[remote_path]
        off = max(0, min(offset, len(data)))
        chunk = data[off : off + max(0, length)]
        return RangeRead(
            path=remote_path,
            offset=off,
            data=chunk,
            size=len(data),
            mtime=self.mtimes[remote_path],
            inode=self.inodes[remote_path],
            eof=off + len(chunk) >= len(data),
        )

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._check("read_tail", remote_path)
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        size = len(self.files[remote_path])
        self.calls.pop()
        return self.read_range(remote_path, max(0, size - length), length)

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        self._check("stat", ",".join(remote_paths))
        out: dict[str, StatInfo] = {}
        for p in remote_paths:
            if p in self.files:
                out[p] = StatInfo(
                    path=p,
                    exists=True,
                    kind="file",
                    size=len(self.files[p]),
                    mtime=self.mtimes[p],
                    inode=self.inodes[p],
                )
            else:
                out[p] = StatInfo(path=p, exists=False, kind="missing")
        return out

    def close(self) -> None:
        pass


class FakeScheduler:
    """Scripted squeue/sacct. ``queue``/``acct`` are the jobs the next calls return."""

    def __init__(self) -> None:
        self.cluster_id = CLUSTER
        self.queue: list[JobObservation] = []
        self.acct: list[JobObservation] = []
        # ``sacct --duplicates`` records (every requeue segment, oldest first) per job id
        self.segments: dict[str, list[JobObservation]] = {}
        self.queue_error: Exception | None = None
        self.acct_error: Exception | None = None
        self.segments_error: Exception | None = None
        self.samples: list[GpuSample] = []
        self.samples_error: Exception | None = None
        self.block: threading.Event | None = None  # query_queue waits on it when set
        self.calls: list[str] = []
        self.clock: FakeClock | None = None

    def _now(self) -> str:
        assert self.clock is not None
        out = iso(self.clock.now())
        assert out is not None
        return out

    def probe_capabilities(self) -> SchedulerCapabilities:
        return SchedulerCapabilities(sacct_available=True, srun_overlap=True)

    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        self.calls.append(f"squeue:{','.join(job_ids) if job_ids else 'user'}")
        if self.block is not None:
            self.block.wait(10)
        if self.queue_error is not None:
            raise self.queue_error
        jobs = [j for j in self.queue if job_ids is None or j.job_id in job_ids]
        return QueryResult(source="squeue", observed_at=self._now(), jobs=jobs)

    def query_accounting(
        self,
        *,
        job_ids: list[str] | None = None,
        user: str | None = None,
        name: str | None = None,
        since: str | None = None,
    ) -> QueryResult:
        self.calls.append(f"sacct:{','.join(job_ids or [])}")
        if self.acct_error is not None:
            raise self.acct_error
        jobs = [j for j in self.acct if job_ids is None or j.job_id in job_ids]
        return QueryResult(source="sacct", observed_at=self._now(), jobs=jobs)

    def accounting_segments(self, job_id: str) -> list[JobObservation]:
        """Like ``SlurmScheduler.accounting_segments``: restart_count = position, oldest first."""
        self.calls.append(f"sacct-D:{job_id}")
        if self.segments_error is not None:
            raise self.segments_error
        segs = self.segments.get(job_id, [])
        return [
            s.model_copy(update={"restart_count": i, "observed_at": self._now()}) for i, s in enumerate(segs)
        ]

    def gpu_samples(self, job_id: str, *, timeout: float = 20.0) -> list[GpuSample]:
        self.calls.append(f"gpu:{job_id}")
        if self.samples_error is not None:
            raise self.samples_error
        return list(self.samples)

    def cancel(self, job_id: str) -> CancelResponse:  # pragma: no cover - never used by observe
        raise AssertionError("observe must never cancel")


@dataclass
class Env:
    ctx: ServiceContext
    clock: FakeClock
    transport: FakeTransport
    scheduler: FakeScheduler
    stubs: dict[str, Any] = field(default_factory=dict)

    def now(self) -> str:
        out = iso(self.clock.now())
        assert out is not None
        return out

    def attempt(self, attempt_id: str) -> dict[str, Any]:
        row = self.ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
        assert row is not None
        return row

    def run(self, run_id: str) -> dict[str, Any]:
        row = self.ctx.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        assert row is not None
        return row

    def events(self, entity_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE entity_id=?"
        params: list[Any] = [entity_id]
        if event_type:
            sql += " AND event_type=?"
            params.append(event_type)
        return self.ctx.db.all(sql + " ORDER BY seq", params)

    def row(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
        """One ledger row that must exist."""
        found = self.ctx.db.one(sql, params)
        assert found is not None, sql
        return found

    def poll_state(self, source: str = "squeue") -> dict[str, Any]:
        return self.row("SELECT * FROM poll_state WHERE cluster_id=? AND source=?", (CLUSTER, source))

    def outbox(self, kind: str) -> list[dict[str, Any]]:
        return self.ctx.db.all("SELECT * FROM outbox WHERE kind=? ORDER BY outbox_id", (kind,))


def profile(**policy: Any) -> ClusterProfile:
    return ClusterProfile(
        cluster_id=CLUSTER,
        display_name="Test cluster",
        transport="local",
        partitions={
            "gpu": PartitionRule(max_time="2-00:00:00", gpu_types=["a100"], default=True),
            "scavenge": PartitionRule(max_time="1-00:00:00", preemptible=True),
        },
        default_partition="gpu",
        policy=SitePolicy(
            **{"poll_min_interval_s": 30.0, "sacct_min_interval_s": 60.0, "accounting_lag_s": 120.0, **policy}
        ),
    )


def make_env(tmp_path: Path, *, settings: Settings | None = None, **policy: Any) -> Env:
    clock = FakeClock()
    transport = FakeTransport()
    scheduler = FakeScheduler()
    scheduler.clock = clock
    paths = get_paths(tmp_path / "hpclab-home").ensure()
    ctx = ServiceContext.create(
        paths,
        clock=clock,
        transport_factory=lambda c, p: transport,  # type: ignore[arg-type,return-value]
        scheduler_factory=lambda t, c, k: scheduler,  # type: ignore[arg-type,return-value]
        settings=settings or Settings(request_deadline_s=2.0),
    )
    ctx.clusters.register(
        profile(**policy), UserOverlay(cluster_id=CLUSTER, user=USER, remote_root="/remote")
    )
    return Env(ctx=ctx, clock=clock, transport=transport, scheduler=scheduler)


def seed(
    env: Env,
    *,
    phase: str = "observing",
    scheduler_state: str = "RUNNING",
    job_id: str | None = "1001",
    claim: bool = True,
    run_id: str | None = None,
    experiment_id: str | None = None,
    cell: dict[str, Any] | None = None,
    seed_value: int | None = 1,
    plan: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
    **attempt: Any,
) -> tuple[str, str]:
    """(run_id, attempt_id) of a fresh attempt; ``claim`` makes it hold the run's claim."""
    with env.ctx.db.tx() as tx:
        if run_id is None:
            eid = experiment_id or make_experiment(tx, cluster_id=CLUSTER)
            pid = make_plan(tx, eid)
            run_id = make_run(tx, eid, pid, cell=cell, seed=seed_value, spec=spec)
        eph = make_exec_plan(tx, run_id, plan=plan) if plan is not None else None
        attempt.setdefault("job_owner", USER if job_id else None)
        attempt.setdefault("gpus", 1)
        aid = make_attempt(
            tx,
            run_id,
            exec_plan_hash=eph,
            phase=phase,
            scheduler_state=scheduler_state,
            cluster_id=CLUSTER,
            job_id=job_id,
            **attempt,
        )
        if claim:
            tx.execute("UPDATE runs SET claim_attempt_id=? WHERE run_id=?", (aid, run_id))
    return run_id, aid


def observation(env: Env, attempt_id: str, state: SchedulerState | str, **kw: Any) -> JobObservation:
    a = env.attempt(attempt_id)
    fields: dict[str, Any] = {
        "job_id": a["job_id"],
        "source": "squeue",
        "observed_at": env.now(),
        "state": SchedulerState(state),
        "raw_state": str(SchedulerState(state).value),
        "name": a["job_name"],
        "user": USER,
        "partition": "gpu",
    }
    fields.update(kw)
    return JobObservation(**fields)


def install_stub(monkeypatch: Any, name: str, **attrs: Any) -> types.ModuleType:
    """Put a stub module in sys.modules (observe imports other areas lazily by name)."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@dataclass
class Recorder:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def stub_budget_and_notify(monkeypatch: Any) -> tuple[Recorder, Recorder]:
    settle, emit = Recorder(), Recorder()
    install_stub(monkeypatch, "hpclab.services.budget", settle=settle)
    install_stub(monkeypatch, "hpclab.services.notify", emit=emit)
    return settle, emit


def auth_error() -> HpcLabError:
    from hpclab.adapters.base import AuthRequiredError

    return AuthRequiredError("ssh needs MFA", details={"login_command": ["ssh", "x"]})


def put_json(env: Env, path: str, doc: dict[str, Any]) -> None:
    env.transport.put(path, jdump(doc))
