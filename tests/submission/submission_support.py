"""Fakes and ledger seeding for the submission tests (imported by name: pytest prepend mode).

``FakeTransport``/``FakeScheduler`` are scripted stand-ins for the adapters so every outcome
branch of the handlers can be driven deterministically with a ``FakeClock``. ``seed_run``
writes a *real* RunSpec + ExecutionPlan (valid for staging/rendering) through the factories.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpclab.adapters.base import (
    AuthRequiredError,
    CancelResponse,
    CommandResult,
    QueryResult,
    RangeRead,
    StatInfo,
    SubmitResponse,
)
from hpclab.domain.clock import Clock, FakeClock, iso
from hpclab.domain.errors import HpcLabError
from hpclab.domain.models import (
    CodeSnapshotRef,
    EntrypointSpec,
    ExecutionPlan,
    JobObservation,
    ResilienceSpec,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.domain.states import normalize_slurm_state
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxRunner
from hpclab.services.submission import outbox_handlers
from hpclab.testing.factories import make_exec_plan, make_experiment, make_plan, make_run

FAKE_DIGEST = "sha256:" + "ab" * 32


class SimulatedCrash(BaseException):
    """Escapes every ``except Exception`` like a killed controller would."""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTransport:
    def __init__(self, cluster_id: str, user: str = "alice") -> None:
        self.cluster_id = cluster_id
        self.user = user
        self.auth_required = False
        self.auth_reason = "auth"
        self.files: dict[str, bytes] = {}
        self.missing: set[str] = set()
        self.scripts: list[tuple[str, list[str]]] = []
        self.calls: list[str] = []

    def _gate(self, name: str) -> None:
        self.calls.append(name)
        if self.auth_required:
            raise AuthRequiredError(
                "simulated: ControlMaster expired",
                details={"reason": self.auth_reason, "login_command_str": "ssh -M unit"},
            )

    def remote_user(self) -> str:
        self._gate("remote_user")
        return self.user

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        self._gate("run")
        return CommandResult(list(argv), 0, "", "", 0.0)

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        self._gate("run_script")
        self.scripts.append((script, list(args or [])))
        return CommandResult(["bash", "-s"], 0, "", "", 0.0)

    def upload_dir(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        self._gate("upload_dir")

    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        self._gate("write_file")
        self.files[remote_path] = content
        self.missing.discard(remote_path)

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._gate("read_range")
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        data = self.files[remote_path][offset : offset + length]
        return RangeRead(remote_path, offset, data, len(self.files[remote_path]), None, 1, True)

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        return self.read_range(remote_path, 0, length)

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        self._gate("stat")
        return {
            p: StatInfo(p, p not in self.missing, "missing" if p in self.missing else "file")
            for p in remote_paths
        }

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        self._gate("sha256")
        return dict.fromkeys(remote_paths)

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        self._gate("download")

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        self._gate("list_dir")
        return []

    def close(self) -> None:
        return None


Behavior = SubmitResponse | BaseException | Callable[..., SubmitResponse] | str


@dataclass
class FakeScheduler:
    cluster_id: str
    clock: Clock
    submit_behaviors: list[Behavior] = field(default_factory=list)
    submit_calls: list[tuple[str, str, str | None]] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    receipt: SubmitResponse | BaseException | None = None
    receipt_calls: int = 0
    queue: list[JobObservation] = field(default_factory=list)
    queue_error: BaseException | None = None
    queue_calls: list[dict[str, Any]] = field(default_factory=list)
    acct: list[JobObservation] = field(default_factory=list)
    acct_error: BaseException | None = None
    acct_calls: list[dict[str, Any]] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)
    cancel_accepts: bool = True
    cancel_error: BaseException | None = None
    signal_calls: list[tuple[str, str, bool]] = field(default_factory=list)
    next_job_id: int = 5000

    def _now(self) -> str:
        now = iso(self.clock.now())
        assert now is not None
        return now

    def submit(
        self, script_path: str, *, workdir: str, test_only: bool = False, receipt_path: str | None = None
    ) -> SubmitResponse:
        self.submit_calls.append((script_path, workdir, receipt_path))
        behavior: Behavior = self.submit_behaviors.pop(0) if self.submit_behaviors else "accept"
        if isinstance(behavior, BaseException):
            raise behavior
        if isinstance(behavior, SubmitResponse):
            return behavior
        if callable(behavior):
            return behavior(script_path, workdir=workdir, receipt_path=receipt_path)
        jid = str(self.next_job_id)
        self.next_job_id += 1
        resp = SubmitResponse(job_id=jid, cluster_name=None, raw=jid, submitted_at=self._now())
        self.accepted.append(jid)
        self.receipt = resp
        return resp

    def read_receipt(self, receipt_path: str) -> SubmitResponse | None:
        self.receipt_calls += 1
        if isinstance(self.receipt, BaseException):
            raise self.receipt
        return self.receipt

    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        self.queue_calls.append({"job_ids": job_ids, "user": user})
        if self.queue_error is not None:
            raise self.queue_error
        jobs = [
            j
            for j in self.queue
            if (job_ids is None or j.job_id in job_ids) and (user is None or j.user == user)
        ]
        return QueryResult(source="squeue", observed_at=self._now(), jobs=jobs)

    def query_accounting(
        self,
        *,
        job_ids: list[str] | None = None,
        user: str | None = None,
        name: str | None = None,
        since: str | None = None,
    ) -> QueryResult:
        self.acct_calls.append({"job_ids": job_ids, "user": user, "name": name, "since": since})
        if self.acct_error is not None:
            raise self.acct_error
        jobs = [
            j
            for j in self.acct
            if (job_ids is None or j.job_id in job_ids)
            and (user is None or j.user == user)
            and (name is None or j.name == name)
        ]
        return QueryResult(source="sacct", observed_at=self._now(), jobs=jobs)

    def cancel(self, job_id: str) -> CancelResponse:
        self.cancel_calls.append(job_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        return CancelResponse(
            job_id=job_id, accepted=self.cancel_accepts, raw="" if self.cancel_accepts else "no"
        )

    def signal_job(self, job_id: str, signal: str, *, batch: bool = True) -> CancelResponse:
        self.signal_calls.append((job_id, signal, batch))
        return CancelResponse(job_id=job_id, accepted=True, raw="")


def obs(
    job_id: str,
    *,
    name: str | None,
    user: str = "alice",
    comment: str | None = None,
    state: str = "PENDING",
    submit_time: str | None = None,
    source: str = "squeue",
    exit_code: int | None = None,
    restart_count: int | None = 0,
) -> JobObservation:
    return JobObservation(
        job_id=job_id,
        source=source,
        observed_at=iso(datetime.now(UTC)) or "",
        state=normalize_slurm_state(state),
        raw_state=state,
        name=name,
        comment=comment,
        user=user,
        submit_time=submit_time,
        exit_code=exit_code,
        restart_count=restart_count,
    )


@dataclass
class UnitEnv:
    ctx: ServiceContext
    transport: FakeTransport
    scheduler: FakeScheduler
    clock: FakeClock
    published: list[str] = field(default_factory=list)
    publish_error: HpcLabError | None = None

    def set_policy(self, **policy: object) -> None:
        cluster = self.ctx.clusters.get("unit")
        profile = cluster.profile.model_copy(
            update={"policy": cluster.profile.policy.model_copy(update=policy)}
        )
        self.ctx.clusters.register(profile, cluster.overlay)


# ---------------------------------------------------------------------------
# Ledger seeding
# ---------------------------------------------------------------------------


@dataclass
class Seeded:
    experiment_id: str
    plan_id: str
    run_id: str
    exec_plan_hash: str
    run_spec: RunSpec
    plan: ExecutionPlan


def make_run_spec(
    *,
    code: CodeSnapshotRef | None = None,
    argv: list[str] | None = None,
    python: str = sys.executable,
    kind: str = "main",
    seed: int | None = 1,
) -> RunSpec:
    code = code or CodeSnapshotRef(digest=FAKE_DIGEST, file_count=1, total_bytes=10)
    argv = argv or [python, "job.py"]
    return RunSpec(
        experiment="exp",
        kind=kind,
        code=code,
        entrypoint=EntrypointSpec(argv=list(argv)),
        argv=list(argv),
        cell={"seed": seed} if seed is not None else {},
        seed=seed,
        runtime=RuntimeSpec(python=python, guard=False),
    )


def make_plan_for(
    run_id: str,
    run_spec: RunSpec,
    *,
    cluster_id: str,
    remote_root: str,
    profile_revision: str = "rev",
    gpus: int = 1,
    time: str = "01:00:00",
    partition: str | None = "gpu",
    gpu_type: str | None = "a100",
    account: str | None = "lab",
    gate: str | None = None,
    resilience: ResilienceSpec | None = None,
    nonce: str | None = None,
    requeue: bool | None = None,
) -> ExecutionPlan:
    hex_ = run_spec.code.digest.split(":", 1)[1]
    return ExecutionPlan(
        run_id=run_id,
        run_spec_hash=run_spec.spec_hash(),
        cluster_id=cluster_id,
        profile_revision=profile_revision if nonce is None else f"{profile_revision}#{nonce}",
        resources=ResourceSpec(
            partition=partition, account=account, gpus=gpus, gpu_type=gpu_type, time=time, mem="1G"
        ),
        remote_root=remote_root,
        snapshot_path=f"{remote_root.rstrip('/')}/snapshots/{hex_}",
        gate=gate,
        resilience=resilience,
        requeue=requeue,
    )


def seed_run(
    ctx: ServiceContext,
    *,
    cluster_id: str = "unit",
    remote_root: str = "/remote/hpclab",
    experiment_id: str | None = None,
    plan_id: str | None = None,
    budget_gpu_hours: float | None = None,
    max_parallel: int | None = None,
    label: str | None = None,
    seed: int | None = 1,
    kind: str = "main",
    code: CodeSnapshotRef | None = None,
    argv: list[str] | None = None,
    profile_revision: str = "rev",
    **plan_kw: Any,
) -> Seeded:
    run_spec = make_run_spec(code=code, argv=argv, kind=kind, seed=seed)
    with ctx.db.tx() as tx:
        eid = experiment_id or make_experiment(
            tx, cluster_id=cluster_id, budget_gpu_hours=budget_gpu_hours, max_parallel=max_parallel
        )
        pid = plan_id or make_plan(tx, eid)
        rid = make_run(
            tx,
            eid,
            pid,
            cell={"seed": seed} if seed is not None else {},
            seed=seed,
            kind=kind,
            label=label or f"cell.seed-{seed}",
            spec=run_spec.model_dump(mode="json"),
        )
        plan = make_plan_for(
            rid,
            run_spec,
            cluster_id=cluster_id,
            remote_root=remote_root,
            profile_revision=profile_revision,
            **plan_kw,
        )
        h = make_exec_plan(tx, rid, plan=plan.model_dump(mode="json"))
    assert h == plan.plan_hash()
    return Seeded(eid, pid, rid, h, run_spec, plan)


def add_plan(ctx: ServiceContext, seeded: Seeded, *, nonce: str, **plan_kw: Any) -> str:
    """Another ready execution plan for the same run (what run_recover would produce)."""
    plan = make_plan_for(
        seeded.run_id,
        seeded.run_spec,
        cluster_id=seeded.plan.cluster_id,
        remote_root=seeded.plan.remote_root,
        nonce=nonce,
        **plan_kw,
    )
    with ctx.db.tx() as tx:
        return make_exec_plan(tx, seeded.run_id, plan=plan.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Ledger reads
# ---------------------------------------------------------------------------


def attempt(ctx: ServiceContext, attempt_id: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    assert row is not None
    return row


def run_row(ctx: ServiceContext, run_id: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    assert row is not None
    return row


def operation(ctx: ServiceContext, operation_id: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM operations WHERE operation_id=?", (operation_id,))
    assert row is not None
    return row


def reservation(ctx: ServiceContext, attempt_id: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM budget_reservations WHERE attempt_id=?", (attempt_id,))
    assert row is not None
    return row


def events(ctx: ServiceContext, entity_id: str, prefix: str = "") -> list[str]:
    rows = ctx.db.all(
        "SELECT event_type FROM events WHERE entity_id=? AND event_type LIKE ? ORDER BY seq",
        (entity_id, prefix + "%"),
    )
    return [r["event_type"] for r in rows]


def outbox(ctx: ServiceContext, attempt_id: str, kind: str | None = None) -> list[dict[str, Any]]:
    rows = ctx.db.all("SELECT * FROM outbox WHERE target_id=? ORDER BY outbox_id", (attempt_id,))
    return [r for r in rows if kind is None or r["kind"] == kind]


def runner(ctx: ServiceContext, *, lease_s: float = 60.0) -> OutboxRunner:
    r = OutboxRunner(ctx, lease_s=lease_s)
    r.register_all(outbox_handlers())
    return r


def drain(ctx: ServiceContext, *, rounds: int = 50) -> int:
    return runner(ctx).drain(max_rounds=rounds)


def expire_leases(ctx: ServiceContext) -> None:
    """What a restarted controller does: previous leases are released immediately."""
    with ctx.db.tx() as tx:
        tx.execute(
            "UPDATE outbox SET state='pending', lease_owner=NULL, lease_until=NULL WHERE state='leased'"
        )


def write_project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "job.py").write_text(
        "import sys, time\nsecs = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0\ntime.sleep(secs)\nprint('done')\n"
    )
    return root
