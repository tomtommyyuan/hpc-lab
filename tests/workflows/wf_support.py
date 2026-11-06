"""Shared helpers for the workflows and recovery suites (imported by name; see conftest).

* ``FakeSubmission`` implements the §7.2 ``submission.admit`` / ``run_cancel`` contracts against
  the ledger (claim per Run, consume-once plans, dispatch-closed check, optional budget and
  smoke-gate checks), so workflow dispatch can be tested before/without the real module. It is
  installed with ``install_fake_submission(monkeypatch)`` (sys.modules), which also shadows a real
  module when one exists.
* ``LabBuilder`` writes experiments/plans/runs/execution plans the way ``experiment_prepare``
  does (valid ``RunSpec`` + ``ExecutionPlan`` JSON), and moves attempts through phases.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpclab.config import get_paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import (
    AdmissionDenied,
    BudgetExceeded,
    NotFound,
    NotReady,
    PreconditionFailed,
    VersionConflict,
)
from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id, submit_token
from hpclab.domain.models import (
    CodeSnapshotRef,
    EntrypointSpec,
    ExecutionPlan,
    OutputDecl,
    ResilienceSpec,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, StorageRole, UserOverlay
from hpclab.services.context import ServiceContext, Settings
from hpclab.services.diagnosis import AttemptFacts, diagnose
from hpclab.storage.db import jdump, jload

CLUSTER = "sim"
SNAP_HEX = "ab" * 32


def one(ctx: ServiceContext, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    """``ctx.db.one`` for rows a test expects to exist."""
    row = ctx.db.one(sql, params)
    assert row is not None, sql
    return row


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def local_profile(cluster_id: str = CLUSTER) -> ClusterProfile:
    return ClusterProfile(
        cluster_id=cluster_id,
        display_name="test cluster",
        transport="local",
        partitions={
            "gpu": PartitionRule(
                max_time="2-00:00:00", gpu_types=["a100", "h100"], max_gpus_per_job=16, default=True
            ),
            "scavenge": PartitionRule(max_time="1-00:00:00", gpu_types=["h100"], preemptible=True),
            "interactive": PartitionRule(max_time="04:00:00", gpu_types=["a100"], interactive_only=True),
        },
        default_partition="gpu",
        storage={"scratch": StorageRole(path="/scratch/{user}")},
        policy=SitePolicy(max_parallel_default=4, poll_min_interval_s=30.0),
    )


def make_ctx(tmp_path: Path, clock: FakeClock | None = None) -> ServiceContext:
    paths = get_paths(tmp_path / "hpclab-home").ensure()
    ctx = ServiceContext.create(paths, clock=clock or FakeClock(), settings=Settings(request_deadline_s=10.0))
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    ctx.clusters.register(
        local_profile(),
        UserOverlay(cluster_id=CLUSTER, user="tester", account="lab", remote_root=str(remote)),
    )
    return ctx


# ---------------------------------------------------------------------------
# Fake submission module (contract of services.submission.admit / run_cancel)
# ---------------------------------------------------------------------------


@dataclass
class AdmitResult:
    attempt_id: str
    operation_id: str


@dataclass
class FakeSubmission:
    budget_gpu_hours: float | None = None  # total admissible GPU-hours (None = unlimited)
    check_gate: bool = True
    # expose ``request_cancel`` (the in-transaction cancel intent of the real module); False
    # models a submission module without it, so workflow_cancel falls back to the handler
    in_tx_cancel: bool = True
    admitted: list[tuple[str, str]] = field(default_factory=list)  # (run_id, attempt_id)
    cancels: list[dict[str, Any]] = field(default_factory=list)

    # -- admit ---------------------------------------------------------------
    def admit(
        self,
        tx: Any,
        ctx: ServiceContext,
        *,
        run_id: str,
        exec_plan_hash: str,
        workflow_id: str | None = None,
        force_without_gate: bool = False,
        reason: str | None = None,
    ) -> AdmitResult:
        ep = tx.one("SELECT * FROM execution_plans WHERE exec_plan_hash=?", (exec_plan_hash,))
        if ep is None:
            raise NotFound(f"execution plan {exec_plan_hash} not found")
        if ep["run_id"] != run_id:
            raise PreconditionFailed("execution plan belongs to another run")
        if ep["consumed_by_attempt_id"]:
            raise AdmissionDenied("plan already consumed", details={"reason": "plan_consumed"})
        if ep["status"] != "ready":
            raise NotReady(f"execution plan is {ep['status']}")
        run = tx.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if run["claim_attempt_id"]:
            raise AdmissionDenied("run has an active execution claim", details={"reason": "claim_held"})
        if workflow_id:
            wf = tx.one("SELECT dispatch_open FROM workflows WHERE workflow_id=?", (workflow_id,))
            if wf is None or not wf["dispatch_open"]:
                raise AdmissionDenied("workflow dispatch is closed", details={"reason": "dispatch_closed"})
        plan = ExecutionPlan.model_validate(jload(ep["plan_json"]))
        if self.check_gate and plan.gate and not force_without_gate:
            g = tx.one(
                "SELECT a.* FROM runs r JOIN attempts a ON a.attempt_id = r.latest_attempt_id WHERE r.run_id=?",
                (plan.gate,),
            )
            if g is None or not (g["scheduler_state"] == "COMPLETED" and g["result_state"] == "valid"):
                raise NotReady("smoke gate has not passed", details={"gate": plan.gate})
        gh = plan.resources.requested_gpu_hours
        if self.budget_gpu_hours is not None:
            used = sum(
                float(r["settled_consumed"] if r["state"] == "settled" else r["reserved"])
                for r in tx.all(
                    "SELECT state, reserved, settled_consumed FROM budget_reservations WHERE state != 'released'"
                )
            )
            if used + gh > self.budget_gpu_hours + 1e-9:
                raise BudgetExceeded(
                    f"needs {gh:.2f} GPU-h, {self.budget_gpu_hours - used:.2f} available",
                    details={"requested": gh, "available": self.budget_gpu_hours - used},
                )
        aid = new_id("attempt")
        seq_row = tx.one("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM attempts WHERE run_id=?", (run_id,))
        token = submit_token()
        op = tx.create_operation("submit", target_type="attempt", target_id=aid, state="queued")
        tx.insert(
            "attempts",
            {
                "attempt_id": aid,
                "run_id": run_id,
                "exec_plan_hash": exec_plan_hash,
                "seq": seq_row["n"],
                "cluster_id": plan.cluster_id,
                "submit_token": token,
                "job_name": f"hl.{run['label'][:20]}.{token[:8]}",
                "remote_dir": f"{plan.remote_root}/attempts/{aid}",
                "phase": "staging",
                "gpus": plan.resources.total_gpus,
                "partition": plan.resources.partition,
                "gpu_type": plan.resources.gpu_type,
                "time_limit_s": plan.resources.time_seconds,
                "workflow_id": workflow_id,
                "operation_id": op,
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.execute(
            "UPDATE runs SET claim_attempt_id=?, latest_attempt_id=?, status='submitting', version=version+1 WHERE run_id=?",
            (aid, aid, run_id),
        )
        tx.execute(
            "UPDATE execution_plans SET consumed_by_attempt_id=?, status='consumed' WHERE exec_plan_hash=?",
            (aid, exec_plan_hash),
        )
        tx.insert(
            "budget_reservations",
            {
                "reservation_id": new_id("reservation"),
                "experiment_id": run["experiment_id"],
                "attempt_id": aid,
                "reserved": gh,
                "state": "reserved",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.event(
            "attempt",
            aid,
            "attempt.created",
            {"run_id": run_id, "workflow_id": workflow_id},
            entity_version=1,
        )
        tx.enqueue("submit.stage", operation_id=op, target_id=aid, payload={"attempt_id": aid})
        self.admitted.append((run_id, aid))
        return AdmitResult(aid, op)

    # -- run_cancel ------------------------------------------------------------
    def run_cancel(
        self,
        ctx: ServiceContext,
        *,
        attempt_id: str,
        expected_version: int,
        idempotency_key: str,
        reason: str | None = None,
        mode: str = "cancel",
    ) -> dict[str, Any]:
        self.cancels.append(
            {
                "attempt_id": attempt_id,
                "expected_version": expected_version,
                "key": idempotency_key,
                "via": "run_cancel",
            }
        )
        with ctx.db.tx() as tx:
            a = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
            assert a is not None
            if a["version"] != expected_version:
                raise VersionConflict("attempt moved", details={"current_version": a["version"]})
            self._intent(tx, a, reason)
        return {"attempt_id": attempt_id}

    def request_cancel(
        self,
        tx: Any,
        ctx: ServiceContext,
        *,
        attempt_id: str,
        reason: str | None = None,
        mode: str = "cancel",
    ) -> dict[str, Any]:
        """Cancel intent inside the caller's transaction (contract of submission.request_cancel)."""
        a = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
        if a is None:
            raise NotFound(f"attempt {attempt_id} not found")
        if a["phase"] in ("terminal", "submission_rejected", "cancelled_before_submit", "abandoned"):
            raise PreconditionFailed(f"attempt {attempt_id} is {a['phase']}; nothing to cancel")
        self.cancels.append({"attempt_id": attempt_id, "key": None, "via": "request_cancel"})
        self._intent(tx, a, reason)
        return {"attempt_id": attempt_id}

    @staticmethod
    def _intent(tx: Any, a: dict[str, Any], reason: str | None) -> None:
        attempt_id = a["attempt_id"]
        if a["phase"] == "staging":
            tx.update(
                "attempts",
                "attempt_id",
                attempt_id,
                {"phase": "cancelled_before_submit", "cancel_state": "confirmed"},
            )
            tx.execute("UPDATE runs SET claim_attempt_id=NULL WHERE run_id=?", (a["run_id"],))
            tx.execute("UPDATE budget_reservations SET state='released' WHERE attempt_id=?", (attempt_id,))
        else:
            tx.update(
                "attempts",
                "attempt_id",
                attempt_id,
                {"cancel_state": "requested", "cancel_reason": reason},
            )
        tx.event("attempt", attempt_id, "attempt.cancel_requested", {"reason": reason})


def install_fake_submission(monkeypatch: Any, fake: FakeSubmission | None = None) -> FakeSubmission:
    fake = fake or FakeSubmission()
    mod = types.ModuleType("hpclab.services.submission")
    mod.admit = fake.admit  # type: ignore[attr-defined]
    mod.run_cancel = fake.run_cancel  # type: ignore[attr-defined]
    if fake.in_tx_cancel:
        mod.request_cancel = fake.request_cancel  # type: ignore[attr-defined]
    mod.AdmitResult = AdmitResult  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hpclab.services.submission", mod)
    return fake


# ---------------------------------------------------------------------------
# Ledger builder
# ---------------------------------------------------------------------------


@dataclass
class Matrix:
    experiment_id: str
    plan_id: str
    plan_hash: str
    run_ids: list[str]  # main runs in order
    smoke_run_id: str | None = None
    exec_plans: dict[str, str] = field(default_factory=dict)  # run_id -> exec plan hash

    @property
    def smoke(self) -> str:
        assert self.smoke_run_id is not None, "matrix built without smoke"
        return self.smoke_run_id


def run_spec(
    *,
    kind: str = "main",
    cell: dict[str, Any] | None = None,
    seed: int | None = 1,
    params: dict[str, Any] | None = None,
    runtime: RuntimeSpec | None = None,
) -> RunSpec:
    cell = dict(cell or {})
    if seed is not None:
        cell.setdefault("seed", seed)
    return RunSpec(
        experiment="exp",
        kind=kind,
        code=CodeSnapshotRef(digest="sha256:" + SNAP_HEX, file_count=1, total_bytes=10),
        entrypoint=EntrypointSpec(argv=["python", "train.py"]),
        argv=["python", "train.py", "--out", "{output_dir}"],
        params={**(params or {}), **{k: v for k, v in cell.items()}},
        cell=cell,
        seed=seed,
        runtime=runtime or RuntimeSpec(),
        outputs=[OutputDecl(name="metrics", path="{output_dir}/metrics.json", role="metrics")],
    )


class LabBuilder:
    def __init__(self, ctx: ServiceContext, remote_root: str) -> None:
        self.ctx = ctx
        self.remote_root = remote_root

    def matrix(
        self,
        *,
        n: int = 3,
        smoke: bool = False,
        max_parallel: int | None = None,
        resources: dict[str, Any] | None = None,
        resilience: ResilienceSpec | None = None,
        require_outputs: list[str] | None = None,
        status: str = "ready",
        runtime: RuntimeSpec | None = None,
        budget_gpu_hours: float | None = None,
    ) -> Matrix:
        ctx = self.ctx
        spec: dict[str, Any] = {
            "name": "exp",
            "cluster": CLUSTER,
            "entrypoint": {"argv": ["python", "train.py"]},
        }
        if smoke:
            spec["smoke"] = {"enabled": True, "require_outputs": list(require_outputs or [])}
        with ctx.db.tx() as tx:
            eid = new_id("experiment")
            tx.insert(
                "experiments",
                {
                    "experiment_id": eid,
                    "name": "exp",
                    "cluster_id": CLUSTER,
                    "spec_json": jdump(spec),
                    "spec_hash": digest(spec),
                    "max_parallel": max_parallel,
                    "budget_gpu_hours": budget_gpu_hours,
                    "status": "ready",
                    "created_at": tx.now,
                    "updated_at": tx.now,
                },
            )
            pid = new_id("plan")
            tx.insert(
                "plans",
                {
                    "plan_id": pid,
                    "experiment_id": eid,
                    "kind": "matrix",
                    "status": status,
                    "plan_hash": None,
                    "created_at": tx.now,
                    "updated_at": tx.now,
                },
            )
            m = Matrix(eid, pid, "", [])
            if smoke:
                m.smoke_run_id = self._run(
                    tx,
                    m,
                    kind="smoke",
                    cell={"method": "smoke"},
                    seed=None,
                    gate=None,
                    resources=resources,
                    resilience=resilience,
                    runtime=runtime,
                )
            for i in range(n):
                rid = self._run(
                    tx,
                    m,
                    kind="main",
                    cell={"method": "m" + str(i % 2)},
                    seed=i,
                    gate=m.smoke_run_id,
                    resources=resources,
                    resilience=resilience,
                    runtime=runtime,
                )
                m.run_ids.append(rid)
            m.plan_hash = digest({"plan": pid, "exec": sorted(m.exec_plans.values())})
            tx.execute(
                "UPDATE plans SET plan_hash=?, smoke_run_id=? WHERE plan_id=?",
                (m.plan_hash if status == "ready" else None, m.smoke_run_id, pid),
            )
        return m

    def _run(
        self,
        tx: Any,
        m: Matrix,
        *,
        kind: str,
        cell: dict[str, Any],
        seed: int | None,
        gate: str | None,
        resources: dict[str, Any] | None,
        resilience: ResilienceSpec | None,
        runtime: RuntimeSpec | None,
    ) -> str:
        rid = new_id("run")
        spec = run_spec(kind=kind, cell=cell, seed=seed, runtime=runtime)
        res = ResourceSpec(
            **{
                "partition": "gpu",
                "gpus": 1,
                "gpu_type": "a100",
                "time": "01:00:00",
                "mem": "32G",
                **(resources or {}),
            }
        )
        plan = ExecutionPlan(
            run_id=rid,
            run_spec_hash=spec.spec_hash(),
            cluster_id=CLUSTER,
            profile_revision="sha256:rev",
            resources=res,
            remote_root=self.remote_root,
            snapshot_path=f"{self.remote_root}/snapshots/{SNAP_HEX}",
            gate=gate,
            resilience=resilience,
        )
        h = plan.plan_hash()
        label = ".".join(f"{k}-{v}" for k, v in sorted(spec.cell.items()))
        tx.insert(
            "runs",
            {
                "run_id": rid,
                "experiment_id": m.experiment_id,
                "plan_id": m.plan_id,
                "kind": kind,
                "label": label or "run",
                "cell_key": jdump(spec.cell),
                "seed": seed,
                "spec_json": jdump(spec.model_dump(mode="json")),
                "spec_hash": spec.spec_hash(),
                "current_exec_plan_hash": h,
                "status": "prepared",
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.insert(
            "execution_plans",
            {
                "exec_plan_hash": h,
                "run_id": rid,
                "plan_json": jdump(plan.model_dump(mode="json")),
                "status": "ready",
                "created_at": tx.now,
            },
        )
        m.exec_plans[rid] = h
        return rid

    def extra_exec_plan(self, run_id: str, **update: Any) -> str:
        """Another ready execution plan of an existing Run (what run_recover would create)."""
        row = one(
            self.ctx,
            "SELECT plan_json FROM execution_plans WHERE run_id=? ORDER BY created_at, rowid LIMIT 1",
            (run_id,),
        )
        base = ExecutionPlan.model_validate(jload(row["plan_json"]))
        plan = ExecutionPlan.model_validate(
            {**base.model_dump(mode="json"), "created_reason": "retry", **update}
        )
        with self.ctx.db.tx() as tx:
            tx.insert(
                "execution_plans",
                {
                    "exec_plan_hash": plan.plan_hash(),
                    "run_id": run_id,
                    "plan_json": jdump(plan.model_dump(mode="json")),
                    "status": "ready",
                    "created_at": tx.now,
                },
            )
        return plan.plan_hash()

    # -- attempts ----------------------------------------------------------------
    def attempt_of(self, run_id: str) -> dict[str, Any]:
        row = one(
            self.ctx,
            "SELECT a.* FROM runs r JOIN attempts a ON a.attempt_id = r.latest_attempt_id WHERE r.run_id=?",
            (run_id,),
        )
        assert row is not None, f"run {run_id} has no attempt"
        return row

    def set_attempt(self, attempt_id: str, **values: Any) -> None:
        """Move an attempt (as observe/submission would), releasing the claim on final phases."""
        diag = values.pop("diagnosis", None)
        if diag is not None:
            values["diagnosis_json"] = jdump(
                diag.model_dump(mode="json") if hasattr(diag, "model_dump") else diag
            )
        with self.ctx.db.tx(source="observer") as tx:
            tx.update("attempts", "attempt_id", attempt_id, values)
            phase = values.get("phase")
            if phase in ("terminal", "submission_rejected", "cancelled_before_submit", "abandoned"):
                a = tx.one("SELECT run_id FROM attempts WHERE attempt_id=?", (attempt_id,))
                assert a is not None
                tx.execute(
                    "UPDATE runs SET claim_attempt_id=NULL WHERE run_id=? AND claim_attempt_id=?",
                    (a["run_id"], attempt_id),
                )
                tx.execute(
                    "UPDATE budget_reservations SET state='settled', settled_consumed=reserved WHERE attempt_id=? AND state='reserved'",
                    (attempt_id,),
                )
            tx.event(
                "attempt", attempt_id, "attempt.test_update", {k: str(v)[:40] for k, v in values.items()}
            )

    def finish(self, run_id: str, state: str = "COMPLETED", result: str = "valid", **extra: Any) -> str:
        a = self.attempt_of(run_id)
        self.set_attempt(
            a["attempt_id"], phase="terminal", scheduler_state=state, result_state=result, **extra
        )
        return str(a["attempt_id"])

    def fail(
        self, run_id: str, *, state: str = "FAILED", stderr: str = "", exit_code: int | None = 1, **facts: Any
    ) -> str:
        """Terminal failure with a *real* diagnosis computed from the given log tail."""
        a = self.attempt_of(run_id)
        diag = diagnose(AttemptFacts(scheduler_state=state, exit_code=exit_code, **facts), "", stderr)
        self.set_attempt(
            a["attempt_id"],
            phase="terminal",
            scheduler_state=state,
            result_state="unchecked",
            exit_code=exit_code,
            diagnosis=diag,
            node_list=facts.get("node_list"),
        )
        return str(a["attempt_id"])

    def slot_holding(self, experiment_id: str | None = None) -> int:
        row = one(
            self.ctx,
            "SELECT COUNT(*) AS n FROM attempts WHERE phase IN "
            "('staging','submission_in_flight','submission_uncertain','duplicate_detected','job_identified','observing')",
        )
        return int(row["n"])


IMPORT_ERR = (
    "Traceback (most recent call last):\n"
    '  File "train.py", line 3, in <module>\n'
    "    import fancylib\n"
    "ModuleNotFoundError: No module named 'fancylib'\n"
)


def other_import_err(name: str) -> str:
    return IMPORT_ERR.replace("fancylib", name)
