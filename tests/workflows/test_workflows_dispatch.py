"""workflow_submit + dispatch rounds: cells up front, slots from persisted phases, gate, budget."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from wf_support import FakeSubmission, LabBuilder, install_fake_submission, make_ctx, one

from hpclab.domain.errors import (
    AdmissionDenied,
    IdempotencyConflict,
    InvalidArgument,
    NotReady,
    PreconditionFailed,
)
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxRunner
from hpclab.storage.db import jload


def _drain(ctx: ServiceContext) -> int:
    runner = OutboxRunner(ctx)
    runner.register_all(wfm.outbox_handlers())
    return runner.drain()


def _cells(ctx: ServiceContext, wf_id: str) -> dict[str, dict]:
    return {r["run_id"]: r for r in ctx.db.all("SELECT * FROM workflow_cells WHERE workflow_id=?", (wf_id,))}


# ---------------------------------------------------------------------------
# workflow_submit
# ---------------------------------------------------------------------------


def test_submit_persists_every_cell_up_front(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    data = wfm.workflow_submit(
        ctx, idempotency_key="k1", plan_id=m.plan_id, plan_hash=m.plan_hash, max_parallel=2
    )
    assert data.planned_run_count == 4 and data.skipped_count == 0
    assert data.state == "dispatching" and data.dispatch_state == "open" and data.version == 1
    assert data.workflow_ref == f"hpclab://workflows/{data.workflow_id}"
    assert data.operation_ref.startswith("hpclab://operations/")
    cells = _cells(ctx, data.workflow_id)
    assert set(cells) == set(m.run_ids)
    assert all(c["cell_state"] == "planned" and c["attempt_id"] is None for c in cells.values())
    assert {c["dispatch_key"] for c in cells.values()} == {f"wf:{data.workflow_id}:{r}" for r in m.run_ids}
    # nothing admitted synchronously: the controller dispatches from the outbox
    assert sub.admitted == []
    item = one(ctx, "SELECT * FROM outbox WHERE kind='workflow.dispatch'")
    assert item is not None and jload(item["payload_json"]) == {"workflow_id": data.workflow_id}
    ev = one(ctx, "SELECT * FROM events WHERE event_type='workflow.created'")
    assert ev is not None and len(jload(ev["payload_json"])["cells"]) == 4


def test_submit_is_idempotent_and_one_group_per_plan(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    a = wfm.workflow_submit(ctx, idempotency_key="k1", plan_id=m.plan_id, max_parallel=2)
    b = wfm.workflow_submit(ctx, idempotency_key="k1", plan_id=m.plan_id, max_parallel=2)
    assert a == b
    with pytest.raises(IdempotencyConflict):
        wfm.workflow_submit(ctx, idempotency_key="k1", plan_id=m.plan_id, max_parallel=3)
    with pytest.raises(AdmissionDenied) as exc:
        wfm.workflow_submit(ctx, idempotency_key="k2", plan_id=m.plan_id)
    assert exc.value.details["workflow_id"] == a.workflow_id
    assert one(ctx, "SELECT COUNT(*) AS n FROM workflows")["n"] == 1


def test_submit_refuses_unready_or_mismatched_plans(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    with pytest.raises(PreconditionFailed):
        wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, plan_hash="sha256:" + "0" * 64)
    pending = lab.matrix(n=1, status="preparing")
    with pytest.raises(NotReady):
        wfm.workflow_submit(ctx, idempotency_key="k2", plan_id=pending.plan_id)
    with pytest.raises(InvalidArgument):
        wfm.workflow_submit(ctx, idempotency_key="k3")
    with pytest.raises(InvalidArgument):  # workflow_id without resume_dispatch
        wfm.workflow_submit(ctx, idempotency_key="k4", workflow_id="wf_x")
    with pytest.raises(InvalidArgument):  # resume without expected_version
        wfm.workflow_submit(ctx, idempotency_key="k5", workflow_id="wf_x", resume_dispatch=True)
    assert one(ctx, "SELECT COUNT(*) AS n FROM workflows")["n"] == 0


def test_prepared_experiment_id_resolves_latest_ready_plan(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    data = wfm.workflow_submit(
        ctx, idempotency_key="k", prepared_experiment_id=m.experiment_id, plan_hash=m.plan_hash
    )
    assert data.plan_id == m.plan_id
    # default max_parallel comes from the site policy when neither call nor spec sets it
    assert data.max_parallel == 4


# ---------------------------------------------------------------------------
# max_parallel
# ---------------------------------------------------------------------------


def test_dispatch_respects_max_parallel_and_refills(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=5)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    r1 = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r1.dispatched == 2 and r1.state == "dispatching"
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0  # slots full
    assert lab.slot_holding() == 2
    # cells dispatched in plan order
    assert [r for r, _ in sub.admitted] == m.run_ids[:2]
    lab.finish(m.run_ids[0])
    r3 = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r3.dispatched == 1 and lab.slot_holding() == 2
    op = one(
        ctx,
        "SELECT * FROM operations WHERE operation_id=(SELECT operation_id FROM workflows WHERE workflow_id=?)",
        (wf,),
    )
    assert op["state"] == "running" and jload(op["progress_json"]) == {"dispatched": 3, "total": 5}


def test_uncertain_and_staging_attempts_hold_slots(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.set_attempt(lab.attempt_of(m.run_ids[0])["attempt_id"], phase="submission_uncertain")
    lab.set_attempt(lab.attempt_of(m.run_ids[1])["attempt_id"], phase="submission_in_flight")
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    cells = _cells(ctx, wf)
    assert cells[m.run_ids[0]]["cell_state"] == "uncertain"
    assert cells[m.run_ids[1]]["cell_state"] == "dispatched"


def test_max_parallel_survives_a_controller_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx(tmp_path)
    sub = install_fake_submission(monkeypatch)
    lab = LabBuilder(ctx, str(tmp_path / "remote"))
    m = lab.matrix(n=6)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=3).workflow_id
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 3
    ctx.clusters.close()
    ctx.db.close_all()
    # "restart": a fresh context on the same ledger, no memory of the previous rounds
    ctx2 = ServiceContext.create(ctx.paths, clock=ctx.clock)
    try:
        assert wfm.dispatch_workflow(ctx2, workflow_id=wf).dispatched == 0
        assert len(sub.admitted) == 3
    finally:
        ctx2.db.close_all()


def test_concurrent_dispatch_rounds_never_exceed_max_parallel(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=12)
    m2 = lab.matrix(n=12)
    wf = wfm.workflow_submit(ctx, idempotency_key="a", plan_id=m.plan_id, max_parallel=3).workflow_id
    wf2 = wfm.workflow_submit(ctx, idempotency_key="b", plan_id=m2.plan_id, max_parallel=2).workflow_id
    barrier = threading.Barrier(8)

    def worker(target: str) -> None:
        barrier.wait()  # all rounds start together; each thread has its own sqlite connection
        for _ in range(3):
            wfm.dispatch_workflow(ctx, workflow_id=target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, wf if i % 2 else wf2) for i in range(8)]
        for f in futures:
            f.result(timeout=30)  # re-raises a worker's exception
    per_wf = {
        r["workflow_id"]: r["n"]
        for r in ctx.db.all("SELECT workflow_id, COUNT(*) AS n FROM attempts GROUP BY workflow_id")
    }
    assert per_wf == {wf: 3, wf2: 2}
    # every admitted run exactly once
    runs = [r for r, _ in sub.admitted]
    assert len(runs) == len(set(runs)) == 5


def test_sweep_enqueues_live_workflows_only(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    live = wfm.workflow_submit(ctx, idempotency_key="a", plan_id=lab.matrix(n=1).plan_id).workflow_id
    ctx.db.conn.execute("DELETE FROM outbox")
    wfm.sweep(ctx)
    wfm.sweep(ctx)  # deduped while pending
    items = ctx.db.all("SELECT kind, target_id FROM outbox")
    assert items == [{"kind": "workflow.dispatch", "target_id": live}]
    assert {t.name for t in wfm.periodic_tasks()} == {"workflow.sweep"}
    assert set(wfm.outbox_handlers()) == {"workflow.dispatch", "workflow.cancel"}


# ---------------------------------------------------------------------------
# partial failure, completion
# ---------------------------------------------------------------------------


def test_partial_failure_never_redispatches_and_completes(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=3)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=3).workflow_id
    _drain(ctx)
    lab.finish(m.run_ids[0])  # succeeded
    lab.fail(m.run_ids[1], state="NODE_FAIL", exit_code=None, node_list="node7")  # infra failure
    lab.finish(m.run_ids[2], result="unchecked")  # completed, outputs not validated yet
    for _ in range(3):
        wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert len(sub.admitted) == 3  # nothing re-dispatched
    cells = _cells(ctx, wf)
    assert [cells[r]["cell_state"] for r in m.run_ids] == ["succeeded", "failed", "validating"]
    data = wfm.workflow_get(ctx, workflow_id=wf)
    assert data.state == "completed"
    assert data.operation_state == "succeeded"
    assert {c.run_id: c.code for c in data.cells} == {
        m.run_ids[0]: "E",  # completed with valid outputs, not evaluated yet
        m.run_ids[1]: "F",
        m.run_ids[2]: "E",  # completed, outputs not checked yet
    }
    assert one(ctx, "SELECT COUNT(*) AS n FROM events WHERE event_type='workflow.completed'")["n"] == 1


def test_run_claimed_outside_the_workflow_waits_and_consumed_plan_is_adopted(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    # the user run_submit-ted cell 0's frozen plan directly (same admission gate)
    with ctx.db.tx() as tx:
        direct = sub.admit(tx, ctx, run_id=m.run_ids[0], exec_plan_hash=m.exec_plans[m.run_ids[0]])
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    cells = _cells(ctx, wf)
    assert cells[m.run_ids[0]]["attempt_id"] == direct.attempt_id  # adopted, not dispatched twice
    assert "adopted" in cells[m.run_ids[0]]["note"]
    assert r.dispatched == 1 and len(sub.admitted) == 2


# ---------------------------------------------------------------------------
# smoke gate
# ---------------------------------------------------------------------------


def test_smoke_gate_is_two_stage(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    m = lab.matrix(n=3, smoke=True)
    data = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4)
    assert data.state == "gating" and data.gate is not None and data.gate.run_id == m.smoke_run_id
    wf = data.workflow_id
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 1
    assert [r for r, _ in sub.admitted] == [m.smoke_run_id]
    # smoke running, then COMPLETED but outputs not validated: still gated
    lab.set_attempt(lab.attempt_of(m.smoke)["attempt_id"], phase="observing", scheduler_state="RUNNING")
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    lab.finish(m.smoke, result="unchecked")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.dispatched == 0 and r.gate_state == "pending"
    # validated -> gate passes and the main cells go out
    lab.set_attempt(lab.attempt_of(m.smoke)["attempt_id"], result_state="valid")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.gate_state == "passed" and r.dispatched == 3 and r.state == "draining"
    assert one(ctx, "SELECT gate_state FROM workflows WHERE workflow_id=?", (wf,))["gate_state"] == "passed"


def test_smoke_failure_blocks_main_cells(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    m = lab.matrix(n=3, smoke=True)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4).workflow_id
    _drain(ctx)
    lab.fail(m.smoke, stderr="Traceback (most recent call last):\nRuntimeError: shapes do not match\n")
    for _ in range(2):
        r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.state == "gate_failed" and r.dispatched == 0
    assert [x for x, _ in sub.admitted] == [m.smoke_run_id]
    data = wfm.workflow_get(ctx, workflow_id=wf)
    assert data.gate is not None and data.gate.state == "failed"
    assert data.operation_state == "blocked" and "smoke gate failed" in (data.dispatch_blocked_reason or "")
    assert "ended FAILED" in (data.dispatch_blocked_reason or "")  # later rounds keep the specific reason
    assert all(
        c.cell_state == "planned" and "smoke gate failed" in (c.note or "")
        for c in data.cells
        if c.kind == "main"
    )


def test_smoke_required_outputs_must_be_finalized(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2, smoke=True, require_outputs=["metrics"])
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4).workflow_id
    _drain(ctx)
    lab.finish(m.smoke, result="valid")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.state == "gate_failed"  # valid result_state alone is not enough: metrics never finalized
    assert len(sub.admitted) == 1


def test_gate_failed_workflow_resumes_after_smoke_recovery(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2, smoke=True)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4).workflow_id
    _drain(ctx)
    lab.fail(m.smoke, state="NODE_FAIL", exit_code=None)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    v = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    with pytest.raises(PreconditionFailed):  # still failing
        wfm.workflow_submit(
            ctx, idempotency_key="r1", workflow_id=wf, resume_dispatch=True, expected_version=v
        )
    # the smoke run was recovered outside this workflow (its own plan, same admission gate) and passed
    retry_plan = lab.extra_exec_plan(m.smoke)
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=m.smoke, exec_plan_hash=retry_plan)
    lab.finish(m.smoke)
    data = wfm.workflow_submit(
        ctx, idempotency_key="r2", workflow_id=wf, resume_dispatch=True, expected_version=v
    )
    assert (
        data.resumed and data.state == "dispatching" and data.gate is not None and data.gate.state == "passed"
    )
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.dispatched == 2


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_exhaustion_stops_dispatch_with_reason(
    ctx: ServiceContext, lab: LabBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = install_fake_submission(monkeypatch, FakeSubmission(budget_gpu_hours=2.0))  # 1 GPU x 1 h per cell
    m = lab.matrix(n=4)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4).workflow_id
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.dispatched == 2 and r.blocked and "budget exhausted" in r.blocked
    data = wfm.workflow_get(ctx, workflow_id=wf)
    assert (
        data.operation_stage == "budget_exhausted" and data.operation_state == "running"
    )  # in-flight may free budget
    assert "budget exhausted" in (data.dispatch_blocked_reason or "")
    assert data.budget is not None and data.budget.reserved_gpu_hours == pytest.approx(2.0)
    # both finish without freeing budget (settled = reserved): nothing in flight -> blocked
    lab.finish(m.run_ids[0])
    lab.finish(m.run_ids[1])
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.dispatched == 0
    op = one(
        ctx,
        "SELECT * FROM operations WHERE operation_id=(SELECT operation_id FROM workflows WHERE workflow_id=?)",
        (wf,),
    )
    assert op["state"] == "blocked" and jload(op["error_json"])["code"] == "budget_exceeded"
    assert len(sub.admitted) == 2
