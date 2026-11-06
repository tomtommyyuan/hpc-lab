"""Selector (batch) recovery -> one recovery plan -> workflow_submit dispatches only eligible cells."""

from __future__ import annotations

import pytest
from wf_support import IMPORT_ERR, FakeSubmission, LabBuilder, one

from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.services import recovery as rec
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext

NAN = "step 10 loss nan\nstep 20 loss nan\nRuntimeError: Loss is NaN at step 20\n"


def _workflow(ctx: ServiceContext, lab: LabBuilder, n: int = 6) -> tuple[str, list[str]]:
    m = lab.matrix(n=n)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=n).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    return wf, m.run_ids


def test_missing_valid_batch_then_dispatch_only_eligible(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    wf, runs = _workflow(ctx, lab)
    ok, node_fail, nan, running, uncertain, preempted = runs
    lab.finish(ok)  # valid: excluded from missing_valid (V only once evaluated)
    with ctx.db.tx() as tx:
        from hpclab.testing.factories import make_evaluation

        make_evaluation(tx, lab.attempt_of(ok)["attempt_id"])
    lab.fail(node_fail, state="NODE_FAIL", exit_code=None, node_list="n1")
    lab.fail(nan, stderr=NAN)
    lab.set_attempt(lab.attempt_of(running)["attempt_id"], phase="observing", scheduler_state="RUNNING")
    lab.set_attempt(lab.attempt_of(uncertain)["attempt_id"], phase="submission_uncertain")
    lab.fail(preempted, state="PREEMPTED", exit_code=None)
    data = rec.run_recover(
        ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf, cells="missing_valid")
    )
    assert data.mode == "batch" and data.eligible and data.plan_id and data.source_workflow_id == wf
    assert data.cells is not None
    cells = {c.run_id: c for c in data.cells}
    assert set(cells) == {node_fail, nan, running, uncertain, preempted}
    assert (
        cells[node_fail].eligible and cells[node_fail].strategy == "retry" and cells[node_fail].exec_plan_hash
    )
    assert cells[preempted].eligible
    assert not cells[nan].eligible and cells[nan].diagnosis_code == "nan_loss"
    assert not cells[running].eligible and "still active" in (cells[running].blocking_reason or "")
    assert not cells[uncertain].eligible and "unresolved" in (cells[uncertain].blocking_reason or "")
    assert data.eligible_count == 2 and data.total == 5 and data.requested_gpu_hours == pytest.approx(2.0)
    assert data.next_step and (
        f'workflow_submit(plan_id="{data.plan_id}", idempotency_key="<new key>")' in data.next_step
    )
    assert rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf)) == data

    before = len(sub.admitted)
    sub_data = wfm.workflow_submit(ctx, idempotency_key="rw", plan_id=data.plan_id, max_parallel=4)
    assert (
        sub_data.plan_kind == "recovery" and sub_data.planned_run_count == 2 and sub_data.skipped_count == 3
    )
    rwf = sub_data.workflow_id
    r = wfm.dispatch_workflow(ctx, workflow_id=rwf)
    assert r.dispatched == 2
    new = [run for run, _ in sub.admitted[before:]]
    assert sorted(new) == sorted([node_fail, preempted])
    # the cells track their own (new) attempts, not the failures they recover
    got = {c.run_id: c for c in wfm.workflow_get(ctx, workflow_id=rwf).cells}
    assert (
        got[node_fail].cell_state == "dispatched" and got[node_fail].attempt_id != cells[node_fail].attempt_id
    )
    assert got[nan].cell_state == "skipped" and "nan_loss" in (got[nan].note or "")
    # a second round never re-dispatches; the original workflow is untouched
    assert wfm.dispatch_workflow(ctx, workflow_id=rwf).dispatched == 0
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0


def test_diagnosis_and_run_id_selectors(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    wf, runs = _workflow(ctx, lab, n=4)
    lab.fail(runs[0], stderr=IMPORT_ERR)
    lab.fail(runs[1], stderr=IMPORT_ERR)
    lab.fail(runs[2], state="NODE_FAIL", exit_code=None)
    only_env = rec.run_recover(
        ctx,
        idempotency_key="a",
        selector=rec.RecoverySelector(
            workflow_id=wf, cells=rec.DiagnosisSelector(diagnosis_in=["import_error"])
        ),
    )
    assert only_env.cells is not None and {c.run_id for c in only_env.cells} == set(runs[:2])
    assert all(c.eligible and c.diagnosis_code == "import_error" for c in only_env.cells)
    picked = rec.run_recover(
        ctx,
        idempotency_key="b",
        selector=rec.RecoverySelector(workflow_id=wf, cells=rec.RunIdsSelector(run_ids=[runs[2]])),
    )
    assert picked.cells is not None and [c.run_id for c in picked.cells] == [runs[2]] and picked.eligible
    with pytest.raises(InvalidArgument):
        rec.run_recover(
            ctx,
            idempotency_key="c",
            selector=rec.RecoverySelector(workflow_id=wf, cells=rec.RunIdsSelector(run_ids=["run_nope"])),
        )
    with pytest.raises(NotFound):
        rec.run_recover(ctx, idempotency_key="d", selector=rec.RecoverySelector(workflow_id="wf_nope"))
    with pytest.raises(InvalidArgument):
        rec.run_recover(
            ctx, idempotency_key="e", attempt_id="att_x", selector=rec.RecoverySelector(workflow_id=wf)
        )


def test_batch_with_nothing_recoverable_writes_nothing(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    wf, runs = _workflow(ctx, lab, n=2)
    lab.fail(runs[0], stderr=NAN)
    lab.set_attempt(lab.attempt_of(runs[1])["attempt_id"], phase="observing", scheduler_state="RUNNING")
    d = rec.run_recover(ctx, idempotency_key="x", selector=rec.RecoverySelector(workflow_id=wf))
    assert not d.eligible and d.plan_id is None and d.blocking and d.cells is not None and len(d.cells) == 2
    assert one(ctx, "SELECT COUNT(*) AS n FROM plans WHERE kind='recovery'")["n"] == 0


def test_never_dispatched_cells_of_a_cancelled_workflow_are_recoverable(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=3)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=1).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    open_wf = rec.run_recover(ctx, idempotency_key="a", selector=rec.RecoverySelector(workflow_id=wf))
    assert open_wf.cells is not None
    assert all(
        not c.eligible and "will dispatch it" in (c.blocking_reason or "")
        for c in open_wf.cells
        if c.attempt_id is None
    )  # the workflow will dispatch them
    v = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=v, idempotency_key="c")
    d = rec.run_recover(ctx, idempotency_key="b", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    initial = [c for c in d.cells if c.strategy == "initial"]
    assert {c.run_id for c in initial} == set(m.run_ids[1:]) and all(c.eligible for c in initial)
    assert all(c.exec_plan_hash == m.exec_plans[c.run_id] for c in initial)
    r = wfm.workflow_submit(ctx, idempotency_key="rw", plan_id=d.plan_id, max_parallel=2)
    assert wfm.dispatch_workflow(ctx, workflow_id=r.workflow_id).dispatched == 2


def test_batch_gpu_hours_include_never_dispatched_cells(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """GPU-hours of 'initial' cells are known from their frozen plan: never reported as 0/None,
    and the budget check counts them."""
    m = lab.matrix(n=3, budget_gpu_hours=1.5)  # 1 GPU x 1 h per cell
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=1).workflow_id
    v = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=v, idempotency_key="c")
    d = rec.run_recover(ctx, idempotency_key="b", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None and d.eligible_count == 3
    assert all(c.strategy == "initial" and c.requested_gpu_hours == pytest.approx(1.0) for c in d.cells)
    assert d.requested_gpu_hours == pytest.approx(3.0)
    budget = next((c for c in d.checks if c.check == "budget"), None)
    assert budget is not None and budget.status in ("warn", "unchecked")


def test_missing_valid_skips_runs_waiting_for_evaluation(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """E (completed, evaluation pending) is not a failure: missing_valid never proposes it."""
    wf, runs = _workflow(ctx, lab, n=4)
    evaluated, awaiting, checking, broken = runs
    lab.finish(evaluated)
    with ctx.db.tx() as tx:
        from hpclab.testing.factories import make_evaluation

        make_evaluation(tx, lab.attempt_of(evaluated)["attempt_id"])
    lab.finish(awaiting)  # valid outputs, not evaluated yet -> E
    lab.finish(checking, result="unchecked")  # outputs being checked -> E
    lab.fail(broken, state="NODE_FAIL", exit_code=None)
    d = rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None and [c.run_id for c in d.cells] == [broken]
    assert d.eligible_count == 1 and d.total == 1
    assert any("2 run(s) are E" in w and "evaluation_collect" in w for w in d.warnings), d.warnings
    # an explicit selection still explains why an E run is not recoverable
    picked = rec.run_recover(
        ctx,
        idempotency_key="p",
        selector=rec.RecoverySelector(workflow_id=wf, cells=rec.RunIdsSelector(run_ids=[awaiting])),
    )
    assert picked.cells is not None and not picked.cells[0].eligible and picked.cells[0].code == "E"


def test_batch_recovery_points_each_eligible_run_at_its_new_plan(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    wf, runs = _workflow(ctx, lab, n=3)
    lab.fail(runs[0], state="NODE_FAIL", exit_code=None)
    lab.fail(runs[1], state="PREEMPTED", exit_code=None)
    lab.fail(runs[2], stderr=NAN)  # not eligible: its pointer stays
    before = {
        r: one(ctx, "SELECT current_exec_plan_hash FROM runs WHERE run_id=?", (r,))["current_exec_plan_hash"]
        for r in runs
    }
    d = rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None and d.eligible_count == 2
    for c in d.cells:
        now = one(ctx, "SELECT current_exec_plan_hash FROM runs WHERE run_id=?", (c.run_id,))
        if c.eligible:
            assert now["current_exec_plan_hash"] == c.exec_plan_hash != before[c.run_id]
            ev = ctx.db.all(
                "SELECT payload_json FROM events WHERE entity_type='run' AND entity_id=? "
                "AND event_type='run.exec_plan_changed'",
                (c.run_id,),
            )
            assert len(ev) == 1 and d.plan_id in ev[0]["payload_json"]
        else:
            assert now["current_exec_plan_hash"] == before[c.run_id]


def _paused(ctx: ServiceContext, lab: LabBuilder) -> tuple[str, list[str]]:
    from wf_support import IMPORT_ERR as ERR

    m = lab.matrix(n=4)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=2).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=ERR)
    lab.fail(m.run_ids[1], stderr=ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped
    return wf, m.run_ids


def test_cells_of_a_paused_workflow_say_how_to_resume_it(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    wf, runs = _paused(ctx, lab)
    version = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    d = rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    never = [c for c in d.cells if c.attempt_id is None]
    assert {c.run_id for c in never} == set(runs[2:])
    for c in never:
        why = c.blocking_reason or ""
        assert not c.eligible and "will dispatch it" not in why, why
        assert (
            "paused" in why
            and (
                f'workflow_submit(workflow_id="{wf}", resume_dispatch=true, expected_version={version}, '
                'idempotency_key="<new key>")'
            )
            in why
        ), why


def test_cells_of_a_gate_failed_workflow_say_what_to_do(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2, smoke=True)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=4).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.smoke, state="NODE_FAIL", exit_code=None)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).state == "gate_failed"
    d = rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    cells = {c.run_id: c for c in d.cells}
    assert cells[m.smoke].eligible  # the smoke run itself is what to recover
    smoke_attempt = cells[m.smoke].attempt_id
    for rid in m.run_ids:
        why = cells[rid].blocking_reason or ""
        assert not cells[rid].eligible and "will dispatch it" not in why
        assert "smoke gate" in why and "resume_dispatch=true" in why, why
        assert f'run_recover(attempt_id="{smoke_attempt}", idempotency_key="<new key>")' in why, why


def test_a_cell_the_open_workflow_skipped_is_not_left_to_that_workflow(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """Only 'planned' cells are still owned by an open workflow: a cell it skipped (no ready
    execution plan at dispatch time) is never dispatched by it, so recovery must not say it will."""
    m = lab.matrix(n=3)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=1).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    waiting, skipped = m.run_ids[1], m.run_ids[2]
    with ctx.db.tx() as tx:  # as _dispatch_cells does when the plan was missing at dispatch time
        tx.execute(
            "UPDATE workflow_cells SET cell_state='skipped', note='no ready execution plan' "
            "WHERE workflow_id=? AND run_id=?",
            (wf, skipped),
        )
    assert one(ctx, "SELECT state FROM workflows WHERE workflow_id=?", (wf,))["state"] == "dispatching"
    d = rec.run_recover(ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    cells = {c.run_id: c for c in d.cells}
    assert not cells[waiting].eligible and "will dispatch it" in (cells[waiting].blocking_reason or "")
    assert "will dispatch it" not in (cells[skipped].blocking_reason or "")
    assert cells[skipped].eligible and cells[skipped].strategy == "initial", cells[skipped]
    assert cells[skipped].exec_plan_hash == m.exec_plans[skipped]


def test_gate_failed_text_follows_the_smoke_runs_retry(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """After the user recovered and resubmitted the smoke run, its latest attempt is running (then
    passed): the main cells must not tell them to run_recover that attempt."""
    m = lab.matrix(n=2, smoke=True)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=4).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    failed = lab.fail(m.smoke, state="NODE_FAIL", exit_code=None)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).state == "gate_failed"
    single = rec.run_recover(ctx, idempotency_key="s", attempt_id=failed)
    assert single.eligible and single.exec_plan_hash
    with ctx.db.tx() as tx:
        retry = sub.admit(tx, ctx, run_id=m.smoke, exec_plan_hash=single.exec_plan_hash).attempt_id
    lab.set_attempt(retry, phase="observing", scheduler_state="RUNNING")
    d = rec.run_recover(ctx, idempotency_key="r1", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    for c in d.cells:
        if c.run_id == m.smoke:
            continue
        why = c.blocking_reason or ""
        assert not c.eligible and "run_recover(" not in why, why
        assert retry in why and "resume_dispatch=true" in why, why
    lab.finish(m.smoke)  # the retry passed: resuming is all that is left
    d = rec.run_recover(ctx, idempotency_key="r2", selector=rec.RecoverySelector(workflow_id=wf))
    assert d.cells is not None
    for c in d.cells:
        if c.run_id == m.smoke:
            continue
        why = c.blocking_reason or ""
        assert "run_recover(" not in why and "passed" in why and "resume_dispatch=true" in why, why
