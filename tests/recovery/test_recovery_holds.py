"""run_recover refuses while the Run's original job may still run (execution holds).

The same rule as admission (``submission.execution_holds``): a job that ended in a state Slurm
may requeue from is watched for ``REQUEUE_WATCH_S`` (observe keeps polling it), and a job that
owns an attempt dir unobserved (finalize's duplicate owner) keeps the Run claimed until it is
confirmed ended (the scheduler's record or a human). A recovery plan made in that window would be
a second execution of one Run.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from wf_support import FakeSubmission, LabBuilder, one

from hpclab.domain.clock import FakeClock, iso
from hpclab.services import observe
from hpclab.services import recovery as rec
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext
from hpclab.services.submission import EVENT_DUPLICATE_JOB, EVENT_END_CONFIRMED

WATCH = observe.REQUEUE_WATCH_S


def _stopped(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock, state: str = "NODE_FAIL"
) -> tuple[str, str]:
    m = lab.matrix(n=1)
    rid = m.run_ids[0]
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=rid, exec_plan_hash=m.exec_plans[rid])
    aid = lab.fail(rid, state=state, exit_code=None, node_list="gpu-7")
    lab.set_attempt(aid, job_id="4242", terminal_at=iso(clock.now()))  # as observe records the end
    return rid, aid


def _counts(ctx: ServiceContext) -> tuple[int, int]:
    return (
        one(ctx, "SELECT COUNT(*) AS n FROM plans WHERE kind IN ('recovery','probe')")["n"],
        one(ctx, "SELECT COUNT(*) AS n FROM execution_plans")["n"],
    )


def _event(ctx: ServiceContext, attempt_id: str, event_type: str, payload: dict[str, Any]) -> None:
    with ctx.db.tx(source="human") as tx:
        tx.event("attempt", attempt_id, event_type, payload)


@pytest.mark.parametrize("state", ["NODE_FAIL", "PREEMPTED", "BOOT_FAIL"])
def test_recover_waits_while_the_original_job_may_be_requeued(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock, state: str
) -> None:
    _rid, aid = _stopped(ctx, lab, sub, clock, state)
    until = iso(clock.now() + timedelta(seconds=WATCH))
    before = _counts(ctx)
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert not d.eligible and d.exec_plan_hash is None and d.plan_id is None
    assert any(
        "original job 4242" in b and f"may be requeued; wait until {until} or confirm it ended" in b
        for b in d.blocking
    ), d.blocking
    assert any(h.strategy == "wait" and "confirm_ended" in h.message for h in d.hints)
    assert _counts(ctx) == before  # nothing written: the same key re-evaluates later
    clock.advance(WATCH)
    d2 = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert d2.eligible and d2.strategy == "retry" and d2.exec_plan_hash


def test_a_confirmed_end_lifts_the_watch(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    _rid, aid = _stopped(ctx, lab, sub, clock)
    assert not rec.run_recover(ctx, idempotency_key="k", attempt_id=aid).eligible
    # what attempt_resolve(resolution='confirm_ended') records after its live check
    _event(ctx, aid, EVENT_END_CONFIRMED, {"job_ids": ["4242"], "restart_count": 0, "reason": "sacct: over"})
    assert rec.run_recover(ctx, idempotency_key="k2", attempt_id=aid).eligible


def test_a_no_requeue_plan_is_not_watched(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    m = lab.matrix(n=1)
    rid = m.run_ids[0]
    h = lab.extra_exec_plan(rid, requeue=False, created_reason="initial")
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=rid, exec_plan_hash=h)
    aid = lab.fail(rid, state="NODE_FAIL", exit_code=None)
    lab.set_attempt(aid, job_id="4242", terminal_at=iso(clock.now()))
    assert rec.run_recover(ctx, idempotency_key="k", attempt_id=aid).eligible


def test_a_duplicate_owner_blocks_recovery_until_confirmed(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    _rid, aid = _stopped(ctx, lab, sub, clock, state="FAILED")
    _event(ctx, aid, EVENT_DUPLICATE_JOB, {"job_id": "4242", "owner_job_id": "4199", "exit_code": 95})
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert not d.eligible
    assert any("Slurm job 4199" in b and "confirm it ended" in b for b in d.blocking), d.blocking
    clock.advance(30 * 24 * 3600)  # no time bound: the owner runs as long as it runs
    assert not rec.run_recover(ctx, idempotency_key="k", attempt_id=aid).eligible
    _event(ctx, aid, EVENT_END_CONFIRMED, {"job_ids": ["4199"], "restart_count": 0, "reason": "over"})
    assert rec.run_recover(ctx, idempotency_key="k2", attempt_id=aid).eligible


def test_batch_recovery_holds_the_cell_back(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    m = lab.matrix(n=2)
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=2).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    held, free = m.run_ids
    held_aid = lab.fail(held, state="PREEMPTED", exit_code=None)
    lab.set_attempt(held_aid, job_id="4300", terminal_at=iso(clock.now()))
    lab.fail(free, state="NODE_FAIL", exit_code=None)
    data = rec.run_recover(
        ctx, idempotency_key="r", selector=rec.RecoverySelector(workflow_id=wf, cells="missing_valid")
    )
    assert data.cells is not None
    cells = {c.run_id: c for c in data.cells}
    assert cells[free].eligible and not cells[held].eligible
    assert "may be requeued" in (cells[held].blocking_reason or "")
    assert data.eligible_count == 1


def test_assess_blocks_on_holds_after_the_phase_rules() -> None:
    held = rec.assess(
        phase="terminal",
        scheduler_state="NODE_FAIL",
        result_state="unchecked",
        diagnosis=None,
        requested=None,
        resumable=False,
        has_checkpoint=False,
        has_hint_patch=False,
        holds=["original job 1 may be requeued; wait until T or confirm it ended"],
    )
    assert not held.eligible and held.blocking == [
        "original job 1 may be requeued; wait until T or confirm it ended"
    ]
    active = rec.assess(
        phase="observing",
        scheduler_state="RUNNING",
        result_state="pending",
        diagnosis=None,
        requested=None,
        resumable=False,
        has_checkpoint=False,
        has_hint_patch=False,
        holds=["x"],
    )
    assert any("still active" in b for b in active.blocking)


def test_recovering_an_abandoned_observing_attempt_names_its_job(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    m = lab.matrix(n=1)
    rid = m.run_ids[0]
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=rid, exec_plan_hash=m.exec_plans[rid])
    aid = lab.attempt_of(rid)["attempt_id"]
    lab.set_attempt(aid, phase="abandoned", job_id="4400", scheduler_state="RUNNING")
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert d.eligible and d.strategy == "retry"
    assert any("job 4400" in w and "scancel 4400" in w for w in d.warnings), d.warnings
