"""Circuit breaker, resume, workflow_cancel and workflow_get."""

from __future__ import annotations

import pytest
from wf_support import (
    IMPORT_ERR,
    FakeSubmission,
    LabBuilder,
    install_fake_submission,
    one,
    other_import_err,
)

from hpclab.domain.errors import NotFound, PreconditionFailed, VersionConflict
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxRunner
from hpclab.storage.db import jload


def _drain(ctx: ServiceContext) -> int:
    runner = OutboxRunner(ctx)
    runner.register_all(wfm.outbox_handlers())
    return runner.drain()


def _version(ctx: ServiceContext, wf: str) -> int:
    return int(one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"])


def _submit(ctx: ServiceContext, m, max_parallel: int = 2, **kw) -> str:
    return wfm.workflow_submit(
        ctx, idempotency_key=f"k-{m.plan_id}", plan_id=m.plan_id, max_parallel=max_parallel, **kw
    ).workflow_id


# ---------------------------------------------------------------------------
# circuit breaker
# ---------------------------------------------------------------------------


def test_breaker_trips_on_two_same_environment_failures(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=6)
    wf = _submit(ctx, m, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.tripped and r.dispatch_state == "paused" and r.dispatched == 0
    assert len(sub.admitted) == 3  # nothing new; the third cell is untouched and still in flight
    assert lab.attempt_of(m.run_ids[2])["phase"] == "staging"
    data = wfm.workflow_get(ctx, workflow_id=wf)
    assert data.dispatch_state == "paused" and data.breaker is not None and data.breaker.state == "tripped"
    assert data.breaker.fingerprint is not None
    assert (
        data.breaker.fingerprint["category"] == "environment"
        and data.breaker.fingerprint["code"] == "import_error"
    )
    assert "fancylib" in data.breaker.fingerprint["error_line"]
    assert {c["run_id"] for c in data.breaker.cells} == set(m.run_ids[:2])
    assert data.breaker.resume_hint and "resume_dispatch=true" in data.breaker.resume_hint
    assert data.operation_state == "blocked" and data.operation_stage == "paused_by_breaker"
    assert data.budget is not None and data.budget.reserved_gpu_hours == pytest.approx(
        1.0
    )  # in-flight keeps its budget
    # further rounds while paused dispatch nothing
    lab.finish(m.run_ids[2])
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    assert one(ctx, "SELECT COUNT(*) AS n FROM events WHERE event_type='workflow.breaker_tripped'")["n"] == 1


def test_breaker_ignores_infrastructure_failures(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], state="NODE_FAIL", exit_code=None, node_list="node3")
    lab.fail(m.run_ids[1], state="NODE_FAIL", exit_code=None, node_list="node3")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert not r.tripped and r.dispatch_state == "open" and r.dispatched == 2


def test_breaker_needs_the_same_fingerprint(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=other_import_err("alpha"))
    lab.fail(m.run_ids[1], stderr=other_import_err("beta"))
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert not r.tripped and r.dispatched == 2


def test_breaker_counts_wrapper_guard_failures(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=3)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    for rid in m.run_ids[:2]:  # no stored diagnosis at all: the wrapper's guard status is enough
        a = lab.attempt_of(rid)
        lab.set_attempt(
            a["attempt_id"],
            phase="terminal",
            scheduler_state="FAILED",
            exit_code=3,
            app_state="guard_failed",
            status_json='{"guard_check": "torch_cuda_available"}',
        )
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.tripped
    bj = jload(one(ctx, "SELECT breaker_json FROM workflows WHERE workflow_id=?", (wf,))["breaker_json"])
    assert bj["fingerprint"]["code"] == "env_guard_failed:torch_cuda_available"


def test_breaker_stays_quiet_when_nothing_is_left_to_dispatch(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    wf = _submit(ctx, m, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert not r.tripped and r.state == "completed" and r.dispatch_state == "open"


def test_breaker_disabled_by_policy(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    m = lab.matrix(n=4)
    policy = wfm.FailurePolicy(circuit_breaker=wfm.CircuitBreakerPolicy(enabled=False))
    wf = _submit(ctx, m, failure_policy=policy)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 2
    assert wfm.workflow_get(ctx, workflow_id=wf).failure_policy.circuit_breaker.enabled is False


def test_identical_sbatch_refusals_trip_the_breaker(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    for rid in m.run_ids[:2]:
        lab.set_attempt(
            lab.attempt_of(rid)["attempt_id"],
            phase="submission_rejected",
            reason="Invalid account or account/partition combination",
        )
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped


def test_resume_with_stale_version_conflicts_then_reopens(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=6)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    v = _version(ctx, wf)
    with pytest.raises(VersionConflict) as exc:
        wfm.workflow_submit(
            ctx, idempotency_key="r0", workflow_id=wf, resume_dispatch=True, expected_version=v - 1
        )
    assert exc.value.details["current_version"] == v
    assert (
        one(ctx, "SELECT dispatch_state FROM workflows WHERE workflow_id=?", (wf,))["dispatch_state"]
        == "paused"
    )
    data = wfm.workflow_submit(
        ctx, idempotency_key="r1", workflow_id=wf, resume_dispatch=True, expected_version=v
    )
    assert data.resumed and data.dispatch_state == "open" and data.version == v + 1
    # same key replays the same answer; the acknowledged failures do not re-trip
    again = wfm.workflow_submit(
        ctx, idempotency_key="r1", workflow_id=wf, resume_dispatch=True, expected_version=v
    )
    assert again == data
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert not r.tripped and r.dispatched == 2
    # but two NEW identical failures do
    lab.fail(m.run_ids[2], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[3], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped
    bj = jload(one(ctx, "SELECT breaker_json FROM workflows WHERE workflow_id=?", (wf,))["breaker_json"])
    assert {c["run_id"] for c in bj["cells"]} == set(m.run_ids[2:4]) and len(bj["history"]) == 1


def test_resume_without_pause_is_a_noop(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    m = lab.matrix(n=2)
    wf = _submit(ctx, m)
    data = wfm.workflow_submit(
        ctx, idempotency_key="r", workflow_id=wf, resume_dispatch=True, expected_version=1
    )
    assert not data.resumed and data.version == 1 and data.warnings


# ---------------------------------------------------------------------------
# workflow_cancel
# ---------------------------------------------------------------------------


def test_cancel_queued_only_keeps_inflight_and_closes_dispatch(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=5)
    wf = _submit(ctx, m, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.set_attempt(lab.attempt_of(m.run_ids[1])["attempt_id"], phase="observing", scheduler_state="RUNNING")
    with pytest.raises(VersionConflict):
        wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=99, idempotency_key="c0")
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=1, scope="queued_only", idempotency_key="c1"
    )
    assert data.state == "cancelling" and data.dispatch_state == "closed" and data.version == 2
    assert data.counts == {"cancelled": 3, "dispatched": 1, "active": 1}
    assert data.budget_kept_gpu_hours == pytest.approx(2.0)
    assert data.cells[0].cell_state in ("dispatched", "active")  # non-final cells listed first
    assert (
        wfm.workflow_cancel(
            ctx, workflow_id=wf, expected_version=1, scope="queued_only", idempotency_key="c1"
        )
        == data
    )
    assert sub.cancels == []  # queued_only never touches in-flight work
    op = one(ctx, "SELECT * FROM operations WHERE operation_id=?", (data.operation_id,))
    assert op["state"] == "succeeded"
    # dispatch is closed in the same transaction: later rounds admit nothing
    lab.finish(m.run_ids[0])
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    lab.finish(m.run_ids[1], state="FAILED", result="unchecked")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.state == "cancelled" and len(sub.admitted) == 2
    with pytest.raises(PreconditionFailed):  # cancelled dispatch is never reopened
        wfm.workflow_submit(
            ctx, idempotency_key="r", workflow_id=wf, resume_dispatch=True, expected_version=_version(ctx, wf)
        )


def test_cancel_all_managed_requests_verified_cancels(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=5)
    wf = _submit(ctx, m, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    staging = lab.attempt_of(m.run_ids[0])["attempt_id"]
    running = lab.attempt_of(m.run_ids[1])["attempt_id"]
    uncertain = lab.attempt_of(m.run_ids[2])["attempt_id"]
    lab.set_attempt(running, phase="observing", scheduler_state="RUNNING", job_id="1234")
    lab.set_attempt(uncertain, phase="submission_uncertain")
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=1, scope="all_managed", idempotency_key="c"
    )
    # 2 never-dispatched cells + the staging attempt, cancelled in the cancel transaction itself
    assert data.state == "cancelling" and data.counts["cancelled"] == 3
    assert any("uncertain" in n for n in data.notes)
    assert data.coverage.notes  # uncertain cancellation is pending reconciliation
    # every active attempt got its intent in that same transaction (request_cancel)
    assert {c["attempt_id"] for c in sub.cancels} == {staging, running, uncertain}
    assert all(c["via"] == "request_cancel" for c in sub.cancels)
    _drain(ctx)  # the workflow.cancel handler has nothing left to request
    assert len(sub.cancels) == 3
    cells = {c.run_id: c for c in wfm.workflow_get(ctx, workflow_id=wf).cells}
    assert cells[m.run_ids[0]].cell_state == "cancelled"  # never reached sbatch
    assert cells[m.run_ids[1]].cell_state == "active" and cells[m.run_ids[1]].cancel_state == "requested"
    assert cells[m.run_ids[2]].cell_state == "uncertain"  # not marked cancelled before evidence
    # a second pass does not re-request
    wfm.cancel_step(ctx, workflow_id=wf)
    assert len(sub.cancels) == 3
    assert one(ctx, "SELECT state FROM workflows WHERE workflow_id=?", (wf,))["state"] == "cancelling"
    # terminal evidence arrives -> cancelled
    lab.finish(m.run_ids[1], state="CANCELLED", result="unchecked")
    lab.set_attempt(uncertain, phase="abandoned")
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.state == "cancelled"
    cop = one(ctx, "SELECT state FROM operations WHERE operation_id=?", (data.operation_id,))
    assert cop["state"] == "succeeded"


def test_cancel_retries_when_an_attempt_moved(
    ctx: ServiceContext, lab: LabBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # handler fallback path: a submission module without the in-transaction request_cancel
    sub = install_fake_submission(monkeypatch, FakeSubmission(in_tx_cancel=False))
    m = lab.matrix(n=1)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    aid = lab.attempt_of(m.run_ids[0])["attempt_id"]
    lab.set_attempt(aid, phase="observing", scheduler_state="PENDING")
    wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c"
    )
    real = sub.run_cancel
    calls = {"n": 0}

    def flaky(ctx_, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            lab.set_attempt(aid, scheduler_state="RUNNING")  # races the request
        return real(ctx_, **kw)

    import sys

    sys.modules["hpclab.services.submission"].run_cancel = flaky  # type: ignore[attr-defined]
    res = wfm.cancel_step(ctx, workflow_id=wf)
    assert res.action == "retry"
    res = wfm.cancel_step(ctx, workflow_id=wf)
    assert res.action == "done"
    assert lab.attempt_of(m.run_ids[0])["cancel_state"] == "requested"


def test_cancel_escalation_and_completed_noop(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    wf = _submit(ctx, m)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.set_attempt(lab.attempt_of(m.run_ids[0])["attempt_id"], phase="observing", scheduler_state="RUNNING")
    # the round moved the workflow dispatching -> draining: a control-state change bumps the version
    assert _version(ctx, wf) == 2
    first = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=2, scope="queued_only", idempotency_key="a"
    )
    again = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=first.version, scope="queued_only", idempotency_key="b"
    )
    assert again.version == first.version and again.notes  # nothing to do
    esc = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=first.version, scope="all_managed", idempotency_key="c"
    )
    assert esc.version == first.version + 1 and esc.scope == "all_managed"
    _drain(ctx)
    assert len(sub.cancels) == 2
    # a completed workflow keeps its terminal states
    m2 = lab.matrix(n=1)
    wf2 = _submit(ctx, m2)
    wfm.dispatch_workflow(ctx, workflow_id=wf2)
    lab.finish(m2.run_ids[0])
    wfm.dispatch_workflow(ctx, workflow_id=wf2)
    v = _version(ctx, wf2)
    d = wfm.workflow_cancel(ctx, workflow_id=wf2, expected_version=v, idempotency_key="d")
    assert d.state == "completed" and d.counts == {"succeeded": 1} and d.version == v


# ---------------------------------------------------------------------------
# workflow_get
# ---------------------------------------------------------------------------


def test_workflow_get_pages_codes_grid_and_coverage(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock
) -> None:
    m = lab.matrix(n=5)
    wf = _submit(ctx, m, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.set_attempt(
        lab.attempt_of(m.run_ids[0])["attempt_id"],
        phase="observing",
        scheduler_state="RUNNING",
        last_observed_at="2026-09-01T12:00:00.000Z",
    )
    page1 = wfm.workflow_get(ctx, workflow_id=wf, limit=3)
    assert [c.run_id for c in page1.cells] == m.run_ids[:3] and page1.next_cursor
    assert page1.grid is not None and len(page1.grid) == 2  # methods m0/m1 x seeds
    assert (
        page1.codes["planned"] == 5
        and page1.codes["R"] == 1
        and page1.codes["S"] == 1
        and page1.codes["P"] == 3
    )
    assert page1.slots_used == 2 and page1.counts == {"active": 1, "dispatched": 1, "planned": 3}
    assert page1.coverage.complete and page1.observed_at == "2026-09-01T12:00:00.000Z"
    page2 = wfm.workflow_get(ctx, workflow_id=wf, cursor=page1.next_cursor, limit=3)
    assert (
        [c.run_id for c in page2.cells] == m.run_ids[3:] and page2.next_cursor is None and page2.grid is None
    )
    clock.advance(3600)  # the running cell has not been observed for an hour
    stale = wfm.workflow_get(ctx, workflow_id=wf)
    assert not stale.coverage.complete
    assert {c.run_id: c.code for c in stale.cells}[m.run_ids[0]] == "?"
    with pytest.raises(NotFound):
        wfm.workflow_get(ctx, workflow_id="wf_missing")


# ---------------------------------------------------------------------------
# version semantics: decisions, not dispatch rounds
# ---------------------------------------------------------------------------


def test_cancel_with_a_version_read_before_several_dispatch_rounds_succeeds(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=5)
    submitted = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=1)
    wf, seen = submitted.workflow_id, submitted.version
    for i in range(3):  # dispatch rounds + per-cell progress while the user reads the workflow
        wfm.dispatch_workflow(ctx, workflow_id=wf)
        aid = lab.attempt_of(m.run_ids[i])["attempt_id"]
        lab.set_attempt(aid, phase="observing", scheduler_state="RUNNING")
        wfm.dispatch_workflow(ctx, workflow_id=wf)
        lab.finish(m.run_ids[i])
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert len(sub.admitted) == 4 and _version(ctx, wf) == seen
    assert wfm.workflow_get(ctx, workflow_id=wf).state == "dispatching"  # one cell still planned
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=seen, scope="queued_only", idempotency_key="c"
    )
    assert data.state in ("cancelling", "cancelled") and data.version > seen


def test_a_real_workflow_state_change_still_conflicts(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = _submit(ctx, m, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    seen = _version(ctx, wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped  # dispatch paused: a decision point
    assert _version(ctx, wf) == seen + 1
    with pytest.raises(VersionConflict) as exc:
        wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=seen, idempotency_key="c")
    assert exc.value.details["current_version"] == seen + 1


def test_resume_hints_are_valid_calls_with_the_current_version(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=4)
    wf = _submit(ctx, m, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped
    v = _version(ctx, wf)
    want = (
        f'workflow_submit(workflow_id="{wf}", resume_dispatch=true, expected_version={v}, '
        'idempotency_key="<new key>")'
    )
    data = wfm.workflow_get(ctx, workflow_id=wf)
    assert data.breaker is not None and data.breaker.resume_hint == want
    op = one(ctx, "SELECT error_json FROM operations WHERE operation_id=?", (data.operation_id,))
    assert want in jload(op["error_json"])["details"]["remediation"]
    # following the hint verbatim works
    resumed = wfm.workflow_submit(
        ctx, idempotency_key="r", workflow_id=wf, resume_dispatch=True, expected_version=v
    )
    assert resumed.resumed and resumed.dispatch_state == "open"
