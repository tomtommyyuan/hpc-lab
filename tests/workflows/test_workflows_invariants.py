"""Invariant checks for workflows found in review (spec §4.9 events, tools.md §15 cancel intent,
honest operation progress, bounded and honest responses)."""

from __future__ import annotations

from typing import Any

import pytest
from wf_support import FakeSubmission, LabBuilder, install_fake_submission, one

from hpclab.api.envelope import encode_cursor
from hpclab.domain.errors import InvalidArgument
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import OutboxItem
from hpclab.storage.db import jload


def _version(ctx: ServiceContext, wf: str) -> int:
    return int(one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"])


def _versions_with_events(ctx: ServiceContext, wf: str) -> set[int]:
    rows = ctx.db.all(
        "SELECT entity_version FROM events WHERE entity_type='workflow' AND entity_id=? "
        "AND entity_version IS NOT NULL",
        (wf,),
    )
    return {int(r["entity_version"]) for r in rows}


def _submit(ctx: ServiceContext, plan_id: str, max_parallel: int = 3) -> str:
    return wfm.workflow_submit(
        ctx, idempotency_key=f"k-{plan_id}", plan_id=plan_id, max_parallel=max_parallel
    ).workflow_id


# ---------------------------------------------------------------------------
# §4.9: every state change of the workflow row writes an event in the same tx
# ---------------------------------------------------------------------------


def test_every_workflow_version_has_an_event(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2, smoke=True)
    wf = _submit(ctx, m.plan_id)
    wfm.dispatch_workflow(ctx, workflow_id=wf)  # gating: smoke only
    lab.finish(m.smoke)
    wfm.dispatch_workflow(ctx, workflow_id=wf)  # gate passed -> dispatching -> draining
    for rid in m.run_ids:
        lab.finish(rid)
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)  # draining -> completed
    assert r.state == "completed"
    v = _version(ctx, wf)
    assert v >= 3
    missing = set(range(1, v + 1)) - _versions_with_events(ctx, wf)
    assert not missing, f"workflow versions without an event: {sorted(missing)}"
    changes = [
        jload(e["payload_json"])
        for e in ctx.db.all(
            "SELECT payload_json FROM events WHERE entity_id=? AND event_type='workflow.state_changed' ORDER BY seq",
            (wf,),
        )
    ]
    assert [(c["from"], c["to"]) for c in changes] == [("gating", "draining"), ("draining", "completed")]


def test_breaker_resume_and_cancel_versions_all_have_events(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    from wf_support import IMPORT_ERR

    m = lab.matrix(n=6)
    wf = _submit(ctx, m.plan_id, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped
    wfm.workflow_submit(
        ctx, idempotency_key="r", workflow_id=wf, resume_dispatch=True, expected_version=_version(ctx, wf)
    )
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.set_attempt(lab.attempt_of(m.run_ids[2])["attempt_id"], phase="observing", scheduler_state="RUNNING")
    wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="queued_only", idempotency_key="c"
    )
    wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c2"
    )
    for rid in m.run_ids[2:4]:
        lab.finish(rid, state="CANCELLED", result="unchecked")
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).state == "cancelled"
    v = _version(ctx, wf)
    missing = set(range(1, v + 1)) - _versions_with_events(ctx, wf)
    assert not missing, f"workflow versions without an event: {sorted(missing)}"


def test_dispatching_to_draining_is_recorded(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    wf = _submit(ctx, m.plan_id)
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.state == "draining" and _version(ctx, wf) == 2
    ev = one(
        ctx,
        "SELECT * FROM events WHERE entity_id=? AND event_type='workflow.state_changed'",
        (wf,),
    )
    assert ev["entity_version"] == 2
    assert jload(ev["payload_json"]) == {"from": "dispatching", "to": "draining"}


# ---------------------------------------------------------------------------
# tools.md §15: all_managed writes the per-attempt cancel intent in the cancel transaction
# ---------------------------------------------------------------------------


def test_all_managed_cancel_intent_is_written_in_the_cancel_transaction(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=5)
    wf = _submit(ctx, m.plan_id, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    staging = lab.attempt_of(m.run_ids[0])["attempt_id"]
    running = lab.attempt_of(m.run_ids[1])["attempt_id"]
    uncertain = lab.attempt_of(m.run_ids[2])["attempt_id"]
    lab.set_attempt(running, phase="observing", scheduler_state="RUNNING", job_id="1234")
    lab.set_attempt(uncertain, phase="submission_uncertain")
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c"
    )
    # no outbox step ran: the submit worker (which re-checks the attempt's cancel intent right
    # before sbatch) must already see the intent, or it would submit a job of a cancelled workflow
    a = {r["attempt_id"]: r for r in ctx.db.all("SELECT * FROM attempts")}
    assert a[staging]["phase"] == "cancelled_before_submit"
    assert a[running]["cancel_state"] == "requested" and a[running]["phase"] == "observing"
    assert a[uncertain]["cancel_state"] == "requested" and a[uncertain]["phase"] == "submission_uncertain"
    assert {c["attempt_id"] for c in sub.cancels} == {staging, running, uncertain}
    # the response reflects it: the staging cell is cancelled, the others are still in flight
    assert data.counts == {"cancelled": 3, "active": 1, "uncertain": 1}
    assert data.state == "cancelling"
    op = one(ctx, "SELECT * FROM operations WHERE operation_id=?", (data.operation_id,))
    prog = jload(op["progress_json"])
    assert prog == {"requested": 2, "total": 2}
    # the handler pass has nothing left to request
    wfm.cancel_step(ctx, workflow_id=wf)
    assert len(sub.cancels) == 3


def test_all_managed_cancel_of_only_staging_work_finishes_at_once(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=3)
    wf = _submit(ctx, m.plan_id, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c"
    )
    assert data.state == "cancelled" and data.counts == {"cancelled": 3}
    assert data.budget_kept_gpu_hours == 0.0
    cop = one(ctx, "SELECT state, stage FROM operations WHERE operation_id=?", (data.operation_id,))
    assert cop["state"] == "succeeded" and cop["stage"] == "done"
    assert one(ctx, "SELECT COUNT(*) AS n FROM outbox WHERE kind='workflow.cancel'")["n"] == 0


def test_cancel_handler_progress_is_never_faked(
    ctx: ServiceContext, lab: LabBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a submission module without request_cancel: the workflow.cancel handler does the requests
    sub = install_fake_submission(monkeypatch, FakeSubmission(in_tx_cancel=False))
    m = lab.matrix(n=3)
    wf = _submit(ctx, m.plan_id, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    for rid in m.run_ids:
        lab.set_attempt(lab.attempt_of(rid)["attempt_id"], phase="observing", scheduler_state="RUNNING")
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c"
    )
    assert sub.cancels == []  # nothing requested in the transaction without request_cancel
    op0 = jload(
        one(ctx, "SELECT progress_json FROM operations WHERE operation_id=?", (data.operation_id,))[
            "progress_json"
        ]
    )
    assert op0 == {"requested": 0, "total": 3}
    assert wfm.cancel_step(ctx, workflow_id=wf).action == "done"
    assert len(sub.cancels) == 3 and all(c["via"] == "run_cancel" for c in sub.cancels)
    prog = jload(
        one(ctx, "SELECT progress_json FROM operations WHERE operation_id=?", (data.operation_id,))[
            "progress_json"
        ]
    )
    assert prog == {"requested": 3, "total": 3}


def test_a_finished_cancellation_is_never_reopened(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=2)
    wf = _submit(ctx, m.plan_id)
    first = wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=1, idempotency_key="a")
    assert first.state == "cancelled"
    ops = one(ctx, "SELECT COUNT(*) AS n FROM operations WHERE kind='workflow_cancel'")["n"]
    esc = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=first.version, scope="all_managed", idempotency_key="b"
    )
    assert esc.state == "cancelled" and esc.version == first.version and esc.notes
    assert one(ctx, "SELECT COUNT(*) AS n FROM operations WHERE kind='workflow_cancel'")["n"] == ops
    assert _version(ctx, wf) == first.version


# ---------------------------------------------------------------------------
# bounded + honest responses
# ---------------------------------------------------------------------------


def test_cancel_response_discloses_truncated_cells(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=wfm.MAX_CANCEL_CELLS + 5)
    wf = _submit(ctx, m.plan_id, max_parallel=2)
    data = wfm.workflow_cancel(ctx, workflow_id=wf, expected_version=1, idempotency_key="c")
    assert len(data.cells) == wfm.MAX_CANCEL_CELLS
    assert sum(data.counts.values()) == wfm.MAX_CANCEL_CELLS + 5
    assert any("workflow_get" in n and str(wfm.MAX_CANCEL_CELLS) in n for n in data.notes)
    assert any(str(wfm.MAX_CANCEL_CELLS) in n for n in data.coverage.notes)
    ev = one(
        ctx,
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='workflow.cancel_requested'",
        (wf,),
    )
    assert set(jload(ev["payload_json"])["cancelled_run_ids"]) == set(m.run_ids)


@pytest.mark.parametrize(
    "cursor", [encode_cursor({"o": "x"}), encode_cursor({"o": [1]}), encode_cursor({"o": -3})]
)
def test_workflow_get_rejects_malformed_cursors(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, cursor: str
) -> None:
    wf = _submit(ctx, lab.matrix(n=2).plan_id)
    with pytest.raises(InvalidArgument):
        wfm.workflow_get(ctx, workflow_id=wf, cursor=cursor)


# ---------------------------------------------------------------------------
# with the real submission module: the submit worker never sbatches a cancelled workflow's cell
# ---------------------------------------------------------------------------


def test_real_submit_worker_skips_a_cell_cancelled_with_all_managed(
    ctx: ServiceContext, lab: LabBuilder
) -> None:
    real: Any = pytest.importorskip("hpclab.services.submission")
    m = lab.matrix(n=3)
    wf = _submit(ctx, m.plan_id, max_parallel=2)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    staged = [lab.attempt_of(r)["attempt_id"] for r in m.run_ids[:2]]
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=_version(ctx, wf), scope="all_managed", idempotency_key="c"
    )
    assert data.state == "cancelled"
    rows = {r["attempt_id"]: r for r in ctx.db.all("SELECT * FROM attempts")}
    assert all(rows[a]["phase"] == "cancelled_before_submit" for a in staged)
    # the queued submit steps now end without touching the cluster
    calls: list[Any] = []

    def no_cluster(*a: Any, **k: Any) -> Any:
        calls.append(a)
        raise AssertionError("the cluster must not be contacted for a cancelled attempt")

    ctx.clusters.transport = no_cluster  # type: ignore[method-assign]
    ctx.clusters.scheduler = no_cluster  # type: ignore[method-assign]
    for aid in staged:
        item = OutboxItem(
            outbox_id=0,
            kind="submit.dispatch",
            operation_id=None,
            target_id=aid,
            payload={"attempt_id": aid},
            tries=1,
        )
        res = real.handle_dispatch(ctx, item)
        assert res is None or res.action == "done"
    assert calls == []
    assert all(r["state"] != "reserved" for r in ctx.db.all("SELECT state FROM budget_reservations"))


# ---------------------------------------------------------------------------
# resume never silently ignores a parameter; stale claims do not park a cell forever
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"max_parallel": 1},
        {"failure_policy": wfm.FailurePolicy(circuit_breaker=wfm.CircuitBreakerPolicy(enabled=False))},
        {"plan_hash": "sha256:" + "0" * 64},
    ],
)
def test_resume_refuses_parameters_it_would_ignore(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, extra: dict[str, Any]
) -> None:
    from wf_support import IMPORT_ERR

    m = lab.matrix(n=6)
    wf = _submit(ctx, m.plan_id, max_parallel=3)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    lab.fail(m.run_ids[1], stderr=IMPORT_ERR)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).tripped
    v = _version(ctx, wf)
    with pytest.raises(InvalidArgument):
        wfm.workflow_submit(
            ctx, idempotency_key="r", workflow_id=wf, resume_dispatch=True, expected_version=v, **extra
        )
    wf_row = one(
        ctx, "SELECT dispatch_state, max_parallel, version FROM workflows WHERE workflow_id=?", (wf,)
    )
    assert wf_row["dispatch_state"] == "paused" and wf_row["max_parallel"] == 3 and wf_row["version"] == v


def test_a_stale_claim_is_left_to_the_admission_gate(ctx: ServiceContext, lab: LabBuilder) -> None:
    """A claim pointing at an attempt that already ended must not park the cell as "waiting":
    the real admission gate re-checks the holder and releases the stale claim."""
    pytest.importorskip("hpclab.services.submission")
    from hpclab.testing.factories import make_attempt

    m = lab.matrix(n=1)
    rid = m.run_ids[0]
    with ctx.db.tx() as tx:
        old = make_attempt(tx, rid, phase="terminal", scheduler_state="FAILED")
        tx.execute(
            "UPDATE runs SET claim_attempt_id=?, current_exec_plan_hash=? WHERE run_id=?",
            (old, m.exec_plans[rid], rid),
        )
    wf = _submit(ctx, m.plan_id, max_parallel=1)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 1
    run = one(ctx, "SELECT claim_attempt_id FROM runs WHERE run_id=?", (rid,))
    cell = one(ctx, "SELECT attempt_id FROM workflow_cells WHERE workflow_id=?", (wf,))
    assert run["claim_attempt_id"] == cell["attempt_id"] != old


def test_a_live_claim_outside_the_workflow_still_waits(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m = lab.matrix(n=1)
    rid = m.run_ids[0]
    other = lab.extra_exec_plan(rid)
    with ctx.db.tx() as tx:
        direct = sub.admit(tx, ctx, run_id=rid, exec_plan_hash=other)
    wf = _submit(ctx, m.plan_id, max_parallel=1)
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    cell = one(ctx, "SELECT attempt_id, note FROM workflow_cells WHERE workflow_id=?", (wf,))
    assert cell["attempt_id"] is None and direct.attempt_id in (cell["note"] or "")


def test_an_invalid_plan_is_not_reported_as_retryable(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    from hpclab.domain.errors import NotReady, PreconditionFailed

    invalid = lab.matrix(n=1, status="invalid")
    with pytest.raises(PreconditionFailed) as exc:
        wfm.workflow_submit(ctx, idempotency_key="a", plan_id=invalid.plan_id)
    assert not exc.value.retryable and exc.value.details["plan_status"] == "invalid"
    preparing = lab.matrix(n=1, status="preparing")
    with pytest.raises(NotReady) as exc2:  # still being prepared: worth retrying later
        wfm.workflow_submit(ctx, idempotency_key="b", plan_id=preparing.plan_id)
    assert exc2.value.retryable
