"""run_cancel + cancel.execute (invariant 4.4): live identity re-verified right before the
scheduler call; any mismatch is REFUSED with zero cancellations."""

from __future__ import annotations

import json
from typing import Any

import pytest
from submission_support import (
    FakeScheduler,
    UnitEnv,
    attempt,
    drain,
    events,
    obs,
    operation,
    outbox,
    reservation,
    run_row,
    seed_run,
)

from hpclab.adapters.base import ConnectionLost, SubmissionRejected
from hpclab.domain.errors import IdempotencyConflict, PreconditionFailed, VersionConflict
from hpclab.domain.models import ResilienceSpec
from hpclab.services import submission
from hpclab.services.submission import attempt_resolve, run_cancel, run_submit


def running(unit: UnitEnv, **seed_kw: Any) -> dict[str, Any]:
    """A submitted attempt bound to fake job 5000, visible in squeue as RUNNING."""
    s = seed_run(unit.ctx, **seed_kw)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key=s.run_id)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "observing"
    unit.scheduler.queue = [live(a)]
    return a


def live(a: dict[str, Any], **kw: Any) -> Any:
    kw.setdefault("submit_time", a["dispatch_at"])
    kw.setdefault("state", "RUNNING")
    return obs(str(a["job_id"] or "5000"), name=a["job_name"], comment=f"hpclab:{a['submit_token']}", **kw)


def cancel(unit: UnitEnv, a: dict[str, Any], key: str = "c", **kw: Any) -> submission.CancelReceiptData:
    version = attempt(unit.ctx, a["attempt_id"])["version"]
    return run_cancel(
        unit.ctx, attempt_id=a["attempt_id"], expected_version=version, idempotency_key=key, **kw
    )


def test_verified_identity_is_cancelled_once(unit: UnitEnv) -> None:
    a = running(unit)
    receipt = cancel(unit, a, reason="wrong lr")
    assert receipt.cancel_state == "requested" and receipt.operation_id is not None
    assert receipt.note.startswith("cancel requested")
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"]
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "sent" and b["cancel_reason"] == "wrong lr" and b["phase"] == "observing"
    op = operation(unit.ctx, receipt.operation_id)
    assert op["kind"] == "cancel" and op["state"] == "succeeded" and op["stage"] == "sent"
    assert events(unit.ctx, a["attempt_id"], "attempt.cancel") == [
        "attempt.cancel_requested",
        "attempt.cancel_sent",
    ]
    # the claim is only released when an observation shows the job terminal (observe's job)
    assert run_row(unit.ctx, a["run_id"])["claim_attempt_id"] == a["attempt_id"]


@pytest.mark.parametrize(
    "change",
    [
        {"user": "mallory"},
        {"name": "hl.cell.seed-1.zzzzzzzz"},
        {"name": "someone-elses-job"},
        {"comment": "hpclab:zzzzzzzzzzzz"},
        {"submit_time": "2020-01-01T00:00:00.000Z"},
        {"submit_time": None},
        {"restart_count": 0, "_attempt": {"restart_count": 2}},
        {"_attempt": {"job_submit_time": "2020-01-01T00:00:00.000Z"}},
        {"_transport_user": "bob"},
    ],
    ids=[
        "owner",
        "name_token",
        "foreign_name",
        "comment_token",
        "submit_time_window",
        "submit_time_unknown",
        "restart_count_went_back",
        "recorded_submit_time",
        "controller_user_changed",
    ],
)
def test_identity_mismatch_is_refused_with_zero_scheduler_calls(
    unit: UnitEnv, change: dict[str, Any]
) -> None:
    a = running(unit)
    change = dict(change)
    if "_attempt" in change:
        with unit.ctx.db.tx() as tx:
            for col, value in change.pop("_attempt").items():
                tx.execute(f"UPDATE attempts SET {col}=? WHERE attempt_id=?", (value, a["attempt_id"]))
    if "_transport_user" in change:
        unit.transport.user = change.pop("_transport_user")
    unit.scheduler.queue = [live(a).model_copy(update=change)]
    receipt = cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert unit.scheduler.signal_calls == []
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "refused"
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "failed" and op["stage"] == "identity_mismatch"
    assert json.loads(op["error_json"])["code"] == "unauthorized"
    assert json.loads(op["error_json"])["details"]["problems"]
    assert "attempt.cancel_refused" in events(unit.ctx, a["attempt_id"])


def test_job_already_finished_is_confirmed_without_scancel(unit: UnitEnv) -> None:
    a = running(unit)
    unit.scheduler.queue = []
    unit.scheduler.acct = [live(a, state="COMPLETED", source="sacct")]
    cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "confirmed"


def test_invisible_job_is_retried_then_the_cancel_fails_honestly(unit: UnitEnv) -> None:
    unit.ctx.settings.extra["submission.cancel_max_tries"] = 3
    a = running(unit)
    unit.scheduler.queue = []
    receipt = cancel(unit, a)
    for _ in range(5):
        drain(unit.ctx)
        unit.clock.advance(6)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "failed"
    assert operation(unit.ctx, receipt.operation_id or "")["stage"] == "job_not_visible"


def test_scheduler_refusal_marks_the_cancel_failed(unit: UnitEnv) -> None:
    a = running(unit)
    unit.scheduler.cancel_accepts = False
    cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"]
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "failed"


def test_lost_scancel_reply_is_retried_after_verifying_again(unit: UnitEnv) -> None:
    a = running(unit)
    unit.scheduler.cancel_error = ConnectionLost("scancel: no answer")
    cancel(unit, a)
    drain(unit.ctx)
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "requested"
    checks = len(unit.scheduler.queue_calls)
    unit.scheduler.cancel_error = None
    unit.clock.advance(600)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000", "5000"]
    assert len(unit.scheduler.queue_calls) > checks  # identity verified again before resending
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "sent"


def test_auth_during_cancel_blocks_without_scancel(unit: UnitEnv) -> None:
    a = running(unit)
    unit.transport.auth_required = True
    receipt = cancel(unit, a)
    drain(unit.ctx)
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "blocked" and op["stage"] == "auth_required"
    assert unit.scheduler.cancel_calls == []
    unit.transport.auth_required = False
    unit.clock.advance(121)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"]
    assert operation(unit.ctx, receipt.operation_id or "")["state"] == "succeeded"


def test_request_checkpoint_needs_checkpoint_and_exit(unit: UnitEnv) -> None:
    a = running(unit)
    with pytest.raises(PreconditionFailed):
        cancel(unit, a, mode="request_checkpoint")
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "none"


def test_request_checkpoint_sends_the_plan_signal_after_verification(unit: UnitEnv) -> None:
    a = running(
        unit, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", signal="USR2", resumable=True)
    )
    receipt = cancel(unit, a, mode="request_checkpoint", reason="deadline")
    drain(unit.ctx)
    assert unit.scheduler.signal_calls == [("5000", "USR2", True)]
    assert unit.scheduler.cancel_calls == []
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "sent" and b["cancel_mode"] == "request_checkpoint" and b["signal_sent_at"]
    assert operation(unit.ctx, receipt.operation_id or "")["stage"] == "signal_sent"


def test_request_checkpoint_without_scheduler_support_fails(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(FakeScheduler, "signal_job")
    a = running(unit, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", resumable=True))
    cancel(unit, a, mode="request_checkpoint")
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "failed"


def test_request_checkpoint_before_submission_is_refused(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", resumable=True))
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    with pytest.raises(PreconditionFailed):
        run_cancel(
            unit.ctx,
            attempt_id=r.attempt_id,
            expected_version=1,
            idempotency_key="c",
            mode="request_checkpoint",
        )


def test_cancel_while_uncertain_waits_for_reconcile(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "submission_uncertain"
    receipt = cancel(unit, a)
    assert receipt.cancel_state == "requested" and "not known yet" in receipt.note
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "queued" and op["stage"] == "waiting_for_reconcile"
    assert outbox(unit.ctx, a["attempt_id"], "cancel.execute") == []
    unit.scheduler.queue = [live(dict(a, job_id="7100"))]
    unit.clock.advance(3)
    drain(unit.ctx)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["job_id"] == "7100" and b["cancel_state"] == "sent"
    assert unit.scheduler.cancel_calls == ["7100"]
    assert operation(unit.ctx, receipt.operation_id or "")["state"] == "succeeded"


def test_cancel_during_the_sbatch_call_is_applied_after_identification(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")

    def submit_while_user_cancels(script: str, *, workdir: str, receipt_path: str | None) -> Any:
        a = attempt(unit.ctx, r.attempt_id)
        assert a["phase"] == "submission_in_flight"
        got = cancel(unit, a)
        assert got.cancel_state == "requested"
        unit.scheduler.submit_behaviors.insert(0, "accept")
        resp = unit.scheduler.submit(script, workdir=workdir, receipt_path=receipt_path)
        unit.scheduler.queue = [live(dict(attempt(unit.ctx, r.attempt_id), job_id=resp.job_id))]
        return resp

    unit.scheduler.submit_behaviors = [submit_while_user_cancels]
    drain(unit.ctx)
    b = attempt(unit.ctx, r.attempt_id)
    assert b["phase"] == "observing" and b["cancel_state"] == "sent"
    assert unit.scheduler.cancel_calls == ["5000"]


def test_cancel_in_flight_then_refused_by_sbatch_needs_nothing_more(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    holder: dict[str, Any] = {}

    def refuse_while_user_cancels(script: str, *, workdir: str, receipt_path: str | None) -> Any:
        holder["receipt"] = cancel(unit, attempt(unit.ctx, r.attempt_id))
        raise SubmissionRejected("refused", details={"reason_code": "invalid_qos"})

    unit.scheduler.submit_behaviors = [refuse_while_user_cancels]
    drain(unit.ctx)
    b = attempt(unit.ctx, r.attempt_id)
    assert b["phase"] == "submission_rejected" and b["cancel_state"] == "confirmed"
    op = operation(unit.ctx, holder["receipt"].operation_id)
    assert op["state"] == "succeeded" and op["stage"] == "nothing_queued"
    assert unit.scheduler.cancel_calls == []


def test_cancel_duplicate_waits_for_the_human_then_cancels_only_the_bound_job(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    unit.scheduler.queue = [live(dict(a, job_id="7200")), live(dict(a, job_id="7201"))]
    unit.clock.advance(3)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "duplicate_detected"
    receipt = cancel(unit, a)
    assert operation(unit.ctx, receipt.operation_id or "")["state"] == "blocked"
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    a = attempt(unit.ctx, r.attempt_id)
    attempt_resolve(
        unit.ctx,
        attempt_id=a["attempt_id"],
        expected_version=a["version"],
        resolution="bind_job",
        reason="7200 is the one I want",
        idempotency_key="r",
        job_id="7200",
    )
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["7200"]
    assert attempt(unit.ctx, r.attempt_id)["cancel_state"] == "sent"


def test_cancel_before_submit_is_immediate_and_idempotent(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    first = run_cancel(
        unit.ctx, attempt_id=r.attempt_id, expected_version=1, idempotency_key="c", reason="oops"
    )
    again = run_cancel(
        unit.ctx, attempt_id=r.attempt_id, expected_version=1, idempotency_key="c", reason="oops"
    )
    assert again == first and first.phase == "cancelled_before_submit" and first.operation_id is None
    with pytest.raises(IdempotencyConflict):
        run_cancel(unit.ctx, attempt_id=r.attempt_id, expected_version=1, idempotency_key="c", reason="other")
    assert run_row(unit.ctx, s.run_id)["claim_attempt_id"] is None
    assert reservation(unit.ctx, r.attempt_id)["state"] == "released"
    assert operation(unit.ctx, r.operation_id)["state"] == "failed"
    drain(unit.ctx)
    assert unit.scheduler.submit_calls == []


def test_version_conflict_and_nothing_to_cancel(unit: UnitEnv) -> None:
    a = running(unit)
    with pytest.raises(VersionConflict):
        run_cancel(unit.ctx, attempt_id=a["attempt_id"], expected_version=1, idempotency_key="c")
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE attempts SET phase='terminal' WHERE attempt_id=?", (a["attempt_id"],))
    with pytest.raises(PreconditionFailed):
        cancel(unit, a, key="c2")


def test_repeated_cancel_request_reuses_the_pending_operation(unit: UnitEnv) -> None:
    a = running(unit)
    unit.transport.auth_required = True  # keeps the first request pending
    first = cancel(unit, a, key="c1")
    drain(unit.ctx)
    second = cancel(unit, a, key="c2")
    assert second.operation_id == first.operation_id and "already pending" in second.note


def test_terminal_before_execution_confirms_without_scancel(unit: UnitEnv) -> None:
    a = running(unit)
    cancel(unit, a)
    with unit.ctx.db.tx() as tx:  # observe recorded the end before the cancel step ran
        tx.execute(
            "UPDATE attempts SET phase='terminal', scheduler_state='COMPLETED' WHERE attempt_id=?",
            (a["attempt_id"],),
        )
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "confirmed"


def test_signal_unsupported_by_the_cluster_fails_the_request(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hpclab.adapters.base import TransportError
    from hpclab.domain.errors import ErrorCode

    def unsupported(job_id: str, signal: str, *, batch: bool = True) -> Any:
        raise TransportError("scancel --signal not allowed here", code=ErrorCode.UNSUPPORTED, retryable=False)

    monkeypatch.setattr(unit.scheduler, "signal_job", unsupported)
    a = running(unit, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", resumable=True))
    receipt = cancel(unit, a, mode="request_checkpoint")
    drain(unit.ctx)
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "failed"
    assert operation(unit.ctx, receipt.operation_id or "")["stage"] == "unsupported"


def test_a_newer_request_supersedes_the_pending_one(unit: UnitEnv) -> None:
    a = running(unit, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", resumable=True))
    unit.transport.auth_required = True  # keep the first request pending
    first = cancel(unit, a, key="c1", mode="request_checkpoint")
    drain(unit.ctx)
    second = cancel(unit, a, key="c2", mode="cancel")
    assert second.operation_id != first.operation_id
    assert operation(unit.ctx, first.operation_id or "")["stage"] == "superseded"
    unit.transport.auth_required = False
    unit.clock.advance(121)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"] and unit.scheduler.signal_calls == []
    assert operation(unit.ctx, second.operation_id or "")["state"] == "succeeded"
    assert operation(unit.ctx, first.operation_id or "")["state"] == "failed"


def test_intent_confirmed_by_an_observation_closes_the_operation(unit: UnitEnv) -> None:
    a = running(unit)
    receipt = cancel(unit, a)
    with unit.ctx.db.tx() as tx:  # observe saw the job end first and confirmed the intent
        tx.execute(
            "UPDATE attempts SET phase='terminal', scheduler_state='CANCELLED', cancel_state='confirmed' "
            "WHERE attempt_id=?",
            (a["attempt_id"],),
        )
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "succeeded" and op["stage"] == "already_terminal"


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def _later(ts: str, seconds: float) -> str:
    from datetime import timedelta

    from hpclab.domain.clock import iso, parse_iso

    parsed = parse_iso(ts)
    assert parsed is not None
    out = iso(parsed + timedelta(seconds=seconds))
    assert out is not None
    return out


@pytest.mark.parametrize(
    ("ledger", "live_restart"),
    [
        ({"restart_count": 1, "job_submit_time": "dispatch"}, 1),  # observe recorded the requeue
        ({"restart_count": 0, "job_submit_time": "dispatch"}, 1),  # requeued since our last poll
        ({"restart_count": 0, "job_submit_time": None}, 2),  # bound from the sbatch reply, then requeued
    ],
    ids=["requeue_recorded", "requeue_not_yet_observed", "no_recorded_submit_time"],
)
def test_a_requeued_job_with_a_new_submit_time_is_still_cancelled(
    unit: UnitEnv, ledger: dict[str, Any], live_restart: int
) -> None:
    """Slurm stamps a new SubmitTime on every requeue (slurmdbd keys a job record by id +
    submit time, which is how ``sacct -D`` lists the segments). Job id, owner and token do not
    change, so a requeued job must stay cancellable - otherwise it burns GPUs, uncancellable."""
    a = running(unit)
    values = {k: (a["dispatch_at"] if v == "dispatch" else v) for k, v in ledger.items()}
    with unit.ctx.db.tx() as tx:
        for col, value in values.items():
            tx.execute(f"UPDATE attempts SET {col}=? WHERE attempt_id=?", (value, a["attempt_id"]))
    unit.clock.advance(3 * 3600 + 60)  # preempted and requeued three hours later
    unit.scheduler.queue = [
        live(a, submit_time=_later(a["dispatch_at"], 3 * 3600), restart_count=live_restart)
    ]
    cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"]
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "sent"


def test_a_finished_requeued_job_seen_only_in_accounting_is_confirmed(unit: UnitEnv) -> None:
    """sacct reports the latest segment's Submit and no restart count."""
    a = running(unit)
    with unit.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET restart_count=1, job_submit_time=? WHERE attempt_id=?",
            (a["dispatch_at"], a["attempt_id"]),
        )
    unit.clock.advance(1900)
    unit.scheduler.queue = []
    unit.scheduler.acct = [
        live(
            a,
            state="COMPLETED",
            source="sacct",
            restart_count=None,
            submit_time=_later(a["dispatch_at"], 1800),
        )
    ]
    cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "confirmed"


@pytest.mark.parametrize(
    "change",
    [
        {"submit_time": "2020-01-01T00:00:00.000Z", "restart_count": 1},  # before the recorded one
        {"restart_count": 1, "name": "hl.cell.seed-1.zzzzzzzz"},  # requeue does not excuse the token
        {"restart_count": 1, "user": "mallory"},
    ],
    ids=["requeued_but_earlier", "requeued_other_token", "requeued_other_owner"],
)
def test_a_requeue_never_excuses_a_real_identity_mismatch(unit: UnitEnv, change: dict[str, Any]) -> None:
    a = running(unit)
    with unit.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET restart_count=1, job_submit_time=? WHERE attempt_id=?",
            (a["dispatch_at"], a["attempt_id"]),
        )
    unit.scheduler.queue = [live(a).model_copy(update=change)]
    cancel(unit, a)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "refused"


def test_intent_confirmed_while_scancel_runs_still_finishes_the_operation(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """observe (another worker) confirms the intent between the identity check and the
    outcome commit: the cancel operation must still reach a final state."""
    a = running(unit)
    receipt = cancel(unit, a)
    real = unit.scheduler.cancel

    def observe_confirms_meanwhile(job_id: str) -> Any:
        with unit.ctx.db.tx() as tx:
            tx.execute(
                "UPDATE attempts SET phase='terminal', scheduler_state='CANCELLED', "
                "cancel_state='confirmed', version=version+1 WHERE attempt_id=?",
                (a["attempt_id"],),
            )
        return real(job_id)

    monkeypatch.setattr(unit.scheduler, "cancel", observe_confirms_meanwhile)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == ["5000"]
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "succeeded", (op["state"], op["stage"])
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "confirmed"
    assert "attempt.cancel_sent" in events(unit.ctx, a["attempt_id"])  # the scancel is audited


def test_a_request_superseded_while_the_step_runs_is_executed_too(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer request (another mode) arrives while cancel.execute is leased: its operation
    must not be left waiting forever, and the old outcome must not overwrite the new intent."""
    a = running(unit, resilience=ResilienceSpec(stop_behavior="checkpoint_and_exit", resumable=True))
    first = cancel(unit, a, key="c1", mode="request_checkpoint")
    real = unit.scheduler.signal_job
    later: list[submission.CancelReceiptData] = []

    def user_changes_mind(job_id: str, signal: str, *, batch: bool = True) -> Any:
        if not later:
            later.append(cancel(unit, a, key="c2", mode="cancel"))
        return real(job_id, signal, batch=batch)

    monkeypatch.setattr(unit.scheduler, "signal_job", user_changes_mind)
    drain(unit.ctx)
    assert unit.scheduler.signal_calls == [("5000", "USR1", True)]
    assert unit.scheduler.cancel_calls == ["5000"]  # the newer request ran as well
    second = operation(unit.ctx, later[0].operation_id or "")
    assert second["state"] == "succeeded" and second["stage"] == "sent"
    assert operation(unit.ctx, first.operation_id or "")["state"] == "failed"  # superseded stays failed
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "sent" and b["cancel_mode"] == "cancel"
    assert "attempt.checkpoint_signal_sent" in events(unit.ctx, a["attempt_id"])  # audited anyway


@pytest.mark.parametrize("outcome", ["duplicate", "unresolved"])
def test_a_waiting_cancel_follows_the_attempt_into_a_human_decision(unit: UnitEnv, outcome: str) -> None:
    """Cancel requested while uncertain; reconcile then needs a human (duplicate jobs, or
    nothing found in the window). The cancel operation must say so (blocked, waiting for the
    resolution) instead of sitting 'queued' forever as if a background step would run it."""
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    receipt = cancel(unit, a)
    if outcome == "duplicate":
        unit.scheduler.queue = [live(dict(a, job_id="7300")), live(dict(a, job_id="7301"))]
        unit.clock.advance(3)
        drain(unit.ctx)
    else:
        for _ in range(20):
            unit.clock.advance(30)
            drain(unit.ctx)
    submit_op = operation(unit.ctx, a["operation_id"])
    assert submit_op["state"] == "blocked" and submit_op["stage"] in ("duplicate_detected", "unresolved")
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "blocked" and op["stage"] == "waiting_for_resolution", (op["state"], op["stage"])
    assert unit.scheduler.cancel_calls == []
    # the human abandons it: the cancel ends honestly (nothing was identified to cancel)
    a = attempt(unit.ctx, r.attempt_id)
    attempt_resolve(
        unit.ctx,
        attempt_id=a["attempt_id"],
        expected_version=a["version"],
        resolution="abandon",
        reason="gave up",
        idempotency_key="r",
    )
    assert operation(unit.ctx, receipt.operation_id or "")["state"] == "failed"


def test_a_cancel_requested_after_the_verdict_unresolved_is_blocked_at_once(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    for _ in range(20):
        unit.clock.advance(30)
        drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert operation(unit.ctx, a["operation_id"])["stage"] == "unresolved"
    receipt = cancel(unit, a)
    op = operation(unit.ctx, receipt.operation_id or "")
    assert op["state"] == "blocked" and op["stage"] == "waiting_for_resolution"
    assert "human" in receipt.note
