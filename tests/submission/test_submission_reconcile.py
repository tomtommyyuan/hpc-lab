"""submit.reconcile + the sweep (invariant 4.3): unique token match binds, duplicates are
flagged, nothing found is *unknown* - never resubmitted."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from submission_support import (
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

from hpclab.adapters.base import (
    CommandFailed,
    CommandResult,
    ConnectionLost,
    SubmissionRejected,
    SubmitResponse,
    TransportError,
)
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import ErrorCode
from hpclab.domain.models import JobObservation
from hpclab.services import budget
from hpclab.services.submission import reconcile_sweep, run_submit

WINDOW = 300 + 60  # unit policy: reconcile_window_s + accounting_lag_s


def uncertain(unit: UnitEnv) -> dict[str, Any]:
    """An attempt whose sbatch reply was lost after the scheduler accepted nothing we know of."""
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key=s.run_id)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "submission_uncertain"
    return a


def ours(a: dict[str, Any], job_id: str, **kw: Any) -> JobObservation:
    """A job carrying this attempt's token, submitted when it was dispatched."""
    kw.setdefault("submit_time", a["dispatch_at"])
    return obs(job_id, name=a["job_name"], comment=f"hpclab:{a['submit_token']}", **kw)


def step(unit: UnitEnv, seconds: float = 3.0) -> None:
    unit.clock.advance(seconds)
    drain(unit.ctx)


def test_receipt_job_confirmed_by_squeue_is_bound(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.receipt = SubmitResponse("7001", None, "7001", iso(unit.clock.now()) or "")
    unit.scheduler.queue = [ours(a, "7001")]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "observing" and b["job_id"] == "7001" and b["job_owner"] == "alice"
    assert b["job_submit_time"] is not None
    op = operation(unit.ctx, a["operation_id"])
    assert (
        op["state"] == "succeeded" and json.loads(op["result_json"])["source"] == "reconcile:receipt+squeue"
    )
    assert len(unit.scheduler.acct_calls) == 1  # accounting consulted too: a bind needs a UNIQUE match
    assert len(unit.scheduler.submit_calls) == 1
    assert run_row(unit.ctx, a["run_id"])["status"] == "queued"


def test_missing_receipt_squeue_token_match_is_bound(unit: UnitEnv) -> None:
    a = uncertain(unit)
    foreign = obs("6999", name="train-baseline", comment=None)
    other_attempt = obs("6998", name="hl.cell.seed-1.zzzzzzzz", comment="hpclab:zzzzzzzzzzzz")
    unit.scheduler.queue = [foreign, other_attempt, ours(a, "7002")]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "observing" and b["job_id"] == "7002"


def test_job_that_already_left_the_queue_is_found_in_accounting(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.acct = [ours(a, "7003", state="COMPLETED", source="sacct", exit_code=0)]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "observing" and b["job_id"] == "7003"
    call = unit.scheduler.acct_calls[0]
    assert call["name"] == a["job_name"] and call["user"] == "alice"
    dispatched = parse_iso(a["dispatch_at"])
    assert dispatched is not None
    assert call["since"] == iso(dispatched - timedelta(seconds=600))


def test_two_token_matches_are_a_duplicate_and_nothing_is_cancelled(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7004"), ours(a, "7005", state="RUNNING")]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "duplicate_detected"
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "duplicate_detected"
    err = json.loads(op["error_json"])
    assert err["code"] == ErrorCode.AMBIGUOUS
    assert [c["job_id"] for c in err["details"]["candidates"]] == ["7004", "7005"]
    assert "--bind-job" in err["details"]["remediation"]
    assert unit.scheduler.cancel_calls == []
    assert run_row(unit.ctx, a["run_id"])["claim_attempt_id"] == a["attempt_id"]
    assert reservation(unit.ctx, a["attempt_id"])["state"] == "reserved"
    # no further reconcile work is scheduled for a duplicate
    step(unit, 30)
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "duplicate_detected"


def test_receipt_job_plus_another_token_job_is_a_duplicate(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.receipt = SubmitResponse("7006", None, "7006", "")
    unit.scheduler.queue = [ours(a, "7007")]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "duplicate_detected"
    payload = unit.ctx.db.one(
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.duplicate_detected'",
        (a["attempt_id"],),
    )
    cands = json.loads(payload["payload_json"])["candidates"]  # type: ignore[index]
    assert {c["job_id"] for c in cands} == {"7006", "7007"}
    assert [c["from_receipt"] for c in cands if c["job_id"] == "7006"] == [True]


def test_exit_95_duplicates_are_marked_harmless_for_the_human(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7010", state="RUNNING"), ours(a, "7011", state="FAILED", exit_code=95)]
    step(unit)
    op = operation(unit.ctx, a["operation_id"])
    cands = {c["job_id"]: c for c in json.loads(op["error_json"])["details"]["candidates"]}
    assert cands["7011"]["harmless_duplicate"] is True and cands["7010"]["harmless_duplicate"] is False


@pytest.mark.parametrize(
    "kw",
    [
        {"user": "mallory"},
        {"submit_time": "2020-01-01T00:00:00.000Z"},
        {"comment": "hpclab:differenttoken"},
    ],
    ids=["other_owner", "submitted_long_before_dispatch", "comment_contradicts"],
)
def test_token_match_with_wrong_identity_is_never_bound(unit: UnitEnv, kw: dict[str, str]) -> None:
    a = uncertain(unit)
    job = ours(a, "7020")
    unit.scheduler.queue = [job.model_copy(update=kw)]
    unit.scheduler.acct = [job.model_copy(update={**kw, "source": "sacct"})]
    step(unit)
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"
    step(unit, WINDOW + 5)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "submission_uncertain"
    op = operation(unit.ctx, a["operation_id"])
    assert op["stage"] == "unresolved"
    mism = json.loads(op["error_json"])["details"]["mismatched"]
    if "user" not in kw:  # another user's jobs are not even listed by squeue/sacct --user
        assert mism and mism[0]["job_id"] == "7020"


def test_nothing_found_keeps_looking_then_blocks_unresolved_keeping_claim_and_budget(unit: UnitEnv) -> None:
    a = uncertain(unit)
    aid = a["attempt_id"]
    step(unit)
    assert attempt(unit.ctx, aid)["phase"] == "submission_uncertain"
    assert operation(unit.ctx, a["operation_id"])["stage"] == "reconciling"
    looks = len(unit.scheduler.queue_calls)
    step(unit, 30)
    assert len(unit.scheduler.queue_calls) > looks  # retried with backoff
    for _ in range(30):
        step(unit, 30)
    b = attempt(unit.ctx, aid)
    assert b["phase"] == "submission_uncertain"
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "unresolved"
    err = json.loads(op["error_json"])
    assert err["code"] == "submission_uncertain" and "attempt resolve" in err["details"]["remediation"]
    assert run_row(unit.ctx, a["run_id"])["claim_attempt_id"] == aid
    assert reservation(unit.ctx, aid)["state"] == "reserved"
    s = budget.summary(unit.ctx.db, experiment_id=run_row(unit.ctx, a["run_id"])["experiment_id"])
    assert s.reserved == pytest.approx(1.0)
    assert events(unit.ctx, aid).count("attempt.unresolved") == 1
    assert len(unit.scheduler.submit_calls) == 1  # never resubmitted
    # the sweep does not restart a judged attempt
    reconcile_sweep(unit.ctx)
    assert [i["state"] for i in outbox(unit.ctx, aid, "submit.reconcile")] == ["done"]


def test_unresolved_is_not_declared_while_sources_are_down(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.queue_error = CommandFailed("squeue broke", CommandResult(["squeue"], 1, "", "boom", 0.0))
    step(unit, WINDOW + 10)
    step(unit, 120)
    assert operation(unit.ctx, a["operation_id"])["stage"] == "reconciling"
    unit.scheduler.queue_error = None
    unit.scheduler.queue = [ours(a, "7030")]
    step(unit, 120)
    assert attempt(unit.ctx, a["attempt_id"])["job_id"] == "7030"


def test_accounting_unsupported_still_reaches_a_verdict(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.acct_error = TransportError("sacct missing", code=ErrorCode.UNSUPPORTED)
    step(unit, WINDOW + 10)
    op = operation(unit.ctx, a["operation_id"])
    assert op["stage"] == "unresolved"
    assert json.loads(op["error_json"])["details"]["sources"]["sacct"] == "unsupported"


def test_receipt_proving_a_refusal_rejects_and_releases(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.receipt = SubmissionRejected(
        "sbatch refused the job", details={"reason_code": "invalid_partition", "retry_later": False}
    )
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "submission_rejected"
    assert run_row(unit.ctx, a["run_id"])["claim_attempt_id"] is None
    assert reservation(unit.ctx, a["attempt_id"])["state"] == "released"
    ev = unit.ctx.db.one(
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.rejected'",
        (a["attempt_id"],),
    )
    assert json.loads(ev["payload_json"])["proof"] == "receipt"  # type: ignore[index]


def test_receipt_job_not_visible_anywhere_waits(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.receipt = SubmitResponse("7040", None, "7040", "")
    step(unit)
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"
    unit.scheduler.acct = [ours(a, "7040", state="COMPLETED", source="sacct")]
    step(unit, 60)
    assert attempt(unit.ctx, a["attempt_id"])["job_id"] == "7040"


def test_auth_during_reconcile_blocks_then_resumes(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.transport.auth_required = True
    step(unit)
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "auth_required"
    step(unit, 121)
    assert events(unit.ctx, a["attempt_id"]).count("attempt.auth_required") == 1
    unit.transport.auth_required = False
    unit.scheduler.queue = [ours(a, "7050")]
    step(unit, 121)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "observing"
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "succeeded" and op["error_json"] is None


def test_sweep_reenqueues_lost_reconcile_work_within_the_window(unit: UnitEnv) -> None:
    a = uncertain(unit)
    aid = a["attempt_id"]
    with unit.ctx.db.tx() as tx:  # the reconcile item died (e.g. dead-lettered)
        tx.execute("UPDATE outbox SET state='dead' WHERE target_id=? AND kind='submit.reconcile'", (aid,))
    reconcile_sweep(unit.ctx)
    states = [i["state"] for i in outbox(unit.ctx, aid, "submit.reconcile")]
    assert states == ["dead", "pending"]
    reconcile_sweep(unit.ctx)  # idempotent while one is pending
    assert len(outbox(unit.ctx, aid, "submit.reconcile")) == 2


def test_sweep_moves_an_orphaned_in_flight_attempt_to_uncertain(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    with unit.ctx.db.tx() as tx:  # in flight, and its dispatch item is gone
        tx.execute(
            "UPDATE attempts SET phase='submission_in_flight', dispatch_at=? WHERE attempt_id=?",
            (iso(unit.clock.now()), r.attempt_id),
        )
        tx.execute("UPDATE outbox SET state='done' WHERE target_id=?", (r.attempt_id,))
    reconcile_sweep(unit.ctx)
    assert attempt(unit.ctx, r.attempt_id)["phase"] == "submission_in_flight"  # too recent
    unit.clock.advance(601)
    reconcile_sweep(unit.ctx)
    assert attempt(unit.ctx, r.attempt_id)["phase"] == "submission_uncertain"
    assert unit.scheduler.submit_calls == []
    assert [i["state"] for i in outbox(unit.ctx, r.attempt_id, "submit.reconcile")] == ["pending"]


def test_reconcile_retry_is_capped_at_the_window(unit: UnitEnv) -> None:
    a = uncertain(unit)
    step(unit)
    item = outbox(unit.ctx, a["attempt_id"], "submit.reconcile")[0]
    assert item["state"] == "pending"
    assert item["not_before"] <= iso(unit.clock.now() + timedelta(seconds=60))


def test_non_retryable_access_problem_during_reconcile_blocks_without_failing_the_operation(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = uncertain(unit)

    def broken() -> str:
        raise TransportError("bad ~/.ssh permissions", code=ErrorCode.PRECONDITION_FAILED, retryable=False)

    monkeypatch.setattr(unit.transport, "remote_user", broken)
    step(unit)
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "cluster_access"
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"
    assert [i["state"] for i in outbox(unit.ctx, a["attempt_id"], "submit.reconcile")] == ["pending"]
    monkeypatch.undo()
    unit.scheduler.queue = [ours(a, "7060")]
    step(unit, 901)
    assert attempt(unit.ctx, a["attempt_id"])["job_id"] == "7060"
    assert operation(unit.ctx, a["operation_id"])["state"] == "succeeded"


def test_every_attempt_and_run_version_has_an_event(unit: UnitEnv) -> None:
    """Invariant 4.9 over a lifecycle: uncertain -> reconciled -> cancel sent."""
    from hpclab.services.submission import run_cancel

    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7070", state="RUNNING")]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    run_cancel(unit.ctx, attempt_id=b["attempt_id"], expected_version=b["version"], idempotency_key="c")
    drain(unit.ctx)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "sent"
    for entity, row in (("attempt", b), ("run", run_row(unit.ctx, a["run_id"]))):
        key = f"{entity}_id"
        versions = {
            r["entity_version"]
            for r in unit.ctx.db.all(
                "SELECT entity_version FROM events WHERE entity_type=? AND entity_id=?", (entity, row[key])
            )
        }
        first = 1 if entity == "attempt" else 2  # runs are created by the factory without an event
        assert set(range(first, row["version"] + 1)) <= versions, (entity, row["version"], versions)


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def test_sweep_ends_a_staging_attempt_whose_background_steps_gave_up(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """submit.stage dead-lettered (a local error, three times): nothing ever reached the
    scheduler, so the attempt must not hold the run's claim, a slot and its budget forever."""
    from hpclab.services import submission

    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")

    def broken(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("local snapshot store unreadable")

    monkeypatch.setattr(submission, "_stage_io", broken)
    for _ in range(6):
        step(unit, 30)
    assert [i["state"] for i in outbox(unit.ctx, r.attempt_id, "submit.stage")] == ["dead"]
    assert attempt(unit.ctx, r.attempt_id)["phase"] == "staging"
    reconcile_sweep(unit.ctx)
    assert attempt(unit.ctx, r.attempt_id)["phase"] == "staging"  # not stale yet
    unit.clock.advance(601)
    reconcile_sweep(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "submission_rejected" and a["reason"].startswith("stage_failed")
    assert run_row(unit.ctx, s.run_id)["claim_attempt_id"] is None
    assert reservation(unit.ctx, r.attempt_id)["state"] == "released"
    assert unit.scheduler.submit_calls == []
    op = operation(unit.ctx, r.operation_id)
    assert op["state"] == "failed" and op["stage"] == "stage_failed"
    assert "attempt.stage_failed" in events(unit.ctx, r.attempt_id)


def test_sweep_restages_a_staging_attempt_left_without_work(unit: UnitEnv) -> None:
    """No live and no dead step (e.g. housekeeping removed finished rows): staging is
    re-runnable, so it is simply scheduled again - and still submits exactly once."""
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    with unit.ctx.db.tx() as tx:
        tx.execute("DELETE FROM outbox WHERE target_id=?", (r.attempt_id,))
    unit.clock.advance(601)
    reconcile_sweep(unit.ctx)
    assert [i["state"] for i in outbox(unit.ctx, r.attempt_id, "submit.stage")] == ["pending"]
    reconcile_sweep(unit.ctx)  # idempotent while it is pending
    assert len(outbox(unit.ctx, r.attempt_id, "submit.stage")) == 1
    drain(unit.ctx)
    assert attempt(unit.ctx, r.attempt_id)["phase"] == "observing"
    assert len(unit.scheduler.submit_calls) == 1


def test_sweep_leaves_staging_attempts_with_live_work_alone(unit: UnitEnv) -> None:
    unit.transport.auth_required = True  # blocked for hours, item keeps retrying
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    unit.clock.advance(3600)
    reconcile_sweep(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "staging"
    assert operation(unit.ctx, r.operation_id)["stage"] == "auth_required"


def test_a_finished_same_token_job_that_left_squeue_still_makes_a_duplicate(unit: UnitEnv) -> None:
    """Invariant 4.3 binds only a UNIQUE match. After a long auth block, the job that did the
    work (7080) finished and left squeue while a same-token duplicate (7081) is still pending:
    squeue alone shows one match. Binding 7081 would attribute the run to a job that will exit
    95 - accounting must be consulted and the human asked."""
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7081", state="PENDING")]
    unit.scheduler.acct = [
        ours(a, "7080", state="COMPLETED", source="sacct", exit_code=0),
        ours(a, "7081", state="PENDING", source="sacct"),
    ]
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "duplicate_detected"
    cands = json.loads(operation(unit.ctx, a["operation_id"])["error_json"])["details"]["candidates"]
    assert sorted(c["job_id"] for c in cands) == ["7080", "7081"]
    assert unit.scheduler.cancel_calls == []


def test_a_contradiction_seen_in_one_source_is_not_erased_by_another(unit: UnitEnv) -> None:
    """squeue shows the job with a contradicting comment token; sacct (no Comment field on this
    cluster) shows only the name. The job must never become a candidate."""
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7090").model_copy(update={"comment": "hpclab:zzzzzzzzzzzz"})]
    unit.scheduler.acct = [ours(a, "7090", source="sacct").model_copy(update={"comment": None})]
    step(unit)
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"
    step(unit, WINDOW + 5)
    op = operation(unit.ctx, a["operation_id"])
    assert op["stage"] == "unresolved"
    assert [m["job_id"] for m in json.loads(op["error_json"])["details"]["mismatched"]] == ["7090"]


def test_a_unique_squeue_match_binds_even_when_accounting_is_down(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7095", state="RUNNING")]
    unit.scheduler.acct_error = CommandFailed("slurmdbd down", CommandResult(["sacct"], 1, "", "x", 0.0))
    step(unit)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "observing" and b["job_id"] == "7095"
    ev = unit.ctx.db.one(
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.job_identified'",
        (a["attempt_id"],),
    )
    assert json.loads(ev["payload_json"])["evidence"]["sources"]["sacct"] == "unavailable"  # type: ignore[index]


def test_a_contradiction_in_accounting_demotes_a_clean_squeue_match(unit: UnitEnv) -> None:
    a = uncertain(unit)
    unit.scheduler.queue = [ours(a, "7097")]
    unit.scheduler.acct = [
        ours(a, "7097", source="sacct").model_copy(update={"comment": "hpclab:zzzzzzzzzzzz"})
    ]
    step(unit)
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"
