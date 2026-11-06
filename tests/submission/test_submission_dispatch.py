"""submit.stage / submit.dispatch outcome classification with scripted adapters (invariant 4.2)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest
from submission_support import (
    SimulatedCrash,
    UnitEnv,
    attempt,
    drain,
    events,
    expire_leases,
    operation,
    outbox,
    reservation,
    run_row,
    seed_run,
)

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    CommandResult,
    ConnectionLost,
    SubmissionRejected,
    TransportTimeout,
)
from hpclab.domain.clock import iso
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument
from hpclab.services import submission
from hpclab.services.submission import run_cancel, run_submit


def submitted(unit: UnitEnv, **seed_kw: object) -> tuple[str, str]:
    s = seed_run(unit.ctx, **seed_kw)  # type: ignore[arg-type]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key=s.run_id)
    return r.attempt_id, s.run_id


def test_stage_writes_the_attempt_files_then_dispatch_binds_the_job(unit: UnitEnv) -> None:
    aid, run_id = submitted(unit)
    drain(unit.ctx)

    a = attempt(unit.ctx, aid)
    assert a["phase"] == "observing" and a["job_id"] == "5000" and a["job_owner"] == "alice"
    assert a["dispatch_at"] is not None and a["identified_at"] is not None
    assert len(unit.scheduler.submit_calls) == 1
    script, workdir, receipt = unit.scheduler.submit_calls[0]
    assert script == f"{a['remote_dir']}/job.sbatch" and workdir == a["remote_dir"]
    assert receipt == f"{a['remote_dir']}/submit.receipt"
    body = unit.transport.files[script].decode()
    assert body.startswith("#!/bin/bash") and f"--job-name={a['job_name']}" in body
    assert f"hpclab:{a['submit_token']}" in body
    plan_doc = json.loads(unit.transport.files[f"{a['remote_dir']}/hpclab-plan.json"])
    assert set(plan_doc) == {"run_id", "attempt_id", "token", "run_spec", "execution_plan", "plan_hash"}
    assert plan_doc["plan_hash"] == a["exec_plan_hash"] and plan_doc["token"] == a["submit_token"]
    mkdirs = [args for script_, args in unit.transport.scripts if "mkdir" in script_]
    assert mkdirs == [[a["remote_dir"], f"{a['remote_dir']}/logs"]]
    assert unit.published == [f"/remote/hpclab/snapshots/{'ab' * 32}"]
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "succeeded" and op["stage"] == "submitted"
    assert run_row(unit.ctx, run_id)["status"] == "queued"
    assert run_row(unit.ctx, run_id)["claim_attempt_id"] == aid  # claim held while the job lives
    assert events(unit.ctx, aid) == [
        "attempt.admitted",
        "attempt.staged",
        "attempt.dispatching",
        "attempt.job_identified",
        "attempt.observing",
    ]


def test_stage_is_rerunnable_with_identical_content(unit: UnitEnv) -> None:
    aid, _ = submitted(unit)
    a = attempt(unit.ctx, aid)
    row = submission._attempt(unit.ctx.db, aid)
    assert row is not None
    inp = submission._load_inputs(unit.ctx, row)
    first = submission._stage_io(unit.ctx, row, inp)
    body1 = unit.transport.files[f"{a['remote_dir']}/job.sbatch"]
    second = submission._stage_io(unit.ctx, row, inp)
    assert first == second
    assert unit.transport.files[f"{a['remote_dir']}/job.sbatch"] == body1


def test_definite_rejection_releases_claim_and_budget(unit: UnitEnv) -> None:
    unit.scheduler.submit_behaviors = [
        SubmissionRejected(
            "sbatch refused the job: Invalid account",
            details={
                "reason_code": "invalid_account",
                "retry_later": False,
                "stderr": "sbatch: error: Invalid account",
            },
        )
    ]
    aid, run_id = submitted(unit)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_rejected" and a["reason"].startswith("invalid_account")
    assert run_row(unit.ctx, run_id)["claim_attempt_id"] is None
    assert run_row(unit.ctx, run_id)["status"] == "rejected"
    assert reservation(unit.ctx, aid)["state"] == "released"
    op = operation(unit.ctx, a["operation_id"])
    assert (
        op["state"] == "failed"
        and json.loads(op["error_json"])["details"]["reason_code"] == "invalid_account"
    )
    assert len(unit.scheduler.submit_calls) == 1
    assert outbox(unit.ctx, aid, "submit.reconcile") == []


@pytest.mark.parametrize(
    "error",
    [
        ConnectionLost("reply lost", details={"reason_code": "no_reply_frame"}),
        TransportTimeout("sbatch timed out"),
        CommandFailed("odd", CommandResult(["x"], 1, "", "", 0.0)),
        RuntimeError("bug in an adapter"),
        InvalidArgument("even an argument error after the in-flight commit is not proof"),
    ],
    ids=["connection_lost", "timeout", "command_failed", "bug", "invalid_argument"],
)
def test_every_non_refusal_after_the_in_flight_commit_is_uncertain(
    unit: UnitEnv, error: BaseException
) -> None:
    unit.scheduler.submit_behaviors = [error]
    aid, run_id = submitted(unit)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_uncertain" and a["uncertain_since"] is not None
    assert run_row(unit.ctx, run_id)["claim_attempt_id"] == aid
    assert run_row(unit.ctx, run_id)["status"] == "unknown"
    assert reservation(unit.ctx, aid)["state"] == "reserved"
    rec = outbox(unit.ctx, aid, "submit.reconcile")
    assert len(rec) == 1 and rec[0]["state"] == "pending"
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "running" and op["stage"] == "reconciling"
    assert len(unit.scheduler.submit_calls) == 1


def test_auth_before_the_call_keeps_staging_and_blocks_the_operation(unit: UnitEnv) -> None:
    aid, _ = submitted(unit)
    unit.transport.auth_required = True
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "staging"
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "auth_required"
    err = json.loads(op["error_json"])
    assert err["code"] == "auth_required" and err["details"]["remediation"] == "hpclab auth unit"
    assert unit.scheduler.submit_calls == []
    assert events(unit.ctx, aid).count("attempt.auth_required") == 1
    # still blocked on the next retry: no duplicate event
    unit.clock.advance(121)
    drain(unit.ctx)
    assert events(unit.ctx, aid).count("attempt.auth_required") == 1

    unit.transport.auth_required = False
    unit.clock.advance(121)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "observing" and len(unit.scheduler.submit_calls) == 1
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "succeeded" and op["error_json"] is None


def test_host_key_problems_are_not_hammered(unit: UnitEnv) -> None:
    aid, _ = submitted(unit)
    unit.transport.auth_required = True
    unit.transport.auth_reason = "host_key"
    drain(unit.ctx)
    op = operation(unit.ctx, attempt(unit.ctx, aid)["operation_id"])
    err = json.loads(op["error_json"])
    assert err["retryable"] is False and "admins" in err["details"]["remediation"]
    item = outbox(unit.ctx, aid, "submit.stage")[0]
    assert item["not_before"] == iso(unit.clock.now() + timedelta(seconds=900))


def test_auth_during_submit_stays_in_flight_then_reconciles_without_resending(unit: UnitEnv) -> None:
    unit.scheduler.submit_behaviors = [AuthRequiredError("mfa expired mid-call")]
    aid, _ = submitted(unit)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_in_flight"
    assert operation(unit.ctx, a["operation_id"])["stage"] == "auth_required"
    unit.clock.advance(121)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_uncertain"
    assert len(unit.scheduler.submit_calls) == 1
    assert "attempt.uncertain" in events(unit.ctx, aid)


def test_crash_after_the_in_flight_commit_never_resubmits(unit: UnitEnv) -> None:
    unit.scheduler.submit_behaviors = [SimulatedCrash("controller killed while sbatch ran")]
    aid, _ = submitted(unit)
    with pytest.raises(SimulatedCrash):
        drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "submission_in_flight"
    expire_leases(unit.ctx)  # the restarted controller picks the dispatch item up again
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_uncertain"
    assert len(unit.scheduler.submit_calls) == 1
    payloads = unit.ctx.db.all(
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.uncertain'", (aid,)
    )
    assert json.loads(payloads[0]["payload_json"])["cause"] == "reentry"


def test_crash_before_the_in_flight_commit_submits_exactly_once(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = submission._transition
    crashed: list[bool] = []

    def crashing(tx: Any, row: Any, dst: Any, *a: Any, **kw: Any) -> Any:
        if dst == submission.AttemptPhase.SUBMISSION_IN_FLIGHT and not crashed:
            crashed.append(True)
            raise SimulatedCrash("killed before the intent commit")
        return real(tx, row, dst, *a, **kw)

    monkeypatch.setattr(submission, "_transition", crashing)
    aid, _ = submitted(unit)
    with pytest.raises(SimulatedCrash):
        drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "staging"  # the intent tx rolled back
    expire_leases(unit.ctx)
    drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "observing"
    assert len(unit.scheduler.submit_calls) == 1


def test_crash_after_sbatch_accepted_before_the_outcome_commit_reconciles_one_job(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    def accept_then_crash(script: str, *, workdir: str, receipt_path: str | None) -> object:
        unit.scheduler.submit_behaviors.insert(0, "accept")
        resp = unit.scheduler.submit(script, workdir=workdir, receipt_path=receipt_path)
        raise SimulatedCrash(f"killed after job {resp.job_id} was queued")

    unit.scheduler.submit_behaviors = [accept_then_crash]  # type: ignore[list-item]
    aid, _ = submitted(unit)
    with pytest.raises(SimulatedCrash):
        drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_in_flight"
    from submission_support import obs

    unit.scheduler.queue = [obs("5000", name=a["job_name"], comment=f"hpclab:{a['submit_token']}")]
    expire_leases(unit.ctx)
    drain(unit.ctx)
    unit.clock.advance(5)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "observing" and a["job_id"] == "5000"
    assert unit.scheduler.accepted == ["5000"]  # one job, the ground truth of the fake


def test_permanent_stage_failure_rejects_without_touching_the_scheduler(unit: UnitEnv) -> None:
    aid, run_id = submitted(unit)
    with unit.ctx.db.tx() as tx:  # a run spec staging cannot render
        tx.execute('UPDATE runs SET spec_json=\'{"experiment": "x"}\' WHERE run_id=?', (run_id,))
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "submission_rejected" and a["reason"].startswith("stage_failed")
    assert run_row(unit.ctx, run_id)["claim_attempt_id"] is None
    assert reservation(unit.ctx, aid)["state"] == "released"
    assert unit.scheduler.submit_calls == []
    assert "attempt.stage_failed" in events(unit.ctx, aid)


def test_tampered_execution_plan_is_refused_at_staging(unit: UnitEnv) -> None:
    aid, _ = submitted(unit)
    a = attempt(unit.ctx, aid)
    with unit.ctx.db.tx() as tx:
        row = tx.one("SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (a["exec_plan_hash"],))
        plan = json.loads(row["plan_json"])  # type: ignore[index]
        plan["resources"]["gpus"] = 8
        tx.execute(
            "UPDATE execution_plans SET plan_json=? WHERE exec_plan_hash=?",
            (json.dumps(plan), a["exec_plan_hash"]),
        )
    drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "submission_rejected"
    assert "does not match its hash" in attempt(unit.ctx, aid)["reason"]
    assert unit.scheduler.submit_calls == []


def test_transient_stage_failure_is_retried(unit: UnitEnv) -> None:
    unit.publish_error = ConnectionLost("rsync dropped")
    aid, _ = submitted(unit)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "staging"
    item = outbox(unit.ctx, aid, "submit.stage")[0]
    assert item["state"] == "pending" and item["tries"] == 1
    op = operation(unit.ctx, a["operation_id"])
    assert "rsync dropped" in json.loads(op["error_json"])["message"]
    unit.publish_error = None
    unit.clock.advance(600)
    drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "observing"


def test_non_retryable_publish_error_ends_the_attempt(unit: UnitEnv) -> None:
    unit.publish_error = HpcLabError("local manifest corrupt", code=ErrorCode.INTERNAL)
    aid, _ = submitted(unit)
    drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "submission_rejected"
    assert unit.scheduler.submit_calls == []


def test_missing_staged_files_are_restaged_before_dispatch(
    unit: UnitEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    aid, _ = submitted(unit)
    a = attempt(unit.ctx, aid)
    script = f"{a['remote_dir']}/job.sbatch"
    real_write = unit.transport.write_file
    purged: list[bool] = []

    def purge_after_stage(path: str, content: bytes, **kw: object) -> None:
        real_write(path, content)
        if path == script and not purged:
            purged.append(True)
            unit.transport.files.pop(path)
            unit.transport.missing.add(path)  # scratch purge between stage and dispatch

    monkeypatch.setattr(unit.transport, "write_file", purge_after_stage)
    drain(unit.ctx)
    assert "attempt.restage" in events(unit.ctx, aid)
    assert attempt(unit.ctx, aid)["phase"] == "observing"
    assert len(unit.scheduler.submit_calls) == 1


def test_cancel_before_dispatch_wins_and_nothing_is_sent(unit: UnitEnv) -> None:
    aid, run_id = submitted(unit)
    a = attempt(unit.ctx, aid)
    receipt = run_cancel(unit.ctx, attempt_id=aid, expected_version=a["version"], idempotency_key="c")
    assert receipt.phase == "cancelled_before_submit" and receipt.cancel_state == "confirmed"
    drain(unit.ctx)
    assert unit.scheduler.submit_calls == []
    assert run_row(unit.ctx, run_id)["status"] == "cancelled"
    assert reservation(unit.ctx, aid)["state"] == "released"


def test_cancel_intent_recorded_between_stage_and_dispatch_is_honoured(unit: UnitEnv) -> None:
    aid, _ = submitted(unit)
    with unit.ctx.db.tx() as tx:  # e.g. a workflow cancel that only recorded the intent
        tx.execute("UPDATE attempts SET cancel_state='requested' WHERE attempt_id=?", (aid,))
    drain(unit.ctx)
    assert attempt(unit.ctx, aid)["phase"] == "cancelled_before_submit"
    assert unit.scheduler.submit_calls == []


def test_submit_rate_limit_defers_dispatch_without_committing(unit: UnitEnv) -> None:
    unit.set_policy(submit_rate_per_min=1)
    first, _ = submitted(unit)
    second, _ = submitted(unit)
    drain(unit.ctx)
    phases = {attempt(unit.ctx, first)["phase"], attempt(unit.ctx, second)["phase"]}
    assert phases == {"observing", "staging"}
    assert len(unit.scheduler.submit_calls) == 1
    unit.clock.advance(61)
    drain(unit.ctx)
    assert attempt(unit.ctx, second)["phase"] == "observing"
    assert len(unit.scheduler.submit_calls) == 2


def test_a_late_reply_after_reentry_binds_the_replied_job(unit: UnitEnv) -> None:
    """Lease expiry while sbatch was slow: the re-entered handler made the attempt uncertain;
    the original call's reply is still authoritative and binds its job."""
    aid, _ = submitted(unit)

    def slow_submit(script: str, *, workdir: str, receipt_path: str | None) -> object:
        submission._dispatch_reentry(unit.ctx, aid)  # a second worker re-entered meanwhile
        unit.scheduler.submit_behaviors.insert(0, "accept")
        return unit.scheduler.submit(script, workdir=workdir, receipt_path=receipt_path)

    unit.scheduler.submit_behaviors = [slow_submit]  # type: ignore[list-item]
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "observing" and a["job_id"] == "5000"
    assert unit.scheduler.accepted == ["5000"]


def test_non_retryable_cluster_problem_blocks_instead_of_failing(unit: UnitEnv) -> None:
    unit.publish_error = CommandFailed(
        "disk quota exceeded",
        CommandResult(["rsync"], 23, "", "Disk quota exceeded", 0.0),
        retryable=False,
        details={"reason": "remote_storage_full"},
    )
    aid, run_id = submitted(unit)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "staging"  # the user can free space; nothing was sent
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "blocked" and op["stage"] == "cluster_access"
    assert "quota" in json.loads(op["error_json"])["details"]["remediation"]
    assert run_row(unit.ctx, run_id)["claim_attempt_id"] == aid
    unit.publish_error = None
    unit.clock.advance(901)
    drain(unit.ctx)
    a = attempt(unit.ctx, aid)
    assert a["phase"] == "observing" and operation(unit.ctx, a["operation_id"])["state"] == "succeeded"
