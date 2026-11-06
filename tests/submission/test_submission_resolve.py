"""attempt_resolve: the explicit, audited human decision for uncertain/duplicate attempts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from submission_support import UnitEnv, add_plan, attempt, drain, obs, operation, run_row, seed_run

from hpclab.adapters.base import ConnectionLost
from hpclab.domain.errors import (
    HpcLabError,
    IdempotencyConflict,
    InvalidArgument,
    PreconditionFailed,
    VersionConflict,
)
from hpclab.domain.states import HUMAN_ONLY_TRANSITIONS, AttemptPhase, can_transition
from hpclab.services import budget, observe, submission
from hpclab.services.submission import attempt_resolve, run_cancel, run_submit


def unresolved(unit: UnitEnv) -> tuple[dict[str, Any], Any]:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    unit.clock.advance(400)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "submission_uncertain"
    assert operation(unit.ctx, a["operation_id"])["stage"] == "unresolved"
    return a, s


def job(a: dict[str, Any], job_id: str, **kw: Any) -> Any:
    kw.setdefault("submit_time", a["dispatch_at"])
    return obs(job_id, name=a["job_name"], comment=f"hpclab:{a['submit_token']}", **kw)


def resolve(unit: UnitEnv, a: dict[str, Any], **kw: Any) -> Any:
    kw.setdefault("reason", "checked squeue by hand")
    kw.setdefault("idempotency_key", "r")
    version = attempt(unit.ctx, a["attempt_id"])["version"]
    return attempt_resolve(unit.ctx, attempt_id=a["attempt_id"], expected_version=version, **kw)


def test_abandon_releases_the_claim_but_keeps_the_budget_as_unknown(unit: UnitEnv) -> None:
    a, s = unresolved(unit)
    data = resolve(unit, a, resolution="abandon")
    assert data.phase == "abandoned" and data.job_id is None
    assert any("cancels nothing" in w for w in data.warnings)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "abandoned" and b["resolution"].startswith("abandon: checked squeue")
    assert run_row(unit.ctx, s.run_id)["claim_attempt_id"] is None
    summary = budget.summary(unit.ctx.db, experiment_id=s.experiment_id)
    assert summary.unknown_count == 1 and summary.unknown_reserved == pytest.approx(1.0)
    assert summary.reserved == 0.0
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == "failed" and op["stage"] == "abandoned"
    ev = unit.ctx.db.one(
        "SELECT source, payload_json FROM events WHERE entity_id=? AND event_type='attempt.abandoned'",
        (a["attempt_id"],),
    )
    assert ev is not None and ev["source"] == "human"
    assert json.loads(ev["payload_json"])["reason"] == "checked squeue by hand"
    # the run can execute again - only through a NEW plan
    fresh = add_plan(unit.ctx, s, nonce="after-abandon")
    unit.scheduler.submit_behaviors = []
    r2 = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=fresh, idempotency_key="k2")
    drain(unit.ctx)
    assert attempt(unit.ctx, r2.attempt_id)["phase"] == "observing"


def test_bind_job_after_a_live_identity_check(unit: UnitEnv) -> None:
    a, _ = unresolved(unit)
    unit.scheduler.acct = [job(a, "7300", state="COMPLETED", source="sacct")]
    data = resolve(unit, a, resolution="bind_job", job_id="7300")
    assert data.phase == "observing" and data.job_id == "7300" and data.warnings == []
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["job_id"] == "7300" and b["resolution"].startswith("bind_job 7300")
    assert operation(unit.ctx, a["operation_id"])["state"] == "succeeded"
    types = [
        r["event_type"]
        for r in unit.ctx.db.all(
            "SELECT event_type FROM events WHERE entity_id=? AND source='human' ORDER BY seq",
            (a["attempt_id"],),
        )
    ]
    assert types == ["attempt.job_identified", "attempt.observing", "attempt.resolved"]


def test_bind_job_on_a_duplicate(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    unit.scheduler.queue = [job(a, "7400"), job(a, "7401")]
    unit.clock.advance(3)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "duplicate_detected"
    data = resolve(unit, a, resolution="bind_job", job_id="7401")
    assert data.job_id == "7401" and data.phase == "observing"
    assert unit.scheduler.cancel_calls == []  # the other duplicate is the human's call


def test_abandon_duplicate_warns_about_live_jobs(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    unit.scheduler.submit_behaviors = [ConnectionLost("reply lost")]
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    unit.scheduler.queue = [job(a, "7500"), job(a, "7501")]
    unit.clock.advance(3)
    drain(unit.ctx)
    data = resolve(unit, attempt(unit.ctx, r.attempt_id), resolution="abandon")
    assert "7500, 7501" in data.warnings[0]
    assert unit.scheduler.cancel_calls == []


@pytest.mark.parametrize(
    "change",
    [{"name": "hl.cell.seed-1.zzzzzzzz", "comment": None}, {"user": "mallory"}],
    ids=["token", "owner"],
)
def test_bind_job_refuses_a_job_that_is_not_provably_ours(unit: UnitEnv, change: dict[str, Any]) -> None:
    a, _ = unresolved(unit)
    unit.scheduler.queue = [job(a, "7600").model_copy(update=change)]
    with pytest.raises(PreconditionFailed) as exc:
        resolve(unit, a, resolution="bind_job", job_id="7600")
    assert exc.value.details["problems"]
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "submission_uncertain"


def test_bind_job_needs_a_visible_job(unit: UnitEnv) -> None:
    a, _ = unresolved(unit)
    with pytest.raises(PreconditionFailed):
        resolve(unit, a, resolution="bind_job", job_id="7700")


def test_bind_job_with_an_early_submit_time_binds_with_a_warning(unit: UnitEnv) -> None:
    a, _ = unresolved(unit)
    unit.scheduler.queue = [job(a, "7800", submit_time="2020-01-01T00:00:00.000Z")]
    data = resolve(unit, a, resolution="bind_job", job_id="7800")
    assert data.job_id == "7800" and data.warnings and "submit time" in data.warnings[0]


def test_argument_version_and_phase_checks(unit: UnitEnv) -> None:
    a, _ = unresolved(unit)
    with pytest.raises(InvalidArgument):
        resolve(unit, a, resolution="abandon", reason="  ")
    with pytest.raises(InvalidArgument):
        resolve(unit, a, resolution="bind_job")
    with pytest.raises(VersionConflict):
        attempt_resolve(
            unit.ctx,
            attempt_id=a["attempt_id"],
            expected_version=1,
            resolution="abandon",
            reason="x",
            idempotency_key="v",
        )
    resolve(unit, a, resolution="abandon", idempotency_key="same")
    with pytest.raises(PreconditionFailed):  # already abandoned: nothing left to resolve
        resolve(unit, a, resolution="abandon", idempotency_key="later")


def test_resolution_is_idempotent_per_key(unit: UnitEnv) -> None:
    a, _ = unresolved(unit)
    version = a["version"]
    first = attempt_resolve(
        unit.ctx,
        attempt_id=a["attempt_id"],
        expected_version=version,
        resolution="abandon",
        reason="gone",
        idempotency_key="same",
    )
    replay = attempt_resolve(
        unit.ctx,
        attempt_id=a["attempt_id"],
        expected_version=version,
        resolution="abandon",
        reason="gone",
        idempotency_key="same",
    )
    assert replay == first
    with pytest.raises(IdempotencyConflict):
        attempt_resolve(
            unit.ctx,
            attempt_id=a["attempt_id"],
            expected_version=version,
            resolution="abandon",
            reason="different",
            idempotency_key="same",
        )


def test_live_jobs_cannot_be_resolved_by_hand(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "observing"
    with pytest.raises(PreconditionFailed):
        resolve(unit, a, resolution="abandon")


# ---------------------------------------------------------------------------
# Last resort: abandoning an OBSERVING attempt hpclab cannot observe any more
# ---------------------------------------------------------------------------

UNVERIFIABLE = (
    "not in squeue; accounting shows job 5000 with a later submit time (a requeue hpclab did not see?) "
    "but its earlier segments could not be verified: per-segment accounting (sacct --duplicates) is not available"
)


def observing_in_gap(unit: UnitEnv, **seed_kw: Any) -> tuple[dict[str, Any], Any]:
    """An identified job whose requeue chain accounting can never verify: observe keeps it
    OBSERVING with an observation gap forever."""
    s = seed_run(unit.ctx, **seed_kw)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    with unit.ctx.db.tx(source="observer") as tx:
        assert observe.mark_observation_gap(tx, attempt_id=r.attempt_id, reason=UNVERIFIABLE) == [
            "gap_opened"
        ]
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "observing" and a["observation_gap"] == 1 and a["job_id"] == "5000"
    return a, s


def test_transition_table_allows_observing_to_abandoned_for_humans_only() -> None:
    assert can_transition(AttemptPhase.OBSERVING, AttemptPhase.ABANDONED)
    assert (AttemptPhase.OBSERVING, AttemptPhase.ABANDONED) in HUMAN_ONLY_TRANSITIONS
    assert not can_transition(AttemptPhase.JOB_IDENTIFIED, AttemptPhase.ABANDONED)
    assert not can_transition(AttemptPhase.TERMINAL, AttemptPhase.ABANDONED)
    assert not can_transition(AttemptPhase.ABANDONED, AttemptPhase.OBSERVING)


def test_no_automated_actor_can_abandon_an_observing_attempt(unit: UnitEnv) -> None:
    a, _ = observing_in_gap(unit)
    for source in ("observer", "worker", "api"):
        with pytest.raises(HpcLabError) as exc, unit.ctx.db.tx(source=source) as tx:
            submission._transition(tx, a, AttemptPhase.ABANDONED, {}, "attempt.abandoned")
        assert "human decision only" in exc.value.message
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "observing"


def test_abandon_an_unobservable_observing_attempt_as_a_last_resort(unit: UnitEnv) -> None:
    a, s = observing_in_gap(unit)
    op_before = operation(unit.ctx, a["operation_id"])
    data = resolve(unit, a, resolution="abandon", reason="sacct --duplicates is disabled on this site")
    assert data.phase == "abandoned" and data.job_id == "5000"
    # the loud warning comes first: the job may still run and must be cancelled by hand
    loud = data.warnings[0]
    assert loud.startswith("WARNING: Slurm job 5000 may still be queued or running")
    assert "cancels nothing" in loud and "scancel 5000" in loud
    assert any("could not be verified" in w for w in data.warnings)  # why it was unobservable
    assert unit.scheduler.cancel_calls == []
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["phase"] == "abandoned" and b["resolution"].startswith("abandon: sacct --duplicates")
    run = run_row(unit.ctx, s.run_id)
    assert run["claim_attempt_id"] is None and run["status"] == "unknown"
    summary = budget.summary(unit.ctx.db, experiment_id=s.experiment_id)
    assert summary.unknown_count == 1 and summary.unknown_reserved == pytest.approx(1.0)
    assert summary.reserved == 0.0
    # the submission itself succeeded: its operation is left as it was
    op = operation(unit.ctx, a["operation_id"])
    assert op["state"] == op_before["state"] == "succeeded" and op["stage"] == op_before["stage"]
    ev = unit.ctx.db.one(
        "SELECT source, payload_json FROM events WHERE entity_id=? AND event_type='attempt.abandoned'",
        (a["attempt_id"],),
    )
    assert ev is not None and ev["source"] == "human"
    payload = json.loads(ev["payload_json"])
    assert payload["from"] == "observing" and payload["job_id"] == "5000" and payload["budget"] == "unknown"
    assert "may still be running" in payload["warning"]
    # observe leaves the abandoned attempt alone from now on
    with unit.ctx.db.tx(source="observer") as tx:
        late = job(b, "5000", state="COMPLETED", submit_time=b["dispatch_at"])
        assert observe.apply_observation(tx, unit.ctx, attempt_id=a["attempt_id"], obs=late) == [
            "ignored:phase=abandoned"
        ]
    # a new attempt needs a new plan, like every other retry
    fresh = add_plan(unit.ctx, s, nonce="after-abandon")
    r2 = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=fresh, idempotency_key="k2")
    assert attempt(unit.ctx, r2.attempt_id)["seq"] == 2


def test_abandoning_an_observing_attempt_drops_its_pending_cancel(unit: UnitEnv) -> None:
    a, _ = observing_in_gap(unit)
    unit.transport.auth_required = True  # the cancel step cannot run
    cancel = run_cancel(
        unit.ctx, attempt_id=a["attempt_id"], expected_version=a["version"], idempotency_key="c"
    )
    drain(unit.ctx)
    assert attempt(unit.ctx, a["attempt_id"])["cancel_state"] == "requested"
    data = resolve(unit, attempt(unit.ctx, a["attempt_id"]), resolution="abandon")
    assert any("pending cancel request was dropped" in w and "5000" in w for w in data.warnings)
    b = attempt(unit.ctx, a["attempt_id"])
    assert b["cancel_state"] == "failed"
    assert operation(unit.ctx, cancel.operation_id or "")["state"] == "failed"
    unit.transport.auth_required = False
    unit.clock.advance(300)
    drain(unit.ctx)
    assert unit.scheduler.cancel_calls == []  # nothing is sent for an abandoned attempt


def test_only_abandon_applies_to_an_observing_attempt_in_a_gap(unit: UnitEnv) -> None:
    a, _ = observing_in_gap(unit)
    unit.scheduler.acct = [job(a, "5000", state="RUNNING", source="sacct")]
    with pytest.raises(PreconditionFailed) as exc:
        resolve(unit, a, resolution="bind_job", job_id="5000")
    assert "can only be abandoned" in exc.value.message
    with pytest.raises(PreconditionFailed):
        resolve(unit, a, resolution="confirm_ended", idempotency_key="c")
    assert attempt(unit.ctx, a["attempt_id"])["phase"] == "observing"


def test_an_observed_job_is_cancelled_not_abandoned(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key="k")
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    with pytest.raises(PreconditionFailed) as exc:
        resolve(unit, a, resolution="abandon")
    assert "run_cancel" in exc.value.message and exc.value.details["observation_gap"] is False
