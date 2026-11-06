"""Execution holds: a Run stays claimed while its original job may still run.

* requeue watch - a job that ended PREEMPTED / NODE_FAIL / BOOT_FAIL may come back after a Slurm
  requeue (observe keeps polling it for ``REQUEUE_WATCH_S``): admission refuses a second
  attempt until the window closes or a human confirms the job ended.
* duplicate owner - finalize found that another Slurm job owns the attempt dir (the bound job
  was a duplicate submission): that job runs the attempt unobserved, so the Run stays claimed
  until squeue/sacct show it ended (the reconcile sweep looks) or a human confirms it ended.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from submission_support import UnitEnv, add_plan, attempt, drain, events, obs, run_row, seed_run

from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import AdmissionDenied, HpcLabError, PreconditionFailed
from hpclab.domain.models import JobObservation
from hpclab.domain.states import normalize_slurm_state
from hpclab.services import observe
from hpclab.services.submission import attempt_resolve, execution_holds, reconcile_sweep, run_submit
from hpclab.testing.factories import make_attempt


def running(unit: UnitEnv, **seed_kw: Any) -> tuple[dict[str, Any], Any]:
    s = seed_run(unit.ctx, **seed_kw)
    r = run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=s.exec_plan_hash, idempotency_key=s.run_id)
    drain(unit.ctx)
    a = attempt(unit.ctx, r.attempt_id)
    assert a["phase"] == "observing" and a["job_id"]
    return a, s


def job_obs(unit: UnitEnv, a: dict[str, Any], state: str, **kw: Any) -> JobObservation:
    kw.setdefault("submit_time", a["dispatch_at"])
    o = obs(str(a["job_id"]), name=a["job_name"], comment=f"hpclab:{a['submit_token']}", state=state, **kw)
    return o.model_copy(update={"observed_at": iso(unit.clock.now())})


def end(unit: UnitEnv, a: dict[str, Any], state: str) -> dict[str, Any]:
    """What observe does when squeue shows the job terminal (claim released, finalize queued)."""
    with unit.ctx.db.tx(source="observer") as tx:
        changes = observe.apply_observation(
            tx, unit.ctx, attempt_id=a["attempt_id"], obs=job_obs(unit, a, state)
        )
    assert f"terminal:{normalize_slurm_state(state).value}" in changes
    b = attempt(unit.ctx, a["attempt_id"])
    assert run_row(unit.ctx, b["run_id"])["claim_attempt_id"] is None
    return b


def resubmit(unit: UnitEnv, s: Any, *, nonce: str, key: str) -> Any:
    fresh = add_plan(unit.ctx, s, nonce=nonce)
    return run_submit(unit.ctx, run_id=s.run_id, execution_plan_hash=fresh, idempotency_key=key)


def confirm(unit: UnitEnv, a: dict[str, Any], key: str = "confirm", **kw: Any) -> Any:
    kw.setdefault("reason", "checked sacct by hand: the job is over")
    version = attempt(unit.ctx, a["attempt_id"])["version"]
    return attempt_resolve(
        unit.ctx,
        attempt_id=a["attempt_id"],
        expected_version=version,
        resolution="confirm_ended",
        idempotency_key=key,
        **kw,
    )


def attempt_count(unit: UnitEnv) -> int:
    row = unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")
    assert row is not None
    return int(row["n"])


# ---------------------------------------------------------------------------
# requeue watch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["PREEMPTED", "NODE_FAIL", "BOOT_FAIL"])
def test_admission_refuses_while_the_original_job_may_be_requeued(unit: UnitEnv, state: str) -> None:
    a, s = running(unit)
    b = end(unit, a, state)
    ended = parse_iso(b["terminal_at"])
    assert ended is not None
    until = iso(ended + timedelta(seconds=observe.REQUEUE_WATCH_S))

    with pytest.raises(AdmissionDenied) as exc:
        resubmit(unit, s, nonce="retry", key="k2")
    err = exc.value
    assert "original job 5000" in err.message
    assert f"may be requeued; wait until {until} or confirm it ended" in err.message
    assert err.retryable
    assert err.details["reason"] == "run_claimed" and err.details["hold"] == "requeue_watch"
    assert err.details["claim_attempt_id"] == a["attempt_id"] and err.details["job_id"] == "5000"
    assert err.details["until"] == until
    assert "confirm_ended" in err.details["hint"]
    assert attempt_count(unit) == 1  # nothing admitted, no second execution

    # one second before the window closes: still held; after it: admitted
    unit.clock.advance(observe.REQUEUE_WATCH_S - 1)
    with pytest.raises(AdmissionDenied):
        resubmit(unit, s, nonce="retry-2", key="k3")
    unit.clock.advance(2)
    r = resubmit(unit, s, nonce="retry-3", key="k4")
    assert attempt(unit.ctx, r.attempt_id)["seq"] == 2


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"])
def test_states_slurm_does_not_requeue_from_are_not_held(unit: UnitEnv, state: str) -> None:
    a, s = running(unit)
    end(unit, a, state)
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []
    r = resubmit(unit, s, nonce="retry", key="k2")
    assert attempt(unit.ctx, r.attempt_id)["seq"] == 2


def test_a_no_requeue_plan_cannot_come_back_and_is_not_held(unit: UnitEnv) -> None:
    """``--no-requeue``: Slurm never requeues the job, not on preemption and not on node failure."""
    a, s = running(unit, requeue=False)
    end(unit, a, "NODE_FAIL")
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []
    resubmit(unit, s, nonce="retry", key="k2")


def test_only_the_latest_attempt_is_watched(unit: UnitEnv) -> None:
    a, s = running(unit)
    end(unit, a, "PREEMPTED")
    unit.clock.advance(observe.REQUEUE_WATCH_S + 1)
    r = resubmit(unit, s, nonce="retry", key="k2")
    assert attempt(unit.ctx, r.attempt_id)["seq"] == 2
    holds = execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())
    assert holds == []


def test_confirm_ended_lifts_the_watch_when_the_scheduler_shows_the_job_over(unit: UnitEnv) -> None:
    a, s = running(unit)
    b = end(unit, a, "PREEMPTED")
    unit.scheduler.acct = [job_obs(unit, a, "PREEMPTED", source="sacct")]
    data = confirm(unit, b)
    assert data.resolution == "confirm_ended" and data.phase == "terminal" and data.job_id == "5000"
    assert "confirm_ended" in data.summary()
    c = attempt(unit.ctx, a["attempt_id"])
    assert c["version"] == b["version"] + 1 and c["resolution"].startswith("confirm_ended: checked sacct")
    ev = unit.ctx.db.one(
        "SELECT source, payload_json FROM events WHERE entity_id=? AND event_type='attempt.end_confirmed'",
        (a["attempt_id"],),
    )
    assert ev is not None and ev["source"] == "human"
    assert '"5000"' in ev["payload_json"] and "PREEMPTED" in ev["payload_json"]
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []
    r = resubmit(unit, s, nonce="retry", key="k2")  # well inside the watch window
    assert attempt(unit.ctx, r.attempt_id)["seq"] == 2


def test_confirm_ended_on_the_word_of_a_human_warns(unit: UnitEnv) -> None:
    a, s = running(unit)
    b = end(unit, a, "NODE_FAIL")
    data = confirm(unit, b)  # neither squeue nor sacct knows the job (any more)
    assert any("not visible" in w and "5000" in w for w in data.warnings)
    resubmit(unit, s, nonce="retry", key="k2")


def test_confirm_ended_refuses_a_job_that_was_requeued(unit: UnitEnv) -> None:
    a, s = running(unit)
    b = end(unit, a, "PREEMPTED")
    unit.scheduler.queue = [job_obs(unit, a, "PENDING", restart_count=1)]
    with pytest.raises(PreconditionFailed) as exc:
        confirm(unit, b)
    assert "5000" in exc.value.message and "PENDING" in exc.value.message
    assert attempt(unit.ctx, a["attempt_id"])["version"] == b["version"]
    assert events(unit.ctx, a["attempt_id"], "attempt.end_confirmed") == []
    with pytest.raises(AdmissionDenied):
        resubmit(unit, s, nonce="retry", key="k2")


def test_confirm_ended_needs_an_open_hold(unit: UnitEnv) -> None:
    a, _ = running(unit)
    b = end(unit, a, "COMPLETED")
    with pytest.raises(PreconditionFailed) as exc:
        confirm(unit, b)
    assert "nothing to confirm" in exc.value.message
    c, _ = running(unit)
    with pytest.raises(PreconditionFailed):  # a job hpclab observes is not confirmed by hand
        confirm(unit, c, key="live")


def test_a_later_requeue_segment_is_watched_again(unit: UnitEnv) -> None:
    """A confirmation covers the segment it was given for: a requeue opens a new watch."""
    a, s = running(unit)
    b = end(unit, a, "PREEMPTED")
    confirm(unit, b)
    with unit.ctx.db.tx(source="observer") as tx:  # Slurm requeued it after all ...
        observe.apply_observation(
            tx, unit.ctx, attempt_id=a["attempt_id"], obs=job_obs(unit, a, "RUNNING", restart_count=1)
        )
    assert attempt(unit.ctx, a["attempt_id"])["restart_count"] == 1
    unit.clock.advance(60)
    with unit.ctx.db.tx(source="observer") as tx:  # ... and it was preempted again
        observe.apply_observation(
            tx, unit.ctx, attempt_id=a["attempt_id"], obs=job_obs(unit, a, "PREEMPTED", restart_count=1)
        )
    holds = execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())
    assert [h.kind for h in holds] == ["requeue_watch"]
    with pytest.raises(AdmissionDenied):
        resubmit(unit, s, nonce="retry", key="k2")


# ---------------------------------------------------------------------------
# duplicate owner
# ---------------------------------------------------------------------------


def duplicate(unit: UnitEnv, owner: str = "4999") -> tuple[dict[str, Any], Any]:
    """The bound job 5000 exited 95 at once: job ``owner`` owns the attempt dir (finalize's record)."""
    a, s = running(unit)
    b = end(unit, a, "FAILED")
    with unit.ctx.db.tx(source="worker") as tx:
        observe._record_duplicate(tx, b, owner, 95)
    return b, s


def test_a_duplicate_owner_keeps_the_run_claimed_until_it_is_confirmed_ended(unit: UnitEnv) -> None:
    b, s = duplicate(unit)
    with pytest.raises(AdmissionDenied) as exc:
        resubmit(unit, s, nonce="retry", key="k2")
    err = exc.value
    assert err.details["reason"] == "run_claimed" and err.details["hold"] == "duplicate_owner"
    assert err.details["job_id"] == "4999" and err.details["until"] is None
    assert "Slurm job 4999" in err.message and "confirm it ended" in err.message
    assert not err.retryable
    # no time window: the owner runs as long as it runs
    unit.clock.advance(7 * 24 * 3600)
    with pytest.raises(AdmissionDenied):
        resubmit(unit, s, nonce="retry-2", key="k3")

    # the owner is still running: hpclab cannot confirm it ended (and never cancels it)
    owner_live = obs("4999", name=b["job_name"], comment=f"hpclab:{b['submit_token']}", state="RUNNING")
    unit.scheduler.queue = [owner_live]
    with pytest.raises(PreconditionFailed) as refused:
        confirm(unit, b)
    assert "4999" in refused.value.message and "scancel 4999" in str(refused.value.details)
    assert unit.scheduler.cancel_calls == []

    # it ended: confirmed with the scheduler's evidence, the run can execute again
    unit.scheduler.queue = []
    unit.scheduler.acct = [
        owner_live.model_copy(update={"state": normalize_slurm_state("COMPLETED"), "source": "sacct"})
    ]
    data = confirm(unit, b, key="c2")
    assert data.resolution == "confirm_ended"
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []
    resubmit(unit, s, nonce="retry-3", key="k4")


def test_a_duplicate_owner_of_an_older_attempt_still_holds_the_run(unit: UnitEnv) -> None:
    b, s = duplicate(unit)
    # (a ledger from before this rule: a second attempt was admitted and finished meanwhile)
    with unit.ctx.db.tx() as tx:
        newer = make_attempt(
            tx, s.run_id, phase="terminal", scheduler_state="COMPLETED", cluster_id="unit", job_id="6000"
        )
    assert attempt(unit.ctx, newer)["seq"] == 2
    holds = execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())
    assert [(h.kind, h.attempt_id, h.job_id) for h in holds] == [("duplicate_owner", b["attempt_id"], "4999")]


def test_workflow_dispatch_treats_a_hold_as_a_claim(unit: UnitEnv) -> None:
    """workflows.dispatch waits per cell on claim denials instead of blocking the whole workflow."""
    from hpclab.services.workflows import _is_claim_denial

    a, s = running(unit)
    end(unit, a, "PREEMPTED")
    with pytest.raises(AdmissionDenied) as exc:
        resubmit(unit, s, nonce="retry", key="k2")
    assert _is_claim_denial(exc.value)


# ---------------------------------------------------------------------------
# duplicate owner: the hold ends when the scheduler shows the owner job ended
# ---------------------------------------------------------------------------


def owner_obs(b: dict[str, Any], state: str, *, job_id: str = "4999", **kw: Any) -> JobObservation:
    return obs(job_id, name=b["job_name"], comment=f"hpclab:{b['submit_token']}", state=state, **kw)


def looked_up(unit: UnitEnv, job_id: str) -> int:
    return sum(1 for c in unit.scheduler.queue_calls if c["job_ids"] == [job_id])


def test_the_sweep_lifts_a_duplicate_owner_hold_once_the_scheduler_shows_the_owner_ended(
    unit: UnitEnv,
) -> None:
    """ "Claimed until the owner job is terminal": positive scheduler evidence ends the hold
    without a human (the CLI offers no confirm_ended), a live owner keeps it."""
    b, s = duplicate(unit)
    unit.scheduler.queue = [owner_obs(b, "RUNNING")]
    reconcile_sweep(unit.ctx)
    assert looked_up(unit, "4999") == 1
    assert [h.kind for h in execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())] == [
        "duplicate_owner"
    ]
    assert events(unit.ctx, b["attempt_id"], "attempt.end_confirmed") == []
    assert unit.scheduler.cancel_calls == []

    unit.clock.advance(3600)
    unit.scheduler.queue = []
    unit.scheduler.acct = [owner_obs(b, "COMPLETED", source="sacct")]
    reconcile_sweep(unit.ctx)
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []
    ev = unit.ctx.db.one(
        "SELECT source, payload_json FROM events WHERE entity_id=? AND event_type='attempt.end_confirmed'",
        (b["attempt_id"],),
    )
    assert ev is not None and ev["source"] == "worker"
    assert '"4999"' in ev["payload_json"] and "COMPLETED" in ev["payload_json"]
    assert attempt(unit.ctx, b["attempt_id"])["version"] == b["version"]  # evidence, not a decision
    resubmit(unit, s, nonce="retry", key="k2")


def test_an_owner_the_scheduler_cannot_see_is_never_confirmed_automatically(unit: UnitEnv) -> None:
    """Not visible is not ended (purged accounting, sacct unavailable): only a human's word."""
    _, s = duplicate(unit)
    reconcile_sweep(unit.ctx)
    unit.clock.advance(3600)
    unit.scheduler.acct_error = HpcLabError("sacct: slurmdbd down")
    reconcile_sweep(unit.ctx)
    assert looked_up(unit, "4999") == 2
    assert [h.kind for h in execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())] == [
        "duplicate_owner"
    ]


def test_an_owner_record_of_another_job_is_not_evidence(unit: UnitEnv) -> None:
    """A reused job id (another token) says nothing about the owner of this attempt dir."""
    _, s = duplicate(unit)
    unit.scheduler.acct = [
        obs("4999", name="hl.other.deadbeef", comment="hpclab:" + "d" * 32, state="COMPLETED", source="sacct")
    ]
    reconcile_sweep(unit.ctx)
    assert [h.kind for h in execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())] == [
        "duplicate_owner"
    ]


def test_an_owner_that_may_still_be_requeued_is_not_confirmed_yet(unit: UnitEnv) -> None:
    b, s = duplicate(unit)
    ended = iso(unit.clock.now())
    unit.scheduler.acct = [owner_obs(b, "PREEMPTED", source="sacct").model_copy(update={"end_time": ended})]
    reconcile_sweep(unit.ctx)
    assert [h.kind for h in execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())] == [
        "duplicate_owner"
    ]
    unit.clock.advance(observe.REQUEUE_WATCH_S + 1)
    reconcile_sweep(unit.ctx)
    assert execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now()) == []


def test_duplicate_owner_lookups_are_rate_limited(unit: UnitEnv) -> None:
    b, _ = duplicate(unit)
    unit.scheduler.queue = [owner_obs(b, "RUNNING")]
    reconcile_sweep(unit.ctx)
    reconcile_sweep(unit.ctx)
    unit.clock.advance(60)
    reconcile_sweep(unit.ctx)
    assert looked_up(unit, "4999") == 1
    unit.clock.advance(3600)
    reconcile_sweep(unit.ctx)
    assert looked_up(unit, "4999") == 2


def test_an_unreachable_cluster_does_not_break_the_sweep(unit: UnitEnv) -> None:
    _, s = duplicate(unit)
    unit.scheduler.queue_error = HpcLabError("ssh: connection refused")
    reconcile_sweep(unit.ctx)  # no exception: the owner stays held and is asked again later
    assert [h.kind for h in execution_holds(unit.ctx.db, s.run_id, now=unit.clock.now())] == [
        "duplicate_owner"
    ]
