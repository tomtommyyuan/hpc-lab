"""The observation state machine (apply_observation): spec §4.5/§4.6, table-driven with factories."""

from __future__ import annotations

from typing import Any

import pytest
from observe_support import USER, Env, observation, seed

from hpclab.domain.states import AttemptPhase, RunStatus, SchedulerState
from hpclab.services.observe import apply_observation, mark_observation_gap, run_status_for


def apply(env: Env, attempt_id: str, obs: Any) -> list[str]:
    with env.ctx.db.tx(source="observer") as tx:
        return apply_observation(tx, env.ctx, attempt_id=attempt_id, obs=obs)


# (initial phase, initial state, observed state, extra obs fields) -> (phase, state, run status)
TRANSITIONS = [
    (
        "job_identified",
        "UNKNOWN",
        "PENDING",
        {"reason": "Priority"},
        "observing",
        "PENDING",
        RunStatus.QUEUED,
    ),
    (
        "observing",
        "PENDING",
        "RUNNING",
        {"elapsed_s": 60, "time_limit_s": 3600},
        "observing",
        "RUNNING",
        RunStatus.RUNNING,
    ),
    ("job_identified", "UNKNOWN", "RUNNING", {}, "observing", "RUNNING", RunStatus.RUNNING),
    ("observing", "RUNNING", "COMPLETED", {}, "terminal", "COMPLETED", RunStatus.COMPLETED_UNVERIFIED),
    (
        "observing",
        "RUNNING",
        "FAILED",
        {"source": "sacct", "exit_code": 1},
        "terminal",
        "FAILED",
        RunStatus.FAILED,
    ),
    ("observing", "RUNNING", "TIMEOUT", {}, "terminal", "TIMEOUT", RunStatus.FAILED),
    ("observing", "RUNNING", "OUT_OF_MEMORY", {}, "terminal", "OUT_OF_MEMORY", RunStatus.FAILED),
    ("observing", "PENDING", "CANCELLED", {}, "terminal", "CANCELLED", RunStatus.CANCELLED),
    ("job_identified", "UNKNOWN", "NODE_FAIL", {}, "terminal", "NODE_FAIL", RunStatus.FAILED),
    ("observing", "RUNNING", "COMPLETING", {}, "observing", "COMPLETING", RunStatus.RUNNING),
]


@pytest.mark.parametrize(
    ("phase", "state", "seen", "extra", "want_phase", "want_state", "want_run"), TRANSITIONS
)
def test_transition_table(
    env: Env,
    phase: str,
    state: str,
    seen: str,
    extra: dict[str, Any],
    want_phase: str,
    want_state: str,
    want_run: RunStatus,
) -> None:
    run_id, aid = seed(env, phase=phase, scheduler_state=state)
    before = env.attempt(aid)["version"]
    changes = apply(env, aid, observation(env, aid, seen, **extra))

    a = env.attempt(aid)
    assert (a["phase"], a["scheduler_state"]) == (want_phase, want_state), changes
    assert a["version"] == before + 1  # material transition: versioned + evented
    assert env.run(run_id)["status"] == want_run.value
    terminal = want_phase == "terminal"
    assert bool(env.events(aid, "attempt.terminal")) is terminal
    assert bool(env.events(aid, "attempt.observed")) is not terminal
    assert len(env.outbox("observe.finalize")) == (1 if terminal else 0)
    run = env.run(run_id)
    assert run["claim_attempt_id"] == (None if terminal else aid)  # claim released only on terminal
    if terminal:
        assert a["terminal_at"] == env.now()
        assert a["time_left_s"] is None and a["pending_reason"] is None
        assert env.events(run_id, "run.claim_released")


def test_pending_reason_and_time_left_are_recorded(env: Env) -> None:
    _, aid = seed(env, phase="job_identified", scheduler_state="UNKNOWN")
    apply(env, aid, observation(env, aid, "PENDING", reason="QOSMaxGRESPerUser", time_limit_s=7200))
    a = env.attempt(aid)
    assert a["pending_reason"] == "QOSMaxGRESPerUser" and a["time_left_s"] is None
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "RUNNING", elapsed_s=600, time_limit_s=7200, node_list="gpu-01"))
    a = env.attempt(aid)
    assert a["pending_reason"] is None
    assert a["time_left_s"] == 6600 and a["node_list"] == "gpu-01"
    assert a["job_submit_time"] is None  # nothing observed yet
    seg = env.ctx.db.all("SELECT * FROM attempt_segments WHERE attempt_id=?", (aid,))
    assert [(s["segment"], s["state"], s["node_list"]) for s in seg] == [(0, "RUNNING", "gpu-01")]


def test_unchanged_state_refreshes_freshness_without_version_or_event(env: Env) -> None:
    _, aid = seed(env, phase="observing", scheduler_state="RUNNING")
    apply(env, aid, observation(env, aid, "RUNNING", elapsed_s=10))
    before = env.attempt(aid)
    n_events = len(env.events(aid))
    env.clock.advance(30)
    changes = apply(env, aid, observation(env, aid, "RUNNING", elapsed_s=40, time_limit_s=100))
    after = env.attempt(aid)
    assert changes == []
    assert after["version"] == before["version"]
    assert len(env.events(aid)) == n_events
    assert after["last_observed_at"] == env.now() and after["elapsed_s"] == 40 and after["time_left_s"] == 60


def test_terminal_is_never_moved_back_by_a_stale_or_late_observation(env: Env) -> None:
    run_id, aid = seed(env)
    apply(env, aid, observation(env, aid, "COMPLETED"))
    snapshot = env.attempt(aid)
    env.clock.advance(5)
    # A late RUNNING (same restart count) from another source must not resurrect the job.
    assert apply(env, aid, observation(env, aid, "RUNNING", restart_count=0)) == ["ignored:terminal"]
    assert apply(env, aid, observation(env, aid, "PENDING", source="sacct")) == ["ignored:terminal"]
    after = env.attempt(aid)
    assert after["phase"] == "terminal" and after["scheduler_state"] == "COMPLETED"
    assert after["version"] == snapshot["version"]
    assert env.run(run_id)["claim_attempt_id"] is None


def test_out_of_order_observations_are_ignored(env: Env) -> None:
    _, aid = seed(env, phase="observing", scheduler_state="PENDING")
    old = observation(env, aid, "PENDING")
    env.clock.advance(60)
    apply(env, aid, observation(env, aid, "RUNNING"))
    assert apply(env, aid, old) == ["ignored:out_of_order"]
    assert env.attempt(aid)["scheduler_state"] == "RUNNING"


@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"user": "mallory"}, "owner"),
        ({"name": "somebody-elses-job"}, "token"),
        ({"name": "hl.run.zzzzzzzz"}, "token"),
        ({"comment": "hpclab:abcdefghjkmn"}, "comment"),
    ],
)
def test_identity_mismatch_is_never_applied_and_recorded_once(
    env: Env, override: dict[str, Any], why: str
) -> None:
    run_id, aid = seed(env, phase="observing", scheduler_state="RUNNING")
    for _ in range(2):
        env.clock.advance(1)
        changes = apply(env, aid, observation(env, aid, "COMPLETED", **override))
        assert changes[0].startswith("ignored:identity"), changes
    a = env.attempt(aid)
    assert a["phase"] == "observing" and a["scheduler_state"] == "RUNNING"
    assert env.run(run_id)["claim_attempt_id"] == aid
    assert len(env.events(aid, "attempt.identity_mismatch")) == 1


def test_submit_time_far_apart_is_another_job(env: Env) -> None:
    _, aid = seed(env, job_submit_time="2026-09-01T11:59:00.000Z")
    ok = observation(env, aid, "RUNNING", submit_time="2026-09-01T11:59:03.000Z")
    reused = observation(env, aid, "COMPLETED", submit_time="2026-08-01T00:00:00.000Z")
    assert apply(env, aid, reused)[0].startswith("ignored:identity")
    assert apply(env, aid, ok) == [] or env.attempt(aid)["scheduler_state"] == "RUNNING"


@pytest.mark.parametrize(
    ("phase", "job_obs_id", "expected"),
    [
        ("submission_uncertain", "1001", "ignored:phase=submission_uncertain"),
        ("staging", "1001", "ignored:phase=staging"),
        ("observing", "9999", "ignored:job_id"),
    ],
)
def test_observations_for_other_phases_or_jobs_are_ignored(
    env: Env, phase: str, job_obs_id: str, expected: str
) -> None:
    _, aid = seed(env, phase=phase, scheduler_state="UNKNOWN")
    obs = observation(env, aid, "COMPLETED", job_id=job_obs_id)
    assert apply(env, aid, obs) == [expected]
    assert env.attempt(aid)["phase"] == phase


def test_cancel_intent_is_confirmed_by_a_terminal_observation(env: Env) -> None:
    run_id, aid = seed(env, cancel_state="sent")
    changes = apply(env, aid, observation(env, aid, "CANCELLED", raw_state="CANCELLED by 1000"))
    assert "cancel:confirmed" in changes
    assert env.attempt(aid)["cancel_state"] == "confirmed"
    assert env.run(run_id)["status"] == RunStatus.CANCELLED.value


def test_accounting_fills_exit_fields_of_a_squeue_terminal_without_new_version(env: Env) -> None:
    _, aid = seed(env)
    apply(env, aid, observation(env, aid, "FAILED"))
    v = env.attempt(aid)["version"]
    env.clock.advance(5)
    changes = apply(
        env,
        aid,
        observation(
            env, aid, "FAILED", source="sacct", exit_code=2, exit_signal=0, elapsed_s=321, end_time=env.now()
        ),
    )
    a = env.attempt(aid)
    assert "filled:exit_code" in changes and a["exit_code"] == 2 and a["elapsed_s"] == 321
    assert a["version"] == v
    # accounting (authoritative for finished jobs) may revise the terminal state, with an event
    env.clock.advance(5)
    changes = apply(env, aid, observation(env, aid, "NODE_FAIL", source="sacct"))
    assert "revised:FAILED->NODE_FAIL" in changes
    assert env.attempt(aid)["version"] == v + 1
    assert env.events(aid, "attempt.terminal_revised")


def test_finalize_is_enqueued_exactly_once(env: Env) -> None:
    _, aid = seed(env)
    apply(env, aid, observation(env, aid, "COMPLETED"))
    env.clock.advance(1)
    apply(env, aid, observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=5))
    items = env.outbox("observe.finalize")
    assert len(items) == 1 and '"attempt_id"' in items[0]["payload_json"]


# ---------------------------------------------------------------------------
# requeue -> segments
# ---------------------------------------------------------------------------


def test_requeue_after_terminal_opens_a_new_segment_and_retakes_the_claim(env: Env) -> None:
    run_id, aid = seed(env)
    apply(env, aid, observation(env, aid, "RUNNING", start_time="2026-09-01T11:00:00.000Z", node_list="n1"))
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "PREEMPTED", end_time=env.now(), restart_count=0))
    assert env.run(run_id)["claim_attempt_id"] is None
    env.clock.advance(10)
    changes = apply(env, aid, observation(env, aid, "PENDING", restart_count=1, reason="BeginTime"))

    assert "requeue:0->1" in changes
    a = env.attempt(aid)
    assert a["phase"] == AttemptPhase.OBSERVING.value and a["scheduler_state"] == "PENDING"
    assert a["restart_count"] == 1 and a["terminal_at"] is None and a["end_time"] is None
    assert a["result_state"] == "pending"
    assert env.run(run_id)["claim_attempt_id"] == aid
    assert env.run(run_id)["status"] == RunStatus.QUEUED.value
    segs = env.ctx.db.all("SELECT * FROM attempt_segments WHERE attempt_id=? ORDER BY segment", (aid,))
    assert [(s["segment"], s["state"], s["node_list"]) for s in segs] == [
        (0, "PREEMPTED", "n1"),
        (1, "PENDING", None),
    ]
    assert segs[0]["start_time"] == "2026-09-01T11:00:00.000Z" and segs[0]["end_time"] is not None
    ev = env.events(aid, "attempt.requeued")
    assert len(ev) == 1 and '"claim":"reclaimed"' in ev[0]["payload_json"]
    # the second segment ends: a second finalize is enqueued for restart 1
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "COMPLETED", restart_count=1))
    assert len(env.outbox("observe.finalize")) == 2


def test_requeue_while_active_closes_the_old_segment(env: Env) -> None:
    _, aid = seed(env)
    apply(env, aid, observation(env, aid, "RUNNING", node_list="n1", start_time="2026-09-01T11:00:00.000Z"))
    env.clock.advance(10)
    changes = apply(env, aid, observation(env, aid, "PENDING", restart_count=1))
    assert "requeue:0->1" in changes
    segs = env.ctx.db.all("SELECT * FROM attempt_segments WHERE attempt_id=? ORDER BY segment", (aid,))
    assert [(s["segment"], s["state"]) for s in segs] == [(0, "REQUEUED"), (1, "PENDING")]
    a = env.attempt(aid)
    assert a["restart_count"] == 1 and a["start_time"] is None and a["node_list"] is None


def test_requeue_with_the_claim_taken_by_another_attempt_is_reported(env: Env) -> None:
    run_id, aid = seed(env)
    apply(env, aid, observation(env, aid, "NODE_FAIL"))
    _, other = seed(env, run_id=run_id, phase="staging", scheduler_state="UNKNOWN", job_id=None)
    env.clock.advance(5)
    changes = apply(env, aid, observation(env, aid, "RUNNING", restart_count=1))
    assert "claim_conflict" in changes
    assert env.run(run_id)["claim_attempt_id"] == other
    assert "conflict" in env.events(aid, "attempt.requeued")[0]["payload_json"]


# ---------------------------------------------------------------------------
# observation gap + run status
# ---------------------------------------------------------------------------


def test_observation_gap_is_flagged_once_and_closed_by_the_next_observation(env: Env) -> None:
    run_id, aid = seed(env)
    with env.ctx.db.tx() as tx:
        assert mark_observation_gap(tx, attempt_id=aid, reason="not in squeue") == ["gap_opened"]
        assert mark_observation_gap(tx, attempt_id=aid, reason="not in squeue") == []
    a = env.attempt(aid)
    assert a["observation_gap"] == 1 and a["scheduler_state"] == "RUNNING" and a["phase"] == "observing"
    assert env.run(run_id)["status"] == RunStatus.UNKNOWN.value
    assert len(env.events(aid, "attempt.observation_gap")) == 1
    changes = apply(env, aid, observation(env, aid, "COMPLETED", source="sacct", exit_code=0))
    assert "gap_closed" in changes
    assert env.attempt(aid)["observation_gap"] == 0


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (None, RunStatus.PREPARED),
        ({"phase": "staging"}, RunStatus.SUBMITTING),
        ({"phase": "submission_in_flight"}, RunStatus.SUBMITTING),
        ({"phase": "submission_uncertain"}, RunStatus.UNKNOWN),
        ({"phase": "duplicate_detected"}, RunStatus.UNKNOWN),
        ({"phase": "submission_rejected"}, RunStatus.REJECTED),
        ({"phase": "cancelled_before_submit"}, RunStatus.CANCELLED),
        ({"phase": "abandoned"}, RunStatus.CANCELLED),
        ({"phase": "job_identified", "scheduler_state": "UNKNOWN"}, RunStatus.QUEUED),
        ({"phase": "observing", "scheduler_state": "PENDING"}, RunStatus.QUEUED),
        ({"phase": "observing", "scheduler_state": "RUNNING"}, RunStatus.RUNNING),
        ({"phase": "observing", "scheduler_state": "RUNNING", "observation_gap": 1}, RunStatus.UNKNOWN),
        ({"phase": "terminal", "scheduler_state": "COMPLETED", "result_state": "valid"}, RunStatus.SUCCEEDED),
        (
            {"phase": "terminal", "scheduler_state": "COMPLETED", "result_state": "unchecked"},
            RunStatus.COMPLETED_UNVERIFIED,
        ),
        ({"phase": "terminal", "scheduler_state": "TIMEOUT"}, RunStatus.FAILED),
        ({"phase": "terminal", "scheduler_state": "CANCELLED"}, RunStatus.CANCELLED),
    ],
)
def test_run_status_mapping(row: dict[str, Any] | None, expected: RunStatus) -> None:
    assert run_status_for(row) == expected


def test_identity_fields_are_filled_from_the_first_observation(env: Env) -> None:
    _, aid = seed(env, job_owner=None)
    apply(env, aid, observation(env, aid, "PENDING", submit_time="2026-09-01T11:58:00.000Z", partition="gpu"))
    a = env.attempt(aid)
    assert a["job_owner"] == USER and a["job_submit_time"] == "2026-09-01T11:58:00.000Z"
    assert a["partition"] == "gpu"
    assert SchedulerState(a["scheduler_state"]) == SchedulerState.PENDING


def test_pending_reason_flips_are_evented_without_invalidating_expected_version(env: Env) -> None:
    _, aid = seed(env, phase="observing", scheduler_state="PENDING", pending_reason="Priority")
    v = env.attempt(aid)["version"]
    env.clock.advance(1)
    assert apply(env, aid, observation(env, aid, "PENDING", reason="Resources")) == [
        "pending_reason:Resources"
    ]
    a = env.attempt(aid)
    assert a["version"] == v and a["pending_reason"] == "Resources"
    ev = env.events(aid, "attempt.pending_reason")
    assert len(ev) == 1 and ev[0]["entity_version"] == v
