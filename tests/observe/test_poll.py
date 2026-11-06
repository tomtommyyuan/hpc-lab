"""poll_cluster / refresh_attempts / poll_all against a scripted scheduler (FakeClock)."""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest
from observe_support import (
    CLUSTER,
    USER,
    Env,
    Recorder,
    auth_error,
    install_stub,
    observation,
    seed,
)

from hpclab.adapters.base import CommandFailed, CommandResult, TransportError
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import ErrorCode
from hpclab.domain.models import JobObservation
from hpclab.domain.states import SchedulerState
from hpclab.services import observe
from hpclab.services.observe import poll_all, poll_cluster, refresh_attempts


def failing(message: str = "squeue: slurm_load_jobs error: Socket timed out") -> CommandFailed:
    return CommandFailed(message, CommandResult(["squeue"], 1, "", message, 0.1))


def test_one_batched_squeue_for_all_attempts_and_foreign_jobs_summary(env: Env) -> None:
    _, a1 = seed(env, job_id="1001", phase="job_identified", scheduler_state="UNKNOWN")
    _, a2 = seed(env, job_id="1002", phase="observing", scheduler_state="PENDING")
    env.scheduler.queue = [
        observation(env, a1, "RUNNING", elapsed_s=5, time_limit_s=100),
        observation(env, a2, "PENDING", reason="Resources"),
        JobObservation(
            job_id="555",
            source="squeue",
            observed_at=env.now(),
            state=SchedulerState.RUNNING,
            name="my-notebook",
            user=USER,
            partition="interactive",
            alloc_gpus=1,
        ),
        JobObservation(
            job_id="556",
            source="squeue",
            observed_at=env.now(),
            state=SchedulerState.PENDING,
            name="eval",
            user=USER,
            partition="gpu",
            alloc_gpus=None,
        ),
        JobObservation(  # an hpclab-named job no attempt is bound to (e.g. a duplicate)
            job_id="557",
            source="squeue",
            observed_at=env.now(),
            state=SchedulerState.PENDING,
            name="hl.x.abcdefgh",
            user=USER,
        ),
    ]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)

    assert res.ok and res.polled and res.attempts_polled == 2
    assert env.scheduler.calls == ["squeue:user"]  # one call; no sacct needed
    assert res.sacct == "not_needed" and res.foreign_jobs == 2
    assert env.attempt(a1)["scheduler_state"] == "RUNNING" and env.attempt(a1)["phase"] == "observing"
    assert env.attempt(a2)["pending_reason"] == "Resources"
    row = env.ctx.db.one(
        "SELECT * FROM observations WHERE cluster_id=? AND obs_key='queue:foreign_jobs'", (CLUSTER,)
    )
    assert row is not None
    assert '"count":2' in row["value_json"] and '"unbound_hpclab_jobs":1' in row["value_json"]
    assert '"gpus":1' in row["value_json"] and '"gpus_unknown_jobs":1' in row["value_json"]
    ps = env.poll_state("squeue")
    assert ps["last_ok_at"] == env.now() and ps["consecutive_failures"] == 0


def test_rate_limit_is_shared_and_force_bypasses_it(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    env.clock.advance(10)  # policy: 30 s
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert not res.polled and res.skipped_reason == "too_soon"
    assert res.next_poll_after == iso(env.clock.now() + timedelta(seconds=20))
    assert env.scheduler.calls == ["squeue:user"]
    assert poll_cluster(env.ctx, cluster_id=CLUSTER, force=True).ok
    env.clock.advance(31)
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    assert env.scheduler.calls == ["squeue:user"] * 3


def test_nothing_to_observe_means_no_scheduler_calls(env: Env) -> None:
    seed(env, phase="terminal", scheduler_state="COMPLETED")
    seed(env, phase="staging", scheduler_state="UNKNOWN", job_id=None)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.skipped_reason == "no_active_attempts" and not res.polled
    assert env.scheduler.calls == []


def test_missing_from_squeue_asks_sacct_once_for_all(env: Env) -> None:
    run1, a1 = seed(env, job_id="1001")
    run2, a2 = seed(env, job_id="1002")
    env.scheduler.queue = []
    env.scheduler.acct = [observation(env, a1, "COMPLETED", source="sacct", exit_code=0, elapsed_s=42)]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)

    assert env.scheduler.calls == ["squeue:user", "sacct:1001,1002"]
    assert res.sacct == "ok" and res.terminal == [a1] and res.gaps == [a2]
    assert env.attempt(a1)["phase"] == "terminal" and env.attempt(a1)["elapsed_s"] == 42
    # not in squeue + no accounting row: flagged, never "COMPLETED"
    a = env.attempt(a2)
    assert a["observation_gap"] == 1 and a["phase"] == "observing" and a["scheduler_state"] == "RUNNING"
    assert env.run(run2)["status"] == "unknown" and env.run(run2)["claim_attempt_id"] == a2
    assert env.run(run1)["claim_attempt_id"] is None


def test_accounting_lag_then_min_job_age_gap_closes_when_sacct_catches_up(env: Env) -> None:
    run_id, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    # job ended and left squeue (MinJobAge) before we looked; accounting lags behind
    env.scheduler.queue = []
    env.clock.advance(61)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.gaps == [aid] and env.attempt(aid)["scheduler_state"] == "RUNNING"
    # sacct is rate limited separately (60 s): the next squeue poll defers it and changes nothing
    env.clock.advance(31)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.sacct == "deferred" and env.attempt(aid)["observation_gap"] == 1
    env.clock.advance(30)
    env.scheduler.acct = [observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=99)]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert res.terminal == [aid]
    assert a["phase"] == "terminal" and a["observation_gap"] == 0 and a["exit_code"] == 0
    assert env.run(run_id)["status"] == "completed_unverified"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (TransportError("accounting storage is disabled", code=ErrorCode.UNSUPPORTED), "unsupported"),
        (failing("sacct: error: Problem talking to the database"), "unavailable"),
    ],
)
def test_sacct_unavailable_never_turns_into_completed(env: Env, error: Exception, status: str) -> None:
    _, aid = seed(env)
    env.scheduler.acct_error = error
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.sacct == status and res.gaps == [aid] and res.ok
    a = env.attempt(aid)
    assert a["scheduler_state"] == "RUNNING" and a["phase"] == "observing" and a["observation_gap"] == 1
    ps = env.poll_state("sacct")
    assert ps["consecutive_failures"] == 1 and ps["last_error"]


def test_squeue_failure_backs_off_and_recovers(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue_error = failing()
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.polled and not res.ok and res.error_code == "unavailable"
    ps = env.poll_state("squeue")
    until = parse_iso(ps["backoff_until"])
    assert (
        ps["consecutive_failures"] == 1
        and until is not None
        and until > env.clock.now() + timedelta(seconds=30)
    )
    assert len(env.events(CLUSTER, "cluster.poll_failed")) == 1
    assert env.attempt(aid)["scheduler_state"] == "RUNNING"  # nothing guessed
    env.clock.advance(31)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.skipped_reason == "backoff"
    env.clock.advance(1000)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)  # still failing: longer backoff, one event only
    ps2 = env.poll_state("squeue")
    assert ps2["consecutive_failures"] == 2 and len(env.events(CLUSTER, "cluster.poll_failed")) == 1
    env.scheduler.queue_error = None
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    env.clock.advance(1000)
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    ps3 = env.poll_state("squeue")
    assert ps3["consecutive_failures"] == 0 and ps3["backoff_until"] is None
    assert env.events(CLUSTER, "cluster.poll_recovered")


def test_auth_required_pauses_polling_and_notifies_once(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    emit = Recorder()
    install_stub(monkeypatch, "hpclab.services.notify", emit=emit)
    _, aid = seed(env)
    env.scheduler.queue_error = auth_error()
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.auth_required and res.error_code == "auth_required"
    ps = env.poll_state("squeue")
    assert parse_iso(ps["backoff_until"]) == env.clock.now() + timedelta(seconds=observe.AUTH_PAUSE_S)
    assert "auth_required" in ps["last_error"]
    assert [c["event_type"] for c in emit.calls] == ["auth_required"]
    assert emit.calls[0]["urgency"] == "immediate" and "hpclab auth c1" in emit.calls[0]["body"]
    env.clock.advance(observe.AUTH_PAUSE_S + 1)
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert len(emit.calls) == 1  # still failing: no second notification
    assert env.attempt(aid)["scheduler_state"] == "RUNNING"


def test_requeued_job_is_watched_after_a_preemption(env: Env) -> None:
    run_id, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "PREEMPTED", restart_count=0)]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert env.attempt(aid)["phase"] == "terminal"
    env.clock.advance(31)
    env.scheduler.queue = [observation(env, aid, "PENDING", restart_count=1, reason="BeginTime")]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert any("requeue:0->1" in c for c in res.changes)
    assert env.attempt(aid)["phase"] == "observing" and env.run(run_id)["claim_attempt_id"] == aid


def test_completed_attempts_are_not_watched(env: Env) -> None:
    seed(env, phase="terminal", scheduler_state="COMPLETED", terminal_at=iso(env.clock.now()))
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).skipped_reason == "no_active_attempts"


def test_live_views_of_running_attempts_are_cached(env: Env) -> None:
    _, aid = seed(env, gpus=1)
    a = env.attempt(aid)
    d = a["remote_dir"]
    now = env.clock.now().timestamp()
    env.transport.put(f"{d}/outputs/metrics.jsonl", f'{{"step": 10, "total_steps": 100, "time": {now}}}\n')
    env.transport.put(f"{d}/telemetry/gpu.csv", "t, 0, GPU-a, 0, 100, 80000, 60\n" * 4)
    env.transport.put(f"{d}/telemetry/heartbeat", f"{int(now)}\n")
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=100, time_limit_s=3600)]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert '"step":10' in a["progress_json"]
    assert '"idle_allocation"' in a["health_json"]


def test_refresh_is_rate_limited_and_bounded(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    first = refresh_attempts(env.ctx, attempt_ids=[aid], deadline_s=5)
    assert first.refreshed == [aid] and not first.stale
    again = refresh_attempts(env.ctx, attempt_ids=[aid], deadline_s=5)
    assert again.refreshed == [] and not again.stale and "polled recently" in again.notes[0]

    env.clock.advance(31)
    env.scheduler.block = threading.Event()
    t0 = time.monotonic()
    slow = refresh_attempts(env.ctx, attempt_ids=[aid], deadline_s=0.3)
    assert time.monotonic() - t0 < 3
    assert slow.stale and "did not finish" in slow.notes[0]
    env.scheduler.block.set()  # the background poll completes later and still records its result
    assert observe.wait_for_background(10)
    env.scheduler.block = None
    ps = env.poll_state("squeue")
    assert ps["last_ok_at"] == env.now()


def test_refresh_reports_failures_as_stale(env: Env) -> None:
    _, aid = seed(env)
    _, done = seed(env, phase="terminal", scheduler_state="COMPLETED")
    env.scheduler.queue_error = failing()
    res = refresh_attempts(env.ctx, attempt_ids=[aid, done], deadline_s=5)
    assert res.stale and any("failed" in n for n in res.notes)
    assert any("is not polled" in n for n in res.notes)


def test_poll_all_survives_a_broken_cluster(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def boom(ctx: object, *, cluster_id: str, force: bool = False) -> None:
        seen.append(cluster_id)
        raise RuntimeError("bug")

    monkeypatch.setattr(observe, "poll_cluster", boom)
    poll_all(env.ctx)
    assert seen == [CLUSTER]


def test_periodic_and_outbox_registration() -> None:
    tasks = observe.periodic_tasks()
    assert [t.name for t in tasks] == ["observe.poll_all"] and tasks[0].fn is observe.poll_all
    assert observe.outbox_handlers() == {
        "observe.finalize": observe.finalize_attempt,
        "observe.settle": observe.resettle_attempt,
    }


def test_running_estimate_is_fed_to_the_budget_throttled(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    update = Recorder()
    install_stub(monkeypatch, "hpclab.services.budget", update_estimate=update)
    _, aid = seed(env, gpus=2)
    for elapsed in (100, 200, 290, 400, 1300):
        env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=elapsed, time_limit_s=3600)]
        env.clock.advance(31)
        assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    # one write per 5-minute bucket of elapsed time: 100, 400, 1300
    assert [round(c["estimated_gpu_hours"] * 3600) for c in update.calls] == [200, 800, 2600]
    assert all(c["attempt_id"] == aid for c in update.calls)


def test_partially_deployed_budget_module_does_not_break_polling(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_stub(monkeypatch, "hpclab.services.budget")  # no functions yet
    _, aid = seed(env, gpus=1)
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=50)]
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    assert env.attempt(aid)["elapsed_s"] == 50


def test_refresh_during_backoff_is_stale(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue_error = failing()
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    res = refresh_attempts(env.ctx, attempt_ids=[aid], deadline_s=5)
    assert res.stale and "backing off" in res.notes[0]
