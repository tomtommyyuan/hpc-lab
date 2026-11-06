"""Reliability review of the observation pipeline (second round): requeues the controller did not
see, requeue budget accounting, duplicate owner jobs, accounting outages, batched/rate-limited
accounting reads, refresh honesty and wall-clock steps. Each test names the defect it pins."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from observe_support import (
    CLUSTER,
    USER,
    Env,
    Recorder,
    install_stub,
    observation,
    put_json,
    seed,
)

from hpclab.adapters.base import TransportTimeout
from hpclab.domain.errors import BudgetExceeded
from hpclab.services import budget, observe
from hpclab.services.jobscript import DUPLICATE_MARKER, EXIT_DUPLICATE_JOB
from hpclab.services.observe import (
    apply_observation,
    finalize_attempt,
    poll_cluster,
    refresh_attempts,
)
from hpclab.services.outbox import OutboxItem, OutboxRunner

T0 = "2026-09-01T12:00:00.000Z"


def apply(env: Env, attempt_id: str, obs: Any) -> list[str]:
    with env.ctx.db.tx(source="observer") as tx:
        return apply_observation(tx, env.ctx, attempt_id=attempt_id, obs=obs)


def item(aid: str, tries: int = 1, kind: str = "observe.finalize", **payload: Any) -> OutboxItem:
    return OutboxItem(
        outbox_id=1,
        kind=kind,
        operation_id=None,
        target_id=aid,
        payload={"attempt_id": aid, **payload},
        tries=tries,
    )


def notify_stub(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    emit = Recorder()
    install_stub(monkeypatch, "hpclab.services.notify", emit=emit)
    return emit


def with_budget(env: Env, run_id: str, aid: str, *, budget_gpu_hours: float, reserved: float) -> str:
    eid = env.run(run_id)["experiment_id"]
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE experiments SET budget_gpu_hours=? WHERE experiment_id=?", (budget_gpu_hours, eid))
        budget.reserve(tx, experiment_id=eid, attempt_id=aid, gpu_hours=reserved)
    return str(eid)


def reservation(env: Env, aid: str) -> dict[str, Any]:
    row = budget.reservation(env.ctx.db, attempt_id=aid)
    assert row is not None
    return row


def status(env: Env, aid: str, **fields: Any) -> None:
    d = env.attempt(aid)["remote_dir"]
    put_json(env, f"{d}/status.json", {"schema": "hpclab.status/v1", "attempt_id": aid, **fields})


def segments(env: Env, aid: str) -> list[dict[str, Any]]:
    return env.ctx.db.all(
        "SELECT segment, start_time, end_time, state FROM attempt_segments WHERE attempt_id=? ORDER BY segment",
        (aid,),
    )


# ---------------------------------------------------------------------------
# A requeue the controller did not see (Slurm restamps SubmitTime; sacct has no restart count)
# ---------------------------------------------------------------------------


def _unseen_requeue(env: Env) -> tuple[str, str]:
    """Attempt observed at 11:00 (restart 0); overnight the job was preempted, requeued (new
    SubmitTime 13:00:05), ran again, COMPLETED and left squeue. sacct (no --duplicates) only has
    the latest record, without a restart count."""
    run_id, aid = seed(
        env,
        job_id="1001",
        job_submit_time="2026-09-01T11:00:00.000Z",
        start_time="2026-09-01T11:01:00.000Z",
        gpus=1,
    )
    env.clock.advance(12 * 3600)
    env.scheduler.queue = []
    latest = observation(
        env,
        aid,
        "COMPLETED",
        source="sacct",
        exit_code=0,
        elapsed_s=25_000,
        submit_time="2026-09-01T13:00:05.000Z",
        start_time="2026-09-01T13:02:00.000Z",
        end_time="2026-09-01T20:00:00.000Z",
        alloc_gpus=1,
    )
    env.scheduler.acct = [latest]
    first = observation(
        env,
        aid,
        "PREEMPTED",
        source="sacct",
        submit_time="2026-09-01T11:00:00.000Z",
        start_time="2026-09-01T11:01:00.000Z",
        end_time="2026-09-01T13:00:04.000Z",
        elapsed_s=7144,
        alloc_gpus=1,
    )
    env.scheduler.segments["1001"] = [first, latest]
    return run_id, aid


def test_unseen_requeue_is_verified_from_per_segment_accounting_and_finalized(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before: the requeued job's own record failed identity_mismatch (submit time drift), the
    attempt stayed OBSERVING in a gap forever, holding its claim, budget and workflow slot."""
    notify_stub(monkeypatch)
    run_id, aid = _unseen_requeue(env)
    with_budget(env, run_id, aid, budget_gpu_hours=40, reserved=8.0)

    res = poll_cluster(env.ctx, cluster_id=CLUSTER, force=True)

    a = env.attempt(aid)
    assert res.terminal == [aid], res
    assert a["phase"] == "terminal" and a["scheduler_state"] == "COMPLETED"
    assert a["restart_count"] == 1 and a["observation_gap"] == 0 and a["exit_code"] == 0
    assert env.events(aid, "attempt.identity_mismatch") == []
    assert len(env.events(aid, "attempt.requeued")) == 1
    assert len(env.events(aid, "attempt.requeue_verified")) == 1
    assert env.run(run_id)["claim_attempt_id"] is None
    segs = segments(env, aid)
    assert [s["segment"] for s in segs] == [0, 1]
    assert segs[0]["end_time"] == "2026-09-01T13:00:04.000Z"
    assert "sacct-D:1001" in env.scheduler.calls

    # the cancel-time identity check accepts the job's latest record now (restart recorded)
    from hpclab.services.submission import identity_problems

    problems = identity_problems(
        env.attempt(aid), env.scheduler.acct[0], remote_user=USER, now=env.clock.now(), skew_s=600
    )
    assert problems == []

    status(env, aid, state="exited", exit_code=0, restart_count=1, job_id="1001")
    out = finalize_attempt(env.ctx, item(aid))
    assert out is not None and out.action == "done", out
    row = reservation(env, aid)
    # 7144 s (segment 0) + 25000 s (segment 1) on 1 GPU, exactly
    assert row["state"] == "settled" and row["settled_consumed"] == round(32_144 / 3600, 6)


def test_unverifiable_later_submit_time_stays_a_gap_with_an_honest_reason(env: Env) -> None:
    _, aid = _unseen_requeue(env)
    env.scheduler.segments["1001"] = []  # --duplicates has nothing that chains to our submit time
    res = poll_cluster(env.ctx, cluster_id=CLUSTER, force=True)
    a = env.attempt(aid)
    assert a["phase"] == "observing" and a["observation_gap"] == 1 and res.gaps == [aid]
    gap = env.events(aid, "attempt.observation_gap")[0]["payload_json"]
    assert "requeue" in gap


def test_an_unverifiable_requeue_is_not_rechecked_on_every_poll(env: Env) -> None:
    """The per-segment check is one more per-job sacct: an attempt whose chain cannot be verified
    must not add a ``sacct --duplicates`` call to every poll for as long as it stays in the gap."""
    _, aid = _unseen_requeue(env)
    good = list(env.scheduler.segments["1001"])
    env.scheduler.segments["1001"] = []
    for _ in range(10):  # ten sacct slots, one a minute
        poll_cluster(env.ctx, cluster_id=CLUSTER)
        env.clock.advance(61)
    checks = [c for c in env.scheduler.calls if c.startswith("sacct-D")]
    assert len(checks) == 1, checks
    assert env.attempt(aid)["phase"] == "observing"

    # accounting catches up (e.g. slurmdbd had not stored the earlier segment yet): rechecked later
    env.scheduler.segments["1001"] = good
    env.clock.advance(observe.REQUEUE_RECHECK_S)
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert a["phase"] == "terminal" and a["restart_count"] == 1
    assert len([c for c in env.scheduler.calls if c.startswith("sacct-D")]) == 2
    assert env.ctx.db.one("SELECT 1 AS x FROM observations WHERE obs_key LIKE 'requeue_check:%'") is None


def test_a_foreign_job_or_an_earlier_submit_time_never_triggers_a_requeue_check(env: Env) -> None:
    _, aid = _unseen_requeue(env)
    env.scheduler.acct = [
        env.scheduler.acct[0].model_copy(update={"name": "hl.other.zzzzzzzz"}),
    ]
    poll_cluster(env.ctx, cluster_id=CLUSTER, force=True)
    assert env.attempt(aid)["phase"] == "observing"
    env.clock.advance(120)
    env.scheduler.acct = [
        observation(
            env, aid, "COMPLETED", source="sacct", submit_time="2026-09-01T09:00:00.000Z", elapsed_s=5
        )
    ]
    poll_cluster(env.ctx, cluster_id=CLUSTER, force=True)
    assert env.attempt(aid)["phase"] == "observing"
    assert not any(c.startswith("sacct-D") for c in env.scheduler.calls)


# ---------------------------------------------------------------------------
# Requeue seen while active: settlement and the reservation as an upper bound
# ---------------------------------------------------------------------------


def _requeue_while_running(env: Env) -> tuple[str, str, str]:
    run_id, aid = seed(
        env, gpus=2, time_limit_s=7200, job_submit_time="2026-09-01T09:59:00.000Z", start_time=None
    )
    eid = with_budget(env, run_id, aid, budget_gpu_hours=10, reserved=4.0)
    apply(env, aid, observation(env, aid, "RUNNING", start_time="2026-09-01T10:00:00.000Z", elapsed_s=6000))
    env.clock.advance(10)
    changes = apply(env, aid, observation(env, aid, "PENDING", restart_count=1, reason="BeginTime"))
    assert "requeue:0->1" in changes
    return run_id, aid, eid


def test_a_requeue_grows_the_reservation_to_one_time_limit_per_segment(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    notify_stub(monkeypatch)
    _, aid, eid = _requeue_while_running(env)
    row = reservation(env, aid)
    assert row["state"] == "reserved" and row["reserved"] == 8.0  # 2 segments x 2 GPUs x 2 h
    assert budget.summary(env.ctx.db, experiment_id=eid).available == 2.0


def test_requeue_while_running_settles_exactly_from_per_segment_accounting(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before: segment 0 had no end time, so settlement was 'unknown' and counted at the single
    4 GPU-h reservation although the job used ~7.2 GPU-h."""
    notify_stub(monkeypatch)
    _, aid, eid = _requeue_while_running(env)
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "RUNNING", restart_count=1, start_time="2026-09-01T12:00:00.000Z"))
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "COMPLETED", restart_count=1))
    status(env, aid, state="exited", exit_code=0, restart_count=1)
    final = observation(
        env,
        aid,
        "COMPLETED",
        source="sacct",
        exit_code=0,
        elapsed_s=7000,
        alloc_gpus=2,
        submit_time="2026-09-01T11:40:01.000Z",
        start_time="2026-09-01T12:00:00.000Z",
        end_time="2026-09-01T13:56:40.000Z",
    )
    env.scheduler.acct = [final]
    env.scheduler.segments[env.attempt(aid)["job_id"]] = [
        observation(
            env,
            aid,
            "PREEMPTED",
            source="sacct",
            elapsed_s=6000,
            alloc_gpus=2,
            submit_time="2026-09-01T09:59:00.000Z",
            start_time="2026-09-01T10:00:00.000Z",
            end_time="2026-09-01T11:40:00.000Z",
        ),
        final,
    ]
    out = finalize_attempt(env.ctx, item(aid))
    assert out is not None and out.action == "done", out
    row = reservation(env, aid)
    assert row["state"] == "settled" and row["settled_consumed"] == round(13_000 * 2 / 3600, 6)
    assert segments(env, aid)[0]["end_time"] == "2026-09-01T11:40:00.000Z"
    assert budget.summary(env.ctx.db, experiment_id=eid).available == pytest.approx(10 - 13_000 * 2 / 3600)


def test_requeue_without_per_segment_accounting_stays_an_upper_bound(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    notify_stub(monkeypatch)
    _, aid, eid = _requeue_while_running(env)
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "COMPLETED", restart_count=1))
    status(env, aid, state="exited", exit_code=0, restart_count=1)
    env.scheduler.acct = [observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=7000)]
    env.scheduler.segments_error = TransportTimeout("sacct timed out")
    out = finalize_attempt(env.ctx, item(aid))
    assert out is not None and out.action == "done"
    row = reservation(env, aid)
    assert row["state"] == "unknown" and row["reserved"] == 8.0
    summary = budget.summary(env.ctx.db, experiment_id=eid)
    assert summary.unknown_reserved == 8.0 and summary.available == 2.0


def test_requeue_observed_after_settlement_makes_the_live_job_hold_budget_again(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before: the reopened attempt ran with a 'settled' reservation, so the live job held no
    budget and admission could overcommit."""
    notify_stub(monkeypatch)
    run_id, aid = seed(env, gpus=2, time_limit_s=7200)
    eid = with_budget(env, run_id, aid, budget_gpu_hours=10, reserved=4.0)
    apply(env, aid, observation(env, aid, "RUNNING", start_time="2026-09-01T11:45:00.000Z"))
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "PREEMPTED", end_time="2026-09-01T12:00:00.000Z"))
    env.scheduler.acct = [
        observation(env, aid, "PREEMPTED", source="sacct", elapsed_s=900, alloc_gpus=2, exit_code=0)
    ]
    status(env, aid, state="started", restart_count=0)
    assert finalize_attempt(env.ctx, item(aid)).action == "done"  # type: ignore[union-attr]
    assert reservation(env, aid)["state"] == "settled"

    env.clock.advance(60)
    changes = apply(env, aid, observation(env, aid, "RUNNING", restart_count=1))
    assert "requeue:0->1" in changes
    row = reservation(env, aid)
    assert row["state"] == "reserved" and row["reserved"] == 8.0
    assert budget.summary(env.ctx.db, experiment_id=eid).available == 2.0
    with pytest.raises(BudgetExceeded):
        budget.check_available(env.ctx.db, experiment_id=eid, gpu_hours=9.4)


# ---------------------------------------------------------------------------
# A duplicate owner job runs the attempt: never settle from the duplicate's seconds
# ---------------------------------------------------------------------------


def test_duplicate_owner_keeps_the_reservation_unknown_at_its_full_amount(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    notify_stub(monkeypatch)
    run_id, aid = seed(env, gpus=4, job_id="1002")
    eid = with_budget(env, run_id, aid, budget_gpu_hours=40, reserved=32.0)
    apply(env, aid, observation(env, aid, "FAILED"))
    d = env.attempt(aid)["remote_dir"]
    env.transport.put(f"{d}/logs/stderr.log", f"{DUPLICATE_MARKER}1001\n")
    status(env, aid, state="started", job_id="1001", restart_count=0)
    env.scheduler.acct = [
        observation(
            env, aid, "FAILED", source="sacct", exit_code=EXIT_DUPLICATE_JOB, elapsed_s=2, alloc_gpus=4
        )
    ]
    out = finalize_attempt(env.ctx, item(aid))
    assert out is not None and out.action == "done"
    row = reservation(env, aid)
    assert row["state"] == "unknown" and row["settled_consumed"] is None and "1001" in row["note"]
    assert budget.summary(env.ctx.db, experiment_id=eid).available == 8.0
    assert env.outbox("observe.settle") == []


# ---------------------------------------------------------------------------
# Accounting outage right after the job ended: wait, then settle late - never a permanent guess
# ---------------------------------------------------------------------------


def _squeue_terminal(env: Env, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    notify_stub(monkeypatch)
    run_id, aid = seed(env, gpus=2)
    eid = with_budget(env, run_id, aid, budget_gpu_hours=10, reserved=8.0)
    apply(env, aid, observation(env, aid, "COMPLETED"))
    status(env, aid, state="exited", exit_code=0, restart_count=0)
    return run_id, aid, eid


def test_an_accounting_outage_is_not_settled_as_missing_and_is_settled_late(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, aid, eid = _squeue_terminal(env, monkeypatch)
    env.scheduler.acct_error = TransportTimeout("sacct: Socket timed out")
    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and res.action == "retry"
    env.clock.advance(181)  # past accounting lag + grace: an outage is still not "no record"
    res = finalize_attempt(env.ctx, item(aid, tries=5))
    assert res is not None and res.action == "retry" and "unavailable" in (res.note or "")

    env.clock.advance(observe.ACCOUNTING_OUTAGE_WAIT_S)
    res = finalize_attempt(env.ctx, item(aid, tries=40))
    assert res is not None and res.action == "done"
    row = reservation(env, aid)
    assert row["state"] == "unknown" and "unavailable" in row["note"]
    assert env.attempt(aid)["exit_code"] is None
    pending = env.outbox("observe.settle")
    assert len(pending) == 1 and pending[0]["state"] == "pending"

    # accounting recovers: the follow-up applies the record and settles exactly
    env.scheduler.acct_error = None
    env.scheduler.acct = [
        observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=600, alloc_gpus=2)
    ]
    env.clock.advance(3600)
    res = observe.resettle_attempt(env.ctx, item(aid, kind="observe.settle", restart_count=0))
    assert res is not None and res.action == "done", res
    a = env.attempt(aid)
    assert a["exit_code"] == 0 and a["elapsed_s"] == 600
    row = reservation(env, aid)
    assert row["state"] == "settled" and row["settled_consumed"] == round(600 * 2 / 3600, 6)
    assert budget.summary(env.ctx.db, experiment_id=eid).available == pytest.approx(10 - 1 / 3)
    assert len(env.events(aid, "attempt.accounting_settled")) == 1
    again = observe.resettle_attempt(env.ctx, item(aid, kind="observe.settle", restart_count=0))
    assert (
        again is not None
        and again.action == "done"
        and len(env.events(aid, "attempt.accounting_settled")) == 1
    )


def test_accounting_that_appears_after_the_lag_window_is_settled_late(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, aid, _ = _squeue_terminal(env, monkeypatch)
    env.scheduler.acct = []
    env.clock.advance(181)
    res = finalize_attempt(env.ctx, item(aid, tries=5))
    assert res is not None and res.action == "done"
    assert reservation(env, aid)["state"] == "unknown"
    env.scheduler.acct = [
        observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=900, alloc_gpus=2)
    ]
    env.clock.advance(600)
    res = observe.resettle_attempt(env.ctx, item(aid, kind="observe.settle", restart_count=0))
    assert res is not None and res.action == "done"
    assert reservation(env, aid)["settled_consumed"] == 0.5


def test_the_late_settlement_gives_up_after_its_horizon(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    _, aid, _ = _squeue_terminal(env, monkeypatch)
    env.scheduler.acct_error = TransportTimeout("sacct: Socket timed out")
    env.clock.advance(observe.RESETTLE_HORIZON_S + 1)
    res = observe.resettle_attempt(env.ctx, item(aid, kind="observe.settle", restart_count=0))
    assert res is not None and res.action == "done"
    assert len(env.events(aid, "attempt.accounting_unresolved")) == 1


def test_an_unreachable_login_node_is_waited_out_longer_than_the_tries_cap(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, aid, _ = _squeue_terminal(env, monkeypatch)
    env.scheduler.acct = [
        observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=600, alloc_gpus=2)
    ]
    env.transport.fail["read_range"] = TransportTimeout("ssh: connect timed out")
    env.clock.advance(3 * 3600)
    res = finalize_attempt(env.ctx, item(aid, tries=99))
    assert res is not None and res.action == "retry"
    del env.transport.fail["read_range"]
    res = finalize_attempt(env.ctx, item(aid, tries=100))
    assert res is not None and res.action == "done"
    assert env.attempt(aid)["app_state"] == "exited_ok"


# ---------------------------------------------------------------------------
# Accounting reads are batched and share the cluster's sacct rate limit
# ---------------------------------------------------------------------------


def test_finalize_accounting_is_batched_and_rate_limited(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before: every finalize queried sacct for its one job and re-queried every 30 s during the
    accounting lag (1024 sacct calls for 256 attempts)."""
    notify_stub(monkeypatch)
    install_stub(monkeypatch, "hpclab.services.budget", settle=Recorder())
    aids = [seed(env, job_id=str(5000 + i), cell={"i": i})[1] for i in range(20)]
    env.scheduler.queue = [observation(env, a, "COMPLETED") for a in aids]
    assert len(poll_cluster(env.ctx, cluster_id=CLUSTER).terminal) == 20
    env.scheduler.calls.clear()
    runner = OutboxRunner(env.ctx, lease_s=60)
    runner.register("observe.finalize", finalize_attempt)
    runner.register("observe.settle", observe.resettle_attempt)
    rows = [observation(env, a, "COMPLETED", source="sacct", exit_code=0, elapsed_s=600) for a in aids]
    for t in range(0, 305, 5):
        if t >= 90:  # slurmdbd has the end records 90 s after the jobs ended
            env.scheduler.acct = rows
        while runner.run_once(limit=100):
            pass
        env.clock.advance(5)
    sacct = [c for c in env.scheduler.calls if c.startswith("sacct:")]
    assert len(sacct) <= 4, sacct
    assert all(len(c.split(",")) == 20 for c in sacct)  # one batch for every attempt waiting on it
    assert all(len(env.events(a, "attempt.finalized")) == 1 for a in aids)
    assert all(env.attempt(a)["exit_code"] == 0 for a in aids)


# ---------------------------------------------------------------------------
# run_inspect(refresh=True) with sacct deferred is not a fresh success
# ---------------------------------------------------------------------------


def test_refresh_with_accounting_deferred_is_stale_not_refreshed(env: Env) -> None:
    _, a1 = seed(env, job_id="1001")
    _, a2 = seed(env, job_id="1002")
    env.scheduler.queue = [observation(env, a1, "RUNNING")]
    env.scheduler.acct = [observation(env, a2, "COMPLETED", source="sacct", exit_code=0, elapsed_s=5)]
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).terminal == [a2]  # uses the sacct slot
    env.clock.advance(31)
    env.scheduler.queue = []
    env.scheduler.acct = [observation(env, a1, "FAILED", source="sacct", exit_code=1, elapsed_s=5)]
    res = refresh_attempts(env.ctx, attempt_ids=[a1], deadline_s=5)
    assert res.refreshed == [] and res.stale
    assert any("sacct" in n and a1 in n for n in res.notes), res.notes


# ---------------------------------------------------------------------------
# A backward wall-clock step
# ---------------------------------------------------------------------------


def test_a_backward_clock_step_neither_freezes_polling_nor_drops_observations(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    assert poll_cluster(env.ctx, cluster_id=CLUSTER).ok
    env.clock._now -= timedelta(hours=1)  # NTP step / manual clock fix
    env.clock.advance(60)
    env.scheduler.queue = [observation(env, aid, "COMPLETED")]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.polled and res.ok, res
    a = env.attempt(aid)
    assert a["phase"] == "terminal" and a["scheduler_state"] == "COMPLETED"


def test_a_backoff_far_in_the_future_is_not_honoured_after_a_clock_step(env: Env) -> None:
    _, aid = seed(env)
    with env.ctx.db.tx() as tx:
        tx.execute(
            "INSERT INTO poll_state (cluster_id, source, last_polled_at, consecutive_failures, backoff_until)"
            " VALUES (?, 'squeue', ?, 3, ?)",
            (CLUSTER, "2026-09-01T13:00:00.000Z", "2026-09-01T13:10:00.000Z"),
        )
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert res.polled and res.ok, res
    assert env.poll_state("squeue")["last_polled_at"] == T0
