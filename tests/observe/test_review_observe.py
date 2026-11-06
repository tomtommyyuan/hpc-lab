"""Adversarial review of the observation pipeline: each test pins down a defect that was found in
review (the invariant it protects is named in the docstring)."""

from __future__ import annotations

from typing import Any

import pytest
from observe_support import (
    CLUSTER,
    Env,
    observation,
    put_json,
    seed,
    stub_budget_and_notify,
)

from hpclab.adapters.base import QueryResult
from hpclab.domain.states import RunStatus
from hpclab.services.jobscript import DUPLICATE_MARKER, EXIT_DUPLICATE_JOB
from hpclab.services.observe import (
    app_state_from_status,
    apply_observation,
    finalize_attempt,
    poll_cluster,
)
from hpclab.services.outbox import OutboxItem


def apply(env: Env, attempt_id: str, obs: Any) -> list[str]:
    with env.ctx.db.tx(source="observer") as tx:
        return apply_observation(tx, env.ctx, attempt_id=attempt_id, obs=obs)


def item(aid: str, tries: int = 1) -> OutboxItem:
    return OutboxItem(
        outbox_id=1,
        kind="observe.finalize",
        operation_id=None,
        target_id=aid,
        payload={"attempt_id": aid},
        tries=tries,
    )


# ---------------------------------------------------------------------------
# §4.6: "not in squeue" is never "still running" either
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("acct_state", ["RUNNING", "PENDING"])
def test_lagging_accounting_for_a_job_gone_from_squeue_is_a_gap_not_fresh_running(
    env: Env, acct_state: str
) -> None:
    """slurmdbd lag: the job left squeue (it ended >= MinJobAge ago) but accounting has not
    received its end record and still says RUNNING. That is *not* a fresh RUNNING observation:
    the attempt must be flagged as an observation gap and keep its last real observation time."""
    run_id, aid = seed(env)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    seen_at = env.attempt(aid)["last_observed_at"]
    env.clock.advance(61)
    env.scheduler.queue = []
    env.scheduler.acct = [observation(env, aid, acct_state, source="sacct")]

    res = poll_cluster(env.ctx, cluster_id=CLUSTER)

    a = env.attempt(aid)
    assert res.gaps == [aid]
    assert a["observation_gap"] == 1 and a["phase"] == "observing"
    assert a["last_observed_at"] == seen_at  # nothing trustworthy was observed now
    assert env.run(run_id)["status"] == RunStatus.UNKNOWN.value
    gap = env.events(aid, "attempt.observation_gap")
    assert len(gap) == 1 and acct_state in gap[0]["payload_json"]


def test_a_gap_is_not_closed_by_lagging_accounting(env: Env) -> None:
    _, aid = seed(env)
    env.scheduler.queue = []
    env.scheduler.acct = []
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert env.attempt(aid)["observation_gap"] == 1
    env.clock.advance(61)
    env.scheduler.acct = [observation(env, aid, "RUNNING", source="sacct")]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    assert env.attempt(aid)["observation_gap"] == 1
    # the final accounting record closes it
    env.clock.advance(61)
    env.scheduler.acct = [observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=9)]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert res.terminal == [aid] and a["observation_gap"] == 0 and a["phase"] == "terminal"


def test_accounting_row_of_another_job_with_the_same_id_still_flags_the_gap(env: Env) -> None:
    """A reused job id in accounting is recorded as an identity mismatch AND the attempt, which is
    gone from squeue, is flagged (not silently left looking RUNNING)."""
    _, aid = seed(env)
    env.scheduler.queue = []
    env.scheduler.acct = [observation(env, aid, "COMPLETED", source="sacct", name="hl.other.zzzzzzzz")]
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert a["observation_gap"] == 1 and a["phase"] == "observing" and res.gaps == [aid]
    assert len(env.events(aid, "attempt.identity_mismatch")) == 1


def test_an_incomplete_queue_listing_never_turns_into_gaps(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """§4.10: a partial squeue result says nothing about the jobs it did not list."""
    _, aid = seed(env)

    def partial(*, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        env.scheduler.calls.append("squeue:partial")
        return QueryResult(
            source="squeue", observed_at=env.now(), jobs=[], complete=False, error="chunk failed"
        )

    monkeypatch.setattr(env.scheduler, "query_queue", partial)
    res = poll_cluster(env.ctx, cluster_id=CLUSTER)
    a = env.attempt(aid)
    assert a["observation_gap"] == 0 and a["scheduler_state"] == "RUNNING"
    assert res.gaps == [] and not res.ok
    assert env.scheduler.calls == ["squeue:partial"]  # no accounting query for unlisted jobs
    assert env.poll_state("squeue")["consecutive_failures"] == 1


# ---------------------------------------------------------------------------
# §4.5: requeue segments / out-of-order segments
# ---------------------------------------------------------------------------


def test_requeued_and_finished_between_two_polls_records_both_segments(env: Env) -> None:
    """PREEMPTED (restart 0) observed, then the requeued segment ran and ended before the next
    poll: the next observation is terminal with restart_count 1. That is a new segment with its
    own finalization - not a 'refinement' of the old terminal record."""
    run_id, aid = seed(env)
    apply(env, aid, observation(env, aid, "RUNNING", start_time="2026-09-01T11:00:00.000Z", node_list="n1"))
    env.clock.advance(10)
    apply(env, aid, observation(env, aid, "PREEMPTED", restart_count=0, end_time="2026-09-01T11:30:00.000Z"))
    env.clock.advance(60)
    changes = apply(
        env,
        aid,
        observation(
            env,
            aid,
            "FAILED",
            restart_count=1,
            start_time="2026-09-01T11:32:00.000Z",
            end_time="2026-09-01T11:33:00.000Z",
            node_list="n2",
        ),
    )

    a = env.attempt(aid)
    assert "requeue:0->1" in changes and "terminal:FAILED" in changes
    assert a["phase"] == "terminal" and a["scheduler_state"] == "FAILED" and a["restart_count"] == 1
    assert a["node_list"] == "n2"
    segs = env.ctx.db.all("SELECT * FROM attempt_segments WHERE attempt_id=? ORDER BY segment", (aid,))
    assert [(s["segment"], s["state"], s["node_list"]) for s in segs] == [
        (0, "PREEMPTED", "n1"),
        (1, "FAILED", "n2"),
    ]
    keys = [r["dedupe_key"] for r in env.outbox("observe.finalize")]
    assert keys == [f"observe.finalize:{aid}:0", f"observe.finalize:{aid}:1"]
    assert env.run(run_id)["claim_attempt_id"] is None
    assert env.run(run_id)["status"] == RunStatus.FAILED.value


def test_an_observation_of_an_older_segment_is_out_of_order(env: Env) -> None:
    """An observation carrying a lower restart count describes an earlier segment; applying it
    would end the current (requeued, running) segment."""
    run_id, aid = seed(env, restart_count=1)
    env.clock.advance(5)
    changes = apply(env, aid, observation(env, aid, "PREEMPTED", restart_count=0))
    assert changes == ["ignored:older_segment"]
    a = env.attempt(aid)
    assert a["phase"] == "observing" and a["scheduler_state"] == "RUNNING"
    assert env.run(run_id)["claim_attempt_id"] == aid and env.outbox("observe.finalize") == []


# ---------------------------------------------------------------------------
# finalize: facts are attributed to the right Slurm job
# ---------------------------------------------------------------------------


def test_status_json_written_by_another_slurm_job_is_not_this_jobs_outcome() -> None:
    attempt = {"attempt_id": "att_x", "job_id": "1001", "restart_count": 0}
    mine = {"attempt_id": "att_x", "job_id": "1001", "state": "exited", "exit_code": 0}
    other = {**mine, "job_id": "2002"}
    assert app_state_from_status(mine, None, attempt)[0].value == "exited_ok"
    state, note = app_state_from_status(other, None, attempt)
    assert state.value == "unknown" and note is not None and "2002" in note
    # the wrapper's fallback writer may leave job_id empty: unknown fields never mismatch
    assert app_state_from_status({**mine, "job_id": ""}, None, attempt)[0].value == "exited_ok"


def test_bound_job_that_exited_as_a_duplicate_is_reported_not_diagnosed_as_the_runs_failure(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jobscript contract: a job that is not the attempt's owner prints the duplicate marker and
    exits 95 without touching the attempt. Its FAILED/95 is not the attempt's failure: finalize
    records the owner job so a human can decide, and keeps the owner's status.json out of it."""
    _, emit = stub_budget_and_notify(monkeypatch)
    _, aid = seed(env)
    apply(env, aid, observation(env, aid, "FAILED"))
    d = env.attempt(aid)["remote_dir"]
    put_json(
        env,
        f"{d}/status.json",
        {"attempt_id": aid, "job_id": "3003", "state": "started", "restart_count": 0},
    )
    env.transport.put(f"{d}/logs/stderr.log", f"{DUPLICATE_MARKER}3003\n[hpclab] duplicate submission\n")
    env.scheduler.acct = [
        observation(
            env, aid, "FAILED", source="sacct", exit_code=EXIT_DUPLICATE_JOB, elapsed_s=2, alloc_gpus=1
        )
    ]
    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and res.action == "done"
    a = env.attempt(aid)
    assert a["app_state"] == "unknown"  # the owner job's status.json is not this job's outcome
    assert a["diagnosis_json"] is None  # exit 95 is not a failure cause of the Run
    dup = env.events(aid, "attempt.duplicate_job")
    assert len(dup) == 1 and '"owner_job_id":"3003"' in dup[0]["payload_json"]
    fin = env.events(aid, "attempt.finalized")[0]["payload_json"]
    assert "duplicate" in fin and "3003" in fin
    # a human must look now: another job may be running this attempt unobserved
    assert [(c["event_type"], c["urgency"]) for c in emit.calls] == [("attempt.uncertain", "immediate")]
