"""attempts.version bumps only on changes a user decision depends on (phase, scheduler state,
app/result state, cancel, job identity, requeue) - never on observation-only refreshes - so an
``expected_version`` read by a human or agent stays valid while the poller keeps working."""

from __future__ import annotations

from typing import Any

import pytest
from observe_support import Env, observation, seed

from hpclab.domain.errors import VersionConflict
from hpclab.services.health import Health
from hpclab.services.observe import LiveViews, apply_observation, mark_observation_gap, store_live_views
from hpclab.services.progress import Progress
from hpclab.services.submission import run_cancel


def apply(env: Env, attempt_id: str, obs: Any) -> list[str]:
    with env.ctx.db.tx(source="observer") as tx:
        return apply_observation(tx, env.ctx, attempt_id=attempt_id, obs=obs)


def _observation_refreshes(env: Env, aid: str) -> None:
    """Everything the poller does to a RUNNING attempt between two user decisions."""
    for i in range(3):
        env.clock.advance(30)
        apply(
            env,
            aid,
            observation(
                env, aid, "RUNNING", elapsed_s=60 * (i + 1), time_limit_s=3600, node_list=f"gpu-0{i}"
            ),
        )
    # the job vanished from squeue, accounting lagged: a gap is flagged ...
    env.clock.advance(30)
    with env.ctx.db.tx(source="observer") as tx:
        assert mark_observation_gap(tx, attempt_id=aid, reason="not in squeue") == ["gap_opened"]
    # ... and closed by the next observation
    env.clock.advance(30)
    assert "gap_closed" in apply(env, aid, observation(env, aid, "RUNNING", elapsed_s=300, time_limit_s=3600))
    # cached derived views (progress / health) are refreshed as well
    views = LiveViews(
        progress=Progress(step=10, total_steps=100, fraction=0.1),
        health=Health(source="telemetry", as_of=env.clock.now().timestamp()),
    )
    store_live_views(env.ctx, aid, views)


def test_run_cancel_with_a_version_read_before_observation_refreshes_succeeds(env: Env) -> None:
    _, aid = seed(env, phase="observing", scheduler_state="RUNNING", job_submit_time=env.now())
    seen = env.attempt(aid)["version"]
    _observation_refreshes(env, aid)
    a = env.attempt(aid)
    assert a["version"] == seen, "observation-only refreshes must not invalidate expected_version"
    assert a["observation_gap"] == 0 and a["elapsed_s"] == 300 and a["progress_json"]
    # the gap was still recorded (events are the audit trail, versions are for decisions)
    assert env.events(aid, "attempt.observation_gap") and env.events(aid, "attempt.gap_closed")
    receipt = run_cancel(env.ctx, attempt_id=aid, expected_version=seen, idempotency_key="cancel-1")
    assert receipt.cancel_state == "requested"
    assert env.attempt(aid)["version"] == seen + 1  # the cancel itself is a decision: versioned


def test_a_real_state_change_still_conflicts(env: Env) -> None:
    _, aid = seed(env, phase="observing", scheduler_state="PENDING", job_submit_time=env.now())
    seen = env.attempt(aid)["version"]
    env.clock.advance(30)
    apply(env, aid, observation(env, aid, "RUNNING", elapsed_s=5, time_limit_s=3600))
    assert env.attempt(aid)["version"] == seen + 1
    with pytest.raises(VersionConflict) as exc:
        run_cancel(env.ctx, attempt_id=aid, expected_version=seen, idempotency_key="cancel-1")
    assert exc.value.details["current_version"] == seen + 1


def test_a_newly_known_job_identity_is_versioned(env: Env) -> None:
    """Owner / submit time are part of the job identity a cancel is verified against."""
    _, aid = seed(env, phase="observing", scheduler_state="RUNNING", job_owner=None)
    seen = env.attempt(aid)["version"]
    env.clock.advance(30)
    apply(env, aid, observation(env, aid, "RUNNING", submit_time="2026-09-01T10:00:00.000Z"))
    a = env.attempt(aid)
    assert a["job_submit_time"] == "2026-09-01T10:00:00.000Z" and a["job_owner"] is not None
    assert a["version"] == seen + 1
    env.clock.advance(30)
    apply(env, aid, observation(env, aid, "RUNNING", submit_time="2026-09-01T10:00:00.000Z"))
    assert env.attempt(aid)["version"] == seen + 1  # known now: nothing new


def test_gap_open_and_close_do_not_bump_the_version(env: Env) -> None:
    run_id, aid = seed(env, phase="observing", scheduler_state="RUNNING")
    seen = env.attempt(aid)["version"]
    with env.ctx.db.tx(source="observer") as tx:
        mark_observation_gap(tx, attempt_id=aid, reason="not in squeue")
    assert env.attempt(aid)["version"] == seen
    assert env.run(run_id)["status"] == "unknown"  # the run summary still reflects the gap
    ev = env.events(aid, "attempt.observation_gap")
    assert len(ev) == 1 and ev[0]["entity_version"] == seen
    env.clock.advance(30)
    apply(env, aid, observation(env, aid, "RUNNING"))
    assert env.attempt(aid)["version"] == seen and env.run(run_id)["status"] == "running"


def test_finalize_versions_the_app_state_but_not_evidence_copies(env: Env, monkeypatch: Any) -> None:
    from observe_support import put_json, stub_budget_and_notify

    from hpclab.services.observe import finalize_attempt
    from hpclab.services.outbox import OutboxItem

    stub_budget_and_notify(monkeypatch)

    def finalize(aid: str) -> None:
        item = OutboxItem(outbox_id=1, kind="observe.finalize", operation_id=None, target_id=aid,
                          payload={"attempt_id": aid}, tries=1)  # fmt: skip
        res = finalize_attempt(env.ctx, item)
        assert res is not None and res.action == "done", res

    def completed(**kw: Any) -> str:
        _, aid = seed(env, **kw)
        apply(env, aid, observation(env, aid, "COMPLETED", node_list="gpu-07"))
        env.scheduler.acct = [observation(env, aid, "COMPLETED", source="sacct", exit_code=0, elapsed_s=60)]
        return aid

    # the program's outcome becomes known: a decision input -> versioned
    ok = completed()
    d = env.attempt(ok)["remote_dir"]
    put_json(
        env, f"{d}/status.json", {"attempt_id": ok, "restart_count": 0, "state": "exited", "exit_code": 0}
    )
    before = env.attempt(ok)["version"]
    finalize(ok)
    assert env.attempt(ok)["app_state"] == "exited_ok" and env.attempt(ok)["version"] == before + 1
    # only evidence copies (env.json) are stored, the app state stays unknown -> not versioned
    env.clock.advance(60)  # finalize shares the cluster's sacct rate limit
    unknown = completed(job_id="1002")
    d = env.attempt(unknown)["remote_dir"]
    put_json(env, f"{d}/status.json", {"attempt_id": "att_someone_else", "state": "exited", "exit_code": 0})
    put_json(env, f"{d}/env.json", {"schema": "hpclab.env/v1", "host": "gpu-07"})
    before = env.attempt(unknown)["version"]
    finalize(unknown)
    a = env.attempt(unknown)
    assert a["app_state"] == "unknown" and a["env_json"] and a["version"] == before
    ev = env.events(unknown, "attempt.finalized")
    assert len(ev) == 1 and ev[0]["entity_version"] == before
