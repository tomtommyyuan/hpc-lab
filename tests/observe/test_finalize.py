"""observe.finalize: app state, accounting, budget settlement, diagnosis, node observations,
follow-ups and notifications - with the other wave-3 modules stubbed by name."""

from __future__ import annotations

from typing import Any

import pytest
from observe_support import (
    CLUSTER,
    Env,
    Recorder,
    install_stub,
    observation,
    put_json,
    seed,
    stub_budget_and_notify,
)

from hpclab.adapters.base import CommandFailed, CommandResult
from hpclab.domain.errors import NotFound
from hpclab.domain.states import SchedulerState
from hpclab.services.observe import (
    app_state_from_status,
    apply_observation,
    expand_hostlist,
    finalize_attempt,
)
from hpclab.services.outbox import OutboxItem


def terminal(env: Env, state: str = "COMPLETED", **seed_kw: Any) -> tuple[str, str]:
    run_id, aid = seed(env, **seed_kw)
    with env.ctx.db.tx() as tx:
        apply_observation(
            tx,
            env.ctx,
            attempt_id=aid,
            obs=observation(env, aid, state, node_list=seed_kw.get("node_list", "gpu-07")),
        )
    return run_id, aid


def item(aid: str, tries: int = 1) -> OutboxItem:
    return OutboxItem(
        outbox_id=1,
        kind="observe.finalize",
        operation_id=None,
        target_id=aid,
        payload={"attempt_id": aid},
        tries=tries,
    )


def status(env: Env, aid: str, **fields: Any) -> None:
    d = env.attempt(aid)["remote_dir"]
    put_json(
        env,
        f"{d}/status.json",
        {"schema": "hpclab.status/v1", "attempt_id": aid, "restart_count": 0, **fields},
    )


def acct(env: Env, aid: str, state: str = "COMPLETED", **kw: Any) -> None:
    env.scheduler.acct = [observation(env, aid, state, source="sacct", **kw)]


def test_completed_attempt_is_finalized_once_with_followups(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle, emit = stub_budget_and_notify(monkeypatch)
    run_id, aid = terminal(env, workflow_id="wf_01TESTWORKFLOW0000000000000")
    d = env.attempt(aid)["remote_dir"]
    status(env, aid, state="exited", exit_code=0)
    put_json(
        env, f"{d}/env.json", {"schema": "hpclab.env/v1", "host": "gpu-07", "torch": {"version": "2.8.0"}}
    )
    put_json(env, f"{d}/outputs.json", {"schema": "hpclab.outputs/v1", "files": []})
    acct(env, aid, exit_code=0, elapsed_s=3600, alloc_gpus=2)

    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and res.action == "done", res
    a = env.attempt(aid)
    assert a["app_state"] == "exited_ok" and a["exit_code"] == 0 and a["elapsed_s"] == 3600
    assert '"torch":{"version":"2.8.0"}' in a["env_json"] and '"exited"' in a["status_json"]
    assert a["diagnosis_json"] is None  # success: nothing to diagnose (results judged by artifacts)
    assert settle.calls == [{"attempt_id": aid, "consumed_gpu_hours": 2.0, "note": settle.calls[0]["note"]}]
    assert "allocated GPUs from accounting" in settle.calls[0]["note"]
    kinds = [
        r["kind"] for r in env.ctx.db.all("SELECT kind FROM outbox WHERE state='pending' ORDER BY outbox_id")
    ]
    assert kinds == ["observe.finalize", "artifacts.index", "vault.capture", "workflow.dispatch"]
    assert [c["event_type"] for c in emit.calls] == ["run.succeeded"]
    assert "/" not in emit.calls[0]["body"].replace(f"job {a['job_id']} on {CLUSTER}", "")  # no paths
    ev = env.events(aid, "attempt.finalized")
    assert len(ev) == 1 and '"outputs_manifest":"present"' in ev[0]["payload_json"]

    again = finalize_attempt(env.ctx, item(aid))
    assert again is not None and again.note == "already finalized"
    assert len(settle.calls) == 1 and len(env.events(aid, "attempt.finalized")) == 1
    assert env.run(run_id)["status"] == "completed_unverified"


def test_accounting_lag_is_waited_out_then_settles_unknown_never_zero(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env)
    status(env, aid, state="exited", exit_code=0)
    env.scheduler.acct = []  # no accounting row yet
    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and res.action == "retry" and "waiting for accounting" in (res.note or "")
    assert settle.calls == [] and env.attempt(aid)["app_state"] == "unknown"
    env.clock.advance(120 + 61)  # policy.accounting_lag_s + grace
    res = finalize_attempt(env.ctx, item(aid, tries=5))
    assert res is not None and res.action == "done"
    assert settle.calls[0]["consumed_gpu_hours"] is None
    assert settle.calls[0]["note"].startswith("unknown")
    payload = env.events(aid, "attempt.finalized")[0]["payload_json"]
    assert '"accounting":"missing"' in payload and '"consumed_gpu_hours":null' in payload


def test_gpu_count_falls_back_to_the_request_when_accounting_has_no_gpu_tres(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, gpus=4)
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=1800, alloc_gpus=None)
    finalize_attempt(env.ctx, item(aid))
    assert settle.calls[0]["consumed_gpu_hours"] == 2.0
    assert "from the request" in settle.calls[0]["note"]


def test_gpu_count_unknown_everywhere_settles_unknown_never_zero(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No GPU TRES in accounting, no recorded request, no execution plan: the job's GPU-hours
    are unknown, never 0 (a GPU job settled as free would leak budget)."""
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, gpus=0, plan={"run_id": "r", "nonce": "no resources recorded"})
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=1800, alloc_gpus=None)
    finalize_attempt(env.ctx, item(aid))
    assert settle.calls[0]["consumed_gpu_hours"] is None
    assert settle.calls[0]["note"].startswith("unknown") and "GPU" in settle.calls[0]["note"]


def test_gpu_count_uses_the_planned_gpus_when_accounting_has_none(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    plan = {"run_id": "r", "resources": {"gpus": 2, "nodes": 2, "time": "01:00:00"}}
    _, aid = terminal(env, gpus=0, plan=plan)
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=900, alloc_gpus=None)
    finalize_attempt(env.ctx, item(aid))
    assert settle.calls[0]["consumed_gpu_hours"] == 1.0  # 900 s x 4 GPUs
    assert "planned GPUs from the execution plan" in settle.calls[0]["note"]


def test_gpu_count_unreadable_plan_is_unknown(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, gpus=0, plan={"run_id": "r", "resources": {"gpus": "many"}})
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=900, alloc_gpus=None)
    finalize_attempt(env.ctx, item(aid))
    assert settle.calls[0]["consumed_gpu_hours"] is None
    assert settle.calls[0]["note"].startswith("unknown")


def test_cancelled_while_pending_consumed_nothing(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    settle, emit = stub_budget_and_notify(monkeypatch)
    _, aid = seed(env, phase="observing", scheduler_state="PENDING")
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "CANCELLED"))
    acct(env, aid, state="CANCELLED", elapsed_s=0, alloc_gpus=None, node_list=None)
    finalize_attempt(env.ctx, item(aid))
    a = env.attempt(aid)
    assert a["app_state"] == "not_started"  # status.json was never written
    assert settle.calls[0]["consumed_gpu_hours"] == 0.0
    assert [c["event_type"] for c in emit.calls] == ["run.cancelled"]
    assert '"primary_code":"cancelled"' in a["diagnosis_json"]


def test_failure_is_diagnosed_from_bounded_log_tails(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    _, emit = stub_budget_and_notify(monkeypatch)
    run_id, aid = terminal(env, state="FAILED")
    d = env.attempt(aid)["remote_dir"]
    status(env, aid, state="exited", exit_code=1)
    env.transport.put(f"{d}/logs/stdout.log", "x" * 200_000 + "\nepoch 1 step 40\n")
    env.transport.put(
        f"{d}/logs/stderr.log",
        'Traceback (most recent call last):\n  File "train.py", line 3, in <module>\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
    )
    acct(env, aid, state="FAILED", exit_code=1, elapsed_s=100, alloc_gpus=1)
    finalize_attempt(env.ctx, item(aid))
    a = env.attempt(aid)
    assert a["app_state"] == "exited_error"
    assert '"primary_code":"cuda_oom"' in a["diagnosis_json"]
    reads = [c for c in env.transport.calls if c[0] in ("read_range", "read_tail") and "stdout" in c[1]]
    assert reads  # tail only: the 200 KB log is never read whole
    assert [c["event_type"] for c in emit.calls] == ["run.failed"]
    assert "cuda_oom" in emit.calls[0]["body"]
    assert env.run(run_id)["status"] == "failed"


def test_guard_failure_status_becomes_app_state_and_diagnosis(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, state="FAILED")
    d = env.attempt(aid)["remote_dir"]
    status(env, aid, state="exited", exit_code=3, guard_failed="torch_cuda_available", failed_phase="guard")
    env.transport.put(
        f"{d}/logs/stderr.log", "HPCLAB_GUARD_FAILED:torch_cuda_available\n[hpclab] guard check failed\n"
    )
    acct(env, aid, state="FAILED", exit_code=3, elapsed_s=12)
    finalize_attempt(env.ctx, item(aid))
    a = env.attempt(aid)
    assert a["app_state"] == "guard_failed"
    assert '"primary_code":"env_guard_failed"' in a["diagnosis_json"]


def test_node_failure_records_a_ttl_node_observation(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, state="NODE_FAIL", node_list="gpu-07")
    status(env, aid, state="started")
    acct(env, aid, state="NODE_FAIL", elapsed_s=500, node_list="gpu-07", alloc_gpus=1)
    finalize_attempt(env.ctx, item(aid))
    rows = env.ctx.db.all(
        "SELECT * FROM observations WHERE cluster_id=? AND obs_key LIKE 'node:%'", (CLUSTER,)
    )
    keys = [r["obs_key"] for r in rows]
    assert f"node:gpu-07:node_fail:{aid}" in keys
    row = next(r for r in rows if r["obs_key"] == f"node:gpu-07:node_fail:{aid}")
    assert row["expires_at"] == "2026-09-04T12:00:00.000Z"  # observed + 72 h
    assert env.attempt(aid)["app_state"] == "killed"


def test_requeued_before_finalize_is_a_no_op(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env, state="PREEMPTED")
    env.clock.advance(5)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "PENDING", restart_count=1))
    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and "requeued" in (res.note or "")
    assert settle.calls == []


def test_transport_trouble_is_retried_not_guessed(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env)
    acct(env, aid, exit_code=0, elapsed_s=10)
    env.transport.fail["read_range"] = CommandFailed(
        "python helper died", CommandResult(["x"], 1, "", "", 0.1)
    )
    res = finalize_attempt(env.ctx, item(aid))
    assert res is not None and res.action == "retry"
    assert env.attempt(aid)["app_state"] == "unknown" and not env.events(aid, "attempt.finalized")
    # after many tries the pipeline proceeds, recording the app state as unknown (never guessed)
    res = finalize_attempt(env.ctx, item(aid, tries=99))
    assert res is not None and res.action == "done" and env.attempt(aid)["app_state"] == "unknown"


def test_auth_required_propagates_for_the_outbox_to_wait(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    from observe_support import auth_error

    stub_budget_and_notify(monkeypatch)
    _, aid = terminal(env)
    env.scheduler.acct_error = auth_error()
    with pytest.raises(Exception, match="MFA") as exc:
        finalize_attempt(env.ctx, item(aid))
    assert getattr(exc.value, "code", None) == "auth_required"


def test_missing_budget_or_reservation_does_not_block_finalization(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_reservation(*args: Any, **kwargs: Any) -> None:
        raise NotFound("no reservation")

    install_stub(monkeypatch, "hpclab.services.budget", settle=no_reservation)
    install_stub(monkeypatch, "hpclab.services.notify", emit=Recorder())
    _, aid = terminal(env)
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=10, alloc_gpus=1)
    assert finalize_attempt(env.ctx, item(aid)) is not None
    assert '"budget":"no reservation"' in env.events(aid, "attempt.finalized")[0]["payload_json"]


def test_requeue_segments_are_summed_for_settlement(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    settle, _ = stub_budget_and_notify(monkeypatch)
    _, aid = seed(env, gpus=2)
    with env.ctx.db.tx() as tx:
        o = observation(env, aid, "RUNNING", start_time="2026-09-01T11:00:00.000Z")
        apply_observation(tx, env.ctx, attempt_id=aid, obs=o)
    env.clock.advance(10)
    with env.ctx.db.tx() as tx:
        o = observation(env, aid, "PREEMPTED", end_time="2026-09-01T11:30:00.000Z", restart_count=0)
        apply_observation(tx, env.ctx, attempt_id=aid, obs=o)
    env.clock.advance(10)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "RUNNING", restart_count=1))
    env.clock.advance(10)
    with env.ctx.db.tx() as tx:
        apply_observation(
            tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "COMPLETED", restart_count=1)
        )
    status(env, aid, state="exited", exit_code=0, restart_count=1)
    acct(env, aid, exit_code=0, elapsed_s=1800, alloc_gpus=2)
    finalize_attempt(env.ctx, item(aid))
    assert settle.calls[0]["consumed_gpu_hours"] == 2.0  # (30 min + 30 min) x 2 GPUs


@pytest.mark.parametrize(
    ("doc", "problem", "expected"),
    [
        (None, "missing", "not_started"),
        (None, "unavailable: x", "unknown"),
        ({"state": "started"}, None, "killed"),
        ({"state": "killed", "signal": "TERM"}, None, "killed"),
        ({"state": "exited", "exit_code": 0}, None, "exited_ok"),
        ({"state": "exited", "exit_code": 2}, None, "exited_error"),
        ({"state": "exited", "exit_code": 3, "guard_failed": "venv"}, None, "guard_failed"),
        ({"state": "exited", "exit_code": 0, "attempt_id": "att_other"}, None, "unknown"),
        ({"state": "exited", "exit_code": 0, "restart_count": 0}, None, "unknown"),  # restart 1 attempt
    ],
)
def test_app_state_from_status(doc: dict[str, Any] | None, problem: str | None, expected: str) -> None:
    attempt = {"attempt_id": "att_x", "restart_count": 1 if doc and "restart_count" in doc else 0}
    if doc is not None and "attempt_id" not in doc:
        doc = {**doc, "attempt_id": "att_x"}
    state, _ = app_state_from_status(doc, problem, attempt)
    assert state.value == expected


def test_expand_hostlist() -> None:
    assert expand_hostlist("gpu-[01-03,07],cpu1") == ["gpu-01", "gpu-02", "gpu-03", "gpu-07", "cpu1"]
    assert expand_hostlist("n1") == ["n1"]
    assert expand_hostlist(None) == []
    assert len(expand_hostlist("x[1-1000]")) == 16  # bounded


def test_state_enum_sanity() -> None:
    assert SchedulerState("NODE_FAIL") in (SchedulerState.NODE_FAIL,)


def test_settlement_with_the_real_budget_module(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration with services/budget.py (skipped until that area is deployed)."""
    budget = pytest.importorskip("hpclab.services.budget")
    install_stub(monkeypatch, "hpclab.services.notify", emit=Recorder())
    run_id, aid = seed(env, gpus=2)
    exp = env.run(run_id)["experiment_id"]
    with env.ctx.db.tx() as tx:
        budget.reserve(tx, experiment_id=exp, attempt_id=aid, gpu_hours=8.0)
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "COMPLETED"))
    status(env, aid, state="exited", exit_code=0)
    acct(env, aid, exit_code=0, elapsed_s=5400, alloc_gpus=2)
    finalize_attempt(env.ctx, item(aid))
    row = env.ctx.db.one("SELECT * FROM budget_reservations WHERE attempt_id=?", (aid,))
    assert row is not None and row["state"] == "settled" and row["settled_consumed"] == 3.0
    assert '"budget":"settled"' in env.events(aid, "attempt.finalized")[0]["payload_json"]

    # no accounting -> unknown, and the reservation is kept (never released as 0)
    run2, aid2 = seed(env, gpus=1)
    with env.ctx.db.tx() as tx:
        budget.reserve(tx, experiment_id=env.run(run2)["experiment_id"], attempt_id=aid2, gpu_hours=1.0)
        apply_observation(tx, env.ctx, attempt_id=aid2, obs=observation(env, aid2, "FAILED"))
    env.scheduler.acct = []
    env.clock.advance(3600)
    finalize_attempt(env.ctx, item(aid2, tries=3))
    row2 = env.ctx.db.one("SELECT * FROM budget_reservations WHERE attempt_id=?", (aid2,))
    assert row2 is not None and row2["state"] == "unknown" and row2["settled_consumed"] is None
