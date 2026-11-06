"""run_inspect (history, job_ref lookup, refresh, gpu probe, progress/health, diagnosis, env drift),
diagnose_attempt and operation_get."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
from observe_support import CLUSTER, USER, Env, observation, seed

from hpclab.adapters.base import CommandFailed, CommandResult, GpuSample, TransportTimeout
from hpclab.domain.errors import Ambiguous, InvalidArgument, NotFound
from hpclab.services.observe import apply_observation
from hpclab.services.registry import Registry
from hpclab.services.runs import JobRef, diagnose_attempt, operation_get, run_inspect
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_exec_plan, make_experiment, make_plan, make_run


def test_needs_exactly_one_identifier(env: Env) -> None:
    run_id, aid = seed(env)
    with pytest.raises(InvalidArgument):
        run_inspect(env.ctx)
    with pytest.raises(InvalidArgument):
        run_inspect(env.ctx, run_id=run_id, attempt_id=aid)
    with pytest.raises(NotFound):
        run_inspect(env.ctx, attempt_id="att_01NOPE0000000000000000000000")


def test_history_segments_budget_operations_and_pending(env: Env) -> None:
    run_id, first = seed(
        env,
        phase="terminal",
        scheduler_state="NODE_FAIL",
        claim=False,
        diagnosis_json=jdump({"primary_code": "node_fail", "items": [], "summary": "x"}),
    )
    _, second = seed(
        env,
        run_id=run_id,
        phase="observing",
        scheduler_state="PENDING",
        pending_reason="PartitionTimeLimit",
        last_observed_at=env.now(),
    )
    with env.ctx.db.tx() as tx:
        tx.insert(
            "budget_reservations",
            {
                "reservation_id": "bud_1",
                "experiment_id": "exp",
                "attempt_id": second,
                "reserved": 4.0,
                "estimated_consumed": None,
                "settled_consumed": None,
                "state": "reserved",
                "note": None,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        op = tx.create_operation("submit", target_type="attempt", target_id=second, stage="dispatch")
        tx.execute("UPDATE attempts SET operation_id=? WHERE attempt_id=?", (op, second))
        tx.execute(
            "INSERT INTO attempt_segments (attempt_id, segment, job_id, state, observed_at) VALUES (?,0,'1001','NODE_FAIL',?)",
            (first, tx.now),
        )
    data = run_inspect(env.ctx, run_id=run_id)

    assert data.focus_attempt_id == second and data.attempts_total == 2
    assert [a.attempt_id for a in data.attempts] == [second, first]  # newest first
    newest, oldest = data.attempts
    assert newest.budget is not None and newest.budget.reserved == 4.0
    assert newest.job.job_id == "1001" and newest.job.owner == USER
    assert oldest.diagnosis_code == "node_fail" and oldest.segments[0].state == "NODE_FAIL"
    assert [o.operation_id for o in data.operations] == [op] and data.operations[0].ref.endswith(op)
    assert data.pending is not None and data.pending.category == "never_starts"
    assert any("cannot start as submitted" in h and "run_cancel" in h for h in data.recommended_next)
    assert data.refs.plan_ref.startswith("hpclab://plans/")
    assert data.progress is None  # never ran: nothing is read remotely
    assert not [c for c in env.transport.calls if c[0] in ("read_range", "read_tail")]


def test_job_ref_reverse_lookup_and_ambiguity(env: Env) -> None:
    _, a1 = seed(
        env,
        job_id="777",
        job_submit_time="2026-08-01T10:00:00.000Z",
        phase="terminal",
        scheduler_state="COMPLETED",
        claim=False,
    )
    _, a2 = seed(env, job_id="777", job_submit_time="2026-09-01T11:00:00.000Z", claim=False)
    with pytest.raises(Ambiguous) as exc:
        run_inspect(env.ctx, job_ref=JobRef(cluster_id=CLUSTER, job_id="777"))
    assert {c["attempt_id"] for c in exc.value.details["candidates"]} == {a1, a2}
    got = run_inspect(
        env.ctx, job_ref=JobRef(cluster_id=CLUSTER, job_id="777", submit_time="2026-09-01T11:00:01Z")
    )
    assert got.focus_attempt_id == a2
    with pytest.raises(NotFound):
        run_inspect(env.ctx, job_ref=JobRef(cluster_id=CLUSTER, job_id="778"))
    with pytest.raises(NotFound):
        run_inspect(
            env.ctx, job_ref=JobRef(cluster_id=CLUSTER, job_id="777", submit_time="2025-01-01T00:00:00Z")
        )


def test_refresh_polls_then_serves_cached_state_when_too_soon(env: Env) -> None:
    run_id, aid = seed(env, phase="observing", scheduler_state="PENDING")
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=10, time_limit_s=100)]
    fresh = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert fresh.attempts[0].scheduler_state == "RUNNING" and not fresh.stale
    assert any(s.name == "refresh" and s.status == "ok" for s in fresh.coverage.sources)
    env.scheduler.queue = [observation(env, aid, "COMPLETED")]
    cached = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert cached.attempts[0].scheduler_state == "RUNNING" and not cached.stale
    assert any("polled recently" in w for w in cached.warnings)


def test_refresh_timeout_returns_the_last_known_state_flagged_stale(env: Env) -> None:
    env.ctx.settings.request_deadline_s = 1.0
    run_id, aid = seed(env, last_observed_at=env.now())
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    env.scheduler.block = threading.Event()
    data = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert data.stale and not data.coverage.complete
    assert any(f'run_inspect(run_id="{run_id}", refresh=true)' in h for h in data.recommended_next)


def test_gpu_probe_is_identity_checked_and_rate_limited(env: Env) -> None:
    run_id, aid = seed(env, last_observed_at=env.now())
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    env.scheduler.samples = [
        GpuSample(
            index=0, name="A100", util_percent=97.0, mem_used_mib=1000.0, mem_total_mib=81920.0, power_w=300.0
        )
    ]
    first = run_inspect(env.ctx, run_id=run_id, probe="gpu")
    assert first.gpu_probe is not None and first.gpu_probe.status == "ok"
    assert first.gpu_probe.samples[0].util_percent == 97.0
    assert env.scheduler.calls == [
        f"squeue:{env.attempt(aid)['job_id']}",
        f"gpu:{env.attempt(aid)['job_id']}",
    ]
    env.clock.advance(60)
    second = run_inspect(env.ctx, run_id=run_id, probe="gpu")
    assert second.gpu_probe is not None and second.gpu_probe.status == "rate_limited"
    assert second.gpu_probe.samples[0].util_percent == 97.0  # cached samples, no new srun
    assert len(env.scheduler.calls) == 2
    env.clock.advance(600)
    env.scheduler.queue = [observation(env, aid, "RUNNING", user="mallory")]
    refused = run_inspect(env.ctx, run_id=run_id, probe="gpu")
    assert refused.gpu_probe is not None and refused.gpu_probe.status == "refused"
    assert env.scheduler.calls[-1].startswith("squeue:")  # never ran srun on a mismatched job


@pytest.mark.parametrize(
    ("phase", "state", "error", "expected"),
    [
        ("observing", "PENDING", None, "not_running"),
        ("terminal", "COMPLETED", None, "not_running"),
        (
            "observing",
            "RUNNING",
            CommandFailed("srun: error", CommandResult(["srun"], 1, "", "", 0.1)),
            "unavailable",
        ),
    ],
)
def test_gpu_probe_outcomes(env: Env, phase: str, state: str, error: Exception | None, expected: str) -> None:
    run_id, aid = seed(env, phase=phase, scheduler_state=state)
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    env.scheduler.samples_error = error
    data = run_inspect(env.ctx, run_id=run_id, probe="gpu")
    assert data.gpu_probe is not None and data.gpu_probe.status == expected


def test_progress_and_health_are_read_for_the_running_attempt(env: Env) -> None:
    run_id, aid = seed(env, elapsed_s=500, time_limit_s=3600, last_observed_at=env.now(), gpus=2)
    d = env.attempt(aid)["remote_dir"]
    now = env.clock.now().timestamp()
    lines = "".join(
        jdump({"schema": "hpclab.metrics/v1", "step": s, "total_steps": 1000, "time": now - 500 + s}) + "\n"
        for s in range(0, 450, 50)
    )
    env.transport.put(f"{d}/outputs/metrics.jsonl", lines, mtime=now - 5)
    env.transport.put(f"{d}/logs/stdout.log", "ok\n", mtime=now - 5)
    env.transport.put(
        f"{d}/telemetry/gpu.csv", "t, 0, G0, 99, 1000, 81920, 300\nt, 1, G1, 0, 10, 81920, 60\n" * 5
    )
    env.transport.put(f"{d}/telemetry/heartbeat", f"{int(now) - 20}\n")
    data = run_inspect(env.ctx, run_id=run_id)
    assert data.progress is not None and data.progress.step == 400 and data.progress.total_steps == 1000
    assert data.health is not None and data.health.flags == ["partial_gpu_use"]
    assert any(f'run_inspect(attempt_id="{aid}", probe="gpu")' in h for h in data.recommended_next)
    cached = env.attempt(aid)
    assert '"step":400' in cached["progress_json"] and "partial_gpu_use" in cached["health_json"]
    # metrics unreadable: cached views are used and the source is reported unavailable (the
    # files are re-read once the last read is a poll interval old, see test_read_load.py)
    env.transport.fail["read_range"] = CommandFailed("down", CommandResult(["x"], 1, "", "", 0.1))
    env.clock.advance(31)
    again = run_inspect(env.ctx, run_id=run_id)
    assert again.progress is not None and again.progress.step == 400
    assert any(s.name == "metrics" and s.status == "unavailable" for s in again.coverage.sources)


def test_failed_attempt_is_diagnosed_on_demand_and_cached(env: Env) -> None:
    run_id, aid = seed(
        env,
        phase="terminal",
        scheduler_state="FAILED",
        exit_code=1,
        node_list="n1",
        app_state="exited_error",
        elapsed_s=30,
    )
    d = env.attempt(aid)["remote_dir"]
    env.transport.put(
        f"{d}/logs/stderr.log",
        'Traceback (most recent call last):\n  File "t.py", line 1\n'
        "ModuleNotFoundError: No module named 'timm'\n",
    )
    data = run_inspect(env.ctx, run_id=run_id)
    assert data.diagnosis is not None and data.diagnosis.primary_code == "import_error"
    assert '"primary_code":"import_error"' in env.attempt(aid)["diagnosis_json"]
    assert any("run_logs(" in h and "stderr" in h for h in data.recommended_next)
    assert any("fix the code/environment" in h for h in data.recommended_next)
    # stored diagnosis is returned without logs
    assert diagnose_attempt(env.ctx, attempt_id=aid, fetch_logs=False).primary_code == "import_error"


def test_env_drift_between_attempts(env: Env) -> None:
    def env_doc(torch: str, digest: str) -> str:
        return jdump(
            {
                "schema": "hpclab.env/v1",
                "host": "n1",
                "torch": {"version": torch, "device_names": ["NVIDIA H100"]},
                "packages_digest": digest,
                "python": {"version": "3.11.9"},
            }
        )

    run_id, _ = seed(
        env, phase="terminal", scheduler_state="FAILED", claim=False, env_json=env_doc("2.7.0", "sha256:a")
    )
    _, a2 = seed(
        env,
        run_id=run_id,
        phase="terminal",
        scheduler_state="COMPLETED",
        result_state="valid",
        env_json=env_doc("2.8.0", "sha256:b"),
        gpu_type="a100",
    )
    data = run_inspect(env.ctx, run_id=run_id)
    assert data.env is not None and data.env.source == "env.json"
    drift = " | ".join(data.env.drift)
    assert "torch 2.8.0 (attempt #1 had 2.7.0)" in drift and "installed packages differ" in drift
    assert "declared gpu_type 'a100'" in drift
    assert data.focus_attempt_id == a2
    assert any("evaluation_collect" in h for h in data.recommended_next)


def test_uncertain_attempt_hints_never_suggest_resubmitting(env: Env) -> None:
    run_id, _ = seed(env, phase="submission_uncertain", scheduler_state="UNKNOWN", job_id=None)
    data = run_inspect(env.ctx, run_id=run_id)
    assert any("never resubmit" in h for h in data.recommended_next)
    assert not any("run_submit" in h for h in data.recommended_next)


def test_operation_get(env: Env) -> None:
    with env.ctx.db.tx() as tx:
        op = tx.create_operation("workflow_dispatch", target_type="workflow", target_id="wf_1")
        tx.set_operation(
            op, state="blocked", stage="reconcile", error={"code": "auth_required", "message": "mfa"}
        )
    data = operation_get(env.ctx, operation_id=op)
    assert data.state == "blocked" and data.stage == "reconcile" and data.progress == {}
    assert data.target_ref == "hpclab://workflows/wf_1" and data.error is not None
    assert [e.event_type for e in data.recent_events] == ["operation.updated", "operation.created"]
    with pytest.raises(NotFound):
        operation_get(env.ctx, operation_id="op_nope")


def test_read_side_functions_register_with_the_method_registry(env: Env) -> None:
    from hpclab.services.logs import run_logs
    from hpclab.services.runs import runs_list

    reg = Registry()
    methods: list[tuple[str, Callable[..., Any]]] = [
        ("runs_list", runs_list),
        ("run_inspect", run_inspect),
        ("run_logs", run_logs),
        ("operation_get", operation_get),
        ("diagnose_attempt", diagnose_attempt),
    ]
    for name, fn in methods:
        reg.register(name, fn)
    run_id, _ = seed(env, last_observed_at=env.now())
    out = reg.dispatch(env.ctx, "run_inspect", {"run_id": run_id}, request_id="r1")
    assert out["ok"], out
    assert out["coverage"] is not None and "coverage" not in out["data"]
    listed = reg.dispatch(env.ctx, "runs_list", {"states": ["prepared", "running"]}, request_id="r2")
    assert listed["ok"], listed
    bad = reg.dispatch(env.ctx, "run_inspect", {"job_ref": {"cluster_id": CLUSTER}}, request_id="r3")
    assert not bad["ok"] and bad["error"]["code"] == "invalid_argument"


def test_apply_then_inspect_shows_terminal_follow_up_state(env: Env) -> None:
    _, aid = seed(env)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, "COMPLETED"))
    data = run_inspect(env.ctx, attempt_id=aid)
    assert data.run.claim_attempt_id is None and data.run.status == "completed_unverified"
    assert any("artifacts_list" in h or "being indexed" in h for h in data.recommended_next)


# ---------------------------------------------------------------------------
# refresh recomputes the run summary; hints are valid calls
# ---------------------------------------------------------------------------


def test_refresh_recomputes_the_run_status_and_code(env: Env) -> None:
    """The returned run.status/code describe the refreshed attempt, not the pre-refresh row."""
    run_id, aid = seed(env, phase="observing", scheduler_state="PENDING", last_observed_at=env.now())
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE runs SET status='queued' WHERE run_id=?", (run_id,))
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=10, time_limit_s=100)]
    data = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert data.attempts[0].scheduler_state == "RUNNING"
    assert data.run.status == "running" and data.run.code == "R"
    assert "[R] running" in data.summary()

    env.clock.advance(120)
    env.scheduler.queue = [observation(env, aid, "COMPLETED", exit_code=0)]
    done = run_inspect(env.ctx, run_id=run_id, refresh=True)
    assert done.attempts[0].phase == "terminal"
    assert done.run.status == "completed_unverified" and done.run.code == "E"
    assert done.run.claim_attempt_id is None  # the claim release is visible too


_MUTATIONS = {"run_cancel", "run_recover", "run_submit", "evaluation_collect", "experiment_prepare"}
_CALL = re.compile(r"\b([a-z_]+)\((.*?)\)(?=$|[\s;,.:])")
_JSON_VALUE = r'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null|\{.*?\}|\[.*?\]'
_ARGS = re.compile(rf"^\w+=(?:{_JSON_VALUE})(?:, \w+=(?:{_JSON_VALUE}))*$")


def _calls(hint: str) -> list[tuple[str, dict[str, Any]]]:
    """Every ``tool(k=v, ...)`` in a hint, with its arguments parsed as JSON (asserting they are)."""
    out: list[tuple[str, dict[str, Any]]] = []
    for m in _CALL.finditer(hint):
        tool, inner = m.group(1), m.group(2)
        if tool not in {"run_cancel", "run_recover", "run_submit", "evaluation_collect", "experiment_prepare",
                        "run_inspect", "run_logs", "artifacts_list"}:  # fmt: skip
            continue
        assert _ARGS.match(inner), f"not JSON-literal arguments: {hint!r}"
        args = json.loads("{" + re.sub(r"(?:^|(?<=, ))(\w+)=", r'"\1": ', inner) + "}")
        out.append((tool, args))
    return out


def _check_hints(hints: list[str]) -> set[str]:
    tools: set[str] = set()
    for h in hints:
        for tool, args in _calls(h):
            tools.add(tool)
            if tool in _MUTATIONS:
                assert args.get("idempotency_key") == "<new key>", h
            else:
                assert "idempotency_key" not in args, h
            if "attempt_id" in args:
                assert str(args["attempt_id"]).startswith("att"), h
    return tools


def test_recommended_next_hints_are_valid_mcp_calls(env: Env) -> None:
    from hpclab.services.diagnosis import AttemptFacts, diagnose

    seen: set[str] = set()
    # never starts as submitted -> run_cancel with the version and a key placeholder
    run_id, aid = seed(env, phase="observing", scheduler_state="PENDING", pending_reason="PartitionTimeLimit",
                       last_observed_at=env.now())  # fmt: skip
    data = run_inspect(env.ctx, run_id=run_id)
    seen |= _check_hints(data.recommended_next)
    version = env.attempt(aid)["version"]
    assert any(
        f'run_cancel(attempt_id="{aid}", expected_version={version}, idempotency_key="<new key>")' in h
        for h in data.recommended_next
    ), data.recommended_next
    # a recoverable failure -> run_logs + run_recover
    diag = diagnose(AttemptFacts(scheduler_state="NODE_FAIL", node_list="n1"), "", "")
    run_f, aid_f = seed(env, phase="terminal", scheduler_state="NODE_FAIL", claim=False,
                        diagnosis_json=jdump(diag.model_dump(mode="json")))  # fmt: skip
    failed = run_inspect(env.ctx, run_id=run_f)
    seen |= _check_hints(failed.recommended_next)
    assert any(
        f'run_logs(attempt_id="{aid_f}", stream="stderr", tail_lines=200)' in h
        for h in failed.recommended_next
    )
    assert any(
        f'run_recover(attempt_id="{aid_f}", strategy="retry", idempotency_key="<new key>")' in h
        for h in failed.recommended_next
    ), failed.recommended_next
    # a semantic failure -> derive a Run (a complete experiment_prepare call)
    nan = diagnose(
        AttemptFacts(scheduler_state="FAILED", exit_code=1), "", "RuntimeError: Loss is NaN at step 20\n"
    )
    run_n, _ = seed(env, phase="terminal", scheduler_state="FAILED", claim=False,
                    diagnosis_json=jdump(nan.model_dump(mode="json")))  # fmt: skip
    seen |= _check_hints(run_inspect(env.ctx, run_id=run_n).recommended_next)
    # completed with valid outputs but no evaluation -> evaluation_collect with a key
    run_v, aid_v = seed(env, phase="terminal", scheduler_state="COMPLETED", result_state="valid", claim=False)
    valid = run_inspect(env.ctx, run_id=run_v)
    seen |= _check_hints(valid.recommended_next)
    assert any(
        f'evaluation_collect(attempt_id="{aid_v}", idempotency_key="<new key>")' in h
        for h in valid.recommended_next
    )
    # outputs still being checked -> artifacts_list
    run_u, _ = seed(env, phase="terminal", scheduler_state="COMPLETED", result_state="unchecked", claim=False)
    seen |= _check_hints(run_inspect(env.ctx, run_id=run_u).recommended_next)
    # never dispatched -> run_submit with the run's execution plan
    with env.ctx.db.tx() as tx:
        e = make_experiment(tx, cluster_id=CLUSTER)
        r = make_run(tx, e, make_plan(tx, e))
        h = make_exec_plan(tx, r)
    fresh = run_inspect(env.ctx, run_id=r)
    seen |= _check_hints(fresh.recommended_next)
    assert any(
        f'run_submit(run_id="{r}", execution_plan_hash="{h}", idempotency_key="<new key>")' in x
        for x in fresh.recommended_next
    )
    assert {"run_cancel", "run_logs", "run_recover", "evaluation_collect", "artifacts_list", "run_submit",
            "experiment_prepare"} <= seen  # fmt: skip


def test_running_hints_for_health_flags_and_walltime_risk_are_valid_calls(env: Env) -> None:
    run_id, aid = seed(env, elapsed_s=3000, time_limit_s=3600, last_observed_at=env.now(), gpus=2)
    now = env.clock.now().timestamp()
    with env.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET progress_json=?, health_json=? WHERE attempt_id=?",
            (
                jdump({"step": 10, "total_steps": 1000, "fraction": 0.01, "timelimit_risk": "high"}),
                jdump({"source": "telemetry", "flags": ["idle_allocation"], "as_of": now}),
                aid,
            ),
        )
    env.transport.fail["read_tail"] = TransportTimeout("the login node did not answer")
    data = run_inspect(env.ctx, run_id=run_id)  # unreachable: the cached views are shown
    tools = _check_hints(data.recommended_next)
    assert {"run_inspect", "run_logs", "run_cancel"} <= tools, data.recommended_next
    version = env.attempt(aid)["version"]
    assert any(
        f'run_cancel(attempt_id="{aid}", expected_version={version}, mode="request_checkpoint", '
        'idempotency_key="<new key>")' in h
        for h in data.recommended_next
    ), data.recommended_next


def test_a_bad_evaluation_is_not_answered_with_collect_again(env: Env) -> None:
    """C! from an invalid/partial evaluation: evaluation_collect with the same inputs returns the
    same record (idempotent), so the hint points at the evaluation's findings first."""
    from hpclab.testing.factories import make_evaluation

    run_id, aid = seed(env, phase="terminal", scheduler_state="COMPLETED", result_state="valid", claim=False)
    with env.ctx.db.tx() as tx:
        evid = make_evaluation(tx, aid, status="invalid")
    data = run_inspect(env.ctx, run_id=run_id)
    assert data.run.code == "C!"
    _check_hints(data.recommended_next)
    assert not any(h.startswith("collect the metrics") for h in data.recommended_next), data.recommended_next
    hint = next((h for h in data.recommended_next if evid in h), None)
    assert hint is not None and f"hpclab://evaluations/{evid}" in hint and "invalid" in hint, (
        data.recommended_next
    )


def _workflow_owning(
    env: Env, experiment_id: str, plan_id: str, run_id: str, *, state: str, cell: str
) -> str:
    from hpclab.domain.ids import new_id

    wid = new_id("workflow")
    with env.ctx.db.tx() as tx:
        closed = state in ("cancelling", "cancelled")
        tx.insert(
            "workflows",
            {"workflow_id": wid, "experiment_id": experiment_id, "plan_id": plan_id, "plan_hash": "sha256:x",
             "max_parallel": 1, "dispatch_open": 0 if closed else 1,
             "dispatch_state": "closed" if closed else "open", "state": state,
             "created_at": tx.now, "updated_at": tx.now},
        )  # fmt: skip
        tx.insert(
            "workflow_cells",
            {"workflow_id": wid, "run_id": run_id, "ordinal": 0, "cell_state": cell,
             "dispatch_key": f"wf:{wid}:{run_id}", "updated_at": tx.now},
        )  # fmt: skip
    return wid


def test_a_run_its_workflow_will_dispatch_is_not_submitted_by_hand(env: Env) -> None:
    """run_submit on a cell an open workflow still owns bypasses its max_parallel: the hint says
    the workflow dispatches it; once that workflow no longer does, run_submit is the next step."""
    with env.ctx.db.tx() as tx:
        e = make_experiment(tx, cluster_id=CLUSTER)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        h = make_exec_plan(tx, r)
    wid = _workflow_owning(env, e, p, r, state="dispatching", cell="planned")
    hints = run_inspect(env.ctx, run_id=r).recommended_next
    _check_hints(hints)
    assert not any("run_submit(" in x for x in hints), hints
    assert any(wid in x and "will dispatch it" in x for x in hints), hints
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE workflows SET state='cancelled', dispatch_open=0, dispatch_state='closed'")
        tx.execute("UPDATE workflow_cells SET cell_state='cancelled'")
    hints = run_inspect(env.ctx, run_id=r).recommended_next
    assert any(
        f'run_submit(run_id="{r}", execution_plan_hash="{h}", idempotency_key="<new key>")' in x
        for x in hints
    ), hints
    # a run without an execution plan points at the plan resource (there is no plan_get tool)
    with env.ctx.db.tx() as tx:
        bare = make_run(tx, e, p, seed=2)
    hints = run_inspect(env.ctx, run_id=bare).recommended_next
    assert any(f"hpclab://plans/{p}" in x for x in hints) and not any("plan_get" in x for x in hints), hints
