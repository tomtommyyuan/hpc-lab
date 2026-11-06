"""Integration with the real neighbours (no simulator): the real ``submission.admit`` + budget
ledger behind dispatch, and the method registry / facade that serves the tools."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from wf_support import IMPORT_ERR, LabBuilder, one

from hpclab.services import recovery as rec
from hpclab.services import workflows as wfm
from hpclab.services.context import ServiceContext
from hpclab.services.registry import Method
from hpclab.storage.db import jload


@pytest.fixture
def real_submission() -> Any:
    return pytest.importorskip("hpclab.services.submission")


def _staging(ctx: ServiceContext) -> int:
    return int(one(ctx, "SELECT COUNT(*) AS n FROM attempts WHERE phase='staging'")["n"])


def test_dispatch_with_the_real_admission_gate(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    m = lab.matrix(n=4)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 2
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 0
    rows = ctx.db.all("SELECT * FROM attempts ORDER BY created_at")
    assert [r["workflow_id"] for r in rows] == [wf, wf] and _staging(ctx) == 2
    # the real admission reserved budget and queued staging for each
    assert one(ctx, "SELECT COUNT(*) AS n FROM budget_reservations WHERE state='reserved'")["n"] == 2
    assert one(ctx, "SELECT COUNT(*) AS n FROM outbox WHERE kind='submit.stage'")["n"] == 2
    keys = {
        r["idem_key"]
        for r in ctx.db.all("SELECT idem_key FROM idempotency WHERE tool='workflow.dispatch_cell'")
    }
    assert keys == {f"wf:{wf}:{r}" for r in m.run_ids[:2]}


def test_experiment_limit_and_budget_from_the_real_gate(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    capped = lab.matrix(n=3, max_parallel=1)
    wf = wfm.workflow_submit(ctx, idempotency_key="a", plan_id=capped.plan_id, max_parallel=3).workflow_id
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.dispatched == 1 and r.blocked and "max_parallel" in r.blocked
    got = wfm.workflow_get(ctx, workflow_id=wf)
    assert got.operation_stage == "admission_denied" and got.operation_state == "running"
    poor = lab.matrix(n=3, budget_gpu_hours=2.0)
    wf2 = wfm.workflow_submit(ctx, idempotency_key="b", plan_id=poor.plan_id, max_parallel=3).workflow_id
    r2 = wfm.dispatch_workflow(ctx, workflow_id=wf2)
    assert r2.dispatched == 2 and r2.blocked and "budget" in r2.blocked
    assert wfm.workflow_get(ctx, workflow_id=wf2).operation_stage == "budget_exhausted"


def test_real_gate_agrees_with_the_workflow_gate(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    m = lab.matrix(n=2, smoke=True)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=3).workflow_id
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 1  # smoke only
    lab.finish(m.smoke)
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    assert r.gate_state == "passed" and r.dispatched == 2


def test_direct_run_submit_of_a_cell_is_adopted(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    m = lab.matrix(n=2)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    direct = real_submission.run_submit(
        ctx, run_id=m.run_ids[0], execution_plan_hash=m.exec_plans[m.run_ids[0]], idempotency_key="direct"
    )
    r = wfm.dispatch_workflow(ctx, workflow_id=wf)
    cells = {c.run_id: c for c in wfm.workflow_get(ctx, workflow_id=wf).cells}
    assert cells[m.run_ids[0]].attempt_id == direct.attempt_id and r.dispatched == 1
    assert one(ctx, "SELECT COUNT(*) AS n FROM attempts WHERE run_id=?", (m.run_ids[0],))["n"] == 1


def test_cancel_all_managed_through_the_real_request_cancel(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    m = lab.matrix(n=3)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    running = lab.attempt_of(m.run_ids[1])["attempt_id"]
    lab.set_attempt(running, phase="observing", scheduler_state="RUNNING", job_id="4242")
    v = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=v, scope="all_managed", idempotency_key="c"
    )
    # the staging attempt ended before sbatch in the cancel transaction; the running one has a
    # verified cancel step queued (identity re-checked right before scancel)
    assert data.counts == {"cancelled": 2, "active": 1} and data.state == "cancelling"
    assert one(ctx, "SELECT phase FROM attempts WHERE run_id=?", (m.run_ids[0],))["phase"] == (
        "cancelled_before_submit"
    )
    assert one(ctx, "SELECT cancel_state FROM attempts WHERE attempt_id=?", (running,))["cancel_state"] == (
        "requested"
    )
    assert one(ctx, "SELECT COUNT(*) AS n FROM outbox WHERE kind='cancel.execute'")["n"] == 1
    wfm.cancel_step(ctx, workflow_id=wf)  # nothing left to request
    assert one(ctx, "SELECT COUNT(*) AS n FROM outbox WHERE kind='cancel.execute'")["n"] == 1
    lab.finish(m.run_ids[1], state="CANCELLED", result="unchecked")
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).state == "cancelled"


def test_cancel_all_managed_through_the_real_run_cancel(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # fallback: a submission module without request_cancel -> the workflow.cancel handler uses
    # the real run_cancel per attempt
    view = types.ModuleType("hpclab.services.submission")
    view.admit = real_submission.admit  # type: ignore[attr-defined]
    view.run_cancel = real_submission.run_cancel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hpclab.services.submission", view)
    m = lab.matrix(n=3)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=2).workflow_id
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    v = one(ctx, "SELECT version FROM workflows WHERE workflow_id=?", (wf,))["version"]
    data = wfm.workflow_cancel(
        ctx, workflow_id=wf, expected_version=v, scope="all_managed", idempotency_key="c"
    )
    assert data.counts == {"cancelled": 1, "dispatched": 2}
    wfm.cancel_step(ctx, workflow_id=wf)
    # both staging attempts ended before sbatch: nothing left in flight, workflow cancelled
    assert one(ctx, "SELECT COUNT(*) AS n FROM attempts WHERE phase='cancelled_before_submit'")["n"] == 2
    assert one(ctx, "SELECT state FROM workflows WHERE workflow_id=?", (wf,))["state"] == "cancelled"
    released = ctx.db.all("SELECT state FROM budget_reservations")
    assert released and all(r["state"] != "reserved" for r in released)


# ---------------------------------------------------------------------------
# registry / facade
# ---------------------------------------------------------------------------


def test_methods_register_and_validate_structured_params(
    ctx: ServiceContext, lab: LabBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wf_support import install_fake_submission

    install_fake_submission(monkeypatch)
    m = lab.matrix(n=3)
    submit = Method.from_function("workflow_submit", wfm.workflow_submit, mutation=True)
    env = submit.call(
        ctx,
        {
            "idempotency_key": "k",
            "plan_id": m.plan_id,
            "max_parallel": 2,
            "failure_policy": {"circuit_breaker": {"enabled": True, "threshold": 3}},
        },
        request_id="r1",
    )
    assert env["ok"], env
    wf = env["data"]["workflow_id"]
    bad = submit.call(
        ctx, {"idempotency_key": "k2", "plan_id": m.plan_id, "max_parallel": 0}, request_id="r2"
    )
    assert not bad["ok"] and bad["error"]["code"] == "invalid_argument"
    get = Method.from_function("workflow_get", wfm.workflow_get)
    g = get.call(ctx, {"workflow_id": wf, "limit": 2}, request_id="r3")
    assert g["ok"] and g["coverage"]["complete"] and g["next_cursor"] and "coverage" not in g["data"]
    assert g["data"]["failure_policy"]["circuit_breaker"]["threshold"] == 3
    cancel = Method.from_function("workflow_cancel", wfm.workflow_cancel, mutation=True)
    c = cancel.call(
        ctx,
        {"workflow_id": wf, "expected_version": 1, "scope": "everything", "idempotency_key": "c"},
        request_id="r4",
    )
    assert not c["ok"] and c["error"]["code"] == "invalid_argument"
    c = cancel.call(ctx, {"workflow_id": wf, "expected_version": 7, "idempotency_key": "c"}, request_id="r5")
    assert not c["ok"] and c["error"]["code"] == "version_conflict"
    recover = Method.from_function("run_recover", rec.run_recover, mutation=True)
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    lab.fail(m.run_ids[0], stderr=IMPORT_ERR)
    r = recover.call(
        ctx,
        {"idempotency_key": "x", "selector": {"workflow_id": wf, "cells": {"run_ids": [m.run_ids[0]]}}},
        request_id="r6",
    )
    assert r["ok"] and r["data"]["mode"] == "batch" and r["data"]["cells"][0]["eligible"]
    r = recover.call(
        ctx,
        {
            "idempotency_key": "y",
            "attempt_id": lab.attempt_of(m.run_ids[0])["attempt_id"],
            "resource_patch": {"mem": "64G", "bogus": 1},
        },
        request_id="r7",
    )
    assert not r["ok"] and r["error"]["code"] == "invalid_argument"


def test_facade_summaries(ctx: ServiceContext, lab: LabBuilder, monkeypatch: pytest.MonkeyPatch) -> None:
    app = pytest.importorskip("hpclab.services.app")
    from wf_support import install_fake_submission

    install_fake_submission(monkeypatch)
    reg = app.build_registry()
    m = lab.matrix(n=1)
    env = reg.dispatch(ctx, "workflow_submit", {"idempotency_key": "k", "plan_id": m.plan_id}, request_id="r")
    assert env["ok"] and env["summary"].startswith(
        f"workflow {env['data']['workflow_id']} dispatching (dispatch open)"
    )
    wf = env["data"]["workflow_id"]
    wfm.dispatch_workflow(ctx, workflow_id=wf)
    aid = lab.fail(m.run_ids[0], state="NODE_FAIL", exit_code=None)
    out = reg.dispatch(ctx, "run_recover", {"idempotency_key": "r", "attempt_id": aid}, request_id="r2")
    assert out["ok"] and out["summary"].startswith("eligible: retry plan ready")
    got = reg.dispatch(ctx, "workflow_get", {"workflow_id": wf}, request_id="r3")
    assert got["ok"] and jload(__import__("json").dumps(got["data"]))["workflow_id"] == wf
    handlers = set(app.App(ctx, reg).outbox_handlers())
    assert {"workflow.dispatch", "workflow.cancel"} <= handlers
    names = {t.name for t in app.App(ctx, reg).periodic_tasks()}
    assert "workflow.sweep" in names


def test_concurrent_rounds_with_the_real_gate_respect_both_limits(
    ctx: ServiceContext, lab: LabBuilder, real_submission: Any
) -> None:
    """Rounds from sweep / finalize / resume race in several controller threads: the experiment
    limit (2) is below the workflow limit (4) and neither may ever be exceeded; no Run gets two
    attempts. Then cells finish one by one and refills stay within the limits."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    m = lab.matrix(n=8, max_parallel=2)
    wf = wfm.workflow_submit(ctx, idempotency_key="k", plan_id=m.plan_id, max_parallel=4).workflow_id
    barrier = threading.Barrier(6)

    def worker() -> None:
        barrier.wait()
        for _ in range(3):
            wfm.dispatch_workflow(ctx, workflow_id=wf)

    def holding() -> int:
        return int(
            one(
                ctx,
                "SELECT COUNT(*) AS n FROM attempts WHERE phase IN ('staging','submission_in_flight',"
                "'submission_uncertain','duplicate_detected','job_identified','observing')",
            )["n"]
        )

    for _round in range(4):
        with ThreadPoolExecutor(max_workers=6) as pool:
            for f in [pool.submit(worker) for _ in range(6)]:
                f.result(timeout=30)
        assert holding() <= 2
        per_run = ctx.db.all("SELECT run_id, COUNT(*) AS n FROM attempts GROUP BY run_id")
        assert all(r["n"] == 1 for r in per_run)
        for r in ctx.db.all("SELECT run_id FROM attempts WHERE phase='staging'")[:1]:
            lab.finish(r["run_id"])
    assert one(ctx, "SELECT COUNT(*) AS n FROM attempts")["n"] == 5  # 2 + one refill per round
