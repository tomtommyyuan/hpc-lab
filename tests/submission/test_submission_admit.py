"""Admission gate + run_submit idempotency (invariants 4.1, 4.7, 4.8)."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from submission_support import (
    UnitEnv,
    add_plan,
    attempt,
    events,
    operation,
    outbox,
    reservation,
    run_row,
    seed_run,
)

from hpclab.domain.errors import (
    AdmissionDenied,
    BudgetExceeded,
    ErrorCode,
    HpcLabError,
    IdempotencyConflict,
    InvalidArgument,
    NotFound,
    NotReady,
    PreconditionFailed,
)
from hpclab.services import budget, submission
from hpclab.services.submission import admit, run_cancel, run_submit
from hpclab.testing.factories import make_attempt, make_exec_plan, make_experiment, make_plan, make_run


def submit(
    env: UnitEnv, seeded_run: str, plan_hash: str, key: str = "k1", **kw: object
) -> submission.SubmitReceiptData:
    return run_submit(env.ctx, run_id=seeded_run, execution_plan_hash=plan_hash, idempotency_key=key, **kw)  # type: ignore[arg-type]


def test_submit_creates_attempt_claim_consumption_reservation_and_stage_step(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx, gpus=2, time="03:00:00")
    receipt = submit(unit, s.run_id, s.exec_plan_hash)

    assert receipt.phase == "staging"
    assert receipt.reserved_gpu_hours == pytest.approx(6.0)
    assert receipt.operation_ref == f"hpclab://operations/{receipt.operation_id}"
    a = attempt(unit.ctx, receipt.attempt_id)
    assert a["phase"] == "staging" and a["seq"] == 1 and a["gpus"] == 2
    assert a["partition"] == "gpu" and a["gpu_type"] == "a100" and a["time_limit_s"] == 3 * 3600
    assert a["remote_dir"] == f"/remote/hpclab/attempts/{receipt.attempt_id}"
    assert a["job_name"].startswith("hl.cell.seed-1.") and a["job_name"].endswith(a["submit_token"][:8])
    run = run_row(unit.ctx, s.run_id)
    assert run["claim_attempt_id"] == receipt.attempt_id and run["latest_attempt_id"] == receipt.attempt_id
    assert run["status"] == "submitting"
    ep = unit.ctx.db.one("SELECT * FROM execution_plans WHERE exec_plan_hash=?", (s.exec_plan_hash,))
    assert (
        ep is not None and ep["consumed_by_attempt_id"] == receipt.attempt_id and ep["status"] == "consumed"
    )
    res = reservation(unit.ctx, receipt.attempt_id)
    assert res["state"] == "reserved" and res["reserved"] == pytest.approx(6.0)
    op = operation(unit.ctx, receipt.operation_id)
    assert op["kind"] == "submit" and op["state"] == "queued" and op["target_id"] == receipt.attempt_id
    items = outbox(unit.ctx, receipt.attempt_id)
    assert [i["kind"] for i in items] == ["submit.stage"]
    assert events(unit.ctx, receipt.attempt_id) == ["attempt.admitted"]
    assert events(unit.ctx, s.run_id) == ["run.claimed"]
    assert "budget.reserved" in events(unit.ctx, s.experiment_id)


def test_same_key_same_payload_replays_the_stored_receipt(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    first = submit(unit, s.run_id, s.exec_plan_hash, key="same")
    unit.clock.advance(30)
    again = submit(unit, s.run_id, s.exec_plan_hash, key="same")
    assert again == first
    assert unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")["n"] == 1  # type: ignore[index]


def test_same_key_different_payload_is_a_conflict(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    other = seed_run(unit.ctx)
    submit(unit, s.run_id, s.exec_plan_hash, key="k")
    with pytest.raises(IdempotencyConflict):
        submit(unit, other.run_id, other.exec_plan_hash, key="k")


def test_different_key_cannot_open_a_second_execution(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    first = submit(unit, s.run_id, s.exec_plan_hash, key="a")
    second_plan = add_plan(unit.ctx, s, nonce="retry")
    with pytest.raises(AdmissionDenied) as exc:
        submit(unit, s.run_id, second_plan, key="b")
    assert exc.value.details["reason"] == "run_claimed"
    assert exc.value.details["claim_attempt_id"] == first.attempt_id
    assert unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")["n"] == 1  # type: ignore[index]


def test_plan_is_consumed_once_even_after_the_claim_is_released(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    first = submit(unit, s.run_id, s.exec_plan_hash, key="a")
    a = attempt(unit.ctx, first.attempt_id)
    run_cancel(unit.ctx, attempt_id=first.attempt_id, expected_version=a["version"], idempotency_key="c")
    assert run_row(unit.ctx, s.run_id)["claim_attempt_id"] is None

    with pytest.raises(AdmissionDenied) as exc:
        submit(unit, s.run_id, s.exec_plan_hash, key="b")
    assert exc.value.details["reason"] == "plan_consumed"
    assert exc.value.details["consumed_by_attempt_id"] == first.attempt_id

    fresh = add_plan(unit.ctx, s, nonce="retry")
    second = submit(unit, s.run_id, fresh, key="c")
    assert attempt(unit.ctx, second.attempt_id)["seq"] == 2


def test_unknown_foreign_and_unready_plans_are_refused(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    other = seed_run(unit.ctx)
    with pytest.raises(NotFound):
        submit(unit, s.run_id, "sha256:" + "0" * 64)
    with pytest.raises(InvalidArgument):
        submit(unit, s.run_id, other.exec_plan_hash)
    with pytest.raises(NotFound):
        submit(unit, "run_missing", s.exec_plan_hash)
    with unit.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE execution_plans SET status='not_ready' WHERE exec_plan_hash=?", (s.exec_plan_hash,)
        )
    with pytest.raises(NotReady):
        submit(unit, s.run_id, s.exec_plan_hash)
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE execution_plans SET status='invalid' WHERE exec_plan_hash=?", (s.exec_plan_hash,))
    with pytest.raises(PreconditionFailed):
        submit(unit, s.run_id, s.exec_plan_hash)
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE execution_plans SET status='ready' WHERE exec_plan_hash=?", (s.exec_plan_hash,))
        tx.execute("UPDATE plans SET status='preparing' WHERE plan_id=?", (s.plan_id,))
    with pytest.raises(NotReady):
        submit(unit, s.run_id, s.exec_plan_hash)
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE plans SET status='invalid' WHERE plan_id=?", (s.plan_id,))
    with pytest.raises(PreconditionFailed):
        submit(unit, s.run_id, s.exec_plan_hash)
    assert unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")["n"] == 0  # type: ignore[index]


def test_failed_admission_stores_no_idempotency_record(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    with unit.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE execution_plans SET status='not_ready' WHERE exec_plan_hash=?", (s.exec_plan_hash,)
        )
    with pytest.raises(NotReady):
        submit(unit, s.run_id, s.exec_plan_hash, key="k")
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE execution_plans SET status='ready' WHERE exec_plan_hash=?", (s.exec_plan_hash,))
    assert submit(unit, s.run_id, s.exec_plan_hash, key="k").phase == "staging"


def _gate(unit: UnitEnv, **attempt_kw: Any) -> tuple[str, str]:
    """A smoke run (optionally with one attempt) and a main run gated on it."""
    smoke = seed_run(unit.ctx, kind="smoke", label="smoke")
    if attempt_kw:
        with unit.ctx.db.tx() as tx:
            make_attempt(
                tx, smoke.run_id, exec_plan_hash=smoke.exec_plan_hash, cluster_id="unit", **attempt_kw
            )
    main = seed_run(
        unit.ctx, experiment_id=smoke.experiment_id, plan_id=smoke.plan_id, seed=2, gate=smoke.run_id
    )
    return smoke.run_id, main.exec_plan_hash


def test_smoke_gate_not_started_or_running_is_not_ready(unit: UnitEnv) -> None:
    smoke_id, _ = _gate(unit)
    main = unit.ctx.db.one("SELECT run_id FROM runs WHERE run_id != ?", (smoke_id,))
    assert main is not None
    plan = unit.ctx.db.one("SELECT exec_plan_hash FROM execution_plans WHERE run_id=?", (main["run_id"],))
    assert plan is not None
    with pytest.raises(NotReady) as exc:
        submit(unit, main["run_id"], plan["exec_plan_hash"])
    assert exc.value.details["gate_state"] == "not_started"


def test_smoke_gate_states(unit: UnitEnv) -> None:
    cases = {
        "pending": {"phase": "observing", "scheduler_state": "RUNNING"},
        "failed": {"phase": "terminal", "scheduler_state": "FAILED"},
        "pending_validation": {
            "phase": "terminal",
            "scheduler_state": "COMPLETED",
            "result_state": "unchecked",
        },
        "passed": {"phase": "terminal", "scheduler_state": "COMPLETED", "result_state": "valid"},
    }
    for name, kw in cases.items():
        _smoke_id, plan_hash = _gate(unit, **kw)
        run_id = unit.ctx.db.one("SELECT run_id FROM execution_plans WHERE exec_plan_hash=?", (plan_hash,))[
            "run_id"
        ]  # type: ignore[index]
        if name == "passed":
            assert submit(unit, run_id, plan_hash, key=name).phase == "staging"
        elif name == "failed":
            with pytest.raises(PreconditionFailed) as exc:
                submit(unit, run_id, plan_hash, key=name)
            assert exc.value.details["gate_state"] == "failed"
        else:
            with pytest.raises(NotReady):
                submit(unit, run_id, plan_hash, key=name)


def test_forcing_past_the_gate_needs_a_reason_and_is_recorded(unit: UnitEnv) -> None:
    _smoke, plan_hash = _gate(unit, phase="terminal", scheduler_state="FAILED")
    run_id = unit.ctx.db.one("SELECT run_id FROM execution_plans WHERE exec_plan_hash=?", (plan_hash,))[
        "run_id"
    ]  # type: ignore[index]
    with pytest.raises(InvalidArgument):
        submit(unit, run_id, plan_hash, key="f1", force_without_gate=True)
    r = submit(
        unit, run_id, plan_hash, key="f2", force_without_gate=True, reason="smoke failed on a flaky node"
    )
    assert "attempt.gate_forced" in events(unit.ctx, r.attempt_id)


def test_experiment_slots_count_staging_and_uncertain_attempts(unit: UnitEnv) -> None:
    a = seed_run(unit.ctx, max_parallel=1)
    b = seed_run(unit.ctx, experiment_id=a.experiment_id, plan_id=a.plan_id, seed=2)
    first = submit(unit, a.run_id, a.exec_plan_hash, key="a")
    with pytest.raises(AdmissionDenied) as exc:
        submit(unit, b.run_id, b.exec_plan_hash, key="b")
    assert exc.value.details["reason"] == "experiment_slots_full"
    assert exc.value.retryable is True
    # uncertain still holds the slot
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE attempts SET phase='submission_uncertain' WHERE attempt_id=?", (first.attempt_id,))
    with pytest.raises(AdmissionDenied):
        submit(unit, b.run_id, b.exec_plan_hash, key="b")
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE attempts SET phase='terminal' WHERE attempt_id=?", (first.attempt_id,))
    assert submit(unit, b.run_id, b.exec_plan_hash, key="b").phase == "staging"


def test_workflow_slots_and_closed_dispatch(unit: UnitEnv) -> None:
    a = seed_run(unit.ctx)
    b = seed_run(unit.ctx, experiment_id=a.experiment_id, plan_id=a.plan_id, seed=2)
    c = seed_run(unit.ctx, experiment_id=a.experiment_id, plan_id=a.plan_id, seed=3)
    with unit.ctx.db.tx() as tx:
        tx.insert(
            "workflows",
            {
                "workflow_id": "wf_1",
                "experiment_id": a.experiment_id,
                "plan_id": a.plan_id,
                "plan_hash": "sha256:x",
                "max_parallel": 1,
                "state": "dispatching",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        res = admit(tx, unit.ctx, run_id=a.run_id, exec_plan_hash=a.exec_plan_hash, workflow_id="wf_1")
    assert attempt(unit.ctx, res.attempt_id)["workflow_id"] == "wf_1"
    with pytest.raises(AdmissionDenied) as exc, unit.ctx.db.tx() as tx:
        admit(tx, unit.ctx, run_id=b.run_id, exec_plan_hash=b.exec_plan_hash, workflow_id="wf_1")
    assert exc.value.details["reason"] == "workflow_slots_full"
    with unit.ctx.db.tx() as tx:
        tx.execute("UPDATE workflows SET max_parallel=5, dispatch_open=0 WHERE workflow_id='wf_1'")
    with pytest.raises(AdmissionDenied) as exc, unit.ctx.db.tx() as tx:
        admit(tx, unit.ctx, run_id=c.run_id, exec_plan_hash=c.exec_plan_hash, workflow_id="wf_1")
    assert exc.value.details["reason"] == "workflow_dispatch_closed"
    with pytest.raises(NotFound), unit.ctx.db.tx() as tx:
        admit(tx, unit.ctx, run_id=c.run_id, exec_plan_hash=c.exec_plan_hash, workflow_id="wf_nope")


def test_budget_is_checked_and_reserved_at_admission(unit: UnitEnv) -> None:
    a = seed_run(unit.ctx, budget_gpu_hours=3.0, gpus=1, time="02:00:00")
    b = seed_run(unit.ctx, experiment_id=a.experiment_id, plan_id=a.plan_id, seed=2, gpus=1, time="02:00:00")
    first = submit(unit, a.run_id, a.exec_plan_hash, key="a")
    with pytest.raises(BudgetExceeded) as exc:
        submit(unit, b.run_id, b.exec_plan_hash, key="b")
    assert exc.value.code == ErrorCode.BUDGET_EXCEEDED
    assert exc.value.details["summary"]["available"] == pytest.approx(1.0)
    assert unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")["n"] == 1  # type: ignore[index]
    # an abandoned (unknown) attempt keeps counting at its reservation: still no room
    with unit.ctx.db.tx() as tx:
        budget.settle(tx, attempt_id=first.attempt_id, consumed_gpu_hours=None, note="unknown")
        tx.execute("UPDATE attempts SET phase='abandoned' WHERE attempt_id=?", (first.attempt_id,))
        tx.execute("UPDATE runs SET claim_attempt_id=NULL WHERE run_id=?", (a.run_id,))
    with pytest.raises(BudgetExceeded):
        submit(unit, b.run_id, b.exec_plan_hash, key="b")
    # once settled at the real (smaller) consumption, the difference is available again
    with unit.ctx.db.tx() as tx:
        budget.settle(tx, attempt_id=first.attempt_id, consumed_gpu_hours=0.5, note="sacct")
    assert submit(unit, b.run_id, b.exec_plan_hash, key="b").reserved_gpu_hours == pytest.approx(2.0)


def test_concurrent_submits_with_different_keys_create_exactly_one_attempt(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    plans = [s.exec_plan_hash, *(add_plan(unit.ctx, s, nonce=f"n{i}") for i in range(7))]
    start = threading.Barrier(len(plans))
    results: list[object] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        start.wait()
        try:
            r: object = submit(unit, s.run_id, plans[i], key=f"key-{i}")
        except HpcLabError as exc:
            r = exc
        finally:
            unit.ctx.db.close()
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(plans))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    ok = [r for r in results if isinstance(r, submission.SubmitReceiptData)]
    denied = [r for r in results if isinstance(r, AdmissionDenied)]
    assert len(ok) == 1 and len(denied) == len(plans) - 1
    assert all(d.details["reason"] == "run_claimed" for d in denied)
    assert unit.ctx.db.one("SELECT COUNT(*) AS n FROM attempts")["n"] == 1  # type: ignore[index]
    assert run_row(unit.ctx, s.run_id)["claim_attempt_id"] == ok[0].attempt_id


def test_concurrent_submits_with_the_same_key_return_one_receipt(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    start = threading.Barrier(6)
    results: list[object] = []

    def worker() -> None:
        start.wait()
        try:
            results.append(submit(unit, s.run_id, s.exec_plan_hash, key="shared"))
        finally:
            unit.ctx.db.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == 6 and len({r.attempt_id for r in results}) == 1  # type: ignore[attr-defined]


def test_a_stale_claim_is_healed_with_an_event(unit: UnitEnv) -> None:
    s = seed_run(unit.ctx)
    with unit.ctx.db.tx() as tx:
        old = make_attempt(
            tx, s.run_id, exec_plan_hash=make_exec_plan(tx, s.run_id), phase="terminal", cluster_id="unit"
        )
        tx.execute("UPDATE runs SET claim_attempt_id=? WHERE run_id=?", (old, s.run_id))
    r = submit(unit, s.run_id, s.exec_plan_hash)
    assert attempt(unit.ctx, r.attempt_id)["seq"] == 2
    assert events(unit.ctx, s.run_id) == ["run.claim_released", "run.claimed"]


def test_admit_tolerates_partial_factory_plans(unit: UnitEnv) -> None:
    """Other areas' unit tests admit factory rows; staging (not admission) needs full plans."""
    with unit.ctx.db.tx() as tx:
        e = make_experiment(tx, cluster_id="unit")
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        h = make_exec_plan(tx, r)
        res = admit(tx, unit.ctx, run_id=r, exec_plan_hash=h)
    a = attempt(unit.ctx, res.attempt_id)
    assert a["gpus"] == 1 and a["remote_dir"].endswith(f"/attempts/{res.attempt_id}")
    assert res.reserved_gpu_hours == pytest.approx(1.0)
