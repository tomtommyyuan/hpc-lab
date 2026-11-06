"""A just-failed cell keeps its dispatch slot until its failure is classified, so the circuit
breaker sees identical environment failures before their slots are refilled (bounded hold)."""

from __future__ import annotations

from datetime import timedelta

from hpclab.domain.clock import iso
from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id
from hpclab.services import workflows as wfm
from hpclab.storage.db import Database, jdump
from hpclab.testing.factories import make_attempt, make_experiment, make_plan, make_run


def _workflow_with_failed_cell(db: Database, **attempt: object) -> tuple[str, str]:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        wf = new_id("workflow")
        tx.insert(
            "workflows",
            {
                "workflow_id": wf,
                "experiment_id": e,
                "plan_id": p,
                "plan_hash": digest({"p": p}),
                "max_parallel": 2,
                "state": "dispatching",
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        a = make_attempt(tx, r, phase="terminal", workflow_id=wf, terminal_at=tx.now, **attempt)
    return wf, a


def test_unclassified_failure_holds_its_slot_until_diagnosed(db: Database) -> None:
    wf, a = _workflow_with_failed_cell(db, scheduler_state="FAILED", app_state="exited_error")
    now = iso(db.clock.now())
    assert wfm._slots_used(db, wf) == 0  # without the breaker the slot is free at once
    assert wfm._slots_used(db, wf, now=now) == 1
    with db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET diagnosis_json=? WHERE attempt_id=?",
            (jdump({"primary_code": "import_error"}), a),
        )
    assert wfm._slots_used(db, wf, now=now) == 0


def test_the_hold_is_bounded_and_never_applies_to_cancellations(db: Database) -> None:
    wf, _ = _workflow_with_failed_cell(db, scheduler_state="TIMEOUT")
    later = iso(db.clock.now() + timedelta(seconds=wfm.UNCLASSIFIED_FAILURE_HOLD_S + 5))
    assert wfm._slots_used(db, wf, now=later) == 0
    wf2, _ = _workflow_with_failed_cell(db, scheduler_state="CANCELLED")
    assert wfm._slots_used(db, wf2, now=iso(db.clock.now())) == 0
