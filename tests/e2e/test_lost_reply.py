"""Design scenario 1: a single run whose sbatch reply is lost.

Slurm (the simulator) accepts the job but the reply never reaches hpclab. The receipt must
not claim success, the attempt must be reconciled to exactly the one job Slurm accepted, the
idempotency key must replay the same receipt, a second submit of the run must be refused
(single execution claim), and the run must still flow through logs, automatic evaluation and
evidence export like any other.
"""

from __future__ import annotations

import pytest
from e2e_support import Lab

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(120)]


def test_lost_sbatch_reply_single_run(lab: Lab) -> None:
    spec = lab.spec(matrix={"method": ["baseline"]}, seeds=[1], smoke=None)
    prep = lab.prepare(spec)
    (run,) = prep["runs"]
    assert run["kind"] == "main" and run["exec_plan_hash"]

    lab.sim.set_faults(sbatch_lose_reply=1)
    key = lab.key("submit")
    receipt = lab.call(
        "run_submit", run_id=run["run_id"], execution_plan_hash=run["exec_plan_hash"], idempotency_key=key
    )
    aid = receipt["attempt_id"]
    assert receipt["run_id"] == run["run_id"] and receipt["operation_ref"].startswith("hpclab://operations/")
    assert receipt["phase"] == "staging"  # a receipt is never "accepted by Slurm"

    # same key, same payload -> the stored receipt (no second attempt)
    assert (
        lab.call(
            "run_submit", run_id=run["run_id"], execution_plan_hash=run["exec_plan_hash"], idempotency_key=key
        )
        == receipt
    )
    # another key while the attempt holds the run's execution claim -> refused
    err = lab.error(
        "run_submit",
        run_id=run["run_id"],
        execution_plan_hash=run["exec_plan_hash"],
        idempotency_key=lab.key("submit-again"),
    )
    # the plan was consumed by (and the run's claim is held by) the first attempt
    assert err["code"] == "admission_denied", err
    assert err["details"]["reason"] in ("plan_consumed", "run_claimed"), err
    assert aid in (err["details"].get("consumed_by_attempt_id"), err["details"].get("claim_attempt_id"))

    # the lost reply makes the outcome uncertain; reconciliation binds the one real job
    bound = lab.wait_attempt(aid, lambda a: a["job_id"] is not None, timeout=45, what="bound to a job")
    events = [
        e["event_type"]
        for e in lab.ledger("SELECT event_type FROM events WHERE entity_id=? ORDER BY seq", (aid,))
    ]
    assert "attempt.uncertain" in events, events
    assert lab.sim.faults().sbatch_lose_reply == 0  # the fault was consumed by our sbatch

    final = lab.wait_settled(aid)
    jobs = lab.sim.jobs()
    assert len(jobs) == 1, jobs  # ground truth: Slurm accepted exactly one job
    assert str(jobs[0]["job_id"]) == final["job_id"] == bound["job_id"]
    assert jobs[0]["name"] == final["job_name"] and jobs[0]["state"] == "COMPLETED"
    assert final["scheduler_state"] == "COMPLETED" and final["result_state"] == "valid"

    op = lab.call("operation_get", operation_id=receipt["operation_id"])
    assert op["state"] == "succeeded", op
    # replay after completion still returns the original receipt, and nothing new is queued
    assert (
        lab.call(
            "run_submit", run_id=run["run_id"], execution_plan_hash=run["exec_plan_hash"], idempotency_key=key
        )
        == receipt
    )
    assert len(lab.sim.jobs()) == 1

    logs = lab.call("run_logs", attempt_id=aid, stream="stdout", tail_lines=5)
    assert "done: method=baseline seed=1" in logs["text"], logs

    # evaluation is collected automatically when the outputs validate
    evals = lab.wait(lambda: lab.evaluated(run["run_id"]), timeout=30, what="automatic evaluation")
    assert evals[0]["attempt_id"] == aid and evals[0]["selected"] is True

    ev = lab.call(
        "evidence_export",
        purpose="lost-reply e2e",
        idempotency_key=lab.key("evidence"),
        planned_run_set_ref=prep["experiment_id"],
        metric_names=["acc"],
    )
    assert ev["counts"] == {"planned_runs": 1, "included": 1, "missing": 0, "failed": 0, "excluded": 0}
    tex = lab.call("evidence_get", bundle_id=ev["bundle_id"], file="table.tex")["content"]
    assert f"\\hpcval{{{ev['bundle_id']}:baseline.acc.mean}}" in tex, tex
    trace = lab.call("trace", cell_ref=f"{ev['bundle_id']}:baseline.acc.mean")
    (seed,) = trace["seeds"]
    assert seed["status"] == "included" and seed["chain"]["job"]["job_id"] == final["job_id"], seed
