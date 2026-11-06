"""Cancel identity: scancel only the job that provably belongs to the attempt.

Three runs are running. Cancelling the first must reach Slurm (ground truth: the simulator
recorded the scancel) and be confirmed by an observation. The job ids of the other two are
then made to point at jobs that are no longer the attempts' own -- one renamed (token gone),
one owned by another user (as after a job-id reuse) -- via simulator internals. Cancelling them
must be refused with the reason, and Slurm must receive zero scancels.
"""

from __future__ import annotations

from typing import Any

import pytest
from e2e_support import Lab, argv_with

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(120)]


def _tamper(lab: Lab, job_id: str, **changes: Any) -> None:
    state = lab.sim.state
    with state.tx() as conn:
        job = state.load_job(conn, int(job_id))
        assert job is not None and job.state == "RUNNING", job
        for k, v in changes.items():
            setattr(job, k, v)
        state.put_job(conn, job)


def _cancel(lab: Lab, aid: str) -> dict[str, Any]:
    version = lab.attempts("a.attempt_id=?", (aid,))[0]["version"]
    return lab.call(
        "run_cancel",
        attempt_id=aid,
        expected_version=version,
        idempotency_key=lab.key("cancel"),
        reason="e2e",
    )


def test_cancel_verifies_identity_before_scancel(lab: Lab) -> None:
    base = lab.spec()
    spec = lab.spec(
        matrix={"method": ["baseline"]},
        seeds=[1, 2, 3],
        smoke=None,
        budget={"gpu_hours": 5, "max_parallel": 3},
        # ~60 s jobs: still running when cancelled (all are killed at teardown)
        entrypoint={"argv": argv_with(base, "--sleep-per-step", "0.2")},
    )
    prep = lab.prepare(spec)
    aids = [
        lab.call(
            "run_submit",
            run_id=r["run_id"],
            execution_plan_hash=r["exec_plan_hash"],
            idempotency_key=lab.key("s"),
        )["attempt_id"]
        for r in prep["runs"]
    ]
    rows = [
        lab.wait_attempt(a, lambda r: r["scheduler_state"] == "RUNNING", timeout=40, what="RUNNING")
        for a in aids
    ]
    ok, renamed, foreign = rows

    # 1) a genuine cancel: verified, scancel reaches Slurm, confirmed by observation
    receipt = _cancel(lab, ok["attempt_id"])
    assert receipt["cancel_state"] in ("requested", "sent"), receipt
    job = lab.sim.wait_for(ok["job_id"], {"CANCELLED"}, timeout=30)
    assert job is not None and "CANCELLED" in (job["raw_state"] or job["state"]), job
    done = lab.wait_attempt(
        ok["attempt_id"], lambda a: a["cancel_state"] == "confirmed", timeout=30, what="cancel confirmed"
    )
    assert done["phase"] == "terminal" and done["scheduler_state"] == "CANCELLED"

    # 2) job ids that no longer identify the attempts' own jobs
    _tamper(lab, renamed["job_id"], name="train_llm_70b")
    _tamper(lab, foreign["job_id"], user="mallory")
    for row, expect in ((renamed, "token"), (foreign, "owner")):
        _cancel(lab, row["attempt_id"])
        refused = lab.wait_attempt(
            row["attempt_id"], lambda a: a["cancel_state"] == "refused", timeout=30, what="cancel refused"
        )
        events = lab.ledger(
            "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.cancel_refused'",
            (row["attempt_id"],),
        )
        assert events and expect in events[0]["payload_json"], events
        sim_job = lab.sim.state.job(int(row["job_id"]))
        assert sim_job is not None and sim_job.state == "RUNNING" and not sim_job.cancel_requested, sim_job
        assert refused["phase"] != "terminal"
        insp = lab.call("run_inspect", attempt_id=row["attempt_id"])
        op = next(o for o in insp["operations"] if o["kind"] == "cancel")
        assert op["state"] == "failed" and op["error"], op

    # ground truth: exactly one job was ever cancelled
    cancelled = [j for j in lab.sim.jobs() if j["state"] == "CANCELLED"]
    assert [str(j["job_id"]) for j in cancelled] == [ok["job_id"]], cancelled
