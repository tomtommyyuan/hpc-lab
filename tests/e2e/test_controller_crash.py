"""Controller crashes (kill -9) while a 2x3 workflow is dispatching.

Slurm jobs keep running while the controller is dead; after a restart the workflow must finish,
the simulator (ground truth) must show at most one job per attempt, and every attempt must
either complete or carry an explanation a human can act on -- never sit silently in flight.

1. an external ``kill -9`` at an arbitrary moment of dispatch;
2. a deterministic crash right after a ``submit.dispatch`` committed ``submission_in_flight``
   and before sbatch ran (``benchmarks/crash_controller.py``). hpclab cannot know whether sbatch
   ran, so it reconciles, finds nothing, and asks a human instead of resubmitting; the human
   abandons the attempt and a recovery plan completes the matrix.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from e2e_support import UNRESOLVED_PHASES, Lab, argv_with, crash_controller_cmd

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]


def _assert_one_job_per_attempt(lab: Lab) -> None:
    names = lab.jobs_by_name()
    attempts = lab.attempts()
    assert not lab.duplicate_jobs(), names
    for a in attempts:
        expected = 1 if a["job_id"] else 0
        assert names.get(a["job_name"], 0) == expected, (a["attempt_id"], a["phase"], names)
    assert sum(names.values()) == sum(1 for a in attempts if a["job_id"]), (names, attempts)


def test_kill_9_mid_dispatch_then_restart_completes(lab: Lab) -> None:
    base = lab.spec()
    # ~2 s jobs so the kill lands while jobs are queued/running and cells are still planned
    spec = lab.spec(entrypoint={"argv": argv_with(base, "--sleep-per-step", "0.006")})
    prep = lab.prepare(spec)
    wf = lab.call(
        "workflow_submit", idempotency_key=lab.key("wf"), plan_id=prep["plan_id"], plan_hash=prep["plan_hash"]
    )
    wid = wf["workflow_id"]

    def mid_dispatch() -> bool:
        w = lab.workflow(wid)
        return w["state"] == "dispatching" and len(lab.sim.jobs()) >= 3 and w["counts"].get("planned", 0) > 0

    lab.wait(mid_dispatch, timeout=40, what="workflow dispatching main cells", interval=0.2)
    old_pid = lab.kill_controller()
    jobs_at_kill = len(lab.sim.jobs())

    # the next call autostarts a fresh controller, which releases the dead one's leases
    new_pid = lab.controller_pid()
    assert new_pid != old_pid
    done = lab.wait_workflow(wid, timeout=75)
    assert done["counts"] == {"succeeded": 7}, done["counts"]
    assert len(lab.wait_all_valid(workflow_id=wid)) == 2
    attempts = lab.attempts()
    assert len(attempts) == 7 and len(lab.sim.jobs()) == 7 >= jobs_at_kill
    assert all(a["phase"] == "terminal" and a["result_state"] == "valid" for a in attempts), lab.snapshot()
    _assert_one_job_per_attempt(lab)


def _unresolved(lab: Lab) -> Callable[[], dict[str, Any] | None]:
    def check() -> dict[str, Any] | None:
        for a in lab.attempts("a.phase=?", ("submission_uncertain",)):
            ops = lab.ledger(
                "SELECT state, stage, error_json FROM operations WHERE operation_id=?", (a["operation_id"],)
            )
            if ops and ops[0]["state"] == "blocked" and ops[0]["stage"] == "unresolved":
                return a
        return None

    return check


def test_crash_after_inflight_commit_is_never_resubmitted(
    make_lab: Callable[..., Lab], tmp_path: Path
) -> None:
    marker = tmp_path / "crashed.txt"
    # the 3rd sbatch of the workflow = the 2nd main cell (the smoke run is the 1st)
    lab: Lab = make_lab(controller_cmd=crash_controller_cmd("after_inflight", marker, nth=3))
    prep = lab.prepare(lab.spec())
    wf = lab.call(
        "workflow_submit", idempotency_key=lab.key("wf"), plan_id=prep["plan_id"], plan_hash=prep["plan_hash"]
    )
    wid = wf["workflow_id"]
    lab.wait(marker.exists, timeout=45, what="the controller to crash after an in-flight commit")
    lab.wait(lambda: not lab.client.ping(), timeout=10, what="the crashed controller to go away")

    in_flight = lab.attempts("a.phase=?", ("submission_in_flight",))
    assert len(in_flight) == 1, lab.snapshot()  # committed intent, then the crash
    victim = in_flight[0]
    assert victim["job_id"] is None and victim["job_name"] not in lab.jobs_by_name()

    lab.use_plain_controller()  # a normal restart
    blocked = lab.wait(_unresolved(lab), timeout=60, what="the uncertain attempt to be flagged unresolved")
    assert blocked["attempt_id"] == victim["attempt_id"]
    events = [
        e["event_type"]
        for e in lab.ledger(
            "SELECT event_type FROM events WHERE entity_id=? ORDER BY seq", (victim["attempt_id"],)
        )
    ]
    assert "attempt.uncertain" in events and "attempt.unresolved" in events, events
    run = lab.call("run_inspect", run_id=victim["run_id"])
    op = next(o for o in run["operations"] if o["operation_id"] == victim["operation_id"])
    assert op["state"] == "blocked" and "attempt resolve" in op["error"]["details"]["remediation"], op
    assert run["run"]["code"] == "U"
    # nothing resubmitted while the question is open: exactly the other cells ran
    lab.wait(lambda: lab.workflow(wid)["counts"].get("succeeded") == 6, timeout=60, what="the other 6 cells")
    assert victim["job_name"] not in lab.jobs_by_name()
    assert lab.workflow(wid)["state"] == "draining"  # waits for the human, does not pretend completion

    # the human decision: nothing was queued, abandon it; then recover the missing cell
    cur = lab.attempts("a.attempt_id=?", (victim["attempt_id"],))[0]
    res = lab.call(
        "attempt_resolve",
        attempt_id=victim["attempt_id"],
        expected_version=cur["version"],
        resolution="abandon",
        reason="squeue/sacct show no job with this token (controller crashed before sbatch)",
        idempotency_key=lab.key("resolve"),
    )
    assert res["phase"] == "abandoned", res
    lab.wait_workflow(wid, timeout=30)
    rec = lab.call(
        "run_recover", idempotency_key=lab.key("rec"), selector={"workflow_id": wid, "cells": "missing_valid"}
    )
    assert [c["run_id"] for c in rec["cells"] if c["eligible"]] == [victim["run_id"]], rec["cells"]
    wf2 = lab.call("workflow_submit", idempotency_key=lab.key("wf"), plan_id=rec["plan_id"])
    lab.wait_workflow(wf2["workflow_id"], timeout=45)
    lab.wait_all_valid(experiment_id=prep["experiment_id"])

    _assert_one_job_per_attempt(lab)
    assert len(lab.sim.jobs()) == 7  # smoke + 6 cells; the crashed attempt never reached Slurm
    stuck = [a for a in lab.attempts() if a["phase"] in UNRESOLVED_PHASES]
    assert not stuck, stuck
