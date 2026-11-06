"""Circuit breaker: the same environment failure in 2 cells pauses dispatch.

The cluster environment lacks a package the job's ``runtime.setup`` imports (a missing module,
the classic 'job waited a day, then died in 10 s'). After two cells fail with the same
fingerprint, dispatch must pause instead of burning through the matrix; after the environment
is fixed, dispatch resumes and a recovery plan re-runs the two failed cells.
"""

from __future__ import annotations

import time

import pytest
from e2e_support import Lab

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]


def test_breaker_pauses_on_env_failure_and_resumes_after_fix(lab: Lab) -> None:
    site = lab.root / "cluster-site-packages"  # part of the cluster's environment, not of the Run
    site.mkdir()
    spec = lab.spec(
        smoke=None,
        runtime={"python": "python3", "setup": [f"export PYTHONPATH={site}", "python3 -c 'import labdeps'"]},
    )
    prep = lab.prepare(spec)
    wf = lab.call(
        "workflow_submit",
        idempotency_key=lab.key("wf"),
        plan_id=prep["plan_id"],
        plan_hash=prep["plan_hash"],
        max_parallel=2,
    )
    wid = wf["workflow_id"]

    paused = lab.wait(
        lambda: (w := lab.workflow(wid))["dispatch_state"] == "paused" and w,
        timeout=45,
        what="the circuit breaker to pause dispatch",
    )
    breaker = paused["breaker"]
    assert breaker["state"] == "tripped" and len(breaker["cells"]) >= 2, breaker
    fp = breaker["fingerprint"]
    assert (fp["category"], fp["code"]) == ("environment", "import_error") and "labdeps" in fp[
        "error_line"
    ], fp
    assert breaker["resume_hint"] and "resume_dispatch" in breaker["resume_hint"]
    # Normally exactly the 2 cells in flight ran. A 3rd can legitimately slip in when the two
    # failures (~0.1 s apart) straddle an observation cycle: the first frees a slot before the
    # second is seen, and the breaker needs 2 classified failures to trip.
    dispatched = len(lab.sim.jobs())
    assert dispatched in (2, 3), lab.snapshot()
    assert paused["counts"].get("planned") == 6 - dispatched, paused["counts"]

    # paused means paused: nothing new reaches Slurm (jobs already queued would keep running)
    time.sleep(6)  # more than one observe/dispatch cycle of the controller
    assert len(lab.sim.jobs()) == dispatched and lab.workflow(wid)["dispatch_state"] == "paused"
    after = lab.wait(
        lambda: (w := lab.workflow(wid))["counts"].get("failed") == dispatched and w,
        timeout=20,
        what="every dispatched cell to fail with the same error",
    )
    failed_runs = [c["run_id"] for c in after["cells"] if c["cell_state"] == "failed"]
    diag = lab.call("run_inspect", run_id=failed_runs[0])["diagnosis"]
    assert diag["primary_code"] == "import_error" and "labdeps" in (diag["last_error_line"] or ""), diag

    # fix the environment on the cluster, then resume dispatch of the remaining cells
    (site / "labdeps.py").write_text("VERSION = '1.0'\n")
    current = lab.workflow(wid)
    resumed = lab.call(
        "workflow_submit",
        idempotency_key=lab.key("resume"),
        workflow_id=wid,
        resume_dispatch=True,
        expected_version=current["version"],
    )
    assert resumed["resumed"] is True and resumed["dispatch_state"] == "open", resumed
    done = lab.wait_workflow(wid, timeout=60)
    assert done["counts"] == {"succeeded": 6 - dispatched, "failed": dispatched}, done["counts"]

    # the cells that failed before the fix: a recovery plan re-runs exactly them
    rec = lab.call(
        "run_recover", idempotency_key=lab.key("rec"), selector={"workflow_id": wid, "cells": "missing_valid"}
    )
    eligible = sorted(c["run_id"] for c in rec["cells"] if c["eligible"])
    assert eligible == sorted(failed_runs), rec["cells"]
    assert all(c["strategy"] == "retry" for c in rec["cells"] if c["eligible"])
    wf2 = lab.call("workflow_submit", idempotency_key=lab.key("wf"), plan_id=rec["plan_id"])
    lab.wait_workflow(wf2["workflow_id"], timeout=45)
    lab.wait_all_valid(experiment_id=prep["experiment_id"])
    # the failed cells + the rest after the fix + one retry per failed cell
    assert len(lab.sim.jobs()) == 6 + dispatched and not lab.duplicate_jobs()
