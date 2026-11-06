"""Regression tests for service fixes found by the end-to-end suite (in-process, fast).

A workflow cell whose attempt ends WITHOUT reaching Slurm's terminal states -- sbatch refused
it, it was cancelled before dispatch, or a human abandoned an unresolved submission -- frees a
slot and settles the cell. The workflow must get a dispatch round right away (like
observe.finalize gives one for terminal attempts), not only from the 60 s safety-net sweep:
otherwise the next cell waits up to a minute and `attempt resolve --abandon` leaves the
workflow 'draining' for that long.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from hpclab.services import workflows as wfm
from hpclab.services.app import App
from hpclab.services.experiments import experiment_prepare
from hpclab.services.submission import outbox_handlers as submission_handlers
from hpclab.testing.simctx import SimEnv, sim_context

pytestmark = [pytest.mark.sim]


def _spec(project: Path) -> dict[str, Any]:
    (project / "job.py").write_text("print('hello')\n")
    return {
        "name": "fix-wf",
        "cluster": "sim",
        "code": {"project_root": str(project)},
        "entrypoint": {"argv": [sys.executable, "job.py"]},
        "runtime": {"python": sys.executable, "guard": False},
        "matrix": {"method": ["a"]},
        "resources": {"partition": "gpu", "gpus": 0, "cpus": 1, "mem": "1G", "time": "00:05:00"},
        "outputs": [{"name": "log", "path": "{output_dir}/x.txt", "role": "other", "required": False}],
    }


def _pending_dispatch(env: SimEnv, wf_id: str) -> bool:
    row = env.ctx.db.one(
        "SELECT state FROM outbox WHERE kind='workflow.dispatch' AND dedupe_key=?",
        (f"workflow.dispatch:{wf_id}",),
    )
    return row is not None and row["state"] == "pending"


def test_rejected_workflow_cell_triggers_a_dispatch_round(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    with sim_context(tmp_path) as env:
        env.scheduler.probe_capabilities()
        prep = experiment_prepare(env.ctx, idempotency_key="p", spec=_spec(project), validate_remote=False)
        assert prep.status == "ready", prep.validation
        wf = wfm.workflow_submit(env.ctx, idempotency_key="w", plan_id=prep.plan_id)
        dispatch_only = {"workflow.dispatch": App(env.ctx).outbox_handlers()["workflow.dispatch"]}
        env.drain(dispatch_only)  # the first dispatch round admits the cell
        assert not _pending_dispatch(env, wf.workflow_id)

        env.sim.set_faults(sbatch_reject="sbatch: error: Batch job submission failed: Invalid account")
        env.drain(submission_handlers())  # stage + dispatch -> definite refusal
        row = env.ctx.db.one("SELECT phase FROM attempts")
        assert row is not None and row["phase"] == "submission_rejected"
        assert _pending_dispatch(env, wf.workflow_id), "no dispatch round after the rejection"

        env.drain(App(env.ctx).outbox_handlers())  # the round settles the workflow; no sweep needed
        assert wfm.workflow_get(env.ctx, workflow_id=wf.workflow_id).state == "completed"
