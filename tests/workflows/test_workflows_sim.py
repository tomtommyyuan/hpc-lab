"""End to end on the Slurm simulator with the production wiring: real experiment_prepare,
real submission/observe/artifacts handlers (``App.outbox_handlers()``), real SlurmScheduler.
Skipped when a service module of another area is not importable yet."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from wf_support import one

from hpclab.services import workflows as wfm
from hpclab.testing.simctx import SimEnv, sim_context, wait_until

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]

JOB = r"""
import json, os, sys, time
out = os.environ["HPCLAB_OUTPUT_DIR"]
os.makedirs(out, exist_ok=True)
time.sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
rec = {"schema": "hpclab.metrics/v1", "step": 1, "total_steps": 1, "split": "val", "final": True,
       "time": time.time(), "metrics": {"acc": 0.5}}
with open(os.path.join(out, "metrics.json"), "w") as fh:
    json.dump(rec, fh)
if sys.argv[1] == "boom":
    raise SystemExit("ModuleNotFoundError: No module named 'fancylib'")
"""


def _modules() -> dict[str, Any]:
    return {
        "experiments": pytest.importorskip("hpclab.services.experiments"),
        "observe": pytest.importorskip("hpclab.services.observe"),
        "submission": pytest.importorskip("hpclab.services.submission"),
        "app": pytest.importorskip("hpclab.services.app"),
    }


def _spec(
    project: Path, *, methods: list[str], seeds: list[int], smoke: bool, sleep: float = 0.5
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": "wf-sim",
        "cluster": "sim",
        "code": {"project_root": str(project)},
        "entrypoint": {"argv": [sys.executable, "job.py", "{method}", "{seed}", str(sleep)]},
        "runtime": {"python": sys.executable, "guard": False},
        "matrix": {"method": methods},
        "seeds": seeds,
        "resources": {"partition": "gpu", "gpus": 0, "cpus": 1, "mem": "1G", "time": "00:05:00"},
        "outputs": [{"name": "metrics", "path": "{output_dir}/metrics.json", "role": "metrics"}],
    }
    if smoke:
        spec["smoke"] = {"require_outputs": ["metrics"]}
    return spec


def _driver(env: SimEnv, mods: dict[str, Any], wf: str) -> tuple[Callable[[], str], list[int]]:
    runner = env.runner(mods["app"].App(env.ctx).outbox_handlers())
    samples: list[int] = []

    def tick() -> str:
        runner.drain(max_rounds=100)
        mods["observe"].poll_cluster(env.ctx, cluster_id=env.cluster_id, force=True)
        runner.drain(max_rounds=100)
        wfm.dispatch_workflow(env.ctx, workflow_id=wf)
        runner.drain(max_rounds=100)
        samples.append(wfm.workflow_get(env.ctx, workflow_id=wf).slots_used)
        return str(one(env.ctx, "SELECT state FROM workflows WHERE workflow_id=?", (wf,))["state"])

    return tick, samples


def _max_overlap(jobs: list[dict[str, Any]]) -> int:
    """Max number of jobs submitted-but-not-ended at once (what a slot holds), from ground truth."""
    points: list[tuple[float, int]] = []
    for j in jobs:
        points.append((float(j["submit_time"]), 1))
        points.append((float(j["end_time"]) if j["end_time"] is not None else float("inf"), -1))
    cur = best = 0
    for _t, d in sorted(points, key=lambda p: (p[0], p[1])):
        cur += d
        best = max(best, cur)
    return best


def test_workflow_on_sim_gates_on_smoke_and_respects_max_parallel(tmp_path: Path) -> None:
    mods = _modules()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "job.py").write_text(JOB)
    with sim_context(tmp_path) as env:
        env.scheduler.probe_capabilities()
        prep = mods["experiments"].experiment_prepare(
            env.ctx,
            idempotency_key="p",
            spec=_spec(project, methods=["a", "b"], seeds=[1, 2], smoke=True),
            validate_remote=False,
        )
        assert prep.status == "ready", prep.validation
        data = wfm.workflow_submit(
            env.ctx, idempotency_key="w", plan_id=prep.plan_id, plan_hash=prep.plan_hash, max_parallel=2
        )
        assert data.state == "gating" and data.planned_run_count == 5
        tick, samples = _driver(env, mods, data.workflow_id)
        wait_until(lambda: tick() == "completed", timeout=45, interval=0.2, what="workflow completion")
        jobs = env.sim.jobs()
        assert len(jobs) == 5  # every cell exactly once (ground truth: what sbatch accepted)
        assert max(samples) <= 2 and _max_overlap(jobs) <= 2
        smoke_job = one(
            env.ctx, "SELECT a.job_id FROM attempts a JOIN runs r ON r.run_id=a.run_id WHERE r.kind='smoke'"
        )["job_id"]
        smoke = next(j for j in jobs if str(j["job_id"]) == str(smoke_job))
        mains = [j for j in jobs if j is not smoke]
        assert all(float(j["submit_time"]) >= float(smoke["end_time"]) for j in mains)  # two-stage gate
        final = wfm.workflow_get(env.ctx, workflow_id=data.workflow_id)
        assert final.gate is not None and final.gate.state == "passed"
        assert final.counts.get("succeeded", 0) + final.counts.get("validating", 0) == 5
        assert final.operation_state == "succeeded"


def test_workflow_on_sim_breaker_pauses_on_identical_env_failures(tmp_path: Path) -> None:
    mods = _modules()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "job.py").write_text(JOB)
    with sim_context(tmp_path) as env:
        env.scheduler.probe_capabilities()
        prep = mods["experiments"].experiment_prepare(
            env.ctx,
            idempotency_key="p",
            spec=_spec(project, methods=["boom"], seeds=[1, 2, 3, 4], smoke=False),
            validate_remote=False,
        )
        assert prep.status == "ready", prep.validation
        data = wfm.workflow_submit(env.ctx, idempotency_key="w", plan_id=prep.plan_id, max_parallel=2)
        tick, _samples = _driver(env, mods, data.workflow_id)

        def paused() -> bool:
            tick()
            return bool(
                one(env.ctx, "SELECT dispatch_state FROM workflows WHERE workflow_id=?", (data.workflow_id,))[
                    "dispatch_state"
                ]
                == "paused"
            )

        wait_until(paused, timeout=45, interval=0.2, what="breaker pause")
        for _ in range(5):
            tick()
        # The breaker trips on the 2nd identical failure. If the two failures straddle an observe
        # tick, one more cell may already have been admitted (documented race) — but never all 4.
        submitted = len(env.sim.jobs())
        assert submitted in (2, 3), submitted
        got = wfm.workflow_get(env.ctx, workflow_id=data.workflow_id)
        assert got.breaker is not None and got.breaker.fingerprint is not None
        assert got.breaker.fingerprint["category"] == "environment"
        assert got.counts.get("planned") == 4 - submitted
