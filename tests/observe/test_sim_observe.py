"""Observation against the Slurm simulator: real wrapper jobs submitted with the real scheduler,
attempts created with factories, then poll -> terminal -> finalize -> logs/inspect."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from observe_support import stub_budget_and_notify

from hpclab.domain.clock import iso
from hpclab.domain.models import (
    CodeSnapshotRef,
    EntrypointSpec,
    ExecutionPlan,
    OutputDecl,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.services.jobscript import attempt_layout, job_name_for, render_job_script
from hpclab.services.logs import run_logs
from hpclab.services.observe import finalize_attempt, poll_cluster, wait_for_background
from hpclab.services.runs import run_inspect
from hpclab.testing.factories import make_attempt, make_exec_plan, make_experiment, make_plan, make_run
from hpclab.testing.simctx import SimEnv, sim_context, wait_until

pytestmark = pytest.mark.sim

TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
)
SNAPSHOT_HEX = "cd" * 32

PROGRAM = r"""
import json, os, sys, time
with open(os.environ["HPCLAB_METRICS_PATH"], "a") as fh:
    for step in range(1, 6):
        rec = {"schema": "hpclab.metrics/v1", "step": step, "total_steps": 5, "time": time.time(),
               "split": "train", "metrics": {"loss": 1.0 / step}}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        time.sleep(float(os.environ.get("PROG_STEP_S", "0.3")))
print("training done")
if os.environ.get("PROG_FAIL"):
    raise ValueError("synthetic failure for the diagnosis test")
"""


@contextmanager
def sim_env(tmp_path: Path, **config: Any) -> Iterator[SimEnv]:
    with sim_context(tmp_path, **config) as env:
        try:
            env.scheduler.probe_capabilities()  # the one-time probe must not race MinJobAge
            yield env
        finally:
            wait_for_background(10)


def submit(env: SimEnv, *, extra_env: dict[str, str] | None = None) -> tuple[str, str, str]:
    """Render the real wrapper for a CPU-only attempt, sbatch it, record it. (run, attempt, job)."""
    root = env.remote_root
    snap = Path(root) / "snapshots" / SNAPSHOT_HEX
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "prog.py").write_text(PROGRAM)
    (snap / ".hpclab-complete").write_text("{}\n")
    run_spec = RunSpec(
        experiment="exp",
        code=CodeSnapshotRef(digest="sha256:" + SNAPSHOT_HEX, file_count=1, total_bytes=len(PROGRAM)),
        entrypoint=EntrypointSpec(argv=["python", "prog.py"]),
        argv=[sys.executable, "prog.py"],
        runtime=RuntimeSpec(
            python=sys.executable,
            env=dict(extra_env or {}),
        ),
        outputs=[OutputDecl(name="metrics", path="{output_dir}/metrics.jsonl", role="metrics")],
    )
    ctx = env.ctx
    with ctx.db.tx() as tx:
        eid = make_experiment(tx, cluster_id=env.cluster_id)
        pid = make_plan(tx, eid)
        run_id = make_run(tx, eid, pid, spec=run_spec.model_dump(mode="json"), seed=None)
    with ctx.db.tx() as tx:
        aid = make_attempt(
            tx, run_id, phase="staging", scheduler_state="UNKNOWN", cluster_id=env.cluster_id, job_id=None
        )
        found = tx.one("SELECT submit_token FROM attempts WHERE attempt_id=?", (aid,))
        assert found is not None
        token = found["submit_token"]
    layout = attempt_layout(root, aid)
    plan = ExecutionPlan(
        run_id=run_id,
        run_spec_hash=run_spec.spec_hash(),
        cluster_id=env.cluster_id,
        profile_revision=env.cluster.revision,
        resources=ResourceSpec(gpus=0, time="00:05:00", partition="gpu"),
        remote_root=root,
        snapshot_path=str(snap),
    )
    job_name = job_name_for("simtest", token)
    script = render_job_script(
        run_spec=run_spec,
        plan=plan,
        cluster=env.cluster,
        run_id=run_id,
        attempt_id=aid,
        token=token,
        job_name=job_name,
        layout=layout,
    )
    for d in layout.dirs_to_create():
        os.makedirs(d, exist_ok=True)
    Path(layout.script).write_text(script)
    resp = env.scheduler.submit(layout.script, workdir=layout.dir, receipt_path=layout.receipt)
    with ctx.db.tx() as tx:
        eph = make_exec_plan(tx, run_id, plan=plan.model_dump(mode="json"))
        tx.execute(
            "UPDATE attempts SET phase='job_identified', job_id=?, job_owner=?, job_name=?, remote_dir=?,"
            " exec_plan_hash=?, identified_at=?, gpus=0 WHERE attempt_id=?",
            (resp.job_id, env.sim.user, job_name, layout.dir, eph, tx.now, aid),
        )
        tx.execute("UPDATE runs SET claim_attempt_id=?, status='queued' WHERE run_id=?", (aid, run_id))
    return run_id, aid, resp.job_id


def one(env: SimEnv, sql: str, params: tuple[Any, ...]) -> dict[str, Any]:
    row = env.ctx.db.one(sql, params)
    assert row is not None, sql
    return row


def attempt(env: SimEnv, aid: str) -> dict[str, Any]:
    return one(env, "SELECT * FROM attempts WHERE attempt_id=?", (aid,))


def poll_until_terminal(env: SimEnv, aid: str, timeout: float = 20.0, *, force: bool = False) -> list[str]:
    seen: list[str] = []

    def step() -> bool:
        poll_cluster(env.ctx, cluster_id=env.cluster_id, force=force)
        a = attempt(env, aid)
        if not seen or seen[-1] != a["scheduler_state"]:
            seen.append(a["scheduler_state"])
        return a["phase"] == "terminal"

    wait_until(step, timeout=timeout, interval=0.25, what=f"attempt {aid} to become terminal")
    return seen


def test_real_job_through_its_whole_life(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settle, emit = stub_budget_and_notify(monkeypatch)
    with sim_env(tmp_path) as env:
        run_id, aid, job_id = submit(env)
        seen = poll_until_terminal(env, aid)
        assert "RUNNING" in seen and seen[-1] == "COMPLETED", seen
        a = attempt(env, aid)
        run = one(env, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        assert run["claim_attempt_id"] is None and run["status"] == "completed_unverified"
        assert a["job_submit_time"] is not None and a["node_list"]
        segs = env.ctx.db.all("SELECT * FROM attempt_segments WHERE attempt_id=?", (aid,))
        assert [s["segment"] for s in segs] == [0] and segs[0]["start_time"]

        env.drain({"observe.finalize": finalize_attempt})
        a = attempt(env, aid)
        assert a["app_state"] == "exited_ok" and a["exit_code"] == 0 and a["elapsed_s"] is not None
        assert '"schema":"hpclab.env/v1"' in a["env_json"] and '"state":"exited"' in a["status_json"]
        assert settle.calls and settle.calls[0]["consumed_gpu_hours"] == 0.0  # CPU-only attempt
        assert [c["event_type"] for c in emit.calls] == ["run.succeeded"]
        pending = env.ctx.db.all("SELECT kind FROM outbox WHERE state='pending' ORDER BY outbox_id")
        assert [p["kind"] for p in pending] == ["artifacts.index", "vault.capture"]

        out = run_logs(env.ctx, attempt_id=aid, tail_lines=5)
        assert "training done" in out.text
        head = run_logs(env.ctx, attempt_id=aid, max_bytes=4096)
        assert f"hpclab attempt {aid}" in head.text and f"job {job_id}" in head.text
        err = run_logs(env.ctx, attempt_id=aid, stream="stderr")
        assert "[hpclab] finished: state=exited exit_code=0" in err.text

        data = run_inspect(env.ctx, run_id=run_id)
        assert data.progress is not None and data.progress.step == 5 and data.progress.final_seen is False
        assert data.progress.fraction == 1.0
        assert data.env is not None and data.env.observed is not None
        assert data.attempts[0].job.job_id == job_id and data.attempts[0].job.owner == env.sim.user


def test_failed_job_is_diagnosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, emit = stub_budget_and_notify(monkeypatch)
    with sim_env(tmp_path) as env:
        run_id, aid, _ = submit(env, extra_env={"PROG_FAIL": "1", "PROG_STEP_S": "0.05"})
        seen = poll_until_terminal(env, aid)
        assert seen[-1] == "FAILED"
        env.drain({"observe.finalize": finalize_attempt})
        a = attempt(env, aid)
        assert a["app_state"] == "exited_error" and a["exit_code"] == 1
        assert '"primary_code":"python_exception"' in a["diagnosis_json"]
        assert [c["event_type"] for c in emit.calls] == ["run.failed"]
        data = run_inspect(env.ctx, run_id=run_id)
        assert data.diagnosis is not None and data.diagnosis.primary_code == "python_exception"
        assert data.run.status == "failed"


def test_min_job_age_and_accounting_lag_make_a_gap_not_a_completion(tmp_path: Path) -> None:
    with sim_env(tmp_path, min_job_age_s=0.0, accounting_lag_s=3.0) as env:
        run_id, aid, job_id = submit(env, extra_env={"PROG_STEP_S": "0.05"})
        wait_until(
            lambda: (
                poll_cluster(env.ctx, cluster_id=env.cluster_id) and attempt(env, aid)["phase"] == "observing"
            ),
            timeout=15,
            interval=0.2,
            what="first observation",
        )
        job = env.sim.wait_for(job_id, TERMINAL, 15)
        assert job is not None and job["state"] == "COMPLETED"
        time.sleep(0.3)  # past the job end, inside the accounting lag
        poll_cluster(env.ctx, cluster_id=env.cluster_id, force=True)
        a = attempt(env, aid)
        assert a["observation_gap"] == 1 and a["phase"] == "observing" and a["scheduler_state"] != "COMPLETED"
        run = one(env, "SELECT status FROM runs WHERE run_id=?", (run_id,))
        assert run["status"] == "unknown"
        seen = poll_until_terminal(env, aid, timeout=15)
        a = attempt(env, aid)
        assert seen[-1] == "COMPLETED" and a["observation_gap"] == 0 and a["exit_code"] == 0
        gaps = env.ctx.db.all(
            "SELECT * FROM events WHERE entity_id=? AND event_type='attempt.observation_gap'", (aid,)
        )
        assert len(gaps) == 1


def test_sacct_outage_never_reports_completion(tmp_path: Path) -> None:
    with sim_env(tmp_path, min_job_age_s=0.0) as env:
        _, aid, job_id = submit(env, extra_env={"PROG_STEP_S": "0.05"})
        env.sim.set_faults(sacct_unavailable=True)
        assert env.sim.wait_for(job_id, TERMINAL, 15) is not None
        time.sleep(0.3)
        res = poll_cluster(env.ctx, cluster_id=env.cluster_id, force=True)
        a = attempt(env, aid)
        assert res.sacct in ("unavailable", "unsupported") and a["observation_gap"] == 1
        assert a["phase"] != "terminal"
        ps = one(env, "SELECT * FROM poll_state WHERE cluster_id=? AND source='sacct'", (env.cluster_id,))
        assert ps["consecutive_failures"] >= 1 and ps["last_error"]
        env.sim.clear_faults()
        # sacct is backing off after the failure; an explicit (forced) poll does not wait for it
        poll_until_terminal(env, aid, timeout=15, force=True)
        assert attempt(env, aid)["scheduler_state"] == "COMPLETED"
        assert iso(env.ctx.clock.now()) is not None
