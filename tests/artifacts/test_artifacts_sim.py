"""End to end on the simulator: examples/tiny-classifier runs as a Slurm job through the real
job wrapper; its outputs are indexed, listed, read, evaluated, vaulted and backed up."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hpclab.domain.clock import iso
from hpclab.domain.models import (
    CodeOptions,
    EntrypointSpec,
    EvaluationDefaults,
    ExecutionPlan,
    OutputDecl,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.services.artifacts import artifact_read, artifacts_list, index_attempt
from hpclab.services.evaluations import evaluation_collect
from hpclab.services.jobscript import attempt_layout, job_name_for, render_job_script
from hpclab.services.snapshot import SnapshotStore, capture_snapshot, publish_snapshot
from hpclab.services.vault import backup, capture, vault_status
from hpclab.testing.factories import make_attempt, make_exec_plan, make_experiment, make_plan, make_run
from hpclab.testing.simctx import SimEnv, sim_context

pytestmark = pytest.mark.sim

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"
TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
)


def run_example_job(
    env: SimEnv, tmp_path: Path, *, extra_argv: list[str] | None = None
) -> tuple[str, str, str]:
    """Snapshot + publish the example, render the wrapper, sbatch it; return (experiment, attempt, job)."""
    ctx = env.ctx
    project = tmp_path / "project"
    project.mkdir()
    for name in ("train.py", "hpclab_io.py"):
        shutil.copy2(EXAMPLE / name, project / name)
    store = SnapshotStore(ctx.paths.snapshots)
    snap = capture_snapshot(project, CodeOptions(), store)
    snapshot_path = publish_snapshot(env.transport, store, snap.ref, env.remote_root)
    outputs = [
        OutputDecl(
            name="metrics", path="{output_dir}/metrics.jsonl", role="metrics", format="hpclab.metrics/v1"
        ),
        OutputDecl(
            name="summary", path="{output_dir}/metrics.json", role="metrics", format="hpclab.metrics/v1"
        ),
        OutputDecl(
            name="checkpoints",
            path="{checkpoint_dir}",
            kind="dir",
            role="checkpoint",
            format="hpclab.checkpoint/v1",
            required=False,
        ),
    ]
    argv = [
        sys.executable, "train.py", "--method", "baseline", "--seed", "1", "--steps", "60", "--eval-every", "20",
        "--ckpt-every", "20", "--out", "{output_dir}", "--ckpt-dir", "{checkpoint_dir}", *(extra_argv or []),
    ]  # fmt: skip
    run_spec = RunSpec(
        experiment="tiny-e2e",
        code=snap.ref,
        entrypoint=EntrypointSpec(argv=["python3", "train.py", "--method", "{method}", "--seed", "{seed}"]),
        argv=argv,
        params={"method": "baseline"},
        cell={"method": "baseline", "seed": 1},
        seed=1,
        runtime=RuntimeSpec(python=sys.executable, guard=False),
        outputs=outputs,
        evaluation=EvaluationDefaults(config={"split": "val"}),
    )
    with ctx.db.tx() as tx:
        exp = make_experiment(tx, name="tiny-e2e", cluster_id=env.cluster_id)
        plan_id = make_plan(tx, exp)
        run_id = make_run(
            tx,
            exp,
            plan_id,
            seed=1,
            cell={"method": "baseline"},
            spec=run_spec.model_dump(mode="json"),
            status="submitting",
        )
    plan = ExecutionPlan(
        run_id=run_id,
        run_spec_hash=run_spec.spec_hash(),
        cluster_id=env.cluster_id,
        profile_revision=env.cluster.revision,
        resources=ResourceSpec(partition="gpu", gpus=1, gpu_type="a100", cpus=1, mem="1G", time="00:05:00"),
        remote_root=env.remote_root,
        snapshot_path=snapshot_path,
    )
    with ctx.db.tx() as tx:
        eph = make_exec_plan(tx, run_id, plan=plan.model_dump(mode="json"))
        attempt_id = make_attempt(
            tx,
            run_id,
            exec_plan_hash=eph,
            cluster_id=env.cluster_id,
            phase="staging",
            scheduler_state="UNKNOWN",
            job_id=None,
        )
        token = tx.one("SELECT submit_token FROM attempts WHERE attempt_id=?", (attempt_id,))["submit_token"]  # type: ignore[index]
    layout = attempt_layout(env.remote_root, attempt_id)
    script = render_job_script(
        run_spec=run_spec,
        plan=plan,
        cluster=env.cluster,
        run_id=run_id,
        attempt_id=attempt_id,
        token=token,
        job_name=job_name_for("tiny-e2e", token),
        layout=layout,
    )
    for d in layout.dirs_to_create():
        Path(d).mkdir(parents=True, exist_ok=True)
    env.transport.write_file(layout.script, script.encode())
    env.transport.write_file(
        layout.token_file,
        json.dumps(
            {"run_id": run_id, "attempt_id": attempt_id, "token": token, "plan_hash": plan.plan_hash()}
        ).encode(),
    )
    resp = env.scheduler.submit(layout.script, workdir=layout.dir, receipt_path=layout.receipt)
    with ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET remote_dir=?, job_id=?, phase='observing', scheduler_state='PENDING' WHERE attempt_id=?",
            (layout.dir, resp.job_id, attempt_id),
        )
    job = env.sim.wait_for(resp.job_id, TERMINAL, 25.0)
    assert job is not None, env.sim.job(resp.job_id)
    return exp, attempt_id, resp.job_id


def mark_terminal(env: SimEnv, attempt_id: str, state: str) -> None:
    """What services/observe.py records when the job reaches a terminal state."""
    with env.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET phase='terminal', scheduler_state=?, terminal_at=? WHERE attempt_id=?",
            (state, iso(env.ctx.clock.now()), attempt_id),
        )


def test_tiny_classifier_outputs_are_indexed_read_evaluated_and_vaulted(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        ctx = env.ctx
        try:
            exp, attempt_id, job_id = run_example_job(env, tmp_path)
            job = env.sim.job(job_id)
            assert job is not None and job["state"] == "COMPLETED", job

            # while the ledger still says "observing", everything is mutable
            early = index_attempt(ctx, attempt_id=attempt_id)
            assert early.result_state == "pending" and {o.integrity for o in early.outputs} == {"mutable"}

            mark_terminal(env, attempt_id, "COMPLETED")
            res = index_attempt(ctx, attempt_id=attempt_id, background=True)
            assert res.result_state == "valid", res
            assert res.manifest == "present" and res.manifest_hash_state == "complete"
            by = {o.name: o for o in res.outputs}
            assert {o.integrity for o in res.outputs} == {"finalized"}
            assert (
                by["metrics"].digest_source == "compute_node"
                and by["summary"].digest_source == "compute_node"
            )
            ckpt_dir = Path(attempt_layout(env.remote_root, attempt_id).checkpoint_dir)
            on_disk = [p for p in ckpt_dir.rglob("*") if p.is_file()]
            assert len({p.parent for p in on_disk}) == 3 and by["checkpoints"].members == len(on_disk)

            listing = artifacts_list(ctx, attempt_id=attempt_id, include_members=True, limit=100)
            assert listing.coverage.complete and listing.result_state == "valid"
            members = [a for a in listing.artifacts if a.member_path]
            assert members and all(a.integrity == "finalized" and a.digest for a in members)
            assert not any("/." in a.name for a in members)

            metrics = next(a for a in listing.artifacts if a.name == "metrics")
            tail = artifact_read(ctx, artifact_ref=metrics.artifact_ref)
            assert tail.view == "jsonl_tail" and tail.digest_verified
            finals = [r for r in tail.records or [] if r.get("final")]
            assert len(finals) == 1 and finals[0]["split"] == "val" and finals[0]["step"] == 60
            summary = next(a for a in listing.artifacts if a.name == "summary")
            doc = artifact_read(ctx, artifact_ref=summary.artifact_ref)
            assert doc.view == "json" and isinstance(doc.value, dict)

            collected = evaluation_collect(ctx, idempotency_key="e2e", attempt_id=attempt_id)
            ev = collected.evaluation
            assert ev is not None and ev.status == "valid", ev
            acc = next(m for m in ev.metrics if m["name"] == "acc")
            assert acc["value"] == finals[0]["metrics"]["acc"] and acc["split"] == "val"
            assert ev.provenance["attempt"]["job"]["job_id"] == job_id
            assert ev.inputs[0]["digest"] == metrics.digest
            batch = evaluation_collect(ctx, idempotency_key="e2e-batch", run_set_ref=exp)
            assert batch.collected == [] and [s.reason for s in batch.skipped] == ["already_evaluated"]

            vaulted = capture(ctx, attempt_id=attempt_id)
            kinds = {i.kind for i in vaulted.captured}
            assert {
                "script",
                "plan",
                "status",
                "env",
                "outputs_manifest",
                "log_head",
                "metrics_file",
                "sacct_raw",
            } <= kinds
            assert vaulted.coverage.complete, vaulted
            sacct = next(i for i in vaulted.captured if i.kind == "sacct_raw")
            record = json.loads(Path(sacct.local_path).read_text())["records"][0]
            assert record["job_id"] == job_id and record["state"] == "COMPLETED"
            status = vault_status(ctx, experiment_id=exp)
            assert status.fully_captured == 1 and status.not_captured == 0

            bk = backup(ctx, dest=str(tmp_path / "backups"))
            assert bk.integrity_check == "ok" and bk.vault_files == len(vaulted.captured)
            assert bk.snapshot_manifests == 1
        finally:
            subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=False)


def test_failed_job_without_final_outputs_is_incomplete_and_not_collected(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        try:
            exp, attempt_id, job_id = run_example_job(env, tmp_path, extra_argv=["--fail-at-step", "30"])
            job = env.sim.job(job_id)
            assert job is not None and job["state"] == "FAILED", job
            mark_terminal(env, attempt_id, "FAILED")
            res = index_attempt(env.ctx, attempt_id=attempt_id)
            by = {o.name: o for o in res.outputs}
            assert by["summary"].integrity == "missing" and res.result_state == "incomplete"
            batch = evaluation_collect(env.ctx, idempotency_key="fail-batch", run_set_ref=exp)
            assert batch.collected == [] and batch.skipped[0].reason == "scheduler_state_not_completed"
        finally:
            subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=False)
