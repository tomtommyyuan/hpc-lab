"""Recovery end to end on the Slurm simulator: a job writes a checkpoint and is cancelled; the
checkpoint's metadata is validated from the login node, the built-in probe program runs through
the real job wrapper on a (simulated) compute node, and the resume attempt starts with
HPCLAB_RESUME_FROM pointing at that checkpoint. Production wiring (App.outbox_handlers())."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from wf_support import one

from hpclab.domain.models import ExecutionPlan
from hpclab.services import recovery as rec
from hpclab.storage.db import jload
from hpclab.testing.simctx import sim_context, wait_until

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]

TRAIN = r"""
import hashlib, json, os, sys, time
out = os.environ["HPCLAB_OUTPUT_DIR"]
os.makedirs(out, exist_ok=True)
resume = os.environ.get("HPCLAB_RESUME_FROM", "")
if resume:
    rec = {"schema": "hpclab.metrics/v1", "step": 10, "total_steps": 10, "split": "val", "final": True,
           "time": time.time(), "metrics": {"acc": 0.9}, "resumed_from": resume}
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(rec, fh)
    sys.exit(0)
d = os.path.join(os.environ["HPCLAB_CHECKPOINT_DIR"], "ckpt-000000005")
os.makedirs(d, exist_ok=True)
data = b"weights" * 64
with open(os.path.join(d, "model.pt"), "wb") as fh:
    fh.write(data)
manifest = {"schema": "hpclab.checkpoint/v1", "step": 5, "complete": True, "created_at": time.time(),
            "run_id": os.environ["HPCLAB_RUN_ID"], "attempt_id": os.environ["HPCLAB_ATTEMPT_ID"],
            "run_spec_hash": os.environ["HPCLAB_RUN_SPEC_HASH"], "config": json.loads(os.environ["HPCLAB_PARAMS_JSON"]),
            "semantic_keys": ["method"], "resumable": True,
            "files": [{"path": "model.pt", "size": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}]}
tmp = os.path.join(d, ".manifest.tmp")
with open(tmp, "w") as fh:
    json.dump(manifest, fh)
os.replace(tmp, os.path.join(d, "hpclab-ckpt.json"))
time.sleep(60)
"""


def test_cancel_checkpoint_probe_resume_on_sim(tmp_path: Path) -> None:
    experiments = pytest.importorskip("hpclab.services.experiments")
    observe = pytest.importorskip("hpclab.services.observe")
    submission = pytest.importorskip("hpclab.services.submission")
    app = pytest.importorskip("hpclab.services.app")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "train.py").write_text(TRAIN)
    spec: dict[str, Any] = {
        "name": "resume-sim",
        "cluster": "sim",
        "code": {"project_root": str(project)},
        "entrypoint": {"argv": [sys.executable, "train.py", "{method}"]},
        "runtime": {"python": sys.executable, "guard": False},
        "matrix": {"method": ["a"]},
        "resources": {"partition": "gpu", "gpus": 0, "cpus": 1, "mem": "1G", "time": "00:05:00"},
        "outputs": [
            {"name": "metrics", "path": "{output_dir}/metrics.json", "role": "metrics"},
            {
                "name": "checkpoints",
                "path": "{checkpoint_dir}",
                "kind": "dir",
                "role": "checkpoint",
                "required": False,
            },
        ],
        "resilience": {"resumable": True},
    }
    with sim_context(tmp_path) as env:
        env.scheduler.probe_capabilities()
        ctx = env.ctx
        runner = env.runner(app.App(ctx).outbox_handlers())

        def step() -> None:
            runner.drain(max_rounds=100)
            observe.poll_cluster(ctx, cluster_id=env.cluster_id, force=True)
            runner.drain(max_rounds=100)

        def attempt(aid: str) -> dict[str, Any]:
            row = one(ctx, "SELECT * FROM attempts WHERE attempt_id=?", (aid,))
            assert row is not None
            return row

        def until_terminal(aid: str) -> dict[str, Any]:
            def done() -> bool:
                step()
                a = attempt(aid)
                return a["phase"] == "terminal" and a["result_state"] not in ("unchecked", "pending")

            wait_until(done, timeout=30, interval=0.2, what=f"{aid} terminal")
            return attempt(aid)

        prep = experiments.experiment_prepare(ctx, idempotency_key="p", spec=spec, validate_remote=False)
        assert prep.status == "ready", prep.validation
        run_id = prep.runs[0].run_id
        first = submission.run_submit(
            ctx, run_id=run_id, execution_plan_hash=prep.runs[0].exec_plan_hash, idempotency_key="s1"
        )
        aid = first.attempt_id
        manifest = (
            Path(env.remote_root) / "attempts" / aid / "checkpoints" / "ckpt-000000005" / "hpclab-ckpt.json"
        )

        def checkpointed() -> bool:
            step()
            return manifest.exists()

        wait_until(checkpointed, timeout=30, interval=0.2, what="checkpoint written")
        # recovering a running attempt is refused
        assert not rec.run_recover(ctx, idempotency_key="early", attempt_id=aid).eligible
        submission.run_cancel(
            ctx, attempt_id=aid, expected_version=attempt(aid)["version"], idempotency_key="c1"
        )
        stopped = until_terminal(aid)
        assert stopped["scheduler_state"] == "CANCELLED"
        if ctx.db.one("SELECT 1 AS x FROM artifacts WHERE run_id=? AND role='checkpoint'", (run_id,)) is None:
            pytest.fail("artifacts.index did not record the declared checkpoint directory")

        # 1) metadata valid -> a probe run is created (never submitted by recovery itself)
        r1 = rec.run_recover(ctx, idempotency_key="r1", attempt_id=aid, strategy="resume")
        assert not r1.eligible and r1.probe is not None and r1.probe.created, r1.blocking
        assert r1.checkpoint is not None and r1.checkpoint.step == 5 and r1.checkpoint.continuation == "exact"
        assert len(env.sim.jobs()) == 1
        # 2) the probe runs through the real wrapper on a compute node
        assert r1.probe.run_id and r1.probe.exec_plan_hash
        probe = submission.run_submit(
            ctx, run_id=r1.probe.run_id, execution_plan_hash=r1.probe.exec_plan_hash, idempotency_key="s2"
        )
        pa = until_terminal(probe.attempt_id)
        assert pa["scheduler_state"] == "COMPLETED", pa
        report = json.loads((Path(pa["remote_dir"]) / "outputs" / "probe.json").read_text())
        assert report["ok"] is True and report["manifest_digest"] == r1.checkpoint.manifest_digest
        # 3) now a ready resume plan of the same Run
        r2 = rec.run_recover(ctx, idempotency_key="r2", attempt_id=aid, strategy="resume")
        assert r2.eligible and r2.exec_plan_hash and r2.probe and r2.probe.state == "passed", r2.blocking
        plan = ExecutionPlan.model_validate(
            jload(ctx.db.get("execution_plans", "exec_plan_hash", r2.exec_plan_hash)["plan_json"])
        )
        assert (
            plan.run_id == run_id
            and plan.recovery
            and plan.recovery.checkpoint_remote_path == str(manifest.parent)
        )
        # 4) the resume attempt starts from exactly that checkpoint
        resumed = submission.run_submit(
            ctx, run_id=run_id, execution_plan_hash=r2.exec_plan_hash, idempotency_key="s3"
        )
        ra = until_terminal(resumed.attempt_id)
        assert ra["scheduler_state"] == "COMPLETED" and ra["seq"] == 2
        out = json.loads((Path(ra["remote_dir"]) / "outputs" / "metrics.json").read_text())
        assert out["resumed_from"] == str(manifest.parent)
        assert len(env.sim.jobs()) == 3  # original, probe, resume: nothing submitted twice
