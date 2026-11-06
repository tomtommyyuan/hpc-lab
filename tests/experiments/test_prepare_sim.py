"""experiment_prepare against the Slurm simulator: the real example project, the real
SlurmScheduler (``sbatch --test-only``) and remote reads through SimTransport."""

from __future__ import annotations

from pathlib import Path

import pytest
from prepare_support import make_venv

from hpclab.domain.models import ExecutionPlan, RunSpec
from hpclab.profiles.schema import SiteObservation
from hpclab.services import experiments as ex
from hpclab.storage.db import jload
from hpclab.testing.simctx import sim_context

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"
SPEC_PATH = EXAMPLE / "hpclab.yaml"

pytestmark = pytest.mark.sim


def test_prepare_tiny_classifier_example_on_sim(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        data = ex.experiment_prepare(env.ctx, idempotency_key="tiny-1", spec_path=str(SPEC_PATH))
        assert data.status == "ready", data.validation.blocking
        f = {x.check: x for x in data.validation.findings}
        assert f["scheduler_test"].status == "pass", f["scheduler_test"]
        assert f["scheduler_test"].evidence["partition"] == "gpu"
        assert f["env_guard_static"].status == "unchecked"  # the example declares no venv
        assert f["entrypoint_in_snapshot"].status == "pass"
        assert f["cluster_registered"].status == "pass" and f["partition_exists"].status == "pass"
        assert data.runs_total == 7 and data.smoke_run_id
        labels = [r.label for r in data.runs]
        assert labels[0] == "smoke.method-baseline.seed-1" and "method-mixup.seed-3" in labels
        # real snapshot of the example directory
        snap = data.snapshot
        assert snap and snap.ref and snap.mode == "captured"
        store_manifest = jload(
            (env.ctx.paths.snapshots / "manifests" / f"{snap.ref.digest.split(':')[1]}.json").read_text()
        )
        paths = {e["path"] for e in store_manifest["files"]}
        assert {"train.py", "hpclab_io.py", "hpclab.yaml"} <= paths
        assert snap.ref.project_root == str(EXAMPLE)
        # frozen exec plans resolve the sim overlay (account lab) and the remote root
        row = env.ctx.db.one(
            "SELECT * FROM runs WHERE plan_id=? AND label='method-mixup.seed-2'", (data.plan_id,)
        )
        assert row is not None
        rs = RunSpec.model_validate(jload(row["spec_json"]))
        assert rs.argv[:6] == ["python3", "train.py", "--method", "mixup", "--seed", "2"]
        ep = ExecutionPlan.model_validate(
            jload(
                env.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])[
                    "plan_json"
                ]
            )
        )
        assert ep.resources.account == env.cluster.overlay.account and ep.resources.partition == "gpu"
        assert ep.remote_root == env.remote_root and ep.gate == data.smoke_run_id
        # the reviewable script is accepted by the (simulated) scheduler as-is; it fits one page or
        # comes in pages of whole lines (the sim's long temp paths may push it past 64 KiB)
        script = ex.plan_script(env.ctx, plan_id=data.plan_id)
        text = script.script
        while script.next_cursor:
            assert script.truncated and script.script.endswith("\n")
            script = ex.plan_script(env.ctx, plan_id=data.plan_id, cursor=script.next_cursor)
            text += script.script
        assert len(text.encode()) == script.script_bytes
        path = Path(env.remote_root) / "review" / "job.sbatch"
        path.parent.mkdir(parents=True)
        path.write_text(text)
        est = env.scheduler.submit(str(path), workdir=str(path.parent), test_only=True)
        assert "to start at" in est.raw
        assert env.sim.jobs() == []  # nothing was ever queued
        # idempotent retry
        again = ex.experiment_prepare(env.ctx, idempotency_key="tiny-1", spec_path=str(SPEC_PATH))
        assert again.plan_id == data.plan_id
        # no scheduler_test leftovers on the cluster
        assert not (Path(env.remote_root) / ".prepare" / data.plan_id).exists()


def test_prepare_on_sim_blocks_bad_env_and_bad_account(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        profile = env.cluster.profile.model_copy(
            update={
                "facts": [
                    SiteObservation(
                        key="driver_cuda", value="12.8", source="test", observed_at="2026-08-01T00:00:00Z"
                    )
                ]
            }
        )
        env.ctx.clusters.register(profile, env.cluster.overlay)
        scratch = env.ctx.clusters.get("sim").storage_path("scratch")
        assert scratch
        make_venv(Path(scratch) / "venvs" / "tiny", torch_cuda="13.0")
        raw, base = ex.load_spec_file(str(SPEC_PATH))
        raw["runtime"] = {"python": "python3", "venv": "{scratch}/venvs/tiny", "guard": True}
        raw["resources"]["account"] = "nope"
        spec = ex.parse_spec(raw, base_dir=base).model_dump(mode="json")
        data = ex.experiment_prepare(env.ctx, idempotency_key="bad-1", spec=spec)
        f = {x.check: x for x in data.validation.findings}
        assert f["env_guard_static"].status == "fail", f["env_guard_static"]
        assert "13.0" in f["env_guard_static"].message
        assert (
            f["scheduler_test"].status == "fail"
            and f["scheduler_test"].evidence["reason_code"] == "invalid_account"
        )
        assert data.status == "invalid"
        assert set(data.validation.blocking) == {"env_guard_static", "scheduler_test"}
        assert env.ctx.db.one("SELECT COUNT(*) AS n FROM runs")["n"] == 0  # type: ignore[index]
        assert env.sim.jobs() == []


def test_prepare_on_sim_with_auth_down_reports_unchecked(tmp_path: Path) -> None:
    with sim_context(tmp_path) as env:
        env.sim.set_faults(auth_required=True)
        data = ex.experiment_prepare(env.ctx, idempotency_key="auth-1", spec_path=str(SPEC_PATH))
        f = {x.check: x for x in data.validation.findings}
        assert f["scheduler_test"].status == "unchecked"
        assert f["scheduler_test"].remediation == "hpclab auth sim"
        assert data.status == "ready" and data.validation.status == "warn"


def test_background_prepare_on_sim_runs_remote_checks_in_the_worker(tmp_path: Path) -> None:
    """Large snapshot -> 'preparing' receipt; the outbox worker captures, runs the real
    SlurmScheduler --test-only through SimTransport, and freezes the plan. Nothing is queued."""
    with sim_context(tmp_path) as env:
        env.ctx.settings.prepare_sync_max_bytes = 1  # force the background path
        data = ex.experiment_prepare(env.ctx, idempotency_key="bg-1", spec_path=str(SPEC_PATH))
        assert data.status == "preparing" and data.plan_hash is None and data.operation_id
        assert data.runs_total == 7 and data.runs[0].kind == "smoke"
        assert env.drain(ex.outbox_handlers()) >= 1
        plan = ex.plan_get(env.ctx, plan_id=data.plan_id, limit=100)
        assert plan.status == "ready" and plan.plan_hash and plan.runs_total == 7
        f = {x.check: x for x in plan.validation.findings}
        assert f["scheduler_test"].status == "pass", f["scheduler_test"]
        assert plan.budget and plan.budget["requested_gpu_hours_total"] == pytest.approx(
            data.budget.requested_gpu_hours_total
        )
        op = env.ctx.db.get("operations", "operation_id", data.operation_id)
        assert op["state"] == "succeeded" and jload(op["progress_json"]) == {}
        assert env.sim.jobs() == []
        assert not (Path(env.remote_root) / ".prepare" / data.plan_id).exists()
