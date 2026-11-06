"""experiment_prepare flows against a local fake cluster (no simulator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from prepare_support import Lab

from hpclab.domain.errors import (
    IdempotencyConflict,
    InvalidArgument,
    LimitExceeded,
    NotFound,
    NotReady,
    PreconditionFailed,
)
from hpclab.domain.models import ExecutionPlan, RunSpec
from hpclab.services import experiments as ex
from hpclab.services.outbox import OutboxRunner
from hpclab.services.registry import Registry
from hpclab.storage.db import jload
from hpclab.testing.factories import make_attempt, make_evaluation


def prepare(lab: Lab, key: str = "k1", **kw: Any) -> ex.PrepareData:
    kw.setdefault("validate_remote", False)
    if "spec" not in kw and "spec_path" not in kw and "extends" not in kw and "derive_from" not in kw:
        kw["spec"] = lab.spec()
    return ex.experiment_prepare(lab.ctx, idempotency_key=key, **kw)


def findings(data: ex.PrepareData) -> dict[str, Any]:
    return {f.check: f for f in data.validation.findings}


def count(lab: Lab, table: str, where: str = "1=1", params: tuple[Any, ...] = ()) -> int:
    row = lab.ctx.db.one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params)
    assert row is not None
    return int(row["n"])


def events(lab: Lab, event_type: str) -> list[dict[str, Any]]:
    return lab.ctx.db.all("SELECT * FROM events WHERE event_type=? ORDER BY seq", (event_type,))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_prepare_ready_plan_freezes_runs_and_exec_plans(lab: Lab) -> None:
    data = prepare(lab)
    assert data.status == "ready", data.validation.blocking
    assert data.plan_ref == f"hpclab://plans/{data.plan_id}"
    assert data.plan_hash and data.plan_hash.startswith("sha256:")
    assert data.runs_total == 7 and len(data.runs) == 7
    smoke, *main = data.runs
    assert smoke.kind == "smoke" and smoke.label == "smoke.method-baseline.seed-1"
    assert [r.label for r in main][:2] == ["method-baseline.seed-1", "method-baseline.seed-2"]
    assert data.smoke_run_id == smoke.run_id
    assert data.snapshot and data.snapshot.mode == "captured" and data.snapshot.ref
    assert data.budget.requested_gpu_hours_total == pytest.approx(6 * 1.0 + 0.25)
    assert data.budget.smoke_gpu_hours == pytest.approx(0.25)
    assert data.budget.budget_gpu_hours == 50
    # ledger rows
    assert count(lab, "runs", "plan_id=?", (data.plan_id,)) == 7
    assert count(lab, "execution_plans") == 7
    exp = lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)
    assert exp["status"] == "ready" and exp["budget_gpu_hours"] == 50 and exp["max_parallel"] == 2
    plan = lab.ctx.db.get("plans", "plan_id", data.plan_id)
    assert plan["status"] == "ready" and plan["plan_hash"] == data.plan_hash
    assert plan["smoke_run_id"] == smoke.run_id and plan["snapshot_digest"] == data.snapshot.ref.digest
    # insertion order = matrix order (smoke first)
    labels = [
        r["label"]
        for r in lab.ctx.db.all("SELECT label FROM runs WHERE plan_id=? ORDER BY rowid", (data.plan_id,))
    ]
    assert labels == [r.label for r in data.runs]
    assert len(events(lab, "run.created")) == 7 and len(events(lab, "plan.ready")) == 1
    assert len(events(lab, "experiment.created")) == 1


def test_frozen_run_specs_and_execution_plans(lab: Lab) -> None:
    data = prepare(lab)
    main = next(r for r in data.runs if r.label == "method-mixup.seed-2")
    row = lab.ctx.db.get("runs", "run_id", main.run_id or "")
    rs = RunSpec.model_validate(jload(row["spec_json"]))
    assert rs.argv == ["python", "train.py", "--method", "mixup", "--seed", "2", "--out", "{output_dir}"]
    assert rs.params == {"method": "mixup"} and rs.seed == 2 and rs.cell == {"method": "mixup", "seed": 2}
    assert row["cell_key"] == '{"method":"mixup","seed":2}' and row["status"] == "prepared"
    assert row["spec_hash"] == rs.spec_hash()
    ep_row = lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])
    ep = ExecutionPlan.model_validate(jload(ep_row["plan_json"]))
    assert ep.plan_hash() == row["current_exec_plan_hash"] == main.exec_plan_hash
    assert ep.run_spec_hash == rs.spec_hash()
    assert ep.resources.partition == "gpu" and ep.resources.account == "lab-acct"  # defaults resolved
    assert ep.gate == data.smoke_run_id
    assert ep.remote_root == str(lab.remote_root)
    assert ep.snapshot_path == f"{lab.remote_root}/snapshots/{rs.code.digest.split(':')[1]}"
    assert ep.requeue is False and ep.cluster_id == "lab"  # not resumable: --no-requeue everywhere
    assert ep.profile_revision == lab.ctx.clusters.get("lab").revision
    smoke_row = lab.ctx.db.get("runs", "run_id", data.smoke_run_id or "")
    smoke_spec = RunSpec.model_validate(jload(smoke_row["spec_json"]))
    assert smoke_spec.kind == "smoke" and smoke_spec.argv[-2:] == ["--steps", "5"]
    smoke_ep = ExecutionPlan.model_validate(
        jload(
            lab.ctx.db.get("execution_plans", "exec_plan_hash", smoke_row["current_exec_plan_hash"])[
                "plan_json"
            ]
        )
    )
    assert smoke_ep.gate is None and smoke_ep.resources.time == "00:15:00"


def test_plan_hash_is_deterministic_over_run_and_plan_hashes(lab: Lab) -> None:
    data = prepare(lab)
    rows = lab.ctx.db.all("SELECT * FROM runs WHERE plan_id=? ORDER BY rowid", (data.plan_id,))
    from hpclab.domain.hashing import digest

    expected = digest(
        {
            "kind": "matrix",
            "runs": [
                {
                    "run_id": r["run_id"],
                    "kind": r["kind"],
                    "run_spec_hash": r["spec_hash"],
                    "exec_plan_hash": r["current_exec_plan_hash"],
                }
                for r in rows
            ],
        }
    )
    assert data.plan_hash == expected


def test_validation_findings_have_contract_shape(lab: Lab) -> None:
    data = prepare(lab)
    f = findings(data)
    for name in (
        "spec_schema",
        "placeholders",
        "cluster_registered",
        "partition_exists",
        "time_within_partition_max",
        "gpu_type_in_partition",
        "gpus_limits",
        "account_present",
        "qos_required",
        "interactive_partition_batch",
        "storage_home_avoidance",
        "outputs_declared",
        "evaluation_output_declared",
        "preemptible_gate",
        "walltime_signal",
        "budget",
        "lockfiles_present",
        "inputs_pinning",
        "env_guard_static",
        "snapshot_bypass",
        "scheduler_test",
        "resource_advice",
        "script_render",
    ):
        assert name in f, name
    assert f["env_guard_static"].status == "unchecked"  # validate_remote=False: never "pass"
    assert f["scheduler_test"].status == "unchecked"
    assert f["resource_advice"].status == "unchecked"  # no history
    assert f["lockfiles_present"].status == "pass"
    assert data.validation.status == "warn"  # unchecked findings are not a clean pass
    cov = {x.name: x.status for x in data.coverage.sources}
    assert cov == {"ledger": "ok", "snapshot": "ok", "cluster:lab": "skipped"} and data.coverage.complete
    assert data.observed_at == "2026-09-01T12:00:00.000Z"
    for item in data.validation.findings:
        assert item.status in ("pass", "warn", "fail", "unchecked")
        assert item.severity in ("block", "warn", "info")


def test_spec_path_yaml_relative_project_root(lab: Lab, tmp_path: Path) -> None:
    spec = lab.spec()
    spec["code"] = {"project_root": "."}
    spec["resources"]["time"] = "01:00:00"
    path = lab.project / "hpclab.yaml"
    # unquoted HH:MM:SS must stay a string (not base-60 int)
    text = yaml.safe_dump(spec).replace("time: 01:00:00", "time: 01:00:00").replace("'01:00:00'", "01:00:00")
    path.write_text(text)
    data = prepare(lab, spec_path=str(path))
    assert data.status == "ready"
    exp = lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)
    assert exp["project_root"] == str(lab.project.resolve())
    # the same content inline (absolute root) is the same request -> same stored response
    inline = lab.spec()
    inline["code"] = {"project_root": str(lab.project.resolve())}
    again = prepare(lab, spec=inline)
    assert again.plan_id == data.plan_id


def test_spec_path_errors(lab: Lab, tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument, match="absolute"):
        prepare(lab, spec_path="relative/hpclab.yaml")
    with pytest.raises(NotFound):
        prepare(lab, spec_path=str(tmp_path / "missing.yaml"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(InvalidArgument, match="YAML"):
        prepare(lab, spec_path=str(bad))
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n- b\n")
    with pytest.raises(InvalidArgument, match="mapping"):
        prepare(lab, spec_path=str(lst))


def test_inline_spec_needs_absolute_project_root(lab: Lab) -> None:
    spec = lab.spec()
    spec["code"] = {"project_root": "."}
    with pytest.raises(InvalidArgument, match="absolute"):
        prepare(lab, spec=spec)


def test_invalid_spec_is_an_error_not_a_plan(lab: Lab) -> None:
    with pytest.raises(InvalidArgument) as ei:
        prepare(lab, spec=lab.spec(entrypoint={"argv": []}))
    assert ei.value.details["errors"]
    with pytest.raises(InvalidArgument):
        prepare(lab, spec=lab.spec(seeds=None, matrix={"method": ["a"], "seed": ["x"]}))
    assert count(lab, "experiments") == 0


def test_request_overrides_cluster_experiment_matrix(lab: Lab) -> None:
    data = prepare(lab, experiment="renamed", matrix={"method": ["a", "b", "c"]})
    assert data.runs_total == 1 + 9
    exp = lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)
    assert exp["name"] == "renamed"
    with pytest.raises(InvalidArgument):
        prepare(lab, key="k2", spec=lab.spec(), spec_path="/x.yaml")
    with pytest.raises(InvalidArgument):
        ex.experiment_prepare(lab.ctx, idempotency_key="k3")


def test_matrix_over_limit_is_limit_error(lab: Lab) -> None:
    with pytest.raises(InvalidArgument):
        prepare(lab, matrix={"a": list(range(100))})  # 100 x 3 seeds > 256 (spec model refuses)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_same_key_same_payload_returns_stored_response(lab: Lab) -> None:
    first = prepare(lab)
    second = prepare(lab)
    assert second.model_dump() == first.model_dump()
    assert count(lab, "plans") == 1 and count(lab, "runs") == 7


def test_same_key_different_payload_conflicts(lab: Lab) -> None:
    prepare(lab)
    with pytest.raises(IdempotencyConflict):
        prepare(lab, spec=lab.spec(seeds=[1, 2]))
    with pytest.raises(IdempotencyConflict):
        prepare(lab, validate_remote=True)
    assert count(lab, "plans") == 1


def test_different_key_creates_a_new_experiment(lab: Lab) -> None:
    a = prepare(lab, key="a")
    b = prepare(lab, key="b")
    assert a.experiment_id != b.experiment_id and a.plan_id != b.plan_id


def test_idempotency_stored_in_same_transaction(lab: Lab) -> None:
    data = prepare(lab)
    row = lab.ctx.db.one(
        "SELECT response_json FROM idempotency WHERE tool='experiment_prepare' AND idem_key='k1'"
    )
    assert row is not None and jload(row["response_json"])["plan_id"] == data.plan_id


# ---------------------------------------------------------------------------
# Invalid plans and overrides
# ---------------------------------------------------------------------------


def test_unknown_placeholder_makes_plan_invalid_without_runs(lab: Lab) -> None:
    spec = lab.spec(entrypoint={"argv": ["python", "train.py", "--lr", "{lr}"]})
    data = prepare(lab, spec=spec)
    assert data.status == "invalid" and data.plan_hash is None and data.smoke_run_id is None
    assert "placeholders" in data.validation.blocking
    assert all(r.run_id is None for r in data.runs) and data.runs_total == 7
    assert count(lab, "runs") == 0 and count(lab, "execution_plans") == 0
    exp = lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)
    assert exp["status"] == "invalid"
    pg = ex.plan_get(lab.ctx, plan_id=data.plan_id)
    assert pg.status == "invalid" and pg.runs == [] and len(pg.planned_runs) == 7
    assert pg.validation.status == "fail"
    with pytest.raises(PreconditionFailed):
        ex.plan_script(lab.ctx, plan_id=data.plan_id)
    # a non-overridable check cannot be overridden
    again = prepare(lab, key="k2", spec=spec, overrides=[{"check": "placeholders", "reason": "trust me"}])
    assert again.status == "invalid"
    assert any("cannot be overridden" in w for w in again.warnings)


def test_budget_block_and_override_recorded_as_event(lab: Lab) -> None:
    spec = lab.spec(budget={"gpu_hours": 2})
    data = prepare(lab, spec=spec)
    assert data.status == "invalid" and data.validation.blocking == ["budget"]
    over = prepare(
        lab, key="k2", spec=spec, overrides=[{"check": "budget", "reason": "PI approved extra hours"}]
    )
    assert over.status == "ready"
    f = findings(over)["budget"]
    assert f.status == "warn" and f.severity == "warn" and f.override_reason == "PI approved extra hours"
    assert over.validation.overridden == ["budget"]
    assert over.overrides[0].applied is True
    evs = events(lab, "validation.override")
    assert len(evs) == 1 and jload(evs[0]["payload_json"])["reason"] == "PI approved extra hours"
    assert evs[0]["entity_id"] == over.plan_id


def test_override_of_passing_or_unknown_check_is_reported(lab: Lab) -> None:
    data = prepare(
        lab,
        overrides=[{"check": "budget", "reason": "not needed"}, {"check": "no_such_check", "reason": "typo"}],
    )
    assert data.status == "ready"
    assert {o.check: o.applied for o in data.overrides} == {"budget": False, "no_such_check": False}
    assert len(data.warnings) == 2
    assert len(events(lab, "validation.override_rejected")) == 2
    with pytest.raises(InvalidArgument):
        prepare(lab, key="dupe", overrides=[{"check": "a", "reason": "xyz"}, {"check": "a", "reason": "xyz"}])


def test_unregistered_cluster_is_invalid_plan(lab: Lab) -> None:
    data = prepare(lab, cluster_id="nowhere")
    assert data.status == "invalid"
    f = findings(data)
    assert f["cluster_registered"].status == "fail"
    assert f["partition_exists"].status == "unchecked"
    assert f["env_guard_static"].status == "unchecked"
    assert all(r.exec_plan_hash is None for r in data.runs)


def test_interactive_partition_blocks(lab: Lab) -> None:
    spec = lab.spec()
    spec["resources"]["partition"] = "interactive"
    data = prepare(lab, spec=spec)
    assert data.status == "invalid"
    assert "interactive_partition_batch" in data.validation.blocking


def test_preemptible_partition_requeue_rules(lab: Lab) -> None:
    spec = lab.spec(smoke=None)
    spec["resources"].update({"partition": "scavenge", "gpu_type": "h100"})
    data = prepare(lab, spec=spec)
    assert "preemptible_gate" in data.validation.blocking
    spec["resilience"] = {"resumable": True, "stop_behavior": "checkpoint_and_exit", "lead_s": 300}
    ok = prepare(lab, key="k2", spec=spec)
    assert ok.status == "ready"
    row = lab.ctx.db.get("runs", "run_id", ok.runs[0].run_id or "")
    ep = ExecutionPlan.model_validate(
        jload(lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])["plan_json"])
    )
    assert ep.requeue is True and ep.resilience and ep.resilience.stop_behavior == "checkpoint_and_exit"
    spec["resilience"] = {"resumable": True, "requeue_on_preemption": False}
    no = prepare(lab, key="k3", spec=spec)
    row = lab.ctx.db.get("runs", "run_id", no.runs[0].run_id or "")
    ep = ExecutionPlan.model_validate(
        jload(lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])["plan_json"])
    )
    assert ep.requeue is False


def test_smoke_shorter_than_signal_lead_drops_resilience(lab: Lab) -> None:
    spec = lab.spec(resilience={"stop_behavior": "checkpoint_and_exit", "lead_s": 1200, "resumable": True})
    data = prepare(lab, spec=spec)
    assert data.status == "ready"
    smoke_row = lab.ctx.db.get("runs", "run_id", data.smoke_run_id or "")
    ep = ExecutionPlan.model_validate(
        jload(
            lab.ctx.db.get("execution_plans", "exec_plan_hash", smoke_row["current_exec_plan_hash"])[
                "plan_json"
            ]
        )
    )
    assert ep.resilience is None
    assert findings(data)["walltime_signal"].status == "pass"


def test_entrypoint_missing_from_snapshot_blocks(lab: Lab) -> None:
    (lab.project / "train.py").unlink()
    data = prepare(lab)
    assert data.status == "invalid" and "entrypoint_in_snapshot" in data.validation.blocking


def test_smoke_cell_not_in_matrix_blocks(lab: Lab) -> None:
    spec = lab.spec(smoke={"cell": {"method": "nope"}, "require_outputs": ["metrics"]})
    data = prepare(lab, spec=spec)
    assert data.status == "invalid" and "smoke_gate" in data.validation.blocking


def test_no_smoke_means_no_gate(lab: Lab) -> None:
    data = prepare(lab, spec=lab.spec(smoke=None))
    assert data.status == "ready" and data.smoke_run_id is None and data.runs_total == 6
    row = lab.ctx.db.get("runs", "run_id", data.runs[0].run_id or "")
    ep = jload(
        lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])["plan_json"]
    )
    assert ep["gate"] is None


def test_script_render_failure_is_a_blocking_finding(lab: Lab) -> None:
    spec = lab.spec(runtime={"python": "python3", "env": {"IFS": "x"}})
    data = prepare(lab, spec=spec)
    assert data.status == "invalid" and "script_render" in data.validation.blocking


# ---------------------------------------------------------------------------
# Background snapshot
# ---------------------------------------------------------------------------


def run_outbox(lab: Lab) -> int:
    runner = OutboxRunner(lab.ctx)
    runner.register_all(ex.outbox_handlers())
    return runner.drain()


def test_large_snapshot_prepares_in_background(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab)
    assert data.status == "preparing" and data.plan_hash is None and data.smoke_run_id is None
    assert data.operation_ref == f"hpclab://operations/{data.operation_id}"
    assert data.validation.status == "pending" and data.snapshot and data.snapshot.mode == "background"
    assert data.coverage.notes and {x.name: x.status for x in data.coverage.sources}["snapshot"] == "skipped"
    assert all(r.run_id is None for r in data.runs) and data.runs_total == 7
    assert data.runs[0].kind == "smoke" and data.budget.smoke_gpu_hours == pytest.approx(0.25)
    assert count(lab, "runs") == 0
    assert lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)["status"] == "preparing"
    op = lab.ctx.db.get("operations", "operation_id", data.operation_id or "")
    assert op["kind"] == "prepare" and op["state"] == "queued" and jload(op["progress_json"]) == {}
    item = lab.ctx.db.one("SELECT * FROM outbox WHERE kind='prepare.snapshot'")
    assert item is not None and jload(item["payload_json"]) == {"plan_id": data.plan_id}
    # retry with the same key: same receipt, nothing new
    assert prepare(lab).model_dump() == data.model_dump()
    with pytest.raises(NotReady):
        ex.plan_script(lab.ctx, plan_id=data.plan_id)
    pg = ex.plan_get(lab.ctx, plan_id=data.plan_id)
    assert pg.status == "preparing" and pg.planned_runs and pg.coverage.notes
    assert run_outbox(lab) == 1
    plan = lab.ctx.db.get("plans", "plan_id", data.plan_id)
    assert plan["status"] == "ready" and plan["plan_hash"] and plan["smoke_run_id"]
    assert count(lab, "runs", "plan_id=?", (data.plan_id,)) == 7
    assert lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)["status"] == "ready"
    op = lab.ctx.db.get("operations", "operation_id", data.operation_id or "")
    assert op["state"] == "succeeded" and jload(op["result_json"])["status"] == "ready"
    # re-running the handler is a no-op (crash-safe)
    from hpclab.services.outbox import OutboxItem

    res = ex.handle_prepare_snapshot(
        lab.ctx,
        OutboxItem(
            outbox_id=0,
            kind="prepare.snapshot",
            operation_id=data.operation_id,
            target_id=data.plan_id,
            payload={"plan_id": data.plan_id},
            tries=1,
        ),
    )
    assert res is not None and res.note == "already finished"
    assert count(lab, "runs") == 7
    full = ex.plan_get(lab.ctx, plan_id=data.plan_id)
    assert full.validation.status in ("pass", "warn") and full.runs_total == 7


def test_background_failure_marks_plan_invalid(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab)
    import shutil

    shutil.rmtree(lab.project)
    run_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", data.plan_id)
    assert plan["status"] == "invalid"
    doc = jload(plan["validation_json"])
    assert doc["status"] == "fail" and doc["findings"][0]["check"] == "code_snapshot"
    assert lab.ctx.db.get("experiments", "experiment_id", data.experiment_id)["status"] == "invalid"
    op = lab.ctx.db.get("operations", "operation_id", data.operation_id or "")
    assert op["state"] == "failed" and jload(op["error_json"])["message"]
    assert count(lab, "runs") == 0


def test_background_invalid_plan_from_validation(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    data = prepare(lab, spec=lab.spec(budget={"gpu_hours": 1}))
    run_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", data.plan_id)
    assert plan["status"] == "invalid" and plan["plan_hash"] is None
    op = lab.ctx.db.get("operations", "operation_id", data.operation_id or "")
    assert op["state"] == "succeeded" and jload(op["result_json"])["blocking"] == ["budget"]


def test_outbox_and_periodic_exports() -> None:
    assert set(ex.outbox_handlers()) == {"prepare.snapshot"}
    assert ex.periodic_tasks() == []


# ---------------------------------------------------------------------------
# extends
# ---------------------------------------------------------------------------


def test_extends_adds_only_new_seed_cells(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1, 2]))
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2, 3])
    assert ext.status == "ready" and ext.experiment_id == base.experiment_id and ext.plan_id != base.plan_id
    main = [r for r in ext.runs if r.kind == "main"]
    assert [r.label for r in main] == ["method-baseline.seed-3", "method-mixup.seed-3"]
    assert all(not r.added_post_hoc for r in ext.runs)
    smoke = [r for r in ext.runs if r.kind == "smoke"]
    assert len(smoke) == 1 and smoke[0].cell == {"method": "baseline", "seed": 3}
    keys = [
        r["cell_key"]
        for r in lab.ctx.db.all(
            "SELECT cell_key FROM runs WHERE experiment_id=? AND kind='main'", (base.experiment_id,)
        )
    ]
    assert len(keys) == len(set(keys)) == 6
    exp = lab.ctx.db.get("experiments", "experiment_id", base.experiment_id)
    assert jload(exp["spec_json"])["seeds"] == [1, 2, 3] and exp["version"] == 2
    ev = events(lab, "experiment.extended")
    assert len(ev) == 1 and jload(ev[0]["payload_json"])["added"]["seeds"] == [3]
    assert findings(ext)["code_drift"].status == "pass"
    assert ext.budget.existing_planned_gpu_hours == pytest.approx(4.25)


def test_extends_add_matrix_value(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None))
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_matrix={"method": ["cutmix", "mixup"]})
    assert [r.label for r in ext.runs] == [f"method-cutmix.seed-{s}" for s in (1, 2, 3)]


def test_extends_errors(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1, 2]))
    with pytest.raises(InvalidArgument, match="nothing to add"):
        prepare(lab, key="e1", extends=base.experiment_id, add_seeds=[1])
    with pytest.raises(InvalidArgument, match="not an axis"):
        prepare(lab, key="e2", extends=base.experiment_id, add_matrix={"lr": [0.1]})
    with pytest.raises(InvalidArgument):
        prepare(lab, key="e3", extends=base.experiment_id)
    with pytest.raises(InvalidArgument):
        prepare(lab, key="e4", extends=base.experiment_id, add_seeds=[5], spec=lab.spec())
    with pytest.raises(InvalidArgument):
        prepare(lab, key="e5", add_seeds=[5])
    with pytest.raises(NotFound):
        prepare(lab, key="e6", extends="exp_00000000000000000000000000", add_seeds=[5])
    with pytest.raises(LimitExceeded):
        prepare(lab, key="e7", extends=base.experiment_id, add_seeds=list(range(10, 140)))


def test_extends_reports_code_drift_with_changed_files(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1]))
    (lab.project / "train.py").write_text("print('changed')\n")
    (lab.project / "new_module.py").write_text("x = 1\n")
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    f = findings(ext)["code_drift"]
    assert f.status == "warn" and f.severity == "warn"
    diff = next(iter(f.evidence["diff"].values()))
    assert diff["changed"] == ["train.py"] and diff["added"] == ["new_module.py"]
    assert ext.status == "ready"


def test_extends_after_evaluations_flags_post_hoc(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1]))
    run_id = next(r.run_id for r in base.runs if r.kind == "main")
    with lab.ctx.db.tx() as tx:
        ep = tx.one("SELECT current_exec_plan_hash AS h FROM runs WHERE run_id=?", (run_id,))
        att = make_attempt(
            tx,
            run_id or "",
            exec_plan_hash=ep["h"] if ep else None,
            phase="terminal",
            scheduler_state="COMPLETED",
            result_state="valid",
            cluster_id="lab",
        )
        make_evaluation(tx, att)
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    assert all(r.added_post_hoc for r in ext.runs)
    rows = lab.ctx.db.all("SELECT added_post_hoc FROM runs WHERE plan_id=?", (ext.plan_id,))
    assert rows and all(r["added_post_hoc"] == 1 for r in rows)
    assert findings(ext)["post_hoc"].status == "warn"


def test_extends_while_preparing_is_not_ready(lab: Lab) -> None:
    lab.ctx.settings.prepare_sync_max_bytes = 10
    base = prepare(lab)
    with pytest.raises(NotReady):
        prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[9])
    run_outbox(lab)
    lab.ctx.settings.prepare_sync_max_bytes = 1 << 30
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[9])
    assert ext.status == "ready"


def test_extends_invalid_experiment_refused(lab: Lab) -> None:
    bad = prepare(lab, spec=lab.spec(budget={"gpu_hours": 1}))
    with pytest.raises(PreconditionFailed):
        prepare(lab, key="ext", extends=bad.experiment_id, add_seeds=[9])


def test_extends_in_background(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(seeds=[1]))
    lab.ctx.settings.prepare_sync_max_bytes = 10
    ext = prepare(lab, key="ext", extends=base.experiment_id, add_seeds=[2])
    assert ext.status == "preparing" and [r.label for r in ext.runs] == [
        "smoke.method-baseline.seed-2",
        "method-baseline.seed-2",
        "method-mixup.seed-2",
    ]
    run_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", ext.plan_id)
    assert plan["status"] == "ready"
    exp = lab.ctx.db.get("experiments", "experiment_id", base.experiment_id)
    assert exp["status"] == "ready" and jload(exp["spec_json"])["seeds"] == [1, 2]


# ---------------------------------------------------------------------------
# derive_from
# ---------------------------------------------------------------------------


def test_derive_from_creates_new_run_with_lineage(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(parameters={"lr": 0.1}))
    src = next(r for r in base.runs if r.label == "method-mixup.seed-1")
    der = prepare(
        lab,
        key="d1",
        derive_from=ex.DeriveFrom(
            run_id=src.run_id or "",
            overrides=ex.DeriveOverrides(
                params={"lr": 0.05}, resources={"time": "02:00:00"}, argv_append=["--lr", "{lr}"]
            ),
        ),
    )
    assert der.status == "ready" and der.kind == "derived" and der.runs_total == 1
    run = der.runs[0]
    assert run.derived_from == src.run_id and run.label.startswith("method-mixup.seed-1.derived-d")
    assert run.cell["method"] == "mixup" and run.cell["seed"] == 1 and run.cell["variant"].startswith("d")
    row = lab.ctx.db.get("runs", "run_id", run.run_id or "")
    rs = RunSpec.model_validate(jload(row["spec_json"]))
    assert rs.derived_from == src.run_id and rs.params["lr"] == 0.05
    assert rs.argv[-2:] == ["--lr", "0.05"]
    assert row["derived_from"] == src.run_id
    ep = ExecutionPlan.model_validate(
        jload(lab.ctx.db.get("execution_plans", "exec_plan_hash", row["current_exec_plan_hash"])["plan_json"])
    )
    assert ep.resources.time == "02:00:00" and ep.gate is None
    assert der.snapshot and der.snapshot.mode == "reused"
    src_spec = RunSpec.model_validate(jload(lab.ctx.db.get("runs", "run_id", src.run_id or "")["spec_json"]))
    assert rs.code == src_spec.code
    assert lab.ctx.db.get("plans", "plan_id", der.plan_id)["kind"] == "derived"


def test_derive_seed_override_and_recapture_drift(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None))
    src = base.runs[0]
    (lab.project / "train.py").write_text("print('v2')\n")
    der = prepare(
        lab,
        key="d1",
        derive_from={"run_id": src.run_id, "overrides": {"params": {"seed": 99}}, "recapture_code": True},
    )
    assert der.runs[0].seed == 99 and der.runs[0].cell["seed"] == 99
    assert der.snapshot and der.snapshot.mode == "captured"
    assert findings(der)["code_drift"].status == "warn"


def test_derive_with_recapture_in_background(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None))
    src = base.runs[0]
    lab.ctx.settings.prepare_sync_max_bytes = 10
    der = prepare(
        lab,
        key="d1",
        derive_from={
            "run_id": src.run_id,
            "overrides": {"resources": {"time": "03:00:00"}},
            "recapture_code": True,
        },
    )
    assert der.status == "preparing" and der.runs_total == 1
    assert der.runs[0].derived_from == src.run_id and der.runs[0].requested_gpu_hours == pytest.approx(3.0)
    run_outbox(lab)
    plan = lab.ctx.db.get("plans", "plan_id", der.plan_id)
    assert plan["status"] == "ready" and plan["kind"] == "derived"
    rows = lab.ctx.db.all("SELECT derived_from FROM runs WHERE plan_id=?", (der.plan_id,))
    assert [r["derived_from"] for r in rows] == [src.run_id]


def test_derive_errors(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None))
    with pytest.raises(NotFound):
        prepare(lab, key="d1", derive_from={"run_id": "run_00000000000000000000000000"})
    with pytest.raises(InvalidArgument):
        prepare(
            lab,
            key="d2",
            derive_from={"run_id": base.runs[0].run_id, "overrides": {"resources": {"gpus": -1}}},
        )
    with pytest.raises(InvalidArgument):
        prepare(lab, key="d3", derive_from={"run_id": base.runs[0].run_id}, extends=base.experiment_id)
    with pytest.raises(InvalidArgument):
        prepare(
            lab, key="d4", derive_from={"run_id": base.runs[0].run_id, "overrides": {"params": {"seed": "x"}}}
        )


def test_derive_needs_source_snapshot_in_store(lab: Lab) -> None:
    base = prepare(lab, spec=lab.spec(smoke=None))
    import shutil

    shutil.rmtree(lab.ctx.paths.snapshots / "manifests")
    with pytest.raises(PreconditionFailed, match="recapture_code"):
        prepare(lab, key="d1", derive_from={"run_id": base.runs[0].run_id})


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def test_experiments_list_and_get(lab: Lab) -> None:
    a = prepare(lab, key="a", experiment="alpha")
    b = prepare(lab, key="b", experiment="beta-run")
    lab.clock.advance(1)
    c = prepare(lab, key="c", experiment="beta-two")
    page = ex.experiments_list(lab.ctx, limit=2)
    assert page.experiments[0].experiment_id == c.experiment_id
    assert page.next_cursor is not None and len(page.experiments) == 2
    rest = ex.experiments_list(lab.ctx, limit=2, cursor=page.next_cursor)
    ids = {e.experiment_id for e in page.experiments} | {e.experiment_id for e in rest.experiments}
    assert ids == {a.experiment_id, b.experiment_id, c.experiment_id} and rest.next_cursor is None
    assert {e.name for e in ex.experiments_list(lab.ctx, name="beta").experiments} == {"beta-run", "beta-two"}
    assert ex.experiments_list(lab.ctx, name="%").experiments == []
    got = ex.experiment_get(lab.ctx, experiment_id=a.experiment_id)
    assert got.name == "alpha" and got.plans_total == 1 and got.plans[0].status == "ready"
    assert got.run_counts["planned"] == 6 and got.run_counts.get("P") == 6
    assert got.runs_total == 7 and got.planned_gpu_hours == pytest.approx(6.25)
    assert got.budget is not None and got.budget["budget"] == 50  # services.budget.summary
    with pytest.raises(NotFound):
        ex.experiment_get(lab.ctx, experiment_id="exp_00000000000000000000000000")


def test_experiment_get_budget_unavailable_is_reported(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import hpclab.services as services_pkg

    monkeypatch.setitem(sys.modules, "hpclab.services.budget", None)  # import -> ImportError
    monkeypatch.delattr(services_pkg, "budget", raising=False)
    a = prepare(lab)
    got = ex.experiment_get(lab.ctx, experiment_id=a.experiment_id)
    assert got.budget is None and got.coverage.complete is False
    assert {s.name: s.status for s in got.coverage.sources}["budget"] == "unsupported"


def test_experiment_get_uses_budget_summary_when_available(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from dataclasses import dataclass

    @dataclass
    class Summary:
        budget: float | None
        reserved: float

    mod = types.ModuleType("hpclab.services.budget")
    mod.summary = lambda reader, *, experiment_id: Summary(budget=50.0, reserved=1.5)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hpclab.services.budget", mod)
    import hpclab.services as services_pkg

    monkeypatch.setattr(services_pkg, "budget", mod, raising=False)
    a = prepare(lab)
    got = ex.experiment_get(lab.ctx, experiment_id=a.experiment_id)
    assert got.budget == {"budget": 50.0, "reserved": 1.5} and got.coverage.complete is True


def test_plan_get_pages_runs(lab: Lab) -> None:
    data = prepare(lab)
    first = ex.plan_get(lab.ctx, plan_id=data.plan_id, limit=3)
    assert first.runs_total == 7 and len(first.runs) == 3 and first.next_cursor
    assert first.runs[0].kind == "smoke" and first.runs[0].execution_plan
    assert first.runs[0].exec_plan_hash == first.runs[0].current_exec_plan_hash
    labels = [r.label for r in first.runs]
    cursor: str | None = first.next_cursor
    while cursor:
        page = ex.plan_get(lab.ctx, plan_id=data.plan_id, limit=3, cursor=cursor)
        labels += [r.label for r in page.runs]
        cursor = page.next_cursor
    assert labels == [r.label for r in data.runs]
    assert first.validation.status == data.validation.status
    assert first.budget and first.resource_advice and first.snapshot
    with pytest.raises(NotFound):
        ex.plan_get(lab.ctx, plan_id="pln_00000000000000000000000000")


def test_plan_get_tolerates_foreign_plan_documents(lab: Lab) -> None:
    from hpclab.testing.factories import make_experiment, make_plan

    with lab.ctx.db.tx() as tx:
        eid = make_experiment(tx, cluster_id="lab")
        pid = make_plan(tx, eid, kind="recovery")
        tx.execute(
            "UPDATE plans SET validation_json=? WHERE plan_id=?",
            ('{"cells":[{"run_id":"run_x","eligible":false}],"findings":[{"odd":1}],"status":"ready"}', pid),
        )
    pg = ex.plan_get(lab.ctx, plan_id=pid)
    assert pg.kind == "recovery" and pg.validation.status == "pending" and pg.validation.findings == []
    assert pg.details["cells"][0]["run_id"] == "run_x" and pg.runs == []


def test_plan_script_views(lab: Lab) -> None:
    data = prepare(lab)
    full = ex.plan_script(lab.ctx, plan_id=data.plan_id)
    main_first = next(r for r in data.runs if r.kind == "main")
    assert full.run_id == main_first.run_id and full.view == "full"
    assert not full.truncated and full.next_cursor is None  # the whole script fits in one 64 KiB page
    assert len(full.script.encode()) == full.script_bytes
    from hpclab.domain.hashing import sha256_hex

    assert "sha256:" + sha256_hex(full.script) == full.script_digest
    assert full.script.startswith("#!/bin/bash\n#SBATCH --job-name=hl.method-baseline.seed-1.previewt")
    assert "--job-name=hl.method-baseline.seed-1.previewt" in full.directives
    assert any(d.startswith("--gres=gpu:a100:1") for d in full.directives)
    assert "--account=lab-acct" in full.directives and "--partition=gpu" in full.directives
    assert full.attempt_id_placeholder in full.script
    gen = ex.plan_script(lab.ctx, plan_id=data.plan_id, view="generated")
    assert "static wrapper runtime" in gen.script and len(gen.script) < full.script_bytes // 4
    paged = ""
    cursor = None
    while True:
        page = ex.plan_script(lab.ctx, plan_id=data.plan_id, cursor=cursor, max_bytes=20000)
        paged += page.script
        cursor = page.next_cursor
        if cursor is None:
            break
        assert page.truncated
    assert paged == full.script
    smoke = ex.plan_script(lab.ctx, plan_id=data.plan_id, run_id=data.smoke_run_id)
    assert "--time=00:15:00" in smoke.directives
    with pytest.raises(NotFound):
        ex.plan_script(lab.ctx, plan_id=data.plan_id, run_id="run_00000000000000000000000000")


def test_plan_script_default_pages_end_at_line_boundaries(lab: Lab) -> None:
    """A script larger than one page (a long setup, many env vars) is paged at line boundaries by
    default: every page is whole lines and the pages concatenate to the exact script."""
    from hpclab.domain.hashing import sha256_hex

    setup = [f"export STEP_{i}=" + "x" * 180 for i in range(60)]  # ~11 KB of run-specific script
    data = prepare(lab, spec=lab.spec(runtime={"python": "python3", "setup": setup}))
    assert data.status == "ready", data.validation.blocking
    pages: list[ex.PlanScriptData] = []
    cursor = None
    while True:
        page = ex.plan_script(lab.ctx, plan_id=data.plan_id, cursor=cursor)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert len(pages) == 2 and pages[0].truncated and not pages[1].truncated
    first = pages[0].script.encode()
    assert len(first) <= 64 * 1024 and first.endswith(b"\n")
    assert any("line boundary" in w and "next_cursor" in w for w in pages[0].warnings)
    assert pages[1].offset == len(first)
    whole = "".join(p.script for p in pages)
    assert len(whole.encode()) == pages[0].script_bytes
    assert "sha256:" + sha256_hex(whole) == pages[0].script_digest
    # an explicit max_bytes still pages byte-exactly
    exact = ex.plan_script(lab.ctx, plan_id=data.plan_id, max_bytes=1000)
    assert exact.script.encode() == whole.encode()[:1000]


def test_plan_script_resource_route_reads_script_field(lab: Lab) -> None:
    from hpclab.mcp.server import RESOURCES

    route = next(r for r in RESOURCES if r.method == "plan_script")
    data = prepare(lab)
    out = ex.plan_script(lab.ctx, **route.params({"plan_id": data.plan_id}))
    assert route.text_field is not None
    assert getattr(out, route.text_field).startswith("#!/bin/bash")


def test_registry_derives_params_from_signatures(lab: Lab) -> None:
    reg = Registry()
    reg.register("experiment_prepare", ex.experiment_prepare, mutation=True)
    reg.register("experiments_list", ex.experiments_list)
    reg.register("experiment_get", ex.experiment_get)
    reg.register("plan_get", ex.plan_get)
    reg.register("plan_script", ex.plan_script)
    env = reg.dispatch(
        lab.ctx,
        "experiment_prepare",
        {
            "idempotency_key": "rk",
            "spec": lab.spec(),
            "validate_remote": False,
            "overrides": [{"check": "budget", "reason": "fine"}],
        },
        request_id="r1",
    )
    assert env["ok"] is True, env
    assert env["data"]["status"] == "ready" and env["summary"].startswith("plan ")
    assert env["warnings"]  # the unused override
    assert env["coverage"]["sources"][0]["name"] == "ledger" and env["observed_at"]
    bad = reg.dispatch(
        lab.ctx, "experiment_prepare", {"idempotency_key": "x", "derive_from": {"nope": 1}}, request_id="r2"
    )
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_argument"
    lst = reg.dispatch(lab.ctx, "experiments_list", {}, request_id="r3")
    assert lst["ok"] and lst["coverage"]["complete"] is True
    plan = reg.dispatch(lab.ctx, "plan_get", {"plan_id": env["data"]["plan_id"]}, request_id="r4")
    assert plan["ok"] and plan["data"]["runs_total"] == 7
