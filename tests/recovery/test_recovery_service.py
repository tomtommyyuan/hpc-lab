"""run_recover for single attempts: plans, idempotency, node exclusion, resource patches."""

from __future__ import annotations

from datetime import timedelta

import pytest
from wf_support import IMPORT_ERR, FakeSubmission, LabBuilder, one

from hpclab.domain.clock import FakeClock, iso
from hpclab.domain.errors import IdempotencyConflict, InvalidArgument, NotFound
from hpclab.domain.models import ExecutionPlan, ResilienceSpec
from hpclab.services import recovery as rec
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jload


def _one_attempt(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, **matrix_kw):
    m = lab.matrix(n=1, **matrix_kw)
    rid = m.run_ids[0]
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=rid, exec_plan_hash=m.exec_plans[rid])
    return m, rid


def _plan(ctx: ServiceContext, h: str) -> ExecutionPlan:
    return ExecutionPlan.model_validate(
        jload(one(ctx, "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (h,))["plan_json"])
    )


def _counts(ctx: ServiceContext) -> tuple[int, int]:
    return (
        one(ctx, "SELECT COUNT(*) AS n FROM plans WHERE kind IN ('recovery','probe')")["n"],
        one(ctx, "SELECT COUNT(*) AS n FROM execution_plans")["n"],
    )


def test_retry_after_node_failure_builds_a_ready_plan_for_the_same_run(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, state="NODE_FAIL", exit_code=None, node_list="gpu-7")
    data = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert data.eligible and data.mode == "single" and data.strategy == "retry" and data.cells is None
    assert (
        data.diagnosis is not None
        and data.diagnosis.primary_code == "node_fail"
        and data.diagnosis.source == "stored"
    )
    assert data.plan_ref == f"hpclab://plans/{data.plan_id}" and data.exec_plan_hash
    assert data.next_step and "run_submit" in data.next_step
    plan = _plan(ctx, data.exec_plan_hash)
    old = _plan(ctx, m.exec_plans[rid])
    assert plan.run_id == rid and plan.run_spec_hash == old.run_spec_hash  # same Run, same science
    assert (
        plan.created_reason == "retry" and plan.recovery is not None and plan.recovery.from_attempt_id == aid
    )
    assert plan.resources == old.resources  # retry = same resources (single node obs -> no exclusion)
    assert "gpu-7" in data.suspect_nodes and data.exclude_nodes == []
    row = one(ctx, "SELECT * FROM plans WHERE plan_id=?", (data.plan_id,))
    assert row["kind"] == "recovery" and row["status"] == "ready"
    cells = rec.recovery_plan_cells(row)
    assert (
        cells[0]["run_id"] == rid
        and cells[0]["eligible"]
        and cells[0]["exec_plan_hash"] == data.exec_plan_hash
    )
    # the run now points at its new ready plan (same tx, audited); nothing was submitted
    assert (
        one(ctx, "SELECT current_exec_plan_hash FROM runs WHERE run_id=?", (rid,))["current_exec_plan_hash"]
        == data.exec_plan_hash
    )
    changed = ctx.db.all(
        "SELECT payload_json FROM events WHERE entity_type='run' AND entity_id=? AND event_type='run.exec_plan_changed'",
        (rid,),
    )
    assert len(changed) == 1
    payload = jload(changed[0]["payload_json"])
    assert payload["from"] == m.exec_plans[rid] and payload["to"] == data.exec_plan_hash
    assert payload["from_attempt_id"] == aid and payload["plan_id"] == data.plan_id
    assert data.next_step == (
        f'run_submit(run_id="{rid}", execution_plan_hash="{data.exec_plan_hash}", idempotency_key="<new key>")'
    )
    assert (
        one(
            ctx,
            "SELECT consumed_by_attempt_id FROM execution_plans WHERE exec_plan_hash=?",
            (data.exec_plan_hash,),
        )["consumed_by_attempt_id"]
        is None
    )
    assert len(sub.admitted) == 1
    # idempotency
    assert rec.run_recover(ctx, idempotency_key="k", attempt_id=aid) == data
    with pytest.raises(IdempotencyConflict):
        rec.run_recover(ctx, idempotency_key="k", attempt_id=aid, strategy="adjust_resources")
    # the recovery plan is admissible exactly once through the normal gate
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=rid, exec_plan_hash=data.exec_plan_hash)
    again = rec.run_recover(ctx, idempotency_key="k2", attempt_id=aid)
    assert not again.eligible and any("newer attempt" in b for b in again.blocking)


def test_ineligible_attempts_write_nothing(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.attempt_of(rid)["attempt_id"]
    before = _counts(ctx)
    lab.set_attempt(aid, phase="submission_uncertain")
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert not d.eligible and d.exec_plan_hash is None and any("unresolved" in b for b in d.blocking)
    lab.set_attempt(aid, phase="duplicate_detected")
    assert not rec.run_recover(ctx, idempotency_key="k", attempt_id=aid).eligible
    lab.set_attempt(aid, phase="job_identified")
    lab.set_attempt(aid, phase="observing", scheduler_state="RUNNING")
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert not d.eligible and any("still active" in b for b in d.blocking)
    assert _counts(ctx) == before
    # the same key re-evaluates once the attempt has stopped (nothing was stored)
    lab.fail(rid, state="PREEMPTED", exit_code=None)
    assert rec.run_recover(ctx, idempotency_key="k", attempt_id=aid).eligible
    with pytest.raises(NotFound):
        rec.run_recover(ctx, idempotency_key="z", attempt_id="att_nope")
    with pytest.raises(InvalidArgument):
        rec.run_recover(ctx, idempotency_key="z")


def test_nan_loss_is_a_derived_run_not_a_retry(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(
        rid, stderr="step 100 loss nan\nstep 110 loss nan\nRuntimeError: Loss is NaN at step 110\n"
    )
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid, strategy="retry")
    assert not d.eligible and d.diagnosis is not None and d.diagnosis.primary_code == "nan_loss"
    assert any(h.strategy == "derive_run" and h.tool == "experiment_prepare" for h in d.hints)


def test_environment_failure_is_retryable_with_a_warning(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, stderr=IMPORT_ERR)
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert d.eligible and d.strategy == "retry" and d.warnings
    assert any(h.strategy == "fix_environment" for h in d.hints)


def _observe_node(
    ctx: ServiceContext,
    node: str,
    attempt: str,
    *,
    age_h: float = 1.0,
    expires_h: float | None = 71.0,
    cls: str = "node_fail",
) -> None:
    now = ctx.clock.now()
    with ctx.db.tx() as tx:
        tx.insert(
            "observations",
            {
                "cluster_id": "sim",
                "obs_key": f"node:{node}:{cls}:{attempt}",
                "value_json": "{}",
                "source": "observe",
                "observed_at": iso(now - timedelta(hours=age_h)),
                "expires_at": iso(now + timedelta(hours=expires_h)) if expires_h is not None else None,
            },
        )


def test_node_exclusion_needs_two_distinct_attempts_within_ttl(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, clock: FakeClock
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, state="NODE_FAIL", exit_code=None, node_list="gpu-1")
    _observe_node(ctx, "gpu-1", "att_a")
    _observe_node(ctx, "gpu-1", "att_a", cls="nfs_stale")  # same attempt twice (another class) counts once
    _observe_node(ctx, "gpu-2", "att_b")
    _observe_node(ctx, "gpu-2", "att_c")
    _observe_node(ctx, "gpu-3", "att_d", expires_h=-1)  # expired
    _observe_node(ctx, "gpu-3", "att_e", expires_h=-1)
    _observe_node(ctx, "gpu-4", "att_f", age_h=100, expires_h=None)  # no expiry, older than the 72 h TTL
    _observe_node(ctx, "gpu-4", "att_g", age_h=100, expires_h=None)
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert d.eligible and d.exclude_nodes == ["gpu-2"]
    assert "gpu-1" in d.suspect_nodes
    assert d.resource_patch == {"exclude_nodes": ["gpu-2"]}
    assert any(c.check == "node_exclusion" and "gpu-2 (2 attempts)" in c.message for c in d.checks)
    assert _plan(ctx, d.exec_plan_hash or "").resources.exclude_nodes == ["gpu-2"]


def test_suspect_node_observations_are_reported_never_excluded(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """A multi-node failure cannot say WHICH node failed (``*_suspect`` classes): even two such
    attempts sharing nodes are not evidence of a failure *on* a node, so nothing is excluded."""
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, state="NODE_FAIL", exit_code=None, node_list="gpu-[5-6]")
    for att in ("att_a", "att_b"):
        _observe_node(ctx, "gpu-5", att, cls="node_fail_suspect")
        _observe_node(ctx, "gpu-6", att, cls="nfs_suspect")
    _observe_node(ctx, "gpu-9", "att_c")  # a definite single-node failure, twice
    _observe_node(ctx, "gpu-9", "att_d")
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert d.eligible and d.exclude_nodes == ["gpu-9"]
    assert {"gpu-5", "gpu-6"} <= set(d.suspect_nodes) and "gpu-9" not in d.suspect_nodes


def test_adjust_resources_after_timeout_uses_the_hint_and_partition_limits(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, state="TIMEOUT", exit_code=None, elapsed_s=3600, time_limit_s=3600)
    auto = rec.run_recover(ctx, idempotency_key="k", attempt_id=aid)
    assert auto.eligible and auto.strategy == "adjust_resources"
    assert (
        auto.resource_patch["time"] == "01:30:00" and auto.resources and auto.resources["time"] == "01:30:00"
    )
    assert "extra_sbatch" not in auto.resource_patch  # --signal is never appended blindly
    assert any(c.check == "partition_limits" and c.status == "pass" for c in auto.checks)
    too_long = rec.run_recover(
        ctx, idempotency_key="k2", attempt_id=aid, resource_patch=rec.ResourcePatch(time="5-00:00:00")
    )
    assert not too_long.eligible and any("time_within_partition_max" in b for b in too_long.blocking)
    shorter = rec.run_recover(
        ctx, idempotency_key="k3", attempt_id=aid, resource_patch=rec.ResourcePatch(time="00:30:00")
    )
    assert not shorter.eligible and any("must raise resources.time" in b for b in shorter.blocking)
    retry = rec.run_recover(ctx, idempotency_key="k4", attempt_id=aid, strategy="retry")
    assert not retry.eligible and any("times out again" in b for b in retry.blocking)
    with pytest.raises(InvalidArgument):
        rec.run_recover(
            ctx,
            idempotency_key="k5",
            attempt_id=aid,
            strategy="retry",
            resource_patch=rec.ResourcePatch(time="02:00:00"),
        )
    with pytest.raises(InvalidArgument):
        rec.run_recover(
            ctx, idempotency_key="k6", attempt_id=aid, resource_patch=rec.ResourcePatch(time="bogus")
        )


def test_gpu_count_changes_need_batch_invariance(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    oom = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 39.39 GiB\n"
    aid = lab.fail(rid, stderr=oom, gpus=1)
    more_gpus = rec.run_recover(
        ctx, idempotency_key="a", attempt_id=aid, resource_patch=rec.ResourcePatch(gpus=2)
    )
    assert not more_gpus.eligible and any("batch_invariant" in b for b in more_gpus.blocking)
    assert any(h.strategy == "derive_run" for h in more_gpus.hints)
    mem_only = rec.run_recover(
        ctx, idempotency_key="b", attempt_id=aid, resource_patch=rec.ResourcePatch(mem="64G")
    )
    assert not mem_only.eligible and any("GPU type with more memory" in b for b in mem_only.blocking)
    bigger_gpu = rec.run_recover(
        ctx, idempotency_key="c", attempt_id=aid, resource_patch=rec.ResourcePatch(gpu_type="h100")
    )
    assert bigger_gpu.eligible and bigger_gpu.resources and bigger_gpu.resources["gpu_type"] == "h100"


def test_batch_invariant_program_may_scale_out_after_cuda_oom(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub, resilience=ResilienceSpec(batch_invariant_to_gpus=True))
    oom = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.\n"
    aid = lab.fail(rid, stderr=oom, gpus=1)
    d = rec.run_recover(ctx, idempotency_key="a", attempt_id=aid, resource_patch=rec.ResourcePatch(gpus=2))
    assert d.eligible and d.requested_gpu_hours == pytest.approx(2.0)


def test_host_oom_resume_requires_more_memory(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    _m, rid = _one_attempt(ctx, lab, sub)
    aid = lab.fail(rid, state="OUT_OF_MEMORY", exit_code=None)
    same = rec.run_recover(
        ctx,
        idempotency_key="a",
        attempt_id=aid,
        strategy="adjust_resources",
        resource_patch=rec.ResourcePatch(time="02:00:00"),
    )
    assert not same.eligible and any("raise resources.mem" in b for b in same.blocking)
    assert not any(h.strategy == "derive_run" for h in same.hints)  # more memory is not a new experiment
    more = rec.run_recover(
        ctx, idempotency_key="b", attempt_id=aid, resource_patch=rec.ResourcePatch(mem="96G")
    )
    assert more.eligible and more.resources and more.resources["mem"] == "96G"
