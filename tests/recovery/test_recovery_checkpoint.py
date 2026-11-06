"""resume: checkpoint metadata validation + compute probe flow (local transport, real files)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from ckpt_support import add_artifact, fake_torch, manifest_digest, run_probe, write_checkpoint
from wf_support import FakeSubmission, LabBuilder, one

from hpclab.domain.errors import NotFound
from hpclab.domain.models import ExecutionPlan, ResilienceSpec, RunSpec
from hpclab.services import recovery as rec
from hpclab.services.context import ServiceContext
from hpclab.services.probe import PROBE_OUTPUT_FILE, PROBE_SOURCE
from hpclab.storage.db import jload


@dataclass
class Stopped:
    run_id: str
    attempt_id: str
    attempt_dir: Path
    ckpt_dir: Path
    manifest: dict
    artifact_id: str


def _stopped_with_checkpoint(
    ctx: ServiceContext,
    lab: LabBuilder,
    sub: FakeSubmission,
    *,
    resumable: bool = True,
    config_override: dict | None = None,
    semantic_keys: list[str] | None = None,
    files: dict[str, bytes] | None = None,
    requeue: bool | None = None,
) -> Stopped:
    m = lab.matrix(n=1, resilience=ResilienceSpec(resumable=resumable, stop_behavior="none"))
    rid = m.run_ids[0]
    h = m.exec_plans[rid]
    if requeue is not None:  # the run's own plan asks Slurm to requeue it (--requeue / --no-requeue)
        h = lab.extra_exec_plan(rid, requeue=requeue, created_reason="initial")
    with ctx.db.tx() as tx:
        res = sub.admit(tx, ctx, run_id=rid, exec_plan_hash=h)
    aid = lab.fail(rid, state="TIMEOUT", exit_code=None, elapsed_s=3600, time_limit_s=3600)
    a = one(ctx, "SELECT remote_dir FROM attempts WHERE attempt_id=?", (aid,))
    attempt_dir = Path(a["remote_dir"])
    run = one(ctx, "SELECT spec_json, spec_hash FROM runs WHERE run_id=?", (rid,))
    spec = RunSpec.model_validate(jload(run["spec_json"]))
    config = {**spec.params, "world_size": 1, **(config_override or {})}
    ckpt = attempt_dir / "checkpoints" / "ckpt-000000500"
    manifest = write_checkpoint(
        ckpt,
        files or {"model.pt": b"weights" * 100, "optim.pt": b"optimizer" * 50},
        step=500,
        run_id=rid,
        attempt_id=aid,
        run_spec_hash=run["spec_hash"],
        config=config,
        semantic_keys=semantic_keys if semantic_keys is not None else ["method", "world_size"],
        resumable=True,
    )
    # an older, lower-step checkpoint and a half-written staging dir that must be ignored
    write_checkpoint(
        attempt_dir / "checkpoints" / "ckpt-000000100",
        {"model.pt": b"old"},
        step=100,
        run_id=rid,
        attempt_id=aid,
        run_spec_hash=run["spec_hash"],
        config=config,
        semantic_keys=["method"],
    )
    (attempt_dir / "checkpoints" / ".staging-ckpt-600").mkdir(parents=True)
    art = add_artifact(ctx, attempt_id=aid, run_id=rid, remote_path=str(attempt_dir / "checkpoints"))
    del res
    return Stopped(rid, aid, attempt_dir, ckpt, manifest, art)


def _complete_probe(
    ctx: ServiceContext,
    lab: LabBuilder,
    sub: FakeSubmission,
    probe_run: str,
    exec_plan_hash: str,
    ckpt: Path,
    expected: str,
    *,
    extra_path: Path | None = None,
) -> int:
    with ctx.db.tx() as tx:
        res = sub.admit(tx, ctx, run_id=probe_run, exec_plan_hash=exec_plan_hash)
    a = one(ctx, "SELECT remote_dir FROM attempts WHERE attempt_id=?", (res.attempt_id,))
    rc = run_probe(str(ckpt), expected, Path(a["remote_dir"]) / "outputs", extra_path=extra_path)
    state = "COMPLETED" if rc == 0 else "FAILED"
    lab.set_attempt(
        res.attempt_id, phase="terminal", scheduler_state=state, exit_code=rc, result_state="valid"
    )
    return rc


def test_resume_needs_a_passed_probe_then_yields_a_ready_plan(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub)
    digest = manifest_digest(s.manifest)
    first = rec.run_recover(ctx, idempotency_key="k1", attempt_id=s.attempt_id, strategy="resume")
    assert not first.eligible and first.strategy == "resume"
    assert (
        first.checkpoint is not None and first.checkpoint.step == 500 and first.checkpoint.status == "valid"
    )
    assert first.checkpoint.continuation == "exact" and first.checkpoint.manifest_digest == digest
    assert first.checkpoint.remote_dir == str(s.ckpt_dir)
    assert (
        first.probe is not None and first.probe.created and first.probe.run_id and first.probe.exec_plan_hash
    )
    assert first.next_step and first.probe.run_id in first.next_step
    assert any(c.check == "checkpoint_load_probe" and c.status == "pending" for c in first.checks)
    probe_run = one(ctx, "SELECT * FROM runs WHERE run_id=?", (first.probe.run_id,))
    assert probe_run["kind"] == "probe" and probe_run["experiment_id"] == first.experiment_id
    spec = RunSpec.model_validate(jload(probe_run["spec_json"]))
    assert (
        spec.kind == "probe" and spec.argv[2] == PROBE_SOURCE and spec.argv[3:] == [str(s.ckpt_dir), digest]
    )
    assert spec.params["target_run_id"] == s.run_id and spec.params["checkpoint_artifact_id"] == s.artifact_id
    probe_plan = ExecutionPlan.model_validate(
        jload(
            one(
                ctx,
                "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?",
                (first.probe.exec_plan_hash,),
            )["plan_json"]
        )
    )
    assert probe_plan.resources.gpus == 0 and probe_plan.run_spec_hash == spec.spec_hash()
    assert probe_plan.requeue is False  # a probe only loads a checkpoint: never auto-requeued
    assert one(ctx, "SELECT kind FROM plans WHERE plan_id=?", (first.probe.plan_id,))["kind"] == "probe"
    # asking again before the probe ran reuses the same probe run (no second one)
    second = rec.run_recover(ctx, idempotency_key="k2", attempt_id=s.attempt_id, strategy="resume")
    assert not second.eligible and second.probe and second.probe.run_id == first.probe.run_id
    assert second.probe.state == "not_submitted" and not second.probe.created
    assert one(ctx, "SELECT COUNT(*) AS n FROM runs WHERE kind='probe'")["n"] == 1
    # the probe runs on the "compute node" and passes
    assert (
        _complete_probe(ctx, lab, sub, first.probe.run_id, first.probe.exec_plan_hash, s.ckpt_dir, digest)
        == 0
    )
    probe_json = jload(
        (
            Path(
                one(ctx, "SELECT remote_dir FROM attempts WHERE run_id=?", (first.probe.run_id,))[
                    "remote_dir"
                ]
            )
            / "outputs"
            / PROBE_OUTPUT_FILE
        ).read_text()
    )
    assert probe_json["ok"] is True and probe_json["manifest_digest"] == digest
    third = rec.run_recover(ctx, idempotency_key="k3", attempt_id=s.attempt_id, strategy="resume")
    assert third.eligible and third.exec_plan_hash and third.created_reason == "resume"
    assert any("load not checked" in w or "could not load" in w for w in third.warnings)  # no torch locally
    plan = ExecutionPlan.model_validate(
        jload(
            one(ctx, "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (third.exec_plan_hash,))[
                "plan_json"
            ]
        )
    )
    assert plan.recovery is not None and plan.recovery.strategy == "resume"
    assert plan.recovery.checkpoint_remote_path == str(s.ckpt_dir)
    assert plan.recovery.checkpoint_digest == digest and plan.recovery.checkpoint_artifact_id == s.artifact_id
    assert plan.resources.time == "01:00:00"  # a resume keeps the limit; it continues from step 500


def test_tampered_checkpoint_fails_the_probe(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub)
    digest = manifest_digest(s.manifest)
    first = rec.run_recover(ctx, idempotency_key="k1", attempt_id=s.attempt_id, strategy="resume")
    assert first.probe and first.probe.run_id and first.probe.exec_plan_hash
    model = s.ckpt_dir / "model.pt"
    model.write_bytes(b"X" * model.stat().st_size)  # same size: only the compute-node hash can tell
    rc = _complete_probe(ctx, lab, sub, first.probe.run_id, first.probe.exec_plan_hash, s.ckpt_dir, digest)
    assert rc == 2
    after = rec.run_recover(ctx, idempotency_key="k2", attempt_id=s.attempt_id, strategy="resume")
    assert not after.eligible and after.probe and after.probe.state == "failed"
    assert any("probe" in b and "failed" in b for b in after.blocking)
    assert one(ctx, "SELECT COUNT(*) AS n FROM plans WHERE kind='recovery'")["n"] == 0


def test_probe_with_torch_loads_every_file(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission, tmp_path: Path
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub)
    digest = manifest_digest(s.manifest)
    first = rec.run_recover(ctx, idempotency_key="k1", attempt_id=s.attempt_id, strategy="resume")
    assert first.probe and first.probe.run_id and first.probe.exec_plan_hash
    rc = _complete_probe(
        ctx,
        lab,
        sub,
        first.probe.run_id,
        first.probe.exec_plan_hash,
        s.ckpt_dir,
        digest,
        extra_path=fake_torch(tmp_path / "torchlib"),
    )
    assert rc == 0
    ok = rec.run_recover(ctx, idempotency_key="k2", attempt_id=s.attempt_id, strategy="resume")
    assert ok.eligible and ok.probe and ok.probe.state == "passed" and ok.probe.load_checked
    assert not any("could not load" in w for w in ok.warnings)


def test_incompatible_or_foreign_checkpoints_are_derived_runs(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub, config_override={"method": "other"})
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=s.attempt_id, strategy="resume")
    assert not d.eligible and d.checkpoint and d.checkpoint.continuation == "incompatible"
    assert any("not an exact continuation" in b for b in d.blocking)
    assert any(h.strategy == "derive_run" for h in d.hints)
    assert one(ctx, "SELECT COUNT(*) AS n FROM runs WHERE kind='probe'")["n"] == 0
    other = _stopped_with_checkpoint(ctx, lab, sub)
    foreign = rec.run_recover(
        ctx,
        idempotency_key="k2",
        attempt_id=s.attempt_id,
        checkpoint_ref=f"hpclab://artifacts/{other.artifact_id}",
    )
    assert not foreign.eligible and any("belongs to run" in b for b in foreign.blocking)
    with pytest.raises(NotFound):
        rec.run_recover(ctx, idempotency_key="k3", attempt_id=s.attempt_id, checkpoint_ref="art_missing")


def test_explicit_member_ref_picks_that_checkpoint(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub)
    d = rec.run_recover(
        ctx, idempotency_key="k", attempt_id=s.attempt_id, checkpoint_ref=f"{s.artifact_id}#ckpt-000000100"
    )
    assert d.strategy == "resume" and d.checkpoint and d.checkpoint.step == 100
    with pytest.raises(Exception, match="escapes"):
        rec.run_recover(
            ctx, idempotency_key="k2", attempt_id=s.attempt_id, checkpoint_ref=f"{s.artifact_id}#../../x"
        )


def test_resume_blockers(ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission) -> None:
    not_resumable = _stopped_with_checkpoint(ctx, lab, sub, resumable=False)
    d = rec.run_recover(ctx, idempotency_key="a", attempt_id=not_resumable.attempt_id, strategy="resume")
    assert not d.eligible and any("resumable" in b for b in d.blocking)
    # no manifest at all: an existing .pt proves nothing
    s = _stopped_with_checkpoint(ctx, lab, sub)
    for p in s.attempt_dir.joinpath("checkpoints").rglob("hpclab-ckpt.json"):
        p.unlink()
    d = rec.run_recover(ctx, idempotency_key="b", attempt_id=s.attempt_id, strategy="resume")
    assert not d.eligible and any("manifest" in b for b in d.blocking)
    # no checkpoint artifact indexed
    m = lab.matrix(n=1, resilience=ResilienceSpec(resumable=True))
    with ctx.db.tx() as tx:
        sub.admit(tx, ctx, run_id=m.run_ids[0], exec_plan_hash=m.exec_plans[m.run_ids[0]])
    aid = lab.fail(m.run_ids[0], state="PREEMPTED", exit_code=None)
    d = rec.run_recover(ctx, idempotency_key="c", attempt_id=aid, strategy="resume")
    assert not d.eligible and any("no checkpoint artifact" in b for b in d.blocking)
    auto = rec.run_recover(ctx, idempotency_key="d", attempt_id=aid)  # without a checkpoint: retry
    assert auto.eligible and auto.strategy == "retry"


def test_world_size_change_blocks_an_exact_resume(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    s = _stopped_with_checkpoint(ctx, lab, sub)
    d = rec.run_recover(
        ctx,
        idempotency_key="k",
        attempt_id=s.attempt_id,
        strategy="resume",
        resource_patch=rec.ResourcePatch(gpus=2),
    )
    assert not d.eligible and any("batch_invariant" in b for b in d.blocking)


def test_batch_resume_creates_one_probe_plan_dispatchable_as_a_workflow(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    from hpclab.services import workflows as wfm

    m = lab.matrix(n=2, resilience=ResilienceSpec(resumable=True))
    wf = wfm.workflow_submit(ctx, idempotency_key="w", plan_id=m.plan_id, max_parallel=2).workflow_id
    assert wfm.dispatch_workflow(ctx, workflow_id=wf).dispatched == 2
    for rid in m.run_ids:
        aid = lab.fail(rid, state="TIMEOUT", exit_code=None, elapsed_s=3600, time_limit_s=3600)
        adir = Path(one(ctx, "SELECT remote_dir FROM attempts WHERE attempt_id=?", (aid,))["remote_dir"])
        run = one(ctx, "SELECT spec_json, spec_hash FROM runs WHERE run_id=?", (rid,))
        spec = RunSpec.model_validate(jload(run["spec_json"]))
        write_checkpoint(
            adir / "checkpoints" / "ckpt-000000050",
            {"model.pt": b"w" * 10},
            step=50,
            run_id=rid,
            attempt_id=aid,
            run_spec_hash=run["spec_hash"],
            config={**spec.params, "world_size": 1},
            semantic_keys=["method"],
        )
        add_artifact(ctx, attempt_id=aid, run_id=rid, remote_path=str(adir / "checkpoints"))
    d = rec.run_recover(
        ctx, idempotency_key="k", selector=rec.RecoverySelector(workflow_id=wf), strategy="resume"
    )
    assert not d.eligible and d.plan_id is None and d.probe_plan_id
    assert d.cells is not None and all(c.probe_run_id and not c.eligible for c in d.cells)
    assert d.next_step and d.probe_plan_id in d.next_step
    probes = wfm.workflow_submit(ctx, idempotency_key="p", plan_id=d.probe_plan_id, max_parallel=2)
    assert probes.plan_kind == "probe" and probes.planned_run_count == 2
    assert wfm.dispatch_workflow(ctx, workflow_id=probes.workflow_id).dispatched == 2
    # asking again reuses the same probe runs (found by run + manifest + probe version)
    again = rec.run_recover(
        ctx, idempotency_key="k2", selector=rec.RecoverySelector(workflow_id=wf), strategy="resume"
    )
    assert again.probe_plan_id is None and again.cells is not None
    assert {c.probe_run_id for c in again.cells} == {c.probe_run_id for c in d.cells}


def test_a_probe_vouches_only_for_the_directory_it_checked(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """A copied checkpoint dir has the same manifest (same digest) but its own bytes: the probe
    that verified the original must not make the copy resumable."""
    import shutil

    s = _stopped_with_checkpoint(ctx, lab, sub)
    digest = manifest_digest(s.manifest)
    first = rec.run_recover(ctx, idempotency_key="k1", attempt_id=s.attempt_id, strategy="resume")
    assert first.probe is not None and first.probe.run_id and first.probe.exec_plan_hash
    rc = _complete_probe(ctx, lab, sub, first.probe.run_id, first.probe.exec_plan_hash, s.ckpt_dir, digest)
    assert rc == 0
    # the researcher keeps a copy of the checkpoint; the copy's weights are damaged (same size)
    copy = s.ckpt_dir.parent / "ckpt-000000500-copy"
    shutil.copytree(s.ckpt_dir, copy)
    model = copy / "model.pt"
    model.write_bytes(b"X" * model.stat().st_size)
    d = rec.run_recover(
        ctx,
        idempotency_key="k2",
        attempt_id=s.attempt_id,
        checkpoint_ref=f"{s.artifact_id}#ckpt-000000500-copy",
    )
    assert d.checkpoint is not None and d.checkpoint.remote_dir == str(copy)
    assert d.checkpoint.manifest_digest == digest  # indistinguishable by metadata
    assert not d.eligible and d.exec_plan_hash is None
    assert d.probe is not None and d.probe.created and d.probe.run_id != first.probe.run_id
    spec = RunSpec.model_validate(
        jload(one(ctx, "SELECT spec_json FROM runs WHERE run_id=?", (d.probe.run_id,))["spec_json"])
    )
    assert spec.argv[3] == str(copy)
    # the original directory is still covered by its own passed probe
    orig = rec.run_recover(
        ctx, idempotency_key="k3", attempt_id=s.attempt_id, checkpoint_ref=f"{s.artifact_id}#ckpt-000000500"
    )
    assert orig.eligible and orig.probe is not None and orig.probe.run_id == first.probe.run_id


def test_checkpoint_ref_must_name_a_checkpoint_artifact(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    from hpclab.domain.ids import new_id

    s = _stopped_with_checkpoint(ctx, lab, sub)
    metrics = new_id("artifact")
    with ctx.db.tx() as tx:
        tx.insert(
            "artifacts",
            {
                "artifact_id": metrics,
                "attempt_id": s.attempt_id,
                "run_id": s.run_id,
                "name": "metrics",
                "role": "metrics",
                "kind": "file",
                # a metrics file that sits next to the checkpoint dirs
                "remote_path": str(s.attempt_dir / "checkpoints" / "metrics.json"),
                "digest_state": "computed",
                "integrity": "finalized",
                "required": 1,
                "observed_at": tx.now,
            },
        )
    d = rec.run_recover(ctx, idempotency_key="k", attempt_id=s.attempt_id, checkpoint_ref=metrics)
    assert not d.eligible and any("not a checkpoint" in b for b in d.blocking)
    assert one(ctx, "SELECT COUNT(*) AS n FROM runs WHERE kind='probe'")["n"] == 0


def test_a_probe_is_never_requeued_even_when_the_run_is(
    ctx: ServiceContext, lab: LabBuilder, sub: FakeSubmission
) -> None:
    """A resumable Run on a preemptible partition gets ``--requeue``; its load probe must not:
    a requeued probe reruns by itself and its verdict would describe some later execution."""
    s = _stopped_with_checkpoint(ctx, lab, sub, requeue=True)
    first = rec.run_recover(ctx, idempotency_key="k1", attempt_id=s.attempt_id, strategy="resume")
    assert first.probe is not None and first.probe.created and first.probe.exec_plan_hash
    probe_plan = ExecutionPlan.model_validate(
        jload(
            one(
                ctx,
                "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?",
                (first.probe.exec_plan_hash,),
            )["plan_json"]
        )
    )
    assert probe_plan.requeue is False and probe_plan.resources.extra_sbatch == []
    from hpclab.services.jobscript import _resolve_requeue

    assert _resolve_requeue(probe_plan, probe_plan.resources.extra_sbatch) == "--no-requeue"
    # the resumed Run itself keeps its own --requeue
    digest = manifest_digest(s.manifest)
    assert (
        _complete_probe(ctx, lab, sub, first.probe.run_id, first.probe.exec_plan_hash, s.ckpt_dir, digest)
        == 0
    )
    ok = rec.run_recover(ctx, idempotency_key="k2", attempt_id=s.attempt_id, strategy="resume")
    assert ok.eligible and ok.exec_plan_hash
    resumed = ExecutionPlan.model_validate(
        jload(
            one(ctx, "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (ok.exec_plan_hash,))[
                "plan_json"
            ]
        )
    )
    assert resumed.requeue is True
