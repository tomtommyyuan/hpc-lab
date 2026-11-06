"""Design scenario 2: failure -> diagnosis -> resume from a checkpoint.

A resumable run is preempted after it wrote checkpoints (sim fault ``preempt_next``; the example
checkpoints on SIGTERM too). hpclab must diagnose the preemption, refuse to resume from an
unverified checkpoint without a compute-side probe, resume once the probe passed, and end with
metrics bit-identical to an uninterrupted run -- while the preempted attempt and the GPU-hours
it burned stay on record.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from e2e_support import Lab, argv_with, final_val_record, run_example_locally

pytestmark = [pytest.mark.sim, pytest.mark.slow, pytest.mark.timeout(150)]


def test_preempted_run_resumes_exactly(make_lab: Callable[..., Lab], tmp_path: Path) -> None:
    # the preemption strikes 2.5 s after the job starts: mid-training, after a few checkpoints
    lab: Lab = make_lab(sim={"fate_delay_s": 2.5})
    base = lab.spec()
    spec = lab.spec(
        matrix={"method": ["mixup"]},
        seeds=[2],
        smoke=None,
        resilience={"resumable": True},
        entrypoint={"argv": argv_with(base, "--sleep-per-step", "0.02", "--ckpt-every", "25")},
    )
    prep = lab.prepare(spec)
    (run,) = prep["runs"]
    rid = run["run_id"]

    lab.sim.set_faults(preempt_next=1)
    first = lab.call(
        "run_submit", run_id=rid, execution_plan_hash=run["exec_plan_hash"], idempotency_key=lab.key("s")
    )
    aid1 = first["attempt_id"]
    stopped = lab.wait_settled(aid1, timeout=60)
    assert stopped["scheduler_state"] == "PREEMPTED", stopped

    # diagnosis: an infrastructure event, not the program's fault
    insp = lab.call("run_inspect", run_id=rid)
    assert insp["diagnosis"]["primary_code"] == "preempted", insp["diagnosis"]
    assert insp["run"]["code"] == "F"
    ckpts = sorted((Path(stopped["remote_dir"]) / "checkpoints").glob("ckpt-*/hpclab-ckpt.json"))
    assert ckpts, "the preempted job left no complete checkpoint"

    # Slurm may requeue a preempted job: the run stays claimed for the requeue watch ...
    held = lab.call("run_recover", idempotency_key=lab.key("r"), attempt_id=aid1, strategy="resume")
    assert held["eligible"] is False and any("may be requeued" in b for b in held["blocking"]), held
    # ... unless a human confirms the job is over (hpclab looks at squeue/sacct first)
    cur = lab.attempts("a.attempt_id=?", (aid1,))[0]
    confirmed = lab.call(
        "attempt_resolve",
        attempt_id=aid1,
        expected_version=cur["version"],
        resolution="confirm_ended",
        reason="the partition cancels preempted jobs (no requeue); sacct shows PREEMPTED",
        idempotency_key=lab.key("resolve"),
    )
    assert confirmed["phase"] == "terminal", confirmed

    # resume needs proof the checkpoint loads on a compute node: a probe run comes first
    r1 = lab.call("run_recover", idempotency_key=lab.key("r"), attempt_id=aid1, strategy="resume")
    assert r1["eligible"] is False and r1["probe"] and r1["probe"]["created"], r1
    assert r1["checkpoint"]["continuation"] == "exact" and r1["checkpoint"]["status"] == "valid", r1[
        "checkpoint"
    ]
    jobs_before_probe = len(lab.sim.jobs())
    assert jobs_before_probe == 1  # recovery never submits by itself
    probe = lab.call(
        "run_submit",
        run_id=r1["probe"]["run_id"],
        execution_plan_hash=r1["probe"]["exec_plan_hash"],
        idempotency_key=lab.key("s"),
    )
    probe_att = lab.wait_settled(probe["attempt_id"])
    assert probe_att["scheduler_state"] == "COMPLETED" and probe_att["kind"] == "probe", probe_att

    r2 = lab.call("run_recover", idempotency_key=lab.key("r"), attempt_id=aid1, strategy="resume")
    assert r2["eligible"] is True and r2["exec_plan_hash"], r2
    assert r2["probe"]["state"] == "passed" and r2["run_id"] == rid
    resumed = lab.call(
        "run_submit", run_id=rid, execution_plan_hash=r2["exec_plan_hash"], idempotency_key=lab.key("s")
    )
    aid2 = resumed["attempt_id"]
    done = lab.wait_settled(aid2, timeout=60)
    assert done["scheduler_state"] == "COMPLETED" and done["result_state"] == "valid" and done["seq"] == 2

    logs = lab.call("run_logs", attempt_id=aid2, stream="stdout")["text"]
    assert "resumed from " in logs, logs

    # exact resume: identical final metrics to an uninterrupted run of the same config
    resumed_final = final_val_record(Path(done["remote_dir"]) / "outputs" / "metrics.jsonl")
    reference = run_example_locally(tmp_path / "reference", "--method", "mixup", "--seed", "2")
    assert resumed_final["metrics"] == reference["metrics"], (resumed_final, reference)
    evals = lab.wait(lambda: lab.evaluated(rid), timeout=30, what="evaluation of the resumed attempt")
    ev = lab.call("evaluation_get", evaluation_id=evals[0]["evaluation_id"])
    values = {m["name"]: m["value"] for m in ev["metrics"]}
    assert ev["attempt_id"] == aid2 and values["acc"] == reference["metrics"]["acc"], ev["metrics"]

    # history: the preempted attempt and its GPU-hours stay recorded next to the resumed one
    final = lab.call("run_inspect", run_id=rid)
    by_id = {a["attempt_id"]: a for a in final["attempts"]}
    assert set(by_id) == {aid1, aid2} and final["run"]["code"] == "V"
    old = by_id[aid1]
    assert old["scheduler_state"] == "PREEMPTED" and old["phase"] == "terminal"
    assert old["budget"]["state"] == "settled" and old["budget"]["settled_consumed"] is not None, old[
        "budget"
    ]
    assert old["elapsed_s"] is not None and old["elapsed_s"] >= 2, old
    assert by_id[aid2]["created_reason"] and "resume" in by_id[aid2]["created_reason"], by_id[aid2]
    # ground truth: original, probe and resume -- each submitted exactly once
    assert len(lab.sim.jobs()) == 3
    assert not lab.duplicate_jobs()
