"""The real SlurmScheduler driving the simulator through SimTransport (shims + bash snippets).

Jobs are real local processes; every test keeps them short (sub-second to a few seconds).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    ConnectionLost,
    SubmissionRejected,
    TransportError,
    TransportTimeout,
)
from hpclab.adapters.slurm import SlurmScheduler
from hpclab.domain.clock import iso
from hpclab.domain.errors import ErrorCode
from hpclab.domain.models import JobObservation
from hpclab.domain.states import SchedulerState
from hpclab.sim.cluster import SimCluster
from hpclab.sim.testing import sim_cluster, sim_scheduler
from hpclab.sim.transport import SimTransport

pytestmark = pytest.mark.sim
TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
)


def wait_until[T](predicate: Callable[[], T | None], timeout: float = 10.0, what: str = "condition") -> T:
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def queued(slurm: SlurmScheduler, job_id: str) -> JobObservation | None:
    return next((j for j in slurm.query_queue(job_ids=[job_id]).jobs if j.job_id == job_id), None)


def accounted(slurm: SlurmScheduler, job_id: str) -> JobObservation | None:
    return next((j for j in slurm.query_accounting(job_ids=[job_id]).jobs if j.job_id == job_id), None)


def wait_state(
    slurm: SlurmScheduler, job_id: str, state: SchedulerState, timeout: float = 10.0
) -> JobObservation:
    def check() -> JobObservation | None:
        obs = queued(slurm, job_id) or accounted(slurm, job_id)
        return obs if obs is not None and obs.state == state else None

    return wait_until(check, timeout, f"job {job_id} to reach {state}")


def finished(sim: SimCluster, job_id: str, timeout: float = 10.0) -> dict:
    job = sim.wait_for(job_id, TERMINAL, timeout)
    assert job is not None, f"job {job_id} did not finish: {sim.job(job_id)}"
    return job


def ago(seconds: float) -> str:
    stamp = iso(datetime.now(UTC) - timedelta(seconds=seconds))
    assert stamp is not None
    return stamp


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_submit_run_complete_with_outputs(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    run_dir = workdir / "run"
    (run_dir / "logs").mkdir(parents=True)
    script = write_script(
        'echo "id=$SLURM_JOB_ID name=$SLURM_JOB_NAME part=$SLURM_JOB_PARTITION acct=$SLURM_JOB_ACCOUNT"\n'
        'echo "nodes=$SLURM_JOB_NODELIST cpt=$SLURM_CPUS_PER_TASK gpus=$SLURM_GPUS_ON_NODE cvd=$CUDA_VISIBLE_DEVICES"\n'
        'echo "restart=$SLURM_RESTART_COUNT submit=$SLURM_SUBMIT_DIR pwd=$(pwd)"\n'
        "echo to-stderr >&2",
        "--job-name=hello",
        "--gres=gpu:a100:2",
        "--cpus-per-task=2",
        "--time=00:05:00",
        f"--chdir={run_dir}",
        "--output=logs/%x-%j.out",
        "--error=logs/%x-%j.err",
        "--comment=hpclab:tok123",
    )
    receipt = workdir / "submit.receipt"
    resp = slurm.submit(str(script), workdir=str(workdir), receipt_path=str(receipt))
    job_id = resp.job_id
    assert job_id == "1000" and sim.job(job_id) is not None
    assert finished(sim, job_id)["state"] == "COMPLETED"

    out = (run_dir / "logs" / f"hello-{job_id}.out").read_text()
    assert f"id={job_id} name=hello part=gpu acct=lab" in out
    assert "nodes=sim-a100-01 cpt=2 gpus=2 cvd=0,1" in out
    assert f"restart=0 submit={workdir} pwd={run_dir}" in out
    assert (run_dir / "logs" / f"hello-{job_id}.err").read_text() == "to-stderr\n"

    obs = queued(slurm, job_id)  # still within MinJobAge
    assert obs is not None and obs.state == SchedulerState.COMPLETED and obs.end_time is not None
    acct = accounted(slurm, job_id)
    assert acct is not None
    assert (acct.state, acct.exit_code, acct.exit_signal, acct.alloc_gpus) == (
        SchedulerState.COMPLETED,
        0,
        0,
        2,
    )
    assert (acct.comment, acct.name, acct.partition, acct.account) == ("hpclab:tok123", "hello", "gpu", "lab")
    assert acct.submit_line is not None and acct.submit_line.startswith("sbatch --parsable")
    assert acct.work_dir == str(run_dir) and acct.time_limit_s == 300
    wait_until(lambda: queued(slurm, job_id) is None, 5.0, "squeue to forget the job (MinJobAge)")
    recovered = slurm.read_receipt(str(receipt))
    assert recovered is not None and recovered.job_id == job_id


def test_default_output_and_open_mode_append(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    first = slurm.submit(str(write_script("echo first")), workdir=str(workdir))
    finished(sim, first.job_id)
    assert (workdir / f"slurm-{first.job_id}.out").read_text() == "first\n"
    log = workdir / "shared.log"
    log.write_text("previous\n")
    appended = write_script("echo second", "--output=shared.log", "--open-mode=append", name="append.sbatch")
    finished(sim, slurm.submit(str(appended), workdir=str(workdir)).job_id)
    assert log.read_text() == "previous\nsecond\n"


def test_timeout_sends_term_then_records_timeout(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    script = write_script(
        "trap 'echo got-TERM; exit 143' TERM\nsleep 30 &\nwait", "--time=0:01", "--output=t.out"
    )
    job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
    job = finished(sim, job_id, timeout=8.0)
    assert job["state"] == "TIMEOUT"
    out = (workdir / "t.out").read_text()
    assert "DUE TO TIME LIMIT" in out and "got-TERM" in out
    acct = accounted(slurm, job_id)
    assert acct is not None and acct.state == SchedulerState.TIMEOUT and acct.elapsed_s is not None


def test_scancel_running_and_pending(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    script = write_script("trap 'echo got-TERM; exit 143' TERM\nsleep 30 &\nwait", "--output=c.out")
    job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
    wait_state(slurm, job_id, SchedulerState.RUNNING)
    resp = slurm.cancel(job_id)
    assert resp.accepted and resp.raw == ""
    assert finished(sim, job_id)["state"] == "CANCELLED"
    acct = accounted(slurm, job_id)
    assert acct is not None and acct.state == SchedulerState.CANCELLED
    assert acct.raw_state == f"CANCELLED by {os.getuid()}"
    out = (workdir / "c.out").read_text()
    assert f"*** JOB {job_id} ON sim-a100-01 CANCELLED AT" in out and "got-TERM" in out
    again = slurm.cancel(job_id)
    assert not again.accepted and "already completing or completed" in again.raw

    held = slurm.submit(
        str(write_script("echo never", "--hold", name="held.sbatch")), workdir=str(workdir)
    ).job_id
    pending = wait_state(slurm, held, SchedulerState.PENDING)
    assert pending.reason == "JobHeldUser"
    assert slurm.cancel(held).accepted
    cancelled = finished(sim, held)
    assert cancelled["state"] == "CANCELLED" and cancelled["node_list"] == []


def test_cancelled_while_pending_reports_start_equal_to_end_like_slurmctld(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    # slurmctld sets start_time = end_time = <cancel time> for a pending job it cancels, so
    # squeue/sacct show a Start for a job that never ran. Consumers must look at node_list /
    # elapsed, not at "start_time is not None"; the simulator must not hide that trap.
    held = slurm.submit(
        str(write_script("echo never", "--hold", "--gres=gpu:a100:2", name="held.sbatch")),
        workdir=str(workdir),
    ).job_id
    wait_state(slurm, held, SchedulerState.PENDING)
    assert slurm.cancel(held).accepted
    finished(sim, held)
    for obs in (queued(slurm, held), accounted(slurm, held)):
        assert obs is not None and obs.state == SchedulerState.CANCELLED
        assert obs.start_time is not None and obs.start_time == obs.end_time
        assert obs.node_list is None
    acct = accounted(slurm, held)
    assert acct is not None and acct.elapsed_s == 0 and acct.alloc_gpus is None  # AllocTRES empty
    steps = sim.run("sacct", "-n", "-P", "-j", held, "--format=JobID,State")
    assert steps.stdout.splitlines() == [f"{held}|CANCELLED by {os.getuid()}"]  # no .batch/.extern rows


@pytest.mark.parametrize("body", ["exit 0", "exit 7"])
def test_processes_left_behind_by_the_batch_script_die_with_the_job(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path, body: str
) -> None:
    # slurmstepd kills everything left in the step when the batch shell exits; a leaked
    # background sidecar (telemetry loop, dataloader worker) must not outlive the job.
    marker = workdir / f"bg-{body[-1]}.pid"
    script = write_script(f"sleep 30 &\necho $! > {marker}\n{body}", name=f"bg{body[-1]}.sbatch")
    job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
    finished(sim, job_id)
    pid = int(marker.read_text())
    assert wait_until(lambda: not _pid_alive(pid), 3.0, "the background process to be killed")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_failing_exit_codes(sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path) -> None:
    bad = slurm.submit(str(write_script("exit 3", name="exit3.sbatch")), workdir=str(workdir)).job_id
    segv = slurm.submit(str(write_script("kill -SEGV $$", name="segv.sbatch")), workdir=str(workdir)).job_id
    assert finished(sim, bad)["state"] == "FAILED"
    assert finished(sim, segv)["state"] == "FAILED"
    acct = {o.job_id: o for o in slurm.query_accounting(job_ids=[bad, segv]).jobs}
    assert (acct[bad].exit_code, acct[bad].exit_signal) == (3, 0)
    assert (acct[segv].exit_code, acct[segv].exit_signal) == (0, 11)


# ---------------------------------------------------------------------------
# submission failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("directive", "reason"),
    [
        ("--partition=nope", "invalid_partition"),
        ("--account=ghost", "invalid_account"),
        ("--time=3-00:00:00", "invalid_time_limit"),
        ("--gres=gpu:v100:1", "invalid_gres"),
        ("--gres=gpu:a100:8", "node_config_unavailable"),
        ("--mem-per-gpu=10XB", "invalid_option"),  # client-side parse error: never queued
        ("--open-mode=sometimes", "invalid_option"),
    ],
)
def test_rejections_through_the_scheduler(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path, directive: str, reason: str
) -> None:
    receipt = workdir / "submit.receipt"
    with pytest.raises(SubmissionRejected) as exc:
        slurm.submit(str(write_script("echo hi", directive)), workdir=str(workdir), receipt_path=str(receipt))
    assert exc.value.details["reason_code"] == reason
    assert sim.jobs() == []
    with pytest.raises(SubmissionRejected):
        slurm.read_receipt(str(receipt))


def test_lost_reply_job_exists_and_receipt_recovers_it(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(sbatch_lose_reply=1)
    receipt = workdir / "submit.receipt"
    script = write_script("sleep 0.2", "--job-name=hl.lost.tok99999")
    with pytest.raises(ConnectionLost) as exc:
        slurm.submit(str(script), workdir=str(workdir), receipt_path=str(receipt))
    assert exc.value.maybe_executed is True
    jobs = sim.jobs()
    assert len(jobs) == 1  # ground truth: sbatch accepted exactly one job
    recovered = slurm.read_receipt(str(receipt))
    assert recovered is not None and recovered.job_id == str(jobs[0]["job_id"])
    seen = queued(slurm, recovered.job_id) or accounted(slurm, recovered.job_id)
    assert seen is not None and seen.name == "hl.lost.tok99999"
    assert sim.faults().sbatch_lose_reply == 0
    ok = slurm.submit(str(script), workdir=str(workdir))  # the fault was consumed
    assert ok.job_id == str(jobs[0]["job_id"] + 1)


def test_ambiguous_error_may_or_may_not_queue(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(sbatch_ambiguous_error=1)
    receipt = workdir / "amb.receipt"
    script = write_script("sleep 0.1", "--job-name=hl.amb.tok11111")
    with pytest.raises(ConnectionLost) as exc:
        slurm.submit(str(script), workdir=str(workdir), receipt_path=str(receipt))
    assert exc.value.details["reason_code"] == "socket_timeout"
    assert len(sim.jobs()) == 1
    assert slurm.read_receipt(str(receipt)) is None  # the journal identifies no job...
    found = slurm.query_accounting(name="hl.amb.tok11111").jobs  # ...but reconciliation by name does
    assert [j.job_id for j in found] == [str(sim.jobs()[0]["job_id"])]

    sim.set_faults(sbatch_ambiguous_error=1, sbatch_ambiguous_creates_job=False)
    with pytest.raises(ConnectionLost):
        slurm.submit(str(script), workdir=str(workdir))
    assert len(sim.jobs()) == 1


def test_test_only_through_the_scheduler(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    est = slurm.test_submit(
        str(write_script("echo hi", "--gres=gpu:h100:8", "--partition=scavenge")), workdir=str(workdir)
    )
    assert est.partition == "scavenge" and est.nodes == "sim-h100-01" and est.start_time is not None
    assert sim.jobs() == []


# ---------------------------------------------------------------------------
# observation semantics
# ---------------------------------------------------------------------------


def test_accounting_lag_and_min_job_age_leave_an_observation_gap(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(accounting_lag_s=3.0, min_job_age_s=0.2)
    job_id = slurm.submit(str(write_script("true")), workdir=str(workdir)).job_id
    job = finished(sim, job_id)
    wait_until(lambda: time.time() > job["end_time"] + 0.4, 5.0, "MinJobAge to pass")
    q = slurm.query_queue(job_ids=[job_id])
    assert q.complete and q.jobs == []  # gone from the controller...
    assert slurm.query_accounting(job_ids=[job_id]).jobs == []  # ...and not yet in slurmdbd
    # slurmdbd catches up gradually: first a stale PENDING/RUNNING record, then the terminal one.
    first = wait_until(lambda: accounted(slurm, job_id), 6.0, "the first accounting record")
    assert first.state in (SchedulerState.PENDING, SchedulerState.RUNNING, SchedulerState.COMPLETED)
    final = wait_until(
        lambda: (o := accounted(slurm, job_id)) is not None and o.state == SchedulerState.COMPLETED and o,
        6.0,
        "the terminal accounting record",
    )
    assert final.end_time is not None


def test_gpu_capacity_queueing(sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path) -> None:
    big = write_script("sleep 30", "--gres=gpu:a100:4", name="big.sbatch")
    ids = [slurm.submit(str(big), workdir=str(workdir)).job_id for _ in range(5)]
    small = slurm.submit(
        str(write_script("true", "--gres=gpu:1", name="small.sbatch")), workdir=str(workdir)
    ).job_id

    def settled() -> dict[str, JobObservation] | None:
        obs = {o.job_id: o for o in slurm.query_queue(job_ids=[*ids, small]).jobs}
        running = [i for i in ids if i in obs and obs[i].state == SchedulerState.RUNNING]
        return obs if len(running) == 4 and small in obs and obs[small].reason == "Priority" else None

    obs = wait_until(settled, 10.0, "four big jobs running and the queue settled")
    assert obs[ids[4]].state == SchedulerState.PENDING and obs[ids[4]].reason == "Resources"
    assert obs[small].state == SchedulerState.PENDING  # strict FIFO: no backfill past the blocked head
    gpu = next(p for p in slurm.partitions() if p.name == "gpu")
    assert gpu.gpu_types["a100"] == {"total": 16, "used": 16, "free": 0, "unavailable": 0}
    est = {e.job_id: e for e in slurm.start_estimates([ids[4], small])}
    assert est[ids[4]].start_time is not None and est[ids[4]].reason == "Resources"

    assert slurm.cancel(ids[0]).accepted  # one node frees up: it goes to the head of the queue...
    wait_state(slurm, ids[4], SchedulerState.RUNNING)
    head = wait_until(
        lambda: (o := queued(slurm, small)) and o.reason == "Resources" and o, 5.0, "small job at head"
    )
    assert head.state == SchedulerState.PENDING  # ...and the small job is now the blocked head
    assert slurm.cancel(ids[1]).accepted
    assert finished(sim, small)["state"] == "COMPLETED"
    for job_id in ids[2:]:
        slurm.cancel(job_id)
    assert sim.wait_idle(10.0)


def test_partition_availability_reports_drained_nodes(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_node_state("sim-a100-04", "drain", "bad NVLink")
    job_id = slurm.submit(str(write_script("sleep 5", "--gres=gpu:2")), workdir=str(workdir)).job_id
    wait_state(slurm, job_id, SchedulerState.RUNNING)
    parts = {p.name: p for p in slurm.partitions()}
    assert parts["gpu"].gpu_types["a100"] == {"total": 16, "used": 2, "free": 10, "unavailable": 4}
    assert parts["gpu"].default and parts["gpu"].max_time_s == 2 * 86400 and parts["gpu"].nodes_total == 4
    assert parts["scavenge"].preemptible is True and parts["interactive"].preemptible is False
    nodes = {n.name: n for n in slurm.nodes("gpu")}
    assert nodes["sim-a100-04"].state == "drained" and nodes["sim-a100-04"].reason == "bad NVLink"
    assert nodes["sim-a100-01"].state == "mixed" and nodes["sim-a100-01"].gres_used == "gpu:a100:2(IDX:0-1)"
    slurm.cancel(job_id)
    finished(sim, job_id)


def test_associations_and_capabilities(
    sim: SimCluster, slurm: SlurmScheduler, transport: SimTransport
) -> None:
    caps = slurm.probe_capabilities()
    assert caps.slurm_version == "23.02.7" and caps.sacct_available and caps.sacct_comment_field
    assert caps.sstat_available and caps.srun_overlap
    assocs = slurm.associations(transport.remote_user())
    assert [(a.account, a.qos, a.default_qos) for a in assocs] == [
        ("lab", ["normal"], "normal"),
        ("other", ["normal"], "normal"),
    ]


def test_gpu_samples_and_step_stats(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(gpu_util=42.0)
    job_id = slurm.submit(
        str(write_script("sleep 5", "--gres=gpu:h100:2", "--partition=scavenge")), workdir=str(workdir)
    ).job_id
    wait_state(slurm, job_id, SchedulerState.RUNNING)
    samples = slurm.gpu_samples(job_id)
    assert [s.index for s in samples] == [0, 1]
    assert all(s.util_percent == 42.0 and s.name == "NVIDIA H100 80GB HBM3" for s in samples)
    steps = {s.step: s for s in slurm.step_stats(job_id)}
    assert set(steps) == {"batch", "extern"} and (steps["batch"].max_rss_bytes or 0) > 0
    slurm.cancel(job_id)
    finished(sim, job_id)
    with pytest.raises(TransportError) as exc:
        slurm.gpu_samples(job_id)
    assert exc.value.code == ErrorCode.PRECONDITION_FAILED
    assert slurm.step_stats(job_id) == []

    held = slurm.submit(str(write_script("true", "--hold", name="held.sbatch")), workdir=str(workdir)).job_id
    with pytest.raises(TransportError) as exc2:
        slurm.gpu_samples(held)
    assert exc2.value.code == ErrorCode.NOT_READY


def test_usage_counts_allocated_gpus(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    a = slurm.submit(
        str(write_script("sleep 0.3", "--gres=gpu:2", name="a.sbatch")), workdir=str(workdir)
    ).job_id
    b = slurm.submit(
        str(write_script("true", "--gres=gpu:1", "--account=other", name="b.sbatch")), workdir=str(workdir)
    ).job_id
    finished(sim, a)
    finished(sim, b)
    rows = {r.job_id: r for r in slurm.usage(account="lab", users=None, since=ago(600), until=None)}
    assert set(rows) == {a}
    assert rows[a].gpus == 2 and rows[a].state == "COMPLETED" and rows[a].user == sim.user
    everyone = {r.job_id: r for r in slurm.usage(account=None, users=[sim.user], since=ago(600), until=None)}
    assert set(everyone) == {a, b} and everyone[b].account == "other" and everyone[b].gpus == 1


def test_untracked_gpu_tres_makes_gpu_counts_unknown(tmp_path: Path) -> None:
    # Slurm's default AccountingStorageTRES has no gres/gpu: the TRES strings of a GPU job then
    # carry no GPU entry. The simulator reproduces that; the scheduler must say "unknown".
    with sim_cluster(tmp_path, track_gpu_tres=False) as cluster:
        _, sched, _ = sim_scheduler(cluster)
        work = cluster.fs_root / "w"
        work.mkdir(parents=True)
        script = work / "gpu.sbatch"
        script.write_text("#!/bin/bash\n#SBATCH --gres=gpu:a100:2\nsleep 2\n")
        job_id = sched.submit(str(script), workdir=str(work)).job_id
        running = wait_state(sched, job_id, SchedulerState.RUNNING)
        assert running.alloc_gpus is None
        finished(cluster, job_id)
        acct = accounted(sched, job_id)
        assert acct is not None and acct.alloc_gpus is None and acct.elapsed_s is not None
        assert any("gres/gpu" in n for n in sched.probe_capabilities().notes)
        with pytest.raises(TransportError) as exc:
            sched.usage(account=None, users=None, since=ago(600), until=None)
        assert exc.value.code == ErrorCode.UNSUPPORTED


def test_injected_fates(sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path) -> None:
    # --no-requeue: with JobRequeue=1 a node failure would requeue the job (tested separately)
    script = write_script("sleep 10", "--output=fate-%j.out", "--no-requeue")
    fates = {}
    for fault in ("node_fail_next", "preempt_next", "oom_next"):
        sim.set_faults(**{fault: 1})
        fates[fault] = slurm.submit(str(script), workdir=str(workdir)).job_id
    states = {fault: finished(sim, job_id)["state"] for fault, job_id in fates.items()}
    assert states == {"node_fail_next": "NODE_FAIL", "preempt_next": "PREEMPTED", "oom_next": "OUT_OF_MEMORY"}
    acct = {o.job_id: o for o in slurm.query_accounting(job_ids=list(fates.values())).jobs}
    assert (
        acct[fates["oom_next"]].state == SchedulerState.OUT_OF_MEMORY
        and acct[fates["oom_next"]].exit_signal == 125
    )
    assert acct[fates["preempt_next"]].state == SchedulerState.PREEMPTED
    assert acct[fates["node_fail_next"]].state == SchedulerState.NODE_FAIL
    assert "oom-kill event" in (workdir / f"fate-{fates['oom_next']}.out").read_text()


def test_dependencies(sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path) -> None:
    first = slurm.submit(str(write_script("sleep 1", name="first.sbatch")), workdir=str(workdir)).job_id
    then = write_script("echo after", f"--dependency=afterok:{first}", name="then.sbatch")
    second = slurm.submit(str(then), workdir=str(workdir)).job_id
    waiting = wait_state(slurm, second, SchedulerState.PENDING)
    assert waiting.reason == "Dependency"
    done = finished(sim, second)
    assert done["state"] == "COMPLETED" and done["start_time"] >= finished(sim, first)["end_time"]

    failing = slurm.submit(str(write_script("exit 1", name="fail.sbatch")), workdir=str(workdir)).job_id
    orphan = write_script("echo never", f"--dependency=afterok:{failing}", name="orphan.sbatch")
    stuck = slurm.submit(str(orphan), workdir=str(workdir)).job_id
    finished(sim, failing)
    obs = wait_until(
        lambda: (o := queued(slurm, stuck)) and o.reason == "DependencyNeverSatisfied" and o,
        5.0,
        "DependencyNeverSatisfied",
    )
    assert obs.state == SchedulerState.PENDING


def test_array_jobs_use_underscore_ids(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    script = write_script(
        'echo "task=$SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_JOB_ID"', "--array=0-2", "--output=a-%A_%a.out"
    )
    slurm.probe_capabilities()  # one-time probe first: the tasks finish within MinJobAge
    base = slurm.submit(str(script), workdir=str(workdir)).job_id
    obs = slurm.query_queue(job_ids=[base]).jobs
    assert sorted(o.job_id for o in obs) == [f"{base}_0", f"{base}_1", f"{base}_2"]
    assert sorted(o.array_task_id or "" for o in obs) == ["0", "1", "2"]
    for task in range(3):
        finished(sim, f"{base}_{task}")
        assert (workdir / f"a-{base}_{task}.out").read_text() == f"task={task} of {base}\n"
    acct = slurm.query_accounting(job_ids=[f"{base}_1"]).jobs
    assert [(o.job_id, o.array_task_id) for o in acct] == [(f"{base}_1", "1")]


# ---------------------------------------------------------------------------
# transport-level faults and robustness
# ---------------------------------------------------------------------------


def test_transport_faults(
    sim: SimCluster, slurm: SlurmScheduler, transport: SimTransport, write_script, workdir: Path
) -> None:
    slurm.probe_capabilities()  # probing runs `sbatch --version`, which the faults below would block
    sim.set_faults(auth_required=True)
    with pytest.raises(AuthRequiredError) as exc:
        slurm.query_queue(job_ids=["1"])
    assert exc.value.code == ErrorCode.AUTH_REQUIRED and "login_command" in exc.value.details
    with pytest.raises(AuthRequiredError):
        transport.stat([str(workdir)])
    sim.clear_faults()

    sim.set_faults(transport_timeout_cmds=["squeue", "sbatch"])
    with pytest.raises(TransportTimeout):
        slurm.query_queue(job_ids=["1"])
    with pytest.raises(TransportTimeout):
        slurm.submit(str(write_script("true")), workdir=str(workdir))
    assert sim.jobs() == []  # the timeout fault fires before anything runs
    assert slurm.query_accounting(job_ids=["1"]).jobs == []  # other commands still work
    sim.clear_faults()

    sim.set_faults(squeue_unavailable=True, sacct_unavailable=True)
    with pytest.raises(CommandFailed) as qexc:
        slurm.query_queue(job_ids=["1"])
    assert qexc.value.code == ErrorCode.UNAVAILABLE
    with pytest.raises(CommandFailed) as aexc:
        slurm.query_accounting(job_ids=["1"])
    assert aexc.value.code == ErrorCode.UNAVAILABLE


def test_shutdown_kills_running_jobs(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    job_id = slurm.submit(str(write_script("sleep 60")), workdir=str(workdir)).job_id
    wait_state(slurm, job_id, SchedulerState.RUNNING)
    pgid = wait_until(lambda: (j := sim.state.job(int(job_id))) and j.pgid, 5.0, "job process group")
    killed = sim.shutdown(grace_s=2.0)
    assert int(job_id) in killed
    assert sim.job(job_id)["state"] == "CANCELLED"
    assert wait_until(lambda: _group_gone(pgid), 3.0, "the job's process group to exit")


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def test_concurrent_submissions_get_unique_ids(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    script = write_script("true")
    results: list[str] = []
    errors: list[BaseException] = []

    def submit() -> None:
        try:
            results.append(slurm.submit(str(script), workdir=str(workdir)).job_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sorted(results) == [str(i) for i in range(1000, 1006)]
    assert sim.wait_idle(10.0)
    assert {j["state"] for j in sim.jobs()} == {"COMPLETED"}


def test_pending_delay_keeps_jobs_queued(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(pending_delay_s=1.0)
    job_id = slurm.submit(str(write_script("true")), workdir=str(workdir)).job_id
    obs = queued(slurm, job_id)
    assert obs is not None and obs.state == SchedulerState.PENDING
    wait_until(
        lambda: (o := queued(slurm, job_id)) is not None and o.reason == "Priority", 2.0, "Reason=Priority"
    )
    job = finished(sim, job_id, timeout=6.0)
    assert job["state"] == "COMPLETED" and job["start_time"] - job["submit_time"] >= 1.0


def test_lose_reply_only_applies_to_transport_calls(sim: SimCluster, run_cli, write_script) -> None:
    sim.set_faults(sbatch_lose_reply=1)
    res = run_cli("sbatch", "--parsable", str(write_script("true")))  # no SimTransport call id
    assert res.rc == 0 and res.stdout.strip() == "1000"
    assert sim.faults().sbatch_lose_reply == 1
    finished(sim, "1000")


def test_unenforced_partition_limit_pends_with_reason(tmp_path: Path) -> None:
    from hpclab.sim.testing import sim_cluster, sim_scheduler

    with sim_cluster(tmp_path, enforce_part_limits=False) as sim:
        transport, slurm, resolved = sim_scheduler(sim, cluster_id="sim-lax")
        assert resolved.cluster_id == "sim-lax" and resolved.profile.transport == "sim"
        assert resolved.remote_root() == str(sim.fs_root / "scratch" / sim.user / "hpclab")
        assert transport.remote_user() == sim.user
        workdir = sim.fs_root / "w"
        workdir.mkdir()
        script = workdir / "long.sbatch"
        script.write_text("#!/bin/bash\n#SBATCH --partition=interactive\n#SBATCH --time=5:00:00\ntrue\n")
        job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
        obs = wait_until(
            lambda: (o := queued(slurm, job_id)) is not None and o.reason == "PartitionTimeLimit" and o,
            5.0,
            "Reason=PartitionTimeLimit",
        )
        assert obs.state == SchedulerState.PENDING and obs.time_limit_s == 5 * 3600


def test_protocol_conformance(sim: SimCluster, slurm: SlurmScheduler, transport: SimTransport) -> None:
    from hpclab.adapters.base import Scheduler, Transport

    assert isinstance(slurm, Scheduler)
    assert isinstance(transport, Transport)
