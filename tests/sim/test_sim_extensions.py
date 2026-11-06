"""Simulator extensions: config facts, QOS, signals, requeue segments, PrivateData, injected
jobs and the per-job fake ``nvidia-smi`` (so the real job wrapper's GPU guard and telemetry run).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from hpclab.adapters.slurm import JobShape, SlurmScheduler
from hpclab.domain.models import (
    CodeOptions,
    EntrypointSpec,
    ExecutionPlan,
    OutputDecl,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.domain.states import SchedulerState
from hpclab.services.jobscript import attempt_layout, job_name_for, render_job_script
from hpclab.services.snapshot import SnapshotStore, capture_snapshot, publish_snapshot
from hpclab.sim.cluster import SimCluster
from hpclab.sim.config import PartitionConfig, SimConfig, SimFaults
from hpclab.sim.testing import sim_cluster, sim_scheduler

TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
)


def finished(sim: SimCluster, job_id: str, timeout: float = 15.0) -> dict:
    job = sim.wait_for(job_id, TERMINAL, timeout)
    assert job is not None, f"job {job_id} did not finish: {sim.job(job_id)}"
    return job


def wait_for(pred, timeout: float = 10.0, what: str = "condition"):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# config facts, partitions, QOS (in-process fake commands)
# ---------------------------------------------------------------------------


def test_scontrol_config_reports_visibility_and_requeue_facts(sim: SimCluster, run_cli) -> None:
    config = run_cli("scontrol", "show", "config").stdout
    for line in (
        "PrivateData             = none",
        "JobRequeue              = 1",
        "MailProg                = /bin/mail",
        "JobAcctGatherType       = jobacct_gather/cgroup",
        "MinJobAge               = 1 sec",
    ):
        assert line in config


def test_scontrol_config_overrides(tmp_path: Path) -> None:
    with sim_cluster(tmp_path, private_data="jobs,usage", mail_prog=None, job_requeue=False) as sim:
        out = sim.run("scontrol", "show", "config").stdout
        assert "PrivateData             = jobs,usage" in out
        assert "MailProg                = (null)" in out and "JobRequeue              = 0" in out


def test_partition_grace_time_and_preempt_mode(tmp_path: Path) -> None:
    parts = (
        PartitionConfig(name="gpu", nodes=("sim-a100-01",), max_time_s=3600, default=True),
        PartitionConfig(
            name="lo", nodes=("sim-h100-01",), preemptible=True, priority_tier=1, grace_time_s=300,
            preempt_mode="requeue",
        ),
    )  # fmt: skip
    with sim_cluster(tmp_path, partitions=parts) as sim:
        _, sched, _ = sim_scheduler(sim)
        details = {p.name: p for p in sched.partition_details()}
        assert details["lo"].preempt_mode == "REQUEUE" and details["lo"].grace_time_s == 300
        assert details["gpu"].preempt_mode == "OFF" and details["gpu"].grace_time_s == 0
        assert details["gpu"].default and details["gpu"].max_time_s == 3600
        assert details["lo"].priority_tier == 1 and details["lo"].total_gpus == 8


def test_config_round_trip_keeps_new_fields() -> None:
    cfg = SimConfig(user="u", private_data="jobs", qos_max_gpus_per_user=4, driver_cuda="12.4")
    assert SimConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.hides_other_jobs() and not SimConfig(user="u").hides_other_jobs()
    assert "requeue_next" in SimFaults.COUNTERS and SimFaults.coerce("requeue_next", "2") == 2


def test_sacctmgr_show_qos(tmp_path: Path) -> None:
    with sim_cluster(
        tmp_path, qos=("normal", "long"), qos_max_gpus_per_user=8, qos_max_wall_s=86400,
        qos_max_jobs_per_user=4, max_submit_jobs_per_user=10,
    ) as sim:  # fmt: skip
        _, sched, _ = sim_scheduler(sim)
        limits = {q.name: q for q in sched.qos_limits()}
        assert set(limits) == {"normal", "long"}
        n = limits["normal"]
        assert (n.max_gpus_per_user, n.max_wall_s, n.max_jobs_per_user, n.max_submit_per_user) == (
            8,
            86400,
            4,
            10,
        )
        assert n.max_gpus_per_job is None and n.grp_gpus is None
        assert [q.name for q in sched.qos_limits(["long"])] == ["long"]
        bad = sim.run("sacctmgr", "-n", "-P", "show", "qos", "format=Bogus")
        assert bad.returncode == 1 and "Unknown field" in bad.stderr


def test_default_qos_has_no_limits(sim: SimCluster, run_cli) -> None:
    res = run_cli("sacctmgr", "-n", "-P", "show", "qos", "format=Name,MaxTRESPU,MaxWall")
    assert res.rc == 0 and res.stdout == "normal||\n"


# ---------------------------------------------------------------------------
# injected jobs + PrivateData
# ---------------------------------------------------------------------------


def test_injected_jobs_are_history_not_processes(sim: SimCluster, slurm: SlurmScheduler) -> None:
    now = time.time()
    done = sim.inject_job(
        user="alice", submit_time=now - 7200, start_time=now - 7000, end_time=now - 3400, gpus=4,
        gpu_type="h100", partition="scavenge", state="TIMEOUT",
    )  # fmt: skip
    live = sim.inject_job(user="bob", submit_time=now - 3600, start_time=now - 3500, gpus=2, state="RUNNING")
    sim.state.reap()  # an hour without a runner process must not become NODE_FAIL
    assert sim.job(live)["state"] == "RUNNING" and sim.job(done)["state"] == "TIMEOUT"
    assert sim.wait_idle(1.0)  # injected records never block "idle"
    running = {j.job_id: j for j in slurm.query_account_queue("lab").jobs}
    assert running[str(live)].user == "bob" and running[str(live)].alloc_gpus == 2
    capacity = {p.name: p for p in slurm.partitions()}
    assert capacity["gpu"].gpu_types["a100"]["used"] == 2  # the injected job holds its GPUs
    slurm.cancel(str(live))
    assert sim.job(live)["state"] == "CANCELLED"  # no runner: the cancel lands at once
    with pytest.raises(ValueError):
        sim.inject_job(submit_time=now, state="RUNNING")  # needs start_time
    with pytest.raises(ValueError):
        sim.inject_job(submit_time=now, state="BOGUS")


def test_private_data_jobs_hides_other_users(tmp_path: Path) -> None:
    now = time.time()
    for private in ("none", "jobs"):
        with sim_cluster(tmp_path / private, private_data=private) as sim:
            _, sched, _ = sim_scheduler(sim)
            sim.inject_job(
                user="alice", submit_time=now - 600, start_time=now - 500, end_time=now - 100, gpus=1
            )
            sim.inject_job(
                user=sim.user, submit_time=now - 600, start_time=now - 500, end_time=now - 100, gpus=1
            )
            sim.inject_job(user="bob", submit_time=now - 60, start_time=now - 50, gpus=1, state="RUNNING")
            users = {
                r.user for r in sched.usage_segments(account="lab", users=None, since=_ago(3600), until=None)
            }
            queue = {j.user for j in sched.query_account_queue("lab").jobs}
            if private == "none":
                assert users == {"alice", sim.user, "bob"} and queue == {"bob"}
                assert sched.probe_capabilities().private_data == "none"
            else:
                assert users == {sim.user} and queue == set()
                assert sched.probe_capabilities().private_data == "jobs"


def _ago(seconds: float) -> str:
    from datetime import UTC, datetime, timedelta

    from hpclab.domain.clock import iso

    stamp = iso(datetime.now(UTC) - timedelta(seconds=seconds))
    assert stamp is not None
    return stamp


def test_injected_requeue_segments_in_sacct_duplicates(
    sim: SimCluster, slurm: SlurmScheduler, run_cli
) -> None:
    now = float(int(time.time()))
    job = sim.inject_job(
        submit_time=now - 5000, start_time=now - 3000, end_time=now - 1000, gpus=4, state="COMPLETED",
        segments=[{"start": now - 4800, "end": now - 4200, "state": "PREEMPTED", "node_list": ["sim-a100-02"]}],
    )  # fmt: skip
    plain = run_cli("sacct", "-X", "-n", "-P", "-j", str(job), "--format=JobID,State,NodeList")
    assert plain.stdout == f"{job}|COMPLETED|sim-a100-01\n"
    dup = run_cli("sacct", "-X", "-D", "-n", "-P", "-j", str(job), "--format=JobID,State,NodeList,ElapsedRaw")
    assert dup.stdout.splitlines() == [
        f"{job}|PREEMPTED|sim-a100-02|600",
        f"{job}|COMPLETED|sim-a100-01|2000",
    ]
    segs = slurm.accounting_segments(str(job))
    assert [(s.restart_count, s.state, s.elapsed_s) for s in segs] == [
        (0, SchedulerState.PREEMPTED, 600),
        (1, SchedulerState.COMPLETED, 2000),
    ]


# ---------------------------------------------------------------------------
# live jobs: requeue, signals, nvidia-smi
# ---------------------------------------------------------------------------

pytest_sim = pytest.mark.sim


@pytest_sim
def test_requeue_fault_reruns_the_job_as_a_new_segment(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(requeue_next=1)
    script = write_script(
        'echo "restart=$SLURM_RESTART_COUNT"; sleep 0.8', "--gres=gpu:1", "--open-mode=append"
    )
    job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
    final = finished(sim, job_id)
    assert final["state"] == "COMPLETED"
    record = sim.state.job(int(job_id))
    assert record is not None and record.restart_count == 1 and len(record.segments) == 1
    assert record.segments[0]["state"] == "PREEMPTED"
    out = (workdir / f"slurm-{job_id}.out").read_text()
    assert "restart=0" in out and "restart=1" in out and "DUE TO JOB REQUEUE" in out
    segs = slurm.accounting_segments(job_id)
    assert [s.state for s in segs] == [SchedulerState.PREEMPTED, SchedulerState.COMPLETED]
    assert [s.restart_count for s in segs] == [0, 1] and all(s.alloc_gpus == 1 for s in segs)
    assert len(sim.jobs()) == 1  # one submission, two segments


@pytest_sim
def test_requeue_restamps_the_submit_time_like_slurm(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path, run_cli
) -> None:
    """Slurm's batch_requeue_fini stamps the requeued job submitted *now* (one second later when
    that is the same second): accounting keeps one record per submission, the old one keeping
    the old submit time."""
    sim.set_faults(requeue_next=1)
    job_id = slurm.submit(str(write_script("sleep 0.8", "--gres=gpu:1")), workdir=str(workdir)).job_id
    first = sim.state.job(int(job_id))
    assert first is not None
    submitted = first.submit_time
    assert finished(sim, job_id)["state"] == "COMPLETED"
    record = sim.state.job(int(job_id))
    assert record is not None and record.restart_count == 1
    assert record.segments[0]["submit"] == submitted
    assert int(record.submit_time) >= int(submitted) + 1  # a distinct second: a distinct record
    assert record.submit_time in (record.segments[0]["end"], float(int(submitted) + 1))
    dup = run_cli("sacct", "-X", "-D", "-n", "-P", "-j", job_id, "--format=JobID,Submit,State")
    rows = [line.split("|") for line in dup.stdout.splitlines()]
    assert [r[2] for r in rows] == ["PREEMPTED", "COMPLETED"]
    assert int(rows[0][1]) == int(submitted) and int(rows[1][1]) == int(record.submit_time)
    plain = run_cli("sacct", "-X", "-n", "-P", "-j", job_id, "--format=JobID,Submit,State")
    assert plain.stdout.split("|")[1] == str(int(record.submit_time))
    segs = slurm.accounting_segments(job_id)
    assert [s.state for s in segs] == [SchedulerState.PREEMPTED, SchedulerState.COMPLETED]
    assert segs[0].submit_time is not None and segs[1].submit_time is not None
    assert segs[0].submit_time < segs[1].submit_time


@pytest_sim
def test_node_failure_requeues_a_requeueable_job(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    """JobRequeue=1 (the default): a batch job whose node fails is requeued - a NODE_FAIL
    segment, then the script runs again with SLURM_RESTART_COUNT=1."""
    sim.set_faults(node_fail_next=1)
    script = write_script(
        'echo "restart=$SLURM_RESTART_COUNT"; sleep 0.8', "--gres=gpu:1", "--open-mode=append"
    )
    job_id = slurm.submit(str(script), workdir=str(workdir)).job_id
    assert finished(sim, job_id)["state"] == "COMPLETED"
    record = sim.state.job(int(job_id))
    assert record is not None and record.restart_count == 1
    assert record.segments[0]["state"] == "NODE_FAIL" and record.segments[0]["exit_signal"] == 0
    out = (workdir / f"slurm-{job_id}.out").read_text()
    assert "restart=0" in out and "restart=1" in out
    segs = slurm.accounting_segments(job_id)
    assert [s.state for s in segs] == [SchedulerState.NODE_FAIL, SchedulerState.COMPLETED]
    assert [s.restart_count for s in segs] == [0, 1]


@pytest_sim
def test_node_failure_of_a_no_requeue_job_is_final(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(node_fail_next=1)
    job_id = slurm.submit(str(write_script("sleep 5", "--no-requeue")), workdir=str(workdir)).job_id
    assert finished(sim, job_id)["state"] == "NODE_FAIL"
    record = sim.state.job(int(job_id))
    assert record is not None and record.restart_count == 0 and record.segments == []
    assert [s.state for s in slurm.accounting_segments(job_id)] == [SchedulerState.NODE_FAIL]


@pytest_sim
def test_no_requeue_job_ends_preempted(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(requeue_next=1)
    job_id = slurm.submit(str(write_script("sleep 5", "--no-requeue")), workdir=str(workdir)).job_id
    assert finished(sim, job_id)["state"] == "PREEMPTED"
    assert [s.state for s in slurm.accounting_segments(job_id)] == [SchedulerState.PREEMPTED]


@pytest_sim
def test_signal_batch_shell_and_pending_refusal(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    # The batch shell traps USR1 and records it; its child must NOT receive the signal (--batch).
    body = (
        "trap 'echo shell-got-usr1 >> sig.log' USR1\n"
        "python3 -c 'import signal,time,sys\n"
        'signal.signal(signal.SIGUSR1, lambda *a: (open("sig.log","a").write("child-got-usr1\\n"), sys.exit(3)))\n'
        'open("ready","w").close()\n'
        "time.sleep(2.0)' &\n"
        "wait $!; wait $!\n"
    )
    job_id = slurm.submit(str(write_script(body)), workdir=str(workdir)).job_id
    wait_for(lambda: (workdir / "ready").exists(), what="child ready")
    assert slurm.signal_job(job_id, "USR1").accepted
    assert finished(sim, job_id)["state"] == "COMPLETED"
    log = (workdir / "sig.log").read_text()
    assert "shell-got-usr1" in log and "child-got-usr1" not in log

    held = slurm.submit(str(write_script("true", "--hold", name="held.sbatch")), workdir=str(workdir)).job_id
    resp = slurm.signal_job(held, "USR1")
    assert not resp.accepted and "pending" in resp.raw.lower()
    assert sim.job(held)["state"] == "PENDING"  # a signal is not a cancel
    slurm.cancel(held)


@pytest_sim
def test_fake_nvidia_smi_is_per_job(
    sim: SimCluster, slurm: SlurmScheduler, write_script, workdir: Path
) -> None:
    sim.set_faults(gpu_util=37.0)
    body = (
        "nvidia-smi -L > list.txt\n"
        "nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw "
        "--format=csv,noheader,nounits > query.txt\n"
        "nvidia-smi > header.txt\n"
    )
    job_id = slurm.submit(
        str(write_script(body, "--gres=gpu:h100:2", "--partition=scavenge")), workdir=str(workdir)
    ).job_id
    assert finished(sim, job_id)["state"] == "COMPLETED"
    listing = (workdir / "list.txt").read_text().splitlines()
    assert len(listing) == 2 and all("NVIDIA H100 80GB HBM3 (UUID: GPU-sim-" in ln for ln in listing)
    rows = [r.split(", ") for r in (workdir / "query.txt").read_text().splitlines()]
    assert len(rows) == 2 and all(len(r) == 7 and r[3] == "37" for r in rows)
    assert "/" in rows[0][0] and ":" in rows[0][0]  # 2026/09/27 12:00:00.123
    assert "CUDA Version: 12.8" in (workdir / "header.txt").read_text()

    cpu_only = slurm.submit(
        str(write_script("nvidia-smi -L; echo rc=$? > rc.txt", name="cpu.sbatch")), workdir=str(workdir)
    ).job_id
    finished(sim, cpu_only)
    assert (workdir / "rc.txt").read_text().strip() == "rc=6"  # GPU node, but no GPU allocated
    # the login node (what hpclab's transport sees) has no nvidia-smi
    assert shutil.which("nvidia-smi", path=str(sim.bin_dir)) is None


@pytest_sim
def test_real_wrapper_guard_and_telemetry_pass_in_the_sim(tmp_path: Path) -> None:
    """The KNOWN GAP: with a per-job nvidia-smi the wrapper's GPU guard passes (torch absent ->
    unchecked, not failed) and the telemetry sidecar records samples."""
    with sim_cluster(tmp_path, gpu_util=55.0) as sim:
        transport, sched, cluster = sim_scheduler(sim)
        sched.probe_capabilities()
        project = tmp_path / "project"
        project.mkdir()
        (project / "train.py").write_text(
            "import os, time\n"
            "time.sleep(1.5)\n"
            "open(os.path.join(os.environ['HPCLAB_OUTPUT_DIR'], 'done.txt'), 'w').write('ok')\n"
        )
        store = SnapshotStore(tmp_path / "store")
        captured = capture_snapshot(project, CodeOptions(), store)
        remote_root = cluster.remote_root()
        snapshot_path = publish_snapshot(transport, store, captured.ref, remote_root)
        run_spec = RunSpec(
            experiment="guard",
            code=captured.ref,
            entrypoint=EntrypointSpec(argv=["python", "train.py"]),
            argv=[sys.executable, "train.py"],
            runtime=RuntimeSpec(python=sys.executable, guard=True),  # the sidecar samples once at start
            outputs=[OutputDecl(name="done", path="{output_dir}/done.txt", role="other")],
        )
        plan = ExecutionPlan(
            run_id="run_GUARD",
            run_spec_hash=run_spec.spec_hash(),
            cluster_id=cluster.cluster_id,
            profile_revision=cluster.revision,
            resources=ResourceSpec(partition="gpu", gpus=1, gpu_type="a100", time="00:05:00", account="lab"),
            remote_root=remote_root,
            snapshot_path=snapshot_path,
        )
        token = "k7m2p9q4r8s3"
        layout = attempt_layout(remote_root, "att_GUARD")
        script = render_job_script(
            run_spec=run_spec,
            plan=plan,
            cluster=cluster,
            run_id="run_GUARD",
            attempt_id="att_GUARD",
            token=token,
            job_name=job_name_for("guard", token),
            layout=layout,
        )
        for d in layout.dirs_to_create():
            os.makedirs(d, exist_ok=True)
        Path(layout.script).write_text(script)
        job_id = sched.submit(layout.script, workdir=layout.dir, receipt_path=layout.receipt).job_id
        assert finished(sim, job_id, timeout=25)["state"] == "COMPLETED", Path(layout.stderr).read_text()
        status = json.loads(Path(layout.status_json).read_text())
        assert status["state"] == "exited" and status["exit_code"] == 0, status
        assert "HPCLAB_GUARD_FAILED" not in Path(layout.stderr).read_text()
        env = json.loads(Path(layout.env_json).read_text())
        assert env["nvidia_smi"] and len(env["nvidia_smi"]["gpus"]) == 1
        assert env["nvidia_smi"]["gpus"][0]["name"] == "NVIDIA A100-SXM4-80GB"
        telemetry = Path(layout.dir) / "telemetry" / "gpu.csv"
        samples = [ln for ln in telemetry.read_text().splitlines() if ln.strip()]
        assert samples and all(ln.split(", ")[3] == "55" for ln in samples)
        assert (Path(layout.output_dir) / "done.txt").read_text() == "ok"


def test_test_only_through_the_sim(sim: SimCluster, slurm: SlurmScheduler) -> None:
    est = slurm.test_only(JobShape(partition="gpu", gpus=2, gpu_type="a100", time="00:30:00", account="lab"))
    assert est.partition == "gpu" and est.start_time is not None and est.job_id
    assert sim.jobs() == []  # nothing queued
