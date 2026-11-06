"""Pure pieces of the simulator: formats, config/faults, FIFO planning."""

from __future__ import annotations

import time

import pytest

from hpclab.sim.config import NodeConfig, PartitionConfig, SimConfig, SimFaults
from hpclab.sim.formats import (
    FieldSpec,
    compress_hostlist,
    compress_ranges,
    elapsed_str,
    expand_hostlist,
    format_mem,
    parse_long_format,
    parse_mem_mb,
    parse_short_format,
    parse_time_limit,
    parse_timespec,
    render_value,
    secs2time_str,
    slurm_time,
)
from hpclab.sim.jobrunner import expand_pattern, job_environment
from hpclab.sim.scheduler import dependency_status, estimate_start, plan
from hpclab.sim.state import SimJob
from hpclab.sim.transport import command_name

# ---------------------------------------------------------------------------
# formats
# ---------------------------------------------------------------------------


def test_slurm_time_honors_time_format() -> None:
    assert slurm_time(1790000000.7, {"SLURM_TIME_FORMAT": "%s"}) == "1790000000"
    assert slurm_time(None, {"SLURM_TIME_FORMAT": "%s"}) == "Unknown"
    assert slurm_time(0, {}, missing="N/A") == "N/A"
    standard = slurm_time(1790000000, {})
    assert len(standard) == 19 and standard[10] == "T"
    assert slurm_time(1790000000, {"SLURM_TIME_FORMAT": "%Y"}) == time.strftime(
        "%Y", time.localtime(1790000000)
    )


@pytest.mark.parametrize(
    ("secs", "squeue", "sacct"),
    [
        (5, "0:05", "00:00:05"),
        (754, "12:34", "00:12:34"),
        (3600, "1:00:00", "01:00:00"),
        (93784, "1-02:03:04", "1-02:03:04"),
        (None, "UNLIMITED", "UNLIMITED"),
    ],
)
def test_duration_styles(secs: int | None, squeue: str, sacct: str) -> None:
    assert secs2time_str(secs) == squeue
    assert elapsed_str(secs) == sacct


@pytest.mark.parametrize(
    ("raw", "secs"),
    [
        ("30", 1800),
        ("0:02", 2),
        ("1:30:00", 5400),
        ("2-00", 172800),
        ("1-02:03", 93780),
        ("1-02:03:04", 93784),
        ("UNLIMITED", None),
        ("0", None),
    ],
)
def test_parse_time_limit(raw: str, secs: int | None) -> None:
    assert parse_time_limit(raw) == secs


@pytest.mark.parametrize("raw", ["abc", "1:2:3:4", "-5", "1-"])
def test_parse_time_limit_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_time_limit(raw)


def test_memory_helpers() -> None:
    assert parse_mem_mb("64G") == 65536
    assert parse_mem_mb("512M") == 512
    assert parse_mem_mb("4096") == 4096
    assert parse_mem_mb("1T") == 1024 * 1024
    with pytest.raises(ValueError):
        parse_mem_mb("lots")
    assert format_mem(65536) == "64G" and format_mem(500) == "500M"


def test_hostlists_round_trip() -> None:
    names = ["sim-a100-01", "sim-a100-02", "sim-a100-03", "sim-a100-07", "login"]
    compressed = compress_hostlist(names)
    assert compressed == "sim-a100-[01-03,07],login"
    assert expand_hostlist(compressed) == names
    assert compress_hostlist(["sim-int-01"]) == "sim-int-01"
    assert compress_ranges([0, 1, 2, 5]) == "0-2,5"


def test_long_format_semantics() -> None:
    specs = parse_long_format("JobID:0|,Name,State:.8,Reason:5x")
    assert specs == [
        FieldSpec("JobID", 0, False, "|"),
        FieldSpec("Name", 20, False, ""),
        FieldSpec("State", 8, True, ""),
        FieldSpec("Reason", 5, False, "x"),
    ]
    assert render_value("1234", specs[0]) == "1234|"
    assert render_value("train", specs[1]) == "train" + " " * 15
    assert render_value("RUNNING", specs[2]) == " RUNNING"
    assert render_value("Resources", specs[3]) == "Resoux"  # cut to 5, then suffix


def test_short_format_keeps_literals() -> None:
    specs = parse_short_format("%.18i|%T %j%%")
    assert specs == [
        FieldSpec("i", 18, True, ""),
        "|",
        FieldSpec("T", 0, False, ""),
        " ",
        FieldSpec("j", 0, False, ""),
        "%",
    ]


def test_parse_timespec() -> None:
    now = 1_790_000_000.0
    assert parse_timespec("now", now) == now
    assert parse_timespec("now-3600", now) == now - 3600
    assert parse_timespec("now-2hours", now) == now - 7200
    assert parse_timespec("now+1days", now) == now + 86400
    assert parse_timespec("2026-09-01", now) > 0
    with pytest.raises(ValueError):
        parse_timespec("now-3fortnights", now)
    with pytest.raises(ValueError):
        parse_timespec("tomorrow", now)


def test_command_name_detection() -> None:
    assert command_name(["env", "SLURM_TIME_FORMAT=%s", "squeue", "-h"]) == "squeue"
    assert (
        command_name(["bash", "-s", "--", "/r", "/w", "env", "SLURM_TIME_FORMAT=%s", "sbatch", "x"])
        == "sbatch"
    )
    assert command_name(["/usr/bin/scancel", "1"]) == "scancel"
    assert command_name(["bash", "-s"]) == "bash"
    assert command_name([]) == ""


# ---------------------------------------------------------------------------
# config / faults
# ---------------------------------------------------------------------------


def test_default_config_matches_spec() -> None:
    cfg = SimConfig()
    gpu, inter, scav = cfg.partition("gpu"), cfg.partition("interactive"), cfg.partition("scavenge")
    assert gpu is not None and gpu.default and gpu.max_time_s == 2 * 86400 and len(gpu.nodes) == 4
    assert all((n := cfg.node(x)) is not None and n.gpus == 4 and n.gpu_type == "a100" for x in gpu.nodes)
    assert inter is not None and inter.max_time_s == 4 * 3600 and inter.max_gpus_per_user == 1
    assert scav is not None and scav.preemptible and scav.max_time_s == 86400
    assert all((n := cfg.node(x)) is not None and n.gpus == 8 and n.gpu_type == "h100" for x in scav.nodes)
    assert cfg.accounts == ("lab", "other")


def test_config_round_trip_and_validation() -> None:
    cfg = SimConfig().replace(min_job_age_s=0.5)
    assert SimConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown nodes"):
        SimConfig(partitions=(PartitionConfig(name="p", nodes=("ghost",)),))
    with pytest.raises(ValueError, match="unknown simulator config"):
        SimConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match="default_account"):
        SimConfig(default_account="nope")


def test_fault_coercion() -> None:
    assert SimFaults.coerce("sbatch_lose_reply", "2") == 2
    assert SimFaults.coerce("auth_required", "true") is True
    assert SimFaults.coerce("auth_required", "off") is False
    assert SimFaults.coerce("transport_timeout_cmds", "squeue, sacct") == ["squeue", "sacct"]
    assert SimFaults.coerce("accounting_lag_s", "none") is None
    assert SimFaults.coerce("gpu_util", "12.5") == 12.5
    assert SimFaults.coerce("sbatch_reject", "Invalid qos") == "Invalid qos"
    for key, value in [
        ("nope", 1),
        ("oom_next", -1),
        ("auth_required", "maybe"),
        ("pending_delay_s", "none"),
    ]:
        with pytest.raises(ValueError):
            SimFaults.coerce(key, value)


# ---------------------------------------------------------------------------
# scheduling (pure)
# ---------------------------------------------------------------------------


def _job(
    job_id: int, *, gpus: int = 1, partition: str = "gpu", state: str = "PENDING", **kw: object
) -> SimJob:
    job = SimJob(
        job_id=job_id,
        name=f"j{job_id}",
        user="u",
        uid=1,
        account="lab",
        partition=partition,
        qos="normal",
        submit_time=float(job_id),
        time_limit_s=600,
        gpus_per_node=gpus,
        state=state,
    )
    for key, value in kw.items():
        setattr(job, key, value)
    return job


def test_accounting_view_of_a_requeued_job_follows_its_records() -> None:
    """A requeued job is one accounting record per submission: until the requeue reaches
    slurmdbd (lagging view) sacct shows the previous record, then the new one (new submit time,
    the earlier segment listed by --duplicates)."""
    from hpclab.sim.cmd_sacct import accounting_view, segment_views

    seg = {"submit": 900.0, "start": 910.0, "end": 950.0, "state": "NODE_FAIL", "node_list": ["n1"]}
    job = _job(7, state="PENDING", submit_time=951.0, restart_count=1, segments=[dict(seg)])
    assert accounting_view(job, 899.0) is None
    running = accounting_view(job, 940.0)
    assert running is not None and running.state == "RUNNING" and running.submit_time == 900.0
    assert running.segments == [] and running.restart_count == 0 and running.node_list == ["n1"]
    requeued = accounting_view(job, 950.5)  # the requeue (at 950) arrived; its submit stamp is 951
    assert requeued is not None and requeued.state == "PENDING" and requeued.submit_time == 951.0
    earlier = segment_views(requeued, 950.5)
    assert [(v.state, v.submit_time, v.restart_count) for v in earlier] == [("NODE_FAIL", 900.0, 0)]


def _small_cfg() -> SimConfig:
    return SimConfig(
        nodes=(
            NodeConfig("n1", cpus=8, gpus=4, gpu_type="a100"),
            NodeConfig("n2", cpus=8, gpus=4, gpu_type="a100"),
        ),
        partitions=(
            PartitionConfig("gpu", ("n1", "n2"), max_time_s=3600, default=True, max_gpus_per_user=None),
        ),
    )


def test_plan_fifo_resources_then_priority() -> None:
    cfg = _small_cfg()
    jobs = [_job(1, gpus=4), _job(2, gpus=4), _job(3, gpus=4), _job(4, gpus=1)]
    decisions = {d.job_id: d for d in plan(cfg, jobs, {}, now=100.0)}
    assert decisions[1].alloc == {"n1": {"gpus": [0, 1, 2, 3], "cpus": 1, "mem": 4096}}
    assert decisions[2].alloc is not None and list(decisions[2].alloc) == ["n2"]
    assert decisions[3].reason == "Resources"
    assert decisions[4].reason == "Priority"  # strict FIFO: no backfill past the blocked head


def test_plan_packs_gpu_indices_on_shared_nodes() -> None:
    cfg = _small_cfg()
    running = _job(1, gpus=2, state="RUNNING", alloc={"n1": {"gpus": [0, 1], "cpus": 1, "mem": 1}})
    decisions = plan(cfg, [running, _job(2, gpus=2)], {}, now=100.0)
    assert decisions[0].alloc == {"n1": {"gpus": [2, 3], "cpus": 1, "mem": 4096}}


def test_plan_skips_drained_nodes_and_reports_unavailable() -> None:
    cfg = _small_cfg()
    admin = {"n1": ("drain", "bad disk"), "n2": ("down", None)}
    (decision,) = plan(cfg, [_job(1, gpus=1)], admin, now=100.0)
    assert decision.alloc is None and decision.reason == "ReqNodeNotAvail, UnavailableNodes:n[1-2]"
    (ok,) = plan(cfg, [_job(1, gpus=1)], {"n1": ("drain", "x")}, now=100.0)
    assert ok.alloc is not None and list(ok.alloc) == ["n2"]


def test_plan_holds_do_not_block_others() -> None:
    cfg = _small_cfg()
    dep_target = _job(1, gpus=0, state="RUNNING")
    jobs = [
        dep_target,
        _job(2, dependency="afterok:1"),
        _job(3, hold=True),
        _job(4, time_limit_s=7200),
        _job(5),
    ]
    decisions = {d.job_id: d for d in plan(cfg, jobs, {}, now=100.0)}
    assert decisions[2].reason == "Dependency"
    assert decisions[3].reason == "JobHeldUser"
    assert decisions[4].reason == "PartitionTimeLimit"
    assert decisions[5].alloc is not None


def test_plan_pending_delay_and_user_gpu_limit() -> None:
    cfg = SimConfig()
    delayed = plan(cfg, [_job(1)], {}, now=1.5, pending_delay_s=5.0)
    assert delayed[0].reason == "Priority"
    running = _job(
        1, partition="interactive", state="RUNNING", alloc={"sim-int-01": {"gpus": [0], "cpus": 1, "mem": 1}}
    )
    (limited,) = plan(cfg, [running, _job(2, partition="interactive")], {}, now=100.0)
    assert limited.reason == "QOSMaxGRESPerUser"


@pytest.mark.parametrize(
    ("dependency", "dep_state", "status"),
    [
        ("afterok:1", "RUNNING", "Dependency"),
        ("afterok:1", "COMPLETED", None),
        ("afterok:1", "FAILED", "DependencyNeverSatisfied"),
        ("afterany:1", "FAILED", None),
        ("afternotok:1", "COMPLETED", "DependencyNeverSatisfied"),
        ("afternotok:1", "TIMEOUT", None),
        ("after:1", "RUNNING", None),
        ("after:1", "PENDING", "Dependency"),
        ("afterok:99", "RUNNING", None),  # purged dependency counts as satisfied
    ],
)
def test_dependency_status(dependency: str, dep_state: str, status: str | None) -> None:
    dep = _job(1, state=dep_state)
    job = _job(2, dependency=dependency)
    assert dependency_status(job, {1: dep, 2: job}) == status


def test_estimate_start_uses_running_time_limits() -> None:
    cfg = _small_cfg()
    running = _job(1, state="RUNNING", start_time=1000.0)
    pending = _job(2)
    assert estimate_start(cfg, pending, [running, pending], now=1100.0) == 1600.0
    assert estimate_start(cfg, running, [running], now=1100.0) is None


def test_expand_pattern_and_environment() -> None:
    job = _job(
        1234,
        gpus=2,
        state="RUNNING",
        name="train",
        node_list=["sim-a100-02"],
        alloc={"sim-a100-02": {"gpus": [2, 3], "cpus": 8, "mem": 1}},
        cpus_per_task=8,
        submit_dir="/w",
        start_time=100.0,
        env={"PATH": "/bin", "SLURM_JOB_ID": "stale", "HPCLAB_SIM_CALL_ID": "x", "KEEP": "1"},
    )
    assert expand_pattern("logs/%x-%j.%u.%%.out", job) == "logs/train-1234.u.%.out"
    env = job_environment(job, "simcluster")
    assert env["SLURM_JOB_ID"] == "1234" and env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["SLURM_GPUS_ON_NODE"] == "2" and env["SLURM_CPUS_PER_TASK"] == "8"
    assert env["SLURM_JOB_NODELIST"] == "sim-a100-02" and env["SLURM_RESTART_COUNT"] == "0"
    assert env["SLURM_JOB_ACCOUNT"] == "lab" and env["SLURM_SUBMIT_DIR"] == "/w" and env["KEEP"] == "1"
    assert "HPCLAB_SIM_CALL_ID" not in env
    assert env["SLURM_JOB_END_TIME"] == "700"
