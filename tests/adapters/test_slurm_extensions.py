"""Wave-3 SlurmScheduler extensions against scripted, real-format Slurm output.

signal_job, cluster_config, qos_limits, association_limits, partition_details, test_only,
accounting_segments, usage_segments, query_account_queue and the new capability fields.
"""

from __future__ import annotations

import pytest
from slurm_fake import ENV, FakeTransport, load, probe_ok

from hpclab.adapters.base import CommandFailed, ConnectionLost, SubmissionRejected, TransportError
from hpclab.adapters.slurm import (
    ACTIVE_QUEUE_STATES,
    FS,
    JobShape,
    SlurmScheduler,
    gpu_types_from_tres,
    normalize_signal,
    parse_config_seconds,
    private_data_flags,
    tres_gpu_limit,
)
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import ErrorCode, InvalidArgument
from hpclab.domain.states import SchedulerState
from hpclab.profiles.schema import ClusterProfile, PartitionRule, ResolvedCluster, UserOverlay

CONFIG_PRIVATE = (
    load("scontrol_show_config.txt")
    .replace("MinJobAge               = 300 sec", "MinJobAge               = 30 sec")
    .replace(
        "ClusterName             = cluster1",
        "ClusterName             = cluster1\nPrivateData             = accounts,jobs,usage\nMailProg                = (null)\nJobRequeue              = 1\nJobAcctGatherType       = jobacct_gather/cgroup",
    )
)


@pytest.fixture
def cluster() -> ResolvedCluster:
    profile = ClusterProfile(
        cluster_id="cluster1", display_name="ExampleHPC", partitions={"gpu": PartitionRule(default=True)}
    )
    return ResolvedCluster(
        profile=profile, overlay=UserOverlay(cluster_id="cluster1", user="jdoe", account="mylab")
    )


@pytest.fixture
def fake() -> FakeTransport:
    t = FakeTransport()
    probe_ok(t)
    return t


@pytest.fixture
def sched(fake: FakeTransport, cluster: ResolvedCluster) -> SlurmScheduler:
    return SlurmScheduler(fake, cluster, clock=FakeClock())


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_limit_and_config_helpers() -> None:
    assert tres_gpu_limit("cpu=64,gres/gpu=8") == 8
    assert tres_gpu_limit("cpu=64") is None  # GPUs not limited: never "0 GPUs allowed"
    assert tres_gpu_limit("gres/gpu:h100=2,gres/gpu:a100=2") == 4
    assert tres_gpu_limit("") is None
    assert gpu_types_from_tres("gres/gpu=4,gres/gpu:h100=4") == {"h100": 4}
    assert gpu_types_from_tres("cpu=1") == {}
    assert private_data_flags(None) is None  # unknown
    assert private_data_flags("none") == frozenset()
    assert private_data_flags("Accounts,JOBS") == {"accounts", "jobs"}
    assert parse_config_seconds("300 sec") == 300 and parse_config_seconds("n/a") is None
    assert normalize_signal("SIGUSR1") == "USR1" and normalize_signal("usr2") == "USR2"
    for bad in ("10", "KILL", "STOP", "USR1; rm -rf /"):
        with pytest.raises(InvalidArgument):
            normalize_signal(bad)


# ---------------------------------------------------------------------------
# capabilities + config
# ---------------------------------------------------------------------------


def test_probe_populates_new_capability_fields(sched: SlurmScheduler) -> None:
    caps = sched.probe_capabilities()
    assert caps.cluster_name == "cluster1"
    assert caps.min_job_age_s == 300
    assert caps.accounting_store_flags == "job_comment,job_script"
    assert caps.private_data is None  # not in this config dump: unknown, not "none"


def test_probe_reads_private_data(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake, config=CONFIG_PRIVATE)
    caps = SlurmScheduler(fake, cluster).probe_capabilities()
    assert caps.private_data == "accounts,jobs,usage" and caps.min_job_age_s == 30
    assert private_data_flags(caps.private_data) == {"accounts", "jobs", "usage"}


def test_probe_without_controller_config_leaves_fields_unknown(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake)
    fake.on("scontrol", "", "slurm_load_ctl_conf error: Unable to contact slurm controller", rc=1,
            when=lambda a: "config" in a)  # fmt: skip
    caps = SlurmScheduler(fake, cluster).probe_capabilities()
    assert caps.cluster_name is None and caps.private_data is None and caps.min_job_age_s is None


def test_cluster_config_selected_keys(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake, config=CONFIG_PRIVATE)
    conf = SlurmScheduler(fake, cluster).cluster_config()
    assert fake.last("scontrol") == [*ENV, "scontrol", "show", "config"]
    assert conf["PrivateData"] == "accounts,jobs,usage"
    assert conf["MailProg"] is None  # "(null)"
    assert conf["JobRequeue"] == "1" and conf["MinJobAge"] == "30 sec"
    assert conf["AccountingStorageTRES"].endswith("gres/gpu")
    assert conf["SLURM_VERSION"] == "23.02.7" and conf["ClusterName"] == "cluster1"


def test_cluster_config_failures_are_typed(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    fake.on("scontrol", "", "Unable to contact slurm controller (connect failure)", rc=1)
    with pytest.raises(CommandFailed) as exc:
        SlurmScheduler(fake, cluster).cluster_config()
    assert exc.value.code == ErrorCode.UNAVAILABLE
    missing = FakeTransport()  # no scontrol at all -> 127
    with pytest.raises(CommandFailed) as exc2:
        SlurmScheduler(missing, cluster).cluster_config()
    assert exc2.value.code == ErrorCode.UNSUPPORTED


# ---------------------------------------------------------------------------
# signal_job
# ---------------------------------------------------------------------------


def test_signal_job_argv_and_answers(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("scancel", "")
    resp = sched.signal_job("4242", "SIGUSR1")
    assert resp.accepted and fake.last("scancel") == [*ENV, "scancel", "--batch", "--signal=USR1", "4242"]
    sched.signal_job("4242", "TERM", batch=False)
    assert fake.last("scancel") == [*ENV, "scancel", "--full", "--signal=TERM", "4242"]
    fake.on("scancel", "", "scancel: error: Kill job error on job id 4242: Job is pending execution", rc=1)
    resp = sched.signal_job("4242", "USR1")
    assert not resp.accepted and "pending" in resp.raw
    fake.on(
        "scancel",
        "",
        "scancel: error: Kill job error on job id 4242: Socket timed out on send/recv operation",
        rc=1,
    )
    with pytest.raises(ConnectionLost) as exc:
        sched.signal_job("4242", "USR1")
    assert exc.value.maybe_executed and exc.value.retryable
    with pytest.raises(InvalidArgument):
        sched.signal_job("4242 4243", "USR1")
    with pytest.raises(InvalidArgument):
        sched.signal_job("4242", "9")


# ---------------------------------------------------------------------------
# sacctmgr: qos + association limits
# ---------------------------------------------------------------------------

QOS_ROWS = (
    "normal|2-00:00:00|cpu=64,gres/gpu=8|4|10|||||cluster|0\n"
    "interactive|04:00:00|gres/gpu=1|1|1|gres/gpu=1|gres/gpu=16|20|normal,low|requeue|100\n"
)


def test_qos_limits(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sacctmgr", QOS_ROWS)
    limits = {q.name: q for q in sched.qos_limits(["normal", "interactive"])}
    argv = fake.last("sacctmgr")
    assert argv[2:8] == ["sacctmgr", "--noheader", "--parsable2", "show", "qos", "where"]
    assert "name=normal,interactive" in argv and argv[-1].startswith("format=Name,MaxWall,MaxTRESPU")
    n = limits["normal"]
    assert (n.max_wall_s, n.max_gpus_per_user, n.max_jobs_per_user, n.max_submit_per_user) == (
        172800,
        8,
        4,
        10,
    )
    assert n.max_gpus_per_job is None and n.grp_gpus is None and n.preempt == ()  # unset = no limit
    i = limits["interactive"]
    assert i.max_gpus_per_job == 1 and i.grp_gpus == 16 and i.grp_jobs == 20
    assert i.preempt == ("normal", "low") and i.preempt_mode == "requeue" and i.priority == 100
    sched.qos_limits()
    assert "where" not in fake.last("sacctmgr")


@pytest.mark.parametrize(
    ("stderr", "rc", "code"),
    [
        (
            "sacctmgr: error: Access denied: PrivateData prevents you from viewing QOS",
            1,
            ErrorCode.UNAVAILABLE,
        ),
        ("You are not running a supported accounting_storage plugin", 1, ErrorCode.UNSUPPORTED),
        ("sacctmgr: error: Problem talking to the database: Connection refused", 1, ErrorCode.UNAVAILABLE),
    ],
)
def test_hidden_or_failed_limits_raise(
    fake: FakeTransport, sched: SlurmScheduler, stderr: str, rc: int, code: ErrorCode
) -> None:
    # Hidden limits are "unavailable", never an empty list that would read as "no limits".
    fake.on("sacctmgr", "", stderr, rc=rc)
    with pytest.raises(CommandFailed) as exc:
        sched.qos_limits()
    assert exc.value.code == code
    if "Access denied" in stderr:
        assert exc.value.details["reason"] == "access_denied"
    with pytest.raises(CommandFailed):
        sched.association_limits("jdoe")


def test_association_limits_keep_only_this_cluster(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on(
        "sacctmgr",
        "cluster1|mylab|jdoe||normal,interactive|normal|gres/gpu=32|||4|20|1-00:00:00\n"
        "other|mylab|jdoe||normal|normal||||||\n"
        "cluster1|mylab||||||||||\n",
    )
    rows = sched.association_limits("jdoe")
    assert len(rows) == 1
    a = rows[0]
    assert (a.account, a.qos, a.default_qos, a.grp_gpus, a.max_jobs, a.max_submit, a.max_wall_s) == (
        "mylab",
        ("normal", "interactive"),
        "normal",
        32,
        4,
        20,
        86400,
    )
    assert a.max_gpus_per_job is None
    assert "user=jdoe" in fake.last("sacctmgr")


# ---------------------------------------------------------------------------
# partition details
# ---------------------------------------------------------------------------


def test_partition_details(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("scontrol", load("scontrol_partitions.txt"), when=lambda a: "partition" in a)
    parts = {p.name: p for p in sched.partition_details()}
    assert fake.last("scontrol") == [*ENV, "scontrol", "--oneliner", "show", "partition"]
    gpu, lo, inter = parts["gpu"], parts["scavenge"], parts["interactive"]
    assert gpu.default and gpu.max_time_s == 7 * 86400 and gpu.preempt_mode == "OFF" and gpu.total_gpus == 48
    assert gpu.default_time_s is None and gpu.grace_time_s == 0 and gpu.priority_tier == 10
    assert lo.preempt_mode == "REQUEUE" and lo.priority_tier == 1
    assert inter.default_time_s == 3600 and inter.qos == "interactive"
    assert parts["cpu"].total_gpus is None
    assert parts["gpu-new"].state == "INACTIVE" and parts["gpu-new"].max_time_s is None


def test_partition_details_failure_raises(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("scontrol", "", "slurm_load_partitions: Unable to contact slurm controller", rc=1,
            when=lambda a: "partition" in a)  # fmt: skip
    with pytest.raises(CommandFailed) as exc:
        sched.partition_details()
    assert exc.value.code == ErrorCode.UNAVAILABLE


# ---------------------------------------------------------------------------
# test_only
# ---------------------------------------------------------------------------


def test_test_only_argv_and_estimate(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on(
        "sbatch",
        "",
        "sbatch: Job 4999 to start at 1790003600 using 16 processors on nodes gpu-node-[1-2] in partition gpu\n",
        when=lambda a: "--test-only" in a,
    )
    est = sched.test_only(
        {
            "partition": "gpu",
            "gpus": 4,
            "gpu_type": "h100",
            "time": "02:00:00",
            "nodes": 2,
            "account": "mylab",
        }
    )
    argv = fake.last("sbatch")
    assert argv[:2] == ENV
    assert argv[2:8] == [
        "sbatch",
        "--test-only",
        "--wrap=true",
        "--job-name=hpclab-test-only",
        "--output=/dev/null",
        "--partition=gpu",
    ]
    assert "--account=mylab" in argv and "--nodes=2" in argv and "--gres=gpu:h100:2" in argv
    assert argv[-1] == "--time=02:00:00"
    assert est.job_id == "4999" and est.start_time == "2026-09-21T15:13:20.000Z"
    assert est.partition == "gpu" and est.nodes == "gpu-node-[1-2]" and est.processors == 16
    sched.test_only(JobShape(partition="gpu", gpus=1, time="10"))
    assert "--gres=gpu:1" in fake.last("sbatch")


def test_test_only_failures(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on(
        "sbatch",
        "",
        "sbatch: error: Batch job submission failed: Requested node configuration is not available\n",
        rc=1,
        when=lambda a: "--test-only" in a,
    )
    with pytest.raises(SubmissionRejected) as exc:
        sched.test_only(JobShape(partition="gpu", gpus=64))
    assert exc.value.details["reason_code"] == "node_config_unavailable"
    fake.on(
        "sbatch",
        "",
        "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n",
        rc=1,
        when=lambda a: "--test-only" in a,
    )
    with pytest.raises(TransportError) as exc2:
        sched.test_only(JobShape(partition="gpu"))
    assert not isinstance(exc2.value, SubmissionRejected)
    for bad in (
        {"time": "soon"},
        {"mem": "64 GB"},
        {"partition": "gpu --wrap=rm"},
        {"bogus": 1},
        {"nodes": 0},
    ):
        with pytest.raises(InvalidArgument):
            sched.test_only(bad)


# ---------------------------------------------------------------------------
# accounting segments + usage segments
# ---------------------------------------------------------------------------


def _sacct_row(
    job: str,
    state: str,
    start: str,
    end: str,
    elapsed: str,
    nodes: str,
    tres: str,
    name: str = "hl.x.abcd2345",
) -> str:
    fields = [job, job, state, "0:0", "jdoe", "mylab", "scavenge", "1790000000", start, end, elapsed, "1440",
              nodes, tres, "/w", name, "hpclab:abcd2345efgh", "sbatch job.sbatch"]  # fmt: skip
    return FS.join(fields)


def test_accounting_segments_orders_requeue_records(fake: FakeTransport, sched: SlurmScheduler) -> None:
    tres = "billing=8,cpu=8,gres/gpu=4,gres/gpu:h200=4,mem=64G,node=1"
    rows = [
        _sacct_row("4300", "COMPLETED", "1790001000", "1790004600", "3600", "gpu-node-2", tres),
        _sacct_row("4300", "PREEMPTED", "1790000100", "1790000700", "600", "gpu-node-6", tres),
        _sacct_row("43001", "COMPLETED", "1790000100", "1790000200", "100", "gpu-node-1", tres),
    ]
    fake.on("sacct", "\n".join(rows) + "\n", when=lambda a: "--duplicates" in a)
    segs = sched.accounting_segments("4300")
    argv = fake.last("sacct")
    assert "--duplicates" in argv and "--allocations" in argv and argv[-1] == "--jobs=4300"
    assert [(s.restart_count, s.state, s.node_list, s.alloc_gpus) for s in segs] == [
        (0, SchedulerState.PREEMPTED, "gpu-node-6", 4),
        (1, SchedulerState.COMPLETED, "gpu-node-2", 4),
    ]


USAGE_ROWS = "\n".join(
    FS.join(f)
    for f in [
        # requeued job: two records, typed + untyped TRES (4 GPUs, not 8)
        ["4300", "4300", "jdoe", "mylab", "scavenge", "normal", "PREEMPTED", "0:15", "1790000000",
         "1790000100", "1790000700", "600", "1440", "gpu-node-6", "billing=8,cpu=8,gres/gpu=4,gres/gpu:h100=4,mem=64G,node=1", "sweep a"],
        ["4300", "4300", "jdoe", "mylab", "scavenge", "normal", "COMPLETED", "0:0", "1790000000",
         "1790001000", "1790004600", "3600", "1440", "gpu-node-2", "billing=8,cpu=8,gres/gpu=4,gres/gpu:h100=4,mem=64G,node=1", "sweep a"],
        # typed-only TRES: fall back to the typed sum
        ["4301_2", "4303", "alice", "mylab", "gpu", "normal", "TIMEOUT", "0:0", "1790000000",
         "1790000100", "1790003700", "3600", "60", "gpu-node-1", "billing=4,cpu=4,gres/gpu:h200=2,mem=32G,node=1", "arr"],
        # cancelled while pending: Start == End, no nodes, no TRES -> 0 GPUs, known
        ["4302", "4302", "bob", "mylab", "gpu", "normal", "CANCELLED by 1234", "0:0", "1790000000",
         "1790000500", "1790000500", "0", "60", "None assigned", "", "never ran"],
        # still running
        ["4303", "4304", "bob", "mylab", "gpu", "normal", "RUNNING", "0:0", "1790000000",
         "1790002000", "Unknown", "1200", "UNLIMITED", "gpu-node-3", "billing=2,cpu=2,gres/gpu=1,mem=8G,node=1", "live"],
    ]
) + "\n"  # fmt: skip


def test_usage_segments(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sacct", USAGE_ROWS, when=lambda a: "--duplicates" in a)
    rows = sched.usage_segments(account="mylab", users=None, since="2026-09-01T12:00:00Z", until=None)
    argv = fake.last("sacct")
    assert "--duplicates" in argv and "--allusers" in argv and "--accounts=mylab" in argv
    assert argv[-1].startswith("--format=JobID,JobIDRaw,User,Account,Partition,QOS,State,ExitCode")
    assert [(r.job_id, r.state, r.gpus, r.gpus_known) for r in rows] == [
        ("4300", "PREEMPTED", 4, True),
        ("4300", "COMPLETED", 4, True),
        ("4301_2", "TIMEOUT", 2, True),
        ("4302", "CANCELLED", 0, True),
        ("4303", "RUNNING", 1, True),
    ]
    first, _, arr, never, live = rows
    assert first.gpu_types == (("h100", 4),) and first.exit_signal == 15 and first.time_limit_s == 1440 * 60
    assert arr.job_id_raw == "4303" and arr.gpu_types == (("h200", 2),)
    assert never.node_list is None and never.raw_state == "CANCELLED by 1234"
    assert live.end_time is None and live.time_limit_s is None
    sched.usage_segments(
        account=None, users=["alice"], since="2026-08-20T00:00:00Z", until="2026-08-25T00:00:00Z"
    )
    argv = fake.last("sacct")
    assert (
        "--user=alice" in argv and "--allusers" not in argv and any(a.startswith("--endtime=") for a in argv)
    )


def test_usage_segments_refuse_untracked_gpus(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake, config=load("scontrol_show_config.txt").replace(",gres/gpu", ""))
    with pytest.raises(TransportError) as exc:
        SlurmScheduler(fake, cluster).usage_segments(
            account=None, users=None, since="2026-09-01T12:00:00Z", until=None
        )
    assert exc.value.code == ErrorCode.UNSUPPORTED


def test_usage_segments_gpu_count_unknown_when_tracking_unknown(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake, config=load("scontrol_show_config.txt").replace("AccountingStorageTRES", "X-Unrelated"))
    row = FS.join(["9", "9", "jdoe", "mylab", "gpu", "normal", "COMPLETED", "0:0", "1790000000",
                   "1790000100", "1790000200", "100", "60", "gpu-node-1", "billing=8,cpu=8,mem=8G,node=1", "x"])  # fmt: skip
    fake.on("sacct", row + "\n", when=lambda a: "--duplicates" in a)
    (seg,) = SlurmScheduler(fake, cluster).usage_segments(
        account=None, users=None, since="2026-09-01T12:00:00Z", until=None
    )
    assert seg.gpus == 0 and seg.gpus_known is False


# ---------------------------------------------------------------------------
# account queue
# ---------------------------------------------------------------------------


def test_query_account_queue(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", load("squeue_jobs.txt"), when=lambda a: any(x.startswith("--account=") for x in a))
    res = sched.query_account_queue("mylab")
    argv = fake.last("squeue")
    assert f"--states={ACTIVE_QUEUE_STATES}" in argv and "--account=mylab" in argv and "--noheader" in argv
    assert res.complete and {j.job_id for j in res.jobs} >= {"4242", "4243"}
    fake.on("squeue", "", "slurm_load_jobs error: Unable to contact slurm controller", rc=1,
            when=lambda a: any(x.startswith("--account=") for x in a))  # fmt: skip
    with pytest.raises(CommandFailed):
        sched.query_account_queue("mylab")
    with pytest.raises(InvalidArgument):
        sched.query_account_queue("in grai")
