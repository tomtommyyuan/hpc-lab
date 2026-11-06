"""Fakes for cluster_inspect unit tests (imported by name, pytest prepend mode): a scripted
login node and scheduler behind a real ServiceContext (ledger, registry, FakeClock)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hpclab.adapters.base import (
    CommandResult,
    NodeInfo,
    PartitionInfo,
    QueryResult,
    SchedulerCapabilities,
)
from hpclab.adapters.slurm import AssocLimit, PartitionDetail, QosLimit, ShapeEstimate
from hpclab.domain.clock import FakeClock
from hpclab.domain.models import JobObservation
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, StorageRole, UserOverlay
from hpclab.services.context import ServiceContext


@dataclass
class FakeLogin:
    """Scripted login node: ``id -un``, storage/runtime snippets, ``quota``."""

    cluster_id: str = "fake"
    user: str = "alice"
    id_error: Exception | None = None
    master: bool | None = True
    capacity_pct: int = 50
    tools: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "python3": ("/usr/bin/python3", "Python 3.11.4"),
            "uv": ("", ""),
            "rsync": ("/usr/bin/rsync", "rsync  version 3.2.7  protocol version 31"),
        }
    )
    quota_text: str = "Disk quotas for user alice (uid 1000): none\n"
    calls: list[list[str]] = field(default_factory=list)

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        self.calls.append(list(argv))
        if argv == ["id", "-un"]:
            if self.id_error is not None:
                raise self.id_error
            return CommandResult(argv, 0, self.user + "\n", "", 0.01)
        if argv[:1] == ["quota"]:
            return CommandResult(argv, 0, self.quota_text, "", 0.01)
        return CommandResult(argv, 127, "", f"{argv[0]}: command not found", 0.01)

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        self.calls.append(["<script>", *(args or [])])
        out: list[str] = []
        if "TOOL" in script:
            for name in args or []:
                path, version = self.tools.get(name, ("", ""))
                out.append(f"TOOL\t{name}\t{path}\t{version}")
        else:
            for p in args or []:
                out.append(f"PATH\t{p}\t{p}")
                used = 10_000_000 * self.capacity_pct // 100
                out.append(f"DF\t/dev/sda1 10000000 {used} {10_000_000 - used} {self.capacity_pct}% /scratch")
        return CommandResult(["bash"], 0, "\n".join(out) + "\n", "", 0.01)

    def check_master(self) -> bool | None:
        return self.master

    def remote_user(self) -> str:
        return self.user

    def close(self) -> None:
        return None


def job(job_id: str, state: str, *, user: str = "alice", gpus: int | None = 1, name: str = "x",
        reason: str | None = None, partition: str = "gpu") -> JobObservation:  # fmt: skip
    return JobObservation(
        job_id=job_id,
        source="squeue",
        observed_at="2026-09-01T12:00:00.000Z",
        state=state,  # type: ignore[arg-type]
        user=user,
        name=name,
        reason=reason,
        partition=partition,
        alloc_gpus=gpus,
    )


@dataclass
class FakeSched:
    cluster_id: str = "fake"
    caps: SchedulerCapabilities = field(
        default_factory=lambda: SchedulerCapabilities(
            slurm_version="23.02.7",
            sacct_available=True,
            sacct_comment_field=True,
            sstat_available=True,
            srun_overlap=True,
            cluster_name="fakecluster",
            min_job_age_s=300,
            accounting_store_flags="job_comment",
            private_data="none",
        )
    )
    config: dict[str, str | None] = field(
        default_factory=lambda: {"PrivateData": "none", "MailProg": "/bin/mail", "JobRequeue": "1"}
    )
    parts: list[PartitionInfo] = field(
        default_factory=lambda: [
            PartitionInfo(
                name="gpu",
                state="up",
                max_time_s=2 * 86400,
                default=True,
                nodes_total=4,
                gpu_types={"a100": {"total": 16, "used": 4, "free": 8, "unavailable": 4}},
                preemptible=False,
            ),
            PartitionInfo(
                name="lo",
                state="up",
                max_time_s=86400,
                default=False,
                nodes_total=2,
                gpu_types={"h100": {"total": 16, "used": 0, "free": 16, "unavailable": 0}},
                preemptible=True,
            ),
        ]
    )
    details: list[PartitionDetail] = field(
        default_factory=lambda: [
            PartitionDetail(name="gpu", preempt_mode="OFF", grace_time_s=0, max_time_s=2 * 86400),
            PartitionDetail(name="lo", preempt_mode="REQUEUE", grace_time_s=120, max_time_s=86400),
        ]
    )
    node_list: list[NodeInfo] = field(
        default_factory=lambda: [
            NodeInfo("n1", "mixed", ["gpu"], "gpu:a100:4", "gpu:a100:2", None),
            NodeInfo("n2", "drained", ["gpu"], "gpu:a100:4", "gpu:a100:0", "bad gpu"),
        ]
    )
    assocs: list[AssocLimit] = field(
        default_factory=lambda: [
            AssocLimit(
                cluster="fakecluster",
                account="lab",
                user="alice",
                partition=None,
                qos=("normal",),
                default_qos="normal",
            )
        ]
    )
    qos: list[QosLimit] = field(default_factory=lambda: [QosLimit(name="normal", max_gpus_per_user=8)])
    my_jobs: list[JobObservation] = field(default_factory=list)
    account_jobs: list[JobObservation] = field(default_factory=list)
    errors: dict[str, Exception] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def _hit(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if name in self.errors:
            raise self.errors[name]

    def probe_capabilities(self, *, refresh: bool = False) -> SchedulerCapabilities:
        self._hit("probe_capabilities")
        return self.caps

    def cluster_config(self) -> dict[str, str | None]:
        self._hit("cluster_config")
        return dict(self.config)

    def partitions(self) -> list[PartitionInfo]:
        self._hit("partitions")
        return list(self.parts)

    def partition_details(self) -> list[PartitionDetail]:
        self._hit("partition_details")
        return list(self.details)

    def nodes(self, partition: str | None = None) -> list[NodeInfo]:
        self._hit("nodes")
        return list(self.node_list)

    def association_limits(self, user: str) -> list[AssocLimit]:
        self._hit("association_limits")
        return list(self.assocs)

    def qos_limits(self, names: list[str] | None = None) -> list[QosLimit]:
        self._hit("qos_limits")
        return [q for q in self.qos if not names or q.name in names]

    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        self._hit("query_queue")
        return QueryResult(source="squeue", observed_at="2026-09-01T12:00:00.000Z", jobs=list(self.my_jobs))

    def query_account_queue(self, account: str) -> QueryResult:
        self._hit("query_account_queue")
        return QueryResult(
            source="squeue", observed_at="2026-09-01T12:00:00.000Z", jobs=list(self.account_jobs)
        )

    def test_only(self, shape: Any) -> ShapeEstimate:
        self._hit("test_only")
        return ShapeEstimate(
            job_id="999", start_time="2026-09-01T13:00:00.000Z", reason=None, partition=shape.partition
        )


@dataclass
class Env:
    ctx: ServiceContext
    login: FakeLogin
    sched: FakeSched
    clock: FakeClock

    def register(self, **overlay: Any) -> None:
        self.ctx.clusters.register(profile(), UserOverlay(cluster_id="fake", **{**OVERLAY, **overlay}))


OVERLAY: dict[str, Any] = {
    "ssh_host": "login.example.org",
    "user": "alice",
    "account": "lab",
    "remote_root": "/scratch/alice/hpclab",
}


def profile() -> ClusterProfile:
    return ClusterProfile(
        cluster_id="fake",
        display_name="Fake cluster",
        transport="ssh",
        login_host="login.example.org",
        partitions={
            "gpu": PartitionRule(default=True, gpu_types=["a100"], max_time="2-00:00:00"),
            "lo": PartitionRule(preemptible=True, gpu_types=["h100"], max_time="1-00:00:00"),
            "interactive": PartitionRule(interactive_only=True, gpu_types=["a100"]),
        },
        storage={
            "scratch": StorageRole(path="/scratch/{user}"),
            "home": StorageRole(path="/home/{user}", avoid_for=["outputs", "checkpoints"]),
        },
        policy=SitePolicy(),
    )
