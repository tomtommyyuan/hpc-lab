"""Simulator configuration and fault switches.

Kept dependency-free (dataclasses + json) because every fake Slurm command is a fresh Python
process: import time is latency for every simulated ``squeue``.

``SimConfig`` describes the cluster (nodes, partitions, accounts, timings) and is persisted as
``<sim_root>/config.json``. ``SimFaults`` are runtime switches persisted in the state database;
``None`` on a timing fault means "use the config value".
"""

from __future__ import annotations

import dataclasses
import getpass
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NodeConfig:
    name: str
    cpus: int = 64
    mem_mb: int = 512_000
    gpus: int = 0
    gpu_type: str | None = None
    features: tuple[str, ...] = ()

    def gres(self) -> str:
        """``gpu:a100:4(S:0-1)`` as sinfo prints it; ``(null)`` without GPUs."""
        if not self.gpus:
            return "(null)"
        typed = f"gpu:{self.gpu_type}:{self.gpus}" if self.gpu_type else f"gpu:{self.gpus}"
        return f"{typed}(S:0-1)"


@dataclass(frozen=True)
class PartitionConfig:
    name: str
    nodes: tuple[str, ...]
    max_time_s: int | None = None  # None = UNLIMITED
    default_time_s: int | None = None  # None = MaxTime
    default: bool = False
    max_gpus_per_user: int | None = None  # running GPUs per user (QOSMaxGRESPerUser)
    preemptible: bool = False
    priority_tier: int = 10
    state: str = "UP"
    grace_time_s: int = 0  # GraceTime: SIGTERM -> SIGKILL delay for preempted jobs
    preempt_mode: str | None = None  # None: CANCEL when preemptible, else OFF

    def effective_preempt_mode(self) -> str:
        """``PreemptMode`` as scontrol/sinfo print it."""
        if self.preempt_mode:
            return self.preempt_mode.upper()
        return "CANCEL" if self.preemptible else "OFF"


def _default_nodes() -> tuple[NodeConfig, ...]:
    a100 = tuple(
        NodeConfig(
            name=f"sim-a100-{i:02d}", cpus=64, mem_mb=512_000, gpus=4, gpu_type="a100", features=("a100",)
        )
        for i in range(1, 5)
    )
    inter = (
        NodeConfig(name="sim-int-01", cpus=32, mem_mb=256_000, gpus=2, gpu_type="a100", features=("a100",)),
    )
    h100 = tuple(
        NodeConfig(
            name=f"sim-h100-{i:02d}", cpus=96, mem_mb=1_024_000, gpus=8, gpu_type="h100", features=("h100",)
        )
        for i in range(1, 3)
    )
    return a100 + inter + h100


def _default_partitions() -> tuple[PartitionConfig, ...]:
    return (
        PartitionConfig(
            name="gpu",
            nodes=tuple(f"sim-a100-{i:02d}" for i in range(1, 5)),
            max_time_s=2 * 86400,
            default=True,
        ),
        PartitionConfig(name="interactive", nodes=("sim-int-01",), max_time_s=4 * 3600, max_gpus_per_user=1),
        PartitionConfig(
            name="scavenge",
            nodes=("sim-h100-01", "sim-h100-02"),
            max_time_s=86400,
            preemptible=True,
            priority_tier=1,
        ),
    )


@dataclass(frozen=True)
class SimConfig:
    cluster_name: str = "simcluster"
    slurm_version: str = "23.02.7"
    user: str = field(default_factory=getpass.getuser)
    accounts: tuple[str, ...] = ("lab", "other")
    default_account: str = "lab"
    qos: tuple[str, ...] = ("normal",)
    default_qos: str = "normal"
    nodes: tuple[NodeConfig, ...] = field(default_factory=_default_nodes)
    partitions: tuple[PartitionConfig, ...] = field(default_factory=_default_partitions)
    first_job_id: int = 1000
    min_job_age_s: float = 2.0  # squeue keeps finished jobs this long (MinJobAge)
    accounting_lag_s: float = 0.0  # slurmdbd view lags the controller by this much
    kill_wait_s: float = 1.0  # SIGTERM -> SIGKILL grace (KillWait)
    poll_interval_s: float = 0.1  # job runner scheduling/supervision tick
    fate_delay_s: float = 0.5  # when injected node-fail/preempt/OOM strikes after start
    enforce_part_limits: bool = True  # False: over-limit jobs pend with PartitionTimeLimit
    max_submit_jobs_per_user: int | None = None
    def_mem_per_cpu_mb: int = 4096
    gpu_util: float = 85.0  # default utilization.gpu for fake nvidia-smi
    accounting_enabled: bool = True
    store_job_comment: bool = True
    # AccountingStorageTRES includes gres/gpu. Slurm's default does not: TRES strings (sacct
    # AllocTRES, squeue tres-alloc) then carry no GPU entry even for GPU jobs.
    track_gpu_tres: bool = True
    parsable_cluster_suffix: bool = False  # sbatch --parsable prints "<id>;<cluster>" (multi-cluster)
    # PrivateData (scontrol show config). "jobs" hides other users' jobs from squeue and sacct,
    # as on a real cluster for a non-admin, non-coordinator user.
    private_data: str = "none"
    job_acct_gather_type: str = "jobacct_gather/cgroup"
    job_requeue: bool = True  # JobRequeue=1: batch jobs are requeueable unless --no-requeue
    mail_prog: str | None = "/bin/mail"
    # QOS limits reported by `sacctmgr show qos` (same for every configured QOS; None = unset).
    qos_max_wall_s: int | None = None
    qos_max_gpus_per_user: int | None = None
    qos_max_jobs_per_user: int | None = None
    # What the fake per-job nvidia-smi reports.
    driver_version: str = "570.124.06"
    driver_cuda: str = "12.8"

    def __post_init__(self) -> None:
        self.validate()

    # -- lookups --------------------------------------------------------------
    def node(self, name: str) -> NodeConfig | None:
        return next((n for n in self.nodes if n.name == name), None)

    def partition(self, name: str) -> PartitionConfig | None:
        return next((p for p in self.partitions if p.name == name), None)

    def default_partition(self) -> PartitionConfig | None:
        return next(
            (p for p in self.partitions if p.default), self.partitions[0] if self.partitions else None
        )

    def gpu_types(self) -> set[str]:
        return {n.gpu_type for n in self.nodes if n.gpu_type}

    def private_data_flags(self) -> frozenset[str]:
        return frozenset(f.strip().lower() for f in (self.private_data or "").split(",") if f.strip())

    def hides_other_jobs(self) -> bool:
        """PrivateData=jobs: a regular user only sees their own jobs (squeue, sacct)."""
        return "jobs" in self.private_data_flags()

    def validate(self) -> SimConfig:
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate node names")
        pnames = [p.name for p in self.partitions]
        if len(set(pnames)) != len(pnames):
            raise ValueError("duplicate partition names")
        for p in self.partitions:
            missing = [n for n in p.nodes if n not in names]
            if missing:
                raise ValueError(f"partition {p.name} references unknown nodes {missing}")
        if self.default_account not in self.accounts:
            raise ValueError("default_account must be one of accounts")
        return self

    def replace(self, **changes: Any) -> SimConfig:
        return dataclasses.replace(self, **changes)

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimConfig:
        data = dict(data)
        if "nodes" in data:
            data["nodes"] = tuple(
                NodeConfig(**{**n, "features": tuple(n.get("features", ()))}) for n in data["nodes"]
            )
        if "partitions" in data:
            data["partitions"] = tuple(
                PartitionConfig(**{**p, "nodes": tuple(p["nodes"])}) for p in data["partitions"]
            )
        for key in ("accounts", "qos"):
            if key in data:
                data[key] = tuple(data[key])
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown simulator config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> SimConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class SimFaults:
    """Injected failures. Counters (``*_next``, ``sbatch_lose_reply``...) are consumed per use."""

    sbatch_reject: str | None = None  # stderr text; sbatch exits 1, nothing queued
    sbatch_lose_reply: int = 0  # job IS queued, SimTransport raises ConnectionLost
    sbatch_ambiguous_error: int = 0  # sbatch prints a socket timeout and exits 1
    sbatch_ambiguous_creates_job: bool = True  # ... after (True) or without (False) queuing the job
    squeue_unavailable: bool = False
    sacct_unavailable: bool = False
    accounting_lag_s: float | None = None
    min_job_age_s: float | None = None
    auth_required: bool = False  # every SimTransport call raises AuthRequiredError
    # Command names (squeue, sbatch...; "sbatch" also blocks the `sbatch --version` probe) or
    # transport method names (stat, read_range...), or "*": those calls raise TransportTimeout.
    transport_timeout_cmds: list[str] = field(default_factory=list)
    node_fail_next: int = (
        0  # next submitted jobs' node fails after start: NODE_FAIL (requeued if requeueable)
    )
    preempt_next: int = 0  # ... PREEMPTED
    oom_next: int = 0  # ... OUT_OF_MEMORY
    requeue_next: int = 0  # ... preempted once and requeued (PREEMPTED segment, then runs again)
    pending_delay_s: float = 0.0  # jobs stay PENDING (Reason=Priority) at least this long
    gpu_util: float | None = None  # utilization.gpu reported by fake nvidia-smi

    COUNTERS = (
        "sbatch_lose_reply",
        "sbatch_ambiguous_error",
        "node_fail_next",
        "preempt_next",
        "oom_next",
        "requeue_next",
    )
    BOOLS = ("squeue_unavailable", "sacct_unavailable", "auth_required", "sbatch_ambiguous_creates_job")
    OPTIONAL_FLOATS = ("accounting_lag_s", "min_job_age_s", "gpu_util")

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimFaults:
        known = cls.field_names()
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def coerce(cls, key: str, value: Any) -> Any:
        """Validate/convert one fault value (CLI strings included). Raises ValueError."""
        if key not in cls.field_names():
            raise ValueError(f"unknown fault {key!r}; known: {sorted(cls.field_names())}")
        if key == "transport_timeout_cmds":
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return [str(v) for v in value or ()]
        is_null = value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", ""))
        if key == "sbatch_reject":
            return None if is_null else str(value)
        if key in cls.COUNTERS:
            n = int(value)
            if n < 0:
                raise ValueError(f"{key} must be >= 0")
            return n
        if key in cls.BOOLS:
            return _as_bool(key, value)
        if is_null:
            if key in cls.OPTIONAL_FLOATS:
                return None
            raise ValueError(f"{key} may not be null")
        number = float(value)
        if number < 0:
            raise ValueError(f"{key} must be >= 0")
        return number


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key} expects a boolean, got {value!r}")
