"""Adapter contracts.

``Transport`` moves bytes and runs *bounded metadata commands* on a login node.
``Scheduler`` turns Slurm commands into typed facts via a Transport.

Failure semantics matter more than the happy path:

* ``AuthRequiredError``   – SSH needs interactive auth (MFA / expired ControlMaster). Never
  retried in the background; surfaced as ``auth_required``.
* ``TransportTimeout``    – the command may or may not have run remotely.
* ``ConnectionLost``      – the connection dropped; the command may or may not have run.
  For sbatch both of these mean *submission uncertain*, never "not submitted".
* ``CommandFailed``       – the command ran and returned non-zero (remote said no).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from hpclab.domain.errors import ErrorCode, HpcLabError
from hpclab.domain.models import JobObservation


class TransportError(HpcLabError):
    code = ErrorCode.UNAVAILABLE
    # Subclass default for "the remote side may have executed the command (reply lost)".
    maybe_executed_default: bool = False

    def __init__(self, message: str, *, maybe_executed: bool | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        if maybe_executed is not None:
            self.maybe_executed = maybe_executed
        elif "maybe_executed" not in self.details:
            self.maybe_executed = self.maybe_executed_default

    @property
    def maybe_executed(self) -> bool:
        """Stored in ``details`` so it survives the controller's JSON boundary."""
        return bool(self.details.get("maybe_executed", self.maybe_executed_default))

    @maybe_executed.setter
    def maybe_executed(self, value: bool) -> None:
        self.details["maybe_executed"] = bool(value)


class AuthRequiredError(TransportError):
    code = ErrorCode.AUTH_REQUIRED


class TransportTimeout(TransportError):
    code = ErrorCode.UNAVAILABLE
    maybe_executed_default = True


class ConnectionLost(TransportError):
    code = ErrorCode.UNAVAILABLE
    maybe_executed_default = True


class CommandFailed(TransportError):
    """Remote command ran and exited non-zero. ``result`` holds stdout/stderr."""

    code = ErrorCode.UNAVAILABLE

    def __init__(self, message: str, result: CommandResult, **kw: object) -> None:
        super().__init__(message, **kw)  # type: ignore[arg-type]
        self.result = result


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True)
class StatInfo:
    path: str
    exists: bool
    kind: Literal["file", "dir", "symlink", "other", "missing"]
    size: int | None = None
    mtime: float | None = None  # epoch seconds
    inode: int | None = None
    realpath: str | None = None  # resolved path (symlinks followed)


@dataclass(frozen=True)
class RangeRead:
    path: str
    offset: int
    data: bytes
    size: int  # current file size
    mtime: float | None
    inode: int | None  # generation marker: a replaced/rotated file gets a new inode
    eof: bool


@runtime_checkable
class Transport(Protocol):
    """Login-node access. Implementations: OpenSSH (real), Local, Sim."""

    cluster_id: str

    def remote_user(self) -> str:
        """The actual Unix user on the cluster (``id -un``), cached after first success."""
        ...

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        """Run argv on the login node. Arguments are quoted by the transport; never pass
        shell strings built from user data. Returns even on non-zero exit (caller decides)."""
        ...

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        """Run a fixed, hpclab-authored bash script (fed on stdin to ``bash -s -- args``)."""
        ...

    def upload_dir(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        """Copy a local directory tree to ``remote_dir`` (rsync -a semantics; creates parents)."""
        ...

    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        """Atomically write a small file (temp + rename)."""
        ...

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        """Read at most ``length`` bytes from ``offset``. Missing file -> FileNotFoundError."""
        ...

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        """Read the last ``length`` bytes."""
        ...

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]: ...

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        """``sha256:<hex>`` per file; None for missing/unreadable."""
        ...

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None: ...

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        """Recursive listing of files under a directory (bounded)."""
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass
class SchedulerCapabilities:
    slurm_version: str | None = None
    squeue_json: bool = False
    sacct_available: bool = False
    sacct_comment_field: bool = False  # AccountingStoreFlags=job_comment
    sstat_available: bool = False
    srun_overlap: bool = False
    test_only: bool = True
    parsable_submit: bool = True
    cluster_name: str | None = None
    min_job_age_s: int | None = None  # how long finished jobs stay in squeue
    accounting_store_flags: str | None = None
    private_data: str | None = None  # PrivateData (e.g. "jobs,usage") -> visibility limits
    probed_at: str | None = None
    notes: list[str] = field(default_factory=list)


class SubmissionRejected(HpcLabError):
    """sbatch ran and refused the job (bad account/partition/limits). Nothing was queued."""

    code = ErrorCode.PRECONDITION_FAILED


@dataclass(frozen=True)
class SubmitResponse:
    job_id: str
    cluster_name: str | None
    raw: str
    submitted_at: str  # controller-side timestamp when the reply was received


@dataclass(frozen=True)
class QueryResult:
    """Observations from one source. ``complete`` is False if the source was partial."""

    source: Literal["squeue", "sacct", "scontrol", "sim"]
    observed_at: str
    jobs: list[JobObservation]
    complete: bool = True
    error: str | None = None


@dataclass(frozen=True)
class CancelResponse:
    job_id: str
    accepted: bool
    raw: str


@dataclass(frozen=True)
class PartitionInfo:
    name: str
    state: str | None
    max_time_s: int | None
    default: bool
    nodes_total: int | None
    gpu_types: dict[str, dict[str, int]]  # gpu_type -> {"total": n, "used": m, "free": k}
    preemptible: bool | None = None


@dataclass(frozen=True)
class NodeInfo:
    name: str
    state: str
    partitions: list[str]
    gres: str | None
    gres_used: str | None
    reason: str | None


@dataclass(frozen=True)
class Association:
    user: str
    account: str
    partition: str | None
    qos: list[str]
    default_qos: str | None


@dataclass(frozen=True)
class StartEstimate:
    job_id: str
    start_time: str | None
    reason: str | None


@dataclass(frozen=True)
class JobStepStats:
    job_id: str
    step: str
    max_rss_bytes: int | None
    ave_cpu_s: int | None
    tres_usage: str | None


@dataclass(frozen=True)
class GpuSample:
    index: int
    name: str | None
    util_percent: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    power_w: float | None


@dataclass(frozen=True)
class UsageRecord:
    """One accounting row aggregated for usage reports."""

    job_id: str
    user: str
    account: str | None
    partition: str | None
    state: str
    gpus: int
    elapsed_s: int
    start_time: str | None
    end_time: str | None
    submit_time: str | None
    job_name: str | None


@runtime_checkable
class Scheduler(Protocol):
    cluster_id: str

    def probe_capabilities(self) -> SchedulerCapabilities: ...

    def submit(
        self, script_path: str, *, workdir: str, test_only: bool = False, receipt_path: str | None = None
    ) -> SubmitResponse:
        """Call sbatch. ``receipt_path``: remote file where the raw sbatch reply is journaled
        *on the cluster* before being returned, so a lost SSH reply can be recovered.
        Raises SubmissionRejected on explicit refusal; TransportTimeout/ConnectionLost when
        the outcome is unknown."""
        ...

    def read_receipt(self, receipt_path: str) -> SubmitResponse | None: ...

    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        """squeue for the given job ids (or all jobs of ``user``)."""
        ...

    def query_accounting(
        self,
        *,
        job_ids: list[str] | None = None,
        user: str | None = None,
        name: str | None = None,
        since: str | None = None,
    ) -> QueryResult:
        """sacct, allocation lines only (-X). Unavailable accounting raises TransportError
        (code unavailable/unsupported), never returns an empty 'success'."""
        ...

    def cancel(self, job_id: str) -> CancelResponse: ...

    def partitions(self) -> list[PartitionInfo]: ...

    def nodes(self, partition: str | None = None) -> list[NodeInfo]: ...

    def associations(self, user: str) -> list[Association]: ...

    def start_estimates(self, job_ids: list[str]) -> list[StartEstimate]: ...

    def step_stats(self, job_id: str) -> list[JobStepStats]: ...

    def gpu_samples(self, job_id: str, *, timeout: float = 20.0) -> list[GpuSample]:
        """nvidia-smi inside the job's own allocation (``srun --jobid --overlap``)."""
        ...

    def usage(
        self, *, account: str | None, users: list[str] | None, since: str, until: str | None
    ) -> list[UsageRecord]: ...
