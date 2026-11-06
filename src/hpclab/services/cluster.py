"""Cluster doctor: ``cluster_inspect``.

Read-only. Runs only bounded metadata commands on the login node (``id``, ``sbatch --version``,
``scontrol show config|partition``, ``sinfo``, ``squeue``, ``sacctmgr``, ``df``, ``quota``,
``command -v``) and, only for explicit ``candidate_shapes``, ``sbatch --test-only`` (cached 5 min
per shape in ``report_cache``). Never creates compute jobs, never lists the whole queue.

Every check returns ``{name, status, summary, findings, data, source, observed_at}`` with
``status`` in ok|warn|fail|unavailable|unsupported|skipped. Rules:

* a source that could not answer is ``unavailable``/``unsupported``, never ``ok`` and never 0;
* a hidden limit is reported as unavailable (``data`` holds None), not as "no limit";
* when SSH authentication is missing, every remote check is skipped with the auth
  remediation (``hpclab auth <cluster>``) instead of hanging on a prompt (BatchMode);
* the whole call is bounded by ``settings.request_deadline_s``: checks that would start after
  the budget are ``skipped`` with a note (re-run them with ``checks=[...]``).
"""

from __future__ import annotations

import copy
import logging
import shlex
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hpclab.adapters.base import AuthRequiredError, ConnectionLost, TransportError, TransportTimeout
from hpclab.api.envelope import Coverage, SourceStatusValue
from hpclab.domain.clock import iso, parse_iso, parse_slurm_duration
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument
from hpclab.domain.hashing import digest
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jdump, jload

log = logging.getLogger("hpclab.cluster")

CheckName = Literal[
    "auth",
    "identity",
    "scheduler",
    "storage",
    "runtime",
    "queue",
    "capacity",
    "limits",
    "nodes",
    "profile_facts",
]
CheckStatus = Literal["ok", "warn", "fail", "unavailable", "unsupported", "skipped"]
AuthState = Literal["ok", "auth_required", "vpn_unreachable", "host_key_changed", "unknown"]
ALL_CHECKS: tuple[CheckName, ...] = (
    "auth",
    "identity",
    "scheduler",
    "storage",
    "runtime",
    "queue",
    "capacity",
    "limits",
    "nodes",
    "profile_facts",
)
LOCAL_CHECKS = frozenset({"profile_facts"})
MAX_CANDIDATES = 3
TEST_ONLY_TTL_S = 300.0
CACHE_PREFIX = "test_only:v1:"
MAX_LISTED = 50

# Pending reasons that can never resolve by waiting (fallback when services.pending is absent).
_NEVER_STARTS = (
    "PartitionTimeLimit",
    "QOSMaxWallDurationPerJobLimit",
    "DependencyNeverSatisfied",
    "BadConstraints",
    "PartitionNodeLimit",
    "InvalidAccount",
    "InvalidQOS",
)

STORAGE_SNIPPET = r"""set -u
for p in "$@"; do
  q=$p
  while [ ! -e "$q" ] && [ "$q" != "/" ] && [ -n "$q" ]; do q=$(dirname -- "$q"); done
  printf 'PATH\t%s\t%s\n' "$p" "$q"
  df -Pk -- "$q" 2>/dev/null | tail -n +2 | head -n 1 | sed 's/^/DF\t/'
done
exit 0
"""

RUNTIME_SNIPPET = r"""set -u
for c in "$@"; do
  p=$(command -v -- "$c" 2>/dev/null) || p=""
  v=""
  if [ -n "$p" ]; then v=$("$c" --version 2>&1 | head -n 1 | tr -d '\t\r'); fi
  printf 'TOOL\t%s\t%s\t%s\n' "$c" "$p" "$v"
done
exit 0
"""
RUNTIME_TOOLS = ("python3", "uv", "rsync")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warn", "block"]
    message: str
    remediation: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: CheckName
    status: CheckStatus
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    source: str
    observed_at: str | None = None


class CandidateShape(BaseModel):
    """A placement to compare: ``gpus`` is the job total (spread over ``nodes``)."""

    model_config = ConfigDict(extra="forbid")

    partition: str
    gpu_type: str | None = None
    gpus: int = Field(default=1, ge=0, le=64)
    time: str = "01:00:00"
    nodes: int = Field(default=1, ge=1, le=64)

    @field_validator("time")
    @classmethod
    def _time(cls, v: str) -> str:
        if parse_slurm_duration(v) in (None, 0):
            raise ValueError(f"invalid Slurm time limit {v!r}")
        return v


class CandidateEstimate(BaseModel):
    """``sbatch --test-only`` guess for a shape: low confidence, never an ETA or a ranking."""

    model_config = ConfigDict(extra="forbid")

    shape: CandidateShape
    status: Literal["ok", "blocked", "unavailable"]
    est_start: str | None = None
    est_source: Literal["test_only", "none"] = "none"
    confidence: Literal["low"] = "low"
    cached: bool = False
    observed_at: str | None = None
    free_now: int | None = None
    max_time_s: int | None = None
    preemptible: bool | None = None
    preempt_mode: str | None = None
    grace_time_s: int | None = None
    blockers: list[str] = Field(default_factory=list)
    note: str | None = None


class ClusterInspectData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    status: Literal["ok", "warn", "fail"]
    checks: list[CheckResult]
    candidates: list[CandidateEstimate] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    coverage: Coverage
    observed_at: str

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        parts = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        head = f"cluster {self.cluster_id}: {self.status} ({parts})"
        return f"{head}; next: {self.remediation[0]}" if self.remediation else head


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class _BudgetedTransport:
    """Transport proxy for one inspect: no command may outlive the call's time budget.

    Every ``run``/``run_script`` timeout is capped to what is left of the budget, and a command
    that would start after it is refused at once (``TransportTimeout``, nothing executed). The
    scheduler adapter's own per-command timeouts (20-60 s each) would otherwise let a hung
    slurmctld - exactly when people run the doctor - stretch the call past the RPC deadline.
    """

    def __init__(self, inner: Any, remaining: Callable[[], float]) -> None:
        self._inner = inner
        self._remaining = remaining

    def _timeout(self, timeout: float) -> float:
        left = self._remaining()
        if left <= 0:
            raise TransportTimeout(
                "not run: the request time budget was used up",
                maybe_executed=False,
                details={"reason": "deadline"},
            )
        return min(float(timeout), max(1.0, left))

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> Any:
        return self._inner.run(argv, timeout=self._timeout(timeout), stdin=stdin)

    def run_script(self, script: str, *, timeout: float = 30.0, args: list[str] | None = None) -> Any:
        return self._inner.run_script(script, timeout=self._timeout(timeout), args=args)

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:  # e.g. during copy/pickle, before __init__ ran: no recursion
            raise AttributeError(name)
        return getattr(inner, name)


@dataclass
class _Conn:
    state: AuthState
    remote_user: str | None
    master_alive: bool | None
    error: HpcLabError | None
    detail: str | None
    remediation: str | None

    @property
    def ok(self) -> bool:
        return self.state == "ok"


class _Inspector:
    def __init__(self, ctx: ServiceContext, rc: ResolvedCluster) -> None:
        self.ctx = ctx
        self.rc = rc
        self.cid = rc.cluster_id
        self._conn: _Conn | None = None
        self._sched: Any = None
        self._transport: Any = None
        self._partitions: list[Any] | None = None
        self._details: dict[str, Any] | None = None
        self._caps: Any = None
        budget = max(1.0, float(ctx.settings.request_deadline_s) * 0.85)
        self._deadline = ctx.clock.monotonic() + budget

    # -- plumbing ---------------------------------------------------------------
    def now(self) -> str:
        return iso(self.ctx.clock.now()) or ""

    def auth_remediation(self, extra: str | None = None) -> str:
        base = f"run `hpclab auth {self.cid}` in a terminal and complete the password/Duo prompt"
        return f"{base} ({extra})" if extra else base

    def remaining(self) -> float:
        return self._deadline - self.ctx.clock.monotonic()

    def transport(self) -> Any:
        if self._transport is None:
            self._transport = _BudgetedTransport(self.ctx.clusters.transport(self.cid), self.remaining)
        return self._transport

    def sched(self) -> Any:
        """A private shallow copy of the cluster's scheduler whose commands go through the
        budgeted transport (the shared instance keeps its own timeouts and probe cache)."""
        if self._sched is None:
            sched = copy.copy(self.ctx.clusters.scheduler(self.cid))
            if hasattr(sched, "transport"):
                sched.transport = _BudgetedTransport(sched.transport, self.remaining)
            self._sched = sched
        return self._sched

    def user(self) -> str | None:
        conn = self.connect()
        return conn.remote_user or self.rc.overlay.user

    def caps(self) -> Any:
        if self._caps is None:
            self._caps = self.sched().probe_capabilities()
        return self._caps

    def connect(self) -> _Conn:
        """One cheap round trip (``id -un``) that classifies the SSH/auth state."""
        if self._conn is not None:
            return self._conn
        host = self.rc.overlay.ssh_host or self.rc.profile.login_host
        try:
            transport = self.transport()
        except HpcLabError as exc:
            self._conn = _Conn(
                "unknown", None, None, exc, exc.message, "fix the cluster registration (ssh_host)"
            )
            return self._conn
        master = getattr(transport, "check_master", None)
        try:
            res = transport.run(["id", "-un"], timeout=15.0)
        except AuthRequiredError as exc:
            reason = str(exc.details.get("reason") or "auth")
            login = exc.details.get("login_command_str") or (
                shlex.join(exc.details["login_command"])
                if isinstance(exc.details.get("login_command"), list)
                else None
            )
            if reason == "host_key":
                self._conn = _Conn(
                    "host_key_changed",
                    None,
                    None,
                    exc,
                    exc.message,
                    f"the SSH host key of {host or self.cid} is unknown or CHANGED: confirm it with your cluster "
                    f"admins before accepting it (`hpclab auth {self.cid}` shows the prompt)",
                )
            else:
                self._conn = _Conn(
                    "auth_required",
                    None,
                    bool(master()) if callable(master) else None,
                    exc,
                    exc.message,
                    self.auth_remediation(f"command: {login}" if login else None),
                )
            return self._conn
        except ConnectionLost as exc:
            state: AuthState = "vpn_unreachable" if not exc.maybe_executed else "unknown"
            self._conn = _Conn(
                state,
                None,
                None,
                exc,
                exc.message,
                f"cannot reach {host or self.cid}: check your network/VPN, then retry"
                if state == "vpn_unreachable"
                else "the connection dropped; retry, and check the network if it persists",
            )
            return self._conn
        except TransportTimeout as exc:
            self._conn = _Conn(
                "unknown", None, None, exc, exc.message, "the login node did not answer in time; retry"
            )
            return self._conn
        except TransportError as exc:
            remediation = (
                "fix the SSH configuration (overlay ssh_options or ~/.ssh/config)"
                if exc.code == ErrorCode.PRECONDITION_FAILED
                else "see the error details"
            )
            self._conn = _Conn("unknown", None, None, exc, exc.message, remediation)
            return self._conn
        user = res.stdout.strip() if res.exit_code == 0 and res.stdout.strip() else None
        alive = bool(master()) if callable(master) else None
        self._conn = _Conn(
            "ok", user, alive, None, None if user else (res.stderr.strip()[-300:] or None), None
        )
        return self._conn

    def over_budget(self) -> bool:
        return self.remaining() <= 0

    # -- runner -------------------------------------------------------------------
    def run(self, name: CheckName, fn: Callable[[], CheckResult]) -> tuple[CheckResult, bool]:
        """(result, ran). ``ran`` is False when the check was skipped (auth, deadline)."""
        if name not in LOCAL_CHECKS and name != "auth":
            if self.over_budget():
                return (
                    CheckResult(
                        name=name,
                        status="skipped",
                        summary="not run: the request time budget was used up",
                        findings=[
                            Finding(
                                code="deadline",
                                severity="info",
                                message="skipped to stay within the request deadline",
                                remediation=f"re-run with checks=['{name}']",
                            )
                        ],
                        source="-",
                    ),
                    False,
                )
            conn = self.connect()
            if not conn.ok:
                return (
                    CheckResult(
                        name=name,
                        status="skipped",
                        summary=f"not run: SSH state is {conn.state}",
                        findings=[
                            Finding(
                                code="blocked_by_auth",
                                severity="warn",
                                message=f"needs a working SSH connection (state: {conn.state})",
                                remediation=conn.remediation,
                            )
                        ],
                        source="-",
                    ),
                    False,
                )
        try:
            return fn(), True
        except HpcLabError as exc:
            status: CheckStatus = "unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable"
            remediation = self.auth_remediation() if exc.code == ErrorCode.AUTH_REQUIRED else None
            return (
                CheckResult(
                    name=name,
                    status=status,
                    summary=f"{name} {status}: {exc.message}"[:300],
                    findings=[
                        Finding(
                            code=f"{name}_{status}",
                            severity="warn",
                            message=exc.message[:500],
                            remediation=remediation,
                            evidence={"error_code": str(exc.code)},
                        )
                    ],
                    source=name,
                    observed_at=self.now(),
                ),
                True,
            )
        except Exception as exc:
            log.exception("cluster_inspect %s: check %s crashed", self.cid, name)
            return (
                CheckResult(
                    name=name,
                    status="unavailable",
                    summary=f"{name} could not be checked (internal error: {type(exc).__name__})",
                    findings=[
                        Finding(
                            code="internal_error",
                            severity="warn",
                            message=f"{type(exc).__name__} while checking {name}; see the controller log",
                        )
                    ],
                    source=name,
                    observed_at=self.now(),
                ),
                True,
            )

    # -- checks -------------------------------------------------------------------
    def check_auth(self) -> CheckResult:
        conn = self.connect()
        transport_kind = self.rc.profile.transport
        data: dict[str, Any] = {
            "state": conn.state,
            "master_alive": conn.master_alive,
            "remote_user": conn.remote_user,
            "transport": transport_kind,
            "host": self.rc.overlay.ssh_host or self.rc.profile.login_host,
        }
        if conn.error is not None:
            data["error"] = {"code": str(conn.error.code), "message": conn.error.message[:500]}
        if conn.ok:
            return CheckResult(
                name="auth",
                status="ok",
                summary=f"connected as {conn.remote_user or '?'}"
                + (" (SSH master open)" if conn.master_alive else ""),
                data=data,
                source="ssh: id -un" if transport_kind == "ssh" else f"{transport_kind}: id -un",
                observed_at=self.now(),
            )
        status: CheckStatus = "unavailable" if conn.state == "unknown" else "fail"
        return CheckResult(
            name="auth",
            status=status,
            summary=f"SSH state {conn.state}: {conn.detail or 'no detail'}"[:300],
            findings=[
                Finding(
                    code=conn.state,
                    severity="block",
                    message=(conn.detail or conn.state)[:500],
                    remediation=conn.remediation,
                )
            ],
            data=data,
            source="ssh: id -un",
            observed_at=self.now(),
        )

    def check_identity(self) -> CheckResult:
        conn = self.connect()
        expected = self.rc.overlay.user
        data = {"remote_user": conn.remote_user, "expected_user": expected}
        if conn.remote_user is None:
            return CheckResult(
                name="identity",
                status="unavailable",
                summary="`id -un` gave no user name",
                findings=[
                    Finding(code="identity_unknown", severity="warn", message=conn.detail or "no output")
                ],
                data=data,
                source="id -un",
                observed_at=self.now(),
            )
        if expected is None:
            return CheckResult(
                name="identity",
                status="warn",
                summary=f"remote user is {conn.remote_user}; the overlay does not set one",
                findings=[
                    Finding(
                        code="user_not_set",
                        severity="warn",
                        message="the cluster overlay has no user: ownership checks (reconcile, cancel) and "
                        "{user} paths depend on it",
                        remediation=f"hpclab cluster add {self.cid} --user {conn.remote_user}",
                    )
                ],
                data=data,
                source="id -un",
                observed_at=self.now(),
            )
        if expected != conn.remote_user:
            return CheckResult(
                name="identity",
                status="fail",
                summary=f"identity mismatch: logged in as {conn.remote_user}, overlay says {expected}",
                findings=[
                    Finding(
                        code="identity_mismatch",
                        severity="block",
                        message=f"SSH logs in as {conn.remote_user!r} but the overlay expects {expected!r}: jobs would "
                        "be owned by another user, so hpclab's ownership checks would refuse to bind or cancel them",
                        remediation=f"fix the SSH user (ssh_host / ~/.ssh/config User) or run "
                        f"`hpclab cluster add {self.cid} --user {conn.remote_user}`",
                        evidence={"remote_user": conn.remote_user, "expected_user": expected},
                    )
                ],
                data=data,
                source="id -un",
                observed_at=self.now(),
            )
        return CheckResult(
            name="identity",
            status="ok",
            summary=f"remote user {conn.remote_user} matches the overlay",
            data=data,
            source="id -un",
            observed_at=self.now(),
        )

    def check_scheduler(self) -> CheckResult:
        sched = self.sched()
        probe = sched.probe_capabilities
        try:
            try:
                caps = probe(refresh=True)
            except TypeError:  # a Scheduler without the refresh knob
                caps = probe()
        except HpcLabError as exc:
            command = str(exc.details.get("command") or "")
            if exc.code == ErrorCode.UNSUPPORTED and command.startswith("sbatch"):
                return self._slurm_missing(exc)
            raise
        self._caps = caps
        shared = self.ctx.clusters.scheduler(self.cid)
        adopt = getattr(shared, "adopt_probe", None)
        if shared is not sched and callable(adopt):
            adopt(sched)  # the refreshed facts serve every later call, not just this inspect
        observed = self.now()
        findings: list[Finding] = []
        config: dict[str, Any] | None = None
        cfg_fn = getattr(sched, "cluster_config", None)
        if callable(cfg_fn):
            try:
                config = {
                    k: {"value": v, "source": "scontrol show config", "observed_at": observed}
                    for k, v in cfg_fn().items()
                }
            except HpcLabError as exc:
                findings.append(
                    Finding(
                        code="config_unavailable",
                        severity="warn",
                        message=f"scontrol show config failed: {exc.message}"[:300],
                    )
                )
        if not caps.sacct_available:
            findings.append(
                Finding(
                    code="accounting_unavailable",
                    severity="warn",
                    message="Slurm accounting (sacct) is unavailable: finished jobs can only be confirmed while "
                    "squeue still shows them, budgets settle as unknown and usage reports are unavailable",
                )
            )
        if any("gres/gpu" in n for n in caps.notes):
            findings.append(
                Finding(
                    code="gpu_accounting_untracked",
                    severity="warn",
                    message="AccountingStorageTRES does not track gres/gpu: GPU counts of jobs are unknown "
                    "(budget settlement and GPU-hour reports cannot be computed)",
                    remediation="ask the admins to add gres/gpu to AccountingStorageTRES",
                )
            )
        if caps.sacct_available and not caps.sacct_comment_field:
            findings.append(
                Finding(
                    code="job_comment_not_stored",
                    severity="info",
                    message="job comments are not stored in accounting; reconciliation relies on the job name token",
                )
            )
        if not caps.srun_overlap:
            findings.append(
                Finding(
                    code="no_srun_overlap",
                    severity="info",
                    message="live GPU sampling (srun --overlap) unavailable",
                )
            )
        from hpclab.adapters.slurm import private_data_flags

        flags = private_data_flags(caps.private_data)
        if flags and ("jobs" in flags or "usage" in flags):
            findings.append(
                Finding(
                    code="private_data",
                    severity="info",
                    message=f"PrivateData={caps.private_data}: other users' jobs/usage are hidden, so account views "
                    "(queue, usage) show your own jobs only",
                )
            )
        if caps.min_job_age_s is not None and caps.min_job_age_s < 60:
            findings.append(
                Finding(
                    code="short_min_job_age",
                    severity="info",
                    message=f"MinJobAge={caps.min_job_age_s}s: finished jobs leave squeue quickly; hpclab relies on sacct",
                )
            )
        if config is not None and (config.get("MailProg") or {}).get("value") is None:
            findings.append(
                Finding(code="no_mail_prog", severity="info", message="MailProg is unset: Slurm mail is off")
            )
        status: CheckStatus = "warn" if any(f.severity != "info" for f in findings) else "ok"
        return CheckResult(
            name="scheduler",
            status=status,
            summary=f"Slurm {caps.slurm_version or '?'} ({caps.cluster_name or 'cluster name unknown'}); "
            f"accounting {'on' if caps.sacct_available else 'OFF'}",
            findings=findings,
            data={"capabilities": asdict(caps), "config": config},
            source="sbatch/squeue/sacct/sstat/srun --version|--help + scontrol show config",
            observed_at=observed,
        )

    def _slurm_missing(self, exc: HpcLabError) -> CheckResult:
        """``sbatch --version`` failed: nothing can be submitted. Most often the login node's
        non-interactive shell (what SSH commands get) lacks ``module load slurm``/PATH setup."""
        missing = exc.details.get("exit_code") == 127 or "not found" in exc.message.lower()
        return CheckResult(
            name="scheduler",
            status="fail",
            summary=f"Slurm is not usable from a non-interactive shell: {exc.message}"[:300],
            findings=[
                Finding(
                    code="slurm_commands_missing" if missing else "slurm_unusable",
                    severity="block",
                    message=(
                        "sbatch is not on the PATH of non-interactive SSH shells on the login node, so "
                        "hpclab cannot submit, observe or cancel jobs"
                        if missing
                        else f"`sbatch --version` fails on the login node: {exc.message}"
                    )[:500],
                    remediation="make Slurm available to non-interactive shells: put the site's "
                    "`module load slurm` (or PATH setup) in ~/.bashrc above its interactive-only guard, "
                    "or ask the admins; then re-run cluster_inspect",
                    evidence={"exit_code": exc.details.get("exit_code")},
                )
            ],
            source="sbatch --version",
            observed_at=self.now(),
        )

    def check_storage(self) -> CheckResult:
        findings: list[Finding] = []
        paths: list[tuple[str, str]] = []
        try:
            remote_root = self.rc.remote_root()
            paths.append(("remote_root", remote_root))
        except HpcLabError as exc:
            remote_root = None
            findings.append(
                Finding(
                    code="remote_root_unresolved",
                    severity="block",
                    message=exc.message,
                    remediation=str(exc.details.get("hint") or "") or None,
                )
            )
        for role in sorted(self.rc.profile.storage):
            p = self.rc.storage_path(role)
            if not p or "{" in p or not p.startswith("/"):
                findings.append(
                    Finding(
                        code="storage_role_unresolved",
                        severity="info",
                        message=f"storage role {role!r} path cannot be resolved ({p!r}); set the overlay user",
                    )
                )
                continue
            paths.append((role, p))
        rows: dict[str, dict[str, Any]] = {}
        if paths:
            res = self.transport().run_script(STORAGE_SNIPPET, timeout=20.0, args=[p for _, p in paths])
            current: str | None = None
            for line in res.stdout.splitlines():
                parts = line.split("\t")
                if parts[0] == "PATH" and len(parts) >= 3:
                    current = parts[1]
                    rows[current] = {"path": parts[1], "checked": parts[2], "exists": parts[1] == parts[2]}
                elif parts[0] == "DF" and len(parts) >= 2 and current is not None:
                    rows[current].update(_parse_df(parts[1]))
        entries: list[dict[str, Any]] = []
        by_mount: dict[str, list[dict[str, Any]]] = {}
        for role, p in paths:
            info = {"role": role, **rows.get(p, {"path": p})}
            entries.append(info)
            if info.get("capacity_pct") is None:
                findings.append(
                    Finding(
                        code="df_unavailable",
                        severity="warn",
                        message=f"free space of {role} ({p}) is unknown",
                    )
                )
                continue
            by_mount.setdefault(str(info.get("mount") or info.get("filesystem")), []).append(info)
        for mount, infos in by_mount.items():  # one finding per filesystem, naming every role on it
            cap = int(infos[0]["capacity_pct"])
            avail_gib = (infos[0].get("avail_kib") or 0) / (1024 * 1024)
            roles = ", ".join(i["role"] for i in infos)
            if cap >= 90:
                findings.append(
                    Finding(
                        code="filesystem_full" if cap >= 99 else "filesystem_nearly_full",
                        severity="block" if cap >= 99 else "warn",
                        message=f"filesystem {mount} ({roles}) is {cap}% full ({avail_gib:.1f} GiB free)",
                        evidence={"mount": mount, "roles": [i["role"] for i in infos], "capacity_pct": cap},
                    )
                )
        if remote_root is not None:
            for role, rule in self.rc.profile.storage.items():
                base = self.rc.storage_path(role)
                if (
                    base
                    and remote_root.startswith(base.rstrip("/") + "/")
                    and {"outputs", "checkpoints"} & set(rule.avoid_for)
                ):
                    findings.append(
                        Finding(
                            code="remote_root_on_avoided_storage",
                            severity="warn",
                            message=f"remote_root is under the {role!r} storage, which the site profile says to avoid "
                            f"for {', '.join(rule.avoid_for)}",
                            remediation=f"hpclab cluster add {self.cid} --remote-root <scratch path>",
                        )
                    )
        quota = _quota(self.transport())
        if quota.get("over_quota"):
            findings.append(
                Finding(code="over_quota", severity="warn", message="`quota` reports a filesystem over quota")
            )
        status: CheckStatus
        if any(f.severity == "block" for f in findings):
            status = "fail"
        elif paths and all(e.get("capacity_pct") is None for e in entries):
            status = "unavailable"
        elif any(f.severity == "warn" for f in findings):
            status = "warn"
        else:
            status = "ok"
        return CheckResult(
            name="storage",
            status=status,
            summary=", ".join(
                f"{e['role']} {e['capacity_pct']}% used"
                if e.get("capacity_pct") is not None
                else f"{e['role']} ?"
                for e in entries
            )
            or "no storage paths to check",
            findings=findings,
            data={"paths": entries, "quota": quota},
            source="df -Pk (nearest existing directory) + quota -s",
            observed_at=self.now(),
        )

    def check_runtime(self) -> CheckResult:
        res = self.transport().run_script(RUNTIME_SNIPPET, timeout=20.0, args=list(RUNTIME_TOOLS))
        tools: dict[str, dict[str, str | None]] = {}
        for line in res.stdout.splitlines():
            parts = line.split("\t")
            if parts[0] == "TOOL" and len(parts) >= 4:
                tools[parts[1]] = {"path": parts[2] or None, "version": parts[3] or None}
        if not tools:
            return CheckResult(
                name="runtime",
                status="unavailable",
                summary="could not list login-node tools",
                findings=[
                    Finding(
                        code="runtime_unavailable",
                        severity="warn",
                        message=res.stderr.strip()[-300:] or "no output",
                    )
                ],
                source="command -v",
                observed_at=self.now(),
            )
        findings: list[Finding] = []
        ssh = self.rc.profile.transport == "ssh"
        if not (tools.get("rsync") or {}).get("path"):
            findings.append(
                Finding(
                    code="rsync_missing",
                    severity="block" if ssh else "info",
                    message="rsync is not on the login node's PATH: code snapshots cannot be uploaded",
                    remediation="ask the admins for rsync, or `module load rsync` in ~/.bashrc for non-interactive shells",
                )
            )
        if not (tools.get("python3") or {}).get("path"):
            findings.append(
                Finding(
                    code="python3_missing",
                    severity="warn",
                    message="python3 is missing on the login node: hpclab's file helper falls back to coreutils "
                    "(directory listings unsupported)",
                )
            )
        if not (tools.get("uv") or {}).get("path"):
            findings.append(
                Finding(
                    code="uv_missing",
                    severity="info",
                    message="uv is not installed on the login node (optional)",
                )
            )
        status: CheckStatus = (
            "fail"
            if any(f.severity == "block" for f in findings)
            else ("warn" if any(f.severity == "warn" for f in findings) else "ok")
        )
        return CheckResult(
            name="runtime",
            status=status,
            summary=", ".join(f"{k} {'ok' if v.get('path') else 'missing'}" for k, v in tools.items()),
            findings=findings,
            data={"tools": tools},
            source="command -v + --version (login node; no compute)",
            observed_at=self.now(),
        )

    def check_queue(self) -> CheckResult:
        sched = self.sched()
        user = self.user()
        if not user:
            raise HpcLabError("the remote user is unknown", code=ErrorCode.UNAVAILABLE)
        res = sched.query_queue(user=user)
        jobs = [
            j
            for j in res.jobs
            if str(j.state) in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED")
        ]
        managed = _managed_job_ids(self.ctx, self.cid, jobs)
        findings: list[Finding] = []
        mine: dict[str, Any] = {
            "running_jobs": 0,
            "pending_jobs": 0,
            "running_gpus": 0,
            "pending_gpus": 0,
            "managed_jobs": 0,
            "foreign_jobs": 0,
            "foreign_gpus": 0,
            "gpu_counts_complete": True,
            "partitions": {},
        }
        never: list[str] = []
        for j in jobs:
            running = str(j.state) != "PENDING"
            key = "running" if running else "pending"
            mine[f"{key}_jobs"] += 1
            if j.alloc_gpus is None:
                mine["gpu_counts_complete"] = False
            else:
                mine[f"{key}_gpus"] += j.alloc_gpus
            is_managed = j.job_id in managed
            mine["managed_jobs" if is_managed else "foreign_jobs"] += 1
            if not is_managed and j.alloc_gpus:
                mine["foreign_gpus"] += j.alloc_gpus
            if j.partition:
                mine["partitions"][j.partition] = mine["partitions"].get(j.partition, 0) + 1
            if not running and j.reason:
                category, explanation = _pending_category(j.reason)
                if category == "never_starts":
                    never.append(f"{'hpclab' if is_managed else 'foreign'} job: {j.reason} ({explanation})")
        if not mine["gpu_counts_complete"]:
            for k in ("running_gpus", "pending_gpus", "foreign_gpus"):
                mine[k] = None  # some GPU counts are unknown: a sum would understate
        if never:
            findings.append(
                Finding(
                    code="pending_never_starts",
                    severity="warn",
                    message=f"{len(never)} pending job(s) cannot start as requested: " + "; ".join(never[:5]),
                    remediation="cancel and resubmit with an allowed shape (run_recover adjusts resources)",
                )
            )
        if mine["foreign_jobs"]:
            findings.append(
                Finding(
                    code="foreign_jobs",
                    severity="info",
                    message=f"{mine['foreign_jobs']} of your jobs were not submitted through hpclab; they count toward "
                    "your per-user limits",
                )
            )
        account_now = self._account_now(sched, user)
        if account_now is not None and account_now.get("visibility") == "unavailable":
            findings.append(
                Finding(
                    code="account_queue_unavailable",
                    severity="info",
                    message=f"the queue of account {account_now['account']} could not be read: "
                    f"{account_now.get('error') or 'unknown error'}"[:300],
                )
            )
        status: CheckStatus = "warn" if never else "ok"
        return CheckResult(
            name="queue",
            status=status,
            summary=f"{mine['running_jobs']} running / {mine['pending_jobs']} pending job(s) of {user} "
            f"({mine['managed_jobs']} via hpclab)",
            findings=findings,
            data={"user": user, "mine": mine, "account_now": account_now},
            source="squeue --user" + (" + squeue --account" if account_now else ""),
            observed_at=res.observed_at,
        )

    def _account_now(self, sched: Any, user: str) -> dict[str, Any] | None:
        account = self.rc.overlay.account
        query = getattr(sched, "query_account_queue", None)
        if not account or not callable(query):
            return None
        try:
            res = query(account)
        except HpcLabError as exc:
            if exc.code == ErrorCode.AUTH_REQUIRED:
                raise
            return {"account": account, "visibility": "unavailable", "error": exc.message[:300]}
        from hpclab.services.usage import visibility_for

        users = {j.user for j in res.jobs if j.user}
        caps = self.caps()
        vis, note = visibility_for(
            scope="account", me=user, users_seen=users, private_data=caps.private_data, source="squeue"
        )
        running = [j for j in res.jobs if str(j.state) != "PENDING"]
        pending = [j for j in res.jobs if str(j.state) == "PENDING"]

        def gpus(js: list[Any]) -> int | None:
            return None if any(j.alloc_gpus is None for j in js) else sum(int(j.alloc_gpus) for j in js)

        return {
            "account": account,
            "visibility": vis,
            "note": note,
            "running_gpus": gpus(running),
            "pending_gpus": gpus(pending),
            "running_jobs": len(running),
            "pending_jobs": len(pending),
            "users_visible": len(users),
            "observed_at": res.observed_at,
        }

    def partitions(self) -> list[Any]:
        if self._partitions is None:
            self._partitions = list(self.sched().partitions())
        return self._partitions

    def details(self) -> dict[str, Any]:
        if self._details is None:
            fn = getattr(self.sched(), "partition_details", None)
            self._details = {d.name: d for d in fn()} if callable(fn) else {}
        return self._details

    def check_capacity(self) -> CheckResult:
        parts = self.partitions()
        findings: list[Finding] = []
        try:
            details = self.details()
        except HpcLabError as exc:
            details = {}
            findings.append(
                Finding(code="partition_details_unavailable", severity="info", message=exc.message[:300])
            )
        observed = self.now()
        entries: list[dict[str, Any]] = []
        for p in parts:
            d = details.get(p.name)
            entries.append(
                {
                    "name": p.name,
                    "state": p.state,
                    "default": p.default,
                    "max_time_s": p.max_time_s,
                    "nodes_total": p.nodes_total,
                    "preemptible": p.preemptible,
                    "preempt_mode": getattr(d, "preempt_mode", None),
                    "grace_time_s": getattr(d, "grace_time_s", None),
                    "gpus": p.gpu_types,
                }
            )
            if p.state and p.state.lower() not in ("up",):
                findings.append(
                    Finding(
                        code="partition_not_up", severity="warn", message=f"partition {p.name} is {p.state}"
                    )
                )
            unavailable = sum(v.get("unavailable", 0) for v in p.gpu_types.values())
            if unavailable:
                findings.append(
                    Finding(
                        code="gpus_unavailable",
                        severity="info",
                        message=f"{p.name}: {unavailable} GPU(s) on drained/down nodes",
                    )
                )
        seen = {p.name for p in parts}
        missing = sorted(set(self.rc.profile.partitions) - seen)
        if missing:
            findings.append(
                Finding(
                    code="profile_partition_missing",
                    severity="warn",
                    message=f"partition(s) in the site profile not reported by sinfo: {', '.join(missing)}",
                    remediation="update the site profile (partitions may have been renamed or hidden from you)",
                )
            )
        status: CheckStatus = "warn" if any(f.severity == "warn" for f in findings) else "ok"
        total_free = sum(v.get("free", 0) for p in parts for v in p.gpu_types.values())
        return CheckResult(
            name="capacity",
            status=status,
            summary=f"{len(parts)} partition(s); {total_free} GPU(s) idle on healthy nodes right now",
            findings=findings,
            data={"partitions": entries},
            source="sinfo --Node + scontrol show partition",
            observed_at=observed,
        )

    def check_limits(self) -> CheckResult:
        sched = self.sched()
        user = self.user()
        if not user:
            raise HpcLabError("the remote user is unknown", code=ErrorCode.UNAVAILABLE)
        findings: list[Finding] = []
        assoc_fn = getattr(sched, "association_limits", None)
        assocs = assoc_fn(user) if callable(assoc_fn) else sched.associations(user)
        assoc_rows = [asdict(a) if hasattr(a, "__dataclass_fields__") else dict(a) for a in assocs]
        accounts = sorted({a["account"] for a in assoc_rows})
        allowed_qos = sorted({q for a in assoc_rows for q in (a.get("qos") or [])})
        want_account, want_qos = self.rc.overlay.account, self.rc.overlay.qos
        if not assoc_rows:
            findings.append(
                Finding(
                    code="no_associations",
                    severity="block",
                    message=f"user {user} has no Slurm association on this cluster: sbatch will be refused",
                    remediation="ask your PI/admins to add you to an account",
                )
            )
        elif want_account and want_account not in accounts:
            findings.append(
                Finding(
                    code="account_not_associated",
                    severity="block",
                    message=f"overlay account {want_account!r} is not one of your accounts ({', '.join(accounts)})",
                    remediation=f"hpclab cluster add {self.cid} --account <one of {', '.join(accounts)}>",
                )
            )
        if want_qos and allowed_qos and want_qos not in allowed_qos:
            findings.append(
                Finding(
                    code="qos_not_allowed",
                    severity="block",
                    message=f"overlay QOS {want_qos!r} is not allowed for you ({', '.join(allowed_qos)})",
                )
            )
        qos_rows: list[dict[str, Any]] | None = None
        qos_fn = getattr(sched, "qos_limits", None)
        names = sorted({*allowed_qos, *([want_qos] if want_qos else [])})
        if callable(qos_fn) and names:
            try:
                qos_rows = [asdict(q) for q in qos_fn(names)]
            except HpcLabError as exc:
                if exc.code == ErrorCode.AUTH_REQUIRED:
                    raise
                findings.append(
                    Finding(
                        code="qos_limits_unavailable",
                        severity="warn",
                        message=f"QOS limits are hidden or unavailable ({exc.message[:200]}); they are unknown, "
                        "not unlimited",
                    )
                )
        for q in qos_rows or []:
            if q.get("max_gpus_per_user") is not None:
                findings.append(
                    Finding(
                        code="per_user_gpu_limit",
                        severity="info",
                        message=f"QOS {q['name']}: at most {q['max_gpus_per_user']} GPU(s) per user at a time",
                    )
                )
        status: CheckStatus
        if any(f.severity == "block" for f in findings):
            status = "fail"
        elif any(f.severity == "warn" for f in findings):
            status = "warn"
        else:
            status = "ok"
        return CheckResult(
            name="limits",
            status=status,
            summary=f"{len(assoc_rows)} association(s), account(s): {', '.join(accounts) or 'none'}; "
            f"QOS limits {'unavailable' if qos_rows is None and names else 'read'}",
            findings=findings,
            data={
                "associations": assoc_rows,
                "qos": qos_rows,
                "accounts": accounts,
                "allowed_qos": allowed_qos,
            },
            source="sacctmgr show associations + sacctmgr show qos",
            observed_at=self.now(),
        )

    def check_nodes(self) -> CheckResult:
        from hpclab.adapters.slurm import node_availability
        from hpclab.profiles.loader import active_observations

        now = self.ctx.clock.now()
        findings: list[Finding] = []
        local = [
            {
                "key": r["obs_key"],
                "value": jload(r["value_json"], None),
                "source": r["source"],
                "observed_at": r["observed_at"],
                "expires_at": r["expires_at"],
            }
            for r in self.ctx.db.all(
                "SELECT * FROM observations WHERE cluster_id=? AND obs_key LIKE 'node:%' "
                "AND (expires_at IS NULL OR expires_at > ?) ORDER BY observed_at DESC LIMIT ?",
                (self.cid, iso(now), MAX_LISTED),
            )
        ]
        for obs in local:
            findings.append(
                Finding(
                    code="node_suspect",
                    severity="warn",
                    message=f"local observation {obs['key']} (source {obs['source']}, expires {obs['expires_at'] or 'never'})",
                    remediation="run_recover can exclude the node for retries",
                )
            )
        issues = [
            {
                "key": o.key,
                "value": o.value,
                "severity": o.severity,
                "source": o.source,
                "expires_at": o.expires_at,
            }
            for o in active_observations(self.rc.profile, now)
        ]
        for issue in issues:
            findings.append(
                Finding(
                    code="known_issue",
                    severity="warn" if issue["severity"] in ("warning", "critical") else "info",
                    message=f"site known issue {issue['key']}: {issue['value']}"[:300],
                )
            )
        data: dict[str, Any] = {"local_observations": local, "known_issues": issues}
        try:
            nodes = list(self.sched().nodes())
        except HpcLabError as exc:
            if exc.code == ErrorCode.AUTH_REQUIRED:
                raise
            data["cluster_nodes"] = None
            return CheckResult(
                name="nodes",
                status="unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable",
                summary=f"sinfo failed: {exc.message}"[:300],
                findings=[
                    Finding(code="sinfo_unavailable", severity="warn", message=exc.message[:300]),
                    *findings,
                ],
                data=data,
                source="sinfo --Node + local observations",
                observed_at=self.now(),
            )
        bad = [n for n in nodes if node_availability(n.state) == "unavailable"]
        data["nodes_total"] = len(nodes)
        data["unavailable_total"] = len(bad)
        data["unavailable"] = [
            {"name": n.name, "state": n.state, "reason": n.reason, "partitions": n.partitions}
            for n in bad[:MAX_LISTED]
        ]
        status: CheckStatus = "warn" if any(f.severity == "warn" for f in findings) else "ok"
        return CheckResult(
            name="nodes",
            status=status,
            summary=f"{len(bad)} of {len(nodes)} node(s) drained/down; {len(local)} local node observation(s)",
            findings=findings,
            data=data,
            source="sinfo --Node + local observations + profile known issues",
            observed_at=self.now(),
        )

    def check_profile_facts(self) -> CheckResult:
        from hpclab.profiles.loader import active_facts

        now = self.ctx.clock.now()
        active_keys = {(f.key, f.observed_at) for f in active_facts(self.rc.profile, now)}
        facts: list[dict[str, Any]] = []
        findings: list[Finding] = []
        for f in self.rc.profile.facts:
            observed = parse_iso(f.observed_at) if f.observed_at else None
            age = (now - observed).total_seconds() if observed else None
            expired = (f.key, f.observed_at) not in active_keys
            facts.append(
                {
                    "key": f.key,
                    "value": f.value,
                    "source": f.source,
                    "observed_at": f.observed_at,
                    "age_s": int(age) if age is not None else None,
                    "expires_at": f.expires_at,
                    "expired": expired,
                }
            )
            if expired:
                findings.append(
                    Finding(
                        code="fact_expired",
                        severity="info",
                        message=f"profile fact {f.key!r} expired at {f.expires_at}: treat it as unknown until re-checked",
                    )
                )
            elif age is not None and age > 180 * 86400:
                findings.append(
                    Finding(
                        code="fact_old",
                        severity="info",
                        message=f"profile fact {f.key!r} is {int(age // 86400)} days old",
                    )
                )
        return CheckResult(
            name="profile_facts",
            status="ok",
            summary=f"{len(facts)} site fact(s), {sum(1 for x in facts if x['expired'])} expired",
            findings=findings,
            data={"facts": facts, "profile_revision": self.rc.revision},
            source="site profile",
            observed_at=iso(now),
        )

    # -- candidate shapes ------------------------------------------------------------
    def candidates(self, shapes: list[CandidateShape]) -> list[CandidateEstimate]:
        conn = self.connect()
        out: list[CandidateEstimate] = []
        try:
            parts = {p.name: p for p in self.partitions()} if conn.ok else {}
        except HpcLabError:
            parts = {}
        try:
            details = self.details() if conn.ok else {}
        except HpcLabError:
            details = {}
        for shape in shapes:
            est = self._candidate(shape, conn, parts, details)
            out.append(est)
        return out

    def _candidate(
        self, shape: CandidateShape, conn: _Conn, parts: dict[str, Any], details: dict[str, Any]
    ) -> CandidateEstimate:
        part = parts.get(shape.partition)
        det = details.get(shape.partition)
        rule = self.rc.profile.partitions.get(shape.partition)
        blockers: list[str] = []
        limit_s = parse_slurm_duration(shape.time)
        max_time = (
            part.max_time_s
            if part is not None
            else (parse_slurm_duration(rule.max_time) if rule and rule.max_time else None)
        )
        if parts and part is None:
            blockers.append(f"partition {shape.partition!r} is not reported by sinfo")
        if rule is not None and rule.interactive_only:
            blockers.append(f"{shape.partition} is interactive-only (not for batch jobs)")
        if max_time is not None and limit_s is not None and limit_s > max_time:
            blockers.append(f"time {shape.time} exceeds the partition maximum ({max_time}s)")
        types = set(part.gpu_types) if part is not None else set(rule.gpu_types if rule else [])
        if shape.gpu_type and types and shape.gpu_type not in types:
            blockers.append(
                f"gpu_type {shape.gpu_type!r} is not in {shape.partition} ({', '.join(sorted(types))})"
            )
        free: int | None = None
        if part is not None:
            vals = [v for t, v in part.gpu_types.items() if shape.gpu_type is None or t == shape.gpu_type]
            free = sum(v.get("free", 0) for v in vals)
        base = CandidateEstimate(
            shape=shape,
            status="blocked" if blockers else "ok",
            free_now=free,
            max_time_s=max_time,
            preemptible=part.preemptible if part is not None else (rule.preemptible if rule else None),
            preempt_mode=getattr(det, "preempt_mode", None),
            grace_time_s=getattr(det, "grace_time_s", None),
            blockers=blockers,
            note="scheduler guess from sbatch --test-only: low confidence, not an ETA; nothing was submitted",
        )
        if blockers:
            return base
        if not conn.ok:
            return base.model_copy(
                update={"status": "unavailable", "note": f"not estimated: SSH state {conn.state}"}
            )
        fn = getattr(self.sched(), "test_only", None)
        if not callable(fn):
            return base.model_copy(
                update={"status": "unavailable", "note": "test-only estimates are not supported"}
            )
        account, qos = self.rc.overlay.account, self.rc.overlay.qos
        key = (
            CACHE_PREFIX
            + self.cid
            + ":"
            + digest({"shape": shape.model_dump(mode="json"), "account": account, "qos": qos}).split(":", 1)[
                1
            ][:32]
        )
        now = self.ctx.clock.now()
        row = self.ctx.db.one(
            "SELECT payload_json, created_at, expires_at FROM report_cache WHERE cache_key=?", (key,)
        )
        if row is not None and (row["expires_at"] or "") > (iso(now) or ""):
            cached_payload = jload(row["payload_json"], {}) or {}
            return _apply_estimate(base, cached_payload, cached=True, observed_at=row["created_at"])
        if self.over_budget():
            return base.model_copy(
                update={"status": "unavailable", "note": "skipped: request time budget used up"}
            )
        from hpclab.adapters.base import SubmissionRejected
        from hpclab.adapters.slurm import JobShape

        payload: dict[str, Any]
        try:
            est = fn(
                JobShape(
                    partition=shape.partition,
                    gpus=shape.gpus,
                    gpu_type=shape.gpu_type,
                    time=shape.time,
                    nodes=shape.nodes,
                    account=account,
                    qos=qos,
                )
            )
            payload = {"start_time": est.start_time, "nodes": getattr(est, "nodes", None), "rejected": None}
        except SubmissionRejected as exc:
            payload = {
                "start_time": None,
                "rejected": str(exc.details.get("reason_code") or exc.message)[:200],
            }
        except HpcLabError as exc:
            if exc.code == ErrorCode.AUTH_REQUIRED:
                raise
            return base.model_copy(
                update={"status": "unavailable", "note": f"test-only failed: {exc.message}"[:300]}
            )
        observed = iso(self.ctx.clock.now())
        with self.ctx.db.tx(source="api") as tx:
            tx.execute(
                "INSERT INTO report_cache(cache_key, payload_json, created_at, expires_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET payload_json=excluded.payload_json, "
                "created_at=excluded.created_at, expires_at=excluded.expires_at",
                (key, jdump(payload), observed, iso(now + timedelta(seconds=TEST_ONLY_TTL_S))),
            )
        return _apply_estimate(base, payload, cached=False, observed_at=observed)


def _apply_estimate(
    base: CandidateEstimate, payload: dict[str, Any], *, cached: bool, observed_at: str | None
) -> CandidateEstimate:
    rejected = payload.get("rejected")
    if rejected:
        return base.model_copy(
            update={
                "status": "blocked",
                "blockers": [*base.blockers, f"sbatch --test-only refused: {rejected}"],
                "cached": cached,
                "observed_at": observed_at,
            }
        )
    start = payload.get("start_time")
    return base.model_copy(
        update={
            "est_start": start,
            "est_source": "test_only" if start else "none",
            "cached": cached,
            "observed_at": observed_at,
        }
    )


def _parse_df(line: str) -> dict[str, Any]:
    """``Filesystem 1024-blocks Used Available Capacity Mounted-on`` (POSIX df -P -k)."""
    parts = line.split()
    if len(parts) < 6:
        return {}
    try:
        total, used, avail = int(parts[1]), int(parts[2]), int(parts[3])
        cap = int(parts[4].rstrip("%"))
    except ValueError:
        return {}
    return {
        "filesystem": parts[0],
        "total_kib": total,
        "used_kib": used,
        "avail_kib": avail,
        "capacity_pct": cap,
        "mount": " ".join(parts[5:]),
    }


def _quota(transport: Any) -> dict[str, Any]:
    """Best effort ``quota -s`` (bounded); unknown formats are kept as text, never parsed to 0."""
    try:
        res = transport.run(["quota", "-s"], timeout=10.0)
    except (TransportTimeout, ConnectionLost):
        return {"status": "unavailable", "detail": "quota timed out"}
    except AuthRequiredError:
        raise
    except TransportError as exc:
        return {"status": "unavailable", "detail": exc.message[:200]}
    if res.exit_code == 127 or "not found" in res.stderr.lower():
        return {"status": "unsupported", "detail": "quota is not installed"}
    lines = [ln.rstrip() for ln in (res.stdout + res.stderr).splitlines() if ln.strip()][:20]
    over = any("*" in ln and not ln.lower().startswith("disk quotas") for ln in lines)
    return {"status": "ok" if res.exit_code in (0, 1) else "unavailable", "text": lines, "over_quota": over}


def _managed_job_ids(ctx: ServiceContext, cluster_id: str, jobs: list[Any]) -> set[str]:
    """Job ids bound to hpclab attempts, plus jobs whose name carries an attempt's token."""
    ids = [j.job_id for j in jobs]
    out: set[str] = set()
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        out.update(
            str(r["job_id"])
            for r in ctx.db.all(
                f"SELECT job_id FROM attempts WHERE cluster_id=? AND job_id IN ({marks})",
                [cluster_id, *chunk],
            )
        )
    try:
        from hpclab.services.jobscript import token_from_job_name
    except ImportError:  # pragma: no cover - jobscript is a wave-2 module
        return out
    tokens: dict[str, str] = {}
    for j in jobs:
        tok = token_from_job_name(j.name or "") if j.job_id not in out else None
        if tok:
            tokens[tok] = j.job_id
    for tok, job_id in tokens.items():
        row = ctx.db.one(
            "SELECT 1 AS x FROM attempts WHERE cluster_id=? AND substr(submit_token, 1, ?) = ?",
            (cluster_id, len(tok), tok),
        )
        if row is not None:
            out.add(job_id)
    return out


def _pending_category(reason: str) -> tuple[str, str]:
    """(category, explanation) via services.pending when available, else a small table."""
    try:
        from hpclab.services.pending import decode_pending

        info = decode_pending(reason)
        return str(info.category), str(info.explanation)
    except ImportError:
        pass
    except Exception:  # a decoder bug must not break the doctor
        log.debug("decode_pending failed for %r", reason, exc_info=True)
    if any(reason.startswith(r) for r in _NEVER_STARTS) or (
        "ReqNodeNotAvail" in reason and "Reserved" in reason
    ):
        return "never_starts", "the request can never be satisfied as submitted"
    return "unknown", reason


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------

# Findings that mean "this check answered, but one of its sources did not": coverage is partial.
_PARTIAL_CODES = frozenset(
    {
        "config_unavailable",
        "qos_limits_unavailable",
        "partition_details_unavailable",
        "df_unavailable",
        "account_queue_unavailable",
    }
)

_SOURCE_STATUS: dict[str, SourceStatusValue] = {
    "ok": "ok",
    "warn": "ok",
    "fail": "ok",
    "unavailable": "unavailable",
    "unsupported": "unsupported",
}


def cluster_inspect(
    ctx: ServiceContext,
    *,
    cluster_id: str,
    checks: list[CheckName] | None = None,
    candidate_shapes: list[CandidateShape] | None = None,
) -> ClusterInspectData:
    """Check a registered cluster before using it: SSH auth state (with the exact remediation
    when MFA is needed), remote identity vs the overlay, Slurm capabilities and config facts,
    storage headroom, login-node tools, my jobs (managed vs foreign), GPU capacity, limits,
    unhealthy nodes and profile facts. Read-only. ``candidate_shapes`` (<= 3) adds cached,
    low-confidence ``sbatch --test-only`` start estimates."""
    rc = ctx.clusters.get(cluster_id)
    wanted: list[CheckName] = list(dict.fromkeys(checks)) if checks else list(ALL_CHECKS)
    shapes = list(candidate_shapes or [])
    if len(shapes) > MAX_CANDIDATES:
        raise InvalidArgument(
            f"at most {MAX_CANDIDATES} candidate_shapes per call", details={"given": len(shapes)}
        )
    shapes = [s if isinstance(s, CandidateShape) else CandidateShape.model_validate(s) for s in shapes]
    insp = _Inspector(ctx, rc)
    runners: dict[str, Callable[[], CheckResult]] = {
        "auth": insp.check_auth,
        "identity": insp.check_identity,
        "scheduler": insp.check_scheduler,
        "storage": insp.check_storage,
        "runtime": insp.check_runtime,
        "queue": insp.check_queue,
        "capacity": insp.check_capacity,
        "limits": insp.check_limits,
        "nodes": insp.check_nodes,
        "profile_facts": insp.check_profile_facts,
    }
    results: dict[str, CheckResult] = {}
    coverage = Coverage()
    order = [c for c in ALL_CHECKS if c in wanted]  # auth first, then cheap-to-expensive
    for name in order:
        result, ran = insp.run(name, runners[name])
        results[name] = result
        if not ran:
            blocked = bool(result.findings) and result.findings[0].code == "blocked_by_auth"
            coverage.add(name, "unavailable" if blocked else "skipped", detail=result.summary)
            coverage.complete = False  # asked for, not answered (auth or time budget)
            coverage.missing.append(name)
        else:
            status = _SOURCE_STATUS.get(result.status, "ok")
            if any(f.code in ("slurm_commands_missing", "slurm_unusable") for f in result.findings):
                status = "unsupported"  # the scheduler facts themselves are unknown
            elif status == "ok" and any(f.code in _PARTIAL_CODES for f in result.findings):
                status = "partial"
            coverage.add(name, status, observed_at=result.observed_at, detail=result.source)
    candidates = insp.candidates(shapes) if shapes else []
    checks_out = [results[n] for n in wanted]
    statuses = {c.status for c in checks_out}
    overall: Literal["ok", "warn", "fail"] = (
        "fail"
        if "fail" in statuses
        else ("warn" if statuses & {"warn", "unavailable", "unsupported", "skipped"} else "ok")
    )
    if any(
        c.status == "skipped" and c.findings and c.findings[0].code == "blocked_by_auth" for c in checks_out
    ):
        overall = "fail"
    remediation: list[str] = []
    for c in checks_out:
        for f in c.findings:
            if f.remediation and f.severity in ("block", "warn") and f.remediation not in remediation:
                remediation.append(f.remediation)
    return ClusterInspectData(
        cluster_id=cluster_id,
        status=overall,
        checks=checks_out,
        candidates=candidates,
        remediation=remediation[:10],
        coverage=coverage,
        observed_at=iso(ctx.clock.now()) or "",
    )


# ---------------------------------------------------------------------------
# Controller wiring
# ---------------------------------------------------------------------------


def outbox_handlers() -> dict[str, Any]:
    return {}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    return [PeriodicTask("cluster.cache_gc", 3600.0, gc_cache, jitter_s=60.0, run_at_start=False)]


def gc_cache(ctx: ServiceContext) -> None:
    """Drop expired test-only estimates (only this module's cache keys)."""
    with ctx.db.tx(source="worker") as tx:
        tx.execute(
            "DELETE FROM report_cache WHERE cache_key LIKE ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (f"{CACHE_PREFIX}%", tx.now),
        )
