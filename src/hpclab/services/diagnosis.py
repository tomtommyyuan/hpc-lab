"""Failure diagnosis for one Attempt: scheduler facts + log tails -> ranked, actionable causes.

Pure and deterministic (no I/O, no clock). ``observe.finalize`` stores the result in
``attempts.diagnosis_json``; ``run_inspect`` shows it; ``run_recover`` reads ``primary_code`` and
``resource_hint`` to pre-fill a recovery plan; the workflow circuit breaker groups failures by
``CODE_CATEGORY`` and ``normalize_error_line(last_error_line)``.

Ranking rules (why the first item is the primary cause):

1. Scheduler truth first. A job the scheduler stopped from outside (TIMEOUT, OUT_OF_MEMORY,
   PREEMPTED, NODE_FAIL, CANCELLED, or the equivalent ``slurmstepd`` kill markers in the log) is
   diagnosed by that cause; tracebacks printed while it was being killed are consequences.
2. Specific beats generic: a CUDA OOM traceback is ``cuda_oom``, never ``python_exception``; a
   rank that crashed beats the NCCL errors it caused on the other ranks.
3. Then confidence, then a fixed specificity order (infrastructure > environment > program).
4. The hpclab wrapper's own stderr lines (``HPCLAB_GUARD_FAILED:<check>``, ``[hpclab] finished:
   ... failed_phase=setup|prepare``, ``[hpclab] received SIGUSR1``) are trusted like exit
   statuses. A program that stops on the walltime warning signal (checkpoint_and_exit) exits
   128+USR1 *before* the limit, so Slurm says FAILED; it is diagnosed as a planned stop to
   resume (code ``timeout``), never as a crash to retry from scratch.
5. Generic file/permission lines are common harmless noise: when an unexplained traceback ends
   the log, they are capped at medium. Side observations (a transient fp16 NaN) never become
   the primary cause of a failed job; ``unknown_failure`` does.

Every item carries at most five redacted evidence lines (<= 300 chars each) so a human or LLM can
check the claim. Suggested strategies never hide semantic changes: shrinking the batch after a
CUDA OOM is a new experiment (``derive_run``), not a resource adjustment, and a NaN loss is never
answered with "retry the same config".

``resource_hint`` keys named like ``ResourceSpec`` fields (``RESOURCE_HINT_KEYS``) can be turned
into a resource patch with :func:`resource_patch_from_hint`; all other keys are informational.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hpclab.domain.clock import format_slurm_duration
from hpclab.domain.states import (
    SCHEDULER_ACTIVE,
    SCHEDULER_TERMINAL,
    AppState,
    SchedulerState,
    normalize_slurm_state,
)

__all__ = [
    "CODE_CATEGORY",
    "DIAGNOSIS_CODES",
    "MAX_EVIDENCE_CHARS",
    "MAX_EVIDENCE_LINES",
    "RESOURCE_HINT_KEYS",
    "STRATEGIES",
    "AttemptFacts",
    "Confidence",
    "Diagnosis",
    "DiagnosisItem",
    "Strategy",
    "diagnose",
    "normalize_error_line",
    "redact_secrets",
    "resource_patch_from_hint",
]

Confidence = Literal["high", "medium", "low"]
Strategy = Literal[
    "retry",
    "resume",
    "adjust_resources",
    "derive_run",
    "fix_code",
    "fix_environment",
    "wait",
    "contact_support",
    "none",
]
STRATEGIES: frozenset[str] = frozenset(get_args(Strategy))
Category = Literal[
    "infrastructure", "resources", "environment", "program", "numerics", "cancelled", "results", "unknown"
]

MAX_EVIDENCE_LINES = 5
MAX_EVIDENCE_CHARS = 300
MAX_ITEMS = 8
_MAX_SCAN_CHARS = 512 * 1024
_MAX_SCAN_LINES = 5000
_MAX_LINE_CHARS = 2000
_GIB = 1024**3

# ResourceSpec-compatible hint keys (see resource_patch_from_hint); everything else is informational.
RESOURCE_HINT_KEYS: frozenset[str] = frozenset(
    {"time", "mem", "gpus", "gpu_type", "partition", "exclude_nodes", "extra_sbatch"}
)

_TITLES: dict[str, str] = {
    "preempted": "Preempted by the scheduler",
    "node_fail": "Node failure",
    "host_oom": "Host memory (RAM) exhausted",
    "timeout": "Walltime limit reached",
    "cancelled": "Cancelled",
    "gpu_hardware_error": "GPU hardware fault",
    "nfs_stale": "Network filesystem error",
    "env_guard_failed": "Environment guard failed before the program started",
    "cuda_driver_mismatch": "PyTorch CUDA build does not match the node's driver or GPU",
    "no_gpu": "No usable GPU visible to the program",
    "command_not_found": "Command not found on the compute node",
    "import_error": "Missing or broken Python package",
    "disk_quota": "Disk quota exceeded or disk full",
    "network_unreachable": "No network access from the compute node",
    "cuda_oom": "GPU memory exhausted (CUDA OOM)",
    "nan_loss": "Loss became NaN/Inf",
    "cuda_device_assert": "CUDA kernel assertion / illegal memory access",
    "state_dict_mismatch": "Checkpoint does not match the model",
    "weights_only_load": "torch.load refused the checkpoint (weights_only)",
    "checkpoint_corrupt": "Checkpoint file truncated or corrupt",
    "file_not_found": "File or directory not found",
    "permission_denied": "Permission denied / read-only filesystem",
    "port_in_use": "Rendezvous port already in use",
    "segfault": "Native crash (segmentation fault)",
    "python_exception": "Unhandled Python exception",
    "nccl_error": "Distributed communication (NCCL / c10d) failure",
    "killed_signal": "Killed by a signal",
    "success_no_outputs": "Completed, but results are missing or unverified",
    "unknown_failure": "Failure without a recognised signature",
}
DIAGNOSIS_CODES: frozenset[str] = frozenset(_TITLES)

CODE_CATEGORY: dict[str, Category] = {
    "preempted": "infrastructure",
    "node_fail": "infrastructure",
    "gpu_hardware_error": "infrastructure",
    "nfs_stale": "infrastructure",
    "host_oom": "resources",
    "timeout": "resources",
    "cuda_oom": "resources",
    "env_guard_failed": "environment",
    "cuda_driver_mismatch": "environment",
    "no_gpu": "environment",
    "command_not_found": "environment",
    "import_error": "environment",
    "disk_quota": "environment",
    "network_unreachable": "environment",
    "file_not_found": "environment",
    "permission_denied": "environment",
    "port_in_use": "environment",
    "segfault": "environment",
    "python_exception": "program",
    "cuda_device_assert": "program",
    "state_dict_mismatch": "program",
    "weights_only_load": "program",
    "checkpoint_corrupt": "program",
    "nccl_error": "program",
    "nan_loss": "numerics",
    "cancelled": "cancelled",
    "success_no_outputs": "results",
    "killed_signal": "unknown",
    "unknown_failure": "unknown",
}

# Specificity order among causes of equal authority and confidence (lower = shown first).
_RANK: dict[str, int] = {
    "preempted": 0,
    "node_fail": 1,
    "host_oom": 2,
    "timeout": 3,
    "cancelled": 4,
    "gpu_hardware_error": 10,
    "nfs_stale": 11,
    "env_guard_failed": 12,
    "cuda_driver_mismatch": 13,
    "no_gpu": 14,
    "command_not_found": 15,
    "import_error": 16,
    "disk_quota": 17,
    "network_unreachable": 18,
    "cuda_oom": 20,
    "nan_loss": 21,
    "cuda_device_assert": 22,
    "state_dict_mismatch": 23,
    "weights_only_load": 24,
    "checkpoint_corrupt": 25,
    "file_not_found": 26,
    "permission_denied": 27,
    "port_in_use": 28,
    "segfault": 30,
    "python_exception": 31,
    "nccl_error": 32,
    "killed_signal": 40,
    "success_no_outputs": 45,
    "unknown_failure": 99,
}
_CONF_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AttemptFacts(_Frozen):
    """What the ledger/scheduler/wrapper know about one Attempt. Unknown = ``None``.

    Strings from the ledger are accepted: ``scheduler_state`` may be raw Slurm text
    (``"CANCELLED by 1234"``, ``"OUT_OF_ME+"``) and ``app_state`` any ``AppState`` value.
    The optional fields after ``preemptible`` sharpen the advice when the caller knows them.
    """

    scheduler_state: SchedulerState = SchedulerState.UNKNOWN
    raw_state: str | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    app_state: AppState = AppState.UNKNOWN
    reason: str | None = None
    elapsed_s: int | None = None
    time_limit_s: int | None = None
    max_rss_bytes: int | None = None
    mem_request_bytes: int | None = None
    gpus: int | None = None
    gpu_type: str | None = None
    node_list: str | None = None
    restart_count: int = 0
    partition: str | None = None
    partition_max_time_s: int | None = None
    preemptible: bool | None = None
    # True/False when the caller checked for a usable checkpoint of this Run; None = not checked.
    checkpoint_available: bool | None = None
    # Training progress (0..1) at the end of the attempt, from services.progress; sizes walltime hints.
    progress_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    # Declared required outputs that are missing (from artifacts indexing).
    missing_outputs: tuple[str, ...] = ()
    result_state: str | None = None
    # Primary diagnosis codes of earlier attempts of the same Run (oldest first).
    previous_codes: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        state = data.get("scheduler_state")
        if state is None or state == "" or state == SchedulerState.UNKNOWN:
            data["scheduler_state"] = normalize_slurm_state(data.get("raw_state"))
        elif not isinstance(state, SchedulerState):
            data["scheduler_state"] = normalize_slurm_state(str(state))
        app = data.get("app_state")
        if app is None:
            data["app_state"] = AppState.UNKNOWN
        elif not isinstance(app, AppState):
            try:
                data["app_state"] = AppState(str(app).lower())
            except ValueError:
                data["app_state"] = AppState.UNKNOWN
        return data

    @field_validator("restart_count", mode="before")
    @classmethod
    def _restart(cls, v: Any) -> Any:
        return 0 if v is None else v


class DiagnosisItem(_Frozen):
    """One candidate cause with its evidence and what to do about it."""

    code: str
    title: str
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)
    explanation: str
    suggested_strategies: list[Strategy] = Field(default_factory=list)
    resource_hint: dict[str, str] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def _bound_evidence(cls, v: list[str]) -> list[str]:
        """Evidence is bounded and redacted however the item was built (incl. from stored JSON)."""
        return [_clip(redact_secrets(e), MAX_EVIDENCE_CHARS) for e in v[:MAX_EVIDENCE_LINES]]

    @field_validator("title", "explanation")
    @classmethod
    def _redact_text(cls, v: str) -> str:
        """Prose quotes log fragments (paths, module names, exception messages): redact it too."""
        return redact_secrets(v)

    @field_validator("resource_hint")
    @classmethod
    def _redact_hint(cls, v: dict[str, str]) -> dict[str, str]:
        return {k: _clip(redact_secrets(val), MAX_EVIDENCE_CHARS) for k, val in v.items()}

    @property
    def category(self) -> Category:
        return CODE_CATEGORY.get(self.code, "unknown")


class Diagnosis(_Frozen):
    """Ranked causes; ``items[0]`` is the primary one (``primary_code``), or none at all.

    ``last_error_line`` is the most relevant error line (redacted), for fingerprinting failures
    across cells with :func:`normalize_error_line`.
    """

    items: list[DiagnosisItem] = Field(default_factory=list)
    primary_code: str | None = None
    summary: str
    last_error_line: str | None = None

    @field_validator("summary", "last_error_line")
    @classmethod
    def _redact(cls, v: str | None) -> str | None:
        return None if v is None else redact_secrets(v)

    @property
    def primary(self) -> DiagnosisItem | None:
        return self.items[0] if self.items else None

    def item(self, code: str) -> DiagnosisItem | None:
        return next((i for i in self.items if i.code == code), None)

    def codes(self) -> list[str]:
        return [i.code for i in self.items]


def resource_patch_from_hint(hint: dict[str, str]) -> dict[str, Any]:
    """Turn the ``ResourceSpec``-compatible keys of a ``resource_hint`` into a resource patch.

    ``exclude_nodes`` is split on commas outside hostlist brackets (``a[1-2],b`` -> 2 entries),
    ``extra_sbatch`` on whitespace, ``gpus`` to int. Informational keys are dropped.
    """
    patch: dict[str, Any] = {}
    for key, value in hint.items():
        if key not in RESOURCE_HINT_KEYS or not value:
            continue
        if key == "exclude_nodes":
            patch[key] = _split_hostlist(value)
        elif key == "extra_sbatch":
            patch[key] = value.split()
        elif key == "gpus":
            try:
                patch[key] = int(value)
            except ValueError:
                continue
        else:
            patch[key] = value
    return patch


def _split_hostlist(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in value:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Redaction and evidence formatting
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A whole PEM block when the text spans lines; otherwise the rest of the BEGIN line.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----|.*)", re.DOTALL
        ),
        "[REDACTED PRIVATE KEY]",
    ),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_[REDACTED]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "github_pat_[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "gh_[REDACTED]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), "glpat-[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "sk-[REDACTED]"),
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), "xox-[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "AKIA[REDACTED]"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}"), r"\1 [REDACTED]"),
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1[REDACTED]@"),
    (
        re.compile(
            r"(?i)\b([\w.-]{0,40}(?:password|passwd|pwd|secret)[\w.-]{0,40}[\"']?\s*[:=]\s*[\"']?)[^\s\"',;]+"
        ),
        r"\1[REDACTED]",
    ),
    (
        # token/api-key assignments; the value must look like a credential (letters AND digits,
        # >= 8 chars) so "max_tokens=2048" stays readable.
        re.compile(
            r"(?i)\b([\w.-]{0,40}(?:token|api[_-]?key|access[_-]?key|auth[_-]?key)[\w.-]{0,40}[\"']?\s*[:=]\s*[\"']?)"
            r"(?=[^\s\"',;]*[A-Za-z])(?=[^\s\"',;]*\d)[^\s\"',;]{8,}"
        ),
        r"\1[REDACTED]",
    ),
)


def redact_secrets(text: str) -> str:
    """Replace credential-looking substrings (HF/GitHub/OpenAI/Slack/AWS tokens, bearer headers,
    URL userinfo, ``password=``/``token=`` assignments, private keys) with ``[REDACTED]``."""
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _clip(text: str, limit: int, focus: int = 0) -> str:
    """Shorten to ``limit`` chars, keeping the region around ``focus`` visible."""
    if len(text) <= limit:
        return text
    if focus + 40 <= limit - 3:
        return text[: limit - 3] + "..."
    start = max(0, focus - 40)
    return ("..." + text[start : start + limit - 6] + "...")[:limit]


def _evidence_line(prefix: str, text: str, needle: str = "") -> str:
    clean = redact_secrets(text.strip())
    focus = clean.find(needle[:24]) if needle else 0
    return prefix + _clip(clean, MAX_EVIDENCE_CHARS - len(prefix), max(focus, 0))


def _fmt_dur(seconds: int | None) -> str:
    return format_slurm_duration(seconds) if seconds is not None and seconds >= 0 else "?"


def _fmt_gib(n_bytes: float) -> str:
    return f"{n_bytes / _GIB:.1f} GiB"


def _round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def _first_sentence(text: str) -> str:
    idx = text.find(". ")
    return text if idx < 0 else text[: idx + 1]


_SIZE_RE = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s*(?P<u>[KMGT]i?B|B)\b")
_SIZE_MULT = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _parse_size(text: str) -> float | None:
    m = _SIZE_RE.search(text)
    if not m:
        return None
    return float(m["n"]) * _SIZE_MULT[m["u"][0]]


# ---------------------------------------------------------------------------
# Log normalization and scanning
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# torchrun/PyTorch rank prefixes ("[rank0]: ", "[default0]:") and srun --label ("0: ").
_PREFIX_RE = re.compile(r"^(?:\[(?:rank|default)\d+\]:[ ]?|\d{1,5}: )")


@dataclass(frozen=True)
class _LogLine:
    stream: str
    text: str


_PEM_REDACTED = "[REDACTED PRIVATE KEY]"


def _split_log(text: str, stream: str) -> list[_LogLine]:
    """Normalised lines of one stream. Lines inside a PEM private-key block are replaced
    here, because evidence is quoted line by line and a key body line alone matches no
    single-line secret pattern."""
    if not text:
        return []
    if len(text) > _MAX_SCAN_CHARS:
        text = text[-_MAX_SCAN_CHARS:]
        text = text[text.find("\n") + 1 :]
    raw_lines = text.split("\n")[-_MAX_SCAN_LINES:]
    out: list[_LogLine] = []
    in_key = False
    for raw in raw_lines:
        line = raw.rstrip("\r")
        if "\r" in line:  # progress bars redraw with CR: keep the last frame only
            line = line.rsplit("\r", 1)[-1]
        line = _PREFIX_RE.sub("", _ANSI_RE.sub("", line), count=1)
        if in_key or ("-----BEGIN" in line and "PRIVATE KEY-----" in line):
            in_key = not ("-----END" in line and "PRIVATE KEY-----" in line.rsplit("-----END", 1)[1])
            line = _PEM_REDACTED
        out.append(_LogLine(stream, line[:_MAX_LINE_CHARS]))
    return out


@dataclass(frozen=True)
class _Rule:
    code: str
    pattern: re.Pattern[str]
    confidence: Confidence
    generic: bool = False  # only matches lines no specific rule claimed
    exclude: re.Pattern[str] | None = None


def _rule(
    code: str,
    pattern: str,
    confidence: Confidence = "high",
    *,
    flags: int = 0,
    generic: bool = False,
    exclude: str | None = None,
) -> _Rule:
    return _Rule(
        code, re.compile(pattern, flags), confidence, generic, re.compile(exclude, flags) if exclude else None
    )


_I = re.IGNORECASE

# Codes starting with "_" are private markers consumed by the scheduler-level builders.
_RULES: tuple[_Rule, ...] = (
    _rule(
        "_cancel_marker",
        r"\*\*\*\s*(?P<scope>JOB|STEP)\s+(?P<job>\S+)\s+ON\s+(?P<node>\S+)\s+CANCELLED AT\s+(?P<at>\S+)"
        r"(?:\s+DUE\s+TO\s+(?P<why>[A-Z][A-Z ]*[A-Z]))?",
    ),
    _rule("_stepd_unkillable", r"STEPD TERMINATED ON\s+(?P<node>\S+).*DUE TO JOB NOT ENDING WITH SIGNALS"),
    # hpclab wrapper log lines (services/jobscript.py, ``[hpclab] ...`` on stderr)
    _rule("_hpclab_signal", r"^\[hpclab\] received SIG(?P<sig>[A-Z0-9]+)\b"),
    _rule("_guard_msg", r"^\[hpclab\] guard check '(?P<check>[A-Za-z0-9_.-]+)' failed: "),
    _rule(
        "_wrapper_finished",
        r"^\[hpclab\] finished: state=(?P<state>\w+) exit_code=(?P<rc>-?\d+)(?: signal=\S+)?"
        r"(?: failed_phase=(?P<phase>\w+))?(?: \((?P<err>.*)\))?\s*$",
    ),
    _rule("_setup_failed", r"^\[hpclab\] setup\[(?P<idx>\d+)\] failed with exit code (?P<rc>\d+)"),
    _rule(
        "_oom_kill",
        r"oom[-_ ]kill|cgroup out-of-memory handler|Exceeded (?:job|step) memory limit|exceeded memory limit"
        r"|Memory cgroup out of memory|have been OOM Killed",
        flags=_I,
    ),
    _rule("_killed", r"(?:^|[\s:])Killed(?:\s|$)"),
    _rule(
        "_dl_killed",
        r"DataLoader worker \(pid(?:\(s\))?\s[^)]*\)\s(?:is killed by signal|exited unexpectedly)",
        flags=_I,
    ),
    _rule(
        "_shm",
        r"insufficient shared memory|(?:No space left on device|Bus error).{0,120}/dev/shm"
        r"|/dev/shm.{0,120}(?:No space left|Bus error)",
        flags=_I,
    ),
    _rule("_guard", r"HPCLAB_GUARD_FAILED:(?P<check>[A-Za-z0-9_.-]+)"),
    _rule(
        "gpu_hardware_error",
        r"uncorrectable ECC error|ECC error encountered|fallen off the bus|GPU is lost|NVML_ERROR_GPU_IS_LOST"
        r"|NVRM: Xid|\bXid \(PCI",
    ),
    _rule("gpu_hardware_error", r"CUDA-capable device\(s\) is/are busy or unavailable", "medium"),
    _rule("nfs_stale", r"Stale file handle|\bESTALE\b|\[Errno 116\]"),
    _rule("nfs_stale", r"\[Errno 5\] Input/output error|: Input/output error", "medium"),
    _rule(
        "cuda_driver_mismatch",
        r"driver (?:on your system )?is too old|driver too old|\(found version \d{4,5}\)"
        r"|CUDA driver version is insufficient|no kernel image is available for execution"
        r"|sm_\d+ is not compatible|not compatible with the current PyTorch installation"
        r"|unsupported PTX version|UNSUPPORTED_PTX_VERSION|forward compatibility was attempted on non supported HW",
        flags=_I,
    ),
    _rule(
        "no_gpu",
        r"Found no NVIDIA driver|No CUDA GPUs are available|Torch not compiled with CUDA enabled"
        r"|no CUDA-capable device is detected|CUDA driver initialization failed",
        flags=_I,
    ),
    _rule(
        "command_not_found",
        # literal anchors only: a leading greedy class here backtracks quadratically on long lines
        r": command not found|execve\(\): [^:]{1,300}: No such file or directory|env: \S{1,300}: No such file or directory",
    ),
    _rule(
        "import_error",
        r"ModuleNotFoundError|No module named|ImportError|cannot open shared object file|undefined symbol: "
        r"|GLIBC(?:XX)?_[\d.]+' not found",
    ),
    _rule(
        "disk_quota",
        r"Disk quota exceeded|No space left on device|\[Errno 122\]|\[Errno 28\]",
        flags=_I,
        exclude=r"/dev/shm|shared memory",  # noqa: S108 - regex text, not a temp path
    ),
    _rule("disk_quota", r"PytorchStreamWriter failed writing|unexpected pos \d+ vs \d+", "medium"),
    _rule(
        "network_unreachable",
        r"couldn't connect to 'https?://huggingface\.co|LocalEntryNotFoundError|Temporary failure in name resolution"
        r"|Failed to resolve '|Name or service not known|Could not resolve host|Network is unreachable",
        flags=_I,
    ),
    _rule(
        "network_unreachable",
        r"Max retries exceeded with url|Failed to establish a new connection|offline mode is enabled",
        "medium",
        flags=_I,
    ),
    _rule(
        "cuda_oom",
        r"CUDA out of memory|OutOfMemoryError: CUDA|CUDA error: out of memory|Cuda failure 'out of memory'"
        r"|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|RESOURCE_EXHAUSTED: Out of memory"
        r"|OOM when allocating tensor|HIP out of memory",
        flags=_I,
    ),
    _rule(
        "nan_loss",
        r"\bloss (?:is|became|has become|went to) (?:nan|inf)|\bnan loss|non-finite loss|loss is not finite"
        r"|FloatingPointError|Detected (?:a )?NaN|NaN detected|nan or inf found|returned nan values"
        r"|probability tensor contains either|loss scale already at minimum|\"nonfinite\": ?\{[^}]*\"(?:[\w/]*_)?loss\"",
        flags=_I,
    ),
    _rule(
        "cuda_device_assert",
        r"device-side assert triggered|Assertion `[^`]*` failed|an illegal memory access was encountered"
        r"|CUBLAS_STATUS_EXECUTION_FAILED|index out of range in self",
    ),
    _rule(
        "state_dict_mismatch",
        r"Error\(s\) in loading state_dict|size mismatch for [\w.]+|Missing key\(s\) in state_dict"
        r"|Unexpected key\(s\) in state_dict",
    ),
    _rule("weights_only_load", r"Weights only load failed|WeightsUnpickler error"),
    _rule(
        "checkpoint_corrupt",
        r"PytorchStreamReader failed|failed finding central directory|unexpected EOF, expected \d+ more bytes"
        r"|invalid load key|SafetensorError|HeaderTooLarge|MetadataIncompleteBuffer|UnpicklingError",
        exclude=r"Weights only load failed|weights_only|WeightsUnpickler",
    ),
    _rule("checkpoint_corrupt", r"EOFError: Ran out of input", "medium"),
    _rule("port_in_use", r"Address already in use|EADDRINUSE|\[Errno 98\]|errno: 98", flags=_I),
    _rule("segfault", r"Segmentation fault|SIGSEGV|Signal 11\b"),
    _rule(
        "nccl_error",
        r"Watchdog caught collective operation timeout|NCCL error|ncclInternalError|ncclSystemError"
        r"|ncclUnhandledCudaError|ncclRemoteError|ncclInvalidUsage|NCCL communicator was aborted"
        r"|Some NCCL operations have failed or timed out|DistBackendError|RendezvousTimeoutError|DistStoreError"
        r"|store based barrier",
    ),
    _rule(
        "file_not_found",
        r"FileNotFoundError|No such file or directory|can't open file|couldn't chdir to|does not appear to have a file named",
        generic=True,
    ),
    _rule(
        "permission_denied",
        r"PermissionError|Permission denied|Read-only file system|\[Errno 13\]|\[Errno 30\]|Operation not permitted",
        generic=True,
    ),
)
_ORDERED_RULES: tuple[_Rule, ...] = tuple(sorted(_RULES, key=lambda r: r.generic))

# Loss values in free-text and JSON logs: "loss 0.53", "loss=nan", "'loss': nan", "train/loss: inf".
_LOSS_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,24}[_/])?loss(?![A-Za-z0-9_])[\"']?\s*[:=]?\s*[\"']?(?:tensor\()?"
    r"(?P<v>[-+]?(?:nan|inf(?:inity)?)|[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Hit:
    index: int  # into _Logs.lines
    line: _LogLine
    match: re.Match[str]
    confidence: Confidence


@dataclass(frozen=True)
class _Exc:
    type: str
    message: str
    index: int
    frame: str | None


@dataclass
class _Logs:
    lines: list[_LogLine]  # stderr first (evidence priority), then stdout
    n_stderr: int
    hits: dict[str, list[_Hit]] = field(default_factory=dict)
    claimed: set[int] = field(default_factory=set)  # lines matched by a specific rule
    explained: set[int] = field(default_factory=set)  # lines matched by any rule

    @classmethod
    def build(cls, stdout: str, stderr: str) -> _Logs:
        err = _split_log(stderr or "", "stderr")
        out = _split_log(stdout or "", "stdout")
        logs = cls(lines=[*err, *out], n_stderr=len(err))
        logs._scan()
        return logs

    def _source_lines(self) -> set[int]:
        """Source-code lines quoted under traceback frames (``raise FileNotFoundError(...)`` is
        code, not an error that happened) plus their ``^^^``/``~~~`` marker lines."""
        skip: set[int] = set()
        for idx, line in enumerate(self.lines):
            fm = _FRAME_RE.match(line.text)
            if fm is None:
                continue
            indent = len(line.text) - len(line.text.lstrip())
            for nxt in range(idx + 1, min(idx + 3, len(self.lines))):
                text = self.lines[nxt].text
                deeper = len(text) - len(text.lstrip()) > indent and text.strip()
                if not deeper or _FRAME_RE.match(text):
                    break
                if nxt == idx + 2 and not _MARKER_LINE_RE.match(text):
                    break
                skip.add(nxt)
        return skip

    def _scan(self) -> None:
        source = self._source_lines()
        for rule in _ORDERED_RULES:
            for idx, line in enumerate(self.lines):
                if idx in source or (rule.generic and idx in self.claimed):
                    continue
                m = rule.pattern.search(line.text)
                if m is None or (rule.exclude is not None and rule.exclude.search(line.text)):
                    continue
                self.hits.setdefault(rule.code, []).append(_Hit(idx, line, m, rule.confidence))
                self.explained.add(idx)
                if not rule.generic:
                    self.claimed.add(idx)

    @property
    def empty(self) -> bool:
        return not any(line.text.strip() for line in self.lines)

    def streams(self) -> Iterator[tuple[int, list[_LogLine]]]:
        """(offset into ``lines``, lines) for stderr, then stdout."""
        yield 0, self.lines[: self.n_stderr]
        yield self.n_stderr, self.lines[self.n_stderr :]


def _tracebacks(lines: Sequence[_LogLine], offset: int) -> list[_Exc]:
    """Final exception line + innermost frame of every ``Traceback (most recent call last):``."""
    out: list[_Exc] = []
    i, n = 0, len(lines)
    while i < n:
        if not _TB_START_RE.match(lines[i].text):
            i += 1
            continue
        j, frame = i + 1, None
        while j < n:
            text = lines[j].text
            fm = _FRAME_RE.match(text)
            if fm:
                frame = f"{fm['file']}:{fm['line']}" + (f" in {fm['func']}" if fm["func"] else "")
            elif text.strip() and text[0] not in " \t":
                break
            j += 1
        if j < n and frame is not None:
            em = _EXC_RE.match(lines[j].text.strip())
            if em:
                out.append(_Exc(em["type"], (em["msg"] or "").strip(), offset + j, frame))
        i = j
    return out


_TB_START_RE = re.compile(r"^\s*Traceback \(most recent call last\):")
_MARKER_LINE_RE = re.compile(r"^\s*[\^~]+[\s\^~]*$")
_FRAME_RE = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>\S+))?')
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::[ \t]?(?P<msg>.*))?$")
# Launcher / signal wrappers: they report *that* a child failed, never why.
_DERIVATIVE_EXC = frozenset(
    {
        "ChildFailedError",
        "SignalException",
        "SIGTERMException",
        "CalledProcessError",
        "KeyboardInterrupt",
        "SystemExit",
        "BrokenPipeError",
    }
)
_TRANSIENT_EXC = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "TimeoutError",
        "ReadTimeout",
        "ReadTimeoutError",
    }
)


# ---------------------------------------------------------------------------
# Context shared by the builders
# ---------------------------------------------------------------------------

_Outcome = Literal["pending", "active", "succeeded", "failed", "unknown"]


@dataclass(frozen=True)
class _Stop:
    """Why the scheduler (not the program) ended the job."""

    code: Literal["timeout", "preempted", "node_fail", "host_oom", "cancelled"]
    confidence: Confidence
    evidence: tuple[str, ...]
    node: str | None = None
    variant: str = ""


@dataclass
class _Draft:
    code: str
    confidence: Confidence
    evidence: list[str]
    explanation: str
    strategies: list[Strategy]
    hint: dict[str, str] = field(default_factory=dict)
    # "log" items are weakened for running/succeeded jobs (_adjust_for_outcome); the others are not
    source: Literal["scheduler", "log", "exit", "result", "wrapper"] = "log"
    authoritative: bool = False
    title: str | None = None
    # A side observation that cannot explain a failure (e.g. a transient fp16 NaN step): it
    # never becomes the primary cause of a failed job.
    incidental: bool = False


@dataclass
class _Ctx:
    facts: AttemptFacts
    logs: _Logs
    outcome: _Outcome
    signal: int | None
    sched_line: str
    stop: _Stop | None = None
    _root: tuple[_Exc | None] | None = None  # memo for root_exception()

    def root_exception(self) -> _Exc | None:
        """Last exception that is not a launcher wrapper or an NCCL cascade (stderr before stdout)."""
        if self._root is None:
            self._root = (_find_root_exception(self.logs),)
        return self._root[0]

    def hits(self, code: str) -> list[_Hit]:
        return self.logs.hits.get(code, [])

    def has(self, code: str) -> bool:
        return bool(self.logs.hits.get(code))

    def evidence(self, code: str, limit: int = 2) -> list[str]:
        """Evidence for the first and last hit of ``code`` (the start and the end of the story)."""
        hits = self.hits(code)
        chosen = hits if len(hits) <= limit else [*hits[: limit - 1], hits[-1]]
        return [_evidence_line(f"{h.line.stream}: ", h.line.text, h.match.group(0)) for h in chosen]

    def best_confidence(self, code: str) -> Confidence:
        return min((h.confidence for h in self.hits(code)), key=_CONF_ORDER.__getitem__, default="low")

    @property
    def rss_high(self) -> bool:
        f = self.facts
        return bool(f.max_rss_bytes and f.mem_request_bytes and f.max_rss_bytes >= 0.9 * f.mem_request_bytes)

    @property
    def node(self) -> str | None:
        """The job's node when it ran on exactly one host (safe to ``--exclude``), else None."""
        return _single_host(self.facts.node_list)

    @property
    def node_label(self) -> str | None:
        """The job's node list as reported (may be a multi-host expression)."""
        return _clean_node(self.facts.node_list)

    def node_hint(self, hint: dict[str, str]) -> str:
        """Add ``exclude_nodes`` for a single-node job, or ``suspect_nodes`` for a multi-node one
        (excluding every node of an allocation for one bad node can make the job unschedulable).
        Returns a sentence for the explanation."""
        if self.node:
            hint["exclude_nodes"] = self.node
            return f" Exclude {self.node} for the next attempt."
        if self.node_label:
            hint["suspect_nodes"] = self.node_label
            return (
                f" The job ran on several nodes ({self.node_label}) and the log does not say which one is at "
                "fault; if it recurs, find the common node and exclude only that one."
            )
        return ""


_BuilderFn = Callable[[_Ctx], _Draft | None]


def _find_root_exception(logs: _Logs) -> _Exc | None:
    nccl_lines = {h.index for h in logs.hits.get("nccl_error", [])}
    for offset, lines in logs.streams():
        for exc in reversed(_tracebacks(lines, offset)):
            if exc.type.rsplit(".", 1)[-1] in _DERIVATIVE_EXC or exc.index in nccl_lines:
                continue
            return exc
    return None


def _clean_node(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if not v or v.lower() in {"none", "none assigned", "(null)", "n/a", "unknown"}:
        return None
    return v


def _single_host(value: str | None) -> str | None:
    """``value`` if it names exactly one host (no hostlist brackets, commas or spaces)."""
    v = _clean_node(value)
    if v is None or any(c in v for c in "[],") or any(c.isspace() for c in v):
        return None
    return v


def _effective_signal(f: AttemptFacts) -> int | None:
    if f.exit_signal:
        return f.exit_signal
    if f.exit_code is not None and 128 < f.exit_code < 160:
        return f.exit_code - 128
    return None


def _outcome(f: AttemptFacts) -> _Outcome:
    st = f.scheduler_state
    if st is SchedulerState.PENDING:
        return "pending"
    if st in SCHEDULER_ACTIVE:
        return "active"
    if st is SchedulerState.COMPLETED:
        return "succeeded" if f.exit_code in (None, 0) else "failed"
    if st in SCHEDULER_TERMINAL:
        return "failed"
    if (
        (f.exit_code not in (None, 0))
        or f.exit_signal
        or f.app_state in (AppState.EXITED_ERROR, AppState.KILLED)
    ):
        return "failed"
    if f.app_state is AppState.EXITED_OK:
        return "succeeded"
    return "unknown"


def _sched_line(f: AttemptFacts) -> str:
    parts = [f"state={f.raw_state or f.scheduler_state.value}"]
    if f.exit_code is not None or f.exit_signal is not None:
        parts.append(f"exit={'?' if f.exit_code is None else f.exit_code}:{f.exit_signal or 0}")
    if f.elapsed_s is not None:
        parts.append(f"elapsed={_fmt_dur(f.elapsed_s)}")
    if f.time_limit_s:
        parts.append(f"limit={_fmt_dur(f.time_limit_s)}")
    if f.partition:
        parts.append(f"partition={f.partition}")
    if _clean_node(f.node_list):
        parts.append(f"nodes={f.node_list}")
    if f.max_rss_bytes:
        parts.append(f"max_rss={_fmt_gib(f.max_rss_bytes)}")
    if f.mem_request_bytes:
        parts.append(f"mem_request={_fmt_gib(f.mem_request_bytes)}")
    if f.restart_count:
        parts.append(f"restarts={f.restart_count}")
    return _clip("scheduler: " + " ".join(parts), MAX_EVIDENCE_CHARS)


def _stop_cause(ctx: _Ctx) -> _Stop | None:
    """The external stop cause, from the scheduler state first and ``slurmstepd`` markers second."""
    f, st = ctx.facts, ctx.facts.scheduler_state
    sched = (ctx.sched_line,)
    marker = ctx.hits("_cancel_marker")[-1] if ctx.has("_cancel_marker") else None
    why = (marker.match["why"] or "").upper() if marker else ""
    mev = tuple(ctx.evidence("_cancel_marker", 1)) if marker else ()
    mnode = marker.match["node"] if marker else None
    conf: Confidence = "high" if st in SCHEDULER_TERMINAL or st is SchedulerState.UNKNOWN else "medium"

    if st in (SchedulerState.TIMEOUT, SchedulerState.DEADLINE):
        variant = "deadline" if st is SchedulerState.DEADLINE else ""
        return _Stop("timeout", "high", sched + (mev if why == "TIME LIMIT" else ()), mnode, variant)
    if st is SchedulerState.PREEMPTED:
        return _Stop("preempted", "high", sched + (mev if why else ()), mnode)
    if st in (SchedulerState.NODE_FAIL, SchedulerState.BOOT_FAIL):
        variant = "boot_fail" if st is SchedulerState.BOOT_FAIL else ""
        return _Stop("node_fail", "high", sched + mev, mnode, variant)
    if st is SchedulerState.OUT_OF_MEMORY:
        return _Stop("host_oom", "high", sched)
    if st is SchedulerState.COMPLETED:
        return None
    if why == "TIME LIMIT":
        return _Stop("timeout", conf, sched + mev, mnode, "marker")
    if why in ("PREEMPTION", "JOB REQUEUE"):
        return _Stop("preempted", conf, sched + mev, mnode, "requeue" if why == "JOB REQUEUE" else "")
    if why.startswith("NODE FAIL") or ctx.has("_stepd_unkillable"):
        stepd = ctx.hits("_stepd_unkillable")
        # the unkillable node is named exactly; a NODE FAILURE marker names the (healthy) batch host
        node = stepd[-1].match["node"] if stepd else mnode
        variant = "unkillable" if stepd else ""
        return _Stop(
            "node_fail", conf, sched + mev + tuple(ctx.evidence("_stepd_unkillable", 1)), node, variant
        )
    failed_like = st in (SchedulerState.FAILED, SchedulerState.UNKNOWN)
    if (
        failed_like
        and _outcome(f) != "unknown"
        and (ctx.has("_oom_kill") or (ctx.signal == 9 and ctx.rss_high))
    ):
        return _Stop("host_oom", "high" if ctx.has("_oom_kill") or ctx.has("_killed") else "medium", sched)
    if st is SchedulerState.CANCELLED:
        return _Stop("cancelled", "high", sched + mev, mnode)
    # A plain STEP marker is what srun prints for the surviving tasks after one task failed
    # (--kill-on-bad-exit): a consequence, not a cancel. A plain JOB marker only overrides a
    # stale or missing state, never a FAILED one reported by the scheduler.
    if marker is not None and not why and marker.match["scope"] == "JOB" and st is not SchedulerState.FAILED:
        return _Stop("cancelled", "medium", sched + mev, mnode)
    return None


# ---------------------------------------------------------------------------
# Builders: one per code; each returns a draft or None
# ---------------------------------------------------------------------------


def _ckpt_strategies(ctx: _Ctx, base: Strategy) -> list[Strategy]:
    """Resume first when a checkpoint is known to exist, after ``base`` when unknown."""
    ckpt = ctx.facts.checkpoint_available
    if ckpt is True:
        return ["resume", base]
    if ckpt is None:
        return [base, "resume"]
    return [base]


@dataclass
class _TimeAdvice:
    """Walltime advice after a timeout.

    ``at_max``: the limit already is the partition maximum (no ``time`` hint possible).
    ``needs_chain``: the projected full run is longer than the partition maximum, so even the
    capped ``time`` hint cannot finish it in one attempt (checkpoint + resume is required).
    """

    hint: dict[str, str]
    sentences: list[str]
    at_max: bool = False
    needs_chain: bool = False
    target_s: int | None = None


def _timeout_hint(f: AttemptFacts) -> _TimeAdvice:
    base = max((v for v in (f.elapsed_s, f.time_limit_s) if v), default=None)
    if base is None:
        return _TimeAdvice({}, ["Request a longer time limit (about 1.5x the current one)."])
    target = math.ceil(base * 1.5)
    sentences: list[str] = []
    projected: int | None = None
    pf = f.progress_fraction
    if pf is not None and 0 < pf < 1 and f.elapsed_s:
        projected = math.ceil(f.elapsed_s / pf * 1.15)
        if projected > target:
            target = projected
            sentences.append(
                f"Training was about {pf:.0%} done, so a full run needs roughly {_fmt_dur(projected)} "
                "including a 15% margin."
            )
    target = _round_up(target, 300)
    cap = f.partition_max_time_s
    if cap is None or target <= cap:
        return _TimeAdvice({"time": format_slurm_duration(target)}, sentences, target_s=target)
    needs_chain = projected is not None and projected > cap
    if base >= cap:
        sentences.append(
            f"The limit already equals the partition maximum ({_fmt_dur(cap)}), so more walltime is not "
            "possible here: checkpoint and resume, or use a partition with a longer limit."
        )
        return _TimeAdvice({}, sentences, at_max=True, needs_chain=needs_chain, target_s=cap)
    sentences.append(f"The suggested limit is capped at the partition maximum ({_fmt_dur(cap)}).")
    if needs_chain:
        sentences.append(
            "Even that is shorter than the projected run, so it needs more than one attempt: write checkpoints "
            "(at least on the warning signal) and resume from them."
        )
    return _TimeAdvice({"time": format_slurm_duration(cap)}, sentences, needs_chain=needs_chain, target_s=cap)


def _signal_lead(target_s: int | None) -> int:
    """Warning lead for ``--signal``: 10% of the limit, clamped to 2..10 min (ResilienceSpec default 600)."""
    return min(600, max(120, _round_up((target_s or 3600) * 0.1, 60)))


def _b_timeout(ctx: _Ctx) -> _Draft | None:
    stop = ctx.stop
    if stop is None or stop.code != "timeout":
        return None
    f = ctx.facts
    advice = _timeout_hint(f)
    hint = advice.hint
    lead = _signal_lead(advice.target_s)
    if stop.variant == "deadline":
        text = "The job reached its --deadline and Slurm terminated it; the program itself did not fail."
    else:
        text = (
            f"The job ran {_fmt_dur(f.elapsed_s)} and hit its walltime limit ({_fmt_dur(f.time_limit_s)}); Slurm "
            "stopped it with SIGTERM, then SIGKILL after the grace period. The program itself did not fail."
        )
    parts = [text, *advice.sentences]
    ckpt = f.checkpoint_available
    if ckpt is True or advice.at_max or advice.needs_chain:
        hint["extra_sbatch"] = f"--signal=B:USR1@{lead}"
        parts.append(
            f"Add --signal=B:USR1@{lead} (or resilience.stop_behavior=checkpoint_and_exit) so the program gets "
            f"{lead}s warning to write a final checkpoint; the hpclab wrapper forwards USR1 to it."
        )
    if ckpt is True:
        parts.append("A checkpoint exists: resume from it instead of starting over.")
        if f.progress_fraction and 0 < f.progress_fraction < 1 and f.elapsed_s:
            remaining = f.elapsed_s * (1 - f.progress_fraction) / f.progress_fraction
            resume_t = _round_up(remaining * 1.25 + 600, 300)
            if f.partition_max_time_s:
                resume_t = min(resume_t, f.partition_max_time_s)
            hint["resume_time"] = format_slurm_duration(resume_t)
    elif ckpt is None:
        parts.append(
            "If the program wrote checkpoints, resume from the latest valid one rather than restarting."
        )
    else:
        parts.append(
            "No checkpoint was found, so the next attempt starts from zero; periodic checkpoints to "
            "$HPCLAB_CHECKPOINT_DIR would make future timeouts resumable."
        )
    warned = _warning_signal(ctx)
    if warned and ctx.has("_hpclab_signal"):
        parts.append(
            f"The program did receive the SIG{warned} warning but was still running at the limit: writing the "
            "checkpoint takes longer than the lead time, or the handler never stops the training loop. Increase "
            "resilience.lead_s (Slurm may deliver the signal up to a minute late)."
        )
    evidence = [*stop.evidence, *ctx.evidence("_cancel_marker", 1), *ctx.evidence("_hpclab_signal", 1)]
    strategies = _timeout_strategies(advice, ckpt)
    return _Draft("timeout", stop.confidence, evidence, " ".join(parts), strategies, hint, "scheduler", True)


def _timeout_strategies(advice: _TimeAdvice, ckpt: bool | None) -> list[Strategy]:
    """Recovery order after a walltime stop, by what more walltime can achieve and whether a
    checkpoint is known to exist (True), unknown (None) or known to be absent (False)."""
    table: dict[bool | None, list[Strategy]]
    if advice.at_max:  # more walltime is impossible here
        table = {True: ["resume", "fix_code"], None: ["fix_code", "resume"], False: ["fix_code"]}
    elif advice.needs_chain:  # more walltime helps, but only together with resume
        table = {
            True: ["resume", "adjust_resources"],
            None: ["resume", "fix_code", "adjust_resources"],
            False: ["fix_code", "adjust_resources"],
        }
    else:
        table = {
            True: ["resume", "adjust_resources"],
            None: ["adjust_resources", "resume"],
            False: ["adjust_resources", "fix_code"],
        }
    return list(table[ckpt])


def _b_preempted(ctx: _Ctx) -> _Draft | None:
    stop = ctx.stop
    if stop is None or stop.code != "preempted":
        return None
    f = ctx.facts
    parts = [
        "The scheduler preempted the job: its GPUs went to a higher-priority job. Nothing was wrong with the job."
    ]
    if stop.variant == "requeue":
        parts = ["Slurm requeued the job (preemption or an administrator); nothing was wrong with the job."]
    hint: dict[str, str] = {}
    if f.preemptible and f.partition:
        hint["avoid_partition"] = f.partition
        parts.append(
            f"Partition {f.partition} is preemptible by design: for runs longer than a few hours prefer a "
            "non-preemptible partition, and keep preemptible ones for short or frequently checkpointing jobs."
        )
    if f.restart_count:
        parts.append(f"Slurm already requeued it {f.restart_count} time(s).")
    if f.checkpoint_available is True:
        parts.append("Resume from the latest valid checkpoint; only progress since that checkpoint is lost.")
    elif f.checkpoint_available is False:
        parts.append(
            "Without checkpoints every preemption restarts from zero: write a checkpoint on SIGTERM."
        )
    strategies = [*_ckpt_strategies(ctx, "retry"), "adjust_resources"]
    return _Draft(
        "preempted",
        stop.confidence,
        list(stop.evidence),
        " ".join(parts),
        strategies,
        hint,
        "scheduler",
        True,
    )


def _node_fail_culprit(ctx: _Ctx, stop: _Stop) -> str | None:
    """The node to exclude after a node failure, only when it is known with confidence.

    The unkillable-stepd marker names the stuck node itself. A ``DUE TO NODE FAILURE`` marker
    is printed by a *surviving* slurmstepd, so its node is the failed one only for a single-node
    job; with an unknown node list the marker node is still the best (single-node) guess.
    """
    if stop.variant == "unkillable" and stop.node:
        return stop.node
    if ctx.node:
        return ctx.node
    return stop.node if ctx.node_label is None else None


def _b_node_fail(ctx: _Ctx) -> _Draft | None:
    stop = ctx.stop
    if stop is None or stop.code != "node_fail":
        return None
    node = _node_fail_culprit(ctx, stop)
    title = None
    if stop.variant == "boot_fail":
        title = "Node failed to boot"
        parts = ["The allocated node failed to boot, so the job never ran. This is an infrastructure fault."]
    elif stop.variant == "unkillable":
        title = "Job processes could not be killed"
        parts = [
            "Slurm could not kill the job's processes (usually stuck in uninterruptible I/O on a network "
            "filesystem); such nodes are normally drained. This is an infrastructure fault."
        ]
    else:
        parts = [
            "The node running the job failed (hardware, kernel or network filesystem). Your code did not cause this."
        ]
    if ctx.logs.empty and stop.variant != "boot_fail":
        parts.append(
            "No log output was captured at all: the node probably could not even create the log files, which "
            "is typical of a broken network-filesystem client on that node."
        )
    hint: dict[str, str] = {}
    if node:
        hint["exclude_nodes"] = node
        parts.append(
            f"Retry excluding {node}; if a node fails repeatedly, report it to cluster support with the job id."
        )
    elif ctx.node_label:
        hint["suspect_nodes"] = ctx.node_label
        parts.append(
            f"Slurm does not say which of the job's nodes ({ctx.node_label}) failed: look up the node reason "
            "(scontrol show node / sinfo -R) and exclude only that one; excluding the whole allocation can make "
            "the job unschedulable."
        )
    strategies: list[Strategy] = [*_ckpt_strategies(ctx, "retry"), "contact_support"]
    return _Draft(
        "node_fail",
        stop.confidence,
        list(stop.evidence),
        " ".join(parts),
        strategies,
        hint,
        "scheduler",
        True,
        title,
    )


def _b_cancelled(ctx: _Ctx) -> _Draft | None:
    stop = ctx.stop
    if stop is None or stop.code != "cancelled":
        return None
    f = ctx.facts
    who = re.search(r"by (\S+)", f.raw_state or "")
    by = f" by uid {who.group(1)}" if who else ""
    parts = [f"The job was cancelled{by}."]
    if f.elapsed_s == 0 or f.app_state is AppState.NOT_STARTED:
        parts.append("It was cancelled before it started running.")
    parts.append(
        "If neither you nor an hpclab cancel request did this, an administrator or another tool did; "
        "check with cluster support before resubmitting."
    )
    return _Draft(
        "cancelled",
        stop.confidence,
        list(stop.evidence),
        " ".join(parts),
        ["none", "retry"],
        {},
        "scheduler",
        True,
    )


def _host_oom_confidence(ctx: _Ctx) -> Confidence | None:
    if ctx.facts.scheduler_state is SchedulerState.OUT_OF_MEMORY or ctx.has("_oom_kill"):
        return "high"
    if ctx.signal == 9 and ctx.rss_high:
        return "high" if ctx.has("_killed") else "medium"
    if ctx.has("_dl_killed") or ctx.has("_shm"):
        return "high" if ctx.rss_high else "medium"
    return None


def _b_host_oom(ctx: _Ctx) -> _Draft | None:
    conf = _host_oom_confidence(ctx)
    if conf is None:
        return None
    f = ctx.facts
    evidence = [ctx.sched_line]
    for code in ("_oom_kill", "_killed", "_dl_killed", "_shm"):
        evidence.extend(ctx.evidence(code, 1))
    req = f" (--mem {_fmt_gib(f.mem_request_bytes)})" if f.mem_request_bytes else ""
    parts = [
        f"The job used more host (CPU) RAM than it requested{req} and the memory cgroup killed it. "
        "This is system RAM, not GPU memory."
    ]
    if f.max_rss_bytes and f.mem_request_bytes:
        parts.append(
            f"Peak RSS was {_fmt_gib(f.max_rss_bytes)} of {_fmt_gib(f.mem_request_bytes)} requested."
        )
    if ctx.has("_shm"):
        parts.append(
            "DataLoader workers pass batches through /dev/shm, which counts against the job's memory: "
            "lower num_workers or request more memory."
        )
    parts.append(
        "Request more memory, or use less: fewer DataLoader workers (each holds a copy of Python-side dataset "
        "objects), memory-map large arrays instead of loading them, and log loss.item() rather than tensors "
        "that keep the autograd graph alive."
    )
    hint: dict[str, str] = {}
    candidates = [v for v in ((f.mem_request_bytes or 0) * 1.5, (f.max_rss_bytes or 0) * 1.3) if v > 0]
    if candidates:
        hint["mem"] = f"{max(1, math.ceil(max(candidates) / _GIB))}G"
    strategies: list[Strategy] = ["adjust_resources", "resume", "fix_code"]
    if f.checkpoint_available is not True:
        strategies.remove("resume")
    if f.scheduler_state is SchedulerState.OUT_OF_MEMORY and f.app_state is AppState.EXITED_OK:
        # Slurm reports OUT_OF_MEMORY for *any* OOM kill in the job's cgroup, e.g. a DataLoader
        # worker that was restarted, even though the program went on and exited 0.
        conf = "medium"
        parts.insert(
            1,
            "The program itself exited 0 (status.json), so the kill hit a helper process such as a DataLoader "
            "worker; check the results before rerunning anything.",
        )
        if f.result_state == "valid":
            strategies = ["none", "adjust_resources"]
    auth = ctx.stop is not None and ctx.stop.code == "host_oom"
    source: Literal["scheduler", "log"] = (
        "scheduler" if auth or f.scheduler_state is SchedulerState.OUT_OF_MEMORY else "log"
    )
    return _Draft("host_oom", conf, evidence, " ".join(parts), strategies, hint, source, auth)


# Keys are services/jobscript.GUARD_CHECKS; older/alternative spellings map onto them.
_GUARD_ADVICE: dict[str, tuple[str, tuple[Strategy, ...]]] = {
    "torch_gpu_arch": (
        "the installed PyTorch has no kernels for this GPU's architecture (e.g. a cu124 build on B200/sm_100): "
        "install a build that covers it (cu128+) or request a GPU type the build supports.",
        ("fix_environment", "adjust_resources"),
    ),
    "torch_cuda_available": (
        "PyTorch could not use CUDA on the node (a wheel built for a newer CUDA than the driver, a CPU-only wheel, "
        "or no GPU visible): install a wheel matching the cluster driver.",
        ("fix_environment",),
    ),
    "nvidia_smi": (
        "nvidia-smi is missing or failing on the node although GPUs were requested, which usually means a node "
        "problem.",
        ("retry", "contact_support"),
    ),
    "venv": (
        "the runtime.venv virtualenv does not exist on the compute node or could not be activated: create it on a "
        "filesystem the compute nodes mount (e.g. scratch) and give its absolute path.",
        ("fix_environment",),
    ),
}
_GUARD_ALIASES = {
    "sm_arch": "torch_gpu_arch",
    "cuda_available": "torch_cuda_available",
    "cuda": "torch_cuda_available",
    "driver_compat": "torch_cuda_available",
}


def _b_guard(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("_guard")
    if not hits:
        return None
    check = hits[-1].match["check"]
    advice, strategies = _GUARD_ADVICE.get(
        _GUARD_ALIASES.get(check, check),
        (
            "see the wrapper's '[hpclab] guard check' line in stderr for the failing check.",
            ("fix_environment",),
        ),
    )
    text = (
        f"The hpclab wrapper's fail-fast check '{check}' failed before the program started (exit 3), so no GPU time "
        f"was spent on training: {advice}"
    )
    hint = {"guard_check": check}
    if check == "nvidia_smi":
        text += ctx.node_hint(hint)
    detail = [h for h in ctx.hits("_guard_msg") if h.match["check"] == check][-1:]
    evidence = [
        *ctx.evidence("_guard"),
        *(_evidence_line(f"{h.line.stream}: ", h.line.text) for h in detail),
    ]
    return _Draft("env_guard_failed", "high", evidence, text, list(strategies), hint, "wrapper")


def _b_wrapper_phase(ctx: _Ctx) -> _Draft | None:
    """The wrapper stopped before the program ran (exit 97 setup / 96 prepare).

    Medium confidence on purpose: this says *where* it failed; a specific rule on the lines
    above (command not found, missing file, ...) says *why* and should rank first.
    """
    if ctx.has("_guard"):
        return None  # the guard builder covers it
    hits = [h for h in ctx.hits("_wrapper_finished") if h.match["state"] == "exited" and h.match["phase"]]
    hits = [h for h in hits if h.match["phase"] in ("setup", "prepare")]
    if not hits:
        return None
    last = hits[-1]
    phase, err = last.match["phase"], last.match["err"] or ""
    hint = {"guard_check": phase}
    strategies: list[Strategy]
    if phase == "setup":
        title = "Runtime setup failed before the program started"
        text = (
            f"A runtime.setup line failed ({err or 'see stderr'}), so the program never started and no GPU time was "
            "spent on training. Setup lines run in a non-interactive batch shell that does not read ~/.bashrc: "
            "'conda activate' needs 'source <conda>/etc/profile.d/conda.sh' first and 'module' may need "
            "'source /etc/profile.d/modules.sh'. The stderr lines just above show the actual error."
        )
        strategies = ["fix_environment"]
    elif "entrypoint workdir" in err:
        title = "Entrypoint directory missing from the code snapshot"
        text = (
            f"hpclab could not start the program: {err}. entrypoint.workdir is relative to the project root and "
            "must be tracked by git (or listed in code.include_untracked) to be part of the snapshot."
        )
        strategies = ["fix_code"]
    else:
        title = "Attempt preparation failed before the program started"
        text = (
            f"hpclab could not prepare the attempt on the compute node ({err or 'see stderr'}): the scratch "
            "filesystem may be full, over quota or not mounted on that node, or the published snapshot was removed. "
            "No GPU time was spent on training."
        )
        strategies = ["retry", "fix_environment"]
    evidence = [*ctx.evidence("_setup_failed", 1), _evidence_line(f"{last.line.stream}: ", last.line.text)]
    return _Draft("env_guard_failed", "medium", evidence, text, strategies, hint, "wrapper", title=title)


def _b_simple(code: str, text: str, strategies: list[Strategy]) -> _BuilderFn:
    def build(ctx: _Ctx) -> _Draft | None:
        if not ctx.has(code):
            return None
        return _Draft(code, ctx.best_confidence(code), ctx.evidence(code), text, list(strategies))

    return build


def _b_node_scoped(code: str, text: str, strategies: list[Strategy]) -> _BuilderFn:
    """Infrastructure faults tied to a node: add ``exclude_nodes``."""

    def build(ctx: _Ctx) -> _Draft | None:
        if not ctx.has(code):
            return None
        hint: dict[str, str] = {}
        suffix = ctx.node_hint(hint)
        return _Draft(
            code, ctx.best_confidence(code), ctx.evidence(code), text + suffix, list(strategies), hint
        )

    return build


_DRIVER_RE = re.compile(r"found version (?P<v>\d{4,5})")
_TORCH_INDEX = "https://download.pytorch.org/whl"
# CUDA versions PyTorch publishes wheel indexes for (download.pytorch.org/whl/cuXYZ), as
# (major, minor). A driver runs any build of the same or an older CUDA version, so the advice
# is the newest index not above the driver; there is e.g. no cu122 index to point at.
_TORCH_CUDA_BUILDS: tuple[tuple[int, int], ...] = (
    (11, 3), (11, 6), (11, 7), (11, 8), (12, 1), (12, 4), (12, 6), (12, 8), (12, 9), (13, 0),
)  # fmt: skip


def _torch_cuda_tag(major: int, minor: int) -> str | None:
    usable = [b for b in _TORCH_CUDA_BUILDS if b <= (major, minor)]
    return f"cu{usable[-1][0]}{usable[-1][1]}" if usable else None


_SM_RE = re.compile(r"sm_(?P<arch>\d+) is not compatible")
_SUPPORTED_RE = re.compile(r"supports CUDA capabilities (?P<list>(?:sm_\d+\s*)+)")


def _b_driver(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("cuda_driver_mismatch")
    if not hits:
        return None
    text_all = "\n".join(line.text for line in ctx.logs.lines)
    hint: dict[str, str] = {}
    parts: list[str] = []
    strategies: list[Strategy] = ["fix_environment"]
    drv = _DRIVER_RE.search(text_all)
    sm = _SM_RE.search(text_all)
    if drv:
        v = int(drv["v"])
        major, minor = v // 1000, (v % 1000) // 10
        hint["driver_cuda"] = f"{major}.{minor}"
        text = (
            f"The node's driver supports CUDA {major}.{minor}, but the installed PyTorch wheel was built for a newer "
            "CUDA (plain PyPI wheels track the newest CUDA), so no GPU is usable."
        )
        tag = _torch_cuda_tag(major, minor)
        if tag:
            hint["torch_index_url"] = f"{_TORCH_INDEX}/{tag}"
            text += (
                f" Reinstall with pip install torch --index-url {_TORCH_INDEX}/{tag} (the newest PyTorch CUDA "
                "build that is not newer than the driver)."
            )
        else:
            text += " The driver is older than any current PyTorch CUDA build: use nodes with a newer driver."
            strategies.append("adjust_resources")
        parts.append(text)
    if sm:
        arch = f"sm_{sm['arch']}"
        hint["gpu_arch"] = arch
        supported = _SUPPORTED_RE.search(text_all)
        have = f" (this build supports {supported['list'].strip()})" if supported else ""
        parts.append(
            f"The GPU ({arch}) is newer than the architectures compiled into this PyTorch build{have}: install a build "
            "that includes it (e.g. cu128+ for sm_100/B200) or request a GPU type the build supports."
        )
        strategies.append("adjust_resources")
    if not parts:
        parts.append(
            "The PyTorch CUDA build and the node's driver or GPU architecture are incompatible "
            "(no kernel image / unsupported PTX): install a PyTorch build that matches the cluster's driver and GPUs."
        )
        strategies.append("adjust_resources")
    parts.append("Check torch.version.cuda in env.json against the driver version reported by nvidia-smi.")
    return _Draft(
        "cuda_driver_mismatch",
        "high",
        ctx.evidence("cuda_driver_mismatch"),
        " ".join(parts),
        strategies,
        hint,
    )


def _b_no_gpu(ctx: _Ctx) -> _Draft | None:
    if not ctx.has("no_gpu"):
        return None
    text_all = " ".join(h.line.text for h in ctx.hits("no_gpu"))
    hint: dict[str, str] = {}
    strategies: list[Strategy]
    if ctx.facts.gpus == 0:
        text = "The job requested no GPUs, so CUDA sees no device. Request at least one GPU."
        hint["gpus"] = "1"
        strategies = ["adjust_resources"]
    elif "not compiled with CUDA" in text_all:
        text = (
            "The environment has a CPU-only PyTorch build. Install a CUDA wheel that matches the cluster driver "
            "(e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128)."
        )
        strategies = ["fix_environment"]
    else:
        text = (
            "GPUs were requested but CUDA could not see any: the setup script may override CUDA_VISIBLE_DEVICES, "
            "the PyTorch wheel may not match the driver, or the node's GPU driver is unhealthy."
        )
        strategies = ["fix_environment", "contact_support"]
    return _Draft("no_gpu", ctx.best_confidence("no_gpu"), ctx.evidence("no_gpu"), text, strategies, hint)


_COMMAND_RE = re.compile(
    r"(?P<cmd>[\w./+-]{1,200}): command not found|execve\(\): (?P<cmd2>[^:]{1,300}): No such file"
    r"|env: (?P<cmd3>\S{1,300}): No such file"
)


def _b_command_not_found(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("command_not_found")
    if not hits:
        return None
    cmd_match = _COMMAND_RE.search(hits[0].line.text)
    raw_cmd = (cmd_match["cmd"] or cmd_match["cmd2"] or cmd_match["cmd3"] or "") if cmd_match else ""
    cmd = raw_cmd.strip().strip("\u2018\u2019'\"").rsplit("/", 1)[-1]
    if cmd in ("module", "ml"):
        text = (
            "The 'module' command is not initialised in the job's non-interactive shell: add "
            "'source /etc/profile.d/modules.sh' (or your site's equivalent) before 'module load' in runtime.setup."
        )
    else:
        text = (
            f"The shell could not find '{cmd}' on the compute node. Batch jobs run a non-interactive shell that "
            "does not read ~/.bashrc, so activate the environment explicitly (runtime.venv, or runtime.setup lines "
            "such as 'source /abs/path/.venv/bin/activate' / 'module load ...') or call the interpreter by absolute "
            "path."
        )
        missing = _quoted_path(ctx.hits("file_not_found"))
        if missing and "activate" in missing:
            text = (
                f"The environment activation script '{missing}' does not exist on the compute node, so '{cmd}' was "
                "not found afterwards. " + text
            )
    hint = {"command": cmd} if cmd else {}
    return _Draft(
        "command_not_found", "high", ctx.evidence("command_not_found"), text, ["fix_environment"], hint
    )


_MODULE_RE = re.compile(r"No module named ['\"]?(?P<m>[\w.]+)")
_IMPORT_NAME_RE = re.compile(r"cannot import name ['\"]?(?P<n>\w+)['\"]? from ['\"]?(?P<m>[\w.]+)")
_SO_RE = re.compile(r"(?P<lib>lib[\w.+-]+\.so[\w.]*): cannot open shared object file")
_SYMBOL_RE = re.compile(r"undefined symbol: (?P<sym>\S+)")
_GLIBC_RE = re.compile(r"(?P<v>GLIBC(?:XX)?_[\d.]+)' not found")


def _b_import(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("import_error")
    if not hits:
        return None
    joined = "\n".join(h.line.text for h in hits)
    hint: dict[str, str] = {}
    if m := _MODULE_RE.search(joined):
        module = m["m"]
        hint["missing_module"] = module
        text = (
            f"Python module '{module}' is not installed in the environment the job used. Check which interpreter "
            "ran (env.json python.executable) and that runtime.venv/setup activates the right environment. "
            f"If '{module.split('.')[0]}' is part of your project, it may be untracked by git and therefore missing "
            "from the code snapshot: commit it or list it in code.include_untracked."
        )
    elif m := _IMPORT_NAME_RE.search(joined):
        hint["missing_module"] = m["m"]
        text = (
            f"'{m['n']}' cannot be imported from '{m['m']}': the package version on the cluster differs from the one "
            "your code expects. Pin versions with a lockfile (runtime.lockfiles) and rebuild the environment."
        )
    elif m := _SO_RE.search(joined):
        hint["missing_library"] = m["lib"]
        text = (
            f"The native library {m['lib']} is missing on the compute node: load the matching module (e.g. "
            "'module load cuda/12.x') in runtime.setup or install the CUDA-enabled wheel that bundles it."
        )
    elif m := _SYMBOL_RE.search(joined):
        text = (
            "A compiled extension (e.g. flash-attn, apex, a custom CUDA op) was built against a different "
            "torch/CUDA version than the one installed: reinstall or rebuild it inside this environment."
        )
    elif m := _GLIBC_RE.search(joined):
        text = (
            f"A binary needs {m['v']}, which this node's OS does not provide: install a manylinux2014-compatible "
            "wheel or build the package on the cluster."
        )
    else:
        text = "An import failed: a package is missing or incompatible in the job's Python environment."
    return _Draft(
        "import_error", "high", ctx.evidence("import_error"), text, ["fix_environment", "fix_code"], hint
    )


_PATH_IN_QUOTES_RE = re.compile("(?:: |to |file )['\"`\u2018](?P<p>[^'\"`\u2019]{1,400})['\"`\u2019]")
# bash style: "line 11: .venv/bin/activate: No such file or directory"
_BASH_PATH_RE = re.compile(
    r"(?:^|: )(?P<p>[^\s:'\"`]{1,400}): (?:No such file or directory|Permission denied)"
)


def _quoted_path(hits: Sequence[_Hit]) -> str | None:
    """The path an error line talks about: quoted (Python) first, then bash-style."""
    for h in hits:
        if m := _PATH_IN_QUOTES_RE.search(h.line.text):
            return m["p"]
    for h in hits:
        if m := _BASH_PATH_RE.search(h.line.text):
            return m["p"]
    return None


def _b_disk_quota(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("disk_quota")
    if not hits:
        return None
    path = _quoted_path(hits)
    where = f" (writing {path})" if path else ""
    parts = [f"A write failed because a filesystem quota was exceeded or the disk is full{where}."]
    if path and path.startswith("/tmp"):  # noqa: S108 - classifying a reported path
        parts.append("Node-local /tmp is small: point TMPDIR at scratch or the attempt directory.")
    parts.append(
        "Home directories on clusters usually have small quotas: keep outputs, checkpoints and caches (HF_HOME, "
        "TORCH_HOME, pip/uv cache, WANDB_DIR, TMPDIR) on scratch, and delete old checkpoints. Checkpoints written "
        "while the disk was full are probably truncated: do not resume from one without validating it."
    )
    hint = {"path": redact_secrets(path)} if path else {}
    return _Draft(
        "disk_quota",
        ctx.best_confidence("disk_quota"),
        ctx.evidence("disk_quota"),
        " ".join(parts),
        ["fix_environment"],
        hint,
    )


def _generic_confidence(ctx: _Ctx, code: str) -> Confidence:
    """Confidence of a generic (file/permission) signature.

    Such lines are common harmless noise ('ls: cannot access ...', cleanup 'rm' errors). When
    the log also ends in a traceback that no rule explains, that traceback is the fatal error:
    the generic finding is capped at medium so the exception ranks first.
    """
    conf = ctx.best_confidence(code)
    exc = ctx.root_exception()
    if conf == "high" and exc is not None and exc.index not in ctx.logs.explained:
        return "medium"
    return conf


def _b_file_not_found(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("file_not_found")
    if not hits:
        return None
    path = _quoted_path(hits)
    chdir = any("couldn't chdir" in h.line.text for h in hits)
    if chdir:
        text = (
            "Slurm could not enter the job's working directory on the node: the filesystem is not mounted there "
            "or the directory was removed before the job started."
        )
        strategies: list[Strategy] = ["fix_environment", "retry"]
    elif path and ("activate" in path or "venv" in path):
        text = (
            f"The environment setup points at '{path}', which does not exist on the compute node. Virtualenvs are "
            "never part of the code snapshot: use runtime.venv with an absolute path to a venv created on the cluster."
        )
        strategies = ["fix_environment"]
    elif path and not path.startswith("/"):
        text = (
            f"'{path}' was not found. Relative paths resolve against the attempt's work dir, a copy of the code "
            "snapshot; data and output folders (data/, outputs/, checkpoints/, wandb/) are excluded from snapshots. "
            "Use an absolute cluster path (e.g. on scratch) or declare the data as an input."
        )
        strategies = ["fix_code", "fix_environment"]
    else:
        where = f"'{path}' " if path else "A required file "
        text = (
            f"{where}does not exist on the compute node. Check the path on the cluster itself: files on your laptop "
            "or on a login node's /tmp are not visible to compute nodes."
        )
        strategies = ["fix_code", "fix_environment"]
    hint = {"path": redact_secrets(path)} if path else {}
    return _Draft(
        "file_not_found",
        _generic_confidence(ctx, "file_not_found"),
        ctx.evidence("file_not_found"),
        text,
        strategies,
        hint,
    )


def _b_permission(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("permission_denied")
    if not hits:
        return None
    path = _quoted_path(hits)
    joined = " ".join(h.line.text for h in hits)
    if "Read-only file system" in joined or "Errno 30" in joined:
        text = (
            "The program tried to write to a read-only location. hpclab publishes code snapshots read-only on "
            "purpose: write outputs to $HPCLAB_OUTPUT_DIR ({output_dir}) and checkpoints to $HPCLAB_CHECKPOINT_DIR."
        )
    elif "execve" in joined:
        text = "The entrypoint is not executable: invoke it through its interpreter (python train.py, bash run.sh)."
    else:
        where = f" on '{path}'" if path else ""
        text = (
            f"Access was denied{where}. Check ownership and permissions on the cluster, and write only below "
            "your scratch directory or $HPCLAB_OUTPUT_DIR."
        )
    hint = {"path": redact_secrets(path)} if path else {}
    return _Draft(
        "permission_denied",
        _generic_confidence(ctx, "permission_denied"),
        ctx.evidence("permission_denied"),
        text,
        ["fix_code", "fix_environment"],
        hint,
    )


_OOM_ALLOC_RE = re.compile(r"Tried to allocate (?P<n>\d+(?:\.\d+)?\s*[KMGT]i?B)")
_OOM_CAP_RE = re.compile(r"total capacity of (?P<n>\d+(?:\.\d+)?\s*[KMGT]i?B)")
_OOM_RESERVED_RE = re.compile(r"(?P<n>\d+(?:\.\d+)?\s*[KMGT]i?B) is reserved by PyTorch but unallocated")
_OOM_PROCESS_RE = re.compile(r"Process (?P<pid>\d+) has (?P<n>\d+(?:\.\d+)?\s*[KMGT]i?B) memory in use")


def _b_cuda_oom(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("cuda_oom")
    if not hits:
        return None
    joined = "\n".join(h.line.text for h in hits)
    hint: dict[str, str] = {}
    alloc = _OOM_ALLOC_RE.search(joined)
    cap = _OOM_CAP_RE.search(joined)
    on_gpu = f" of a {cap['n']} GPU" if cap else ""
    tried = f" while allocating {alloc['n']}" if alloc else ""
    parts = [f"The program ran out of GPU memory{on_gpu}{tried}."]
    if alloc:
        hint["tried_to_allocate"] = alloc["n"]
    if cap:
        hint["gpu_capacity"] = cap["n"]
    if len({m["pid"] for m in _OOM_PROCESS_RE.finditer(joined)}) >= 2:
        parts.append(
            "Several processes share this GPU: usually every rank is using cuda:0. Call "
            "torch.cuda.set_device(local_rank) (or use LOCAL_RANK) before allocating."
        )
    reserved = _OOM_RESERVED_RE.search(joined)
    cap_b = _parse_size(cap["n"]) if cap else None
    res_b = _parse_size(reserved["n"]) if reserved else None
    if res_b and (res_b >= _GIB or (cap_b and res_b >= 0.1 * cap_b)):
        hint["env"] = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        parts.append(
            f"{reserved['n'] if reserved else ''} was reserved but unallocated (fragmentation): setting "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True may be enough on its own."
        )
    parts.append(
        "Changes that keep the experiment identical: a GPU type with more memory, activation checkpointing, or "
        "memory-efficient kernels."
    )
    if ctx.facts.gpus and ctx.facts.gpus > 1:
        parts.append(
            "With several GPUs, sharding parameters and optimizer state (FSDP/ZeRO) also keeps the math unchanged."
        )
    parts.append(
        "Shrinking the per-GPU batch size or sequence length changes the training semantics, so it is a new derived "
        "Run (derive_run), not a resource adjustment — unless gradient accumulation grows by the same factor so the "
        "effective batch size stays the same; say so in the derived Run's notes."
    )
    return _Draft(
        "cuda_oom", "high", ctx.evidence("cuda_oom"), " ".join(parts), ["derive_run", "fix_code"], hint
    )


def _loss_trend(ctx: _Ctx) -> tuple[int, bool, list[_Hit]]:
    """(# non-finite loss values, whether a stream's *last* loss is non-finite, evidence hits)."""
    nonfinite, ends_bad, evidence = 0, False, []
    for offset, lines in ctx.logs.streams():
        last_bad = False
        for pos, line in enumerate(lines):
            for m in _LOSS_VALUE_RE.finditer(line.text):
                last_bad = not math.isfinite(float(m["v"].lower().replace("infinity", "inf")))
                if last_bad:
                    nonfinite += 1
                    evidence.append(_Hit(offset + pos, line, m, "high"))
        ends_bad = ends_bad or last_bad
    return nonfinite, ends_bad, evidence


def _b_nan(ctx: _Ctx) -> _Draft | None:
    explicit = ctx.hits("nan_loss")
    nonfinite, ends_bad, value_hits = _loss_trend(ctx)
    if not explicit and not nonfinite:
        return None
    chosen = [
        *explicit[:1],
        *value_hits[:1],
        *(value_hits[-1:] if len(value_hits) > 1 else []),
        *explicit[-1:],
    ]
    evidence = list(
        dict.fromkeys(_evidence_line(f"{h.line.stream}: ", h.line.text, h.match.group(0)) for h in chosen)
    )
    if explicit or ends_bad:
        conf: Confidence = "high" if explicit or nonfinite >= 2 else "medium"
        tail = " and stayed non-finite until the end of the log" if ends_bad else ""
        text = (
            f"The training loss became NaN/Inf{tail}. Retrying the same configuration usually diverges again. "
            "Typical fixes: lower the learning rate or lengthen warmup, clip gradients (e.g. max_norm=1.0), prefer "
            "bf16 over fp16 (or keep the loss scaler), and check inputs for NaN/Inf, division by zero or log(0). "
            "Changing hyperparameters is a new derived Run; so is resuming from a pre-divergence checkpoint with a "
            "different learning rate."
        )
        return _Draft("nan_loss", conf, evidence, text, ["fix_code", "derive_run"])
    text = (
        f"{nonfinite} non-finite loss value(s) appeared but later losses are finite. fp16 dynamic loss scaling skips "
        "such steps and this is usually harmless if rare; if it keeps happening, lower the learning rate or use bf16."
    )
    return _Draft("nan_loss", "low", evidence, text, ["none", "fix_code"], incidental=True)


def _b_device_assert(ctx: _Ctx) -> _Draft | None:
    if not ctx.has("cuda_device_assert"):
        return None
    joined = " ".join(h.line.text for h in ctx.hits("cuda_device_assert"))
    parts = [
        "A CUDA kernel assertion failed or memory was accessed out of bounds. In training code this is almost "
        "always an index out of range: class labels outside [0, num_classes) for cross-entropy, token ids >= the "
        "embedding/vocabulary size, or a wrong padding id. CUDA reports errors asynchronously, so the Python stack "
        "may point at a later line: reproduce with CUDA_LAUNCH_BLOCKING=1 (or on CPU) in a short smoke run."
    ]
    if "illegal memory access" in joined and "assert" not in joined.lower():
        parts.append("If it happens at random steps on one node only, it can also be a hardware fault.")
    return _Draft(
        "cuda_device_assert", "high", ctx.evidence("cuda_device_assert"), " ".join(parts), ["fix_code"]
    )


_WATCHDOG_RE = re.compile(r"OpType=(?P<op>\w+).*?Timeout\(ms\)=(?P<ms>\d+)")


def _b_nccl(ctx: _Ctx) -> _Draft | None:
    hits = ctx.hits("nccl_error")
    if not hits:
        return None
    joined = "\n".join(h.line.text for h in hits)
    hint: dict[str, str] = {}
    strategies: list[Strategy]
    if wd := _WATCHDOG_RE.search(joined):
        minutes = int(wd["ms"]) / 60000
        text = (
            f"A collective ({wd['op']}) waited {minutes:g} min for the other ranks and timed out. This almost always "
            "means one rank crashed, hung or ran much slower (uneven data per rank, rank-0-only evaluation or "
            "checkpointing, a stuck DataLoader), not a network fault. Look for the first error of any rank above it "
            "and make sure every rank issues the same collectives."
        )
        strategies = ["fix_code", "retry", "contact_support"]
    elif re.search(r"RendezvousTimeoutError|DistStoreError|store based barrier", joined):
        text = (
            "Not all ranks joined the process group: a rank failed to start, MASTER_ADDR/MASTER_PORT or WORLD_SIZE "
            "is wrong, or the nodes cannot reach each other."
        )
        strategies = ["fix_code", "retry"]
    else:
        text = (
            "NCCL failed while ranks were communicating. If another error appears first on some rank, fix that one; "
            "otherwise rerun with NCCL_DEBUG=INFO to see the failing link, and if it recurs on the same node(s) "
            "exclude them and report it to support."
        )
        strategies = ["retry", "contact_support", "fix_environment"]
        text += ctx.node_hint(hint)
    return _Draft(
        "nccl_error", ctx.best_confidence("nccl_error"), ctx.evidence("nccl_error"), text, strategies, hint
    )


def _b_segfault(ctx: _Ctx) -> _Draft | None:
    from_exit = ctx.signal == 11
    if not from_exit and not ctx.has("segfault"):
        return None
    evidence = ([ctx.sched_line] if from_exit else []) + ctx.evidence("segfault")
    text = (
        "The process crashed in native code (SIGSEGV). Common causes on GPU clusters: a compiled extension "
        "(flash-attn, apex, custom CUDA ops) built against a different torch/CUDA, DataLoader workers forked after "
        "CUDA initialisation, or mismatched numpy/ABI versions. Reinstall compiled extensions inside this "
        "environment and set PYTHONFAULTHANDLER=1 to get a Python stack on the next crash."
    )
    source: Literal["exit", "log"] = "exit" if from_exit else "log"
    return _Draft("segfault", "high", evidence, text, ["fix_environment", "fix_code", "retry"], {}, source)


def _b_python_exception(ctx: _Ctx) -> _Draft | None:
    exc = ctx.root_exception()
    if exc is None or exc.index in ctx.logs.explained:
        return None  # nothing, or a specific rule already explains this exception line
    short_type = exc.type.rsplit(".", 1)[-1]
    line = ctx.logs.lines[exc.index]
    evidence = [_evidence_line(f"{line.stream}: ", line.text)]
    where = ""
    if exc.frame:
        evidence.append(_evidence_line(f"{line.stream}: at ", exc.frame))
        where = f" at {redact_secrets(exc.frame)}"
    message = _clip(redact_secrets(exc.message), 200)
    text = f"The program raised {exc.type}{': ' + message if message else ''}{where}."
    strategies: list[Strategy]
    if short_type in _TRANSIENT_EXC:
        text += " This looks like a transient network error: a retry may succeed, but make the code retry it itself."
        strategies = ["retry", "fix_code"]
    else:
        text += " This is a code, data or configuration error; an unchanged retry will fail the same way."
        strategies = ["fix_code"]
    hint = {"exception": short_type}
    if exc.frame:
        hint["location"] = redact_secrets(exc.frame)[:200]
    title = f"Unhandled Python exception ({short_type})"
    return _Draft("python_exception", "high", evidence, text, strategies, hint, title=title)


_SIGNAL_TEXT: dict[int, tuple[str, str, list[Strategy]]] = {
    9: (
        "SIGKILL",
        "It was killed with SIGKILL: typically the memory cgroup (host OOM without a recorded event) or Slurm's "
        "KillWait after an earlier SIGTERM. Compare sacct MaxRSS with --mem.",
        ["adjust_resources", "retry"],
    ),
    15: (
        "SIGTERM",
        "It was terminated with SIGTERM, which scancel, time limits, preemption and node shutdowns send.",
        ["retry"],
    ),
    6: (
        "SIGABRT",
        "It aborted (SIGABRT): a C++ exception or assertion, e.g. 'terminate called after throwing ...' from the "
        "NCCL watchdog or a CUDA error. Look for the first error line above the abort.",
        ["fix_code", "fix_environment"],
    ),
    7: (
        "SIGBUS",
        "It died with a bus error (SIGBUS): usually /dev/shm exhaustion by DataLoader workers or a memory-mapped "
        "file that changed or vanished on a network filesystem.",
        ["fix_code", "adjust_resources"],
    ),
    4: (
        "SIGILL",
        "It hit an illegal instruction (SIGILL): a binary built for a newer CPU (e.g. AVX-512) than this node.",
        ["fix_environment", "adjust_resources"],
    ),
    2: ("SIGINT", "It was interrupted (SIGINT).", ["none", "retry"]),
}


def _b_killed_signal(ctx: _Ctx, drafts: Sequence[_Draft]) -> _Draft | None:
    sig = ctx.signal
    if sig is None or sig == 11 or ctx.stop is not None or _host_oom_confidence(ctx):
        return None
    if any(d.confidence != "low" for d in drafts):
        return None  # a specific cause already explains why the process was signalled
    name, text, strategies = _SIGNAL_TEXT.get(
        sig, (f"signal {sig}", f"It was terminated by signal {sig}.", ["retry"])
    )
    evidence = [ctx.sched_line, *ctx.evidence("_killed", 1)]
    return _Draft(
        "killed_signal",
        "medium",
        evidence,
        text,
        list(strategies),
        {"signal": name},
        "exit",
        title=f"Killed by {name}",
    )


def _result_problems(f: AttemptFacts) -> list[str]:
    """Why a job the scheduler calls successful did not produce usable results (may be empty)."""
    reasons: list[str] = []
    if f.missing_outputs:
        shown = ", ".join(f.missing_outputs[:5]) + (" ..." if len(f.missing_outputs) > 5 else "")
        reasons.append(f"required outputs are missing: {shown}")
    if f.app_state not in (AppState.EXITED_OK, AppState.UNKNOWN):
        reasons.append(f"the wrapper recorded app_state={f.app_state.value}")
    if f.result_state == "invalid":
        reasons.append("the declared outputs failed validation")
    elif f.result_state == "incomplete" and not f.missing_outputs:
        reasons.append(
            "some required outputs are missing or only partially written (result_state=incomplete)"
        )
    return reasons


_WARNING_SIGNALS = {10: "USR1", 12: "USR2"}  # Linux numbering (compute nodes)


def _warning_signal(ctx: _Ctx) -> str | None:
    """The walltime-warning signal (USR1/USR2) the program stopped on, from the exit status or
    the wrapper's ``[hpclab] received SIG...`` line (the only portable source on non-Linux)."""
    if ctx.signal in _WARNING_SIGNALS:
        return _WARNING_SIGNALS[ctx.signal]
    logged = [h.match["sig"] for h in ctx.hits("_hpclab_signal") if h.match["sig"] in ("USR1", "USR2")]
    return logged[-1] if logged else None


def _b_warning_signal(ctx: _Ctx) -> _Draft | None:
    """checkpoint_and_exit: the program stopped on the pre-timeout signal, as designed.

    With ``--signal=B:USR1@<lead>`` (or ``run_cancel(mode="request_checkpoint")``) the wrapper
    forwards the signal, the program checkpoints and exits 128+signal *before* the limit, so
    Slurm reports FAILED (or COMPLETED if it exits 0), never TIMEOUT. The recovery is a resume,
    not a retry from scratch.
    """
    if ctx.stop is not None:
        return None
    sig = _warning_signal(ctx)
    if sig is None:
        return None
    f = ctx.facts
    if ctx.outcome == "succeeded":
        if not _result_problems(f):
            return None  # it stopped on the signal but everything it had to produce is there
    elif ctx.outcome != "failed":
        return None
    # Exit status 128+USR1, or a clean exit after the signal, is the designed path. Any other
    # exit status means something else may have gone wrong while stopping (e.g. a crash while
    # writing the checkpoint): medium, so a specific error found in the log ranks first.
    clean_stop = (
        ctx.signal in _WARNING_SIGNALS
        or f.exit_code == 0
        or (f.exit_code is None and f.app_state is AppState.KILLED)
    )
    parts = [
        f"The program received SIG{sig} (the walltime warning requested with --signal, i.e. "
        "resilience.stop_behavior=checkpoint_and_exit, or an hpclab request_checkpoint) and stopped before the time "
        "limit: the planned checkpoint-and-exit path, not a crash."
    ]
    advice = _timeout_hint(f)
    ckpt = f.checkpoint_available
    strategies: list[Strategy]
    if ckpt is False:
        parts.append(
            "But no checkpoint was found: the signal handler did not write one to $HPCLAB_CHECKPOINT_DIR, or the "
            "lead time was too short (keep resilience.lead_s >= checkpoint time + 60 s). Fix that before resubmitting."
        )
        strategies = ["fix_code", "adjust_resources"]
    else:
        parts.append(
            "Resume from the checkpoint it wrote; request more time if you want fewer resume cycles."
            if ckpt
            else "Resume from the checkpoint it wrote on the signal after validating it (a program without a "
            f"SIG{sig} handler is simply killed by it and writes none); request more time for fewer resume cycles."
        )
        strategies = ["resume", "adjust_resources"]
    parts.extend(advice.sentences)
    evidence = [ctx.sched_line, *ctx.evidence("_hpclab_signal", 1), *ctx.evidence("_wrapper_finished", 1)]
    return _Draft(
        "timeout",
        "high" if clean_stop else "medium",
        evidence,
        " ".join(parts),
        strategies,
        dict(advice.hint),
        "wrapper",
        title=f"Stopped on the walltime warning signal (SIG{sig})",
    )


def _b_success_no_outputs(ctx: _Ctx) -> _Draft | None:
    f = ctx.facts
    if ctx.outcome != "succeeded" and not (
        f.scheduler_state is SchedulerState.COMPLETED and f.app_state is not AppState.EXITED_OK
    ):
        return None
    reasons = _result_problems(f)
    if not reasons:
        return None
    text = (
        "The scheduler reports success, but "
        + "; ".join(reasons)
        + ". COMPLETED only means the batch script "
        "exited 0. Common causes: outputs written somewhere other than the declared path (write to "
        "$HPCLAB_OUTPUT_DIR / {output_dir}), an exception swallowed by try/except, or the training command started "
        "in the background without 'wait'."
    )
    evidence = [ctx.sched_line]
    if f.missing_outputs:
        evidence.append(_clip("results: missing " + ", ".join(f.missing_outputs), MAX_EVIDENCE_CHARS))
    hint = {"missing_outputs": ",".join(f.missing_outputs)[:300]} if f.missing_outputs else {}
    return _Draft("success_no_outputs", "high", evidence, text, ["fix_code"], hint, "result")


def _b_unknown(ctx: _Ctx) -> _Draft:
    tail: list[str] = []
    for _, lines in ctx.logs.streams():
        tail = [f"{line.stream}: {line.text}" for line in lines if line.text.strip()][-3:]
        if tail:
            break
    evidence = [ctx.sched_line, *(_evidence_line("", t) for t in tail)]
    if ctx.logs.empty:
        text = (
            "No known failure signature matched and both log tails are empty: the program may have died before "
            "producing output (runtime setup, wrong interpreter path) or the logs could not be read."
        )
    else:
        text = (
            "No known failure signature matched. Read the end of the full logs (run_logs, stream=stderr); retry only "
            "if the failure looks transient."
        )
    return _Draft("unknown_failure", "low", evidence, text, ["fix_code", "retry"], {}, "exit")


_BUILDERS: tuple[_BuilderFn, ...] = (
    _b_timeout,
    _b_preempted,
    _b_node_fail,
    _b_cancelled,
    _b_warning_signal,
    _b_host_oom,
    _b_guard,
    _b_wrapper_phase,
    _b_node_scoped(
        "gpu_hardware_error",
        "The GPU reported a hardware fault (ECC error, fell off the bus, Xid). Your code did not cause this; "
        "retry on another node and report the node to cluster support with the job id.",
        ["retry", "contact_support"],
    ),
    _b_node_scoped(
        "nfs_stale",
        "A file on the network filesystem changed or vanished under the running job (stale handle / I/O error), "
        "or the node's filesystem client is unhealthy. Do not modify or re-sync inputs of running jobs; if it "
        "happens on one node, report that node to support.",
        ["retry", "contact_support"],
    ),
    _b_driver,
    _b_no_gpu,
    _b_command_not_found,
    _b_import,
    _b_disk_quota,
    _b_simple(
        "network_unreachable",
        "The job tried to reach the internet (e.g. to download a model or dataset from Hugging Face, or to sync "
        "W&B), but compute nodes often have no outbound network. Download on the login node into a cache on "
        "scratch (HF_HOME), then run with HF_HUB_OFFLINE=1 (and WANDB_MODE=offline).",
        ["fix_environment"],
    ),
    _b_cuda_oom,
    _b_nan,
    _b_device_assert,
    _b_simple(
        "state_dict_mismatch",
        "The checkpoint's parameter names or shapes do not match the model: it was saved from a different model "
        "configuration, or with a wrapper prefix ('module.' from DDP, '_orig_mod.' from torch.compile) that must "
        "be stripped when saving or loading.",
        ["fix_code", "derive_run"],
    ),
    _b_simple(
        "weights_only_load",
        "torch.load refused to unpickle non-tensor objects: since PyTorch 2.6 it defaults to weights_only=True. The "
        "file is not corrupt. For checkpoints you wrote yourself pass weights_only=False, or allow-list the needed "
        "classes with torch.serialization.add_safe_globals([...]).",
        ["fix_code"],
    ),
    _b_simple(
        "checkpoint_corrupt",
        "Loading a checkpoint failed because the file is truncated or not a valid archive — usually it was being "
        "written when an earlier job was killed or the disk filled up. Resume from an earlier checkpoint whose "
        "hpclab manifest validates, and write checkpoints atomically (temp file, fsync, rename; manifest last).",
        ["resume", "fix_code"],
    ),
    _b_file_not_found,
    _b_permission,
    _b_simple(
        "port_in_use",
        "Another process on the node already listens on the rendezvous port (29500 by default, often another job "
        "on a shared node). Use a per-job port, e.g. MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000)), or "
        "torchrun --standalone for single-node jobs.",
        ["fix_code", "retry"],
    ),
    _b_segfault,
    _b_python_exception,
    _b_nccl,
    _b_success_no_outputs,
)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_HANG_CAUSES = frozenset(
    {
        "cuda_oom",
        "nan_loss",
        "nccl_error",
        "python_exception",
        "cuda_device_assert",
        "segfault",
        "import_error",
    }
)


def _adjust_for_outcome(ctx: _Ctx, draft: _Draft) -> None:
    """Log signatures of jobs that are still running or that succeeded are weaker evidence."""
    if draft.source != "log":
        return
    if ctx.outcome == "active" and draft.confidence == "high":
        draft.confidence = "medium"
        draft.explanation = "The job is still running. " + draft.explanation
    elif ctx.outcome == "succeeded" and draft.code not in ("nan_loss", "host_oom"):
        draft.confidence = "low"
        draft.explanation = "The job completed; this was logged but did not stop it. " + draft.explanation
    elif ctx.outcome == "succeeded" and draft.code == "host_oom" and draft.confidence == "high":
        draft.confidence = "medium"


def _adjust_timeout_for_errors(ctx: _Ctx, drafts: list[_Draft]) -> None:
    """A TIMEOUT after a fatal error is usually a hang: more walltime would be wasted.

    NCCL errors only count when a collective watchdog fired; aborted communicators are the
    normal side effect of Slurm killing all ranks at the limit.
    """
    timeout = next((d for d in drafts if d.code == "timeout" and d.authoritative), None)
    if timeout is None:
        return
    watchdog = any("Watchdog caught collective" in h.line.text for h in ctx.hits("nccl_error"))
    errors = [
        d.code
        for d in drafts
        if d.code in _HANG_CAUSES and d.confidence != "low" and (d.code != "nccl_error" or watchdog)
    ]
    if not errors:
        return
    timeout.confidence = "medium"
    timeout.explanation += (
        f" The log also shows {', '.join(sorted(set(errors)))} before the kill: if the program stopped making "
        "progress after that error, more walltime will not help, so look at it first."
    )
    timeout.strategies = ["fix_code", *[s for s in timeout.strategies if s != "fix_code"]]


def _apply_history(facts: AttemptFacts, draft: _Draft) -> None:
    repeats = sum(1 for c in facts.previous_codes if c == draft.code)
    if not repeats:
        return
    if draft.code in ("timeout", "preempted") and "resume" in draft.strategies and facts.checkpoint_available:
        # a resume chain: stopping again is expected, each attempt continues from a checkpoint
        draft.explanation += (
            f" This Run already stopped this way {repeats} time(s) before; each resume continues from the latest "
            "checkpoint, but if an attempt makes little progress, request more time (or a non-preemptible partition)."
        )
    else:
        draft.explanation += (
            f" Earlier attempts of this Run already failed the same way {repeats} time(s); an unchanged retry is "
            "unlikely to help."
        )
    strategies: list[Strategy] = [s for s in draft.strategies if s != "retry"]
    if draft.code in ("node_fail", "gpu_hardware_error", "nfs_stale", "segfault", "nccl_error"):
        strategies = [s for s in strategies if s != "contact_support"]
        strategies.insert(min(1, len(strategies)), "contact_support")
    if "retry" in draft.strategies:
        strategies.append("retry")
    draft.strategies = strategies


def _one_per_code(drafts: Sequence[_Draft]) -> list[_Draft]:
    """Keep the best-ranked draft of each code (input already sorted), e.g. when a requeued
    job's appended log shows both a guard failure and a setup failure."""
    seen: set[str] = set()
    out: list[_Draft] = []
    for d in drafts:
        if d.code not in seen:
            seen.add(d.code)
            out.append(d)
    return out


def _freeze(d: _Draft) -> DiagnosisItem:
    evidence = list(dict.fromkeys(e for e in d.evidence if e))[:MAX_EVIDENCE_LINES]
    strategies = list(dict.fromkeys(d.strategies)) or ["none"]
    hint = {k: _clip(str(v), MAX_EVIDENCE_CHARS) for k, v in d.hint.items() if v}
    return DiagnosisItem(
        code=d.code,
        title=d.title or _TITLES[d.code],
        confidence=d.confidence,
        evidence=evidence,
        explanation=d.explanation,
        suggested_strategies=strategies,
        resource_hint=hint,
    )


def _sort_key(d: _Draft) -> tuple[int, int, int, int]:
    return (
        0 if d.authoritative else 1,
        1 if d.incidental else 0,
        _CONF_ORDER[d.confidence],
        _RANK.get(d.code, 50),
    )


def _summary(ctx: _Ctx, items: Sequence[DiagnosisItem]) -> str:
    if not items:
        f = ctx.facts
        return {
            "pending": f"Pending in the queue{f' (reason: {f.reason})' if f.reason else ''}; nothing has failed.",
            "active": "Running; no failure signatures in the scheduler state or log tails.",
            "succeeded": "Completed; no problems found in the scheduler state or log tails.",
            "failed": "Failed, but no cause could be determined.",
            "unknown": "No scheduler outcome yet and no failure signatures in the log tails.",
        }[ctx.outcome]
    primary = items[0]
    text = f"{primary.title} ({primary.confidence} confidence). {_first_sentence(primary.explanation)}"
    others = [i.code for i in items[1:]]
    if others:
        text += f" Also found: {', '.join(others[:4])}{' ...' if len(others) > 4 else ''}."
    return _clip(text, 500)


def _last_error_line(ctx: _Ctx, items: Sequence[DiagnosisItem]) -> str | None:
    exc = ctx.root_exception()
    if exc is not None:
        return _clip(redact_secrets(ctx.logs.lines[exc.index].text.strip()), MAX_EVIDENCE_CHARS)
    for item in items:
        for ev in item.evidence:
            if ev.startswith(("stderr: ", "stdout: ")):
                return ev.split(": ", 1)[1]
    return None


def diagnose(facts: AttemptFacts, stdout_tail: str, stderr_tail: str) -> Diagnosis:
    """Explain why an Attempt failed (or what looks wrong while it runs).

    ``stdout_tail``/``stderr_tail`` are bounded tails of the job's logs (the function itself also
    caps what it scans). Returns an empty ``items`` list when nothing is wrong; never raises for
    odd input.
    """
    logs = _Logs.build(stdout_tail, stderr_tail)
    ctx = _Ctx(facts, logs, _outcome(facts), _effective_signal(facts), _sched_line(facts))
    ctx.stop = _stop_cause(ctx)
    drafts = [d for d in (build(ctx) for build in _BUILDERS) if d is not None]
    for d in drafts:
        _adjust_for_outcome(ctx, d)
    if (killed := _b_killed_signal(ctx, drafts)) is not None:
        drafts.append(killed)
    if ctx.outcome == "failed" and all(d.incidental for d in drafts):
        drafts.append(_b_unknown(ctx))  # side observations never explain a failure
    _adjust_timeout_for_errors(ctx, drafts)
    drafts.sort(key=_sort_key)
    drafts = _one_per_code(drafts)
    if drafts:
        _apply_history(facts, drafts[0])
    items = [_freeze(d) for d in drafts[:MAX_ITEMS]]
    return Diagnosis(
        items=items,
        primary_code=items[0].code if items else None,
        summary=_summary(ctx, items),
        last_error_line=_last_error_line(ctx, items),
    )


_NORMALIZE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0x#"),
    (re.compile(r"\b(?:att|run|exp|op)_[0-9A-Za-z]+"), "<id>"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "<hex>"),
    (re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?"), "#"),
    (re.compile(r"\s+"), " "),
)


def normalize_error_line(line: str | None) -> str:
    """Stable form of an error line for grouping failures across cells: ids, hex, numbers and
    whitespace runs are replaced so 'rank 3 ... 2.00 GiB' and 'rank 5 ... 4.00 GiB' compare equal."""
    if not line:
        return ""
    text = redact_secrets(line.strip())
    for pattern, repl in _NORMALIZE_PATTERNS:
        text = pattern.sub(repl, text)
    return text[:MAX_EVIDENCE_CHARS]
