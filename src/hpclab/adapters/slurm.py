"""Slurm scheduler adapter: typed facts from real Slurm (22.05 - 25.x) command output.

Design rules
------------
* Every command runs as ``env SLURM_TIME_FORMAT=%s <cmd> ...`` so all timestamps are epoch
  seconds; no timezone guessing. ``Unknown``/``None``/``N/A``/``0``/empty become ``None``.
  The one exception is a real ``sbatch`` submission: it prints no times, and with the default
  ``--export=ALL`` the variable would leak into the job's environment.
* Field-delimited output only. The delimiter is ASCII unit separator ``\\x1f`` (``FS``), which
  cannot occur in job names, comments, paths or reasons. ``squeue``/``sinfo`` get it as the
  ``--Format`` field *suffix* with width ``0`` (= print the full value, no padding);
  ``sacct`` via ``--delimiter``. ``sstat``/``sacctmgr`` only have ``|`` (``--parsable2``) and
  are only asked for fields that cannot contain ``|``. Free-text fields always come last.
* A source that cannot answer raises a typed error; it never looks like an empty success.
* Transport errors (auth, timeouts, dropped connections) propagate unchanged.

COMMANDS (exact argv; ``<FS>`` is ``\\x1f``; prefixed by ``env SLURM_TIME_FORMAT=%s`` unless noted)
--------------------------------------------------------------------------------------------------
============== =======================================================================================
purpose        command
============== =======================================================================================
probe          ``sbatch --version`` | ``squeue --help`` | ``sacct --helpformat`` |
               ``scontrol show config`` | ``sstat --helpformat`` | ``srun --version``
submit         ``bash -s -- <receipt|''> <workdir> sbatch --parsable <script>`` (NO env prefix) or
               ``bash -s -- '' <workdir> env SLURM_TIME_FORMAT=%s sbatch --parsable --test-only
               <script>``, with ``SUBMIT_SNIPPET`` on stdin (journals before replying)
receipt        ``Transport.read_range(<receipt>, 0, 64 KiB)``
queue          ``squeue --noheader --array --states=all (--jobs=<id,...> | --user=<u>)
               --Format=JobArrayID:0<FS>,State:0<FS>,UserName:0<FS>,Account:0<FS>,Partition:0<FS>,
               SubmitTime:0<FS>,StartTime:0<FS>,EndTime:0<FS>,TimeLimit:0<FS>,TimeUsed:0<FS>,
               NodeList:0<FS>,RestartCnt:0<FS>,tres-alloc:0<FS>,WorkDir:0<FS>,Name:0<FS>,
               Comment:0<FS>,Reason:0``  (at most 100 ids per call)
accounting     ``sacct --allocations --noheader --parsable2 --delimiter=<FS>
               (--jobs=<id,...> | [--user=<u>] [--name=<n>] --starttime=now-<N>)
               --format=JobID,JobIDRaw,State,ExitCode,User,Account,Partition,Submit,Start,End,
               ElapsedRaw,TimelimitRaw,NodeList,AllocTRES[,WorkDir],JobName[,Comment][,SubmitLine]``
               (bracketed fields only when ``sacct --helpformat`` lists them; ``Comment`` only when
               ``AccountingStoreFlags`` contains ``job_comment``)
cancel         ``scancel <id>``
partitions     ``sinfo --noheader --Node --Format=NodeList:0<FS>,Partition:0<FS>,Available:0<FS>,
               StateLong:0<FS>,Time:0<FS>,CPUsState:0<FS>,Memory:0<FS>,Gres:0<FS>,GresUsed:0<FS>,
               Reason:0`` + ``scontrol --oneliner show partition`` (PriorityTier/PreemptMode)
nodes          the same ``sinfo`` line, plus ``--partition=<p>`` when filtering
associations   ``sacctmgr --noheader --parsable2 show associations where user=<u>
               format=Cluster,Account,User,Partition,QOS,DefaultQOS``
start estimate ``squeue --noheader --start --jobs=<id,...> --Format=JobArrayID:0<FS>,StartTime:0<FS>,
               Reason:0``
step stats     ``sstat --noheader --parsable2 --allsteps --jobs=<id>
               --format=JobID,MaxRSS,AveCPU,TRESUsageInTot``
gpu samples    ``srun --jobid=<id> --overlap --nodes=1 --ntasks=1 --quiet nvidia-smi
               --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw
               --format=csv,noheader,nounits``
usage          ``sacct --allocations --noheader --parsable2 --delimiter=<FS> (--allusers | --user=<u,...>)
               [--accounts=<a>] --starttime=now-<N> [--endtime=now-<M>] --format=JobID,JobIDRaw,User,
               Account,Partition,State,Submit,Start,End,ElapsedRaw,AllocTRES,JobName``
usage segments the same with ``--duplicates`` (one record per requeue segment) and
               ``--format=JobID,JobIDRaw,User,Account,Partition,QOS,State,ExitCode,Submit,Start,End,
               ElapsedRaw,TimelimitRaw,NodeList,AllocTRES,JobName``
segments       ``sacct --allocations --duplicates ... --jobs=<id>`` (same fields as accounting)
account queue  ``squeue --noheader --array --states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED
               --account=<a> --Format=<queue fields>``
signal         ``scancel (--batch | --full) --signal=<SIG> <id>``
config         ``scontrol show config`` (selected keys, see ``CONFIG_KEYS``)
qos limits     ``sacctmgr --noheader --parsable2 show qos [where name=<q,...>] format=Name,MaxWall,
               MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxTRES,GrpTRES,GrpJobs,Preempt,PreemptMode,Priority``
assoc limits   ``sacctmgr --noheader --parsable2 show associations where user=<u> format=Cluster,
               Account,User,Partition,QOS,DefaultQOS,GrpTRES,GrpJobs,MaxTRES,MaxJobs,MaxSubmit,MaxWall``
partitions     ``scontrol --oneliner show partition`` (``partition_details``: MaxTime, DefaultTime,
               PreemptMode, GraceTime, PriorityTier, State, Default, TRES)
test only      ``sbatch --test-only --wrap=true --job-name=hpclab-test-only --output=/dev/null
               [--partition=] [--account=] [--qos=] --nodes=<n> [--gres=gpu[:<type>]:<n>]
               [--cpus-per-task=] [--mem=] --time=<t>`` (start estimate only; nothing is queued)
============== =======================================================================================

``--starttime``/``--endtime`` use Slurm's relative ``now-<seconds>`` form so the window does not
depend on the login node's timezone.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from hpclab.adapters.base import (
    Association,
    CancelResponse,
    CommandFailed,
    CommandResult,
    ConnectionLost,
    GpuSample,
    JobStepStats,
    NodeInfo,
    PartitionInfo,
    QueryResult,
    SchedulerCapabilities,
    StartEstimate,
    SubmissionRejected,
    SubmitResponse,
    Transport,
    TransportError,
    UsageRecord,
)
from hpclab.domain.clock import Clock, SystemClock, iso, parse_iso, parse_slurm_duration
from hpclab.domain.errors import ErrorCode, InvalidArgument
from hpclab.domain.models import JobObservation
from hpclab.domain.states import SCHEDULER_TERMINAL, SchedulerState, normalize_slurm_state
from hpclab.profiles.schema import ResolvedCluster

log = logging.getLogger("hpclab.adapters.slurm")

FS = "\x1f"
ENV_PREFIX: tuple[str, ...] = ("env", "SLURM_TIME_FORMAT=%s")
MAX_IDS_PER_CALL = 100
RECEIPT_MAX_BYTES = 64 * 1024
SUBMIT_FRAME = "HPCLAB-SUBMIT/1"
DEFAULT_ACCOUNTING_WINDOW = timedelta(days=7)

# Fixed bash snippet run via ``Transport.run_script``. Arguments: $1 receipt path ('' = no
# journal), $2 working directory, $3.. the sbatch argv. The journal is written (tmp + mv) on the
# cluster BEFORE anything is printed, so a reply lost on the way back can be recovered from it.
# Receipt format (shared contract, spec 6.1): line 1 rc, line 2 sbatch stdout (newlines folded),
# rest sbatch stderr. stdout of the snippet: one frame line, then sbatch's stdout; stderr: sbatch's.
# sbatch applies SBATCH_* input variables OVER the script's #SBATCH lines; the login shell may
# export some (~/.bashrc, /etc/profile.d). sbatch runs without any of them, so the rendered script
# the ledger records is the whole request; the frame's ``ignored_env`` names what was dropped.
SUBMIT_SNIPPET = r"""set -u
receipt=$1
workdir=$2
shift 2
if ! cd -- "$workdir" 2>/dev/null; then
  printf 'HPCLAB-SUBMIT/1 stage=chdir rc=1 journal=skipped\n'
  printf 'hpclab: cannot change directory to %s\n' "$workdir" >&2
  exit 0
fi
errf=""
if [ -n "$receipt" ]; then
  mkdir -p -- "$(dirname -- "$receipt")" 2>/dev/null
  errf="$receipt.err.$$"
  : >"$errf" 2>/dev/null || errf=""
fi
if [ -z "$errf" ]; then
  errf=$(mktemp 2>/dev/null) || errf="/tmp/hpclab-sbatch-err.$$"
fi
ignored=""
for v in ${!SBATCH_@}; do
  unset "$v" 2>/dev/null && ignored="$ignored${ignored:+,}$v"
done
out=$("$@" </dev/null 2>"$errf")
rc=$?
err=$(cat -- "$errf" 2>/dev/null)
rm -f -- "$errf"
journal=skipped
if [ -n "$receipt" ]; then
  tmp="$receipt.tmp.$$"
  if { printf '%s\n' "$rc"; printf '%s\n' "$(printf '%s' "$out" | tr '\n' ' ')"; printf '%s' "$err"; } >"$tmp" 2>/dev/null \
     && mv -f -- "$tmp" "$receipt" 2>/dev/null; then
    journal=ok
  else
    journal=failed
    rm -f -- "$tmp" 2>/dev/null
  fi
fi
printf 'HPCLAB-SUBMIT/1 stage=sbatch rc=%s journal=%s%s\n' "$rc" "$journal" "${ignored:+ ignored_env=$ignored}"
printf '%s\n' "$out"
printf '%s' "$err" >&2
exit 0
"""

SQUEUE_FIELDS: tuple[str, ...] = (
    "JobArrayID",
    "State",
    "UserName",
    "Account",
    "Partition",
    "SubmitTime",
    "StartTime",
    "EndTime",
    "TimeLimit",
    "TimeUsed",
    "NodeList",
    "RestartCnt",
    "tres-alloc",
    # free text last
    "WorkDir",
    "Name",
    "Comment",
    "Reason",
)
SQUEUE_START_FIELDS: tuple[str, ...] = ("JobArrayID", "StartTime", "Reason")
SACCT_BASE_FIELDS: tuple[str, ...] = (
    "JobID",
    "JobIDRaw",
    "State",
    "ExitCode",
    "User",
    "Account",
    "Partition",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "TimelimitRaw",
    "NodeList",
    "AllocTRES",
)
USAGE_FIELDS: tuple[str, ...] = (
    "JobID",
    "JobIDRaw",
    "User",
    "Account",
    "Partition",
    "State",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "AllocTRES",
    "JobName",
)
SINFO_FIELDS: tuple[str, ...] = (
    "NodeList",
    "Partition",
    "Available",
    "StateLong",
    "Time",
    "CPUsState",
    "Memory",
    "Gres",
    "GresUsed",
    "Reason",
)
SACCTMGR_ASSOC_FIELDS: tuple[str, ...] = ("Cluster", "Account", "User", "Partition", "QOS", "DefaultQOS")
SSTAT_FIELDS: tuple[str, ...] = ("JobID", "MaxRSS", "AveCPU", "TRESUsageInTot")
NVIDIA_QUERY = "index,name,utilization.gpu,memory.used,memory.total,power.draw"
USAGE_SEGMENT_FIELDS: tuple[str, ...] = (
    "JobID",
    "JobIDRaw",
    "User",
    "Account",
    "Partition",
    "QOS",
    "State",
    "ExitCode",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "TimelimitRaw",
    "NodeList",
    "AllocTRES",
    # free text last
    "JobName",
)
QOS_FIELDS: tuple[str, ...] = (
    "Name",
    "MaxWall",
    "MaxTRESPU",
    "MaxJobsPU",
    "MaxSubmitPU",
    "MaxTRES",
    "GrpTRES",
    "GrpJobs",
    "Preempt",
    "PreemptMode",
    "Priority",
)
SACCTMGR_ASSOC_LIMIT_FIELDS: tuple[str, ...] = (
    "Cluster",
    "Account",
    "User",
    "Partition",
    "QOS",
    "DefaultQOS",
    "GrpTRES",
    "GrpJobs",
    "MaxTRES",
    "MaxJobs",
    "MaxSubmit",
    "MaxWall",
)
# ``scontrol show config`` keys reported by ``cluster_config`` (cluster_inspect's scheduler facts).
CONFIG_KEYS: tuple[str, ...] = (
    "ClusterName",
    "SLURM_VERSION",
    "PrivateData",
    "AccountingStorageType",
    "AccountingStorageTRES",
    "AccountingStoreFlags",
    "JobAcctGatherType",
    "JobRequeue",
    "MailProg",
    "MinJobAge",
    "PreemptType",
    "PreemptMode",
)
ACTIVE_QUEUE_STATES = "PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED"
# Signals hpclab may send to a job (walltime/checkpoint requests and stop requests only).
ALLOWED_SIGNALS = frozenset({"USR1", "USR2", "TERM", "INT", "HUP", "CONT", "QUIT"})

_JOB_ID_RE = re.compile(r"^[0-9]+(?:_[0-9]+|\+[0-9]+)?$")
_NAME_RE = re.compile(r"^[A-Za-z0-9._@-]{1,128}$")
_PARSABLE_ID_RE = re.compile(r"^\s*(?P<id>[0-9]+)(?:;(?P<cluster>[^\s;]+))?\s*$")
_TEST_ONLY_RE = re.compile(
    r"Job (?P<id>\d+) to start at (?P<start>\S+) using (?P<procs>\d+) processors? on nodes? "
    r"(?P<nodes>\S+) in partition (?P<part>\S+)"
)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_SECONDS_RE = re.compile(r"^\s*(\d+)\s*(?:sec|secs|s)?\s*$", re.IGNORECASE)
_MEM_RE = re.compile(r"^[0-9]+[KMGT]?$")
_NONE_TOKENS = frozenset({"", "unknown", "none", "n/a", "(null)", "none assigned", "invalid"})
_NOT_IN_QUEUE_RE = re.compile(r"Invalid job id specified", re.IGNORECASE)
_SACCT_DB_ERROR_RE = re.compile(
    r"Problem talking to the database|persistent connection|slurmdbd.*(?:refused|fail|timed out)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# sbatch failure taxonomy
# ---------------------------------------------------------------------------

# Communication-level failures: the controller may or may not have created the job.
_AMBIGUOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (code, re.compile(pat, re.IGNORECASE))
    for code, pat in (
        ("socket_timeout", r"Socket timed out"),
        ("controller_unreachable", r"Unable to contact slurm controller"),
        ("zero_bytes", r"Zero Bytes were transmitted"),
        ("controller_busy", r"temporarily unable to accept job|Resource temporarily unavailable"),
        ("communication_failure", r"Communication (?:connection|send|receive|shutdown) failure"),
        ("standby_controller", r"backup controller in standby mode"),
        (
            "protocol_error",
            r"Unexpected message received|Message (?:receive|send) failure|Protocol version"
            r"|Insane message length|Header lengths are longer than data received",
        ),
        ("connection_error", r"Connection (?:refused|reset|timed out)|Transport endpoint|Broken pipe"),
    )
)

# Definite refusals: sbatch (or the controller) answered "no"; nothing was queued.
_REJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (code, re.compile(pat, re.IGNORECASE))
    for code, pat in (
        ("invalid_account", r"Invalid account"),
        ("invalid_partition", r"invalid partition"),
        ("invalid_time_limit", r"Requested time limit is invalid|Invalid --time specification"),
        ("memory_unsatisfiable", r"Memory specification can not be satisfied"),
        ("node_config_unavailable", r"Requested node configuration is not available"),
        (
            "submit_limit",
            r"QOSMaxSubmitJob|AssocMaxSubmitJob|MaxSubmitJob|Job violates accounting/QOS policy",
        ),
        ("invalid_qos", r"Invalid qos"),
        (
            "invalid_gres",
            r"Invalid generic resource|Invalid GRES|Requested GRES option unsupported|Invalid Trackable RESource",
        ),
        (
            "access_denied",
            r"Access/permission denied|not permitted to use this partition|group not permitted",
        ),
        ("dependency_invalid", r"Job dependency problem|invalid dependency"),
        ("invalid_feature", r"Invalid feature specification"),
        ("invalid_node_name", r"Invalid node name specified"),
        ("invalid_node_count", r"Node count specification invalid"),
        ("too_many_cpus", r"More processors requested than permitted"),
        ("partition_unavailable", r"Requested partition configuration not available now"),
        ("invalid_array", r"Invalid job array specification"),
        ("invalid_reservation", r"Requested reservation is invalid|Access denied to requested reservation"),
        ("invalid_license", r"Invalid license specification"),
        ("invalid_wckey", r"Invalid wckey"),
        (
            "script_unreadable",
            r"Unable to open file|does not look like a batch script|Batch script is empty|DOS line breaks",
        ),
        (
            # getopt and sbatch's own option parsing (command line and #SBATCH lines alike): sbatch
            # prints these and exits before it contacts the controller, e.g. `--signal=B:USR1@70000`
            # (the warning time is a uint16), `--mem-per-gpu=10XB`, `--mail-type=BOGUS`.
            "invalid_option",
            r"unrecognized option|invalid option|Invalid numeric value|is not a valid number"
            r"|requires an argument|doesn't allow an argument|option '--[\w-]+' is ambiguous"
            r"|Invalid --[\w-]+ (?:specification|argument|value)|invalid memory constraint"
            r"|--[\w-]+=\S* invalid\b",
        ),
    )
)
# Refusals worth retrying later without changing the request (site limits, busy partitions).
RETRY_LATER_REASONS = frozenset({"submit_limit", "partition_unavailable"})

# ``env`` (or a shell) could not exec sbatch at all: GNU env quotes the name ('x' or ‘x’),
# BSD env does not; exit 127/126. Nothing can have been submitted.
_SBATCH_MISSING_RE = re.compile(
    r"^env: \S?sbatch\S?: (?:No such file or directory|Permission denied)"
    r"|(?:^|: )sbatch: (?:command )?not found",
    re.MULTILINE,
)
# Lines sbatch itself wrote (``sbatch: error: ...``, getopt's ``sbatch: unrecognized option``).
# Only these may prove a refusal; login-shell noise on stderr must never do so.
_SBATCH_LINE_RE = re.compile(r"^\s*(?:\S*/)?sbatch:")


@dataclass(frozen=True)
class SbatchFailure:
    """Classification of a non-successful sbatch reply."""

    kind: Literal["rejected", "ambiguous"]
    reason_code: str
    message: str


def classify_sbatch_failure(stderr: str, stdout: str = "") -> SbatchFailure:
    """Decide whether a failed sbatch definitely queued nothing.

    ``Batch job submission failed: <reason>`` means the RPC completed and the controller
    refused, unless ``<reason>`` is itself a communication error. Without that line only known
    client-side refusals are definite; anything unrecognized is ambiguous, because treating an
    unknown failure as "not submitted" is how duplicate training runs happen. Refusal wording
    only counts on lines sbatch wrote (``sbatch: ...``): stderr of an SSH session also carries
    login-shell noise, which may push a verdict towards "ambiguous" but never towards "rejected".
    """
    text = "\n".join(p for p in (stderr, stdout) if p)
    own = "\n".join(ln for ln in text.splitlines() if _SBATCH_LINE_RE.match(ln))
    message = _headline(text) or "sbatch failed without output"
    final = None
    for line in own.splitlines():
        if "Batch job submission failed:" in line:
            final = line.split("Batch job submission failed:", 1)[1].strip()
    if final is not None:
        for code, pat in _AMBIGUOUS_PATTERNS:
            if pat.search(final):
                return SbatchFailure("ambiguous", code, message)
        return SbatchFailure("rejected", _rejection_code(own) or "controller_refused", message)
    if _SBATCH_MISSING_RE.search(text):
        return SbatchFailure("rejected", "sbatch_unavailable", message)
    for code, pat in _AMBIGUOUS_PATTERNS:
        if pat.search(text):
            return SbatchFailure("ambiguous", code, message)
    refusal = _rejection_code(own)
    if refusal is not None:
        return SbatchFailure("rejected", refusal, message)
    return SbatchFailure("ambiguous", "unrecognized_failure", message)


def _rejection_code(sbatch_lines: str) -> str | None:
    for code, pat in _REJECTION_PATTERNS:
        if pat.search(sbatch_lines):
            return code
    return None


def _headline(text: str) -> str:
    """The most informative line: the final 'submission failed' line, else the last error line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "submission failed" in ln:
            return ln
    for ln in reversed(lines):
        if "error" in ln.lower():
            return ln
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Value parsers (pure, unit-tested against real output fixtures)
# ---------------------------------------------------------------------------


def opt(value: str | None) -> str | None:
    """Normalize Slurm's many spellings of "no value" to None."""
    if value is None:
        return None
    v = value.strip()
    return None if v.lower() in _NONE_TOKENS else v


def text_or_none(value: str | None) -> str | None:
    """For free text (names, comments): only empty and ``(null)`` mean absent."""
    if value is None:
        return None
    v = value.strip()
    return None if v in ("", "(null)") else v


def epoch_to_iso(value: str | None) -> str | None:
    """Epoch seconds (``SLURM_TIME_FORMAT=%s``) to ISO UTC; placeholders and ``0`` -> None."""
    v = opt(value)
    if v is None:
        return None
    try:
        secs = int(v)
    except ValueError:
        log.warning("unexpected non-epoch Slurm time %r (SLURM_TIME_FORMAT not honored?)", v)
        return None
    if secs <= 0:
        return None
    return iso(datetime.fromtimestamp(secs, UTC))


def parse_int(value: str | None) -> int | None:
    """Integer field; placeholders and garbage -> None (``0`` is a real value)."""
    v = opt(value)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def parse_duration(value: str | None) -> int | None:
    """Slurm duration (``1-02:03:04``, ``12:34``, ``00:01.234``) to seconds; UNLIMITED -> None."""
    v = opt(value)
    if v is None:
        return None
    if "." in v:
        v = v.split(".", 1)[0]
    return parse_slurm_duration(v)


def parse_exit_code(value: str | None) -> tuple[int | None, int | None]:
    """sacct ``ExitCode`` ``code:signal`` -> (code, signal)."""
    if value is None or not value.strip():
        return None, None
    code_s, _, sig_s = value.strip().partition(":")
    try:
        code = int(code_s)
    except ValueError:
        return None, None
    try:
        sig = int(sig_s) if sig_s else 0
    except ValueError:
        sig = None
    return code, sig


def split_job_id(job_id: str) -> tuple[str, str | None]:
    """``123_4`` -> (``123_4``, ``4``); ``123_[5-9]`` -> (``123_[5-9]``, ``[5-9]``); ``123`` -> (``123``, None)."""
    jid = job_id.strip()
    if "_" in jid:
        return jid, jid.split("_", 1)[1] or None
    return jid, None


def gpus_from_tres(tres: str | None) -> int | None:
    """GPU count from a TRES string (``cpu=8,mem=64G,gres/gpu=2,gres/gpu:h100=2``).

    Prefers the untyped ``gres/gpu`` total; falls back to the sum of typed entries so a
    site that only tracks typed TRES is not reported as 0. Empty string -> None (unknown);
    a TRES string without GPUs -> 0.
    """
    t = opt(tres)
    if t is None:
        return None
    total: int | None = None
    typed = 0
    for item in t.split(","):
        key, _, val = item.partition("=")
        key = key.strip()
        if key == "gres/gpu":
            total = _count(val)
        elif key.startswith("gres/gpu:"):
            typed += _count(val) or 0
    if total is not None:
        return total
    return typed


def _count(value: str) -> int | None:
    """Gres/TRES counts may carry K/M/G multipliers (``mps:400K``)."""
    v = value.strip()
    mult = 1
    if v and v[-1] in "KMGkmg":
        mult = {"k": 1024, "m": 1024**2, "g": 1024**3}[v[-1].lower()]
        v = v[:-1]
    try:
        return int(v) * mult
    except ValueError:
        return None


def parse_size_bytes(value: str | None) -> int | None:
    """sstat/sacct memory values (``1048576K``, ``12.5M``, ``3G``, ``0``) to bytes."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    mult = 1
    if v[-1].upper() in "KMGTP":
        mult = 1024 ** ("KMGTP".index(v[-1].upper()) + 1)
        v = v[:-1]
    try:
        return int(float(v) * mult)
    except ValueError:
        return None


@dataclass(frozen=True)
class GresItem:
    """One generic resource entry, e.g. ``gpu:h100:8(S:0-1)`` -> name gpu, type h100, count 8."""

    name: str
    type: str | None
    count: int
    detail: str | None = None


def split_top_level(value: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside parentheses (``gpu:h100:3(IDX:0,2-3),shard:0`` has 2 items)."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def parse_gres(value: str | None) -> list[GresItem]:
    """Parse sinfo ``Gres``/``GresUsed`` strings.

    Handles ``gpu:8``, ``gpu:h100:8(S:0-1)``, ``gpu:b200:8,shard:b200:32``,
    ``gpu:h100:3(IDX:0-2)``, ``gpu:0``, ``gpu:(null):0(IDX:N/A)``, ``gpu:a100:no_consume:4``
    and ``(null)``. Unparsable items are skipped (logged), never guessed.
    """
    v = opt(value)
    if v is None:
        return []
    items: list[GresItem] = []
    for raw in split_top_level(v.replace("(null)", "")):
        core, _, rest = raw.partition("(")
        detail = rest[:-1] if rest.endswith(")") else (rest or None)
        parts = [p for p in core.split(":") if p != "no_consume"]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        gtype: str | None = None
        count: int | None = 1
        if len(parts) == 2:
            maybe = _count(parts[1])
            if maybe is None:
                gtype = parts[1]
            else:
                count = maybe
        elif len(parts) >= 3:
            gtype = parts[1]
            count = _count(parts[-1])
        if gtype in ("(null)", ""):
            gtype = None
        if count is None:
            log.debug("skipping unparsable gres item %r", raw)
            continue
        items.append(GresItem(name=name, type=gtype, count=count, detail=detail))
    return items


UNTYPED_GPU = "untyped"
_UNAVAILABLE_STATE_WORDS = (
    "drain",
    "down",
    "fail",
    "maint",
    "unknown",
    "inval",
    "reboot",
    "power",
    "future",
    "boot",
)
_NOT_RESPONDING_FLAGS = "*~#!%@^"


def node_availability(state_long: str | None) -> Literal["free", "busy", "unavailable"]:
    """Classify a sinfo ``StateLong`` value.

    ``free``: idle/mixed and healthy (its idle GPUs can be allocated now); ``unavailable``:
    drain/down/fail/maint/not-responding/powered-down etc.; ``busy``: everything else
    (allocated, completing, reserved, planned).
    """
    s = (state_long or "").strip().lower()
    if not s:
        return "unavailable"
    if any(flag in s for flag in _NOT_RESPONDING_FLAGS):
        return "unavailable"
    if any(word in s for word in _UNAVAILABLE_STATE_WORDS):
        return "unavailable"
    base = s.split("+", 1)[0]
    if base in ("idle", "mixed", "mix"):
        return "free"
    return "busy"


def gpu_counts_by_type(gres: str | None, gres_used: str | None) -> dict[str, tuple[int, int]]:
    """Per GPU type ``(total, used)`` on one node.

    Untyped used entries are attributed to the node's single GPU type; typed used entries win
    over untyped ones so a node is never double counted.
    """
    totals: dict[str, int] = {}
    for item in parse_gres(gres):
        if item.name == "gpu":
            key = item.type or UNTYPED_GPU
            totals[key] = totals.get(key, 0) + item.count
    used_items = [i for i in parse_gres(gres_used) if i.name == "gpu"]
    used: dict[str, int] = {}
    typed = [i for i in used_items if i.type]
    if typed:
        for item in typed:
            key = item.type or UNTYPED_GPU
            used[key] = used.get(key, 0) + item.count
    else:
        untyped_total = sum(i.count for i in used_items)
        if untyped_total:
            key = next(iter(totals)) if len(totals) == 1 else UNTYPED_GPU
            used[key] = untyped_total
    keys = list(totals) + [k for k in used if k not in totals]
    return {k: (totals.get(k, 0), min(used.get(k, 0), totals.get(k, used.get(k, 0)))) for k in keys}


def split_records(stdout: str, n_fields: int, sep: str = FS) -> list[list[str]]:
    """Split delimiter-separated output into records of exactly ``n_fields``.

    Slurm prints free text (job names, comments, submit lines) raw, so ANY field may contain a
    newline. Physical lines are joined until the record holds its ``n_fields - 1`` separators;
    after that a line without a separator continues the last field, and a line with one starts
    the next record. This is unambiguous because the first field (a job id or node name) never
    contains a newline. Separators beyond ``n_fields - 1`` belong to the last field; trailing
    newlines of the last field are dropped. Lines before the first record that carry no
    separator (e.g. ``CLUSTER: x`` banners) and a final record cut short are dropped (logged),
    never padded into a fake record.
    """
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in stdout.split("\n")]
    if n_fields <= 1:
        return [[ln] for ln in lines if ln.strip()]
    need = n_fields - 1
    records: list[list[str]] = []
    buf: str | None = None
    for line in lines:
        if buf is None:
            if sep in line:
                buf = line
            elif line.strip():
                log.debug("dropping stray line before the first record: %r", line)
        elif buf.count(sep) < need or sep not in line:
            buf += "\n" + line
        else:
            _close_record(buf, need, sep, records)
            buf = line
    if buf is not None:
        _close_record(buf, need, sep, records)
    return records


def _close_record(buf: str, need: int, sep: str, records: list[list[str]]) -> None:
    parts = buf.split(sep)
    if len(parts) <= need:
        log.warning("dropping truncated record (%d of %d fields): %r", len(parts), need + 1, buf[:200])
        return
    records.append([*parts[:need], sep.join(parts[need:]).rstrip("\n")])


def format_spec(fields: Sequence[str], sep: str = FS) -> str:
    """squeue/sinfo ``--Format`` value: full-width fields, ``sep`` as suffix except on the last."""
    return ",".join(f"{f}:0{sep}" for f in fields[:-1]) + f",{fields[-1]}:0"


def parse_test_only(text: str) -> SubmitEstimate | None:
    """Parse ``sbatch --test-only`` output (``sbatch: Job N to start at T using P processors on
    nodes X in partition Y``)."""
    m = _TEST_ONLY_RE.search(text or "")
    if not m:
        return None
    start = m.group("start")
    start_iso = epoch_to_iso(start) if start.isdigit() else _local_time_or_none(start)
    return SubmitEstimate(
        job_id=m.group("id"),
        start_time=start_iso,
        processors=int(m.group("procs")),
        nodes=m.group("nodes"),
        partition=m.group("part"),
        raw=m.group(0),
    )


def _local_time_or_none(value: str) -> str | None:
    # A non-epoch time means SLURM_TIME_FORMAT was ignored; its timezone is unknown, so it
    # is only returned when it carries an explicit offset.
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return iso(dt) if dt.tzinfo is not None else None


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def parse_key_values(line: str) -> dict[str, str]:
    """``Key=Value Key2=Value2`` (scontrol --oneliner) into a dict. Values have no spaces."""
    out: dict[str, str] = {}
    for token in line.split():
        key, sep, value = token.partition("=")
        if sep:
            out[key] = value
    return out


def parse_scontrol_config(text: str) -> dict[str, str]:
    """``scontrol show config`` lines ``Key    = Value``."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and " " not in key.strip():
            out[key.strip()] = value.strip()
    return out


def parse_nvidia_csv(text: str) -> list[GpuSample]:
    """``nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw
    --format=csv,noheader,nounits`` rows; ``[N/A]``/``[Not Supported]`` -> None."""
    samples: list[GpuSample] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index = parse_int(parts[0])
        if index is None:
            continue
        name = ",".join(parts[1 : len(parts) - 4]).strip() or None
        util, mem_used, mem_total, power = (_float(p) for p in parts[-4:])
        samples.append(
            GpuSample(
                index=index,
                name=name,
                util_percent=util,
                mem_used_mib=mem_used,
                mem_total_mib=mem_total,
                power_w=power,
            )
        )
    return samples


def _float(value: str) -> float | None:
    v = value.strip()
    if not v or v.startswith("["):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_config_seconds(value: str | None) -> int | None:
    """``scontrol show config`` durations such as ``MinJobAge = 300 sec`` -> 300."""
    if value is None:
        return None
    m = _SECONDS_RE.match(value)
    return int(m.group(1)) if m else None


def config_value(value: str | None) -> str | None:
    """A ``scontrol show config`` value; absent/``(null)``/empty -> None. ``none`` is kept: for
    PrivateData it is a fact ("nothing hidden"), not a missing value."""
    if value is None:
        return None
    v = value.strip()
    return None if v in ("", "(null)") else v


def private_data_flags(value: str | None) -> frozenset[str] | None:
    """``PrivateData`` -> the hidden kinds (``{"jobs", "usage"}``); empty set = nothing hidden;
    None = unknown (the configuration could not be read)."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("", "none", "(null)"):
        return frozenset()
    return frozenset(f.strip() for f in v.split(",") if f.strip())


def gpu_types_from_tres(tres: str | None) -> dict[str, int]:
    """Typed GPU entries of a TRES string: ``gres/gpu:h100=4`` -> ``{"h100": 4}``."""
    out: dict[str, int] = {}
    for item in (opt(tres) or "").split(","):
        key, _, val = item.partition("=")
        key = key.strip()
        if key.startswith("gres/gpu:"):
            n = _count(val)
            if n is not None:
                gtype = key.split(":", 1)[1]
                out[gtype] = out.get(gtype, 0) + n
    return out


def tres_gpu_limit(tres: str | None) -> int | None:
    """GPU count in a *limit* TRES string (``cpu=64,gres/gpu=8``); None when GPUs are not limited
    (unlike ``gpus_from_tres``, a missing entry here means "no limit", never 0)."""
    t = opt(tres)
    if t is None:
        return None
    total: int | None = None
    typed: int | None = None
    for item in t.split(","):
        key, _, val = item.partition("=")
        key = key.strip()
        if key == "gres/gpu":
            total = _count(val)
        elif key.startswith("gres/gpu:"):
            n = _count(val)
            if n is not None:
                typed = (typed or 0) + n
    return total if total is not None else typed


def normalize_signal(signal: str) -> str:
    """``SIGUSR1``/``usr1``/``USR1`` -> ``USR1``. Numbers are refused: they differ between
    platforms (USR1 is 10 on Linux, 30 on macOS). Only ``ALLOWED_SIGNALS`` pass."""
    s = str(signal).strip().upper().removeprefix("SIG")
    if s not in ALLOWED_SIGNALS:
        raise InvalidArgument(
            f"signal {signal!r} is not allowed", details={"allowed": sorted(ALLOWED_SIGNALS)}
        )
    return s


# ---------------------------------------------------------------------------
# Result types that do not exist in adapters.base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubmitEstimate:
    """What ``sbatch --test-only`` predicted. Nothing was queued."""

    job_id: str | None
    start_time: str | None
    processors: int | None
    nodes: str | None
    partition: str | None
    raw: str


@dataclass(frozen=True)
class SubmitReceipt:
    """Parsed on-cluster sbatch journal (``submit.receipt``)."""

    path: str
    exit_code: int
    stdout: str
    stderr: str
    job_id: str | None
    cluster_name: str | None
    written_at: str | None
    failure: SbatchFailure | None


@dataclass(frozen=True)
class QosLimit:
    """One QOS from ``sacctmgr show qos``. ``None`` limits are unset (unlimited), because the
    query itself succeeded; a hidden QOS table raises instead of returning Nones."""

    name: str
    max_wall_s: int | None = None
    max_tres_per_user: str | None = None
    max_gpus_per_user: int | None = None
    max_jobs_per_user: int | None = None
    max_submit_per_user: int | None = None
    max_tres_per_job: str | None = None
    max_gpus_per_job: int | None = None
    grp_tres: str | None = None
    grp_gpus: int | None = None
    grp_jobs: int | None = None
    preempt: tuple[str, ...] = ()
    preempt_mode: str | None = None
    priority: int | None = None


@dataclass(frozen=True)
class AssocLimit:
    """A user association with its limits (``None`` = not set on this association)."""

    cluster: str | None
    account: str
    user: str
    partition: str | None
    qos: tuple[str, ...]
    default_qos: str | None
    grp_tres: str | None = None
    grp_gpus: int | None = None
    grp_jobs: int | None = None
    max_tres: str | None = None
    max_gpus_per_job: int | None = None
    max_jobs: int | None = None
    max_submit: int | None = None
    max_wall_s: int | None = None


@dataclass(frozen=True)
class PartitionDetail:
    """``scontrol show partition`` attributes sinfo does not show."""

    name: str
    state: str | None = None
    default: bool = False
    max_time_s: int | None = None
    default_time_s: int | None = None
    preempt_mode: str | None = None
    grace_time_s: int | None = None
    priority_tier: int | None = None
    total_nodes: int | None = None
    total_gpus: int | None = None
    allow_accounts: str | None = None
    allow_qos: str | None = None
    qos: str | None = None


@dataclass(frozen=True)
class JobShape:
    """A resource shape for ``test_only`` (``gpus`` is the job total, spread over ``nodes``)."""

    partition: str | None = None
    gpus: int = 0
    gpu_type: str | None = None
    time: str = "01:00:00"
    nodes: int = 1
    account: str | None = None
    qos: str | None = None
    cpus: int | None = None
    mem: str | None = None

    @classmethod
    def coerce(cls, value: JobShape | Mapping[str, Any]) -> JobShape:
        if isinstance(value, JobShape):
            return value
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(value) - known
        if unknown:
            raise InvalidArgument(f"unknown job shape fields {sorted(unknown)}")
        return cls(**{k: v for k, v in value.items() if v is not None})


@dataclass(frozen=True)
class ShapeEstimate(StartEstimate):
    """``sbatch --test-only`` for a shape: a *low-confidence* scheduler guess, not an ETA.
    ``job_id`` is the id the job would have received (nothing was queued)."""

    partition: str | None = None
    nodes: str | None = None
    processors: int | None = None
    raw: str = ""


@dataclass(frozen=True)
class UsageSegment(UsageRecord):
    """One accounting record of ``sacct --allocations --duplicates``: a job, or one requeue
    segment of it (same ``job_id``, own Start/End/State). ``gpus_known=False`` means the GPU count
    could not be read (``gpus`` is then 0 and must not be summed as a fact)."""

    job_id_raw: str = ""
    qos: str | None = None
    raw_state: str | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    time_limit_s: int | None = None
    node_list: str | None = None
    gpus_known: bool = True
    gpu_types: tuple[tuple[str, int], ...] = ()


@dataclass
class _ProbeFacts:
    sacct_fields: frozenset[str] = frozenset()
    accounting_store_flags: frozenset[str] = frozenset()
    accounting_storage_type: str | None = None
    cluster_name: str | None = None
    preempt_type: str | None = None  # e.g. preempt/partition_prio, preempt/qos, preempt/none
    # AccountingStorageTRES lists untyped gres/gpu (None: unknown). Slurm's default does NOT,
    # and then TRES strings carry no GPU entry even for GPU jobs.
    gpu_tres_tracked: bool | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------


class SlurmScheduler:
    """``adapters.base.Scheduler`` for Slurm over any ``Transport``.

    Stateless apart from the cached capability probe; safe to share between threads as long as
    the transport is.
    """

    query_timeout_s: float = 30.0
    submit_timeout_s: float = 60.0
    probe_timeout_s: float = 20.0
    accounting_timeout_s: float = 60.0
    cancel_timeout_s: float = 30.0
    # A probe that could not read the controller configuration is cached only this long, so a
    # controller restart during the first probe does not degrade a long-lived process for good.
    partial_probe_retry_s: float = 300.0

    def __init__(self, transport: Transport, cluster: ResolvedCluster, *, clock: Clock | None = None) -> None:
        self.transport = transport
        self.cluster = cluster
        self.cluster_id = cluster.cluster_id
        self.clock: Clock = clock or SystemClock()
        self._caps: SchedulerCapabilities | None = None
        self._facts = _ProbeFacts()
        self._reprobe_at: float | None = None  # monotonic deadline of a partial probe

    # -- plumbing -----------------------------------------------------------
    def _now(self) -> str:
        now = iso(self.clock.now())
        assert now is not None
        return now

    def _run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        return self.transport.run([*ENV_PREFIX, *argv], timeout=timeout)

    def _fail(
        self, what: str, res: CommandResult, *, code: ErrorCode = ErrorCode.UNAVAILABLE, **details: object
    ) -> CommandFailed:
        line = _headline(res.stderr) or _headline(res.stdout) or f"exit status {res.exit_code}"
        return CommandFailed(
            f"{what} failed: {line}",
            res,
            code=code,
            details={
                "command": what,
                "exit_code": res.exit_code,
                "stderr": res.stderr[-2000:],
                **details,
            },
        )

    # -- capabilities -------------------------------------------------------
    def probe_capabilities(self, *, refresh: bool = False) -> SchedulerCapabilities:
        """Probe once per instance (``refresh=True`` to redo). Missing sbatch -> Unsupported.

        Only ``scontrol show config`` talks to the controller; if it fails the result is
        cached for ``partial_probe_retry_s`` only (see ``notes``), then probed again.
        """
        if self._caps is not None and not refresh and not self._probe_expired():
            return self._caps
        caps = SchedulerCapabilities(probed_at=self._now())
        facts = _ProbeFacts()
        partial = False

        res = self._run(["sbatch", "--version"], timeout=self.probe_timeout_s)
        if res.exit_code != 0:
            raise self._fail(
                "sbatch --version", res, code=ErrorCode.UNSUPPORTED, hint="is this a Slurm login node?"
            )
        caps.slurm_version = _version_string(res.stdout)
        version = parse_version(res.stdout)

        res = self._run(["squeue", "--help"], timeout=self.probe_timeout_s)
        caps.squeue_json = res.exit_code == 0 and "--json" in res.stdout

        res = self._run(["scontrol", "show", "config"], timeout=self.probe_timeout_s)
        if res.exit_code == 0:
            conf = parse_scontrol_config(res.stdout)
            caps.cluster_name = opt(conf.get("ClusterName"))
            caps.min_job_age_s = parse_config_seconds(conf.get("MinJobAge"))
            caps.accounting_store_flags = config_value(conf.get("AccountingStoreFlags"))
            caps.private_data = config_value(conf.get("PrivateData"))
            facts.accounting_storage_type = conf.get("AccountingStorageType")
            facts.accounting_store_flags = frozenset(
                f.strip() for f in (conf.get("AccountingStoreFlags") or "").split(",") if f.strip()
            )
            facts.cluster_name = opt(conf.get("ClusterName"))
            facts.preempt_type = opt(conf.get("PreemptType"))
            tracked_tres = conf.get("AccountingStorageTRES")
            if tracked_tres is not None:
                names = {t.strip().lower() for t in tracked_tres.split(",")}
                facts.gpu_tres_tracked = "gres/gpu" in names
                if not facts.gpu_tres_tracked:
                    facts.notes.append(
                        "AccountingStorageTRES does not track gres/gpu: job GPU counts are unknown"
                    )
        else:
            partial = res.exit_code != 127  # a missing scontrol will not come back by itself
            facts.notes.append(
                "scontrol show config failed; accounting configuration unknown"
                + (f" (probed again after {self.partial_probe_retry_s:.0f}s)" if partial else "")
            )

        res = self._run(["sacct", "--helpformat"], timeout=self.probe_timeout_s)
        if res.exit_code == 0:
            facts.sacct_fields = frozenset(res.stdout.split())
            caps.sacct_available = True
            if facts.accounting_storage_type and facts.accounting_storage_type.endswith("/none"):
                caps.sacct_available = False
                facts.notes.append(f"accounting storage disabled ({facts.accounting_storage_type})")
        else:
            facts.notes.append("sacct not available")
        comment_listed = "Comment" in facts.sacct_fields
        caps.sacct_comment_field = (
            caps.sacct_available and comment_listed and "job_comment" in facts.accounting_store_flags
        )
        if caps.sacct_available and comment_listed and not caps.sacct_comment_field:
            facts.notes.append(
                "job comments are not stored in accounting (AccountingStoreFlags lacks job_comment)"
            )

        res = self._run(["sstat", "--helpformat"], timeout=self.probe_timeout_s)
        caps.sstat_available = res.exit_code == 0

        res = self._run(["srun", "--version"], timeout=self.probe_timeout_s)
        caps.srun_overlap = res.exit_code == 0 and version is not None and version >= (20, 11, 0)
        if res.exit_code == 0 and not caps.srun_overlap:
            facts.notes.append("srun --overlap needs Slurm >= 20.11; live GPU sampling disabled")

        caps.test_only = True
        caps.parsable_submit = True
        caps.notes = list(facts.notes)
        # Facts first: a concurrent reader that sees the new caps must also see their facts.
        self._facts = facts
        self._reprobe_at = self.clock.monotonic() + self.partial_probe_retry_s if partial else None
        self._caps = caps
        return caps

    def adopt_probe(self, other: SlurmScheduler) -> None:
        """Take over ``other``'s completed capability probe (e.g. a budget-limited copy used by
        cluster_inspect), so a fresh probe is not lost with the copy. Facts before caps, as in
        ``probe_capabilities``."""
        if other._caps is None:
            return
        self._facts = other._facts
        self._reprobe_at = other._reprobe_at
        self._caps = other._caps

    def _probe_expired(self) -> bool:
        return self._reprobe_at is not None and self.clock.monotonic() >= self._reprobe_at

    # -- submission ---------------------------------------------------------
    def submit(
        self, script_path: str, *, workdir: str, test_only: bool = False, receipt_path: str | None = None
    ) -> SubmitResponse:
        """Run sbatch through ``SUBMIT_SNIPPET`` (journal first, then reply).

        Returns ``SubmitResponse``; with ``test_only=True`` nothing is queued and the response
        has ``job_id == ""`` and ``raw`` = the ``Job N to start at ...`` line (see
        ``parse_test_only``). Raises ``SubmissionRejected`` (details: ``reason_code``, ``stderr``,
        ``retry_later``) on a definite refusal and ``ConnectionLost`` (``maybe_executed=True``)
        when the outcome is unknown. For ``test_only`` an unknown outcome is a plain
        ``TransportError``: a test-only call can never queue a job.

        ``receipt_path`` must be absolute: the snippet quotes it, so ``~``/``$USER`` would not
        be expanded and the journal would silently land somewhere ``read_receipt`` never looks.
        """
        _check_submit_paths(script_path, workdir, receipt_path)
        # Only --test-only prints a time. A real submission runs sbatch in the user's own
        # environment: with --export=ALL, SLURM_TIME_FORMAT=%s would be exported into the job
        # and change what scontrol/squeue/sacct print inside it.
        sbatch = (
            [*ENV_PREFIX, "sbatch", "--parsable", "--test-only", script_path]
            if test_only
            else ["sbatch", "--parsable", script_path]
        )
        res = self.transport.run_script(
            SUBMIT_SNIPPET,
            timeout=self.submit_timeout_s,
            args=[receipt_path or "", workdir, *sbatch],
        )
        received_at = self._now()
        frame, sbatch_stdout = _parse_frame(res.stdout)
        details: dict[str, object] = {
            "exit_code": res.exit_code,
            "stdout": res.stdout[-2000:],
            "stderr": res.stderr[-2000:],
            "receipt_path": receipt_path,
        }
        if frame is None:
            if test_only:
                raise TransportError("sbatch --test-only: no reply from the submit wrapper", details=details)
            raise ConnectionLost(
                "sbatch outcome unknown: the submit wrapper produced no reply frame",
                details={**details, "reason_code": "no_reply_frame"},
            )
        if frame.get("stage") == "chdir":
            raise SubmissionRejected(
                f"submission working directory is not accessible: {workdir}",
                details={**details, "reason_code": "workdir_unavailable", "retry_later": False},
            )
        if frame.get("journal") == "failed":
            log.warning("sbatch receipt could not be journaled at %s", receipt_path)
        ignored_env = sorted(v for v in frame.get("ignored_env", "").split(",") if v)
        if ignored_env:
            details["ignored_env"] = ignored_env
            log.warning(
                "sbatch ran without %s from the cluster login environment: sbatch would have let "
                "them override the job script's #SBATCH lines (remove them from ~/.bashrc)",
                ",".join(ignored_env),
            )
        rc = parse_int(frame.get("rc"))
        details["sbatch_exit_code"] = rc
        if test_only:
            return self._test_only_response(rc, res.stderr, sbatch_stdout, received_at, details)
        parsed = _parse_job_id(sbatch_stdout)
        if parsed is not None:
            if rc != 0:
                log.warning(
                    "sbatch exited %s but printed job id %s; treating the job as queued", rc, parsed[0]
                )
            return SubmitResponse(
                job_id=parsed[0], cluster_name=parsed[1], raw=sbatch_stdout.strip(), submitted_at=received_at
            )
        if rc == 0:
            raise ConnectionLost(
                "sbatch returned success without a parsable job id; outcome unknown",
                details={**details, "reason_code": "unparsable_reply"},
            )
        failure = classify_sbatch_failure(res.stderr, sbatch_stdout)
        raise _failure_error(failure, details)

    def _test_only_response(
        self, rc: int | None, stderr: str, stdout: str, received_at: str, details: dict[str, object]
    ) -> SubmitResponse:
        if rc == 0:
            est = parse_test_only(stderr) or parse_test_only(stdout)
            raw = est.raw if est else (stderr.strip() or stdout.strip())
            return SubmitResponse(job_id="", cluster_name=None, raw=raw, submitted_at=received_at)
        failure = classify_sbatch_failure(stderr, stdout)
        if failure.kind == "rejected":
            raise _failure_error(failure, details)
        raise TransportError(f"sbatch --test-only could not be evaluated: {failure.message}", details=details)

    def test_submit(self, script_path: str, *, workdir: str) -> SubmitEstimate:
        """``sbatch --test-only``: validate a script against live limits without queuing it."""
        resp = self.submit(script_path, workdir=workdir, test_only=True)
        est = parse_test_only(resp.raw)
        return est or SubmitEstimate(
            job_id=None, start_time=None, processors=None, nodes=None, partition=None, raw=resp.raw
        )

    def read_receipt_record(self, receipt_path: str) -> SubmitReceipt | None:
        """Parse the sbatch journal written by ``SUBMIT_SNIPPET``; None if it does not exist."""
        try:
            rr = self.transport.read_range(receipt_path, 0, RECEIPT_MAX_BYTES, timeout=self.query_timeout_s)
        except FileNotFoundError:
            return None
        text = rr.data.decode("utf-8", "replace")
        first, _, rest = text.partition("\n")
        stdout, _, stderr = rest.partition("\n")
        try:
            rc = int(first.strip())
        except ValueError:
            log.warning("malformed sbatch receipt at %s", receipt_path)
            return SubmitReceipt(
                path=receipt_path,
                exit_code=-1,
                stdout=stdout,
                stderr=text,
                job_id=None,
                cluster_name=None,
                written_at=_mtime_iso(rr.mtime),
                failure=SbatchFailure("ambiguous", "malformed_receipt", "receipt is not in hpclab format"),
            )
        parsed = _parse_job_id(stdout)
        failure = None
        if parsed is None:
            failure = (
                SbatchFailure("ambiguous", "unparsable_reply", "sbatch succeeded without a job id")
                if rc == 0
                else classify_sbatch_failure(stderr, stdout)
            )
        return SubmitReceipt(
            path=receipt_path,
            exit_code=rc,
            stdout=stdout,
            stderr=stderr,
            job_id=parsed[0] if parsed else None,
            cluster_name=parsed[1] if parsed else None,
            written_at=_mtime_iso(rr.mtime),
            failure=failure,
        )

    def read_receipt(self, receipt_path: str) -> SubmitResponse | None:
        """Recover an sbatch reply from its on-cluster journal.

        * journal missing -> None (sbatch may not have run, or the journal write failed);
        * journal with a job id -> ``SubmitResponse`` (``submitted_at`` = journal mtime);
        * journal of a definite refusal -> raises ``SubmissionRejected`` (nothing was queued);
        * journal of an ambiguous failure -> None: it identifies no job, keep reconciling.
        """
        rec = self.read_receipt_record(receipt_path)
        if rec is None:
            return None
        if rec.job_id is not None:
            return SubmitResponse(
                job_id=rec.job_id,
                cluster_name=rec.cluster_name,
                raw=rec.stdout.strip(),
                submitted_at=rec.written_at or self._now(),
            )
        if rec.failure is not None and rec.failure.kind == "rejected":
            raise _failure_error(
                rec.failure,
                {"exit_code": rec.exit_code, "stderr": rec.stderr[-2000:], "receipt_path": receipt_path},
            )
        return None

    # -- queue ---------------------------------------------------------------
    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        """squeue (all states still in controller memory) for ``job_ids`` or all of ``user``'s jobs.

        Ids unknown to the controller are simply absent (not an error). If every call fails the
        error is raised; if only some chunks fail, the result is ``complete=False`` with the jobs
        that were observed.
        """
        observed_at = self._now()
        if job_ids is not None:
            ids = _validated_ids(job_ids)
            if not ids:
                return QueryResult(source="squeue", observed_at=observed_at, jobs=[], complete=True)
            selectors = [[f"--jobs={','.join(chunk)}"] for chunk in _chunks(ids, MAX_IDS_PER_CALL)]
        else:
            who = _validated_name(user or self.transport.remote_user(), "user")
            selectors = [[f"--user={who}"]]
        self.probe_capabilities()  # cached; tells whether tres-alloc can count GPUs
        jobs: list[JobObservation] = []
        errors: list[CommandFailed] = []
        for selector in selectors:
            argv = [
                "squeue",
                "--noheader",
                "--array",
                "--states=all",
                *selector,
                f"--Format={format_spec(SQUEUE_FIELDS)}",
            ]
            res = self._run(argv, timeout=self.query_timeout_s)
            if res.exit_code != 0 and not _NOT_IN_QUEUE_RE.search(res.stderr):
                errors.append(self._fail("squeue", res))
                continue
            for rec in split_records(res.stdout, len(SQUEUE_FIELDS)):
                jobs.append(self._squeue_observation(dict(zip(SQUEUE_FIELDS, rec, strict=True)), observed_at))
        if errors and len(errors) == len(selectors):
            raise errors[0]
        return QueryResult(
            source="squeue",
            observed_at=observed_at,
            jobs=jobs,
            complete=not errors,
            error=errors[0].message if errors else None,
        )

    def _squeue_observation(self, row: dict[str, str], observed_at: str) -> JobObservation:
        job_id, task = split_job_id(row["JobArrayID"])
        state = normalize_slurm_state(row["State"])
        terminal = state in SCHEDULER_TERMINAL
        return JobObservation(
            job_id=job_id,
            source="squeue",
            observed_at=observed_at,
            state=state,
            raw_state=opt(row["State"]),
            name=text_or_none(row["Name"]),
            comment=text_or_none(row["Comment"]),
            user=opt(row["UserName"]),
            account=opt(row["Account"]),
            partition=opt(row["Partition"]),
            reason=opt(row["Reason"]),
            submit_time=epoch_to_iso(row["SubmitTime"]),
            # For pending jobs StartTime/EndTime are estimates, not facts.
            start_time=epoch_to_iso(row["StartTime"]) if state != SchedulerState.PENDING else None,
            end_time=epoch_to_iso(row["EndTime"]) if terminal else None,
            elapsed_s=parse_duration(row["TimeUsed"]),
            time_limit_s=parse_duration(row["TimeLimit"]),
            node_list=opt(row["NodeList"]),
            alloc_gpus=self._alloc_gpus(row["tres-alloc"]),
            restart_count=parse_int(row["RestartCnt"]),
            array_task_id=task,
            work_dir=text_or_none(row["WorkDir"]),
        )

    def _alloc_gpus(self, tres: str | None) -> int | None:
        """GPU count of a TRES string, or None when it cannot be known.

        Without untyped ``gres/gpu`` in AccountingStorageTRES (Slurm's default) a GPU job's TRES
        has no GPU entry at all, so "no entry" means unknown, never 0 GPUs. That holds until the
        probe has confirmed the tracking (``scontrol show config`` may have failed): only then is
        a TRES string without GPUs a CPU-only job (spec §4 inv. 7 - unknown usage is never 0).
        """
        gpus = gpus_from_tres(tres)
        if not gpus and self._facts.gpu_tres_tracked is not True:
            return None
        return gpus

    # -- accounting ------------------------------------------------------------
    def _require_accounting(self) -> SchedulerCapabilities:
        caps = self.probe_capabilities()
        if not caps.sacct_available:
            raise TransportError(
                "Slurm accounting (sacct) is not available on this cluster",
                code=ErrorCode.UNSUPPORTED,
                details={"notes": caps.notes},
            )
        return caps

    def _sacct_fields(self, caps: SchedulerCapabilities) -> tuple[str, ...]:
        fields = list(SACCT_BASE_FIELDS)
        known = self._facts.sacct_fields
        if "WorkDir" in known:
            fields.append("WorkDir")
        fields.append("JobName")
        if caps.sacct_comment_field:
            fields.append("Comment")
        if "SubmitLine" in known:
            fields.append("SubmitLine")
        return tuple(fields)

    def _run_sacct(self, argv: list[str], what: str) -> CommandResult:
        res = self._run(argv, timeout=self.accounting_timeout_s)
        if res.exit_code == 0:
            if not res.stdout.strip() and _SACCT_DB_ERROR_RE.search(res.stderr):
                # "no rows" is only a fact if slurmdbd answered; never an empty success.
                raise self._fail(what, res, code=ErrorCode.UNAVAILABLE)
            return res
        err = res.stderr.lower()
        if (
            res.exit_code == 127
            or "accounting storage is disabled" in err
            or "not running a supported accounting" in err
        ):
            raise self._fail(what, res, code=ErrorCode.UNSUPPORTED)
        raise self._fail(what, res, code=ErrorCode.UNAVAILABLE)

    def query_accounting(
        self,
        *,
        job_ids: list[str] | None = None,
        user: str | None = None,
        name: str | None = None,
        since: str | None = None,
    ) -> QueryResult:
        """sacct allocation lines. By ids: all time unless ``since``; otherwise the window starts at
        ``since`` (default now - 7 days). Accounting unavailable -> raises (never empty success)."""
        caps = self._require_accounting()
        observed_at = self._now()
        fields = self._sacct_fields(caps)
        base = [
            "sacct",
            "--allocations",
            "--noheader",
            "--parsable2",
            f"--delimiter={FS}",
            f"--format={','.join(fields)}",
        ]
        selectors: list[list[str]]
        if job_ids is not None:
            ids = _validated_ids(job_ids)
            if not ids:
                return QueryResult(source="sacct", observed_at=observed_at, jobs=[], complete=True)
            window = [self._starttime_arg(since)] if since else []
            selectors = [[f"--jobs={','.join(chunk)}", *window] for chunk in _chunks(ids, MAX_IDS_PER_CALL)]
        else:
            sel = [f"--user={_validated_name(user or self.transport.remote_user(), 'user')}"]
            if name:
                if "," in name or "\n" in name:
                    raise InvalidArgument("job name filter may not contain ',' or newlines")
                sel.append(f"--name={name}")
            sel.append(self._starttime_arg(since))
            selectors = [sel]
        jobs: list[JobObservation] = []
        for selector in selectors:
            res = self._run_sacct([*base, *selector], "sacct")
            for rec in split_records(res.stdout, len(fields)):
                jobs.append(self._sacct_observation(dict(zip(fields, rec, strict=True)), observed_at))
        return QueryResult(source="sacct", observed_at=observed_at, jobs=jobs, complete=True)

    def _starttime_arg(self, since: str | None) -> str:
        return f"--starttime={self._relative(since, default=DEFAULT_ACCOUNTING_WINDOW)}"

    def _relative(self, when: str | None, *, default: timedelta | None = None) -> str:
        """ISO time -> Slurm ``now-<seconds>`` (timezone-free; rounded to include the boundary)."""
        now = self.clock.now()
        if when is None and default is not None:
            delta = default.total_seconds()
        else:
            delta = (now - _parse_when(when)).total_seconds()
        return f"now-{max(0, math.ceil(delta))}"

    def _sacct_observation(self, row: dict[str, str], observed_at: str) -> JobObservation:
        job_id, task = split_job_id(row["JobID"])
        code, sig = parse_exit_code(row.get("ExitCode"))
        limit_min = parse_int(row.get("TimelimitRaw"))
        return JobObservation(
            job_id=job_id,
            source="sacct",
            observed_at=observed_at,
            state=normalize_slurm_state(row["State"]),
            raw_state=opt(row["State"]),
            name=text_or_none(row.get("JobName")),
            comment=text_or_none(row.get("Comment")),
            user=opt(row.get("User")),
            account=opt(row.get("Account")),
            partition=opt(row.get("Partition")),
            submit_time=epoch_to_iso(row.get("Submit")),
            start_time=epoch_to_iso(row.get("Start")),
            end_time=epoch_to_iso(row.get("End")),
            elapsed_s=parse_int(row.get("ElapsedRaw")),
            time_limit_s=limit_min * 60 if limit_min is not None else None,
            exit_code=code,
            exit_signal=sig,
            node_list=opt(row.get("NodeList")),
            alloc_gpus=self._alloc_gpus(row.get("AllocTRES")),
            array_task_id=task,
            submit_line=text_or_none(row.get("SubmitLine")),
            work_dir=text_or_none(row.get("WorkDir")),
        )

    # -- cancel -------------------------------------------------------------------
    def cancel(self, job_id: str) -> CancelResponse:
        """``scancel <id>``. Identity verification is the caller's job.

        ``accepted=False`` means the controller answered "no" (already completed, invalid id,
        permission). A communication failure (overloaded or unreachable controller) is not an
        answer: it raises ``ConnectionLost`` (retryable, maybe executed) so the cancel is retried
        instead of being recorded as refused while the job keeps running.
        """
        (jid,) = _validated_ids([job_id])
        res = self._run(["scancel", jid], timeout=self.cancel_timeout_s)
        raw = "\n".join(p for p in (res.stdout.strip(), res.stderr.strip()) if p)
        if res.exit_code != 0:
            for code, pat in _AMBIGUOUS_PATTERNS:
                if pat.search(res.stderr):
                    raise ConnectionLost(
                        f"scancel {jid}: no answer from the controller ({_headline(res.stderr)})",
                        details={"job_id": jid, "reason_code": code, "stderr": res.stderr[-2000:]},
                    )
        return CancelResponse(job_id=jid, accepted=res.exit_code == 0, raw=raw)

    # -- cluster resources -----------------------------------------------------------
    def _sinfo_rows(self, partition: str | None) -> list[dict[str, str]]:
        argv = ["sinfo", "--noheader", "--Node"]
        if partition:
            argv.append(f"--partition={_validated_name(partition, 'partition')}")
        argv.append(f"--Format={format_spec(SINFO_FIELDS)}")
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            raise self._fail("sinfo", res)
        return [
            dict(zip(SINFO_FIELDS, rec, strict=True)) for rec in split_records(res.stdout, len(SINFO_FIELDS))
        ]

    def nodes(self, partition: str | None = None) -> list[NodeInfo]:
        """One entry per node (partitions merged), in sinfo order."""
        merged: dict[str, NodeInfo] = {}
        for row in self._sinfo_rows(partition):
            name = row["NodeList"].strip()
            part = row["Partition"].strip().rstrip("*")
            prev = merged.get(name)
            if prev is not None:
                if part not in prev.partitions:
                    prev.partitions.append(part)
                continue
            merged[name] = NodeInfo(
                name=name,
                state=row["StateLong"].strip(),
                partitions=[part],
                gres=opt(row["Gres"]),
                gres_used=opt(row["GresUsed"]),
                reason=opt(row["Reason"]),
            )
        return list(merged.values())

    def partitions(self) -> list[PartitionInfo]:
        """GPU capacity per partition and type: total, used, free (idle/mixed healthy nodes only)
        and unavailable (idle GPUs on drained/down/failed/not-responding nodes)."""
        self.probe_capabilities()  # PreemptType decides how partition tiers are read
        rows = self._sinfo_rows(None)
        extra = self._scontrol_partitions()
        order: list[str] = []
        acc: dict[str, _PartitionAcc] = {}
        for row in rows:
            raw_name = row["Partition"].strip()
            name = raw_name.rstrip("*")
            pa = acc.get(name)
            if pa is None:
                pa = acc[name] = _PartitionAcc(name=name)
                order.append(name)
            pa.default = pa.default or raw_name.endswith("*")
            pa.state = opt(row["Available"]) or pa.state
            pa.max_time_s = parse_duration(row["Time"])
            node = row["NodeList"].strip()
            if node in pa.nodes:
                continue
            pa.nodes.add(node)
            availability = node_availability(row["StateLong"])
            for gtype, (total, used) in gpu_counts_by_type(row["Gres"], row["GresUsed"]).items():
                slot = pa.gpus.setdefault(gtype, {"total": 0, "used": 0, "free": 0, "unavailable": 0})
                slot["total"] += total
                slot["used"] += used
                idle = max(0, total - used)
                if availability == "free":
                    slot["free"] += idle
                elif availability == "unavailable":
                    slot["unavailable"] += idle
        for name, kv in extra.items():
            if name not in acc:
                acc[name] = _PartitionAcc(name=name)
                order.append(name)
            pa = acc[name]
            pa.default = pa.default or kv.get("Default", "").upper() == "YES"
            if pa.max_time_s is None:
                pa.max_time_s = parse_duration(kv.get("MaxTime"))
            if pa.state is None:
                pa.state = opt(kv.get("State"))
        tiers = {n: parse_int(kv.get("PriorityTier")) for n, kv in extra.items()}
        return [
            PartitionInfo(
                name=name,
                state=acc[name].state,
                max_time_s=acc[name].max_time_s,
                default=acc[name].default,
                nodes_total=len(acc[name].nodes),
                gpu_types=acc[name].gpus,
                preemptible=self._preemptible(name, extra.get(name), tiers),
            )
            for name in order
        ]

    def _scontrol_partitions(self) -> dict[str, dict[str, str]]:
        """Best effort: partition attributes sinfo does not show. Failure -> {} (logged)."""
        res = self._run(["scontrol", "--oneliner", "show", "partition"], timeout=self.query_timeout_s)
        if res.exit_code != 0:
            log.info("scontrol show partition failed (rc=%s); preemption info unknown", res.exit_code)
            return {}
        out: dict[str, dict[str, str]] = {}
        for line in res.stdout.splitlines():
            kv = parse_key_values(line)
            if "PartitionName" in kv:
                out[kv["PartitionName"]] = kv
        return out

    def _preemptible(self, name: str, kv: dict[str, str] | None, tiers: dict[str, int | None]) -> bool | None:
        """Profile rule wins. Otherwise: PreemptMode OFF (or PreemptType none) -> False; a
        PriorityTier below another partition's -> True. A top-tier partition is only known to be
        safe under ``preempt/partition_prio``; with QOS-based (or unknown) preemption the answer
        depends on the job's QOS, so it is None, never a confident False."""
        rule = self.cluster.profile.partitions.get(name)
        if rule is not None:
            return rule.preemptible
        if kv is None:
            return None
        mode = (kv.get("PreemptMode") or "").upper()
        preempt_type = (self._facts.preempt_type or "").lower()
        if mode == "OFF" or preempt_type == "preempt/none":
            return False
        tier = tiers.get(name)
        if not mode or tier is None:
            return None
        others = [t for n, t in tiers.items() if n != name and t is not None]
        if others and tier < max(others):
            return True
        return False if preempt_type == "preempt/partition_prio" else None

    # -- associations / estimates / live stats -------------------------------------
    def associations(self, user: str) -> list[Association]:
        """The user's associations on THIS cluster.

        One slurmdbd often serves several clusters and sacctmgr lists all of them, so rows are
        kept only for the probed ``ClusterName`` (all rows when it is unknown).
        """
        self.probe_capabilities()
        cluster_name = (self._facts.cluster_name or "").lower()
        who = _validated_name(user, "user")
        argv = [
            "sacctmgr",
            "--noheader",
            "--parsable2",
            "show",
            "associations",
            "where",
            f"user={who}",
            f"format={','.join(SACCTMGR_ASSOC_FIELDS)}",
        ]
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            err = res.stderr.lower()
            unsupported = res.exit_code == 127 or "not running a supported accounting" in err
            raise self._fail(
                "sacctmgr", res, code=ErrorCode.UNSUPPORTED if unsupported else ErrorCode.UNAVAILABLE
            )
        out: list[Association] = []
        for rec in split_records(res.stdout, len(SACCTMGR_ASSOC_FIELDS), sep="|"):
            row = dict(zip(SACCTMGR_ASSOC_FIELDS, rec, strict=True))
            if not row["User"].strip():
                continue
            row_cluster = row["Cluster"].strip().lower()
            if cluster_name and row_cluster and row_cluster != cluster_name:
                continue
            out.append(
                Association(
                    user=row["User"].strip(),
                    account=row["Account"].strip(),
                    partition=opt(row["Partition"]),
                    qos=[q for q in (s.strip() for s in row["QOS"].split(",")) if q],
                    default_qos=opt(row["DefaultQOS"]),
                )
            )
        return out

    def start_estimates(self, job_ids: list[str]) -> list[StartEstimate]:
        """Scheduler's expected start for PENDING jobs among ``job_ids`` (others are omitted)."""
        ids = _validated_ids(job_ids)
        out: list[StartEstimate] = []
        for chunk in _chunks(ids, MAX_IDS_PER_CALL):
            argv = [
                "squeue",
                "--noheader",
                "--start",
                f"--jobs={','.join(chunk)}",
                f"--Format={format_spec(SQUEUE_START_FIELDS)}",
            ]
            res = self._run(argv, timeout=self.query_timeout_s)
            if res.exit_code != 0 and not _NOT_IN_QUEUE_RE.search(res.stderr):
                raise self._fail("squeue --start", res)
            for rec in split_records(res.stdout, len(SQUEUE_START_FIELDS)):
                row = dict(zip(SQUEUE_START_FIELDS, rec, strict=True))
                out.append(
                    StartEstimate(
                        job_id=row["JobArrayID"].strip(),
                        start_time=epoch_to_iso(row["StartTime"]),
                        reason=opt(row["Reason"]),
                    )
                )
        return out

    def step_stats(self, job_id: str) -> list[JobStepStats]:
        """Live per-step stats of a RUNNING job; [] when the job has no running steps."""
        caps = self.probe_capabilities()
        if not caps.sstat_available:
            raise TransportError("sstat is not available on this cluster", code=ErrorCode.UNSUPPORTED)
        (jid,) = _validated_ids([job_id])
        argv = [
            "sstat",
            "--noheader",
            "--parsable2",
            "--allsteps",
            f"--jobs={jid}",
            f"--format={','.join(SSTAT_FIELDS)}",
        ]
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            if re.search(
                r"couldn't get steps|no steps running|Invalid job id|not running", res.stderr, re.IGNORECASE
            ):
                return []
            raise self._fail("sstat", res)
        out: list[JobStepStats] = []
        for rec in split_records(res.stdout, len(SSTAT_FIELDS), sep="|"):
            row = dict(zip(SSTAT_FIELDS, rec, strict=True))
            step_id = row["JobID"].strip()
            base, _, step = step_id.partition(".")
            out.append(
                JobStepStats(
                    job_id=base,
                    step=step or step_id,
                    max_rss_bytes=parse_size_bytes(row["MaxRSS"]),
                    ave_cpu_s=parse_duration(row["AveCPU"]),
                    tres_usage=opt(row["TRESUsageInTot"]),
                )
            )
        return out

    def gpu_samples(self, job_id: str, *, timeout: float = 20.0) -> list[GpuSample]:
        """nvidia-smi inside the job's allocation (first node only).

        Errors: job pending -> code not_ready; job finished/unknown -> precondition_failed;
        srun without ``--overlap`` or no nvidia-smi -> unsupported; anything else -> unavailable.
        """
        caps = self.probe_capabilities()
        if not caps.srun_overlap:
            raise TransportError(
                "live GPU sampling needs 'srun --overlap' (Slurm >= 20.11)", code=ErrorCode.UNSUPPORTED
            )
        (jid,) = _validated_ids([job_id])
        argv = [
            "srun",
            f"--jobid={jid}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--quiet",
            "nvidia-smi",
            f"--query-gpu={NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
        ]
        res = self._run(argv, timeout=timeout)
        if res.exit_code != 0:
            err = res.stderr
            if re.search(r"pending execution|is pending", err, re.IGNORECASE):
                raise self._fail("srun nvidia-smi", res, code=ErrorCode.NOT_READY, reason="job_pending")
            if re.search(
                r"Invalid job id|already completing or completed|has already finished|Expired",
                err,
                re.IGNORECASE,
            ):
                raise self._fail(
                    "srun nvidia-smi", res, code=ErrorCode.PRECONDITION_FAILED, reason="job_not_running"
                )
            if re.search(
                r"nvidia-smi.*(No such file|not found)|No devices were found", err + res.stdout, re.IGNORECASE
            ):
                raise self._fail("srun nvidia-smi", res, code=ErrorCode.UNSUPPORTED, reason="no_gpus_visible")
            raise self._fail("srun nvidia-smi", res)
        return parse_nvidia_csv(res.stdout)

    def usage(
        self, *, account: str | None, users: list[str] | None, since: str, until: str | None
    ) -> list[UsageRecord]:
        """Allocation rows for GPU-hour reports (raw elapsed; callers clip to their window).

        Raises ``unsupported`` when accounting does not track ``gres/gpu``: ``UsageRecord.gpus``
        cannot say "unknown", and 0 would report a GPU-heavy lab as idle.
        """
        self._require_accounting()
        if self._facts.gpu_tres_tracked is False:
            raise TransportError(
                "GPU usage is not recorded by this cluster's accounting "
                "(AccountingStorageTRES lacks gres/gpu); GPU-hours cannot be computed",
                code=ErrorCode.UNSUPPORTED,
                details={"remediation": "ask the admins to add gres/gpu to AccountingStorageTRES"},
            )
        argv = [
            "sacct",
            "--allocations",
            "--noheader",
            "--parsable2",
            f"--delimiter={FS}",
        ]
        if users:
            argv.append(f"--user={','.join(_validated_name(u, 'user') for u in users)}")
        else:
            argv.append("--allusers")
        if account:
            argv.append(f"--accounts={_validated_name(account, 'account')}")
        argv.append(f"--starttime={self._relative(since)}")
        if until is not None and _parse_when(until) < self.clock.now():
            argv.append(f"--endtime={self._relative(until)}")
        argv.append(f"--format={','.join(USAGE_FIELDS)}")
        res = self._run_sacct(argv, "sacct (usage)")
        out: list[UsageRecord] = []
        for rec in split_records(res.stdout, len(USAGE_FIELDS)):
            row = dict(zip(USAGE_FIELDS, rec, strict=True))
            out.append(
                UsageRecord(
                    job_id=row["JobID"].strip(),
                    user=row["User"].strip(),
                    account=opt(row["Account"]),
                    partition=opt(row["Partition"]),
                    state=str(normalize_slurm_state(row["State"])),
                    gpus=gpus_from_tres(row["AllocTRES"]) or 0,
                    elapsed_s=parse_int(row["ElapsedRaw"]) or 0,
                    start_time=epoch_to_iso(row["Start"]),
                    end_time=epoch_to_iso(row["End"]),
                    submit_time=epoch_to_iso(row["Submit"]),
                    job_name=text_or_none(row["JobName"]),
                )
            )
        return out

    # -- wave-3 extensions: signals, config facts, limits, shapes, segments ---------------------
    def signal_job(self, job_id: str, signal: str, *, batch: bool = True) -> CancelResponse:
        """``scancel --batch --signal=<SIG> <id>`` (``batch=False``: ``--full``, every step).

        ``--batch`` reaches only the batch shell, i.e. hpclab's wrapper, which records the
        request and forwards it to the program. ``accepted=False`` is the controller's "no" (job
        pending or finished, invalid id); communication failures raise ``ConnectionLost``.
        Identity verification is the caller's job, exactly as for ``cancel``.
        """
        (jid,) = _validated_ids([job_id])
        sig = normalize_signal(signal)
        argv = ["scancel", "--batch" if batch else "--full", f"--signal={sig}", jid]
        res = self._run(argv, timeout=self.cancel_timeout_s)
        raw = "\n".join(p for p in (res.stdout.strip(), res.stderr.strip()) if p)
        if res.exit_code != 0:
            for code, pat in _AMBIGUOUS_PATTERNS:
                if pat.search(res.stderr):
                    raise ConnectionLost(
                        f"scancel --signal={sig} {jid}: no answer from the controller ({_headline(res.stderr)})",
                        details={
                            "job_id": jid,
                            "signal": sig,
                            "reason_code": code,
                            "stderr": res.stderr[-2000:],
                        },
                    )
        return CancelResponse(job_id=jid, accepted=res.exit_code == 0, raw=raw)

    def cluster_config(self) -> dict[str, str | None]:
        """Fresh ``scontrol show config`` values of ``CONFIG_KEYS`` (None: not reported).

        Raises (unsupported without scontrol, unavailable when the controller does not answer).
        """
        res = self._run(["scontrol", "show", "config"], timeout=self.probe_timeout_s)
        if res.exit_code != 0:
            code = ErrorCode.UNSUPPORTED if res.exit_code == 127 else ErrorCode.UNAVAILABLE
            raise self._fail("scontrol show config", res, code=code)
        conf = parse_scontrol_config(res.stdout)
        return {key: config_value(conf.get(key)) for key in CONFIG_KEYS}

    def _sacctmgr_failure(self, what: str, res: CommandResult) -> CommandFailed:
        err = res.stderr.lower()
        if res.exit_code == 127 or "not running a supported accounting" in err:
            return self._fail(what, res, code=ErrorCode.UNSUPPORTED)
        if re.search(r"access denied|permission denied|not permitted|privatedata", err):
            return self._fail(what, res, code=ErrorCode.UNAVAILABLE, reason="access_denied")
        return self._fail(what, res, code=ErrorCode.UNAVAILABLE)

    def qos_limits(self, names: Sequence[str] | None = None) -> list[QosLimit]:
        """``sacctmgr show qos`` for ``names`` (all when empty). A refused or failed query raises:
        hidden limits are unavailable, never "no limit"."""
        wanted = [_validated_name(n, "qos") for n in (names or [])]
        argv = ["sacctmgr", "--noheader", "--parsable2", "show", "qos"]
        if wanted:
            argv += ["where", f"name={','.join(wanted)}"]
        argv.append(f"format={','.join(QOS_FIELDS)}")
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            raise self._sacctmgr_failure("sacctmgr show qos", res)
        out: list[QosLimit] = []
        for rec in split_records(res.stdout, len(QOS_FIELDS), sep="|"):
            row = dict(zip(QOS_FIELDS, rec, strict=True))
            name = row["Name"].strip()
            if not name:
                continue
            out.append(
                QosLimit(
                    name=name,
                    max_wall_s=parse_duration(row["MaxWall"]),
                    max_tres_per_user=opt(row["MaxTRESPU"]),
                    max_gpus_per_user=tres_gpu_limit(row["MaxTRESPU"]),
                    max_jobs_per_user=parse_int(row["MaxJobsPU"]),
                    max_submit_per_user=parse_int(row["MaxSubmitPU"]),
                    max_tres_per_job=opt(row["MaxTRES"]),
                    max_gpus_per_job=tres_gpu_limit(row["MaxTRES"]),
                    grp_tres=opt(row["GrpTRES"]),
                    grp_gpus=tres_gpu_limit(row["GrpTRES"]),
                    grp_jobs=parse_int(row["GrpJobs"]),
                    preempt=tuple(q for q in (x.strip() for x in row["Preempt"].split(",")) if q),
                    preempt_mode=opt(row["PreemptMode"]),
                    priority=parse_int(row["Priority"]),
                )
            )
        return out

    def association_limits(self, user: str) -> list[AssocLimit]:
        """The user's associations on THIS cluster with their limits (see ``associations``)."""
        self.probe_capabilities()
        cluster_name = (self._facts.cluster_name or "").lower()
        who = _validated_name(user, "user")
        argv = [
            "sacctmgr",
            "--noheader",
            "--parsable2",
            "show",
            "associations",
            "where",
            f"user={who}",
            f"format={','.join(SACCTMGR_ASSOC_LIMIT_FIELDS)}",
        ]
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            raise self._sacctmgr_failure("sacctmgr show associations", res)
        out: list[AssocLimit] = []
        for rec in split_records(res.stdout, len(SACCTMGR_ASSOC_LIMIT_FIELDS), sep="|"):
            row = dict(zip(SACCTMGR_ASSOC_LIMIT_FIELDS, rec, strict=True))
            if not row["User"].strip():
                continue
            row_cluster = row["Cluster"].strip().lower()
            if cluster_name and row_cluster and row_cluster != cluster_name:
                continue
            out.append(
                AssocLimit(
                    cluster=opt(row["Cluster"]),
                    account=row["Account"].strip(),
                    user=row["User"].strip(),
                    partition=opt(row["Partition"]),
                    qos=tuple(q for q in (x.strip() for x in row["QOS"].split(",")) if q),
                    default_qos=opt(row["DefaultQOS"]),
                    grp_tres=opt(row["GrpTRES"]),
                    grp_gpus=tres_gpu_limit(row["GrpTRES"]),
                    grp_jobs=parse_int(row["GrpJobs"]),
                    max_tres=opt(row["MaxTRES"]),
                    max_gpus_per_job=tres_gpu_limit(row["MaxTRES"]),
                    max_jobs=parse_int(row["MaxJobs"]),
                    max_submit=parse_int(row["MaxSubmit"]),
                    max_wall_s=parse_duration(row["MaxWall"]),
                )
            )
        return out

    def partition_details(self) -> list[PartitionDetail]:
        """Per-partition MaxTime/DefaultTime/PreemptMode/GraceTime/PriorityTier/State/Default.

        Unlike the best-effort lookup inside ``partitions()``, a failure raises."""
        res = self._run(["scontrol", "--oneliner", "show", "partition"], timeout=self.query_timeout_s)
        if res.exit_code != 0:
            code = ErrorCode.UNSUPPORTED if res.exit_code == 127 else ErrorCode.UNAVAILABLE
            raise self._fail("scontrol show partition", res, code=code)
        out: list[PartitionDetail] = []
        for line in res.stdout.splitlines():
            kv = parse_key_values(line)
            name = kv.get("PartitionName")
            if not name:
                continue
            out.append(
                PartitionDetail(
                    name=name,
                    state=opt(kv.get("State")),
                    default=kv.get("Default", "").upper() == "YES",
                    max_time_s=parse_duration(kv.get("MaxTime")),
                    default_time_s=parse_duration(kv.get("DefaultTime")),
                    preempt_mode=opt(kv.get("PreemptMode")),
                    grace_time_s=parse_int(kv.get("GraceTime")),
                    priority_tier=parse_int(kv.get("PriorityTier")),
                    total_nodes=parse_int(kv.get("TotalNodes")),
                    total_gpus=tres_gpu_limit(kv.get("TRES")),
                    allow_accounts=opt(kv.get("AllowAccounts")),
                    allow_qos=opt(kv.get("AllowQos")),
                    qos=opt(kv.get("QoS")),
                )
            )
        return out

    def test_only(self, shape: JobShape | Mapping[str, Any]) -> ShapeEstimate:
        """``sbatch --test-only --wrap=true`` for a resource shape: nothing is queued.

        The start time is the scheduler's current guess (low confidence; backfill and new
        submissions change it). A definite refusal raises ``SubmissionRejected``; an unclear
        failure a plain ``TransportError`` (a test-only call can never queue a job).
        """
        sh = JobShape.coerce(shape)
        if sh.nodes < 1 or sh.gpus < 0 or (sh.cpus is not None and sh.cpus < 1):
            raise InvalidArgument("job shape needs nodes >= 1, gpus >= 0 and cpus >= 1")
        if parse_slurm_duration(sh.time) in (None, 0):
            raise InvalidArgument(f"invalid time limit {sh.time!r}")
        argv = [
            *ENV_PREFIX,
            "sbatch",
            "--test-only",
            "--wrap=true",
            "--job-name=hpclab-test-only",
            "--output=/dev/null",
        ]
        if sh.partition:
            argv.append(f"--partition={_validated_name(sh.partition, 'partition')}")
        if sh.account:
            argv.append(f"--account={_validated_name(sh.account, 'account')}")
        if sh.qos:
            argv.append(f"--qos={_validated_name(sh.qos, 'qos')}")
        argv.append(f"--nodes={sh.nodes}")
        if sh.gpus:
            per_node = math.ceil(sh.gpus / sh.nodes)
            gtype = f"{_validated_name(sh.gpu_type, 'gpu type')}:" if sh.gpu_type else ""
            argv.append(f"--gres=gpu:{gtype}{per_node}")
        if sh.cpus:
            argv.append(f"--cpus-per-task={sh.cpus}")
        if sh.mem:
            if not _MEM_RE.match(sh.mem):
                raise InvalidArgument(f"invalid memory {sh.mem!r} (use e.g. 64G)")
            argv.append(f"--mem={sh.mem}")
        argv.append(f"--time={sh.time.strip()}")
        res = self.transport.run(argv, timeout=self.submit_timeout_s)
        details: dict[str, object] = {"exit_code": res.exit_code, "stderr": res.stderr[-2000:]}
        if res.exit_code != 0:
            failure = classify_sbatch_failure(res.stderr, res.stdout)
            if failure.kind == "rejected":
                raise _failure_error(failure, details)
            raise TransportError(
                f"sbatch --test-only could not be evaluated: {failure.message}", details=details
            )
        est = parse_test_only(res.stderr) or parse_test_only(res.stdout)
        if est is None:
            return ShapeEstimate(
                job_id="",
                start_time=None,
                reason="unparsed test-only reply",
                raw=(res.stderr or res.stdout)[-500:],
            )
        return ShapeEstimate(
            job_id=est.job_id or "",
            start_time=est.start_time,
            reason=None,
            partition=est.partition,
            nodes=est.nodes,
            processors=est.processors,
            raw=est.raw,
        )

    def accounting_segments(self, job_id: str) -> list[JobObservation]:
        """Every accounting record of ``job_id`` (``sacct --duplicates``): one per requeue
        segment, oldest first, ``restart_count`` = position in that order.

        Job ids are reused after a slurmdbd reset, so records of an unrelated older job can
        appear too: callers must match ``submit_time``/``user`` against what they submitted.
        """
        caps = self._require_accounting()
        (jid,) = _validated_ids([job_id])
        observed_at = self._now()
        fields = self._sacct_fields(caps)
        argv = [
            "sacct",
            "--allocations",
            "--duplicates",
            "--noheader",
            "--parsable2",
            f"--delimiter={FS}",
            f"--format={','.join(fields)}",
            f"--jobs={jid}",
        ]
        res = self._run_sacct(argv, "sacct --duplicates")
        obs = [
            self._sacct_observation(dict(zip(fields, rec, strict=True)), observed_at)
            for rec in split_records(res.stdout, len(fields))
        ]
        obs = [o for o in obs if o.job_id == jid]
        obs.sort(key=lambda o: (o.start_time is None, o.start_time or "", o.submit_time or ""))
        return [o.model_copy(update={"restart_count": i}) for i, o in enumerate(obs)]

    def usage_segments(
        self, *, account: str | None, users: list[str] | None, since: str, until: str | None
    ) -> list[UsageSegment]:
        """``usage`` with requeue segments (``--duplicates``), QOS, exit code, time limit, node
        list and typed GPU entries. GPU count: the untyped ``gres/gpu`` total (typed entries only
        when it is absent), so ``gres/gpu=4,gres/gpu:h100=4`` is 4 GPUs, never 8.
        """
        self._require_accounting()
        if self._facts.gpu_tres_tracked is False:
            raise TransportError(
                "GPU usage is not recorded by this cluster's accounting "
                "(AccountingStorageTRES lacks gres/gpu); GPU-hours cannot be computed",
                code=ErrorCode.UNSUPPORTED,
                details={"remediation": "ask the admins to add gres/gpu to AccountingStorageTRES"},
            )
        argv = ["sacct", "--allocations", "--duplicates", "--noheader", "--parsable2", f"--delimiter={FS}"]
        if users:
            argv.append(f"--user={','.join(_validated_name(u, 'user') for u in users)}")
        else:
            argv.append("--allusers")
        if account:
            argv.append(f"--accounts={_validated_name(account, 'account')}")
        argv.append(f"--starttime={self._relative(since)}")
        if until is not None and _parse_when(until) < self.clock.now():
            argv.append(f"--endtime={self._relative(until)}")
        argv.append(f"--format={','.join(USAGE_SEGMENT_FIELDS)}")
        res = self._run_sacct(argv, "sacct (usage segments)")
        out: list[UsageSegment] = []
        for rec in split_records(res.stdout, len(USAGE_SEGMENT_FIELDS)):
            row = dict(zip(USAGE_SEGMENT_FIELDS, rec, strict=True))
            tres = row["AllocTRES"]
            node_list = opt(row["NodeList"])
            gpus = gpus_from_tres(tres)
            known = True
            if gpus is None:  # no TRES at all: never allocated (0) unless it ran (then unknown)
                gpus, known = 0, node_list is None
            elif gpus == 0 and node_list is not None and self._facts.gpu_tres_tracked is None:
                known = False  # GPU tracking unknown: "no GPU entry" could hide GPUs
            code, sig = parse_exit_code(row["ExitCode"])
            limit_min = parse_int(row["TimelimitRaw"])
            out.append(
                UsageSegment(
                    job_id=row["JobID"].strip(),
                    user=row["User"].strip(),
                    account=opt(row["Account"]),
                    partition=opt(row["Partition"]),
                    state=str(normalize_slurm_state(row["State"])),
                    gpus=gpus,
                    elapsed_s=parse_int(row["ElapsedRaw"]) or 0,
                    start_time=epoch_to_iso(row["Start"]),
                    end_time=epoch_to_iso(row["End"]),
                    submit_time=epoch_to_iso(row["Submit"]),
                    job_name=text_or_none(row["JobName"]),
                    job_id_raw=row["JobIDRaw"].strip(),
                    qos=opt(row["QOS"]),
                    raw_state=opt(row["State"]),
                    exit_code=code,
                    exit_signal=sig,
                    time_limit_s=limit_min * 60 if limit_min is not None else None,
                    node_list=node_list,
                    gpus_known=known,
                    gpu_types=tuple(sorted(gpu_types_from_tres(tres).items())),
                )
            )
        return out

    def query_account_queue(self, account: str) -> QueryResult:
        """Active (pending/running) jobs of an account, all users (``squeue --account``).

        Under ``PrivateData=jobs`` Slurm shows a regular user only their own jobs; the result
        cannot tell, so callers compare users against ``SchedulerCapabilities.private_data``.
        """
        acct = _validated_name(account, "account")
        self.probe_capabilities()
        observed_at = self._now()
        argv = [
            "squeue",
            "--noheader",
            "--array",
            f"--states={ACTIVE_QUEUE_STATES}",
            f"--account={acct}",
            f"--Format={format_spec(SQUEUE_FIELDS)}",
        ]
        res = self._run(argv, timeout=self.query_timeout_s)
        if res.exit_code != 0:
            raise self._fail("squeue --account", res)
        jobs = [
            self._squeue_observation(dict(zip(SQUEUE_FIELDS, rec, strict=True)), observed_at)
            for rec in split_records(res.stdout, len(SQUEUE_FIELDS))
        ]
        return QueryResult(source="squeue", observed_at=observed_at, jobs=jobs, complete=True)


@dataclass
class _PartitionAcc:
    name: str
    default: bool = False
    state: str | None = None
    max_time_s: int | None = None
    nodes: set[str] = field(default_factory=set)
    gpus: dict[str, dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------


def _parse_when(value: str | None) -> datetime:
    """Caller-supplied ISO timestamp -> aware datetime; anything else is InvalidArgument."""
    try:
        dt = parse_iso(value)
    except ValueError:
        dt = None
    if dt is None:
        raise InvalidArgument(f"invalid ISO timestamp {value!r}")
    return dt


def _check_submit_paths(script_path: str, workdir: str, receipt_path: str | None) -> None:
    """Refuse paths the snippet would mis-handle, before anything reaches the cluster."""
    for label, value in (
        ("script_path", script_path),
        ("workdir", workdir),
        ("receipt_path", receipt_path or ""),
    ):
        if "\n" in value or "\x00" in value:
            raise InvalidArgument(f"{label} contains a newline or NUL byte")
    if not script_path or not workdir:
        raise InvalidArgument("script_path and workdir are required")
    if script_path.startswith("-"):
        # sbatch would parse it as an option (e.g. --wrap=...), not as the script to submit.
        raise InvalidArgument(f"script_path may not start with '-': {script_path!r}")
    if receipt_path is not None and receipt_path and not receipt_path.startswith("/"):
        raise InvalidArgument(f"receipt_path must be an absolute path, got {receipt_path!r}")


def _failure_error(failure: SbatchFailure, details: dict[str, object]) -> TransportError | SubmissionRejected:
    if failure.kind == "rejected":
        return SubmissionRejected(
            f"sbatch refused the job: {failure.message}",
            details={
                **details,
                "reason_code": failure.reason_code,
                "retry_later": failure.reason_code in RETRY_LATER_REASONS,
            },
        )
    return ConnectionLost(
        f"sbatch outcome unknown: {failure.message}",
        details={**details, "reason_code": failure.reason_code},
    )


def _parse_frame(stdout: str) -> tuple[dict[str, str] | None, str]:
    """Find the submit wrapper's frame line; return (fields, the sbatch stdout after it).

    The frame may be preceded on its line by login-shell output that lacked a newline
    (``~/.bashrc`` printing a terminal title), so it is searched for anywhere in a line.
    """
    marker = SUBMIT_FRAME + " "
    lines = stdout.split("\n")
    for i, line in enumerate(lines):
        pos = line.find(marker)
        if pos >= 0:
            fields = parse_key_values(line[pos + len(SUBMIT_FRAME) :])
            return fields, "\n".join(lines[i + 1 :]).strip("\n")
    return None, stdout


def _parse_job_id(stdout: str) -> tuple[str, str | None] | None:
    for line in stdout.splitlines():
        m = _PARSABLE_ID_RE.match(line)
        if m:
            return m.group("id"), m.group("cluster")
    return None


def _version_string(text: str) -> str | None:
    m = _VERSION_RE.search(text or "")
    return m.group(0) if m else (text.strip() or None)


def _mtime_iso(mtime: float | None) -> str | None:
    if mtime is None:
        return None
    return iso(datetime.fromtimestamp(mtime, UTC))


def _validated_ids(job_ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for jid in job_ids:
        j = str(jid).strip()
        if not _JOB_ID_RE.match(j):
            raise InvalidArgument(f"invalid Slurm job id {jid!r}")
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def _validated_name(value: str, what: str) -> str:
    v = value.strip()
    if not _NAME_RE.match(v):
        raise InvalidArgument(f"invalid {what} {value!r}")
    return v


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
