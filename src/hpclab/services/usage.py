"""GPU usage report from Slurm accounting (``usage_report``).

What it reports: *allocated* GPU-hours (not utilization) for me (``scope="self"``) or a lab
account (``scope="account"``) over a window, grouped by user/partition/gpu_type/state/...,
with GPU-hours by state, wasted GPU-hours by cause, median queue wait, walltime accuracy and
interactive hours. No rankings or scores.

Honesty rules
-------------
* GPU count per record = the untyped ``gres/gpu`` TRES; typed entries only when it is absent
  (``gres/gpu=4,gres/gpu:h100=4`` is 4 GPUs). A record whose GPU count cannot be read is counted
  in ``gpu_count_unknown_jobs`` and contributes no hours (never 0-as-fact).
* Requeued jobs contribute every segment (``sacct --duplicates``); running segments are charged
  up to the time accounting was read ("now"); everything is clipped to the window.
* ``PrivateData``: with ``scope="account"`` the report is ``visibility="self_only"`` when other
  users' jobs are (or may be) hidden - then there is NO account total (``totals`` only has a
  ``"self"`` entry) and coverage is incomplete. Accounting that returns only my rows is always
  self_only: sacct obeys slurmdbd.conf's PrivateData, which ``scontrol show config`` cannot show.
* Waste: managed (hpclab) jobs are labelled by their diagnosis code, foreign jobs by terminal
  state; earlier requeue segments by their own state; a managed TIMEOUT that left a finalized
  checkpoint is not waste.
* A source that fails is reported in ``coverage`` (partial windows) or raised (all failed /
  unsupported); an empty window is only "no usage" when accounting answered.

Cost control: sacct windows are at most 31 days per call; whole UTC days older than
``max(accounting_lag, 1 h)`` whose records are all terminal are cached in ``report_cache``
(key per day), so a repeated report only re-reads the live tail.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.api.envelope import Coverage, clamp_limit
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import (
    ErrorCode,
    HpcLabError,
    InvalidArgument,
    LimitExceeded,
    Unsupported,
)
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jdump, jload

log = logging.getLogger("hpclab.usage")

MAX_WINDOW_DAYS = 93
SACCT_MAX_DAYS = 31
FINALIZE_MARGIN_S = 3600.0
CACHE_PREFIX = "usage:v1:"
CACHE_TTL = timedelta(days=30)
DEFAULT_WINDOW = timedelta(days=7)

GroupDim = Literal["user", "account", "partition", "qos", "gpu_type", "state", "diagnosis_class"]
IncludeKey = Literal["now", "waste", "queue_wait", "walltime_accuracy"]
Visibility = Literal["full", "self_only", "unavailable"]
DEFAULT_INCLUDE: tuple[IncludeKey, ...] = ("waste", "queue_wait", "walltime_accuracy")

# Terminal states whose allocated hours did not produce a finished job.
WASTE_STATES = frozenset(
    {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "CANCELLED"}
)
TERMINAL_STATES = WASTE_STATES | {"COMPLETED", "REQUEUED", "SPECIAL_EXIT"}
_INTERACTIVE_NAMES = frozenset({"bash", "sh", "zsh", "fish", "interactive", "salloc"})
_INTERACTIVE_PREFIXES = ("sys/dashboard", "jupyter", "ood", "vscode", "code-server", "rstudio")

_REL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


# ---------------------------------------------------------------------------
# Time windows (shared with services.digest)
# ---------------------------------------------------------------------------


def parse_when(value: str, *, now: datetime, what: str = "time") -> datetime:
    """``7d``/``24h``/``30m``/``2w``/``45s`` (before ``now``), ``now``, ``YYYY-MM-DD`` (UTC
    midnight) or an ISO date-time (naive = UTC). Anything else is InvalidArgument."""
    v = (value or "").strip()
    if not v:
        raise InvalidArgument(f"{what} is empty")
    if v.lower() == "now":
        return now
    m = _REL_RE.match(v)
    if m:
        return now - timedelta(seconds=float(m.group(1)) * _UNITS[m.group(2).lower()])
    if _DATE_RE.match(v):
        try:
            return datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError as exc:
            raise InvalidArgument(f"invalid {what} {value!r}") from exc
    try:
        dt = parse_iso(v)
    except ValueError:
        dt = None
    if dt is None:
        raise InvalidArgument(
            f"invalid {what} {value!r}: use an ISO date/time or a relative duration like 7d, 24h, 30m"
        )
    return dt


def resolve_window(
    since: str | None,
    until: str | None,
    *,
    now: datetime,
    default: timedelta = DEFAULT_WINDOW,
    max_days: int = MAX_WINDOW_DAYS,
) -> tuple[datetime, datetime, list[str]]:
    """Validated ``[start, end]`` (end <= now) plus notes about adjustments."""
    notes: list[str] = []
    end = parse_when(until, now=now, what="until") if until else now
    start = parse_when(since, now=now, what="since") if since else end - default
    if end > now:
        end = now
        notes.append("until is in the future; the window ends now")
    if start >= end:
        raise InvalidArgument("the window is empty: since must be before until (and before now)")
    if end - start > timedelta(days=max_days):
        raise LimitExceeded(
            f"window of {(end - start).days} days exceeds the {max_days}-day maximum",
            details={"max_days": max_days, "hint": "split the report into shorter windows"},
        )
    return start, end, notes


def floor_day(dt: datetime) -> datetime:
    d = dt.astimezone(UTC)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class UsageStats(BaseModel):
    """GPU-hours are allocated hours (GPUs x wall time), rounded to 0.001."""

    model_config = ConfigDict(extra="forbid")

    jobs: int
    gpu_hours: float
    gpu_hours_by_state: dict[str, float] = Field(default_factory=dict)
    wasted_gpu_hours_by_cause: dict[str, float] | None = None  # None when "waste" not included
    median_queue_wait_s: float | None = None  # None: not included or no started job in window
    walltime_accuracy: float | None = None  # median elapsed / time limit of COMPLETED jobs
    interactive_hours: float = 0.0  # GPU-hours of interactive jobs
    managed_jobs: int = 0  # jobs submitted through hpclab (ledger match)
    gpu_count_unknown_jobs: int = 0  # jobs whose GPU count accounting could not state


class UsageRow(UsageStats):
    group: dict[str, str]


class NowSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visibility: Visibility
    running_jobs: int | None = None
    pending_jobs: int | None = None
    running_gpus: int | None = None  # None: unavailable or some GPU count unknown
    pending_gpus: int | None = None
    users_visible: int | None = None
    observed_at: str | None = None
    note: str | None = None


class UsageDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visibility: Visibility
    private_data: str | None
    requeue_segments_included: bool = True
    accounting_lag_s: float
    window_truncated: bool
    jobs_not_managed_by_hpclab: int
    gpu_count_unknown_jobs: int
    sacct_calls: int
    cached_days: int
    queried_windows: list[str]
    failed_windows: list[str]
    # not queried because the request time budget ran out; a re-run continues from the cache
    skipped_windows: list[str] = Field(default_factory=list)


class UsageData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    scope: Literal["self", "account"]
    account: str | None
    user: str
    since: str
    until: str
    group_by: list[str]
    include: list[str]
    visibility: Visibility
    private_data: str | None
    rows: list[UsageRow]
    rows_total: int
    rows_truncated: int
    # "account" only when other users' jobs were visible; otherwise just "self". A self_only
    # report never carries an account total.
    totals: dict[str, UsageStats]
    now: NowSection | None = None
    details: UsageDetails
    notes: list[str] = Field(default_factory=list)
    coverage: Coverage
    observed_at: str
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        key = "account" if "account" in self.totals else "self"
        t = self.totals.get(key)
        hours = f"{t.gpu_hours:.1f} GPU-h over {t.jobs} job(s)" if t else "no data"
        who = f"account {self.account}" if self.scope == "account" else f"user {self.user}"
        vis = "" if self.visibility == "full" else f" [visibility {self.visibility}: your jobs only]"
        return f"{who}, {self.since[:10]}..{self.until[:10]}: {hours}{vis}"


# ---------------------------------------------------------------------------
# Accounting records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seg:
    """One accounting record (a job or one requeue segment) as the report uses it."""

    job_id: str
    job_key: str  # JobIDRaw (unique per array task)
    user: str
    account: str | None
    partition: str | None
    qos: str | None
    state: str
    gpus: int
    gpus_known: bool
    gpu_types: tuple[tuple[str, int], ...]
    submit: datetime | None
    start: datetime | None
    end: datetime | None
    time_limit_s: int | None
    node_list: str | None
    job_name: str | None
    elapsed_s: int
    observed_at: datetime

    @property
    def ran(self) -> bool:
        # A job cancelled while pending has Start == End but no nodes and no elapsed time.
        return self.start is not None and (self.node_list is not None or self.elapsed_s > 0)

    @property
    def run_end(self) -> datetime | None:
        """End of the charged interval: End, or the time accounting was read if still running."""
        if self.start is None:
            return None
        return self.end if self.end is not None else self.observed_at

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("submit", "start", "end", "observed_at"):
            d[k] = iso(d[k]) if d[k] is not None else None
        d["gpu_types"] = [list(x) for x in self.gpu_types]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Seg:
        data = dict(d)
        for k in ("submit", "start", "end", "observed_at"):
            data[k] = parse_iso(data.get(k))
        data["gpu_types"] = tuple((str(t), int(n)) for t, n in data.get("gpu_types") or [])
        return cls(**data)


def seg_from_record(rec: Any, observed_at: datetime) -> Seg:
    """Adapter ``UsageSegment`` (or plain ``UsageRecord``) -> ``Seg``."""
    return Seg(
        job_id=str(rec.job_id),
        job_key=str(getattr(rec, "job_id_raw", "") or rec.job_id),
        user=str(rec.user),
        account=rec.account,
        partition=rec.partition,
        qos=getattr(rec, "qos", None),
        state=str(rec.state).upper(),
        gpus=int(rec.gpus or 0),
        gpus_known=bool(getattr(rec, "gpus_known", True)),
        gpu_types=tuple(getattr(rec, "gpu_types", ()) or ()),
        submit=parse_iso(rec.submit_time),
        start=parse_iso(rec.start_time),
        end=parse_iso(rec.end_time),
        time_limit_s=getattr(rec, "time_limit_s", None),
        node_list=getattr(rec, "node_list", None),
        job_name=rec.job_name,
        elapsed_s=int(rec.elapsed_s or 0),
        observed_at=observed_at,
    )


def _overlaps(seg: Seg, lo: datetime, hi: datetime) -> bool:
    end = seg.run_end
    return seg.start is not None and end is not None and seg.start < hi and end > lo


def _terminal(seg: Seg) -> bool:
    return seg.end is not None and seg.state in TERMINAL_STATES


@dataclass
class _Fetch:
    segs: list[Seg] = field(default_factory=list)
    users_seen: set[str] = field(default_factory=set)
    sacct_calls: int = 0
    cached_days: int = 0
    queried: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[HpcLabError] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)


def _day_key(cluster_id: str, scope_key: str, day: datetime) -> str:
    return f"{CACHE_PREFIX}{cluster_id}:{scope_key}:{day.date().isoformat()}"


def _label(lo: datetime, hi: datetime) -> str:
    return f"{iso(lo)}..{iso(hi)}"


def _fetch(
    ctx: ServiceContext,
    sched: Any,
    *,
    cluster_id: str,
    scope_key: str,
    users: list[str] | None,
    account: str | None,
    start: datetime,
    end: datetime,
    lag_s: float,
    deadline: float | None = None,
) -> _Fetch:
    """Accounting records overlapping ``[start, end]``: finalized whole days from the cache,
    the rest from sacct (<= 31 days per call). Network I/O happens outside any transaction.

    ``deadline`` (``ctx.clock.monotonic()`` value): once passed, no further sacct window is
    started (the first always is); those windows are reported in ``skipped``. Finalized days
    already read are cached, so a re-run continues where this one stopped."""
    now = ctx.clock.now()
    out = _Fetch()
    boundary = floor_day(now - timedelta(seconds=max(lag_s, FINALIZE_MARGIN_S)))
    days: list[datetime] = []
    d = floor_day(start)
    while d < min(end, boundary):
        days.append(d)
        d += timedelta(days=1)

    cached: dict[datetime, list[Seg]] = {}
    if days:
        keys = {_day_key(cluster_id, scope_key, day): day for day in days}
        marks = ",".join("?" for _ in keys)
        rows = ctx.db.all(
            f"SELECT cache_key, payload_json, expires_at FROM report_cache WHERE cache_key IN ({marks})",
            list(keys),
        )
        now_iso = iso(now) or ""
        for r in rows:
            if r["expires_at"] and r["expires_at"] <= now_iso:
                continue
            payload = jload(r["payload_json"], {}) or {}
            try:
                cached[keys[r["cache_key"]]] = [Seg.from_json(x) for x in payload.get("segments", [])]
                out.users_seen.update(payload.get("users", []))
            except (TypeError, ValueError, KeyError):
                log.warning("ignoring unreadable usage cache entry %s", r["cache_key"])
    out.cached_days = len(cached)
    if cached:
        for segs in cached.values():
            out.segs.extend(segs)
        out.coverage.add(
            "report_cache",
            "cached",
            detail=f"{len(cached)} finalized day(s) of accounting from the local cache",
        )

    # windows still to query: contiguous uncached days (<= 31 per call) + the live tail
    windows: list[tuple[datetime, datetime, list[datetime]]] = []
    run: list[datetime] = []
    for day in days:
        if day in cached:
            if run:
                windows.extend(_chunk_days(run))
                run = []
            continue
        run.append(day)
    if run:
        windows.extend(_chunk_days(run))
    if end > boundary:
        windows.append((max(start, boundary), end, []))

    to_cache: list[tuple[str, dict[str, Any]]] = []
    for lo, hi, win_days in windows:
        label = _label(lo, hi)
        if deadline is not None and (out.queried or out.failed) and ctx.clock.monotonic() >= deadline:
            out.skipped.append(label)
            continue
        try:
            records = sched.usage_segments(
                account=account, users=users, since=iso(lo), until=iso(hi) if hi < now else None
            )
        except HpcLabError as exc:
            if exc.code in (ErrorCode.AUTH_REQUIRED, ErrorCode.UNSUPPORTED, ErrorCode.INVALID_ARGUMENT):
                raise  # the whole report depends on it; never a partial "success"
            out.errors.append(exc)
            out.failed.append(label)
            out.coverage.add("sacct", "unavailable", detail=f"{label}: {exc.message}"[:300])
            continue
        observed = ctx.clock.now()
        out.sacct_calls += 1
        out.queried.append(label)
        out.coverage.add("sacct", "ok", observed_at=iso(observed), detail=label)
        segs = [seg_from_record(r, observed) for r in records]
        out.users_seen.update(s.user for s in segs)
        out.segs.extend(segs)
        for day in win_days:
            day_end = day + timedelta(days=1)
            in_day = [s for s in segs if s.ran and _overlaps(s, day, day_end)]
            if all(_terminal(s) for s in in_day):
                users_day = sorted({s.user for s in segs})
                to_cache.append(
                    (
                        _day_key(cluster_id, scope_key, day),
                        {"segments": [s.to_json() for s in in_day], "users": users_day},
                    )
                )
    if to_cache:
        expires = iso(now + CACHE_TTL)
        with ctx.db.tx(source="api") as tx:
            for key, payload in to_cache:
                tx.execute(
                    "INSERT INTO report_cache(cache_key, payload_json, created_at, expires_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(cache_key) DO UPDATE SET payload_json=excluded.payload_json, "
                    "created_at=excluded.created_at, expires_at=excluded.expires_at",
                    (key, jdump(payload), tx.now, expires),
                )
    if windows and not out.queried and not cached and out.errors:
        raise out.errors[0]  # accounting never answered: an error, never an empty report
    return out


def _chunk_days(days: list[datetime]) -> list[tuple[datetime, datetime, list[datetime]]]:
    out: list[tuple[datetime, datetime, list[datetime]]] = []
    for i in range(0, len(days), SACCT_MAX_DAYS):
        part = days[i : i + SACCT_MAX_DAYS]
        out.append((part[0], part[-1] + timedelta(days=1), part))
    return out


def dedupe(segs: Iterable[Seg]) -> list[Seg]:
    """One record per (job, segment start): the same segment appears in several windows."""
    best: dict[tuple[str, str, str], Seg] = {}
    for s in segs:
        key = (s.job_key, iso(s.start) or "-", iso(s.submit) or "-")
        cur = best.get(key)
        if cur is None or _fresher(s, cur):
            best[key] = s
    return list(best.values())


def _fresher(a: Seg, b: Seg) -> bool:
    if (a.end is not None) != (b.end is not None):
        return a.end is not None
    return a.observed_at > b.observed_at


# ---------------------------------------------------------------------------
# Ledger context: which jobs are hpclab's, their diagnosis, checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Managed:
    attempt_id: str
    diagnosis_code: str | None
    result_state: str | None
    checkpoint_ok: bool


def _managed(ctx: ServiceContext, cluster_id: str, segs: Sequence[Seg]) -> dict[str, _Managed]:
    """job_key -> managed attempt, matched on (cluster, job id) + owner + submit time."""
    by_id: dict[str, list[Seg]] = {}
    for s in segs:
        by_id.setdefault(s.job_id, []).append(s)
    ids = sorted(by_id)
    rows: list[dict[str, Any]] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        rows.extend(
            ctx.db.all(
                "SELECT attempt_id, job_id, job_owner, job_submit_time, diagnosis_json, result_state, created_at "
                f"FROM attempts WHERE cluster_id=? AND job_id IN ({marks})",
                [cluster_id, *chunk],
            )
        )
    candidates: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        candidates.setdefault(str(r["job_id"]), []).append(r)
    ckpt: set[str] = set()
    attempt_ids = [r["attempt_id"] for r in rows]
    for i in range(0, len(attempt_ids), 500):
        chunk = attempt_ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        for r in ctx.db.all(
            f"SELECT DISTINCT attempt_id FROM artifacts WHERE attempt_id IN ({marks}) "
            "AND role='checkpoint' AND integrity='finalized'",
            chunk,
        ):
            ckpt.add(r["attempt_id"])
    out: dict[str, _Managed] = {}
    for job_id, job_segs in by_id.items():
        seg = job_segs[0]
        options = []
        for r in candidates.get(job_id, []):
            if r["job_owner"] and r["job_owner"] != seg.user:
                continue
            submitted = parse_iso(r["job_submit_time"])
            if (
                submitted is not None
                and seg.submit is not None
                and abs((submitted - seg.submit).total_seconds()) > 300
            ):
                continue
            options.append(r)
        if not options:
            continue
        r = max(options, key=lambda x: x["created_at"])
        diag = jload(r["diagnosis_json"], None) or {}
        managed = _Managed(
            attempt_id=r["attempt_id"],
            diagnosis_code=diag.get("primary_code") if isinstance(diag, dict) else None,
            result_state=r["result_state"],
            checkpoint_ok=r["attempt_id"] in ckpt,
        )
        for s in job_segs:
            out[s.job_key] = managed
    return out


def _category(code: str | None) -> str:
    if code is None:
        return "none"
    try:
        from hpclab.services.diagnosis import CODE_CATEGORY
    except ImportError:  # pragma: no cover - diagnosis is a wave-2 foundation module
        return "unknown"
    return str(CODE_CATEGORY.get(code, "unknown"))


def waste_cause(seg: Seg, *, final: bool, managed: _Managed | None) -> str | None:
    """Why this record's GPU-hours did not produce a finished job (None = not waste)."""
    if not final:  # an earlier requeue segment: interrupted by preemption/failure/requeue
        return seg.state.lower() if seg.state in WASTE_STATES else "requeued"
    if seg.end is None or seg.state not in TERMINAL_STATES:
        return None  # still running
    if managed is not None:
        if seg.state == "COMPLETED":
            if managed.result_state in ("invalid", "incomplete"):
                return managed.diagnosis_code or "success_no_outputs"
            return None
        if seg.state == "TIMEOUT" and managed.checkpoint_ok:
            return None  # stopped at the walltime with a finalized checkpoint: resumable progress
        return managed.diagnosis_code or seg.state.lower()
    return seg.state.lower() if seg.state in WASTE_STATES else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class _Acc:
    jobs: set[str] = field(default_factory=set)
    gpu_h: float = 0.0
    by_state: dict[str, float] = field(default_factory=dict)
    waste: dict[str, float] = field(default_factory=dict)
    waits: list[float] = field(default_factory=list)
    walltime: list[float] = field(default_factory=list)
    interactive: float = 0.0
    managed: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)

    def stats(self, include: set[str]) -> UsageStats:
        return UsageStats(
            jobs=len(self.jobs),
            gpu_hours=round(self.gpu_h, 3),
            gpu_hours_by_state={k: round(v, 3) for k, v in sorted(self.by_state.items())},
            wasted_gpu_hours_by_cause=(
                {k: round(v, 3) for k, v in sorted(self.waste.items())} if "waste" in include else None
            ),
            median_queue_wait_s=(
                round(median(self.waits), 1) if "queue_wait" in include and self.waits else None
            ),
            walltime_accuracy=(
                round(median(self.walltime), 3) if "walltime_accuracy" in include and self.walltime else None
            ),
            interactive_hours=round(self.interactive, 3),
            managed_jobs=len(self.managed),
            gpu_count_unknown_jobs=len(self.unknown),
        )


def _gpu_type(seg: Seg) -> str:
    if not seg.gpu_types:
        return "untyped" if seg.gpus else "none"
    types = {t for t, _ in seg.gpu_types}
    return next(iter(types)) if len(types) == 1 else "mixed"


def _is_interactive(seg: Seg, interactive_partitions: set[str]) -> bool:
    if seg.partition and seg.partition in interactive_partitions:
        return True
    name = (seg.job_name or "").strip().lower()
    return name in _INTERACTIVE_NAMES or name.startswith(_INTERACTIVE_PREFIXES)


def aggregate(
    segs: Sequence[Seg],
    *,
    start: datetime,
    end: datetime,
    dims: Sequence[str],
    include: set[str],
    managed: dict[str, _Managed],
    interactive_partitions: set[str],
) -> tuple[dict[tuple[str, ...], _Acc], _Acc]:
    """Group accumulators + the overall accumulator over records clipped to ``[start, end]``."""
    by_job: dict[str, list[Seg]] = {}
    for s in segs:
        if s.ran:
            by_job.setdefault(s.job_key, []).append(s)
    groups: dict[tuple[str, ...], _Acc] = {}
    total = _Acc()
    for job_key, job_segs in by_job.items():
        job_segs.sort(key=lambda s: s.start or s.observed_at)
        final = job_segs[-1]
        m = managed.get(job_key)
        for i, seg in enumerate(job_segs):
            lo = max(seg.start or start, start)
            run_end = seg.run_end
            if run_end is None:
                continue
            hi = min(run_end, end)
            secs = (hi - lo).total_seconds()
            if secs <= 0 or (seg.gpus_known and seg.gpus <= 0):
                continue  # outside the window, or a CPU-only record (not GPU usage)
            key = tuple(_dim(seg, d, m) for d in dims)
            accs = (groups.setdefault(key, _Acc()), total)
            for acc in accs:
                acc.jobs.add(job_key)
                if m is not None:
                    acc.managed.add(job_key)
            if not seg.gpus_known:
                for acc in accs:
                    acc.unknown.add(job_key)
                continue
            gh = seg.gpus * secs / 3600.0
            # A REQUEUED record is never the job's last run, even when the window cuts off the rerun.
            is_final = seg is final and seg.state != "REQUEUED"
            cause = waste_cause(seg, final=is_final, managed=m) if gh > 0 else None
            for acc in accs:
                acc.gpu_h += gh
                acc.by_state[seg.state] = acc.by_state.get(seg.state, 0.0) + gh
                if cause is not None:
                    acc.waste[cause] = acc.waste.get(cause, 0.0) + gh
                if _is_interactive(seg, interactive_partitions):
                    acc.interactive += gh
            if i == 0 and seg.start is not None and seg.submit is not None and start <= seg.start <= end:
                for acc in accs:
                    acc.waits.append(max(0.0, (seg.start - seg.submit).total_seconds()))
            if (
                seg is final
                and seg.state == "COMPLETED"
                and seg.end is not None
                and seg.start is not None
                and seg.time_limit_s
                and start <= seg.end <= end
            ):
                ratio = (seg.end - seg.start).total_seconds() / seg.time_limit_s
                for acc in accs:
                    acc.walltime.append(ratio)
    return groups, total


def _dim(seg: Seg, dim: str, managed: _Managed | None) -> str:
    if dim == "user":
        return seg.user
    if dim == "account":
        return seg.account or "unknown"
    if dim == "partition":
        return seg.partition or "unknown"
    if dim == "qos":
        return seg.qos or "unknown"
    if dim == "gpu_type":
        return _gpu_type(seg)
    if dim == "state":
        return seg.state
    if dim == "diagnosis_class":
        return _category(managed.diagnosis_code) if managed is not None else "foreign"
    raise InvalidArgument(f"unknown group_by dimension {dim!r}")


def visibility_for(
    *,
    scope: str,
    me: str,
    users_seen: set[str],
    private_data: str | None,
    source: Literal["sacct", "squeue"] = "sacct",
) -> tuple[Visibility, str | None]:
    """(visibility, note) for what Slurm showed us.

    ``private_data`` is slurm.conf's PrivateData (``scontrol show config``). slurmctld enforces
    it, so for ``squeue`` a known "nothing hidden" makes a me-only queue the full picture. sacct
    is answered by slurmdbd, which enforces its OWN PrivateData (slurmdbd.conf, not visible
    here): for accounting, "only my rows" is always ``self_only`` (spec: no account total)."""
    if scope == "self":
        return "full", None
    from hpclab.adapters.slurm import private_data_flags

    flags = private_data_flags(private_data)
    others = users_seen - {me}
    if others:
        note = (
            "PrivateData hides jobs, yet other users' jobs are visible (operator/coordinator rights)"
            if flags and "jobs" in flags
            else None
        )
        return "full", note
    if flags is not None and "jobs" in flags:
        return "self_only", (
            f"PrivateData={private_data}: Slurm shows you only your own jobs, so these numbers are NOT "
            "account totals"
        )
    if flags is None:
        return "self_only", (
            "only your own jobs were returned and PrivateData could not be read: other users' jobs may be "
            "hidden, so these numbers are NOT account totals"
        )
    if source == "squeue":
        return "full", "PrivateData hides nothing; only your jobs are in this account's queue"
    return "self_only", (
        "accounting returned no other user's jobs for this account: either nobody else ran in the "
        "window or slurmdbd's PrivateData hides them (it cannot be read from here), so these numbers "
        "are NOT account totals"
    )


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def usage_report(
    ctx: ServiceContext,
    *,
    cluster_id: str,
    scope: Literal["self", "account"] = "self",
    account: str | None = None,
    since: str | None = None,
    until: str | None = None,
    group_by: list[GroupDim] | None = None,
    include: list[IncludeKey] | None = None,
    limit: int | None = None,
) -> UsageData:
    """Allocated GPU-hours from Slurm accounting for me or a lab account over a window
    (default: the last 7 days; ``since``/``until`` take ISO dates/times or ``7d``/``24h``/``30m``;
    at most 93 days). Grouped by ``group_by`` (default user for accounts, partition for self),
    with waste by cause, queue wait and walltime accuracy. ``visibility`` says whether other
    users' jobs were visible; a ``self_only`` report has no account total."""
    rc = ctx.clusters.get(cluster_id)
    now = ctx.clock.now()
    # Stop starting sacct windows after 85% of the request deadline (like cluster_inspect).
    budget_deadline = ctx.clock.monotonic() + max(1.0, float(ctx.settings.request_deadline_s) * 0.85)
    start, end, notes = resolve_window(since, until, now=now)
    inc: set[str] = set(include) if include is not None else set(DEFAULT_INCLUDE)
    dims: list[str] = (
        list(dict.fromkeys(group_by)) if group_by else (["user"] if scope == "account" else ["partition"])
    )
    lim = clamp_limit(limit)
    acct = account or (rc.overlay.account if scope == "account" else None)
    if scope == "account" and not acct:
        raise InvalidArgument(
            "scope='account' needs an account (none given and none in the cluster overlay)",
            details={"hint": f"hpclab cluster add {cluster_id} --account <acct>, or pass account="},
        )
    transport = ctx.clusters.transport(cluster_id)
    sched = ctx.clusters.scheduler(cluster_id)
    me = transport.remote_user()
    caps = sched.probe_capabilities()
    if not hasattr(sched, "usage_segments"):
        raise Unsupported(f"the scheduler adapter of {cluster_id} cannot read usage records")
    lag_s = float(rc.profile.policy.accounting_lag_s)
    # Who asks is part of the key: what accounting shows depends on it (PrivateData, coordinators).
    scope_key = (
        f"account={acct},as={me}"
        if scope == "account"
        else f"user={me}" + (f",account={acct}" if acct else "")
    )

    fetched = _fetch(
        ctx,
        sched,
        cluster_id=cluster_id,
        scope_key=scope_key,
        users=None if scope == "account" else [me],
        account=acct,
        start=start,
        end=end,
        lag_s=lag_s,
        deadline=budget_deadline,
    )
    segs = dedupe(fetched.segs)
    visibility, vis_note = visibility_for(
        scope=scope, me=me, users_seen=fetched.users_seen, private_data=caps.private_data
    )
    if visibility == "self_only":
        segs = [s for s in segs if s.user == me]
    managed = _managed(ctx, cluster_id, segs)
    interactive_parts = {name for name, rule in rc.profile.partitions.items() if rule.interactive_only}
    groups, total = aggregate(
        segs,
        start=start,
        end=end,
        dims=dims,
        include=inc,
        managed=managed,
        interactive_partitions=interactive_parts,
    )

    rows = [
        UsageRow(group=dict(zip(dims, key, strict=True)), **acc.stats(inc).model_dump())
        for key, acc in sorted(groups.items())
    ]
    total_stats = total.stats(inc)
    totals = {"account" if scope == "account" and visibility == "full" else "self": total_stats}

    coverage = fetched.coverage
    coverage.add("ledger", "ok", observed_at=iso(now), detail="hpclab attempts, diagnoses, checkpoints")
    if fetched.failed:
        coverage.missing.extend(f"sacct window {w}" for w in fetched.failed)
    if fetched.skipped:
        coverage.complete = False
        coverage.add("sacct", "skipped", detail=f"{len(fetched.skipped)} window(s) not read: time budget")
        coverage.missing.extend(f"sacct window {w} (not read: time budget)" for w in fetched.skipped)
    if visibility == "self_only":
        coverage.complete = False
        coverage.missing.append("other users' jobs (hidden by PrivateData or not confirmable)")
    all_notes = list(notes)
    if vis_note:
        all_notes.append(vis_note)
    all_notes.append("GPU-hours are allocated hours (GPUs x wall time), not measured utilization")
    warnings: list[str] = []
    if fetched.failed:
        warnings.append(
            f"{len(fetched.failed)} accounting window(s) could not be read; totals cover the rest only"
        )
    if fetched.skipped:
        warnings.append(
            f"{len(fetched.skipped)} accounting window(s) were not read within the request time budget; "
            "totals cover the rest only - re-run to continue (finished days are cached)"
        )
    if total_stats.gpu_count_unknown_jobs:
        warnings.append(
            f"{total_stats.gpu_count_unknown_jobs} job(s) have an unknown GPU count and add no GPU-hours"
        )
        # The GPU-hour totals are then a lower bound: never presented as complete.
        coverage.complete = False
        coverage.missing.append(
            f"GPU count of {total_stats.gpu_count_unknown_jobs} job(s) (accounting has no GPU TRES for them)"
        )

    now_section: NowSection | None = None
    if "now" in inc:
        now_section = _now_section(sched, scope=scope, account=acct, me=me, private_data=caps.private_data)
        coverage.add(
            "squeue",
            "ok" if now_section.visibility != "unavailable" else "unavailable",
            observed_at=now_section.observed_at,
            detail=now_section.note,
        )

    foreign = len({s.job_key for s in segs if s.ran and s.job_key not in managed})
    details = UsageDetails(
        visibility=visibility,
        private_data=caps.private_data,
        accounting_lag_s=lag_s,
        window_truncated=bool(fetched.failed or fetched.skipped),
        jobs_not_managed_by_hpclab=foreign,
        gpu_count_unknown_jobs=total_stats.gpu_count_unknown_jobs,
        sacct_calls=fetched.sacct_calls,
        cached_days=fetched.cached_days,
        queried_windows=fetched.queried,
        failed_windows=fetched.failed,
        skipped_windows=fetched.skipped,
    )
    return UsageData(
        cluster_id=cluster_id,
        scope=scope,
        account=acct,
        user=me,
        since=iso(start) or "",
        until=iso(end) or "",
        group_by=dims,
        include=sorted(inc),
        visibility=visibility,
        private_data=caps.private_data,
        rows=rows[:lim],
        rows_total=len(rows),
        rows_truncated=max(0, len(rows) - lim),
        totals=totals,
        now=now_section,
        details=details,
        notes=all_notes,
        coverage=coverage,
        observed_at=iso(now) or "",
        warnings=warnings,
    )


def _now_section(
    sched: Any, *, scope: str, account: str | None, me: str, private_data: str | None
) -> NowSection:
    try:
        if scope == "account":
            query = getattr(sched, "query_account_queue", None)
            if query is None or account is None:
                return NowSection(visibility="unavailable", note="account queue view not supported")
            res = query(account)
        else:
            res = sched.query_queue(user=me)
    except HpcLabError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return NowSection(visibility="unavailable", note=f"squeue failed: {exc.message}"[:300])
    running = [j for j in res.jobs if str(j.state) in ("RUNNING", "COMPLETING", "CONFIGURING")]
    pending = [j for j in res.jobs if str(j.state) == "PENDING"]

    def gpus(jobs: list[Any]) -> int | None:
        if any(j.alloc_gpus is None for j in jobs):
            return None
        return sum(int(j.alloc_gpus) for j in jobs)

    users = {j.user for j in res.jobs if j.user}
    vis, note = visibility_for(
        scope=scope, me=me, users_seen=users, private_data=private_data, source="squeue"
    )
    return NowSection(
        visibility=vis,
        running_jobs=len(running),
        pending_jobs=len(pending),
        running_gpus=gpus(running),
        pending_gpus=gpus(pending),
        users_visible=len(users),
        observed_at=res.observed_at,
        note=note,
    )


# ---------------------------------------------------------------------------
# Controller wiring
# ---------------------------------------------------------------------------


def outbox_handlers() -> dict[str, Any]:
    return {}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    return [PeriodicTask("usage.cache_gc", 6 * 3600.0, gc_cache, jitter_s=60.0, run_at_start=False)]


def gc_cache(ctx: ServiceContext) -> None:
    """Drop expired usage cache entries (only this module's keys)."""
    with ctx.db.tx(source="worker") as tx:
        tx.execute(
            "DELETE FROM report_cache WHERE cache_key LIKE ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (f"{CACHE_PREFIX}%", tx.now),
        )
