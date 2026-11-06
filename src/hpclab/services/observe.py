"""Observation of submitted jobs: batched polling, the observation state machine, finalization.

Who calls what
--------------
* ``poll_cluster`` - one batched, rate-limited poll of a cluster: ONE ``squeue --user=<me>`` for
  every observed attempt (plus the user's other jobs, summarised as *foreign jobs*), then ONE
  ``sacct`` for the attempts that vanished from the queue. The rate limit lives in ``poll_state``
  (per cluster + source), so the periodic poller, ``run_inspect(refresh=True)`` and every MCP/CLI
  client share it. Failures back off exponentially; ``auth_required`` pauses polling (and notifies)
  until the user re-authenticates.
* ``apply_observation`` - the state machine (design §6.3, spec §4.5/§4.6): out-of-order
  observations are ignored, a TERMINAL attempt never goes back to running unless the scheduler's
  restart count increased (requeue -> a new ``attempt_segments`` row), identity (owner, token in
  job name/comment, submit time) is checked before anything is applied.
* ``observe.finalize`` (outbox) - after an attempt became TERMINAL: reads ``status.json`` /
  ``env.json`` (app state), accounting (exit code, elapsed), settles the budget, stores a
  diagnosis for failures, records node observations, and enqueues ``artifacts.index``,
  ``vault.capture``, ``workflow.dispatch`` and a notification.

Honesty rules
-------------
* "not in squeue" is never "finished": sacct is asked; no row yet -> ``observation_gap=1`` and the
  state is left alone (spec §4.6).
* Unknown stays unknown: no accounting -> the budget settles as *unknown* (``None``), never 0.
* Network I/O never happens inside ``db.tx()``: read -> tx(apply).

Versioning: the attempt's ``version`` changes only with what a user decision depends on -
phase, scheduler state, app/result state, cancel state, job identity (owner, submit time, array
task) and requeue (restart count / segments); each of those writes an event. Observation-only
refreshes (``last_observed_at``, elapsed, time left, pending reason, node list, cached
progress/health views, the observation-gap flag) are written with ``bump_version=False`` (the
gap still gets its events), so a user's ``expected_version`` for ``run_cancel`` is not
invalidated by every poll.
"""

from __future__ import annotations

import importlib
import logging
import random
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.adapters.base import (
    AuthRequiredError,
    ConnectionLost,
    Scheduler,
    StatInfo,
    Transport,
    TransportError,
    TransportTimeout,
)
from hpclab.domain.clock import iso, parse_iso, parse_slurm_duration
from hpclab.domain.errors import ErrorCode, HpcLabError, NotFound
from hpclab.domain.models import JobObservation, ResourceSpec
from hpclab.domain.states import (
    SCHEDULER_TERMINAL,
    AppState,
    AttemptPhase,
    CancelState,
    ResultState,
    RunStatus,
    SchedulerState,
)
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services.context import ServiceContext
from hpclab.services.diagnosis import AttemptFacts, Diagnosis, diagnose
from hpclab.services.health import Health, health_from
from hpclab.services.jobscript import (
    DUPLICATE_MARKER,
    EXIT_DUPLICATE_JOB,
    AttemptLayout,
    token_from_comment,
    token_from_job_name,
)
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.services.progress import Progress, estimate_progress
from hpclab.storage.db import Row, Tx, jdump, jload

log = logging.getLogger("hpclab.observe")

__all__ = [
    "PollResult",
    "RefreshResult",
    "apply_observation",
    "attempt_facts",
    "compute_live_views",
    "finalize_attempt",
    "layout_for",
    "mark_observation_gap",
    "outbox_handlers",
    "periodic_tasks",
    "pick_observation",
    "poll_all",
    "poll_cluster",
    "read_log_tails",
    "refresh_attempts",
    "refresh_run_status",
    "resettle_attempt",
    "run_status_for",
    "stored_accounting",
    "sub_dict",
    "wait_for_background",
]

OBSERVED_PHASES = (AttemptPhase.JOB_IDENTIFIED.value, AttemptPhase.OBSERVING.value)
# Job identity a live cancel is verified against: learning one of them is versioned (a user's
# decision may depend on which job the attempt is), unlike freshness fields.
IDENTITY_COLUMNS = ("job_id", "job_owner", "job_submit_time", "array_task_id")
# Terminal states Slurm may requeue from (preemption with --requeue, JobRequeue=1 node failure):
# such attempts stay on the poll list for a while so a requeue opens a new segment.
REQUEUE_STATES = (
    SchedulerState.PREEMPTED.value,
    SchedulerState.NODE_FAIL.value,
    SchedulerState.BOOT_FAIL.value,
)
REQUEUE_WATCH_S = 900.0
SUBMIT_TIME_TOLERANCE_S = 600.0  # job ids are reused only after wrap-around/DB resets (days apart)
MAX_BACKOFF_S = 600.0
AUTH_PAUSE_S = 120.0
POLL_TICK_S = 5.0  # periodic tick; the per-cluster minimum interval is enforced by poll_state
LIVE_VIEW_TTL_S = 300.0
BACKGROUND_READ_TIMEOUT_S = 15.0
MAX_LIVE_VIEWS_PER_POLL = 8
METRICS_TAIL_BYTES = 16 * 1024
TELEMETRY_TAIL_BYTES = 8 * 1024
HEARTBEAT_TAIL_BYTES = 512
LOG_TAIL_BYTES = 64 * 1024
STATUS_MAX_BYTES = 64 * 1024
ENV_MAX_BYTES = 512 * 1024
FINALIZE_MAX_TRIES = 30
ACCOUNTING_GRACE_S = 60.0
# sacct *failing* (slurmdbd restarting, "Socket timed out") is not "no record": finalize waits this
# long before it finalizes provisionally; ``observe.settle`` then keeps asking for RESETTLE_HORIZON_S.
ACCOUNTING_OUTAGE_WAIT_S = 1800.0
UNREACHABLE_WAIT_S = 24 * 3600.0  # wrapper files while the login node does not answer at all
RESETTLE_HORIZON_S = 7 * 24 * 3600.0
RESETTLE_FIRST_DELAY_S = 300.0
RESETTLE_MAX_DELAY_S = 3600.0
MAX_ACCOUNTING_BATCH = 200  # terminal attempts waiting for accounting, asked in one sacct call
ACCOUNTING_CHECK_TTL_S = 24 * 3600.0
ACCOUNTING_CHECK_PREFIX = "acct:"  # observations cache: the last accounting check of an attempt
MAX_REQUEUE_CHECKS_PER_POLL = 4  # sacct --duplicates calls for requeues the controller did not see
REQUEUE_RECHECK_S = 900.0  # an unverifiable requeue chain is asked for again after this long
REQUEUE_CHECK_PREFIX = "requeue_check:"  # observations cache: the last failed per-segment check
# A stored poll/observation time this far in the future means the wall clock stepped back.
CLOCK_SKEW_S = 30.0
KIND_FINALIZE = "observe.finalize"
KIND_RESETTLE = "observe.settle"
NODE_OBS_TTL_S = 72 * 3600.0
FOREIGN_JOBS_KEY = "queue:foreign_jobs"
FOREIGN_JOBS_TTL_S = 3600.0
MAX_RESULT_ITEMS = 100

_REFRESH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hpclab-refresh")
_INFLIGHT: set[Future[Any]] = set()
_INFLIGHT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class PollResult(BaseModel):
    """Outcome of one ``poll_cluster`` call (bounded lists)."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    polled: bool  # a scheduler query was attempted
    ok: bool = False
    skipped_reason: Literal["too_soon", "backoff", "no_active_attempts"] | None = None
    next_poll_after: str | None = None
    observed_at: str | None = None
    attempts_polled: int = 0
    changes: list[str] = Field(default_factory=list)
    terminal: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    # left squeue while the sacct check was deferred by its rate limit: their state is not known
    deferred: list[str] = Field(default_factory=list)
    sacct: Literal["not_needed", "ok", "deferred", "unavailable", "unsupported", "error"] = "not_needed"
    error_code: str | None = None
    error: str | None = None
    auth_required: bool = False
    foreign_jobs: int | None = None
    notes: list[str] = Field(default_factory=list)


class RefreshResult(BaseModel):
    """Outcome of a bounded on-demand refresh (``run_inspect(refresh=True)``)."""

    model_config = ConfigDict(extra="forbid")

    attempt_ids: list[str]
    refreshed: list[str] = Field(default_factory=list)
    stale: bool = False
    notes: list[str] = Field(default_factory=list)
    observed_at: str | None = None


# ---------------------------------------------------------------------------
# Small helpers shared with runs.py / logs.py
# ---------------------------------------------------------------------------


def layout_for(attempt: Mapping[str, Any]) -> AttemptLayout:
    """Remote paths of an attempt from its ledger row (``remote_dir`` is the attempt dir)."""
    d = str(attempt["remote_dir"]).rstrip("/")
    return AttemptLayout(
        dir=d,
        script=f"{d}/job.sbatch",
        stdout=f"{d}/logs/stdout.log",
        stderr=f"{d}/logs/stderr.log",
        status_json=f"{d}/status.json",
        env_json=f"{d}/env.json",
        outputs_manifest=f"{d}/outputs.json",
        receipt=f"{d}/submit.receipt",
        work=f"{d}/work",
        output_dir=f"{d}/outputs",
        checkpoint_dir=f"{d}/checkpoints",
        token_file=f"{d}/hpclab-plan.json",
    )


def plan_dict(reader: Any, exec_plan_hash: str | None) -> dict[str, Any]:
    if not exec_plan_hash:
        return {}
    row = reader.one("SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (exec_plan_hash,))
    doc = jload(row["plan_json"], {}) if row else {}
    return doc if isinstance(doc, dict) else {}


def sub_dict(doc: Any, key: str) -> dict[str, Any]:
    """``doc[key]`` when it is a mapping, else ``{}`` (defensive reads of stored JSON)."""
    value = doc.get(key) if isinstance(doc, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_module(name: str, *, quiet: bool = False) -> ModuleType | None:
    """Lazy import of another wave-3 service module (may not exist in every deployment/test)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        (log.debug if quiet else log.warning)("optional module %s is not available", name)
        return None


def _optional_function(module: str, name: str) -> Any:
    """``module.name`` if both exist (a partially deployed area must not break observation)."""
    mod = _optional_module(module, quiet=True)
    fn = getattr(mod, name, None) if mod is not None else None
    if mod is not None and fn is None:
        log.warning("%s.%s is not available", module, name)
    return fn


def _state(value: str | None) -> SchedulerState:
    try:
        return SchedulerState(value or "UNKNOWN")
    except ValueError:
        return SchedulerState.UNKNOWN


def _now_iso(ctx: ServiceContext) -> str:
    out = iso(ctx.clock.now())
    assert out is not None
    return out


def _bounded(items: list[str], n: int = MAX_RESULT_ITEMS) -> list[str]:
    return items[:n]


# ---------------------------------------------------------------------------
# Run status (derived summary of the latest attempt)
# ---------------------------------------------------------------------------


def run_status_for(attempt: Mapping[str, Any] | None) -> RunStatus:
    """``RunStatus`` of a Run given its latest attempt row (None = never dispatched)."""
    if attempt is None:
        return RunStatus.PREPARED
    try:
        phase = AttemptPhase(attempt["phase"])
    except ValueError:
        return RunStatus.UNKNOWN
    if phase in (AttemptPhase.STAGING, AttemptPhase.SUBMISSION_IN_FLIGHT):
        return RunStatus.SUBMITTING
    if phase in (AttemptPhase.SUBMISSION_UNCERTAIN, AttemptPhase.DUPLICATE_DETECTED):
        return RunStatus.UNKNOWN
    if phase == AttemptPhase.SUBMISSION_REJECTED:
        return RunStatus.REJECTED
    if phase in (AttemptPhase.CANCELLED_BEFORE_SUBMIT, AttemptPhase.ABANDONED):
        return RunStatus.CANCELLED
    sched = _state(attempt.get("scheduler_state"))
    if phase in (AttemptPhase.JOB_IDENTIFIED, AttemptPhase.OBSERVING):
        if attempt.get("observation_gap"):
            return RunStatus.UNKNOWN
        if sched in (SchedulerState.PENDING, SchedulerState.REQUEUED, SchedulerState.UNKNOWN):
            return RunStatus.QUEUED
        return RunStatus.RUNNING
    if sched == SchedulerState.COMPLETED:
        if attempt.get("result_state") == ResultState.VALID.value:
            return RunStatus.SUCCEEDED
        return RunStatus.COMPLETED_UNVERIFIED
    if sched == SchedulerState.CANCELLED:
        return RunStatus.CANCELLED
    return RunStatus.FAILED


def refresh_run_status(tx: Tx, run_id: str) -> str:
    """Recompute ``runs.status`` from the Run's latest attempt; event on change. Returns it.

    Public: services that change an attempt's phase/result_state (submission, artifacts) call it
    in the same transaction.
    """
    run = tx.one("SELECT run_id, status FROM runs WHERE run_id=?", (run_id,))
    if run is None:
        raise NotFound(f"run {run_id} not found")
    latest = tx.one("SELECT * FROM attempts WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,))
    new = run_status_for(latest).value
    if new != run["status"]:
        version = tx.update("runs", "run_id", run_id, {"status": new})
        tx.event(
            "run",
            run_id,
            "run.status_changed",
            {"from": run["status"], "to": new, "attempt_id": latest["attempt_id"] if latest else None},
            entity_version=version,
            source="observer",
        )
    return new


# ---------------------------------------------------------------------------
# The observation state machine
# ---------------------------------------------------------------------------


def identity_mismatch(attempt: Mapping[str, Any], obs: JobObservation) -> str | None:
    """Why ``obs`` cannot be this attempt's job (None when it can). Unknown fields never mismatch."""
    return _identity_problem(attempt, obs, check_submit=True)


def _identity_problem(attempt: Mapping[str, Any], obs: JobObservation, *, check_submit: bool) -> str | None:
    owner = attempt.get("job_owner")
    if owner and obs.user and obs.user != owner:
        return f"owner {obs.user!r} is not {owner!r}"
    token = str(attempt.get("submit_token") or "")
    if obs.name:
        found = token_from_job_name(obs.name)
        if found is None or found != token[:8]:
            return f"job name {obs.name!r} does not carry this attempt's token"
    if obs.comment:
        full = token_from_comment(obs.comment)
        if full is not None and full != token:
            return "job comment carries another hpclab token"
    if not check_submit:
        return None
    restart = int(attempt.get("restart_count") or 0)
    if attempt.get("job_submit_time") and obs.submit_time and restart == 0 and (obs.restart_count or 0) == 0:
        a_t, o_t = parse_iso(attempt["job_submit_time"]), parse_iso(obs.submit_time)
        if a_t and o_t and abs((o_t - a_t).total_seconds()) > SUBMIT_TIME_TOLERANCE_S:
            return f"submit time {obs.submit_time} differs from {attempt['job_submit_time']}"
    return None


def later_submit_only(attempt: Mapping[str, Any], obs: JobObservation) -> bool:
    """True when the only thing that makes ``obs`` look like another job is a *later* submit time
    with no restart count known on either side: what a requeue the controller never saw looks like
    in accounting (Slurm stamps a new SubmitTime on every requeue; sacct reports no restart count).
    Such a record is only accepted once the per-segment accounting proves the chain
    (:func:`requeue_chain`)."""
    if int(attempt.get("restart_count") or 0) or (obs.restart_count or 0):
        return False
    if _identity_problem(attempt, obs, check_submit=False) is not None:
        return False
    a_t, o_t = parse_iso(attempt.get("job_submit_time")), parse_iso(obs.submit_time)
    return a_t is not None and o_t is not None and (o_t - a_t).total_seconds() > SUBMIT_TIME_TOLERANCE_S


def requeue_chain(
    attempt: Mapping[str, Any], records: Sequence[JobObservation]
) -> list[JobObservation] | None:
    """This attempt's execution segments among ``sacct --duplicates`` records (oldest first,
    ``restart_count`` renumbered 0..n), or None when they do not provably belong to it.

    Job ids are reused, so the chain starts at the record whose submit time is the one recorded
    for the attempt (older records of another job are skipped) and continues while the records
    carry the attempt's owner and token and their submit times do not go back (Slurm restamps
    SubmitTime on every requeue; it never moves it backwards)."""
    rows = [
        r
        for r in records
        if r.job_id == attempt.get("job_id")
        and (r.array_task_id or None) == (attempt.get("array_task_id") or None)
    ]
    recorded = parse_iso(attempt.get("job_submit_time"))
    start = 0
    if recorded is not None:
        found = next(
            (
                i
                for i, r in enumerate(rows)
                if (t := parse_iso(r.submit_time)) is not None
                and abs((t - recorded).total_seconds()) <= SUBMIT_TIME_TOLERANCE_S
            ),
            None,
        )
        if found is None:
            return None
        start = found
    chain: list[JobObservation] = []
    previous: datetime | None = None
    for r in rows[start:]:
        if _identity_problem(attempt, r, check_submit=False) is not None:
            break
        submitted = parse_iso(r.submit_time)
        if submitted is not None and previous is not None and submitted < previous:
            break
        previous = submitted or previous
        chain.append(r)
    if not chain:
        return None
    return [r.model_copy(update={"restart_count": i}) for i, r in enumerate(chain)]


def _same_time(a: str | None, b: str | None, tolerance_s: float = 2.0) -> bool:
    ta, tb = parse_iso(a), parse_iso(b)
    return ta is not None and tb is not None and abs((ta - tb).total_seconds()) <= tolerance_s


def _fetch_requeue_chain(
    scheduler: Any, attempt: Mapping[str, Any], *, raise_auth: bool
) -> tuple[list[JobObservation] | None, str | None]:
    """(verified segment chain | None, why not) from ``accounting_segments`` (sacct --duplicates)."""
    fetch = getattr(scheduler, "accounting_segments", None)
    if fetch is None:
        return None, "per-segment accounting (sacct --duplicates) is not available"
    try:
        records = fetch(str(attempt["job_id"]))
    except HpcLabError as exc:
        if raise_auth and exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return None, f"per-segment accounting unavailable: {exc.message[:200]}"
    chain = requeue_chain(attempt, records)
    if chain is None:
        return (
            None,
            "per-segment accounting (sacct --duplicates) does not chain to this attempt's submit time",
        )
    return chain, None


def _record_chain_segments(
    tx: Tx, attempt_id: str, job_id: str | None, chain: Sequence[JobObservation], observed_at: str
) -> None:
    """Accounting's start/end/state of requeue segments (authoritative: overrides observed values)."""
    for seg in chain:
        _upsert_segment(
            tx,
            attempt_id,
            int(seg.restart_count or 0),
            job_id=job_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            state=seg.state.value if seg.state != SchedulerState.UNKNOWN else None,
            node_list=seg.node_list,
            observed_at=observed_at,
        )


def _segments_complete(reader: Any, attempt_id: str, restart: int) -> bool:
    """Every segment before ``restart`` has a start and an end time in the ledger."""
    if restart <= 0:
        return True
    rows = reader.all(
        "SELECT start_time, end_time FROM attempt_segments WHERE attempt_id=? AND segment < ?",
        (attempt_id, restart),
    )
    return len(rows) >= restart and all(r["start_time"] and r["end_time"] for r in rows)


def _obs_json(obs: JobObservation) -> str:
    data = obs.model_dump(mode="json", exclude_none=True)
    for key in ("submit_line", "comment", "name", "work_dir", "reason"):
        if isinstance(data.get(key), str) and len(data[key]) > 512:
            data[key] = data[key][:512] + "..."
    return jdump(data)


def _upsert_segment(
    tx: Tx,
    attempt_id: str,
    segment: int,
    *,
    job_id: str | None,
    start_time: str | None,
    end_time: str | None,
    state: str | None,
    node_list: str | None,
    observed_at: str,
) -> None:
    tx.execute(
        "INSERT INTO attempt_segments (attempt_id, segment, job_id, start_time, end_time, state, node_list,"
        " observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(attempt_id, segment) DO UPDATE SET"
        " job_id=COALESCE(excluded.job_id, job_id), start_time=COALESCE(excluded.start_time, start_time),"
        " end_time=COALESCE(excluded.end_time, end_time), state=COALESCE(excluded.state, state),"
        " node_list=COALESCE(excluded.node_list, node_list), observed_at=excluded.observed_at",
        (attempt_id, segment, job_id, start_time, end_time, state, node_list, observed_at),
    )


def _time_left(state: SchedulerState, limit: int | None, elapsed: int | None) -> int | None:
    if state in SCHEDULER_TERMINAL or state == SchedulerState.PENDING or limit is None or elapsed is None:
        return None
    return max(0, limit - elapsed)


def apply_observation(tx: Tx, ctx: ServiceContext, *, attempt_id: str, obs: JobObservation) -> list[str]:
    """Apply one scheduler observation to an attempt (inside the caller's transaction).

    Returns short change descriptions (``state:PENDING->RUNNING``, ``terminal:COMPLETED``,
    ``requeue:0->1``) or one ``ignored:<why>`` entry. On TERMINAL (same tx): exit fields, claim
    released, run status refreshed, cancel intent confirmed, ``observe.finalize`` enqueued.
    """
    a = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    if a is None:
        raise NotFound(f"attempt {attempt_id} not found")
    phase = a["phase"]
    if phase not in (*OBSERVED_PHASES, AttemptPhase.TERMINAL.value):
        return [f"ignored:phase={phase}"]
    if not a["job_id"] or obs.job_id != a["job_id"]:
        return ["ignored:job_id"]
    if obs.state == SchedulerState.UNKNOWN:
        return ["ignored:unknown_state"]
    why = identity_mismatch(a, obs)
    if why is not None:
        _record_identity_mismatch(tx, a, obs, why)
        return [f"ignored:identity ({why})"]
    last, seen = parse_iso(a["last_observed_at"]), parse_iso(obs.observed_at)
    if (
        last is not None
        and seen is not None
        and seen < last
        and last <= ctx.clock.now() + timedelta(seconds=CLOCK_SKEW_S)
    ):
        # (a stored time in the future means the wall clock stepped back: ordering by it would
        # drop every fresh observation until the clock catches up)
        return ["ignored:out_of_order"]
    prev_restart = int(a["restart_count"] or 0)
    if obs.restart_count is not None and obs.restart_count < prev_restart:
        # A record of an earlier requeue segment (restart counts only grow): applying it would
        # end or rewrite the current segment.
        return ["ignored:older_segment"]
    new_restart = obs.restart_count if obs.restart_count is not None else prev_restart
    terminal_obs = obs.state in SCHEDULER_TERMINAL
    if phase == AttemptPhase.TERMINAL.value:
        if new_restart > prev_restart:
            changes = _reopen_for_requeue(tx, a, obs, new_restart)
            if terminal_obs:
                # Requeued AND finished between two observations: the new segment ends too
                # (its own terminal event, claim release and finalize for this restart count).
                reopened = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
                assert reopened is not None
                changes += _apply_active(tx, reopened, obs, new_restart, new_restart)
            return changes
        if terminal_obs:
            return _refine_terminal(tx, a, obs)
        return ["ignored:terminal"]
    return _apply_active(tx, a, obs, prev_restart, new_restart)


def _record_identity_mismatch(tx: Tx, a: Row, obs: JobObservation, why: str) -> None:
    seen = tx.one(
        "SELECT 1 AS x FROM events WHERE entity_type='attempt' AND entity_id=? "
        "AND event_type='attempt.identity_mismatch' LIMIT 1",
        (a["attempt_id"],),
    )
    if seen is None:
        tx.event(
            "attempt",
            a["attempt_id"],
            "attempt.identity_mismatch",
            {"job_id": obs.job_id, "source": obs.source, "why": why},
            observed_at=obs.observed_at,
            source="observer",
        )


def _apply_active(tx: Tx, a: Row, obs: JobObservation, prev_restart: int, new_restart: int) -> list[str]:
    aid = a["attempt_id"]
    changes: list[str] = []
    old_state = _state(a["scheduler_state"])
    state = obs.state
    terminal = state in SCHEDULER_TERMINAL
    limit = obs.time_limit_s if obs.time_limit_s is not None else a["time_limit_s"]
    elapsed = obs.elapsed_s if obs.elapsed_s is not None else a["elapsed_s"]
    values: dict[str, Any] = {
        "scheduler_state": state.value,
        "raw_scheduler_state": obs.raw_state,
        "reason": obs.reason if obs.reason is not None else a["reason"],
        "node_list": obs.node_list or a["node_list"],
        "start_time": obs.start_time or a["start_time"],
        "elapsed_s": elapsed,
        "time_limit_s": limit,
        "pending_reason": obs.reason if state == SchedulerState.PENDING else None,
        "time_left_s": _time_left(state, limit, elapsed),
        "last_observed_at": obs.observed_at,
        "last_observation_json": _obs_json(obs),
        "observation_gap": 0,
    }
    for col, val in (
        ("job_submit_time", obs.submit_time),
        ("job_owner", obs.user),
        ("partition", obs.partition),
        ("array_task_id", obs.array_task_id),
    ):
        if a[col] is None and val is not None:
            values[col] = val
    identity = sorted(c for c in IDENTITY_COLUMNS if c in values)
    material = bool(identity)
    if identity:
        changes.append("identity:" + ",".join(identity))
    if new_restart > prev_restart:
        # Requeued while we were not looking (or straight from RUNNING): close the old segment.
        # Its end time is not observed here; finalize settles it from per-segment accounting.
        _upsert_segment(
            tx,
            aid,
            prev_restart,
            job_id=a["job_id"],
            start_time=a["start_time"],
            end_time=None,
            state=SchedulerState.REQUEUED.value,
            node_list=a["node_list"],
            observed_at=obs.observed_at,
        )
        values["restart_count"] = new_restart
        values["start_time"] = obs.start_time  # the new segment's start (None while pending)
        values["node_list"] = obs.node_list
        changes.append(f"requeue:{prev_restart}->{new_restart}")
        material = True
        _cover_requeue_budget(tx, a, new_restart, limit)
    if state != old_state:
        changes.append(f"state:{old_state.value}->{state.value}")
        material = True
    # The job is seen again: an event, but no version bump (the gap flag is freshness, not a
    # state a decision depends on).
    gap_closed = bool(a["observation_gap"])
    if gap_closed:
        changes.append("gap_closed")
    if not terminal and a["phase"] == AttemptPhase.JOB_IDENTIFIED.value:
        values["phase"] = AttemptPhase.OBSERVING.value
        changes.append("phase:observing")
        material = True
    # A new PENDING reason is worth an event but not a version bump: Priority <-> Resources flips
    # every scheduling cycle and must not invalidate a user's expected_version for run_cancel.
    reason_changed = state == SchedulerState.PENDING and obs.reason != a["pending_reason"]
    if reason_changed:
        changes.append(f"pending_reason:{obs.reason}")
    if terminal:
        values.update(
            phase=AttemptPhase.TERMINAL.value,
            terminal_at=tx.now,
            end_time=obs.end_time or a["end_time"],
            exit_code=obs.exit_code if obs.exit_code is not None else a["exit_code"],
            exit_signal=obs.exit_signal if obs.exit_signal is not None else a["exit_signal"],
            pending_reason=None,
            time_left_s=None,
        )
        if a["cancel_state"] in (CancelState.REQUESTED.value, CancelState.SENT.value):
            values["cancel_state"] = CancelState.CONFIRMED.value
            changes.append("cancel:confirmed")
        changes.append(f"terminal:{state.value}")
        material = True
    version = tx.update("attempts", "attempt_id", aid, values, bump_version=material)
    newly_known = (obs.start_time and not a["start_time"]) or (obs.node_list and not a["node_list"])
    has_segment = tx.one(
        "SELECT 1 AS x FROM attempt_segments WHERE attempt_id=? AND segment=?", (aid, new_restart)
    )
    if material or newly_known or has_segment is None:
        _upsert_segment(
            tx,
            aid,
            new_restart,
            job_id=a["job_id"],
            start_time=values.get("start_time"),
            end_time=values.get("end_time") if terminal else None,
            state=state.value,
            node_list=values.get("node_list"),
            observed_at=obs.observed_at,
        )
    if not terminal and elapsed is not None and a["gpus"]:
        _update_estimate(tx, a, elapsed, first=material)
    if not material:
        if reason_changed:
            tx.event(
                "attempt",
                aid,
                "attempt.pending_reason",
                {"reason": obs.reason, "previous": a["pending_reason"]},
                entity_version=a["version"],
                observed_at=obs.observed_at,
                source="observer",
            )
        if gap_closed:
            tx.event(
                "attempt",
                aid,
                "attempt.gap_closed",
                {"state": state.value, "source": obs.source},
                entity_version=a["version"],
                observed_at=obs.observed_at,
                source="observer",
            )
        refresh_run_status(tx, a["run_id"])  # cheap; writes only if the summary drifted
        return changes
    event_type = "attempt.terminal" if terminal else "attempt.observed"
    if new_restart > prev_restart:
        tx.event(
            "attempt",
            aid,
            "attempt.requeued",
            {"from_restart": prev_restart, "restart_count": new_restart, "state": state.value},
            entity_version=version,
            observed_at=obs.observed_at,
            source="observer",
        )
    tx.event(
        "attempt",
        aid,
        event_type,
        {
            "from": old_state.value,
            "to": state.value,
            "source": obs.source,
            "job_id": obs.job_id,
            "reason": obs.reason,
            "exit_code": values.get("exit_code"),
            "changes": changes,
        },
        entity_version=version,
        observed_at=obs.observed_at,
        source="observer",
    )
    if terminal:
        _on_terminal(tx, a, new_restart)
    refresh_run_status(tx, a["run_id"])
    return changes


ESTIMATE_BUCKET_S = 300  # running budget estimate refreshed every 5 minutes of elapsed time


def _update_estimate(tx: Tx, a: Row, elapsed: int, *, first: bool) -> None:
    """Running GPU-hour estimate (``budget.update_estimate``) while the job runs, throttled to
    one write per ``ESTIMATE_BUCKET_S`` of elapsed time. Settlement happens in finalize."""
    previous = a["elapsed_s"]
    if not first and previous is not None and previous // ESTIMATE_BUCKET_S == elapsed // ESTIMATE_BUCKET_S:
        return
    if int(a["restart_count"] or 0):
        return  # earlier segments' time is unknown here; finalize settles from accounting
    update = _optional_function("hpclab.services.budget", "update_estimate")
    if update is None:
        return
    try:
        update(
            tx, attempt_id=a["attempt_id"], estimated_gpu_hours=round(elapsed * int(a["gpus"]) / 3600.0, 6)
        )
    except NotFound:
        return


def _cover_requeue_budget(tx: Tx, a: Row, new_restart: int, limit_s: int | None) -> None:
    """Slurm may charge a requeued job up to its time limit once per segment: the reservation
    grows to ``(restart + 1) x gpus x limit`` and, when finalize had already settled it, becomes
    active again (``budget.cover_requeue``), so an unknown settlement stays an upper bound and a
    live requeued job always holds budget."""
    cover = _optional_function("hpclab.services.budget", "cover_requeue")
    if cover is None:
        return
    gpus = int(a["gpus"] or 0)
    per = round(gpus * float(limit_s) / 3600.0, 6) if limit_s is not None else None
    try:
        cover(
            tx,
            attempt_id=a["attempt_id"],
            segments=new_restart + 1,
            segment_gpu_hours=per,
            note=f"requeued (restart {new_restart}): up to one time limit per segment is held",
        )
    except NotFound:
        return


def _on_terminal(tx: Tx, a: Row, restart: int) -> None:
    """Terminal follow-ups in the same tx: release the execution claim, enqueue finalize."""
    cur = tx.execute(
        "UPDATE runs SET claim_attempt_id=NULL, version=version+1, updated_at=? "
        "WHERE run_id=? AND claim_attempt_id=?",
        (tx.now, a["run_id"], a["attempt_id"]),
    )
    if cur.rowcount:
        tx.event(
            "run",
            a["run_id"],
            "run.claim_released",
            {"attempt_id": a["attempt_id"], "reason": "terminal"},
            source="observer",
        )
    aid = a["attempt_id"]
    tx.enqueue(
        KIND_FINALIZE,
        target_id=aid,
        payload={"attempt_id": aid},
        dedupe_key=f"{KIND_FINALIZE}:{aid}:{restart}",
    )


def _reopen_for_requeue(tx: Tx, a: Row, obs: JobObservation, new_restart: int) -> list[str]:
    """TERMINAL -> OBSERVING only because the scheduler restarted the job (spec §4.5)."""
    aid = a["attempt_id"]
    prev_restart = int(a["restart_count"] or 0)
    _upsert_segment(
        tx,
        aid,
        prev_restart,
        job_id=a["job_id"],
        start_time=a["start_time"],
        end_time=a["end_time"],
        state=a["scheduler_state"],
        node_list=a["node_list"],
        observed_at=obs.observed_at,
    )
    values: dict[str, Any] = {
        "phase": AttemptPhase.OBSERVING.value,
        "scheduler_state": obs.state.value,
        "raw_scheduler_state": obs.raw_state,
        "restart_count": new_restart,
        "terminal_at": None,
        "end_time": None,
        "start_time": obs.start_time,
        "node_list": obs.node_list,
        "exit_code": None,
        "exit_signal": None,
        "elapsed_s": obs.elapsed_s,
        "time_limit_s": obs.time_limit_s if obs.time_limit_s is not None else a["time_limit_s"],
        "pending_reason": obs.reason if obs.state == SchedulerState.PENDING else None,
        "time_left_s": _time_left(obs.state, obs.time_limit_s, obs.elapsed_s),
        "reason": obs.reason,
        "result_state": ResultState.PENDING.value,  # outputs are mutable again
        "last_observed_at": obs.observed_at,
        "last_observation_json": _obs_json(obs),
        "observation_gap": 0,
    }
    version = tx.update("attempts", "attempt_id", aid, values)
    _upsert_segment(
        tx,
        aid,
        new_restart,
        job_id=a["job_id"],
        start_time=obs.start_time,
        end_time=None,
        state=obs.state.value,
        node_list=obs.node_list,
        observed_at=obs.observed_at,
    )
    changes = [f"requeue:{prev_restart}->{new_restart}", f"state:{a['scheduler_state']}->{obs.state.value}"]
    run = tx.one("SELECT claim_attempt_id FROM runs WHERE run_id=?", (a["run_id"],))
    claim_note = "reclaimed"
    if run is not None and run["claim_attempt_id"] is None:
        tx.execute(
            "UPDATE runs SET claim_attempt_id=?, version=version+1, updated_at=? WHERE run_id=?",
            (aid, tx.now, a["run_id"]),
        )
        tx.event("run", a["run_id"], "run.claim_retaken", {"attempt_id": aid, "reason": "requeue"})
    elif run is not None and run["claim_attempt_id"] != aid:
        claim_note = f"conflict: claim held by {run['claim_attempt_id']}"
        changes.append("claim_conflict")
    # finalize may already have settled the earlier segment: the live job must hold budget again
    _cover_requeue_budget(tx, a, new_restart, values["time_limit_s"])
    tx.event(
        "attempt",
        aid,
        "attempt.requeued",
        {
            "from_state": a["scheduler_state"],
            "to_state": obs.state.value,
            "from_restart": prev_restart,
            "restart_count": new_restart,
            "claim": claim_note,
            "source": obs.source,
        },
        entity_version=version,
        observed_at=obs.observed_at,
        source="observer",
    )
    refresh_run_status(tx, a["run_id"])
    return changes


def _refine_terminal(tx: Tx, a: Row, obs: JobObservation) -> list[str]:
    """A later terminal observation (typically sacct after squeue) fills exit/elapsed fields.
    The state itself is only revised by accounting, the authority for finished jobs."""
    fill = {
        "exit_code": obs.exit_code,
        "exit_signal": obs.exit_signal,
        "elapsed_s": obs.elapsed_s,
        "end_time": obs.end_time,
        "start_time": obs.start_time,
        "node_list": obs.node_list,
        "time_limit_s": obs.time_limit_s,
        "job_submit_time": obs.submit_time,
        "job_owner": obs.user,
        "partition": obs.partition,
    }
    values: dict[str, Any] = {k: v for k, v in fill.items() if v is not None and a[k] is None}
    if obs.source == "sacct" and obs.elapsed_s is not None and a["elapsed_s"] != obs.elapsed_s:
        values["elapsed_s"] = obs.elapsed_s  # squeue TimeUsed was sampled before the end
    changes = [f"filled:{k}" for k in sorted(values)]
    revised = obs.source == "sacct" and obs.state.value != a["scheduler_state"]
    if revised:
        values["scheduler_state"] = obs.state.value
        values["raw_scheduler_state"] = obs.raw_state
        changes.append(f"revised:{a['scheduler_state']}->{obs.state.value}")
    if not values:
        return ["ignored:terminal_no_change"]
    # exit fields, elapsed, node list: accounting detail; a revised state or a newly known job
    # identity: versioned
    material = revised or any(c in values for c in IDENTITY_COLUMNS)
    values["last_observed_at"] = obs.observed_at
    values["last_observation_json"] = _obs_json(obs)
    version = tx.update("attempts", "attempt_id", a["attempt_id"], values, bump_version=material)
    _upsert_segment(
        tx,
        a["attempt_id"],
        int(a["restart_count"] or 0),
        job_id=a["job_id"],
        start_time=values.get("start_time"),
        end_time=values.get("end_time"),
        state=obs.state.value if revised else None,
        node_list=values.get("node_list"),
        observed_at=obs.observed_at,
    )
    if revised:
        tx.event(
            "attempt",
            a["attempt_id"],
            "attempt.terminal_revised",
            {"from": a["scheduler_state"], "to": obs.state.value, "source": obs.source},
            entity_version=version,
            observed_at=obs.observed_at,
            source="observer",
        )
        refresh_run_status(tx, a["run_id"])
    elif material:
        tx.event(
            "attempt",
            a["attempt_id"],
            "attempt.identity_filled",
            {k: values[k] for k in IDENTITY_COLUMNS if k in values},
            entity_version=version,
            observed_at=obs.observed_at,
            source="observer",
        )
    return changes


def mark_observation_gap(
    tx: Tx, *, attempt_id: str, reason: str, observed_at: str | None = None
) -> list[str]:
    """The job left squeue but accounting has no row (yet): flag it, never guess a state.

    The flag is freshness information (evented, not versioned): the attempt's phase and
    scheduler state are unchanged, so a user's ``expected_version`` stays valid."""
    a = tx.one(
        "SELECT attempt_id, run_id, phase, observation_gap, version FROM attempts WHERE attempt_id=?",
        (attempt_id,),
    )
    if a is None:
        raise NotFound(f"attempt {attempt_id} not found")
    if a["phase"] not in OBSERVED_PHASES or a["observation_gap"]:
        return []
    tx.update("attempts", "attempt_id", attempt_id, {"observation_gap": 1}, bump_version=False)
    tx.event(
        "attempt",
        attempt_id,
        "attempt.observation_gap",
        {"reason": reason},
        entity_version=a["version"],
        observed_at=observed_at,
        source="observer",
    )
    refresh_run_status(tx, a["run_id"])
    return ["gap_opened"]


# ---------------------------------------------------------------------------
# Poll bookkeeping (shared rate limit per cluster + source)
# ---------------------------------------------------------------------------


def _intervals(ctx: ServiceContext, cluster: ResolvedCluster) -> tuple[float, float]:
    policy = cluster.profile.policy
    override = ctx.settings.min_poll_interval_s
    if override is not None:
        return float(override), float(override)
    return float(policy.poll_min_interval_s), float(policy.sacct_min_interval_s)


def _claim_slot(
    ctx: ServiceContext, cluster_id: str, source: str, interval_s: float, *, force: bool
) -> tuple[bool, Literal["too_soon", "backoff"] | None, str | None]:
    """Atomically take the next poll slot (last_polled_at := now) unless too soon / backing off.

    Stored times beyond what this code ever writes (a poll after now, a backoff longer than the
    longest one) mean the wall clock stepped back (NTP step, manual fix, VM restore): they are
    not honoured, or polling would freeze until the clock caught up with them."""
    now = ctx.clock.now()
    with ctx.db.tx(source="observer") as tx:
        row = tx.one("SELECT * FROM poll_state WHERE cluster_id=? AND source=?", (cluster_id, source))
        if row is not None and not force:
            backoff = parse_iso(row["backoff_until"])
            longest = max(MAX_BACKOFF_S * 1.2, AUTH_PAUSE_S) + CLOCK_SKEW_S
            if backoff is not None and now < backoff <= now + timedelta(seconds=longest):
                return False, "backoff", row["backoff_until"]
            last = parse_iso(row["last_polled_at"])
            if (
                last is not None
                and last <= now + timedelta(seconds=CLOCK_SKEW_S)
                and last + timedelta(seconds=interval_s) > now
            ):
                return False, "too_soon", iso(last + timedelta(seconds=interval_s))
        tx.execute(
            "INSERT INTO poll_state (cluster_id, source, last_polled_at, consecutive_failures) VALUES (?, ?, ?, 0)"
            " ON CONFLICT(cluster_id, source) DO UPDATE SET last_polled_at=excluded.last_polled_at",
            (cluster_id, source, tx.now),
        )
    return True, None, None


def _record_ok(tx: Tx, cluster_id: str, source: str) -> None:
    row = tx.one(
        "SELECT consecutive_failures FROM poll_state WHERE cluster_id=? AND source=?", (cluster_id, source)
    )
    tx.execute(
        "INSERT INTO poll_state (cluster_id, source, last_polled_at, last_ok_at, consecutive_failures)"
        " VALUES (?, ?, ?, ?, 0) ON CONFLICT(cluster_id, source) DO UPDATE SET last_ok_at=excluded.last_ok_at,"
        " consecutive_failures=0, backoff_until=NULL, last_error=NULL",
        (cluster_id, source, tx.now, tx.now),
    )
    if row is not None and row["consecutive_failures"]:
        tx.event(
            "cluster",
            cluster_id,
            "cluster.poll_recovered",
            {"source": source, "after_failures": row["consecutive_failures"]},
            source="observer",
        )


def _record_failure(
    ctx: ServiceContext, cluster_id: str, source: str, exc: HpcLabError, interval_s: float
) -> str:
    """Backoff bookkeeping; events/notifications only on the ok -> failing transition."""
    auth = exc.code == ErrorCode.AUTH_REQUIRED
    with ctx.db.tx(source="observer") as tx:
        row = tx.one("SELECT * FROM poll_state WHERE cluster_id=? AND source=?", (cluster_id, source))
        failures = int(row["consecutive_failures"]) + 1 if row else 1
        if auth:
            delay = AUTH_PAUSE_S
        else:
            base = min(MAX_BACKOFF_S, max(interval_s, 5.0) * 2 ** min(failures, 6))
            delay = base * (0.8 + random.random() * 0.4)  # noqa: S311 - jitter only
        until = iso(ctx.clock.now() + timedelta(seconds=delay))
        message = f"{exc.code}: {exc.message}"[:1000]
        tx.execute(
            "INSERT INTO poll_state (cluster_id, source, last_polled_at, consecutive_failures, backoff_until,"
            " last_error) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(cluster_id, source) DO UPDATE SET"
            " consecutive_failures=excluded.consecutive_failures, backoff_until=excluded.backoff_until,"
            " last_error=excluded.last_error",
            (cluster_id, source, tx.now, failures, until, message),
        )
        if failures == 1:
            tx.event(
                "cluster",
                cluster_id,
                "cluster.poll_failed",
                {"source": source, "code": str(exc.code), "message": exc.message[:300]},
                source="observer",
            )
            if auth:
                emit = _optional_function("hpclab.services.notify", "emit")
                if emit is not None:
                    last_ok = row["last_ok_at"] if row else None
                    emit(
                        tx,
                        event_type="auth_required",
                        entity_id=cluster_id,
                        title=f"hpclab needs you to log in to {cluster_id}",
                        body=f"Polling {cluster_id} is paused until you run: hpclab auth {cluster_id}",
                        urgency="immediate",
                        dedupe_key=f"auth_required:{cluster_id}:{last_ok or 'never'}",
                    )
    return message


# ---------------------------------------------------------------------------
# poll_cluster
# ---------------------------------------------------------------------------


def _poll_targets(ctx: ServiceContext, cluster_id: str) -> tuple[list[Row], list[Row]]:
    active = ctx.db.all(
        "SELECT * FROM attempts WHERE cluster_id=? AND phase IN (?, ?) AND job_id IS NOT NULL ORDER BY attempt_id",
        (cluster_id, *OBSERVED_PHASES),
    )
    since = iso(ctx.clock.now() - timedelta(seconds=REQUEUE_WATCH_S))
    marks = ",".join("?" for _ in REQUEUE_STATES)
    watch = ctx.db.all(
        f"SELECT * FROM attempts WHERE cluster_id=? AND phase='terminal' AND job_id IS NOT NULL "
        f"AND terminal_at >= ? AND scheduler_state IN ({marks}) ORDER BY attempt_id",
        (cluster_id, since, *REQUEUE_STATES),
    )
    return active, watch


def pick_observation(
    attempt: Mapping[str, Any], candidates: Sequence[JobObservation]
) -> JobObservation | None:
    """The observation that is this attempt's job among rows with its job id (array task, identity,
    then the newest submit time). An identity-mismatched row is still returned when it is the only
    candidate, so the mismatch gets recorded instead of silently looking like a gap."""
    rows = [
        o
        for o in candidates
        if o.job_id == attempt["job_id"]
        and (o.array_task_id or None) == (attempt.get("array_task_id") or None)
    ]
    if not rows:
        return None
    good = [o for o in rows if identity_mismatch(attempt, o) is None]
    pool = good or rows
    return max(pool, key=lambda o: o.submit_time or "")


def _foreign_summary(jobs: Sequence[JobObservation], bound_ids: set[str], observed_at: str) -> dict[str, Any]:
    """The user's jobs not managed by hpclab, as counts only (their ids are not actionable here)."""
    count = running = pending = gpus = gpus_unknown = unbound = 0
    partitions: dict[str, int] = {}
    for o in jobs:
        if o.state in SCHEDULER_TERMINAL:
            continue
        managed = token_from_job_name(o.name or "") is not None or token_from_comment(o.comment) is not None
        if managed:
            if o.job_id not in bound_ids:
                unbound += 1
            continue
        count += 1
        if o.state == SchedulerState.PENDING:
            pending += 1
        else:
            running += 1
        if o.alloc_gpus is None:
            gpus_unknown += 1
        else:
            gpus += o.alloc_gpus
        if o.partition:
            partitions[o.partition] = partitions.get(o.partition, 0) + 1
    return {
        "count": count,
        "running": running,
        "pending": pending,
        "gpus": gpus,
        "gpus_unknown_jobs": gpus_unknown,
        "partitions": dict(sorted(partitions.items())[:20]),
        "unbound_hpclab_jobs": unbound,
        "observed_at": observed_at,
    }


def poll_cluster(ctx: ServiceContext, *, cluster_id: str, force: bool = False) -> PollResult:
    """One batched, rate-limited observation pass over a cluster's submitted attempts.

    ``force=True`` ignores the minimum interval and backoff (explicit human request) but still
    records the poll so other clients share it.
    """
    return _poll_cluster(ctx, cluster_id, force=force, refresh_views=True)


def _poll_cluster(ctx: ServiceContext, cluster_id: str, *, force: bool, refresh_views: bool) -> PollResult:
    """``refresh_views=False`` (on-demand refresh): scheduler state only, so the caller's deadline
    is not spent re-reading other attempts' telemetry (the periodic poll keeps those fresh)."""
    cluster = ctx.clusters.get(cluster_id)
    sq_interval, sa_interval = _intervals(ctx, cluster)
    active, watch = _poll_targets(ctx, cluster_id)
    if not active and not watch and not force:
        return PollResult(cluster_id=cluster_id, polled=False, skipped_reason="no_active_attempts")
    claimed, why, next_at = _claim_slot(ctx, cluster_id, "squeue", sq_interval, force=force)
    if not claimed:
        return PollResult(cluster_id=cluster_id, polled=False, skipped_reason=why, next_poll_after=next_at)
    result = PollResult(cluster_id=cluster_id, polled=True, attempts_polled=len(active) + len(watch))
    try:
        scheduler = ctx.clusters.scheduler(cluster_id)
        queue = scheduler.query_queue(user=None)
    except HpcLabError as exc:
        result.error_code = str(exc.code)
        result.error = _record_failure(ctx, cluster_id, "squeue", exc, sq_interval)
        result.auth_required = exc.code == ErrorCode.AUTH_REQUIRED
        return result
    result.observed_at = queue.observed_at
    by_job: dict[str, list[JobObservation]] = {}
    for obs in queue.jobs:
        by_job.setdefault(obs.job_id, []).append(obs)
    found: list[tuple[Row, JobObservation]] = []
    missing: list[Row] = []
    for a in active:
        picked = pick_observation(a, by_job.get(a["job_id"], []))
        if picked is not None:
            found.append((a, picked))
        else:
            missing.append(a)
    for a in watch:
        picked = pick_observation(a, by_job.get(a["job_id"], []))
        if picked is not None:
            found.append((a, picked))
    if not queue.complete:
        # A partial listing says nothing about the jobs it did not list: they are neither
        # "gone" (no sacct, no gap) nor observed. The listed ones are still genuine observations.
        result.notes.append(
            f"squeue listing incomplete ({queue.error}); {len(missing)} attempt(s) not observed"
        )
        missing = []

    accounting: dict[str, list[JobObservation]] = {}
    chains: dict[str, list[JobObservation]] = {}
    chain_notes: dict[str, str] = {}
    sacct_note: str | None = None
    if missing:
        result.sacct, sacct_note = _query_missing(
            ctx, scheduler, cluster_id, missing, sa_interval, force, accounting
        )
        if sacct_note:
            result.notes.append(sacct_note)
        if result.sacct == "ok":
            _check_unseen_requeues(ctx, scheduler, cluster_id, missing, accounting, chains, chain_notes)

    bound_ids = {
        r["job_id"]
        for r in ctx.db.all(
            "SELECT job_id FROM attempts WHERE cluster_id=? AND job_id IS NOT NULL", (cluster_id,)
        )
    }
    foreign = _foreign_summary(queue.jobs, bound_ids, queue.observed_at)
    result.foreign_jobs = foreign["count"] if queue.complete else None
    with ctx.db.tx(source="observer") as tx:
        for a, obs in found:
            changes = apply_observation(tx, ctx, attempt_id=a["attempt_id"], obs=obs)
            _collect(result, a, changes)
        for a in missing:
            if result.sacct == "deferred":
                result.deferred.append(a["attempt_id"])
            reason = _apply_accounting_for_missing(tx, ctx, result, a, accounting, chains, chain_notes)
            if reason is not None and mark_observation_gap(
                tx, attempt_id=a["attempt_id"], reason=reason, observed_at=queue.observed_at
            ):
                result.gaps.append(a["attempt_id"])
        if queue.complete:
            tx.execute(
                "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at,"
                " expires_at) VALUES (?, ?, ?, 'squeue', ?, ?)",
                (
                    cluster_id,
                    FOREIGN_JOBS_KEY,
                    jdump(foreign),
                    queue.observed_at,
                    iso(ctx.clock.now() + timedelta(seconds=FOREIGN_JOBS_TTL_S)),
                ),
            )
            _record_ok(tx, cluster_id, "squeue")
    result.changes = _bounded(result.changes)
    result.terminal = _bounded(result.terminal)
    result.gaps = _bounded(result.gaps)
    result.deferred = _bounded(result.deferred)
    if not queue.complete:
        partial = TransportError(f"squeue returned an incomplete listing: {queue.error}")
        result.error_code = str(partial.code)
        result.error = _record_failure(ctx, cluster_id, "squeue", partial, sq_interval)
        return result
    result.ok = True
    if refresh_views:
        _refresh_running_views(ctx, cluster_id)
    return result


def _apply_accounting_for_missing(
    tx: Tx,
    ctx: ServiceContext,
    result: PollResult,
    a: Row,
    accounting: Mapping[str, list[JobObservation]],
    chains: Mapping[str, list[JobObservation]] | None = None,
    chain_notes: Mapping[str, str] | None = None,
) -> str | None:
    """An attempt that is not in squeue: apply its *final* accounting record, else return why it
    is an observation gap (None = nothing to flag, e.g. sacct deferred by its rate limit).

    Not in ``squeue --states=all`` means the job ended at least MinJobAge ago (or the controller
    lost it). Accounting that still says PENDING/RUNNING is lagging behind (slurmdbd has not
    received the end record): it is not a fresh "still running" observation, so it is not applied.

    A record with a later submit time is the job's own record after a requeue the controller did
    not see only when the per-segment accounting proved it (``chains``): it is then applied as a
    requeue with the verified restart count, and the earlier segments are recorded.
    """
    if result.sacct == "deferred":
        return None
    if result.sacct != "ok":
        return f"not in squeue and accounting is {result.sacct}"
    acct = pick_observation(a, accounting.get(a["job_id"], []))
    if acct is None:
        return "not in squeue and sacct has no record yet (accounting lag or MinJobAge)"
    if acct.state not in SCHEDULER_TERMINAL:
        return f"not in squeue but accounting still reports {acct.state.value} (accounting lag)"
    chain = (chains or {}).get(a["attempt_id"])
    if chain:
        acct = acct.model_copy(update={"restart_count": len(chain) - 1})
    changes = apply_observation(tx, ctx, attempt_id=a["attempt_id"], obs=acct)
    if chain and not any(c.startswith("ignored:") for c in changes):
        _record_chain_segments(tx, a["attempt_id"], a["job_id"], chain[:-1], acct.observed_at)
        tx.event(
            "attempt",
            a["attempt_id"],
            "attempt.requeue_verified",
            {
                "restart_count": len(chain) - 1,
                "source": "sacct --duplicates",
                "why": "accounting showed a later submit time: Slurm requeued the job while hpclab "
                "was not observing it",
                "segments": [
                    {
                        "segment": s.restart_count,
                        "submit_time": s.submit_time,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "state": s.state.value,
                    }
                    for s in chain[:10]
                ],
            },
            observed_at=acct.observed_at,
            source="observer",
        )
        changes.append(f"requeue_verified:{len(chain) - 1}")
        tx.execute(
            "DELETE FROM observations WHERE cluster_id=? AND obs_key=?",
            (a["cluster_id"], f"{REQUEUE_CHECK_PREFIX}{a['attempt_id']}"),
        )
    _collect(result, a, changes)
    if any(c.startswith("ignored:identity") for c in changes):
        note = (chain_notes or {}).get(a["attempt_id"])
        if note:
            return (
                f"not in squeue; accounting shows job {a['job_id']} with a later submit time (a requeue "
                f"hpclab did not see?) but its earlier segments could not be verified: {note}"
            )
        return "not in squeue and accounting only has another job with this id"
    return None


def _check_unseen_requeues(
    ctx: ServiceContext,
    scheduler: Scheduler,
    cluster_id: str,
    missing: Sequence[Row],
    accounting: Mapping[str, list[JobObservation]],
    chains: dict[str, list[JobObservation]],
    notes: dict[str, str],
) -> None:
    """For attempts whose final accounting record differs only by a later submit time (a requeue
    the controller did not see: Slurm restamps SubmitTime, sacct has no restart count), fetch the
    per-segment records (``sacct --duplicates``) and keep the chain when it provably starts at the
    recorded submit time and ends with that record. Bounded per poll; outside any transaction.

    A chain that could not be verified is asked for again only after ``REQUEUE_RECHECK_S`` (or
    when accounting shows a newer record): the check is one more per-job sacct call and must not
    ride along with every poll while the attempt stays in its gap."""
    now = ctx.clock.now()
    checked = 0
    failed: list[tuple[str, str | None, str]] = []
    for a in missing:
        acct = pick_observation(a, accounting.get(a["job_id"], []))
        if acct is None or acct.state not in SCHEDULER_TERMINAL or not later_submit_only(a, acct):
            continue
        recent = _recent_requeue_check(ctx.db, cluster_id, a["attempt_id"], acct.submit_time, now)
        if recent is not None:
            notes[a["attempt_id"]] = recent
            continue
        if checked >= MAX_REQUEUE_CHECKS_PER_POLL:
            notes[a["attempt_id"]] = "check deferred to a later poll"
            continue
        checked += 1
        try:
            chain, why = _fetch_requeue_chain(scheduler, a, raise_auth=False)
        except Exception as exc:  # noqa: BLE001 - an optional check must never break the poll
            chain, why = None, f"per-segment accounting failed: {exc}"
        if chain is not None and (len(chain) < 2 or not _same_time(chain[-1].submit_time, acct.submit_time)):
            chain, why = None, "per-segment accounting does not end with the job's latest record"
        if chain is not None:
            chains[a["attempt_id"]] = chain
            continue
        why = why or "per-segment accounting could not be verified"
        notes[a["attempt_id"]] = why
        failed.append((a["attempt_id"], acct.submit_time, why))
    if failed:
        _store_requeue_checks(ctx, cluster_id, failed)


def _recent_requeue_check(
    reader: Any, cluster_id: str, attempt_id: str, submit_time: str | None, now: datetime
) -> str | None:
    """Why the last per-segment check of this attempt failed, when it was made less than
    ``REQUEUE_RECHECK_S`` ago for the same latest accounting record (None = check again)."""
    row = reader.one(
        "SELECT value_json, observed_at FROM observations WHERE cluster_id=? AND obs_key=?",
        (cluster_id, f"{REQUEUE_CHECK_PREFIX}{attempt_id}"),
    )
    if row is None:
        return None
    checked = parse_iso(row["observed_at"])
    doc = jload(row["value_json"], None)
    if (
        checked is None
        or not isinstance(doc, dict)
        or not now - timedelta(seconds=REQUEUE_RECHECK_S) < checked <= now + timedelta(seconds=CLOCK_SKEW_S)
        or not _same_time(doc.get("submit_time"), submit_time)
    ):
        return None
    return f"{doc.get('note') or 'per-segment accounting could not be verified'} (checked at {row['observed_at']})"


def _store_requeue_checks(
    ctx: ServiceContext, cluster_id: str, failed: Sequence[tuple[str, str | None, str]]
) -> None:
    expires = iso(ctx.clock.now() + timedelta(seconds=REQUEUE_RECHECK_S))
    with ctx.db.tx(source="observer") as tx:
        tx.execute(
            "DELETE FROM observations WHERE cluster_id=? AND obs_key LIKE ? AND expires_at < ?",
            (cluster_id, f"{REQUEUE_CHECK_PREFIX}%", tx.now),
        )
        for attempt_id, submit_time, why in failed:
            tx.execute(
                "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at,"
                " expires_at) VALUES (?, ?, ?, 'sacct', ?, ?)",
                (
                    cluster_id,
                    f"{REQUEUE_CHECK_PREFIX}{attempt_id}",
                    jdump({"submit_time": submit_time, "note": why[:300]}),
                    tx.now,
                    expires,
                ),
            )


def _collect(result: PollResult, a: Row, changes: list[str]) -> None:
    for c in changes:
        if not c.startswith("ignored:terminal") and len(result.changes) < MAX_RESULT_ITEMS:
            result.changes.append(f"{a['attempt_id']}:{c}")
        if c.startswith("terminal:"):
            result.terminal.append(a["attempt_id"])


def _query_missing(
    ctx: ServiceContext,
    scheduler: Scheduler,
    cluster_id: str,
    missing: list[Row],
    interval_s: float,
    force: bool,
    out: dict[str, list[JobObservation]],
) -> tuple[Literal["ok", "deferred", "unavailable", "unsupported", "error"], str | None]:
    """One sacct for the attempts that left squeue, plus the terminal attempts whose finalize /
    late settlement waits for accounting (their result is cached for them: one call serves all)."""
    claimed, why, next_at = _claim_slot(ctx, cluster_id, "sacct", interval_s, force=force)
    if not claimed:
        return "deferred", f"sacct deferred ({why}; next at {next_at})"
    ids = {a["attempt_id"] for a in missing}
    waiting = _awaiting_accounting(ctx.db, cluster_id, exclude=ids)
    try:
        res = scheduler.query_accounting(job_ids=sorted({a["job_id"] for a in [*missing, *waiting]}))
    except HpcLabError as exc:
        message = _record_failure(ctx, cluster_id, "sacct", exc, interval_s)
        state = "unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable"
        _store_accounting_checks(ctx, cluster_id, waiting, None, state, f"accounting {state}: {message}")
        if exc.code == ErrorCode.UNSUPPORTED:
            return "unsupported", f"accounting unsupported: {message}"
        if exc.code == ErrorCode.AUTH_REQUIRED:
            return "error", f"accounting needs auth: {message}"
        return "unavailable", f"accounting unavailable: {message}"
    for obs in res.jobs:
        out.setdefault(obs.job_id, []).append(obs)
    _store_accounting_checks(ctx, cluster_id, waiting, res.jobs, "ok", None, record_ok=True)
    return "ok", None


# ---------------------------------------------------------------------------
# Accounting for terminal attempts (finalize, late settlement): batched, rate-limited, cached
# ---------------------------------------------------------------------------


class AccountingCheck(BaseModel):
    """What accounting says about a terminal attempt's job right now."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["ok", "missing", "unavailable", "unsupported", "deferred"]
    record: JobObservation | None = None
    note: str | None = None
    retry_at: str | None = None


def _final_sacct(attempt: Mapping[str, Any]) -> JobObservation | None:
    """The attempt's stored final accounting record (a terminal sacct observation with elapsed)."""
    acct = _last_sacct_obs(attempt)
    if acct is None or acct.elapsed_s is None or acct.state not in SCHEDULER_TERMINAL:
        return None  # an earlier, non-final accounting view: ask again
    return acct


def _final_record(attempt: Mapping[str, Any], jobs: Sequence[JobObservation]) -> JobObservation | None:
    rec = pick_observation(attempt, jobs)
    if rec is None or rec.state not in SCHEDULER_TERMINAL or identity_mismatch(attempt, rec) is not None:
        return None  # another job, or accounting has not caught up with the end yet
    return rec


def _awaiting_accounting(reader: Any, cluster_id: str, *, exclude: set[str]) -> list[Row]:
    """Terminal attempts whose ``observe.finalize`` / ``observe.settle`` still needs a final
    accounting record (bounded; oldest first)."""
    rows = reader.all(
        "SELECT * FROM attempts WHERE cluster_id=? AND phase='terminal' AND job_id IS NOT NULL AND attempt_id IN "
        "(SELECT target_id FROM outbox WHERE kind IN (?, ?) AND state IN ('pending', 'leased')) "
        "ORDER BY terminal_at, attempt_id LIMIT ?",
        (cluster_id, KIND_FINALIZE, KIND_RESETTLE, MAX_ACCOUNTING_BATCH),
    )
    return [r for r in rows if r["attempt_id"] not in exclude and _final_sacct(r) is None]


def _store_accounting_checks(
    ctx: ServiceContext,
    cluster_id: str,
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[JobObservation] | None,
    state: str,
    note: str | None,
    *,
    record_ok: bool = False,
) -> None:
    """Cache one accounting answer per attempt (``observations``; checked-at = now) so the other
    waiting attempts use it instead of each asking sacct for its own job."""
    if not rows and not record_ok:
        return
    expires = iso(ctx.clock.now() + timedelta(seconds=ACCOUNTING_CHECK_TTL_S))
    with ctx.db.tx(source="observer") as tx:
        if record_ok:
            _record_ok(tx, cluster_id, "sacct")
        tx.execute(
            "DELETE FROM observations WHERE cluster_id=? AND obs_key LIKE ? AND expires_at < ?",
            (cluster_id, f"{ACCOUNTING_CHECK_PREFIX}%", tx.now),
        )
        for a in rows:
            rec = _final_record(a, jobs) if jobs is not None else None
            doc = {
                "state": "ok" if rec is not None else ("missing" if jobs is not None else state),
                "note": note,
                "record": jload(_obs_json(rec)) if rec is not None else None,
            }
            tx.execute(
                "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at,"
                " expires_at) VALUES (?, ?, ?, 'sacct', ?, ?)",
                (cluster_id, f"{ACCOUNTING_CHECK_PREFIX}{a['attempt_id']}", jdump(doc), tx.now, expires),
            )


def _cached_check(
    reader: Any, attempt: Mapping[str, Any], *, now: datetime, max_age_s: float | None
) -> AccountingCheck | None:
    """The cached accounting answer for this attempt, made after it became terminal."""
    row = reader.one(
        "SELECT value_json, observed_at FROM observations WHERE cluster_id=? AND obs_key=?",
        (attempt["cluster_id"], f"{ACCOUNTING_CHECK_PREFIX}{attempt['attempt_id']}"),
    )
    if row is None:
        return None
    checked, ended = parse_iso(row["observed_at"]), parse_iso(attempt.get("terminal_at"))
    if checked is None or (ended is not None and checked < ended):
        return None
    if max_age_s is not None and checked < now - timedelta(seconds=max_age_s):
        return None
    doc = jload(row["value_json"], None)
    if not isinstance(doc, dict) or doc.get("state") not in ("ok", "missing", "unavailable", "unsupported"):
        return None
    record: JobObservation | None = None
    if isinstance(doc.get("record"), dict):
        try:
            record = JobObservation.model_validate(doc["record"])
        except ValueError:
            return None
    if doc["state"] == "ok" and record is None:
        return None
    note = doc.get("note")
    return AccountingCheck(state=doc["state"], record=record, note=note if isinstance(note, str) else None)


def _accounting_check(
    ctx: ServiceContext, cluster: ResolvedCluster, attempt: Mapping[str, Any]
) -> AccountingCheck:
    """Final accounting for a terminal attempt, sharing the cluster's sacct rate limit.

    A cached final record is used as is. Otherwise the caller takes the ``sacct`` poll slot and
    asks, in one call, for every terminal attempt still waiting on accounting (the answers are
    cached for them); when the slot was taken recently, the answer cached by whoever took it is
    used, else the check is ``deferred`` until the slot frees. Auth problems raise."""
    now = ctx.clock.now()
    cached = _cached_check(ctx.db, attempt, now=now, max_age_s=None)
    if cached is not None and cached.state == "ok":
        return cached
    if not attempt.get("job_id"):
        return AccountingCheck(state="missing", note="the attempt has no job id")
    cluster_id = attempt["cluster_id"]
    _, interval = _intervals(ctx, cluster)
    claimed, why, next_at = _claim_slot(ctx, cluster_id, "sacct", interval, force=False)
    if not claimed:
        fresh = _cached_check(ctx.db, attempt, now=now, max_age_s=interval + CLOCK_SKEW_S)
        if fresh is not None:
            return fresh
        if why == "backoff":
            row = ctx.db.one(
                "SELECT last_error FROM poll_state WHERE cluster_id=? AND source='sacct'", (cluster_id,)
            )
            if row is not None and str(row["last_error"] or "").startswith(str(ErrorCode.AUTH_REQUIRED)):
                # the slot's holder needed a login: wait for it like every other background step
                # (parked until `hpclab auth`), without an SSH attempt of our own
                raise AuthRequiredError(
                    f"{cluster_id}: accounting (sacct) needs a login again: run `hpclab auth {cluster_id}`",
                    details={
                        "cluster_id": cluster_id,
                        "gated": True,
                        "maybe_executed": False,
                        "retry_at": next_at,
                    },
                )
            return AccountingCheck(
                state="unavailable",
                note=f"sacct is backing off after errors (until {next_at})",
                retry_at=next_at,
            )
        return AccountingCheck(
            state="deferred", note=f"sacct rate limit (next at {next_at})", retry_at=next_at
        )
    batch = [attempt, *_awaiting_accounting(ctx.db, cluster_id, exclude={attempt["attempt_id"]})]
    try:
        res = ctx.clusters.scheduler(cluster_id).query_accounting(
            job_ids=sorted({r["job_id"] for r in batch})
        )
    except HpcLabError as exc:
        _record_failure(ctx, cluster_id, "sacct", exc, interval)
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        state = "unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable"
        note = f"accounting {state}: {exc.message[:200]}"
        _store_accounting_checks(ctx, cluster_id, batch, None, state, note)
        return AccountingCheck(state=state, note=note)
    _store_accounting_checks(ctx, cluster_id, batch, res.jobs, "ok", None, record_ok=True)
    rec = _final_record(attempt, res.jobs)
    return AccountingCheck(state="ok" if rec is not None else "missing", record=rec)


def _seconds_until(ctx: ServiceContext, when: str | None, *, floor: float, cap: float) -> float:
    target = parse_iso(when)
    delta = (target - ctx.clock.now()).total_seconds() if target is not None else cap
    return min(cap, max(floor, delta)) + random.random()  # noqa: S311 - jitter only


def _accounting_wait(
    ctx: ServiceContext, cluster: ResolvedCluster, check: AccountingCheck, ended: datetime, tries: int
) -> HandlerResult | None:
    """Whether finalize waits for accounting (a retry) or goes on without it (None).

    * ``deferred`` (the shared sacct slot was taken): retry when it frees - whoever took it
      cached the answer for this attempt too.
    * ``missing``: slurmdbd lags behind the job end; waited out for ``accounting_lag_s`` plus a
      grace period, then finalize goes on and ``observe.settle`` keeps asking.
    * ``unavailable`` (sacct failing: slurmdbd restarting, "Socket timed out"): that is not "no
      record" - waited for ``ACCOUNTING_OUTAGE_WAIT_S``, then the same provisional path.
    Retries follow the cluster's sacct interval, never a fixed short period."""
    if check.state in ("ok", "unsupported"):
        return None
    _, interval = _intervals(ctx, cluster)
    if check.state == "deferred":
        delay = _seconds_until(ctx, check.retry_at, floor=1.0, cap=interval + CLOCK_SKEW_S)
        return HandlerResult.retry(delay, note=f"waiting for accounting (deferred): {check.note}")
    if check.state == "missing":
        wait_s = float(cluster.profile.policy.accounting_lag_s) + ACCOUNTING_GRACE_S
    else:
        wait_s = ACCOUNTING_OUTAGE_WAIT_S
    if ctx.clock.now() >= ended + timedelta(seconds=wait_s):
        return None
    if check.retry_at is not None:
        delay = _seconds_until(ctx, check.retry_at, floor=max(1.0, interval), cap=MAX_BACKOFF_S * 1.2)
    elif check.state == "missing":
        delay = max(1.0, interval) + random.random()  # noqa: S311 - jitter only
    else:
        delay = min(300.0, max(1.0, interval) * 2 ** min(max(tries - 1, 0), 3))
    note = f"waiting for accounting ({check.state})" + (f": {check.note}" if check.note else "")
    return HandlerResult.retry(delay, note=note)


def _settlement_chain(
    ctx: ServiceContext, attempt: Mapping[str, Any], restart: int
) -> tuple[list[JobObservation] | None, str | None, bool]:
    """(earlier segments, why not, worth retrying) for a requeued attempt whose earlier segments'
    start/end times were not all observed (a requeue between two polls goes straight from
    RUNNING to PENDING): accounting's per-segment records (``sacct --duplicates``) settle them
    exactly. Auth problems raise."""
    try:
        scheduler = ctx.clusters.scheduler(attempt["cluster_id"])
        chain, why = _fetch_requeue_chain(scheduler, attempt, raise_auth=True)
    except HpcLabError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return None, f"per-segment accounting unavailable: {exc.message[:200]}", True
    if chain is None:
        return None, why, bool(why and why.startswith("per-segment accounting unavailable"))
    if len(chain) < restart + 1:
        return (
            None,
            f"per-segment accounting (sacct --duplicates) has {len(chain)} segment(s) of this job; "
            f"the attempt ran {restart + 1}",
            True,  # accounting may still lag behind the latest segment
        )
    return chain[:restart], None, False


def stored_accounting(reader: Any, attempt: Mapping[str, Any]) -> JobObservation | None:
    """The attempt's final accounting record already known to hpclab (the one applied by the
    observer / finalize, or the batched check cached for it), or None: other services use this
    instead of asking sacct for one job each."""
    acct = _final_sacct(attempt)
    if acct is not None:
        return acct
    cached = _cached_check(reader, attempt, now=datetime.max, max_age_s=None)
    return cached.record if cached is not None and cached.state == "ok" else None


def _forget_accounting_check(tx: Tx, attempt: Mapping[str, Any]) -> None:
    tx.execute(
        "DELETE FROM observations WHERE cluster_id=? AND obs_key=?",
        (attempt["cluster_id"], f"{ACCOUNTING_CHECK_PREFIX}{attempt['attempt_id']}"),
    )


# ---------------------------------------------------------------------------
# On-demand refresh + periodic poll
# ---------------------------------------------------------------------------


def refresh_attempts(ctx: ServiceContext, *, attempt_ids: list[str], deadline_s: float) -> RefreshResult:
    """Bounded refresh for ``run_inspect(refresh=True)``: polls the attempts' clusters (rate limits
    honoured) and waits at most ``deadline_s``; a slower poll keeps running in the background and
    the caller gets the cached state with ``stale=True``."""
    out = RefreshResult(attempt_ids=list(attempt_ids))
    if not attempt_ids:
        return out
    marks = ",".join("?" for _ in attempt_ids)
    rows = ctx.db.all(
        f"SELECT attempt_id, cluster_id, phase, job_id FROM attempts WHERE attempt_id IN ({marks})",
        list(attempt_ids),
    )
    active = [r for r in rows if r["phase"] in OBSERVED_PHASES and r["job_id"]]
    for r in rows:
        if r["phase"] not in OBSERVED_PHASES:
            out.notes.append(f"{r['attempt_id']}: phase {r['phase']} is not polled")
    clusters = sorted({r["cluster_id"] for r in active})
    futures = {
        cid: _submit_background(_poll_cluster, ctx, cid, force=False, refresh_views=False) for cid in clusters
    }
    if futures:
        wait(list(futures.values()), timeout=max(0.1, deadline_s))
    for cid, fut in futures.items():
        ids = [r["attempt_id"] for r in active if r["cluster_id"] == cid]
        if not fut.done():
            out.stale = True
            out.notes.append(
                f"refresh of {cid} did not finish within {deadline_s:.0f}s; showing the last known state "
                "(the poll continues in the background)"
            )
            continue
        try:
            res = fut.result()
        except Exception as exc:  # noqa: BLE001 - reported as a stale refresh, never an empty success
            out.stale = True
            out.notes.append(f"refresh of {cid} failed: {exc}")
            continue
        if res.skipped_reason == "backoff":
            out.stale = True
            out.notes.append(
                f"polling {cid} is backing off after errors (until {res.next_poll_after}); showing the "
                "last known state"
            )
            continue
        if res.skipped_reason is not None:
            out.notes.append(
                f"{cid} was polled recently; showing cached state"
                + (f", next poll allowed at {res.next_poll_after}" if res.next_poll_after else "")
            )
            continue
        if not res.ok:
            out.stale = True
            out.notes.append(f"refresh of {cid} failed: {res.error}")
            continue
        # Gone from squeue while the sacct check was deferred by its rate limit: the only new
        # fact is "no longer queued" - its final state is unknown, so it is not refreshed.
        unresolved = [i for i in ids if i in res.deferred]
        for aid in unresolved:
            out.stale = True
            out.notes.append(
                f"{aid}: no longer in squeue; its accounting (sacct) check was deferred by the rate limit, "
                "so its final state is not known yet (the next poll asks sacct)"
            )
        out.notes.extend(n for n in res.notes[:5] if n not in out.notes)
        out.refreshed.extend(i for i in ids if i not in unresolved)
        out.observed_at = res.observed_at
    return out


def _submit_background(fn: Any, *args: Any, **kwargs: Any) -> Future[Any]:
    fut: Future[Any] = _REFRESH_POOL.submit(fn, *args, **kwargs)
    with _INFLIGHT_LOCK:
        _INFLIGHT.add(fut)
    fut.add_done_callback(_forget)
    return fut


def _forget(fut: Future[Any]) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(fut)


def wait_for_background(timeout: float = 10.0) -> bool:
    """Wait for refresh polls that outlived their caller's deadline (they still write the ledger).

    Call before closing the database (controller shutdown, tests). True when none is left.
    """
    with _INFLIGHT_LOCK:
        pending = list(_INFLIGHT)
    if not pending:
        return True
    _done, not_done = wait(pending, timeout=timeout)
    return not not_done


def poll_all(ctx: ServiceContext) -> None:
    """Periodic task: poll every registered cluster that has something to observe."""
    for cluster_id in ctx.clusters.ids():
        try:
            poll_cluster(ctx, cluster_id=cluster_id)
        except Exception:  # one broken cluster must not stop the others
            log.exception("poll of cluster %s failed", cluster_id)


# ---------------------------------------------------------------------------
# Remote reads (bounded) + live views
# ---------------------------------------------------------------------------


def _decode_tail(data: bytes, started_mid_file: bool) -> str:
    text = data.decode("utf-8", "replace")
    if started_mid_file and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


UNREACHABLE = "unreachable"  # problem prefix: the login node/filesystem did not answer in time


def _tail(
    transport: Transport, path: str, limit: int, *, timeout: float = 30.0
) -> tuple[str | None, str | None]:
    """(text, problem) - problem is 'missing' / 'unreachable: ...' (timeout or lost connection:
    further reads would hang the same way) / 'unavailable: ...'. Auth problems raise."""
    try:
        rr = transport.read_tail(path, limit, timeout=timeout)
    except FileNotFoundError:
        return None, "missing"
    except (TransportTimeout, ConnectionLost) as exc:
        return None, f"{UNREACHABLE}: {exc.message}"
    except TransportError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return None, f"unavailable: {exc.message}"
    except OSError as exc:
        return None, f"unavailable: {exc}"
    return _decode_tail(rr.data, rr.offset > 0), None


def read_log_tails(
    transport: Transport, layout: AttemptLayout, limit: int = LOG_TAIL_BYTES, *, timeout: float = 30.0
) -> tuple[str, str, list[str]]:
    """Bounded stdout/stderr tails for diagnosis (the stderr tail carries the wrapper's lines).
    Notes name each log that could not be read ('missing' / 'unreachable' / 'unavailable')."""
    notes: list[str] = []
    out, p1 = _tail(transport, layout.stdout, limit, timeout=timeout)
    if p1 is not None and p1.startswith(UNREACHABLE):
        return "", "", [f"stdout log {p1}", "stderr log not read (the cluster did not answer)"]
    err, p2 = _tail(transport, layout.stderr, limit, timeout=timeout)
    for name, problem in (("stdout", p1), ("stderr", p2)):
        if problem:
            notes.append(f"{name} log {problem}")
    return out or "", err or "", notes


def _stat(transport: Transport, paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
    try:
        return transport.stat(paths, timeout=timeout)
    except TransportError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return {}
    except OSError:
        return {}


def metrics_path(reader: Any, attempt: Mapping[str, Any], layout: AttemptLayout) -> str:
    """The metrics JSONL of an attempt: a declared ``role: metrics`` ``.jsonl`` output, else
    ``$HPCLAB_METRICS_PATH`` (``<output_dir>/metrics.jsonl``)."""
    run = reader.one("SELECT spec_json FROM runs WHERE run_id=?", (attempt["run_id"],))
    spec = jload(run["spec_json"], {}) if run else {}
    for out in (spec.get("outputs") or []) if isinstance(spec, dict) else []:
        if not isinstance(out, dict) or out.get("role") != "metrics" or out.get("kind", "file") != "file":
            continue
        path = str(out.get("path") or "")
        if not path.endswith(".jsonl"):
            continue
        path = path.replace("{output_dir}", layout.output_dir).replace(
            "{checkpoint_dir}", layout.checkpoint_dir
        )
        path = path.replace("{attempt_dir}", layout.dir).replace("{code_dir}", layout.work)
        if "{" in path:
            continue
        return path if path.startswith("/") else f"{layout.work}/{path}"
    return f"{layout.output_dir}/metrics.jsonl"


class LiveViews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    progress: Progress | None = None
    health: Health | None = None
    metrics_source: Literal["ok", "missing", "unavailable", "skipped"] = "skipped"
    telemetry_source: Literal["ok", "missing", "unavailable", "skipped"] = "skipped"
    unreachable: bool = False  # a read timed out / lost the connection: the other reads were skipped
    notes: list[str] = Field(default_factory=list)


def compute_live_views(
    ctx: ServiceContext,
    attempt: Mapping[str, Any],
    *,
    with_health: bool | None = None,
    timeout: float = 30.0,
) -> LiveViews:
    """Progress (metrics tail <= 16 KiB) and, for running attempts, advisory health (telemetry
    tails + log/metrics mtimes). Bounded remote reads (``timeout`` each); auth problems raise,
    others are notes. The first read that times out (e.g. a hung NFS mount) stops the others."""
    views = LiveViews()
    transport = ctx.clusters.transport(attempt["cluster_id"])
    layout = layout_for(attempt)
    now = ctx.clock.now().timestamp()
    mpath = metrics_path(ctx.db, attempt, layout)
    text, problem = _tail(transport, mpath, METRICS_TAIL_BYTES, timeout=timeout)
    if text is not None:
        views.metrics_source = "ok"
        views.progress = estimate_progress(
            text.splitlines(),
            elapsed_s=attempt.get("elapsed_s"),
            time_limit_s=attempt.get("time_limit_s"),
            now_epoch=now,
        )
    else:
        views.metrics_source = "missing" if problem == "missing" else "unavailable"
        if problem and problem != "missing":
            views.notes.append(f"metrics {problem}")
    running = (
        _state(attempt.get("scheduler_state")) == SchedulerState.RUNNING
        and attempt.get("phase") in OBSERVED_PHASES
    )
    if with_health is None:
        with_health = running
    if problem is not None and problem.startswith(UNREACHABLE):
        views.unreachable = True
        if with_health:
            views.telemetry_source = "unavailable"
            views.notes.append("telemetry not read (the cluster did not answer)")
        return views
    if not with_health:
        return views
    csv_text, p_csv = _tail(transport, layout.gpu_telemetry, TELEMETRY_TAIL_BYTES, timeout=timeout)
    if p_csv is not None and p_csv.startswith(UNREACHABLE):
        views.unreachable = True
        views.telemetry_source = "unavailable"
        views.notes.append(f"telemetry {p_csv}")
        return views
    hb_text, _ = _tail(transport, layout.heartbeat, HEARTBEAT_TAIL_BYTES, timeout=timeout)
    stats = _stat(transport, [layout.stdout, mpath], timeout=timeout)
    views.telemetry_source = (
        "ok" if csv_text is not None else ("missing" if p_csv == "missing" else "unavailable")
    )
    log_st, met_st = stats.get(layout.stdout), stats.get(mpath)
    views.health = health_from(
        csv_text or "",
        hb_text or "",
        now=now,
        gpus_allocated=attempt.get("gpus") or None,
        log_mtime=log_st.mtime if log_st is not None and log_st.exists else None,
        metrics_mtime=met_st.mtime if met_st is not None and met_st.exists else None,
    )
    return views


def store_live_views(ctx: ServiceContext, attempt_id: str, views: LiveViews) -> None:
    """Cache derived views on the attempt (no version bump / event: not domain state)."""
    values: dict[str, Any] = {}
    if views.progress is not None:
        values["progress_json"] = jdump(views.progress.model_dump(mode="json"))
    if views.health is not None:
        values["health_json"] = jdump(views.health.model_dump(mode="json"))
    if not values:
        return
    with ctx.db.tx(source="observer") as tx:
        if tx.one("SELECT 1 AS x FROM attempts WHERE attempt_id=?", (attempt_id,)) is not None:
            tx.update("attempts", "attempt_id", attempt_id, values, bump_version=False)


def _refresh_running_views(ctx: ServiceContext, cluster_id: str) -> None:
    """After a poll: refresh cached progress/health of a few RUNNING attempts whose view is old."""
    rows = ctx.db.all(
        "SELECT * FROM attempts WHERE cluster_id=? AND phase IN (?, ?) AND scheduler_state='RUNNING' "
        "ORDER BY attempt_id",
        (cluster_id, *OBSERVED_PHASES),
    )
    now = ctx.clock.now().timestamp()
    done = 0
    for a in rows:
        if done >= MAX_LIVE_VIEWS_PER_POLL:
            break
        cached = jload(a["health_json"], None) or {}
        as_of = cached.get("as_of") if isinstance(cached, dict) else None
        if isinstance(as_of, (int, float)) and now - as_of < LIVE_VIEW_TTL_S:
            continue
        done += 1
        try:
            views = compute_live_views(ctx, a, timeout=BACKGROUND_READ_TIMEOUT_S)
        except HpcLabError as exc:
            log.debug("live views of %s skipped: %s", a["attempt_id"], exc.message)
            return  # auth / transport trouble: the next poll tries again
        store_live_views(ctx, a["attempt_id"], views)
        if views.unreachable:
            return  # the other attempts' reads would hang the same way


# ---------------------------------------------------------------------------
# Facts for diagnosis
# ---------------------------------------------------------------------------


def _cluster_or_none(ctx: ServiceContext, cluster_id: str) -> ResolvedCluster | None:
    try:
        return ctx.clusters.get(cluster_id)
    except HpcLabError:
        return None


def attempt_facts(
    reader: Any,
    ctx: ServiceContext,
    attempt: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> AttemptFacts:
    """``AttemptFacts`` from the ledger (exact field names; ``overrides`` win)."""
    plan = plan_dict(reader, attempt.get("exec_plan_hash"))
    res = sub_dict(plan, "resources")
    mem_bytes: int | None = None
    try:
        mem_bytes = ResourceSpec.model_validate(res).mem_bytes()
    except ValueError:
        mem_bytes = None
    partition = attempt.get("partition") or res.get("partition")
    cluster = _cluster_or_none(ctx, attempt["cluster_id"])
    rule = cluster.profile.partitions.get(partition) if cluster is not None and partition else None
    previous: list[str] = []
    for row in reader.all(
        "SELECT diagnosis_json FROM attempts WHERE run_id=? AND seq < ? ORDER BY seq",
        (attempt["run_id"], attempt.get("seq") or 0),
    ):
        diag = jload(row["diagnosis_json"], None)
        if isinstance(diag, dict) and diag.get("primary_code"):
            previous.append(str(diag["primary_code"]))
    progress = jload(attempt.get("progress_json"), None)
    fraction = progress.get("fraction") if isinstance(progress, dict) else None
    missing = [
        r["name"]
        for r in reader.all(
            "SELECT name FROM artifacts WHERE attempt_id=? AND required=1 AND integrity='missing' ORDER BY name",
            (attempt["attempt_id"],),
        )
    ]
    fields: dict[str, Any] = {
        "scheduler_state": attempt.get("scheduler_state"),
        "raw_state": attempt.get("raw_scheduler_state"),
        "exit_code": attempt.get("exit_code"),
        "exit_signal": attempt.get("exit_signal"),
        "app_state": attempt.get("app_state"),
        "reason": attempt.get("reason"),
        "elapsed_s": attempt.get("elapsed_s"),
        "time_limit_s": attempt.get("time_limit_s"),
        "mem_request_bytes": mem_bytes,
        "gpus": attempt.get("gpus"),
        "gpu_type": attempt.get("gpu_type") or res.get("gpu_type"),
        "node_list": attempt.get("node_list"),
        "restart_count": attempt.get("restart_count") or 0,
        "partition": partition,
        "partition_max_time_s": parse_slurm_duration(rule.max_time) if rule and rule.max_time else None,
        "preemptible": rule.preemptible if rule else None,
        "progress_fraction": min(1.0, max(0.0, float(fraction)))
        if isinstance(fraction, (int, float))
        else None,
        "missing_outputs": tuple(missing[:50]),
        "result_state": attempt.get("result_state"),
        "previous_codes": tuple(previous),
    }
    fields.update(overrides or {})
    return AttemptFacts.model_validate(fields)


# ---------------------------------------------------------------------------
# observe.finalize
# ---------------------------------------------------------------------------


def _read_json(transport: Transport, path: str, limit: int) -> tuple[dict[str, Any] | None, str | None]:
    """(document, problem): problem in missing | too_large | invalid | unreachable: ... (timeout,
    lost connection) | unavailable: ..."""
    try:
        rr = transport.read_range(path, 0, limit + 1)
    except FileNotFoundError:
        return None, "missing"
    except (TransportTimeout, ConnectionLost) as exc:
        return None, f"{UNREACHABLE}: {exc.message}"
    except TransportError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED:
            raise
        return None, f"unavailable: {exc.message}"
    except OSError as exc:
        return None, f"unavailable: {exc}"
    if rr.size > limit:
        return None, "too_large"
    try:
        doc = jload(rr.data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, "invalid"
    return (doc, None) if isinstance(doc, dict) else (None, "invalid")


def app_state_from_status(
    status: Mapping[str, Any] | None, problem: str | None, attempt: Mapping[str, Any]
) -> tuple[AppState, str | None]:
    """App state from the wrapper's status.json (None/'missing' = the wrapper never started)."""
    if status is None:
        if problem == "missing":
            return AppState.NOT_STARTED, "status.json was never written (the wrapper did not start)"
        return AppState.UNKNOWN, f"status.json {problem}"
    if status.get("attempt_id") not in (None, attempt["attempt_id"]):
        return AppState.UNKNOWN, "status.json belongs to another attempt"
    written_by, bound = str(status.get("job_id") or ""), str(attempt.get("job_id") or "")
    if written_by and bound and written_by != bound:
        # e.g. a duplicate submission of the same attempt ran it: that job's outcome is not ours
        return AppState.UNKNOWN, f"status.json was written by Slurm job {written_by}, not by job {bound}"
    restart = status.get("restart_count")
    if isinstance(restart, int) and restart < int(attempt.get("restart_count") or 0):
        return AppState.UNKNOWN, "status.json is from an earlier requeue segment"
    state = status.get("state")
    if state == "started":
        return AppState.KILLED, "the wrapper recorded a start but no end (killed, node lost or timed out)"
    if state == "killed":
        return AppState.KILLED, None
    if state == "exited":
        if status.get("guard_failed"):
            return AppState.GUARD_FAILED, None
        code = status.get("exit_code")
        if code == 0:
            return AppState.EXITED_OK, None
        if isinstance(code, int):
            return AppState.EXITED_ERROR, None
    return AppState.UNKNOWN, f"unrecognised status.json state {state!r}"


def _last_sacct_obs(attempt: Mapping[str, Any]) -> JobObservation | None:
    doc = jload(attempt.get("last_observation_json"), None)
    if not isinstance(doc, dict) or doc.get("source") != "sacct":
        return None
    try:
        return JobObservation.model_validate(doc)
    except ValueError:
        return None


def _gpu_count(
    reader: Any, attempt: Mapping[str, Any], acct: JobObservation | None
) -> tuple[int | None, str]:
    """(GPUs the job held | None = unknown, where the count came from).

    Accounting's allocation first (the adapter reports None, never 0, when the cluster does not
    track GPU TRES); else the request recorded on the attempt; else the planned GPUs of the
    execution plan. With none of them readable the count is unknown - never 0, so a GPU job is
    never settled as 0 GPU-hours."""
    if acct is not None and acct.alloc_gpus is not None:
        return acct.alloc_gpus, "allocated GPUs from accounting"
    if attempt.get("gpus"):
        return int(attempt["gpus"]), "GPU count from the request (accounting does not report GPUs)"
    res = plan_dict(reader, attempt.get("exec_plan_hash")).get("resources")
    if not isinstance(res, Mapping):
        return None, (
            "unknown GPU count: accounting does not report the allocation's GPUs and no execution plan "
            "records the request"
        )
    try:
        spec = ResourceSpec.model_validate(dict(res))
    except ValueError:
        return None, (
            "unknown GPU count: accounting does not report the allocation's GPUs and the execution "
            "plan's resources could not be read"
        )
    return (
        spec.total_gpus,
        "planned GPUs from the execution plan (accounting does not report the allocation's GPUs)",
    )


def settlement(
    reader: Any, attempt: Mapping[str, Any], acct: JobObservation | None
) -> tuple[float | None, str]:
    """(consumed GPU-hours | None = unknown, note). Accounting elapsed only - never a guess."""
    if acct is None or acct.elapsed_s is None:
        return None, "unknown: no accounting record for the job"
    gpus, gpu_note = _gpu_count(reader, attempt, acct)
    total_s = float(acct.elapsed_s)
    restart = int(attempt.get("restart_count") or 0)
    if restart > 0:
        segs = reader.all(
            "SELECT segment, start_time, end_time FROM attempt_segments WHERE attempt_id=? AND segment < ?"
            " ORDER BY segment",
            (attempt["attempt_id"], restart),
        )
        if len(segs) < restart:
            return None, "unknown: earlier requeue segments were not all observed"
        for s in segs:
            start, end = parse_iso(s["start_time"]), parse_iso(s["end_time"])
            if start is None or end is None:
                return None, f"unknown: requeue segment {s['segment']} has no start/end time"
            total_s += max(0.0, (end - start).total_seconds())
    if gpus is None:
        if total_s <= 0:
            return 0.0, f"0 s: the job never ran ({gpu_note})"
        return None, f"unknown: {int(total_s)} s x ? GPU(s) ({gpu_note})"
    hours = total_s * gpus / 3600.0
    return round(hours, 6), f"{int(total_s)} s x {gpus} GPU(s) ({gpu_note})"


def _settling_row(tx: Tx, attempt_id: str, restart: int) -> Row | None:
    """The attempt inside a settling transaction, None unless it is still the terminal segment
    ``restart`` (a requeue since then has its own finalize)."""
    cur = tx.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    if (
        cur is None
        or cur["phase"] != AttemptPhase.TERMINAL.value
        or int(cur["restart_count"] or 0) != restart
    ):
        return None
    return cur


def _apply_final_accounting(tx: Tx, ctx: ServiceContext, cur: Row, acct: JobObservation | None) -> Row | None:
    """Apply the final accounting record (exit code, elapsed, revised state); the re-read row, or
    None when it showed a later requeue segment."""
    if acct is None:
        return cur
    aid, restart = cur["attempt_id"], int(cur["restart_count"] or 0)
    apply_observation(tx, ctx, attempt_id=aid, obs=acct.model_copy(update={"observed_at": tx.now}))
    return _settling_row(tx, aid, restart)


def _finalized(reader: Any, attempt_id: str, restart: int) -> bool:
    row = reader.one(
        "SELECT 1 AS x FROM events WHERE entity_type='attempt' AND entity_id=? AND event_type='attempt.finalized'"
        " AND json_extract(payload_json, '$.restart_count') = ? LIMIT 1",
        (attempt_id, restart),
    )
    return row is not None


_NODE_CLASSES = {"node_fail": "node_fail", "nfs_stale": "nfs", "gpu_hardware_error": "gpu_hw"}
_HOSTLIST_RE = re.compile(r"^(?P<prefix>[^\[\]]*)\[(?P<body>[^\]]+)\](?P<suffix>[^\[\]]*)$")


def expand_hostlist(value: str | None, limit: int = 16) -> list[str]:
    """Expand ``a-[01-03,07],b`` style node lists (one bracket group per entry; bounded)."""
    if not value:
        return []
    entries: list[str] = []
    depth, cur = 0, ""
    for ch in value:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            entries.append(cur)
            cur = ""
        else:
            cur += ch
    entries.append(cur)
    out: list[str] = []
    for entry in (e.strip() for e in entries if e.strip()):
        m = _HOSTLIST_RE.match(entry)
        if not m:
            out.append(entry)
            continue
        for part in m.group("body").split(","):
            lo, _, hi = part.partition("-")
            if hi and lo.isdigit() and hi.isdigit():
                for n in range(int(lo), min(int(hi), int(lo) + limit) + 1):
                    out.append(f"{m.group('prefix')}{str(n).zfill(len(lo))}{m.group('suffix')}")
            else:
                out.append(f"{m.group('prefix')}{part}{m.group('suffix')}")
        if len(out) >= limit:
            break
    return out[:limit]


def _node_observations(
    tx: Tx, attempt: Mapping[str, Any], diagnosis: Diagnosis | None, now: datetime
) -> list[str]:
    """Time-limited observations about nodes (TTL 72 h) that recovery may use for ``--exclude``."""
    found: dict[tuple[str, str], list[str]] = {}
    nodes = expand_hostlist(attempt.get("node_list"))
    if attempt.get("scheduler_state") == SchedulerState.NODE_FAIL.value:
        cls = "node_fail" if len(nodes) == 1 else "node_fail_suspect"
        for node in nodes:
            found[(node, cls)] = ["Slurm state NODE_FAIL"]
    for item in diagnosis.items[:3] if diagnosis else []:
        base = _NODE_CLASSES.get(item.code)
        if base is None or item.confidence == "low":
            continue
        hinted = item.resource_hint.get("exclude_nodes") or item.resource_hint.get("suspect_nodes")
        targets = expand_hostlist(hinted) if hinted else nodes
        kind = f"{base}_suspect" if len(targets) > 1 and not hinted else base
        for node in targets:
            found.setdefault((node, kind), list(item.evidence[:2]))
    keys: list[str] = []
    expires = iso(now + timedelta(seconds=NODE_OBS_TTL_S))
    for (node, cls), evidence in sorted(found.items()):
        key = f"node:{node}:{cls}:{attempt['attempt_id']}"
        tx.execute(
            "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at, expires_at)"
            " VALUES (?, ?, ?, 'observe.finalize', ?, ?)",
            (
                attempt["cluster_id"],
                key,
                jdump(
                    {
                        "node": node,
                        "class": cls,
                        "attempt_id": attempt["attempt_id"],
                        "run_id": attempt["run_id"],
                        "job_id": attempt.get("job_id"),
                        "evidence": evidence,
                    }
                ),
                tx.now,
                expires,
            ),
        )
        keys.append(key)
    return keys


def _needs_diagnosis(state: SchedulerState, app: AppState) -> bool:
    return state != SchedulerState.COMPLETED or app in (
        AppState.EXITED_ERROR,
        AppState.GUARD_FAILED,
        AppState.KILLED,
        AppState.NOT_STARTED,
    )


def finalize_attempt(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """Outbox ``observe.finalize`` {attempt_id}: the terminal bookkeeping of one attempt.

    Crash-safe: every remote read happens before one final transaction, which is skipped when an
    ``attempt.finalized`` event for this attempt + restart already exists.
    """
    aid = str(item.payload.get("attempt_id") or item.target_id or "")
    a = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (aid,))
    if a is None:
        return HandlerResult.done(f"attempt {aid} not found")
    if a["phase"] != AttemptPhase.TERMINAL.value:
        return HandlerResult.done("attempt is not terminal any more (requeued)")
    restart = int(a["restart_count"] or 0)
    if _finalized(ctx.db, aid, restart):
        return HandlerResult.done("already finalized")
    cluster = ctx.clusters.get(a["cluster_id"])
    transport = ctx.clusters.transport(a["cluster_id"])
    notes: list[str] = []
    ended = parse_iso(a["terminal_at"]) or ctx.clock.now()

    # 1. accounting: exit code + elapsed (authoritative for budget settlement). Batched with the
    #    other attempts waiting on it and rate-limited by the cluster's shared sacct slot.
    acct = _final_sacct(a)
    acct_state = "ok" if acct is not None else "not_queried"
    notes_acct: str | None = None
    if acct is None:
        check = _accounting_check(ctx, cluster, a)
        acct, acct_state = check.record, check.state
        wait = _accounting_wait(ctx, cluster, check, ended, item.tries)
        if wait is not None:
            return wait
        if check.note and acct_state != "ok":
            notes_acct = check.note[:300]
            notes.append(check.note)
    # A requeue seen only as RUNNING -> PENDING(restart+1) leaves the earlier segment without an
    # end time: per-segment accounting settles it exactly.
    chain: list[JobObservation] | None = None
    chain_retry = False
    if acct is not None and restart > 0 and not _segments_complete(ctx.db, aid, restart):
        chain, chain_note, chain_retry = _settlement_chain(ctx, a, restart)
        if chain_note:
            notes.append(chain_note)

    # 2. wrapper files. An unreachable login node is waited out (bounded by time, not tries).
    layout = layout_for(a)
    status, status_problem = _read_json(transport, layout.status_json, STATUS_MAX_BYTES)
    env, env_problem = _read_json(transport, layout.env_json, ENV_MAX_BYTES)
    transient = [p for p in (status_problem, env_problem) if p and p.startswith(("unavailable", UNREACHABLE))]
    unreachable = any(p.startswith(UNREACHABLE) for p in transient)
    if transient and (
        item.tries < FINALIZE_MAX_TRIES
        or (unreachable and ctx.clock.now() < ended + timedelta(seconds=UNREACHABLE_WAIT_S))
    ):
        return HandlerResult.retry(min(300.0, 5.0 * 2 ** min(item.tries, 6)), note=transient[0])
    outputs_stat = _stat(transport, [layout.outputs_manifest]).get(layout.outputs_manifest)
    app, app_note = app_state_from_status(status, status_problem, a)
    if app_note:
        notes.append(app_note)
    if env_problem and env_problem != "missing":
        notes.append(f"env.json {env_problem}")

    # 3. diagnosis over bounded log tails (failures only)
    final_state = (
        acct.state if acct is not None and acct.state in SCHEDULER_TERMINAL else _state(a["scheduler_state"])
    )
    exit_code = acct.exit_code if acct is not None and acct.exit_code is not None else a["exit_code"]
    diagnosis: Diagnosis | None = None
    owner_job: str | None = None
    stdout_tail = stderr_tail = ""
    if _needs_diagnosis(final_state, app) or exit_code == EXIT_DUPLICATE_JOB:
        stdout_tail, stderr_tail, log_notes = read_log_tails(transport, layout)
        notes.extend(log_notes)
        owner_job = duplicate_owner(stderr_tail, a["job_id"])
        if owner_job is not None:
            notes.append(
                f"Slurm job {owner_job} owns this attempt; job {a['job_id']} was a duplicate submission "
                f"(exit {exit_code}). hpclab never cancels either job automatically"
            )
    if owner_job is None and _needs_diagnosis(final_state, app):
        overrides: dict[str, Any] = {"app_state": app, "scheduler_state": final_state.value}
        if acct is not None:
            for key in ("exit_code", "exit_signal", "elapsed_s", "node_list", "time_limit_s"):
                if getattr(acct, key) is not None:
                    overrides[key] = getattr(acct, key)
            overrides["raw_state"] = acct.raw_state
        diagnosis = diagnose(attempt_facts(ctx.db, ctx, a, overrides=overrides), stdout_tail, stderr_tail)

    # 4. one transaction: apply accounting, record, settle, follow-ups
    with ctx.db.tx(source="worker") as tx:
        cur = _settling_row(tx, aid, restart)
        if cur is None:
            return HandlerResult.done("attempt changed while finalizing")
        if _finalized(tx, aid, restart):
            return HandlerResult.done("already finalized")
        cur = _apply_final_accounting(tx, ctx, cur, acct)
        if cur is None:
            # accounting showed a later requeue segment: its own finalize item takes over
            return HandlerResult.done("accounting shows a later requeue segment")
        if chain:
            _record_chain_segments(tx, aid, cur["job_id"], chain, tx.now)
        if owner_job is not None:
            # Another Slurm job runs this attempt: the bound job's few seconds say nothing about
            # what the attempt consumes. The reservation stays unknown at its full amount (an
            # upper bound: the owner runs the same script under the same limits).
            consumed, settle_note = (
                None,
                f"unknown: Slurm job {owner_job} runs this attempt (job {cur['job_id']} was a duplicate); "
                "counted at the full reservation",
            )
        else:
            consumed, settle_note = settlement(tx, cur, acct)
            if acct is None:
                settle_note = f"unknown: accounting {acct_state}" + (f" ({notes_acct})" if notes_acct else "")
        late = (
            consumed is None
            and owner_job is None
            and (acct is None or chain_retry)
            and acct_state != "unsupported"
        )
        values: dict[str, Any] = {"app_state": app.value}
        if status is not None:
            values["status_json"] = jdump(status)
        if env is not None:
            values["env_json"] = jdump(env)
        elif env_problem == "too_large":
            values["env_json"] = jdump({"schema": "hpclab.env/v1", "unreadable": "too_large"})
        if diagnosis is not None:
            values["diagnosis_json"] = jdump(diagnosis.model_dump(mode="json"))
        # app state and diagnosis steer what a user does next (recover, derive, fix); the
        # status.json / env.json copies are evidence only
        decided = cur["app_state"] != app.value or (
            "diagnosis_json" in values and values["diagnosis_json"] != cur["diagnosis_json"]
        )
        tx.update("attempts", "attempt_id", aid, values, bump_version=decided)
        version = int(cur["version"]) + (1 if decided else 0)
        budget_note = _settle_budget(tx, aid, consumed, settle_note)
        node_keys = _node_observations(tx, cur, diagnosis, ctx.clock.now())
        followups = _enqueue_followups(tx, cur, restart)
        if late:
            # finalized provisionally: accounting (or its per-segment records) is still asked
            # for, so an outage or a slow slurmdbd never becomes a permanent full-reservation charge
            tx.enqueue(
                KIND_RESETTLE,
                target_id=aid,
                payload={"attempt_id": aid, "restart_count": restart},
                delay_s=RESETTLE_FIRST_DELAY_S,
                dedupe_key=f"{KIND_RESETTLE}:{aid}:{restart}",
            )
            followups.append(KIND_RESETTLE)
        else:
            _forget_accounting_check(tx, cur)
        tx.event(
            "attempt",
            aid,
            "attempt.finalized",
            {
                "restart_count": restart,
                "scheduler_state": cur["scheduler_state"],
                "app_state": app.value,
                "exit_code": cur["exit_code"],
                "accounting": acct_state,
                "consumed_gpu_hours": consumed,
                "settlement": settle_note,
                "budget": budget_note,
                "outputs_manifest": "present"
                if outputs_stat is not None and outputs_stat.exists
                else "absent",
                "diagnosis": diagnosis.primary_code if diagnosis else None,
                "duplicate_of_job_id": owner_job,
                "node_observations": node_keys,
                "followups": followups,
                "notes": notes[:10],
            },
            entity_version=version,
            source="worker",
        )
        if owner_job is not None:
            _record_duplicate(tx, cur, owner_job, exit_code)
        else:
            _notify_terminal(tx, cur, app, diagnosis)
        refresh_run_status(tx, cur["run_id"])
    return HandlerResult.done()


def _attempt_event(reader: Any, attempt_id: str, event_type: str, restart: int) -> bool:
    row = reader.one(
        "SELECT 1 AS x FROM events WHERE entity_type='attempt' AND entity_id=? AND event_type=?"
        " AND json_extract(payload_json, '$.restart_count') = ? LIMIT 1",
        (attempt_id, event_type, restart),
    )
    return row is not None


def resettle_attempt(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """Outbox ``observe.settle`` {attempt_id, restart_count}: late settlement of an attempt that
    finalize had to settle as *unknown* (accounting unavailable or not there yet, per-segment
    accounting of a requeue unreadable).

    Asks accounting again - batched with every other attempt waiting on it and within the
    cluster's shared sacct rate limit - with a growing delay, for ``RESETTLE_HORIZON_S``. When the
    final record arrives it is applied (exit code, elapsed, revised state) and the budget is
    settled from it; after the horizon the attempt stays unknown (counted at its reservation)
    and an ``attempt.accounting_unresolved`` event says so. A requeue since then is settled by
    the new segment's own finalize."""
    aid = str(item.payload.get("attempt_id") or item.target_id or "")
    restart = int(item.payload.get("restart_count") or 0)
    a = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (aid,))
    if a is None:
        return HandlerResult.done(f"attempt {aid} not found")
    if a["phase"] != AttemptPhase.TERMINAL.value or int(a["restart_count"] or 0) != restart:
        return HandlerResult.done("attempt was requeued: the new segment's finalize settles it")
    if _attempt_event(ctx.db, aid, "attempt.accounting_settled", restart):
        return HandlerResult.done("already settled")
    cluster = ctx.clusters.get(a["cluster_id"])
    ended = parse_iso(a["terminal_at"]) or ctx.clock.now()
    acct = _final_sacct(a)
    state, why = "ok", None
    if acct is None:
        check = _accounting_check(ctx, cluster, a)
        acct, state, why = check.record, check.state, check.note
    chain: list[JobObservation] | None = None
    if acct is not None and restart > 0 and not _segments_complete(ctx.db, aid, restart):
        chain, why, retry_chain = _settlement_chain(ctx, a, restart)
        if chain is None and not retry_chain:
            state = "segments_unverified"
    if acct is not None:
        with ctx.db.tx(source="worker") as tx:
            cur = _settling_row(tx, aid, restart)
            if cur is None:
                return HandlerResult.done("attempt was requeued: the new segment's finalize settles it")
            cur = _apply_final_accounting(tx, ctx, cur, acct)
            if cur is None:
                return HandlerResult.done("accounting shows a later requeue segment")
            if chain:
                _record_chain_segments(tx, aid, cur["job_id"], chain, tx.now)
            consumed, settle_note = settlement(tx, cur, acct)
            if consumed is not None:
                budget_note = _settle_budget(tx, aid, consumed, f"{settle_note} (settled late)")
                _forget_accounting_check(tx, cur)
                tx.event(
                    "attempt",
                    aid,
                    "attempt.accounting_settled",
                    {
                        "restart_count": restart,
                        "scheduler_state": cur["scheduler_state"],
                        "exit_code": cur["exit_code"],
                        "consumed_gpu_hours": consumed,
                        "settlement": settle_note,
                        "budget": budget_note,
                    },
                    source="worker",
                )
                refresh_run_status(tx, cur["run_id"])
                return HandlerResult.done()
            why = why or settle_note
    if state in ("unsupported", "segments_unverified") or ctx.clock.now() >= ended + timedelta(
        seconds=RESETTLE_HORIZON_S
    ):
        with ctx.db.tx(source="worker") as tx:
            if _settling_row(tx, aid, restart) is not None and not _attempt_event(
                tx, aid, "attempt.accounting_unresolved", restart
            ):
                tx.event(
                    "attempt",
                    aid,
                    "attempt.accounting_unresolved",
                    {
                        "restart_count": restart,
                        "accounting": state,
                        "note": (why or "")[:300],
                        "budget": "stays unknown: counted at its reservation, never as 0",
                    },
                    source="worker",
                )
        return HandlerResult.done(f"accounting unresolved ({state})")
    delay = min(RESETTLE_MAX_DELAY_S, RESETTLE_FIRST_DELAY_S * 2 ** min(max(item.tries - 1, 0), 4))
    return HandlerResult.retry(
        delay + random.random(),  # noqa: S311 - jitter only
        note=f"late settlement waits for accounting ({state})" + (f": {why}" if why else ""),
    )


_DUPLICATE_RE = re.compile(re.escape(DUPLICATE_MARKER) + r"(\S+)")


def duplicate_owner(stderr_tail: str, job_id: str | None) -> str | None:
    """The Slurm job that owns the attempt dir when it is not ``job_id`` (the wrapper of a
    duplicate submission prints ``HPCLAB_DUPLICATE_JOB:<owner>`` and exits 95 untouched)."""
    owners = [m.group(1) for m in _DUPLICATE_RE.finditer(stderr_tail or "")]
    others = [o for o in owners if o != str(job_id or "")]
    return others[-1][:64] if others else None


def _record_duplicate(tx: Tx, attempt: Mapping[str, Any], owner_job: str, exit_code: int | None) -> None:
    """Another Slurm job runs this attempt unobserved: event + immediate notification (a human
    decides; hpclab never cancels a job whose identity it did not bind)."""
    tx.event(
        "attempt",
        attempt["attempt_id"],
        "attempt.duplicate_job",
        {"job_id": attempt["job_id"], "owner_job_id": owner_job, "exit_code": exit_code},
        source="worker",
    )
    emit = _optional_function("hpclab.services.notify", "emit")
    if emit is None:
        return
    run = tx.one("SELECT label FROM runs WHERE run_id=?", (attempt["run_id"],))
    emit(
        tx,
        event_type="attempt.uncertain",
        entity_id=attempt["run_id"],
        title=f"Run {run['label'] if run else attempt['run_id']}: job {attempt['job_id']} was a duplicate",
        body=(
            f"attempt {attempt['attempt_id']} on {attempt['cluster_id']}: Slurm job {owner_job} owns the "
            f"attempt and is not tracked by hpclab; decide with run_inspect before resubmitting"
        ),
        urgency="immediate",
        dedupe_key=f"attempt.duplicate_job:{attempt['attempt_id']}:{owner_job}",
    )


def _settle_budget(tx: Tx, attempt_id: str, consumed: float | None, note: str) -> str:
    settle = _optional_function("hpclab.services.budget", "settle")
    if settle is None:
        return "budget module unavailable"
    try:
        state = settle(tx, attempt_id=attempt_id, consumed_gpu_hours=consumed, note=note)
    except NotFound:
        return "no reservation"
    return str(state) if state else ("settled" if consumed is not None else "unknown")


def _enqueue_followups(tx: Tx, attempt: Mapping[str, Any], restart: int) -> list[str]:
    aid = attempt["attempt_id"]
    kinds = ["artifacts.index", "vault.capture"]
    for kind in kinds:
        tx.enqueue(kind, target_id=aid, payload={"attempt_id": aid}, dedupe_key=f"{kind}:{aid}:{restart}")
    wf = attempt.get("workflow_id")
    if wf:
        tx.enqueue(
            "workflow.dispatch",
            target_id=wf,
            payload={"workflow_id": wf},
            dedupe_key=f"workflow.dispatch:{wf}",
        )
        kinds.append("workflow.dispatch")
    return kinds


def _notify_terminal(tx: Tx, attempt: Mapping[str, Any], app: AppState, diagnosis: Diagnosis | None) -> None:
    emit = _optional_function("hpclab.services.notify", "emit")
    if emit is None:
        return
    run = tx.one("SELECT label FROM runs WHERE run_id=?", (attempt["run_id"],))
    label = run["label"] if run else attempt["run_id"]
    state = attempt["scheduler_state"]
    if state == SchedulerState.COMPLETED.value and app == AppState.EXITED_OK:
        event_type, verb = "run.succeeded", "completed (exit 0; outputs are being validated)"
    elif state == SchedulerState.CANCELLED.value:
        event_type, verb = "run.cancelled", "was cancelled"
    else:
        event_type, verb = "run.failed", f"ended {state}"
    body = (
        f"attempt {attempt['attempt_id']} (job {attempt['job_id']} on {attempt['cluster_id']}): "
        f"scheduler {state}, program {app.value}"
    )
    if diagnosis is not None and diagnosis.primary_code:
        body += f", diagnosis {diagnosis.primary_code}"
    emit(
        tx,
        event_type=event_type,
        entity_id=attempt["run_id"],
        title=f"Run {label} {verb}",
        body=body,
        urgency="batch",
        dedupe_key=f"{event_type}:{attempt['attempt_id']}:{attempt['restart_count'] or 0}",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def outbox_handlers() -> dict[str, Handler]:
    return {KIND_FINALIZE: finalize_attempt, KIND_RESETTLE: resettle_attempt}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    return [PeriodicTask(name="observe.poll_all", interval_s=POLL_TICK_S, fn=poll_all, jitter_s=1.0)]
