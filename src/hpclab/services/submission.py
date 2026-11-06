"""Submission, reconciliation and cancellation (design §6.2/§6.4, build spec §4.1-4.4, 4.7-4.9).

The promise of this module is *never blindly resubmit, never cancel the wrong job*.

Lifecycle of one Attempt (``AttemptPhase``)::

    admit ──> STAGING ──(submit.stage: snapshot, attempt dir, job.sbatch)──> submit.dispatch
                 │                                                             │
                 │ cancel before dispatch                 commit SUBMISSION_IN_FLIGHT (+dispatch_at)
                 v                                                  THEN the one sbatch call
       CANCELLED_BEFORE_SUBMIT            ┌───────────────┬──────────────────┴────────────┐
                                    job id reply    definite refusal         lost reply / timeout /
                                          │               │                  crash / re-entry
                                          v               v                         v
                                JOB_IDENTIFIED   SUBMISSION_REJECTED     SUBMISSION_UNCERTAIN
                                      -> OBSERVING   (claim + budget       (submit.reconcile:
                                                      released)             receipt -> squeue -> sacct)
                                                                  unique ─> bind │ >1 ─> DUPLICATE_DETECTED
                                                                  none ─> retry until the window,
                                                                          then operation BLOCKED 'unresolved'

Rules that make it safe:

* ``admit`` is THE admission gate for every entry point (``run_submit``, workflow dispatch,
  recovery): one execution claim per Run, a plan is consumed once, smoke gate, experiment /
  workflow slots, budget - all in the caller's transaction.
* Network I/O never happens inside a transaction: intent tx -> I/O -> outcome tx; every
  outcome tx re-reads the attempt, so concurrent actors (cancel, a re-entered handler) are
  serialized by SQLite's ``BEGIN IMMEDIATE``.
* ``submit.dispatch`` commits ``SUBMISSION_IN_FLIGHT`` BEFORE calling sbatch. Finding an
  attempt already in flight (crash, lease expiry) means the call may have happened: the
  attempt becomes ``SUBMISSION_UNCERTAIN`` and is reconciled - sbatch is never called twice.
* Only ``SubmissionRejected`` (a definite refusal) turns a dispatched attempt into
  ``SUBMISSION_REJECTED``; everything else after the in-flight commit is uncertain.
* Reconciliation binds only a UNIQUE job whose owner is the cluster user, whose job name /
  comment carry this attempt's token and which was submitted after ``dispatch_at - skew``.
  Nothing found means *unknown*, never "not submitted": after the reconcile window the
  submit operation is ``blocked`` (claim and budget kept) until a human runs
  ``attempt_resolve``.
* Cancel re-verifies the live identity (job id, owner, token, submit time, restart count)
  right before ``scancel``; any mismatch is ``REFUSED`` with zero scheduler calls.
* ``AuthRequired`` (MFA expired) and non-retryable transport problems (ssh misconfiguration,
  full quota) block the operation (``auth_required`` / ``cluster_access``) with a remediation
  and retry later; the attempt state is untouched. Transient errors are retried with backoff.
* A *permanent* staging failure (the stored plan cannot render, the snapshot is gone locally
  and remotely) happens before anything reaches the scheduler, so the attempt ends as
  ``SUBMISSION_REJECTED`` with reason ``stage_failed:...`` and frees its claim and budget.
"""

from __future__ import annotations

import importlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    SubmissionRejected,
    SubmitResponse,
    TransportError,
)
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import (
    AdmissionDenied,
    ErrorCode,
    HpcLabError,
    InvalidArgument,
    NotFound,
    NotReady,
    PreconditionFailed,
    Unsupported,
    VersionConflict,
)
from hpclab.domain.hashing import digest, sha256_hex
from hpclab.domain.ids import new_id, submit_token
from hpclab.domain.models import ExecutionPlan, JobObservation, ResourceSpec, RunSpec
from hpclab.domain.states import (
    CLAIM_HOLDING_PHASES,
    FINAL_PHASES,
    HUMAN_ONLY_TRANSITIONS,
    SCHEDULER_TERMINAL,
    AttemptPhase,
    CancelState,
    can_transition,
)
from hpclab.profiles.schema import ResolvedCluster, SitePolicy
from hpclab.services import budget
from hpclab.services.context import ServiceContext
from hpclab.services.jobscript import (
    EXIT_DUPLICATE_JOB,
    AttemptLayout,
    attempt_layout,
    job_name_for,
    render_job_script,
    token_from_comment,
    token_from_job_name,
)
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.services.snapshot import SnapshotStore, publish_snapshot
from hpclab.storage.db import Tx, jdump, jload

if TYPE_CHECKING:
    from hpclab.adapters.base import Scheduler
    from hpclab.services.periodic import PeriodicTask

log = logging.getLogger("hpclab.submit")

# Outbox kinds owned by this module (build spec §3).
KIND_STAGE = "submit.stage"
KIND_DISPATCH = "submit.dispatch"
KIND_RECONCILE = "submit.reconcile"
KIND_CANCEL = "cancel.execute"
SWEEP_TASK = "submit.reconcile_sweep"

# Idempotency tool names (principal, tool, key).
TOOL_SUBMIT = "run_submit"
TOOL_CANCEL = "run_cancel"
TOOL_RESOLVE = "attempt_resolve"

MARKER_NAME = ".hpclab-complete"  # snapshot completion marker (services.snapshot)
_MKDIR_SCRIPT = 'set -eu\nfor d in "$@"; do mkdir -p -- "$d"; done\n'

# Tunables; each can be overridden with ``ctx.settings.extra["submission.<name>"]``.
DEFAULTS: dict[str, float] = {
    "poll_after_ms": 2000,
    "reconcile_first_delay_s": 2.0,  # squeue needs a moment after a lost reply
    "reconcile_base_delay_s": 5.0,
    "reconcile_max_delay_s": 60.0,
    "submit_time_skew_s": 600.0,  # controller <-> cluster clock skew tolerated in identity checks
    "auth_retry_s": 120.0,
    "host_key_retry_s": 900.0,
    "stale_in_flight_s": 600.0,  # in-flight without a live dispatch item -> uncertain (sweep)
    "cancel_retry_s": 5.0,
    "cancel_max_tries": 12.0,  # job invisible in squeue and sacct this many times -> cancel failed
    "sweep_interval_s": 60.0,
    "owner_check_s": 600.0,  # an unconfirmed duplicate-owner job is looked up at most this often
    "owner_checks_per_sweep": 4.0,
}

_ACTIVE_BOUND = (AttemptPhase.JOB_IDENTIFIED, AttemptPhase.OBSERVING)
_UNRESOLVED = (AttemptPhase.SUBMISSION_UNCERTAIN, AttemptPhase.DUPLICATE_DETECTED)
_OP_FINAL = ("succeeded", "failed")
MAX_CANDIDATES = 50
SUBMIT_TIME_TOLERANCE_S = 2.0  # recorded vs live SubmitTime of the same job (rounding)

# Execution holds (a Run stays claimed while an original job may still run; see ``execution_holds``).
EVENT_DUPLICATE_JOB = "attempt.duplicate_job"  # observe.finalize: another job owns the attempt dir
EVENT_END_CONFIRMED = "attempt.end_confirmed"  # confirm_ended, or the sweep's scheduler evidence
OWNER_CHECK_PREFIX = "owner_check:"  # observations cache: the last live look at a duplicate owner
HoldKind = Literal["requeue_watch", "duplicate_owner"]
Resolution = Literal["abandon", "bind_job", "confirm_ended"]
_RESOLUTIONS: tuple[str, ...] = ("abandon", "bind_job", "confirm_ended")


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


class AdmitResult(BaseModel):
    """What ``admit`` created (attempt in STAGING, submit operation, budget reservation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str
    operation_id: str
    reserved_gpu_hours: float
    seq: int
    job_name: str


class SubmitReceiptData(BaseModel):
    """Receipt of ``run_submit``. A receipt is NOT scheduler acceptance: poll the operation."""

    model_config = ConfigDict(extra="forbid")

    operation_ref: str
    operation_id: str
    attempt_id: str
    run_id: str
    phase: str
    poll_after_ms: int
    reserved_gpu_hours: float

    def summary(self) -> str:
        return (
            f"attempt {self.attempt_id} admitted ({self.phase}); {self.reserved_gpu_hours:g} GPU-h reserved; "
            f"poll {self.operation_ref}"
        )


class CancelReceiptData(BaseModel):
    """Receipt of ``run_cancel``. ``cancel_state`` is ``requested``/``sent`` until an
    observation shows the job terminal; only a pre-dispatch cancel is final immediately."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    phase: str
    cancel_state: str
    cancel_mode: str | None = None
    attempt_version: int
    operation_ref: str | None = None
    operation_id: str | None = None
    poll_after_ms: int | None = None
    note: str = ""

    def summary(self) -> str:
        return f"attempt {self.attempt_id}: cancel {self.cancel_state} ({self.phase}). {self.note}".strip()


class AttemptResolveData(BaseModel):
    """Result of a human resolution: an uncertain/duplicate attempt bound or abandoned, an
    unobservable OBSERVING attempt abandoned, or a terminal attempt's job confirmed ended."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    resolution: Resolution
    phase: str
    job_id: str | None = None
    attempt_version: int
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        if self.resolution == "bind_job":
            what = f"bound to job {self.job_id}"
        elif self.resolution == "confirm_ended":
            what = "confirm_ended: its job's end is confirmed, the run may execute again"
        else:
            what = "abandoned"
        return f"attempt {self.attempt_id} {what} ({self.phase})"


class ExecutionHold(BaseModel):
    """Why a Run stays claimed although no attempt holds its claim: a Slurm job that ran (or
    runs) one of its attempts may still execute. ``until`` is when the hold ends by itself
    (None: no time bound - it ends when squeue/sacct show the job ended, which the reconcile
    sweep checks, or by a human ``attempt_resolve(resolution='confirm_ended')``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: HoldKind
    attempt_id: str
    job_id: str  # the Slurm job that may still run
    until: str | None = None
    message: str
    hint: str


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _setting(ctx: ServiceContext, name: str) -> float:
    value = ctx.settings.extra.get(f"submission.{name}", DEFAULTS[name])
    return float(value)


def operation_ref(operation_id: str) -> str:
    return f"hpclab://operations/{operation_id}"


def _require_key(key: str) -> None:
    if not isinstance(key, str) or not key.strip() or len(key) > 200:
        raise InvalidArgument("idempotency_key must be a non-empty string of at most 200 characters")


def _attempt(reader: Any, attempt_id: str) -> dict[str, Any] | None:
    row: dict[str, Any] | None = reader.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    return row


def _require_attempt(reader: Any, attempt_id: str) -> dict[str, Any]:
    row = _attempt(reader, attempt_id)
    if row is None:
        raise NotFound(f"attempt {attempt_id} not found", details={"attempt_id": attempt_id})
    return row


def _payload_attempt(item: OutboxItem) -> str:
    aid = item.payload.get("attempt_id") or item.target_id
    if not isinstance(aid, str) or not aid:
        raise InvalidArgument(f"outbox item {item.outbox_id} ({item.kind}) has no attempt_id")
    return aid


def _err(code: ErrorCode | str, message: str, *, retryable: bool = False, **details: Any) -> dict[str, Any]:
    return {"code": str(code), "message": message, "retryable": retryable, "details": details}


def _exc_error(exc: HpcLabError) -> dict[str, Any]:
    data = exc.to_dict()
    details = dict(data.get("details") or {})
    for key in ("stderr", "stdout"):
        if isinstance(details.get(key), str):
            details[key] = details[key][-2000:]
    data["details"] = details
    return data


def _transition(
    tx: Tx,
    row: dict[str, Any],
    dst: AttemptPhase,
    values: dict[str, Any],
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Versioned phase change + its event, in the caller's transaction."""
    src = AttemptPhase(row["phase"])
    if not can_transition(src, dst):
        raise HpcLabError(
            f"illegal attempt transition {src} -> {dst} for {row['attempt_id']}",
            details={"attempt_id": row["attempt_id"], "from": src.value, "to": dst.value},
        )
    if (src, dst) in HUMAN_ONLY_TRANSITIONS and tx.source != "human":
        raise HpcLabError(
            f"attempt transition {src} -> {dst} for {row['attempt_id']} is an explicit human decision only",
            details={
                "attempt_id": row["attempt_id"],
                "from": src.value,
                "to": dst.value,
                "source": tx.source,
            },
        )
    vals = {"phase": dst.value, **values}
    version = tx.update("attempts", "attempt_id", row["attempt_id"], vals, expected_version=row["version"])
    tx.event(
        "attempt",
        row["attempt_id"],
        event_type,
        {"from": src.value, "to": dst.value, **(payload or {})},
        entity_version=version,
    )
    new = dict(row)
    new.update(vals)
    new["version"] = version
    return new


def _update_attempt(
    tx: Tx,
    row: dict[str, Any],
    values: dict[str, Any],
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = tx.update("attempts", "attempt_id", row["attempt_id"], values, expected_version=row["version"])
    tx.event("attempt", row["attempt_id"], event_type, dict(payload or {}), entity_version=version)
    new = dict(row)
    new.update(values)
    new["version"] = version
    return new


def _op(
    tx: Tx,
    operation_id: str | None,
    *,
    state: str | None = None,
    stage: str | None = None,
    result: Any = None,
    error: dict[str, Any] | None = None,
    clear_error: bool = False,
) -> None:
    """Update an operation (+ event) only when something actually changes."""
    if not operation_id:
        return
    cur = tx.one("SELECT state, stage, error_json FROM operations WHERE operation_id=?", (operation_id,))
    if cur is None:
        return
    values: dict[str, Any] = {}
    if state is not None and state != cur["state"]:
        values["state"] = state
        if state in _OP_FINAL:
            values["finished_at"] = tx.now
    if stage is not None and stage != cur["stage"]:
        values["stage"] = stage
    if result is not None:
        values["result_json"] = jdump(result)
    if error is not None:
        encoded = jdump(error)
        if encoded != cur["error_json"]:
            values["error_json"] = encoded
    elif clear_error and cur["error_json"] is not None:
        values["error_json"] = None
    if not values:
        return
    version = tx.update("operations", "operation_id", operation_id, values)
    tx.event(
        "operation",
        operation_id,
        "operation.updated",
        {k: values[k] for k in ("state", "stage") if k in values},
        entity_version=version,
    )


def _set_run_status(tx: Tx, attempt: dict[str, Any], status: str) -> None:
    """Refresh the run's summary status (only when this attempt is its latest one)."""
    run = tx.one("SELECT latest_attempt_id, status FROM runs WHERE run_id=?", (attempt["run_id"],))
    if run is None or run["latest_attempt_id"] != attempt["attempt_id"] or run["status"] == status:
        return
    version = tx.update("runs", "run_id", attempt["run_id"], {"status": status})
    tx.event(
        "run",
        attempt["run_id"],
        "run.status_changed",
        {"from": run["status"], "to": status, "attempt_id": attempt["attempt_id"]},
        entity_version=version,
    )


def _release_claim(tx: Tx, attempt: dict[str, Any], *, run_status: str | None) -> None:
    wf = attempt.get("workflow_id")
    if wf:
        # The attempt ended without a terminal observation (rejected, cancelled before dispatch,
        # abandoned): its workflow slot is free now, so give the workflow a dispatch round at once
        # like observe.finalize does, instead of waiting for the periodic sweep.
        tx.enqueue(
            "workflow.dispatch",
            target_id=wf,
            payload={"workflow_id": wf},
            dedupe_key=f"workflow.dispatch:{wf}",
        )
    run = tx.one(
        "SELECT claim_attempt_id, latest_attempt_id, status FROM runs WHERE run_id=?", (attempt["run_id"],)
    )
    if run is None:
        return
    values: dict[str, Any] = {}
    if run["claim_attempt_id"] == attempt["attempt_id"]:
        values["claim_attempt_id"] = None
    if run_status and run["latest_attempt_id"] == attempt["attempt_id"] and run["status"] != run_status:
        values["status"] = run_status
    if not values:
        return
    version = tx.update("runs", "run_id", attempt["run_id"], values)
    if "claim_attempt_id" in values:
        tx.event(
            "run",
            attempt["run_id"],
            "run.claim_released",
            {"attempt_id": attempt["attempt_id"], "phase": attempt["phase"], "status": values.get("status")},
            entity_version=version,
        )
    else:
        tx.event(
            "run",
            attempt["run_id"],
            "run.status_changed",
            {"from": run["status"], "to": values["status"], "attempt_id": attempt["attempt_id"]},
            entity_version=version,
        )


def _notify(
    tx: Tx, event_type: str, entity_id: str, title: str, body: str, *, urgency: str = "immediate"
) -> None:
    """Best-effort notification through services.notify (written by another area); a missing
    or broken notifier must never fail the state change it reports."""
    try:
        notify = importlib.import_module("hpclab.services.notify")
    except ImportError:
        return
    try:
        notify.emit(
            tx,
            event_type=event_type,
            entity_id=entity_id,
            title=title,
            body=body,
            urgency=urgency,
            dedupe_key=f"{event_type}:{entity_id}",
        )
    except (AttributeError, TypeError, HpcLabError, sqlite3.Error) as exc:
        log.warning("notify.emit(%s, %s) failed: %s", event_type, entity_id, exc)


def _policy(ctx: ServiceContext, cluster_id: str) -> SitePolicy:
    try:
        return ctx.clusters.get(cluster_id).profile.policy
    except HpcLabError:
        return SitePolicy()


def _window_s(policy: SitePolicy) -> float:
    return float(policy.reconcile_window_s) + float(policy.accounting_lag_s)


def _dispatch_time(row: dict[str, Any]) -> datetime:
    when = parse_iso(row.get("dispatch_at") or row.get("uncertain_since") or row.get("created_at"))
    assert when is not None
    return when


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlanFacts:
    resources: ResourceSpec
    cluster_id: str
    remote_root: str | None
    gate: str | None


def _plan_facts(plan_json: str, default_cluster: str) -> _PlanFacts:
    """What admission needs from a stored plan. Strict ExecutionPlan first; a partial plan
    (hand-made rows) is tolerated here and rejected by staging, which needs the full plan."""
    raw = jload(plan_json, {})
    if not isinstance(raw, dict):
        raise PreconditionFailed("stored execution plan is not a JSON object")
    try:
        plan = ExecutionPlan.model_validate(raw)
    except ValidationError:
        plan = None
    if plan is not None:
        return _PlanFacts(plan.resources, plan.cluster_id, plan.remote_root, plan.gate)
    try:
        resources = ResourceSpec.model_validate(raw.get("resources") or {})
    except ValidationError as exc:
        raise PreconditionFailed(
            "execution plan resources are invalid",
            details={"errors": [e.get("msg") for e in exc.errors()[:10]]},
        ) from exc
    root = raw.get("remote_root")
    gate = raw.get("gate")
    return _PlanFacts(
        resources=resources,
        cluster_id=str(raw.get("cluster_id") or default_cluster),
        remote_root=root if isinstance(root, str) else None,
        gate=gate if isinstance(gate, str) else None,
    )


def _remote_dir(ctx: ServiceContext, facts: _PlanFacts, attempt_id: str) -> str:
    root = facts.remote_root
    if root is None:
        try:
            root = ctx.clusters.get(facts.cluster_id).remote_root()
        except HpcLabError:
            root = None
    if root is None:
        return f"<unresolved>/attempts/{attempt_id}"
    try:
        return attempt_layout(root, attempt_id).dir
    except InvalidArgument:
        return f"{root.rstrip('/')}/attempts/{attempt_id}"


# ---------------------------------------------------------------------------
# Execution holds: the Run stays claimed while an original job may still run
# ---------------------------------------------------------------------------


def _requeue_watch() -> tuple[frozenset[str], float]:
    """observe's requeue watch: terminal states Slurm may requeue from and how long it polls
    such an attempt after it ended (imported lazily: observe must never need this module)."""
    observe = importlib.import_module("hpclab.services.observe")
    return frozenset(observe.REQUEUE_STATES), float(observe.REQUEUE_WATCH_S)


def _plan_requeue(reader: Any, exec_plan_hash: str | None) -> bool | None:
    """The plan's explicit ``--requeue`` (True) / ``--no-requeue`` (False); None = site default."""
    ep = reader.one("SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (exec_plan_hash,))
    raw = jload(ep["plan_json"], {}) if ep else {}
    value = raw.get("requeue") if isinstance(raw, dict) else None
    return value if isinstance(value, bool) else None


def _end_confirmations(reader: Any, attempt_id: str) -> list[dict[str, Any]]:
    rows = reader.all(
        "SELECT payload_json FROM events WHERE entity_type='attempt' AND entity_id=? AND event_type=? ORDER BY seq",
        (attempt_id, EVENT_END_CONFIRMED),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        payload = jload(r["payload_json"], {})
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _is_confirmed(confirmations: list[dict[str, Any]], job_id: str, *, restart: int | None) -> bool:
    """A human confirmed ``job_id`` ended (for requeue segment ``restart`` when given: a later
    requeue of the job is a new execution that needs its own confirmation)."""
    for c in confirmations:
        if job_id not in [str(j) for j in c.get("job_ids") or []]:
            continue
        if restart is not None and int(c.get("restart_count") or 0) != restart:
            continue
        return True
    return False


def _confirm_hint(attempt_id: str) -> str:
    return (
        f"attempt_resolve(attempt_id={attempt_id!r}, resolution='confirm_ended', reason=<how you know>) "
        "records a human confirmation after a live squeue/sacct check (a job still queued or running is refused)"
    )


def _requeue_hold(reader: Any, a: dict[str, Any], *, now: datetime) -> ExecutionHold | None:
    """The requeue watch of a TERMINAL attempt (mirrors observe's poll targets): its job ended in a
    state Slurm may requeue from less than ``REQUEUE_WATCH_S`` ago, so it may run again in the
    same attempt dir. A ``--no-requeue`` plan cannot come back (Slurm never requeues it)."""
    states, watch_s = _requeue_watch()
    if a["phase"] != AttemptPhase.TERMINAL.value or not a["job_id"] or a["scheduler_state"] not in states:
        return None
    ended = parse_iso(a["terminal_at"])
    if ended is None:
        return None
    until = ended + timedelta(seconds=watch_s)
    if now >= until or _plan_requeue(reader, a["exec_plan_hash"]) is False:
        return None
    job_id = str(a["job_id"])
    restart = int(a["restart_count"] or 0)
    if _is_confirmed(_end_confirmations(reader, a["attempt_id"]), job_id, restart=restart):
        return None
    until_s = iso(until)
    return ExecutionHold(
        kind="requeue_watch",
        attempt_id=a["attempt_id"],
        job_id=job_id,
        until=until_s,
        message=(
            f"original job {job_id} (attempt {a['attempt_id']}, {a['scheduler_state']}) may be requeued; "
            f"wait until {until_s} or confirm it ended"
        ),
        hint=(
            f"Slurm may requeue a job that ended {a['scheduler_state']} and rerun it in the same attempt dir; "
            f"hpclab watches it until {until_s}. If it cannot come back (squeue/sacct): {_confirm_hint(a['attempt_id'])}"
        ),
    )


def _duplicate_holds(reader: Any, run_id: str, *, attempt_id: str | None = None) -> list[ExecutionHold]:
    """Jobs that own an attempt dir of this Run while hpclab observed another (duplicate) job:
    finalize recorded ``attempt.duplicate_job``. Such a job runs the attempt unobserved, so the
    Run stays claimed until it is known to have ended - there is no time bound to wait for:
    squeue/sacct showing it terminal (``check_duplicate_owners``, run by the reconcile sweep) or
    a human ``confirm_ended`` records ``attempt.end_confirmed``."""
    sql = (
        "SELECT e.entity_id AS attempt_id, e.payload_json FROM events e JOIN attempts a ON a.attempt_id = e.entity_id "
        "WHERE e.entity_type='attempt' AND e.event_type=? AND a.run_id=?"
    )
    params: tuple[Any, ...] = (EVENT_DUPLICATE_JOB, run_id)
    if attempt_id is not None:
        sql += " AND a.attempt_id=?"
        params = (*params, attempt_id)
    holds: list[ExecutionHold] = []
    seen: set[tuple[str, str]] = set()
    confirmations: dict[str, list[dict[str, Any]]] = {}
    for r in reader.all(sql + " ORDER BY e.seq", params):
        payload = jload(r["payload_json"], {})
        owner = str(payload.get("owner_job_id") or "") if isinstance(payload, dict) else ""
        aid = str(r["attempt_id"])
        if not owner or (aid, owner) in seen:
            continue
        seen.add((aid, owner))
        if aid not in confirmations:
            confirmations[aid] = _end_confirmations(reader, aid)
        if _is_confirmed(confirmations[aid], owner, restart=None):
            continue
        duplicate = payload.get("job_id")
        holds.append(
            ExecutionHold(
                kind="duplicate_owner",
                attempt_id=aid,
                job_id=owner,
                message=(
                    f"Slurm job {owner} runs attempt {aid} outside hpclab (job {duplicate} was a duplicate "
                    "submission) and may still be running; wait until it ends (hpclab checks squeue/sacct) "
                    "or confirm it ended"
                ),
                hint=(
                    f"hpclab looks the job up every few minutes and lifts this hold once squeue/sacct show it "
                    f"ended; it never cancels a job it did not bind (`scancel {owner}` yourself if it must stop). "
                    f"If the scheduler no longer knows the job (`sacct -j {owner}` shows nothing): {_confirm_hint(aid)}"
                ),
            )
        )
    return holds


def execution_holds(reader: Any, run_id: str, *, now: datetime) -> list[ExecutionHold]:
    """Why the Run must not start another execution although no attempt holds its claim (empty =
    none): the latest attempt's original job may be requeued (observe's requeue watch), or a job
    finalize found owning an attempt dir of the Run may still run (duplicate owner). ``admit``
    and ``run_recover`` refuse while any is open; each ends by time (requeue watch), by the
    scheduler showing the owner job ended (duplicate owner, ``check_duplicate_owners``) or by an
    explicit human ``attempt_resolve(resolution='confirm_ended')`` after a live check."""
    holds: list[ExecutionHold] = []
    latest = reader.one("SELECT * FROM attempts WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,))
    if latest is not None:
        hold = _requeue_hold(reader, latest, now=now)
        if hold is not None:
            holds.append(hold)
    holds.extend(_duplicate_holds(reader, run_id))
    return holds


def _attempt_holds(reader: Any, a: dict[str, Any], *, now: datetime) -> list[ExecutionHold]:
    """The open holds a confirmation of attempt ``a`` can lift."""
    holds: list[ExecutionHold] = []
    hold = _requeue_hold(reader, a, now=now)
    if hold is not None:
        holds.append(hold)
    holds.extend(_duplicate_holds(reader, a["run_id"], attempt_id=a["attempt_id"]))
    return holds


def _owner_ended(
    row: dict[str, Any], found: JobObservation | None, *, requeue: bool | None, now: datetime
) -> bool:
    """Whether the scheduler's own record proves a duplicate owner job ended: visible, carrying
    this attempt's token (a reused job id is another job), terminal, and not in a state Slurm
    may still requeue it from (inside the requeue watch, unless the plan is ``--no-requeue``).
    A job nobody can see is not proven ended (purged accounting is no end record)."""
    if found is None or found.state not in SCHEDULER_TERMINAL:
        return False
    ours, problems = _token_problems(found, str(row["submit_token"] or ""))
    if not ours or problems:
        return False
    if row["job_owner"] and found.user and found.user != row["job_owner"]:
        return False
    states, watch_s = _requeue_watch()
    if found.state.value in states and requeue is not False:
        ended = parse_iso(found.end_time)
        if ended is None or now < ended + timedelta(seconds=watch_s):
            return False
    return True


def _open_owner_holds(reader: Any) -> list[tuple[dict[str, Any], ExecutionHold]]:
    """Every duplicate-owner hold not confirmed yet, as (attempt row, hold)."""
    out: list[tuple[dict[str, Any], ExecutionHold]] = []
    for r in reader.all(
        "SELECT DISTINCT entity_id FROM events WHERE event_type=? AND entity_type='attempt' ORDER BY entity_id",
        (EVENT_DUPLICATE_JOB,),
    ):
        a = _attempt(reader, str(r["entity_id"]))
        if a is not None:
            out.extend((a, h) for h in _duplicate_holds(reader, a["run_id"], attempt_id=a["attempt_id"]))
    return out


def check_duplicate_owners(ctx: ServiceContext) -> int:
    """Lift the duplicate-owner holds whose owner job squeue/sacct show ended (reconcile sweep).

    A job that owns an attempt dir unobserved keeps its Run claimed until it is terminal. Only
    positive evidence ends the hold here (``_owner_ended``): ``attempt.end_confirmed`` (source
    worker, the scheduler's record as evidence). A job the scheduler does not show is never
    confirmed automatically - that needs a human ``confirm_ended``. Each owner is looked up at
    most every ``owner_check_s`` and at most ``owner_checks_per_sweep`` per sweep; lookup
    failures (auth, transport) are retried at the next check; nothing is ever cancelled.
    Returns the number of holds lifted."""
    now = ctx.clock.now()
    every_s = _setting(ctx, "owner_check_s")
    left = int(_setting(ctx, "owner_checks_per_sweep"))
    lifted = 0
    for a, hold in _open_owner_holds(ctx.db):
        if left <= 0:
            break
        key = f"{OWNER_CHECK_PREFIX}{a['attempt_id']}:{hold.job_id}"
        last = ctx.db.one(
            "SELECT observed_at FROM observations WHERE cluster_id=? AND obs_key=?", (a["cluster_id"], key)
        )
        seen = parse_iso(last["observed_at"]) if last else None
        if seen is not None and abs((now - seen).total_seconds()) < every_s:
            continue  # (a check time in the future means the clock stepped back: look again)
        left -= 1
        found: JobObservation | None = None
        sources: dict[str, str] = {}
        error: str | None = None
        try:
            found, sources = _live_lookup(ctx.clusters.scheduler(a["cluster_id"]), hold.job_id)
        except HpcLabError as exc:
            error = f"{exc.code}: {exc.message}"[:300]
        ended = error is None and _owner_ended(
            a, found, requeue=_plan_requeue(ctx.db, a["exec_plan_hash"]), now=now
        )
        with ctx.db.tx(source="worker") as tx:
            cur = _attempt(tx, a["attempt_id"])
            still = cur is not None and any(
                h.job_id == hold.job_id
                for h in _duplicate_holds(tx, cur["run_id"], attempt_id=cur["attempt_id"])
            )
            if ended and found is not None and cur is not None and still:
                tx.event(
                    "attempt",
                    cur["attempt_id"],
                    EVENT_END_CONFIRMED,
                    {
                        "job_ids": [hold.job_id],
                        "holds": [hold.kind],
                        "restart_count": int(cur["restart_count"] or 0),
                        "reason": f"{found.source} shows job {hold.job_id} {found.state.value}",
                        "automatic": True,
                        "evidence": [
                            {
                                "job_id": hold.job_id,
                                "hold": hold.kind,
                                "state": found.state.value,
                                "source": found.source,
                                "end_time": found.end_time,
                                "restart_count": found.restart_count,
                            }
                        ],
                    },
                    observed_at=found.observed_at,
                )
                tx.execute(
                    "DELETE FROM observations WHERE cluster_id=? AND obs_key=?", (a["cluster_id"], key)
                )
                lifted += 1
                continue
            tx.execute(
                "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at,"
                " expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    a["cluster_id"],
                    key,
                    jdump(
                        {
                            "job_id": hold.job_id,
                            "state": found.state.value if found is not None else None,
                            "sources": sources,
                            "error": error,
                        }
                    ),
                    found.source if found is not None else "squeue",
                    iso(now),
                    iso(now + timedelta(seconds=every_s)),
                ),
            )
    if lifted:
        log.info("sweep: %d duplicate-owner hold(s) lifted (the scheduler shows the owner job ended)", lifted)
    return lifted


GateState = Literal["passed", "pending", "not_started", "failed", "missing"]


def gate_state(reader: Any, gate_run_id: str) -> GateState:
    """Smoke gate of a plan: passed only when some attempt of the gate run is TERMINAL with
    scheduler COMPLETED *and* validated results."""
    if reader.one("SELECT 1 FROM runs WHERE run_id=?", (gate_run_id,)) is None:
        return "missing"
    rows = reader.all(
        "SELECT phase, scheduler_state, result_state FROM attempts WHERE run_id=? ORDER BY seq DESC",
        (gate_run_id,),
    )
    if not rows:
        return "not_started"
    for r in rows:
        if (
            r["phase"] == AttemptPhase.TERMINAL
            and r["scheduler_state"] == "COMPLETED"
            and r["result_state"] == "valid"
        ):
            return "passed"
    latest = rows[0]
    if latest["phase"] in CLAIM_HOLDING_PHASES:
        return "pending"
    if (
        latest["phase"] == AttemptPhase.TERMINAL
        and latest["scheduler_state"] == "COMPLETED"
        and latest["result_state"] in ("unchecked", "pending")
    ):
        return "pending"  # finished; outputs not validated yet
    return "failed"


def _count_active(reader: Any, sql_where: str, params: tuple[Any, ...]) -> int:
    phases = tuple(p.value for p in CLAIM_HOLDING_PHASES)
    marks = ",".join("?" for _ in phases)
    row = reader.one(
        f"SELECT COUNT(*) AS n FROM attempts a JOIN runs r ON r.run_id = a.run_id "
        f"WHERE {sql_where} AND a.phase IN ({marks})",
        (*params, *phases),
    )
    return int(row["n"]) if row else 0


def admit(
    tx: Tx,
    ctx: ServiceContext,
    *,
    run_id: str,
    exec_plan_hash: str,
    workflow_id: str | None = None,
    force_without_gate: bool = False,
    reason: str | None = None,
) -> AdmitResult:
    """THE admission gate (run_submit, workflow dispatch, recovery), inside the caller's tx.

    Checks, in order: plan exists / belongs to the run / ready / not consumed; the run holds no
    execution claim and no execution hold (``execution_holds``: an original job that may be
    requeued or a duplicate-owner job that may still run keeps the run claimed); smoke gate passed (or ``force_without_gate`` with a ``reason``, recorded as
    an event); workflow dispatch open + workflow slots; experiment slots (a slot = an attempt in
    a claim-holding phase); budget. Then creates the Attempt (STAGING, seq, token, job name,
    remote dir, gpus, partition, gpu type), sets ``runs.claim_attempt_id`` and
    ``execution_plans.consumed_by_attempt_id``, reserves ``gpus x time limit`` GPU-hours,
    creates the submit Operation and enqueues ``submit.stage``.

    Raises NotFound, InvalidArgument, NotReady, PreconditionFailed, AdmissionDenied,
    BudgetExceeded - all with ``details`` saying what to do.
    """
    run = tx.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    if run is None:
        raise NotFound(f"run {run_id} not found", details={"run_id": run_id})
    ep = tx.one("SELECT * FROM execution_plans WHERE exec_plan_hash=?", (exec_plan_hash,))
    if ep is None:
        raise NotFound(
            f"execution plan {exec_plan_hash} not found",
            details={
                "run_id": run_id,
                "execution_plan_hash": exec_plan_hash,
                "current_exec_plan_hash": run["current_exec_plan_hash"],
            },
        )
    if ep["run_id"] != run_id:
        raise InvalidArgument(
            f"execution plan {exec_plan_hash} belongs to run {ep['run_id']}, not {run_id}",
            details={"plan_run_id": ep["run_id"], "run_id": run_id},
        )
    if ep["consumed_by_attempt_id"] is not None or ep["status"] == "consumed":
        raise AdmissionDenied(
            "this execution plan was already consumed by an attempt; a plan runs at most once",
            details={
                "reason": "plan_consumed",
                "consumed_by_attempt_id": ep["consumed_by_attempt_id"],
                "hint": "use run_recover to create a new execution plan for another attempt",
            },
        )
    if ep["status"] == "not_ready":
        raise NotReady("execution plan is not ready yet", details={"execution_plan_hash": exec_plan_hash})
    if ep["status"] != "ready":
        raise PreconditionFailed(
            f"execution plan is {ep['status']}",
            details={"execution_plan_hash": exec_plan_hash, "validation": jload(ep["validation_json"], {})},
        )
    parent = tx.one("SELECT status FROM plans WHERE plan_id=?", (run["plan_id"],))
    if parent is not None and parent["status"] == "preparing":
        raise NotReady("the plan is still being prepared", details={"plan_id": run["plan_id"]})
    if parent is not None and parent["status"] != "ready":
        raise PreconditionFailed(
            f"the plan is {parent['status']}; fix the validation findings and prepare again",
            details={"plan_id": run["plan_id"], "plan_status": parent["status"]},
        )
    exp = tx.one("SELECT * FROM experiments WHERE experiment_id=?", (run["experiment_id"],))
    if exp is None:
        raise NotFound(f"experiment {run['experiment_id']} not found")

    # -- one execution claim per run --------------------------------------------------
    if run["claim_attempt_id"]:
        holder = tx.one(
            "SELECT attempt_id, phase FROM attempts WHERE attempt_id=?", (run["claim_attempt_id"],)
        )
        if holder is not None and holder["phase"] in CLAIM_HOLDING_PHASES:
            unresolved = holder["phase"] in _UNRESOLVED
            raise AdmissionDenied(
                f"run {run_id} already has an {'unresolved' if unresolved else 'active'} attempt "
                f"{holder['attempt_id']} ({holder['phase']})",
                details={
                    "reason": "run_claimed",
                    "claim_attempt_id": holder["attempt_id"],
                    "phase": holder["phase"],
                    "hint": (
                        "the previous submission is uncertain: wait for reconciliation or resolve it with "
                        f"`hpclab attempt resolve {holder['attempt_id']}`"
                        if unresolved
                        else "wait for it to finish or cancel it; parallel executions need a derived Run"
                    ),
                },
            )
        # A claim left behind by an attempt that no longer holds one (defensive self-heal).
        version = tx.update(
            "runs", "run_id", run_id, {"claim_attempt_id": None}, expected_version=run["version"]
        )
        tx.event(
            "run",
            run_id,
            "run.claim_released",
            {
                "attempt_id": run["claim_attempt_id"],
                "phase": holder["phase"] if holder else None,
                "stale": True,
            },
            entity_version=version,
        )
        run = dict(run, claim_attempt_id=None, version=version)

    # -- the run stays claimed while an original job may still run (requeue / duplicate owner) --
    holds = execution_holds(tx, run_id, now=ctx.clock.now())
    if holds:
        hold = holds[0]
        raise AdmissionDenied(
            f"run {run_id} stays claimed: {hold.message}",
            retryable=all(h.until is not None for h in holds),
            details={
                "reason": "run_claimed",
                "hold": hold.kind,
                "claim_attempt_id": hold.attempt_id,
                "job_id": hold.job_id,
                "until": hold.until,
                "hint": hold.hint,
                "holds": [h.model_dump(mode="json") for h in holds],
            },
        )

    facts = _plan_facts(ep["plan_json"], exp["cluster_id"])

    # -- smoke gate ----------------------------------------------------------------------
    gate_forced: dict[str, Any] | None = None
    if facts.gate:
        state = gate_state(tx, facts.gate)
        if state != "passed":
            if force_without_gate:
                if not (reason and reason.strip()):
                    raise InvalidArgument("force_without_gate requires a reason (it is recorded as an event)")
                gate_forced = {"gate_run_id": facts.gate, "gate_state": state, "reason": reason}
            elif state in ("pending", "not_started"):
                raise NotReady(
                    f"smoke gate run {facts.gate} has not passed yet ({state})",
                    details={
                        "gate_run_id": facts.gate,
                        "gate_state": state,
                        "hint": "submit/wait for the smoke run",
                    },
                )
            else:
                raise PreconditionFailed(
                    f"smoke gate run {facts.gate} {'does not exist' if state == 'missing' else 'failed'}",
                    details={
                        "gate_run_id": facts.gate,
                        "gate_state": state,
                        "hint": "fix the smoke failure (run_inspect) or pass force_without_gate with a reason",
                    },
                )

    # -- workflow + experiment slots -------------------------------------------------------
    if workflow_id is not None:
        wf = tx.one("SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,))
        if wf is None:
            raise NotFound(f"workflow {workflow_id} not found", details={"workflow_id": workflow_id})
        if not wf["dispatch_open"] or wf.get("dispatch_state", "open") in ("paused", "closed"):
            raise AdmissionDenied(
                f"workflow {workflow_id} dispatch is closed ({wf.get('dispatch_state')})",
                details={"reason": "workflow_dispatch_closed", "workflow_id": workflow_id},
            )
        active = _count_active(tx, "a.workflow_id = ?", (workflow_id,))
        if active >= int(wf["max_parallel"]):
            raise AdmissionDenied(
                f"workflow {workflow_id} already runs {active} of max_parallel={wf['max_parallel']} attempts",
                retryable=True,
                details={
                    "reason": "workflow_slots_full",
                    "active": active,
                    "max_parallel": wf["max_parallel"],
                },
            )
    if exp["max_parallel"] is not None:
        active = _count_active(tx, "r.experiment_id = ?", (exp["experiment_id"],))
        if active >= int(exp["max_parallel"]):
            raise AdmissionDenied(
                f"experiment already has {active} of max_parallel={exp['max_parallel']} attempts in flight",
                retryable=True,
                details={
                    "reason": "experiment_slots_full",
                    "active": active,
                    "max_parallel": exp["max_parallel"],
                    "hint": "staging/in-flight/uncertain attempts hold a slot too; retry when one finishes",
                },
            )

    # -- budget -----------------------------------------------------------------------------
    gpu_hours = round(facts.resources.requested_gpu_hours, 6)
    budget.check_available(tx, experiment_id=exp["experiment_id"], gpu_hours=gpu_hours)

    # -- create -----------------------------------------------------------------------------
    attempt_id = new_id("attempt")
    token = submit_token()
    job_name = job_name_for(run["label"] or "run", token)
    seq_row = tx.one("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM attempts WHERE run_id=?", (run_id,))
    seq = int(seq_row["n"]) if seq_row else 1
    poll_ms = int(_setting(ctx, "poll_after_ms"))
    op_id = tx.create_operation(
        "submit", target_type="attempt", target_id=attempt_id, stage="admitted", poll_after_ms=poll_ms
    )
    res = facts.resources
    tx.insert(
        "attempts",
        {
            "attempt_id": attempt_id,
            "run_id": run_id,
            "exec_plan_hash": exec_plan_hash,
            "seq": seq,
            "cluster_id": facts.cluster_id,
            "submit_token": token,
            "job_name": job_name,
            "remote_dir": _remote_dir(ctx, facts, attempt_id),
            "phase": AttemptPhase.STAGING.value,
            "gpus": res.total_gpus,
            "gpu_type": res.gpu_type,
            "partition": res.partition,
            "time_limit_s": res.time_seconds,
            "workflow_id": workflow_id,
            "operation_id": op_id,
            "version": 1,
            "created_at": tx.now,
            "updated_at": tx.now,
        },
    )
    cur = tx.execute(
        "UPDATE execution_plans SET consumed_by_attempt_id=?, status='consumed' "
        "WHERE exec_plan_hash=? AND consumed_by_attempt_id IS NULL",
        (attempt_id, exec_plan_hash),
    )
    if cur.rowcount != 1:  # unreachable under BEGIN IMMEDIATE; kept as the invariant's last line
        raise AdmissionDenied("execution plan was consumed concurrently", details={"reason": "plan_consumed"})
    run_version = tx.update(
        "runs",
        "run_id",
        run_id,
        {"claim_attempt_id": attempt_id, "latest_attempt_id": attempt_id, "status": "submitting"},
        expected_version=run["version"],
    )
    tx.event(
        "attempt",
        attempt_id,
        "attempt.admitted",
        {
            "run_id": run_id,
            "exec_plan_hash": exec_plan_hash,
            "seq": seq,
            "job_name": job_name,
            "cluster_id": facts.cluster_id,
            "gpus": res.total_gpus,
            "gpu_type": res.gpu_type,
            "partition": res.partition,
            "time_limit_s": res.time_seconds,
            "reserved_gpu_hours": gpu_hours,
            "workflow_id": workflow_id,
            "operation_id": op_id,
        },
        entity_version=1,
    )
    tx.event("run", run_id, "run.claimed", {"attempt_id": attempt_id, "seq": seq}, entity_version=run_version)
    tx.event(
        "execution_plan",
        exec_plan_hash,
        "execution_plan.consumed",
        {"run_id": run_id, "attempt_id": attempt_id},
    )
    if gate_forced is not None:
        tx.event("attempt", attempt_id, "attempt.gate_forced", gate_forced, entity_version=1)
    budget.reserve(tx, experiment_id=exp["experiment_id"], attempt_id=attempt_id, gpu_hours=gpu_hours)
    tx.enqueue(
        KIND_STAGE,
        operation_id=op_id,
        target_id=attempt_id,
        payload={"attempt_id": attempt_id},
        dedupe_key=f"{KIND_STAGE}:{attempt_id}",
    )
    return AdmitResult(
        attempt_id=attempt_id, operation_id=op_id, reserved_gpu_hours=gpu_hours, seq=seq, job_name=job_name
    )


def run_submit(
    ctx: ServiceContext,
    *,
    run_id: str,
    execution_plan_hash: str,
    idempotency_key: str,
    force_without_gate: bool = False,
    reason: str | None = None,
) -> SubmitReceiptData:
    """Submit one prepared Run with a frozen execution plan (asynchronous).

    Returns a receipt at once: the snapshot is staged and sbatch is called by the controller;
    poll ``operation_ref`` (or run_inspect). The same ``idempotency_key`` with the same payload
    returns the same receipt; a different payload is ``idempotency_conflict``. Different keys
    cannot double-run a Run: one active/unresolved attempt per Run, each plan consumed once.
    A lost sbatch reply is reconciled, never resubmitted.
    """
    _require_key(idempotency_key)
    payload_hash = digest(
        {
            "run_id": run_id,
            "execution_plan_hash": execution_plan_hash,
            "force_without_gate": bool(force_without_gate),
            "reason": reason,
        }
    )
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL_SUBMIT, idempotency_key, payload_hash)
        if stored is not None:
            return SubmitReceiptData.model_validate(stored)
        res = admit(
            tx,
            ctx,
            run_id=run_id,
            exec_plan_hash=execution_plan_hash,
            force_without_gate=force_without_gate,
            reason=reason,
        )
        receipt = SubmitReceiptData(
            operation_ref=operation_ref(res.operation_id),
            operation_id=res.operation_id,
            attempt_id=res.attempt_id,
            run_id=run_id,
            phase=AttemptPhase.STAGING.value,
            poll_after_ms=int(_setting(ctx, "poll_after_ms")),
            reserved_gpu_hours=res.reserved_gpu_hours,
        )
        tx.idem_store(
            ctx.principal, TOOL_SUBMIT, idempotency_key, payload_hash, receipt.model_dump(mode="json")
        )
    return receipt


# ---------------------------------------------------------------------------
# Terminal-before-scheduler outcomes (rejected / cancelled before submit)
# ---------------------------------------------------------------------------


def _finish_unsubmitted(
    tx: Tx,
    row: dict[str, Any],
    dst: AttemptPhase,
    *,
    event_type: str,
    reason: str,
    run_status: str,
    op_state: str,
    op_stage: str,
    op_error: dict[str, Any] | None,
    budget_note: str,
    payload: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Nothing is (or ever was) queued: final phase, claim + reservation released."""
    values: dict[str, Any] = {"reason": reason[:2000], **(extra or {})}
    if row["cancel_state"] == CancelState.REQUESTED and "cancel_state" not in values:
        values["cancel_state"] = CancelState.CONFIRMED.value
    new = _transition(tx, row, dst, values, event_type, payload)
    _release_claim(tx, new, run_status=run_status)
    budget.release(tx, attempt_id=new["attempt_id"], note=budget_note)
    _op(tx, new["operation_id"], state=op_state, stage=op_stage, error=op_error)
    if values.get("cancel_state") == CancelState.CONFIRMED:
        for op in _waiting_cancel_ops(tx, new["attempt_id"]):
            _op(
                tx,
                op,
                state="succeeded",
                stage="nothing_queued",
                result={"attempt_id": new["attempt_id"], "cancel_state": "confirmed", "phase": dst.value},
            )
    return new


def _waiting_cancel_ops(reader: Any, attempt_id: str) -> list[str]:
    rows = reader.all(
        "SELECT operation_id FROM operations WHERE kind='cancel' AND target_type='attempt' AND target_id=? "
        "AND state NOT IN ('succeeded', 'failed') ORDER BY created_at",
        (attempt_id,),
    )
    return [r["operation_id"] for r in rows]


def _cancel_waits_for_human(tx: Tx, attempt_id: str, why: str) -> None:
    """A pending cancel of an attempt that now needs a human decision (duplicate jobs, or an
    unresolved submission) cannot run by itself: say so instead of looking queued."""
    for op in _waiting_cancel_ops(tx, attempt_id):
        _op(
            tx,
            op,
            state="blocked",
            stage="waiting_for_resolution",
            error=_err(
                ErrorCode.SUBMISSION_UNCERTAIN,
                f"the cancel waits for a human decision on the submission ({why})",
                remediation=f"hpclab attempt resolve {attempt_id} --bind-job <job_id> (the cancel then runs) "
                "or --abandon",
            ),
        )


def _enqueue_cancel(tx: Tx, attempt: dict[str, Any]) -> str:
    ops = _waiting_cancel_ops(tx, attempt["attempt_id"])
    op_id = ops[-1] if ops else None
    if op_id is None:
        op_id = tx.create_operation(
            "cancel", target_type="attempt", target_id=attempt["attempt_id"], stage="verifying_identity"
        )
    else:
        _op(tx, op_id, state="running", stage="verifying_identity", clear_error=True)
    tx.enqueue(
        KIND_CANCEL,
        operation_id=op_id,
        target_id=attempt["attempt_id"],
        payload={"attempt_id": attempt["attempt_id"]},
        dedupe_key=f"{KIND_CANCEL}:{attempt['attempt_id']}",
    )
    return op_id


def _bind_job(
    tx: Tx,
    row: dict[str, Any],
    *,
    job_id: str,
    owner: str | None,
    submit_time: str | None,
    source: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Bind the scheduler job to the attempt: JOB_IDENTIFIED -> OBSERVING in one tx."""
    new = _transition(
        tx,
        row,
        AttemptPhase.JOB_IDENTIFIED,
        {
            "job_id": job_id,
            "job_owner": owner,
            "job_submit_time": submit_time,
            "identified_at": tx.now,
            "reason": None,
        },
        "attempt.job_identified",
        {
            "job_id": job_id,
            "owner": owner,
            "submit_time": submit_time,
            "source": source,
            "evidence": evidence,
        },
    )
    new = _transition(tx, new, AttemptPhase.OBSERVING, {}, "attempt.observing", {"job_id": job_id})
    _set_run_status(tx, new, "queued")
    _op(
        tx,
        new["operation_id"],
        state="succeeded",
        stage="submitted",
        result={"attempt_id": new["attempt_id"], "job_id": job_id, "source": source},
        clear_error=True,
    )
    if new["cancel_state"] == CancelState.REQUESTED:
        _enqueue_cancel(tx, new)
    return new


# ---------------------------------------------------------------------------
# Handler plumbing: loading inputs, auth blocks, retries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Inputs:
    run_spec: RunSpec
    plan: ExecutionPlan
    cluster: ResolvedCluster
    layout: AttemptLayout
    label: str


def _load_inputs(ctx: ServiceContext, row: dict[str, Any]) -> _Inputs:
    """Strictly parse what staging/dispatch need. Every failure here is permanent
    (non-retryable): the stored plan/spec cannot produce a job."""
    run = ctx.db.one("SELECT * FROM runs WHERE run_id=?", (row["run_id"],))
    ep = ctx.db.one("SELECT * FROM execution_plans WHERE exec_plan_hash=?", (row["exec_plan_hash"],))
    if run is None or ep is None:
        raise PreconditionFailed(
            "run or execution plan row is missing", details={"attempt_id": row["attempt_id"]}
        )
    raw = jload(ep["plan_json"], {})
    try:
        plan = ExecutionPlan.model_validate(raw)
    except ValidationError as exc:
        raise PreconditionFailed(
            "stored execution plan is not a complete ExecutionPlan",
            details={"errors": [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10]]},
        ) from exc
    if row["exec_plan_hash"] not in (plan.plan_hash(), digest(raw)):
        raise PreconditionFailed(
            "execution plan content does not match its hash (ledger tampered or corrupted)",
            details={"exec_plan_hash": row["exec_plan_hash"], "content_hash": plan.plan_hash()},
        )
    try:
        run_spec = RunSpec.model_validate(jload(run["spec_json"], {}))
    except ValidationError as exc:
        raise PreconditionFailed(
            "stored run spec is not a complete RunSpec",
            details={"errors": [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10]]},
        ) from exc
    if plan.cluster_id != row["cluster_id"]:
        raise PreconditionFailed("attempt and execution plan name different clusters")
    cluster = ctx.clusters.get(plan.cluster_id)
    layout = attempt_layout(plan.remote_root, row["attempt_id"])
    if layout.dir != row["remote_dir"]:
        raise PreconditionFailed(
            "attempt remote dir does not match the execution plan",
            details={"remote_dir": row["remote_dir"], "expected": layout.dir},
        )
    return _Inputs(run_spec=run_spec, plan=plan, cluster=cluster, layout=layout, label=run["label"])


def _blocked(ctx: ServiceContext, row: dict[str, Any], exc: TransportError, *, step: str) -> HandlerResult:
    """Operation -> ``blocked`` with a remediation, retried later; the attempt state is untouched.

    ``AuthRequiredError`` (MFA / expired ControlMaster) blocks as ``auth_required``; any other
    non-retryable transport problem (ssh misconfiguration, full quota) as ``cluster_access`` -
    both are fixable by the user, so the step waits instead of failing the attempt.
    """
    cid = row["cluster_id"]
    auth = isinstance(exc, AuthRequiredError)
    host_key = auth and exc.details.get("reason") == "host_key"
    stage = "auth_required" if auth else "cluster_access"
    if host_key:
        remediation = (
            f"the host key of {cid} changed: verify it with your cluster admins, then run `hpclab auth {cid}`"
        )
    elif auth:
        remediation = f"hpclab auth {cid}"
    else:
        remediation = f"fix this cluster-side problem ({exc.message}); hpclab retries automatically"
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, row["attempt_id"])
        op_id = cur["operation_id"] if cur else None
        if step == "cancel":
            ops = _waiting_cancel_ops(tx, row["attempt_id"])
            op_id = ops[-1] if ops else None
        op = tx.one("SELECT state, stage FROM operations WHERE operation_id=?", (op_id,)) if op_id else None
        if op is not None and not (op["state"] == "blocked" and op["stage"] == stage):
            _op(
                tx,
                op_id,
                state="blocked",
                stage=stage,
                error=_err(
                    exc.code,
                    exc.message,
                    retryable=auth and not host_key,
                    remediation=remediation,
                    cluster_id=cid,
                    blocked_step=step,
                    login_command=exc.details.get("login_command_str") or exc.details.get("login_command"),
                ),
            )
            if cur is not None:
                tx.event(
                    "attempt",
                    cur["attempt_id"],
                    f"attempt.{stage}",
                    {"step": step, "phase": cur["phase"], "remediation": remediation},
                    entity_version=cur["version"],
                )
            _notify(
                tx,
                "auth_required" if auth else "cluster_access",
                cid,
                f"hpclab cannot reach {cid}" if not auth else f"hpclab needs SSH authentication for {cid}",
                f"Run `{remediation}`; background work for attempt {row['attempt_id']} is waiting."
                if auth
                else f"{remediation} (attempt {row['attempt_id']} is waiting).",
            )
    if host_key or not auth:
        delay = _setting(ctx, "host_key_retry_s")
    else:
        delay = _setting(ctx, "auth_retry_s")
    return HandlerResult.retry(delay, note=f"{stage} ({step})")


def _blocked_op(reader: Any, op_id: str | None) -> bool:
    if not op_id:
        return False
    op = reader.one("SELECT state, stage FROM operations WHERE operation_id=?", (op_id,))
    return op is not None and op["state"] == "blocked" and op["stage"] in ("auth_required", "cluster_access")


def _unblock(ctx: ServiceContext, op_id: str | None, *, stage: str) -> None:
    """Called after the step's remote I/O worked again: blocked(auth/access) -> running."""
    if not _blocked_op(ctx.db, op_id):
        return
    with ctx.db.tx(source="worker") as tx:
        if _blocked_op(tx, op_id):
            _op(tx, op_id, state="running", stage=stage, clear_error=True)


def _is_access_problem(exc: HpcLabError) -> bool:
    """A transport failure a human must fix (not a transient one the outbox backs off)."""
    return isinstance(exc, TransportError) and not exc.retryable


def _note_retry(ctx: ServiceContext, row: dict[str, Any], exc: HpcLabError, *, step: str) -> None:
    """Record a transient failure on the operation (once per distinct message) before the
    outbox retries the step with backoff."""
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, row["attempt_id"])
        if cur is None:
            return
        _op(
            tx,
            cur["operation_id"],
            state="running",
            error=_err(exc.code, f"{step}: {exc.message} (retrying)", retryable=True, step=step),
        )


# ---------------------------------------------------------------------------
# submit.stage
# ---------------------------------------------------------------------------


def _stage_failed(ctx: ServiceContext, attempt_id: str, exc: HpcLabError) -> None:
    """A permanent staging failure: nothing was sent to the scheduler, so the attempt ends
    (SUBMISSION_REJECTED with reason ``stage_failed``) and frees its claim and budget."""
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None or cur["phase"] != AttemptPhase.STAGING:
            return
        _finish_unsubmitted(
            tx,
            cur,
            AttemptPhase.SUBMISSION_REJECTED,
            event_type="attempt.stage_failed",
            reason=f"stage_failed: {exc.message}",
            run_status="rejected",
            op_state="failed",
            op_stage="stage_failed",
            op_error=_exc_error(exc),
            budget_note="staging failed; nothing was submitted",
            payload={"error": _exc_error(exc)},
        )
    log.warning("attempt %s: staging failed permanently: %s", attempt_id, exc.message)


def _stage_io(ctx: ServiceContext, row: dict[str, Any], inp: _Inputs) -> dict[str, Any]:
    """Re-runnable by digest: rendering is deterministic, the snapshot is content-addressed and
    every write is an atomic replace, so a crashed stage can simply run again."""
    script = render_job_script(
        run_spec=inp.run_spec,
        plan=inp.plan,
        cluster=inp.cluster,
        run_id=row["run_id"],
        attempt_id=row["attempt_id"],
        token=row["submit_token"],
        job_name=row["job_name"],
        layout=inp.layout,
    )
    plan_doc = {
        "run_id": row["run_id"],
        "attempt_id": row["attempt_id"],
        "token": row["submit_token"],
        "run_spec": inp.run_spec.model_dump(mode="json"),
        "execution_plan": inp.plan.model_dump(mode="json"),
        "plan_hash": row["exec_plan_hash"],
    }
    plan_bytes = (json.dumps(plan_doc, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    transport = ctx.clusters.transport(inp.plan.cluster_id)
    store = SnapshotStore(ctx.paths.snapshots)
    published = publish_snapshot(transport, store, inp.run_spec.code, inp.plan.remote_root)
    if published.rstrip("/") != inp.plan.snapshot_path.rstrip("/"):
        raise PreconditionFailed(
            "published snapshot path differs from the execution plan's snapshot_path",
            details={"published": published, "plan_snapshot_path": inp.plan.snapshot_path},
        )
    res = transport.run_script(_MKDIR_SCRIPT, args=list(inp.layout.dirs_to_create()), timeout=60.0)
    if res.exit_code != 0:
        raise CommandFailed(
            f"could not create the attempt directory: {res.stderr.strip()[-300:]}",
            res,
            details={"dirs": list(inp.layout.dirs_to_create()), "stderr": res.stderr[-2000:]},
        )
    transport.write_file(inp.layout.token_file, plan_bytes, mode=0o644)
    transport.write_file(inp.layout.script, script.encode("utf-8"), mode=0o644)  # last: dispatch checks it
    return {
        "snapshot_path": published,
        "script_sha256": "sha256:" + sha256_hex(script),
        "plan_file_sha256": "sha256:" + sha256_hex(plan_bytes),
        "profile_revision_changed": inp.plan.profile_revision != inp.cluster.revision,
    }


def handle_stage(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """``submit.stage``: publish the code snapshot, create the attempt dir, write
    ``job.sbatch`` + ``hpclab-plan.json``, then enqueue ``submit.dispatch``."""
    aid = _payload_attempt(item)
    row = _attempt(ctx.db, aid)
    if row is None or row["phase"] != AttemptPhase.STAGING:
        return HandlerResult.done("attempt is not staging")
    try:
        inp = _load_inputs(ctx, row)
    except HpcLabError as exc:
        if exc.retryable:
            raise
        _stage_failed(ctx, aid, exc)
        return HandlerResult.done("stage failed")
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, aid)
        if cur is None or cur["phase"] != AttemptPhase.STAGING:
            return HandlerResult.done("attempt left staging")
        if not _blocked_op(tx, cur["operation_id"]):  # stays blocked until the I/O works
            _op(tx, cur["operation_id"], state="running", stage="staging")
    try:
        staged = _stage_io(ctx, row, inp)
    except AuthRequiredError as exc:
        return _blocked(ctx, row, exc, step="staging")
    except HpcLabError as exc:
        if isinstance(exc, TransportError) and _is_access_problem(exc):
            return _blocked(ctx, row, exc, step="staging")
        if exc.retryable:
            _note_retry(ctx, row, exc, step="staging")
            raise
        _stage_failed(ctx, aid, exc)
        return HandlerResult.done("stage failed")
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, aid)
        if cur is None or cur["phase"] != AttemptPhase.STAGING:
            return HandlerResult.done("attempt left staging while it was staged")
        tx.event("attempt", aid, "attempt.staged", staged, entity_version=cur["version"])
        _op(tx, cur["operation_id"], state="running", stage="staged", clear_error=True)
        tx.enqueue(
            KIND_DISPATCH,
            operation_id=cur["operation_id"],
            target_id=aid,
            payload={"attempt_id": aid},
            dedupe_key=f"{KIND_DISPATCH}:{aid}",
        )
    return HandlerResult.done()


# ---------------------------------------------------------------------------
# submit.dispatch
# ---------------------------------------------------------------------------


def _make_uncertain(
    tx: Tx, ctx: ServiceContext, row: dict[str, Any], *, reason: str, payload: dict[str, Any]
) -> dict[str, Any]:
    new = _transition(
        tx,
        row,
        AttemptPhase.SUBMISSION_UNCERTAIN,
        {"uncertain_since": tx.now, "reason": reason[:2000]},
        "attempt.uncertain",
        payload,
    )
    _set_run_status(tx, new, "unknown")
    _op(tx, new["operation_id"], state="running", stage="reconciling")
    tx.enqueue(
        KIND_RECONCILE,
        operation_id=new["operation_id"],
        target_id=new["attempt_id"],
        payload={"attempt_id": new["attempt_id"]},
        delay_s=_setting(ctx, "reconcile_first_delay_s"),
        dedupe_key=f"{KIND_RECONCILE}:{new['attempt_id']}",
    )
    _notify(
        tx,
        "attempt.uncertain",
        new["attempt_id"],
        "Submission outcome unknown",
        f"attempt {new['attempt_id']} (run {new['run_id']}): sbatch outcome unknown; reconciling "
        "(receipt, squeue, sacct). It is never resubmitted automatically.",
    )
    return new


def _dispatch_reentry(ctx: ServiceContext, attempt_id: str) -> HandlerResult:
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_IN_FLIGHT:
            return HandlerResult.done()
        _make_uncertain(
            tx,
            ctx,
            cur,
            reason="dispatch re-entered while sbatch may already have been called (crash, lease expiry "
            "or lost auth); reconciling instead of resubmitting",
            payload={"cause": "reentry", "dispatch_at": cur["dispatch_at"]},
        )
    log.warning("attempt %s: dispatch re-entered in flight -> uncertain (no second sbatch)", attempt_id)
    return HandlerResult.done("re-entry: uncertain")


def _dispatch_rejected(ctx: ServiceContext, attempt_id: str, exc: SubmissionRejected, *, proof: str) -> None:
    details = exc.details or {}
    code = str(details.get("reason_code") or "rejected")
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None:
            return
        if cur["phase"] not in (AttemptPhase.SUBMISSION_IN_FLIGHT, AttemptPhase.SUBMISSION_UNCERTAIN):
            tx.event(
                "attempt",
                attempt_id,
                "attempt.late_sbatch_refusal",
                {"phase": cur["phase"], "reason_code": code, "proof": proof},
                entity_version=cur["version"],
            )
            return
        err = _err(
            exc.code,
            exc.message,
            reason_code=code,
            retry_later=bool(details.get("retry_later")),
            stderr=str(details.get("stderr") or "")[-2000:],
            proof=proof,
            hint="nothing was queued; fix the cause and create a new plan (run_recover) to try again",
        )
        _finish_unsubmitted(
            tx,
            cur,
            AttemptPhase.SUBMISSION_REJECTED,
            event_type="attempt.rejected",
            reason=f"{code}: {exc.message}",
            run_status="rejected",
            op_state="failed",
            op_stage="rejected",
            op_error=err,
            budget_note=f"sbatch refused the job ({code}); nothing was queued",
            payload={"reason_code": code, "proof": proof, "retry_later": bool(details.get("retry_later"))},
        )
        _notify(
            tx,
            "attempt.rejected",
            attempt_id,
            "Submission rejected",
            f"attempt {attempt_id} (run {cur['run_id']}): sbatch refused the job ({code}).",
            urgency="batch",
        )


def _dispatch_accepted(ctx: ServiceContext, attempt_id: str, resp: SubmitResponse, owner: str) -> None:
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None:
            return
        if cur["phase"] in (AttemptPhase.SUBMISSION_IN_FLIGHT, AttemptPhase.SUBMISSION_UNCERTAIN):
            _bind_job(
                tx,
                cur,
                job_id=resp.job_id,
                owner=owner,
                submit_time=None,  # Slurm's own SubmitTime comes with the first observation
                source="sbatch_reply",
                evidence={
                    "raw": resp.raw[:200],
                    "received_at": resp.submitted_at,
                    "cluster": resp.cluster_name,
                },
            )
        elif cur["job_id"] != resp.job_id:
            # e.g. a human already bound another job: record the conflicting evidence, act on nothing
            tx.event(
                "attempt",
                attempt_id,
                "attempt.late_sbatch_reply",
                {"job_id": resp.job_id, "phase": cur["phase"], "bound_job_id": cur["job_id"]},
                entity_version=cur["version"],
            )


def _dispatch_uncertain(ctx: ServiceContext, attempt_id: str, exc: BaseException) -> None:
    if isinstance(exc, HpcLabError):
        cause = {"error": _exc_error(exc)}
        message = exc.message
    else:
        cause = {"error": {"code": "internal", "message": repr(exc)[:500]}}
        message = repr(exc)[:500]
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_IN_FLIGHT:
            return
        _make_uncertain(
            tx, ctx, cur, reason=f"sbatch outcome unknown: {message}", payload={"cause": "sbatch", **cause}
        )
    log.warning("attempt %s: sbatch outcome unknown (%s) -> reconcile", attempt_id, message)


def handle_dispatch(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """``submit.dispatch``: commit SUBMISSION_IN_FLIGHT, then the ONE sbatch call.

    Re-entry with the attempt already in flight never calls sbatch again: it becomes
    SUBMISSION_UNCERTAIN and is reconciled.
    """
    aid = _payload_attempt(item)
    row = _attempt(ctx.db, aid)
    if row is None:
        return HandlerResult.done("attempt not found")
    if row["phase"] == AttemptPhase.SUBMISSION_IN_FLIGHT:
        return _dispatch_reentry(ctx, aid)
    if row["phase"] != AttemptPhase.STAGING:
        return HandlerResult.done(f"attempt is {row['phase']}")
    try:
        inp = _load_inputs(ctx, row)
    except HpcLabError as exc:
        if exc.retryable:
            raise
        _stage_failed(ctx, aid, exc)
        return HandlerResult.done("stage failed")

    # Pre-flight while still STAGING: a failure here cannot have submitted anything.
    try:
        transport = ctx.clusters.transport(row["cluster_id"])
        scheduler = ctx.clusters.scheduler(row["cluster_id"])
        owner = transport.remote_user()
        marker = f"{inp.plan.snapshot_path.rstrip('/')}/{MARKER_NAME}"
        st = transport.stat([inp.layout.script, marker])
    except AuthRequiredError as exc:
        return _blocked(ctx, row, exc, step="dispatch_preflight")
    except HpcLabError as exc:
        if isinstance(exc, TransportError) and _is_access_problem(exc):
            return _blocked(ctx, row, exc, step="dispatch_preflight")
        if exc.retryable:
            _note_retry(ctx, row, exc, step="dispatch_preflight")
            raise
        _stage_failed(ctx, aid, exc)
        return HandlerResult.done("stage failed")
    missing = [p for p, info in st.items() if not info.exists]
    if missing:
        # scratch purge / manual cleanup between stage and dispatch: stage again (re-runnable)
        with ctx.db.tx(source="worker") as tx:
            cur = _attempt(tx, aid)
            if cur is not None and cur["phase"] == AttemptPhase.STAGING:
                tx.event(
                    "attempt", aid, "attempt.restage", {"missing": missing}, entity_version=cur["version"]
                )
                tx.enqueue(
                    KIND_STAGE,
                    operation_id=cur["operation_id"],
                    target_id=aid,
                    payload={"attempt_id": aid},
                    dedupe_key=f"{KIND_STAGE}:{aid}",
                )
        return HandlerResult.done("staged files missing; re-staging")

    policy = inp.cluster.profile.policy
    reentered = False
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, aid)
        if cur is None:
            return HandlerResult.done()
        if cur["phase"] == AttemptPhase.SUBMISSION_IN_FLIGHT:
            reentered = True
        elif cur["phase"] != AttemptPhase.STAGING:
            return HandlerResult.done(f"attempt is {cur['phase']}")
        elif cur["cancel_state"] == CancelState.REQUESTED:
            _cancel_before_submit(tx, cur, reason=cur["cancel_reason"])
            return HandlerResult.done("cancelled before dispatch")
        else:
            since = iso(ctx.clock.now() - timedelta(seconds=60))
            recent = tx.one(
                "SELECT COUNT(*) AS n FROM attempts WHERE cluster_id=? AND dispatch_at >= ?",
                (cur["cluster_id"], since),
            )
            if recent is not None and int(recent["n"]) >= max(1, int(policy.submit_rate_per_min)):
                return HandlerResult.retry(
                    max(1.0, 60.0 / max(1, policy.submit_rate_per_min)), "submit rate limit"
                )
            cur = _transition(
                tx,
                cur,
                AttemptPhase.SUBMISSION_IN_FLIGHT,
                {"dispatch_at": tx.now, "job_owner": owner},
                "attempt.dispatching",
                {"script": inp.layout.script, "receipt": inp.layout.receipt, "owner": owner},
            )
            _op(tx, cur["operation_id"], state="running", stage="dispatching", clear_error=True)
    if reentered:
        return _dispatch_reentry(ctx, aid)

    # The one sbatch call. From here on only a definite refusal is "not submitted".
    try:
        resp = scheduler.submit(inp.layout.script, workdir=inp.layout.dir, receipt_path=inp.layout.receipt)
    except SubmissionRejected as exc:
        _dispatch_rejected(ctx, aid, exc, proof="sbatch_reply")
        return HandlerResult.done("rejected")
    except AuthRequiredError as exc:
        # attempt stays in flight; the retried item re-enters and reconciles (never re-sends)
        return _blocked(ctx, row, exc, step="dispatching")
    except Exception as exc:  # noqa: BLE001 - after the in-flight commit every failure is "unknown"
        _dispatch_uncertain(ctx, aid, exc)
        return HandlerResult.done("uncertain")
    _dispatch_accepted(ctx, aid, resp, owner)
    return HandlerResult.done("submitted")


# ---------------------------------------------------------------------------
# submit.reconcile
# ---------------------------------------------------------------------------


@dataclass
class _Look:
    """One reconciliation pass over the three sources."""

    receipt_job_id: str | None = None
    rejected: SubmissionRejected | None = None
    candidates: dict[str, JobObservation] = field(default_factory=dict)
    mismatched: dict[str, list[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.sources.get("squeue") == "ok" and self.sources.get("sacct", "skipped") in (
            "ok",
            "skipped",
            "unsupported",
        )


def _token_problems(obs: JobObservation, token: str) -> tuple[bool, list[str]]:
    """(carries this token at all, contradictions)."""
    ntok = token_from_job_name(obs.name or "")
    ctok = token_from_comment(obs.comment)
    ours = ntok == token[:8] or ctok == token
    problems = []
    if ntok is not None and ntok != token[:8]:
        problems.append(f"job name {obs.name!r} carries another token")
    if ctok is not None and ctok != token:
        problems.append("job comment carries another token")
    if ntok is None and ctok is None:
        problems.append("no hpclab token in job name or comment")
    return ours, problems


def _consider(look: _Look, obs: JobObservation, *, token: str, owner: str, not_before: datetime) -> None:
    """Sort one observation into candidates / mismatched. Sources are considered freshest
    first (squeue, then sacct). A contradiction seen in ANY source is final: it is never
    erased by a poorer view of the same job elsewhere, and it demotes an earlier candidate."""
    if obs.job_id in look.mismatched:
        return
    known = obs.job_id in look.candidates
    ours, problems = _token_problems(obs, token)
    if not ours:
        if not known:
            return
        problems.append("another source shows this job without this attempt's token")
    if obs.user != owner:
        problems.append(f"owner {obs.user!r} is not the cluster user {owner!r}")
    submitted = parse_iso(obs.submit_time)
    if submitted is not None and submitted < not_before:
        problems.append(f"submitted at {obs.submit_time}, before this attempt was dispatched")
    if problems:
        look.candidates.pop(obs.job_id, None)
        look.mismatched[obs.job_id] = problems
    elif not known:
        look.candidates[obs.job_id] = obs  # squeue (fresher) is considered first


def _look(ctx: ServiceContext, row: dict[str, Any]) -> _Look:
    transport = ctx.clusters.transport(row["cluster_id"])
    scheduler = ctx.clusters.scheduler(row["cluster_id"])
    owner = transport.remote_user()
    token = row["submit_token"]
    not_before = _dispatch_time(row) - timedelta(seconds=_setting(ctx, "submit_time_skew_s"))
    look = _Look()
    receipt = f"{row['remote_dir'].rstrip('/')}/submit.receipt"
    # 1. the on-cluster journal of our own sbatch call
    try:
        resp = scheduler.read_receipt(receipt)
        look.sources["receipt"] = "ok" if resp is not None else "missing_or_ambiguous"
        if resp is not None:
            look.receipt_job_id = resp.job_id
    except SubmissionRejected as exc:
        look.rejected = exc
        look.sources["receipt"] = "rejected"
        return look
    except AuthRequiredError:
        raise
    except (HpcLabError, OSError) as exc:
        look.sources["receipt"] = "unavailable"
        look.errors.append(f"receipt: {exc}")
    # 2. the live queue, all of the user's jobs
    try:
        q = scheduler.query_queue(user=owner)
        look.sources["squeue"] = "ok" if q.complete else "partial"
        for obs in q.jobs:
            _consider(look, obs, token=token, owner=owner, not_before=not_before)
    except AuthRequiredError:
        raise
    except HpcLabError as exc:
        look.sources["squeue"] = "unavailable"
        look.errors.append(f"squeue: {exc.message}")
    # 3. accounting by job name: jobs that already left the queue. Always consulted, even with
    #    one squeue match: a same-token job that already finished (e.g. the one that did the
    #    work, while a duplicate still pends) is only visible here, and a bind needs a UNIQUE
    #    match. When accounting is down, a lone squeue match still binds (recorded in sources).
    try:
        acct = scheduler.query_accounting(user=owner, name=row["job_name"], since=iso(not_before))
        look.sources["sacct"] = "ok"
        for obs in acct.jobs:
            _consider(look, obs, token=token, owner=owner, not_before=not_before)
    except AuthRequiredError:
        raise
    except HpcLabError as exc:
        look.sources["sacct"] = "unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable"
        look.errors.append(f"sacct: {exc.message}")
    return look


def _candidate_summary(obs: JobObservation, *, from_receipt: bool = False) -> dict[str, Any]:
    terminal = obs.state in SCHEDULER_TERMINAL
    return {
        "job_id": obs.job_id,
        "source": obs.source,
        "state": obs.state.value,
        "name": obs.name,
        "user": obs.user,
        "submit_time": obs.submit_time,
        "exit_code": obs.exit_code,
        "from_receipt": from_receipt,
        # the wrapper's own duplicate guard: this job exited 95 without touching the attempt
        "harmless_duplicate": terminal and obs.exit_code == EXIT_DUPLICATE_JOB,
    }


def _look_details(look: _Look, *, deadline: datetime) -> dict[str, Any]:
    return {
        "sources": dict(look.sources),
        "receipt_job_id": look.receipt_job_id,
        "mismatched": [{"job_id": j, "problems": p} for j, p in sorted(look.mismatched.items())][:20],
        "errors": look.errors[:10],
        "window_ends_at": iso(deadline),
    }


def _reconcile_delay(ctx: ServiceContext, tries: int, *, now: datetime, deadline: datetime) -> float:
    base = _setting(ctx, "reconcile_base_delay_s")
    delay = min(_setting(ctx, "reconcile_max_delay_s"), base * (2 ** max(0, min(tries - 1, 10))))
    to_deadline = (deadline - now).total_seconds()
    if to_deadline > 0:
        delay = min(delay, to_deadline + 1.0)  # reach a verdict right after the window closes
    return max(1.0, delay)


def handle_reconcile(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """``submit.reconcile``: receipt -> squeue -> sacct; bind a unique token match, flag
    duplicates, keep looking until the window closes, then block as unresolved. Never
    resubmits."""
    aid = _payload_attempt(item)
    row = _attempt(ctx.db, aid)
    if row is None or row["phase"] != AttemptPhase.SUBMISSION_UNCERTAIN:
        return HandlerResult.done("attempt is not uncertain")
    policy = _policy(ctx, row["cluster_id"])
    deadline = _dispatch_time(row) + timedelta(seconds=_window_s(policy))
    try:
        look = _look(ctx, row)
    except AuthRequiredError as exc:
        return _blocked(ctx, row, exc, step="reconcile")
    except HpcLabError as exc:  # e.g. remote_user() unreachable: nothing could be checked
        if isinstance(exc, TransportError) and _is_access_problem(exc):
            return _blocked(ctx, row, exc, step="reconcile")
        return HandlerResult.retry(
            _reconcile_delay(ctx, item.tries, now=ctx.clock.now(), deadline=deadline),
            f"reconcile: {exc.message}",
        )
    _unblock(ctx, row["operation_id"], stage="reconciling")
    now = ctx.clock.now()
    details = _look_details(look, deadline=deadline)

    if look.rejected is not None:
        _dispatch_rejected(ctx, aid, look.rejected, proof="receipt")
        return HandlerResult.done("receipt proves refusal")

    cands = dict(look.candidates)
    rid = look.receipt_job_id
    if rid is not None and rid not in cands and cands:
        _mark_duplicate(ctx, aid, cands, receipt_job_id=rid, details=details)
        return HandlerResult.done("duplicate")
    if len(cands) > 1:
        _mark_duplicate(ctx, aid, cands, receipt_job_id=rid, details=details)
        return HandlerResult.done("duplicate")
    if len(cands) == 1 and (rid is None or rid in cands):
        obs = next(iter(cands.values()))
        with ctx.db.tx(source="reconcile") as tx:
            cur = _attempt(tx, aid)
            if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_UNCERTAIN:
                return HandlerResult.done()
            _bind_job(
                tx,
                cur,
                job_id=obs.job_id,
                owner=obs.user,
                submit_time=obs.submit_time,
                source=f"reconcile:{'receipt+' if rid else ''}{obs.source}",
                evidence={**details, "job": _candidate_summary(obs, from_receipt=rid == obs.job_id)},
            )
        log.info("attempt %s: reconciled to job %s", aid, obs.job_id)
        return HandlerResult.done("bound")

    # nothing (confirmed) yet
    if now < deadline or not look.complete:
        return HandlerResult.retry(
            _reconcile_delay(ctx, item.tries, now=now, deadline=deadline), note="no unique match yet"
        )
    _mark_unresolved(ctx, aid, details=details, window_s=_window_s(policy))
    return HandlerResult.done("unresolved")


def _mark_duplicate(
    ctx: ServiceContext,
    attempt_id: str,
    cands: dict[str, JobObservation],
    *,
    receipt_job_id: str | None,
    details: dict[str, Any],
) -> None:
    listed = [_candidate_summary(o, from_receipt=o.job_id == receipt_job_id) for o in cands.values()]
    if receipt_job_id is not None and receipt_job_id not in cands:
        listed.append(
            {"job_id": receipt_job_id, "source": "receipt", "from_receipt": True, "state": "UNKNOWN"}
        )
    listed.sort(key=lambda c: str(c["job_id"]))
    total = len(listed)
    listed = listed[:MAX_CANDIDATES]  # bounded payload; the count below is the real one
    with ctx.db.tx(source="reconcile") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_UNCERTAIN:
            return
        ids = [str(c["job_id"]) for c in listed]
        new = _transition(
            tx,
            cur,
            AttemptPhase.DUPLICATE_DETECTED,
            {"reason": f"{total} jobs carry this attempt's token: {', '.join(ids)}"},
            "attempt.duplicate_detected",
            {"candidates": listed, "candidate_count": total, **details},
        )
        _set_run_status(tx, new, "unknown")
        _op(
            tx,
            new["operation_id"],
            state="blocked",
            stage="duplicate_detected",
            error=_err(
                ErrorCode.AMBIGUOUS,
                f"{total} scheduler jobs match this submission; nothing was cancelled automatically",
                candidates=listed,
                candidate_count=total,
                remediation=(
                    f"hpclab attempt resolve {attempt_id} --bind-job <job_id>  (then scancel the other job "
                    f"yourself if needed), or --abandon"
                ),
            ),
        )
        _cancel_waits_for_human(tx, attempt_id, "duplicate jobs")
        _notify(
            tx,
            "attempt.duplicate_detected",
            attempt_id,
            "Duplicate submission detected",
            f"attempt {attempt_id} (run {cur['run_id']}): jobs {', '.join(ids)} carry the same token; "
            "a human decision is needed (nothing was cancelled).",
        )
    log.warning("attempt %s: duplicate jobs %s", attempt_id, [c["job_id"] for c in listed])


def _mark_unresolved(
    ctx: ServiceContext, attempt_id: str, *, details: dict[str, Any], window_s: float
) -> None:
    with ctx.db.tx(source="reconcile") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_UNCERTAIN:
            return
        payload = {**details, "window_s": window_s}
        remediation = (
            f"check the cluster (squeue -u <you>; sacct --name={cur['job_name']}); then "
            f"`hpclab attempt resolve {attempt_id} --bind-job <job_id>` or `--abandon`"
        )
        _op(
            tx,
            cur["operation_id"],
            state="blocked",
            stage="unresolved",
            error=_err(
                ErrorCode.SUBMISSION_UNCERTAIN,
                "no job with this attempt's token was found within the reconcile window; the submission "
                "stays unresolved (claim and budget kept) and is never resubmitted automatically",
                remediation=remediation,
                **payload,
            ),
        )
        _cancel_waits_for_human(tx, attempt_id, "unresolved")
        if not _has_event(tx, attempt_id, "attempt.unresolved"):
            tx.event("attempt", attempt_id, "attempt.unresolved", payload, entity_version=cur["version"])
            _notify(
                tx,
                "attempt.unresolved",
                attempt_id,
                "Submission unresolved",
                f"attempt {attempt_id} (run {cur['run_id']}): no matching job found after the reconcile "
                "window; a human decision is needed.",
            )
    log.warning("attempt %s: unresolved after the reconcile window", attempt_id)


def _has_event(reader: Any, attempt_id: str, event_type: str) -> bool:
    return (
        reader.one(
            "SELECT 1 FROM events WHERE entity_type='attempt' AND entity_id=? AND event_type=? LIMIT 1",
            (attempt_id, event_type),
        )
        is not None
    )


def _sweep_orphaned_staging(ctx: ServiceContext, *, stale_before: str) -> None:
    """A STAGING attempt with no pending/leased ``submit.stage``/``submit.dispatch`` item would
    hold its run's claim, a workflow slot and its budget forever. STAGING proves sbatch was
    never called (the in-flight commit precedes it), so this is safe to settle: when the outbox
    gave up on a step (dead item) the attempt ends ``stage_failed``; otherwise (rows removed)
    staging, which is re-runnable, is scheduled again."""
    for r in ctx.db.all(
        "SELECT attempt_id FROM attempts WHERE phase=? AND created_at < ?",
        (AttemptPhase.STAGING.value, stale_before),
    ):
        aid = r["attempt_id"]
        # the newest item of each step holds its dedupe key (UNIQUE index; runs every minute)
        items = ctx.db.all(
            "SELECT state, last_error FROM outbox WHERE dedupe_key IN (?, ?) ORDER BY outbox_id",
            (f"{KIND_STAGE}:{aid}", f"{KIND_DISPATCH}:{aid}"),
        )
        if any(i["state"] in ("pending", "leased") for i in items):
            continue
        dead = [i for i in items if i["state"] == "dead"]
        if dead:
            last = str(dead[-1]["last_error"] or "unknown error")[:500]
            _stage_failed(
                ctx,
                aid,
                HpcLabError(
                    f"background staging gave up after repeated failures: {last}",
                    details={"attempt_id": aid, "last_error": last},
                ),
            )
            continue
        with ctx.db.tx(source="worker") as tx:
            cur = _attempt(tx, aid)
            if cur is None or cur["phase"] != AttemptPhase.STAGING:
                continue
            if tx.enqueue(
                KIND_STAGE,
                operation_id=cur["operation_id"],
                target_id=aid,
                payload={"attempt_id": aid},
                dedupe_key=f"{KIND_STAGE}:{aid}",
            ):
                tx.event(
                    "attempt",
                    aid,
                    "attempt.restage",
                    {"cause": "no_live_step"},
                    entity_version=cur["version"],
                )
                log.info("sweep: re-enqueued staging for %s (no live step)", aid)


def reconcile_sweep(ctx: ServiceContext) -> None:
    """Periodic ``submit.reconcile_sweep``: every uncertain attempt inside its window (or not
    yet judged) has a pending reconcile item, an in-flight attempt whose dispatch item is gone
    (dead worker, dead item) is moved to uncertain - never re-dispatched - and a staging
    attempt without live work is re-staged or, if the outbox gave up on it, ended. Last, a
    duplicate-owner job the scheduler shows ended releases its hold (``check_duplicate_owners``)."""
    now = ctx.clock.now()
    policies: dict[str, SitePolicy] = {}

    def policy(cid: str) -> SitePolicy:
        if cid not in policies:
            policies[cid] = _policy(ctx, cid)
        return policies[cid]

    for r in ctx.db.all(
        "SELECT * FROM attempts WHERE phase=? ORDER BY dispatch_at",
        (AttemptPhase.SUBMISSION_UNCERTAIN.value,),
    ):
        within = now < _dispatch_time(r) + timedelta(seconds=_window_s(policy(r["cluster_id"])))
        if not within and _has_event(ctx.db, r["attempt_id"], "attempt.unresolved"):
            continue
        with ctx.db.tx(source="worker") as tx:
            cur = _attempt(tx, r["attempt_id"])
            if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_UNCERTAIN:
                continue
            if tx.enqueue(
                KIND_RECONCILE,
                operation_id=cur["operation_id"],
                target_id=cur["attempt_id"],
                payload={"attempt_id": cur["attempt_id"]},
                dedupe_key=f"{KIND_RECONCILE}:{cur['attempt_id']}",
            ):
                log.info("sweep: re-enqueued reconcile for %s", cur["attempt_id"])

    stale_before = iso(now - timedelta(seconds=_setting(ctx, "stale_in_flight_s")))
    _sweep_orphaned_staging(ctx, stale_before=stale_before or "")
    for r in ctx.db.all(
        "SELECT attempt_id FROM attempts WHERE phase=? AND dispatch_at < ?",
        (AttemptPhase.SUBMISSION_IN_FLIGHT.value, stale_before),
    ):
        live = ctx.db.one(
            "SELECT 1 FROM outbox WHERE kind=? AND target_id=? AND state IN ('pending', 'leased') LIMIT 1",
            (KIND_DISPATCH, r["attempt_id"]),
        )
        if live is not None:
            continue
        with ctx.db.tx(source="worker") as tx:
            cur = _attempt(tx, r["attempt_id"])
            if cur is None or cur["phase"] != AttemptPhase.SUBMISSION_IN_FLIGHT:
                continue
            _make_uncertain(
                tx,
                ctx,
                cur,
                reason="in flight without a live dispatch step (worker died mid-submission); reconciling",
                payload={"cause": "orphaned_in_flight", "dispatch_at": cur["dispatch_at"]},
            )
    check_duplicate_owners(ctx)


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def _stop_behavior(reader: Any, row: dict[str, Any]) -> tuple[str | None, str]:
    ep = reader.one("SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (row["exec_plan_hash"],))
    raw = jload(ep["plan_json"], {}) if ep else {}
    res = raw.get("resilience") if isinstance(raw, dict) else None
    if not isinstance(res, dict):
        return None, "USR1"
    return res.get("stop_behavior"), str(res.get("signal") or "USR1")


def _cancel_before_submit(tx: Tx, row: dict[str, Any], *, reason: str | None) -> dict[str, Any]:
    return _finish_unsubmitted(
        tx,
        row,
        AttemptPhase.CANCELLED_BEFORE_SUBMIT,
        event_type="attempt.cancelled_before_submit",
        reason=f"cancelled before submission: {reason or 'no reason given'}",
        run_status="cancelled",
        op_state="failed",
        op_stage="cancelled",
        op_error=_err(
            ErrorCode.PRECONDITION_FAILED, "cancelled before submission; nothing was sent to the scheduler"
        ),
        budget_note="cancelled before submission",
        extra={"cancel_state": CancelState.CONFIRMED.value, "cancel_reason": reason, "cancel_mode": "cancel"},
    )


def request_cancel(
    tx: Tx,
    ctx: ServiceContext,
    *,
    attempt_id: str,
    reason: str | None = None,
    mode: Literal["cancel", "request_checkpoint"] = "cancel",
) -> CancelReceiptData:
    """Record a cancel intent inside the caller's transaction (used by ``run_cancel`` and
    ``workflow_cancel``). Before dispatch the attempt ends at once (CANCELLED_BEFORE_SUBMIT);
    in flight / uncertain / duplicate the intent waits for the job to be identified;
    identified attempts get a ``cancel.execute`` step (live identity check, then scancel)."""
    row = _require_attempt(tx, attempt_id)
    phase = AttemptPhase(row["phase"])
    poll_ms = int(_setting(ctx, "poll_after_ms"))
    if phase in FINAL_PHASES or phase == AttemptPhase.TERMINAL:
        raise PreconditionFailed(
            f"attempt {attempt_id} is {phase.value}; there is nothing to cancel",
            details={"phase": phase.value, "scheduler_state": row["scheduler_state"]},
        )
    if mode == "request_checkpoint":
        stop, _sig = _stop_behavior(tx, row)
        if stop != "checkpoint_and_exit":
            raise PreconditionFailed(
                "request_checkpoint needs resilience.stop_behavior=checkpoint_and_exit in the execution plan",
                details={"stop_behavior": stop, "hint": "use mode='cancel'"},
            )
        if phase == AttemptPhase.STAGING:
            raise PreconditionFailed(
                "the attempt has not been submitted yet; there is no program to checkpoint",
                details={"phase": phase.value, "hint": "use mode='cancel'"},
            )
    if phase == AttemptPhase.STAGING:
        new = _cancel_before_submit(tx, row, reason=reason)
        return CancelReceiptData(
            attempt_id=attempt_id,
            run_id=row["run_id"],
            phase=new["phase"],
            cancel_state=new["cancel_state"],
            cancel_mode="cancel",
            attempt_version=new["version"],
            note="cancelled before submission; nothing was sent to the scheduler",
        )
    existing = _waiting_cancel_ops(tx, attempt_id)
    if row["cancel_state"] == CancelState.REQUESTED and row["cancel_mode"] == mode and existing:
        op_id = existing[-1]
        return CancelReceiptData(
            attempt_id=attempt_id,
            run_id=row["run_id"],
            phase=row["phase"],
            cancel_state=row["cancel_state"],
            cancel_mode=mode,
            attempt_version=row["version"],
            operation_ref=operation_ref(op_id),
            operation_id=op_id,
            poll_after_ms=poll_ms,
            note="a cancel request is already pending",
        )
    new = _update_attempt(
        tx,
        row,
        {"cancel_state": CancelState.REQUESTED.value, "cancel_reason": reason, "cancel_mode": mode},
        "attempt.cancel_requested",
        {"mode": mode, "reason": reason, "phase": phase.value},
    )
    for op in existing:  # a newer request supersedes an older pending one
        _op(
            tx,
            op,
            state="failed",
            stage="superseded",
            error=_err(ErrorCode.PRECONDITION_FAILED, "superseded"),
        )
    if phase in _ACTIVE_BOUND:
        op_id = _enqueue_cancel(tx, new)
        note = "cancel requested; the job identity is re-verified right before the scheduler call"
    else:
        waiting = phase == AttemptPhase.DUPLICATE_DETECTED or (
            phase == AttemptPhase.SUBMISSION_UNCERTAIN and _has_event(tx, attempt_id, "attempt.unresolved")
        )
        op_id = tx.create_operation(
            "cancel",
            target_type="attempt",
            target_id=attempt_id,
            state="blocked" if waiting else "queued",
            stage="waiting_for_resolution" if waiting else "waiting_for_reconcile",
            poll_after_ms=poll_ms,
        )
        note = (
            "the submission needs a human decision (duplicate jobs or unresolved): the cancel runs after "
            "a human binds the job (`hpclab attempt resolve`)"
            if waiting
            else "the submission outcome is not known yet: the cancel runs once the job is identified"
        )
    return CancelReceiptData(
        attempt_id=attempt_id,
        run_id=row["run_id"],
        phase=new["phase"],
        cancel_state=new["cancel_state"],
        cancel_mode=mode,
        attempt_version=new["version"],
        operation_ref=operation_ref(op_id),
        operation_id=op_id,
        poll_after_ms=poll_ms,
        note=note,
    )


def run_cancel(
    ctx: ServiceContext,
    *,
    attempt_id: str,
    expected_version: int,
    idempotency_key: str,
    reason: str | None = None,
    mode: Literal["cancel", "request_checkpoint"] = "cancel",
) -> CancelReceiptData:
    """Cancel one Attempt (or ask it to checkpoint and exit).

    Before dispatch it is cancelled immediately; an in-flight or uncertain submission records
    the intent and cancels once reconciliation identifies the job; an identified job is
    cancelled only after its live identity (job id, owner, token, submit time) is re-verified
    - on any mismatch nothing is cancelled (``cancel_state=refused``). ``request_checkpoint``
    sends the plan's resilience signal instead (only with stop_behavior=checkpoint_and_exit).
    Until an observation shows the job terminal the state is ``requested``/``sent``.
    """
    _require_key(idempotency_key)
    payload_hash = digest(
        {"attempt_id": attempt_id, "expected_version": expected_version, "reason": reason, "mode": mode}
    )
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL_CANCEL, idempotency_key, payload_hash)
        if stored is not None:
            return CancelReceiptData.model_validate(stored)
        row = _require_attempt(tx, attempt_id)
        if int(row["version"]) != int(expected_version):
            raise VersionConflict(
                f"attempt {attempt_id} is at version {row['version']}, expected {expected_version}",
                details={"current_version": row["version"], "expected_version": expected_version},
            )
        data = request_cancel(tx, ctx, attempt_id=attempt_id, reason=reason, mode=mode)
        tx.idem_store(ctx.principal, TOOL_CANCEL, idempotency_key, payload_hash, data.model_dump(mode="json"))
    return data


def _live_lookup(scheduler: Scheduler, job_id: str) -> tuple[JobObservation | None, dict[str, str]]:
    """squeue by id, else sacct by id. squeue failures raise (the caller retries); sacct
    trouble only means "not visible there"."""
    sources: dict[str, str] = {}
    q = scheduler.query_queue(job_ids=[job_id])
    sources["squeue"] = "ok" if q.complete else "partial"
    for obs in q.jobs:
        if obs.job_id == job_id and not obs.array_task_id:
            return obs, sources
    try:
        acct = scheduler.query_accounting(job_ids=[job_id])
        sources["sacct"] = "ok"
        for obs in acct.jobs:
            if obs.job_id == job_id and not obs.array_task_id:
                return obs, sources
    except AuthRequiredError:
        raise
    except HpcLabError as exc:
        sources["sacct"] = "unsupported" if exc.code == ErrorCode.UNSUPPORTED else "unavailable"
    return None, sources


def identity_problems(
    row: dict[str, Any], obs: JobObservation, *, remote_user: str, now: datetime, skew_s: float
) -> list[str]:
    """Why ``obs`` is not provably the job bound to attempt ``row`` (empty = verified).

    Checks job id, owner (== the bound owner == the controller's cluster user), the token in
    the job name (and comment when present), the submit time (== the recorded one, or inside
    [dispatch_at - skew, identified_at + skew]) and that the restart count did not go back.

    Slurm stamps a new SubmitTime on every requeue (slurmdbd keys a job record by id + submit
    time). Once the job has been requeued (recorded or live restart count > 0, or sacct, which
    reports no restart count, for an attempt known to be requeued) the submit time may only
    move forward: it must not be earlier than the recorded one / the dispatch window, and the
    id, owner and token checks still apply in full.
    """
    problems: list[str] = []
    if obs.job_id != row["job_id"]:
        problems.append(f"job id {obs.job_id} != bound job {row['job_id']}")
    expected_owner = row["job_owner"] or remote_user
    if row["job_owner"] and row["job_owner"] != remote_user:
        problems.append(
            f"controller's cluster user {remote_user!r} is not the job owner {row['job_owner']!r}"
        )
    if obs.user != expected_owner:
        problems.append(f"owner {obs.user!r} != {expected_owner!r}")
    token = row["submit_token"]
    if token_from_job_name(obs.name or "") != token[:8]:
        problems.append(f"job name {obs.name!r} does not carry this attempt's token")
    ctok = token_from_comment(obs.comment)
    if ctok is not None and ctok != token:
        problems.append("job comment carries another token")
    recorded_restarts = int(row["restart_count"] or 0)
    requeued = recorded_restarts > 0 or (obs.restart_count or 0) > 0
    submitted = parse_iso(obs.submit_time)
    if submitted is None:
        problems.append("submit time unknown: cannot verify identity")
    elif row["job_submit_time"]:
        recorded = parse_iso(row["job_submit_time"])
        if recorded is not None:
            drift = (submitted - recorded).total_seconds()
            if drift < -SUBMIT_TIME_TOLERANCE_S or (not requeued and drift > SUBMIT_TIME_TOLERANCE_S):
                problems.append(f"submit time {obs.submit_time} != recorded {row['job_submit_time']}")
    else:
        lo = _dispatch_time(row) - timedelta(seconds=skew_s)
        hi = (parse_iso(row["identified_at"]) or now) + timedelta(seconds=skew_s)
        if submitted < lo or (not requeued and submitted > hi):
            problems.append(f"submit time {obs.submit_time} is outside this attempt's dispatch window")
    if obs.restart_count is not None and obs.restart_count < recorded_restarts:
        problems.append(f"restart count {obs.restart_count} < recorded {row['restart_count']}")
    return problems


def _cancel_outcome(
    ctx: ServiceContext,
    attempt_id: str,
    *,
    mode: str,
    cancel_state: CancelState,
    event_type: str,
    payload: dict[str, Any],
    op_state: str,
    op_stage: str,
    op_error: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Record what this ``cancel.execute`` run did for the intent it acted on (``mode``).

    The intent is re-read in the outcome tx because observe or a newer ``run_cancel`` may have
    changed it while the scheduler call ran. Settled elsewhere (observe confirmed it) -> the
    pending operations are finished. Superseded by a request with another mode -> the action
    is audited as an event only and False is returned: the caller re-runs the step for the new
    intent (its ``cancel.execute`` enqueue was deduplicated against this leased item).
    """
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is None:
            return True
        current_mode = cur["cancel_mode"] or "cancel"
        if cur["cancel_state"] != CancelState.REQUESTED or current_mode != mode:
            if op_stage in ("sent", "signal_sent"):  # a scheduler action happened: keep it on record
                tx.event(
                    "attempt",
                    attempt_id,
                    event_type,
                    {
                        **payload,
                        "intent_changed": {"cancel_state": cur["cancel_state"], "mode": current_mode},
                    },
                    entity_version=cur["version"],
                )
            if cur["cancel_state"] != CancelState.REQUESTED:
                _close_waiting_cancel_ops(tx, attempt_id, cur["cancel_state"])
                return True
            return False
        _update_attempt(tx, cur, {"cancel_state": cancel_state.value, **(extra or {})}, event_type, payload)
        pending = _waiting_cancel_ops(tx, attempt_id)
        _op(
            tx,
            pending[-1] if pending else None,
            state=op_state,
            stage=op_stage,
            error=op_error,
            result={"attempt_id": attempt_id, "cancel_state": cancel_state.value, **payload}
            if op_state == "succeeded"
            else None,
        )
    return True


def _close_waiting_cancel_ops(tx: Tx, attempt_id: str, cancel_state: str) -> None:
    """The intent was settled elsewhere (e.g. observe saw the job end and confirmed it):
    finish cancel operations that would otherwise wait forever."""
    for op in _waiting_cancel_ops(tx, attempt_id):
        if cancel_state == CancelState.CONFIRMED:
            _op(
                tx,
                op,
                state="succeeded",
                stage="already_terminal",
                result={"attempt_id": attempt_id, "cancel_state": cancel_state},
            )
        elif cancel_state in (CancelState.REFUSED, CancelState.FAILED, CancelState.NONE):
            _op(
                tx,
                op,
                state="failed",
                stage=f"cancel_{cancel_state}",
                error=_err(ErrorCode.PRECONDITION_FAILED, f"cancel ended as {cancel_state}"),
            )


def _close_cancel_ops(ctx: ServiceContext, attempt_id: str, cancel_state: str) -> None:
    if not _waiting_cancel_ops(ctx.db, attempt_id):
        return
    with ctx.db.tx(source="worker") as tx:
        cur = _attempt(tx, attempt_id)
        if cur is not None and cur["cancel_state"] == cancel_state:
            _close_waiting_cancel_ops(tx, attempt_id, cancel_state)


def handle_cancel(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """``cancel.execute``: live identity check, then scancel (or the checkpoint signal)."""
    aid = _payload_attempt(item)
    row = _attempt(ctx.db, aid)
    if row is None:
        return HandlerResult.done("attempt not found")
    if row["cancel_state"] != CancelState.REQUESTED:
        _close_cancel_ops(ctx, aid, row["cancel_state"])
        return HandlerResult.done("no pending cancel")
    # the newest pending cancel operation (a later request may have superseded the item's one)
    pending = _waiting_cancel_ops(ctx.db, aid)
    op_id = pending[-1] if pending else item.operation_id
    mode = row["cancel_mode"] or "cancel"

    def outcome(note: str, **kw: Any) -> HandlerResult:
        if _cancel_outcome(ctx, aid, mode=mode, **kw):
            return HandlerResult.done(note)
        return HandlerResult.retry(0.0, "a newer cancel request replaced the one just executed")

    phase = AttemptPhase(row["phase"])
    if phase not in _ACTIVE_BOUND:
        if phase == AttemptPhase.TERMINAL:
            return outcome(
                f"attempt is {phase.value}",
                cancel_state=CancelState.CONFIRMED,
                event_type="attempt.cancel_confirmed",
                payload={"note": f"job already terminal ({row['scheduler_state']}); nothing sent"},
                op_state="succeeded",
                op_stage="already_terminal",
            )
        return HandlerResult.done(f"attempt is {phase.value}")
    transport = ctx.clusters.transport(row["cluster_id"])
    scheduler = ctx.clusters.scheduler(row["cluster_id"])
    try:
        remote_user = transport.remote_user()
        obs, sources = _live_lookup(scheduler, row["job_id"])
    except AuthRequiredError as exc:
        return _blocked(ctx, row, exc, step="cancel")
    except TransportError as exc:
        if _is_access_problem(exc):
            return _blocked(ctx, row, exc, step="cancel")
        raise  # transient: the outbox retries and the identity is verified again
    _unblock(ctx, op_id, stage="verifying_identity")
    if obs is None:
        if item.tries >= int(_setting(ctx, "cancel_max_tries")):
            return outcome(
                "job not visible",
                cancel_state=CancelState.FAILED,
                event_type="attempt.cancel_failed",
                payload={
                    "reason": "job not visible in squeue or sacct; nothing was cancelled",
                    "sources": sources,
                },
                op_state="failed",
                op_stage="job_not_visible",
                op_error=_err(
                    ErrorCode.UNAVAILABLE, "job not visible in squeue or sacct; nothing was cancelled"
                ),
            )
        return HandlerResult.retry(_setting(ctx, "cancel_retry_s"), "job not visible yet")
    problems = identity_problems(
        row, obs, remote_user=remote_user, now=ctx.clock.now(), skew_s=_setting(ctx, "submit_time_skew_s")
    )
    observed = {
        "job_id": obs.job_id,
        "user": obs.user,
        "name": obs.name,
        "submit_time": obs.submit_time,
        "restart_count": obs.restart_count,
        "source": obs.source,
    }
    if problems:
        log.warning("attempt %s: cancel refused, identity mismatch: %s", aid, problems)
        return outcome(
            "refused",
            cancel_state=CancelState.REFUSED,
            event_type="attempt.cancel_refused",
            payload={"problems": problems, "observed": observed},
            op_state="failed",
            op_stage="identity_mismatch",
            op_error=_err(
                ErrorCode.UNAUTHORIZED,
                "live job identity does not match the attempt; nothing was cancelled",
                problems=problems,
                observed=observed,
            ),
        )
    if obs.state in SCHEDULER_TERMINAL:
        return outcome(
            "already terminal",
            cancel_state=CancelState.CONFIRMED,
            event_type="attempt.cancel_confirmed",
            payload={"note": f"job already {obs.state.value}; nothing sent", "observed": observed},
            op_state="succeeded",
            op_stage="already_terminal",
        )
    try:
        if mode == "request_checkpoint":
            _stop, sig = _stop_behavior(ctx.db, row)
            signal_job = getattr(scheduler, "signal_job", None)
            if signal_job is None:
                raise Unsupported("this scheduler adapter cannot send signals (signal_job missing)")
            resp = signal_job(row["job_id"], sig, batch=True)
        else:
            sig = None
            resp = scheduler.cancel(row["job_id"])
    except AuthRequiredError as exc:
        return _blocked(ctx, row, exc, step="cancel")
    except (Unsupported, TransportError) as exc:
        if isinstance(exc, TransportError) and exc.code != ErrorCode.UNSUPPORTED:
            if _is_access_problem(exc):
                return _blocked(ctx, row, exc, step="cancel")
            raise  # e.g. ConnectionLost from scancel: retried, identity verified again first
        return outcome(
            "unsupported",
            cancel_state=CancelState.FAILED,
            event_type="attempt.cancel_failed",
            payload={"reason": exc.message},
            op_state="failed",
            op_stage="unsupported",
            op_error=_exc_error(exc),
        )
    if resp.accepted:
        if mode == "request_checkpoint":
            return outcome(
                "sent",
                cancel_state=CancelState.SENT,
                event_type="attempt.checkpoint_signal_sent",
                payload={"signal": sig, "job_id": row["job_id"], "observed": observed},
                op_state="succeeded",
                op_stage="signal_sent",
                extra={"signal_sent_at": iso(ctx.clock.now())},
            )
        return outcome(
            "sent",
            cancel_state=CancelState.SENT,
            event_type="attempt.cancel_sent",
            payload={"job_id": row["job_id"], "observed": observed},
            op_state="succeeded",
            op_stage="sent",
        )
    return outcome(
        "scheduler refused",
        cancel_state=CancelState.FAILED,
        event_type="attempt.cancel_failed",
        payload={"raw": resp.raw[-500:], "job_id": row["job_id"]},
        op_state="failed",
        op_stage="scheduler_refused",
        op_error=_err(ErrorCode.UNAVAILABLE, f"the scheduler did not accept the request: {resp.raw[-300:]}"),
    )


# ---------------------------------------------------------------------------
# Human resolution
# ---------------------------------------------------------------------------


def _last_gap_reason(reader: Any, attempt_id: str) -> str | None:
    ev = reader.one(
        "SELECT payload_json FROM events WHERE entity_type='attempt' AND entity_id=? "
        "AND event_type='attempt.observation_gap' ORDER BY seq DESC LIMIT 1",
        (attempt_id,),
    )
    payload = jload(ev["payload_json"], {}) if ev else {}
    why = payload.get("reason") if isinstance(payload, dict) else None
    return str(why) if why else None


def _abandon(tx: Tx, row: dict[str, Any], reason: str) -> tuple[dict[str, Any], list[str]]:
    """ABANDONED: claim released, budget kept as *unknown*. From SUBMISSION_UNCERTAIN /
    DUPLICATE_DETECTED (no job bound) or - the last resort - from OBSERVING while hpclab cannot
    observe the bound job (observation gap, e.g. a requeue chain accounting can never verify):
    that job may still run and is never cancelled here."""
    observing = row["phase"] == AttemptPhase.OBSERVING
    job = row["job_id"]
    if observing:
        warnings = [
            f"WARNING: Slurm job {job} may still be queued or running. hpclab stops observing it now and "
            f"cancels nothing: check `squeue -j {job}` / `sacct -j {job}` and, if it is still there, cancel it "
            f"by hand (`scancel {job}`) - otherwise it keeps running (and using GPU-hours) outside hpclab, "
            "next to any new attempt of this run",
            "the budget reservation is kept as 'unknown' consumption (never released to zero)",
        ]
        gap = _last_gap_reason(tx, row["attempt_id"])
        if gap:
            warnings.append(f"why hpclab could not observe it: {gap}")
        if row["cancel_state"] == CancelState.SENT:
            warnings.append(
                f"a cancel was sent to job {job} earlier, but no observation confirmed that it ended"
            )
    else:
        warnings = [
            "abandoning cancels nothing on the cluster: if a job named "
            f"{row['job_name']} exists, verify it (squeue/sacct) and cancel it yourself",
            "the budget reservation is kept as 'unknown' consumption (never released to zero)",
        ]
    if row["phase"] == AttemptPhase.DUPLICATE_DETECTED:
        ev = tx.one(
            "SELECT payload_json FROM events WHERE entity_type='attempt' AND entity_id=? "
            "AND event_type='attempt.duplicate_detected' ORDER BY seq DESC LIMIT 1",
            (row["attempt_id"],),
        )
        ids = [
            str(c.get("job_id")) for c in (jload(ev["payload_json"], {}) if ev else {}).get("candidates", [])
        ]
        if ids:
            warnings.insert(0, f"jobs {', '.join(ids)} may still be queued or running")
    values: dict[str, Any] = {
        "resolution": f"abandon: {reason}"[:2000],
        "reason": f"abandoned: {reason}"[:2000],
    }
    if row["cancel_state"] == CancelState.REQUESTED:
        values["cancel_state"] = CancelState.FAILED.value
        warnings.append(
            f"the pending cancel request was dropped: nothing was sent to job {job}; cancel it yourself"
            if observing
            else "the pending cancel request could not run: no job was identified"
        )
    payload: dict[str, Any] = {"reason": reason, "budget": "unknown"}
    if observing:
        payload.update(job_id=job, warning=f"job {job} may still be running; nothing was cancelled")
    new = _transition(tx, row, AttemptPhase.ABANDONED, values, "attempt.abandoned", payload)
    _release_claim(tx, new, run_status="unknown")
    try:
        budget.settle(
            tx, attempt_id=new["attempt_id"], consumed_gpu_hours=None, note=f"abandoned: {reason}"[:500]
        )
    except NotFound:
        warnings.append("the attempt had no budget reservation")
    if not observing:  # an observed attempt's submission succeeded: its operation stays so
        _op(
            tx,
            new["operation_id"],
            state="failed",
            stage="abandoned",
            error=_err(ErrorCode.SUBMISSION_UNCERTAIN, "abandoned by a human decision", reason=reason),
        )
    for op in _waiting_cancel_ops(tx, new["attempt_id"]):
        _op(
            tx,
            op,
            state="failed",
            stage="abandoned",
            error=_err(ErrorCode.PRECONDITION_FAILED, "attempt abandoned"),
        )
    return new, warnings


def _check_resolvable(tx: Tx, row: dict[str, Any], resolution: str, *, now: datetime) -> list[ExecutionHold]:
    """Raise PreconditionFailed unless ``resolution`` applies to the attempt's phase; for
    ``confirm_ended`` returns the open holds the confirmation lifts."""
    aid, phase = row["attempt_id"], row["phase"]
    if resolution == "confirm_ended":
        if phase != AttemptPhase.TERMINAL.value:
            raise PreconditionFailed(
                f"attempt {aid} is {phase}; confirm_ended applies to a terminal attempt whose job may still run",
                details={"phase": phase},
            )
        holds = _attempt_holds(tx, row, now=now)
        if not holds:
            raise PreconditionFailed(
                f"nothing to confirm: attempt {aid} has no open requeue watch and no duplicate-owner job",
                details={"phase": phase, "scheduler_state": row["scheduler_state"]},
            )
        return holds
    if phase in _UNRESOLVED:
        return []
    if phase == AttemptPhase.OBSERVING.value and resolution == "abandon":
        if not row["observation_gap"]:
            raise PreconditionFailed(
                f"attempt {aid} is observing job {row['job_id']} and hpclab still sees it: cancel it with "
                "run_cancel; abandon is the last resort for a job hpclab cannot observe any more (observation gap)",
                details={"phase": phase, "observation_gap": False, "job_id": row["job_id"]},
            )
        return []
    raise PreconditionFailed(
        f"attempt {aid} is {phase}; only uncertain or duplicate attempts are resolved by hand"
        + (
            " (an unobservable observing attempt can only be abandoned)"
            if phase == AttemptPhase.OBSERVING
            else ""
        ),
        details={"phase": phase},
    )


def _confirm_evidence(
    scheduler: Scheduler, row: dict[str, Any], holds: list[ExecutionHold]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Live look (squeue, else sacct) at each job whose end a human confirms. A job the scheduler
    still shows queued/running, or requeued since hpclab saw it end, is refused: a confirmation
    records what nobody can observe, never overrides what the scheduler says."""
    evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    token = str(row["submit_token"] or "")
    for hold in holds:
        obs, sources = _live_lookup(scheduler, hold.job_id)
        if obs is not None and token and token_from_job_name(obs.name or "") not in (None, token[:8]):
            obs = None  # the id now names another job (id reuse): the one in question is gone
        if obs is None:
            warnings.append(
                f"job {hold.job_id} is not visible in squeue or sacct ({', '.join(f'{k}={v}' for k, v in sources.items())}): "
                "its end is recorded on your word"
            )
            evidence.append({"job_id": hold.job_id, "hold": hold.kind, "state": None, "sources": sources})
            continue
        requeued = hold.kind == "requeue_watch" and (
            (obs.restart_count or 0) > int(row["restart_count"] or 0)
            or _submitted_later(obs.submit_time, row["job_submit_time"])
        )
        if obs.state not in SCHEDULER_TERMINAL or requeued:
            what = "was requeued" if requeued else "has not ended"
            raise PreconditionFailed(
                f"job {hold.job_id} {what}: {obs.source} shows it {obs.state.value}"
                + (f" (restart {obs.restart_count})" if obs.restart_count else "")
                + "; nothing is confirmed",
                details={
                    "job_id": hold.job_id,
                    "hold": hold.kind,
                    "observed_state": obs.state.value,
                    "restart_count": obs.restart_count,
                    "hint": (
                        "hpclab observes the requeued job and reopens the attempt by itself"
                        if hold.kind == "requeue_watch"
                        else f"wait for it to end, or cancel it yourself (`scancel {hold.job_id}`; hpclab never "
                        "cancels a job it did not bind), then confirm again"
                    ),
                },
            )
        evidence.append(
            {
                "job_id": hold.job_id,
                "hold": hold.kind,
                "state": obs.state.value,
                "source": obs.source,
                "end_time": obs.end_time,
                "restart_count": obs.restart_count,
            }
        )
    return evidence, warnings


def _submitted_later(observed: str | None, recorded: str | None) -> bool:
    seen, known = parse_iso(observed), parse_iso(recorded)
    return seen is not None and known is not None and (seen - known).total_seconds() > SUBMIT_TIME_TOLERANCE_S


def attempt_resolve(
    ctx: ServiceContext,
    *,
    attempt_id: str,
    expected_version: int,
    resolution: Resolution,
    reason: str,
    idempotency_key: str,
    job_id: str | None = None,
) -> AttemptResolveData:
    """Explicit, audited HUMAN decision (CLI only).

    ``abandon``: an uncertain or duplicate submission ends ABANDONED, the run's claim is
    released, the budget reservation stays as *unknown* consumption. It is also the LAST RESORT
    for an OBSERVING attempt whose job hpclab cannot observe any more (observation gap, e.g. a
    requeue chain accounting can never verify): same outcome, and the response warns loudly
    that the job may still be running and must be cancelled by hand - hpclab cancels nothing.
    ``bind_job``: binds ``job_id`` to an uncertain/duplicate attempt after a live identity check
    (owner + token; a submit time outside the dispatch window is a warning).
    ``confirm_ended``: lifts the execution hold of a TERMINAL attempt (its job ended in a state
    Slurm may requeue from, or another job owns its attempt dir) after a live squeue/sacct look:
    a job still queued/running or requeued is refused; one nobody can see is confirmed on the
    human's word with a warning.
    """
    _require_key(idempotency_key)
    if resolution not in _RESOLUTIONS:
        raise InvalidArgument(f"resolution must be one of {', '.join(_RESOLUTIONS)}")
    if not reason or not reason.strip():
        raise InvalidArgument("a reason is required for a human resolution (it is recorded in the audit log)")
    if resolution == "bind_job" and not job_id:
        raise InvalidArgument("resolution=bind_job requires job_id")
    payload_hash = digest(
        {
            "attempt_id": attempt_id,
            "expected_version": expected_version,
            "resolution": resolution,
            "reason": reason,
            "job_id": job_id,
        }
    )

    def check(tx: Tx) -> tuple[AttemptResolveData | None, dict[str, Any], list[ExecutionHold]]:
        stored = tx.idem_lookup(ctx.principal, TOOL_RESOLVE, idempotency_key, payload_hash)
        if stored is not None:
            return AttemptResolveData.model_validate(stored), {}, []
        row = _require_attempt(tx, attempt_id)
        if int(row["version"]) != int(expected_version):
            raise VersionConflict(
                f"attempt {attempt_id} is at version {row['version']}, expected {expected_version}",
                details={"current_version": row["version"], "expected_version": expected_version},
            )
        return None, row, _check_resolvable(tx, row, resolution, now=ctx.clock.now())

    with ctx.db.tx(source="human") as tx:
        stored, row, holds = check(tx)
    if stored is not None:
        return stored

    obs: JobObservation | None = None
    warnings: list[str] = []
    evidence: list[dict[str, Any]] = []
    if resolution == "bind_job":
        assert job_id is not None
        transport = ctx.clusters.transport(row["cluster_id"])
        scheduler = ctx.clusters.scheduler(row["cluster_id"])
        remote_user = transport.remote_user()
        obs, sources = _live_lookup(scheduler, job_id)
        if obs is None:
            raise PreconditionFailed(
                f"job {job_id} is not visible in squeue or sacct; cannot verify its identity",
                details={"job_id": job_id, "sources": sources},
            )
        ours, problems = _token_problems(obs, row["submit_token"])
        if not ours:
            problems = [*problems, "job does not carry this attempt's token"]
        if obs.user != remote_user:
            problems.append(f"owner {obs.user!r} is not the cluster user {remote_user!r}")
        if problems:
            raise PreconditionFailed(
                f"job {job_id} is not provably this attempt's job; not bound",
                details={
                    "job_id": job_id,
                    "problems": problems,
                    "observed_name": obs.name,
                    "observed_user": obs.user,
                },
            )
        submitted = parse_iso(obs.submit_time)
        skew = timedelta(seconds=_setting(ctx, "submit_time_skew_s"))
        if submitted is None or submitted < _dispatch_time(row) - skew:
            warnings.append(f"job submit time {obs.submit_time} is not after this attempt's dispatch")
    elif resolution == "confirm_ended":
        evidence, warnings = _confirm_evidence(ctx.clusters.scheduler(row["cluster_id"]), row, holds)

    with ctx.db.tx(source="human") as tx:
        stored, row, holds_now = check(tx)
        if stored is not None:
            return stored
        if resolution == "abandon":
            new, warnings = _abandon(tx, row, reason)
        elif resolution == "confirm_ended":
            confirmed = {h.job_id for h in holds}
            unchecked = [h.job_id for h in holds_now if h.job_id not in confirmed]
            if unchecked:  # a hold appeared between the live look and now: look again
                raise PreconditionFailed(
                    f"job(s) {', '.join(unchecked)} of attempt {attempt_id} were not checked; confirm again",
                    retryable=True,
                    details={"job_ids": unchecked},
                )
            new = _update_attempt(
                tx,
                row,
                {"resolution": f"confirm_ended: {reason}"[:2000]},
                EVENT_END_CONFIRMED,
                {
                    "job_ids": [h.job_id for h in holds_now],
                    "holds": [h.kind for h in holds_now],
                    "restart_count": int(row["restart_count"] or 0),
                    "reason": reason,
                    "evidence": evidence,
                    "warnings": warnings,
                },
            )
        else:
            assert obs is not None
            new = _bind_job(
                tx,
                row,
                job_id=obs.job_id,
                owner=obs.user,
                submit_time=obs.submit_time,
                source="human",
                evidence={"reason": reason, "warnings": warnings, "observed_state": obs.state.value},
            )
            version = tx.update(
                "attempts",
                "attempt_id",
                attempt_id,
                {"resolution": f"bind_job {obs.job_id}: {reason}"[:2000]},
                expected_version=new["version"],
            )
            tx.event(
                "attempt",
                attempt_id,
                "attempt.resolved",
                {"resolution": "bind_job", "job_id": obs.job_id, "reason": reason},
                entity_version=version,
            )
            new = dict(new, version=version)
        data = AttemptResolveData(
            attempt_id=attempt_id,
            run_id=row["run_id"],
            resolution=resolution,
            phase=new["phase"],
            job_id=new.get("job_id"),
            attempt_version=int(new["version"]),
            warnings=warnings,
        )
        tx.idem_store(
            ctx.principal, TOOL_RESOLVE, idempotency_key, payload_hash, data.model_dump(mode="json")
        )
    return data


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def outbox_handlers() -> dict[str, Handler]:
    return {
        KIND_STAGE: handle_stage,
        KIND_DISPATCH: handle_dispatch,
        KIND_RECONCILE: handle_reconcile,
        KIND_CANCEL: handle_cancel,
    }


def periodic_tasks() -> list[PeriodicTask]:
    from hpclab.services.periodic import PeriodicTask

    return [
        PeriodicTask(
            name=SWEEP_TASK, interval_s=DEFAULTS["sweep_interval_s"], fn=reconcile_sweep, jitter_s=5.0
        )
    ]


__all__ = [
    "AdmitResult",
    "AttemptResolveData",
    "CancelReceiptData",
    "ExecutionHold",
    "SubmitReceiptData",
    "admit",
    "attempt_resolve",
    "check_duplicate_owners",
    "execution_holds",
    "gate_state",
    "handle_cancel",
    "handle_dispatch",
    "handle_reconcile",
    "handle_stage",
    "identity_problems",
    "outbox_handlers",
    "periodic_tasks",
    "reconcile_sweep",
    "request_cancel",
    "run_cancel",
    "run_submit",
]
