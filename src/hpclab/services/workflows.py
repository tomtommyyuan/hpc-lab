"""Workflows: one durable execution group per frozen plan (design tools.md §04/§15, system §6.5).

``workflow_submit`` persists every cell of the plan up front (``workflow_cells``, state
``planned``) and hands dispatching to the controller (outbox ``workflow.dispatch``), so the
concurrency limit never depends on an agent's conversation loop or on controller memory:

* **Slots** are recomputed inside every dispatch transaction from persisted attempt phases
  (``SLOT_HOLDING_PHASES`` of attempts this workflow dispatched or adopted). Dispatch rounds
  run under ``BEGIN IMMEDIATE``, so concurrent rounds (sweep + finalize + restart) serialize and
  ``max_parallel`` holds across restarts.
* **Admission** goes through ``submission.admit`` - the single gate shared with ``run_submit``
  (claim per Run, consume-once plans, experiment limits, budget) - one savepoint per cell, and
  the per-cell dispatch key ``wf:<workflow_id>:<run_id>`` is recorded as an idempotency row in
  the same transaction. A cell is dispatched at most once; failed or cancelled cells are never
  re-dispatched automatically (recovery is an explicit ``run_recover`` + a new plan).
* **Smoke gate** (two-stage): only the smoke cell is dispatched until its attempt COMPLETED with
  ``result_state=valid`` and the spec's ``smoke.require_outputs`` finalized; a failed smoke sets
  ``gate_failed`` and no main cell is dispatched. ``resume_dispatch`` re-evaluates the gate on the
  smoke Run's latest attempt (e.g. after a recovery of the smoke run).
* **Circuit breaker** (default on): >= 2 distinct cells whose failures share an
  environment-class fingerprint (diagnosis category + code + normalized last error line; the
  wrapper guard counts) pause dispatch - in-flight work and reservations are untouched and the
  evidence is stored in ``breaker_json``. Infrastructure failures (node_fail, preempted,
  timeout, ...) never trip it. ``resume_dispatch`` + ``expected_version`` reopens dispatch and
  acknowledges the evidence so the same failures do not trip it again.
* **Budget** exhaustion (``BudgetExceeded`` from admission) stops the round with the reason on
  the workflow's operation (``blocked`` when nothing is in flight to free budget).

``workflow_cancel`` closes dispatch in the same transaction that bumps the version, so a
dispatch round either committed before (its submissions count as in flight - never pretended
revoked) or sees dispatch closed. Never-dispatched cells become ``cancelled``; ``all_managed``
additionally requests ``run_cancel`` for every active attempt from the ``workflow.cancel``
handler (live identity checks and reconcile-first for uncertain submissions are the submission
service's contract). In-flight/uncertain attempts keep their slots and budget until terminal.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.api.envelope import Coverage, clamp_limit, decode_cursor, encode_cursor
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import (
    AdmissionDenied,
    BudgetExceeded,
    ErrorCode,
    HpcLabError,
    InvalidArgument,
    NotFound,
    NotReady,
    PreconditionFailed,
    VersionConflict,
)
from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id
from hpclab.domain.states import SLOT_HOLDING_PHASES, AttemptPhase, SchedulerState
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import CellStatus, cell_statuses, summarize_codes
from hpclab.services.diagnosis import CODE_CATEGORY, normalize_error_line
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.services.recovery import recovery_plan_cells
from hpclab.services.runs import NEW_KEY, call
from hpclab.storage.db import Database, Tx, jdump, jload

log = logging.getLogger("hpclab.workflows")

__all__ = [
    "CircuitBreakerPolicy",
    "DispatchRound",
    "FailurePolicy",
    "WorkflowCancelData",
    "WorkflowCell",
    "WorkflowData",
    "WorkflowSubmitData",
    "cancel_step",
    "cell_state_for",
    "dispatch_workflow",
    "outbox_handlers",
    "periodic_tasks",
    "sweep",
    "workflow_cancel",
    "workflow_get",
    "workflow_submit",
]

DISPATCH_KIND = "workflow.dispatch"
CANCEL_KIND = "workflow.cancel"
SWEEP_TASK = "workflow.sweep"
SWEEP_INTERVAL_S = 60.0
TOOL_SUBMIT = "workflow_submit"
TOOL_CANCEL = "workflow_cancel"
TOOL_DISPATCH_CELL = "workflow.dispatch_cell"
OP_DISPATCH = "workflow_dispatch"
OP_CANCEL = "workflow_cancel"
POLL_AFTER_MS = 2000
MAX_BREAKER_HISTORY = 10
MAX_CANCEL_CELLS = 100
DEFAULT_MAX_PARALLEL = 4
DEFAULT_STALE_AFTER_S = 900.0

WorkflowState = Literal[
    "gating", "dispatching", "draining", "completed", "cancelling", "cancelled", "gate_failed"
]
CellState = Literal[
    "planned",
    "dispatched",
    "active",
    "uncertain",
    "validating",
    "succeeded",
    "failed",
    "cancelled",
    "skipped",
]
CancelScope = Literal["queued_only", "all_managed"]

_OPEN_STATES = ("gating", "dispatching", "draining")
_INFLIGHT_CELLS = frozenset({"dispatched", "active", "uncertain"})
_SLOT_PHASES = tuple(sorted(p.value for p in SLOT_HOLDING_PHASES))
# sbatch refusals that are about *when*, not *what*: they never trip the breaker
_RETRY_LATER_REJECTIONS = frozenset({"submit_limit", "partition_unavailable", "controller_busy"})


def workflow_ref(workflow_id: str) -> str:
    return f"hpclab://workflows/{workflow_id}"


def _op_ref(op_id: str | None) -> str | None:
    return f"hpclab://operations/{op_id}" if op_id else None


def _dispatch_dedupe(workflow_id: str) -> str:
    return f"workflow.dispatch:{workflow_id}"


def _cancel_dedupe(workflow_id: str) -> str:
    return f"workflow.cancel:{workflow_id}"


def dispatch_key(workflow_id: str, run_id: str) -> str:
    return f"wf:{workflow_id}:{run_id}"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


BreakerCategory = Literal["environment", "program", "numerics", "results"]


def _default_categories() -> list[BreakerCategory]:
    return ["environment"]


class CircuitBreakerPolicy(BaseModel):
    """Pause dispatch when ``threshold`` distinct cells fail with the same fingerprint of a
    listed diagnosis category. Infrastructure/resource failures are never counted."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threshold: int = Field(default=2, ge=2, le=64)
    categories: list[BreakerCategory] = Field(default_factory=_default_categories)
    include_rejections: bool = True  # identical definite sbatch refusals (bad account, bad gres, ...)


class FailurePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkflowCell(_Out):
    ordinal: int
    run_id: str
    label: str
    kind: str
    cell: dict[str, Any]
    seed: int | None
    cell_state: str
    code: str  # services.coverage cell code of the Run (V E R Q S P F C! X U ?); E is not a failure
    reason: str | None
    attempt_id: str | None
    phase: str | None
    scheduler_state: str | None
    result_state: str | None
    job_id: str | None
    cancel_state: str | None
    note: str | None
    dispatch_key: str


class GateInfo(_Out):
    run_id: str
    state: str  # pending | passed | failed
    attempt_id: str | None = None
    reason: str | None = None


class BreakerInfo(_Out):
    state: str  # tripped | cleared
    tripped_at: str | None = None
    cleared_at: str | None = None
    fingerprint: dict[str, Any] | None = None
    cells: list[dict[str, Any]] = Field(default_factory=list)
    threshold: int | None = None
    acknowledged_attempts: int = 0
    history_count: int = 0
    resume_hint: str | None = None


class CancelInfo(_Out):
    scope: str | None
    operation_id: str | None
    operation_ref: str | None
    operation_state: str | None


class WorkflowBudget(_Out):
    reserved_gpu_hours: float
    settled_gpu_hours: float
    unknown_reservations: int
    released_reservations: int
    attempts_with_reservation: int


class WorkflowSubmitData(_Out):
    workflow_id: str
    workflow_ref: str
    experiment_id: str
    plan_id: str
    plan_kind: str
    plan_hash: str
    operation_id: str
    operation_ref: str
    poll_after_ms: int = POLL_AFTER_MS
    state: str
    dispatch_state: str
    version: int
    max_parallel: int
    planned_run_count: int
    skipped_count: int
    gate: GateInfo | None = None
    resumed: bool = False
    cells: list[WorkflowCell] = Field(default_factory=list)
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)


class CancelCellView(_Out):
    run_id: str
    label: str
    cell_state: str
    attempt_id: str | None
    phase: str | None
    cancel_state: str | None
    note: str | None


class WorkflowCancelData(_Out):
    workflow_id: str
    workflow_ref: str
    operation_id: str | None
    operation_ref: str | None
    poll_after_ms: int = POLL_AFTER_MS
    scope: str
    state: str
    dispatch_state: str
    version: int
    counts: dict[str, int]
    cells: list[CancelCellView] = Field(default_factory=list)
    budget_kept_gpu_hours: float | None = None
    notes: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)


class GridRow(_Out):
    treatment: dict[str, Any]
    codes: dict[str, str]  # seed -> cell code


class WorkflowData(_Out):
    workflow_id: str
    workflow_ref: str
    experiment_id: str
    plan_id: str
    plan_kind: str | None
    plan_hash: str
    state: str
    dispatch_state: str
    dispatch_open: bool
    version: int
    max_parallel: int
    slots_used: int
    gate: GateInfo | None = None
    breaker: BreakerInfo | None = None
    failure_policy: FailurePolicy
    cancel: CancelInfo | None = None
    operation_id: str | None = None
    operation_ref: str | None = None
    operation_state: str | None = None
    operation_stage: str | None = None
    dispatch_blocked_reason: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    codes: dict[str, int] = Field(default_factory=dict)
    budget: WorkflowBudget | None = None
    cells: list[WorkflowCell] = Field(default_factory=list)
    grid: list[GridRow] | None = None
    created_at: str
    updated_at: str
    next_cursor: str | None = None
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _submission() -> Any:
    """``services.submission`` (another area), imported lazily so tests can stub it."""
    try:
        return importlib.import_module("hpclab.services.submission")
    except ImportError as exc:
        raise HpcLabError(
            "the submission service is not available", code=ErrorCode.UNAVAILABLE, retryable=True
        ) from exc


@contextmanager
def _savepoint(tx: Tx, name: str) -> Iterator[None]:
    """Nested rollback scope: a refused admission or a failing notification must not leave
    partial writes in the surrounding dispatch transaction."""
    tx.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        tx.execute(f"ROLLBACK TO SAVEPOINT {name}")
        tx.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        tx.execute(f"RELEASE SAVEPOINT {name}")


def _emit(tx: Tx, *, event_type: str, entity_id: str, title: str, body: str, urgency: str = "batch") -> None:
    """Best-effort notification (services.notify.emit); never breaks the state change."""
    try:
        emit = importlib.import_module("hpclab.services.notify").emit
    except (ImportError, AttributeError):
        return
    try:
        with _savepoint(tx, "hl_notify"):
            emit(
                tx,
                event_type=event_type,
                entity_id=entity_id,
                title=title,
                body=body,
                urgency=urgency,
            )
    except Exception:  # notifications are advisory; the dispatch decision stands
        log.warning("notify.emit(%s) failed for %s", event_type, entity_id, exc_info=True)


_PHASE_CELL: dict[str, str] = {
    AttemptPhase.STAGING.value: "dispatched",
    AttemptPhase.SUBMISSION_IN_FLIGHT.value: "dispatched",
    AttemptPhase.SUBMISSION_UNCERTAIN.value: "uncertain",
    AttemptPhase.DUPLICATE_DETECTED.value: "uncertain",
    AttemptPhase.JOB_IDENTIFIED.value: "active",
    AttemptPhase.OBSERVING.value: "active",
    AttemptPhase.SUBMISSION_REJECTED.value: "failed",
    AttemptPhase.CANCELLED_BEFORE_SUBMIT.value: "cancelled",
    AttemptPhase.ABANDONED.value: "failed",
}


def cell_state_for(phase: str | None, scheduler_state: str | None, result_state: str | None) -> str:
    """Dispatch-lifecycle state of a cell from its own attempt (not the Run's latest attempt:
    a recovery workflow's cell must not look failed because of the attempt it recovers).
    ``validating`` = scheduler COMPLETED, outputs not yet validated (never counted as success)."""
    if phase is None:
        return "planned"
    if phase != AttemptPhase.TERMINAL.value:
        return _PHASE_CELL.get(phase, "uncertain")
    if scheduler_state == SchedulerState.CANCELLED.value:
        return "cancelled"
    if scheduler_state == SchedulerState.COMPLETED.value:
        if result_state == "valid":
            return "succeeded"
        if result_state in ("invalid", "incomplete"):
            return "failed"
        return "validating"
    return "failed"


_CELLS_SQL = """
SELECT c.workflow_id, c.run_id, c.ordinal, c.cell_state, c.attempt_id, c.dispatch_key, c.note,
       r.label, r.kind AS run_kind, r.cell_key, r.seed, r.claim_attempt_id, r.latest_attempt_id,
       r.current_exec_plan_hash,
       a.phase, a.scheduler_state, a.result_state, a.job_id, a.cancel_state, a.version AS attempt_version,
       a.diagnosis_json, a.app_state, a.status_json, a.operation_id AS attempt_operation_id,
       a.reason AS attempt_reason, a.exit_code, a.last_observed_at
FROM workflow_cells c
JOIN runs r ON r.run_id = c.run_id
LEFT JOIN attempts a ON a.attempt_id = c.attempt_id
WHERE c.workflow_id = ?
ORDER BY c.ordinal
"""


def _cells(reader: Database | Tx, workflow_id: str) -> list[dict[str, Any]]:
    return reader.all(_CELLS_SQL, (workflow_id,))


def _load_wf(reader: Database | Tx, workflow_id: str) -> dict[str, Any]:
    wf = reader.one("SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,))
    if wf is None:
        raise NotFound(f"workflow {workflow_id} not found", details={"workflow_id": workflow_id})
    return wf


def _policy(wf: dict[str, Any]) -> FailurePolicy:
    raw = jload(wf.get("failure_policy_json"), {}) or {}
    try:
        return FailurePolicy.model_validate(raw)
    except ValueError:
        return FailurePolicy()


def _counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cells:
        out[c["cell_state"]] = out.get(c["cell_state"], 0) + 1
    return out


def _inflight(cells: list[dict[str, Any]]) -> int:
    return sum(1 for c in cells if c["cell_state"] in _INFLIGHT_CELLS)


# A cell that just failed keeps its slot until its failure is classified (observe.finalize
# stores the diagnosis), so the circuit breaker sees identical environment failures before
# their slots are refilled. Bounded, so a delayed finalize can never wedge a sweep.
UNCLASSIFIED_FAILURE_HOLD_S = 120.0


def _slots_used(reader: Database | Tx, workflow_id: str, *, now: str | None = None) -> int:
    marks = ",".join("?" for _ in _SLOT_PHASES)
    mine = (
        "(workflow_id = ? OR attempt_id IN "
        "(SELECT attempt_id FROM workflow_cells WHERE workflow_id = ? AND attempt_id IS NOT NULL))"
    )
    row = reader.one(
        f"SELECT COUNT(*) AS n FROM attempts WHERE phase IN ({marks}) AND {mine}",
        (*_SLOT_PHASES, workflow_id, workflow_id),
    )
    used = int(row["n"]) if row else 0
    if now is not None:
        base = parse_iso(now)
        assert base is not None
        cutoff = iso(base - timedelta(seconds=UNCLASSIFIED_FAILURE_HOLD_S))
        held = reader.one(
            "SELECT COUNT(*) AS n FROM attempts WHERE phase = 'terminal' AND diagnosis_json IS NULL "
            "AND (scheduler_state != 'COMPLETED' OR app_state IN ('exited_error', 'guard_failed')) "
            f"AND scheduler_state != 'CANCELLED' AND COALESCE(terminal_at, updated_at) >= ? AND {mine}",
            (cutoff, workflow_id, workflow_id),
        )
        used += int(held["n"]) if held else 0
    return used


def _refresh(tx: Tx, workflow_id: str, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-derive each dispatched cell's state from its attempt; one event per round of changes."""
    changes: list[dict[str, Any]] = []
    for c in cells:
        if c["attempt_id"] is None:
            continue
        new = cell_state_for(c["phase"], c["scheduler_state"], c["result_state"])
        if new != c["cell_state"]:
            tx.execute(
                "UPDATE workflow_cells SET cell_state=?, updated_at=? WHERE workflow_id=? AND run_id=?",
                (new, tx.now, workflow_id, c["run_id"]),
            )
            changes.append(
                {"run_id": c["run_id"], "from": c["cell_state"], "to": new, "attempt_id": c["attempt_id"]}
            )
            c["cell_state"] = new
    if changes:
        tx.event("workflow", workflow_id, "workflow.cells_updated", {"changes": changes[:256]})
    return changes


def _set_cell(
    tx: Tx,
    workflow_id: str,
    cell: dict[str, Any],
    *,
    state: str | None = None,
    note: str | None = None,
    attempt_id: str | None = None,
) -> None:
    """Update one cell (only when something changes) and record it."""
    values: dict[str, Any] = {}
    if state is not None and state != cell["cell_state"]:
        values["cell_state"] = state
    if note != cell.get("note"):
        values["note"] = note
    if attempt_id is not None and attempt_id != cell.get("attempt_id"):
        values["attempt_id"] = attempt_id
    if not values:
        return
    values["updated_at"] = tx.now
    sets = ", ".join(f"{k}=?" for k in values)
    tx.execute(
        f"UPDATE workflow_cells SET {sets} WHERE workflow_id=? AND run_id=?",
        (*values.values(), workflow_id, cell["run_id"]),
    )
    tx.event(
        "workflow",
        workflow_id,
        "workflow.cell_updated",
        {"run_id": cell["run_id"], **{k: v for k, v in values.items() if k != "updated_at"}},
    )
    cell.update(values)


def _set_op(
    tx: Tx,
    op_id: str | None,
    *,
    state: str,
    stage: str,
    progress: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Operation update that only writes (and records an event) when something changed, and
    clears a previous error when the operation is no longer blocked/failed."""
    if not op_id:
        return
    row = tx.one(
        "SELECT state, stage, progress_json, error_json, result_json FROM operations WHERE operation_id=?",
        (op_id,),
    )
    if row is None:
        return
    values: dict[str, Any] = {}
    if row["state"] != state:
        values["state"] = state
        if state in ("succeeded", "failed"):
            values["finished_at"] = tx.now
    if row["stage"] != stage:
        values["stage"] = stage
    if progress is not None and jdump(progress) != row["progress_json"]:
        values["progress_json"] = jdump(progress)
    err_json = jdump(error) if error is not None else None
    if err_json != row["error_json"] and (error is not None or state not in ("blocked", "failed")):
        values["error_json"] = err_json
    if result is not None and jdump(result) != row["result_json"]:
        values["result_json"] = jdump(result)
    if not values:
        return
    version = tx.update("operations", "operation_id", op_id, values)
    tx.event(
        "operation",
        op_id,
        "operation.updated",
        {k: v for k, v in values.items() if k in ("state", "stage")},
        entity_version=version,
    )


def _cell_dict(cell_key: str | None) -> dict[str, Any]:
    val = jload(cell_key, {}) or {}
    return val if isinstance(val, dict) else {}


def _views(
    reader: Database | Tx, cells: list[dict[str, Any]], *, now: Any, stale_after_s: float | None
) -> tuple[list[WorkflowCell], dict[str, CellStatus]]:
    statuses = cell_statuses(reader, [c["run_id"] for c in cells], stale_after_s=stale_after_s, now=now)
    out = [
        WorkflowCell(
            ordinal=c["ordinal"],
            run_id=c["run_id"],
            label=c["label"],
            kind=c["run_kind"],
            cell=_cell_dict(c["cell_key"]),
            seed=c["seed"],
            cell_state=c["cell_state"],
            code=statuses[c["run_id"]].code,
            reason=statuses[c["run_id"]].reason,
            attempt_id=c["attempt_id"],
            phase=c["phase"],
            scheduler_state=c["scheduler_state"],
            result_state=c["result_state"],
            job_id=c["job_id"],
            cancel_state=c["cancel_state"],
            note=c["note"],
            dispatch_key=c["dispatch_key"],
        )
        for c in cells
    ]
    return out, statuses


# ---------------------------------------------------------------------------
# Plan -> cells
# ---------------------------------------------------------------------------


def _initial_exec_plan(tx: Tx, run: dict[str, Any]) -> str | None:
    """The Run's frozen initial execution plan (never a later recovery plan of the same Run)."""
    rows = tx.all(
        "SELECT exec_plan_hash, plan_json FROM execution_plans WHERE run_id=? ORDER BY created_at, rowid",
        (run["run_id"],),
    )
    initial = [
        r["exec_plan_hash"]
        for r in rows
        if (jload(r["plan_json"], {}) or {}).get("created_reason", "initial") == "initial"
    ]
    cur = run.get("current_exec_plan_hash")
    if cur and cur in initial:
        return str(cur)
    return initial[0] if initial else None


@dataclass
class _PlannedCell:
    run_id: str
    kind: str
    exec_plan_hash: str | None
    skip_note: str | None = None


def _plan_cells(tx: Tx, plan: dict[str, Any]) -> list[_PlannedCell]:
    out: list[_PlannedCell] = []
    if plan["kind"] == "recovery":
        for c in recovery_plan_cells(plan):
            run = tx.one("SELECT run_id, kind FROM runs WHERE run_id=?", (c["run_id"],))
            if run is None:
                continue
            if c.get("eligible") and c.get("exec_plan_hash"):
                out.append(_PlannedCell(run["run_id"], run["kind"], str(c["exec_plan_hash"])))
            else:
                out.append(
                    _PlannedCell(
                        run["run_id"],
                        run["kind"],
                        None,
                        c.get("blocking_reason") or "not eligible for recovery",
                    )
                )
        return out
    runs = tx.all(
        "SELECT * FROM runs WHERE plan_id=? ORDER BY CASE kind WHEN 'smoke' THEN 0 ELSE 1 END, rowid",
        (plan["plan_id"],),
    )
    for run in runs:
        h = _initial_exec_plan(tx, run)
        out.append(
            _PlannedCell(run["run_id"], run["kind"], h, None if h else "no execution plan for this run")
        )
    return out


def _cell_exec_plan(
    tx: Tx, plan: dict[str, Any], cell: dict[str, Any], recovery_map: dict[str, dict[str, Any]] | None
) -> str | None:
    if recovery_map is not None:
        c = recovery_map.get(cell["run_id"])
        return str(c["exec_plan_hash"]) if c and c.get("eligible") and c.get("exec_plan_hash") else None
    run = tx.one("SELECT run_id, current_exec_plan_hash FROM runs WHERE run_id=?", (cell["run_id"],))
    return _initial_exec_plan(tx, run) if run else None


def _resolve_plan(
    tx: Tx, *, plan_id: str | None, experiment_id: str | None, plan_hash: str | None
) -> dict[str, Any]:
    if plan_id is not None:
        plan = tx.one("SELECT * FROM plans WHERE plan_id=?", (plan_id,))
        if plan is None:
            raise NotFound(f"plan {plan_id} not found", details={"plan_id": plan_id})
        if experiment_id is not None and plan["experiment_id"] != experiment_id:
            raise InvalidArgument(
                f"plan {plan_id} belongs to experiment {plan['experiment_id']}, not {experiment_id}"
            )
        return plan
    assert experiment_id is not None
    if tx.one("SELECT 1 AS x FROM experiments WHERE experiment_id=?", (experiment_id,)) is None:
        raise NotFound(f"experiment {experiment_id} not found", details={"experiment_id": experiment_id})
    if plan_hash is not None:
        rows = tx.all(
            "SELECT * FROM plans WHERE experiment_id=? AND plan_hash=? ORDER BY created_at DESC, rowid DESC",
            (experiment_id, plan_hash),
        )
        if not rows:
            raise NotFound(
                f"experiment {experiment_id} has no plan with hash {plan_hash}",
                details={"experiment_id": experiment_id, "plan_hash": plan_hash},
            )
        return rows[0]
    rows = tx.all(
        "SELECT * FROM plans WHERE experiment_id=? AND status='ready' AND kind IN ('matrix', 'derived') "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (experiment_id,),
    )
    if not rows:
        raise NotReady(
            f"experiment {experiment_id} has no ready plan", details={"experiment_id": experiment_id}
        )
    return rows[0]


def _default_max_parallel(ctx: ServiceContext, exp: dict[str, Any]) -> int:
    if exp.get("max_parallel"):
        return int(exp["max_parallel"])
    spec = jload(exp.get("spec_json"), {}) or {}
    mp = ((spec.get("budget") or {}) if isinstance(spec, dict) else {}).get("max_parallel")
    if isinstance(mp, int) and mp > 0:
        return mp
    try:
        return int(ctx.clusters.get(exp["cluster_id"]).profile.policy.max_parallel_default)
    except HpcLabError:
        return DEFAULT_MAX_PARALLEL


# ---------------------------------------------------------------------------
# workflow_submit
# ---------------------------------------------------------------------------


def workflow_submit(
    ctx: ServiceContext,
    *,
    idempotency_key: str,
    plan_id: str | None = None,
    prepared_experiment_id: str | None = None,
    plan_hash: str | None = None,
    max_parallel: Annotated[int, Field(ge=1, le=256)] | None = None,
    failure_policy: FailurePolicy | None = None,
    workflow_id: str | None = None,
    resume_dispatch: bool = False,
    expected_version: int | None = None,
) -> WorkflowSubmitData:
    """Dispatch every cell of a frozen plan (matrix, recovery, probe or derived) with at most
    ``max_parallel`` active at once, smoke gate first, circuit breaker on. One execution group
    per plan: the same plan under a different key is refused with the existing workflow id.
    ``workflow_id`` + ``resume_dispatch=true`` + ``expected_version`` reopens a paused
    (breaker) or gate-failed workflow."""
    payload_hash = digest(
        {
            "plan_id": plan_id,
            "prepared_experiment_id": prepared_experiment_id,
            "plan_hash": plan_hash,
            "max_parallel": max_parallel,
            "failure_policy": failure_policy.model_dump(mode="json") if failure_policy else None,
            "workflow_id": workflow_id,
            "resume_dispatch": resume_dispatch,
            "expected_version": expected_version,
        }
    )
    if resume_dispatch or workflow_id is not None:
        if not resume_dispatch:
            raise InvalidArgument(
                "workflow_id is only used with resume_dispatch=true (read workflows with workflow_get)"
            )
        if workflow_id is None:
            raise InvalidArgument("resume_dispatch needs workflow_id")
        if plan_id is not None or prepared_experiment_id is not None:
            raise InvalidArgument(
                "resume_dispatch takes workflow_id only (not plan_id / prepared_experiment_id)"
            )
        ignored = [
            name
            for name, value in (
                ("max_parallel", max_parallel),
                ("failure_policy", failure_policy),
                ("plan_hash", plan_hash),
            )
            if value is not None
        ]
        if ignored:
            # resuming only reopens dispatch; silently keeping the old limit/policy would run
            # more (or differently guarded) jobs than the caller asked for
            raise InvalidArgument(
                f"resume_dispatch does not change {', '.join(ignored)}: the workflow keeps the values "
                "it was submitted with (read them with workflow_get)",
                details={"ignored": ignored},
            )
        if expected_version is None:
            raise InvalidArgument("resume_dispatch needs expected_version (the workflow version you saw)")
        return _resume(
            ctx,
            workflow_id=workflow_id,
            expected_version=expected_version,
            key=idempotency_key,
            ph=payload_hash,
        )
    if plan_id is None and prepared_experiment_id is None:
        raise InvalidArgument("give plan_id or prepared_experiment_id")
    if max_parallel is not None and not 1 <= max_parallel <= 256:
        # checked here too: the method registry drops Annotated constraints from the signature
        raise InvalidArgument(
            "max_parallel must be between 1 and 256", details={"max_parallel": max_parallel}
        )
    return _create(
        ctx,
        plan_id=plan_id,
        experiment_id=prepared_experiment_id,
        plan_hash=plan_hash,
        max_parallel=max_parallel,
        policy=failure_policy or FailurePolicy(),
        key=idempotency_key,
        ph=payload_hash,
    )


def _create(
    ctx: ServiceContext,
    *,
    plan_id: str | None,
    experiment_id: str | None,
    plan_hash: str | None,
    max_parallel: int | None,
    policy: FailurePolicy,
    key: str,
    ph: str,
) -> WorkflowSubmitData:
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL_SUBMIT, key, ph)
        if stored is not None:
            return WorkflowSubmitData.model_validate(stored)
        plan = _resolve_plan(tx, plan_id=plan_id, experiment_id=experiment_id, plan_hash=plan_hash)
        existing = tx.one(
            "SELECT workflow_id, version, state FROM workflows WHERE plan_id=?", (plan["plan_id"],)
        )
        if existing is not None:
            raise AdmissionDenied(
                f"plan {plan['plan_id']} already has its execution group {existing['workflow_id']} "
                "(one workflow per frozen plan)",
                details={
                    "workflow_id": existing["workflow_id"],
                    "workflow_ref": workflow_ref(existing["workflow_id"]),
                    "version": existing["version"],
                    "state": existing["state"],
                },
            )
        if plan["status"] not in ("ready", "preparing"):
            # an invalid plan never becomes ready: not retryable (fix the findings, prepare again)
            raise PreconditionFailed(
                f"plan {plan['plan_id']} is {plan['status']}; fix its validation findings and prepare again",
                details={"plan_id": plan["plan_id"], "plan_status": plan["status"]},
            )
        if plan["status"] != "ready" or not plan["plan_hash"]:
            raise NotReady(
                f"plan {plan['plan_id']} is {plan['status']}, not ready",
                details={"plan_id": plan["plan_id"], "plan_status": plan["status"]},
            )
        if plan_hash is not None and plan["plan_hash"] != plan_hash:
            raise PreconditionFailed(
                "plan_hash does not match the plan (it is not the plan you reviewed)",
                details={"plan_id": plan["plan_id"], "expected": plan_hash, "actual": plan["plan_hash"]},
            )
        exp = tx.one("SELECT * FROM experiments WHERE experiment_id=?", (plan["experiment_id"],))
        if exp is None:
            raise NotFound(f"experiment {plan['experiment_id']} not found")
        planned = _plan_cells(tx, plan)
        runnable = [c for c in planned if c.skip_note is None]
        if not runnable:
            raise PreconditionFailed(
                f"plan {plan['plan_id']} has no dispatchable run",
                details={"skipped": [{"run_id": c.run_id, "reason": c.skip_note} for c in planned[:50]]},
            )
        warnings: list[str] = []
        mp = max_parallel or _default_max_parallel(ctx, exp)
        if exp.get("max_parallel") and mp > int(exp["max_parallel"]):
            warnings.append(
                f"max_parallel {mp} exceeds the experiment limit {exp['max_parallel']}; admission enforces the lower one"
            )
        gate_run: str | None = None
        smoke_ids = {c.run_id for c in runnable if c.kind == "smoke"}
        if plan.get("smoke_run_id") and plan["smoke_run_id"] in smoke_ids:
            gate_run = plan["smoke_run_id"]
        elif len(smoke_ids) == 1 and len(runnable) > 1:
            gate_run = next(iter(smoke_ids))
        skipped = [c for c in planned if c.skip_note is not None]
        if skipped:
            warnings.append(f"{len(skipped)} cell(s) are not dispatchable and were recorded as skipped")
        wf_id = new_id("workflow")
        op_id = tx.create_operation(
            OP_DISPATCH,
            target_type="workflow",
            target_id=wf_id,
            state="queued",
            stage="queued",
            poll_after_ms=POLL_AFTER_MS,
        )
        state = "gating" if gate_run else "dispatching"
        tx.insert(
            "workflows",
            {
                "workflow_id": wf_id,
                "experiment_id": plan["experiment_id"],
                "plan_id": plan["plan_id"],
                "plan_hash": plan["plan_hash"],
                "max_parallel": mp,
                "dispatch_open": 1,
                "state": state,
                "gate_run_id": gate_run,
                "gate_state": "pending" if gate_run else "none",
                "cancel_scope": None,
                "cancel_operation_id": None,
                "operation_id": op_id,
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
                "dispatch_state": "open",
                "breaker_json": None,
                "failure_policy_json": jdump(policy.model_dump(mode="json")),
            },
        )
        for i, c in enumerate(planned):
            tx.insert(
                "workflow_cells",
                {
                    "workflow_id": wf_id,
                    "run_id": c.run_id,
                    "ordinal": i,
                    "cell_state": "skipped" if c.skip_note else "planned",
                    "attempt_id": None,
                    "dispatch_key": dispatch_key(wf_id, c.run_id),
                    "note": c.skip_note,
                    "updated_at": tx.now,
                },
            )
        tx.event(
            "workflow",
            wf_id,
            "workflow.created",
            {
                "plan_id": plan["plan_id"],
                "plan_kind": plan["kind"],
                "plan_hash": plan["plan_hash"],
                "max_parallel": mp,
                "gate_run_id": gate_run,
                "cells": [[c.run_id, c.exec_plan_hash] for c in planned],
                "skipped": len(skipped),
                "failure_policy": policy.model_dump(mode="json"),
            },
            entity_version=1,
        )
        tx.enqueue(
            DISPATCH_KIND, target_id=wf_id, payload={"workflow_id": wf_id}, dedupe_key=_dispatch_dedupe(wf_id)
        )
        data = _submit_data(ctx, tx, wf_id, warnings=warnings, resumed=False)
        tx.idem_store(ctx.principal, TOOL_SUBMIT, key, ph, data.model_dump(mode="json"))
    return data


def _gate_info(wf: dict[str, Any], cells: list[dict[str, Any]], reason: str | None = None) -> GateInfo | None:
    if not wf.get("gate_run_id"):
        return None
    cell = next((c for c in cells if c["run_id"] == wf["gate_run_id"]), None)
    return GateInfo(
        run_id=wf["gate_run_id"],
        state=wf.get("gate_state") or "pending",
        attempt_id=cell["attempt_id"] if cell else None,
        reason=reason or (cell.get("note") if cell else None),
    )


def _submit_data(
    ctx: ServiceContext, tx: Tx, wf_id: str, *, warnings: list[str], resumed: bool
) -> WorkflowSubmitData:
    wf = _load_wf(tx, wf_id)
    plan = tx.one("SELECT kind FROM plans WHERE plan_id=?", (wf["plan_id"],))
    cells = _cells(tx, wf_id)
    limit = 20
    views, _ = _views(tx, cells[:limit], now=ctx.clock.now(), stale_after_s=None)
    return WorkflowSubmitData(
        workflow_id=wf_id,
        workflow_ref=workflow_ref(wf_id),
        experiment_id=wf["experiment_id"],
        plan_id=wf["plan_id"],
        plan_kind=plan["kind"] if plan else "unknown",
        plan_hash=wf["plan_hash"],
        operation_id=wf["operation_id"],
        operation_ref=_op_ref(wf["operation_id"]) or "",
        state=wf["state"],
        dispatch_state=wf["dispatch_state"],
        version=wf["version"],
        max_parallel=wf["max_parallel"],
        planned_run_count=sum(1 for c in cells if c["cell_state"] != "skipped"),
        skipped_count=sum(1 for c in cells if c["cell_state"] == "skipped"),
        gate=_gate_info(wf, cells),
        resumed=resumed,
        cells=views,
        next_cursor=encode_cursor({"o": limit}) if len(cells) > limit else None,
        warnings=warnings,
    )


def _failed_attempt_ids(cells: list[dict[str, Any]]) -> set[str]:
    return {c["attempt_id"] for c in cells if c["attempt_id"] and c["cell_state"] == "failed"}


def _resume(
    ctx: ServiceContext, *, workflow_id: str, expected_version: int, key: str, ph: str
) -> WorkflowSubmitData:
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL_SUBMIT, key, ph)
        if stored is not None:
            return WorkflowSubmitData.model_validate(stored)
        wf = _load_wf(tx, workflow_id)
        if wf["version"] != expected_version:
            raise VersionConflict(
                f"workflow {workflow_id} is at version {wf['version']}, expected {expected_version}",
                details={
                    "current_version": wf["version"],
                    "expected_version": expected_version,
                    "state": wf["state"],
                },
            )
        if (
            wf["state"] in ("cancelling", "cancelled")
            or wf["dispatch_state"] == "closed"
            or not wf["dispatch_open"]
        ):
            raise PreconditionFailed(
                f"workflow {workflow_id} was cancelled; cancelled dispatch is never reopened (recover the "
                "missing cells with "
                + call(
                    "run_recover",
                    selector={"workflow_id": workflow_id, "cells": "missing_valid"},
                    idempotency_key=NEW_KEY,
                )
                + " and submit that plan)",
                details={"state": wf["state"], "dispatch_state": wf["dispatch_state"]},
            )
        cells = _cells(tx, workflow_id)
        _refresh(tx, workflow_id, cells)
        values: dict[str, Any] = {}
        events: list[tuple[str, dict[str, Any]]] = []
        warnings: list[str] = []
        if wf["dispatch_state"] == "paused":
            bj = jload(wf.get("breaker_json"), {}) or {}
            ack = set(bj.get("acknowledged_attempt_ids") or []) | _failed_attempt_ids(cells)
            history = list(bj.get("history") or [])
            if bj.get("state") == "tripped":
                history.append({k: bj.get(k) for k in ("tripped_at", "fingerprint", "cells", "threshold")})
            values["dispatch_state"] = "open"
            values["breaker_json"] = jdump(
                {
                    "state": "cleared",
                    "cleared_at": tx.now,
                    "acknowledged_attempt_ids": sorted(ack),
                    "history": history[-MAX_BREAKER_HISTORY:],
                }
            )
            events.append(("workflow.resumed", {"acknowledged_failures": len(ack), "by": ctx.principal}))
        if wf["state"] == "gate_failed":
            gate = next((c for c in cells if c["run_id"] == wf["gate_run_id"]), None)
            if (
                gate is not None
                and gate["latest_attempt_id"]
                and gate["latest_attempt_id"] != gate["attempt_id"]
            ):
                _set_cell(
                    tx,
                    workflow_id,
                    gate,
                    attempt_id=gate["latest_attempt_id"],
                    note="gate re-evaluated on the smoke run's latest attempt",
                )
                cells = _cells(tx, workflow_id)
                _refresh(tx, workflow_id, cells)
                gate = next((c for c in cells if c["run_id"] == wf["gate_run_id"]), None)
            verdict, reason = _gate_verdict(tx, wf, gate)
            if verdict == "failed":
                raise PreconditionFailed(
                    f"the smoke gate still fails ({reason}): recover the smoke run (run_recover), then resume again",
                    details={"gate_run_id": wf["gate_run_id"], "reason": reason},
                )
            values["gate_state"] = verdict
            values["state"] = "dispatching" if verdict == "passed" else "gating"
            events.append(("workflow.gate_reevaluated", {"gate_state": verdict, "reason": reason}))
        if values:
            version = tx.update(
                "workflows", "workflow_id", workflow_id, values, expected_version=expected_version
            )
            for etype, payload in events:
                tx.event("workflow", workflow_id, etype, payload, entity_version=version)
            _set_op(tx, wf["operation_id"], state="running", stage="dispatching", error=None)
        else:
            warnings.append(
                f"dispatch is not paused (state {wf['state']}, dispatch {wf['dispatch_state']}); nothing to resume"
            )
        tx.enqueue(
            DISPATCH_KIND,
            target_id=workflow_id,
            payload={"workflow_id": workflow_id},
            dedupe_key=_dispatch_dedupe(workflow_id),
        )
        data = _submit_data(ctx, tx, workflow_id, warnings=warnings, resumed=bool(values))
        tx.idem_store(ctx.principal, TOOL_SUBMIT, key, ph, data.model_dump(mode="json"))
    return data


# ---------------------------------------------------------------------------
# Gate and breaker
# ---------------------------------------------------------------------------


def _require_outputs(tx: Tx, experiment_id: str) -> list[str]:
    exp = tx.one("SELECT spec_json FROM experiments WHERE experiment_id=?", (experiment_id,))
    spec = jload(exp["spec_json"], {}) if exp else {}
    smoke = spec.get("smoke") if isinstance(spec, dict) else None
    req = smoke.get("require_outputs") if isinstance(smoke, dict) else None
    return [str(x) for x in req] if isinstance(req, list) else []


def _gate_verdict(tx: Tx, wf: dict[str, Any], gate: dict[str, Any] | None) -> tuple[str, str]:
    """(pending | passed | failed, reason). Passed only on COMPLETED + valid results + the
    smoke's required outputs finalized; unvalidated outputs keep it pending (never a pass)."""
    if gate is None:
        return "failed", "the smoke run is not a cell of this workflow"
    if gate["attempt_id"] is None:
        return "pending", "smoke not dispatched yet"
    phase, sched, rs = gate["phase"], gate["scheduler_state"], gate["result_state"]
    if phase in _SLOT_PHASES:
        return "pending", f"smoke attempt is {phase}"
    if phase != AttemptPhase.TERMINAL.value:
        return "failed", f"smoke attempt ended {phase}"
    if sched != SchedulerState.COMPLETED.value:
        return "failed", f"smoke attempt ended {sched}"
    if rs in (None, "unchecked", "pending", "unknown"):
        return "pending", f"waiting for the smoke outputs to be validated (result_state={rs})"
    if rs != "valid":
        return "failed", f"smoke outputs are {rs}"
    required = _require_outputs(tx, wf["experiment_id"])
    if required:
        marks = ",".join("?" for _ in required)
        rows = tx.all(
            f"SELECT name, integrity FROM artifacts WHERE attempt_id=? AND name IN ({marks})",
            (gate["attempt_id"], *required),
        )
        final = {r["name"] for r in rows if r["integrity"] == "finalized"}
        missing = [n for n in required if n not in final]
        if missing:
            return "failed", f"smoke did not produce finalized required outputs: {', '.join(missing)}"
    return "passed", "smoke completed with valid outputs"


def _rejection_info(tx: Tx, cell: dict[str, Any]) -> tuple[str | None, str, bool]:
    """(reason_code, message, retry_later) of a refused submission."""
    code: str | None = None
    message = str(cell.get("attempt_reason") or "")
    retry_later = False
    if cell.get("attempt_operation_id"):
        op = tx.one("SELECT error_json FROM operations WHERE operation_id=?", (cell["attempt_operation_id"],))
        err = jload(op["error_json"], None) if op else None
        if isinstance(err, dict):
            raw = err.get("details")
            details: dict[str, Any] = raw if isinstance(raw, dict) else {}
            code = details.get("reason_code") or None
            retry_later = bool(details.get("retry_later"))
            message = message or str(err.get("message") or "")
    if code in _RETRY_LATER_REJECTIONS:
        retry_later = True
    return code, message, retry_later


def _fingerprint(tx: Tx, cell: dict[str, Any], policy: CircuitBreakerPolicy) -> tuple[str, str, str] | None:
    """(category, code, normalized error line) of a failed cell, or None when it must not count."""
    phase = cell["phase"]
    if phase == AttemptPhase.SUBMISSION_REJECTED.value:
        if not policy.include_rejections:
            return None
        code, message, retry_later = _rejection_info(tx, cell)
        if retry_later:
            return None
        return ("submission", f"submission_rejected:{code or '?'}", normalize_error_line(message))
    if phase != AttemptPhase.TERMINAL.value or cell["cell_state"] != "failed":
        return None
    diag = jload(cell.get("diagnosis_json"), None)
    diag = diag if isinstance(diag, dict) else {}
    code = diag.get("primary_code")
    line = str(diag.get("last_error_line") or "")
    guard = ""
    items = diag.get("items") if isinstance(diag.get("items"), list) else []
    if items and isinstance(items[0], dict):
        guard = str((items[0].get("resource_hint") or {}).get("guard_check") or "")
    if not code and cell.get("app_state") == "guard_failed":
        status = jload(cell.get("status_json"), None)
        status = status if isinstance(status, dict) else {}
        code = "env_guard_failed"
        guard = str(status.get("guard_check") or status.get("failed_check") or "")
        line = line or str(status.get("error") or "")
    if not code:
        return None
    category = CODE_CATEGORY.get(str(code), "unknown")
    if category not in policy.categories:
        return None
    return (category, f"{code}:{guard}" if guard else str(code), normalize_error_line(line))


def _breaker_trip(
    tx: Tx, wf: dict[str, Any], cells: list[dict[str, Any]], policy: CircuitBreakerPolicy
) -> dict[str, Any] | None:
    bj = jload(wf.get("breaker_json"), {}) or {}
    ack = set(bj.get("acknowledged_attempt_ids") or [])
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for c in cells:
        if not c["attempt_id"] or c["attempt_id"] in ack:
            continue
        fp = _fingerprint(tx, c, policy)
        if fp is not None:
            groups.setdefault(fp, []).append(c)
    ranked = sorted(groups.items(), key=lambda kv: (-len({m["run_id"] for m in kv[1]}), kv[0]))
    for fp, members in ranked:
        runs = {m["run_id"] for m in members}
        if len(runs) < policy.threshold:
            continue
        return {
            "state": "tripped",
            "tripped_at": tx.now,
            "reason": "same_failure_fingerprint",
            "fingerprint": {"category": fp[0], "code": fp[1], "error_line": fp[2]},
            "threshold": policy.threshold,
            "cells": [
                {"run_id": m["run_id"], "label": m["label"], "attempt_id": m["attempt_id"]}
                for m in members[:20]
            ],
            "acknowledged_attempt_ids": sorted(ack),
            "history": list(bj.get("history") or [])[-MAX_BREAKER_HISTORY:],
        }
    return None


# ---------------------------------------------------------------------------
# Dispatch round
# ---------------------------------------------------------------------------


@dataclass
class DispatchRound:
    """What one dispatch round did (for the handler log and tests)."""

    workflow_id: str
    state: str
    dispatched: int = 0
    gate_state: str | None = None
    dispatch_state: str | None = None
    blocked: str | None = None
    tripped: bool = False


@dataclass
class _Block:
    code: str
    message: str
    details: dict[str, Any]


def dispatch_workflow(ctx: ServiceContext, *, workflow_id: str) -> DispatchRound:
    """One dispatch round in ONE transaction: refresh cells, evaluate gate and breaker, admit
    planned cells into free slots, derive the workflow state. Safe to run concurrently and
    repeatedly (rounds serialize on the database write lock)."""
    with ctx.db.tx(source="worker") as tx:
        return _round(tx, ctx, workflow_id)


def _round(tx: Tx, ctx: ServiceContext, workflow_id: str) -> DispatchRound:
    wf = tx.one("SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,))
    if wf is None:
        return DispatchRound(workflow_id, state="missing")
    cells = _cells(tx, workflow_id)
    _refresh(tx, workflow_id, cells)
    state, gate_state, dispatch_state = wf["state"], wf["gate_state"], wf["dispatch_state"]
    values: dict[str, Any] = {}
    events: list[tuple[str, dict[str, Any]]] = []
    rnd = DispatchRound(workflow_id, state=state, gate_state=gate_state, dispatch_state=dispatch_state)
    block: _Block | None = None

    if state in ("cancelling", "cancelled"):
        if state == "cancelling" and _inflight(cells) == 0:
            values["state"] = "cancelled"
            events.append(("workflow.cancelled", {"counts": _counts(cells)}))
            _set_op(
                tx,
                wf["cancel_operation_id"],
                state="succeeded",
                stage="done",
                result={"counts": _counts(cells)},
            )
    else:
        policy = _policy(wf).circuit_breaker
        gate_reason: str | None = None
        if gate_state == "pending":
            gate = next((c for c in cells if c["run_id"] == wf["gate_run_id"]), None)
            verdict, gate_reason = _gate_verdict(tx, wf, gate)
            if verdict != "pending":
                gate_state = verdict
                values["gate_state"] = verdict
                events.append(
                    (f"workflow.gate_{verdict}", {"run_id": wf["gate_run_id"], "reason": gate_reason})
                )
                if verdict == "failed":
                    _emit(
                        tx,
                        event_type="workflow.gate_failed",
                        entity_id=workflow_id,
                        title="Smoke gate failed",
                        body=f"workflow {workflow_id}: smoke run failed ({gate_reason}); main cells were not dispatched",
                        urgency="immediate",
                    )
                    for c in cells:
                        if c["cell_state"] == "planned" and c["run_id"] != wf["gate_run_id"]:
                            _set_cell(tx, workflow_id, c, note="blocked: smoke gate failed")
        has_planned = any(c["cell_state"] == "planned" for c in cells)
        if dispatch_state == "open" and policy.enabled and gate_state != "failed" and has_planned:
            trip = _breaker_trip(tx, wf, cells, policy)
            if trip is not None:
                dispatch_state = "paused"
                values["dispatch_state"] = "paused"
                values["breaker_json"] = jdump(trip)
                rnd.tripped = True
                events.append(
                    ("workflow.breaker_tripped", {"fingerprint": trip["fingerprint"], "cells": trip["cells"]})
                )
                fp = trip["fingerprint"]
                _emit(
                    tx,
                    event_type="workflow.breaker_tripped",
                    entity_id=workflow_id,
                    title="Workflow paused by the circuit breaker",
                    body=f"workflow {workflow_id}: {len(trip['cells'])} cells failed with {fp['code']} ({fp['category']}); "
                    "dispatch paused, running jobs untouched",
                    urgency="immediate",
                )
        if dispatch_state == "open" and wf["dispatch_open"] and gate_state != "failed":
            rnd.dispatched, block = _dispatch_cells(tx, ctx, wf, cells, gate_state)
            if rnd.dispatched:
                cells = _cells(tx, workflow_id)
        counts = _counts(cells)
        inflight = _inflight(cells)
        if gate_state == "failed":
            new_state = "gate_failed"
        elif gate_state == "pending":
            new_state = "gating"
        elif counts.get("planned", 0) > 0:
            new_state = "dispatching"
        elif inflight > 0:
            new_state = "draining"
        else:
            new_state = "completed"
        if new_state != state:
            values["state"] = new_state
            # the version bump of a pure state change must be auditable too (spec §4.9)
            events.append(("workflow.state_changed", {"from": state, "to": new_state}))
            if new_state == "completed":
                events.append(("workflow.completed", {"counts": counts}))
                _emit(
                    tx,
                    event_type="workflow.finished",
                    entity_id=workflow_id,
                    title="Workflow finished",
                    body=f"workflow {workflow_id}: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                )
        state = new_state
        _update_dispatch_op(
            tx,
            wf,
            state=state,
            dispatch_state=dispatch_state,
            counts=counts,
            inflight=inflight,
            block=block,
            gate_reason=gate_reason,
            breaker=values.get("breaker_json") or wf.get("breaker_json"),
            version=int(wf["version"]) + (1 if values else 0),  # after this round's update below
        )
    if values:
        version = tx.update("workflows", "workflow_id", workflow_id, values)
        for etype, payload in events:
            tx.event("workflow", workflow_id, etype, payload, entity_version=version)
    rnd.state = str(values.get("state", wf["state"]))
    rnd.gate_state = gate_state
    rnd.dispatch_state = dispatch_state
    rnd.blocked = block.message if block else None
    return rnd


def _update_dispatch_op(
    tx: Tx,
    wf: dict[str, Any],
    *,
    state: str,
    dispatch_state: str,
    counts: dict[str, int],
    inflight: int,
    block: _Block | None,
    gate_reason: str | None,
    breaker: str | None,
    version: int,
) -> None:
    total = sum(v for k, v in counts.items() if k != "skipped")
    progress = {"dispatched": total - counts.get("planned", 0), "total": total}
    op = wf["operation_id"]
    resume = _resume_call(wf["workflow_id"], version)
    if state == "completed":
        _set_op(tx, op, state="succeeded", stage="completed", progress=progress, result={"counts": counts})
    elif state == "gate_failed":
        # later rounds do not know the reason any more: keep the stored, specific error then
        _set_op(
            tx,
            op,
            state="blocked",
            stage="gate_failed",
            progress=progress,
            error=(
                {
                    "code": "precondition_failed",
                    "message": f"smoke gate failed: {gate_reason}",
                    "retryable": False,
                    "details": {
                        "gate_run_id": wf["gate_run_id"],
                        "remediation": f"recover the smoke run, then {resume}",
                    },
                }
                if gate_reason
                else None
            ),
        )
    elif dispatch_state == "paused":
        bj = jload(breaker, {}) or {}
        _set_op(
            tx,
            op,
            state="blocked",
            stage="paused_by_breaker",
            progress=progress,
            error={
                "code": "precondition_failed",
                "message": (
                    f"circuit breaker tripped: cells fail with the same "
                    f"{(bj.get('fingerprint') or {}).get('category', 'environment')} error; dispatch paused"
                ),
                "retryable": False,
                "details": {
                    "fingerprint": bj.get("fingerprint"),
                    "remediation": f"fix the cause, then {resume}",
                },
            },
        )
    elif block is not None:
        blocked = inflight == 0
        _set_op(
            tx,
            op,
            state="blocked" if blocked else "running",
            stage="budget_exhausted" if block.code == ErrorCode.BUDGET_EXCEEDED.value else "admission_denied",
            progress=progress,
            error={
                "code": block.code,
                "message": block.message,
                # slot limits clear by themselves; an exhausted budget with nothing in flight does not
                "retryable": block.code != ErrorCode.BUDGET_EXCEEDED.value or not blocked,
                "details": block.details,
            },
        )
    else:
        _set_op(tx, op, state="running", stage=state, progress=progress, error=None)


def _resume_call(workflow_id: str, version: int) -> str:
    """The exact call that reopens a paused / gate-failed workflow at ``version``."""
    return call(
        "workflow_submit",
        workflow_id=workflow_id,
        resume_dispatch=True,
        expected_version=version,
        idempotency_key=NEW_KEY,
    )


def _is_claim_denial(exc: HpcLabError) -> bool:
    text = f"{exc.message} {exc.details.get('reason', '')} {exc.details.get('check', '')}".lower()
    return "claim" in text or "consumed" in text


def _dispatch_cells(
    tx: Tx, ctx: ServiceContext, wf: dict[str, Any], cells: list[dict[str, Any]], gate_state: str
) -> tuple[int, _Block | None]:
    workflow_id = wf["workflow_id"]
    breaker_on = _policy(wf).circuit_breaker.enabled
    free = int(wf["max_parallel"]) - _slots_used(tx, workflow_id, now=tx.now if breaker_on else None)
    candidates = [c for c in cells if c["attempt_id"] is None and c["cell_state"] == "planned"]
    if gate_state == "pending":
        candidates = [c for c in candidates if c["run_id"] == wf["gate_run_id"]]
    if free <= 0 or not candidates:
        return 0, None
    plan = tx.one("SELECT * FROM plans WHERE plan_id=?", (wf["plan_id"],))
    if plan is None:
        return 0, _Block(ErrorCode.NOT_FOUND.value, f"plan {wf['plan_id']} disappeared", {})
    recovery_map = {c["run_id"]: c for c in recovery_plan_cells(plan)} if plan["kind"] == "recovery" else None
    sub: Any = None
    dispatched = 0
    for c in candidates:
        if free <= 0:
            break
        h = _cell_exec_plan(tx, plan, c, recovery_map)
        if h is None:
            _set_cell(tx, workflow_id, c, state="skipped", note="no ready execution plan")
            continue
        prev = tx.one(
            "SELECT response_json FROM idempotency WHERE principal=? AND tool=? AND idem_key=?",
            (ctx.principal, TOOL_DISPATCH_CELL, c["dispatch_key"]),
        )
        if prev is not None:  # dispatched before (repair a cell row that lost its attempt link)
            resp = jload(prev["response_json"], {}) or {}
            if resp.get("attempt_id"):
                _set_cell(
                    tx, workflow_id, c, state="dispatched", attempt_id=str(resp["attempt_id"]), note=None
                )
            continue
        ep = tx.one("SELECT status, consumed_by_attempt_id FROM execution_plans WHERE exec_plan_hash=?", (h,))
        if ep is None:
            _set_cell(tx, workflow_id, c, state="skipped", note=f"execution plan {h} is missing")
            continue
        if ep["consumed_by_attempt_id"]:
            other = tx.one(
                "SELECT attempt_id, run_id, phase, scheduler_state, result_state FROM attempts WHERE attempt_id=?",
                (ep["consumed_by_attempt_id"],),
            )
            if other is not None and other["run_id"] == c["run_id"]:
                st = cell_state_for(other["phase"], other["scheduler_state"], other["result_state"])
                _set_cell(
                    tx,
                    workflow_id,
                    c,
                    state=st,
                    attempt_id=other["attempt_id"],
                    note="adopted: this plan was submitted outside the workflow",
                )
                if other["phase"] in _SLOT_PHASES:
                    free -= 1
            else:
                _set_cell(tx, workflow_id, c, state="skipped", note="execution plan already consumed")
            continue
        if ep["status"] != "ready":
            _set_cell(tx, workflow_id, c, note=f"execution plan is {ep['status']}")
            continue
        if c["claim_attempt_id"]:
            holder = tx.one("SELECT phase FROM attempts WHERE attempt_id=?", (c["claim_attempt_id"],))
            if holder is not None and holder["phase"] in _SLOT_PHASES:
                _set_cell(
                    tx, workflow_id, c, note=f"waiting: run has an active attempt {c['claim_attempt_id']}"
                )
                continue
            # a claim left behind by an attempt that already ended: admission re-checks and heals it
        if sub is None:
            sub = _submission()
        try:
            with _savepoint(tx, "hl_admit"):
                res = sub.admit(tx, ctx, run_id=c["run_id"], exec_plan_hash=h, workflow_id=workflow_id)
        except BudgetExceeded as exc:
            _set_cell(tx, workflow_id, c, note=f"waiting for budget: {exc.message}")
            return dispatched, _Block(
                ErrorCode.BUDGET_EXCEEDED.value, f"budget exhausted: {exc.message}", dict(exc.details)
            )
        except AdmissionDenied as exc:
            if _is_claim_denial(exc):
                _set_cell(tx, workflow_id, c, note=f"admission denied: {exc.message}")
                continue
            return dispatched, _Block(ErrorCode.ADMISSION_DENIED.value, exc.message, dict(exc.details))
        except (NotReady, PreconditionFailed, NotFound, InvalidArgument) as exc:
            _set_cell(tx, workflow_id, c, note=f"{exc.code}: {exc.message}")
            continue
        attempt_id = str(res.attempt_id)
        _set_cell(tx, workflow_id, c, state="dispatched", attempt_id=attempt_id, note=None)
        tx.idem_store(
            ctx.principal,
            TOOL_DISPATCH_CELL,
            c["dispatch_key"],
            digest({"run_id": c["run_id"], "exec_plan_hash": h}),
            {
                "attempt_id": attempt_id,
                "operation_id": getattr(res, "operation_id", None),
                "exec_plan_hash": h,
            },
        )
        tx.event(
            "workflow",
            workflow_id,
            "workflow.cell_dispatched",
            {
                "run_id": c["run_id"],
                "attempt_id": attempt_id,
                "exec_plan_hash": h,
                "dispatch_key": c["dispatch_key"],
            },
        )
        dispatched += 1
        free -= 1
    return dispatched, None


# ---------------------------------------------------------------------------
# workflow_cancel
# ---------------------------------------------------------------------------


def workflow_cancel(
    ctx: ServiceContext,
    *,
    workflow_id: str,
    expected_version: int,
    scope: CancelScope = "queued_only",
    idempotency_key: str,
) -> WorkflowCancelData:
    """Stop a workflow: dispatch is closed in the same transaction that bumps the version, and
    never-dispatched cells become ``cancelled``. ``all_managed`` also requests a verified
    cancellation of every active attempt; in-flight and uncertain submissions are reconciled
    first and keep their budget until terminal evidence - nothing is marked cancelled early."""
    ph = digest({"workflow_id": workflow_id, "expected_version": expected_version, "scope": scope})
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL_CANCEL, idempotency_key, ph)
        if stored is not None:
            return WorkflowCancelData.model_validate(stored)
        wf = _load_wf(tx, workflow_id)
        if wf["version"] != expected_version:
            raise VersionConflict(
                f"workflow {workflow_id} is at version {wf['version']}, expected {expected_version}",
                details={
                    "current_version": wf["version"],
                    "expected_version": expected_version,
                    "state": wf["state"],
                },
            )
        cells = _cells(tx, workflow_id)
        _refresh(tx, workflow_id, cells)
        notes: list[str] = []
        already = wf["state"] in ("cancelling", "cancelled")
        # escalating queued_only -> all_managed only makes sense while something is still in
        # flight; a finished cancellation is never re-opened
        escalate = (
            wf["state"] == "cancelling" and scope == "all_managed" and wf["cancel_scope"] != "all_managed"
        )
        if wf["state"] == "completed" or (already and not escalate):
            notes.append(
                f"workflow is already {wf['state']}"
                + (f" (scope {wf['cancel_scope']})" if already else "")
                + "; existing terminal states are kept"
            )
            data = _cancel_data(tx, wf, cells, scope=scope, notes=notes)
            tx.idem_store(ctx.principal, TOOL_CANCEL, idempotency_key, ph, data.model_dump(mode="json"))
            return data
        cancel_op = tx.create_operation(
            OP_CANCEL,
            target_type="workflow",
            target_id=workflow_id,
            state="running",
            stage="closing_dispatch",
        )
        version = tx.update(
            "workflows",
            "workflow_id",
            workflow_id,
            {
                "dispatch_open": 0,
                "dispatch_state": "closed",
                "state": "cancelling",
                "cancel_scope": scope,
                "cancel_operation_id": cancel_op,
            },
            expected_version=expected_version,
        )
        cancelled_ids: list[str] = []
        for c in cells:
            if c["attempt_id"] is None and c["cell_state"] == "planned":
                tx.execute(
                    "UPDATE workflow_cells SET cell_state='cancelled', note=?, updated_at=? WHERE workflow_id=? AND run_id=?",
                    ("cancelled before dispatch", tx.now, workflow_id, c["run_id"]),
                )
                c["cell_state"], c["note"] = "cancelled", "cancelled before dispatch"
                cancelled_ids.append(c["run_id"])
        requested_now = 0
        if scope == "all_managed":
            # The cancel intent goes on each attempt in THIS transaction: the submit worker re-checks
            # the attempt's intent right before sbatch, so a cell staged before the cancel is never
            # submitted afterwards (tools.md §15). Staging attempts end at once; in-flight / uncertain
            # ones wait for reconcile; identified ones get a verified cancel step (no I/O here).
            requested_now = _request_cancels_in_tx(tx, ctx, workflow_id, cells)
            if requested_now:
                cells = _cells(tx, workflow_id)
                _refresh(tx, workflow_id, cells)
        inflight = [c for c in cells if c["cell_state"] in _INFLIGHT_CELLS]
        tx.event(
            "workflow",
            workflow_id,
            "workflow.cancel_requested",
            {
                "scope": scope,
                "operation_id": cancel_op,
                "undispatched_cancelled": len(cancelled_ids),
                "cancelled_run_ids": cancelled_ids[:512],
                "attempt_cancels_requested": requested_now,
                "inflight": len(inflight),
                "by": ctx.principal,
            },
            entity_version=version,
        )
        _set_op(
            tx, wf["operation_id"], state="succeeded", stage="cancelled", result={"cancelled_by": cancel_op}
        )
        counts = _counts(cells)
        if not inflight:
            version = tx.update("workflows", "workflow_id", workflow_id, {"state": "cancelled"})
            tx.event(
                "workflow", workflow_id, "workflow.cancelled", {"counts": counts}, entity_version=version
            )
            _set_op(
                tx,
                cancel_op,
                state="succeeded",
                stage="done",
                progress={"requested": 0, "total": 0},
                result={"counts": counts},
            )
        elif scope == "all_managed":
            progress = _cancel_progress(inflight)
            _set_op(
                tx,
                cancel_op,
                state="running",
                stage="waiting_for_terminal"
                if progress["requested"] == progress["total"]
                else "requesting_cancels",
                progress=progress,
            )
            # the handler requests what could not be requested here and finalizes the workflow
            tx.enqueue(
                CANCEL_KIND,
                operation_id=cancel_op,
                target_id=workflow_id,
                payload={"workflow_id": workflow_id},
                dedupe_key=_cancel_dedupe(workflow_id),
            )
            notes.append(
                f"{len(inflight)} active cell(s): cancellation requested; identified jobs are cancelled only "
                "after a live identity check, nothing is marked cancelled before terminal evidence"
            )
        else:
            _set_op(
                tx,
                cancel_op,
                state="succeeded",
                stage="queue_closed",
                result={"counts": counts, "inflight_untouched": len(inflight)},
            )
            notes.append(
                f"queued_only: {len(inflight)} in-flight/active cell(s) keep running (use all_managed to stop them)"
            )
        uncertain = sum(1 for c in inflight if c["cell_state"] == "uncertain")
        if uncertain:
            notes.append(f"{uncertain} uncertain submission(s) are reconciled first; never assumed cancelled")
        wf = _load_wf(tx, workflow_id)
        data = _cancel_data(tx, wf, cells, scope=scope, notes=notes)
        tx.idem_store(ctx.principal, TOOL_CANCEL, idempotency_key, ph, data.model_dump(mode="json"))
    return data


def _needs_cancel_request(c: dict[str, Any]) -> bool:
    """An active attempt of the workflow without any cancel intent yet (a refused or failed
    cancel is never re-requested automatically: its identity did not verify)."""
    return bool(c["attempt_id"]) and c["phase"] in _SLOT_PHASES and (c["cancel_state"] or "none") == "none"


def _cancel_progress(inflight: list[dict[str, Any]]) -> dict[str, int]:
    """Real counts only: in-flight cells that carry a cancel intent, of all in-flight cells."""
    return {
        "requested": sum(1 for c in inflight if (c["cancel_state"] or "none") != "none"),
        "total": len(inflight),
    }


def _request_cancels_in_tx(tx: Tx, ctx: ServiceContext, workflow_id: str, cells: list[dict[str, Any]]) -> int:
    """``submission.request_cancel`` for every active attempt without an intent, inside the
    caller's transaction (one savepoint each). Without that function (or on an error for one
    attempt) the ``workflow.cancel`` handler requests it later via ``run_cancel``."""
    targets = [c for c in cells if _needs_cancel_request(c)]
    if not targets:
        return 0
    try:
        request = getattr(_submission(), "request_cancel", None)
    except HpcLabError:
        request = None
    if request is None:
        return 0
    done = 0
    for c in targets:
        try:
            with _savepoint(tx, "hl_cancel"):
                request(
                    tx,
                    ctx,
                    attempt_id=c["attempt_id"],
                    reason=f"workflow_cancel {workflow_id} (all_managed)",
                )
            done += 1
        except HpcLabError as exc:
            _set_cell(tx, workflow_id, c, note=f"cancel request deferred: {exc.code}: {exc.message}")
    return done


def _kept_budget(reader: Database | Tx, attempt_ids: list[str]) -> float | None:
    if not attempt_ids:
        return 0.0
    marks = ",".join("?" for _ in attempt_ids)
    row = reader.one(
        f"SELECT SUM(reserved) AS r, COUNT(*) AS n FROM budget_reservations WHERE attempt_id IN ({marks}) "
        "AND state IN ('reserved', 'unknown')",
        tuple(attempt_ids),
    )
    if row is None or not row["n"]:
        return None  # no reservation rows: unknown, never "0 kept"
    return round(float(row["r"] or 0.0), 4)


def _cancel_data(
    tx: Tx, wf: dict[str, Any], cells: list[dict[str, Any]], *, scope: str, notes: list[str]
) -> WorkflowCancelData:
    order = {"uncertain": 0, "dispatched": 1, "active": 2}
    ordered = sorted(cells, key=lambda c: (order.get(c["cell_state"], 9), c["ordinal"]))
    active_ids = [c["attempt_id"] for c in cells if c["attempt_id"] and c["cell_state"] in _INFLIGHT_CELLS]
    cov = Coverage()
    cov.add("ledger", "ok")
    if any(c["cell_state"] == "uncertain" for c in cells):
        cov.notes.append("uncertain submissions: their cancellation is pending reconciliation")
    if len(ordered) > MAX_CANCEL_CELLS:
        shown = (
            f"cells lists {MAX_CANCEL_CELLS} of {len(ordered)} cells (in-flight first); "
            "counts cover all of them and workflow_get pages the rest"
        )
        cov.notes.append(shown)
        notes = [*notes, shown]
    return WorkflowCancelData(
        workflow_id=wf["workflow_id"],
        workflow_ref=workflow_ref(wf["workflow_id"]),
        operation_id=wf.get("cancel_operation_id"),
        operation_ref=_op_ref(wf.get("cancel_operation_id")),
        scope=wf.get("cancel_scope") or scope,
        state=wf["state"],
        dispatch_state=wf["dispatch_state"],
        version=wf["version"],
        counts=_counts(cells),
        cells=[
            CancelCellView(
                run_id=c["run_id"],
                label=c["label"],
                cell_state=c["cell_state"],
                attempt_id=c["attempt_id"],
                phase=c["phase"],
                cancel_state=c["cancel_state"],
                note=c["note"],
            )
            for c in ordered[:MAX_CANCEL_CELLS]
        ],
        budget_kept_gpu_hours=_kept_budget(tx, active_ids),
        notes=notes,
        coverage=cov,
    )


def cancel_step(ctx: ServiceContext, *, workflow_id: str) -> HandlerResult:
    """``workflow.cancel`` handler body: request ``run_cancel`` (submission service) for every
    active attempt of an ``all_managed`` cancellation that has no cancel intent yet. Network I/O
    happens inside run_cancel's own background steps, never in our transaction."""
    wf = ctx.db.one("SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,))
    if wf is None or wf["state"] != "cancelling":
        return HandlerResult.done("nothing to cancel")
    cells = _cells(ctx.db, workflow_id)
    retry = False
    notes: dict[str, str] = {}
    if wf["cancel_scope"] == "all_managed":
        targets = [c for c in cells if _needs_cancel_request(c)]
        sub = _submission() if targets else None
        for c in targets:
            assert sub is not None
            try:
                sub.run_cancel(
                    ctx,
                    attempt_id=c["attempt_id"],
                    expected_version=c["attempt_version"],
                    idempotency_key=f"wf-cancel:{workflow_id}:{c['attempt_id']}:v{c['attempt_version']}",
                    reason=f"workflow_cancel {workflow_id} (all_managed)",
                )
            except VersionConflict:
                retry = True  # the attempt moved; re-read and ask again with its new version
            except HpcLabError as exc:
                if exc.code == ErrorCode.AUTH_REQUIRED:
                    raise
                notes[c["run_id"]] = f"cancel request failed: {exc.code}: {exc.message}"
                retry = retry or exc.retryable
    with ctx.db.tx(source="worker") as tx:
        wf = _load_wf(tx, workflow_id)
        cells = _cells(tx, workflow_id)
        _refresh(tx, workflow_id, cells)
        for c in cells:
            if c["run_id"] in notes:
                _set_cell(tx, workflow_id, c, note=notes[c["run_id"]])
        pending = [c for c in cells if c["cell_state"] in _INFLIGHT_CELLS]
        # re-read after the requests: counting ``requested`` again on top would double count
        progress = _cancel_progress(pending)
        if wf["state"] == "cancelling" and not pending:
            version = tx.update("workflows", "workflow_id", workflow_id, {"state": "cancelled"})
            tx.event(
                "workflow",
                workflow_id,
                "workflow.cancelled",
                {"counts": _counts(cells)},
                entity_version=version,
            )
            _set_op(
                tx,
                wf["cancel_operation_id"],
                state="succeeded",
                stage="done",
                result={"counts": _counts(cells)},
            )
        elif wf["cancel_scope"] == "all_managed":
            _set_op(
                tx,
                wf["cancel_operation_id"],
                state="running",
                stage="waiting_for_terminal",
                progress=progress,
            )
    if retry:
        return HandlerResult.retry(5.0, note="some cancel requests must be retried")
    return HandlerResult.done()


# ---------------------------------------------------------------------------
# workflow_get
# ---------------------------------------------------------------------------


def _stale_after(ctx: ServiceContext, experiment_id: str) -> float:
    exp = ctx.db.one("SELECT cluster_id FROM experiments WHERE experiment_id=?", (experiment_id,))
    if exp is None:
        return DEFAULT_STALE_AFTER_S
    try:
        pol = ctx.clusters.get(exp["cluster_id"]).profile.policy
    except HpcLabError:
        return DEFAULT_STALE_AFTER_S
    return max(300.0, 10.0 * float(pol.poll_min_interval_s))


def _breaker_info(wf: dict[str, Any]) -> BreakerInfo | None:
    bj = jload(wf.get("breaker_json"), None)
    if not isinstance(bj, dict) or not bj:
        return None
    return BreakerInfo(
        state=str(bj.get("state") or "tripped"),
        tripped_at=bj.get("tripped_at"),
        cleared_at=bj.get("cleared_at"),
        fingerprint=bj.get("fingerprint"),
        cells=list(bj.get("cells") or []),
        threshold=bj.get("threshold"),
        acknowledged_attempts=len(bj.get("acknowledged_attempt_ids") or []),
        history_count=len(bj.get("history") or []),
        resume_hint=(
            _resume_call(wf["workflow_id"], int(wf["version"]))
            if wf.get("dispatch_state") == "paused"
            else None
        ),
    )


def _budget(reader: Database, attempt_ids: list[str]) -> WorkflowBudget:
    reserved = settled = 0.0
    unknown = released = n = 0
    for i in range(0, len(attempt_ids), 500):
        chunk = attempt_ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        for r in reader.all(
            f"SELECT state, reserved, settled_consumed FROM budget_reservations WHERE attempt_id IN ({marks})",
            tuple(chunk),
        ):
            n += 1
            if r["state"] == "reserved":
                reserved += float(r["reserved"] or 0.0)
            elif r["state"] == "settled":
                settled += float(r["settled_consumed"] or 0.0)
            elif r["state"] == "unknown":
                unknown += 1
            elif r["state"] == "released":
                released += 1
    return WorkflowBudget(
        reserved_gpu_hours=round(reserved, 4),
        settled_gpu_hours=round(settled, 4),
        unknown_reservations=unknown,
        released_reservations=released,
        attempts_with_reservation=n,
    )


def _grid(cells: list[WorkflowCell]) -> list[GridRow]:
    rows: dict[str, GridRow] = {}
    for c in cells:
        treatment = {k: v for k, v in c.cell.items() if k != "seed"}
        key = jdump(treatment)
        row = rows.setdefault(key, GridRow(treatment=treatment, codes={}))
        seed = c.cell.get("seed", c.seed)
        row.codes[str(seed) if seed is not None else c.label] = c.code
    return list(rows.values())


def workflow_get(
    ctx: ServiceContext,
    *,
    workflow_id: str,
    cursor: str | None = None,
    limit: int | None = None,
) -> WorkflowData:
    """Workflow resource: dispatch switch and version, gate, breaker evidence, cancel status,
    budget reservations and a page of cells with their coverage codes (plus a treatment x seed
    grid on the first page). Ledger only - no cluster calls."""
    wf = _load_wf(ctx.db, workflow_id)
    n = clamp_limit(limit)
    raw_offset = decode_cursor(cursor).get("o", 0)
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int) or raw_offset < 0:
        raise InvalidArgument("malformed cursor")
    offset = raw_offset
    cells = _cells(ctx.db, workflow_id)
    for c in cells:  # read-only: derive live states instead of the last dispatch round's snapshot
        if c["attempt_id"] is not None:
            c["cell_state"] = cell_state_for(c["phase"], c["scheduler_state"], c["result_state"])
    now = ctx.clock.now()
    stale = _stale_after(ctx, wf["experiment_id"])
    all_views, statuses = _views(ctx.db, cells, now=now, stale_after_s=stale)
    page = all_views[offset : offset + n]
    plan = ctx.db.one("SELECT kind FROM plans WHERE plan_id=?", (wf["plan_id"],))
    op = (
        ctx.db.one(
            "SELECT state, stage, error_json FROM operations WHERE operation_id=?", (wf["operation_id"],)
        )
        if wf["operation_id"]
        else None
    )
    op_err = jload(op["error_json"], None) if op else None
    blocked = (
        op_err.get("message")
        if isinstance(op_err, dict)
        and op
        and op["state"] in ("blocked", "running")
        and op["stage"] in ("budget_exhausted", "admission_denied", "gate_failed", "paused_by_breaker")
        else None
    )
    cancel = None
    if wf.get("cancel_operation_id") or wf.get("cancel_scope"):
        cop = (
            ctx.db.one("SELECT state FROM operations WHERE operation_id=?", (wf["cancel_operation_id"],))
            if wf.get("cancel_operation_id")
            else None
        )
        cancel = CancelInfo(
            scope=wf.get("cancel_scope"),
            operation_id=wf.get("cancel_operation_id"),
            operation_ref=_op_ref(wf.get("cancel_operation_id")),
            operation_state=cop["state"] if cop else None,
        )
    cov = Coverage()
    cov.add("ledger", "ok")
    stale_cells = [s for s in statuses.values() if s.stale]
    if stale_cells:
        cov.add(
            "observations",
            "stale",
            detail=f"{len(stale_cells)} active cell(s) not observed for more than {int(stale)} s",
        )
    observed = [
        o for o in (parse_iso(c["last_observed_at"]) for c in cells if c.get("last_observed_at")) if o
    ]
    observed_at = iso(min(observed)) if observed else None
    gate_reason = None
    if wf.get("gate_run_id") and wf.get("gate_state") == "pending":
        gate_cell = next((c for c in cells if c["run_id"] == wf["gate_run_id"]), None)
        if gate_cell is not None and gate_cell["attempt_id"] is not None:
            gate_reason = f"smoke attempt {gate_cell['phase']} / result {gate_cell['result_state']}"
    return WorkflowData(
        workflow_id=workflow_id,
        workflow_ref=workflow_ref(workflow_id),
        experiment_id=wf["experiment_id"],
        plan_id=wf["plan_id"],
        plan_kind=plan["kind"] if plan else None,
        plan_hash=wf["plan_hash"],
        state=wf["state"],
        dispatch_state=wf["dispatch_state"],
        dispatch_open=bool(wf["dispatch_open"]),
        version=wf["version"],
        max_parallel=wf["max_parallel"],
        slots_used=_slots_used(ctx.db, workflow_id),
        gate=_gate_info(wf, cells, gate_reason),
        breaker=_breaker_info(wf),
        failure_policy=_policy(wf),
        cancel=cancel,
        operation_id=wf["operation_id"],
        operation_ref=_op_ref(wf["operation_id"]),
        operation_state=op["state"] if op else None,
        operation_stage=op["stage"] if op else None,
        dispatch_blocked_reason=blocked,
        counts=_counts(cells),
        codes=summarize_codes(statuses.values()),
        budget=_budget(ctx.db, [c["attempt_id"] for c in cells if c["attempt_id"]]),
        cells=page,
        grid=_grid(all_views) if offset == 0 else None,
        created_at=wf["created_at"],
        updated_at=wf["updated_at"],
        next_cursor=encode_cursor({"o": offset + n}) if offset + n < len(all_views) else None,
        coverage=cov,
        observed_at=observed_at,
    )


# ---------------------------------------------------------------------------
# Background wiring
# ---------------------------------------------------------------------------


def _handle_dispatch(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    wf_id = item.payload.get("workflow_id") or item.target_id
    if not wf_id:
        return HandlerResult.done("no workflow id")
    rnd = dispatch_workflow(ctx, workflow_id=str(wf_id))
    log.debug("workflow %s round: %s", wf_id, rnd)
    return HandlerResult.done()


def _handle_cancel(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    wf_id = item.payload.get("workflow_id") or item.target_id
    if not wf_id:
        return HandlerResult.done("no workflow id")
    return cancel_step(ctx, workflow_id=str(wf_id))


def sweep(ctx: ServiceContext) -> None:
    """Periodic safety net: every live workflow gets a dispatch round (and all_managed
    cancellations another cancel pass) even if an enqueue from another service was missed."""
    rows = ctx.db.all(
        "SELECT workflow_id, state, cancel_scope FROM workflows WHERE state IN "
        "('gating', 'dispatching', 'draining', 'cancelling')"
    )
    if not rows:
        return
    with ctx.db.tx(source="worker") as tx:
        for r in rows:
            wf_id = r["workflow_id"]
            tx.enqueue(
                DISPATCH_KIND,
                target_id=wf_id,
                payload={"workflow_id": wf_id},
                dedupe_key=_dispatch_dedupe(wf_id),
            )
            if r["state"] == "cancelling" and r["cancel_scope"] == "all_managed":
                tx.enqueue(
                    CANCEL_KIND,
                    target_id=wf_id,
                    payload={"workflow_id": wf_id},
                    dedupe_key=_cancel_dedupe(wf_id),
                )


def outbox_handlers() -> dict[str, Handler]:
    return {DISPATCH_KIND: _handle_dispatch, CANCEL_KIND: _handle_cancel}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    fn: Callable[[ServiceContext], None] = sweep
    return [PeriodicTask(name=SWEEP_TASK, interval_s=SWEEP_INTERVAL_S, fn=fn, jitter_s=5.0)]
