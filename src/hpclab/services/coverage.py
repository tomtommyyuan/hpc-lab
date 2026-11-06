"""Per-run cell status and evaluation selection.

This is the single function that decides "what state is this planned run in, and which
evaluation (if any) counts for it". The experiment grid, ``runs_list`` summaries,
``results_compare`` coverage and the evidence export all call it, so a grid cell and a
paper-table denominator can never disagree.

Cell codes::

    V   valid: a *validated* evaluation exists for an attempt whose job COMPLETED (terminal)
        with finalized outputs
    R   running          Q  queued (PENDING)        S  submitting (staging / in flight)
    P   prepared, never dispatched                  U  submission uncertain / duplicate
    F   failed (TIMEOUT, OOM, NODE_FAIL, FAILED, PREEMPTED, ...; ``reason`` says which)
    E   scheduler COMPLETED, outputs not known to be bad, no valid evaluation *yet*
        (outputs being checked, or evaluation pending / not run) — not a failure
    C!  scheduler COMPLETED but the result is bad: outputs incomplete/invalid, or the
        evaluation of the latest attempt is partial/invalid
    X   cancelled / rejected / abandoned (incl. cells of a cancelled workflow never dispatched)
    ?   last observation older than the freshness bound (state unknown *now*)

Only a job that COMPLETED can produce a counted result. A job killed at its walltime (or
OOM, preempted, ...) can still leave finalized outputs, since the wrapper traps TERM and
writes ``outputs.json``. Its ``result_state`` is then ``valid``, and an explicit
``select='last'`` evaluation of it is valid too. But its value is a mid-training number, so
it never counts and the cell stays ``F``.

Within the counted attempt, the newest valid evaluation counts. Every other valid evaluation
of that attempt (another evaluator, version or config, i.e. a re-evaluation under another
protocol) is disclosed on the cell as ``superseded`` and in its ``reason``. It is never
dropped silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from hpclab.domain.clock import parse_iso
from hpclab.domain.states import AttemptPhase, SchedulerState
from hpclab.storage.db import jload

AttemptSelection = Literal["latest_valid", "first_valid"]

# The attempts (table alias ``a``) whose valid evaluations can count as a run's result: the job
# reached a confirmed terminal state, COMPLETED, and every required output is finalized. Any
# other query that counts "valid results" (e.g. several valid attempts) must use the same rule.
COUNTED_ATTEMPT_WHERE = (
    "a.phase = 'terminal' AND a.scheduler_state = 'COMPLETED' AND a.result_state = 'valid'"
)


class _Reader(Protocol):
    def all(self, sql: str, params: Any = ...) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class SupersededEvaluation:
    """A valid evaluation of the counted attempt that does not count because a newer one does
    (re-evaluated under another evaluator, version, config or input set)."""

    evaluation_id: str
    evaluator_id: str
    evaluator_version: str
    config_hash: str
    created_at: str | None
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class CellStatus:
    run_id: str
    code: str
    latest_attempt_id: str | None
    attempts: int
    phase: str | None
    scheduler_state: str | None
    result_state: str | None
    reason: str | None
    evaluation_id: str | None
    evaluation_attempt_id: str | None
    metrics: dict[str, float | None] = field(default_factory=dict)
    observed_at: str | None = None
    stale: bool = False
    # other valid evaluations of the counted attempt, oldest first (see the module docstring)
    superseded: tuple[SupersededEvaluation, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.code == "V"


_FAILED_STATES = {
    SchedulerState.FAILED,
    SchedulerState.TIMEOUT,
    SchedulerState.OUT_OF_MEMORY,
    SchedulerState.NODE_FAIL,
    SchedulerState.PREEMPTED,
    SchedulerState.BOOT_FAIL,
    SchedulerState.DEADLINE,
}
_ACTIVE_PHASES = {AttemptPhase.JOB_IDENTIFIED, AttemptPhase.OBSERVING}


def _chunks(items: list[str], n: int = 500) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _metric_map(metrics_json: str | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in jload(metrics_json, []) or []:
        name = m.get("name")
        if name is not None and name not in out:
            out[name] = m.get("value")
    return out


def select_evaluations(
    reader: _Reader,
    run_ids: list[str],
    *,
    attempt_selection: AttemptSelection = "latest_valid",
    evaluator_id: str | None = None,
    evaluation_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the evaluation row that counts for each run (runs without one are absent).

    Only ``status='valid'`` evaluations of attempts that ended COMPLETED (terminal) with
    ``result_state='valid'`` qualify (:data:`COUNTED_ATTEMPT_WHERE`). A job that did not
    complete never provides the run's result, even if its outputs were finalized.
    ``latest_valid`` prefers the highest attempt ``seq`` (then newest evaluation);
    ``first_valid`` the lowest — choosing among several valid attempts is a disclosed
    selection rule, never implicit.
    """
    chosen, _ = _select(
        reader,
        run_ids,
        attempt_selection=attempt_selection,
        evaluator_id=evaluator_id,
        evaluation_ids=evaluation_ids,
    )
    return chosen


def _select(
    reader: _Reader,
    run_ids: list[str],
    *,
    attempt_selection: AttemptSelection,
    evaluator_id: str | None,
    evaluation_ids: set[str] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """The counted evaluation per run, and the other qualifying evaluations *of the counted
    attempt* that it supersedes (oldest first)."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for chunk in _chunks(run_ids):
        marks = ",".join("?" for _ in chunk)
        params: list[Any] = list(chunk)
        sql = (
            "SELECT e.*, a.seq AS attempt_seq FROM evaluations e JOIN attempts a ON a.attempt_id = e.attempt_id "
            f"WHERE e.run_id IN ({marks}) AND e.status = 'valid' AND {COUNTED_ATTEMPT_WHERE}"
        )
        if evaluator_id is not None:
            sql += " AND e.evaluator_id = ?"
            params.append(evaluator_id)
        for r in reader.all(sql, params):
            if evaluation_ids is not None and r["evaluation_id"] not in evaluation_ids:
                continue
            candidates.setdefault(r["run_id"], []).append(r)
    chosen: dict[str, dict[str, Any]] = {}
    superseded: dict[str, list[dict[str, Any]]] = {}
    for rid, rows in candidates.items():
        best = rows[0]
        for r in rows[1:]:
            if _rank(r, attempt_selection) > _rank(best, attempt_selection):
                best = r
        chosen[rid] = best
        others = [
            r
            for r in rows
            if r["attempt_id"] == best["attempt_id"] and r["evaluation_id"] != best["evaluation_id"]
        ]
        if others:
            superseded[rid] = sorted(others, key=lambda r: (str(r["created_at"]), str(r["evaluation_id"])))
    return chosen, superseded


def _rank(row: dict[str, Any], attempt_selection: AttemptSelection) -> tuple[int, str]:
    """Attempt order per the selection rule; within one attempt the newest evaluation wins."""
    seq = int(row["attempt_seq"])
    return (seq if attempt_selection == "latest_valid" else -seq, str(row["created_at"]))


def _code_for_attempt(a: dict[str, Any]) -> tuple[str, str | None]:
    try:
        phase = AttemptPhase(a["phase"])
    except ValueError:
        return "?", f"unknown phase {a['phase']}"
    try:
        sched = SchedulerState(a["scheduler_state"] or "UNKNOWN")
    except ValueError:
        sched = SchedulerState.UNKNOWN
    if phase in (AttemptPhase.STAGING, AttemptPhase.SUBMISSION_IN_FLIGHT):
        return "S", phase.value
    if phase in (AttemptPhase.SUBMISSION_UNCERTAIN, AttemptPhase.DUPLICATE_DETECTED):
        return "U", phase.value
    if phase in (
        AttemptPhase.SUBMISSION_REJECTED,
        AttemptPhase.CANCELLED_BEFORE_SUBMIT,
        AttemptPhase.ABANDONED,
    ):
        return "X", phase.value
    if phase in _ACTIVE_PHASES:
        if sched == SchedulerState.PENDING:
            return "Q", a.get("pending_reason") or "PENDING"
        if sched == SchedulerState.UNKNOWN:
            return "Q", "awaiting first observation"
        return "R", sched.value
    # terminal
    if sched == SchedulerState.COMPLETED:
        rs = a.get("result_state")
        if rs in ("incomplete", "invalid"):
            return "C!", f"outputs {rs}"
        if rs == "valid":
            return "E", "completed; evaluation pending (or run evaluation_collect)"
        return "E", f"completed; outputs being checked (result_state={rs})"
    if sched == SchedulerState.CANCELLED:
        return "X", "CANCELLED"
    if sched in _FAILED_STATES:
        diag = jload(a.get("diagnosis_json"), None) or {}
        return "F", diag.get("primary_code") or sched.value
    return "F", sched.value


# What a cell's status reads from an attempt: never the wrapper blobs (env_json, status_json,
# health_json, last_observation_json) - a status grid over many runs must stay cheap. The one
# blob it needs is diagnosis_json (a failed cell's reason).
_ATTEMPT_COLUMNS = ", ".join(
    (
        "attempt_id",
        "run_id",
        "seq",
        "phase",
        "scheduler_state",
        "result_state",
        "pending_reason",
        "diagnosis_json",
        "last_observed_at",
        "identified_at",
        "dispatch_at",
    )
)


def cell_statuses(
    reader: _Reader,
    run_ids: list[str],
    *,
    attempt_selection: AttemptSelection = "latest_valid",
    evaluator_id: str | None = None,
    evaluation_ids: set[str] | None = None,
    stale_after_s: float | None = None,
    now: datetime | None = None,
) -> dict[str, CellStatus]:
    """Status for each run id (every requested id is present in the result)."""
    evals, replaced = _select(
        reader,
        run_ids,
        attempt_selection=attempt_selection,
        evaluator_id=evaluator_id,
        evaluation_ids=evaluation_ids,
    )
    latest: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    bad_eval: dict[str, str] = {}  # latest attempt id -> status of a non-valid evaluation
    cancelled_cells: dict[str, str] = {}  # run id -> workflow cell state for never-dispatched cells
    for chunk in _chunks(run_ids):
        marks = ",".join("?" for _ in chunk)
        # Only evaluations that *would* count if valid can make a result bad: the same
        # evaluator filter and evaluation_refs restriction as select_evaluations.
        sql = (
            f"SELECT evaluation_id, attempt_id, status FROM evaluations WHERE run_id IN ({marks}) "
            "AND status != 'valid'"
        )
        params: list[Any] = list(chunk)
        if evaluator_id is not None:
            sql += " AND evaluator_id = ?"
            params.append(evaluator_id)
        for row in reader.all(sql, params):
            if evaluation_ids is not None and row["evaluation_id"] not in evaluation_ids:
                continue
            bad_eval.setdefault(row["attempt_id"], row["status"])
        for row in reader.all(
            f"SELECT run_id, cell_state FROM workflow_cells WHERE run_id IN ({marks}) "
            "AND cell_state IN ('cancelled', 'skipped') AND attempt_id IS NULL",
            list(chunk),
        ):
            cancelled_cells[row["run_id"]] = row["cell_state"]
    for chunk in _chunks(run_ids):
        marks = ",".join("?" for _ in chunk)
        for row in reader.all(
            f"SELECT {_ATTEMPT_COLUMNS} FROM attempts WHERE run_id IN ({marks}) ORDER BY run_id, seq",
            list(chunk),
        ):
            counts[row["run_id"]] = counts.get(row["run_id"], 0) + 1
            latest[row["run_id"]] = row  # ordered by seq: last one wins
    out: dict[str, CellStatus] = {}
    for rid in run_ids:
        a = latest.get(rid)
        ev = evals.get(rid)
        metrics = _metric_map(ev["metrics_json"]) if ev else {}
        reason: str | None
        if a is None:
            if rid in cancelled_cells:
                why = f"never dispatched: workflow cell {cancelled_cells[rid]}"
                out[rid] = CellStatus(rid, "X", None, 0, None, None, None, why, None, None)
            else:
                out[rid] = CellStatus(rid, "P", None, 0, None, None, None, "not dispatched", None, None)
            continue
        code, reason = _code_for_attempt(a)
        if code == "E" and a["attempt_id"] in bad_eval:
            code, reason = "C!", f"evaluation {bad_eval[a['attempt_id']]}"
        stale = False
        observed_at = a.get("last_observed_at")
        superseded = tuple(_superseded(r) for r in replaced.get(rid, ()))
        if ev is not None:
            notes = []
            if ev["attempt_id"] != a["attempt_id"]:
                notes.append(_latest_attempt_note(a))
            if superseded:
                notes.append(_superseded_note(ev, superseded))
            code, reason = "V", "; ".join(notes) or None
        elif code in ("Q", "R") and stale_after_s is not None and now is not None:
            # A job never observed yet ages from when it was identified/dispatched, so a cell
            # that was just submitted shows Q ("awaiting first observation"), not "?".
            seen = (
                parse_iso(observed_at) or parse_iso(a.get("identified_at")) or parse_iso(a.get("dispatch_at"))
            )
            if seen is None or (now - seen).total_seconds() > stale_after_s:
                stale = True
                reason = f"last observed {observed_at or 'never'}; was {code}"
                code = "?"
        out[rid] = CellStatus(
            run_id=rid,
            code=code,
            latest_attempt_id=a["attempt_id"],
            attempts=counts.get(rid, 0),
            phase=a["phase"],
            scheduler_state=a["scheduler_state"],
            result_state=a["result_state"],
            reason=reason,
            evaluation_id=ev["evaluation_id"] if ev else None,
            evaluation_attempt_id=ev["attempt_id"] if ev else None,
            metrics=metrics,
            observed_at=observed_at,
            stale=stale,
            superseded=superseded,
        )
    return out


_MAX_NOTED = 3  # superseded evaluations named in a cell reason (all of them are in ``superseded``)


def _superseded(row: dict[str, Any]) -> SupersededEvaluation:
    return SupersededEvaluation(
        evaluation_id=str(row["evaluation_id"]),
        evaluator_id=str(row["evaluator_id"]),
        evaluator_version=str(row["evaluator_version"]),
        config_hash=str(row["config_hash"]),
        created_at=row.get("created_at"),
        metrics=_metric_map(row.get("metrics_json")),
    )


def _latest_attempt_note(a: dict[str, Any]) -> str:
    """Why the counted value is not from the run's latest attempt (its phase, and how the job ended)."""
    sched = a.get("scheduler_state")
    if a["phase"] == AttemptPhase.TERMINAL and sched:
        return f"latest attempt {a['attempt_id']} is {a['phase']} ({sched})"
    return f"latest attempt {a['attempt_id']} is {a['phase']}"


def _superseded_note(ev: dict[str, Any], superseded: tuple[SupersededEvaluation, ...]) -> str:
    def protocol(evaluator_id: str, version: str, config_hash: str) -> str:
        return f"{evaluator_id}@{version} config {config_hash[:15]}"

    listed = ", ".join(
        f"{s.evaluation_id} ({protocol(s.evaluator_id, s.evaluator_version, s.config_hash)}, {s.created_at})"
        for s in superseded[:_MAX_NOTED]
    )
    more = f" and {len(superseded) - _MAX_NOTED} more" if len(superseded) > _MAX_NOTED else ""
    counted = protocol(str(ev["evaluator_id"]), str(ev["evaluator_version"]), str(ev["config_hash"]))
    return (
        f"attempt {ev['attempt_id']} was re-evaluated: the newest valid evaluation {ev['evaluation_id']} "
        f"({counted}, {ev['created_at']}) counts and supersedes {listed}{more}"
    )


def summarize_codes(statuses: Iterable[CellStatus]) -> dict[str, int]:
    """Counts per code plus ``planned``/``valid`` totals (for coverage blocks)."""
    counts: dict[str, int] = {}
    total = 0
    for s in statuses:
        total += 1
        counts[s.code] = counts.get(s.code, 0) + 1
    counts["planned"] = total
    counts["valid"] = counts.get("V", 0)
    return counts
