"""GPU-hour budget ledger (design §6.5, build spec invariant 7).

One ``budget_reservations`` row per Attempt, in GPU-hours:

* ``reserve`` at admission: ``gpus x time limit`` (what the scheduler *may* charge, not what
  will be used). Staging, in-flight and uncertain submissions keep it.
* ``update_estimate`` while running (elapsed so far x GPUs; a cached, derived number).
* ``cover_requeue`` when Slurm requeues the job: one time limit per segment is held, and a row
  settled before the requeue was seen becomes active again.
* ``settle`` when accounting is known: ``settled`` with the consumed amount, or ``unknown``
  when accounting cannot tell (no sacct row, GPUs not tracked, abandoned uncertain
  submission). Unknown is *never* counted as 0: it keeps counting at its reservation.
* ``release`` when nothing can have run (rejected by sbatch, cancelled before submission).

``summary`` is the honest view every tool shows, and ``check_available`` is the admission
test: ``available = budget - active reservations - settled consumption - unknown
reservations``. All writers take the caller's ``Tx`` so budget changes commit atomically with
the attempt state change that caused them; each change writes an ``events`` row on the
experiment (``budget.*``) except ``update_estimate``, which is a derived cache like
``attempts.time_left_s`` and is refreshed on every poll.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hpclab.domain.errors import BudgetExceeded, InvalidArgument, NotFound
from hpclab.domain.ids import new_id
from hpclab.storage.db import Tx

log = logging.getLogger("hpclab.budget")

ReservationState = Literal["reserved", "settled", "released", "unknown"]
_EPS = 1e-9


class _Reader(Protocol):
    def one(self, sql: str, params: Any = ...) -> dict[str, Any] | None: ...

    def all(self, sql: str, params: Any = ...) -> list[dict[str, Any]]: ...


class BudgetSummary(BaseModel):
    """GPU-hour position of one experiment. ``None`` means "no hpclab budget configured"
    (the site quota still applies); it never means zero."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    budget: float | None = Field(description="configured GPU-hour budget (None = no hpclab budget)")
    reserved: float = Field(description="GPU-hours held by active attempts (staging..observing)")
    estimated_consumed: float | None = Field(
        description="running estimate over active attempts that have one (see estimate_unknown_count); "
        "None when attempts are active but none has an estimate yet (unknown, not 0)"
    )
    settled_consumed: float = Field(description="GPU-hours confirmed by accounting")
    unknown_count: int = Field(description="attempts whose consumption is unknown (counted at reservation)")
    unknown_reserved: float = Field(description="reservations of unknown attempts (upper bound of their use)")
    available: float | None = Field(
        description="budget - reserved - settled_consumed - unknown_reserved (None without a budget)"
    )
    active_count: int = 0
    settled_count: int = 0
    released_count: int = 0
    estimate_unknown_count: int = 0
    notes: list[str] = Field(default_factory=list)


def _amount(value: float | None, what: str) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise InvalidArgument(f"{what} must be a finite number >= 0, got {value!r}")
    return v


def _row(reader: _Reader, attempt_id: str) -> dict[str, Any] | None:
    return reader.one("SELECT * FROM budget_reservations WHERE attempt_id=?", (attempt_id,))


def _event(tx: Tx, row: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
    tx.event(
        "experiment",
        row["experiment_id"],
        event_type,
        {"attempt_id": row["attempt_id"], "reservation_id": row["reservation_id"], **payload},
    )


def reserve(tx: Tx, *, experiment_id: str, attempt_id: str, gpu_hours: float) -> str:
    """Hold ``gpu_hours`` for ``attempt_id``. Idempotent per attempt (returns the existing id
    when the same amount is already reserved)."""
    amount = _amount(gpu_hours, "gpu_hours")
    assert amount is not None
    existing = _row(tx, attempt_id)
    if existing is not None:
        if abs(float(existing["reserved"]) - amount) > _EPS or existing["experiment_id"] != experiment_id:
            raise InvalidArgument(
                f"attempt {attempt_id} already has a different budget reservation",
                details={"reservation_id": existing["reservation_id"], "reserved": existing["reserved"]},
            )
        return str(existing["reservation_id"])
    rid = new_id("reservation")
    tx.insert(
        "budget_reservations",
        {
            "reservation_id": rid,
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "reserved": amount,
            "estimated_consumed": None,
            "settled_consumed": None,
            "state": "reserved",
            "note": None,
            "created_at": tx.now,
            "updated_at": tx.now,
        },
    )
    tx.event(
        "experiment",
        experiment_id,
        "budget.reserved",
        {"attempt_id": attempt_id, "reservation_id": rid, "gpu_hours": amount},
    )
    return rid


def settle(tx: Tx, *, attempt_id: str, consumed_gpu_hours: float | None, note: str) -> str:
    """Record what an attempt consumed. ``None`` -> state ``unknown`` (kept at its reservation,
    never 0). A known value always wins; a settled row is never downgraded to unknown.

    Returns the new state. Raises ``NotFound`` when the attempt has no reservation (callers
    such as observe.finalize record "no reservation" instead of pretending it was settled).
    """
    consumed = _amount(consumed_gpu_hours, "consumed_gpu_hours")
    row = _row(tx, attempt_id)
    if row is None:
        raise NotFound(f"attempt {attempt_id} has no budget reservation", details={"attempt_id": attempt_id})
    if consumed is None:
        if row["state"] == "settled":
            return "settled"  # accounting already known; an unknown re-settle adds nothing
        if row["state"] == "unknown" and row["note"] == note:
            return "unknown"
        tx.execute(
            "UPDATE budget_reservations SET state='unknown', settled_consumed=NULL, note=?, updated_at=? "
            "WHERE attempt_id=?",
            (note, tx.now, attempt_id),
        )
        _event(tx, row, "budget.unknown", {"previous_state": row["state"], "note": note})
        return "unknown"
    if (
        row["state"] == "settled"
        and row["settled_consumed"] is not None
        and abs(float(row["settled_consumed"]) - consumed) <= _EPS
    ):
        return "settled"
    tx.execute(
        "UPDATE budget_reservations SET state='settled', settled_consumed=?, note=?, updated_at=? "
        "WHERE attempt_id=?",
        (consumed, note, tx.now, attempt_id),
    )
    _event(
        tx,
        row,
        "budget.settled",
        {
            "previous_state": row["state"],
            "consumed_gpu_hours": consumed,
            "reserved": row["reserved"],
            "released_gpu_hours": max(0.0, float(row["reserved"]) - consumed),
            "note": note,
        },
    )
    return "settled"


def release(tx: Tx, *, attempt_id: str, note: str) -> str | None:
    """Free a reservation because nothing ran (sbatch refused, cancelled before submission).

    Only ``reserved``/``unknown`` rows are released; a settled row keeps its consumption.
    Returns the resulting state (None without a reservation).
    """
    row = _row(tx, attempt_id)
    if row is None:
        log.debug("release: attempt %s has no budget reservation", attempt_id)
        return None
    if row["state"] in ("released", "settled"):
        return str(row["state"])
    tx.execute(
        "UPDATE budget_reservations SET state='released', note=?, updated_at=? WHERE attempt_id=?",
        (note, tx.now, attempt_id),
    )
    _event(
        tx,
        row,
        "budget.released",
        {"previous_state": row["state"], "released_gpu_hours": row["reserved"], "note": note},
    )
    return "released"


def cover_requeue(
    tx: Tx, *, attempt_id: str, segments: int, segment_gpu_hours: float | None, note: str
) -> str | None:
    """A requeued attempt runs again, and each requeue segment may be charged up to the full
    time limit: hold ``segments x segment_gpu_hours`` (never less than already held), so an
    unknown settlement stays an upper bound. A row that finalize had already settled or marked
    unknown becomes active again (the live job must hold budget); the next finalize settles the
    total over all segments. Returns the resulting state (None without a reservation)."""
    per = _amount(segment_gpu_hours, "segment_gpu_hours")
    row = _row(tx, attempt_id)
    if row is None:
        return None
    if row["state"] == "released":
        return "released"
    held = float(row["reserved"])
    amount = max(held, per * max(1, int(segments))) if per is not None else held
    if row["state"] == "reserved" and amount - held <= _EPS:
        return "reserved"
    tx.execute(
        "UPDATE budget_reservations SET state='reserved', reserved=?, settled_consumed=NULL, note=?, "
        "updated_at=? WHERE attempt_id=?",
        (amount, note, tx.now, attempt_id),
    )
    _event(
        tx,
        row,
        "budget.requeue_covered",
        {
            "previous_state": row["state"],
            "previous_reserved": held,
            "previous_settled_consumed": row["settled_consumed"],
            "gpu_hours": amount,
            "segments": int(segments),
            "note": note,
        },
    )
    return "reserved"


def update_estimate(tx: Tx, *, attempt_id: str, estimated_gpu_hours: float) -> bool:
    """Refresh the running estimate of an active reservation (no event: derived cache).
    Returns False when there is no active reservation to update."""
    value = _amount(estimated_gpu_hours, "estimated_gpu_hours")
    cur = tx.execute(
        "UPDATE budget_reservations SET estimated_consumed=?, updated_at=? "
        "WHERE attempt_id=? AND state='reserved'",
        (value, tx.now, attempt_id),
    )
    return cur.rowcount == 1


def reservation(reader: _Reader, *, attempt_id: str) -> dict[str, Any] | None:
    """The raw reservation row of an attempt (None if it has none)."""
    return _row(reader, attempt_id)


def summary(reader: _Reader, *, experiment_id: str) -> BudgetSummary:
    exp = reader.one("SELECT budget_gpu_hours FROM experiments WHERE experiment_id=?", (experiment_id,))
    if exp is None:
        raise NotFound(f"experiment {experiment_id} not found", details={"experiment_id": experiment_id})
    budget = float(exp["budget_gpu_hours"]) if exp["budget_gpu_hours"] is not None else None
    rows = reader.all(
        "SELECT state, COUNT(*) AS n, COALESCE(SUM(reserved), 0) AS reserved, "
        "COALESCE(SUM(estimated_consumed), 0) AS estimated, "
        "COALESCE(SUM(settled_consumed), 0) AS settled, "
        "SUM(CASE WHEN estimated_consumed IS NULL THEN 1 ELSE 0 END) AS no_estimate "
        "FROM budget_reservations WHERE experiment_id=? GROUP BY state",
        (experiment_id,),
    )
    by_state = {r["state"]: r for r in rows}

    def val(state: str, col: str) -> float:
        r = by_state.get(state)
        return float(r[col] or 0) if r else 0.0

    def cnt(state: str, col: str = "n") -> int:
        r = by_state.get(state)
        return int(r[col] or 0) if r else 0

    reserved = val("reserved", "reserved")
    settled = val("settled", "settled")
    unknown_reserved = val("unknown", "reserved")
    notes: list[str] = []
    available = None
    if budget is not None:
        available = budget - reserved - settled - unknown_reserved
        if available < 0:
            notes.append("budget overdrawn: consumption/reservations exceed the configured budget")
    if cnt("unknown"):
        notes.append(
            f"{cnt('unknown')} attempt(s) have unknown consumption; counted at their reservation "
            f"({round(unknown_reserved, 6)} GPU-h), never as 0"
        )
    if cnt("reserved", "no_estimate"):
        notes.append(f"{cnt('reserved', 'no_estimate')} active attempt(s) have no usage estimate yet")
    return BudgetSummary(
        experiment_id=experiment_id,
        budget=budget,
        reserved=round(reserved, 6),
        estimated_consumed=(
            None
            if cnt("reserved") and cnt("reserved", "no_estimate") == cnt("reserved")
            else round(val("reserved", "estimated"), 6)
        ),
        settled_consumed=round(settled, 6),
        unknown_count=cnt("unknown"),
        unknown_reserved=round(unknown_reserved, 6),
        available=round(available, 6) if available is not None else None,
        active_count=cnt("reserved"),
        settled_count=cnt("settled"),
        released_count=cnt("released"),
        estimate_unknown_count=cnt("reserved", "no_estimate"),
        notes=notes,
    )


def check_available(reader: _Reader, *, experiment_id: str, gpu_hours: float) -> None:
    """Raise ``BudgetExceeded`` when reserving ``gpu_hours`` would exceed the budget."""
    amount = _amount(gpu_hours, "gpu_hours")
    assert amount is not None
    s = summary(reader, experiment_id=experiment_id)
    if s.budget is None or s.available is None:
        return
    if amount > s.available + _EPS:
        raise BudgetExceeded(
            f"requesting {amount:.3f} GPU-h exceeds the experiment budget "
            f"({max(0.0, s.available):.3f} of {s.budget:.3f} GPU-h available)",
            details={
                "requested_gpu_hours": round(amount, 6),
                "summary": s.model_dump(mode="json"),
                "hint": "wait for running attempts to settle, lower gpus/time, or raise budget.gpu_hours",
            },
        )
