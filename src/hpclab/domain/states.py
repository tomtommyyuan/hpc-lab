"""State axes and allowed transitions.

Design §6.3: ``scheduler_state``, ``application_completion`` and ``result_validation`` are
independent axes. ``COMPLETED`` is not "results are valid"; ``not in squeue`` is not
"finished"; a checkpoint existing is not "resumable".
"""

from __future__ import annotations

from enum import StrEnum


class AttemptPhase(StrEnum):
    """Submission/observation lifecycle of one Attempt (design §6.2)."""

    STAGING = "staging"  # claim taken, snapshot being staged, sbatch not called
    SUBMISSION_IN_FLIGHT = "submission_in_flight"  # dispatch intent persisted; sbatch may have been called
    SUBMISSION_REJECTED = "submission_rejected"  # scheduler explicitly refused; nothing queued
    SUBMISSION_UNCERTAIN = "submission_uncertain"  # reply lost / crash: must reconcile, never blind-resubmit
    DUPLICATE_DETECTED = "duplicate_detected"  # reconcile found >1 matching jobs; human decision needed
    JOB_IDENTIFIED = "job_identified"  # job id bound to this attempt
    OBSERVING = "observing"  # job is queued/running; observations flowing
    TERMINAL = "terminal"  # scheduler reached a terminal state (confirmed)
    CANCELLED_BEFORE_SUBMIT = "cancelled_before_submit"  # cancel intent won before dispatch
    ABANDONED = "abandoned"  # explicit, audited human resolution of an unresolved attempt


ATTEMPT_TRANSITIONS: dict[AttemptPhase, frozenset[AttemptPhase]] = {
    AttemptPhase.STAGING: frozenset(
        {
            AttemptPhase.SUBMISSION_IN_FLIGHT,
            AttemptPhase.CANCELLED_BEFORE_SUBMIT,
            AttemptPhase.SUBMISSION_REJECTED,
        }
    ),
    AttemptPhase.SUBMISSION_IN_FLIGHT: frozenset(
        {
            AttemptPhase.SUBMISSION_REJECTED,
            AttemptPhase.JOB_IDENTIFIED,
            AttemptPhase.SUBMISSION_UNCERTAIN,
        }
    ),
    # REJECTED only on proof: the on-cluster receipt journaled a definite sbatch refusal.
    AttemptPhase.SUBMISSION_UNCERTAIN: frozenset(
        {
            AttemptPhase.JOB_IDENTIFIED,
            AttemptPhase.DUPLICATE_DETECTED,
            AttemptPhase.ABANDONED,
            AttemptPhase.SUBMISSION_REJECTED,
        }
    ),
    AttemptPhase.DUPLICATE_DETECTED: frozenset({AttemptPhase.JOB_IDENTIFIED, AttemptPhase.ABANDONED}),
    AttemptPhase.JOB_IDENTIFIED: frozenset({AttemptPhase.OBSERVING, AttemptPhase.TERMINAL}),
    # OBSERVING -> ABANDONED is a human last resort only (``attempt_resolve(resolution='abandon')``
    # while the job is in an observation gap, e.g. a requeue chain accounting can never verify):
    # see HUMAN_ONLY_TRANSITIONS. The job may still be running; nothing is cancelled.
    AttemptPhase.OBSERVING: frozenset({AttemptPhase.TERMINAL, AttemptPhase.ABANDONED}),
    # Slurm requeue can bring a job back from a terminal-looking state; that is recorded as a
    # new execution segment (restart_count++) rather than silently overwriting history.
    AttemptPhase.TERMINAL: frozenset({AttemptPhase.OBSERVING}),
    AttemptPhase.SUBMISSION_REJECTED: frozenset(),
    AttemptPhase.CANCELLED_BEFORE_SUBMIT: frozenset(),
    AttemptPhase.ABANDONED: frozenset(),
}

# Transitions that only an explicit, audited human decision may take (``Tx.source == 'human'``):
# no automated actor gives up on an identified job whose end it never observed.
HUMAN_ONLY_TRANSITIONS: frozenset[tuple[AttemptPhase, AttemptPhase]] = frozenset(
    {(AttemptPhase.OBSERVING, AttemptPhase.ABANDONED)}
)

# Phases that hold an execution claim on the Run (design: at most one active/unresolved claim).
CLAIM_HOLDING_PHASES: frozenset[AttemptPhase] = frozenset(
    {
        AttemptPhase.STAGING,
        AttemptPhase.SUBMISSION_IN_FLIGHT,
        AttemptPhase.SUBMISSION_UNCERTAIN,
        AttemptPhase.DUPLICATE_DETECTED,
        AttemptPhase.JOB_IDENTIFIED,
        AttemptPhase.OBSERVING,
    }
)

# Phases that occupy a concurrency slot and keep their budget reservation.
SLOT_HOLDING_PHASES = CLAIM_HOLDING_PHASES

FINAL_PHASES: frozenset[AttemptPhase] = frozenset(
    {
        AttemptPhase.SUBMISSION_REJECTED,
        AttemptPhase.CANCELLED_BEFORE_SUBMIT,
        AttemptPhase.ABANDONED,
    }
)


def can_transition(src: AttemptPhase, dst: AttemptPhase) -> bool:
    return src == dst or dst in ATTEMPT_TRANSITIONS.get(src, frozenset())


class SchedulerState(StrEnum):
    """Normalized Slurm job state. ``raw`` strings are preserved alongside."""

    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    CONFIGURING = "CONFIGURING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETING = "COMPLETING"
    REQUEUED = "REQUEUED"
    RESIZING = "RESIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    SPECIAL_EXIT = "SPECIAL_EXIT"


SCHEDULER_TERMINAL: frozenset[SchedulerState] = frozenset(
    {
        SchedulerState.COMPLETED,
        SchedulerState.FAILED,
        SchedulerState.CANCELLED,
        SchedulerState.TIMEOUT,
        SchedulerState.OUT_OF_MEMORY,
        SchedulerState.NODE_FAIL,
        SchedulerState.PREEMPTED,
        SchedulerState.BOOT_FAIL,
        SchedulerState.DEADLINE,
    }
)

SCHEDULER_ACTIVE: frozenset[SchedulerState] = frozenset(
    {
        SchedulerState.PENDING,
        SchedulerState.CONFIGURING,
        SchedulerState.RUNNING,
        SchedulerState.SUSPENDED,
        SchedulerState.COMPLETING,
        SchedulerState.REQUEUED,
        SchedulerState.RESIZING,
        SchedulerState.SPECIAL_EXIT,
    }
)


def normalize_slurm_state(raw: str | None) -> SchedulerState:
    """Map raw Slurm state text (``CANCELLED by 123``, ``PD``, ``OUT_OF_ME+``) to the enum."""
    if not raw:
        return SchedulerState.UNKNOWN
    tok = raw.strip().split()[0].upper().rstrip("+")
    short = {
        "PD": "PENDING",
        "R": "RUNNING",
        "CG": "COMPLETING",
        "CD": "COMPLETED",
        "CF": "CONFIGURING",
        "CA": "CANCELLED",
        "F": "FAILED",
        "TO": "TIMEOUT",
        "OOM": "OUT_OF_MEMORY",
        "NF": "NODE_FAIL",
        "PR": "PREEMPTED",
        "S": "SUSPENDED",
        "BF": "BOOT_FAIL",
        "DL": "DEADLINE",
        "RQ": "REQUEUED",
        "RS": "RESIZING",
        "SE": "SPECIAL_EXIT",
        "RH": "REQUEUED",
        "RF": "REQUEUED",
    }
    tok = short.get(tok, tok)
    if tok.startswith("OUT_OF_ME"):
        tok = "OUT_OF_MEMORY"
    if tok.startswith("REQUEUE"):
        tok = "REQUEUED"
    try:
        return SchedulerState(tok)
    except ValueError:
        return SchedulerState.UNKNOWN


class AppState(StrEnum):
    """What the job wrapper recorded about the user program (status.json on the cluster)."""

    UNKNOWN = "unknown"  # no evidence yet (not observed, or file unreadable)
    NOT_STARTED = "not_started"  # wrapper never wrote a start record
    STARTED = "started"
    EXITED_OK = "exited_ok"  # user command returned 0
    EXITED_ERROR = "exited_error"  # user command returned non-zero
    GUARD_FAILED = "guard_failed"  # wrapper's fail-fast environment guard stopped the job (exit 3)
    KILLED = "killed"  # started, never finished (signal, node loss, timeout)


class ResultState(StrEnum):
    """Validation of declared outputs; independent from scheduler/app state."""

    UNCHECKED = "unchecked"
    PENDING = "pending"  # job still running; outputs mutable
    VALID = "valid"  # all required outputs present, finalized, digests recorded
    INCOMPLETE = "incomplete"  # some required outputs missing/partial
    INVALID = "invalid"  # outputs present but failed validation (bad JSON, digest mismatch)
    UNKNOWN = "unknown"  # could not check (transport unavailable)


class CancelState(StrEnum):
    NONE = "none"
    REQUESTED = "requested"  # intent recorded, not yet sent (e.g. waiting for reconcile)
    SENT = "sent"  # scancel issued after live identity verification
    CONFIRMED = "confirmed"  # a later observation shows a terminal state
    REFUSED = "refused"  # identity mismatch; nothing was cancelled
    FAILED = "failed"


class OperationState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"  # needs reconcile / auth / human decision; see error + details
    SUCCEEDED = "succeeded"
    FAILED = "failed"


OPERATION_FINAL = frozenset({OperationState.SUCCEEDED, OperationState.FAILED})


class RunStatus(StrEnum):
    """Summary status of a Run for listings, derived from its latest Attempt."""

    PREPARED = "prepared"  # no attempt yet
    SUBMITTING = "submitting"  # staging / in flight
    QUEUED = "queued"  # scheduler PENDING
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # scheduler COMPLETED and result VALID
    COMPLETED_UNVERIFIED = "completed_unverified"  # scheduler COMPLETED, results unchecked/incomplete
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"  # submission uncertain / observation gap / duplicate


class Pin(StrEnum):
    """How firmly an input/environment component is fixed (design §6.1)."""

    PINNED = "pinned"  # content digest known and verified
    FINGERPRINTED = "fingerprinted"  # cheap identity (lockfile digest, size+mtime, version label)
    UNPINNED = "unpinned"  # referenced by mutable path only
    UNRESOLVED = "unresolved"  # could not even resolve the reference
