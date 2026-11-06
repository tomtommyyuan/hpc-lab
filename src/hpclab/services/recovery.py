"""Recovery planning: ``run_recover`` turns a stopped Attempt into a new, ready ExecutionPlan of
the SAME Run - or says precisely why it must not (design tools.md §09, system §6.4).

It never submits anything. The output is a plan (``plans.kind='recovery'``) plus one execution
plan per recoverable Run, dispatched later by ``run_submit`` / ``workflow_submit`` through the one
admission gate (``submission.admit``), which still enforces the single execution claim per Run.

Eligibility, in order (the first failing rule wins and is reported in ``blocking``):

1. The attempt must be *stopped for certain*: STAGING / in-flight / identified / observing are
   active ("wait or cancel"); SUBMISSION_UNCERTAIN / DUPLICATE_DETECTED are unresolved
   ("reconcile first" - an uncertain submission never becomes a blind resubmit). A newer
   attempt of the Run, or another attempt holding the Run's claim, also blocks - and so does an
   open execution hold (``submission.execution_holds``, the rule admission applies too): the
   latest job ended PREEMPTED / NODE_FAIL / BOOT_FAIL inside observe's requeue watch (Slurm may
   still requeue it: "wait until <time> or confirm it ended"), or a job finalize found owning an
   attempt dir (duplicate owner) was not confirmed ended (by squeue/sacct, checked by the
   reconcile sweep, or by a human). A probe plan is always ``--no-requeue``.
2. Success is not recoverable: COMPLETED with valid results -> nothing to do; COMPLETED with
   results not yet validated -> wait for artifact indexing.
3. The diagnosis decides which strategies can possibly help (``_allowed_for``): infrastructure
   and cancellation allow retry/resume/adjust; a timeout needs more time or a resume; host OOM
   needs more memory; a CUDA OOM only allows changes that keep the experiment identical (a GPU
   type with more memory, or more GPUs when ``resilience.batch_invariant_to_gpus``); NaN loss,
   missing outputs and code bugs are *semantic* - the answer is a derived Run (``derive_run``
   hint with ready ``experiment_prepare(derive_from=...)`` arguments), never a retry.
4. Strategy requirements: ``resume`` needs ``resilience.resumable`` and a checkpoint of this Run
   whose manifest validates (``adapters.checkpoint``) as an *exact* continuation and a passed
   compute-node load probe (``services.probe``; created on demand and returned for submission);
   ``adjust_resources`` needs a non-empty patch (explicit, or from the diagnosis resource hint)
   that passes the partition limits of the current profile.

Node exclusion is added automatically only for nodes with failure observations from >= 2
distinct attempts within the TTL (``observations`` rows ``node:<name>:<class>:<attempt>``); a
single suspicious node is reported (``suspect_nodes``), never excluded silently. ``*_suspect``
classes (a multi-node failure that cannot name the node) never count towards an exclusion.
A checkpoint load probe vouches only for the directory it hashed (same manifest digest in a
copied directory is not the same bytes).

Batch mode (``selector``) applies the same rules to the cells of a workflow and writes ONE
recovery plan whose ``validation_json.cells`` lists every selected cell with its verdict;
``workflow_submit(plan_id=<recovery plan>)`` then dispatches only the eligible ones.
``missing_valid`` selects the cells without a valid result that are not merely waiting for
their evaluation (code ``E``: completed, evaluation pending - not a failure).

A new ready execution plan becomes the Run's ``current_exec_plan_hash`` in the transaction that
writes it (event ``run.exec_plan_changed``), so "submit this run" means the recovery plan.
"""

from __future__ import annotations

import importlib
import json
import logging
import posixpath
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hpclab.adapters.base import AuthRequiredError, StatInfo, TransportError
from hpclab.adapters.checkpoint import (
    MANIFEST_NAME,
    CheckpointCheck,
    check_manifest_text,
)
from hpclab.api.envelope import Coverage
from hpclab.domain.clock import iso, parse_iso, parse_slurm_duration
from hpclab.domain.errors import (
    ErrorCode,
    HpcLabError,
    IdempotencyConflict,
    InvalidArgument,
    NotFound,
    Stale,
)
from hpclab.domain.hashing import digest
from hpclab.domain.ids import new_id
from hpclab.domain.models import ExecutionPlan, RecoverySource, ResourceSpec, RunSpec
from hpclab.domain.states import (
    CLAIM_HOLDING_PHASES,
    AttemptPhase,
    SchedulerState,
)
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import cell_statuses
from hpclab.services.diagnosis import (
    CODE_CATEGORY,
    AttemptFacts,
    Diagnosis,
    diagnose,
    resource_patch_from_hint,
)
from hpclab.services.probe import (
    ProbeStatus,
    find_probe_runs,
    probe_resources,
    probe_run_spec,
    probe_status,
)
from hpclab.services.runs import NEW_KEY, call
from hpclab.services.submission import ExecutionHold, execution_holds
from hpclab.storage.db import Database, Tx, jdump, jload

log = logging.getLogger("hpclab.recovery")

__all__ = [
    "DiagnosisSelector",
    "RecoverCell",
    "RecoverData",
    "RecoverySelector",
    "ResourcePatch",
    "RunIdsSelector",
    "assess",
    "outbox_handlers",
    "periodic_tasks",
    "recovery_plan_cells",
    "run_recover",
    "validate_resources",
    "workflow_hold_reason",
]

TOOL = "run_recover"
RECOVERY_PLAN_SCHEMA = "hpclab.recovery-plan/v1"
PROBE_PLAN_SCHEMA = "hpclab.probe-plan/v1"
NODE_OBS_TTL_S = 72 * 3600
MAX_AUTO_EXCLUDE = 8
MAX_BATCH_RESUME = (
    8  # each resume cell costs a few bounded remote reads; keep one call well under the deadline
)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CKPT_FILES = 512
MAX_CANDIDATE_DIRS = 8
MAX_LISTING = 5000

RecoverStrategy = Literal["retry", "resume", "adjust_resources"]
CellStrategy = Literal["retry", "resume", "adjust_resources", "initial"]
CheckStatus = Literal["pass", "warn", "fail", "unchecked", "pending"]

_ACTIVE = frozenset(
    {
        AttemptPhase.STAGING.value,
        AttemptPhase.SUBMISSION_IN_FLIGHT.value,
        AttemptPhase.JOB_IDENTIFIED.value,
        AttemptPhase.OBSERVING.value,
    }
)
_UNRESOLVED = frozenset({AttemptPhase.SUBMISSION_UNCERTAIN.value, AttemptPhase.DUPLICATE_DETECTED.value})
_NOTHING_RAN = frozenset({AttemptPhase.SUBMISSION_REJECTED.value, AttemptPhase.CANCELLED_BEFORE_SUBMIT.value})
_CLAIM_PHASES = frozenset(p.value for p in CLAIM_HOLDING_PHASES)
_ALL: frozenset[str] = frozenset({"retry", "resume", "adjust_resources"})
_CODE_BUGS = frozenset({"cuda_device_assert", "state_dict_mismatch", "weights_only_load"})
_WORLD_KEYS = ("gpus", "nodes", "ntasks_per_node")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class ResourcePatch(BaseModel):
    """Resource fields a recovery may change. Everything else about the Run stays frozen."""

    model_config = ConfigDict(extra="forbid")

    time: str | None = None
    mem: str | None = None
    gpus: int | None = Field(default=None, ge=0, le=64)
    gpu_type: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    cpus: int | None = Field(default=None, ge=1, le=512)
    nodes: int | None = Field(default=None, ge=1, le=64)
    ntasks_per_node: int | None = Field(default=None, ge=1, le=64)
    exclude_nodes: list[str] | None = None
    constraint: str | None = None
    extra_sbatch: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DiagnosisSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis_in: list[str] = Field(min_length=1, max_length=64)


class RunIdsSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(min_length=1, max_length=256)


class RecoverySelector(BaseModel):
    """Batch recovery over the cells of one workflow."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    cells: Literal["missing_valid"] | DiagnosisSelector | RunIdsSelector = "missing_valid"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecoverCheck(_Out):
    check: str
    status: CheckStatus
    message: str


class RecoverHint(_Out):
    strategy: str
    message: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class DiagnosisSummary(_Out):
    primary_code: str | None
    category: str | None
    title: str | None
    confidence: str | None
    summary: str
    suggested_strategies: list[str] = Field(default_factory=list)
    resource_hint: dict[str, str] = Field(default_factory=dict)
    last_error_line: str | None = None
    source: Literal["computed", "stored", "scheduler_only", "none"]


class CheckpointSummary(_Out):
    artifact_id: str | None
    remote_dir: str | None
    step: int | None
    status: str
    continuation: str
    manifest_digest: str | None
    needs_load_probe: bool
    total_bytes: int | None = None
    issues: list[str] = Field(default_factory=list)


class ProbeInfo(_Out):
    state: str
    run_id: str | None = None
    plan_id: str | None = None
    plan_ref: str | None = None
    exec_plan_hash: str | None = None
    attempt_id: str | None = None
    detail: str | None = None
    created: bool = False
    load_checked: bool = False


class RecoverCell(_Out):
    run_id: str
    label: str
    kind: str
    attempt_id: str | None
    code: str
    eligible: bool
    strategy: CellStrategy | None
    exec_plan_hash: str | None
    blocking_reason: str | None
    diagnosis_code: str | None
    probe_run_id: str | None = None
    requested_gpu_hours: float | None = None


class RecoverData(_Out):
    mode: Literal["single", "batch"]
    eligible: bool
    strategy: RecoverStrategy | None
    attempt_id: str | None = None
    run_id: str | None = None
    experiment_id: str | None = None
    diagnosis: DiagnosisSummary | None = None
    checks: list[RecoverCheck] = Field(default_factory=list)
    blocking: list[str] = Field(default_factory=list)
    hints: list[RecoverHint] = Field(default_factory=list)
    plan_id: str | None = None
    plan_ref: str | None = None
    exec_plan_hash: str | None = None
    created_reason: str | None = None
    resources: dict[str, Any] | None = None
    resource_patch: dict[str, Any] = Field(default_factory=dict)
    exclude_nodes: list[str] = Field(default_factory=list)
    suspect_nodes: list[str] = Field(default_factory=list)
    requested_gpu_hours: float | None = None
    checkpoint: CheckpointSummary | None = None
    probe: ProbeInfo | None = None
    cells: list[RecoverCell] | None = None
    eligible_count: int | None = None
    total: int | None = None
    probe_plan_id: str | None = None
    probe_plan_ref: str | None = None
    source_workflow_id: str | None = None
    next_step: str | None = None
    coverage: Coverage = Field(default_factory=Coverage)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure eligibility
# ---------------------------------------------------------------------------


@dataclass
class Assessment:
    """Outcome of the pure eligibility rules (before checkpoint / partition checks)."""

    strategy: RecoverStrategy | None
    allowed: frozenset[str]
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[RecoverHint] = field(default_factory=list)
    requirement: str | None = None  # "time" | "mem" | "oom_semantics" for the resource patch

    @property
    def eligible(self) -> bool:
        return not self.blocking and self.strategy is not None


def _derive_hint(run_id: str | None, message: str, *, recapture_code: bool) -> RecoverHint:
    return RecoverHint(
        strategy="derive_run",
        message=message,
        tool="experiment_prepare",
        arguments={"derive_from": {"run_id": run_id, "overrides": {}, "recapture_code": recapture_code}},
    )


def _allowed_for(
    diag: Diagnosis | None, run_id: str | None
) -> tuple[frozenset[str], list[str], list[RecoverHint], str | None, str | None]:
    """(allowed strategies, warnings, hints, patch requirement, reason when nothing is allowed)."""
    item = diag.primary if diag is not None else None
    if item is None:
        return (
            _ALL,
            ["no failure signature was recognised: retry only if the failure looks transient"],
            [],
            None,
            None,
        )
    code = item.code
    cat = CODE_CATEGORY.get(code, "unknown")
    suggested = set(item.suggested_strategies)
    hints: list[RecoverHint] = []
    if "derive_run" in suggested:
        hints.append(
            _derive_hint(
                run_id,
                f"{item.title}: a change of the training configuration is a new derived Run, not a recovery.",
                recapture_code=False,
            )
        )
    if cat == "infrastructure" or code == "cancelled":
        return _ALL, [], hints, None, None
    if code == "timeout":
        return frozenset({"resume", "adjust_resources"}), [], hints, "time", None
    if code == "host_oom":
        return frozenset({"resume", "adjust_resources"}), [], hints, "mem", None
    if code == "cuda_oom":
        if not any(h.strategy == "derive_run" for h in hints):
            hints.append(
                _derive_hint(
                    run_id,
                    "A smaller per-GPU batch or sequence length after a CUDA OOM changes the experiment: derive a Run.",
                    recapture_code=False,
                )
            )
        return frozenset({"resume", "adjust_resources"}), [], hints, "oom_semantics", None
    if cat == "environment":
        hints.append(
            RecoverHint(
                strategy="fix_environment",
                message="The environment (venv, modules, filesystem) lives outside the Run spec: fix it first, "
                "then retry; an unchanged environment fails the same way.",
            )
        )
        return (
            _ALL,
            [f"{code}: retry only after fixing the environment on the cluster"],
            hints,
            None,
            None,
        )
    if code == "checkpoint_corrupt":
        return frozenset({"retry", "resume"}), [], hints, None, None
    if code == "nccl_error":
        return _ALL, ["nccl_error: retry helps only when the failure was transient"], hints, None, None
    if code == "python_exception" and "retry" in suggested:
        return (
            frozenset({"retry", "resume"}),
            ["the exception looks transient; retry once"],
            hints,
            None,
            None,
        )
    if code in ("python_exception", "success_no_outputs") or code in _CODE_BUGS:
        hints.append(
            _derive_hint(
                run_id,
                "Fixing the program changes the code snapshot: prepare a derived Run with recapture_code=true.",
                recapture_code=True,
            )
        )
        return frozenset(), [], hints, None, f"{code}: a retry reruns the same code and fails the same way"
    if cat == "numerics":
        if not any(h.strategy == "derive_run" for h in hints):
            hints.append(
                _derive_hint(run_id, "Changing hyperparameters is a derived Run.", recapture_code=False)
            )
        return frozenset(), [], hints, None, f"{code}: retrying the same configuration diverges again"
    return (
        _ALL,
        [f"{code}: no specific recovery is known; retry only if it looks transient"],
        hints,
        None,
        None,
    )


def assess(
    *,
    phase: str,
    scheduler_state: str | None,
    result_state: str | None,
    diagnosis: Diagnosis | None,
    requested: RecoverStrategy | None,
    resumable: bool,
    has_checkpoint: bool,
    has_hint_patch: bool,
    run_id: str | None = None,
    newer_attempt_id: str | None = None,
    claim_attempt_id: str | None = None,
    holds: list[str] | None = None,
) -> Assessment:
    """The eligibility table (pure; see the module docstring). ``has_checkpoint`` only steers the
    automatic choice of ``resume``; its validity is checked separately. ``holds`` are the Run's
    open execution holds (``submission.execution_holds``: an original job that may be requeued,
    a duplicate-owner job that may still run) - each blocks like a held claim."""
    if phase in _ACTIVE:
        return Assessment(
            None,
            frozenset(),
            [
                f"attempt is still active ({phase}): wait for it to finish or run_cancel it; a parallel branch needs a derived Run"
            ],
        )
    if phase in _UNRESOLVED:
        return Assessment(
            None,
            frozenset(),
            [
                f"submission is unresolved ({phase}): wait for reconciliation or resolve it explicitly; never a blind resubmit"
            ],
        )
    if newer_attempt_id:
        return Assessment(
            None,
            frozenset(),
            [f"a newer attempt {newer_attempt_id} exists: recover from the latest attempt"],
        )
    if claim_attempt_id:
        return Assessment(
            None, frozenset(), [f"the run's execution claim is held by attempt {claim_attempt_id}"]
        )
    if holds:
        return Assessment(None, frozenset(), list(holds))
    warnings: list[str] = []
    hints: list[RecoverHint] = []
    requirement: str | None = None
    if phase in _NOTHING_RAN:
        allowed = frozenset({"retry", "adjust_resources"})
    elif phase == AttemptPhase.ABANDONED.value:
        allowed = frozenset({"retry", "adjust_resources"})
        warnings.append(
            "the attempt was abandoned by a human decision: its job may still exist on the cluster"
        )
    elif phase == AttemptPhase.TERMINAL.value:
        if scheduler_state == SchedulerState.COMPLETED.value and result_state == "valid":
            return Assessment(
                None,
                frozenset(),
                ["the attempt completed with valid results: nothing to recover"],
                hints=[
                    _derive_hint(
                        run_id, "A new experiment from this Run is a derived Run.", recapture_code=False
                    )
                ],
            )
        if scheduler_state == SchedulerState.COMPLETED.value and result_state in (
            None,
            "unchecked",
            "pending",
            "unknown",
        ):
            return Assessment(
                None,
                frozenset(),
                [
                    f"the attempt completed; its outputs are not validated yet (result_state={result_state}): wait for indexing"
                ],
            )
        allowed, warnings, hints, requirement, reason = _allowed_for(diagnosis, run_id)
        if reason is not None:
            return Assessment(None, frozenset(), [reason], warnings, hints)
    else:
        return Assessment(None, frozenset(), [f"unknown attempt phase {phase!r}"])

    blocking: list[str] = []
    strategy: RecoverStrategy | None = None
    if requested is not None:
        if requested not in allowed:
            blocking.append(_not_allowed_reason(requested, diagnosis, phase))
        strategy = requested
    else:
        order: list[str] = []
        if diagnosis is not None and diagnosis.primary is not None:
            order.extend(s for s in diagnosis.primary.suggested_strategies if s in _ALL)
        order.extend(["resume", "adjust_resources", "retry"])
        for s in dict.fromkeys(order):
            if s not in allowed:
                continue
            if s == "resume" and not (has_checkpoint and resumable):
                continue
            if s == "adjust_resources" and not has_hint_patch:
                continue
            strategy = s  # type: ignore[assignment]
            break
        if strategy is None:
            blocking.append(
                "no automatic strategy applies: "
                + (
                    "pass strategy='adjust_resources' with a resource_patch"
                    if "adjust_resources" in allowed
                    else "see hints"
                )
            )
    if strategy == "resume" and not resumable and not blocking:
        blocking.append(
            "the Run does not declare resilience.resumable: the program would ignore HPCLAB_RESUME_FROM "
            "(prepare a derived Run whose program resumes)"
        )
    return Assessment(strategy, allowed, blocking, warnings, hints, requirement)


def _not_allowed_reason(requested: str, diagnosis: Diagnosis | None, phase: str) -> str:
    code = diagnosis.primary_code if diagnosis is not None else None
    if phase in _NOTHING_RAN and requested == "resume":
        return "nothing ran in this attempt, so there is nothing to resume"
    if code == "timeout" and requested == "retry":
        return "timeout: a retry with the same time limit times out again (use adjust_resources with more time, or resume)"
    if code == "host_oom" and requested == "retry":
        return "host_oom: a retry with the same memory runs out again (use adjust_resources with more mem)"
    if code == "cuda_oom" and requested == "retry":
        return "cuda_oom: a retry with the same GPUs runs out of GPU memory again"
    return f"strategy {requested!r} cannot fix {code or phase}"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def _world_size(res: ResourceSpec) -> int:
    return res.nodes * (res.gpus if res.gpus > 0 else res.ntasks_per_node)


def _partition_of(res: ResourceSpec, cluster: ResolvedCluster) -> str | None:
    return res.partition or cluster.overlay.default_partition or cluster.profile.default_partition


def validate_resources(cluster: ResolvedCluster, res: ResourceSpec, *, resumable: bool) -> list[RecoverCheck]:
    """Partition limits of the *current* profile (deterministic, no cluster calls)."""
    checks: list[RecoverCheck] = []
    name = _partition_of(res, cluster)
    rules = cluster.profile.partitions
    if not rules:
        return [
            RecoverCheck(
                check="partition_limits", status="unchecked", message="the profile declares no partitions"
            )
        ]
    if name is None:
        return [
            RecoverCheck(check="partition_limits", status="fail", message="no partition given and no default")
        ]
    rule = rules.get(name)
    if rule is None:
        return [
            RecoverCheck(
                check="partition_exists",
                status="fail",
                message=f"partition {name!r} is not in the profile ({', '.join(sorted(rules))})",
            )
        ]
    if rule.interactive_only:
        checks.append(
            RecoverCheck(
                check="interactive_partition_batch", status="fail", message=f"{name} is interactive-only"
            )
        )
    max_s = parse_slurm_duration(rule.max_time) if rule.max_time else None
    if max_s is not None and res.time_seconds > max_s:
        checks.append(
            RecoverCheck(
                check="time_within_partition_max",
                status="fail",
                message=f"time {res.time} exceeds the {name} maximum {rule.max_time}",
            )
        )
    if res.gpu_type and rule.gpu_types and res.gpu_type not in rule.gpu_types:
        checks.append(
            RecoverCheck(
                check="gpu_type_in_partition",
                status="fail",
                message=f"gpu_type {res.gpu_type!r} is not offered by {name} ({', '.join(rule.gpu_types)})",
            )
        )
    if rule.max_gpus_per_job is not None and res.total_gpus > rule.max_gpus_per_job:
        checks.append(
            RecoverCheck(
                check="gpus_limits",
                status="fail",
                message=f"{res.total_gpus} GPUs exceed the {name} per-job limit {rule.max_gpus_per_job}",
            )
        )
    if rule.preemptible and not resumable:
        checks.append(
            RecoverCheck(
                check="preemptible_gate",
                status="fail",
                message=f"{name} is preemptible and the Run does not declare resilience.resumable",
            )
        )
    if not checks:
        checks.append(
            RecoverCheck(check="partition_limits", status="pass", message=f"within the {name} limits")
        )
    return checks


def _merge_resources(old: ResourceSpec, patch: dict[str, Any]) -> ResourceSpec:
    try:
        return ResourceSpec.model_validate({**old.model_dump(mode="json"), **patch})
    except ValidationError as exc:
        raise InvalidArgument(
            f"resource_patch is invalid: {exc.error_count()} error(s)",
            details={
                "errors": [{"loc": ".".join(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()[:10]]
            },
        ) from exc


def _hint_patch(diag: Diagnosis | None) -> tuple[dict[str, Any], list[str]]:
    """Resource patch from the primary diagnosis hint, minus what recovery must not apply blindly:
    node exclusion from a single attempt (rule: >= 2 attempts) and ``--signal`` (the program must
    handle it; declare resilience.stop_behavior instead)."""
    notes: list[str] = []
    item = diag.primary if diag is not None else None
    if item is None:
        return {}, notes
    patch = resource_patch_from_hint(dict(item.resource_hint))
    if patch.pop("exclude_nodes", None):
        notes.append(
            "the diagnosis names a suspect node; it is excluded only after failures of >= 2 attempts there"
        )
    extra = [f for f in patch.pop("extra_sbatch", []) if not f.startswith("--signal")]
    if extra:
        patch["extra_sbatch"] = extra
    if "extra_sbatch" in (item.resource_hint or {}) and "--signal" in item.resource_hint.get(
        "extra_sbatch", ""
    ):
        notes.append(
            "the diagnosis suggests a walltime warning signal: declare resilience.stop_behavior=checkpoint_and_exit "
            "in a derived Run whose program checkpoints on it (not applied automatically)"
        )
    return patch, notes


def _semantic_problems(
    old: ResourceSpec, new: ResourceSpec, *, batch_invariant: bool, requirement: str | None
) -> list[tuple[str, bool]]:
    """``(problem, semantic)``: semantic problems change the experiment (answer: a derived Run);
    the others only say the patch does not address the failure."""
    problems: list[tuple[str, bool]] = []
    world_changed = _world_size(old) != _world_size(new) or any(
        getattr(old, k) != getattr(new, k) for k in _WORLD_KEYS
    )
    if world_changed and not batch_invariant:
        problems.append(
            (
                f"changing gpus/nodes/tasks ({_world_size(old)} -> {_world_size(new)} ranks) changes the global "
                "batch unless resilience.batch_invariant_to_gpus: that is a derived Run",
                True,
            )
        )
    if requirement == "oom_semantics":
        if new.gpu_type == old.gpu_type and not (world_changed and batch_invariant):
            problems.append(
                (
                    "cuda_oom: only a GPU type with more memory (or more GPUs with a batch-invariant program) keeps "
                    "the experiment identical; pass resource_patch.gpu_type, or derive a Run",
                    True,
                )
            )
    elif requirement == "mem":
        old_b, new_b = old.mem_bytes(), new.mem_bytes()
        if new_b is None or (old_b is not None and new_b <= old_b):
            problems.append(("host_oom: the patch must raise resources.mem", False))
    elif requirement == "time" and new.time_seconds <= old.time_seconds:
        problems.append(("timeout: the patch must raise resources.time (or use strategy=resume)", False))
    return problems


def _suspect_nodes(
    reader: Database | Tx, *, cluster_id: str, now_iso: str
) -> tuple[dict[str, set[str]], set[str]]:
    """(node -> distinct attempt ids with a fresh failure observed ON that node, nodes only
    named by ``*_suspect`` observations). A ``*_suspect`` class comes from a multi-node failure
    that cannot tell which node failed: it is reported, never counted towards an exclusion."""
    now = parse_iso(now_iso)
    out: dict[str, set[str]] = {}
    suspect_only: set[str] = set()
    rows = reader.all(
        "SELECT obs_key, observed_at, expires_at FROM observations WHERE cluster_id=? AND obs_key LIKE 'node:%'",
        (cluster_id,),
    )
    for r in rows:
        parts = str(r["obs_key"]).split(":")
        if len(parts) < 4 or not parts[1]:
            continue
        node, cls, attempt = parts[1], ":".join(parts[2:-1]), parts[-1]
        expires = parse_iso(r["expires_at"]) if r["expires_at"] else None
        seen = parse_iso(r["observed_at"])
        if now is not None:
            if expires is not None and expires <= now:
                continue
            if expires is None and (seen is None or (now - seen).total_seconds() > NODE_OBS_TTL_S):
                continue
        if cls.endswith("_suspect"):
            suspect_only.add(node)
            continue
        out.setdefault(node, set()).add(attempt)
    return out, suspect_only - set(out)


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------


def _row(reader: Database | Tx, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    return reader.one(sql, params)


def _exec_plan(reader: Database | Tx, h: str) -> tuple[dict[str, Any], ExecutionPlan]:
    row = _row(reader, "SELECT * FROM execution_plans WHERE exec_plan_hash=?", (h,))
    if row is None:
        raise NotFound(f"execution plan {h} not found")
    try:
        return row, ExecutionPlan.model_validate(jload(row["plan_json"]))
    except ValidationError as exc:
        raise HpcLabError(
            f"execution plan {h} is not a valid ExecutionPlan",
            code=ErrorCode.INTERNAL,
            details={"exec_plan_hash": h},
        ) from exc


def _run_spec(run: dict[str, Any]) -> RunSpec:
    try:
        return RunSpec.model_validate(jload(run["spec_json"]))
    except ValidationError as exc:
        raise HpcLabError(
            f"run {run['run_id']} has an invalid stored RunSpec",
            code=ErrorCode.INTERNAL,
            details={"run_id": run["run_id"]},
        ) from exc


def _as_diag(value: Any) -> Diagnosis | None:
    if value is None:
        return None
    if isinstance(value, Diagnosis):
        return value
    try:
        return Diagnosis.model_validate(value.model_dump() if isinstance(value, BaseModel) else value)
    except (ValidationError, AttributeError, TypeError):
        return None


def _facts(a: dict[str, Any], *, previous_codes: list[str], cluster: ResolvedCluster | None) -> AttemptFacts:
    rule = cluster.profile.partitions.get(a.get("partition") or "") if cluster is not None else None
    max_s = parse_slurm_duration(rule.max_time) if rule is not None and rule.max_time else None
    return AttemptFacts.model_validate(
        {
            "scheduler_state": a.get("scheduler_state"),
            "raw_state": a.get("raw_scheduler_state"),
            "exit_code": a.get("exit_code"),
            "exit_signal": a.get("exit_signal"),
            "app_state": a.get("app_state"),
            "reason": a.get("reason"),
            "elapsed_s": a.get("elapsed_s"),
            "time_limit_s": a.get("time_limit_s"),
            "gpus": a.get("gpus"),
            "gpu_type": a.get("gpu_type"),
            "node_list": a.get("node_list"),
            "restart_count": a.get("restart_count") or 0,
            "partition": a.get("partition"),
            "partition_max_time_s": max_s,
            "preemptible": rule.preemptible if rule is not None else None,
            "result_state": a.get("result_state"),
            "previous_codes": tuple(previous_codes),
        }
    )


def _previous_codes(reader: Database | Tx, a: dict[str, Any]) -> list[str]:
    rows = reader.all(
        "SELECT diagnosis_json FROM attempts WHERE run_id=? AND seq < ? ORDER BY seq", (a["run_id"], a["seq"])
    )
    out: list[str] = []
    for r in rows:
        d = jload(r["diagnosis_json"], None) or {}
        if isinstance(d, dict) and d.get("primary_code"):
            out.append(str(d["primary_code"]))
    return out


def _diagnosis_for(
    ctx: ServiceContext,
    a: dict[str, Any],
    *,
    fetch_logs: bool,
    cluster: ResolvedCluster | None,
    coverage: Coverage,
) -> tuple[Diagnosis | None, Literal["computed", "stored", "scheduler_only", "none"]]:
    """Stored diagnosis (computed by observe.finalize while the logs were fresh) -> otherwise
    ``runs.diagnose_attempt`` (contract; reads bounded log tails when ``fetch_logs``) -> otherwise
    scheduler facts only. A stored diagnosis is never replaced by a re-diagnosis from logs that
    may have been purged since (that would turn e.g. a CUDA OOM into ``unknown_failure``)."""
    if a["phase"] != AttemptPhase.TERMINAL.value:
        return None, "none"
    stored = _as_diag(jload(a.get("diagnosis_json"), None))
    if stored is not None:
        return stored, "stored"
    try:
        fn = importlib.import_module("hpclab.services.runs").diagnose_attempt
    except (ImportError, AttributeError):
        fn = None
    if fn is not None:
        try:
            got = _as_diag(fn(ctx, attempt_id=a["attempt_id"], fetch_logs=fetch_logs))
            if got is not None:
                return got, "computed"
        except AuthRequiredError as exc:
            coverage.add("diagnosis", "unauthorized", detail=exc.message)
        except HpcLabError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                raise
            coverage.add("diagnosis", "partial", detail=exc.message)
    coverage.notes.append(f"{a['attempt_id']}: diagnosis from scheduler facts only (no log tails)")
    facts = _facts(a, previous_codes=_previous_codes(ctx.db, a), cluster=cluster)
    return diagnose(facts, "", ""), "scheduler_only"


def _diag_summary(
    diag: Diagnosis | None, source: Literal["computed", "stored", "scheduler_only", "none"]
) -> DiagnosisSummary | None:
    if diag is None:
        return None
    item = diag.primary
    return DiagnosisSummary(
        primary_code=diag.primary_code,
        category=CODE_CATEGORY.get(diag.primary_code or "") if diag.primary_code else None,
        title=item.title if item else None,
        confidence=item.confidence if item else None,
        summary=diag.summary,
        suggested_strategies=list(item.suggested_strategies) if item else [],
        resource_hint=dict(item.resource_hint) if item else {},
        last_error_line=diag.last_error_line,
        source=source,
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class _Ckpt:
    artifact_id: str | None
    remote_dir: str | None
    check: CheckpointCheck | None
    total_bytes: int | None = None
    largest_bytes: int | None = None
    blocking: str | None = None
    unavailable: bool = False


def _parse_ref(ref: str) -> tuple[str, str | None]:
    text = ref.strip()
    text = text.removeprefix("hpclab://artifacts/")
    art, _, member = text.partition("#")
    member_s = member.strip("/") or None
    if member_s is not None:
        norm = posixpath.normpath(member_s)
        if norm.startswith("../") or norm == ".." or norm.startswith("/") or "\x00" in norm:
            raise InvalidArgument(f"checkpoint_ref member {member!r} escapes the artifact")
        member_s = norm
    if not art:
        raise InvalidArgument("checkpoint_ref is empty")
    return art, member_s


def _read_manifest(transport: Any, path: str, timeout: float) -> str | None:
    """Manifest text, or None when it is larger than any sane manifest (treated as unreadable).
    Missing file -> FileNotFoundError (the caller then looks one level down)."""
    rr = transport.read_range(path, 0, MAX_MANIFEST_BYTES, timeout=timeout)
    if not rr.eof:
        return None
    data: bytes = rr.data
    return data.decode("utf-8", errors="replace")


def _candidate_dirs(transport: Any, base: str, timeout: float) -> list[str]:
    """Checkpoint dirs one level below ``base`` that hold a manifest (newest name first)."""
    listing: list[StatInfo] = transport.list_dir(base, max_entries=MAX_LISTING, timeout=timeout)
    dirs: set[str] = set()
    for st in listing:
        if posixpath.basename(st.path) != MANIFEST_NAME:
            continue
        d = posixpath.dirname(st.path)
        if posixpath.dirname(d).rstrip("/") != base.rstrip("/"):
            continue
        if posixpath.basename(d).startswith("."):
            continue  # .staging-*, .replaced-*: never checkpoints
        dirs.add(d)
    return sorted(dirs, reverse=True)[:MAX_CANDIDATE_DIRS]


def _validate_dir(
    transport: Any, d: str, *, run: dict[str, Any], spec: RunSpec, world: int, timeout: float
) -> tuple[CheckpointCheck, int, int]:
    """Metadata validation of one checkpoint dir: manifest + login-node ``stat`` of its files
    (sizes only; content digests are verified by the compute probe)."""
    text = _read_manifest(transport, f"{d}/{MANIFEST_NAME}", timeout)
    if text is None:
        return check_manifest_text(b"\xff", {}, expected_run_spec_hash=None), 0, 0
    try:
        manifest = json.loads(text)
    except (ValueError, RecursionError):
        manifest = None
    files = manifest.get("files") if isinstance(manifest, dict) else None
    rels = (
        [f["path"] for f in files if isinstance(f, dict) and isinstance(f.get("path"), str)][:MAX_CKPT_FILES]
        if isinstance(files, list)
        else []
    )
    safe = [r for r in rels if r and not r.startswith("/") and ".." not in r.split("/")]
    stats: dict[str, StatInfo] = transport.stat([f"{d}/{r}" for r in safe], timeout=timeout) if safe else {}
    file_stats: dict[str, Any] = {}
    for r in safe:
        st = stats.get(f"{d}/{r}")
        if st is not None:
            file_stats[r] = st
    sizes = [st.size or 0 for st in file_stats.values() if st.exists]
    check = check_manifest_text(
        text,
        file_stats,
        expected_run_spec_hash=spec.spec_hash(),
        expected_config=dict(spec.params),
        expected_world_size=world,
        expected_run_id=run["run_id"],
    )
    return check, sum(sizes), max(sizes, default=0)


def _dirs_with_checks(
    transport: Any, base: str, *, run: dict[str, Any], spec: RunSpec, world: int, timeout: float
) -> list[tuple[str, CheckpointCheck, int, int]]:
    """``base`` itself if it holds a manifest, else its checkpoint subdirectories."""
    try:
        check, total, largest = _validate_dir(
            transport, base, run=run, spec=spec, world=world, timeout=timeout
        )
        return [(base, check, total, largest)]
    except (FileNotFoundError, NotADirectoryError):
        pass
    found: list[tuple[str, CheckpointCheck, int, int]] = []
    for d in _candidate_dirs(transport, base, timeout):
        try:
            c, t, lg = _validate_dir(transport, d, run=run, spec=spec, world=world, timeout=timeout)
        except (FileNotFoundError, NotADirectoryError):
            continue
        found.append((d, c, t, lg))
    return found


def _resolve_checkpoint(
    ctx: ServiceContext,
    *,
    run: dict[str, Any],
    spec: RunSpec,
    cluster_id: str,
    checkpoint_ref: str | None,
    world: int,
    coverage: Coverage,
) -> _Ckpt:
    """Find and validate (metadata only) the checkpoint to resume from: the given artifact (or a
    ``#member`` dir of it), else this Run's newest indexed checkpoint artifacts. Within a dir
    artifact the highest-step subdirectory whose manifest is an exact continuation wins."""
    if checkpoint_ref is not None:
        art_id, member = _parse_ref(checkpoint_ref)
        row = ctx.db.one("SELECT * FROM artifacts WHERE artifact_id=?", (art_id,))
        if row is None:
            raise NotFound(
                f"checkpoint artifact {art_id} not found", details={"checkpoint_ref": checkpoint_ref}
            )
        if row["run_id"] != run["run_id"]:
            return _Ckpt(
                art_id,
                None,
                None,
                blocking=f"checkpoint {art_id} belongs to run {row['run_id']}: starting from another Run's checkpoint "
                "is a new experiment (derive a Run)",
            )
        if row["role"] != "checkpoint":
            return _Ckpt(
                art_id,
                None,
                None,
                blocking=f"artifact {art_id} ({row['name']}) is not a checkpoint (role {row['role']}): pass an "
                "artifact with role checkpoint",
            )
        candidates: list[tuple[dict[str, Any], str | None]] = [(row, member)]
    else:
        # declared checkpoint dirs first (newest attempt first), then indexed member files
        rows = ctx.db.all(
            "SELECT a.* FROM artifacts a JOIN attempts t ON t.attempt_id = a.attempt_id "
            "WHERE a.run_id=? AND a.role='checkpoint' AND a.integrity != 'missing' "
            "ORDER BY CASE a.kind WHEN 'dir' THEN 0 ELSE 1 END, t.seq DESC, a.observed_at DESC LIMIT 50",
            (run["run_id"],),
        )
        if not rows:
            return _Ckpt(
                None,
                None,
                None,
                blocking="no checkpoint artifact is indexed for this Run: pass checkpoint_ref",
            )
        candidates = [(r, None) for r in rows]
    transport = ctx.clusters.transport(cluster_id)
    timeout = ctx.settings.request_deadline_s
    fallback: _Ckpt | None = None
    seen: set[str] = set()
    for row, member in candidates[:MAX_CANDIDATE_DIRS]:
        base = str(row["remote_path"]).rstrip("/")
        if member:
            base = f"{base}/{member}"
        elif row["kind"] == "file":
            base = posixpath.dirname(base)  # a member file: its checkpoint directory
        if posixpath.basename(base) == MANIFEST_NAME:
            base = posixpath.dirname(base)
        if base in seen:
            continue
        seen.add(base)
        if row["integrity"] == "missing":
            fallback = fallback or _Ckpt(
                row["artifact_id"], base, None, blocking="the checkpoint artifact is missing"
            )
            continue
        try:
            found = _dirs_with_checks(transport, base, run=run, spec=spec, world=world, timeout=timeout)
        except AuthRequiredError:
            raise
        except (FileNotFoundError, NotADirectoryError):
            fallback = fallback or _Ckpt(
                row["artifact_id"], base, None, blocking=f"checkpoint directory {base} does not exist"
            )
            continue
        except TransportError as exc:
            coverage.add("checkpoint", "unavailable", detail=exc.message)
            return _Ckpt(
                row["artifact_id"],
                base,
                None,
                blocking=f"checkpoint could not be read: {exc.message}",
                unavailable=True,
            )
        if not found:
            fallback = fallback or _Ckpt(
                row["artifact_id"],
                base,
                None,
                blocking=f"no {MANIFEST_NAME} manifest under {base}: an existing weight file proves nothing; write "
                "checkpoints with hpclab_io.write_checkpoint",
            )
            continue
        ok = [f for f in found if f[1].metadata_ok_for_resume]
        pick = max(ok or found, key=lambda f: (f[1].step or 0, f[0]))
        cand = _Ckpt(row["artifact_id"], pick[0], pick[1], pick[2], pick[3])
        if pick[1].metadata_ok_for_resume:
            coverage.add("checkpoint", "ok")
            return cand
        if fallback is None or fallback.check is None:
            fallback = cand
    assert fallback is not None
    if fallback.check is not None and fallback.blocking is None:
        fallback.blocking = _ckpt_blocking(fallback.check)
    return fallback


def _ckpt_blocking(check: CheckpointCheck) -> str:
    if check.status == "invalid":
        errs = "; ".join(i.message for i in check.errors[:3])
        return f"checkpoint is invalid: {errs}"
    if check.continuation != "exact":
        return f"checkpoint is not an exact continuation ({check.continuation}): starting from it is a derived Run"
    return "checkpoint metadata could not be confirmed"


def _ckpt_summary(ck: _Ckpt | None) -> CheckpointSummary | None:
    if ck is None:
        return None
    c = ck.check
    return CheckpointSummary(
        artifact_id=ck.artifact_id,
        remote_dir=ck.remote_dir,
        step=c.step if c else None,
        status=c.status if c else "unknown",
        continuation=c.continuation if c else "unknown",
        manifest_digest=c.manifest_digest if c else None,
        needs_load_probe=c.needs_load_probe if c else True,
        total_bytes=ck.total_bytes,
        issues=[f"{i.severity}:{i.code}: {i.message}" for i in (c.issues if c else [])][:10],
    )


# ---------------------------------------------------------------------------
# Per-attempt planning (no writes)
# ---------------------------------------------------------------------------


@dataclass
class _ProbeRequest:
    run_id: str
    spec: RunSpec
    plan: ExecutionPlan
    label: str
    target_run_id: str


@dataclass
class _CellPlan:
    run: dict[str, Any]
    attempt: dict[str, Any] | None
    strategy: RecoverStrategy | None = None
    cell_strategy: CellStrategy | None = None
    eligible: bool = False
    blocking: list[str] = field(default_factory=list)
    checks: list[RecoverCheck] = field(default_factory=list)
    hints: list[RecoverHint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnosis: DiagnosisSummary | None = None
    plan: ExecutionPlan | None = None
    exec_plan_hash: str | None = None
    reuse_exec_plan: bool = False
    patch: dict[str, Any] = field(default_factory=dict)
    exclude_nodes: list[str] = field(default_factory=list)
    suspect_nodes: list[str] = field(default_factory=list)
    checkpoint: _Ckpt | None = None
    probe: ProbeStatus | None = None
    probe_request: _ProbeRequest | None = None
    code: str = "?"

    def block(self, reason: str) -> None:
        self.blocking.append(reason)
        self.eligible = False


def _hold_hint(hold: ExecutionHold) -> RecoverHint:
    """What ends an execution hold: time (requeue watch) or a human confirmation after a live
    squeue/sacct check (``attempt_resolve`` is a human decision: CLI, not an agent tool)."""
    when = f"run_recover again after {hold.until}, or " if hold.until else ""
    return RecoverHint(
        strategy="wait",
        message=f"{when}{hold.hint}",
        arguments={
            "attempt_id": hold.attempt_id,
            "job_id": hold.job_id,
            "hold": hold.kind,
            **({"until": hold.until} if hold.until else {}),
        },
    )


def _plan_attempt(
    ctx: ServiceContext,
    a: dict[str, Any],
    run: dict[str, Any],
    *,
    requested: RecoverStrategy | None,
    checkpoint_ref: str | None,
    user_patch: dict[str, Any],
    fetch_logs: bool,
    allow_remote: bool,
    coverage: Coverage,
    now_iso: str,
) -> _CellPlan:
    cp = _CellPlan(run=run, attempt=a)
    cluster = ctx.clusters.get(a["cluster_id"])
    _ep_row, old_plan = _exec_plan(ctx.db, a["exec_plan_hash"])
    spec = _run_spec(run)
    resilience = old_plan.resilience
    resumable = bool(resilience and resilience.resumable)
    batch_invariant = bool(resilience and resilience.batch_invariant_to_gpus)

    diag, source = _diagnosis_for(ctx, a, fetch_logs=fetch_logs, cluster=cluster, coverage=coverage)
    cp.diagnosis = _diag_summary(diag, source)
    newer = ctx.db.one(
        "SELECT attempt_id FROM attempts WHERE run_id=? AND seq > ? ORDER BY seq DESC LIMIT 1",
        (a["run_id"], a["seq"]),
    )
    claim = run.get("claim_attempt_id")
    claim_other = claim if claim and claim != a["attempt_id"] else None
    holds = execution_holds(ctx.db, run["run_id"], now=ctx.clock.now())
    hint_patch, hint_notes = _hint_patch(diag)
    has_ckpt = bool(checkpoint_ref) or bool(
        ctx.db.one(
            "SELECT 1 AS x FROM artifacts WHERE run_id=? AND role='checkpoint' LIMIT 1", (run["run_id"],)
        )
    )
    asm = assess(
        phase=a["phase"],
        scheduler_state=a["scheduler_state"],
        result_state=a["result_state"],
        diagnosis=diag,
        requested=requested,
        resumable=resumable,
        has_checkpoint=has_ckpt,
        has_hint_patch=bool(hint_patch),
        run_id=run["run_id"],
        newer_attempt_id=newer["attempt_id"] if newer else None,
        claim_attempt_id=claim_other,
        holds=[h.message for h in holds],
    )
    cp.strategy = asm.strategy
    cp.cell_strategy = asm.strategy
    cp.hints.extend(asm.hints)
    cp.warnings.extend(asm.warnings)
    cp.blocking.extend(asm.blocking)
    if holds and any(h.message in asm.blocking for h in holds):
        cp.hints.extend(_hold_hint(h) for h in holds)
    if a["phase"] == AttemptPhase.ABANDONED.value and a.get("job_id"):
        job = a["job_id"]
        cp.warnings.append(
            f"attempt {a['attempt_id']} was abandoned while bound to Slurm job {job}, which hpclab could not "
            f"observe: make sure it is not running any more (`squeue -j {job}`; `scancel {job}` if it is) "
            "before another attempt executes this run"
        )
    if cp.blocking or asm.strategy is None:
        return cp
    strategy = asm.strategy

    # -- resources ---------------------------------------------------------
    patch: dict[str, Any] = {}
    if strategy == "adjust_resources":
        patch = {**hint_patch, **user_patch}
        cp.warnings.extend(hint_notes)
        if not patch:
            cp.block("adjust_resources needs a resource_patch (the diagnosis suggests none)")
            return cp
    elif strategy == "resume":
        patch = dict(user_patch)
    old_res = old_plan.resources
    suspects, suspect_only = _suspect_nodes(ctx.db, cluster_id=a["cluster_id"], now_iso=now_iso)
    auto_exclude = sorted(
        (n for n, atts in suspects.items() if len(atts) >= 2), key=lambda n: (-len(suspects[n]), n)
    )
    auto_exclude = auto_exclude[:MAX_AUTO_EXCLUDE]
    cp.suspect_nodes = sorted({n for n, atts in suspects.items() if len(atts) < 2} | suspect_only)
    if diag is not None and diag.primary is not None and diag.primary.resource_hint.get("exclude_nodes"):
        for n in resource_patch_from_hint({"exclude_nodes": diag.primary.resource_hint["exclude_nodes"]}).get(
            "exclude_nodes", []
        ):
            if n not in auto_exclude and n not in cp.suspect_nodes:
                cp.suspect_nodes.append(n)
    new_res = _merge_resources(old_res, patch)
    excl = list(dict.fromkeys([*new_res.exclude_nodes, *auto_exclude]))
    added = [
        n for n in auto_exclude if n not in old_res.exclude_nodes and n not in patch.get("exclude_nodes", [])
    ]
    if added:
        new_res = _merge_resources(new_res, {"exclude_nodes": excl})
        cp.checks.append(
            RecoverCheck(
                check="node_exclusion",
                status="pass",
                message="excluding "
                + ", ".join(f"{n} ({len(suspects[n])} attempts)" for n in added)
                + f" from fresh failure observations (>= 2 attempts, {NODE_OBS_TTL_S // 3600} h)",
            )
        )
    cp.exclude_nodes = list(new_res.exclude_nodes)
    cp.patch = {k: v for k, v in patch.items()}
    if added:
        cp.patch["exclude_nodes"] = list(new_res.exclude_nodes)
    # a resume after a timeout makes progress with the same limit; every other requirement holds
    requirement = None if (strategy == "resume" and asm.requirement == "time") else asm.requirement
    for problem, semantic in _semantic_problems(
        old_res, new_res, batch_invariant=batch_invariant, requirement=requirement
    ):
        cp.block(problem)
        if semantic:
            cp.hints.append(_derive_hint(run["run_id"], problem, recapture_code=False))
    if patch or added:
        for chk in validate_resources(cluster, new_res, resumable=resumable):
            cp.checks.append(chk)
            if chk.status == "fail":
                cp.block(f"{chk.check}: {chk.message}")
    if cp.blocking:
        return cp

    # -- checkpoint + probe (resume) -----------------------------------------
    recovery_kw: dict[str, Any] = {}
    if strategy == "resume":
        if not allow_remote:
            cp.block("too many resume cells in one batch: recover them in smaller groups")
            return cp
        ck = _resolve_checkpoint(
            ctx,
            run=run,
            spec=spec,
            cluster_id=a["cluster_id"],
            checkpoint_ref=checkpoint_ref,
            world=_world_size(new_res),
            coverage=coverage,
        )
        cp.checkpoint = ck
        if ck.blocking:
            cp.block(ck.blocking)
            cp.checks.append(
                RecoverCheck(
                    check="checkpoint_metadata",
                    status="unchecked" if ck.unavailable else "fail",
                    message=ck.blocking,
                )
            )
            if ck.check is not None and ck.check.continuation in ("weights_only", "incompatible"):
                cp.hints.append(
                    _derive_hint(
                        run["run_id"],
                        "Initialising from a non-exact checkpoint is a derived Run.",
                        recapture_code=False,
                    )
                )
            return cp
        assert ck.check is not None and ck.check.manifest_digest is not None and ck.remote_dir is not None
        cp.checks.append(
            RecoverCheck(
                check="checkpoint_metadata",
                status="pass",
                message=f"step {ck.check.step}: manifest valid, exact continuation ({ck.check.files_checked} file(s))",
            )
        )
        probes = find_probe_runs(
            ctx.db,
            experiment_id=run["experiment_id"],
            target_run_id=run["run_id"],
            manifest_digest=ck.check.manifest_digest,
            checkpoint_dir=ck.remote_dir,
        )
        ps = probe_status(
            ctx, probe_runs=probes, manifest_digest=ck.check.manifest_digest, cluster_id=a["cluster_id"]
        )
        cp.probe = ps
        if ps.state == "passed":
            cp.checks.append(
                RecoverCheck(check="checkpoint_load_probe", status="pass", message=ps.detail or "passed")
            )
            if not ps.load_checked:
                cp.warnings.append(
                    "the probe verified digests but could not load the files (torch not importable there)"
                )
        elif ps.state in ("running", "not_submitted"):
            cp.checks.append(
                RecoverCheck(check="checkpoint_load_probe", status="pending", message=ps.detail or ps.state)
            )
            cp.block(
                f"checkpoint load probe {ps.run_id} has not passed yet ({ps.state}): submit it / wait for it"
            )
            return cp
        elif ps.state == "failed":
            cp.checks.append(
                RecoverCheck(check="checkpoint_load_probe", status="fail", message=ps.detail or "failed")
            )
            cp.block(f"checkpoint load probe {ps.run_id} failed: {ps.detail}")
            cp.hints.append(
                RecoverHint(
                    strategy="resume",
                    message="Pick an earlier checkpoint (checkpoint_ref=<artifact>#<ckpt dir>) or retry from scratch; "
                    "if the probe itself hit an infrastructure failure, recover the probe run.",
                )
            )
            return cp
        elif ps.state == "unverified":
            cp.checks.append(
                RecoverCheck(
                    check="checkpoint_load_probe", status="unchecked", message=ps.detail or "unverified"
                )
            )
            cp.block(f"the probe result could not be read now: {ps.detail}")
            return cp
        else:  # none: create one
            probe_spec = probe_run_spec(
                spec,
                target_run_id=run["run_id"],
                checkpoint_dir=ck.remote_dir,
                manifest_digest=ck.check.manifest_digest,
                artifact_id=ck.artifact_id,
                step=ck.check.step,
            )
            probe_id = new_id("run")
            probe_plan = ExecutionPlan(
                run_id=probe_id,
                run_spec_hash=probe_spec.spec_hash(),
                cluster_id=old_plan.cluster_id,
                profile_revision=cluster.revision,
                resources=probe_resources(
                    new_res, checkpoint_bytes=ck.total_bytes, largest_file_bytes=ck.largest_bytes
                ),
                remote_root=old_plan.remote_root,
                snapshot_path=old_plan.snapshot_path,
                created_reason="initial",
                # a probe only loads a checkpoint: never let Slurm requeue (rerun) it by itself
                requeue=False,
            )
            cp.probe_request = _ProbeRequest(
                run_id=probe_id,
                spec=probe_spec,
                plan=probe_plan,
                label=f"probe.{run['label']}"[:120],
                target_run_id=run["run_id"],
            )
            cp.checks.append(
                RecoverCheck(
                    check="checkpoint_load_probe",
                    status="pending",
                    message="loadability needs a compute-node probe run (created; submit it)",
                )
            )
            cp.block(
                f"checkpoint loadability is unverified: submit probe run {probe_id} first, then run_recover again"
            )
            return cp
        recovery_kw = {
            "checkpoint_artifact_id": ck.artifact_id,
            "checkpoint_digest": ck.check.manifest_digest,
            "checkpoint_remote_path": ck.remote_dir,
        }

    # -- the new execution plan ----------------------------------------------
    new_plan = ExecutionPlan.model_validate(
        {
            **old_plan.model_dump(mode="json"),
            "resources": new_res.model_dump(mode="json"),
            "profile_revision": cluster.revision,
            "created_reason": strategy,
            "recovery": RecoverySource(
                from_attempt_id=a["attempt_id"],
                strategy=strategy,
                resource_patch=cp.patch,
                diagnosis_codes=[i.code for i in diag.items][:8] if diag is not None else [],
                **recovery_kw,
            ).model_dump(mode="json"),
        }
    )
    h = new_plan.plan_hash()
    existing = ctx.db.one(
        "SELECT status, consumed_by_attempt_id FROM execution_plans WHERE exec_plan_hash=?", (h,)
    )
    if existing is not None and existing["consumed_by_attempt_id"]:
        cp.block(f"this exact recovery plan was already used by attempt {existing['consumed_by_attempt_id']}")
        return cp
    cp.reuse_exec_plan = existing is not None
    cp.plan = new_plan
    cp.exec_plan_hash = h
    cp.eligible = True
    return cp


def _resume_call(wf: dict[str, Any]) -> str:
    return call(
        "workflow_submit",
        workflow_id=wf["workflow_id"],
        resume_dispatch=True,
        expected_version=int(wf["version"]),
        idempotency_key=NEW_KEY,
    )


def workflow_hold_reason(ctx: ServiceContext, wf: dict[str, Any], cell_state: str | None) -> str | None:
    """Why a never-dispatched cell must not be recovered because its workflow still owns it -
    saying what the user has to do when that workflow does not dispatch by itself - or None
    when the workflow no longer dispatches (closed / cancelled / finished: recover it).

    Only a ``planned`` cell is still owned: a cell the workflow skipped (no ready execution plan
    at dispatch time) or cancelled is never dispatched by it, whatever state the workflow is in."""
    if cell_state != "planned":
        return None
    wid = wf["workflow_id"]
    if wf["state"] == "gate_failed":
        held = f"not dispatched: the smoke gate of workflow {wid} failed, so it holds its main cells back; "
        smoke = (
            ctx.db.one(
                "SELECT a.attempt_id, a.phase, a.scheduler_state, a.result_state FROM runs r "
                "JOIN attempts a ON a.attempt_id = r.latest_attempt_id WHERE r.run_id=?",
                (wf["gate_run_id"],),
            )
            if wf.get("gate_run_id")
            else None
        )
        # follow the smoke run's LATEST attempt: after a recovery + resubmit it is the retry, and
        # telling the user to recover that one again would be wrong
        if smoke is not None and (smoke["phase"] in _ACTIVE or smoke["phase"] in _UNRESOLVED):
            return (
                held + f"the smoke run's retry (attempt {smoke['attempt_id']}) is {smoke['phase']}; once it "
                f"passes, resume dispatch with {_resume_call(wf)}"
            )
        if (
            smoke is not None
            and smoke["scheduler_state"] == SchedulerState.COMPLETED.value
            and smoke["result_state"] == "valid"
        ):
            return (
                held + f"the smoke run passed on attempt {smoke['attempt_id']}: resume dispatch with "
                f"{_resume_call(wf)} (the gate is re-evaluated on that attempt)"
            )
        recover = (
            call("run_recover", attempt_id=smoke["attempt_id"], idempotency_key=NEW_KEY)
            if smoke is not None
            else "run_recover"
        )
        return (
            held
            + f"recover the smoke run ({recover}), and once it passes resume dispatch with {_resume_call(wf)}"
        )
    if wf["state"] not in ("gating", "dispatching", "draining") or not wf["dispatch_open"]:
        return None
    if wf["dispatch_state"] == "paused":
        return (
            f"not dispatched: workflow {wid}'s dispatch is paused by its circuit breaker; fix the cause of the "
            f"repeated failures, then resume it with {_resume_call(wf)} - it dispatches this cell then"
        )
    if wf["dispatch_state"] != "open":
        return None
    op = (
        ctx.db.one(
            "SELECT state, stage, error_json FROM operations WHERE operation_id=?", (wf["operation_id"],)
        )
        if wf.get("operation_id")
        else None
    )
    if op is not None and op["state"] == "blocked" and op["stage"] == "budget_exhausted":
        err = jload(op["error_json"], None) or {}
        why = err.get("message") if isinstance(err, dict) else None
        return (
            f"not dispatched yet: workflow {wid} waits for budget ({why or 'budget exhausted'}); it dispatches "
            "this cell once GPU-hours are available (raise the experiment budget or let running attempts settle)"
        )
    return f"not dispatched yet: workflow {wid} ({wf['state']}) will dispatch it"


def _initial_cell(ctx: ServiceContext, run: dict[str, Any], wf: dict[str, Any]) -> _CellPlan:
    """A never-dispatched cell of a workflow that no longer dispatches: its own initial plan."""
    cp = _CellPlan(run=run, attempt=None, cell_strategy="initial")
    owned = workflow_hold_reason(ctx, wf, run.get("wf_cell_state", "planned"))
    if owned is not None:
        cp.block(owned)
        return cp
    h = run.get("current_exec_plan_hash")
    row = (
        ctx.db.one("SELECT status, consumed_by_attempt_id FROM execution_plans WHERE exec_plan_hash=?", (h,))
        if h
        else None
    )
    if row is None or row["status"] != "ready" or row["consumed_by_attempt_id"]:
        cp.block("the run has no unused ready execution plan")
        return cp
    assert h is not None
    try:  # its GPU-hours are known from the frozen plan: never reported as 0 / unknown
        cp.plan = _exec_plan(ctx.db, h)[1]
    except HpcLabError as exc:
        cp.warnings.append(f"GPU-hours unknown: {exc.message}")
    cp.exec_plan_hash = h
    cp.reuse_exec_plan = True
    cp.eligible = True
    return cp


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _plan_ref(plan_id: str | None) -> str | None:
    return f"hpclab://plans/{plan_id}" if plan_id else None


def _insert_plan(
    tx: Tx,
    *,
    experiment_id: str,
    kind: str,
    cells: list[tuple[str, str]],
    snapshot_digest: str | None,
    validation: dict[str, Any],
) -> str:
    plan_id = new_id("plan")
    tx.insert(
        "plans",
        {
            "plan_id": plan_id,
            "experiment_id": experiment_id,
            "kind": kind,
            "status": "ready",
            "plan_hash": digest({"kind": kind, "cells": sorted([list(c) for c in cells])}),
            "snapshot_digest": snapshot_digest,
            "validation_json": jdump(validation),
            "smoke_run_id": None,
            "operation_id": None,
            "created_at": tx.now,
            "updated_at": tx.now,
        },
    )
    tx.event("plan", plan_id, "plan.created", {"kind": kind, "cells": len(cells)}, entity_version=1)
    return plan_id


def _insert_exec_plan(tx: Tx, plan: ExecutionPlan, *, validation: dict[str, Any]) -> str:
    h = plan.plan_hash()
    row = tx.one("SELECT consumed_by_attempt_id FROM execution_plans WHERE exec_plan_hash=?", (h,))
    if row is not None:
        if row["consumed_by_attempt_id"]:
            raise Stale(
                f"execution plan {h} was consumed while the recovery was being planned; run_recover again",
                retryable=True,
            )
        return h
    tx.insert(
        "execution_plans",
        {
            "exec_plan_hash": h,
            "run_id": plan.run_id,
            "plan_json": jdump(plan.model_dump(mode="json")),
            "status": "ready",
            "validation_json": jdump(validation),
            "created_at": tx.now,
        },
    )
    tx.event(
        "execution_plan",
        h,
        "execution_plan.created",
        {"run_id": plan.run_id, "created_reason": plan.created_reason, "recovery": bool(plan.recovery)},
    )
    return h


def _insert_probe_runs(
    tx: Tx, *, experiment_id: str, requests: list[_ProbeRequest]
) -> tuple[str, dict[str, str]]:
    cells = [(r.run_id, r.plan.plan_hash()) for r in requests]
    plan_id = _insert_plan(
        tx,
        experiment_id=experiment_id,
        kind="probe",
        cells=cells,
        snapshot_digest=requests[0].spec.code.digest if requests else None,
        validation={
            "schema": PROBE_PLAN_SCHEMA,
            "status": "pass",
            "cells": [
                {"run_id": r.run_id, "exec_plan_hash": r.plan.plan_hash(), "target_run_id": r.target_run_id}
                for r in requests
            ],
        },
    )
    hashes: dict[str, str] = {}
    for r in requests:
        h = r.plan.plan_hash()
        tx.insert(
            "runs",
            {
                "run_id": r.run_id,
                "experiment_id": experiment_id,
                "plan_id": plan_id,
                "kind": "probe",
                "label": r.label,
                "cell_key": jdump(r.spec.cell),
                "seed": None,
                "spec_json": jdump(r.spec.model_dump(mode="json")),
                "spec_hash": r.spec.spec_hash(),
                "derived_from": None,
                "current_exec_plan_hash": h,
                "claim_attempt_id": None,
                "latest_attempt_id": None,
                "status": "prepared",
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.event(
            "run",
            r.run_id,
            "run.created",
            {"kind": "probe", "target_run_id": r.target_run_id},
            entity_version=1,
        )
        _insert_exec_plan(tx, r.plan, validation={"probe_for": r.target_run_id})
        hashes[r.run_id] = h
    return plan_id, hashes


def _point_run_at(
    tx: Tx,
    run_id: str,
    exec_plan_hash: str,
    *,
    plan_id: str | None,
    strategy: str | None,
    from_attempt_id: str | None,
) -> None:
    """The Run's current execution plan := the new ready recovery plan (same tx, evented), so
    "submit this run" (run_submit, CLI ``submit RUN``, derived Runs) means the recovery plan."""
    row = tx.one("SELECT current_exec_plan_hash FROM runs WHERE run_id=?", (run_id,))
    if row is None or row["current_exec_plan_hash"] == exec_plan_hash:
        return
    version = tx.update("runs", "run_id", run_id, {"current_exec_plan_hash": exec_plan_hash})
    tx.event(
        "run",
        run_id,
        "run.exec_plan_changed",
        {
            "from": row["current_exec_plan_hash"],
            "to": exec_plan_hash,
            "reason": "recovery",
            "strategy": strategy,
            "plan_id": plan_id,
            "from_attempt_id": from_attempt_id,
        },
        entity_version=version,
    )


def recovery_plan_cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells of a ``plans.kind='recovery'`` row: ``[{run_id, eligible, strategy, exec_plan_hash,
    blocking_reason, ...}]`` in dispatch order (used by ``workflow_submit``)."""
    val = jload(plan.get("validation_json"), {}) or {}
    cells = val.get("cells") if isinstance(val, dict) else None
    return [c for c in (cells or []) if isinstance(c, dict) and c.get("run_id")]


# ---------------------------------------------------------------------------
# run_recover
# ---------------------------------------------------------------------------


def _payload_hash(**params: Any) -> str:
    return digest(
        {k: (v.model_dump(mode="json") if isinstance(v, BaseModel) else v) for k, v in params.items()}
    )


def _stored(ctx: ServiceContext, key: str, payload_hash: str) -> RecoverData | None:
    row = ctx.db.one(
        "SELECT payload_hash, response_json FROM idempotency WHERE principal=? AND tool=? AND idem_key=?",
        (ctx.principal, TOOL, key),
    )
    if row is None:
        return None
    if row["payload_hash"] != payload_hash:
        raise IdempotencyConflict(
            f"idempotency_key {key!r} was already used for a different {TOOL} request",
            details={"tool": TOOL, "idempotency_key": key},
        )
    return RecoverData.model_validate(jload(row["response_json"]))


def _check_unchanged(tx: Tx, attempts: list[dict[str, Any]]) -> None:
    for a in attempts:
        cur = tx.one("SELECT version FROM attempts WHERE attempt_id=?", (a["attempt_id"],))
        if cur is None or cur["version"] != a["version"]:
            raise Stale(
                f"attempt {a['attempt_id']} changed while the recovery was being planned; run_recover again",
                retryable=True,
                details={"attempt_id": a["attempt_id"]},
            )


def run_recover(
    ctx: ServiceContext,
    *,
    idempotency_key: str,
    attempt_id: str | None = None,
    selector: RecoverySelector | None = None,
    strategy: RecoverStrategy | None = None,
    checkpoint_ref: str | None = None,
    resource_patch: ResourcePatch | None = None,
) -> RecoverData:
    """Plan a recovery (retry / resume / adjust_resources) for a stopped attempt, or for the broken
    cells of a workflow (``selector``). Never submits: returns a ready execution plan to pass to
    run_submit (single) or a recovery plan for workflow_submit (batch), a probe run to submit
    first when a checkpoint's loadability is unverified, or ``eligible=false`` with the reasons."""
    if (attempt_id is None) == (selector is None):
        raise InvalidArgument("give exactly one of attempt_id or selector")
    if checkpoint_ref is not None and strategy not in (None, "resume"):
        raise InvalidArgument("checkpoint_ref is only used with strategy='resume'")
    if checkpoint_ref is not None and selector is not None:
        raise InvalidArgument("checkpoint_ref applies to a single attempt, not to a selector")
    if checkpoint_ref is not None and strategy is None:
        strategy = "resume"
    user_patch = resource_patch.as_dict() if resource_patch is not None else {}
    if user_patch and strategy is None:
        strategy = "adjust_resources"
    if user_patch and strategy == "retry":
        raise InvalidArgument("resource_patch is only used with strategy 'adjust_resources' or 'resume'")
    payload_hash = _payload_hash(
        attempt_id=attempt_id,
        selector=selector,
        strategy=strategy,
        checkpoint_ref=checkpoint_ref,
        resource_patch=user_patch,
    )
    stored = _stored(ctx, idempotency_key, payload_hash)
    if stored is not None:
        return stored
    if attempt_id is not None:
        return _recover_single(
            ctx,
            attempt_id=attempt_id,
            strategy=strategy,
            checkpoint_ref=checkpoint_ref,
            user_patch=user_patch,
            key=idempotency_key,
            payload_hash=payload_hash,
        )
    assert selector is not None
    return _recover_batch(
        ctx,
        selector=selector,
        strategy=strategy,
        user_patch=user_patch,
        key=idempotency_key,
        payload_hash=payload_hash,
    )


def _single_data(cp: _CellPlan, coverage: Coverage) -> RecoverData:
    a = cp.attempt or {}
    ck = cp.checkpoint
    return RecoverData(
        mode="single",
        eligible=cp.eligible and not cp.blocking,
        strategy=cp.strategy,
        attempt_id=a.get("attempt_id"),
        run_id=cp.run["run_id"],
        experiment_id=cp.run["experiment_id"],
        diagnosis=cp.diagnosis,
        checks=cp.checks,
        blocking=cp.blocking,
        hints=cp.hints,
        exec_plan_hash=cp.exec_plan_hash if cp.eligible else None,
        created_reason=cp.plan.created_reason if cp.plan is not None else None,
        resources=cp.plan.resources.model_dump(mode="json") if cp.plan is not None else None,
        resource_patch=cp.patch,
        exclude_nodes=cp.exclude_nodes,
        suspect_nodes=cp.suspect_nodes,
        requested_gpu_hours=round(cp.plan.resources.requested_gpu_hours, 4) if cp.plan is not None else None,
        checkpoint=_ckpt_summary(ck),
        probe=_probe_info(cp.probe, None),
        coverage=coverage,
        warnings=list(dict.fromkeys(cp.warnings)),
    )


def _probe_info(ps: ProbeStatus | None, created: tuple[str, str, str] | None) -> ProbeInfo | None:
    if created is not None:
        run_id, plan_id, h = created
        return ProbeInfo(
            state="not_submitted",
            run_id=run_id,
            plan_id=plan_id,
            plan_ref=_plan_ref(plan_id),
            exec_plan_hash=h,
            detail="created: submit it with run_submit (or workflow_submit on its plan)",
            created=True,
        )
    if ps is None or ps.state == "none":
        return None
    return ProbeInfo(
        state=ps.state,
        run_id=ps.run_id,
        plan_id=ps.plan_id,
        plan_ref=_plan_ref(ps.plan_id),
        exec_plan_hash=ps.exec_plan_hash,
        attempt_id=ps.attempt_id,
        detail=ps.detail,
        load_checked=ps.load_checked,
    )


def _budget_check(ctx: ServiceContext, experiment_id: str, gpu_hours: float) -> RecoverCheck:
    try:
        budget = importlib.import_module("hpclab.services.budget")
        summ = budget.summary(ctx.db, experiment_id=experiment_id)
    except (ImportError, AttributeError):
        return RecoverCheck(check="budget", status="unchecked", message="budget ledger not available")
    except HpcLabError as exc:
        return RecoverCheck(check="budget", status="unchecked", message=exc.message)
    avail = getattr(summ, "available", None)
    if avail is None:
        return RecoverCheck(check="budget", status="pass", message="no hpclab budget set for the experiment")
    if gpu_hours > float(avail):
        return RecoverCheck(
            check="budget",
            status="warn",
            message=f"needs {gpu_hours:.2f} GPU-h but only {float(avail):.2f} are available: admission will refuse",
        )
    return RecoverCheck(
        check="budget", status="pass", message=f"{gpu_hours:.2f} of {float(avail):.2f} GPU-h available"
    )


def _recover_single(
    ctx: ServiceContext,
    *,
    attempt_id: str,
    strategy: RecoverStrategy | None,
    checkpoint_ref: str | None,
    user_patch: dict[str, Any],
    key: str,
    payload_hash: str,
) -> RecoverData:
    a = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    if a is None:
        raise NotFound(f"attempt {attempt_id} not found", details={"attempt_id": attempt_id})
    run = ctx.db.get("runs", "run_id", a["run_id"])
    coverage = Coverage()
    coverage.add("ledger", "ok")
    now_iso = _now(ctx)
    cp = _plan_attempt(
        ctx,
        a,
        run,
        requested=strategy,
        checkpoint_ref=checkpoint_ref,
        user_patch=user_patch,
        fetch_logs=True,
        allow_remote=True,
        coverage=coverage,
        now_iso=now_iso,
    )
    if cp.eligible and cp.plan is not None:
        cp.checks.append(_budget_check(ctx, run["experiment_id"], cp.plan.resources.requested_gpu_hours))
    data = _single_data(cp, coverage)
    if not (cp.eligible or cp.probe_request is not None):
        return data  # nothing written: a repeat with the same key re-evaluates
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, key, payload_hash)
        if again is not None:
            return RecoverData.model_validate(again)
        _check_unchanged(tx, [a])
        if cp.probe_request is not None:
            probe_plan_id, hashes = _insert_probe_runs(
                tx, experiment_id=run["experiment_id"], requests=[cp.probe_request]
            )
            pr = cp.probe_request
            data.probe = _probe_info(None, (pr.run_id, probe_plan_id, hashes[pr.run_id]))
            data.next_step = (
                call(
                    "run_submit",
                    run_id=pr.run_id,
                    execution_plan_hash=hashes[pr.run_id],
                    idempotency_key=NEW_KEY,
                )
                + "; when it completes, "
                + call("run_recover", attempt_id=attempt_id, strategy="resume", idempotency_key=NEW_KEY)
            )
            tx.event(
                "attempt",
                attempt_id,
                "recovery.probe_created",
                {
                    "probe_run_id": pr.run_id,
                    "probe_plan_id": probe_plan_id,
                    "manifest_digest": data.checkpoint.manifest_digest if data.checkpoint else None,
                },
            )
        else:
            assert cp.plan is not None
            h = _insert_exec_plan(tx, cp.plan, validation={"checks": [c.model_dump() for c in cp.checks]})
            plan_id = _insert_plan(
                tx,
                experiment_id=run["experiment_id"],
                kind="recovery",
                cells=[(run["run_id"], h)],
                snapshot_digest=_run_spec(run).code.digest,
                validation={
                    "schema": RECOVERY_PLAN_SCHEMA,
                    "status": "pass",
                    "source": {"attempt_id": attempt_id},
                    "cells": [_cell_record(cp)],
                },
            )
            data.plan_id = plan_id
            data.plan_ref = _plan_ref(plan_id)
            data.next_step = call(
                "run_submit", run_id=run["run_id"], execution_plan_hash=h, idempotency_key=NEW_KEY
            )
            tx.event(
                "attempt",
                attempt_id,
                "recovery.planned",
                {"strategy": cp.strategy, "plan_id": plan_id, "exec_plan_hash": h, "run_id": run["run_id"]},
            )
            _point_run_at(
                tx, run["run_id"], h, plan_id=plan_id, strategy=cp.strategy, from_attempt_id=attempt_id
            )
        tx.idem_store(ctx.principal, TOOL, key, payload_hash, data.model_dump(mode="json"))
    return data


def _cell_record(cp: _CellPlan) -> dict[str, Any]:
    return {
        "run_id": cp.run["run_id"],
        "label": cp.run["label"],
        "kind": cp.run["kind"],
        "from_attempt_id": cp.attempt["attempt_id"] if cp.attempt else None,
        "eligible": cp.eligible and not cp.blocking,
        "strategy": cp.cell_strategy,
        "exec_plan_hash": cp.exec_plan_hash if cp.eligible and not cp.blocking else None,
        "blocking_reason": "; ".join(cp.blocking) if cp.blocking else None,
        "diagnosis_code": cp.diagnosis.primary_code if cp.diagnosis else None,
        "probe_run_id": cp.probe_request.run_id
        if cp.probe_request
        else (cp.probe.run_id if cp.probe else None),
        "requested_gpu_hours": round(cp.plan.resources.requested_gpu_hours, 4)
        if cp.plan is not None
        else None,
    }


def _now(ctx: ServiceContext) -> str:
    out = iso(ctx.clock.now())
    assert out is not None
    return out


def _select_cells(
    ctx: ServiceContext, selector: RecoverySelector
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any] | None, str]], int]:
    """(workflow, selected (run, latest attempt, code), runs left out by ``missing_valid``
    because they only wait for their evaluation)."""
    wf = ctx.db.one("SELECT * FROM workflows WHERE workflow_id=?", (selector.workflow_id,))
    if wf is None:
        raise NotFound(f"workflow {selector.workflow_id} not found")
    rows = ctx.db.all(
        "SELECT r.*, c.cell_state AS wf_cell_state FROM workflow_cells c JOIN runs r ON r.run_id=c.run_id "
        "WHERE c.workflow_id=? ORDER BY c.ordinal",
        (selector.workflow_id,),
    )
    run_ids = [r["run_id"] for r in rows]
    statuses = cell_statuses(ctx.db, run_ids)
    wanted: set[str] | None = None
    diag_codes: set[str] | None = None
    if isinstance(selector.cells, RunIdsSelector):
        wanted = set(selector.cells.run_ids)
        unknown = sorted(wanted - set(run_ids))
        if unknown:
            raise InvalidArgument(
                f"run(s) not in workflow {selector.workflow_id}: {', '.join(unknown[:10])}",
                details={"unknown_run_ids": unknown[:50]},
            )
    elif isinstance(selector.cells, DiagnosisSelector):
        diag_codes = set(selector.cells.diagnosis_in)
    out: list[tuple[dict[str, Any], dict[str, Any] | None, str]] = []
    awaiting = 0
    for r in rows:
        st = statuses[r["run_id"]]
        latest = (
            ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (st.latest_attempt_id,))
            if st.latest_attempt_id
            else None
        )
        if wanted is not None:
            if r["run_id"] not in wanted:
                continue
        elif diag_codes is not None:
            d = jload(latest.get("diagnosis_json"), None) if latest else None
            code = d.get("primary_code") if isinstance(d, dict) else None
            if code is None and latest is not None and latest["phase"] == AttemptPhase.TERMINAL.value:
                code = st.reason if st.code == "F" else None
            if code not in diag_codes:
                continue
        elif st.code == "V":
            continue
        elif st.code == "E":
            awaiting += 1  # completed, evaluation pending: not missing, not a failure
            continue
        out.append((r, latest, st.code))
    return wf, out, awaiting


def _recover_batch(
    ctx: ServiceContext,
    *,
    selector: RecoverySelector,
    strategy: RecoverStrategy | None,
    user_patch: dict[str, Any],
    key: str,
    payload_hash: str,
) -> RecoverData:
    wf, targets, awaiting = _select_cells(ctx, selector)
    coverage = Coverage()
    coverage.add("ledger", "ok")
    now_iso = _now(ctx)
    plans: list[_CellPlan] = []
    remote_budget = MAX_BATCH_RESUME
    for run, latest, code in targets:
        if latest is None:
            cp = _initial_cell(ctx, run, wf)
        elif latest["phase"] in _ACTIVE or latest["phase"] in _UNRESOLVED:
            cp = _CellPlan(run=run, attempt=latest)
            asm = assess(
                phase=latest["phase"],
                scheduler_state=latest["scheduler_state"],
                result_state=latest["result_state"],
                diagnosis=None,
                requested=strategy,
                resumable=False,
                has_checkpoint=False,
                has_hint_patch=False,
            )
            cp.blocking.extend(asm.blocking)
        else:
            try:
                cp = _plan_attempt(
                    ctx,
                    latest,
                    run,
                    requested=strategy,
                    checkpoint_ref=None,
                    user_patch=user_patch,
                    fetch_logs=False,
                    allow_remote=remote_budget > 0,
                    coverage=coverage,
                    now_iso=now_iso,
                )
            except HpcLabError as exc:
                # a malformed patch is the caller's error; auth must reach the user; anything else
                # is this cell's problem only
                if exc.code in (ErrorCode.AUTH_REQUIRED, ErrorCode.INVALID_ARGUMENT):
                    raise
                cp = _CellPlan(run=run, attempt=latest)
                cp.block(f"{exc.code}: {exc.message}")
            if cp.strategy == "resume" and cp.checkpoint is not None:
                remote_budget -= 1
        cp.code = code
        plans.append(cp)

    eligible = [cp for cp in plans if cp.eligible and not cp.blocking and cp.exec_plan_hash]
    probes = [cp.probe_request for cp in plans if cp.probe_request is not None]
    cells = [
        RecoverCell(
            run_id=cp.run["run_id"],
            label=cp.run["label"],
            kind=cp.run["kind"],
            attempt_id=cp.attempt["attempt_id"] if cp.attempt else None,
            code=cp.code,
            eligible=cp.eligible and not cp.blocking,
            strategy=cp.cell_strategy,
            exec_plan_hash=cp.exec_plan_hash if cp.eligible and not cp.blocking else None,
            blocking_reason="; ".join(cp.blocking) if cp.blocking else None,
            diagnosis_code=cp.diagnosis.primary_code if cp.diagnosis else None,
            probe_run_id=cp.probe_request.run_id
            if cp.probe_request
            else (cp.probe.run_id if cp.probe else None),
            requested_gpu_hours=round(cp.plan.resources.requested_gpu_hours, 4)
            if cp.plan is not None
            else None,
        )
        for cp in plans
    ]
    eligible_gh = [c.requested_gpu_hours for c in cells if c.eligible]
    # a sum over unknowns would understate the request (and pass the budget check): None instead
    total_gh: float | None = (
        sum(g for g in eligible_gh if g is not None) if all(g is not None for g in eligible_gh) else None
    )
    hints: list[RecoverHint] = []
    seen_hint: set[str] = set()
    for cp in plans:
        for h in cp.hints:
            if h.strategy not in seen_hint:
                seen_hint.add(h.strategy)
                hints.append(RecoverHint(strategy=h.strategy, message=h.message, tool=h.tool))
    warnings: list[str] = []
    if awaiting:
        warnings.append(
            f"{awaiting} run(s) are E (completed, evaluation pending) and were not selected: they need an "
            "evaluation, not a recovery - "
            + call("evaluation_collect", run_set_ref=wf["workflow_id"], idempotency_key=NEW_KEY)
        )
    for cp in plans:
        warnings.extend(f"{cp.run['label']}: {w}" for w in cp.warnings)
    data = RecoverData(
        mode="batch",
        eligible=bool(eligible),
        strategy=strategy,
        experiment_id=wf["experiment_id"],
        checks=(
            [
                _budget_check(ctx, wf["experiment_id"], total_gh)
                if total_gh is not None
                else RecoverCheck(
                    check="budget", status="unchecked", message="GPU-hours of some eligible cell are unknown"
                )
            ]
            if eligible
            else []
        ),
        blocking=[] if eligible else ["no selected cell is recoverable (see cells)"],
        hints=hints,
        requested_gpu_hours=round(total_gh, 4) if total_gh is not None else None,
        cells=cells,
        eligible_count=len(eligible),
        total=len(cells),
        source_workflow_id=wf["workflow_id"],
        coverage=coverage,
        warnings=warnings[:50],
    )
    if not eligible and not probes:
        return data
    by_attempt = [cp.attempt for cp in plans if cp.attempt is not None and (cp.eligible or cp.probe_request)]
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, key, payload_hash)
        if again is not None:
            return RecoverData.model_validate(again)
        _check_unchanged(tx, by_attempt)
        if probes:
            probe_plan_id, _hashes = _insert_probe_runs(
                tx, experiment_id=wf["experiment_id"], requests=probes
            )
            data.probe_plan_id = probe_plan_id
            data.probe_plan_ref = _plan_ref(probe_plan_id)
        if eligible:
            for cp in eligible:
                if cp.plan is not None and not cp.reuse_exec_plan:
                    _insert_exec_plan(tx, cp.plan, validation={"checks": [c.model_dump() for c in cp.checks]})
            plan_id = _insert_plan(
                tx,
                experiment_id=wf["experiment_id"],
                kind="recovery",
                cells=[(cp.run["run_id"], str(cp.exec_plan_hash)) for cp in eligible],
                snapshot_digest=None,
                validation={
                    "schema": RECOVERY_PLAN_SCHEMA,
                    "status": "pass",
                    "source": {
                        "workflow_id": wf["workflow_id"],
                        "selector": selector.model_dump(mode="json"),
                    },
                    "cells": [_cell_record(cp) for cp in plans],
                },
            )
            data.plan_id = plan_id
            data.plan_ref = _plan_ref(plan_id)
            tx.event(
                "workflow",
                wf["workflow_id"],
                "recovery.planned",
                {"plan_id": plan_id, "eligible": len(eligible), "total": len(cells)},
            )
            for cp in eligible:
                _point_run_at(
                    tx,
                    cp.run["run_id"],
                    str(cp.exec_plan_hash),
                    plan_id=plan_id,
                    strategy=cp.cell_strategy,
                    from_attempt_id=cp.attempt["attempt_id"] if cp.attempt else None,
                )
        steps = []
        if data.probe_plan_id:
            steps.append(
                call("workflow_submit", plan_id=data.probe_plan_id, idempotency_key=NEW_KEY)
                + " to run the checkpoint probes"
            )
        if data.plan_id:
            steps.append(
                call("workflow_submit", plan_id=data.plan_id, idempotency_key=NEW_KEY)
                + f" dispatches the {len(eligible)} eligible cell(s)"
            )
        data.next_step = "; ".join(steps) or None
        tx.idem_store(ctx.principal, TOOL, key, payload_hash, data.model_dump(mode="json"))
    return data


def outbox_handlers() -> dict[str, Any]:
    """run_recover is synchronous and never submits: no background steps."""
    return {}


def periodic_tasks() -> list[Any]:
    return []
