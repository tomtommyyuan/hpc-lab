"""Planned-set comparisons (tools §13 ``results_compare``) and their immutable archive.

The denominator is always the *planned* run set (experiment, plan or workflow; see
:func:`resolve_run_set`), never "the runs that happened to produce numbers":

* Every planned main run lands in exactly one bucket — ``included`` (coverage code ``V``),
  ``missing`` (``P S Q R U ? E``: no result *yet*; ``E`` = the job COMPLETED and its outputs
  are not known to be bad, but no valid evaluation exists yet — outputs being checked or
  evaluation pending — so it is missing with that reason, never failed), ``failed`` (``F X C!``;
  ``C!`` = COMPLETED with incomplete/invalid outputs or a partial/invalid evaluation of the
  latest attempt by the evaluator the comparison counts, whatever other evaluators report;
  ``X`` includes never-dispatched cells of a cancelled/skipped workflow) or ``excluded``
  (explicit, with a reason). Cell codes and "which evaluation counts" come from
  :mod:`hpclab.services.coverage` only, so a grid cell and a table denominator never disagree.
* A requested metric that no evaluation of the planned run set reports is refused (a typo is
  not a result); with no evaluation at all yet, every run is simply missing (with a warning).
* Per metric an included run can still be ``missing`` (the evaluation lacks the metric) or
  ``failed`` (non-finite value). ``n + |missing| + |failed| + |excluded| == expected_n`` always.
* Statistics are :mod:`hpclab.services.stats` (``n = 1`` → std/CI ``None``, never 0);
  differences vs the baseline are paired by seed or unpaired (Welch) per the contract.
* Comparability problems are reported, not hidden: invariant mismatches (split, evaluator,
  dataset, declared params) become ``incomparable`` entries; a declared selection rule that
  selects on the reported split is refused (``selection_leak``); post-hoc seeds, exclusions
  decided after the metric was observed (evaluated, completed, or its metrics output read — a
  live read included; per exclusion episode, in events order), environment heterogeneity, runs
  with several valid (COMPLETED) attempts, counted evaluations re-evaluated under another
  protocol, validly or not (``protocol_switches``), and included values from an older attempt
  while newer ones are running, awaiting evaluation, failed or diverged (``newer_attempts``, one
  entry per newer attempt) are disclosed in ``integrity`` (and ``warnings``); under
  ``latest_valid`` a value whose newer attempt may still yield a result is disclosed as
  provisional.
* Each distinct result is archived once (``comparisons`` row, ``content_hash``); re-running an
  unchanged comparison returns the newest archived one instead of a duplicate, and records the
  request as a ``comparison.reused`` event (an exclusion's history counts every request). Rows
  are never updated.
* :func:`comparison_request_key` names the *question* a comparison answers (run set, effective
  contract, exclusions, evaluation filter — not the time or the metric list), so a later
  comparison supersedes an earlier one only when it answers the same question.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from hpclab.api.envelope import Coverage, SourceStatus
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import (
    Ambiguous,
    InvalidArgument,
    NotFound,
    PolicyViolation,
    PreconditionFailed,
)
from hpclab.domain.hashing import digest
from hpclab.domain.ids import kind_of, new_id
from hpclab.domain.models import ComparisonContract
from hpclab.domain.states import AttemptPhase, SchedulerState
from hpclab.services import stats
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import COUNTED_ATTEMPT_WHERE, CellStatus, cell_statuses
from hpclab.storage.db import Tx, jdump, jload

log = logging.getLogger("hpclab.compare")

__all__ = [
    "ARCHIVE_SCHEMA",
    "UNCOUNTED_PREFIX",
    "ComparisonData",
    "ExclusionSpec",
    "MetricInfo",
    "RunSet",
    "RunSetCell",
    "ValueChain",
    "awaiting_evaluation",
    "comparison_get",
    "comparison_request_key",
    "load_archive",
    "outbox_handlers",
    "periodic_tasks",
    "request_key",
    "resolve_run_set",
    "results_compare",
    "seed_key",
    "strip_ref",
]

ARCHIVE_SCHEMA = "hpclab.comparison/v1"
MAX_METRICS = 20
MAX_RUNS_RETURNED = 256
MAX_LISTED = 50  # names listed in coverage.missing / details
MAX_REUSE_SCAN = 200  # archived comparisons of a run set searched for the newest same-request one
Scalar = str | int | float | bool | None
Seed = int | str
RunBucket = Literal["included", "missing", "failed", "excluded"]

_MISSING_CODES = frozenset({"P", "S", "Q", "R", "U", "?", "E"})
_FAILED_CODES = frozenset({"F", "X"})
# Reason of a run whose valid result this comparison does not count (other evaluator, or not
# among evaluation_refs): missing, but not waiting for anything.
UNCOUNTED_PREFIX = "a valid evaluation exists"
# The attempt "rule" of a comparison whose caller hand-picked the evaluations (evaluation_refs):
# the pin chose the attempt, not the contract's attempt_selection.
PINNED_RULE = "evaluation_refs"
_BUILTIN_INVARIANTS = frozenset({"split", "evaluator", "dataset"})


class Reader(Protocol):
    """Anything with the ``Database``/``Tx`` read helpers."""

    def one(self, sql: str, params: Any = ...) -> dict[str, Any] | None: ...

    def all(self, sql: str, params: Any = ...) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Refs and run sets
# ---------------------------------------------------------------------------

_URI_PREFIXES = {
    "experiment": "hpclab://experiments/",
    "plan": "hpclab://plans/",
    "workflow": "hpclab://workflows/",
    "comparison": "hpclab://comparisons/",
    "evaluation": "hpclab://evaluations/",
    "evidence": "hpclab://evidence/",
}


def strip_ref(ref: str) -> str:
    """``hpclab://<collection>/<id>[/...]`` or a bare id → the id."""
    text = (ref or "").strip()
    if text.startswith("hpclab://"):
        rest = text[len("hpclab://") :]
        parts = rest.split("/")
        if len(parts) < 2 or not parts[1]:
            raise InvalidArgument(f"malformed hpclab reference {ref!r}")
        return parts[1]
    if not text:
        raise InvalidArgument("empty reference")
    return text


def seed_key(seed: Seed | None) -> str:
    """Seeds are compared by string form everywhere (``"none"`` for seedless runs)."""
    return "none" if seed is None else str(seed)


def _seed_order(key: str) -> tuple[int, int, str]:
    try:
        return (0, int(key), "")
    except ValueError:
        return (1, 0, key)


@dataclass(frozen=True)
class RunSetCell:
    run_id: str
    label: str
    plan_id: str
    cell: dict[str, Any]
    seed: int | None
    treatment_key: dict[str, Any]
    added_post_hoc: bool
    spec_hash: str


@dataclass(frozen=True)
class RunSet:
    """The planned denominator: main runs only (smoke/probe runs never count)."""

    kind: Literal["experiment", "plan", "workflow"]
    id: str
    experiment_id: str
    cells: list[RunSetCell]
    notes: list[str] = field(default_factory=list)

    @property
    def run_ids(self) -> list[str]:
        return [c.run_id for c in self.cells]

    @property
    def ref(self) -> str:
        return _URI_PREFIXES[self.kind] + self.id


def _cell_from_row(row: dict[str, Any]) -> RunSetCell:
    cell = jload(row["cell_key"], {}) or {}
    if not isinstance(cell, dict):
        cell = {}
    seed = row.get("seed")
    return RunSetCell(
        run_id=row["run_id"],
        label=row["label"],
        plan_id=row["plan_id"],
        cell=cell,
        seed=int(seed) if seed is not None else None,
        treatment_key={k: v for k, v in cell.items() if k != "seed"},
        added_post_hoc=bool(row.get("added_post_hoc") or 0),
        spec_hash=row["spec_hash"],
    )


def resolve_run_set(reader: Reader, ref: str) -> RunSet:
    """Resolve an experiment / plan / workflow ref (id or ``hpclab://`` URI) to its planned main runs.

    * experiment: main runs of every plan of the experiment except ``derived`` plans (a derived
      Run changes the science; compare it through its own plan ref);
    * plan: the plan's main runs (for a plan without runs of its own, e.g. a recovery plan, the
      cells of the workflow that dispatched it);
    * workflow: the workflow's cells.
    """
    entity_id = strip_ref(ref)
    kind = kind_of(entity_id)
    notes: list[str] = []
    if kind == "experiment":
        if reader.one("SELECT 1 AS x FROM experiments WHERE experiment_id=?", (entity_id,)) is None:
            raise NotFound(f"experiment {entity_id} not found", details={"ref": ref})
        rows = reader.all(
            "SELECT r.*, p.kind AS plan_kind FROM runs r JOIN plans p ON p.plan_id = r.plan_id "
            "WHERE r.experiment_id=? AND r.kind='main' ORDER BY p.created_at, p.rowid, r.created_at, r.rowid",
            (entity_id,),
        )
        derived = [r for r in rows if r["plan_kind"] == "derived"]
        rows = [r for r in rows if r["plan_kind"] != "derived"]
        if derived:
            notes.append(
                f"{len(derived)} derived run(s) are not part of the experiment's planned set; "
                "compare them through their plan ref"
            )
        experiment_id = entity_id
        rs_kind: Literal["experiment", "plan", "workflow"] = "experiment"
    elif kind == "plan":
        plan = reader.one("SELECT plan_id, experiment_id FROM plans WHERE plan_id=?", (entity_id,))
        if plan is None:
            raise NotFound(f"plan {entity_id} not found", details={"ref": ref})
        rows = reader.all(
            "SELECT * FROM runs WHERE plan_id=? AND kind='main' ORDER BY created_at, rowid", (entity_id,)
        )
        if not rows:
            rows = reader.all(
                "SELECT r.* FROM workflows w JOIN workflow_cells c ON c.workflow_id = w.workflow_id "
                "JOIN runs r ON r.run_id = c.run_id WHERE w.plan_id=? AND r.kind='main' ORDER BY c.ordinal",
                (entity_id,),
            )
        experiment_id = plan["experiment_id"]
        rs_kind = "plan"
    elif kind == "workflow":
        wf = reader.one("SELECT workflow_id, experiment_id FROM workflows WHERE workflow_id=?", (entity_id,))
        if wf is None:
            raise NotFound(f"workflow {entity_id} not found", details={"ref": ref})
        rows = reader.all(
            "SELECT r.* FROM workflow_cells c JOIN runs r ON r.run_id = c.run_id "
            "WHERE c.workflow_id=? AND r.kind='main' ORDER BY c.ordinal",
            (entity_id,),
        )
        experiment_id = wf["experiment_id"]
        rs_kind = "workflow"
    else:
        raise InvalidArgument(
            f"planned_run_set_ref must be an experiment, plan or workflow id (or hpclab:// URI), got {ref!r}"
        )
    if not rows:
        raise PreconditionFailed(
            f"{rs_kind} {entity_id} has no planned main runs to compare", details={"ref": ref}
        )
    return RunSet(
        kind=rs_kind,
        id=entity_id,
        experiment_id=experiment_id,
        cells=[_cell_from_row(r) for r in rows],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExclusionSpec(BaseModel):
    """Remove one planned run from the statistics. The reason is mandatory and disclosed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason(cls, v: str) -> str:
        text = " ".join(str(v).split())
        if not text:
            raise ValueError("an exclusion needs a non-empty reason")
        if len(text) > 500:
            raise ValueError("exclusion reasons are limited to 500 characters")
        return text


class ValueChain(_Model):
    """Provenance of one run's counted value: run → attempt → job → evaluation → artifact → env."""

    run_id: str
    run_spec_hash: str | None = None
    plan_id: str | None = None
    code_digest: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    attempt_id: str | None = None
    attempt_seq: int | None = None
    exec_plan_hash: str | None = None
    cluster_id: str | None = None
    job_id: str | None = None
    job_owner: str | None = None
    job_submit_time: str | None = None
    scheduler_state: str | None = None
    evaluation_id: str | None = None
    evaluator: str | None = None
    config_hash: str | None = None
    evaluation_created_at: str | None = None
    artifact_id: str | None = None
    artifact_digest: str | None = None
    artifact_link: Literal["declared", "metric_source", "inferred"] | None = None
    env_gpu: str | None = None
    env_torch: str | None = None
    gpu_type: str | None = None


class RunSetInfo(_Model):
    kind: Literal["experiment", "plan", "workflow"]
    id: str
    ref: str
    experiment_id: str
    run_count: int


class ExperimentInfo(_Model):
    id: str
    name: str | None = None
    question: str | None = None


class RunCoverage(_Model):
    run_id: str
    label: str
    group_label: str
    seed: int | None
    code: str
    status: RunBucket
    reason: str | None = None
    attempts: int = 0
    latest_attempt_id: str | None = None
    evaluation_id: str | None = None
    evaluation_attempt_id: str | None = None
    values: dict[str, float | None] = Field(default_factory=dict)
    added_post_hoc: bool = False
    exclusion_reason: str | None = None


class MetricInfo(_Model):
    """Unit and direction of a compared metric: declared by the contract, else agreed by every
    included evaluation, else unknown (never guessed)."""

    name: str
    unit: str | None = None
    higher_is_better: bool | None = None
    source: Literal["contract", "evaluations", "unknown"] = "unknown"


class ExcludedSeed(_Model):
    seed: Seed
    reason: str


class MetricStats(_Model):
    """One group × metric cell (field names match the ``hpclab.evidence/v1`` bundle cell)."""

    n: int
    expected_n: int
    mean: float | None = None
    std: float | None = None
    sem: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    min: float | None = None
    max: float | None = None
    values_by_seed: dict[str, float] = Field(default_factory=dict)
    missing_seeds: list[Seed] = Field(default_factory=list)
    failed_seeds: list[Seed] = Field(default_factory=list)
    excluded: list[ExcludedSeed] = Field(default_factory=list)
    run_by_seed: dict[str, str] = Field(default_factory=dict)


class GroupResult(_Model):
    label: str
    key: dict[str, Scalar]
    is_baseline: bool = False
    run_ids: list[str] = Field(default_factory=list)
    seeds: list[Seed] = Field(default_factory=list)
    metrics: dict[str, MetricStats] = Field(default_factory=dict)


class DifferenceResult(_Model):
    group_label: str
    baseline_label: str
    metric: str
    kind: Literal["paired", "unpaired"]
    n: int
    n_baseline: int | None = None
    mean_diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    std_diff: float | None = None
    se_diff: float | None = None
    paired_seeds: list[Seed] = Field(default_factory=list)
    unpaired_group: list[Seed] = Field(default_factory=list)
    unpaired_baseline: list[Seed] = Field(default_factory=list)


class IncomparableEntry(_Model):
    reason: str
    invariant: str | None = None
    metric: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PostHocRun(_Model):
    run_id: str
    group_label: str
    seed: int | None
    status: RunBucket


class ExclusionRecord(_Model):
    """One exclusion and when it was decided relative to the run's result.

    ``first_declared_at`` is when the exclusion *in effect* was declared: the first request of
    the run's current exclusion episode — the comparison requests (archived or answered from the
    archive) that excluded it since the last request that included it, in the events table's
    order (see :func:`_exclusion_records`) — or this comparison's time when nothing earlier still
    stands. ``note`` says why an earlier exclusion of the run does not count as the declaration
    (revoked by a later inclusion, or a blanket exclusion).
    """

    run_id: str
    group_label: str
    seed: int | None
    reason: str
    code: str
    observed_before_exclusion: bool
    first_declared_at: str | None = None
    first_observed_at: str | None = None
    note: str | None = None


class HeterogeneityEntry(_Model):
    dimension: Literal["gpu", "torch", "code_digest"]
    values: dict[str, int] = Field(default_factory=dict)
    unknown: int = 0
    mixed: bool = False
    within_groups: list[str] = Field(default_factory=list)


class MultiValidAttempt(_Model):
    """An included run with several valid attempts (COMPLETED, finalized outputs, a valid
    evaluation). ``rule`` is what chose the reported one: the contract's ``attempt_selection``,
    or ``evaluation_refs`` when the caller hand-picked the evaluations."""

    run_id: str
    group_label: str
    seed: int | None
    valid_attempts: int
    selected_attempt_id: str | None
    rule: str


NewerOutcome = Literal["active", "awaiting_evaluation", "failed", "cancelled", "bad_result"]


class NewerAttempt(_Model):
    """An included run whose value comes from an older attempt while a newer attempt of the run
    has no counted valid result (one entry per such attempt); ``disclosure`` reads "value from
    attempt #k; newer attempt #m is <newer_state>". Listed under every selection rule. ``provisional``: under
    ``latest_valid`` (evaluations not pinned) the reported value changes if that attempt still
    yields a valid result (it is active or awaits evaluation); under ``first_valid`` or pinned
    ``evaluation_refs`` it cannot, but the re-run of the same configuration is disclosed all the
    same. (Archives written before the fields were renamed carry ``counted_attempt_id`` /
    ``latest_attempt_id``; they still load.)"""

    run_id: str
    group_label: str
    seed: int | None
    value_attempt_id: str | None = Field(
        validation_alias=AliasChoices("value_attempt_id", "counted_attempt_id")
    )
    value_attempt_seq: int | None = None
    newer_attempt_id: str = Field(validation_alias=AliasChoices("newer_attempt_id", "latest_attempt_id"))
    newer_attempt_seq: int | None = None
    newer_state: str = ""
    outcome: NewerOutcome
    detail: str
    provisional: bool
    disclosure: str = ""


class EvaluationProtocol(_Model):
    """One evaluation of an attempt: its protocol, its verdict (``status``) and the compared
    metrics it reports."""

    evaluation_id: str
    evaluator: str  # evaluator_id@version
    config_hash: str | None = None
    created_at: str | None = None
    status: str = "valid"
    values: dict[str, float | None] = Field(default_factory=dict)


class ProtocolSwitch(_Model):
    """An included value whose attempt was also evaluated under another protocol (evaluator id,
    version or config hash) that this comparison could have counted. ``counted`` is the reported
    evaluation (the newest valid one, or the pinned one); ``others`` are the evaluations of the
    same attempt under a different protocol, oldest first — valid ones, and ones that judged it
    partial/invalid (``status``; e.g. a new evaluator version found the value diverged: only
    valid evaluations count, so the older valid value is still the reported one)."""

    run_id: str
    group_label: str
    seed: int | None
    attempt_id: str | None
    counted: EvaluationProtocol
    others: list[EvaluationProtocol] = Field(default_factory=list)


class SelectionCheck(_Model):
    rule: str
    select_metric: str | None = None
    select_split: str | None = None
    report_split: str | None = None
    evaluation_selection_on_split: list[str] = Field(default_factory=list)  # run ids


class BelowMinN(_Model):
    group_label: str
    metric: str
    n: int
    min_n: int


class IntegrityData(_Model):
    attempt_selection: str
    post_hoc: list[PostHocRun] = Field(default_factory=list)
    exclusions: list[ExclusionRecord] = Field(default_factory=list)
    heterogeneity: list[HeterogeneityEntry] = Field(default_factory=list)
    multiple_valid_attempts: list[MultiValidAttempt] = Field(default_factory=list)
    selection: SelectionCheck
    below_min_n: list[BelowMinN] = Field(default_factory=list)
    newer_attempts: list[NewerAttempt] = Field(default_factory=list)
    protocol_switches: list[ProtocolSwitch] = Field(default_factory=list)


class CoverageCounts(_Model):
    planned_runs: int
    included: int
    missing: int
    failed: int
    excluded: int
    by_code: dict[str, int] = Field(default_factory=dict)


class ComparisonData(_Model):
    """An archived, immutable comparison (also the payload of ``hpclab://comparisons/<id>``)."""

    comparison_id: str
    comparison_ref: str
    created_at: str
    content_hash: str
    reused: bool = False
    run_set: RunSetInfo
    experiment: ExperimentInfo
    contract: ComparisonContract
    metric_names: list[str]
    metric_info: list[MetricInfo] = Field(default_factory=list)
    expected_seeds: list[Seed]
    treatment_axes: list[str]
    evaluation_refs: list[str] | None = None
    groups: list[GroupResult]
    differences: list[DifferenceResult]
    counts: CoverageCounts
    runs: list[RunCoverage]
    runs_truncated: int = 0
    incomparable: list[IncomparableEntry]
    integrity: IntegrityData
    summary: str
    coverage: Coverage
    warnings: list[str] = Field(default_factory=list)
    observed_at: str | None = None


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def results_compare(
    ctx: ServiceContext,
    *,
    planned_run_set_ref: str,
    metric_names: list[str],
    expected_seeds: list[int] | None = None,
    comparison_contract: ComparisonContract | None = None,
    evaluation_refs: list[str] | None = None,
    exclusions: list[ExclusionSpec] | None = None,
    idempotency_key: str | None = None,
) -> ComparisonData:
    """Compare groups over the PLANNED run set and archive the result immutably.

    Every planned run is included, missing, failed or excluded (with its reason) — nothing
    leaves a denominator silently. Raises ``PreconditionFailed`` when ``expected_seeds`` do not
    match the plan, ``PolicyViolation`` (``details.reason='selection_leak'``) when the
    contract selects on the split it reports, and ``InvalidArgument`` (``details.known_metrics``)
    when a requested metric is reported by none of the run set's evaluations (nothing is
    archived then; with no evaluation at all yet the comparison proceeds, every run missing,
    with a warning). Re-running an unchanged comparison returns the
    newest archived comparison (``reused=True``); ``idempotency_key`` additionally pins the
    response to the key.
    """
    names = _check_metric_names(metric_names)
    contract = _coerce_contract(comparison_contract)
    excl = _coerce_exclusions(exclusions)
    eval_ids = _coerce_evaluation_refs(evaluation_refs)
    request = {
        "planned_run_set_ref": planned_run_set_ref,
        "metric_names": names,
        "expected_seeds": expected_seeds,
        "comparison_contract": contract.model_dump(mode="json") if contract else None,
        "evaluation_refs": sorted(eval_ids) if eval_ids is not None else None,
        "exclusions": [{"run_id": e.run_id, "reason": e.reason} for e in excl],
    }
    payload_hash = digest(request)
    with ctx.db.tx(source="api") as tx:
        if idempotency_key:
            stored = tx.idem_lookup(ctx.principal, "results_compare", idempotency_key, payload_hash)
            if stored is not None:
                return ComparisonData.model_validate(stored)
        now = parse_iso(tx.now)  # the same instant the archive rows and events record
        assert now is not None
        computed = compute_comparison(
            tx,
            planned_run_set_ref=planned_run_set_ref,
            metric_names=names,
            expected_seeds=expected_seeds,
            contract=contract,
            evaluation_ids=eval_ids,
            exclusions=excl,
            now=now,
            require_known_metrics=True,
        )
        data = _archive(tx, computed, request=request)
        if idempotency_key:
            tx.idem_store(
                ctx.principal,
                "results_compare",
                idempotency_key,
                payload_hash,
                _cap_runs(data).model_dump(mode="json"),
            )
    return _cap_runs(data)


def comparison_get(ctx: ServiceContext, *, comparison_id: str) -> ComparisonData:
    """An archived comparison (``hpclab://comparisons/<id>``); runs listing capped at 256."""
    data, _chains = load_archive(ctx.db, comparison_id)
    return _cap_runs(data)


def load_archive(reader: Reader, comparison_ref: str) -> tuple[ComparisonData, dict[str, ValueChain]]:
    """Full archived comparison (all runs) plus the per-run provenance chains."""
    cid = strip_ref(comparison_ref)
    if kind_of(cid) != "comparison":
        raise InvalidArgument(f"not a comparison id: {comparison_ref!r}")
    row = reader.one("SELECT * FROM comparisons WHERE comparison_id=?", (cid,))
    if row is None:
        raise NotFound(f"comparison {cid} not found", details={"comparison_id": cid})
    archive = jload(row["result_json"], {}) or {}
    if archive.get("schema") != ARCHIVE_SCHEMA:
        raise PreconditionFailed(
            f"comparison {cid} has an unsupported archive format", details={"schema": archive.get("schema")}
        )
    data = ComparisonData.model_validate(archive["data"])
    chains = {rid: ValueChain.model_validate(c) for rid, c in (archive.get("chains") or {}).items()}
    return data, chains


def archive_content_hash(archive: Mapping[str, Any]) -> str:
    """Hash of an archive's content without its identity/time fields (used to verify it)."""
    data = dict(archive.get("data") or {})
    for volatile in (
        "comparison_id",
        "comparison_ref",
        "created_at",
        "content_hash",
        "reused",
        "observed_at",
    ):
        data.pop(volatile, None)
    return digest({"data": data, "chains": archive.get("chains") or {}})


def request_key(
    run_set_ref: str,
    contract: ComparisonContract,
    exclusions: Iterable[ExclusionSpec | Mapping[str, Any]],
    evaluation_refs: Iterable[str] | None,
) -> str:
    """Digest of the question a comparison answers (``sha256:...``).

    Two comparisons answer the same question when they share the run set, the effective
    contract's statistical terms (treatment, invariants, baseline, selection, attempt_selection,
    pair_by_seed, ci_level, min_n), the set of excluded runs and the ``evaluation_refs`` filter.
    Not part of it: the time, the metric list (matched per metric by the caller), the contract's
    metric display declarations, ``expected_seeds`` (an assertion that never changes a number)
    and the wording of exclusion reasons (disclosure text, not a different denominator).
    """
    excluded = sorted(
        {str(e.run_id if isinstance(e, ExclusionSpec) else e.get("run_id")) for e in exclusions}
    )
    return digest(
        {
            "run_set_ref": run_set_ref,
            "contract": {
                "treatment": sorted(contract.treatment),
                "invariants": sorted(set(contract.invariants)),
                "baseline": dict(contract.baseline) if contract.baseline else None,
                "selection": contract.selection.model_dump(mode="json"),
                "attempt_selection": contract.attempt_selection,
                "pair_by_seed": contract.pair_by_seed,
                "ci_level": contract.ci_level,
                "min_n": contract.min_n,
            },
            "exclusions": excluded,
            "evaluation_refs": sorted({strip_ref(r) for r in evaluation_refs})
            if evaluation_refs is not None
            else None,
        }
    )


def comparison_request_key(row: Mapping[str, Any]) -> str | None:
    """:func:`request_key` of an archived ``comparisons`` row, derived from its stored request
    and effective contract (``contract_json``: a request that left the contract to the
    experiment spec and one that spelled the same contract out ask the same question).
    None when the row cannot be read (it then matches nothing)."""
    request = jload(row.get("request_json"), {}) or {}
    raw_contract = jload(row.get("contract_json"), None)
    if not isinstance(request, dict):
        return None
    if not isinstance(raw_contract, dict):
        raw_contract = request.get("comparison_contract") or {}
    try:
        contract = ComparisonContract.model_validate(raw_contract)
        exclusions = [e for e in request.get("exclusions") or [] if isinstance(e, dict)]
        refs = request.get("evaluation_refs")
        return request_key(
            str(row["run_set_ref"]),
            contract,
            exclusions,
            [str(r) for r in refs] if isinstance(refs, list) else None,
        )
    except (ValidationError, KeyError, TypeError, InvalidArgument):
        return None


def outbox_handlers() -> dict[str, Any]:
    """Comparisons are synchronous, bounded ledger computations: no background work."""
    return {}


def periodic_tasks() -> list[Any]:
    return []


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def _check_metric_names(metric_names: Sequence[str]) -> list[str]:
    if not metric_names:
        raise InvalidArgument("metric_names must name at least one metric")
    names = [str(m).strip() for m in metric_names]
    if any(not m or len(m) > 128 for m in names):
        raise InvalidArgument("metric names must be 1..128 characters")
    if len(set(names)) != len(names):
        raise InvalidArgument("metric_names contains duplicates", details={"metric_names": names})
    if len(names) > MAX_METRICS:
        raise InvalidArgument(f"at most {MAX_METRICS} metrics per comparison")
    return names


def _coerce_contract(contract: ComparisonContract | Mapping[str, Any] | None) -> ComparisonContract | None:
    if contract is None or isinstance(contract, ComparisonContract):
        return contract
    try:
        return ComparisonContract.model_validate(dict(contract))
    except ValidationError as exc:
        raise InvalidArgument("invalid comparison_contract", details={"errors": _errors(exc)}) from None


def _coerce_exclusions(exclusions: Sequence[ExclusionSpec | Mapping[str, Any]] | None) -> list[ExclusionSpec]:
    out: list[ExclusionSpec] = []
    for e in exclusions or []:
        if isinstance(e, ExclusionSpec):
            out.append(e)
            continue
        try:
            out.append(ExclusionSpec.model_validate(dict(e)))
        except (ValidationError, TypeError, ValueError) as exc:
            details = {"errors": _errors(exc)} if isinstance(exc, ValidationError) else {}
            raise InvalidArgument(
                "every exclusion needs {run_id, reason} with a non-empty reason", details=details
            ) from None
    seen = [e.run_id for e in out]
    dupes = sorted({r for r in seen if seen.count(r) > 1})
    if dupes:
        raise InvalidArgument("a run is excluded more than once", details={"run_ids": dupes})
    return out


def _coerce_evaluation_refs(refs: Sequence[str] | None) -> set[str] | None:
    if refs is None:
        return None
    ids = {strip_ref(r) for r in refs}
    bad = sorted(i for i in ids if kind_of(i) != "evaluation")
    if bad:
        raise InvalidArgument("evaluation_refs must be evaluation ids", details={"invalid": bad[:MAX_LISTED]})
    if not ids:
        raise InvalidArgument("evaluation_refs is empty; omit it to use every valid evaluation")
    return ids


def _errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in exc.errors()[:20]]


def _default_contract(spec: Mapping[str, Any]) -> ComparisonContract:
    raw = spec.get("comparison")
    if raw is None:
        return ComparisonContract()
    try:
        return ComparisonContract.model_validate(raw)
    except ValidationError as exc:
        raise InvalidArgument(
            "the experiment spec's comparison contract is invalid; pass comparison_contract explicitly",
            details={"errors": _errors(exc)},
        ) from None


def _spec_evaluator(spec: Mapping[str, Any]) -> str | None:
    ev = spec.get("evaluation")
    if isinstance(ev, Mapping) and isinstance(ev.get("evaluator_id"), str):
        return str(ev["evaluator_id"])
    return None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


@dataclass
class _Computed:
    data: ComparisonData  # ids/time fields are placeholders until archived
    chains: dict[str, ValueChain]
    run_set: RunSet
    exclusions: list[ExclusionRecord]


@dataclass
class _RunFacts:
    cell: RunSetCell
    status: CellStatus
    bucket: RunBucket
    reason: str | None
    group_label: str
    exclusion_reason: str | None = None
    uncounted: bool = False  # a valid result exists that this comparison does not count
    evaluation: dict[str, Any] | None = None
    metric_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempt: dict[str, Any] | None = None
    chain: ValueChain | None = None
    spec: dict[str, Any] = field(default_factory=dict)


def compute_comparison(
    reader: Reader,
    *,
    planned_run_set_ref: str,
    metric_names: list[str],
    expected_seeds: Sequence[int] | None,
    contract: ComparisonContract | None,
    evaluation_ids: set[str] | None,
    exclusions: Sequence[ExclusionSpec],
    now: datetime,
    require_known_metrics: bool = False,
) -> _Computed:
    """Read-only computation of a comparison (``results_compare`` archives it).

    ``require_known_metrics`` (the request path) refuses metric names that no evaluation of the
    run set reports; a re-derivation of an archived request (``evidence_verify``) leaves it off.
    """
    rs = resolve_run_set(reader, planned_run_set_ref)
    exp = reader.one("SELECT * FROM experiments WHERE experiment_id=?", (rs.experiment_id,))
    if exp is None:  # pragma: no cover - foreign keys make this impossible
        raise NotFound(f"experiment {rs.experiment_id} not found")
    spec = jload(exp["spec_json"], {}) or {}
    if contract is None:
        contract = _default_contract(spec)
    _check_selection_rule(contract)
    warnings: list[str] = list(rs.notes)

    run_ids = rs.run_ids
    known = set(run_ids)
    unknown_excl = sorted(e.run_id for e in exclusions if e.run_id not in known)
    if unknown_excl:
        raise InvalidArgument(
            "exclusions name runs that are not in the planned run set",
            details={"run_ids": unknown_excl[:MAX_LISTED], "run_set": rs.ref},
        )
    excl_reason = {e.run_id: e.reason for e in exclusions}
    if evaluation_ids is not None:
        _check_evaluation_refs(reader, evaluation_ids, known)
    _check_metrics_reported(reader, rs, metric_names, warnings, strict=require_known_metrics)
    evaluator_filter = None if evaluation_ids is not None else _spec_evaluator(spec)

    statuses = cell_statuses(
        reader,
        run_ids,
        attempt_selection=contract.attempt_selection,
        evaluator_id=evaluator_filter,
        evaluation_ids=evaluation_ids,
    )
    # Unfiltered statuses tell "no valid result" (failed/missing) apart from "a valid result that
    # this comparison does not count" (missing, with the reason): a run evaluated by another
    # evaluator did not fail, and a paper footnote must not say it did.
    filtered = evaluation_ids is not None or evaluator_filter is not None
    unrestricted = (
        cell_statuses(reader, run_ids, attempt_selection=contract.attempt_selection) if filtered else statuses
    )
    uncounted_reason = (
        f"{UNCOUNTED_PREFIX} but is not among evaluation_refs"
        if evaluation_ids is not None
        else f"{UNCOUNTED_PREFIX}, but not from the experiment's evaluator {evaluator_filter}; "
        "collect one with that evaluator or pass evaluation_refs"
    )
    counted_by = (
        "a counted evaluation (evaluation_refs)"
        if evaluation_ids is not None
        else f"the experiment's evaluator {evaluator_filter}"
    )
    judged = _judged_bad(reader, statuses, unrestricted, evaluator_filter, evaluation_ids) if filtered else {}

    groups_order, labels, axes = _group_layout(rs.cells)
    declared = list(contract.treatment)
    treatment_axes = declared or axes
    undeclared = [a for a in axes if declared and a not in declared]
    if undeclared:
        warnings.append(
            f"matrix axes {undeclared} vary between groups but are not declared as treatment; "
            "groups are still split by every non-seed axis"
        )
    absent = [a for a in declared if a not in axes]
    if absent:
        warnings.append(f"declared treatment axes {absent} do not vary in this run set")

    _check_expected_seeds(rs, labels, groups_order, expected_seeds)
    facts = _run_facts(
        reader, rs, statuses, unrestricted, excl_reason, labels, uncounted_reason, judged, counted_by
    )
    pinned = evaluation_ids is not None
    newer = _newer_attempts(reader, facts, contract, evaluator_filter, evaluation_ids)
    warnings += _newer_attempt_warnings(newer, contract, pinned=pinned)
    switches = _protocol_switches(reader, facts, metric_names, evaluator_filter)
    warnings += _protocol_switch_warnings(switches, pinned=pinned)

    groups = _build_groups(groups_order, labels, facts, metric_names, contract)
    differences, is_baseline = _differences(groups, contract, warnings)
    for g in groups:
        g.is_baseline = g.label in is_baseline
    incomparable = _invariants(facts, metric_names, contract, warnings)
    exclusion_records = _exclusion_records(reader, facts, now)
    integrity = _integrity(
        reader,
        facts,
        groups,
        contract,
        exclusion_records,
        newer,
        switches,
        pinned=pinned,
        evaluator_filter=evaluator_filter,
    )
    if integrity.selection.evaluation_selection_on_split:
        incomparable.append(
            IncomparableEntry(
                reason="selection_leak: evaluations selected the best record on the reported split",
                invariant="selection",
                details={"run_ids": integrity.selection.evaluation_selection_on_split[:MAX_LISTED]},
            )
        )
    for item in integrity.below_min_n:
        warnings.append(
            f"{item.group_label}/{item.metric}: n={item.n} is below the contract's min_n={item.min_n}"
        )

    counts = _counts(facts)
    runs = [
        RunCoverage(
            run_id=f.cell.run_id,
            label=f.cell.label,
            group_label=f.group_label,
            seed=f.cell.seed,
            code=f.status.code,
            status=f.bucket,
            reason=f.reason,
            attempts=f.status.attempts,
            latest_attempt_id=f.status.latest_attempt_id,
            evaluation_id=f.status.evaluation_id,
            evaluation_attempt_id=f.status.evaluation_attempt_id,
            values={
                m: _finite_or_none(f.status.metrics.get(m)) for m in metric_names if m in f.status.metrics
            },
            added_post_hoc=f.cell.added_post_hoc,
            exclusion_reason=f.exclusion_reason,
        )
        for f in facts
    ]
    coverage = _coverage(facts, counts)
    if expected_seeds is not None:
        exp_seeds: list[Seed] = [int(s) for s in expected_seeds]
    else:
        exp_seeds = [_as_seed(k) for k in sorted({seed_key(c.seed) for c in rs.cells}, key=_seed_order)]
    summary = _summary(groups, metric_names, counts, incomparable, len(_awaiting_evaluation(facts)))
    data = ComparisonData(
        comparison_id="",
        comparison_ref="",
        created_at="",
        content_hash="",
        run_set=RunSetInfo(
            kind=rs.kind, id=rs.id, ref=rs.ref, experiment_id=rs.experiment_id, run_count=len(run_ids)
        ),
        experiment=ExperimentInfo(
            id=exp["experiment_id"], name=exp.get("name"), question=exp.get("question")
        ),
        contract=contract,
        metric_names=list(metric_names),
        metric_info=_metric_info(facts, metric_names, contract, warnings),
        expected_seeds=exp_seeds,
        treatment_axes=treatment_axes,
        evaluation_refs=sorted(evaluation_ids) if evaluation_ids is not None else None,
        groups=groups,
        differences=differences,
        counts=counts,
        runs=runs,
        incomparable=incomparable,
        integrity=integrity,
        summary=summary,
        coverage=coverage,
        warnings=warnings,
        observed_at=iso(now),
    )
    chains = {f.cell.run_id: f.chain for f in facts if f.chain is not None}
    return _Computed(data=data, chains=chains, run_set=rs, exclusions=exclusion_records)


def _check_selection_rule(contract: ComparisonContract) -> None:
    sel = contract.selection
    if sel.rule != "best_by":
        return
    same = sel.select_split is not None and sel.select_split == sel.report_split
    unknown = sel.select_split is None or sel.report_split is None
    if same or unknown:
        raise PolicyViolation(
            "selection_leak: the contract selects the best checkpoint on the split it reports"
            if same
            else "selection_leak: a best_by selection must declare select_split and report_split so the "
            "comparison can prove they differ",
            details={
                "reason": "selection_leak",
                "rule": sel.rule,
                "select_metric": sel.select_metric,
                "select_split": sel.select_split,
                "report_split": sel.report_split,
                "remediation": "select on a validation split and report a held-out split, or use rule "
                "'final'/'last'",
            },
        )


def _check_evaluation_refs(reader: Reader, ids: set[str], run_ids: set[str]) -> None:
    rows: list[dict[str, Any]] = []
    ordered = sorted(ids)
    for i in range(0, len(ordered), 500):
        chunk = ordered[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        rows += reader.all(
            f"SELECT evaluation_id, run_id FROM evaluations WHERE evaluation_id IN ({marks})", chunk
        )
    found = {r["evaluation_id"]: r["run_id"] for r in rows}
    missing = sorted(ids - set(found))
    if missing:
        raise NotFound("evaluation_refs not found", details={"evaluation_ids": missing[:MAX_LISTED]})
    foreign = sorted(e for e, r in found.items() if r not in run_ids)
    if foreign:
        raise InvalidArgument(
            "evaluation_refs belong to runs outside the planned run set",
            details={"evaluation_ids": foreign[:MAX_LISTED]},
        )


def _check_metrics_reported(
    reader: Reader, rs: RunSet, metric_names: Sequence[str], warnings: list[str], *, strict: bool
) -> None:
    """A metric is known when at least one evaluation of the run set (any status or evaluator)
    reports it. ``strict``: refuse unknown names while evaluations exist — a comparison of a
    metric nobody reports would archive a table of gaps for what is usually a typo."""
    n_evals = 0
    known: set[str] = set()
    statuses: dict[str, int] = {}
    for row in _chunked_in(
        reader, "SELECT status, metrics_json FROM evaluations WHERE run_id IN ({marks})", rs.run_ids
    ):
        n_evals += 1
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
        for m in jload(row["metrics_json"], []) or []:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                known.add(m["name"])
    if n_evals == 0:
        warnings.append(
            f"no evaluation exists yet for {rs.ref}: every run is missing and the metric names "
            f"{list(metric_names)} could not be checked against what the evaluations report"
        )
        return
    unknown = [m for m in metric_names if m not in known]
    if unknown and strict:
        listed = sorted(known)
        by_status = ", ".join(f"{s} ×{n}" for s, n in sorted(statuses.items()))
        raise InvalidArgument(
            f"metric(s) {unknown} are reported by none of the {n_evals} evaluation(s) of {rs.ref}; "
            f"known metrics: {', '.join(listed[:20])}{' ...' if len(listed) > 20 else ''}"
            if listed
            # nothing to tell a typo from: the evaluations themselves are the problem
            else f"none of the {n_evals} evaluation(s) of {rs.ref} reports any metric ({by_status}), so "
            f"{unknown} cannot be compared; check the evaluator and collect the evaluations again",
            details={
                "unknown_metrics": unknown,
                "known_metrics": listed[:100],
                "run_set": rs.ref,
                "evaluations": n_evals,
                "evaluation_statuses": dict(sorted(statuses.items())),
            },
        )


def _scalar_text(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return jdump(v)
    return str(v)


def _group_layout(cells: Sequence[RunSetCell]) -> tuple[list[str], dict[str, str], list[str]]:
    """Group keys in first-appearance order, a unique display label per key, and the axes."""
    order: list[str] = []
    keys: dict[str, dict[str, Any]] = {}
    axes: list[str] = []
    for c in cells:
        k = jdump(c.treatment_key)
        if k not in keys:
            keys[k] = c.treatment_key
            order.append(k)
        for a in c.treatment_key:
            if a not in axes:
                axes.append(a)
    varying = [a for a in axes if len({_scalar_text(keys[k].get(a)) for k in order}) > 1] or axes
    labels: dict[str, str] = {}
    short = len(varying) == 1
    for k in order:
        key = keys[k]
        if not key:
            labels[k] = "all"
        elif short:
            labels[k] = _scalar_text(key.get(varying[0]))
        else:
            labels[k] = ", ".join(f"{a}={_scalar_text(key[a])}" for a in varying if a in key)
    if len(set(labels.values())) != len(labels):  # e.g. True vs "True": fall back to full keys
        labels = {k: ", ".join(f"{a}={_scalar_text(v)}" for a, v in keys[k].items()) or "all" for k in order}
    if len(set(labels.values())) != len(labels):
        labels = {k: f"{labels[k]} #{i + 1}" for i, k in enumerate(order)}
    return order, labels, axes


def _judged_bad(
    reader: Reader,
    statuses: Mapping[str, CellStatus],
    unrestricted: Mapping[str, CellStatus],
    evaluator_filter: str | None,
    evaluation_ids: set[str] | None,
) -> dict[str, tuple[str, str]]:
    """Run id → (attempt id, status) of a counted (same evaluator / among ``evaluation_refs``)
    *non-valid* evaluation that keeps a run with an uncounted valid result (from another
    evaluation) from being reported as merely uncounted: the evaluator that counts already judged
    that result partial/invalid (e.g. a diverged seed with a NaN metric). :func:`_classify` makes
    the run failed, or missing while its latest attempt may still yield a result.

    The run's latest attempt is judged first (its ``C!`` cell: the counted evaluator's verdict on
    the latest result stands whatever other evaluators say about it or about older attempts),
    then the attempt whose valid result this comparison does not count. The newest verdict of
    an attempt wins."""
    order: dict[str, list[str]] = {}
    for rid, u in unrestricted.items():
        st = statuses[rid]
        if u.code != "V" or st.code == "V":
            continue
        attempts = [str(st.latest_attempt_id)] if st.code == "C!" and st.latest_attempt_id else []
        if u.evaluation_attempt_id and u.evaluation_attempt_id not in attempts:
            attempts.append(str(u.evaluation_attempt_id))
        if attempts:
            order[rid] = attempts
    if not order:
        return {}
    bad: dict[str, str] = {}
    for r in _chunked_in(
        reader,
        "SELECT evaluation_id, attempt_id, evaluator_id, status, created_at FROM evaluations "
        "WHERE attempt_id IN ({marks}) AND status != 'valid' ORDER BY created_at, evaluation_id",
        [a for attempts in order.values() for a in attempts],
    ):
        if evaluation_ids is not None and r["evaluation_id"] not in evaluation_ids:
            continue
        if evaluation_ids is None and evaluator_filter is not None and r["evaluator_id"] != evaluator_filter:
            continue
        bad[str(r["attempt_id"])] = str(r["status"])  # the newest one wins
    out: dict[str, tuple[str, str]] = {}
    for rid, attempts in order.items():
        judged = next((a for a in attempts if a in bad), None)
        if judged is not None:
            out[rid] = (judged, bad[judged])
    return out


def _classify(
    status: CellStatus,
    unrestricted: CellStatus,
    excluded: str | None,
    uncounted_reason: str,
    judged: tuple[str, str] | None = None,
    counted_by: str = "the counted evaluator",
) -> tuple[RunBucket, str | None]:
    """Bucket and reason of one planned run.

    ``E`` (COMPLETED, no valid evaluation yet) is *missing* with its reason — the result may
    still arrive; ``C!`` (COMPLETED with incomplete/invalid outputs or a partial/invalid
    evaluation of the latest attempt) is *failed*. A valid result that this comparison does not
    count (other evaluator, not among ``evaluation_refs``) is missing, never failed — unless the
    evaluation that counts judged a result partial/invalid (``judged`` = (attempt, status), see
    :func:`_judged_bad`): the run's latest attempt, or the attempt of the uncounted valid result.
    The run is then classified as without the other evaluation: failed when its latest attempt
    ended without a result (``C! F X``), missing with the verdict in the reason while the latest
    attempt may still yield one (running, awaiting evaluation). An included run keeps the
    coverage reason (e.g. the counted value comes from an older attempt than the latest).
    """
    if excluded is not None:
        return "excluded", excluded
    if status.code == "V":
        return "included", status.reason
    if unrestricted.code == "V" and judged is not None:
        attempt, verdict = judged
        other = unrestricted.evaluation_attempt_id
        if status.code == "C!" and status.latest_attempt_id == attempt:
            where = "" if other == attempt else f" (attempt {other})"
            detail = f"evaluation {verdict} from {counted_by}"
            if status.reason and not status.reason.startswith("evaluation "):
                detail = f"{status.reason}, {detail}"  # e.g. outputs incomplete
            return "failed", (
                f"completed but {detail}; another evaluation's valid result{where} is not counted"
            )
        verdict_text = (
            f"{status.reason or status.code}; {counted_by} judged the result of attempt {attempt} "
            f"{verdict} (another evaluation's valid result is not counted)"
        )
        # a newer attempt that is running or awaits evaluation may still yield a counted result
        return ("missing" if status.code in _MISSING_CODES else "failed"), verdict_text
    if unrestricted.code == "V":
        return "missing", uncounted_reason
    if status.code == "C!":
        return "failed", f"completed but {status.reason or 'result invalid'}"
    if status.code in _FAILED_CODES:
        return "failed", status.reason or status.code
    if status.code in _MISSING_CODES:
        return "missing", status.reason or status.code
    return "missing", status.reason or f"code {status.code}"  # pragma: no cover - defensive


def _chunked_in(
    reader: Reader, sql: str, ids: Sequence[str], *, after: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    """``sql`` contains one ``{marks}`` placeholder for an IN list; ``after`` are the parameters of
    any placeholders that follow it."""
    out: list[dict[str, Any]] = []
    uniq = sorted(set(ids))
    for i in range(0, len(uniq), 500):
        chunk = uniq[i : i + 500]
        out += reader.all(sql.format(marks=",".join("?" for _ in chunk)), [*chunk, *after])
    return out


def _metric_rows(metrics_json: str | None) -> dict[str, dict[str, Any]]:
    """First row per metric name — the same rule ``coverage`` uses for the value."""
    out: dict[str, dict[str, Any]] = {}
    for m in jload(metrics_json, []) or []:
        if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"] not in out:
            out[m["name"]] = m
    return out


def env_facts(env_json: str | None) -> tuple[str | None, str | None]:
    """(GPU model(s), torch version) recorded by the job wrapper's env.json; None when unknown."""
    env = jload(env_json, None)
    if not isinstance(env, dict):
        return None, None
    names: list[str] = []
    smi = env.get("nvidia_smi")
    if isinstance(smi, dict):
        names = [str(g["name"]) for g in smi.get("gpus") or [] if isinstance(g, dict) and g.get("name")]
    torch = env.get("torch")
    if not names and isinstance(torch, dict):
        names = [str(n) for n in torch.get("device_names") or [] if isinstance(n, str) and n]
    gpu = " + ".join(sorted(set(names))) if names else None
    version = torch.get("version") if isinstance(torch, dict) else None
    return gpu, (str(version) if version else None)


def _artifact_link(
    ev: dict[str, Any], metric_rows: Mapping[str, Mapping[str, Any]], artifacts: Sequence[dict[str, Any]]
) -> tuple[str | None, str | None, Literal["declared", "metric_source", "inferred"] | None]:
    by_id = {a["artifact_id"]: a for a in artifacts}
    for ref in jload(ev.get("input_refs_json"), []) or []:
        aid: str | None = None
        dig: str | None = None
        if isinstance(ref, str):
            aid = ref
        elif isinstance(ref, dict):
            raw = ref.get("artifact_id") or ref.get("id") or ref.get("ref") or ref.get("artifact_ref")
            aid = str(raw) if raw else None
            d = ref.get("digest") or ref.get("sha256")
            dig = str(d) if d else None
        if aid:
            aid = aid.split("#", 1)[0]
            if aid.startswith("hpclab://"):
                aid = strip_ref(aid)
            known = by_id.get(aid)
            return aid, dig or (known.get("digest") if known else None), "declared"
    sources = {
        str(m["source"]["digest"])
        for m in metric_rows.values()
        if isinstance(m.get("source"), dict) and m["source"].get("digest")
    }
    matches = [a for a in artifacts if a.get("digest") and a["digest"] in sources]
    if len(matches) == 1:
        return matches[0]["artifact_id"], matches[0]["digest"], "metric_source"
    metrics_role = [a for a in artifacts if a.get("role") == "metrics"]
    if len(metrics_role) == 1:
        return metrics_role[0]["artifact_id"], metrics_role[0].get("digest"), "inferred"
    return None, (next(iter(sources)) if len(sources) == 1 else None), None


def _run_facts(
    reader: Reader,
    rs: RunSet,
    statuses: Mapping[str, CellStatus],
    unrestricted: Mapping[str, CellStatus],
    excl_reason: Mapping[str, str],
    labels: Mapping[str, str],
    uncounted_reason: str,
    judged: Mapping[str, tuple[str, str]],
    counted_by: str,
) -> list[_RunFacts]:
    run_ids = rs.run_ids
    run_rows = {
        r["run_id"]: r for r in _chunked_in(reader, "SELECT * FROM runs WHERE run_id IN ({marks})", run_ids)
    }
    ev_ids = [s.evaluation_id for s in statuses.values() if s.evaluation_id]
    evals = {
        r["evaluation_id"]: r
        for r in _chunked_in(reader, "SELECT * FROM evaluations WHERE evaluation_id IN ({marks})", ev_ids)
    }
    attempt_ids = [s.evaluation_attempt_id or s.latest_attempt_id for s in statuses.values()]
    attempts = {
        r["attempt_id"]: r
        for r in _chunked_in(
            reader, "SELECT * FROM attempts WHERE attempt_id IN ({marks})", [a for a in attempt_ids if a]
        )
    }
    artifacts: dict[str, list[dict[str, Any]]] = {}
    for a in _chunked_in(
        reader,
        "SELECT * FROM artifacts WHERE attempt_id IN ({marks})",
        [s.evaluation_attempt_id for s in statuses.values() if s.evaluation_attempt_id],
    ):
        artifacts.setdefault(a["attempt_id"], []).append(a)
    out: list[_RunFacts] = []
    for cell in rs.cells:
        st = statuses[cell.run_id]
        bucket, reason = _classify(
            st,
            unrestricted[cell.run_id],
            excl_reason.get(cell.run_id),
            uncounted_reason,
            judged.get(cell.run_id),
            counted_by,
        )
        run_row = run_rows.get(cell.run_id, {})
        spec = jload(run_row.get("spec_json"), {}) or {}
        ev = evals.get(st.evaluation_id) if st.evaluation_id else None
        attempt_id = st.evaluation_attempt_id or st.latest_attempt_id
        attempt = attempts.get(attempt_id) if attempt_id else None
        facts = _RunFacts(
            cell=cell,
            status=st,
            bucket=bucket,
            reason=reason,
            group_label=labels[jdump(cell.treatment_key)],
            exclusion_reason=excl_reason.get(cell.run_id),
            # the same test as awaiting_evaluation(): a run waiting on its own attempt is not uncounted
            uncounted=bucket == "missing" and (reason or "").startswith(UNCOUNTED_PREFIX),
            evaluation=ev,
            metric_rows=_metric_rows(ev["metrics_json"]) if ev else {},
            attempt=attempt,
            spec=spec if isinstance(spec, dict) else {},
        )
        facts.chain = _chain(facts, run_row, artifacts.get(st.evaluation_attempt_id or "", []))
        out.append(facts)
    return out


_NO_RESULT_PHASES = frozenset(
    {AttemptPhase.SUBMISSION_REJECTED, AttemptPhase.CANCELLED_BEFORE_SUBMIT, AttemptPhase.ABANDONED}
)


def _newer_attempts(
    reader: Reader,
    facts: Sequence[_RunFacts],
    contract: ComparisonContract,
    evaluator_filter: str | None,
    evaluation_ids: set[str] | None,
) -> list[NewerAttempt]:
    """Included runs whose value comes from an older attempt, with what *every* newer attempt of
    the run did — so a table never shows an old number while hiding a re-run of it (a diverged
    re-run is not hidden behind a later running one either). One entry per newer attempt, oldest
    first, under every selection rule ("value from attempt #k; newer attempt #m is <state>").

    * ``active`` (not terminal yet) and ``awaiting_evaluation`` (COMPLETED, outputs not known to
      be bad, no evaluation from the counted evaluator yet — an ``E`` hidden behind the ``V``)
      make the value *provisional* under ``latest_valid``: it changes if that attempt yields a
      valid result. Under ``first_valid`` or pinned ``evaluation_refs`` the value cannot change;
      the newer attempt is still disclosed, not provisional.
    * ``failed`` / ``cancelled`` (terminal without a result) and ``bad_result`` (COMPLETED with
      incomplete/invalid outputs, or judged partial/invalid by the counted evaluator — e.g. a
      diverged re-run) never change the value but are always disclosed.

    A newer attempt with a valid result of its own is not listed (``multiple_valid_attempts``
    covers it).
    """
    candidates = [
        f
        for f in facts
        if f.bucket == "included"
        and f.status.latest_attempt_id
        and f.status.latest_attempt_id != f.status.evaluation_attempt_id
    ]
    if not candidates:
        return []
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in _chunked_in(
        reader,
        "SELECT attempt_id, run_id, seq, phase, scheduler_state, result_state, diagnosis_json FROM attempts "
        "WHERE run_id IN ({marks}) ORDER BY run_id, seq",
        [f.cell.run_id for f in candidates],
    ):
        by_run.setdefault(str(r["run_id"]), []).append(r)
    newer: dict[str, list[dict[str, Any]]] = {}
    for f in candidates:
        rows = by_run.get(f.cell.run_id, [])
        value_seq = _int_or_none(f.attempt.get("seq")) if f.attempt else None
        if value_seq is None:  # pragma: no cover - the counted attempt was read from the same ledger
            newer[f.cell.run_id] = [r for r in rows if r["attempt_id"] == f.status.latest_attempt_id]
        else:
            newer[f.cell.run_id] = [r for r in rows if (_int_or_none(r["seq"]) or 0) > value_seq]
    evals: dict[str, list[dict[str, Any]]] = {}
    for r in _chunked_in(
        reader,
        "SELECT evaluation_id, attempt_id, evaluator_id, status, created_at FROM evaluations "
        "WHERE attempt_id IN ({marks}) ORDER BY created_at, evaluation_id",
        [str(a["attempt_id"]) for rows in newer.values() for a in rows],
    ):
        if evaluation_ids is None and evaluator_filter is not None and r["evaluator_id"] != evaluator_filter:
            continue
        evals.setdefault(r["attempt_id"], []).append(r)
    may_change = contract.attempt_selection == "latest_valid" and evaluation_ids is None
    out: list[NewerAttempt] = []
    for f in candidates:
        value_seq = _int_or_none(f.attempt.get("seq")) if f.attempt else None
        value_name = f"#{value_seq}" if value_seq is not None else str(f.status.evaluation_attempt_id)
        for a in newer[f.cell.run_id]:
            judged = _newer_outcome(a, evals.get(a["attempt_id"], []))
            if judged is None:
                continue
            outcome, detail, state = judged
            newer_seq = _int_or_none(a.get("seq"))
            newer_name = f"#{newer_seq}" if newer_seq is not None else str(a["attempt_id"])
            out.append(
                NewerAttempt(
                    run_id=f.cell.run_id,
                    group_label=f.group_label,
                    seed=f.cell.seed,
                    value_attempt_id=f.status.evaluation_attempt_id,
                    value_attempt_seq=value_seq,
                    newer_attempt_id=str(a["attempt_id"]),
                    newer_attempt_seq=newer_seq,
                    newer_state=state,
                    outcome=outcome,
                    detail=detail,
                    provisional=may_change and outcome in ("active", "awaiting_evaluation"),
                    disclosure=f"value from attempt {value_name}; newer attempt {newer_name} is {state}",
                )
            )
    return out


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _newer_outcome(
    a: Mapping[str, Any], evals: Sequence[Mapping[str, Any]]
) -> tuple[NewerOutcome, str, str] | None:
    """What one (newer) attempt did — (outcome, detail, state) — or None when it has a valid result
    of its own. ``evals`` are its evaluations that would count (oldest first)."""
    aid, phase, sched = a["attempt_id"], a["phase"], a["scheduler_state"]
    if phase != AttemptPhase.TERMINAL:
        if phase in _NO_RESULT_PHASES:
            return "cancelled", f"attempt {aid} is {phase}", f"cancelled ({phase})"
        known = sched if sched and sched != SchedulerState.UNKNOWN else phase
        return (
            "active",
            f"attempt {aid} is {phase}" + (f" ({sched})" if sched else ""),
            f"in progress ({known})",
        )
    if sched == SchedulerState.COMPLETED:
        rs = a["result_state"]
        if rs in ("incomplete", "invalid"):
            return "bad_result", f"attempt {aid} COMPLETED with {rs} outputs", f"COMPLETED with {rs} outputs"
        if rs == "valid" and any(e["status"] == "valid" for e in evals):
            return None
        bad = [e for e in evals if e["status"] != "valid"]
        if bad:
            e = bad[-1]
            return (
                "bad_result",
                f"attempt {aid} COMPLETED but its evaluation is {e['status']} ({e['evaluator_id']})",
                f"COMPLETED with a {e['status']} evaluation ({e['evaluator_id']})",
            )
        if rs == "valid":
            return (
                "awaiting_evaluation",
                f"attempt {aid} COMPLETED and awaits evaluation",
                "COMPLETED, awaiting evaluation",
            )
        return (
            "awaiting_evaluation",
            f"attempt {aid} COMPLETED; outputs being checked (result_state={rs})",
            f"COMPLETED, outputs being checked (result_state={rs})",
        )
    if sched == SchedulerState.CANCELLED:
        return "cancelled", f"attempt {aid} ended CANCELLED", "cancelled"
    diag = jload(a.get("diagnosis_json"), None)
    code = diag.get("primary_code") if isinstance(diag, dict) else None
    return (
        "failed",
        f"attempt {aid} ended {sched or 'without a scheduler state'}" + (f" ({code})" if code else ""),
        f"failed ({sched or 'no scheduler state'}" + (f", {code})" if code else ")"),
    )


def _newer_attempt_warnings(
    entries: Sequence[NewerAttempt], contract: ComparisonContract, pinned: bool
) -> list[str]:
    def who(items: Sequence[NewerAttempt]) -> str:
        text = "; ".join(f"{e.group_label} seed {seed_key(e.seed)}: {e.disclosure}" for e in items[:20])
        return text + ("; ..." if len(items) > 20 else "")

    def runs(items: Sequence[NewerAttempt]) -> int:  # a run can have several newer attempts
        return len({e.run_id for e in items})

    chosen = (
        "pinned by evaluation_refs"
        if pinned
        else f"the one attempt_selection {contract.attempt_selection} selects"
    )
    out = []
    pending = [e for e in entries if e.provisional and e.outcome == "awaiting_evaluation"]
    if pending:
        out.append(
            f"{runs(pending)} included run(s) report an older attempt's value while a newer attempt "
            f"COMPLETED and awaits evaluation ({who(pending)}): with "
            "attempt_selection latest_valid these values change once it is evaluated, so they are "
            "provisional; run evaluation_collect, then compare again"
        )
    active = [e for e in entries if e.provisional and e.outcome == "active"]
    if active:
        out.append(
            f"{runs(active)} included run(s) report an older attempt's value while a newer attempt is "
            f"still in progress ({who(active)}): with attempt_selection latest_valid these values "
            "change if it yields a valid result, so they are provisional; compare again once it has "
            "finished and been evaluated"
        )
    steady = [e for e in entries if not e.provisional and e.outcome in ("active", "awaiting_evaluation")]
    if steady:
        out.append(
            f"{runs(steady)} included run(s) report an older attempt's value while a newer attempt of the "
            f"run has no result yet ({who(steady)}): the reported value is {chosen} and does not "
            "change with it, but the re-run of the same configuration is not reflected in the table"
        )
    ended = [e for e in entries if e.outcome in ("failed", "cancelled", "bad_result")]
    if ended:
        out.append(
            f"{runs(ended)} included run(s) report an older attempt's value although a newer attempt of "
            f"the run ended without a valid result ({who(ended)}): the reported value is {chosen}, "
            "but a re-run of the same configuration failed, was cancelled or produced a bad result"
        )
    return out


def _protocol_text(evaluator: str, config_hash: str | None, status: str = "valid") -> str:
    verdict = "" if status == "valid" else f" ({status})"
    return f"{evaluator} config {(config_hash or 'unknown')[:15]}{verdict}"


def _protocol_switches(
    reader: Reader, facts: Sequence[_RunFacts], metric_names: Sequence[str], evaluator_filter: str | None
) -> list[ProtocolSwitch]:
    """Included values whose counted evaluation's protocol (evaluator id, version, config hash)
    differs from another evaluation of the same attempt — the attempt was re-evaluated under
    another protocol and one of them was reported. A re-evaluation that was not valid is listed
    too (with its status): it cannot be counted, but its verdict on the reported value is not
    hidden.

    Only evaluations this comparison could have counted are compared: the experiment's evaluator
    when its spec declares one (another evaluator never counts, so it is no switch), any evaluator
    otherwise — also under ``evaluation_refs``, where the pin chose between the protocols.
    """
    counted = {
        str(f.status.evaluation_attempt_id): f
        for f in facts
        if f.bucket == "included" and f.evaluation is not None and f.status.evaluation_attempt_id
    }
    if not counted:
        return []

    def protocol(row: Mapping[str, Any]) -> tuple[str, str, str | None]:
        return (str(row["evaluator_id"]), str(row["evaluator_version"]), row.get("config_hash"))

    def entry(row: Mapping[str, Any]) -> EvaluationProtocol:
        metrics = _metric_rows(row.get("metrics_json"))
        return EvaluationProtocol(
            evaluation_id=str(row["evaluation_id"]),
            evaluator=f"{row['evaluator_id']}@{row['evaluator_version']}",
            config_hash=row.get("config_hash"),
            created_at=row.get("created_at"),
            status=str(row.get("status") or "valid"),
            values={m: _finite_or_none(metrics[m].get("value")) for m in metric_names if m in metrics},
        )

    others: dict[str, list[dict[str, Any]]] = {}
    for r in _chunked_in(
        reader,
        "SELECT evaluation_id, attempt_id, evaluator_id, evaluator_version, config_hash, created_at, "
        "status, metrics_json FROM evaluations WHERE attempt_id IN ({marks}) "
        "ORDER BY created_at, evaluation_id",
        list(counted),
    ):
        f = counted[str(r["attempt_id"])]
        ev = f.evaluation or {}
        if r["evaluation_id"] == ev.get("evaluation_id"):
            continue
        if evaluator_filter is not None and r["evaluator_id"] != evaluator_filter:
            continue
        if protocol(r) != protocol(ev):
            others.setdefault(str(r["attempt_id"]), []).append(r)
    out: list[ProtocolSwitch] = []
    for f in facts:
        aid = f.status.evaluation_attempt_id
        if f.bucket != "included" or f.evaluation is None or not aid or aid not in others:
            continue
        out.append(
            ProtocolSwitch(
                run_id=f.cell.run_id,
                group_label=f.group_label,
                seed=f.cell.seed,
                attempt_id=aid,
                counted=entry(f.evaluation),
                others=[entry(r) for r in others[aid]],
            )
        )
    return out


def _protocol_switch_warnings(switches: Sequence[ProtocolSwitch], pinned: bool) -> list[str]:
    if not switches:
        return []
    text = "; ".join(
        f"{s.group_label} seed {seed_key(s.seed)}: {_protocol_text(s.counted.evaluator, s.counted.config_hash)} "
        f"reported, not {', '.join(_protocol_text(o.evaluator, o.config_hash, o.status) for o in s.others[:3])}"
        + (f" and {len(s.others) - 3} more" if len(s.others) > 3 else "")
        for s in switches[:20]
    ) + ("; ..." if len(switches) > 20 else "")
    chosen = "the pinned evaluation (evaluation_refs)" if pinned else "the newest valid evaluation"
    return [
        f"{len(switches)} included value(s) come from an attempt that was evaluated under more than one "
        f"protocol (evaluator, version or config) ({text}): {chosen} is reported; every protocol's "
        "verdict and values are listed in integrity.protocol_switches"
    ]


def _chain(
    f: _RunFacts, run_row: Mapping[str, Any], artifacts: Sequence[dict[str, Any]]
) -> ValueChain | None:
    if f.attempt is None and f.evaluation is None:
        return None
    code = f.spec.get("code") if isinstance(f.spec.get("code"), dict) else {}
    a = f.attempt or {}
    gpu, torch = env_facts(a.get("env_json"))
    chain = ValueChain(
        run_id=f.cell.run_id,
        run_spec_hash=run_row.get("spec_hash") or f.cell.spec_hash,
        plan_id=f.cell.plan_id,
        code_digest=code.get("digest") if code else None,
        git_commit=code.get("git_commit") if code else None,
        git_dirty=code.get("git_dirty") if code else None,
        attempt_id=a.get("attempt_id"),
        attempt_seq=a.get("seq"),
        exec_plan_hash=a.get("exec_plan_hash"),
        cluster_id=a.get("cluster_id"),
        job_id=a.get("job_id"),
        job_owner=a.get("job_owner"),
        job_submit_time=a.get("job_submit_time"),
        scheduler_state=a.get("scheduler_state"),
        env_gpu=gpu,
        env_torch=torch,
        gpu_type=a.get("gpu_type"),
    )
    if f.evaluation is not None:
        ev = f.evaluation
        aid, adig, link = _artifact_link(ev, f.metric_rows, artifacts)
        chain = chain.model_copy(
            update={
                "evaluation_id": ev["evaluation_id"],
                "evaluator": f"{ev['evaluator_id']}@{ev['evaluator_version']}",
                "config_hash": ev.get("config_hash"),
                "evaluation_created_at": ev.get("created_at"),
                "artifact_id": aid,
                "artifact_digest": adig,
                "artifact_link": link,
            }
        )
    return chain


def _metric_info(
    facts: Sequence[_RunFacts], metric_names: Sequence[str], contract: ComparisonContract, warnings: list[str]
) -> list[MetricInfo]:
    declared = {d.name: d for d in contract.metrics}
    out: list[MetricInfo] = []
    for m in metric_names:
        rows = [f.metric_rows[m] for f in facts if f.bucket == "included" and m in f.metric_rows]
        units = {str(r["unit"]) for r in rows if r.get("unit") is not None}
        dirs = {bool(r["higher_is_better"]) for r in rows if isinstance(r.get("higher_is_better"), bool)}
        if len(units) > 1:
            warnings.append(f"{m}: evaluations report different units {sorted(units)}")
        if len(dirs) > 1:
            warnings.append(f"{m}: evaluations disagree on higher_is_better")
        observed_unit = next(iter(units)) if len(units) == 1 else None
        observed_dir = next(iter(dirs)) if len(dirs) == 1 else None
        decl = declared.get(m)
        if decl is not None:
            out.append(
                MetricInfo(
                    name=m,
                    unit=decl.unit if decl.unit is not None else observed_unit,
                    higher_is_better=decl.higher_is_better
                    if decl.higher_is_better is not None
                    else observed_dir,
                    source="contract",
                )
            )
        elif observed_unit is not None or observed_dir is not None:
            out.append(
                MetricInfo(name=m, unit=observed_unit, higher_is_better=observed_dir, source="evaluations")
            )
        else:
            out.append(MetricInfo(name=m))
    return out


def _check_expected_seeds(
    rs: RunSet, labels: Mapping[str, str], order: Sequence[str], expected: Sequence[int] | None
) -> None:
    if expected is None:
        return
    want = {seed_key(s) for s in expected}
    if len(want) != len(expected):
        raise InvalidArgument(
            "expected_seeds contains duplicates", details={"expected_seeds": list(expected)}
        )
    per_group: dict[str, set[str]] = {k: set() for k in order}
    for c in rs.cells:
        per_group[jdump(c.treatment_key)].add(seed_key(c.seed))
    problems = []
    for k in order:
        planned = per_group[k]
        if planned != want:
            problems.append(
                {
                    "group": labels[k],
                    "planned": sorted(planned, key=_seed_order),
                    "expected_but_not_planned": sorted(want - planned, key=_seed_order),
                    "planned_but_not_expected": sorted(planned - want, key=_seed_order),
                }
            )
    if problems:
        raise PreconditionFailed(
            "expected_seeds do not match the planned run set",
            details={
                "expected_seeds": sorted(want, key=_seed_order),
                "groups": problems[:MAX_LISTED],
                "remediation": "extend the experiment with the missing seeds, or pass the planned seeds and "
                "exclude runs explicitly with reasons",
            },
        )


def _finite_or_none(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _seed_value(seed: int | None) -> Seed:
    return "none" if seed is None else seed


def _build_groups(
    order: Sequence[str],
    labels: Mapping[str, str],
    facts: Sequence[_RunFacts],
    metric_names: Sequence[str],
    contract: ComparisonContract,
) -> list[GroupResult]:
    by_group: dict[str, list[_RunFacts]] = {k: [] for k in order}
    for f in facts:
        by_group[jdump(f.cell.treatment_key)].append(f)
    groups: list[GroupResult] = []
    for k in order:
        members = by_group[k]
        per_seed: dict[str, list[_RunFacts]] = {}
        for f in members:
            per_seed.setdefault(seed_key(f.cell.seed), []).append(f)
        chosen: dict[str, _RunFacts] = {}
        for s, runs in per_seed.items():
            active = [r for r in runs if r.bucket != "excluded"]
            if len(active) > 1:
                raise Ambiguous(
                    f"group {labels[k]!r} has {len(active)} planned runs for seed {s}; exclude all but one "
                    "(with a reason) or compare by plan ref",
                    details={"group": labels[k], "seed": s, "run_ids": [r.cell.run_id for r in active]},
                )
            chosen[s] = active[0] if active else runs[0]
        metrics: dict[str, MetricStats] = {}
        for m in metric_names:
            values: dict[str, float] = {}
            missing: list[Seed] = []
            failed: list[Seed] = []
            excluded: list[ExcludedSeed] = []
            run_by_seed: dict[str, str] = {}
            for s in sorted(chosen, key=_seed_order):
                f = chosen[s]
                run_by_seed[s] = f.cell.run_id
                sv = _seed_value(f.cell.seed)
                if f.bucket == "excluded":
                    excluded.append(ExcludedSeed(seed=sv, reason=f.exclusion_reason or ""))
                elif f.bucket == "missing":
                    missing.append(sv)
                elif f.bucket == "failed":
                    failed.append(sv)
                elif m not in f.status.metrics:
                    missing.append(sv)
                else:
                    value = _finite_or_none(f.status.metrics[m])
                    if value is None:
                        failed.append(sv)
                    else:
                        values[s] = value
            summary = stats.summarize([values[s] for s in sorted(values, key=_seed_order)], contract.ci_level)
            metrics[m] = MetricStats(
                n=summary.n,
                expected_n=len(chosen),
                mean=summary.mean,
                std=summary.std,
                sem=summary.sem,
                ci_low=summary.ci_low,
                ci_high=summary.ci_high,
                min=summary.min,
                max=summary.max,
                values_by_seed=values,
                missing_seeds=missing,
                failed_seeds=failed,
                excluded=excluded,
                run_by_seed=run_by_seed,
            )
        groups.append(
            GroupResult(
                label=labels[k],
                key=members[0].cell.treatment_key if members else {},
                run_ids=[f.cell.run_id for f in members],
                seeds=[_seed_value(chosen[s].cell.seed) for s in sorted(chosen, key=_seed_order)],
                metrics=metrics,
            )
        )
    return groups


def _same_scalar(a: Any, b: Any) -> bool:
    if a == b and type(a) is type(b):
        return True
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return float(a) == float(b)
    return _scalar_text(a) == _scalar_text(b)


def _key_matches(key: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    return set(key) == set(target) and all(_same_scalar(key[a], target[a]) for a in key)


def _differences(
    groups: Sequence[GroupResult], contract: ComparisonContract, warnings: list[str]
) -> tuple[list[DifferenceResult], set[str]]:
    """Group − baseline per metric. The baseline of a group is the group whose key equals the
    group's key with the contract's baseline values substituted (multi-axis matrices get the
    baseline at the same values of the other axes)."""
    baseline = contract.baseline
    if not baseline:
        warnings.append("no baseline in the comparison contract; differences were not computed")
        return [], set()
    axes = {a for g in groups for a in g.key}
    foreign = sorted(set(baseline) - axes)
    if foreign:
        warnings.append(f"baseline axes {foreign} are not matrix axes; differences were not computed")
        return [], set()
    is_baseline = {g.label for g in groups if all(_same_scalar(g.key.get(a), v) for a, v in baseline.items())}
    if not is_baseline:
        warnings.append(f"baseline {baseline} matches no group; differences were not computed")
        return [], set()
    diffs: list[DifferenceResult] = []
    for g in groups:
        if g.label in is_baseline:
            continue
        target = {**g.key, **baseline}
        base = next((b for b in groups if b.label in is_baseline and _key_matches(b.key, target)), None)
        if base is None:
            warnings.append(f"group {g.label!r} has no baseline group with the same other axes")
            continue
        for m, cell in g.metrics.items():
            bcell = base.metrics[m]
            if contract.pair_by_seed:
                p = stats.paired_differences(cell.values_by_seed, bcell.values_by_seed, contract.ci_level)
                diffs.append(
                    DifferenceResult(
                        group_label=g.label,
                        baseline_label=base.label,
                        metric=m,
                        kind="paired",
                        n=p.n_pairs,
                        n_baseline=None,
                        mean_diff=p.mean_diff,
                        ci_low=p.ci_low,
                        ci_high=p.ci_high,
                        std_diff=p.std_diff,
                        se_diff=p.sem_diff,
                        paired_seeds=[_as_seed(s) for s in p.paired_seeds],
                        unpaired_group=[_as_seed(s) for s in p.unpaired_a],
                        unpaired_baseline=[_as_seed(s) for s in p.unpaired_b],
                    )
                )
            else:
                ordered_g = [cell.values_by_seed[s] for s in sorted(cell.values_by_seed, key=_seed_order)]
                ordered_b = [bcell.values_by_seed[s] for s in sorted(bcell.values_by_seed, key=_seed_order)]
                u = stats.unpaired_difference(ordered_g, ordered_b, contract.ci_level)
                diffs.append(
                    DifferenceResult(
                        group_label=g.label,
                        baseline_label=base.label,
                        metric=m,
                        kind="unpaired",
                        n=u.n_a,
                        n_baseline=u.n_b,
                        mean_diff=u.mean_diff,
                        ci_low=u.ci_low,
                        ci_high=u.ci_high,
                        se_diff=u.se_diff,
                    )
                )
    return diffs, is_baseline


def _as_seed(s: Seed) -> Seed:
    if isinstance(s, str) and s.lstrip("-").isdigit():
        return int(s)
    return s


def _distinct(values: Iterable[tuple[str | None, str]]) -> tuple[dict[str, list[str]], list[str]]:
    """value → labels, plus labels with an unknown value."""
    known: dict[str, list[str]] = {}
    unknown: list[str] = []
    for value, who in values:
        if value is None:
            unknown.append(who)
        else:
            known.setdefault(value, []).append(who)
    return known, unknown


def _who(f: _RunFacts) -> str:
    return f"{f.group_label}/seed {seed_key(f.cell.seed)}"


def _invariants(
    facts: Sequence[_RunFacts], metric_names: Sequence[str], contract: ComparisonContract, warnings: list[str]
) -> list[IncomparableEntry]:
    included = [f for f in facts if f.bucket == "included" and f.evaluation is not None]
    out: list[IncomparableEntry] = []
    if not included:
        return out
    report_split = contract.selection.report_split
    for inv in contract.invariants:
        if inv in ("split", "dataset"):
            for m in metric_names:
                rows = [(f, f.metric_rows.get(m)) for f in included if m in f.metric_rows]
                known, unknown = _distinct(
                    (str(r[inv]) if r and r.get(inv) is not None else None, _who(f)) for f, r in rows
                )
                if len(known) > 1:
                    out.append(
                        IncomparableEntry(
                            reason=f"{inv} differs between included results for metric {m}",
                            invariant=inv,
                            metric=m,
                            details={"values": {k: v[:10] for k, v in sorted(known.items())}},
                        )
                    )
                elif known and unknown:
                    warnings.append(
                        f"{m}: {inv} is not recorded for {len(unknown)} included result(s); invariant "
                        "only partially checked"
                    )
                elif not known and rows:
                    warnings.append(f"{m}: {inv} is not recorded by the evaluations; invariant unchecked")
                if inv == "split" and report_split is not None:
                    off = {k: v for k, v in known.items() if k != report_split}
                    if off:
                        out.append(
                            IncomparableEntry(
                                reason=f"metric {m} was not measured on the contract's report_split "
                                f"{report_split!r}",
                                invariant="split",
                                metric=m,
                                details={"values": {k: v[:10] for k, v in sorted(off.items())}},
                            )
                        )
        elif inv == "evaluator":
            ids, _ = _distinct(
                (f"{f.evaluation['evaluator_id']}@{f.evaluation['evaluator_version']}", _who(f))
                for f in included
                if f.evaluation
            )
            if len(ids) > 1:
                out.append(
                    IncomparableEntry(
                        reason="evaluator id/version differs between included results",
                        invariant="evaluator",
                        details={"values": {k: v[:10] for k, v in sorted(ids.items())}},
                    )
                )
            else:
                configs, _ = _distinct(
                    (str(f.evaluation.get("config_hash")), _who(f)) for f in included if f.evaluation
                )
                if len(configs) > 1:
                    out.append(
                        IncomparableEntry(
                            reason="evaluator config differs between included results",
                            invariant="evaluator",
                            details={"config_hashes": {k: v[:10] for k, v in sorted(configs.items())}},
                        )
                    )
        else:
            values: list[tuple[str | None, str]] = []
            seen_anywhere = False
            for f in included:
                raw_params = f.spec.get("params")
                params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
                source = params if inv in params else f.cell.cell if inv in f.cell.cell else None
                if source is None:
                    values.append((None, _who(f)))
                else:
                    seen_anywhere = True
                    values.append((_scalar_text(source[inv]), _who(f)))
            known, unknown = _distinct(values)
            if not seen_anywhere:
                warnings.append(f"invariant {inv!r} is not a recorded run parameter; unchecked")
            elif len(known) > 1:
                out.append(
                    IncomparableEntry(
                        reason=f"parameter {inv} differs between included results",
                        invariant=inv,
                        details={"values": {k: v[:10] for k, v in sorted(known.items())}},
                    )
                )
            elif unknown:
                warnings.append(f"invariant {inv!r} is not recorded for {len(unknown)} included run(s)")
    return out


@dataclass(frozen=True)
class _Observation:
    """One moment a run's metric was (or could be) seen: ``seq`` is its event when the ledger
    recorded one (the events table orders it against a declaration), else only ``at`` is known."""

    at: str | None
    seq: int | None

    def before(self, seq: int, at: str) -> bool:
        """Did it precede a declaration (its event ``seq``, time ``at``)? Events order decides
        when both have events; by the clock a tie counts as observed first."""
        if self.seq is not None:
            return self.seq < seq
        return self.at is not None and self.at <= at


# Events of an artifact that mean someone read its content: artifact_read of a live output that
# had grown (artifacts._refresh_mutable), or a download of it to the local cache.
_ARTIFACT_READ_EVENTS = ("artifact.observed", "artifact.cached")


def _observations(reader: Reader, run_id: str) -> list[_Observation]:
    """Every recorded moment the run's metric was observable: each evaluation of the run (any
    attempt, any status — its first event, else its ``created_at``); the end of each attempt that
    COMPLETED with valid outputs (its metrics file is readable from then on: its terminal event,
    else ``terminal_at``); and each read of one of the run's ``metrics`` outputs — a live read of
    a running attempt's metrics included."""
    first_seq: dict[str, int] = {}
    evaluations = reader.all("SELECT evaluation_id, created_at FROM evaluations WHERE run_id=?", (run_id,))
    attempts = reader.all(
        "SELECT attempt_id, terminal_at FROM attempts WHERE run_id=? AND scheduler_state='COMPLETED' "
        "AND result_state='valid' AND terminal_at IS NOT NULL",
        (run_id,),
    )
    for r in _chunked_in(
        reader,
        "SELECT entity_id, MIN(seq) AS seq FROM events WHERE entity_type='evaluation' "
        "AND entity_id IN ({marks}) GROUP BY entity_id",
        [str(e["evaluation_id"]) for e in evaluations],
    ):
        first_seq[str(r["entity_id"])] = int(r["seq"])
    for r in _chunked_in(
        reader,
        "SELECT entity_id, MIN(seq) AS seq FROM events WHERE entity_type='attempt' "
        "AND entity_id IN ({marks}) AND event_type IN ('attempt.terminal', 'attempt.terminal_revised') "
        "GROUP BY entity_id",
        [str(a["attempt_id"]) for a in attempts],
    ):
        first_seq[str(r["entity_id"])] = int(r["seq"])
    out = [
        _Observation(
            at=str(e["created_at"]) if e["created_at"] else None, seq=first_seq.get(str(e["evaluation_id"]))
        )
        for e in evaluations
    ]
    out += [_Observation(at=str(a["terminal_at"]), seq=first_seq.get(str(a["attempt_id"]))) for a in attempts]
    marks = ",".join("?" for _ in _ARTIFACT_READ_EVENTS)
    out += [
        _Observation(at=str(r["recorded_at"]), seq=int(r["seq"]))
        for r in reader.all(
            "SELECT e.seq, e.recorded_at FROM events e JOIN artifacts ar ON ar.artifact_id = e.entity_id "
            f"WHERE e.entity_type='artifact' AND e.event_type IN ({marks}) AND ar.run_id=? "
            "AND ar.role='metrics'",
            (*_ARTIFACT_READ_EVENTS, run_id),
        )
    ]
    return out


@dataclass
class _PastRequest:
    """One earlier comparison request, as the exclusion history of a run sees it: archived
    (``comparison.created``) or answered by reusing an identical archived comparison
    (``comparison.reused``). ``seq`` (its event) orders requests, evaluations and exclusions."""

    comparison_id: str
    run_set_ref: str
    at: str
    seq: int
    excluded: frozenset[str]
    reused: bool

    def describe(self) -> str:
        return f"comparison {self.comparison_id} ({self.at}{', answered from the archive' if self.reused else ''})"


@dataclass
class _Declaration:
    """The request that started the exclusion episode in effect (None: the comparison being computed)."""

    request: _PastRequest | None
    note: str | None


def _run_set_refs(reader: Reader, run_id: str) -> tuple[set[str], str]:
    """Refs of every run set that can contain the run (:func:`resolve_run_set`), and the run's
    creation time (no comparison before it contained it)."""
    run = reader.one(
        "SELECT r.experiment_id, r.plan_id, r.created_at, p.kind AS plan_kind FROM runs r "
        "JOIN plans p ON p.plan_id = r.plan_id WHERE r.run_id=?",
        (run_id,),
    )
    if run is None:  # pragma: no cover - the run was just resolved from the same ledger
        return set(), ""
    refs = {_URI_PREFIXES["plan"] + run["plan_id"]}
    if run["plan_kind"] != "derived":
        refs.add(_URI_PREFIXES["experiment"] + run["experiment_id"])
    for w in reader.all(
        "SELECT w.workflow_id, w.plan_id FROM workflow_cells c JOIN workflows w ON w.workflow_id = c.workflow_id "
        "WHERE c.run_id=?",
        (run_id,),
    ):
        refs.add(_URI_PREFIXES["workflow"] + w["workflow_id"])
        if w["plan_id"] and w["plan_id"] != run["plan_id"]:
            own = reader.one(
                "SELECT 1 AS x FROM runs WHERE plan_id=? AND kind='main' LIMIT 1", (w["plan_id"],)
            )
            if own is None:  # a plan without runs of its own resolves to its workflow's cells
                refs.add(_URI_PREFIXES["plan"] + w["plan_id"])
    return refs, str(run["created_at"] or "")


def _past_requests(reader: Reader, refs: set[str], extra_ids: set[str]) -> list[_PastRequest]:
    """Every recorded request for an archived comparison of the given run sets (plus
    ``extra_ids``) — its creation and each reuse — newest first, in the events table's order."""
    rows = _chunked_in(
        reader,
        "SELECT comparison_id, run_set_ref, request_json FROM comparisons WHERE run_set_ref IN ({marks})",
        sorted(refs),
    )
    seen = {r["comparison_id"] for r in rows}
    rows += _chunked_in(
        reader,
        "SELECT comparison_id, run_set_ref, request_json FROM comparisons WHERE comparison_id IN ({marks})",
        sorted(extra_ids - seen),
    )
    info: dict[str, tuple[str, frozenset[str]]] = {}
    for r in rows:
        request = jload(r["request_json"], {}) or {}
        raw = request.get("exclusions") if isinstance(request, dict) else None
        info[str(r["comparison_id"])] = (
            str(r["run_set_ref"]),
            frozenset(
                str(e["run_id"]) for e in raw or [] if isinstance(e, dict) and e.get("run_id") is not None
            ),
        )
    out = []
    for e in _chunked_in(
        reader,
        "SELECT seq, entity_id, event_type, recorded_at FROM events WHERE entity_type='comparison' "
        "AND event_type IN ('comparison.created', 'comparison.reused') AND entity_id IN ({marks})",
        list(info),
    ):
        ref, excluded = info[str(e["entity_id"])]
        out.append(
            _PastRequest(
                comparison_id=str(e["entity_id"]),
                run_set_ref=ref,
                at=str(e["recorded_at"]),
                seq=int(e["seq"]),
                excluded=excluded,
                reused=e["event_type"] == "comparison.reused",
            )
        )
    out.sort(key=lambda c: c.seq, reverse=True)
    return out


class _History:
    """Lazily-read facts about archived comparisons (shared by every excluded run)."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader
        self._blanket: dict[str, bool] = {}
        self._runs: dict[str, set[str]] = {}

    def blanket(self, c: _PastRequest) -> bool:
        """Did ``c`` exclude every planned run of its run set? Excluding everything costs nothing
        before any result exists, so it is not a decision about any one run."""
        if c.comparison_id not in self._blanket:
            row = self.reader.one(
                "SELECT payload_json FROM events WHERE entity_type='comparison' AND entity_id=? "
                "AND event_type='comparison.created' ORDER BY seq LIMIT 1",
                (c.comparison_id,),
            )
            payload = jload(row["payload_json"] if row else None, {}) or {}
            counts = payload.get("counts") if isinstance(payload, dict) else None
            if isinstance(counts, dict) and isinstance(counts.get("planned_runs"), int):
                planned = int(counts["planned_runs"])
            else:  # pragma: no cover - every archived comparison has its creation event
                got = self.reader.one(
                    "SELECT json_extract(result_json, '$.data.counts.planned_runs') AS n FROM comparisons "
                    "WHERE comparison_id=?",
                    (c.comparison_id,),
                )
                planned = int(got["n"]) if got and got["n"] is not None else len(c.excluded) + 1
            self._blanket[c.comparison_id] = planned > 0 and len(c.excluded) >= planned
        return self._blanket[c.comparison_id]

    def contains(self, c: _PastRequest, run_id: str) -> bool:
        """Was the run part of ``c``'s planned run set (its archived runs list)?"""
        if c.comparison_id not in self._runs:
            self._runs[c.comparison_id] = {
                str(r["run_id"])
                for r in self.reader.all(
                    "SELECT json_extract(j.value, '$.run_id') AS run_id FROM comparisons c, "
                    "json_each(c.result_json, '$.data.runs') j WHERE c.comparison_id=?",
                    (c.comparison_id,),
                )
            }
        return run_id in self._runs[c.comparison_id]


def _declaration(history: _History, run_id: str, past: Sequence[_PastRequest]) -> _Declaration:
    """The first request of the run's current exclusion episode.

    Walking back from now through the recorded requests (newest first, events order) for
    comparisons whose run set contains the run: each one that excluded it extends the episode;
    one that did not exclude it (it was in the denominator, its value shown if it had one) ends
    it — an exclusion before that was revoked. A reused comparison is a request like any other.
    A comparison that excluded every planned run of its run set neither declares nor revokes.
    """
    start: _PastRequest | None = None
    revoker: _PastRequest | None = None
    blankets: list[_PastRequest] = []
    older: list[_PastRequest] = []
    for i, c in enumerate(past):
        if run_id in c.excluded:
            if history.blanket(c):
                blankets.append(c)
            else:
                start = c
            continue
        if history.contains(c, run_id):
            revoker = c
            older = list(past[i + 1 :])
            break
    notes = []
    revoked = next((c for c in older if run_id in c.excluded), None)
    if revoker is not None and revoked is not None:
        notes.append(
            f"an earlier exclusion ({revoked.describe()}) was revoked: {revoker.describe()} included "
            "the run afterwards"
        )
    unused = [b for b in blankets if start is None or b.seq < start.seq]
    if unused:
        b = unused[-1]  # the oldest: it would otherwise have been the declaration
        notes.append(
            f"comparison {b.comparison_id} ({b.at}) excluded every planned run of {b.run_set_ref}, "
            "which declares nothing about this run"
        )
    return _Declaration(request=start, note="; ".join(notes) or None)


def _exclusion_records(reader: Reader, facts: Sequence[_RunFacts], now: datetime) -> list[ExclusionRecord]:
    """Was the excluded run's result observable before the exclusion in effect was declared?

    Judged per exclusion *episode*: the declaration is the first request of the run's current,
    uninterrupted stretch of exclusions (:func:`_declaration`) — every comparison request of a run
    set containing the run counts, also one answered by reusing an archived comparison
    (``comparison.reused``), in the events table's order — or this comparison when nothing
    earlier still stands. An exclusion revoked (the run was in some request's denominator in
    between) and one that excluded every planned run declare nothing; the record's ``note`` says
    so. The metric was observed at each of :func:`_observations` (the run's evaluations, the end of
    an attempt that COMPLETED with valid outputs — its metrics file is readable before anyone
    collects an evaluation — and every recorded read of its metrics output, live ones included).
    Late when any observation precedes the declaration: in the events table's order when both
    are events, else by the clock (a tie counts as "observed first"). Excluding a run whose metric
    was never observable (failed or never ran, nothing read) is not late.
    """
    excluded = [f for f in facts if f.bucket == "excluded"]
    if not excluded:
        return []
    now_at = iso(now)
    assert now_at is not None
    history = _History(reader)
    refs: dict[str, set[str]] = {}
    born: dict[str, str] = {}
    for f in excluded:
        refs[f.cell.run_id], born[f.cell.run_id] = _run_set_refs(reader, f.cell.run_id)
    excluded_by: dict[str, set[str]] = {}  # run id -> comparisons that recorded excluding it
    for e in _chunked_in(
        reader,
        "SELECT seq, entity_id, payload_json FROM events WHERE entity_type='run' AND event_type='run.excluded' "
        "AND entity_id IN ({marks}) ORDER BY seq",
        list(refs),
    ):
        payload = jload(e["payload_json"], {}) or {}
        cid = payload.get("comparison_id") if isinstance(payload, dict) else None
        if isinstance(cid, str):
            excluded_by.setdefault(e["entity_id"], set()).add(cid)
    all_past = _past_requests(
        reader, set().union(*refs.values()), {cid for per_run in excluded_by.values() for cid in per_run}
    )
    out: list[ExclusionRecord] = []
    for f in excluded:
        rid = f.cell.run_id
        seen = _observations(reader, rid)
        obs_time = min((o.at for o in seen if o.at), default=None)
        mine = excluded_by.get(rid, set())
        past = [
            c
            for c in all_past
            if c.comparison_id in mine or (c.run_set_ref in refs[rid] and c.at >= born[rid])
        ]
        decl = _declaration(history, rid, past)
        decl_at = decl.request.at if decl.request is not None else now_at
        if not seen:
            late = False
        elif decl.request is None:
            late = True  # declared now, after the metric was observable
        else:
            late = any(o.before(decl.request.seq, decl.request.at) for o in seen)
        out.append(
            ExclusionRecord(
                run_id=rid,
                group_label=f.group_label,
                seed=f.cell.seed,
                reason=f.exclusion_reason or "",
                code=f.status.code,
                observed_before_exclusion=late,
                first_declared_at=decl_at,
                first_observed_at=obs_time,
                note=decl.note,
            )
        )
    return out


def _integrity(
    reader: Reader,
    facts: Sequence[_RunFacts],
    groups: Sequence[GroupResult],
    contract: ComparisonContract,
    exclusions: list[ExclusionRecord],
    newer_attempts: list[NewerAttempt],
    protocol_switches: list[ProtocolSwitch],
    *,
    pinned: bool,
    evaluator_filter: str | None = None,
) -> IntegrityData:
    included = [f for f in facts if f.bucket == "included"]
    post_hoc = [
        PostHocRun(run_id=f.cell.run_id, group_label=f.group_label, seed=f.cell.seed, status=f.bucket)
        for f in facts
        if f.cell.added_post_hoc
    ]
    heterogeneity = []
    dims: list[tuple[Literal["gpu", "torch", "code_digest"], Any]] = [
        ("gpu", lambda f: f.chain.env_gpu if f.chain else None),
        ("torch", lambda f: f.chain.env_torch if f.chain else None),
        ("code_digest", lambda f: f.chain.code_digest if f.chain else None),
    ]
    for dim, getter in dims:
        counts: dict[str, int] = {}
        unknown = 0
        per_group: dict[str, set[str]] = {}
        for f in included:
            value = getter(f)
            if value is None:
                unknown += 1
                continue
            counts[value] = counts.get(value, 0) + 1
            per_group.setdefault(f.group_label, set()).add(value)
        heterogeneity.append(
            HeterogeneityEntry(
                dimension=dim,
                values=dict(sorted(counts.items())),
                unknown=unknown,
                mixed=len(counts) > 1,
                within_groups=[g.label for g in groups if len(per_group.get(g.label, ())) > 1],
            )
        )
    multi: list[MultiValidAttempt] = []
    run_ids = [f.cell.run_id for f in included]
    # a valid attempt is one whose valid evaluation *could* count (coverage's rule): a job killed
    # at its walltime with finalized outputs and a valid select='last' evaluation is not one, nor
    # is an attempt evaluated only by an evaluator this comparison never counts (its evaluator
    # filter; pinned evaluation_refs could have picked any valid evaluation)
    sql = (
        "SELECT e.run_id, COUNT(DISTINCT e.attempt_id) AS n FROM evaluations e JOIN attempts a "
        "ON a.attempt_id = e.attempt_id WHERE e.run_id IN ({marks}) AND e.status='valid' "
        f"AND {COUNTED_ATTEMPT_WHERE}"
    )
    after: list[Any] = []
    if evaluator_filter is not None:
        sql += " AND e.evaluator_id = ?"
        after.append(evaluator_filter)
    rows = _chunked_in(reader, sql + " GROUP BY e.run_id", run_ids, after=after)
    n_valid = {r["run_id"]: int(r["n"]) for r in rows}
    for f in included:
        if n_valid.get(f.cell.run_id, 0) > 1:
            multi.append(
                MultiValidAttempt(
                    run_id=f.cell.run_id,
                    group_label=f.group_label,
                    seed=f.cell.seed,
                    valid_attempts=n_valid[f.cell.run_id],
                    selected_attempt_id=f.status.evaluation_attempt_id,
                    rule=PINNED_RULE if pinned else contract.attempt_selection,
                )
            )
    leaking = []
    for f in included:
        validation = jload(f.evaluation.get("validation_json"), {}) if f.evaluation else {}
        issues = validation.get("issues") if isinstance(validation, dict) else None
        if any(isinstance(i, dict) and i.get("code") == "selection_on_split" for i in issues or []):
            leaking.append(f.cell.run_id)
    sel = contract.selection
    below = [
        BelowMinN(group_label=g.label, metric=m, n=c.n, min_n=contract.min_n)
        for g in groups
        for m, c in g.metrics.items()
        if c.n < contract.min_n
    ]
    return IntegrityData(
        attempt_selection=contract.attempt_selection,
        post_hoc=post_hoc,
        exclusions=exclusions,
        heterogeneity=heterogeneity,
        multiple_valid_attempts=multi,
        selection=SelectionCheck(
            rule=sel.rule,
            select_metric=sel.select_metric,
            select_split=sel.select_split,
            report_split=sel.report_split,
            evaluation_selection_on_split=leaking,
        ),
        below_min_n=below,
        newer_attempts=newer_attempts,
        protocol_switches=protocol_switches,
    )


def _counts(facts: Sequence[_RunFacts]) -> CoverageCounts:
    buckets = {"included": 0, "missing": 0, "failed": 0, "excluded": 0}
    by_code: dict[str, int] = {}
    for f in facts:
        buckets[f.bucket] += 1
        by_code[f.status.code] = by_code.get(f.status.code, 0) + 1
    return CoverageCounts(planned_runs=len(facts), **buckets, by_code=dict(sorted(by_code.items())))


def awaiting_evaluation(run: RunCoverage) -> bool:
    """A missing run whose job COMPLETED but has no valid evaluation yet (code ``E``): its value
    may still arrive, so a table that counts it as missing is provisional. A run with a valid
    evaluation that the comparison does not count is not waiting for anything."""
    return run.status == "missing" and run.code == "E" and not (run.reason or "").startswith(UNCOUNTED_PREFIX)


def _awaiting_evaluation(facts: Sequence[_RunFacts]) -> list[_RunFacts]:
    """Facts of the runs :func:`awaiting_evaluation` is true for."""
    return [f for f in facts if f.bucket == "missing" and f.status.code == "E" and not f.uncounted]


def _coverage(facts: Sequence[_RunFacts], counts: CoverageCounts) -> Coverage:
    gaps = [f for f in facts if f.bucket != "included"]
    missing = [f"{f.cell.label} [{f.bucket}]: {f.reason or f.status.code}" for f in gaps]
    notes = []
    if gaps:
        notes.append(
            f"{len(gaps)} of {counts.planned_runs} planned runs are not included; they stay in the "
            "denominator as missing/failed/excluded"
        )
    awaiting = _awaiting_evaluation(facts)
    if awaiting:
        notes.append(
            f"{len(awaiting)} run(s) completed but have no valid evaluation yet (E) — outputs being "
            "checked or evaluation pending: counted as missing, not failed; compare again once "
            "evaluation_collect has recorded their evaluations"
        )
    bad = [f for f in facts if f.bucket == "failed" and f.status.code == "C!"]
    if bad:
        notes.append(
            f"{len(bad)} run(s) completed with incomplete/invalid outputs or a partial/invalid "
            "evaluation (C!): counted as failed"
        )
    if len(missing) > MAX_LISTED:
        notes.append(f"{len(missing) - MAX_LISTED} more gaps are not listed here (see runs)")
    return Coverage(
        complete=not gaps,
        sources=[SourceStatus(name="ledger", status="ok")],
        missing=missing[:MAX_LISTED],
        notes=notes,
    )


def _summary(
    groups: Sequence[GroupResult],
    metric_names: Sequence[str],
    counts: CoverageCounts,
    incomparable: Sequence[IncomparableEntry],
    awaiting_evaluation: int = 0,
) -> str:
    text = (
        f"{len(groups)} group(s) × {len(metric_names)} metric(s) over {counts.planned_runs} planned runs: "
        f"{counts.included} included, {counts.missing} missing, {counts.failed} failed, "
        f"{counts.excluded} excluded"
    )
    if awaiting_evaluation:
        text += f" ({awaiting_evaluation} of the missing completed and await evaluation)"
    if incomparable:
        text += f"; {len(incomparable)} incomparability issue(s)"
    return text


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def _archive(tx: Tx, computed: _Computed, *, request: Mapping[str, Any]) -> ComparisonData:
    """Insert the comparison unless the newest archived answer to the same request is identical.

    "Same request" is :func:`request_key` (exploratory comparisons with other exclusions or
    another contract in between do not force a duplicate). Only the *newest* same-request
    comparison may be reused, so the newest answer to a question is always the current one. A
    reused answer is still recorded (``comparison.reused``): which runs a request included or
    excluded, and when, is what :func:`_exclusion_records` judges exclusions by.
    """
    chains = {rid: c.model_dump(mode="json") for rid, c in computed.chains.items()}
    archive = {"schema": ARCHIVE_SCHEMA, "data": computed.data.model_dump(mode="json"), "chains": chains}
    content_hash = archive_content_hash(archive)
    run_set_ref = computed.run_set.ref
    key = request_key(
        run_set_ref, computed.data.contract, request.get("exclusions") or [], request.get("evaluation_refs")
    )
    newest = next(
        (
            row
            for row in tx.all(
                "SELECT comparison_id, run_set_ref, contract_json, request_json, content_hash FROM comparisons "
                "WHERE run_set_ref=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (run_set_ref, MAX_REUSE_SCAN),
            )
            if comparison_request_key(row) == key
        ),
        None,
    )
    if newest is not None and newest["content_hash"] == content_hash:
        stored_row = tx.one(
            "SELECT result_json FROM comparisons WHERE comparison_id=?", (newest["comparison_id"],)
        )
        stored = jload(stored_row["result_json"] if stored_row else None, {}) or {}
        data = ComparisonData.model_validate(stored["data"])
        # The request is part of the ledger's history even though nothing new is archived: a
        # request that includes a run ends that run's exclusion episode (_exclusion_records).
        tx.event(
            "comparison",
            newest["comparison_id"],
            "comparison.reused",
            {
                "run_set_ref": run_set_ref,
                "content_hash": content_hash,
                "excluded": sorted({str(e.get("run_id")) for e in request.get("exclusions") or []}),
            },
        )
        return data.model_copy(update={"reused": True, "observed_at": computed.data.observed_at})
    cid = new_id("comparison")
    data = computed.data.model_copy(
        update={
            "comparison_id": cid,
            "comparison_ref": _URI_PREFIXES["comparison"] + cid,
            "created_at": tx.now,
            "content_hash": content_hash,
        }
    )
    archive["data"] = data.model_dump(mode="json")
    tx.insert(
        "comparisons",
        {
            "comparison_id": cid,
            "run_set_ref": run_set_ref,
            "contract_json": jdump(data.contract.model_dump(mode="json")),
            "request_json": jdump(dict(request)),
            "result_json": jdump(archive),
            "content_hash": content_hash,
            "created_at": tx.now,
        },
    )
    tx.event(
        "comparison",
        cid,
        "comparison.created",
        {
            "run_set_ref": run_set_ref,
            "content_hash": content_hash,
            "metric_names": data.metric_names,
            "counts": data.counts.model_dump(mode="json"),
            "incomparable": len(data.incomparable),
        },
    )
    for rec in computed.exclusions:
        tx.event(
            "run",
            rec.run_id,
            "run.excluded",
            {
                "comparison_id": cid,
                "reason": rec.reason,
                "observed_before_exclusion": rec.observed_before_exclusion,
            },
        )
    return data


def _cap_runs(data: ComparisonData) -> ComparisonData:
    if len(data.runs) <= MAX_RUNS_RETURNED:
        return data
    return data.model_copy(
        update={"runs": data.runs[:MAX_RUNS_RETURNED], "runs_truncated": len(data.runs) - MAX_RUNS_RETURNED}
    )
