"""Experiment preparation: freeze code + configuration into Runs and ExecutionPlans.

``experiment_prepare`` (build spec §7.2 "prepare", tools design §02) turns an ExperimentSpec
into a *plan*: one immutable ``RunSpec`` per matrix cell (+ an optional smoke Run) and one
``ExecutionPlan`` per Run, all content-hashed, validated and stored in one transaction. It never
submits anything; ``run_submit``/``workflow_submit`` consume ready plans.

Flow (network I/O never inside a transaction)::

    normalize request -> idempotency lookup (same key+payload -> stored response)
      -> resolve target (new experiment | extends | derive_from) -> scan the project
      -> small snapshot:  capture -> build runs -> validate (static + bounded remote) -> ONE tx
      -> large snapshot:  ONE tx (experiment/plan 'preparing' + operation + outbox
                          ``prepare.snapshot``) -> the handler does the rest in the background

Honesty rules: a plan is ``ready`` only after its snapshot was captured and no blocking
finding remains; ``plan_hash`` exists only for ready plans; an ``invalid`` plan keeps its
findings and a preview of the cells but creates **no** Run rows (so run sets, grids and
comparisons never count runs that cannot execute). Overrides of blocking findings are
recorded as events. Runs are inserted in matrix order (smoke first), so ``ORDER BY rowid``
within a plan reproduces the deterministic cell order.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hpclab.api.envelope import (
    MAX_TEXT_BYTES,
    Coverage,
    clamp_bytes,
    clamp_limit,
    decode_cursor,
    encode_cursor,
)
from hpclab.domain.clock import iso
from hpclab.domain.errors import (
    HpcLabError,
    InvalidArgument,
    LimitExceeded,
    NotFound,
    NotReady,
    PreconditionFailed,
    Stale,
    VersionConflict,
)
from hpclab.domain.hashing import canonical_json, digest, sha256_hex, short
from hpclab.domain.ids import new_id
from hpclab.domain.models import (
    CodeSnapshotRef,
    ExecutionPlan,
    ExperimentSpec,
    ResilienceSpec,
    ResourceSpec,
    RunSpec,
    Scalar,
)
from hpclab.domain.states import RunStatus
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services import matrix as mx
from hpclab.services import validation as vd
from hpclab.services.context import ServiceContext
from hpclab.services.localfiles import FileTooLarge, NotRegularFile, read_regular_file
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.storage.db import Tx, jdump, jload

log = logging.getLogger("hpclab.prepare")

TOOL = "experiment_prepare"
OUTBOX_KIND = "prepare.snapshot"
MAX_RUNS_RETURNED = mx.MAX_MATRIX_CELLS + 1  # every main cell + the smoke run
MAX_SPEC_BYTES = 1024 * 1024  # a spec_path file: a small hand-written YAML file
BACKGROUND_REMOTE_BUDGET_S = 60.0
# The handler finishes the plan (invalid) itself before the outbox runner would dead-letter the
# item: crashes are dead after 3 tries there, retryable errors much later.
BACKGROUND_MAX_TRIES = 5
BACKGROUND_MAX_CRASHES = 3
BACKGROUND_CONFLICT_RETRY_S = 5.0
_PLAN_COLUMNS = (
    "plan_id, experiment_id, kind, status, plan_hash, snapshot_digest, validation_json, smoke_run_id, "
    "operation_id, created_at, updated_at"
)


def plan_ref(plan_id: str) -> str:
    return f"hpclab://plans/{plan_id}"


def operation_ref(operation_id: str) -> str:
    return f"hpclab://operations/{operation_id}"


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------


class DeriveOverrides(BaseModel):
    """What changes in a derived Run (a changed RunSpec is always a *new* Run)."""

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Scalar] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    argv_append: list[str] = Field(default_factory=list)


class DeriveFrom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    overrides: DeriveOverrides = Field(default_factory=DeriveOverrides)
    recapture_code: bool = False


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class PreparedRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None  # None: not created (plan preparing or invalid)
    label: str
    kind: str
    cell: dict[str, Any]
    seed: int | None
    exec_plan_hash: str | None
    requested_gpu_hours: float
    added_post_hoc: bool = False
    derived_from: str | None = None


class SnapshotInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["captured", "reused", "background"]
    ref: CodeSnapshotRef | None = None
    file_count: int | None = None
    total_bytes: int | None = None
    warnings: list[str] = Field(default_factory=list)
    excluded_count: int = 0
    reused_store: bool = False


class BudgetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_gpu_hours_total: float  # main + smoke runs of this plan (gpus x time limit)
    smoke_gpu_hours: float
    budget_gpu_hours: float | None
    existing_planned_gpu_hours: float = 0.0  # other plans of the experiment (extends/derive)


class OverrideRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    reason: str
    applied: bool
    note: str


class PrepareData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    plan_id: str
    plan_ref: str
    plan_hash: str | None
    status: Literal["ready", "preparing", "invalid"]
    kind: str
    runs: list[PreparedRun]
    runs_total: int
    smoke_run_id: str | None
    validation: vd.ValidationReport
    resource_advice: vd.ResourceAdvice | None
    snapshot: SnapshotInfo | None
    budget: BudgetInfo
    overrides: list[OverrideRecord] = Field(default_factory=list)
    operation_id: str | None = None
    operation_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str | None = None

    def summary(self) -> str:
        if self.status == "preparing":
            return (
                f"plan {self.plan_id} is preparing in the background ({self.runs_total} runs planned); "
                f"poll {self.operation_ref}"
            )
        if self.status == "invalid":
            return f"plan {self.plan_id} is invalid: blocking checks {self.validation.blocking}"
        return (
            f"plan {self.plan_id} ready: {self.runs_total} runs, {self.budget.requested_gpu_hours_total:.2f} GPU-h "
            f"requested, validation {self.validation.status}"
        )


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------


@dataclass
class _Request:
    mode: Literal["new", "extend", "derive"]
    spec: dict[str, Any] | None = None
    extends: str | None = None
    add_seeds: list[int] = field(default_factory=list)
    add_matrix: dict[str, list[Scalar]] = field(default_factory=dict)
    derive_from: dict[str, Any] | None = None
    overrides: list[dict[str, str]] = field(default_factory=list)
    validate_remote: bool = True

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> _Request:
        return cls(**data)

    def override_models(self) -> list[vd.ValidationOverride]:
        return [vd.ValidationOverride.model_validate(o) for o in self.overrides]

    def payload_hash(self) -> str:
        """Idempotency payload hash. Non-finite numbers (NaN/Infinity from YAML ``.nan`` or a
        JSON-RPC client) cannot be frozen into canonical JSON: a caller error, not a crash."""
        try:
            return digest(self.payload())
        except (ValueError, TypeError) as exc:
            raise InvalidArgument(
                "the request contains a value that is not a finite JSON number/string "
                f"(NaN/Infinity are not allowed in specs, parameters or seeds): {exc}",
            ) from exc


def _compact_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {"loc": ".".join(str(x) for x in e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()[:20]
    ]


_YAML_QUOTED_RE = re.compile(r"'([^']*)'")


def _yaml_problem(exc: yaml.YAMLError) -> str:
    """PyYAML's problem text without file content.

    ``str(exc)`` quotes the offending source line, and ``problem``/``context`` quote aliases,
    anchors and tags taken from the file: ``spec_path`` may name any file the user can read
    (``~/.aws/credentials``), so only token kinds (``'<scalar>'``) and single characters
    (``'\\t'``) are kept; anything else quoted becomes ``'…'``.
    """

    def mask(m: re.Match[str]) -> str:
        inner = m.group(1)
        if len(inner) <= 2 or (inner.startswith("<") and inner.endswith(">")):
            return m.group(0)
        return "'…'"

    if isinstance(exc, yaml.reader.ReaderError):
        return "the file contains a character YAML does not accept"
    parts = [getattr(exc, "problem", None), getattr(exc, "context", None)]
    text = "; ".join(p for p in parts if isinstance(p, str) and p)
    return _YAML_QUOTED_RE.sub(mask, text)[:200] or type(exc).__name__


def load_spec_file(spec_path: str) -> tuple[dict[str, Any], Path]:
    """Parse a local YAML spec with the hand-written-YAML loader (``12:00:00`` stays a string).

    Only a regular file of at most :data:`MAX_SPEC_BYTES` is read (never a device or a FIFO:
    see :mod:`hpclab.services.localfiles`), and errors never echo the file's content.
    """
    from hpclab.profiles.loader import yaml_load

    path = Path(spec_path).expanduser()
    if not path.is_absolute():
        raise InvalidArgument(
            f"spec_path must be absolute (got {spec_path!r}); the controller has no working directory of yours",
            details={"spec_path": spec_path},
        )
    try:
        raw = read_regular_file(path, max_bytes=MAX_SPEC_BYTES)
    except FileNotFoundError as exc:
        raise NotFound(f"spec file {path} not found", details={"spec_path": str(path)}) from exc
    except FileTooLarge as exc:
        raise LimitExceeded(
            f"spec file {path} is larger than {MAX_SPEC_BYTES} bytes; an experiment spec is a small "
            "hand-written YAML file",
            details={"spec_path": str(path), "max_bytes": MAX_SPEC_BYTES},
        ) from exc
    except NotRegularFile as exc:
        raise InvalidArgument(f"spec file {exc}", details={"spec_path": str(path)}) from exc
    except OSError as exc:
        raise InvalidArgument(
            f"cannot read spec file {path}: {exc.strerror or type(exc).__name__}",
            details={"spec_path": str(path)},
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidArgument(
            f"spec file {path} is not UTF-8 text (invalid byte at offset {exc.start})",
            details={"spec_path": str(path)},
        ) from exc
    try:
        data = yaml_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        column = mark.column + 1 if mark is not None else None
        where = f" (line {line}, column {column})" if mark is not None else ""
        raise InvalidArgument(
            f"spec file {path} is not valid YAML{where}: {_yaml_problem(exc)}",
            details={"spec_path": str(path), "line": line, "column": column},
        ) from exc
    if not isinstance(data, dict):
        raise InvalidArgument(f"spec file {path} must contain a mapping", details={"spec_path": str(path)})
    return data, path.resolve().parent


def parse_spec(raw: dict[str, Any], *, base_dir: Path | None) -> ExperimentSpec:
    """Validate a raw spec; a relative ``code.project_root`` resolves against ``base_dir``."""
    data = dict(raw)
    code = data.get("code")
    code = dict(code) if isinstance(code, dict) else ({} if code is None else code)
    if isinstance(code, dict):
        root = Path(str(code.get("project_root", "."))).expanduser()
        if not root.is_absolute():
            if base_dir is None:
                raise InvalidArgument(
                    "code.project_root must be an absolute path when the spec is passed inline "
                    "(or pass spec_path so it resolves against the YAML file's directory)",
                    details={"project_root": str(code.get("project_root", "."))},
                )
            root = base_dir / root
        code["project_root"] = str(root.resolve())
        data["code"] = code
    try:
        spec = ExperimentSpec.model_validate(data)
    except ValidationError as exc:
        raise InvalidArgument(
            f"invalid experiment spec: {exc.error_count()} error(s)", details={"errors": _compact_errors(exc)}
        ) from exc
    mx.seed_values(spec)  # integer seeds (raises InvalidArgument)
    return spec


def _normalize(
    *,
    spec: dict[str, Any] | None,
    spec_path: str | None,
    cluster_id: str | None,
    experiment: str | None,
    matrix: dict[str, list[Scalar]] | None,
    extends: str | None,
    add_seeds: list[int] | None,
    add_matrix: dict[str, list[Scalar]] | None,
    derive_from: DeriveFrom | dict[str, Any] | None,
    overrides: list[vd.ValidationOverride] | None,
    validate_remote: bool,
) -> _Request:
    ovs = [
        (o if isinstance(o, vd.ValidationOverride) else vd.ValidationOverride.model_validate(o)).model_dump()
        for o in (overrides or [])
    ]
    if len({o["check"] for o in ovs}) != len(ovs):
        raise InvalidArgument("overrides name the same check twice")
    spec_given = spec is not None or spec_path is not None
    if extends is not None and derive_from is not None:
        raise InvalidArgument("extends and derive_from are mutually exclusive")
    if (extends is not None or derive_from is not None) and (
        spec_given or cluster_id or experiment or matrix is not None
    ):
        raise InvalidArgument(
            "extends/derive_from reuse the stored experiment spec: do not pass spec, spec_path, "
            "cluster_id, experiment or matrix"
        )
    if (add_seeds or add_matrix) and extends is None:
        raise InvalidArgument("add_seeds/add_matrix require extends=<experiment_id>")
    if extends is not None:
        if not add_seeds and not add_matrix:
            raise InvalidArgument("extends needs add_seeds and/or add_matrix")
        for s in add_seeds or []:
            if isinstance(s, bool) or not isinstance(s, int):
                raise InvalidArgument(f"add_seeds must be integers, got {s!r}")
        return _Request(
            mode="extend",
            extends=extends,
            add_seeds=list(add_seeds or []),
            add_matrix={k: list(v) for k, v in (add_matrix or {}).items()},
            overrides=ovs,
            validate_remote=validate_remote,
        )
    if derive_from is not None:
        df = derive_from if isinstance(derive_from, DeriveFrom) else DeriveFrom.model_validate(derive_from)
        return _Request(
            mode="derive",
            derive_from=df.model_dump(mode="json"),
            overrides=ovs,
            validate_remote=validate_remote,
        )
    if spec is not None and spec_path is not None:
        raise InvalidArgument("pass either spec or spec_path, not both")
    if spec is None and spec_path is None:
        raise InvalidArgument("experiment_prepare needs spec, spec_path, extends or derive_from")
    if spec_path is not None:
        raw, base_dir = load_spec_file(spec_path)
    else:
        if not isinstance(spec, dict):
            raise InvalidArgument("spec must be a mapping")
        raw, base_dir = dict(spec), None
    if cluster_id is not None:
        raw["cluster"] = cluster_id
    if experiment is not None:
        raw["name"] = experiment
    if matrix is not None:
        raw["matrix"] = matrix
    parsed = parse_spec(raw, base_dir=base_dir)
    return _Request(
        mode="new", spec=parsed.model_dump(mode="json"), overrides=ovs, validate_remote=validate_remote
    )


# ---------------------------------------------------------------------------
# Target resolution (reads the ledger; no network)
# ---------------------------------------------------------------------------


@dataclass
class _Target:
    mode: Literal["new", "extend", "derive"]
    spec: ExperimentSpec  # the effective (for extend: extended) spec
    experiment_id: str
    plan_kind: str
    existing_row: dict[str, Any] | None = None
    project_root: Path | None = None  # None: reuse ``reuse_code``
    reuse_code: CodeSnapshotRef | None = None
    existing_keys: set[str] = field(default_factory=set)
    base_digests: list[str] = field(default_factory=list)
    added: dict[str, Any] = field(default_factory=dict)
    existing_gpu_hours: float = 0.0
    post_hoc: bool = False
    derive: DeriveFrom | None = None
    source_run: dict[str, Any] | None = None
    source_spec: RunSpec | None = None
    source_plan: ExecutionPlan | None = None


def _load_experiment(ctx: ServiceContext, experiment_id: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
    if row is None:
        raise NotFound(f"experiment {experiment_id} not found", details={"experiment_id": experiment_id})
    return row


def _results_observed(ctx: ServiceContext, experiment_id: str) -> bool:
    """Did a result of the experiment exist already (so new runs are ``added_post_hoc``)?

    The same notion of "observed" as compare's exclusion records: a main run has an
    evaluation, or an attempt of it COMPLETED with valid outputs (its metrics files are
    readable before anyone collects an evaluation: no evaluator declared, or the automatic
    evaluation still queued). Smoke and probe runs do not count: the smoke gate is evaluated
    automatically as soon as it passes, and its few steps are not a result of the experiment.
    """
    row = ctx.db.one(
        "SELECT 1 AS x FROM runs r WHERE r.experiment_id=? AND r.kind='main' AND ("
        "EXISTS (SELECT 1 FROM evaluations e WHERE e.run_id = r.run_id) OR "
        "EXISTS (SELECT 1 FROM attempts a WHERE a.run_id = r.run_id AND a.scheduler_state='COMPLETED' "
        "AND a.result_state='valid' AND a.terminal_at IS NOT NULL)) LIMIT 1",
        (experiment_id,),
    )
    return row is not None


def _existing_runs(ctx: ServiceContext, experiment_id: str) -> list[dict[str, Any]]:
    return ctx.db.all(
        "SELECT r.run_id, r.kind, r.cell_key, r.spec_json, r.current_exec_plan_hash, ep.plan_json "
        "FROM runs r JOIN plans p ON p.plan_id = r.plan_id "
        "LEFT JOIN execution_plans ep ON ep.exec_plan_hash = r.current_exec_plan_hash "
        "WHERE r.experiment_id=? AND p.status='ready'",
        (experiment_id,),
    )


def _planned_gpu_hours(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for r in rows:
        plan = jload(r.get("plan_json"), None)
        if not isinstance(plan, dict):
            continue
        try:
            total += ResourceSpec.model_validate(plan.get("resources") or {}).requested_gpu_hours
        except ValidationError:
            continue
    return total


def _code_digests(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for r in rows:
        d = (jload(r.get("spec_json"), {}) or {}).get("code", {}) or {}
        dg = d.get("digest") if isinstance(d, dict) else None
        if isinstance(dg, str) and dg not in out:
            out.append(dg)
    return out


def _preparing_plans(ctx: ServiceContext, experiment_id: str, *, exclude: str | None) -> list[str]:
    rows = ctx.db.all(
        "SELECT plan_id FROM plans WHERE experiment_id=? AND status='preparing'", (experiment_id,)
    )
    return [r["plan_id"] for r in rows if r["plan_id"] != exclude]


def _resolve_target(
    ctx: ServiceContext, req: _Request, *, experiment_id: str | None = None, plan_id: str | None = None
) -> _Target:
    if req.mode == "new":
        assert req.spec is not None
        spec = ExperimentSpec.model_validate(req.spec)
        return _Target(
            mode="new",
            spec=spec,
            experiment_id=experiment_id or new_id("experiment"),
            plan_kind="matrix",
            project_root=Path(spec.code.project_root),
        )
    if req.mode == "extend":
        assert req.extends is not None
        row = _load_experiment(ctx, req.extends)
        _require_extendable(ctx, row, plan_id)
        base = ExperimentSpec.model_validate(jload(row["spec_json"]))
        data, added = mx.extend_matrix(base, add_seeds=req.add_seeds, add_matrix=req.add_matrix)
        try:
            spec = ExperimentSpec.model_validate(data)
        except ValidationError as exc:
            errs = _compact_errors(exc)
            if any("hard limit" in str(e.get("msg")) or "at most" in str(e.get("msg")) for e in errs):
                raise LimitExceeded(
                    "the extended matrix exceeds the cell limit", details={"errors": errs}
                ) from exc
            raise InvalidArgument("the extended spec is invalid", details={"errors": errs}) from exc
        existing = _existing_runs(ctx, req.extends)
        main_rows = [r for r in existing if r["kind"] == "main"]
        keys = {r["cell_key"] for r in main_rows}
        if not [c for c in mx.expand_spec(spec) if c.key not in keys]:
            raise InvalidArgument(
                "nothing to add: every requested cell already exists in the experiment",
                details={"experiment_id": req.extends, "added": added},
            )
        return _Target(
            mode="extend",
            spec=spec,
            experiment_id=req.extends,
            plan_kind="matrix",
            existing_row=row,
            project_root=Path(spec.code.project_root),
            existing_keys=keys,
            base_digests=_code_digests(main_rows),
            added=added,
            existing_gpu_hours=_planned_gpu_hours(existing),
            post_hoc=_results_observed(ctx, req.extends),
        )
    assert req.derive_from is not None
    df = DeriveFrom.model_validate(req.derive_from)
    run = ctx.db.one("SELECT * FROM runs WHERE run_id=?", (df.run_id,))
    if run is None:
        raise NotFound(f"run {df.run_id} not found", details={"run_id": df.run_id})
    if run["kind"] == "probe":
        raise InvalidArgument("probe runs cannot be derived from")
    row = _load_experiment(ctx, run["experiment_id"])
    try:
        source_spec = RunSpec.model_validate(jload(run["spec_json"]))
    except ValidationError as exc:
        raise PreconditionFailed(
            f"run {df.run_id} has no usable RunSpec", details={"errors": _compact_errors(exc)}
        ) from exc
    ep = ctx.db.one(
        "SELECT plan_json FROM execution_plans WHERE exec_plan_hash=?", (run["current_exec_plan_hash"] or "",)
    )
    if ep is None:
        raise PreconditionFailed(f"run {df.run_id} has no execution plan to derive resources from")
    try:
        source_plan = ExecutionPlan.model_validate(jload(ep["plan_json"]))
    except ValidationError as exc:
        raise PreconditionFailed(
            f"run {df.run_id} has no usable execution plan", details={"errors": _compact_errors(exc)}
        ) from exc
    spec = ExperimentSpec.model_validate(jload(row["spec_json"]))
    existing = _existing_runs(ctx, run["experiment_id"])
    return _Target(
        mode="derive",
        spec=spec.model_copy(update={"smoke": None}),
        experiment_id=run["experiment_id"],
        plan_kind="derived",
        existing_row=row,
        project_root=Path(spec.code.project_root) if df.recapture_code else None,
        reuse_code=None if df.recapture_code else source_spec.code,
        base_digests=[source_spec.code.digest] if df.recapture_code else [],
        existing_gpu_hours=_planned_gpu_hours(existing),
        post_hoc=_results_observed(ctx, run["experiment_id"]),
        derive=df,
        source_run=run,
        source_spec=source_spec,
        source_plan=source_plan,
    )


def _require_extendable(ctx: ServiceContext, row: dict[str, Any], plan_id: str | None) -> None:
    if row["status"] == "invalid":
        raise PreconditionFailed(
            f"experiment {row['experiment_id']} is invalid (it has no runs); prepare a new experiment",
            details={"experiment_id": row["experiment_id"]},
        )
    busy = _preparing_plans(ctx, row["experiment_id"], exclude=plan_id)
    if busy or row["status"] == "preparing":
        raise NotReady(
            f"experiment {row['experiment_id']} has a plan still preparing; extend it when that finishes",
            details={"preparing_plans": busy},
        )


# ---------------------------------------------------------------------------
# Building runs
# ---------------------------------------------------------------------------


def resolve_resources(res: ResourceSpec, cluster: ResolvedCluster | None) -> ResourceSpec:
    """Apply overlay/profile defaults (partition, account, QoS) so the frozen plan is explicit."""
    if cluster is None:
        return res
    updates: dict[str, Any] = {}
    if res.partition is None:
        updates["partition"] = cluster.overlay.default_partition or cluster.profile.default_partition
    if res.account is None:
        updates["account"] = cluster.overlay.account
    if res.qos is None:
        updates["qos"] = cluster.overlay.qos
    updates = {k: v for k, v in updates.items() if v is not None}
    if not updates:
        return res
    return ResourceSpec.model_validate({**res.model_dump(), **updates})


def requeue_for(
    cluster: ResolvedCluster | None, partition: str | None, resilience: ResilienceSpec | None
) -> bool | None:
    """The plan's ``--requeue``/``--no-requeue`` (None = site default).

    A program that is not resumable always gets ``--no-requeue``, on every partition: Slurm
    requeues a batch job not only on preemption but, with the default ``JobRequeue=1``, after
    a node failure too, and a requeued job reruns the script in the *same* attempt dir, where
    a restart from scratch would mix its metrics with the dead segment's and overwrite the
    attempt's outputs. A resumable program gets ``--requeue`` on a preemptible partition
    (unless ``requeue_on_preemption`` is false) and the site default elsewhere: it resumes
    from the attempt's own checkpoint after any requeue."""
    if resilience is None or not resilience.resumable:
        return False
    rule = cluster.profile.partitions.get(partition) if cluster is not None and partition else None
    if rule is None or not rule.preemptible:
        return None
    return resilience.requeue_on_preemption is not False


@dataclass
class _Env:
    cluster: ResolvedCluster | None
    cluster_error: str | None
    remote_root: str | None
    remote_root_error: str | None


def _load_env(ctx: ServiceContext, cluster_id: str) -> _Env:
    try:
        cluster = ctx.clusters.get(cluster_id)
    except NotFound as exc:
        return _Env(None, exc.message, None, None)
    try:
        return _Env(cluster, None, cluster.remote_root(), None)
    except HpcLabError as exc:
        return _Env(cluster, None, None, exc.message)


def _snapshot_path(remote_root: str, code: CodeSnapshotRef) -> str:
    from hpclab.services.snapshot import digest_hex

    return f"{remote_root}/snapshots/{digest_hex(code.digest)}"


def _exec_plan(
    *,
    run_id: str,
    run_spec: RunSpec,
    resources: ResourceSpec,
    env: _Env,
    gate: str | None,
    resilience: ResilienceSpec | None,
) -> ExecutionPlan | None:
    if env.cluster is None or env.remote_root is None:
        return None
    return ExecutionPlan(
        run_id=run_id,
        run_spec_hash=run_spec.spec_hash(),
        cluster_id=env.cluster.cluster_id,
        profile_revision=env.cluster.revision,
        resources=resources,
        remote_root=env.remote_root,
        snapshot_path=_snapshot_path(env.remote_root, run_spec.code),
        gate=gate,
        created_reason="initial",
        resilience=resilience,
        requeue=requeue_for(env.cluster, resources.partition, resilience),
    )


def _inputs(spec: ExperimentSpec) -> list[Any]:
    return [i.model_copy(update={"pin": vd.compute_pin(i)}) for i in spec.inputs]


def _run_spec(
    spec: ExperimentSpec,
    *,
    kind: Literal["main", "smoke"],
    code: CodeSnapshotRef,
    argv: list[str],
    params: dict[str, Scalar],
    cell: dict[str, Scalar],
    seed: int | None,
    derived_from: str | None = None,
) -> RunSpec:
    return RunSpec(
        experiment=spec.name,
        kind=kind,
        code=code,
        entrypoint=spec.entrypoint,
        argv=argv,
        params=params,
        cell=cell,
        seed=seed,
        inputs=_inputs(spec),
        runtime=spec.runtime,
        outputs=spec.outputs,
        evaluation=spec.evaluation,
        derived_from=derived_from,
    )


def _smoke_resources(spec: ExperimentSpec, main: ResourceSpec) -> ResourceSpec:
    smoke = spec.smoke
    assert smoke is not None
    updates: dict[str, Any] = {"time": smoke.time}
    if smoke.gpus is not None:
        updates["gpus"] = smoke.gpus
    if smoke.partition is not None:
        updates["partition"] = smoke.partition
    return ResourceSpec.model_validate({**main.model_dump(), **updates})


def _build_matrix_runs(
    target: _Target, env: _Env, code: CodeSnapshotRef
) -> tuple[list[vd.PlannedRun], str | None]:
    spec = target.spec
    cells = [c for c in mx.expand_spec(spec) if c.key not in target.existing_keys]
    if target.mode == "extend" and not cells:
        raise InvalidArgument(
            "nothing to add: every requested cell already exists in the experiment",
            details={"experiment_id": target.experiment_id, "added": target.added},
        )
    resources = resolve_resources(spec.resources, env.cluster)
    runs: list[vd.PlannedRun] = []
    smoke_error: str | None = None
    smoke_id: str | None = None
    if mx.smoke_enabled(spec):
        assert spec.smoke is not None
        try:
            sc = mx.smoke_cell(cells, spec.smoke)
        except InvalidArgument as exc:
            if target.mode == "extend":
                sc = cells[0]  # smoke.cell may name an existing cell: smoke the first new one
            else:
                smoke_error = exc.message
                sc = None
        if sc is not None:
            smoke_id = new_id("run")
            params = {**mx.cell_params(spec, sc), **spec.smoke.params_override}
            render = mx.render_argv([*spec.entrypoint.argv, *spec.smoke.argv_append], params, sc.seed)
            rs = _run_spec(
                spec, kind="smoke", code=code, argv=render.argv, params=params, cell=sc.cell, seed=sc.seed
            )
            sres = resolve_resources(_smoke_resources(spec, spec.resources), env.cluster)
            res_spec = spec.resilience
            if (
                res_spec is not None
                and res_spec.stop_behavior == "checkpoint_and_exit"
                and res_spec.lead_s >= sres.time_seconds
            ):
                res_spec = None  # the smoke is shorter than the signal lead time
            runs.append(
                vd.PlannedRun(
                    run_id=smoke_id,
                    kind="smoke",
                    label=mx.smoke_label(sc.label),
                    cell=sc.cell,
                    seed=sc.seed,
                    run_spec=rs,
                    resources=sres,
                    exec_plan=_exec_plan(
                        run_id=smoke_id, run_spec=rs, resources=sres, env=env, gate=None, resilience=res_spec
                    ),
                    render=render,
                    added_post_hoc=target.post_hoc,
                )
            )
    for c in cells:
        rid = new_id("run")
        params = mx.cell_params(spec, c)
        render = mx.render_argv(spec.entrypoint.argv, params, c.seed)
        rs = _run_spec(
            spec, kind="main", code=code, argv=render.argv, params=params, cell=c.cell, seed=c.seed
        )
        runs.append(
            vd.PlannedRun(
                run_id=rid,
                kind="main",
                label=c.label,
                cell=c.cell,
                seed=c.seed,
                run_spec=rs,
                resources=resources,
                exec_plan=_exec_plan(
                    run_id=rid,
                    run_spec=rs,
                    resources=resources,
                    env=env,
                    gate=smoke_id,
                    resilience=spec.resilience,
                ),
                render=render,
                added_post_hoc=target.post_hoc,
            )
        )
    return runs, smoke_error


def _build_derived_run(target: _Target, env: _Env, code: CodeSnapshotRef) -> list[vd.PlannedRun]:
    assert target.derive is not None and target.source_spec is not None and target.source_plan is not None
    assert target.source_run is not None
    src = target.source_spec
    ov = target.derive.overrides
    params: dict[str, Scalar] = dict(src.params)
    seed = src.seed
    for k, v in ov.params.items():
        if k == mx.SEED_AXIS:
            seed = mx.as_int_seed(v, what="derive_from.overrides.params.seed")
        else:
            params[k] = v
    template = list(src.entrypoint.argv)
    tail = list(src.argv[len(template) :])
    render = mx.render_argv([*template, *tail, *ov.argv_append], params, seed)
    try:
        resources = ResourceSpec.model_validate({**target.source_plan.resources.model_dump(), **ov.resources})
    except ValidationError as exc:
        raise InvalidArgument(
            "derive_from.overrides.resources is invalid", details={"errors": _compact_errors(exc)}
        ) from exc
    resources = resolve_resources(resources, env.cluster)
    variant = (
        "d"
        + sha256_hex(
            canonical_json(
                {
                    "argv": render.argv,
                    "params": params,
                    "seed": seed,
                    "code": code.digest,
                    "res": resources.model_dump(mode="json"),
                }
            )
        )[:6]
    )
    cell: dict[str, Scalar] = {k: v for k, v in src.cell.items() if k != "variant"}
    for k in list(cell):
        if k in ov.params and k != mx.SEED_AXIS:
            cell[k] = ov.params[k]
    if seed is not None:
        cell[mx.SEED_AXIS] = seed
    else:
        cell.pop(mx.SEED_AXIS, None)
    cell["variant"] = variant
    rid = new_id("run")
    rs = RunSpec(
        experiment=src.experiment,
        kind="main",
        code=code,
        entrypoint=src.entrypoint,
        argv=render.argv,
        params=params,
        cell=cell,
        seed=seed,
        inputs=src.inputs,
        runtime=src.runtime,
        outputs=src.outputs,
        evaluation=src.evaluation,
        derived_from=src_run_id(target),
    )
    base_label = str(target.source_run["label"]).removeprefix("smoke.")
    label = f"{base_label}.derived-{variant}"
    return [
        vd.PlannedRun(
            run_id=rid,
            kind="main",
            label=label,
            cell=cell,
            seed=seed,
            run_spec=rs,
            resources=resources,
            exec_plan=_exec_plan(
                run_id=rid,
                run_spec=rs,
                resources=resources,
                env=env,
                gate=None,
                resilience=target.source_plan.resilience,
            ),
            render=render,
            added_post_hoc=target.post_hoc,
            derived_from=src_run_id(target),
        )
    ]


def src_run_id(target: _Target) -> str:
    assert target.source_run is not None
    return str(target.source_run["run_id"])


# ---------------------------------------------------------------------------
# Materialize: snapshot -> runs -> validation (no DB writes)
# ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    status: Literal["ready", "invalid"]
    runs: list[vd.PlannedRun]
    report: vd.ValidationReport
    advice: vd.ResourceAdvice
    snapshot: SnapshotInfo
    budget: BudgetInfo
    plan_hash: str | None
    smoke_run_id: str | None
    snapshot_digest: str | None
    overrides: list[vd.OverrideOutcome]
    preview: list[PreparedRun]


def _store(ctx: ServiceContext) -> Any:
    from hpclab.services.snapshot import SnapshotStore

    return SnapshotStore(ctx.paths.snapshots)


def _capture(
    ctx: ServiceContext, target: _Target
) -> tuple[CodeSnapshotRef, SnapshotInfo, dict[str, Any], list[str]]:
    """(code ref, info, manifest, excluded paths). Raises the snapshot module's typed errors."""
    from hpclab.services.snapshot import capture_snapshot

    store = _store(ctx)
    if target.project_root is None:
        ref = target.reuse_code
        assert ref is not None
        if not store.has(ref.digest):
            raise PreconditionFailed(
                f"the source run's code snapshot {short(ref.digest)} is not in the local store, so it cannot be "
                "published again; derive with recapture_code=true",
                details={"digest": ref.digest, "hint": "derive with recapture_code=true"},
            )
        manifest = store.manifest(ref.digest)
        info = SnapshotInfo(
            mode="reused", ref=ref, file_count=ref.file_count, total_bytes=ref.total_bytes, reused_store=True
        )
        return ref, info, manifest, []
    result = capture_snapshot(target.project_root, target.spec.code, store, clock=ctx.clock)
    manifest = store.manifest(result.ref.digest)
    info = SnapshotInfo(
        mode="captured",
        ref=result.ref,
        file_count=result.ref.file_count,
        total_bytes=result.ref.total_bytes,
        warnings=list(result.warnings)[:50],
        excluded_count=len(result.excluded),
        reused_store=result.reused,
    )
    return result.ref, info, manifest, list(result.excluded)


def _history(ctx: ServiceContext, cluster_id: str) -> list[vd.HistoryRow]:
    rows = ctx.db.all(
        "SELECT a.attempt_id, a.elapsed_s, a.last_observation_json, r.spec_json, ep.plan_json "
        "FROM attempts a JOIN runs r ON r.run_id = a.run_id "
        "JOIN execution_plans ep ON ep.exec_plan_hash = a.exec_plan_hash "
        "WHERE a.cluster_id=? AND a.scheduler_state='COMPLETED' AND a.result_state='valid' "
        "AND a.elapsed_s IS NOT NULL AND a.elapsed_s > 0 "
        "ORDER BY a.terminal_at DESC, a.attempt_id DESC LIMIT 2000",
        (cluster_id,),
    )
    out: list[vd.HistoryRow] = []
    for r in rows:
        try:
            rs = RunSpec.model_validate(jload(r["spec_json"]))
            plan = jload(r["plan_json"], {}) or {}
            res = ResourceSpec.model_validate(plan.get("resources") or {})
        except (ValidationError, ValueError, AttributeError):
            continue
        obs = jload(r.get("last_observation_json"), None) or {}
        rss = obs.get("max_rss_bytes") if isinstance(obs, dict) else None
        out.append(
            vd.HistoryRow(
                attempt_id=r["attempt_id"],
                family=vd.family_key(rs, res),
                elapsed_s=int(r["elapsed_s"]),
                max_rss_bytes=int(rss) if isinstance(rss, (int, float)) and rss > 0 else None,
            )
        )
    return out


def _remote_findings(
    ctx: ServiceContext,
    env: _Env,
    pc: vd.PlanContext,
    static: list[vd.Finding],
    *,
    plan_id: str,
    validate_remote: bool,
    budget_s: float,
) -> list[vd.Finding]:
    names = ("env_guard_static", "snapshot_bypass", "scheduler_test")
    if not validate_remote:
        return [vd.finding(n, "unchecked", "remote checks skipped (validate_remote=false)") for n in names]
    if env.cluster is None or env.remote_root is None:
        return [
            vd.finding(n, "unchecked", "cluster/remote root unresolved", reason="unresolved") for n in names
        ]
    cluster = env.cluster
    cid = cluster.cluster_id
    deadline = vd.Deadline(budget_s)
    try:
        transport = ctx.clusters.transport(cid)
    except HpcLabError as exc:
        return [
            vd.finding(
                n, "unchecked", f"no transport for {cid}: {exc.message}", error_type=type(exc).__name__
            )
            for n in names
        ]
    out: list[vd.Finding] = []
    main = pc.main_runs or pc.runs
    gpus = max((r.resources.total_gpus for r in pc.runs), default=0)
    gpu_types: list[str] = []
    for r in main:
        if r.resources.gpus <= 0:
            continue
        if r.resources.gpu_type:
            gpu_types.append(r.resources.gpu_type)
        else:
            rule = cluster.profile.partitions.get(r.resources.partition or "")
            gpu_types.extend(rule.gpu_types if rule else [])
    project = _project_names(ctx, pc.manifest)
    try:
        out.extend(
            vd.check_env_guard_static(
                transport=transport,
                cluster=cluster,
                runtime_venv=pc.spec.runtime.venv,
                gpus=gpus,
                gpu_types=sorted(set(gpu_types)),
                project=project,
                now=ctx.clock.now(),
                deadline=deadline,
            )
        )
    except Exception as exc:
        log.warning("env_guard_static failed unexpectedly", exc_info=True)
        out.extend(vd.finding(n, "unchecked", f"check failed: {exc!r}") for n in names[:2])
    render_ok = any(f.check == "script_render" and f.status == "pass" for f in static)
    rep = next((r for r in main if r.exec_plan is not None), None)
    if not render_ok or rep is None:
        out.append(vd.finding("scheduler_test", "unchecked", "skipped: the job script does not render"))
        return out
    try:
        scheduler = ctx.clusters.scheduler(cid)
        out.append(
            vd.check_scheduler_test(
                transport=transport,
                scheduler=scheduler,
                cluster=cluster,
                run=rep,
                remote_root=env.remote_root,
                plan_id=plan_id,
                deadline=deadline,
            )
        )
    except Exception as exc:
        log.warning("scheduler_test failed unexpectedly", exc_info=True)
        out.append(vd.finding("scheduler_test", "unchecked", f"check failed: {exc!r}"))
    return out


def _project_names(ctx: ServiceContext, manifest: dict[str, Any] | None) -> set[str]:
    files = vd.manifest_paths(manifest)
    text = None
    if "pyproject.toml" in files:
        text = vd.read_local_blob(ctx.paths.snapshots, files["pyproject.toml"])
    return vd.project_names(manifest, text)


def _preview(runs: list[vd.PlannedRun], *, created: bool) -> list[PreparedRun]:
    return [
        PreparedRun(
            run_id=r.run_id if created else None,
            label=r.label,
            kind=r.kind,
            cell=dict(r.cell),
            seed=r.seed,
            exec_plan_hash=r.exec_plan.plan_hash() if (created and r.exec_plan is not None) else None,
            requested_gpu_hours=round(r.requested_gpu_hours, 6),
            added_post_hoc=r.added_post_hoc,
            derived_from=r.derived_from,
        )
        for r in runs
    ]


def compute_plan_hash(kind: str, runs: list[vd.PlannedRun]) -> str:
    """Digest over (run id, kind, run spec hash, execution plan hash) in plan order."""
    return digest(
        {
            "kind": kind,
            "runs": [
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "run_spec_hash": r.run_spec.spec_hash(),
                    "exec_plan_hash": r.exec_plan.plan_hash() if r.exec_plan else None,
                }
                for r in runs
            ],
        }
    )


def _materialize(
    ctx: ServiceContext, req: _Request, target: _Target, env: _Env, *, plan_id: str, remote_budget_s: float
) -> _Outcome:
    try:
        code, snap_info, manifest, excluded = _capture(ctx, target)
    except HpcLabError as exc:
        exc.details.setdefault("stage", "snapshot")  # background failures name the right check
        raise
    if target.mode == "derive":
        runs, smoke_error = _build_derived_run(target, env, code), None
    else:
        runs, smoke_error = _build_matrix_runs(target, env, code)
    pc = vd.PlanContext(
        spec=target.spec,
        cluster=env.cluster,
        runs=runs,
        now=ctx.clock.now(),
        cluster_error=env.cluster_error,
        remote_root=env.remote_root,
        remote_root_error=env.remote_root_error,
        manifest=manifest,
        snapshot_warnings=snap_info.warnings,
        snapshot_excluded=excluded,
        git_dirty=code.git_dirty,
        smoke_error=smoke_error,
        existing_gpu_hours=target.existing_gpu_hours,
        base_digests=target.base_digests,
        base_manifests=_base_manifests(ctx, target.base_digests),
    )
    findings = vd.run_static_checks(pc, drift=bool(target.base_digests) or target.mode == "extend")
    if target.post_hoc:
        findings.append(
            vd.finding(
                "post_hoc",
                "warn",
                "results of this experiment already exist (a main run was evaluated or completed with "
                "valid outputs): the new runs are flagged added_post_hoc and comparisons disclose them",
                severity="info",
            )
        )
    try:
        advice = vd.advise_resources(runs, _history(ctx, env.cluster.cluster_id)) if env.cluster else None
    except Exception as exc:
        log.warning("resource advice failed", exc_info=True)
        advice = vd.ResourceAdvice(status="unavailable", note=f"ledger history unavailable: {exc!r}")
    if advice is None:
        advice = vd.ResourceAdvice(status="unavailable", note="cluster not registered")
    findings.append(vd.check_resource_advice(advice))
    findings.extend(
        _remote_findings(
            ctx,
            env,
            pc,
            findings,
            plan_id=plan_id,
            validate_remote=req.validate_remote,
            budget_s=remote_budget_s,
        )
    )
    findings, outcomes = vd.apply_overrides(findings, req.override_models())
    report = vd.build_report(findings)
    status: Literal["ready", "invalid"] = "invalid" if report.blocking else "ready"
    total_hours = sum(r.requested_gpu_hours for r in runs)
    smoke_hours = sum(r.requested_gpu_hours for r in runs if r.kind == "smoke")
    budget = BudgetInfo(
        requested_gpu_hours_total=round(total_hours, 6),
        smoke_gpu_hours=round(smoke_hours, 6),
        budget_gpu_hours=target.spec.budget.gpu_hours,
        existing_planned_gpu_hours=round(target.existing_gpu_hours, 6),
    )
    if status == "ready" and any(r.exec_plan is None for r in runs):  # defensive: never ready without plans
        status = "invalid"
    smoke_id = next((r.run_id for r in runs if r.kind == "smoke"), None)
    return _Outcome(
        status=status,
        runs=runs,
        report=report,
        advice=advice,
        snapshot=snap_info,
        budget=budget,
        plan_hash=compute_plan_hash(target.plan_kind, runs) if status == "ready" else None,
        smoke_run_id=smoke_id if status == "ready" else None,
        snapshot_digest=code.digest,
        overrides=outcomes,
        preview=_preview(runs, created=status == "ready"),
    )


def _base_manifests(ctx: ServiceContext, digests: list[str]) -> dict[str, dict[str, Any] | None]:
    store = _store(ctx)
    out: dict[str, dict[str, Any] | None] = {}
    for d in digests[:5]:
        try:
            out[d] = store.manifest(d)
        except HpcLabError:
            out[d] = None
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _validation_doc(
    report: vd.ValidationReport,
    *,
    advice: vd.ResourceAdvice | None,
    budget: BudgetInfo,
    snapshot: SnapshotInfo | None,
    overrides: list[OverrideRecord],
    preview: list[PreparedRun] | None,
) -> dict[str, Any]:
    doc: dict[str, Any] = report.model_dump(mode="json")
    doc["resource_advice"] = advice.model_dump(mode="json") if advice else None
    doc["budget"] = budget.model_dump(mode="json")
    doc["snapshot"] = snapshot.model_dump(mode="json") if snapshot else None
    doc["overrides"] = [o.model_dump(mode="json") for o in overrides]
    if preview is not None:
        doc["planned_runs"] = [p.model_dump(mode="json") for p in preview]
    return doc


def _override_records(outcomes: list[vd.OverrideOutcome]) -> list[OverrideRecord]:
    return [OverrideRecord(check=o.check, reason=o.reason, applied=o.applied, note=o.note) for o in outcomes]


def _insert_experiment(tx: Tx, experiment_id: str, spec: ExperimentSpec, status: str) -> None:
    data = spec.model_dump(mode="json")
    tx.insert(
        "experiments",
        {
            "experiment_id": experiment_id,
            "name": spec.name,
            "cluster_id": spec.cluster,
            "question": spec.question,
            "spec_json": jdump(data),
            "spec_hash": digest(data),
            "project_root": spec.code.project_root,
            "budget_gpu_hours": spec.budget.gpu_hours,
            "max_parallel": spec.budget.max_parallel,
            "status": status,
            "tags_json": jdump(list(spec.tags)),
            "version": 1,
            "created_at": tx.now,
            "updated_at": tx.now,
        },
    )
    tx.event(
        "experiment",
        experiment_id,
        "experiment.created",
        {"name": spec.name, "cluster_id": spec.cluster, "spec_hash": digest(data), "status": status},
        entity_version=1,
    )


def _insert_runs(tx: Tx, *, experiment_id: str, plan_id: str, runs: list[vd.PlannedRun]) -> None:
    for r in runs:
        assert r.exec_plan is not None
        ep_hash = r.exec_plan.plan_hash()
        spec_json = r.run_spec.model_dump(mode="json")
        tx.insert(
            "runs",
            {
                "run_id": r.run_id,
                "experiment_id": experiment_id,
                "plan_id": plan_id,
                "kind": r.kind,
                "label": r.label,
                "cell_key": mx.cell_key(r.cell),
                "seed": r.seed,
                "spec_json": jdump(spec_json),
                "spec_hash": r.run_spec.spec_hash(),
                "derived_from": r.derived_from,
                "current_exec_plan_hash": ep_hash,
                "status": RunStatus.PREPARED.value,
                "added_post_hoc": 1 if r.added_post_hoc else 0,
                "version": 1,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.insert(
            "execution_plans",
            {
                "exec_plan_hash": ep_hash,
                "run_id": r.run_id,
                "plan_json": jdump(r.exec_plan.model_dump(mode="json")),
                "status": "ready",
                "validation_json": "{}",
                "created_at": tx.now,
            },
        )
        tx.event(
            "run",
            r.run_id,
            "run.created",
            {
                "plan_id": plan_id,
                "kind": r.kind,
                "label": r.label,
                "spec_hash": r.run_spec.spec_hash(),
                "exec_plan_hash": ep_hash,
                "added_post_hoc": r.added_post_hoc,
                "derived_from": r.derived_from,
                "gate": r.exec_plan.gate,
            },
            entity_version=1,
        )


def _record_overrides(tx: Tx, plan_id: str, outcomes: list[vd.OverrideOutcome]) -> None:
    for o in outcomes:
        tx.event(
            "plan",
            plan_id,
            "validation.override" if o.applied else "validation.override_rejected",
            {
                "check": o.check,
                "reason": o.reason,
                "applied": o.applied,
                "note": o.note,
                "original_status": o.original_status,
                "original_severity": o.original_severity,
            },
        )


def _check_no_collisions(tx: Tx, target: _Target, runs: list[vd.PlannedRun]) -> None:
    """Extensions: another prepare may have extended the experiment meanwhile (checked in the tx).

    The extension was built and validated against the spec read at resolve time and will write
    that spec + its additions back; if another extension committed in between, writing it would
    silently drop the other one's seeds/values from the spec (runs the spec no longer lists) and
    the budget check would have ignored its GPU-hours. Conflict: the caller re-validates.
    """
    if target.mode != "extend":
        return
    assert target.existing_row is not None
    now = tx.one("SELECT spec_hash FROM experiments WHERE experiment_id=?", (target.experiment_id,))
    if now is None or now["spec_hash"] != target.existing_row["spec_hash"]:
        raise VersionConflict(
            f"experiment {target.experiment_id} was extended concurrently; retry the extension so it is "
            "validated against the current spec",
            details={
                "experiment_id": target.experiment_id,
                "validated_spec_hash": target.existing_row["spec_hash"],
                "current_spec_hash": now["spec_hash"] if now else None,
            },
        )
    rows = tx.all(
        "SELECT r.cell_key FROM runs r JOIN plans p ON p.plan_id = r.plan_id "
        "WHERE r.experiment_id=? AND r.kind='main' AND p.status='ready'",
        (target.experiment_id,),
    )
    existing = {r["cell_key"] for r in rows}
    clash = [r.label for r in runs if r.kind == "main" and mx.cell_key(r.cell) in existing]
    if clash:
        raise VersionConflict(
            "the experiment gained some of these cells concurrently; retry the extension",
            details={"cells": clash[:20]},
        )


def _apply_extension(tx: Tx, target: _Target, plan_id: str) -> None:
    if target.mode != "extend" or target.existing_row is None:
        return
    row = tx.one("SELECT version, spec_hash FROM experiments WHERE experiment_id=?", (target.experiment_id,))
    assert row is not None
    data = target.spec.model_dump(mode="json")
    version = tx.update(
        "experiments",
        "experiment_id",
        target.experiment_id,
        {"spec_json": jdump(data), "spec_hash": digest(data)},
        expected_version=int(row["version"]),
    )
    tx.event(
        "experiment",
        target.experiment_id,
        "experiment.extended",
        {
            "plan_id": plan_id,
            "added": target.added,
            "previous_spec_hash": row["spec_hash"],
            "spec_hash": digest(data),
        },
        entity_version=version,
    )


def _finalize_plan_tx(
    tx: Tx,
    *,
    target: _Target,
    outcome: _Outcome,
    plan_id: str,
    operation_id: str | None,
    create_plan: bool,
) -> dict[str, Any]:
    """Write the ready/invalid plan (+ runs when ready). Returns the plan values written."""
    overrides = _override_records(outcome.overrides)
    doc = _validation_doc(
        outcome.report,
        advice=outcome.advice,
        budget=outcome.budget,
        snapshot=outcome.snapshot,
        overrides=overrides,
        preview=None if outcome.status == "ready" else outcome.preview,
    )
    values = {
        "status": outcome.status,
        "plan_hash": outcome.plan_hash,
        "snapshot_digest": outcome.snapshot_digest,
        "validation_json": jdump(doc),
        "smoke_run_id": outcome.smoke_run_id,
    }
    if create_plan:
        tx.insert(
            "plans",
            {
                "plan_id": plan_id,
                "experiment_id": target.experiment_id,
                "kind": target.plan_kind,
                "operation_id": operation_id,
                "created_at": tx.now,
                "updated_at": tx.now,
                **values,
            },
        )
        tx.event("plan", plan_id, "plan.created", {"kind": target.plan_kind, "status": outcome.status})
    else:
        cur = tx.execute(
            "UPDATE plans SET status=?, plan_hash=?, snapshot_digest=?, validation_json=?, smoke_run_id=?, "
            "updated_at=? WHERE plan_id=? AND status='preparing'",
            (
                values["status"],
                values["plan_hash"],
                values["snapshot_digest"],
                values["validation_json"],
                values["smoke_run_id"],
                tx.now,
                plan_id,
            ),
        )
        if cur.rowcount != 1:
            raise VersionConflict(f"plan {plan_id} is no longer preparing")
    if outcome.status == "ready":
        _check_no_collisions(tx, target, outcome.runs)
        _insert_runs(tx, experiment_id=target.experiment_id, plan_id=plan_id, runs=outcome.runs)
        _apply_extension(tx, target, plan_id)
    _record_overrides(tx, plan_id, outcome.overrides)
    tx.event(
        "plan",
        plan_id,
        f"plan.{outcome.status}",
        {
            "plan_hash": outcome.plan_hash,
            "run_count": len(outcome.runs),
            "validation_status": outcome.report.status,
            "blocking": outcome.report.blocking,
            "snapshot_digest": outcome.snapshot_digest,
        },
    )
    return values


REMOTE_CHECKS = ("env_guard_static", "snapshot_bypass", "scheduler_test")


def prepare_coverage(
    report: vd.ValidationReport, *, cluster_id: str, validate_remote: bool, observed_at: str | None
) -> Coverage:
    """Which sources validation consulted: the local snapshot/ledger and the cluster (remote
    checks). A cluster that could not be reached makes coverage incomplete (never a silent pass)."""
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=observed_at)
    snap = next((f for f in report.findings if f.check == "code_snapshot"), None)
    if snap is not None:
        cov.add("snapshot", "ok" if snap.status != "unchecked" else "skipped", observed_at=observed_at)
    remote = [f for f in report.findings if f.check in REMOTE_CHECKS]
    name = f"cluster:{cluster_id}"
    if not validate_remote:
        cov.add(name, "skipped", detail="validate_remote=false: remote checks not run")
    elif not remote:
        cov.add(name, "skipped", detail="remote checks pending")
    elif any(f.remediation and f.remediation.startswith("hpclab auth") for f in remote):
        cov.add(name, "unavailable", detail="SSH authentication required")
    elif any("error_type" in f.evidence for f in remote):
        failed = all(f.status == "unchecked" for f in remote)
        cov.add(
            name,
            "unavailable" if failed else "partial",
            observed_at=observed_at,
            detail="remote checks could not reach the cluster"
            if failed
            else "some remote checks could not run",
        )
    elif any(f.evidence.get("reason") == "unresolved" for f in remote):
        cov.add(name, "unavailable", detail="cluster or remote root unresolved")
    elif all(f.status == "unchecked" for f in remote):
        cov.add(name, "skipped", detail="no remote check was applicable")
    else:
        cov.add(name, "ok", observed_at=observed_at)
    return cov


def _data_from_outcome(
    *,
    target: _Target,
    outcome: _Outcome,
    plan_id: str,
    warnings: list[str],
    validate_remote: bool,
    now: str | None,
) -> PrepareData:
    runs = outcome.preview[:MAX_RUNS_RETURNED]
    return PrepareData(
        experiment_id=target.experiment_id,
        plan_id=plan_id,
        plan_ref=plan_ref(plan_id),
        plan_hash=outcome.plan_hash,
        status=outcome.status,
        kind=target.plan_kind,
        runs=runs,
        runs_total=len(outcome.preview),
        smoke_run_id=outcome.smoke_run_id,
        validation=outcome.report,
        resource_advice=outcome.advice,
        snapshot=outcome.snapshot,
        budget=outcome.budget,
        overrides=_override_records(outcome.overrides),
        warnings=warnings,
        coverage=prepare_coverage(
            outcome.report, cluster_id=target.spec.cluster, validate_remote=validate_remote, observed_at=now
        ),
        observed_at=now,
    )


def _override_warnings(outcomes: list[vd.OverrideOutcome]) -> list[str]:
    return [f"override for {o.check!r} not applied: {o.note}" for o in outcomes if not o.applied]


# ---------------------------------------------------------------------------
# experiment_prepare
# ---------------------------------------------------------------------------


def experiment_prepare(
    ctx: ServiceContext,
    *,
    idempotency_key: str,
    spec: dict[str, Any] | None = None,
    spec_path: str | None = None,
    cluster_id: str | None = None,
    experiment: str | None = None,
    matrix: dict[str, list[Scalar]] | None = None,
    extends: str | None = None,
    add_seeds: list[int] | None = None,
    add_matrix: dict[str, list[Scalar]] | None = None,
    derive_from: DeriveFrom | None = None,
    overrides: list[vd.ValidationOverride] | None = None,
    validate_remote: bool = True,
) -> PrepareData:
    """Freeze an experiment into a validated plan of Runs + ExecutionPlans (never submits).

    ``spec`` (inline, absolute ``code.project_root``) or ``spec_path`` (local YAML; a relative
    project root resolves against the file's directory); ``cluster_id``/``experiment``/
    ``matrix`` override the spec. ``extends=<experiment_id>`` + ``add_seeds``/``add_matrix``
    adds only the *new* cells as a new plan of the same experiment. ``derive_from`` creates one
    derived Run (new RunSpec) from an existing Run. ``overrides=[{check, reason}]`` downgrades
    blocking findings to warnings (recorded as events). Large snapshots are prepared in the
    background (``status='preparing'`` + ``operation_ref``); poll the plan or operation.
    Same ``idempotency_key`` + same request returns the stored response.
    """
    if not idempotency_key or len(idempotency_key) > 200:
        raise InvalidArgument("idempotency_key must be 1-200 characters")
    req = _normalize(
        spec=spec,
        spec_path=spec_path,
        cluster_id=cluster_id,
        experiment=experiment,
        matrix=matrix,
        extends=extends,
        add_seeds=add_seeds,
        add_matrix=add_matrix,
        derive_from=derive_from,
        overrides=overrides,
        validate_remote=validate_remote,
    )
    payload_hash = req.payload_hash()
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
    if stored is not None:
        return PrepareData.model_validate(stored)
    target = _resolve_target(ctx, req)
    env = _load_env(ctx, target.spec.cluster)
    plan_id = new_id("plan")
    if target.project_root is not None:
        from hpclab.services.snapshot import scan_project

        scan = scan_project(target.project_root, target.spec.code)
        if scan.total_bytes > ctx.settings.prepare_sync_max_bytes:
            return _start_background(ctx, req, target, scan, plan_id, idempotency_key, payload_hash)
    outcome = _materialize(
        ctx, req, target, env, plan_id=plan_id, remote_budget_s=ctx.settings.request_deadline_s
    )
    warnings = _override_warnings(outcome.overrides)
    data = _data_from_outcome(
        target=target,
        outcome=outcome,
        plan_id=plan_id,
        warnings=warnings,
        validate_remote=req.validate_remote,
        now=iso(ctx.clock.now()),
    )
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
        if again is not None:  # a concurrent identical request won; its work stands
            return PrepareData.model_validate(again)
        if target.mode == "new":
            _insert_experiment(tx, target.experiment_id, target.spec, outcome.status)
        _finalize_plan_tx(
            tx, target=target, outcome=outcome, plan_id=plan_id, operation_id=None, create_plan=True
        )
        tx.idem_store(ctx.principal, TOOL, idempotency_key, payload_hash, data.model_dump(mode="json"))
    log.info("prepared plan %s (%s, %d runs)", plan_id, outcome.status, len(outcome.runs))
    return data


def _start_background(
    ctx: ServiceContext,
    req: _Request,
    target: _Target,
    scan: Any,
    plan_id: str,
    key: str,
    payload_hash: str,
) -> PrepareData:
    """Record intent (experiment/plan 'preparing' + operation + outbox) and return the receipt."""
    try:
        preview_cells = mx.expand_spec(target.spec) if target.mode != "derive" else []
    except HpcLabError:
        preview_cells = []
    preview_cells = [c for c in preview_cells if c.key not in target.existing_keys]
    cluster = _load_env(ctx, target.spec.cluster).cluster
    res = resolve_resources(target.spec.resources, cluster)
    if target.mode == "derive" and target.source_plan is not None and target.source_run is not None:
        return_preview = _derive_preview(target, cluster)
    else:
        return_preview = None
    preview = [
        PreparedRun(
            run_id=None,
            label=c.label,
            kind="main",
            cell=c.cell,
            seed=c.seed,
            exec_plan_hash=None,
            requested_gpu_hours=round(res.requested_gpu_hours, 6),
            added_post_hoc=target.post_hoc,
        )
        for c in preview_cells
    ]
    # Totals from unrounded hours, exactly as the ready plan will compute them.
    main_hours = res.requested_gpu_hours * len(preview_cells)
    smoke_hours = 0.0
    smoke = _smoke_preview(target, preview_cells, cluster)
    if smoke is not None:  # the ready plan will have it: count its GPU-hours now, not 0
        preview.insert(0, smoke[0])
        smoke_hours = smoke[1]
    if return_preview is not None:
        preview = [return_preview]
        main_hours = return_preview.requested_gpu_hours
    snapshot = SnapshotInfo(
        mode="background",
        file_count=scan.file_count,
        total_bytes=scan.total_bytes,
        warnings=list(scan.warnings)[:50],
        excluded_count=len(scan.excluded),
    )
    budget = BudgetInfo(
        requested_gpu_hours_total=round(main_hours + smoke_hours, 6),
        smoke_gpu_hours=round(smoke_hours, 6),
        budget_gpu_hours=target.spec.budget.gpu_hours,
        existing_planned_gpu_hours=round(target.existing_gpu_hours, 6),
    )
    report = vd.build_report([], pending=True)
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, key, payload_hash)
        if again is not None:
            return PrepareData.model_validate(again)
        if target.mode == "new":
            _insert_experiment(tx, target.experiment_id, target.spec, "preparing")
        if target.mode == "extend":
            # Re-checked in the creating transaction: two preparing extensions of one experiment
            # would each wait for the other (NotReady) until both fail.
            busy = [
                r["plan_id"]
                for r in tx.all(
                    "SELECT plan_id FROM plans WHERE experiment_id=? AND status='preparing'",
                    (target.experiment_id,),
                )
            ]
            if busy:
                raise NotReady(
                    f"experiment {target.experiment_id} has a plan still preparing; extend it when that "
                    "finishes",
                    details={"preparing_plans": busy},
                )
        op_id = tx.create_operation(
            "prepare", target_type="plan", target_id=plan_id, stage="snapshot", poll_after_ms=3000
        )
        doc = _validation_doc(
            report, advice=None, budget=budget, snapshot=snapshot, overrides=[], preview=preview
        )
        tx.insert(
            "plans",
            {
                "plan_id": plan_id,
                "experiment_id": target.experiment_id,
                "kind": target.plan_kind,
                "status": "preparing",
                "plan_hash": None,
                "snapshot_digest": None,
                "validation_json": jdump(doc),
                "smoke_run_id": None,
                "operation_id": op_id,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.event(
            "plan",
            plan_id,
            "plan.created",
            {
                "kind": target.plan_kind,
                "status": "preparing",
                "experiment_id": target.experiment_id,
                "operation_id": op_id,
                "request": req.payload(),
                "scan": {"file_count": scan.file_count, "total_bytes": scan.total_bytes},
            },
        )
        tx.enqueue(
            OUTBOX_KIND,
            operation_id=op_id,
            target_id=plan_id,
            payload={"plan_id": plan_id},
            dedupe_key=f"prepare:{plan_id}",
        )
        data = PrepareData(
            experiment_id=target.experiment_id,
            plan_id=plan_id,
            plan_ref=plan_ref(plan_id),
            plan_hash=None,
            status="preparing",
            kind=target.plan_kind,
            runs=preview[:MAX_RUNS_RETURNED],
            runs_total=len(preview),
            smoke_run_id=None,
            validation=report,
            resource_advice=None,
            snapshot=snapshot,
            budget=budget,
            operation_id=op_id,
            operation_ref=operation_ref(op_id),
            warnings=[
                f"the code snapshot is {scan.total_bytes} bytes (> {ctx.settings.prepare_sync_max_bytes}): it is "
                "captured in the background; the plan has no hash and cannot be submitted until it is ready",
                f"the code is captured when the background step runs, not now: do not edit files or switch "
                f"branches in {target.project_root} until the plan is ready",
            ],
            coverage=_preparing_coverage(tx.now),
            observed_at=tx.now,
        )
        tx.idem_store(ctx.principal, TOOL, key, payload_hash, data.model_dump(mode="json"))
    return data


def _smoke_preview(
    target: _Target, cells: list[mx.Cell], cluster: ResolvedCluster | None
) -> tuple[PreparedRun, float] | None:
    """The smoke Run a background matrix plan will get (same choice as ``_build_matrix_runs``)
    and its requested GPU-hours; None when there is none or its cell is invalid (validation
    will say so)."""
    spec = target.spec
    if target.mode == "derive" or not cells or not mx.smoke_enabled(spec):
        return None
    assert spec.smoke is not None
    try:
        sc = mx.smoke_cell(cells, spec.smoke)
    except InvalidArgument:
        if target.mode != "extend":
            return None
        sc = cells[0]
    try:
        sres = resolve_resources(_smoke_resources(spec, spec.resources), cluster)
    except ValidationError:
        return None
    run = PreparedRun(
        run_id=None,
        label=mx.smoke_label(sc.label),
        kind="smoke",
        cell=sc.cell,
        seed=sc.seed,
        exec_plan_hash=None,
        requested_gpu_hours=round(sres.requested_gpu_hours, 6),
        added_post_hoc=target.post_hoc,
    )
    return run, sres.requested_gpu_hours


def _preparing_coverage(now: str) -> Coverage:
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=now)
    cov.add("snapshot", "skipped", detail="captured in the background")
    cov.notes.append("validation is pending: poll the operation or plan for the result")
    return cov


def _derive_preview(target: _Target, cluster: ResolvedCluster | None) -> PreparedRun:
    assert target.source_plan is not None and target.source_run is not None and target.derive is not None
    try:
        res = ResourceSpec.model_validate(
            {**target.source_plan.resources.model_dump(), **target.derive.overrides.resources}
        )
    except ValidationError as exc:
        raise InvalidArgument(
            "derive_from.overrides.resources is invalid", details={"errors": _compact_errors(exc)}
        ) from exc
    res = resolve_resources(res, cluster)
    return PreparedRun(
        run_id=None,
        label=f"{str(target.source_run['label']).removeprefix('smoke.')}.derived",
        kind="main",
        cell={},
        seed=target.source_spec.seed if target.source_spec else None,
        exec_plan_hash=None,
        requested_gpu_hours=round(res.requested_gpu_hours, 6),
        added_post_hoc=target.post_hoc,
        derived_from=str(target.source_run["run_id"]),
    )


# ---------------------------------------------------------------------------
# Background handler: prepare.snapshot
# ---------------------------------------------------------------------------


def handle_prepare_snapshot(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """Capture + validate a 'preparing' plan and make it ready/invalid (crash-safe: re-entry on a
    plan that is no longer 'preparing' is a no-op; the final write is compare-and-set)."""
    plan_id = str(item.payload.get("plan_id") or item.target_id or "")
    plan = ctx.db.one("SELECT * FROM plans WHERE plan_id=?", (plan_id,))
    if plan is None:
        log.warning("prepare.snapshot: plan %s not found", plan_id)
        return HandlerResult.done("plan not found")
    if plan["status"] != "preparing":
        return HandlerResult.done("already finished")
    ev = ctx.db.one(
        "SELECT payload_json FROM events WHERE entity_type='plan' AND entity_id=? AND event_type='plan.created' "
        "ORDER BY seq LIMIT 1",
        (plan_id,),
    )
    payload = jload(ev["payload_json"], {}) if ev else {}
    op_id = plan["operation_id"] or item.operation_id
    if not payload.get("request"):
        _fail_background(
            ctx, plan, op_id, "the prepare request of this plan was not recorded", code="internal"
        )
        return HandlerResult.done("no request")
    req = _Request.from_payload(payload["request"])
    if op_id:
        with ctx.db.tx(source="worker") as tx:
            tx.set_operation(op_id, state="running", stage="snapshot")
    # Every failure below ends in ready/invalid or a retry the handler itself bounds: an item the
    # runner dead-letters would leave the plan 'preparing' forever, and a preparing plan blocks
    # every later extension of the experiment.
    try:
        target = _resolve_target(ctx, req, experiment_id=plan["experiment_id"], plan_id=plan_id)
        env = _load_env(ctx, target.spec.cluster)
        outcome = _materialize(
            ctx, req, target, env, plan_id=plan_id, remote_budget_s=BACKGROUND_REMOTE_BUDGET_S
        )
        with ctx.db.tx(source="worker") as tx:
            cur = tx.one("SELECT status FROM plans WHERE plan_id=?", (plan_id,))
            if cur is None or cur["status"] != "preparing":
                return HandlerResult.done("finished concurrently")
            _finalize_plan_tx(
                tx, target=target, outcome=outcome, plan_id=plan_id, operation_id=op_id, create_plan=False
            )
            if target.mode == "new":
                _set_experiment_status(tx, target.experiment_id, outcome.status)
            if op_id:
                tx.set_operation(
                    op_id,
                    state="succeeded",
                    stage="done",
                    result={
                        "plan_id": plan_id,
                        "plan_ref": plan_ref(plan_id),
                        "status": outcome.status,
                        "plan_hash": outcome.plan_hash,
                        "run_count": len(outcome.runs),
                        "blocking": outcome.report.blocking,
                    },
                )
    except VersionConflict as exc:
        # A concurrent extension changed the experiment while this plan was validated: re-resolve
        # (the new cells may now exist -> 'nothing to add' -> invalid) and re-validate.
        if item.tries < BACKGROUND_MAX_TRIES:
            log.info("prepare.snapshot %s: %s; re-validating", plan_id, exc.message)
            return HandlerResult.retry(BACKGROUND_CONFLICT_RETRY_S, note=exc.message)
        _fail_background(ctx, plan, op_id, exc.message, code=str(exc.code), details=exc.details)
        return HandlerResult.done(exc.message)
    except HpcLabError as exc:
        if exc.retryable and item.tries < BACKGROUND_MAX_TRIES:
            raise
        _fail_background(ctx, plan, op_id, exc.message, code=str(exc.code), details=exc.details)
        return HandlerResult.done(exc.message)
    except Exception as exc:
        if item.tries < BACKGROUND_MAX_CRASHES:
            raise
        log.exception("prepare.snapshot for %s failed", plan_id)
        _fail_background(ctx, plan, op_id, f"preparation crashed: {exc!r}", code="internal")
        return HandlerResult.done("crashed")
    return HandlerResult.done(outcome.status)


def _set_experiment_status(tx: Tx, experiment_id: str, status: str) -> None:
    row = tx.one("SELECT status, version FROM experiments WHERE experiment_id=?", (experiment_id,))
    if row is None or row["status"] != "preparing":
        return
    version = tx.update(
        "experiments",
        "experiment_id",
        experiment_id,
        {"status": status},
        expected_version=int(row["version"]),
    )
    tx.event(
        "experiment", experiment_id, "experiment.status_changed", {"status": status}, entity_version=version
    )


def _fail_background(
    ctx: ServiceContext,
    plan: dict[str, Any],
    op_id: str | None,
    message: str,
    *,
    code: str,
    details: dict[str, Any] | None = None,
) -> None:
    # A snapshot error is a code_snapshot finding; anything else (a lost race, a crash) must not
    # send the user to fix their code.
    check = "code_snapshot" if (details or {}).get("stage") == "snapshot" else "background_prepare"
    f = vd.finding(check, "fail", message, details=details or None)
    report = vd.build_report([f])
    old = jload(plan["validation_json"], {}) or {}
    doc = report.model_dump(mode="json")
    for k in ("budget", "snapshot", "planned_runs"):
        if k in old:
            doc[k] = old[k]
    with ctx.db.tx(source="worker") as tx:
        cur = tx.execute(
            "UPDATE plans SET status='invalid', validation_json=?, updated_at=? WHERE plan_id=? AND status='preparing'",
            (jdump(doc), tx.now, plan["plan_id"]),
        )
        if cur.rowcount != 1:
            return
        tx.event("plan", plan["plan_id"], "plan.invalid", {"error": message, "code": code})
        _set_experiment_status(tx, plan["experiment_id"], "invalid")
        if op_id:
            tx.set_operation(
                op_id,
                state="failed",
                stage="snapshot",
                error={"code": code, "message": message, "retryable": False, "details": details or {}},
            )


def outbox_handlers() -> dict[str, Handler]:
    return {OUTBOX_KIND: handle_prepare_snapshot}


def periodic_tasks() -> list[Any]:
    """prepare has no periodic work (the outbox runner retries/leases ``prepare.snapshot``)."""
    return []


# ---------------------------------------------------------------------------
# Read side: experiments_list / experiment_get / plan_get / plan_script
# ---------------------------------------------------------------------------


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    cluster_id: str
    status: str
    question: str | None
    tags: list[str]
    plan_count: int
    run_count: int
    budget_gpu_hours: float | None
    created_at: str
    updated_at: str


class ExperimentsListData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiments: list[ExperimentSummary]
    next_cursor: str | None = None
    coverage: Coverage = Field(default_factory=Coverage)


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def experiments_list(
    ctx: ServiceContext, *, cursor: str | None = None, limit: int | None = None, name: str | None = None
) -> ExperimentsListData:
    """Experiments, newest first (``name`` = case-insensitive substring filter)."""
    n = clamp_limit(limit)
    cur = decode_cursor(cursor)
    where = ["1=1"]
    params: list[Any] = []
    if name:
        where.append("e.name LIKE ? ESCAPE '\\'")
        params.append(f"%{_like_escape(name)}%")
    if cur:
        try:
            created, eid = str(cur["created_at"]), str(cur["experiment_id"])
        except KeyError as exc:
            raise InvalidArgument("malformed cursor") from exc
        where.append("(e.created_at < ? OR (e.created_at = ? AND e.experiment_id < ?))")
        params.extend([created, created, eid])
    rows = ctx.db.all(
        "SELECT e.*, (SELECT COUNT(*) FROM plans p WHERE p.experiment_id = e.experiment_id) AS plan_count, "
        "(SELECT COUNT(*) FROM runs r WHERE r.experiment_id = e.experiment_id) AS run_count "
        f"FROM experiments e WHERE {' AND '.join(where)} "
        "ORDER BY e.created_at DESC, e.experiment_id DESC LIMIT ?",
        (*params, n + 1),
    )
    more = len(rows) > n
    rows = rows[:n]
    items = [
        ExperimentSummary(
            experiment_id=r["experiment_id"],
            name=r["name"],
            cluster_id=r["cluster_id"],
            status=r["status"],
            question=str(r["question"])[:300] if r["question"] else None,
            tags=list(jload(r["tags_json"], []) or []),
            plan_count=int(r["plan_count"]),
            run_count=int(r["run_count"]),
            budget_gpu_hours=r["budget_gpu_hours"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
    nxt = (
        encode_cursor({"created_at": rows[-1]["created_at"], "experiment_id": rows[-1]["experiment_id"]})
        if more and rows
        else None
    )
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=iso(ctx.clock.now()))
    return ExperimentsListData(experiments=items, next_cursor=nxt, coverage=cov)


class PlanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    plan_ref: str
    kind: str
    status: str
    plan_hash: str | None
    snapshot_digest: str | None
    smoke_run_id: str | None
    operation_id: str | None
    run_count: int
    validation_status: str | None
    created_at: str


class ExperimentData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    cluster_id: str
    question: str | None
    status: str
    version: int
    spec: dict[str, Any]
    spec_hash: str
    project_root: str | None
    budget_gpu_hours: float | None
    max_parallel: int | None
    tags: list[str]
    parent_experiment_id: str | None
    plans: list[PlanSummary]
    plans_total: int
    run_counts: dict[str, int]  # coverage codes over main runs (+ planned / valid totals)
    runs_total: int
    planned_gpu_hours: float
    budget: dict[str, Any] | None
    created_at: str
    updated_at: str
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str | None = None


def _budget_summary(ctx: ServiceContext, experiment_id: str) -> dict[str, Any]:
    """services.budget.summary (submit area); raises ImportError/AttributeError when absent."""
    from hpclab.services import budget as budget_mod

    summ = budget_mod.summary(ctx.db, experiment_id=experiment_id)
    if isinstance(summ, BaseModel):
        return summ.model_dump(mode="json")
    if dataclasses.is_dataclass(summ) and not isinstance(summ, type):
        return dataclasses.asdict(summ)
    if hasattr(summ, "_asdict"):
        return dict(summ._asdict())
    if isinstance(summ, dict):
        return dict(summ)
    return dict(vars(summ))


def _stale_after_s(ctx: ServiceContext, cluster_id: str) -> float:
    """Freshness bound for queued/running observations (mirrors ``runs._stale_after`` so the
    experiment view and the grid never disagree): 4 poll intervals, at least 2 minutes."""
    try:
        interval = ctx.clusters.get(cluster_id).profile.policy.poll_min_interval_s
    except HpcLabError:
        interval = 30.0
    if ctx.settings.min_poll_interval_s is not None:
        interval = ctx.settings.min_poll_interval_s
    return max(120.0, 4.0 * float(interval))


def experiment_get(ctx: ServiceContext, *, experiment_id: str) -> ExperimentData:
    """Spec, plans, run counts by coverage code and the budget summary of one experiment."""
    from hpclab.services.coverage import cell_statuses, summarize_codes

    row = _load_experiment(ctx, experiment_id)
    now = iso(ctx.clock.now())
    plans = ctx.db.all(
        "SELECT p.*, (SELECT COUNT(*) FROM runs r WHERE r.plan_id = p.plan_id) AS run_count "
        "FROM plans p WHERE p.experiment_id=? ORDER BY p.created_at, p.rowid",
        (experiment_id,),
    )
    summaries = [
        PlanSummary(
            plan_id=p["plan_id"],
            plan_ref=plan_ref(p["plan_id"]),
            kind=p["kind"],
            status=p["status"],
            plan_hash=p["plan_hash"],
            snapshot_digest=p["snapshot_digest"],
            smoke_run_id=p["smoke_run_id"],
            operation_id=p["operation_id"],
            run_count=int(p["run_count"]),
            validation_status=(jload(p["validation_json"], {}) or {}).get("status"),
            created_at=p["created_at"],
        )
        for p in plans[-100:]
    ]
    main_ids = [
        r["run_id"]
        for r in ctx.db.all(
            "SELECT run_id FROM runs WHERE experiment_id=? AND kind='main' ORDER BY rowid", (experiment_id,)
        )
    ]
    # Same freshness bound as runs_list/run_inspect: a queued/running attempt whose last
    # observation is too old is '?' (unknown now), never presented as currently running.
    statuses = (
        cell_statuses(
            ctx.db, main_ids, stale_after_s=_stale_after_s(ctx, row["cluster_id"]), now=ctx.clock.now()
        )
        if main_ids
        else {}
    )
    counts = summarize_codes(statuses.values()) if main_ids else {"planned": 0, "valid": 0}
    all_rows = _existing_runs(ctx, experiment_id)
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=now)
    stale = sum(1 for s in statuses.values() if s.stale)
    if stale:
        cov.complete = False
        cov.notes.append(
            f"{stale} run(s) have stale observations: last seen too long ago to know their state now (code '?'); "
            "refresh with runs_list / run_inspect"
        )
    budget: dict[str, Any] | None
    try:
        budget = _budget_summary(ctx, experiment_id)
        cov.add("budget", "ok", observed_at=now)
    except (ImportError, AttributeError) as exc:
        budget = None
        cov.add("budget", "unsupported", detail=f"budget ledger summary unavailable: {exc}")
    except HpcLabError as exc:
        budget = None
        cov.add("budget", "unavailable", detail=exc.message)
    return ExperimentData(
        experiment_id=row["experiment_id"],
        name=row["name"],
        cluster_id=row["cluster_id"],
        question=row["question"],
        status=row["status"],
        version=int(row["version"]),
        spec=jload(row["spec_json"], {}) or {},
        spec_hash=row["spec_hash"],
        project_root=row["project_root"],
        budget_gpu_hours=row["budget_gpu_hours"],
        max_parallel=row["max_parallel"],
        tags=list(jload(row["tags_json"], []) or []),
        parent_experiment_id=row.get("parent_experiment_id"),
        plans=summaries,
        plans_total=len(plans),
        run_counts=counts,
        runs_total=len(all_rows),
        planned_gpu_hours=round(_planned_gpu_hours(all_rows), 6),
        budget=budget,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        coverage=cov,
        observed_at=now,
    )


class PlanRunDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    label: str
    kind: str
    cell: dict[str, Any]
    seed: int | None
    status: str
    spec_hash: str
    exec_plan_hash: str | None  # the frozen initial execution plan of this plan
    current_exec_plan_hash: str | None
    requested_gpu_hours: float | None
    added_post_hoc: bool
    derived_from: str | None
    run_spec: dict[str, Any]
    execution_plan: dict[str, Any] | None


class PlanData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    plan_ref: str
    experiment_id: str
    kind: str
    status: str
    plan_hash: str | None
    snapshot_digest: str | None
    smoke_run_id: str | None
    operation_id: str | None
    operation_ref: str | None
    validation: vd.ValidationReport
    resource_advice: dict[str, Any] | None
    budget: dict[str, Any] | None
    snapshot: dict[str, Any] | None
    overrides: list[dict[str, Any]]
    runs: list[PlanRunDetail]
    planned_runs: list[PreparedRun]  # preview when no Run rows exist (preparing / invalid)
    runs_total: int
    created_at: str
    updated_at: str
    details: dict[str, Any] = Field(default_factory=dict)  # plan-kind specific document (e.g. recovery cells)
    next_cursor: str | None = None
    coverage: Coverage = Field(default_factory=Coverage)


def _load_plan(ctx: ServiceContext, plan_id: str) -> dict[str, Any]:
    row = ctx.db.one(f"SELECT {_PLAN_COLUMNS} FROM plans WHERE plan_id=?", (plan_id,))
    if row is None:
        raise NotFound(f"plan {plan_id} not found", details={"plan_id": plan_id})
    return row


def _initial_exec_plan(ctx: ServiceContext, run_id: str) -> dict[str, Any] | None:
    return ctx.db.one(
        "SELECT exec_plan_hash, plan_json FROM execution_plans WHERE run_id=? ORDER BY rowid LIMIT 1",
        (run_id,),
    )


_DOC_KEYS = frozenset(
    {
        "status",
        "findings",
        "counts",
        "blocking",
        "overridden",
        "resource_advice",
        "budget",
        "snapshot",
        "overrides",
        "planned_runs",
    }
)


def _report_from_doc(doc: dict[str, Any]) -> vd.ValidationReport:
    """The validation report of a plan; tolerant of plans written by other areas (recovery/probe
    plans keep their own document shape, returned in ``PlanData.details``)."""
    fields = {k: doc[k] for k in ("status", "findings", "counts", "blocking", "overridden") if k in doc}
    fields.setdefault("status", "pending")
    try:
        return vd.ValidationReport.model_validate(fields)
    except ValidationError:
        status = fields["status"] if fields["status"] in ("pass", "warn", "fail", "pending") else "pending"
        return vd.ValidationReport(status=status)


def _cursor_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidArgument("malformed cursor", details={"field": key})
    return value


def _plan_cursor(cursor: str | None, *, plan_id: str, source: str) -> int:
    """Position of a ``plan_get`` cursor. Cursors are bound to their plan and to what they page:
    the preview of a preparing/invalid plan (list index) or the frozen runs (rowid)."""
    if not cursor:
        return 0
    data = decode_cursor(cursor)
    if data.get("plan_id") != plan_id or data.get("src") not in ("runs", "preview"):
        raise InvalidArgument(
            "this cursor does not belong to this plan listing", details={"plan_id": plan_id}
        )
    if data["src"] != source:
        raise Stale(
            f"plan {plan_id} changed since this cursor was issued (its runs were frozen); page again "
            "without a cursor",
            details={"plan_id": plan_id},
        )
    return _cursor_int(data, "after")


def plan_get(
    ctx: ServiceContext, *, plan_id: str, cursor: str | None = None, limit: int | None = None
) -> PlanData:
    """A frozen plan: runs with their RunSpec + initial ExecutionPlan (paged), hashes, validation."""
    plan = _load_plan(ctx, plan_id)
    n = clamp_limit(limit)
    doc = jload(plan["validation_json"], {}) or {}
    total = int((ctx.db.one("SELECT COUNT(*) AS n FROM runs WHERE plan_id=?", (plan_id,)) or {"n": 0})["n"])
    preview_raw = (doc.get("planned_runs") or []) if isinstance(doc, dict) else []
    source = "preview" if total == 0 and preview_raw else "runs"
    after = _plan_cursor(cursor, plan_id=plan_id, source=source)
    rows = ctx.db.all(
        "SELECT rowid AS _rowid, * FROM runs WHERE plan_id=? AND rowid > ? ORDER BY rowid LIMIT ?",
        (plan_id, after, n + 1),
    )
    more = len(rows) > n
    rows = rows[:n]
    details: list[PlanRunDetail] = []
    for r in rows:
        ep = _initial_exec_plan(ctx, r["run_id"])
        plan_json = jload(ep["plan_json"], None) if ep else None
        hours = None
        if isinstance(plan_json, dict):
            try:
                hours = round(
                    ResourceSpec.model_validate(plan_json.get("resources") or {}).requested_gpu_hours, 6
                )
            except ValidationError:
                hours = None
        details.append(
            PlanRunDetail(
                run_id=r["run_id"],
                label=r["label"],
                kind=r["kind"],
                cell=jload(r["cell_key"], {}) or {},
                seed=r["seed"],
                status=r["status"],
                spec_hash=r["spec_hash"],
                exec_plan_hash=ep["exec_plan_hash"] if ep else None,
                current_exec_plan_hash=r["current_exec_plan_hash"],
                requested_gpu_hours=hours,
                added_post_hoc=bool(r.get("added_post_hoc")),
                derived_from=r["derived_from"],
                run_spec=jload(r["spec_json"], {}) or {},
                execution_plan=plan_json,
            )
        )
    preview: list[PreparedRun] = []
    preview_next = None
    if source == "preview":
        page = preview_raw[after : after + n]
        preview = [PreparedRun.model_validate(p) for p in page]
        if after + n < len(preview_raw):
            preview_next = encode_cursor({"plan_id": plan_id, "src": "preview", "after": after + n})
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=iso(ctx.clock.now()))
    if plan["status"] == "preparing":
        cov.notes.append("the plan is still preparing: runs and plan_hash appear when it is ready")
    nxt = (
        encode_cursor({"plan_id": plan_id, "src": "runs", "after": rows[-1]["_rowid"]})
        if more and rows
        else preview_next
    )
    return PlanData(
        plan_id=plan_id,
        plan_ref=plan_ref(plan_id),
        experiment_id=plan["experiment_id"],
        kind=plan["kind"],
        status=plan["status"],
        plan_hash=plan["plan_hash"],
        snapshot_digest=plan["snapshot_digest"],
        smoke_run_id=plan["smoke_run_id"],
        operation_id=plan["operation_id"],
        operation_ref=operation_ref(plan["operation_id"]) if plan["operation_id"] else None,
        validation=_report_from_doc(doc),
        resource_advice=doc.get("resource_advice"),
        budget=doc.get("budget"),
        snapshot=doc.get("snapshot"),
        overrides=list(doc.get("overrides") or []),
        runs=details,
        planned_runs=preview,
        runs_total=total if total else len(preview_raw),
        created_at=plan["created_at"],
        updated_at=plan["updated_at"],
        details={k: v for k, v in doc.items() if k not in _DOC_KEYS} if isinstance(doc, dict) else {},
        next_cursor=nxt,
        coverage=cov,
    )


class PlanScriptData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    run_id: str
    label: str
    exec_plan_hash: str
    attempt_id_placeholder: str
    token_placeholder: str
    view: Literal["generated", "full"]
    directives: list[str]
    script: str  # this page of the script (the whole script when not truncated)
    offset: int
    truncated: bool
    script_bytes: int
    script_digest: str
    wrapper_version: str
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)


def plan_script(
    ctx: ServiceContext,
    *,
    plan_id: str,
    run_id: str | None = None,
    view: Literal["full", "generated"] = "full",
    cursor: str | None = None,
    max_bytes: int | None = None,
) -> PlanScriptData:
    """The exact ``job.sbatch`` a Run of this plan would get (placeholder attempt id/token).

    Default: the first main run, the complete script. A typical script fits one 64 KiB page;
    a larger one (long setup, many env vars or outputs) comes in 64 KiB pages that each end at
    a line boundary (a warning says so; ``cursor`` continues, the pages concatenate to the
    exact script). An explicit ``max_bytes`` pages byte-exactly (never inside a character).
    ``view='generated'`` shows only the run-specific part (directives, values, argv, env,
    setup): the static wrapper runtime, identical for every job of this wrapper version, is
    replaced by a marker.
    """
    plan = _load_plan(ctx, plan_id)
    if plan["status"] == "preparing":
        raise NotReady(
            f"plan {plan_id} is still preparing",
            operation_ref=operation_ref(plan["operation_id"]) if plan["operation_id"] else None,
        )
    if plan["status"] != "ready":
        raise PreconditionFailed(f"plan {plan_id} is {plan['status']}: it has no runs to render")
    if run_id is None:
        row = ctx.db.one(
            "SELECT * FROM runs WHERE plan_id=? ORDER BY (kind != 'main'), rowid LIMIT 1", (plan_id,)
        )
    else:
        row = ctx.db.one("SELECT * FROM runs WHERE run_id=? AND plan_id=?", (run_id, plan_id))
    if row is None:
        raise NotFound(
            f"run {run_id} is not part of plan {plan_id}", details={"plan_id": plan_id, "run_id": run_id}
        )
    ep = _initial_exec_plan(ctx, row["run_id"])
    if ep is None:
        raise PreconditionFailed(f"run {row['run_id']} has no execution plan")
    exec_plan = ExecutionPlan.model_validate(jload(ep["plan_json"]))
    run_spec = RunSpec.model_validate(jload(row["spec_json"]))
    cluster = ctx.clusters.get(exec_plan.cluster_id)
    warnings: list[str] = []
    if cluster.revision != exec_plan.profile_revision:
        warnings.append(
            "the cluster registration changed since this plan was frozen; the script is rendered with the "
            "current profile/overlay"
        )
    planned = vd.PlannedRun(
        run_id=row["run_id"],
        kind=row["kind"],
        label=row["label"],
        cell=jload(row["cell_key"], {}) or {},
        seed=row["seed"],
        run_spec=run_spec,
        resources=exec_plan.resources,
        exec_plan=exec_plan,
        render=mx.RenderResult(argv=list(run_spec.argv)),
    )
    script = vd.render_preview(planned, cluster, exec_plan.remote_root)
    from hpclab.services.jobscript import HELPER_SOURCE, WRAPPER_RUNTIME, WRAPPER_VERSION

    raw = script.encode("utf-8")
    directives = [ln[len("#SBATCH ") :] for ln in script.splitlines() if ln.startswith("#SBATCH ")]
    if view == "generated":
        text = script.replace(
            WRAPPER_RUNTIME.strip("\n"),
            f"# ... static wrapper runtime ({WRAPPER_VERSION}, {len(WRAPPER_RUNTIME)} bytes) omitted: view='full' ...",
        ).replace(HELPER_SOURCE, f"# ... embedded python helper ({len(HELPER_SOURCE)} bytes) omitted ...\n")
        body = text.encode("utf-8")
    else:
        body = raw
    limit = clamp_bytes(MAX_TEXT_BYTES if max_bytes is None else max_bytes)
    body_tag = sha256_hex(body)[:16]
    offset = _script_cursor(cursor, run_id=row["run_id"], view=view, body_tag=body_tag, size=len(body))
    end = _utf8_page_end(body, offset, limit)
    if max_bytes is None and end < len(body):
        # default pages hold whole lines: a reader never gets half a command (or a cut path)
        end = _line_page_end(body, offset, end)
        warnings.append(
            f"the script is {len(body)} bytes, more than one {limit}-byte page: this page "
            f"(bytes {offset}-{end}) ends at a line boundary; continue with next_cursor"
        )
    chunk = body[offset:end]
    return PlanScriptData(
        plan_id=plan_id,
        run_id=row["run_id"],
        label=row["label"],
        exec_plan_hash=ep["exec_plan_hash"],
        attempt_id_placeholder=vd.PLACEHOLDER_ATTEMPT_ID,
        token_placeholder=vd.PLACEHOLDER_TOKEN,
        view=view,
        directives=directives,
        script=chunk.decode("utf-8"),
        offset=offset,
        truncated=end < len(body),
        script_bytes=len(raw),
        script_digest="sha256:" + sha256_hex(raw),
        wrapper_version=WRAPPER_VERSION,
        next_cursor=encode_cursor({"offset": end, "run_id": row["run_id"], "view": view, "body": body_tag})
        if end < len(body)
        else None,
        warnings=warnings,
    )


def _script_cursor(cursor: str | None, *, run_id: str, view: str, body_tag: str, size: int) -> int:
    """Offset of a ``plan_script`` cursor, bound to the run, the view and the rendered text (a
    re-render after a cluster registration change must not be stitched onto older pages)."""
    if not cursor:
        return 0
    data = decode_cursor(cursor)
    if data.get("run_id") != run_id or data.get("view") != view:
        raise InvalidArgument(
            "this cursor belongs to a different run or view of the script",
            details={"run_id": run_id, "view": view},
        )
    offset = _cursor_int(data, "offset")
    if data.get("body") != body_tag:
        raise Stale("the rendered script changed since this cursor was issued; page again without a cursor")
    if offset > size:
        raise InvalidArgument("cursor offset out of range")
    return offset


def _line_page_end(body: bytes, offset: int, end: int) -> int:
    """``end`` moved back to just after the last newline of the page ``body[offset:end]``; a page
    that is one line longer than the page keeps its (character-safe) end, so paging progresses."""
    nl = body.rfind(b"\n", offset, end)
    return nl + 1 if nl >= offset else end


def _utf8_page_end(body: bytes, offset: int, limit: int) -> int:
    """End of a page of at most ``limit`` bytes that never splits a UTF-8 code point and always
    makes progress (a page narrower than one character holds that whole character)."""
    if offset < len(body) and (body[offset] & 0xC0) == 0x80:
        raise InvalidArgument("cursor offset is not at a character boundary")
    end = min(len(body), offset + limit)
    while offset < end < len(body) and (body[end] & 0xC0) == 0x80:
        end -= 1
    if end == offset and offset < len(body):
        end += 1
        while end < len(body) and (body[end] & 0xC0) == 0x80:
            end += 1
    return end
