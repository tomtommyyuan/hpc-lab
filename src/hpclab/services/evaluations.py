"""Evaluations: registered, bounded extractors applied to digest-verified artifacts.

``evaluation_collect`` turns a finalized output (by default ``RunSpec.evaluation.output`` of
the attempt, e.g. ``metrics``) into an archived Evaluation:

1. the input artifact must be ``finalized`` with a recorded sha256 (compute-node or
   login-node); mutable/missing/incomplete/unknown inputs are refused, never evaluated;
2. its bytes (at most ``evaluators.MAX_INPUT_BYTES`` = 64 MiB) are fetched through the
   content-addressed cache and verified against that digest (a changed file is ``stale``);
3. ``evaluators.run_evaluator`` extracts metrics; ``partial``/``invalid`` results are archived
   as such, and an evaluation is never ``valid`` while the evaluator reports an error issue;
4. the row is keyed by ``input_key`` = digest(input artifact ids + digests, evaluator id,
   version, config hash): collecting the same inputs with the same protocol again returns the
   same Evaluation. Every row carries a provenance chain (attempt, job, run spec, code digest,
   input digests, evaluator config).

Batch mode (``run_set_ref`` = experiment / plan / workflow id) evaluates every terminal,
COMPLETED attempt whose result is ``valid`` and that has no evaluation for the protocol yet;
everything else is listed in ``skipped`` with a reason. Up to ``INLINE_BATCH_MAX`` attempts are
evaluated in the call; larger batches become an Operation (outbox ``evaluation.collect``).
"""

from __future__ import annotations

import logging
import posixpath
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.api.envelope import Coverage
from hpclab.domain.clock import iso
from hpclab.domain.errors import (
    AuthRequired,
    ErrorCode,
    HpcLabError,
    InvalidArgument,
    NotReady,
    PreconditionFailed,
    Unavailable,
)
from hpclab.domain.hashing import digest
from hpclab.domain.ids import kind_of, new_id
from hpclab.domain.models import EvaluationDefaults
from hpclab.services import evaluators as ev
from hpclab.services.artifacts import (
    IndexResult,
    artifact_ref,
    index_attempt,
    read_verified_bytes,
    resolve_artifact,
)
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.storage.db import Row, Tx, jdump, jload

log = logging.getLogger("hpclab.evaluations")

TOOL = "evaluation_collect"
EVALUATION_URI = "hpclab://evaluations/"
OPERATION_URI = "hpclab://operations/"
PROVENANCE_SCHEMA = "hpclab.provenance/v1"
MAX_BATCH = 256
INLINE_BATCH_MAX = 64  # inline cap per call; the rest goes to a background operation
INLINE_BUDGET_S = 15.0  # seconds of inline collection before handing off
MAX_METRICS_RETURNED = 500
MAX_ITEMS_RETURNED = 256
HANDLER_TIME_BUDGET_S = 120.0
MAX_TRANSIENT_RETRIES = 20
_NEVER_ENDED = ("submission_rejected", "cancelled_before_submit", "abandoned")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvaluationData(BaseModel):
    """An archived evaluation (resource ``hpclab://evaluations/{evaluation_id}``)."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    evaluation_ref: str
    attempt_id: str
    run_id: str
    evaluator_id: str
    evaluator_version: str
    config_hash: str
    input_key: str
    status: Literal["valid", "partial", "invalid"]
    inputs: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    metrics_total: int
    metrics_truncated: bool = False
    validation: dict[str, Any]
    provenance: dict[str, Any]
    created_at: str


class CollectedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    evaluation_id: str
    status: str
    created: bool


class SkippedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    attempt_id: str | None = None
    reason: str
    detail: str | None = None
    evaluation_id: str | None = None


class EvaluationCollectData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "batch"]
    evaluation: EvaluationData | None = None
    created: bool | None = None
    run_set: dict[str, Any] | None = None
    collected: list[CollectedItem] = Field(default_factory=list)
    skipped: list[SkippedItem] = Field(default_factory=list)
    skipped_total: int = 0
    pending: int = 0  # attempts handed to the background operation
    counts: dict[str, int] = Field(default_factory=dict)
    operation_id: str | None = None
    operation_ref: str | None = None
    poll_after_ms: int | None = None
    coverage: Coverage = Field(default_factory=Coverage)
    observed_at: str
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol (evaluator + config) and inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Protocol:
    evaluator_id: str
    evaluator_version: str
    config: dict[str, Any]  # normalized (defaults applied): the thing hashed
    config_hash: str
    output: str


def _version_key(v: str) -> tuple[int, ...] | tuple[str]:
    try:
        return tuple(int(p) for p in v.split("."))
    except ValueError:
        return (v,)


def _defaults_for(ctx: ServiceContext, run: Row) -> EvaluationDefaults:
    for source in (jload(run.get("spec_json"), {}), _experiment_spec(ctx, run)):
        raw = source.get("evaluation") if isinstance(source, dict) else None
        if isinstance(raw, dict):
            try:
                return EvaluationDefaults.model_validate(raw)
            except ValueError:
                log.warning("run %s has an invalid evaluation default: %r", run["run_id"], raw)
    return EvaluationDefaults()


def _experiment_spec(ctx: ServiceContext, run: Row) -> dict[str, Any]:
    row = ctx.db.one("SELECT spec_json FROM experiments WHERE experiment_id=?", (run["experiment_id"],))
    spec = jload(row["spec_json"], {}) if row else {}
    return spec if isinstance(spec, dict) else {}


def _protocol(
    defaults: EvaluationDefaults,
    evaluator_id: str | None,
    evaluator_version: str | None,
    config: dict[str, Any] | None,
) -> _Protocol:
    eid = evaluator_id or defaults.evaluator_id
    if evaluator_version:
        ver = evaluator_version
    elif eid == defaults.evaluator_id:
        ver = defaults.evaluator_version
    else:
        versions = sorted((v for i, v in ev.EVALUATORS if i == eid), key=_version_key)
        if not versions:
            ev.get_evaluator(eid, "?")  # raises NotFound listing what exists
        ver = versions[-1]
    cfg = config if config is not None else (defaults.config if eid == defaults.evaluator_id else {})
    norm = ev.validate_config(eid, ver, cfg)
    return _Protocol(eid, ver, norm, digest(norm), defaults.output)


def _input_format(row: Row) -> Literal["json", "jsonl"] | None:
    ext = posixpath.splitext(str(row["remote_path"]))[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json"
    return None


def _check_input(row: Row) -> None:
    """Only finalized outputs with a verified digest are evaluated."""
    name = row["name"]
    details = {
        "artifact_id": row["artifact_id"],
        "name": name,
        "integrity": row["integrity"],
        "digest_state": row["digest_state"],
    }
    if row["integrity"] == "mutable":
        raise NotReady(
            f"output {name} is still being written (attempt active); collect after it ends", details=details
        )
    if row["kind"] != "file":
        raise InvalidArgument(
            f"output {name} is a directory; evaluators read one file (use {artifact_ref(row['artifact_id'])}#<relpath>)",
            details=details,
        )
    if row["digest_state"] == "mismatch":
        raise PreconditionFailed(f"output {name} no longer matches its recorded digest", details=details)
    if row["integrity"] != "finalized" or not row["digest"]:
        raise PreconditionFailed(
            f"output {name} is {row['integrity']} (digest {row['digest_state']}); only finalized outputs with a "
            "verified digest are evaluated",
            details={**details, "hint": "inspect with artifacts_list(refresh=true); fix or recover the run"},
        )


def _raise_if_unreachable(res: IndexResult, attempt: Row) -> None:
    """An index that could not reach the cluster leaves inputs ``unknown``: that is a retryable
    outage (``unavailable`` / ``auth_required``), not a verdict about the outputs."""
    cluster = [s for s in res.coverage.sources if s.name == "cluster" and s.status != "ok"]
    if not cluster:
        return
    details = {
        "attempt_id": attempt["attempt_id"],
        "cluster_id": attempt["cluster_id"],
        "detail": cluster[0].detail,
    }
    if cluster[0].status == "unauthorized":
        raise AuthRequired(
            f"cannot check the outputs of {attempt['attempt_id']}: the cluster needs authentication",
            details={**details, "remediation": f"hpclab auth {attempt['cluster_id']}"},
        )
    raise Unavailable(
        f"cannot check the outputs of {attempt['attempt_id']}: the cluster is unreachable; retry later",
        details=details,
    )


def compute_input_key(proto: _Protocol, inputs: list[Row]) -> str:
    """Identity of an evaluation: the exact input artifacts (id + digest) and the protocol."""
    return digest(
        {
            "inputs": sorted([str(r["artifact_id"]), str(r["digest"])] for r in inputs),
            "evaluator": proto.evaluator_id,
            "version": proto.evaluator_version,
            "config_hash": proto.config_hash,
        }
    )


# ---------------------------------------------------------------------------
# Prepare (reads + remote I/O + evaluator) and store (one transaction)
# ---------------------------------------------------------------------------


@dataclass
class _Prepared:
    attempt: Row
    existing: Row | None = None
    row: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def _select_inputs(ctx: ServiceContext, attempt: Row, proto: _Protocol, refs: list[str] | None) -> list[Row]:
    if refs:
        rows = [resolve_artifact(ctx, r) for r in refs]
        for r in rows:
            if r["attempt_id"] != attempt["attempt_id"]:
                raise InvalidArgument(
                    f"artifact {r['artifact_id']} belongs to attempt {r['attempt_id']}, not {attempt['attempt_id']}"
                )
        return rows
    row = ctx.db.one(
        "SELECT * FROM artifacts WHERE attempt_id=? AND name=?", (attempt["attempt_id"], proto.output)
    )
    if row is None:
        known = [
            r["name"]
            for r in ctx.db.all(
                "SELECT name FROM artifacts WHERE attempt_id=? AND instr(name,'/')=0",
                (attempt["attempt_id"],),
            )
        ]
        raise PreconditionFailed(
            f"attempt {attempt['attempt_id']} has no indexed output {proto.output!r} to evaluate",
            details={
                "output": proto.output,
                "indexed_outputs": known,
                "hint": "pass input_artifact_refs explicitly",
            },
        )
    return [row]


def _prepare(
    ctx: ServiceContext,
    attempt_id: str,
    *,
    evaluator_id: str | None,
    evaluator_version: str | None,
    config: dict[str, Any] | None,
    input_refs: list[str] | None,
    background: bool,
) -> _Prepared:
    if kind_of(attempt_id) != "attempt":
        raise InvalidArgument(f"{attempt_id!r} is not an attempt id (att_...)")
    attempt = ctx.db.get("attempts", "attempt_id", attempt_id)
    if attempt["phase"] != "terminal":
        details = {"attempt_id": attempt_id, "phase": attempt["phase"]}
        if attempt["phase"] in _NEVER_ENDED:
            raise PreconditionFailed(
                f"attempt {attempt_id} is {attempt['phase']}: it has no job outputs to evaluate",
                details=details,
            )
        raise NotReady(
            f"attempt {attempt_id} is {attempt['phase']}: outputs are evaluated after the job ends",
            details=details,
        )
    run = ctx.db.get("runs", "run_id", attempt["run_id"])
    proto = _protocol(_defaults_for(ctx, run), evaluator_id, evaluator_version, config)
    last_index: IndexResult | None = None
    if ctx.db.one("SELECT 1 AS x FROM artifacts WHERE attempt_id=? LIMIT 1", (attempt_id,)) is None:
        last_index = index_attempt(ctx, attempt_id=attempt_id, background=background)
        attempt = ctx.db.get("attempts", "attempt_id", attempt_id)
    inputs = _select_inputs(ctx, attempt, proto, input_refs)
    if not background and any(r["integrity"] != "finalized" for r in inputs):
        # The attempt is terminal, so a non-final record is an index taken while the job ran
        # (the terminal index is still queued), during NFS attribute lag or during an outage:
        # it is re-checked once before refusing.
        last_index = index_attempt(ctx, attempt_id=attempt_id)
        attempt = ctx.db.get("attempts", "attempt_id", attempt_id)
        inputs = [ctx.db.get("artifacts", "artifact_id", r["artifact_id"]) for r in inputs]
    if last_index is not None and any(r["integrity"] != "finalized" for r in inputs):
        _raise_if_unreachable(last_index, attempt)
    for r in inputs:
        _check_input(r)
    key = compute_input_key(proto, inputs)
    prep = _Prepared(attempt=attempt)
    if attempt["result_state"] != "valid":
        prep.warnings.append(
            f"attempt result_state is {attempt['result_state']}: this evaluation does not count in comparisons "
            "until every required output is finalized"
        )
    if attempt["scheduler_state"] != "COMPLETED":
        prep.warnings.append(
            f"the job ended {attempt['scheduler_state']}, not COMPLETED: this evaluation is archived but "
            "does not count as the run's result (its record may be from mid-training)"
        )
    existing = ctx.db.one("SELECT * FROM evaluations WHERE input_key=?", (key,))
    if existing is not None:
        prep.existing = existing
        return prep
    limit = ev.MAX_INPUT_BYTES
    oversized = [r for r in inputs if (r["size_bytes"] or 0) > limit]
    if oversized:
        output = ev.EvalOutput(
            evaluator_id=proto.evaluator_id,
            evaluator_version=proto.evaluator_version,
            config=proto.config,
            status="invalid",
            metrics=[],
            issues=[
                ev.EvalIssue(
                    severity="error",
                    code="input_too_large",
                    message=f"input {r['name']} is {r['size_bytes']} bytes; the limit is {limit} (not downloaded)",
                )
                for r in oversized
            ],
        )
    else:
        payloads = [
            ev.EvalInput(
                name=str(r["name"]),
                data=read_verified_bytes(ctx, r, max_bytes=limit, timeout=300.0 if background else None),
                digest=str(r["digest"]),
                format=_input_format(r),
            )
            for r in inputs
        ]
        output = ev.run_evaluator(proto.evaluator_id, proto.evaluator_version, proto.config, payloads)
    status = output.status
    errors = [i for i in output.issues if i.severity == "error"]
    if status == "valid" and errors:
        status = "partial"  # an unknown must never be reported as valid
    if status != "valid":
        codes = ", ".join(sorted({i.code for i in errors})[:5]) or "see issues"
        prep.warnings.append(f"evaluation status is {status} ({codes}); it is archived but does not count")
    prep.row = _evaluation_row(ctx, attempt, run, proto, inputs, key, output, status, prep.warnings)
    return prep


def _evaluation_row(
    ctx: ServiceContext,
    attempt: Row,
    run: Row,
    proto: _Protocol,
    inputs: list[Row],
    key: str,
    output: ev.EvalOutput,
    status: str,
    warnings: list[str],
) -> dict[str, Any]:
    spec = jload(run.get("spec_json"), {}) or {}
    code = spec.get("code") if isinstance(spec, dict) else None
    env = jload(attempt.get("env_json"), None)
    input_refs = [
        {
            "artifact_id": r["artifact_id"],
            "artifact_ref": artifact_ref(r["artifact_id"]),
            "name": r["name"],
            "role": r["role"],
            "digest": r["digest"],
            "digest_source": r["digest_source"],
            "size_bytes": r["size_bytes"],
            "mtime": r["mtime"],
            "remote_path": r["remote_path"],
            "integrity": r["integrity"],
        }
        for r in inputs
    ]
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "attempt": {
            "attempt_id": attempt["attempt_id"],
            "seq": attempt["seq"],
            "exec_plan_hash": attempt["exec_plan_hash"],
            "cluster_id": attempt["cluster_id"],
            "job": {
                "job_id": attempt["job_id"],
                "owner": attempt["job_owner"],
                "submit_time": attempt["job_submit_time"],
                "restart_count": attempt["restart_count"],
            },
            "scheduler_state": attempt["scheduler_state"],
            "app_state": attempt["app_state"],
            "result_state": attempt["result_state"],
            "exit_code": attempt["exit_code"],
            "partition": attempt.get("partition"),
            "gpu_type": attempt.get("gpu_type"),
            "node_list": attempt["node_list"],
        },
        "run": {
            "run_id": run["run_id"],
            "experiment_id": run["experiment_id"],
            "plan_id": run["plan_id"],
            "label": run["label"],
            "kind": run["kind"],
            "seed": run["seed"],
            "cell": jload(run.get("cell_key"), {}),
            "spec_hash": run["spec_hash"],
            "code_digest": code.get("digest") if isinstance(code, dict) else None,
            "git_commit": code.get("git_commit") if isinstance(code, dict) else None,
            "git_dirty": code.get("git_dirty") if isinstance(code, dict) else None,
        },
        "env": _env_summary(env),
        "inputs": input_refs,
        "evaluator": {
            "id": proto.evaluator_id,
            "version": proto.evaluator_version,
            "config": proto.config,
            "config_hash": proto.config_hash,
        },
        "selected_record": output.selected_record,
        # how a best:<metric> selection chose the record (select/report split, selected step);
        # compare/verify use it to tell a declared early-stopping protocol from a selection leak
        "selection": output.selection,
        "collected_by": ctx.principal,
    }
    now = iso(ctx.clock.now()) or ""
    provenance["collected_at"] = now
    return {
        "evaluation_id": new_id("evaluation"),
        "attempt_id": attempt["attempt_id"],
        "run_id": attempt["run_id"],
        "evaluator_id": proto.evaluator_id,
        "evaluator_version": proto.evaluator_version,
        "config_hash": proto.config_hash,
        "input_key": key,
        "input_refs_json": jdump(input_refs),
        "metrics_json": jdump([m.model_dump(mode="json") for m in output.metrics]),
        "provenance_json": jdump(provenance),
        "validation_json": jdump(
            {
                "status": status,
                "evaluator_status": output.status,
                "issues": [i.model_dump(mode="json") for i in output.issues],
                "warnings": warnings,
            }
        ),
        "status": status,
        "created_at": now,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _env_summary(env: Any) -> dict[str, Any] | None:
    if not isinstance(env, dict):
        return None
    torch = _dict(env.get("torch"))
    python = _dict(env.get("python"))
    nvidia = _dict(env.get("nvidia_smi"))
    raw_gpus = nvidia.get("gpus")
    gpus: list[Any] = raw_gpus if isinstance(raw_gpus, list) else []
    return {
        "host": env.get("host"),
        "python": python.get("version"),
        "torch": torch.get("version"),
        "torch_cuda": torch.get("cuda"),
        "gpu_names": sorted({str(g.get("name")) for g in gpus if isinstance(g, dict) and g.get("name")})[:8],
        "driver_version": nvidia.get("driver_version"),
        "packages_digest": env.get("packages_digest"),
    }


def _store(tx: Tx, prep: _Prepared) -> tuple[Row, bool]:
    """Insert the prepared evaluation (or return the one already stored for its input_key)."""
    if prep.existing is not None:
        return prep.existing, False
    assert prep.row is not None
    cur = tx.one("SELECT * FROM evaluations WHERE input_key=?", (prep.row["input_key"],))
    if cur is not None:
        return cur, False
    try:
        tx.insert("evaluations", prep.row)
    except sqlite3.IntegrityError:
        cur = tx.one("SELECT * FROM evaluations WHERE input_key=?", (prep.row["input_key"],))
        if cur is None:
            raise
        return cur, False
    tx.event(
        "evaluation",
        prep.row["evaluation_id"],
        "evaluation.created",
        {
            "attempt_id": prep.row["attempt_id"],
            "run_id": prep.row["run_id"],
            "evaluator": f"{prep.row['evaluator_id']}@{prep.row['evaluator_version']}",
            "status": prep.row["status"],
            "input_key": prep.row["input_key"],
        },
    )
    row = tx.one("SELECT * FROM evaluations WHERE evaluation_id=?", (prep.row["evaluation_id"],))
    assert row is not None
    return row, True


def evaluation_from_row(row: Row) -> EvaluationData:
    metrics = jload(row["metrics_json"], []) or []
    return EvaluationData(
        evaluation_id=row["evaluation_id"],
        evaluation_ref=f"{EVALUATION_URI}{row['evaluation_id']}",
        attempt_id=row["attempt_id"],
        run_id=row["run_id"],
        evaluator_id=row["evaluator_id"],
        evaluator_version=row["evaluator_version"],
        config_hash=row["config_hash"],
        input_key=row["input_key"],
        status=row["status"],
        inputs=jload(row["input_refs_json"], []) or [],
        metrics=metrics[:MAX_METRICS_RETURNED],
        metrics_total=len(metrics),
        metrics_truncated=len(metrics) > MAX_METRICS_RETURNED,
        validation=jload(row["validation_json"], {}) or {},
        provenance=jload(row["provenance_json"], {}) or {},
        created_at=row["created_at"],
    )


def evaluation_get(ctx: ServiceContext, *, evaluation_id: str) -> EvaluationData:
    """One evaluation (resource ``hpclab://evaluations/{evaluation_id}``)."""
    eid = evaluation_id.strip()
    if eid.startswith(EVALUATION_URI):
        eid = eid[len(EVALUATION_URI) :]
    if kind_of(eid) != "evaluation":
        raise InvalidArgument(f"{evaluation_id!r} is not an evaluation id (ev_...)")
    return evaluation_from_row(ctx.db.get("evaluations", "evaluation_id", eid))


# ---------------------------------------------------------------------------
# evaluation_collect
# ---------------------------------------------------------------------------


def evaluation_collect(
    ctx: ServiceContext,
    *,
    idempotency_key: str,
    attempt_id: str | None = None,
    run_set_ref: str | None = None,
    input_artifact_refs: list[str] | None = None,
    evaluator_id: str | None = None,
    evaluator_version: str | None = None,
    config: dict[str, Any] | None = None,
) -> EvaluationCollectData:
    """Extract metrics from finalized, digest-verified outputs and archive them.

    Single mode: ``attempt_id`` (and/or ``input_artifact_refs`` of one attempt). Defaults come
    from ``RunSpec.evaluation`` (evaluator, version, config, output). Batch mode:
    ``run_set_ref`` = experiment / plan / workflow id; see the module docstring. Repeating a
    call with the same ``idempotency_key`` returns the stored response.
    """
    if not idempotency_key or not idempotency_key.strip():
        raise InvalidArgument("idempotency_key is required")
    if run_set_ref is not None and (attempt_id is not None or input_artifact_refs):
        raise InvalidArgument("give either run_set_ref (batch) or attempt_id / input_artifact_refs, not both")
    if run_set_ref is None and attempt_id is None and not input_artifact_refs:
        raise InvalidArgument("give attempt_id, input_artifact_refs or run_set_ref")
    request = {
        "attempt_id": attempt_id,
        "run_set_ref": run_set_ref,
        "input_artifact_refs": list(input_artifact_refs or []),
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "config": config,
    }
    try:
        payload_hash = digest(request)
    except (TypeError, ValueError) as exc:
        raise InvalidArgument(f"config must be plain JSON (no NaN/Infinity): {exc}") from exc
    with ctx.db.tx(source="api") as tx:
        stored = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
    if stored is not None:
        return EvaluationCollectData.model_validate(stored)
    if run_set_ref is not None:
        return _collect_batch(
            ctx,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            run_set_ref=run_set_ref,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            config=config,
        )
    if attempt_id is None:
        first = resolve_artifact(ctx, (input_artifact_refs or [""])[0])
        attempt_id = str(first["attempt_id"])
    prep = _prepare(
        ctx,
        attempt_id,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        config=config,
        input_refs=list(input_artifact_refs or []) or None,
        background=False,
    )
    observed_at = iso(ctx.clock.now()) or ""
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    if prep.existing is None:
        cov.add("artifact", "ok", observed_at, "fetched and verified against the recorded digest")
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
        if again is not None:
            return EvaluationCollectData.model_validate(again)
        row, created = _store(tx, prep)
        warnings = list(prep.warnings)
        if not created:
            warnings.append(
                "an evaluation of exactly these input bytes with this protocol already existed; returned it"
            )
        data = EvaluationCollectData(
            mode="single",
            evaluation=evaluation_from_row(row),
            created=created,
            counts={"collected": 1, "created": int(created)},
            coverage=cov,
            observed_at=observed_at,
            warnings=warnings,
        )
        tx.idem_store(ctx.principal, TOOL, idempotency_key, payload_hash, data.model_dump(mode="json"))
    return data


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

_RUN_SET_PREFIXES = ("hpclab://experiments/", "hpclab://plans/", "hpclab://workflows/")


def resolve_run_ids(ctx: ServiceContext, run_set_ref: str) -> tuple[str, str, list[str]]:
    """(kind, id, run ids) of an experiment / plan / workflow reference (probe runs excluded)."""
    text = run_set_ref.strip()
    for prefix in _RUN_SET_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    kind = kind_of(text)
    if kind == "experiment":
        ctx.db.get("experiments", "experiment_id", text)
        rows = ctx.db.all(
            "SELECT run_id FROM runs WHERE experiment_id=? AND kind != 'probe' ORDER BY run_id", (text,)
        )
    elif kind == "plan":
        ctx.db.get("plans", "plan_id", text)
        rows = ctx.db.all(
            "SELECT run_id FROM runs WHERE plan_id=? AND kind != 'probe' ORDER BY run_id", (text,)
        )
    elif kind == "workflow":
        ctx.db.get("workflows", "workflow_id", text)
        rows = ctx.db.all(
            "SELECT c.run_id FROM workflow_cells c JOIN runs r ON r.run_id=c.run_id WHERE c.workflow_id=? "
            "AND r.kind != 'probe' ORDER BY c.ordinal",
            (text,),
        )
    else:
        raise InvalidArgument(f"run_set_ref {run_set_ref!r} must be an experiment, plan or workflow id")
    return kind, text, [r["run_id"] for r in rows]


def _plan_batch(
    ctx: ServiceContext,
    run_ids: list[str],
    *,
    evaluator_id: str | None,
    evaluator_version: str | None,
    config: dict[str, Any] | None,
) -> tuple[list[tuple[str, str]], list[SkippedItem]]:
    eligible: list[tuple[str, str]] = []
    skipped: list[SkippedItem] = []
    # A caller override that does not fit a run's evaluator is a request error for the whole
    # batch (nothing is stored); only a run's own broken default is a per-run skip.
    override = evaluator_id is not None or evaluator_version is not None or config is not None
    runs = {r["run_id"]: r for r in _rows_in(ctx, "runs", "run_id", run_ids)}
    by_run: dict[str, list[Row]] = {}
    for a in _rows_in(ctx, "attempts", "run_id", run_ids, order="run_id, seq"):
        by_run.setdefault(a["run_id"], []).append(a)
    protocols: dict[str, _Protocol] = {}
    for rid in run_ids:
        attempts = by_run.get(rid, [])
        if not attempts:
            skipped.append(SkippedItem(run_id=rid, reason="not_dispatched"))
            continue
        for a in attempts:
            aid = a["attempt_id"]
            if a["phase"] != "terminal":
                skipped.append(
                    SkippedItem(run_id=rid, attempt_id=aid, reason="not_terminal", detail=a["phase"])
                )
                continue
            if a["scheduler_state"] != "COMPLETED":
                skipped.append(
                    SkippedItem(
                        run_id=rid,
                        attempt_id=aid,
                        reason="scheduler_state_not_completed",
                        detail=a["scheduler_state"],
                    )
                )
                continue
            if a["result_state"] != "valid":
                skipped.append(
                    SkippedItem(
                        run_id=rid,
                        attempt_id=aid,
                        reason="result_not_valid",
                        detail=f"result_state={a['result_state']}",
                    )
                )
                continue
            proto = protocols.get(rid)
            if proto is None:
                try:
                    proto = _protocol(_defaults_for(ctx, runs[rid]), evaluator_id, evaluator_version, config)
                except HpcLabError as exc:
                    if override:
                        raise
                    skipped.append(
                        SkippedItem(run_id=rid, attempt_id=aid, reason=str(exc.code), detail=exc.message)
                    )
                    continue
                protocols[rid] = proto
            try:
                inputs = _select_inputs(ctx, a, proto, None)
                for r in inputs:
                    _check_input(r)
            except HpcLabError as exc:
                skipped.append(
                    SkippedItem(run_id=rid, attempt_id=aid, reason=str(exc.code), detail=exc.message)
                )
                continue
            existing = ctx.db.one(
                "SELECT evaluation_id FROM evaluations WHERE input_key=?", (compute_input_key(proto, inputs),)
            )
            if existing is not None:
                skipped.append(
                    SkippedItem(
                        run_id=rid,
                        attempt_id=aid,
                        reason="already_evaluated",
                        evaluation_id=existing["evaluation_id"],
                    )
                )
                continue
            eligible.append((rid, aid))
    return eligible, skipped


def _rows_in(
    ctx: ServiceContext, table: str, col: str, ids: list[str], *, order: str | None = None
) -> list[Row]:
    out: list[Row] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        sql = f"SELECT * FROM {table} WHERE {col} IN ({marks})"
        if order:
            sql += f" ORDER BY {order}"
        out.extend(ctx.db.all(sql, chunk))
    return out


def _run_item(
    ctx: ServiceContext,
    attempt_id: str,
    *,
    evaluator_id: str | None,
    evaluator_version: str | None,
    config: dict[str, Any] | None,
    background: bool,
) -> CollectedItem:
    prep = _prepare(
        ctx,
        attempt_id,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        config=config,
        input_refs=None,
        background=background,
    )
    with ctx.db.tx(source="worker" if background else "api") as tx:
        row, created = _store(tx, prep)
    return CollectedItem(
        attempt_id=attempt_id,
        run_id=row["run_id"],
        evaluation_id=row["evaluation_id"],
        status=row["status"],
        created=created,
    )


def _counts(collected: list[CollectedItem], skipped: list[SkippedItem], pending: int = 0) -> dict[str, int]:
    counts: dict[str, int] = {"collected": len(collected), "skipped": len(skipped), "pending": pending}
    for c in collected:
        counts[f"status_{c.status}"] = counts.get(f"status_{c.status}", 0) + 1
    for s in skipped:
        counts[f"skipped_{s.reason}"] = counts.get(f"skipped_{s.reason}", 0) + 1
    return counts


def _collect_batch(
    ctx: ServiceContext,
    *,
    idempotency_key: str,
    payload_hash: str,
    run_set_ref: str,
    evaluator_id: str | None,
    evaluator_version: str | None,
    config: dict[str, Any] | None,
) -> EvaluationCollectData:
    kind, set_id, run_ids = resolve_run_ids(ctx, run_set_ref)
    if evaluator_id is not None:
        # a bad override is a request error for the whole batch, not N skipped items
        _protocol(EvaluationDefaults(), evaluator_id, evaluator_version, config)
    observed_at = iso(ctx.clock.now()) or ""
    cov = Coverage()
    cov.add("ledger", "ok", observed_at)
    warnings: list[str] = []
    eligible, skipped = _plan_batch(
        ctx, run_ids, evaluator_id=evaluator_id, evaluator_version=evaluator_version, config=config
    )
    if len(eligible) > MAX_BATCH:
        for rid, aid in eligible[MAX_BATCH:]:
            skipped.append(
                SkippedItem(
                    run_id=rid, attempt_id=aid, reason="batch_limit", detail=f"at most {MAX_BATCH} per call"
                )
            )
        warnings.append(
            f"{len(eligible) - MAX_BATCH} attempt(s) over the batch limit; call again to continue"
        )
        eligible = eligible[:MAX_BATCH]
    run_set = {"kind": kind, "id": set_id, "runs": len(run_ids)}
    collected: list[CollectedItem] = []
    # Collect inline while it fits the request budget (small sweeps finish in one call); hand
    # whatever is left to a background operation so the call never blocks for long.
    budget_s = min(max(1.0, ctx.settings.request_deadline_s * 0.6), INLINE_BUDGET_S)
    started = ctx.clock.monotonic()
    rest: list[tuple[str, str]] = []
    for i, (rid, aid) in enumerate(eligible):
        if i >= INLINE_BATCH_MAX or (i > 0 and ctx.clock.monotonic() - started > budget_s):
            rest = eligible[i:]
            break
        try:
            collected.append(
                _run_item(
                    ctx,
                    aid,
                    evaluator_id=evaluator_id,
                    evaluator_version=evaluator_version,
                    config=config,
                    background=False,
                )
            )
        except HpcLabError as exc:
            skipped.append(SkippedItem(run_id=rid, attempt_id=aid, reason=str(exc.code), detail=exc.message))
            if exc.code in (ErrorCode.UNAVAILABLE, ErrorCode.AUTH_REQUIRED):
                cov.add(
                    "cluster",
                    "unauthorized" if exc.code == ErrorCode.AUTH_REQUIRED else "unavailable",
                    observed_at,
                    exc.message,
                )
    with ctx.db.tx(source="api") as tx:
        again = tx.idem_lookup(ctx.principal, TOOL, idempotency_key, payload_hash)
        if again is not None:
            return EvaluationCollectData.model_validate(again)
        op_id: str | None = None
        if rest:
            op_id = tx.create_operation(
                "collect", target_type=kind, target_id=set_id, stage="queued", poll_after_ms=3000
            )
            tx.set_operation(op_id, progress={"done": 0, "total": len(rest)})
            tx.enqueue(
                "evaluation.collect",
                operation_id=op_id,
                target_id=set_id,
                payload={
                    "operation_id": op_id,
                    "run_set": run_set,
                    "attempt_ids": [aid for _rid, aid in rest],
                    "evaluator_id": evaluator_id,
                    "evaluator_version": evaluator_version,
                    "config": config,
                },
            )
            warnings.append(
                f"{len(rest)} attempt(s) are being evaluated in the background; poll {OPERATION_URI}{op_id} "
                "before comparing results"
            )
        data = EvaluationCollectData(
            mode="batch",
            run_set=run_set,
            collected=collected[:MAX_ITEMS_RETURNED],
            skipped=skipped[:MAX_ITEMS_RETURNED],
            skipped_total=len(skipped),
            pending=len(rest),
            counts=_counts(collected, skipped, len(rest)),
            operation_id=op_id,
            operation_ref=f"{OPERATION_URI}{op_id}" if op_id else None,
            poll_after_ms=3000 if op_id else None,
            coverage=cov,
            observed_at=observed_at,
            warnings=warnings,
        )
        tx.idem_store(ctx.principal, TOOL, idempotency_key, payload_hash, data.model_dump(mode="json"))
    return data


# ---------------------------------------------------------------------------
# Outbox handler
# ---------------------------------------------------------------------------


def _save_progress(
    ctx: ServiceContext,
    op_id: str,
    total: int,
    collected: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    **kw: Any,
) -> None:
    with ctx.db.tx(source="worker") as tx:
        op = tx.one("SELECT state FROM operations WHERE operation_id=?", (op_id,))
        if op is None or op["state"] in ("succeeded", "failed"):
            return
        tx.set_operation(
            op_id,
            progress={"done": len(collected) + len(skipped), "total": total},
            result={"collected": collected, "skipped": skipped, "partial": True},
            **kw,
        )


def handle_collect(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """Evaluate a batch in time-boxed slices; partial outcomes live in the operation result,
    so a restarted worker resumes where it stopped (evaluations are keyed by input_key)."""
    p = item.payload
    op_id = str(item.operation_id or p.get("operation_id") or "")
    op = ctx.db.one("SELECT * FROM operations WHERE operation_id=?", (op_id,))
    if op is None or op["state"] in ("succeeded", "failed"):
        return HandlerResult.done("operation finished or missing")
    attempt_ids = [str(a) for a in p.get("attempt_ids") or []]
    prior = jload(op["result_json"], {}) or {}
    collected: list[dict[str, Any]] = list(prior.get("collected") or [])
    skipped: list[dict[str, Any]] = list(prior.get("skipped") or [])
    done = {c.get("attempt_id") for c in collected} | {s.get("attempt_id") for s in skipped}
    started = time.monotonic()
    if op["state"] != "running":
        _save_progress(ctx, op_id, len(attempt_ids), collected, skipped, state="running", stage="evaluating")
    processed = 0
    for aid in attempt_ids:
        if aid in done:
            continue
        if processed and time.monotonic() - started > HANDLER_TIME_BUDGET_S:
            _save_progress(ctx, op_id, len(attempt_ids), collected, skipped)
            return HandlerResult.retry(0.5, "time slice used; continuing")
        processed += 1
        try:
            got = _run_item(
                ctx,
                aid,
                evaluator_id=p.get("evaluator_id"),
                evaluator_version=p.get("evaluator_version"),
                config=p.get("config"),
                background=True,
            )
            collected.append(got.model_dump(mode="json"))
        except HpcLabError as exc:
            # by code, not class: the transport's AuthRequiredError/TransportError and the
            # services' AuthRequired/Unavailable mean the same thing here
            if exc.code == ErrorCode.AUTH_REQUIRED:
                cluster = ctx.db.one("SELECT cluster_id FROM attempts WHERE attempt_id=?", (aid,))
                err = exc.to_dict()
                err["details"] = {
                    **err.get("details", {}),
                    "remediation": f"hpclab auth {cluster['cluster_id'] if cluster else ''}".strip(),
                }
                _save_progress(
                    ctx,
                    op_id,
                    len(attempt_ids),
                    collected,
                    skipped,
                    state="blocked",
                    stage="waiting_for_auth",
                    error=err,
                )
                return HandlerResult.retry(120.0, "auth_required")
            if exc.code == ErrorCode.UNAVAILABLE and exc.retryable and item.tries < MAX_TRANSIENT_RETRIES:
                _save_progress(ctx, op_id, len(attempt_ids), collected, skipped)
                return HandlerResult.retry(min(300.0, 10.0 * item.tries), exc.message)
            run = ctx.db.one("SELECT run_id FROM attempts WHERE attempt_id=?", (aid,))
            skipped.append(
                SkippedItem(
                    run_id=run["run_id"] if run else None,
                    attempt_id=aid,
                    reason=str(exc.code),
                    detail=exc.message,
                ).model_dump(mode="json")
            )
        _save_progress(ctx, op_id, len(attempt_ids), collected, skipped, state="running", stage="evaluating")
    counts = _counts(
        [CollectedItem.model_validate(c) for c in collected], [SkippedItem.model_validate(s) for s in skipped]
    )
    with ctx.db.tx(source="worker") as tx:
        tx.set_operation(
            op_id,
            state="succeeded",
            stage="done",
            progress={"done": len(attempt_ids), "total": len(attempt_ids)},
            result={
                "run_set": p.get("run_set"),
                "collected": collected[:MAX_ITEMS_RETURNED],
                "skipped": skipped[:MAX_ITEMS_RETURNED],
                "counts": counts,
            },
        )
    return HandlerResult.done(f"{len(collected)} evaluation(s)")


def handle_auto(ctx: ServiceContext, item: OutboxItem) -> HandlerResult | None:
    """``evaluation.auto`` {attempt_id}: run the RunSpec's declared default evaluator as soon as
    an attempt's outputs validate, so grids and run listings show metrics without a manual
    step. Idempotent by input key: a later ``evaluation_collect`` returns the same record.
    Anything that is not a transient cluster problem is recorded and skipped, never retried."""
    attempt_id = str(item.payload.get("attempt_id") or item.target_id or "")
    try:
        _run_item(ctx, attempt_id, evaluator_id=None, evaluator_version=None, config=None, background=True)
    except HpcLabError as exc:
        if exc.code in (ErrorCode.UNAVAILABLE, ErrorCode.AUTH_REQUIRED):
            raise
        log.info("auto evaluation of %s skipped: %s: %s", attempt_id, exc.code, exc.message)
        return HandlerResult.done(note=f"skipped: {exc.code}")
    return HandlerResult.done()


def outbox_handlers() -> dict[str, Handler]:
    return {"evaluation.collect": handle_collect, "evaluation.auto": handle_auto}


def periodic_tasks() -> list[Any]:
    return []
