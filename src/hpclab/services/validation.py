"""Prepare-time validation checks (build spec §7.2 prepare, "Validation checks").

Every check is a small function returning one ``Finding``:

* ``status``   what the check found: ``pass | warn | fail | unchecked``. *Unknown is never pass*:
  a check that could not look (cluster not registered, transport down, fact expired, no
  history) says ``unchecked`` and why.
* ``severity`` what it means for the plan: ``block`` (a ``fail`` + ``block`` finding makes the
  plan ``invalid``), ``warn`` or ``info``.

A blocking finding can be downgraded to ``warn`` by an explicit override
(``overrides=[{check, reason}]``); the override is recorded as an event by the caller and
stays visible on the finding (``override_reason``) — it is never a silent boolean. A few
checks cannot be overridden (a plan that cannot be rendered or has unknown placeholders
cannot run at all).

Static checks are pure functions of a ``PlanContext``. The two remote checks
(``env_guard_static``/``snapshot_bypass`` and ``scheduler_test``) are isolated at the bottom:
they only read a handful of small files / run one ``sbatch --test-only``, every call has a
timeout inside an overall deadline, and *any* transport problem turns the finding into
``unchecked`` (with ``hpclab auth <cluster>`` as remediation for auth problems).
"""

from __future__ import annotations

import json
import logging
import math
import posixpath
import re
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.adapters.base import (
    AuthRequiredError,
    Scheduler,
    SubmissionRejected,
    Transport,
    TransportTimeout,
)
from hpclab.domain.clock import format_slurm_duration, parse_slurm_duration
from hpclab.domain.errors import HpcLabError
from hpclab.domain.hashing import short
from hpclab.domain.models import (
    RUNTIME_PLACEHOLDERS,
    ExecutionPlan,
    ExperimentSpec,
    InputRef,
    ResourceSpec,
    RunSpec,
    Scalar,
)
from hpclab.domain.states import Pin
from hpclab.profiles.schema import PartitionRule, ResolvedCluster
from hpclab.services import matrix as mx

log = logging.getLogger("hpclab.prepare")

FindingStatus = Literal["pass", "warn", "fail", "unchecked"]
Severity = Literal["block", "warn", "info"]
ValidationStatus = Literal["pass", "warn", "fail", "pending"]

# A plan that cannot be rendered / has unknown placeholders cannot run at all.
NON_OVERRIDABLE = frozenset(
    {"spec_schema", "placeholders", "cluster_registered", "remote_root", "script_render", "matrix_duplicates"}
)
PLACEHOLDER_ATTEMPT_ID = "att_00000000000000000000000000"
PLACEHOLDER_TOKEN = "previewtoken"  # noqa: S105 - a job-name token placeholder, not a secret
CACHE_VARS = ("HF_HOME", "UV_CACHE_DIR", "TORCH_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR")
LOCKFILE_NAMES = ("uv.lock", "poetry.lock", "pixi.lock", "Pipfile.lock", "requirements.txt", "conda-lock.yml")
LONG_JOB_S = 4 * 3600
MAX_LIST = 20
# Oldest CUDA toolkit that can build kernels for an architecture.
ARCH_MIN_CUDA: dict[str, tuple[int, int]] = {
    "sm_60": (8, 0),
    "sm_61": (8, 0),
    "sm_70": (9, 0),
    "sm_75": (10, 0),
    "sm_80": (11, 0),
    "sm_86": (11, 1),
    "sm_87": (11, 4),
    "sm_89": (11, 8),
    "sm_90": (11, 8),
    "sm_100": (12, 8),
    "sm_101": (12, 8),
    "sm_120": (12, 8),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    status: FindingStatus
    severity: Severity
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: str | None = None
    override_reason: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status == "fail" and self.severity == "block"


class ValidationOverride(BaseModel):
    """Explicit, audited decision to proceed despite a blocking finding."""

    model_config = ConfigDict(extra="forbid")

    check: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=3, max_length=500)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus
    findings: list[Finding] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    blocking: list[str] = Field(default_factory=list)
    overridden: list[str] = Field(default_factory=list)


def finding(
    check: str,
    status: FindingStatus,
    message: str,
    *,
    severity: Severity | None = None,
    remediation: str | None = None,
    **evidence: Any,
) -> Finding:
    """Finding with conventional severities (fail -> block, warn -> warn, else info)."""
    sev: Severity = severity or {"fail": "block", "warn": "warn"}.get(status, "info")  # type: ignore[assignment]
    return Finding(
        check=check,
        status=status,
        severity=sev,
        message=message,
        evidence={k: v for k, v in evidence.items() if v is not None},
        remediation=remediation,
    )


def build_report(findings: Sequence[Finding], *, pending: bool = False) -> ValidationReport:
    """Overall status: ``fail`` if anything blocks, ``warn`` if anything is not a clean pass
    (warnings, non-blocking failures, *unchecked*), else ``pass``."""
    counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0, "unchecked": 0}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    blocking = sorted({f.check for f in findings if f.blocking})
    overridden = sorted({f.check for f in findings if f.override_reason})
    if pending:
        status: ValidationStatus = "pending"
    elif blocking:
        status = "fail"
    elif counts["warn"] or counts["fail"] or counts["unchecked"]:
        status = "warn"
    else:
        status = "pass"
    return ValidationReport(
        status=status, findings=list(findings), counts=counts, blocking=blocking, overridden=overridden
    )


@dataclass(frozen=True)
class OverrideOutcome:
    check: str
    reason: str
    applied: bool
    note: str
    original_status: str | None = None
    original_severity: str | None = None


def apply_overrides(
    findings: Sequence[Finding], overrides: Sequence[ValidationOverride]
) -> tuple[list[Finding], list[OverrideOutcome]]:
    """Downgrade blocking findings named by an override to ``warn`` (reason kept on the finding).

    Overrides for checks that are not blocking, unknown, or non-overridable are *not applied*
    and reported back (the caller surfaces them as warnings).
    """
    by_check = {o.check: o for o in overrides}
    out: list[Finding] = []
    outcomes: list[OverrideOutcome] = []
    used: set[str] = set()
    for f in findings:
        o = by_check.get(f.check)
        if o is not None and f.blocking and f.check not in NON_OVERRIDABLE:
            out.append(
                f.model_copy(update={"status": "warn", "severity": "warn", "override_reason": o.reason})
            )
            if f.check not in used:
                outcomes.append(
                    OverrideOutcome(f.check, o.reason, True, "downgraded block -> warn", f.status, f.severity)
                )
                used.add(f.check)
        else:
            out.append(f)
    names = {f.check for f in findings}
    for o in overrides:
        if o.check in used:
            continue
        if o.check in NON_OVERRIDABLE:
            note = "this check cannot be overridden (the plan could not run)"
        elif o.check not in names:
            note = "no finding with this check name"
        else:
            note = "the finding is not blocking; nothing to override"
        outcomes.append(OverrideOutcome(o.check, o.reason, False, note))
    return out, outcomes


# ---------------------------------------------------------------------------
# Plan context
# ---------------------------------------------------------------------------


@dataclass
class PlannedRun:
    """One Run as prepare intends to freeze it (``exec_plan`` is None when the cluster/remote
    root cannot be resolved; such plans are invalid anyway)."""

    run_id: str
    kind: str  # main | smoke
    label: str
    cell: dict[str, Scalar]
    seed: int | None
    run_spec: RunSpec
    resources: ResourceSpec  # resolved (partition/account/qos defaults applied)
    exec_plan: ExecutionPlan | None
    render: mx.RenderResult
    added_post_hoc: bool = False
    derived_from: str | None = None

    @property
    def requested_gpu_hours(self) -> float:
        return self.resources.requested_gpu_hours


@dataclass
class PlanContext:
    spec: ExperimentSpec
    cluster: ResolvedCluster | None
    runs: list[PlannedRun]
    now: datetime
    cluster_error: str | None = None
    remote_root: str | None = None
    remote_root_error: str | None = None
    manifest: dict[str, Any] | None = None  # snapshot manifest (None while not captured)
    snapshot_warnings: list[str] = field(default_factory=list)
    snapshot_excluded: list[str] = field(default_factory=list)
    git_dirty: bool = False
    smoke_error: str | None = None
    existing_gpu_hours: float = 0.0  # already planned in the experiment (extends)
    base_digests: list[str] = field(default_factory=list)  # snapshot digests of existing runs (drift)
    base_manifests: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    @property
    def main_runs(self) -> list[PlannedRun]:
        return [r for r in self.runs if r.kind == "main"]

    def shapes(self) -> list[tuple[str, ResourceSpec]]:
        """Distinct (role, resolved resources) pairs, main first."""
        seen: set[str] = set()
        out: list[tuple[str, ResourceSpec]] = []
        for r in sorted(self.runs, key=lambda r: r.kind != "main"):
            key = r.kind + r.resources.model_dump_json()
            if key not in seen:
                seen.add(key)
                out.append((r.kind, r.resources))
        return out


def _rule(pc: PlanContext, partition: str | None) -> PartitionRule | None:
    if pc.cluster is None or partition is None:
        return None
    return pc.cluster.profile.partitions.get(partition)


def _worst(statuses: Iterable[str]) -> str:
    order = {"fail": 3, "warn": 2, "unchecked": 1, "pass": 0}
    return max(statuses, key=lambda s: order[s], default="pass")


def _no_cluster(check: str, pc: PlanContext) -> Finding:
    return finding(check, "unchecked", f"cluster {pc.spec.cluster!r} is not registered; not checked")


# ---------------------------------------------------------------------------
# Spec-level checks
# ---------------------------------------------------------------------------


def check_spec_schema(pc: PlanContext) -> Finding:
    spec = pc.spec
    fails: list[str] = []
    warns: list[str] = []
    names = set(spec.parameters) | {k for k in spec.matrix if k != mx.SEED_AXIS}
    clash = sorted(names & RUNTIME_PLACEHOLDERS)
    if clash:
        fails.append(f"parameter/axis names {clash} collide with runtime placeholders")
    if "seed" in spec.parameters:
        fails.append("'seed' must be given as seeds or matrix.seed, not as a parameter")
    shadow = sorted(set(spec.parameters) & set(spec.matrix))
    if shadow:
        warns.append(f"matrix axes {shadow} override fixed parameters of the same name")
    if not spec.question:
        warns.append("no scientific question stated (question): reports cannot say what was tested")
    if spec.comparison is not None:
        axes = set(mx.axis_names(spec.matrix))
        bad = [t for t in spec.comparison.treatment if t not in axes and t not in spec.parameters]
        if bad:
            warns.append(f"comparison.treatment {bad} are not matrix axes")
        for k, v in (spec.comparison.baseline or {}).items():
            vals = spec.matrix.get(k)
            if vals is not None and all(mx.cell_key({"v": x}) != mx.cell_key({"v": v}) for x in vals):
                warns.append(f"comparison.baseline {k}={v!r} is not a value of matrix axis {k!r}")
    if fails:
        return finding("spec_schema", "fail", "; ".join(fails), problems=fails, warnings=warns or None)
    if warns:
        return finding("spec_schema", "warn", "; ".join(warns), severity="info", warnings=warns)
    return finding("spec_schema", "pass", "spec is well formed")


def check_placeholders(pc: PlanContext) -> Finding:
    unknown: dict[str, list[str]] = {}
    nulls: dict[str, list[str]] = {}
    for r in pc.runs:
        for name in r.render.unknown:
            unknown.setdefault(name, []).append(r.label)
        for name in r.render.nulls:
            nulls.setdefault(name, []).append(r.label)
    bad_outputs = []
    for o in pc.spec.outputs:
        for name in mx.find_placeholders(o.path):
            if name not in RUNTIME_PLACEHOLDERS:
                bad_outputs.append(f"{o.name}: {{{name}}}")
    if unknown or bad_outputs:
        known = sorted(set(pc.spec.parameters) | set(mx.axis_names(pc.spec.matrix)))
        msg = []
        if unknown:
            msg.append(f"unknown argv placeholders {sorted('{' + n + '}' for n in unknown)}")
        if bad_outputs:
            msg.append(f"output paths may only use runtime placeholders: {bad_outputs}")
        return finding(
            "placeholders",
            "fail",
            "; ".join(msg),
            unknown={k: v[:5] for k, v in unknown.items()},
            output_paths=bad_outputs or None,
            known_params=known,
            runtime_placeholders=sorted(RUNTIME_PLACEHOLDERS),
            remediation="use {param} names from parameters/matrix, {seed} (needs seeds), or a runtime placeholder",
        )
    unused = []
    used = set().union(*(r.render.used for r in pc.main_runs)) if pc.main_runs else set()
    for axis in mx.axis_names(pc.spec.matrix):
        if axis not in used:
            unused.append(axis)
    if nulls or unused:
        msg = []
        if nulls:
            msg.append(f"params {sorted(nulls)} are null and render as empty argv strings")
        if unused:
            msg.append(
                f"matrix axes {unused} are not referenced in argv: the program must read them from "
                "$HPCLAB_PARAMS_JSON"
            )
        return finding(
            "placeholders",
            "warn",
            "; ".join(msg),
            severity="warn" if nulls else "info",
            nulls={k: v[:5] for k, v in nulls.items()} or None,
            unused_axes=unused or None,
        )
    return finding("placeholders", "pass", "all argv placeholders resolve")


def check_matrix_duplicates(pc: PlanContext) -> Finding:
    idents = [
        (r.label, mx.effective_identity(r.run_spec.argv, r.run_spec.params, r.seed)) for r in pc.main_runs
    ]
    dupes = mx.find_duplicates(idents)
    if dupes:
        return finding(
            "matrix_duplicates",
            "fail",
            f"{len(dupes)} group(s) of cells are indistinguishable to the program (same argv, params and seed)",
            groups=dupes[:MAX_LIST],
            remediation="remove values that only differ in type (e.g. 1 and '1') from the matrix",
        )
    return finding("matrix_duplicates", "pass", f"{len(idents)} distinct cells")


def check_outputs_declared(pc: PlanContext) -> Finding:
    outs = pc.spec.outputs
    if not outs:
        return finding(
            "outputs_declared",
            "warn",
            "no outputs declared: results cannot be validated and completed runs stay unverified (C!)",
            remediation="declare at least the metrics file, e.g. {name: metrics, path: '{output_dir}/metrics.jsonl', role: metrics}",
        )
    required = [o.name for o in outs if o.required]
    if not required:
        return finding(
            "outputs_declared",
            "warn",
            "all outputs are optional: a run that wrote nothing would still count as valid",
            outputs=[o.name for o in outs],
        )
    if not any(o.role == "metrics" for o in outs):
        return finding(
            "outputs_declared",
            "warn",
            "no output has role 'metrics'",
            severity="info",
            outputs=[o.name for o in outs],
        )
    return finding("outputs_declared", "pass", f"{len(outs)} outputs declared ({len(required)} required)")


def check_evaluation_output_declared(pc: PlanContext) -> Finding:
    spec = pc.spec
    ev = spec.evaluation
    if ev is None:
        if spec.comparison is not None:
            return finding(
                "evaluation_output_declared",
                "warn",
                "comparison declared but no evaluation defaults: evaluation_collect needs an explicit evaluator",
            )
        return finding(
            "evaluation_output_declared", "warn", "no evaluation defaults declared", severity="info"
        )
    decl = {o.name: o for o in spec.outputs}
    if ev.output not in decl:
        return finding(
            "evaluation_output_declared",
            "fail",
            f"evaluation.output {ev.output!r} is not a declared output",
            outputs=sorted(decl),
            remediation="set evaluation.output to one of the declared outputs",
        )
    try:
        from hpclab.services.evaluators import validate_config

        validate_config(ev.evaluator_id, ev.evaluator_version, ev.config)
    except HpcLabError as exc:
        return finding(
            "evaluation_output_declared",
            "fail",
            f"evaluator {ev.evaluator_id}@{ev.evaluator_version}: {exc.message}",
            details=exc.details or None,
        )
    except Exception as exc:  # noqa: BLE001 - evaluator registry problems must not fail prepare
        return finding(
            "evaluation_output_declared", "unchecked", f"could not validate the evaluator config: {exc!r}"
        )
    if not decl[ev.output].required:
        return finding(
            "evaluation_output_declared",
            "warn",
            f"evaluation input {ev.output!r} is optional: runs without it cannot be evaluated",
        )
    return finding(
        "evaluation_output_declared",
        "pass",
        f"{ev.evaluator_id}@{ev.evaluator_version} reads declared output {ev.output!r}",
    )


def check_smoke_gate(pc: PlanContext) -> Finding:
    smoke = pc.spec.smoke
    if smoke is None or not smoke.enabled:
        return finding(
            "smoke_gate",
            "warn",
            "no smoke gate: main cells are dispatched without a short compute-node check first",
            severity="info",
        )
    if pc.smoke_error:
        return finding("smoke_gate", "fail", pc.smoke_error)
    names = {o.name for o in pc.spec.outputs}
    missing = [n for n in smoke.require_outputs if n not in names]
    if missing:
        return finding(
            "smoke_gate",
            "fail",
            f"smoke.require_outputs {missing} are not declared outputs",
            outputs=sorted(names),
        )
    if not smoke.require_outputs:
        # the gate (workflows._gate_verdict) is COMPLETED + result_state=valid, and
        # result_state is valid iff every *required* declared output was finalized
        required = [o.name for o in pc.spec.outputs if o.required]
        if required:
            message = (
                "smoke.require_outputs is empty: the smoke gate passes when the smoke job ends "
                f"COMPLETED with result_state=valid (its required outputs {required} finalized)"
            )
        else:
            message = (
                "smoke.require_outputs is empty and no declared output is required: the smoke gate "
                "only checks that the smoke job ends COMPLETED"
            )
        return finding("smoke_gate", "warn", message, severity="info", required_outputs=required)
    return finding("smoke_gate", "pass", f"smoke gate requires outputs {smoke.require_outputs}")


def check_budget(pc: PlanContext) -> Finding:
    requested = sum(r.requested_gpu_hours for r in pc.runs)
    smoke = sum(r.requested_gpu_hours for r in pc.runs if r.kind == "smoke")
    budget = pc.spec.budget.gpu_hours
    total = requested + pc.existing_gpu_hours
    ev: dict[str, Any] = {
        "requested_gpu_hours": round(requested, 4),
        "smoke_gpu_hours": round(smoke, 4),
        "existing_planned_gpu_hours": round(pc.existing_gpu_hours, 4) if pc.existing_gpu_hours else None,
        "budget_gpu_hours": budget,
    }
    if budget is None:
        return finding("budget", "pass", "no hpclab budget set (site quotas still apply)", **ev)
    if total > budget + 1e-9:
        return finding(
            "budget",
            "fail",
            f"requested {total:.2f} GPU-h (gpus x time limit) exceeds the budget of {budget:.2f} GPU-h",
            remediation="raise budget.gpu_hours, shorten resources.time, or shrink the matrix",
            **ev,
        )
    if total > 0.8 * budget:
        return finding(
            "budget", "warn", f"requested {total:.2f} of {budget:.2f} GPU-h: little room for retries", **ev
        )
    return finding("budget", "pass", f"requested {total:.2f} of {budget:.2f} GPU-h", **ev)


def compute_pin(inp: InputRef) -> Pin:
    if inp.digest:
        return Pin.PINNED
    if inp.version:
        return Pin.FINGERPRINTED
    return Pin.UNPINNED


_PIN_RANK = {Pin.UNRESOLVED: 0, Pin.UNPINNED: 1, Pin.FINGERPRINTED: 2, Pin.PINNED: 3}


def check_inputs_pinning(pc: PlanContext) -> Finding:
    inputs = pc.spec.inputs
    if not inputs:
        return finding("inputs_pinning", "pass", "no external inputs declared")
    pins = {i.name: compute_pin(i).value for i in inputs}
    overclaimed = [
        i.name for i in inputs if i.pin is not None and _PIN_RANK[Pin(i.pin)] > _PIN_RANK[compute_pin(i)]
    ]
    if overclaimed:
        return finding(
            "inputs_pinning",
            "warn",
            f"inputs {overclaimed} claim a pin level their declaration does not support "
            "(pinned needs a digest, fingerprinted a version)",
            pins=pins,
        )
    unpinned = [n for n, p in pins.items() if p == Pin.UNPINNED.value]
    if unpinned:
        return finding(
            "inputs_pinning",
            "warn",
            f"inputs {unpinned} are referenced by mutable path only (unpinned)",
            severity="info",
            pins=pins,
            remediation="add a version label or a sha256 digest to inputs you report results on",
        )
    return finding("inputs_pinning", "pass", "all inputs are pinned or fingerprinted", pins=pins)


# ---------------------------------------------------------------------------
# Cluster / partition checks
# ---------------------------------------------------------------------------


def check_cluster_registered(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return finding(
            "cluster_registered",
            "fail",
            pc.cluster_error or f"cluster {pc.spec.cluster!r} is not registered",
            remediation=f"hpclab cluster add {pc.spec.cluster} --profile <profile>",
        )
    return finding(
        "cluster_registered",
        "pass",
        f"cluster {pc.cluster.cluster_id} ({pc.cluster.profile.display_name})",
        profile_revision=pc.cluster.revision,
    )


def check_remote_root(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("remote_root", pc)
    if pc.remote_root is None:
        return finding(
            "remote_root",
            "fail",
            pc.remote_root_error or "the remote root cannot be resolved",
            remediation=f"hpclab cluster add {pc.spec.cluster} --user <you> [--remote-root /abs/path]",
        )
    return finding("remote_root", "pass", f"remote root {pc.remote_root}")


def check_partition_exists(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("partition_exists", pc)
    rules = pc.cluster.profile.partitions
    problems: list[str] = []
    warns: list[str] = []
    for role, res in pc.shapes():
        if res.partition is None:
            warns.append(f"{role}: no partition (Slurm's default partition will be used)")
        elif rules and res.partition not in rules:
            problems.append(f"{role}: partition {res.partition!r} is not in the profile")
    if problems:
        return finding("partition_exists", "fail", "; ".join(problems), known=sorted(rules), severity="block")
    if not rules:
        return finding("partition_exists", "unchecked", "the profile lists no partitions")
    if warns:
        return finding("partition_exists", "warn", "; ".join(warns))
    return finding("partition_exists", "pass", "partitions exist in the profile")


def check_time_within_partition_max(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("time_within_partition_max", pc)
    statuses: list[str] = []
    notes: list[str] = []
    for role, res in pc.shapes():
        rule = _rule(pc, res.partition)
        max_s = parse_slurm_duration(rule.max_time) if rule and rule.max_time else None
        if max_s is None:
            statuses.append("unchecked")
            notes.append(f"{role}: max time of partition {res.partition!r} unknown")
        elif res.time_seconds > max_s:
            statuses.append("fail")
            notes.append(f"{role}: time {res.time} exceeds {res.partition} max {rule.max_time}")  # type: ignore[union-attr]
        else:
            statuses.append("pass")
    status = _worst(statuses)
    msg = "; ".join(notes) or "time limits within partition maxima"
    return finding("time_within_partition_max", status, msg)  # type: ignore[arg-type]


def check_gpu_type_in_partition(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("gpu_type_in_partition", pc)
    statuses: list[str] = []
    notes: list[str] = []
    for role, res in pc.shapes():
        if res.gpus == 0:
            statuses.append("pass")
            continue
        rule = _rule(pc, res.partition)
        if rule is None or not rule.gpu_types:
            statuses.append("unchecked")
            notes.append(f"{role}: GPU types of partition {res.partition!r} unknown")
        elif res.gpu_type is None:
            statuses.append("warn")
            notes.append(
                f"{role}: no gpu_type: cells may land on different GPU models {rule.gpu_types} "
                "(heterogeneous results)"
            )
        elif res.gpu_type not in rule.gpu_types:
            statuses.append("fail")
            notes.append(f"{role}: gpu_type {res.gpu_type!r} not in {res.partition} {rule.gpu_types}")
        else:
            statuses.append("pass")
    status = _worst(statuses)
    return finding("gpu_type_in_partition", status, "; ".join(notes) or "GPU types available")  # type: ignore[arg-type]


def check_gpus_limits(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("gpus_limits", pc)
    statuses: list[str] = []
    notes: list[str] = []
    for role, res in pc.shapes():
        if res.total_gpus == 0:
            statuses.append("pass")
            continue
        rule = _rule(pc, res.partition)
        if rule is None or (rule.max_gpus_per_job is None and rule.max_gpus_per_user is None):
            statuses.append("unchecked")
            notes.append(f"{role}: GPU limits of partition {res.partition!r} unknown")
            continue
        if rule.max_gpus_per_job is not None and res.total_gpus > rule.max_gpus_per_job:
            statuses.append("fail")
            notes.append(f"{role}: {res.total_gpus} GPUs > max_gpus_per_job {rule.max_gpus_per_job}")
        elif rule.max_gpus_per_user is not None and res.total_gpus > rule.max_gpus_per_user:
            statuses.append("fail")
            notes.append(
                f"{role}: {res.total_gpus} GPUs > max_gpus_per_user {rule.max_gpus_per_user} (never starts)"
            )
        else:
            statuses.append("pass")
    return finding("gpus_limits", _worst(statuses), "; ".join(notes) or "GPU counts within limits")  # type: ignore[arg-type]


def check_account_present(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("account_present", pc)
    missing = [role for role, res in pc.shapes() if not res.account]
    if missing and pc.cluster.profile.policy.require_account:
        return finding(
            "account_present",
            "fail",
            "the site requires --account but none is set",
            remediation=f"hpclab cluster add {pc.cluster.cluster_id} --account <account> (or resources.account)",
        )
    if missing:
        return finding("account_present", "pass", "no account set (not required by the site)")
    return finding("account_present", "pass", f"account {pc.shapes()[0][1].account}")


def check_qos_required(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("qos_required", pc)
    missing = [role for role, res in pc.shapes() if not res.qos]
    if missing and pc.cluster.profile.policy.require_qos:
        return finding(
            "qos_required",
            "fail",
            "the site requires --qos but none is set",
            remediation=f"hpclab cluster add {pc.cluster.cluster_id} --qos <qos> (or resources.qos)",
        )
    return finding("qos_required", "pass", "QoS requirement satisfied")


def check_interactive_partition_batch(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("interactive_partition_batch", pc)
    statuses: list[str] = []
    notes: list[str] = []
    for role, res in pc.shapes():
        rule = _rule(pc, res.partition)
        if rule is None:
            statuses.append("unchecked")
        elif rule.interactive_only:
            statuses.append("fail")
            notes.append(f"{role}: partition {res.partition!r} is for interactive srun sessions, not sbatch")
        else:
            statuses.append("pass")
    status = _worst(statuses)
    msg = "; ".join(notes) or ("batch partitions" if status == "pass" else "partition rules unknown")
    return finding("interactive_partition_batch", status, msg)  # type: ignore[arg-type]


def check_preemptible_gate(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("preemptible_gate", pc)
    resumable = bool(pc.spec.resilience and pc.spec.resilience.resumable)
    statuses: list[str] = []
    notes: list[str] = []
    for role, res in pc.shapes():
        rule = _rule(pc, res.partition)
        if rule is None:
            statuses.append("unchecked")
            notes.append(f"{role}: partition {res.partition!r} rules unknown")
        elif rule.preemptible and not resumable:
            statuses.append("fail")
            notes.append(
                f"{role}: partition {res.partition!r} is preemptible but the program is not declared "
                "resumable: a preemption restarts it from scratch"
            )
        else:
            statuses.append("pass")
    status = _worst(statuses)
    return finding(
        "preemptible_gate",
        status,  # type: ignore[arg-type]
        "; ".join(notes) or "no preemption risk without resume",
        remediation="set resilience.resumable (the program resumes from $HPCLAB_RESUME_FROM) or use a "
        "non-preemptible partition"
        if status == "fail"
        else None,
    )


def check_walltime_signal(pc: PlanContext) -> Finding:
    res_spec = pc.spec.resilience
    statuses: list[str] = []
    notes: list[str] = []
    for r in pc.runs:
        plan_res = r.exec_plan.resilience if r.exec_plan is not None else res_spec
        rs = r.resources
        signal_flag = any(f.startswith("--signal=") for f in rs.extra_sbatch)
        if plan_res is not None and plan_res.stop_behavior == "checkpoint_and_exit":
            if plan_res.lead_s >= rs.time_seconds:
                statuses.append("fail")
                notes.append(f"{r.kind}: resilience.lead_s {plan_res.lead_s}s >= time limit {rs.time}")
            elif signal_flag:
                statuses.append("fail")
                notes.append(f"{r.kind}: extra_sbatch --signal conflicts with resilience.stop_behavior")
            else:
                statuses.append("pass")
        elif plan_res is not None and plan_res.resumable:
            statuses.append("warn")
            notes.append(
                f"{r.kind}: resumable but stop_behavior is 'none': a TIMEOUT loses the work since the "
                "last periodic checkpoint"
            )
        elif signal_flag:
            statuses.append("warn")
            notes.append(
                f"{r.kind}: extra_sbatch --signal without resilience.stop_behavior=checkpoint_and_exit"
            )
        elif r.kind == "main" and rs.time_seconds >= LONG_JOB_S:
            statuses.append("warn")
            notes.append(f"{r.kind}: {rs.time} job without a walltime checkpoint signal")
        else:
            statuses.append("pass")
    status = _worst(statuses)
    uniq = list(dict.fromkeys(notes))
    sev: Severity | None = None
    if status == "warn" and all("without a walltime" in n for n in uniq):
        sev = "info"
    return finding(
        "walltime_signal",
        status,  # type: ignore[arg-type]
        "; ".join(uniq) or "walltime handling consistent",
        severity=sev,
        remediation="resilience: {stop_behavior: checkpoint_and_exit, lead_s: <checkpoint time + 60>}"
        if status != "pass"
        else None,
    )


# ---------------------------------------------------------------------------
# Storage roles
# ---------------------------------------------------------------------------


def _role_regex(role_path: str) -> re.Pattern[str]:
    parts = re.split(r"(\{user\})", role_path.rstrip("/"))
    body = "".join("[^/]+" if p == "{user}" else re.escape(p) for p in parts)
    return re.compile("^" + body + "(/|$)")


def _normalize_remote(path: str, cluster: ResolvedCluster) -> str:
    """Best-effort static resolution of a cluster path for role matching (no remote calls)."""
    try:
        out = cluster.expand(path) if "{" in path else path
    except HpcLabError:  # e.g. {remote_root} on a cluster whose remote root cannot be resolved
        out = path
    user = cluster.overlay.user
    if user:
        out = re.sub(r"\$\{USER\}|\$USER(?![A-Za-z0-9_])", user, out)
    home = cluster.storage_path("home")
    if home:
        if out.startswith("~/") or out == "~":
            out = home + out[1:]
        out = re.sub(r"\$\{HOME\}|\$HOME(?![A-Za-z0-9_])", home, out)
    return out


def roles_for(cluster: ResolvedCluster, path: str) -> list[tuple[str, list[str]]]:
    """Storage roles (name, avoid_for) whose path contains ``path``."""
    norm = _normalize_remote(path, cluster)
    out: list[tuple[str, list[str]]] = []
    for name, role in cluster.profile.storage.items():
        rp = cluster.storage_path(name) or role.path
        if _role_regex(rp).match(norm):
            out.append((name, list(role.avoid_for)))
    return out


def check_storage_home_avoidance(pc: PlanContext) -> Finding:
    if pc.cluster is None:
        return _no_cluster("storage_home_avoidance", pc)
    if pc.remote_root is None:
        return finding("storage_home_avoidance", "unchecked", "remote root unresolved")
    if not pc.cluster.profile.storage:
        return finding(
            "storage_home_avoidance",
            "unchecked",
            "the cluster profile declares no storage roles: cannot tell whether attempt data lands on home",
        )
    fails: list[str] = []
    warns: list[str] = []
    for name, avoid in roles_for(pc.cluster, pc.remote_root):
        bad = sorted(set(avoid) & {"outputs", "checkpoints", "code"})
        if bad:
            fails.append(
                f"remote root {pc.remote_root} (attempt outputs/checkpoints/code) is on storage role "
                f"{name!r}, which should not hold {bad}"
            )
    venv = pc.spec.runtime.venv
    if venv:
        for name, avoid in roles_for(pc.cluster, venv):
            if "venv" in avoid:
                warns.append(f"runtime.venv {venv} is on storage role {name!r} (avoid_for venv)")
    for inp in pc.spec.inputs:
        for name, avoid in roles_for(pc.cluster, inp.uri):
            if "datasets" in avoid:
                warns.append(f"input {inp.name} ({inp.uri}) is on storage role {name!r} (avoid_for datasets)")
    if fails:
        return finding(
            "storage_home_avoidance",
            "fail",
            "; ".join(fails),
            warnings=warns or None,
            remediation=f"set a scratch remote root: hpclab cluster add {pc.cluster.cluster_id} --remote-root <scratch path>",
        )
    if warns:
        return finding("storage_home_avoidance", "warn", "; ".join(warns))
    return finding("storage_home_avoidance", "pass", "no attempt data on storage roles that avoid it")


_EXPORT_RE = re.compile(r"(?:^|[\s;&|])(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\S+)")


def check_cache_locations(pc: PlanContext) -> Finding:
    """Cache env vars (HF_HOME, UV_CACHE_DIR, ...) under a storage role that avoids caches."""
    if pc.cluster is None:
        return _no_cluster("cache_locations", pc)
    if not pc.cluster.profile.storage:
        return finding(
            "cache_locations",
            "unchecked",
            "the cluster profile declares no storage roles: cache locations not checked",
        )
    runtime = pc.spec.runtime
    values: dict[str, str] = {k: v for k, v in runtime.env.items() if k in CACHE_VARS}
    for line in runtime.setup:
        for m in _EXPORT_RE.finditer(line):
            if m.group(1) in CACHE_VARS:
                values.setdefault(m.group(1), m.group(2).strip("'\""))
    bad: dict[str, str] = {}
    for var, value in values.items():
        for name, avoid in roles_for(pc.cluster, value):
            if "caches" in avoid:
                bad[var] = f"{value} (storage role {name})"
    if bad:
        return finding(
            "cache_locations",
            "warn",
            f"cache directories on a storage role that avoids caches: {sorted(bad)}",
            locations=bad,
            remediation="point caches at scratch, e.g. HF_HOME: '{scratch}/.cache/huggingface'",
        )
    home_avoids = any(
        "caches" in role.avoid_for for name, role in pc.cluster.profile.storage.items() if name == "home"
    )
    unset = [v for v in CACHE_VARS if v not in values]
    if unset and home_avoids:
        return finding(
            "cache_locations",
            "warn",
            f"{unset} are unset and default to $HOME/.cache on the home role (small quota) "
            "if the program downloads models/datasets or compiles kernels",
            severity="info",
            unset=unset,
            remediation="set them in runtime.env to scratch paths (see the profile's runtime recipes)",
        )
    return finding("cache_locations", "pass", "no cache directories on storage that avoids caches")


# ---------------------------------------------------------------------------
# Snapshot-based checks
# ---------------------------------------------------------------------------


def manifest_paths(manifest: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {str(f["path"]): f for f in manifest.get("files", []) if isinstance(f, dict) and "path" in f}


def check_code_snapshot(pc: PlanContext) -> Finding:
    if pc.manifest is None:
        return finding("code_snapshot", "unchecked", "no code snapshot captured yet")
    n = int(pc.manifest.get("file_count") or len(pc.manifest.get("files", [])))
    if n == 0:
        return finding(
            "code_snapshot",
            "fail",
            "the code snapshot is empty",
            remediation="check code.project_root and code.exclude",
        )
    serious = [w for w in pc.snapshot_warnings if w.startswith("SECRET") or "are NOT in the snapshot" in w]
    ev: dict[str, Any] = {
        "file_count": n,
        "total_bytes": pc.manifest.get("total_bytes"),
        "excluded_count": len(pc.snapshot_excluded),
        "git_dirty": pc.git_dirty or None,
        "warnings": pc.snapshot_warnings[:MAX_LIST] or None,
    }
    if serious:
        return finding("code_snapshot", "warn", "; ".join(serious[:3]), **ev)
    if pc.snapshot_warnings:
        return finding("code_snapshot", "warn", "; ".join(pc.snapshot_warnings[:3]), severity="info", **ev)
    return finding("code_snapshot", "pass", f"{n} files captured", **ev)


def check_lockfiles_present(pc: PlanContext) -> Finding:
    if pc.manifest is None:
        return finding("lockfiles_present", "unchecked", "no code snapshot captured yet")
    files = manifest_paths(pc.manifest)
    declared = list(pc.spec.runtime.lockfiles)
    if declared:
        missing = [p for p in declared if posixpath.normpath(p) not in files]
        if missing:
            return finding(
                "lockfiles_present",
                "fail",
                f"declared lockfiles {missing} are not in the code snapshot",
                remediation="commit them (or list them in code.include_untracked), or remove them from runtime.lockfiles",
            )
        return finding("lockfiles_present", "pass", f"lockfiles {declared} captured")
    found = sorted(p for p in files if posixpath.basename(p) in LOCKFILE_NAMES and p.count("/") <= 1)
    if found:
        return finding(
            "lockfiles_present",
            "pass",
            f"found {found} in the snapshot (fingerprinted by the job wrapper)",
            found=found,
        )
    return finding(
        "lockfiles_present",
        "warn",
        "no lockfile in the snapshot: the Python environment is unpinned",
        severity="info",
        remediation="commit uv.lock / requirements.txt and list it in runtime.lockfiles",
    )


_SCRIPT_SUFFIXES = (".py", ".sh", ".bash")
_INTERPRETERS = re.compile(r"^(python[0-9.]*|bash|sh|torchrun|accelerate|deepspeed|uv|srun)$")


def entrypoint_targets(argv: Sequence[str], workdir: str) -> tuple[list[str], list[str]]:
    """(script paths, ``-m`` module names) the entrypoint refers to, relative to the snapshot."""
    scripts: list[str] = []
    modules: list[str] = []

    def rel(p: str) -> str:
        return posixpath.normpath(posixpath.join(workdir or ".", p))

    first = argv[0] if argv else ""
    if not _INTERPRETERS.match(posixpath.basename(first)) and (
        first.endswith(_SCRIPT_SUFFIXES) or first.startswith("./")
    ):
        if "{" not in first and not first.startswith("/"):
            scripts.append(rel(first))
        return scripts, modules
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-m" and i + 1 < len(argv):
            modules.append(argv[i + 1])
            break
        if a == "-c":  # `bash -c '<command>'` / `python -c '<code>'`: a command string, not a path
            break
        if a.startswith("-") or "{" in a or a.startswith("/") or any(c.isspace() for c in a):
            i += 1
            continue
        if a.endswith(_SCRIPT_SUFFIXES):
            scripts.append(rel(a))
            break
        i += 1
    return scripts, modules


def check_entrypoint_in_snapshot(pc: PlanContext) -> Finding:
    if pc.manifest is None:
        return finding("entrypoint_in_snapshot", "unchecked", "no code snapshot captured yet")
    files = manifest_paths(pc.manifest)
    ep = pc.spec.entrypoint
    scripts, modules = entrypoint_targets(ep.argv, ep.workdir)
    if not scripts and not modules:
        return finding(
            "entrypoint_in_snapshot",
            "unchecked",
            "could not tell which file the entrypoint runs",
            argv=ep.argv,
        )
    missing = [s for s in scripts if s not in files]
    if missing:
        return finding(
            "entrypoint_in_snapshot",
            "fail",
            f"entrypoint file(s) {missing} are not in the code snapshot",
            remediation="git add the file (untracked files are not captured) or list it in code.include_untracked",
        )
    if scripts and ep.argv and scripts[0] == posixpath.normpath(posixpath.join(ep.workdir, ep.argv[0])):
        entry = files[scripts[0]]
        if not int(entry.get("mode", 0o644)) & 0o111:
            return finding(
                "entrypoint_in_snapshot",
                "fail",
                f"{scripts[0]} is run directly but is not executable",
                remediation=f"chmod +x {scripts[0]} or prefix argv with bash/python",
            )
    unresolved = []
    for mod in modules:
        base = mod.replace(".", "/")
        cands = [f"{base}.py", f"{base}/__main__.py", f"src/{base}.py", f"src/{base}/__main__.py"]
        if not any(posixpath.normpath(posixpath.join(ep.workdir, c)) in files for c in cands):
            unresolved.append(mod)
    if unresolved:
        return finding(
            "entrypoint_in_snapshot",
            "warn",
            f"modules {unresolved} are not in the snapshot: they must come from the environment "
            "(an installed package is NOT covered by the code snapshot)",
        )
    return finding("entrypoint_in_snapshot", "pass", f"entrypoint {scripts or modules} is in the snapshot")


def diff_manifests(
    old: Mapping[str, Any] | None, new: Mapping[str, Any] | None, *, limit: int = 50
) -> dict[str, Any]:
    """Changed/added/removed paths between two snapshot manifests (bounded lists + counts)."""
    a, b = manifest_paths(old), manifest_paths(new)
    changed = sorted(
        p
        for p in a.keys() & b.keys()
        if (a[p].get("sha256"), a[p].get("mode")) != (b[p].get("sha256"), b[p].get("mode"))
    )
    added = sorted(b.keys() - a.keys())
    removed = sorted(a.keys() - b.keys())
    return {
        "changed": changed[:limit],
        "added": added[:limit],
        "removed": removed[:limit],
        "counts": {"changed": len(changed), "added": len(added), "removed": len(removed)},
    }


def check_code_drift(pc: PlanContext) -> Finding:
    """Extension/derivation: does the new snapshot differ from the existing runs' code?"""
    if pc.manifest is None:
        return finding("code_drift", "unchecked", "no code snapshot captured yet")
    new_digest = str(pc.manifest.get("digest"))
    others = [d for d in pc.base_digests if d != new_digest]
    if not others:
        return finding("code_drift", "pass", f"same code snapshot as the existing runs ({short(new_digest)})")
    diffs: dict[str, Any] = {}
    for d in others[:3]:
        base = pc.base_manifests.get(d)
        diffs[short(d)] = (
            diff_manifests(base, pc.manifest)
            if base is not None
            else {"note": "base snapshot not in the local store"}
        )
    return finding(
        "code_drift",
        "warn",
        f"the code differs from the existing runs' snapshot(s) {[short(d) for d in others[:3]]}: new runs use "
        f"{short(new_digest)} (comparisons will disclose mixed code digests)",
        new_digest=new_digest,
        base_digests=others[:MAX_LIST],
        diff=diffs,
        remediation="restore the original code (git checkout <commit>) before extending if the new runs must "
        "use identical code",
    )


# ---------------------------------------------------------------------------
# Script render
# ---------------------------------------------------------------------------


def render_preview(
    run: PlannedRun, cluster: ResolvedCluster, remote_root: str, *, attempt_id: str = PLACEHOLDER_ATTEMPT_ID
) -> str:
    """The job script this run would get, with a placeholder attempt id and token."""
    from hpclab.services.jobscript import attempt_layout, job_name_for, render_job_script

    assert run.exec_plan is not None
    return render_job_script(
        run_spec=run.run_spec,
        plan=run.exec_plan,
        cluster=cluster,
        run_id=run.run_id,
        attempt_id=attempt_id,
        token=PLACEHOLDER_TOKEN,
        job_name=job_name_for(run.label, PLACEHOLDER_TOKEN),
        layout=attempt_layout(remote_root, attempt_id),
    )


def check_script_render(pc: PlanContext) -> Finding:
    if pc.cluster is None or pc.remote_root is None:
        return finding("script_render", "unchecked", "cluster/remote root unresolved")
    errors: list[str] = []
    rendered = 0
    for r in pc.runs:
        if r.exec_plan is None:
            continue
        try:
            render_preview(r, pc.cluster, pc.remote_root)
            rendered += 1
        except HpcLabError as exc:
            errors.append(f"{r.label}: {exc.message}")
            if len(errors) >= 5:
                break
    if errors:
        return finding(
            "script_render",
            "fail",
            f"the job script cannot be rendered: {errors[0]}",
            errors=errors,
        )
    if rendered == 0:
        return finding("script_render", "unchecked", "no execution plans to render")
    return finding("script_render", "pass", f"{rendered} job scripts render")


STATIC_CHECKS: tuple[Callable[[PlanContext], Finding], ...] = (
    check_spec_schema,
    check_placeholders,
    check_matrix_duplicates,
    check_cluster_registered,
    check_remote_root,
    check_partition_exists,
    check_time_within_partition_max,
    check_gpu_type_in_partition,
    check_gpus_limits,
    check_account_present,
    check_qos_required,
    check_interactive_partition_batch,
    check_storage_home_avoidance,
    check_cache_locations,
    check_outputs_declared,
    check_evaluation_output_declared,
    check_smoke_gate,
    check_preemptible_gate,
    check_walltime_signal,
    check_budget,
    check_code_snapshot,
    check_entrypoint_in_snapshot,
    check_lockfiles_present,
    check_inputs_pinning,
    check_script_render,
)


def run_static_checks(pc: PlanContext, *, drift: bool = False) -> list[Finding]:
    out: list[Finding] = []
    for fn in STATIC_CHECKS:
        try:
            out.append(fn(pc))
        except Exception as exc:
            name = fn.__name__.removeprefix("check_")
            log.exception("prepare check %s crashed", name)
            out.append(finding(name, "unchecked", f"check crashed: {exc!r}"))
    if drift:
        out.append(check_code_drift(pc))
    return out


# ---------------------------------------------------------------------------
# Resource advice (pure part)
# ---------------------------------------------------------------------------

ADVICE_MIN_N = 3
ADVICE_OVERHEAD_S = 300
ADVICE_TIME_FACTOR = 1.15
ADVICE_MEM_FACTOR = 1.25
ADVICE_OVER_FACTOR = 3.0


class ResourceAdviceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    kind: str
    labels: list[str]
    cells: int
    n: int
    p90_elapsed_s: int | None
    requested_time: str
    suggested_time: str | None
    verdict: Literal["under_request", "over_request", "ok", "insufficient_history"]
    max_rss_bytes: int | None = None
    requested_mem: str | None = None
    suggested_mem: str | None = None
    sample_attempt_ids: list[str] = Field(default_factory=list)


class ResourceAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["advice", "insufficient_history", "unavailable"]
    items: list[ResourceAdviceItem] = Field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class HistoryRow:
    attempt_id: str
    family: str
    elapsed_s: int
    max_rss_bytes: int | None


def family_key(run_spec: RunSpec, resources: ResourceSpec) -> str:
    """Resource family: same program + params (minus seed) + GPU shape => comparable runtimes."""
    from hpclab.domain.hashing import digest

    template = list(run_spec.entrypoint.argv)
    return digest(
        {
            "kind": run_spec.kind,
            "argv": template,
            "argv_tail": list(run_spec.argv[len(template) :]),
            "workdir": run_spec.entrypoint.workdir,
            "params": {k: v for k, v in run_spec.params.items() if k != mx.SEED_AXIS},
            "gpu_type": resources.gpu_type,
            "gpus": resources.total_gpus,
        }
    )


def p90(values: Sequence[int]) -> int:
    """Nearest-rank 90th percentile."""
    s = sorted(values)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def _mem_str(nbytes: int) -> str:
    mib = math.ceil(nbytes / (1024 * 1024))
    return f"{math.ceil(mib / 1024)}G" if mib >= 1024 else f"{mib}M"


def advise_resources(runs: Sequence[PlannedRun], history: Sequence[HistoryRow]) -> ResourceAdvice:
    """Per family of planned runs: p90 of past COMPLETED+valid elapsed times -> time advice."""
    by_family: dict[str, list[HistoryRow]] = {}
    for h in history:
        by_family.setdefault(h.family, []).append(h)
    groups: dict[str, list[PlannedRun]] = {}
    for r in runs:
        groups.setdefault(family_key(r.run_spec, r.resources), []).append(r)
    items: list[ResourceAdviceItem] = []
    for fam, members in groups.items():
        rep = members[0]
        rows = by_family.get(fam, [])
        req = rep.resources
        base = {
            "family": short(fam),
            "kind": rep.kind,
            "labels": [m.label for m in members[:10]],
            "cells": len(members),
            "n": len(rows),
            "requested_time": req.time,
            "requested_mem": req.mem,
            "sample_attempt_ids": [h.attempt_id for h in rows[:10]],
        }
        if len(rows) < ADVICE_MIN_N:
            items.append(
                ResourceAdviceItem(
                    **base, p90_elapsed_s=None, suggested_time=None, verdict="insufficient_history"
                )
            )
            continue
        q = p90([h.elapsed_s for h in rows])
        suggested_s = int(math.ceil((q * ADVICE_TIME_FACTOR + ADVICE_OVERHEAD_S) / 60.0) * 60)
        if suggested_s > req.time_seconds:
            verdict: Literal["under_request", "over_request", "ok"] = "under_request"
        elif req.time_seconds > ADVICE_OVER_FACTOR * q:
            verdict = "over_request"
        else:
            verdict = "ok"
        rss = [h.max_rss_bytes for h in rows if h.max_rss_bytes]
        max_rss = max(rss) if rss else None
        items.append(
            ResourceAdviceItem(
                **base,
                p90_elapsed_s=q,
                suggested_time=format_slurm_duration(suggested_s),
                verdict=verdict,
                max_rss_bytes=max_rss,
                suggested_mem=_mem_str(int(max_rss * ADVICE_MEM_FACTOR)) if max_rss else None,
            )
        )
    if any(i.verdict != "insufficient_history" for i in items):
        return ResourceAdvice(
            status="advice",
            items=items,
            note=f"suggested time = p90(elapsed) x {ADVICE_TIME_FACTOR} + {ADVICE_OVERHEAD_S}s over "
            "COMPLETED attempts with valid results of the same family",
        )
    return ResourceAdvice(
        status="insufficient_history",
        items=items,
        note=f"fewer than {ADVICE_MIN_N} comparable COMPLETED+valid attempts in the ledger",
    )


def check_resource_advice(advice: ResourceAdvice) -> Finding:
    if advice.status == "unavailable":
        return finding("resource_advice", "unchecked", advice.note or "ledger history unavailable")
    if advice.status == "insufficient_history":
        return finding("resource_advice", "unchecked", advice.note)
    under = [i for i in advice.items if i.verdict == "under_request"]
    over = [i for i in advice.items if i.verdict == "over_request"]
    if under:
        i = under[0]
        return finding(
            "resource_advice",
            "warn",
            f"time {i.requested_time} is below the suggested {i.suggested_time} (p90 {i.p90_elapsed_s}s over "
            f"{i.n} past runs): likely TIMEOUT",
            families=[x.family for x in under],
            remediation=f"set resources.time to at least {i.suggested_time}",
        )
    if over:
        i = over[0]
        return finding(
            "resource_advice",
            "warn",
            f"time {i.requested_time} is > {ADVICE_OVER_FACTOR:g}x the p90 elapsed ({i.p90_elapsed_s}s): "
            f"longer queue waits and reserved budget; suggested {i.suggested_time}",
            severity="info",
            families=[x.family for x in over],
        )
    return finding("resource_advice", "pass", "requested times match past runs")


# ---------------------------------------------------------------------------
# Remote checks (bounded; unchecked on any transport problem)
# ---------------------------------------------------------------------------


class Deadline:
    """Real-time budget for remote I/O (monotonic; independent of the injected clock)."""

    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + max(0.0, seconds)

    def remaining(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def timeout(self, cap: float) -> float:
        rem = self.remaining()
        if rem <= 0.5:
            raise TransportTimeout("prepare's remote-check time budget is exhausted", maybe_executed=False)
        return min(cap, rem)


def bounded_call[T](fn: Callable[[], T], timeout: float, what: str) -> T:
    """Run ``fn`` in a helper thread and give up after ``timeout`` seconds (for adapter calls
    without a timeout parameter). A late result is discarded; the call is read-only/test-only."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, name=f"hpclab-prepare-{what}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TransportTimeout(f"{what} did not finish within {timeout:.0f}s", maybe_executed=True)
    if "error" in box:
        raise box["error"]
    value: T = box["value"]
    return value


def _transport_unchecked(check: str, exc: BaseException, cluster_id: str, what: str) -> Finding:
    if isinstance(exc, AuthRequiredError) or (
        isinstance(exc, HpcLabError) and str(exc.code) == "auth_required"
    ):
        return finding(
            check,
            "unchecked",
            f"{what}: SSH authentication required",
            remediation=f"hpclab auth {cluster_id}",
        )
    msg = exc.message if isinstance(exc, HpcLabError) else repr(exc)
    return finding(check, "unchecked", f"{what}: {msg}", error_type=type(exc).__name__)


def parse_pyvenv_cfg(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip().lower()] = v.strip()
    return out


def venv_python_version(cfg: Mapping[str, str]) -> tuple[int, int] | None:
    for key in ("version_info", "version"):
        m = re.match(r"^(\d+)\.(\d+)", cfg.get(key, ""))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


_TORCH_CUDA_RE = re.compile(
    r"^cuda\s*(?::[^=\n]*)?=\s*(?:'(?P<a>[^']*)'|\"(?P<b>[^\"]*)\"|(?P<none>None))", re.MULTILINE
)
_TORCH_VERSION_RE = re.compile(r"^__version__\s*(?::[^=\n]*)?=\s*['\"](?P<v>[^'\"]+)['\"]", re.MULTILINE)


def parse_torch_version_py(text: str) -> tuple[str | None, str | None, bool]:
    """(torch version, cuda build or None, parsed) from ``torch/version.py``."""
    vm = _TORCH_VERSION_RE.search(text)
    cm = _TORCH_CUDA_RE.search(text)
    version = vm.group("v") if vm else None
    if cm is None:
        return version, None, False
    if cm.group("none"):
        return version, None, True
    return version, (cm.group("a") if cm.group("a") is not None else cm.group("b")), True


def cuda_tuple(text: str | None) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d+)\.(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def normalize_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def project_names(manifest: Mapping[str, Any] | None, pyproject_text: str | None = None) -> set[str]:
    """Distribution/package names of the project itself (to spot an editable install of it)."""
    names: set[str] = set()
    for path in manifest_paths(manifest):
        parts = path.split("/")
        if parts[-1] == "__init__.py" and len(parts) == 2:
            names.add(normalize_dist(parts[0]))
        elif parts[-1] == "__init__.py" and len(parts) == 3 and parts[0] == "src":
            names.add(normalize_dist(parts[1]))
    if pyproject_text:
        try:
            data = tomllib.loads(pyproject_text)
            n = data.get("project", {}).get("name") or data.get("tool", {}).get("poetry", {}).get("name")
            if isinstance(n, str):
                names.add(normalize_dist(n))
        except (tomllib.TOMLDecodeError, AttributeError):
            pass
    return names


def _editable_name(path: str) -> tuple[str, str] | None:
    """(normalized dist name, kind) for an editable marker path, else None."""
    base = posixpath.basename(path)
    parent = posixpath.basename(posixpath.dirname(path))
    if base == "direct_url.json" and parent.endswith(".dist-info"):
        return normalize_dist(parent.removesuffix(".dist-info").split("-")[0]), "direct_url"
    if base.startswith("__editable__.") and base.endswith(".pth"):
        return normalize_dist(base.removeprefix("__editable__.").removesuffix(".pth").split("-")[0]), "pth"
    if base.endswith(".egg-link"):
        return normalize_dist(base.removesuffix(".egg-link")), "egg_link"
    return None


def resolve_cluster_path(path: str, cluster: ResolvedCluster, user: Callable[[], str]) -> str | None:
    """Resolve profile placeholders, ``$USER``/``~``/``$HOME`` statically; None if impossible."""
    out = cluster.expand(path)
    if "{user}" in out or "$USER" in out or "${USER}" in out:
        u = cluster.overlay.user or user()
        out = out.replace("{user}", u)
        out = re.sub(r"\$\{USER\}|\$USER(?![A-Za-z0-9_])", u, out)
    home = cluster.storage_path("home")
    if out.startswith("~/") or "$HOME" in out or "${HOME}" in out:
        if not home or "{" in home:
            return None
        if out.startswith("~/"):
            out = home + out[1:]
        out = re.sub(r"\$\{HOME\}|\$HOME(?![A-Za-z0-9_])", home, out)
    if not out.startswith("/") or "{" in out or "$" in out:
        return None
    return posixpath.normpath(out)


def active_fact(cluster: ResolvedCluster, key: str, now: datetime) -> Any:
    from hpclab.profiles.loader import active_facts

    for fact in active_facts(cluster.profile, now):
        if fact.key == key:
            return fact.value
    return None


@dataclass
class _Sub:
    name: str
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def check_env_guard_static(
    *,
    transport: Transport,
    cluster: ResolvedCluster,
    runtime_venv: str | None,
    gpus: int,
    gpu_types: Sequence[str],
    project: set[str],
    now: datetime,
    deadline: Deadline,
) -> list[Finding]:
    """Static look at ``runtime.venv`` on the cluster: exists / ``pyvenv.cfg``; torch CUDA build
    vs the profile fact ``driver_cuda`` and the GPU architectures; editable installs that would
    bypass the code snapshot. Returns ``env_guard_static`` and ``snapshot_bypass`` findings.

    Bounded: at most ~4 + 16 small reads and one ``find -maxdepth 2`` listing."""
    cid = cluster.cluster_id
    if not runtime_venv:
        msg = "no runtime.venv declared: the environment is only checked on the compute node (runtime.guard)"
        return [finding("env_guard_static", "unchecked", msg), finding("snapshot_bypass", "unchecked", msg)]
    try:
        venv = resolve_cluster_path(
            runtime_venv,
            cluster,
            lambda: bounded_call(transport.remote_user, deadline.timeout(10.0), "remote_user"),
        )
    except (HpcLabError, OSError) as exc:
        f = _transport_unchecked("env_guard_static", exc, cid, "resolving runtime.venv")
        return [f, f.model_copy(update={"check": "snapshot_bypass"})]
    if venv is None:
        msg = f"cannot resolve runtime.venv {runtime_venv!r} to an absolute cluster path"
        return [finding("env_guard_static", "unchecked", msg), finding("snapshot_bypass", "unchecked", msg)]
    subs: list[_Sub] = []
    cfg_path = f"{venv}/pyvenv.cfg"
    try:
        st = transport.stat([venv, cfg_path], timeout=deadline.timeout(10.0))
    except (HpcLabError, OSError) as exc:
        f = _transport_unchecked("env_guard_static", exc, cid, "stat runtime.venv")
        return [f, f.model_copy(update={"check": "snapshot_bypass"})]
    if not st.get(venv) or not st[venv].exists:
        return [
            finding(
                "env_guard_static",
                "fail",
                f"runtime.venv {venv} does not exist on the cluster (the job guard would fail with exit 3)",
                venv=venv,
                remediation=f"create it on the login node, e.g. uv venv {venv}",
            ),
            finding("snapshot_bypass", "unchecked", "venv missing", venv=venv),
        ]
    if not st.get(cfg_path) or not st[cfg_path].exists:
        return [
            finding(
                "env_guard_static",
                "fail",
                f"{venv} has no pyvenv.cfg: it is not a virtualenv",
                venv=venv,
            ),
            finding("snapshot_bypass", "unchecked", "not a virtualenv", venv=venv),
        ]
    subs.append(_Sub("venv", "pass", f"{venv} exists"))
    try:
        cfg = parse_pyvenv_cfg(
            transport.read_range(cfg_path, 0, 8192, timeout=deadline.timeout(10.0)).data.decode(
                "utf-8", "replace"
            )
        )
    except (HpcLabError, OSError) as exc:
        f = _transport_unchecked("env_guard_static", exc, cid, "reading pyvenv.cfg")
        return [f, f.model_copy(update={"check": "snapshot_bypass"})]
    pyver = venv_python_version(cfg)
    if pyver is None:
        subs.append(_Sub("python", "unchecked", "pyvenv.cfg has no version"))
        return _finish_env(
            subs, cid, venv, bypass=finding("snapshot_bypass", "unchecked", "python version unknown")
        )
    site = f"{venv}/lib/python{pyver[0]}.{pyver[1]}/site-packages"
    subs.append(_Sub("python", "pass", f"python {pyver[0]}.{pyver[1]}", {"site_packages": site}))

    # torch build vs driver / GPU architecture
    torch_cuda: tuple[int, int] | None = None
    try:
        text = transport.read_range(
            f"{site}/torch/version.py", 0, 16384, timeout=deadline.timeout(10.0)
        ).data.decode("utf-8", "replace")
        tver, tcuda, parsed = parse_torch_version_py(text)
        if not parsed:
            subs.append(_Sub("torch", "unchecked", "torch/version.py has no cuda field", {"torch": tver}))
        elif tcuda is None:
            status = "fail" if gpus > 0 else "pass"
            subs.append(
                _Sub(
                    "torch",
                    status,
                    f"torch {tver} is a CPU-only build" + (" but GPUs are requested" if gpus else ""),
                )
            )
        else:
            torch_cuda = cuda_tuple(tcuda)
            subs.append(
                _Sub("torch", "pass", f"torch {tver} built for CUDA {tcuda}", {"torch": tver, "cuda": tcuda})
            )
    except FileNotFoundError:
        try:
            site_st = transport.stat([site], timeout=deadline.timeout(10.0)).get(site)
        except (HpcLabError, OSError) as exc:
            site_st = None
            subs.append(
                _Sub("torch", "unchecked", f"cannot stat site-packages: {getattr(exc, 'message', exc)!r}")
            )
        if site_st is not None and site_st.exists:
            subs.append(_Sub("torch", "pass", "torch is not installed in the venv"))
        elif site_st is not None:
            subs.append(_Sub("site_packages", "unchecked", f"{site} does not exist (unusual venv layout)"))
    except (HpcLabError, OSError) as exc:
        subs.append(
            _Sub("torch", "unchecked", f"cannot read torch/version.py: {getattr(exc, 'message', exc)!r}")
        )
    if torch_cuda is not None and gpus > 0:
        driver = active_fact(cluster, "driver_cuda", now)
        dtuple = cuda_tuple(str(driver)) if driver is not None else None
        if dtuple is None:
            subs.append(_Sub("driver", "unchecked", "no current driver_cuda fact in the profile"))
        elif torch_cuda[0] > dtuple[0]:
            subs.append(
                _Sub(
                    "driver",
                    "fail",
                    f"torch is built for CUDA {torch_cuda[0]}.{torch_cuda[1]} but the driver supports CUDA "
                    f"{dtuple[0]}.{dtuple[1]}: torch.cuda.is_available() will be False",
                    {"driver_cuda": str(driver)},
                )
            )
        elif torch_cuda > dtuple:
            subs.append(
                _Sub(
                    "driver",
                    "warn",
                    f"torch CUDA {torch_cuda[0]}.{torch_cuda[1]} is newer than the driver's {dtuple[0]}.{dtuple[1]} "
                    "(minor-version compatibility; PTX JIT/new features may fail)",
                    {"driver_cuda": str(driver)},
                )
            )
        else:
            subs.append(_Sub("driver", "pass", f"driver supports CUDA {driver}"))
        if not gpu_types:
            subs.append(
                _Sub(
                    "arch",
                    "unchecked",
                    "GPU type unknown (no resources.gpu_type and the partition lists no GPU types): "
                    "architecture support of the torch build not checked",
                )
            )
        for gt in gpu_types:
            arch = cluster.profile.gpu_arch.get(gt)
            need = ARCH_MIN_CUDA.get(arch or "")
            if arch is None or need is None:
                subs.append(_Sub(f"arch:{gt}", "unchecked", f"architecture of {gt} unknown"))
            elif torch_cuda < need:
                subs.append(
                    _Sub(
                        f"arch:{gt}",
                        "fail",
                        f"{gt} ({arch}) needs a CUDA >= {need[0]}.{need[1]} build; torch has "
                        f"{torch_cuda[0]}.{torch_cuda[1]} (no kernels for {arch})",
                    )
                )
            else:
                subs.append(_Sub(f"arch:{gt}", "pass", f"{gt} ({arch}) supported"))
    bypass = _check_editables(transport, site, project, deadline, cid)
    return _finish_env(subs, cid, venv, bypass=bypass)


def _finish_env(subs: list[_Sub], cid: str, venv: str, *, bypass: Finding) -> list[Finding]:
    status = _worst(s.status for s in subs)
    bad = [s.message for s in subs if s.status in ("fail", "warn", "unchecked")]
    evidence = {
        "venv": venv,
        "checks": [{"name": s.name, "status": s.status, "message": s.message, **s.data} for s in subs],
    }
    msg = "; ".join(bad) if bad else "; ".join(s.message for s in subs)
    rem = None
    if status == "fail":
        rem = (
            "install a torch wheel matching the driver, e.g. uv pip install 'torch>=2.7' "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    main = Finding(
        check="env_guard_static",
        status=status,
        severity="block" if status == "fail" else ("warn" if status == "warn" else "info"),
        message=msg,
        evidence=evidence,
        remediation=rem,
    )
    return [main, bypass]


def _check_editables(
    transport: Transport, site: str, project: set[str], deadline: Deadline, cid: str
) -> Finding:
    argv = [
        "find",
        site,
        "-maxdepth",
        "2",
        "(",
        "-name",
        "direct_url.json",
        "-o",
        "-name",
        "__editable__*.pth",
        "-o",
        "-name",
        "*.egg-link",
        ")",
    ]
    try:
        res = transport.run(argv, timeout=deadline.timeout(15.0))
    except (HpcLabError, OSError) as exc:
        return _transport_unchecked("snapshot_bypass", exc, cid, "listing site-packages")
    if res.exit_code != 0 and not res.stdout.strip():
        return finding(
            "snapshot_bypass",
            "unchecked",
            f"cannot list {site}: {res.stderr.strip()[:200] or f'exit {res.exit_code}'}",
        )
    # A non-zero exit with output (e.g. an unreadable sub-directory) is a *partial* listing: the
    # project's own editable install may be in the part that was not listed.
    partial = (
        f"the listing of {site} is incomplete: {res.stderr.strip()[-200:] or f'exit {res.exit_code}'}"
        if res.exit_code != 0
        else None
    )
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if len(lines) > 500 and partial is None:
        partial = f"the listing of {site} has {len(lines)} install records; only 500 were inspected"
    paths = lines[:500]
    editables: dict[str, str] = {}
    unread = 0
    reads = 0
    for p in paths:
        parsed = _editable_name(p)
        if parsed is None:
            continue
        name, kind = parsed
        if kind != "direct_url":
            editables[name] = p
            continue
        if reads >= 16:
            unread += 1
            continue
        reads += 1
        try:
            data = json.loads(transport.read_range(p, 0, 4096, timeout=deadline.timeout(10.0)).data or b"{}")
        except (HpcLabError, OSError, ValueError):
            unread += 1
            continue
        if (
            isinstance(data, dict)
            and isinstance(data.get("dir_info"), dict)
            and data["dir_info"].get("editable")
        ):
            editables[name] = str(data.get("url") or p)
    own = sorted(n for n in editables if n in project)
    others = sorted(n for n in editables if n not in project)
    if own:
        return finding(
            "snapshot_bypass",
            "fail",
            f"the project itself is installed editable in the venv ({own}): imports resolve to the mutable "
            "cluster checkout, not the frozen code snapshot",
            editable={n: editables[n] for n in own},
            remediation="reinstall it non-editable (uv pip install .) or remove it from the venv; the snapshot "
            "puts the project on the path via the work dir",
        )
    if partial is not None:
        return finding(
            "snapshot_bypass",
            "unchecked",
            partial + (f"; editable installs seen so far: {others[:10]}" if others else ""),
            editable={n: editables[n] for n in others[:MAX_LIST]} or None,
        )
    if others:
        return finding(
            "snapshot_bypass",
            "warn",
            f"editable installs {others[:10]} run mutable code that is not captured by the snapshot",
            editable={n: editables[n] for n in others[:MAX_LIST]},
        )
    if unread:
        return finding("snapshot_bypass", "unchecked", f"{unread} install records could not be read")
    return finding("snapshot_bypass", "pass", "no editable installs in the venv")


def check_scheduler_test(
    *,
    transport: Transport,
    scheduler: Scheduler,
    cluster: ResolvedCluster,
    run: PlannedRun,
    remote_root: str,
    plan_id: str,
    deadline: Deadline,
) -> Finding:
    """``sbatch --test-only`` of one representative rendered script (nothing is queued)."""
    cid = cluster.cluster_id
    if run.exec_plan is None:
        return finding("scheduler_test", "unchecked", "no execution plan to test")
    try:
        script = render_preview(run, cluster, remote_root)
    except HpcLabError as exc:
        return finding("scheduler_test", "unchecked", f"script does not render: {exc.message}")
    workdir = f"{remote_root}/.prepare/{plan_id}"
    path = f"{workdir}/test-only.sbatch"
    try:
        transport.write_file(path, script.encode("utf-8"), mode=0o644, timeout=deadline.timeout(15.0))
    except (HpcLabError, OSError) as exc:
        return _transport_unchecked("scheduler_test", exc, cid, "uploading the test script")
    try:
        resp = bounded_call(
            lambda: scheduler.submit(path, workdir=workdir, test_only=True),
            deadline.timeout(30.0),
            "sbatch --test-only",
        )
    except SubmissionRejected as exc:
        retry_later = bool(exc.details.get("retry_later"))
        return finding(
            "scheduler_test",
            "warn" if retry_later else "fail",
            f"sbatch --test-only refused the job: {exc.message}",
            reason_code=exc.details.get("reason_code"),
            stderr=str(exc.details.get("stderr") or "")[-500:] or None,
            label=run.label,
        )
    except (HpcLabError, OSError) as exc:
        return _transport_unchecked("scheduler_test", exc, cid, "sbatch --test-only")
    finally:
        _cleanup(transport, path, workdir, deadline)
    from hpclab.adapters.slurm import parse_test_only

    est = parse_test_only(resp.raw)
    return finding(
        "scheduler_test",
        "pass",
        f"sbatch --test-only accepted {run.label}"
        + (f" (estimated start {est.start_time} on {est.partition})" if est and est.start_time else ""),
        estimated_start=est.start_time if est else None,
        partition=est.partition if est else None,
        raw=resp.raw[:300],
        note="--test-only validates the request against current limits; it is not an environment smoke test",
    )


def _cleanup(transport: Transport, path: str, workdir: str, deadline: Deadline) -> None:
    try:
        t = min(5.0, max(1.0, deadline.remaining()))
        transport.run(["rm", "-f", "--", path], timeout=t)
        transport.run(["rmdir", "--", workdir], timeout=t)
    except (HpcLabError, OSError):
        log.debug("could not remove %s", path, exc_info=True)


def read_local_blob(store_root: Path, entry: Mapping[str, Any], limit: int = 256 * 1024) -> str | None:
    """Text of a small file from the local snapshot blob store (e.g. pyproject.toml)."""
    sha = str(entry.get("sha256", "")).removeprefix("sha256:")
    if not re.match(r"^[0-9a-f]{64}$", sha) or int(entry.get("size") or 0) > limit:
        return None
    for suffix in ("", ".x"):
        p = store_root / "objects" / sha[:2] / (sha + suffix)
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


__all__ = [
    "Deadline",
    "Finding",
    "PlanContext",
    "PlannedRun",
    "ResourceAdvice",
    "ResourceAdviceItem",
    "ValidationOverride",
    "ValidationReport",
    "advise_resources",
    "apply_overrides",
    "build_report",
    "check_env_guard_static",
    "check_scheduler_test",
    "run_static_checks",
]
