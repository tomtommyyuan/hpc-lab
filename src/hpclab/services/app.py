"""The application facade the controller serves.

``build_registry()`` maps every public method name (MCP tools, resources, CLI commands) to
its service function; ``create_app(ctx)`` wraps it for the controller: ``dispatch`` plus the
union of all service modules' outbox handlers and periodic tasks.

Method table entries are ``"module:function"`` strings resolved at import time. Summarizers
produce the one-line ``summary`` of each Envelope — the line an agent reads first — so they
favour state and next step over restating inputs.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hpclab.services.context import ServiceContext
from hpclab.services.outbox import Handler
from hpclab.services.periodic import PeriodicTask
from hpclab.services.registry import Registry

log = logging.getLogger("hpclab.app")


def _g(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else getattr(cur, p, None)
    return default if cur is None else cur


def _n(obj: Any, name: str) -> int:
    v = _g(obj, name)
    return len(v) if isinstance(v, (list, tuple, dict)) else 0


# ---------------------------------------------------------------------------
# Summarizers (defensive: never raise; the registry falls back if they do)
# ---------------------------------------------------------------------------


def _num(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _codes(counts: Any) -> str:
    """Cell-code counts without the aggregate keys ('planned', 'valid')."""
    if not isinstance(counts, dict):
        return ""
    return ", ".join(f"{k}={v}" for k, v in counts.items() if v and k not in ("planned", "valid"))


def _s_prepare(d: Any) -> str:
    status = _g(d, "status", default="?")
    plan = _g(d, "plan_id")
    if status == "preparing":
        return (
            f"plan {plan} preparing (large code snapshot in the background) — poll {_g(d, 'operation_ref')}"
        )
    runs = _g(d, "runs", default=[]) or []
    main = sum(1 for r in runs if _g(r, "kind") == "main")
    other = len(runs) - main
    kinds = f"{main} main" + (f" + {other} smoke/other" if other else "")
    findings = _g(d, "validation", "findings", default=[]) or []
    blocking = [
        str(_g(f, "check"))
        for f in findings
        if _g(f, "severity") == "block" and _g(f, "status") == "fail" and not _g(f, "override_reason")
    ]
    attention = sum(1 for f in findings if _g(f, "status") in ("warn", "fail", "unchecked"))
    gh = _g(d, "budget", "requested_gpu_hours_total")
    cap = _g(d, "budget", "budget_gpu_hours")
    gh_s = f", {_num(gh)} GPU-h requested" if isinstance(gh, (int, float)) else ""
    gh_s += f" of {_num(cap)} budget" if isinstance(cap, (int, float)) and gh_s else ""
    val = f"validation {_g(d, 'validation', 'status', default='?')}"
    if attention:
        val += f" ({attention} finding(s) to read)"
    if blocking or status == "invalid":
        what = ", ".join(blocking) or "see findings"
        return f"plan {plan} {status}: {len(runs)} run(s); BLOCKED by {what} — fix and prepare again"
    return f"plan {plan} {status}: {len(runs)} run(s) ({kinds}){gh_s}; {val} — nothing submitted yet"


def _s_submit(d: Any) -> str:
    return (
        f"attempt {_g(d, 'attempt_id')} {_g(d, 'phase', default='staging')} — receipt only, not yet accepted by Slurm; "
        f"poll {_g(d, 'operation_ref')} after {_g(d, 'poll_after_ms', default=2000)} ms"
    )


def _s_runs_list(d: Any) -> str:
    counts = _g(d, "counts") or {}
    total = _g(d, "total", default=_n(d, "runs"))
    codes = _codes(counts) or "none"
    if _g(d, "view") == "grid":
        planned = counts.get("planned", total) if isinstance(counts, dict) else total
        valid = counts.get("valid", counts.get("V", 0)) if isinstance(counts, dict) else 0
        return f"grid: {valid}/{planned} planned main run(s) valid ({codes})"
    shown = _n(d, "runs")
    in_scope = (
        sum(v for k, v in counts.items() if k not in ("planned", "valid") and isinstance(v, int))
        if isinstance(counts, dict)
        else 0
    )
    if _g(d, "changes_cursor") and isinstance(total, int) and in_scope > total:
        # a changed_since listing: `total` counts the changed runs, `counts` the whole scope
        changed = f"{total} run(s) changed" if total else "no run changed"
        return f"{changed} since the cursor (all {in_scope} run(s): {codes})"
    more = f" (showing {shown})" if isinstance(total, int) and shown < total else ""
    return f"{total} run(s){more}: {codes}"


def _s_inspect(d: Any) -> str:
    run = _g(d, "run", "label") or _g(d, "run", "run_id")
    status = _g(d, "run", "status") or _g(d, "status")
    code = _g(d, "run", "code")
    head = f"run {run}: {status}" + (f" [{code}]" if code else "")
    focus = _g(d, "focus_attempt_id")
    att = next((a for a in _g(d, "attempts", default=[]) or [] if _g(a, "attempt_id") == focus), None)
    if att is not None:
        job = _g(att, "job", "job_id")
        head += f"; attempt #{_g(att, 'seq')} {_g(att, 'scheduler_state')}" + (f" (job {job})" if job else "")
        if _g(att, "phase") in ("submission_uncertain", "duplicate_detected"):
            head += " — submission uncertain: do NOT resubmit"
    diag = _g(d, "diagnosis", "items")
    if diag:
        first = diag[0]
        head += f"; diagnosis {_g(first, 'code')}: {_g(first, 'title')} ({_g(first, 'confidence')})"
    for op in _g(d, "operations", default=[]) or []:
        if _g(op, "state") != "blocked":
            continue
        fix = str(_g(op, "error", "details", "remediation", default="") or "")
        if fix.startswith("hpclab "):
            fix = f"user must run `{fix}`"
        head += f"; {_g(op, 'kind')} blocked: {_g(op, 'stage')}" + (f" — {fix}" if fix else "")
        break
    pending = _g(d, "pending")
    if pending is not None:
        head += f"; pending {_g(pending, 'reason_code')} ({_g(pending, 'category')})"
    prog = _g(d, "progress")
    if prog is not None and _g(prog, "step") is not None and status == "running":
        head += f"; step {_g(prog, 'step')}/{_g(prog, 'total_steps', default='?')}"
    return head


def _s_generic_state(label: str) -> Callable[[Any], str]:
    def f(d: Any) -> str:
        for key in ("state", "status", "phase", "cancel_state"):
            v = _g(d, key)
            if v:
                counts = _codes(_g(d, "counts"))
                return f"{label}: {v}" + (f" ({counts})" if counts else "")
        return label

    return f


def _s_recover(d: Any) -> str:
    if _g(d, "mode") == "batch" or (_g(d, "cells") is not None and _g(d, "plan_id")):
        cells = _g(d, "cells", default=[]) or []
        ok = _g(d, "eligible_count", default=sum(1 for c in cells if _g(c, "eligible")))
        total = _g(d, "total", default=len(cells))
        return f"recovery plan {_g(d, 'plan_id')}: {ok}/{total} cell(s) eligible — nothing submitted"
    diag = _g(d, "diagnosis", "primary_code")
    why = f" for {diag}" if diag else ""
    if _g(d, "eligible"):
        patch = _g(d, "resource_patch") or {}
        change = (
            f" ({', '.join(f'{k}={v}' for k, v in patch.items())})"
            if isinstance(patch, dict) and patch
            else ""
        )
        return f"eligible: {_g(d, 'strategy')} plan ready{change}{why} — nothing submitted yet"
    blocking = "; ".join(str(x) for x in (_g(d, "blocking", default=[]) or [])[:2])
    probe = _g(d, "probe", "run_id")
    if probe:
        return f"not eligible yet{why}: submit checkpoint probe run {probe} first" + (
            f" ({blocking})" if blocking else ""
        )
    return f"not eligible{why}: {blocking or 'see hints'}"


def _s_collect(d: Any) -> str:
    if _g(d, "mode") == "single":
        ev = _g(d, "evaluation")
        verb = "created" if _g(d, "created") else "existing"
        return f"evaluation {_g(ev, 'evaluation_id')} ({verb}): {_g(ev, 'status')}"
    c = _g(d, "counts", default={}) or {}
    reasons: dict[str, int] = {}
    for item in _g(d, "skipped", default=[]) or []:
        r = str(_g(item, "reason", default="other"))
        reasons[r] = reasons.get(r, 0) + 1
    why = f" ({', '.join(f'{k}={v}' for k, v in sorted(reasons.items()))})" if reasons else ""
    tail = ""
    if _g(d, "pending"):
        tail = f"; {_g(d, 'pending')} still running in background — poll {_g(d, 'operation_ref')} before comparing"
    return f"collected {c.get('collected', 0)} new evaluation(s), skipped {c.get('skipped', 0)}{why}{tail}"


def _s_logs(d: Any) -> str:
    size = _g(d, "size")
    flags = [f for f in ("truncated", "rotated", "eof") if _g(d, f)]
    note = f" — {_g(d, 'note')}" if _g(d, "note") else ""
    text = _g(d, "text", default="") or ""
    return f"{_g(d, 'stream')} of {_g(d, 'attempt_id')}: {len(text)} chars (file {size} B){' ' + ','.join(flags) if flags else ''}{note}"


def _s_artifacts(d: Any) -> str:
    arts = _g(d, "artifacts", default=[]) or []
    by: dict[str, int] = {}
    for a in arts:
        k = str(_g(a, "integrity", default="?"))
        by[k] = by.get(k, 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
    return f"{_g(d, 'total', default=len(arts))} artifact(s) ({parts}); results {_g(d, 'result_state')}"


def _s_artifact_read(d: Any) -> str:
    a = _g(d, "artifact") or {}
    name = _g(a, "name")
    if _g(d, "members") is not None:
        return f"{name}: directory with {_g(d, 'members_total', default=0)} member file(s)"
    size = _g(d, "size")
    size_s = f"{size} B" if size is not None else "size unknown"
    head = f"{name} [{_g(d, 'view')}]: {_g(d, 'bytes_read', default=0)} of {size_s} read"
    if _g(d, "records") is not None:
        head += f", {_n(d, 'records')} record(s)"
    if _g(d, "truncated"):
        head += ", truncated"
    if _g(d, "download") is not None:
        head += f"; binary — download {_g(d, 'download', 'state')}"
    return head


def _s_inspect_cluster(d: Any) -> str:
    checks = _g(d, "checks") or {}
    items = list(checks.values() if isinstance(checks, dict) else checks)
    bad = [c for c in items if _g(c, "status") not in ("ok", "skipped")]
    tail = ""
    if bad:
        first = bad[0]
        finding = next(
            (f for f in _g(first, "findings", default=[]) or [] if _g(f, "severity") in ("block", "warn")),
            None,
        )
        why = _g(finding, "message") if finding is not None else _g(first, "summary")
        names = ", ".join(f"{_g(c, 'name')}={_g(c, 'status')}" for c in bad)
        tail = f"; attention: {names}" + (f" — {why}" if why else "")
    counts: dict[str, int] = {}
    for c in items:
        counts[str(_g(c, "status"))] = counts.get(str(_g(c, "status")), 0) + 1
    parts = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    remediation = _g(d, "remediation", default=[]) or []
    nxt = f"; next: {remediation[0]}" if remediation else ""
    return f"cluster {_g(d, 'cluster_id')}: {_g(d, 'status')} ({parts}){tail}{nxt}"


def _s_usage(d: Any) -> str:
    totals = _g(d, "totals") or {}
    key = "account" if isinstance(totals, dict) and "account" in totals else "self"
    tot = totals.get(key) if isinstance(totals, dict) else None
    gh = _g(tot, "gpu_hours")
    gh_s = (
        f"{_num(gh)} allocated GPU-h ({key}, {_g(tot, 'jobs')} jobs)"
        if isinstance(gh, (int, float))
        else "GPU-h unknown"
    )
    vis = _g(d, "visibility")
    tail = (
        "; only your own jobs are visible — no lab total"
        if _g(d, "scope") == "account" and vis != "full"
        else ""
    )
    return f"{_g(d, 'scope')} usage on {_g(d, 'cluster_id')} {_g(d, 'since')} → {_g(d, 'until')}: {gh_s}; visibility {vis}{tail}"


def _s_workflow(d: Any) -> str:
    counts = _codes(_g(d, "codes") or _g(d, "counts"))
    blocked = _g(d, "dispatch_blocked_reason")
    gate = _g(d, "gate")
    gate_s = f"; smoke gate {_g(gate, 'state')}" if gate is not None else ""
    planned = _g(d, "planned_run_count")
    planned_s = f"; {planned} planned run(s)" if planned is not None else ""
    return (
        f"workflow {_g(d, 'workflow_id')} {_g(d, 'state')} (dispatch {_g(d, 'dispatch_state')}); version {_g(d, 'version')}"
        + gate_s
        + planned_s
        + (f"; cells {counts}" if counts else "")
        + (f"; blocked: {blocked}" if blocked else "")
    )


def _s_operation(d: Any) -> str:
    err = _g(d, "error", "message")
    return (
        f"operation {_g(d, 'operation_id')} ({_g(d, 'kind')}) {_g(d, 'state')}"
        + (f" at {_g(d, 'stage')}" if _g(d, "stage") else "")
        + (f": {err}" if err else "")
    )


def _s_cancel(d: Any) -> str:
    return f"attempt {_g(d, 'attempt_id')}: cancel {_g(d, 'cancel_state')} ({_g(d, 'phase')})" + (
        f" — {_g(d, 'note')}" if _g(d, "note") else ""
    )


def _s_experiment(d: Any) -> str:
    rc = _g(d, "run_counts") or {}
    parts = ", ".join(f"{k}={v}" for k, v in rc.items() if v) if isinstance(rc, dict) else ""
    return f"experiment {_g(d, 'name')} ({_g(d, 'experiment_id')}) {_g(d, 'status')}: {parts}"


def _s_cluster(d: Any) -> str:
    c = _g(d, "cluster") or d
    missing = _g(c, "missing", default=[]) or []
    verb = "registered" if _g(d, "created") else "updated"
    tail = f"; missing {', '.join(missing)}" if missing else ""
    return f"cluster {_g(c, 'cluster_id')} {verb}{tail}"


def _s_clusters(d: Any) -> str:
    return f"{_n(d, 'clusters')} cluster(s) registered"


def _s_profiles(d: Any) -> str:
    return f"{_n(d, 'profiles')} site profile(s)"


def _s_verify(d: Any) -> str:
    findings = _g(d, "findings", default=[]) or []
    sev = _g(d, "counts_by_severity") or {}
    errors = sev.get("error") if isinstance(sev, dict) and "error" in sev else None
    if errors is None:
        errors = sum(1 for f in findings if _g(f, "severity") == "error")
    warns = sev.get("warn", 0) if isinstance(sev, dict) else 0
    by_code = _g(d, "counts_by_code") or {}
    codes = ", ".join(f"{k}×{v}" for k, v in by_code.items()) if isinstance(by_code, dict) else ""
    counted = f"{errors} error(s), {warns} warning(s)" + (f" ({codes})" if codes else "")
    tail = "; record consistency only, not scientific correctness"
    if not _g(d, "ok", default=not errors):
        return f"bundle {_g(d, 'bundle_id')}: NOT OK — {counted}{tail}"
    # No errors is not "OK" when part of what was asked could not be checked: an unreadable or
    # skipped manuscript, or one without \hpcval tags (e.g. the table is pulled in by \input).
    sources = _g(d, "coverage", "sources", default=[]) or []
    unchecked = [
        str(_g(s, "name")).removeprefix("tex:")
        for s in sources
        if _g(s, "status") not in ("ok", "cached", "skipped")
        or (str(_g(s, "name")).startswith("tex:") and _g(s, "status") != "ok")
    ]
    if unchecked:
        return (
            f"bundle {_g(d, 'bundle_id')}: INCOMPLETE — not checked: {', '.join(unchecked[:3])}"
            f"{' …' if len(unchecked) > 3 else ''}; {counted} in what was checked{tail}"
        )
    reports = _g(d, "tex", default=[]) or []
    untagged = [str(_g(t, "source")) for t in reports if not _g(t, "tags")]
    if reports and len(untagged) == len(reports):
        return (
            f"bundle {_g(d, 'bundle_id')}: manuscript NOT checked — no \\hpcval-tagged numbers in "
            f"{', '.join(untagged[:3])} (files they \\input/\\include were scanned too); bundle itself: {counted}{tail}"
        )
    return f"bundle {_g(d, 'bundle_id')}: OK — {counted}{tail}"


# ---------------------------------------------------------------------------
# Method table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    name: str
    target: str  # "module:function"
    mutation: bool = False
    summarize: Callable[[Any], str] | None = None


S = "hpclab.services"

METHODS: list[Entry] = [
    # clusters / setup (orchestrator-owned)
    Entry("cluster_register", f"{S}.clusters:cluster_register", True, _s_cluster),
    Entry("clusters_list", f"{S}.clusters:clusters_list", False, _s_clusters),
    Entry("cluster_get", f"{S}.clusters:cluster_get"),
    Entry("profiles_list", f"{S}.clusters:profiles_list", False, _s_profiles),
    Entry("init_sim", f"{S}.clusters:init_sim", True, _s_cluster),
    Entry("auth_command", f"{S}.clusters:auth_command"),
    Entry("auth_resume", f"{S}.clusters:auth_resume", True),
    # 01 cluster_inspect + usage/notify/digest
    Entry("cluster_inspect", f"{S}.cluster:cluster_inspect", False, _s_inspect_cluster),
    Entry("usage_report", f"{S}.usage:usage_report", False, _s_usage),
    Entry("notify_test", f"{S}.notify:notify_test", True),
    Entry("lab_digest", f"{S}.digest:lab_digest"),
    # 02 prepare
    Entry("experiment_prepare", f"{S}.experiments:experiment_prepare", True, _s_prepare),
    Entry("experiments_list", f"{S}.experiments:experiments_list"),
    Entry("experiment_get", f"{S}.experiments:experiment_get", False, _s_experiment),
    Entry("plan_get", f"{S}.experiments:plan_get"),
    Entry("plan_script", f"{S}.experiments:plan_script"),
    # 03 / 08 submit, cancel, human resolution
    Entry("run_submit", f"{S}.submission:run_submit", True, _s_submit),
    Entry("run_cancel", f"{S}.submission:run_cancel", True, _s_cancel),
    Entry("attempt_resolve", f"{S}.submission:attempt_resolve", True, _s_generic_state("resolution")),
    # 04 / 15 workflows
    Entry("workflow_submit", f"{S}.workflows:workflow_submit", True, _s_workflow),
    Entry("workflow_cancel", f"{S}.workflows:workflow_cancel", True, _s_generic_state("workflow cancel")),
    Entry("workflow_get", f"{S}.workflows:workflow_get", False, _s_workflow),
    # 05 / 06 / 07 read side
    Entry("runs_list", f"{S}.runs:runs_list", False, _s_runs_list),
    Entry("run_inspect", f"{S}.runs:run_inspect", False, _s_inspect),
    Entry("operation_get", f"{S}.runs:operation_get", False, _s_operation),
    Entry("run_logs", f"{S}.logs:run_logs", False, _s_logs),
    # 09 recovery
    Entry("run_recover", f"{S}.recovery:run_recover", True, _s_recover),
    # 10 / 11 artifacts, 12 evaluations, vault
    Entry("artifacts_list", f"{S}.artifacts:artifacts_list", False, _s_artifacts),
    Entry("artifact_read", f"{S}.artifacts:artifact_read", False, _s_artifact_read),
    Entry("artifact_get", f"{S}.artifacts:artifact_get"),
    Entry("evaluation_collect", f"{S}.evaluations:evaluation_collect", True, _s_collect),
    Entry("evaluation_get", f"{S}.evaluations:evaluation_get"),
    Entry("vault_status", f"{S}.vault:vault_status"),
    Entry("backup", f"{S}.vault:backup", True),
    # 13 / 14 compare, evidence, verify, trace
    Entry("results_compare", f"{S}.compare:results_compare", True),
    Entry("comparison_get", f"{S}.compare:comparison_get"),
    Entry("evidence_export", f"{S}.evidence:evidence_export", True),
    Entry("evidence_get", f"{S}.evidence:evidence_get"),
    Entry("evidence_write", f"{S}.evidence:evidence_write", True),
    Entry("evidence_verify", f"{S}.verify:evidence_verify", False, _s_verify),
    Entry("trace", f"{S}.evidence:trace"),
    # audit trail (read-only; CLI `hpclab events`)
    Entry("events_list", f"{S}.audit:events_list"),
]

# Modules whose outbox handlers / periodic tasks the controller runs.
WORKER_MODULES = [
    f"{S}.experiments",
    f"{S}.submission",
    f"{S}.observe",
    f"{S}.artifacts",
    f"{S}.evaluations",
    f"{S}.vault",
    f"{S}.workflows",
    f"{S}.recovery",
    f"{S}.notify",
    f"{S}.cluster",
    f"{S}.usage",
]


def _resolve(target: str) -> Callable[..., Any]:
    mod_name, _, attr = target.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{target} is not callable")
    return fn


def build_registry(*, strict: bool = False) -> Registry:
    """Registry of every public method. ``strict`` raises on unresolvable entries (tests);
    otherwise they are logged and skipped so a partially built tree still serves the rest."""
    reg = Registry()
    missing: list[str] = []
    for e in METHODS:
        try:
            fn = _resolve(e.target)
        except (ImportError, AttributeError, TypeError) as exc:
            missing.append(f"{e.name} -> {e.target}: {exc}")
            continue
        reg.register(e.name, fn, mutation=e.mutation, summarize=e.summarize)
    if missing:
        if strict:
            raise RuntimeError("unresolved methods:\n  " + "\n  ".join(missing))
        log.warning("methods not available: %s", "; ".join(missing))
    return reg


def _collect(fn_name: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for mod_name in WORKER_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            log.warning("worker module %s unavailable: %s", mod_name, exc)
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None:
            out.append((mod_name, fn()))
    return out


class App:
    """What the controller serves: dispatch + background work of every service module."""

    def __init__(self, ctx: ServiceContext, registry: Registry | None = None) -> None:
        self.ctx = ctx
        self.registry = registry or build_registry()

    def dispatch(self, method: str, params: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        return self.registry.dispatch(self.ctx, method, params, request_id=request_id)

    def outbox_handlers(self) -> dict[str, Handler]:
        handlers: dict[str, Handler] = {}
        for mod_name, hs in _collect("outbox_handlers"):
            for kind, h in (hs or {}).items():
                if kind in handlers:
                    raise RuntimeError(f"outbox kind {kind!r} registered twice (again by {mod_name})")
                handlers[kind] = h
        return handlers

    def periodic_tasks(self) -> list[PeriodicTask]:
        tasks: list[PeriodicTask] = []
        seen: set[str] = set()
        for mod_name, ts in _collect("periodic_tasks"):
            for t in ts or []:
                name = t.name if hasattr(t, "name") else t[0]
                if name in seen:
                    raise RuntimeError(f"periodic task {name!r} registered twice (again by {mod_name})")
                seen.add(name)
                tasks.append(t)
        return tasks

    def close(self) -> None:
        # Refresh polls that outlived their caller's deadline still write the ledger: let them
        # finish before connections are torn down.
        try:
            from hpclab.services.observe import wait_for_background

            wait_for_background(5.0)
        except Exception:  # shutdown must proceed
            log.exception("waiting for background refreshes failed")
        self.ctx.clusters.close()


def create_app(ctx: ServiceContext) -> App:
    return App(ctx)
