"""HPC Lab MCP server (stdio).

A thin frontend: tools, resources and prompts are generated from ``catalog.TOOLS`` and the
service registry, and every call is forwarded to the local controller over its Unix socket.
No scheduler logic lives here. The server can be restarted or run several times (one per
agent session); all of them share one controller and one ledger.

Output of a tool call:

* ``structuredContent`` is the complete Envelope (``request_id, ok, data, error, coverage,
  observed_at, next_cursor, warnings, summary``) and validates against the tool's
  ``outputSchema``.
* The text block is what an agent reads first, so it is kept small (a few KB for common
  calls): the summary line, a compact tool-specific rendering that keeps every id needed to
  chain the next call, warnings / incomplete coverage, and ``Next:`` suggestions with exact
  arguments. Errors render their details and a remediation. Bulky fields stay in
  ``structuredContent`` and in the ``hpclab://`` resources.

Nothing here prints: stdout is the MCP protocol channel.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from hpclab import __version__
from hpclab.api.envelope import Envelope
from hpclab.bundled import skill_text
from hpclab.domain.errors import HpcLabError
from hpclab.mcp.catalog import DEF_DOCS, SERVER_INSTRUCTIONS, TOOLS, TOOLS_BY_NAME, ToolSpec
from hpclab.services.registry import Registry

log = logging.getLogger("hpclab.mcp")

TEXT_BUDGET = 8 * 1024  # target size of the text block of a common call
TEXT_HARD_LIMIT = 16 * 1024  # never more, except for content the agent asked to read
CONTENT_TOOLS = frozenset({"run_logs", "artifact_read"})  # the text block carries the content itself
CONTENT_HARD_LIMIT = 72 * 1024  # 64 KiB page + header
_LIFTED = ("coverage", "observed_at", "warnings", "next_cursor")
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class ClientLike(Protocol):
    def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _is_plain_type(s: Any) -> bool:
    return isinstance(s, dict) and set(s) == {"type"} and isinstance(s["type"], str)


def _compact_schema(schema: Any) -> None:
    """Remove pydantic's generated ``title`` annotations (noise for a model) and fold
    ``anyOf`` lists of bare types into ``type: [...]``, walking the schema structurally so a
    *property* named ``title`` is never touched."""
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("title"), str):
        del schema["title"]
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and len(any_of) > 1 and all(_is_plain_type(x) for x in any_of):
        del schema["anyOf"]
        schema["type"] = [x["type"] for x in any_of]
    for key in ("properties", "$defs", "patternProperties"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            for v in sub.values():
                _compact_schema(v)
    for key in ("items", "additionalProperties", "not", "contains"):
        _compact_schema(schema.get(key))
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for v in schema.get(key) or []:
            _compact_schema(v)


def _collapse_optional(prop: dict[str, Any]) -> dict[str, Any]:
    """``anyOf: [X, {type: null}]`` with ``default: null`` -> ``X``.

    Omitting an optional argument is how "not set" is expressed; the server also drops
    explicit nulls before forwarding (``normalize_arguments``), so the simpler schema is exact.
    """
    any_of = prop.get("anyOf")
    if not isinstance(any_of, list) or len(any_of) != 2 or prop.get("default", ...) is not None:
        return prop
    non_null = [s for s in any_of if s != {"type": "null"}]
    if len(non_null) != 1:
        return prop
    out = {k: v for k, v in prop.items() if k not in ("anyOf", "default")}
    merged = dict(non_null[0])
    merged.update(out)
    return merged


def _polish_properties(props: dict[str, Any], docs: dict[str, str]) -> None:
    for name in list(props):
        prop = _collapse_optional(props[name])
        if name in docs:
            prop["description"] = docs[name]
        props[name] = prop


def input_schema(spec: ToolSpec, registry: Registry) -> dict[str, Any]:
    method = registry.get(spec.method)
    schema = copy.deepcopy(method.params_model.model_json_schema())
    _polish_properties(schema.get("properties", {}), spec.params)
    for def_name, def_schema in (schema.get("$defs") or {}).items():
        if isinstance(def_schema, dict) and isinstance(def_schema.get("properties"), dict):
            _polish_properties(def_schema["properties"], DEF_DOCS.get(def_name, {}))
    _compact_schema(schema)
    schema["additionalProperties"] = False
    return schema


def output_schema(spec: ToolSpec, registry: Registry) -> dict[str, Any]:
    """``Envelope`` schema whose ``data`` is the method's Data model minus lifted fields."""
    env = copy.deepcopy(Envelope[Any].model_json_schema())
    data_model = registry.get(spec.method).data_model
    defs: dict[str, Any] = dict(env.pop("$defs", {}))
    if data_model is not None:
        ds = copy.deepcopy(data_model.model_json_schema())
        defs.update(ds.pop("$defs", {}))
        for f in _LIFTED:
            ds.get("properties", {}).pop(f, None)
        if "required" in ds:
            ds["required"] = [r for r in ds["required"] if r not in _LIFTED]
        env["properties"]["data"] = {"anyOf": [ds, {"type": "null"}]}
    if defs:
        env["$defs"] = defs
    env.pop("title", None)
    return env


def build_tools(registry: Registry) -> list[types.Tool]:
    tools = []
    for spec in TOOLS:
        if spec.method not in registry.methods:
            log.warning("tool %s: method %s not registered; skipping", spec.name, spec.method)
            continue
        tools.append(
            types.Tool(
                name=spec.name,
                title=spec.title,
                description=spec.description,
                input_schema=input_schema(spec, registry),
                output_schema=output_schema(spec, registry),
                annotations=types.ToolAnnotations(
                    title=spec.title,
                    read_only_hint=spec.read_only,
                    destructive_hint=spec.destructive,
                    idempotent_hint=spec.idempotent,
                    open_world_hint=spec.open_world,
                ),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Argument normalization (paths are relative to the agent's working directory)
# ---------------------------------------------------------------------------


def _abs(path: str, cwd: Path) -> str:
    p = Path(os.path.expanduser(path))
    return str(p if p.is_absolute() else (cwd / p).resolve())


def normalize_arguments(tool: str, args: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Drop explicit nulls (= "not set"; every parameter's null means its default) and make
    paths absolute: the controller's working directory is not the agent's."""
    out = {k: v for k, v in args.items() if v is not None}
    if tool == "experiment_prepare":
        if isinstance(out.get("spec_path"), str):
            out["spec_path"] = _abs(out["spec_path"], cwd)
        spec = out.get("spec")
        if isinstance(spec, dict):
            spec = dict(spec)
            code = dict(spec.get("code") or {})
            code["project_root"] = _abs(str(code.get("project_root") or "."), cwd)
            spec["code"] = code
            out["spec"] = spec
    elif tool == "evidence_verify" and isinstance(out.get("tex_paths"), list):
        out["tex_paths"] = [_abs(str(p), cwd) for p in out["tex_paths"]]
    return out


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------

Lines = list[str]


def _num(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        return f"{v:.0f}" if abs(v) >= 1e4 else f"{v:.4g}"
    return str(v)


def _gh(v: Any) -> str:
    """GPU-hours: unknown stays unknown."""
    return f"{v:.3g} GPU-h" if isinstance(v, (int, float)) and not isinstance(v, bool) else "? GPU-h"


def _short_digest(d: Any) -> str:
    s = str(d or "")
    return s[:19] + "…" if s.startswith("sha256:") and len(s) > 20 else s


def _val(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)


def call_hint(tool: str, **args: Any) -> str:
    """``tool(a="x", b=[1, 2])`` with JSON literals; None arguments are left out."""
    parts = [f"{k}={_val(v)}" for k, v in args.items() if v is not None]
    return f"{tool}({', '.join(parts)})"


def _kv_line(pairs: Iterable[tuple[str, Any]], sep: str = ", ") -> str:
    return sep.join(f"{k} {_num(v)}" for k, v in pairs if v not in (None, "", [], {}))


def _cell_str(cell: Any) -> str:
    if isinstance(cell, dict):
        return " ".join(f"{k}={v}" for k, v in cell.items())
    return str(cell or "")


def _counts_str(counts: Any, skip: Iterable[str] = ()) -> str:
    if not isinstance(counts, dict):
        return ""
    skipped = set(skip)
    return ", ".join(f"{k}={v}" for k, v in counts.items() if v and k not in skipped)


def _table(rows: list[list[str]], header: list[str] | None = None, max_rows: int = 40) -> Lines:
    """Left-aligned plain-text table (monospace); rows beyond ``max_rows`` are elided."""
    body = rows[:max_rows]
    allrows = ([header] if header else []) + body
    if not allrows:
        return []
    widths = [max(len(r[i]) if i < len(r) else 0 for r in allrows) for i in range(len(allrows[0]))]
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in allrows]
    if len(rows) > max_rows:
        out.append(f"… {len(rows) - max_rows} more (complete list in structuredContent)")
    return out


_DROP_KEYS = frozenset({"schema_version", "request_id", "legend", "dispatch_key", "content_hash"})


def _prune(value: Any, *, max_list: int, max_str: int) -> Any:
    """Compact copy for text: no nulls/empties, no refs duplicating ids, bounded lists/strings,
    floats to 6 significant digits. Hashes (<= 71 chars) are never cut."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        ids = {str(v) for k, v in value.items() if k.endswith("_id") and isinstance(v, str) and v}
        for k, v in value.items():
            if k in _DROP_KEYS or v is None or v == [] or v == {} or v == "":
                continue
            if (k.endswith("_ref") or k == "ref") and isinstance(v, str) and any(v.endswith(i) for i in ids):
                continue
            out[k] = _prune(v, max_list=max_list, max_str=max_str)
        return out
    if isinstance(value, list):
        items = [_prune(v, max_list=max_list, max_str=max_str) for v in value[:max_list]]
        if len(value) > max_list:
            items.append(f"… {len(value) - max_list} more")
        return items
    if isinstance(value, str) and len(value) > max(max_str, 80):  # ids and sha256 digests stay whole
        return value[:max_str] + f"… [{len(value)} chars]"
    if isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")):
        return float(f"{value:.6g}")
    return value


def compact_json(value: Any, budget: int = TEXT_BUDGET) -> str:
    """The smallest-loss compact JSON of ``value`` that fits ``budget`` bytes (best effort)."""
    text = ""
    for max_list, max_str in ((50, 600), (20, 300), (10, 160), (5, 100), (3, 60)):
        text = json.dumps(
            _prune(value, max_list=max_list, max_str=max_str),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if len(text.encode()) <= budget:
            return text
    return text


def _fit(text: str, limit: int) -> str:
    raw = text.encode()
    if len(raw) <= limit:
        return text
    cut = raw[: limit - 120].decode("utf-8", errors="ignore")
    return cut + f"\n… [text truncated at {limit} bytes; the complete result is in structuredContent]"


# ---------------------------------------------------------------------------
# Per-tool renderers: (data, envelope, args) -> body lines
# ---------------------------------------------------------------------------

Renderer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Lines]


def _r_cluster_inspect(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = []
    fixes: set[str] = set()
    for c in d.get("checks") or []:
        out.append(f"- {c.get('name')}: {c.get('status')} — {c.get('summary') or ''}".rstrip(" —"))
        for f in c.get("findings") or []:
            if f.get("severity") == "info":
                out.append(f"    info {f.get('code')}: {f.get('message')}")
                continue
            if f.get("code") == "blocked_by_auth":  # one per skipped check; the auth line says it
                continue
            out.append(f"    {f.get('severity')} {f.get('code')}: {f.get('message')}")
            if f.get("remediation"):
                out.append(
                    f"      fix: {f['remediation']}"
                    if f["remediation"] not in fixes
                    else "      fix: as above"
                )
                fixes.add(f["remediation"])
    for cand in d.get("candidates") or []:
        shape = cand.get("shape") or {}
        where = f"{shape.get('partition')} {shape.get('gpus')}×{shape.get('gpu_type') or 'gpu'} {shape.get('time')}"
        est = cand.get("est_start") or "no estimate"
        extra = _kv_line([("free now", cand.get("free_now"))])
        blockers = "; blocked: " + ", ".join(cand["blockers"]) if cand.get("blockers") else ""
        out.append(
            f"- candidate {where}: {cand.get('status')}, start ~{est} (low confidence){', ' + extra if extra else ''}{blockers}"
        )
    for r in d.get("remediation") or []:
        if r not in fixes:
            out.append(f"remediation: {r}")
    return out


def _r_prepare(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = [
        f"experiment_id {d.get('experiment_id')}  plan_id {d.get('plan_id')}  status {d.get('status')} ({d.get('kind')})"
    ]
    if d.get("plan_hash"):
        out.append(f"plan_hash {d['plan_hash']}")
    b = d.get("budget") or {}
    budget = (
        f" of {_gh(b.get('budget_gpu_hours'))} budget"
        if b.get("budget_gpu_hours") is not None
        else " (no budget set)"
    )
    out.append(
        f"GPU-hours requested (gpus × time limit): {_gh(b.get('requested_gpu_hours_total'))}{budget}; "
        f"smoke {_gh(b.get('smoke_gpu_hours'))}"
    )
    runs = d.get("runs") or []
    if runs:
        out.append(f"runs ({d.get('runs_total', len(runs))}; smoke run {d.get('smoke_run_id') or 'none'}):")
        rows = [
            [
                str(r.get("kind")),
                str(r.get("label")),
                str(r.get("run_id") or "-"),
                str(r.get("exec_plan_hash") or "-"),
            ]
            for r in runs
        ]
        out += [
            "  " + line for line in _table(rows, ["kind", "label", "run_id", "exec_plan_hash"], max_rows=24)
        ]
    val = d.get("validation") or {}
    counts = _counts_str(val.get("counts"))
    out.append(f"validation {val.get('status')}" + (f" ({counts})" if counts else ""))
    for f in val.get("findings") or []:
        if f.get("status") == "pass":
            continue
        line = f"- [{f.get('status')}/{f.get('severity')}] {f.get('check')}: {f.get('message')}"
        out.append(line)
        if f.get("remediation"):
            out.append(f"    fix: {f['remediation']}")
        if f.get("override_reason"):
            out.append(f"    overridden: {f['override_reason']}")
    if val.get("blocking"):
        out.append("blocking: " + ", ".join(val["blocking"]))
    snap = d.get("snapshot") or {}
    ref = snap.get("ref") or {}
    if snap:
        git = ref.get("git_commit")
        git_s = (
            f", git {str(git)[:12]}{' (dirty)' if ref.get('git_dirty') else ''}"
            if git
            else ", not a git checkout"
        )
        out.append(
            f"code snapshot {snap.get('mode')}: {_short_digest(ref.get('digest'))} "
            f"({snap.get('file_count') or ref.get('file_count')} files{git_s}; {snap.get('excluded_count', 0)} excluded)"
        )
        shown = {str(f.get("message")) for f in val.get("findings") or []}
        out += [f"  snapshot warning: {w}" for w in snap.get("warnings") or [] if w not in shown]
    adv = d.get("resource_advice") or {}
    for item in (adv.get("items") or [])[:6]:
        if item.get("verdict") in ("under_request", "over_request"):
            out.append(
                f"resource advice ({item.get('kind')}): {item.get('verdict')} — requested {item.get('requested_time')}, "
                f"suggested {item.get('suggested_time')} (p90 of {item.get('n')} past runs)"
            )
    for o in d.get("overrides") or []:
        out.append(
            f"override {o.get('check')}: {o.get('reason')} ({'applied' if o.get('applied') else 'not applied'})"
        )
    if d.get("operation_ref"):
        out.append(f"operation {d['operation_ref']} (background snapshot)")
    return out


def _r_submit(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    return [
        f"run_id {d.get('run_id')}  attempt_id {d.get('attempt_id')}  phase {d.get('phase')}",
        f"operation {d.get('operation_ref')}  reserved {_gh(d.get('reserved_gpu_hours'))}",
    ]


def _workflow_cells(cells: list[dict[str, Any]]) -> Lines:
    rows = []
    for c in cells:
        detail = c.get("scheduler_state") or c.get("phase") or ""
        if c.get("reason") and c.get("code") != "P":
            detail = f"{detail} {c['reason']}".strip()
        rows.append(
            [
                str(c.get("code")),
                str(c.get("label")),
                str(c.get("run_id")),
                str(c.get("attempt_id") or "-"),
                str(c.get("job_id") or "-"),
                detail or str(c.get("reason") or ""),
            ]
        )
    return [
        "  " + x
        for x in _table(rows, ["code", "label", "run_id", "attempt_id", "job", "detail"], max_rows=30)
    ]


def _r_workflow(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = [
        f"workflow_id {d.get('workflow_id')}  version {d.get('version')}  state {d.get('state')}  "
        f"dispatch {d.get('dispatch_state')}  max_parallel {d.get('max_parallel')}",
        f"experiment_id {d.get('experiment_id')}  plan_id {d.get('plan_id')} ({d.get('plan_kind')}); "
        f"{d.get('planned_run_count')} planned run(s)",
    ]
    gate = d.get("gate") or {}
    if gate:
        g = f"smoke gate: {gate.get('state')} (run {gate.get('run_id')}"
        g += f", attempt {gate['attempt_id']})" if gate.get("attempt_id") else ")"
        if gate.get("reason"):
            g += f" — {gate['reason']}"
        out.append(g)
    if d.get("skipped_count"):
        out.append(f"skipped (already done): {d['skipped_count']}")
    cells = d.get("cells") or []
    if cells:
        out.append("cells:")
        out += _workflow_cells(cells)
    return out


_PENDING_EVAL = "completed, evaluation pending (not a failure)"
_CODES_LINE = (
    "codes: V valid, R running, Q queued, S submitting, P not dispatched, "
    f"E {_PENDING_EVAL}, F failed, C! completed but outputs or evaluation invalid, "
    "U submission uncertain, X cancelled/rejected, ? stale"
)


def _grid_lines(d: dict[str, Any]) -> Lines:
    g = d.get("grid") or {}
    row_axes = g.get("row_axes") or []
    col_axes = g.get("col_axes") or []
    one_col = len(col_axes) == 1
    col_labels = [
        (c.split("=", 1)[1] if one_col and c.startswith(f"{col_axes[0]}=") else c)
        for c in g.get("cols") or []
    ]
    header = [f"{'/'.join(row_axes) or 'row'} \\ {'/'.join(col_axes) or 'col'}", *col_labels]
    rows: list[list[str]] = []
    attention: Lines = []
    pending: Lines = []
    for r in g.get("rows") or []:
        vals = r.get("values") or {}
        label = (
            str(vals.get(row_axes[0])) if len(row_axes) == 1 and row_axes[0] in vals else str(r.get("key"))
        )
        cells = []
        for i, cell in enumerate(r.get("cells") or []):
            code = str(cell.get("code"))
            text = code + (f" {_num(cell['value'])}" if cell.get("value") is not None else "")
            cells.append(text)
            if code in ("F", "C!", "U", "X", "?", "E"):
                where = (
                    f"{r.get('key')} {g.get('cols', [])[i] if i < len(g.get('cols') or []) else ''}".strip()
                )
                line = f"- {code} {where}: {cell.get('reason') or '-'} (run_id {cell.get('run_id')})"
                (pending if code == "E" else attention).append(line)
        rows.append([label, *cells])
    out: Lines = [f"cell values: {g['metric']} (validated evaluations only)"] if g.get("metric") else []
    out += _table(rows, header, max_rows=40)
    if g.get("total_rows") and g["total_rows"] > len(g.get("rows") or []):
        out.append(f"(rows {g.get('offset', 0) + 1}–{g.get('offset', 0) + len(rows)} of {g['total_rows']})")
    if attention:
        out.append("needs attention:")
        out += attention[:20]
        if len(attention) > 20:
            out.append(f"… {len(attention) - 20} more")
    if pending:
        out.append(f"{_PENDING_EVAL}:")
        out += pending[:10]
        if len(pending) > 10:
            out.append(f"… {len(pending) - 10} more")
    out += [f"note: {n}" for n in g.get("notes") or []]
    out.append(_CODES_LINE)
    return out


def _foreign(d: dict[str, Any]) -> Lines:
    out: Lines = []
    for f in d.get("foreign_jobs") or []:
        if f.get("count"):
            out.append(
                f"other jobs of yours on {f.get('cluster_id')} (not managed by hpclab): {f.get('count')} "
                f"({f.get('running')} running, {f.get('pending')} pending, {f.get('gpus')} GPUs)"
            )
    return out


def _r_runs_list(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    if d.get("view") == "grid" and d.get("grid"):
        return _grid_lines(d) + _foreign(d)
    rows = []
    for r in d.get("runs") or []:
        la = r.get("latest_attempt") or {}
        state = la.get("scheduler_state") or la.get("phase") or "-"
        extra = [
            x for x in (r.get("reason"), la.get("pending_reason"), la.get("progress")) if x and x != state
        ]
        if la.get("time_left_s") is not None:
            extra.append(f"{la['time_left_s']}s left")
        if la.get("observation_gap"):
            extra.append("observation gap")
        if r.get("stale"):
            extra.append("stale")
        rows.append(
            [
                str(r.get("code")),
                str(r.get("label")),
                str(r.get("run_id")),
                f"{la.get('attempt_id')} #{la.get('seq')}" if la else "-",
                str(la.get("job_id") or "-"),
                state,
                "; ".join(str(x) for x in extra),
            ]
        )
    rows.sort(key=lambda r: (not r[1].startswith("smoke"), r[1]))
    out = _table(rows, ["code", "label", "run_id", "latest attempt", "job", "state", "detail"], max_rows=40)
    out += _foreign(d)
    if d.get("changes_cursor"):
        out.append(
            f"changes_cursor {d['changes_cursor']} (pass as changed_since to get only runs that changed)"
        )
    return out


def _r_run_inspect(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    run = d.get("run") or {}
    spec = d.get("spec") or {}
    refs = d.get("refs") or {}
    out: Lines = [
        f"run_id {run.get('run_id')} ({run.get('kind')})"
        + (
            f"; E ({_PENDING_EVAL}): {run.get('code_reason') or '-'}"
            if run.get("code") == "E"
            else f"; why {run.get('code')}: {run['code_reason']}"
            if run.get("code_reason") and run.get("code") not in ("V", "R", "Q", "S", "P")
            else ""
        ),
        f"experiment_id {run.get('experiment_id')}  plan_id {run.get('plan_id')}  cell {_cell_str(run.get('cell'))}",
    ]
    if spec.get("argv"):
        out.append("argv: " + " ".join(str(a) for a in spec["argv"]))
    res = refs.get("resources") or {}
    if res:
        out.append("resources: " + ", ".join(f"{k}={v}" for k, v in res.items() if v not in (None, "", [])))
    git = refs.get("git_commit")
    out.append(
        f"code {_short_digest(refs.get('snapshot_digest'))}"
        + (f", git {str(git)[:12]}{' (dirty)' if refs.get('git_dirty') else ''}" if git else "")
        + (
            f"; current exec_plan_hash {run.get('current_exec_plan_hash')}"
            if run.get("current_exec_plan_hash")
            else ""
        )
    )
    attempts = d.get("attempts") or []
    focus = d.get("focus_attempt_id")
    for a in attempts:
        mark = "*" if a.get("attempt_id") == focus else " "
        job = a.get("job") or {}
        out.append(
            f"{mark}attempt #{a.get('seq')} {a.get('attempt_id')} (version {a.get('version')}): phase {a.get('phase')}; "
            f"scheduler {a.get('scheduler_state')}; app {a.get('app_state')}; results {a.get('result_state')}"
            + (f"; cancel {a.get('cancel_state')}" if a.get("cancel_state") not in (None, "none") else "")
            + (f"; diagnosis {a['diagnosis_code']}" if a.get("diagnosis_code") else "")
        )
        if a.get("attempt_id") != focus:
            continue
        if job.get("job_id"):
            out.append(
                f"   job {job.get('job_id')} on {job.get('cluster_id')} nodes {job.get('node_list') or '?'} "
                f"(owner {job.get('owner')}, submitted {job.get('submit_time')}, restarts {job.get('restart_count', 0)})"
            )
        timing = _kv_line(
            [
                ("elapsed_s", a.get("elapsed_s")),
                ("time_limit_s", a.get("time_limit_s")),
                ("time_left_s", a.get("time_left_s")),
            ]
        )
        exit_s = f"exit {a.get('exit_code')}:{a.get('exit_signal')}" if a.get("exit_code") is not None else ""
        if timing or exit_s:
            out.append("   " + ", ".join(x for x in (exit_s, timing) if x))
        if a.get("observation_gap"):
            out.append(
                "   observation gap: the job left the queue and accounting has no final record yet (not finished, not failed)"
            )
        if a.get("uncertain_since") and a.get("phase") in ("submission_uncertain", "duplicate_detected"):
            out.append(
                f"   submission uncertain since {a['uncertain_since']}: do NOT resubmit; the controller reconciles"
            )
        elif a.get("uncertain_since"):
            out.append(
                f"   submission was uncertain from {a['uncertain_since']} until it was reconciled (phase {a.get('phase')})"
            )
        bud = a.get("budget") or {}
        if bud:
            out.append(
                f"   budget {bud.get('state')}: reserved {_gh(bud.get('reserved'))}, settled {_gh(bud.get('settled_consumed'))}"
            )
    if d.get("attempts_total", len(attempts)) > len(attempts):
        out.append(f"({d['attempts_total'] - len(attempts)} older attempt(s) not shown)")
    for op in _attention_ops(d)[:3]:
        err = op.get("error") or {}
        fix = (err.get("details") or {}).get("remediation")
        out.append(
            f"operation {op.get('operation_id')} ({op.get('kind')}) {op.get('state')}"
            + (f" at {op['stage']}" if op.get("stage") else "")
            + (f": {err.get('message')}" if err.get("message") else "")
            + (f" — fix: {fix}" if fix else "")
        )
    p = d.get("pending") or {}
    if p:
        out.append(
            f"pending: {p.get('reason_code')} ({p.get('category')}) — {p.get('explanation')}"
            + (f" Suggestion: {p['suggestion']}" if p.get("suggestion") else "")
        )
    pr = d.get("progress") or {}
    if pr and (pr.get("step") is not None or pr.get("latest_metrics")):
        frac = f" ({pr['fraction'] * 100:.0f}%)" if isinstance(pr.get("fraction"), (int, float)) else ""
        metrics = ", ".join(f"{k}={_num(v)}" for k, v in list((pr.get("latest_metrics") or {}).items())[:8])
        out.append(
            f"progress: step {pr.get('step')}/{pr.get('total_steps') or '?'}{frac}"
            + (f"; eta {pr['eta_s']}s" if pr.get("eta_s") is not None else "")
            + "; "
            f"time-limit risk {pr.get('timelimit_risk')}"
            + ("; STALLED?" if pr.get("stalled_suspect") else "")
        )
        if metrics:
            out.append(f"   latest {metrics}")
    h = d.get("health") or {}
    if h and (h.get("flags") or h.get("source") == "telemetry"):
        out.append(
            f"health (advisory): flags {', '.join(h.get('flags') or []) or 'none'}; "
            + _kv_line([("idle_gpus", h.get("idle_gpus")), ("heartbeat_age_s", h.get("heartbeat_age_s"))])
        )
    dg = d.get("diagnosis") or {}
    if dg:
        out.append(f"diagnosis: {dg.get('summary')}")
        for item in (dg.get("items") or [])[:2]:
            out.append(f"   {item.get('code')} ({item.get('confidence')}): {item.get('explanation')}")
            for ev in (item.get("evidence") or [])[:3]:
                out.append(f"     evidence: {ev}")
            if item.get("suggested_strategies"):
                hint = item.get("resource_hint") or {}
                hint_s = f"; resource hint {', '.join(f'{k}={v}' for k, v in hint.items())}" if hint else ""
                out.append(f"     suggested: {', '.join(item['suggested_strategies'])}{hint_s}")
    env_v = d.get("env") or {}
    if env_v.get("drift"):
        out.append("environment drift: " + "; ".join(env_v["drift"]))
    arts = d.get("artifacts") or {}
    if arts:
        out.append(
            f"artifacts: {arts.get('total')}"
            + (f" ({_counts_str(arts.get('by_integrity'))})" if arts.get("by_integrity") else "")
            + f"; results {arts.get('result_state')}"
            + (
                f"; required missing: {', '.join(arts['required_missing'])}"
                if arts.get("required_missing")
                else ""
            )
        )
    for ev in d.get("evaluations") or []:
        out.append(
            f"evaluation {ev.get('evaluation_id')} {ev.get('evaluator')} {ev.get('status')}"
            + (" (selected)" if ev.get("selected") else "")
        )
    gp = d.get("gpu_probe") or {}
    if gp:
        samples = "; ".join(
            f"gpu{s.get('index')} {_num(s.get('util_percent'))}% {_num(s.get('mem_used_mib'))}/{_num(s.get('mem_total_mib'))} MiB"
            for s in gp.get("samples") or []
        )
        out.append(
            f"gpu probe: {gp.get('status')}"
            + (f" — {samples}" if samples else "")
            + (f" ({gp['detail']})" if gp.get("detail") else "")
        )
    if d.get("stale"):
        out.append("stale: the refresh did not finish in time; this is the last known state")
    return out


def _attention_ops(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Operations of the inspected run that need someone: blocked ones (SSH auth, cluster access,
    a human decision) and failed ones on the focus attempt. Finished work is not repeated."""
    focus = d.get("focus_attempt_id")
    return [
        op
        for op in d.get("operations") or []
        if op.get("state") == "blocked"
        or (op.get("state") == "failed" and op.get("target_id") == focus and op.get("error"))
    ]


def _r_logs(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    text = d.get("text") or ""
    flags = [f for f in ("truncated", "rotated", "eof") if d.get(f)]
    head = (
        f"----- {d.get('stream')} of {d.get('attempt_id')}: bytes {d.get('offset')}–{d.get('next_offset')} "
        f"of {d.get('size') if d.get('size') is not None else '?'}{' (' + ', '.join(flags) + ')' if flags else ''} -----"
    )
    out = [head]
    if d.get("note"):
        out.insert(0, f"note: {d['note']}")
    out.append(text.rstrip("\n") if text else "(no output)")
    out.append("----- end -----")
    return out


def _r_cancel(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    return [
        f"attempt_id {d.get('attempt_id')} (run {d.get('run_id')}): cancel_state {d.get('cancel_state')}, "
        f"mode {d.get('cancel_mode') or 'cancel'}, phase {d.get('phase')}, attempt version {d.get('attempt_version')}",
        *([f"operation {d['operation_ref']}"] if d.get("operation_ref") else []),
        *([f"note: {d['note']}"] if d.get("note") and d["note"] not in str(env.get("summary")) else []),
    ]


def _r_recover(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = []
    dg = d.get("diagnosis") or {}
    if dg.get("primary_code"):
        out.append(
            f"diagnosis {dg.get('primary_code')} ({dg.get('category')}, {dg.get('confidence')} confidence, {dg.get('source')}): "
            f"{dg.get('summary')}"
        )
    elif dg.get("summary"):
        out.append(f"diagnosis: {dg['summary']}")
    if dg.get("last_error_line"):
        out.append(f"   last error line: {dg['last_error_line']}")
    if d.get("mode") == "batch":
        out.append(
            f"batch recovery plan {d.get('plan_id')} from workflow {d.get('source_workflow_id')}: "
            f"{d.get('eligible_count')}/{d.get('total')} cell(s) eligible, {_gh(d.get('requested_gpu_hours'))}"
        )
        rows = [
            [
                str(c.get("label")),
                str(c.get("run_id")),
                "yes" if c.get("eligible") else "no",
                str(c.get("strategy") or "-"),
                str(c.get("diagnosis_code") or c.get("code") or "-"),
                str(c.get("blocking_reason") or ""),
            ]
            for c in d.get("cells") or []
        ]
        out += [
            "  " + x
            for x in _table(
                rows, ["label", "run_id", "eligible", "strategy", "diagnosis", "blocking"], max_rows=30
            )
        ]
    else:
        out.append(
            f"eligible {'yes' if d.get('eligible') else 'NO'}; strategy {d.get('strategy') or '-'}; "
            f"run_id {d.get('run_id')}; from attempt {d.get('attempt_id')}"
        )
        if d.get("resource_patch"):
            out.append("resource change: " + ", ".join(f"{k}={v}" for k, v in d["resource_patch"].items()))
        if d.get("exclude_nodes"):
            out.append("excluded nodes: " + ", ".join(d["exclude_nodes"]))
        if d.get("suspect_nodes"):
            out.append("suspect nodes (not excluded): " + ", ".join(d["suspect_nodes"]))
        if d.get("exec_plan_hash"):
            out.append(
                f"new execution plan: plan_id {d.get('plan_id')}  exec_plan_hash {d['exec_plan_hash']}  ({_gh(d.get('requested_gpu_hours'))})"
            )
        ck = d.get("checkpoint") or {}
        if ck:
            out.append(
                f"checkpoint {ck.get('artifact_id')} step {ck.get('step')}: {ck.get('status')}, {ck.get('continuation')}"
                + ("; needs a load probe" if ck.get("needs_load_probe") else "")
                + (f"; issues: {'; '.join(ck['issues'])}" if ck.get("issues") else "")
            )
        pb = d.get("probe") or {}
        if pb:
            out.append(
                f"checkpoint probe run {pb.get('run_id')}: {pb.get('state')}"
                + (f", exec_plan_hash {pb['exec_plan_hash']}" if pb.get("exec_plan_hash") else "")
                + (f" — {pb['detail']}" if pb.get("detail") else "")
            )
    for c in d.get("checks") or []:
        out.append(f"check {c.get('check')}: {c.get('status')} — {c.get('message')}")
    for b in d.get("blocking") or []:
        out.append(f"blocking: {b}")
    for h in d.get("hints") or []:  # the exact calls are under Next:
        out.append(f"hint ({h.get('strategy')}): {h.get('message')}")
    return out


def _r_artifacts(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out = [
        f"attempt {d.get('attempt_id')} (run {d.get('run_id')}): phase {d.get('phase')}, scheduler {d.get('scheduler_state')}, "
        f"results {d.get('result_state')}"
    ]
    rows = [
        [
            str(a.get("name")),
            f"{a.get('role')}/{a.get('kind')}{'' if a.get('required') else ' (optional)'}",
            str(a.get("integrity")),
            f"{a.get('size_bytes')} B" if a.get("size_bytes") is not None else "? B",
            _short_digest(a.get("digest")) or str(a.get("digest_state")),
            str(a.get("artifact_id")),
        ]
        for a in d.get("artifacts") or []
    ]
    out += _table(rows, ["name", "role/kind", "integrity", "size", "digest", "artifact_id"], max_rows=40)
    return out


def _r_artifact_read(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    a = d.get("artifact") or {}
    out = [
        f"artifact {a.get('name')} ({a.get('artifact_id')}) {a.get('role')}/{a.get('kind')} integrity {a.get('integrity')}"
    ]
    if d.get("members") is not None:
        out.append(
            f"directory with {d.get('members_total')} member file(s); read one with artifact_read(artifact_ref=<artifact_id>):"
        )
        rows = [
            [
                str(m.get("member_path") or m.get("name")),
                f"{m.get('size_bytes')} B",
                str(m.get("integrity")),
                str(m.get("artifact_id")),
            ]
            for m in d.get("members") or []
        ]
        out += _table(rows, ["member", "size", "integrity", "artifact_id"], max_rows=30)
        return out
    size = d.get("size") if d.get("size") is not None else "?"
    if d.get("offset") is not None:
        end = d.get("next_offset")
        if end is None:
            end = int(d["offset"]) + int(d.get("bytes_read") or 0)
        rng = f"bytes {d['offset']}–{end} of {size}"
    else:
        rng = f"{d.get('bytes_read')} of {size} bytes read"
    flags = [f for f in ("truncated", "eof", "binary", "digest_verified") if d.get(f)]
    out.append(
        f"----- view {d.get('view')} ({d.get('media_type')}), {rng}{', ' + ', '.join(flags) if flags else ''} -----"
    )
    if d.get("records") is not None:
        out += [json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str) for r in d["records"]]
        if d.get("parse_errors"):
            out.append(f"({d['parse_errors']} line(s) could not be parsed)")
    elif d.get("value") is not None:
        out.append(json.dumps(d["value"], ensure_ascii=False, separators=(",", ":"), default=str))
    elif d.get("text") is not None:
        out.append(str(d["text"]).rstrip("\n"))
    elif d.get("value_summary") is not None:
        out.append("summary: " + json.dumps(d["value_summary"], ensure_ascii=False, default=str))
    else:
        out.append("(no content returned)")
    out.append("----- end -----")
    dl = d.get("download") or {}
    if dl:
        out.append(
            f"download: {dl.get('state')}"
            + (f" (operation {dl['operation_ref']})" if dl.get("operation_ref") else "")
        )
    if d.get("note"):
        out.append(f"note: {d['note']}")
    return out


def _r_collect(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    if d.get("mode") == "single":
        ev = d.get("evaluation") or {}
        out = [
            f"evaluation {ev.get('evaluation_id')} ({'created' if d.get('created') else 'existing'}) of attempt {ev.get('attempt_id')}: "
            f"{ev.get('status')} via {ev.get('evaluator_id')}@{ev.get('evaluator_version')}"
        ]
        for m in (ev.get("metrics") or [])[:20]:
            where = ", ".join(f"{k} {m[k]}" for k in ("split", "step", "dataset") if m.get(k) is not None)
            out.append(f"- {m.get('name')} = {_num(m.get('value'))}" + (f" ({where})" if where else ""))
        if ev.get("metrics_truncated"):
            out.append(f"… {ev.get('metrics_total')} metrics in total")
        problems = (ev.get("validation") or {}).get("problems") or (ev.get("validation") or {}).get("issues")
        if problems:
            out.append("validation problems: " + json.dumps(problems, default=str)[:600])
        return out
    rs = d.get("run_set") or {}
    out = [f"run set {rs.get('kind')} {rs.get('id')} ({rs.get('runs')} runs)"]
    for c in (d.get("collected") or [])[:20]:
        out.append(f"- collected {c.get('evaluation_id')} for run {c.get('run_id')} ({c.get('status')})")
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for s in d.get("skipped") or []:
        by_reason.setdefault(str(s.get("reason")), []).append(s)
    for reason, items in sorted(by_reason.items()):
        if reason == "already_evaluated":
            out.append(f"- already evaluated (fine): {len(items)} run(s)")
            continue
        out.append(f"- skipped {reason}: {len(items)}")
        for s in items[:8]:
            out.append(
                f"    run {s.get('run_id')}"
                + (f" attempt {s['attempt_id']}" if s.get("attempt_id") else "")
                + (f": {s['detail']}" if s.get("detail") else "")
            )
    if d.get("pending"):
        out.append(f"{d['pending']} still evaluating in the background: operation {d.get('operation_ref')}")
    return out


def _stat(m: dict[str, Any]) -> str:
    if m.get("mean") is None:
        return f"— (n {m.get('n')}/{m.get('expected_n')})"
    std = f" ± {_num(m['std'])}" if m.get("std") is not None else " (std undefined, n<2)"
    ci = (
        f" CI [{_num(m['ci_low'])}, {_num(m['ci_high'])}]"
        if m.get("ci_low") is not None and m.get("ci_high") is not None
        else ""
    )
    return f"{_num(m['mean'])}{std} (n {m.get('n')}/{m.get('expected_n')}){ci}"


def _r_compare(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    exp = d.get("experiment") or {}
    rs = d.get("run_set") or {}
    c = d.get("counts") or {}
    out: Lines = [
        f"comparison_id {d.get('comparison_id')}{' (reused: identical to an archived comparison)' if d.get('reused') else ''}"
        f" — {exp.get('name')} ({rs.get('kind')} {rs.get('id')}); planned runs {c.get('planned_runs')}: "
        f"{c.get('included')} included, {c.get('missing')} missing, {c.get('failed')} failed, {c.get('excluded')} excluded",
    ]
    info = {m.get("name"): m for m in d.get("metric_info") or []}
    declared = _declared_metrics(d)
    for metric in _unknown_metrics(d):
        out.append(
            f"UNKNOWN METRIC {metric}: not declared in the comparison contract and reported by no selected "
            "evaluation, so every seed shows as missing — check the name"
            + (f" (declared: {', '.join(declared)})" if declared else "")
        )
    seeds = d.get("expected_seeds") or []
    for metric in d.get("metric_names") or []:
        mi = info.get(metric) or {}
        hib = mi.get("higher_is_better")
        arrow = " ↑" if hib is True else " ↓" if hib is False else ""
        unit = f", {mi['unit']}" if mi.get("unit") else ""
        out.append(
            f"{metric}{arrow}{unit} (mean ± sample std over seeds {', '.join(str(s) for s in seeds)}):"
        )
        rows = []
        for g in d.get("groups") or []:
            m = (g.get("metrics") or {}).get(metric) or {}
            per_seed = " ".join(f"{s}:{_num(v)}" for s, v in (m.get("values_by_seed") or {}).items())
            gaps: list[str] = []
            if m.get("missing_seeds"):
                gaps.append("missing " + ",".join(str(s) for s in m["missing_seeds"]))
            if m.get("failed_seeds"):
                gaps.append("failed " + ",".join(str(s) for s in m["failed_seeds"]))
            if m.get("excluded"):
                gaps.append("excluded " + ",".join(str(e.get("seed")) for e in m["excluded"]))
            rows.append(
                [
                    f"{g.get('label')}{' (baseline)' if g.get('is_baseline') else ''}",
                    _stat(m),
                    per_seed,
                    "; ".join(gaps),
                ]
            )
        out += ["  " + x for x in _table(rows, max_rows=30)]
        for diff in d.get("differences") or []:
            if diff.get("metric") != metric:
                continue
            ci = (
                f" CI [{_num(diff['ci_low'])}, {_num(diff['ci_high'])}]"
                if diff.get("ci_low") is not None and diff.get("ci_high") is not None
                else ""
            )
            n = (
                f"{diff.get('n')} pairs"
                if diff.get("kind") == "paired"
                else f"n {diff.get('n')} vs {diff.get('n_baseline')}"
            )
            md = _num(diff["mean_diff"]) if diff.get("mean_diff") is not None else "—"
            out.append(
                f"  {diff.get('group_label')} − {diff.get('baseline_label')}: {md}{ci} ({diff.get('kind')}, {n}; descriptive)"
            )
    not_included = [r for r in d.get("runs") or [] if r.get("status") != "included"]
    if not_included:
        out.append("not included:")
        for r in not_included[:20]:
            out.append(
                f"- {r.get('status')} {r.get('label')} (run_id {r.get('run_id')}, code {r.get('code')}"
                + (f", latest attempt {r['latest_attempt_id']})" if r.get("latest_attempt_id") else ")")
                + (
                    f": {r.get('exclusion_reason') or r.get('reason')}"
                    if r.get("exclusion_reason") or r.get("reason")
                    else ""
                )
            )
        if len(not_included) > 20:
            out.append(f"… {len(not_included) - 20} more (runs[] in structuredContent)")
    for inc in d.get("incomparable") or []:
        out.append(
            f"INCOMPARABLE: {inc.get('reason')} (invariant {inc.get('invariant')}, metric {inc.get('metric')})"
        )
    integ = d.get("integrity") or {}
    for het in integ.get("heterogeneity") or []:
        if het.get("mixed"):
            out.append(f"integrity: mixed {het.get('dimension')}: {_counts_str(het.get('values'))}")
    if integ.get("post_hoc"):
        out.append(f"integrity: {len(integ['post_hoc'])} run(s) added after results existed (post hoc)")
    for ex in integ.get("exclusions") or []:
        if ex.get("observed_before_exclusion"):
            out.append(
                f"integrity: exclusion of {ex.get('run_id')} was decided after its result was observed"
            )
    for bm in integ.get("below_min_n") or []:
        out.append(
            f"integrity: {bm.get('group_label')}/{bm.get('metric')} has n {bm.get('n')} < min_n {bm.get('min_n')}"
        )
    return out


def _declared_metrics(d: dict[str, Any]) -> list[str]:
    return [str(m.get("name")) for m in (d.get("contract") or {}).get("metrics") or [] if m.get("name")]


def _unknown_metrics(d: dict[str, Any]) -> list[str]:
    """Requested metrics that neither the contract declares nor any selected evaluation reports
    (typically a typo: 'accuracy' for 'acc'); every seed then looks 'missing'."""
    return [str(m.get("name")) for m in d.get("metric_info") or [] if m.get("source") == "unknown"]


def _results_section(md: str) -> Lines:
    m = re.search(r"^## Results\n(.*?)(?=^## |\Z)", md or "", re.S | re.M)
    if not m:
        return []
    return [line for line in m.group(1).strip().splitlines() if line.startswith("|")]


def _r_evidence(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    c = d.get("counts") or {}
    out: Lines = [
        f"bundle_id {d.get('bundle_id')}  from comparison {d.get('comparison_id')}  purpose: {d.get('purpose')}",
        f"coverage: {c.get('included')}/{c.get('planned_runs')} planned runs included "
        f"({c.get('missing')} missing, {c.get('failed')} failed, {c.get('excluded')} excluded)",
        "files (read as resources):",
    ]
    out += [f"- {f.get('uri')} ({f.get('bytes')} B)" for f in d.get("files") or []]
    table = _results_section(d.get("summary_markdown") or "")
    if table:
        out.append("results (mean ± sample std (included/expected)):")
        out += table[:30]
    if d.get("incomparable"):
        out.append(f"INCOMPARABLE results: {d['incomparable']}")
    out += [f"integrity: {f}" for f in d.get("integrity_flags") or []]
    return out


def _r_verify(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = [
        f"{d.get('values_checked')} value(s) recomputed from per-seed evaluations; "
        f"{d.get('tags_checked')} \\hpcval tag(s) checked; artifacts re-hashed: {d.get('artifacts_rechecked', 0)}"
    ]
    for f in d.get("findings") or []:
        where = f" @ {f['refs'][0]}" if f.get("refs") else ""
        out.append(f"- [{f.get('severity')}] {f.get('code')}: {f.get('message')}{where}")
    if d.get("findings_truncated"):
        out.append(f"… {d['findings_truncated']} more finding(s)")
    for t in (d.get("tex") or [])[:10]:
        out.append(
            f"tex {t.get('source')}: {t.get('tags')} tag(s), {t.get('untagged_numbers')} untagged number(s)"
        )
    scanned = [str(f) for f in d.get("files_scanned") or []]
    if scanned:
        more = f" … {len(scanned) - 10} more" if len(scanned) > 10 else ""
        out.append(f"files scanned ({len(scanned)}): {', '.join(scanned[:10])}{more}")
    if _manuscript_given(d) and not _verify_checked(d):
        out.append("manuscript NOT checked: no \\hpcval-tagged number in the scanned files")
    if d.get("ledger_changed"):
        out.append("the ledger changed since this bundle's comparison was archived")
    elif d.get("ledger_changed") is None:
        out.append("could not re-derive the comparison from the ledger (see warnings)")
    if d.get("newer_comparison_ref"):
        out.append(f"newer comparison: {d['newer_comparison_ref']}")
    out.append(d.get("statement") or "Verifies record consistency, not scientific correctness.")
    return out


def _r_workflow_cancel(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = [
        f"workflow {d.get('workflow_id')} version {d.get('version')}: state {d.get('state')}, dispatch {d.get('dispatch_state')}, "
        f"scope {d.get('scope')}; {_counts_str(d.get('counts'))}"
    ]
    rows = [
        [
            str(c.get("label")),
            str(c.get("run_id")),
            str(c.get("cell_state")),
            str(c.get("cancel_state") or "-"),
            str(c.get("note") or ""),
        ]
        for c in d.get("cells") or []
    ]
    out += ["  " + x for x in _table(rows, ["label", "run_id", "cell", "cancel", "note"], max_rows=30)]
    if d.get("budget_kept_gpu_hours"):
        out.append(f"budget still reserved for in-flight cells: {_gh(d['budget_kept_gpu_hours'])}")
    out += [f"note: {n}" for n in d.get("notes") or []]
    return out


def _usage_stats(s: dict[str, Any]) -> str:
    parts = [f"{s.get('jobs')} jobs", _gh(s.get("gpu_hours")) + " allocated"]
    if s.get("gpu_hours_by_state"):
        parts.append("by state " + ", ".join(f"{k} {_num(v)}" for k, v in s["gpu_hours_by_state"].items()))
    if s.get("wasted_gpu_hours_by_cause"):
        parts.append(
            "waste " + ", ".join(f"{k} {_num(v)}" for k, v in s["wasted_gpu_hours_by_cause"].items())
        )
    if s.get("median_queue_wait_s") is not None:
        parts.append(f"median queue wait {_num(s['median_queue_wait_s'])} s")
    if s.get("walltime_accuracy") is not None:
        parts.append(f"walltime used/requested {_num(s['walltime_accuracy'])}")
    if s.get("gpu_count_unknown_jobs"):
        parts.append(f"{s['gpu_count_unknown_jobs']} job(s) with unknown GPU count (lower bound)")
    return "; ".join(parts)


def _r_usage(d: dict[str, Any], env: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = [
        f"{d.get('scope')} usage on {d.get('cluster_id')} ({d.get('account') or d.get('user')}), {d.get('since')} → {d.get('until')}; "
        f"visibility {d.get('visibility')} (PrivateData {d.get('private_data') or '?'})"
    ]
    for k, s in (d.get("totals") or {}).items():
        out.append(f"total ({k}): {_usage_stats(s)}")
    if d.get("visibility") != "full" and d.get("scope") == "account":
        out.append(
            "other users' jobs are not visible: there is no lab total; do not present these numbers as the lab's usage"
        )
    for r in d.get("rows") or []:
        grp = " ".join(f"{k}={v}" for k, v in (r.get("group") or {}).items())
        out.append(f"- {grp}: {_usage_stats(r)}")
    if d.get("rows_truncated"):
        out.append(f"… {d['rows_truncated']} more row(s)")
    now = d.get("now") or {}
    if now:
        out.append(
            f"now: {now.get('running_jobs')} running ({now.get('running_gpus')} GPUs), {now.get('pending_jobs')} pending "
            f"({now.get('pending_gpus')} GPUs); visibility {now.get('visibility')}"
        )
    out += [f"note: {n}" for n in d.get("notes") or []]
    return out


RENDERERS: dict[str, Renderer] = {
    "cluster_inspect": _r_cluster_inspect,
    "experiment_prepare": _r_prepare,
    "run_submit": _r_submit,
    "workflow_submit": _r_workflow,
    "runs_list": _r_runs_list,
    "run_inspect": _r_run_inspect,
    "run_logs": _r_logs,
    "run_cancel": _r_cancel,
    "run_recover": _r_recover,
    "artifacts_list": _r_artifacts,
    "artifact_read": _r_artifact_read,
    "evaluation_collect": _r_collect,
    "results_compare": _r_compare,
    "evidence_export": _r_evidence,
    "evidence_verify": _r_verify,
    "workflow_cancel": _r_workflow_cancel,
    "usage_report": _r_usage,
}


# ---------------------------------------------------------------------------
# Next-step suggestions (exact calls; never actions taken on the user's behalf)
# ---------------------------------------------------------------------------

_FRESH = "<new key>"


def _n_cluster(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    auth = next((c for c in d.get("checks") or [] if c.get("name") == "auth"), None)
    if auth and auth.get("status") in ("fail", "unavailable"):
        return [
            "ask the user to authenticate in their own terminal (see the auth fix above), then call cluster_inspect again"
        ]
    if d.get("status") == "fail":
        return ["fix the failing checks above before preparing experiments on this cluster"]
    return [
        f"{call_hint('experiment_prepare', spec_path='hpclab.yaml', idempotency_key=_FRESH)} (spec with cluster: {d.get('cluster_id')})"
    ]


def _n_prepare(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    status = d.get("status")
    blocking = (d.get("validation") or {}).get("blocking") or []
    if status == "preparing":
        return [
            f"read {d.get('operation_ref')} after a few seconds (snapshot in progress), then hpclab://plans/{d.get('plan_id')}"
        ]
    if status == "invalid" or blocking:
        return [
            "fix the blocking findings (see fix: lines) and call experiment_prepare again with a new idempotency_key",
            "overrides=[{check, reason}] only with the user's explicit agreement",
        ]
    runs = [r for r in d.get("runs") or [] if r.get("kind") == "main"]
    auth = next(
        (
            str(f["remediation"])
            for f in (d.get("validation") or {}).get("findings") or []
            if str(f.get("remediation") or "").startswith("hpclab auth ")
        ),
        None,
    )
    out = (
        [
            f"the cluster needs SSH authentication (some checks could not run): ask the user to run `{auth}` in "
            "their own terminal before dispatching — submissions wait until then"
        ]
        if auth
        else []
    )
    out += [
        f"{call_hint('workflow_submit', plan_id=d.get('plan_id'), plan_hash=d.get('plan_hash'), idempotency_key=f'dispatch-{d.get("plan_id")}')} "
        f"— dispatches {'the smoke run first, then ' if d.get('smoke_run_id') else ''}{len(runs)} main run(s); "
        f"confirm the GPU-hours with the user first",
        f"optional review of the job script: resource hpclab://plans/{d.get('plan_id')}/script",
    ]
    return out


def _n_submit(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    secs = max(1, int((d.get("poll_after_ms") or 2000) / 1000))
    return [
        f"after ~{secs}s: {call_hint('run_inspect', run_id=d.get('run_id'))} (shows the Slurm job id once accepted)",
        "do not submit this run again: it holds the run's single execution claim",
    ]


def _n_workflow(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    state = str(d.get("state") or "")
    exp = d.get("experiment_id")
    if state in ("completed", "cancelled", "gate_failed"):
        nxt = [call_hint("runs_list", experiment_id=exp, view="grid")]
        if state == "gate_failed":
            gate = d.get("gate") or {}
            nxt.insert(0, f"the smoke run failed: {call_hint('run_inspect', run_id=gate.get('run_id'))}")
        return nxt
    if d.get("dispatch_state") == "paused":
        return [
            f"dispatch is paused (circuit breaker): inspect the failing cells, fix the cause, then "
            f"{call_hint('workflow_submit', workflow_id=d.get('workflow_id'), resume_dispatch=True, expected_version=d.get('version'), idempotency_key=_FRESH)}"
        ]
    return [
        f"check progress later: {call_hint('runs_list', experiment_id=exp, view='grid')} "
        "(jobs take minutes to hours; do not poll in a tight loop)",
        f"dispatch state and version: resource hpclab://workflows/{d.get('workflow_id')}",
    ]


def _n_runs_list(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    counts = d.get("counts") or {}
    ref = (
        {"experiment_id": args.get("experiment_id")}
        if args.get("experiment_id")
        else {"workflow_id": args.get("workflow_id")}
    )
    out: Lines = []
    if d.get("view") == "grid":
        items = [c for r in (d.get("grid") or {}).get("rows") or [] for c in r.get("cells") or []]
    else:
        items = list(d.get("runs") or [])
    # counts cover every run; the items only this page (grid rows are paginated)
    unevaluated = max(sum(1 for c in items if c.get("code") == "E"), int(counts.get("E") or 0))
    broken = [c for c in items if c.get("code") in ("F", "C!")]
    if broken:
        out.append(f"why it failed: {call_hint('run_inspect', run_id=broken[0].get('run_id'))}")
    if unevaluated:
        run_set = ref.get("experiment_id") or ref.get("workflow_id")
        out.append(
            f"{unevaluated} completed run(s) are not evaluated yet (normally automatic within seconds): "
            f"check again shortly or {call_hint('evaluation_collect', run_set_ref=run_set, idempotency_key=_FRESH)}"
        )
    if counts.get("U"):
        out.append("U cells: submission uncertain — wait, never resubmit; the controller reconciles them")
    notes = (d.get("grid") or {}).get("notes") or []
    workflow_stopped = any(re.match(r"workflow wf_\w+: state", n) for n in notes)
    for n in notes:
        smoke = re.match(r"smoke run .*?\((run_\w+)\): (\S+) ?(.*?) - ", n)
        if smoke:
            sid, scode = smoke.group(1), smoke.group(2)
            if scode in ("R", "Q", "?"):
                out.append(
                    "the smoke run has not passed yet; main cells start after it does — check again later"
                )
            elif scode == "S":
                out.append(
                    "the smoke run is being submitted; check again in a minute — if it stays S, "
                    f"{call_hint('run_inspect', run_id=sid)} shows what blocks it (e.g. SSH authentication)"
                )
            elif scode == "U":
                out.append(
                    "the smoke run's submission is uncertain: wait, never resubmit — the controller reconciles it"
                )
            elif scode == "E":
                out.append(
                    f"the smoke run completed ({_PENDING_EVAL}); its outputs are being checked and evaluated — "
                    f"check again shortly; if it stays E for minutes, {call_hint('run_inspect', run_id=sid)} "
                    "shows why (the main cells wait for it)"
                )
            elif scode == "P" and not workflow_stopped:
                if ref.get("experiment_id"):
                    out.append(
                        "nothing is running and the smoke run was never dispatched: if no workflow was started "
                        f"for this experiment, {call_hint('workflow_submit', prepared_experiment_id=ref['experiment_id'], idempotency_key=_FRESH)} "
                        "dispatches it (smoke first; confirm the GPU-hours with the user); a workflow started "
                        "seconds ago dispatches it shortly"
                    )
                else:
                    out.append("the smoke run is about to be dispatched — check again shortly")
            elif scode in ("F", "C!", "X"):
                out.append(
                    f"the smoke gate failed, so main cells stay P: {call_hint('run_inspect', run_id=sid)}; "
                    "fix the cause, recover the smoke run (run_recover + run_submit), then resume the workflow"
                )
        stopped = re.match(r"workflow (wf_\w+): state (\S+), dispatch (\S+)", n)
        if stopped and (stopped.group(2) == "gate_failed" or stopped.group(3) == "paused"):
            out.append(
                f"after the fix: {call_hint('workflow_submit', workflow_id=stopped.group(1), resume_dispatch=True, expected_version='<version from hpclab://workflows/' + stopped.group(1) + '>', idempotency_key=_FRESH)}"
            )
    running = sum(counts.get(k, 0) for k in ("R", "Q", "S", "?"))
    waiting = counts.get("P", 0)
    if running:
        out.append(
            f"{running} run(s) queued/running: check again later (runs_list is cheap; jobs take a while)"
        )
        submitting = next((c for c in items if c.get("code") == "S" and c.get("run_id")), None)
        if submitting is not None:
            out.append(
                f"if cells stay S for more than a minute: {call_hint('run_inspect', run_id=submitting['run_id'])} "
                "shows what blocks the submission (e.g. SSH authentication)"
            )
    elif waiting and not workflow_stopped and not any(n.startswith("smoke run ") for n in notes):
        out.append(
            f"{waiting} run(s) not dispatched and nothing is running: they start only through workflow_submit "
            "(or run_submit); check the workflow before waiting"
        )
    elif counts.get("V") and not waiting and not unevaluated and ref.get("experiment_id"):
        out.append(
            f"{call_hint('results_compare', planned_run_set_ref=ref['experiment_id'], metric_names=['<metric>'])} (missing/failed seeds stay visible)"
        )
    return out


# Tools whose schema requires idempotency_key (a test keeps this in sync with the registry).
_KEYED_TOOLS = (
    "evaluation_collect",
    "evidence_export",
    "experiment_prepare",
    "run_cancel",
    "run_recover",
    "run_submit",
    "workflow_cancel",
    "workflow_submit",
)


def _with_idempotency_key(hint: str) -> str:
    """The service's hints are shared with the CLI and leave out ``idempotency_key``, which these
    MCP tools require; an agent following one verbatim got 'idempotency_key: Field required'."""
    for tool in _KEYED_TOOLS:
        start = hint.find(f"{tool}(")
        if start < 0:
            continue
        end = hint.find(")", start)
        if end < 0 or "idempotency_key" in hint[start:end]:
            continue
        inner = hint[start + len(tool) + 1 : end].rstrip()
        trailing = re.search(r"(^|,\s*)\.\.\.$", inner)  # "run_cancel(attempt_id='a', ...)"
        if trailing:
            inner = inner[: trailing.start()]
            key = f'idempotency_key="{_FRESH}", ...'
        else:
            key = f'idempotency_key="{_FRESH}"'
        inner = f"{inner}, {key}" if inner else key
        hint = f"{hint[: start + len(tool) + 1]}{inner}{hint[end:]}"
    return hint


def _n_inspect(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = []
    for op in _attention_ops(d):
        if op.get("state") != "blocked":
            continue
        err = op.get("error") or {}
        fix = str((err.get("details") or {}).get("remediation") or "")
        if op.get("stage") == "auth_required" or err.get("code") == "auth_required":
            out.append(
                f"the {op.get('kind')} is waiting for SSH authentication: ask the user to run `{fix or 'hpclab auth <cluster>'}` "
                "in their own terminal (MFA prompts go to them); it resumes by itself afterwards — do not resubmit"
            )
        else:
            out.append(
                f"the {op.get('kind')} is blocked at {op.get('stage')}: {fix or err.get('message') or 'see the operation'} "
                f"(resource {op.get('ref')})"
            )
        break
    rec = list(d.get("recommended_next") or [])
    if rec:
        return out + [_with_idempotency_key(r) for r in rec[:5]]
    if out:
        return out
    run = d.get("run") or {}
    focus = d.get("focus_attempt_id")
    code = run.get("code")
    if code == "S":
        return [
            "the submission is in progress: check again in a minute (the Slurm job id appears once accepted)"
        ]
    if code in ("R", "Q"):
        return [
            f"{call_hint('run_logs', attempt_id=focus, tail_lines=40)} to watch output; or check again later"
        ]
    if code == "E":
        return [
            f"the job {_PENDING_EVAL}: its outputs are checked and evaluated automatically — check again "
            f"shortly, or {call_hint('evaluation_collect', attempt_id=focus, idempotency_key=f'collect-{focus}')}"
        ]
    if code in ("F", "C!"):
        return [call_hint("run_recover", attempt_id=focus, idempotency_key=f"recover-{focus}")]
    return []


def _n_logs(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    if not d.get("next_cursor"):
        return []
    call = call_hint(
        "run_logs", attempt_id=d.get("attempt_id"), stream=d.get("stream"), cursor=d.get("next_cursor")
    )
    if d.get("eof"):
        return [f"new output later (the job may still be writing): {call}"]
    return [f"continue reading: {call}"]


def _n_cancel(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    return [
        f"confirm later: {call_hint('run_inspect', run_id=d.get('run_id'))} (cancel_state stays 'requested' until the job is seen ending)"
    ]


def _n_recover(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    out: Lines = []
    pb = d.get("probe") or {}
    if (
        pb.get("run_id")
        and pb.get("exec_plan_hash")
        and pb.get("state") in ("created", "ready", "prepared", "pending")
    ):
        out.append(
            f"submit the checkpoint probe first: {call_hint('run_submit', run_id=pb['run_id'], execution_plan_hash=pb['exec_plan_hash'], idempotency_key=f'probe-{pb["run_id"]}')}, "
            "then call run_recover again"
        )
    fix_first = [
        str(h.get("strategy")).removeprefix("fix_")
        for h in d.get("hints") or []
        if str(h.get("strategy") or "").startswith("fix_")
    ]
    if d.get("mode") == "batch" and d.get("plan_id") and d.get("eligible_count"):
        caveat = (
            f"only after the {'/'.join(fix_first)} is fixed (see the hint; unchanged, it fails the same way): "
            if fix_first
            else ""
        )
        out.append(
            f"{caveat}{call_hint('workflow_submit', plan_id=d['plan_id'], idempotency_key=f'dispatch-{d["plan_id"]}')} "
            f"— {d.get('eligible_count')} cell(s), {_gh(d.get('requested_gpu_hours'))}; confirm with the user"
        )
    elif d.get("eligible") and d.get("exec_plan_hash") and d.get("run_id"):
        out.append(
            f"{call_hint('run_submit', run_id=d['run_id'], execution_plan_hash=d['exec_plan_hash'], idempotency_key=f'submit-{d["exec_plan_hash"][7:23]}')} "
            f"— {_gh(d.get('requested_gpu_hours'))}; confirm with the user"
        )
    for h in d.get("hints") or []:
        if h.get("tool"):
            hint_args = dict(h.get("arguments") or {})
            if h["tool"] in _KEYED_TOOLS:
                hint_args.setdefault("idempotency_key", _FRESH)
            out.append(call_hint(h["tool"], **hint_args))
    if not out and d.get("next_step"):
        out.append(str(d["next_step"]))
    return out


def _n_artifacts(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    arts = d.get("artifacts") or []
    pick = next((a for a in arts if a.get("role") == "metrics" and a.get("kind") == "file"), None) or next(
        (a for a in arts if a.get("kind") == "file"), None
    )
    return [call_hint("artifact_read", artifact_ref=pick.get("artifact_id"))] if pick else []


def _n_artifact_read(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    if d.get("truncated") and d.get("next_offset") is not None:
        a = d.get("artifact") or {}
        return [
            call_hint(
                "artifact_read",
                artifact_ref=a.get("artifact_id"),
                view="text",
                range={"offset": d["next_offset"]},
            )
        ]
    return []


def _n_collect(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    if d.get("pending"):
        return [
            f"read {d.get('operation_ref')} after ~{int((d.get('poll_after_ms') or 5000) / 1000)}s, before comparing"
        ]
    rs = d.get("run_set") or {}
    ref = rs.get("id") or args.get("run_set_ref")
    if d.get("mode") == "batch" and ref:
        return [call_hint("results_compare", planned_run_set_ref=ref, metric_names=["<metric>"])]
    return []


def _n_compare(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    cid = d.get("comparison_id")
    unknown = _unknown_metrics(d)
    if unknown:
        rs = d.get("run_set") or {}
        names = [m for m in d.get("metric_names") or [] if m not in unknown] or _declared_metrics(d)
        return [
            f"compare again with the right metric names: "
            f"{call_hint('results_compare', planned_run_set_ref=rs.get('id') or args.get('planned_run_set_ref'), metric_names=names or ['<metric>'])} "
            "(do not export a table of an unknown metric)"
        ]
    included = (d.get("counts") or {}).get("included")
    out = (
        [
            f"{call_hint('evidence_export', comparison_ref=cid, purpose='<what the table is for>', idempotency_key=f'export-{cid}')} "
            "→ LaTeX/CSV/Markdown with traceable cells"
        ]
        if included != 0
        else ["nothing to export yet: no planned run has a valid result"]
    )
    runs = d.get("runs") or []
    unevaluated = [r for r in runs if r.get("code") == "E"]
    if unevaluated:
        rs = (d.get("run_set") or {}).get("id") or args.get("planned_run_set_ref")
        out.append(
            f"{len(unevaluated)} run(s) {_PENDING_EVAL}, counted as missing: "
            f"{call_hint('evaluation_collect', run_set_ref=rs, idempotency_key=_FRESH)}, then compare again"
        )
    failed = [
        r
        for r in runs
        if r.get("status") in ("failed", "missing")
        and r.get("latest_attempt_id")
        and r.get("code") not in ("E", "R", "Q", "S", "U", "?")  # still running or evaluating
    ]
    if failed:
        r = failed[0]
        att = r["latest_attempt_id"]
        out.append(
            f"to fill the gap: {call_hint('run_recover', attempt_id=att, idempotency_key=f'recover-{att}')} "
            f"(details: {call_hint('run_inspect', run_id=r.get('run_id'))}); report gaps that cannot be recovered"
        )
    return out


def _n_evidence(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    bid = d.get("bundle_id")
    gaps = [w for w in d.get("_warnings") or [] if "lack values" in str(w)]
    caution = (
        [
            "do not paste yet: some cells have no value although their runs are included (see the warning) — "
            "usually a metric name the evaluations do not report; check metric_names and export again"
        ]
        if gaps
        else []
    )
    return [
        *caution,
        f"read resource hpclab://evidence/{bid}/table.tex and paste it unchanged into the paper (keep the \\hpcval tags)",
        f"before submission: {call_hint('evidence_verify', bundle_ref=bid, tex_paths=['<paper>.tex'])}",
        f"provenance of one number: resource hpclab://trace/{bid}:<cell_id> (cell ids appear in the \\hpcval tags)",
    ]


def _manuscript_given(d: dict[str, Any]) -> bool:
    return bool(
        d.get("files_scanned") or d.get("tex") or (d.get("counts_by_code") or {}).get("NO_TAGS_FOUND")
    )


def _verify_checked(d: dict[str, Any]) -> bool:
    """Was any \\hpcval tag found (``checked``; older services: per-source tag counts)?"""
    if d.get("checked") is not None:
        return bool(d["checked"])
    return bool(d.get("tags_checked")) or any(t.get("tags") for t in d.get("tex") or [])


def _n_verify(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    codes = d.get("counts_by_code") or {}
    out: Lines = []
    newer = str(d.get("newer_comparison_ref") or "").removeprefix("hpclab://comparisons/")
    if newer:
        out.append(
            f"results changed since this bundle: {call_hint('evidence_export', comparison_ref=newer, purpose='<same purpose>', idempotency_key=f'export-{newer}')} "
            "then replace the table in the .tex with the new table.tex"
        )
    elif codes.get("STALE_TEX_NUMBER") or d.get("ledger_changed"):
        out.append(
            "results changed since this bundle: results_compare(planned_run_set_ref=<experiment_id>, metric_names=[...]) "
            "→ evidence_export(comparison_ref=<new comparison_id>) → replace the table in the .tex"
        )
    if codes.get("VALUE_MISMATCH"):
        out.append(
            f"restore hand-edited numbers from resource hpclab://evidence/{d.get('bundle_id')}/table.tex (never edit tagged values)"
        )
    if codes.get("N_BELOW_EXPECTED_UNFOOTNOTED"):
        out.append("add the missing-seed footnote from the bundle's table.tex next to the affected cells")
    unread = [  # given files that could not be read (an unreadable include is UNRESOLVED_INPUT, below)
        str(s.get("name")).removeprefix("tex:")
        for s in (d.get("_coverage") or {}).get("sources") or []
        if str(s.get("name")).startswith("tex:")
        and s.get("status") != "ok"
        and not str(s.get("detail") or "").startswith("\\")
    ]
    if unread:
        out.append(
            f"not checked: {', '.join(unread[:3])} (see the warning) — fix the path in tex_paths (relative paths "
            "resolve against the agent's working directory) or pass the source as tex_content, and verify again"
        )
    if codes.get("UNRESOLVED_INPUT"):
        out.append(
            "an \\input/\\include could not be read (UNRESOLVED_INPUT; its numbers were NOT checked): fix the "
            "path, or pass that file in tex_paths (absolute, or relative to the agent's working directory), "
            "and verify again"
        )
    if _manuscript_given(d) and not _verify_checked(d) and not (unread and not d.get("files_scanned")):
        out.append(
            "no \\hpcval-tagged number was found, so nothing in the manuscript was checked (\\input/\\include "
            "files are followed; files_scanned lists what was read) — pass the .tex that contains or \\inputs "
            f"the table (or paste the bundle's table.tex, resource hpclab://evidence/{d.get('bundle_id')}/table.tex, "
            "unchanged)"
        )
    return out


def _n_workflow_cancel(d: dict[str, Any], args: dict[str, Any]) -> Lines:
    if d.get("state") == "cancelled":
        return [
            f"done: nothing more is dispatched; {call_hint('runs_list', workflow_id=d.get('workflow_id'), view='grid')}"
        ]
    return [f"resource hpclab://workflows/{d.get('workflow_id')} shows the cancellation progress"]


NEXT_STEPS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Lines]] = {
    "cluster_inspect": _n_cluster,
    "experiment_prepare": _n_prepare,
    "run_submit": _n_submit,
    "workflow_submit": _n_workflow,
    "runs_list": _n_runs_list,
    "run_inspect": _n_inspect,
    "run_logs": _n_logs,
    "run_cancel": _n_cancel,
    "run_recover": _n_recover,
    "artifacts_list": _n_artifacts,
    "artifact_read": _n_artifact_read,
    "evaluation_collect": _n_collect,
    "results_compare": _n_compare,
    "evidence_export": _n_evidence,
    "evidence_verify": _n_verify,
    "workflow_cancel": _n_workflow_cancel,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

_ERROR_REMEDIATION: dict[str, str] = {
    "invalid_argument": "fix the arguments (see the tool's input schema) and call again",
    "not_found": "check the id (ids are prefixed: exp_, pln_, run_, att_, wf_, cmp_, evb_, art_, ev_, op_); runs_list(experiment_id=...) lists runs",
    "version_conflict": "the object changed since you read it: re-read its current version and retry only if the action still makes sense",
    "idempotency_conflict": "this idempotency_key was already used for a different request: use a new key for a new request",
    "auth_required": "ask the user to authenticate in their own terminal (`hpclab auth <cluster>`; MFA prompts go to them), then retry",
    "controller_unavailable": "the local hpclab controller is not running or not answering: ask the user to run `hpclab status` / `hpclab controller start`",
    "not_ready": "not ready yet: retry after the operation finishes (see operation_ref / poll_after_ms)",
    "unavailable": "a source was unreachable; retry later — nothing was assumed",
    "budget_exceeded": "the experiment budget does not cover this: tell the user; raising the budget is their decision",
    "internal": "an internal error; the controller log has the traceback (`hpclab status` shows where)",
    "stale": "the data changed while it was read: read it again",
}


_ID_KINDS: dict[str, tuple[str, str]] = {
    # prefix -> (what it is, where it goes)
    "exp_": ("an experiment id", "runs_list(experiment_id=...)"),
    "pln_": ("a plan id", "resource hpclab://plans/<plan_id>"),
    "run_": ("a run id", "run_inspect(run_id=...)"),
    "att_": ("an attempt id", "attempt_id"),
    "wf_": ("a workflow id", "runs_list(workflow_id=...)"),
    "cmp_": ("a comparison id", "evidence_export(comparison_ref=...)"),
    "evb_": ("an evidence bundle id", "evidence_verify(bundle_ref=...)"),
    "art_": ("an artifact id", "artifact_read(artifact_ref=...)"),
    "ev_": ("an evaluation id", "resource hpclab://evaluations/<evaluation_id>"),
    "op_": ("an operation id", "resource hpclab://operations/<operation_id>"),
}
_PARAM_PREFIX = {
    "experiment_id": "exp_",
    "plan_id": "pln_",
    "run_id": "run_",
    "attempt_id": "att_",
    "workflow_id": "wf_",
    "comparison_id": "cmp_",
    "bundle_id": "evb_",
    "artifact_id": "art_",
    "evaluation_id": "ev_",
    "operation_id": "op_",
}


def _wrong_kind(details: dict[str, Any]) -> str | None:
    """'run att_… not found': the agent passed an id of another kind; say where it belongs."""
    for param, prefix in _PARAM_PREFIX.items():
        value = details.get(param)
        if not isinstance(value, str) or value.startswith(prefix):
            continue
        kind = next((k for k in _ID_KINDS if value.startswith(k)), None)
        if kind is None:
            continue
        what, where = _ID_KINDS[kind]
        use = f"pass it as {where}" if "(" not in where and " " not in where else f"use {where}"
        return f"{value!r} is {what} ({param} takes {prefix}… ids): {use}"
    return None


def _what_to_do(code: str, details: dict[str, Any]) -> str | None:
    """Remediation for an error code, specialised by what the details say."""
    if code == "not_found":
        if any(k.endswith("_path") or k == "path" for k in details):
            return "check the path (relative paths resolve against the agent's working directory)"
        if "known_clusters" in details:
            known = ", ".join(str(c) for c in details["known_clusters"]) or "none"
            return f"registered clusters: {known}; the user registers others with `hpclab cluster add`"
        wrong = _wrong_kind(details)
        if wrong:
            return wrong
        listed = next(
            (k for k in details if (k.startswith("known_") or k == "available") and details[k]), None
        )
        if listed:
            return f"use one of {listed} (in the details above)"
    if code == "invalid_argument" and details.get("unknown_metrics") and details.get("known_metrics"):
        run_set = str(details.get("run_set") or "<experiment_id>").rstrip("/").rsplit("/", 1)[-1]
        return (
            call_hint(
                "results_compare",
                planned_run_set_ref=run_set,
                metric_names=[str(m) for m in details["known_metrics"]],
            )
            + " (or a subset of those metric names)"
        )
    if code == "admission_denied" and details.get("workflow_id"):
        wid = details["workflow_id"]
        return (
            f"this plan is already dispatched by workflow {wid}: follow it with "
            f"{call_hint('runs_list', workflow_id=wid, view='grid')} instead of dispatching again"
        )
    return _ERROR_REMEDIATION.get(code)


def _render_error(tool: str | None, env: dict[str, Any]) -> str:
    err = env.get("error") or {}
    code = str(err.get("code") or "error")
    lines: Lines = [f"ERROR {code}: {err.get('message') or env.get('summary') or ''}"]
    details = dict(err.get("details") or {})
    for e in details.pop("errors", None) or []:
        if isinstance(e, dict):
            lines.append(f"- {e.get('loc') or '(arguments)'}: {e.get('msg')}")
        else:
            lines.append(f"- {e}")
    for key in ("remediation", "hint", "login_command_str"):
        if details.get(key):
            lines.append(f"{'run' if key == 'login_command_str' else key}: {details.pop(key)}")
    details.pop("login_command", None)
    if code == "version_conflict" and details.get("current_version") is not None:
        lines.append(
            f"current version is {details['current_version']}: retry with expected_version={details['current_version']} if still appropriate"
        )
    if details:
        lines.append("details: " + compact_json(details, budget=2048))
    if err.get("operation_ref"):
        lines.append(f"operation: {err['operation_ref']}")
    lines.append(f"retryable: {'yes (same idempotency_key)' if err.get('retryable') else 'no'}")
    what = _what_to_do(code, dict(err.get("details") or {}))
    if what:
        lines.append(f"what to do: {what}")
    if isinstance(env.get("data"), dict) and env["data"]:
        lines.append("partial data: " + compact_json(env["data"], budget=2048))
    return _fit("\n".join(lines), TEXT_HARD_LIMIT)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _notes(env: dict[str, Any], tool: str | None = None) -> Lines:
    out: Lines = []
    for w in env.get("warnings") or []:
        out.append(f"warning: {w}")
    cov = env.get("coverage") or {}
    if cov and not cov.get("complete", True):
        bad = [
            f"{s.get('name')}={s.get('status')}" + (f" ({s['detail']})" if s.get("detail") else "")
            for s in cov.get("sources") or []
            if s.get("status") not in ("ok", "cached", "skipped")
        ]
        out.append(
            "INCOMPLETE: "
            + ("; ".join(bad) if bad else "the result does not cover everything (see below)")
            + " — do not report missing parts as zero or as done"
        )
        missing = cov.get("missing") or []
        if tool == "results_compare" and missing:
            out.append(f"  missing: {len(missing)} planned run(s), listed under 'not included' above")
        else:
            out += [f"  missing: {m}" for m in missing]
        out += [f"  note: {n}" for n in cov.get("notes") or []]
    if env.get("next_cursor") and tool not in CONTENT_TOOLS:
        out.append(f"more results: call again with the same arguments plus cursor={_val(env['next_cursor'])}")
    return out


def render_text(tool: str | None, env: dict[str, Any], args: dict[str, Any] | None = None) -> str:
    """The text block of a tool result (see the module docstring)."""
    if not env.get("ok"):
        return _render_error(tool, env)
    args = args or {}
    data = env.get("data")
    head = str(env.get("summary") or "ok")
    body: Lines | None = None
    renderer = RENDERERS.get(tool or "")
    if renderer is not None and isinstance(data, dict):
        try:
            body = renderer(data, env, args)
        except Exception:  # a renderer bug must never hide the result
            log.exception("text renderer for %s failed; using compact JSON", tool)
            body = None
    if body is None and data not in (None, {}, []):
        body = [compact_json(data)]
    tail = _notes(env, tool)
    nxt: Lines = []
    step = NEXT_STEPS.get(tool or "")
    if step is not None and isinstance(data, dict):
        try:
            nxt = step(
                {
                    **data,
                    "next_cursor": env.get("next_cursor"),
                    "_coverage": env.get("coverage"),
                    "_warnings": env.get("warnings"),
                },
                args,
            )
        except Exception:
            log.exception("next-step hints for %s failed", tool)
    if nxt:
        tail += ["Next:", *(f"- {n}" for n in nxt)]
    limit = CONTENT_HARD_LIMIT if tool in CONTENT_TOOLS else TEXT_HARD_LIMIT
    return _assemble(head, body or [], tail, limit)


def _assemble(head: str, body: Lines, tail: Lines, limit: int) -> str:
    """Summary + body + (notes, cursor, Next:) within ``limit`` bytes. When it does not fit, the
    body is cut (at a line boundary) so the summary, the page cursor and the next steps survive."""
    text = "\n".join([head, *body, *tail])
    if len(text.encode()) <= limit:
        return text
    marker = f"… [truncated to {limit} bytes; the complete result is in structuredContent]"
    fixed = len("\n".join([head, marker, *tail]).encode()) + 1
    room = limit - fixed
    if room < 512:  # the notes alone are too long: plain cut
        return _fit(text, limit)
    cut = "\n".join(body).encode()[:room].decode("utf-8", errors="ignore")
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return "\n".join([head, cut, marker, *tail])


def to_call_result(
    env: dict[str, Any], tool: str | None = None, args: dict[str, Any] | None = None
) -> types.CallToolResult:
    ok = bool(env.get("ok"))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=render_text(tool, env, args))],
        structured_content=env,
        is_error=not ok,
    )


def error_result(message: str, code: str = "internal", **details: Any) -> types.CallToolResult:
    env = {
        "schema_version": "hpclab/v1",
        "request_id": "",
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": code in ("controller_unavailable", "unavailable"),
            "operation_ref": None,
            "details": details,
        },
        "coverage": None,
        "observed_at": None,
        "next_cursor": None,
        "summary": f"{code}: {message}",
        "warnings": [],
    }
    return to_call_result(env)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceRoute:
    template: str
    name: str
    description: str
    method: str
    params: Callable[[dict[str, str]], dict[str, Any]]
    mime_type: str = "application/json"
    text_field: str | None = None  # serve data[text_field] as raw text instead of JSON
    greedy: str | None = None  # placeholder that may contain '/'

    @property
    def regex(self) -> re.Pattern[str]:
        pattern = re.escape(self.template)

        def group(m: re.Match[str]) -> str:
            name = m.group(1)
            return f"(?P<{name}>.+)" if name == self.greedy else f"(?P<{name}>[^/]+)"

        pattern = re.sub(r"\\\{(\w+)\\\}", group, pattern)
        return re.compile("^" + pattern + "$")


RESOURCES: list[ResourceRoute] = [
    ResourceRoute(
        "hpclab://operations/{operation_id}",
        "operation",
        "Background operation: kind, state, stage, error, progress (never faked), result refs, suggested poll interval.",
        "operation_get",
        lambda m: {"operation_id": m["operation_id"]},
    ),
    ResourceRoute(
        "hpclab://experiments",
        "experiments",
        "Experiments in the ledger, newest first (experiment_id, name, cluster, status, run and plan counts).",
        "experiments_list",
        lambda m: {"limit": 50},
    ),
    ResourceRoute(
        "hpclab://experiments/{experiment_id}",
        "experiment",
        "Experiment spec, plans, run counts by cell code, budget summary.",
        "experiment_get",
        lambda m: {"experiment_id": m["experiment_id"]},
    ),
    ResourceRoute(
        "hpclab://plans/{plan_id}/script/full",
        "plan-script-full",
        "The complete sbatch script (incl. the static hpclab wrapper runtime) of the plan's first main run.",
        "plan_script",
        lambda m: {"plan_id": m["plan_id"], "view": "full", "max_bytes": 65536},
        "text/x-shellscript",
        "script",
    ),
    ResourceRoute(
        "hpclab://plans/{plan_id}/script",
        "plan-script",
        "The run-specific part of the sbatch script of the plan's first main run (#SBATCH directives, argv, env, setup) "
        "for review; the static wrapper is replaced by a marker (see /script/full).",
        "plan_script",
        lambda m: {"plan_id": m["plan_id"], "view": "generated", "max_bytes": 65536},
        "text/x-shellscript",
        "script",
    ),
    ResourceRoute(
        "hpclab://plans/{plan_id}",
        "plan",
        "Frozen runs with their RunSpec and execution plan, hashes, validation findings, budget (first 20 runs).",
        "plan_get",
        lambda m: {"plan_id": m["plan_id"]},
    ),
    ResourceRoute(
        "hpclab://workflows/{workflow_id}/grid",
        "workflow-grid",
        "Treatment × seed grid of a workflow with cell codes.",
        "runs_list",
        lambda m: {"workflow_id": m["workflow_id"], "view": "grid"},
    ),
    ResourceRoute(
        "hpclab://workflows/{workflow_id}",
        "workflow",
        "Workflow state, dispatch state and current version (for workflow_cancel / resume), gate, breaker evidence, per-cell states, budget.",
        "workflow_get",
        lambda m: {"workflow_id": m["workflow_id"]},
    ),
    ResourceRoute(
        "hpclab://runs/{run_id}",
        "run",
        "Same as run_inspect(run_id) without refresh.",
        "run_inspect",
        lambda m: {"run_id": m["run_id"]},
    ),
    ResourceRoute(
        "hpclab://artifacts/{artifact_id}",
        "artifact",
        "Artifact metadata: path, size, digest, integrity and producing attempt.",
        "artifact_get",
        lambda m: {"artifact_id": m["artifact_id"]},
    ),
    ResourceRoute(
        "hpclab://evaluations/{evaluation_id}",
        "evaluation",
        "Evaluation: metrics, evaluator and config, input digests, validation, provenance.",
        "evaluation_get",
        lambda m: {"evaluation_id": m["evaluation_id"]},
    ),
    ResourceRoute(
        "hpclab://comparisons/{comparison_id}",
        "comparison",
        "Archived comparison: planned run set, coverage, statistics, differences, contract, integrity.",
        "comparison_get",
        lambda m: {"comparison_id": m["comparison_id"]},
    ),
    ResourceRoute(
        "hpclab://evidence/{bundle_id}/{file}",
        "evidence-file",
        "One evidence file: summary.md, table.tex, table.csv or manifest.json.",
        "evidence_get",
        lambda m: {"bundle_id": m["bundle_id"], "file": m["file"], "max_bytes": 65536},
        "text/plain",
        "content",
    ),
    ResourceRoute(
        "hpclab://evidence/{bundle_id}",
        "evidence",
        "Evidence bundle metadata, coverage and file list.",
        "evidence_get",
        lambda m: {"bundle_id": m["bundle_id"]},
    ),
    ResourceRoute(
        "hpclab://trace/{cell_ref}",
        "trace",
        "Full provenance chain of one table cell (<bundle_id>:<cell_id>, as in the \\hpcval tags): per-seed value → "
        "evaluation → artifact digest → attempt → Slurm job → run spec → code → environment.",
        "trace",
        lambda m: {"cell_ref": m["cell_ref"]},
        greedy="cell_ref",
    ),
]

_MIME_BY_SUFFIX = {
    ".md": "text/markdown",
    ".tex": "application/x-tex",
    ".csv": "text/csv",
    ".json": "application/json",
}


def match_resource(uri: str) -> tuple[ResourceRoute, dict[str, str]] | None:
    for route in RESOURCES:
        m = route.regex.match(uri)
        if m:
            return route, m.groupdict()
    return None


def _resource_error(uri: str, env: dict[str, Any]) -> MCPError:
    err = env.get("error") or {}
    code = str(err.get("code") or "error")
    rpc = _INVALID_PARAMS if code in ("not_found", "invalid_argument") else _INTERNAL_ERROR
    return MCPError(rpc, f"{uri}: {_render_error(None, env)}", data={"hpclab_error": err})


def _truncation_note(route: ResourceRoute, data: dict[str, Any]) -> str:
    """A visible marker when a text resource is only the first page of the file."""
    if not data.get("truncated"):
        return ""
    size = data.get("size") or data.get("script_bytes")
    if route.mime_type == "text/x-shellscript":
        prefix = "# "
    elif str(data.get("file") or "").endswith(".tex"):
        prefix = "% "
    else:
        prefix = ""
    return f"\n{prefix}[hpclab: truncated — this is the first part of {size} bytes]\n"


# ---------------------------------------------------------------------------
# Prompts (skills for clients without skill support)
# ---------------------------------------------------------------------------

PROMPTS: dict[str, tuple[str, str, list[types.PromptArgument]]] = {
    "plan_experiment": (
        "hpclab-experiment-design",
        "Design and launch an experiment (baseline, treatments × seeds, metrics, smoke gate, budget).",
        [
            types.PromptArgument(
                name="goal", description="Research question or what to compare", required=True
            )
        ],
    ),
    "diagnose_run": (
        "hpclab-run-doctor",
        "Triage a failed, pending, stuck or uncertain run and plan the smallest correct recovery.",
        [
            types.PromptArgument(
                name="run_id",
                description="Run id (run_...), or an experiment id (exp_...) to triage all its runs",
                required=True,
            )
        ],
    ),
    "write_results": (
        "hpclab-paper-evidence",
        "Collect, compare and export a traceable results table and paragraph; verify a manuscript's numbers.",
        [
            types.PromptArgument(
                name="experiment_id", description="Experiment to report (exp_...)", required=True
            ),
            types.PromptArgument(
                name="tex_path", description="Manuscript to check, e.g. paper/main.tex", required=False
            ),
        ],
    ),
    "onboard_cluster": (
        "hpclab-cluster-onboarding",
        "Set up and verify a cluster for HPC Lab.",
        [types.PromptArgument(name="cluster_id", description="Cluster to set up, e.g. hai", required=False)],
    ),
    "lab_overview": (
        "hpclab-lab-overview",
        "Honest GPU usage and experiment progress overview for a lab.",
        [
            types.PromptArgument(name="account", description="Slurm account of the lab", required=False),
            types.PromptArgument(name="cluster_id", description="Cluster, e.g. hai", required=False),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _default_registry() -> Registry:
    from hpclab.services.app import build_registry

    return build_registry()


def _default_client() -> ClientLike:
    from hpclab.controller.client import HpcLabClient

    return HpcLabClient()


def create_server(
    *,
    client: ClientLike | None = None,
    client_factory: Callable[[], ClientLike] | None = None,
    registry: Registry | None = None,
    cwd: Path | None = None,
) -> Server[Any]:
    reg = registry or _default_registry()
    workdir = (cwd or Path.cwd()).resolve()
    state: dict[str, Any] = {"client": client}
    factory = client_factory or _default_client
    tools = build_tools(reg)

    def get_client() -> ClientLike:
        if state["client"] is None:
            state["client"] = factory()
        return state["client"]

    async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        c = get_client()
        return await anyio.to_thread.run_sync(partial(c.call, method, params))

    async def on_list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        spec = TOOLS_BY_NAME.get(params.name)
        if spec is None or spec.method not in reg.methods:
            return error_result(
                f"unknown tool {params.name!r}", "invalid_argument", known_tools=sorted(TOOLS_BY_NAME)
            )
        args = normalize_arguments(spec.name, dict(params.arguments or {}), workdir)
        try:
            env = await call(spec.method, args)
        except HpcLabError as exc:
            return error_result(exc.message, str(exc.code), **exc.details)
        except Exception as exc:  # never crash the MCP session
            log.exception("tool %s failed in the MCP frontend", spec.name)
            return error_result(f"MCP frontend error: {type(exc).__name__}: {exc}", "internal")
        return to_call_result(env, spec.name, args)

    async def on_list_resource_templates(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(
                    name=r.name, uri_template=r.template, description=r.description, mime_type=r.mime_type
                )
                for r in RESOURCES
                if r.method in reg.methods and "{" in r.template
            ]
        )

    async def on_list_resources(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListResourcesResult:
        """The experiments index plus the most recent experiments, so an agent in a fresh session
        (or a user @-mentioning) can find experiment ids. Never fails: an unreachable controller
        just yields the static entry."""
        out = [
            types.Resource(
                uri="hpclab://experiments",
                name="experiments",
                description="Experiments in the ledger, newest first.",
                mime_type="application/json",
            )
        ]
        if "experiments_list" not in reg.methods:
            return types.ListResourcesResult(resources=out)
        try:
            env = await call("experiments_list", {"limit": 20})
        except Exception:
            log.info("listing experiments for resources/list failed", exc_info=True)
            return types.ListResourcesResult(resources=out)
        for e in ((env.get("data") or {}).get("experiments") or []) if env.get("ok") else []:
            desc = f"{e.get('status')} on {e.get('cluster_id')}: {e.get('run_count')} run(s)"
            if e.get("question"):
                desc += f" — {str(e['question'])[:160]}"
            out.append(
                types.Resource(
                    uri=f"hpclab://experiments/{e.get('experiment_id')}",
                    name=f"experiment {e.get('name')}",
                    description=desc,
                    mime_type="application/json",
                )
            )
        return types.ListResourcesResult(resources=out)

    async def on_read_resource(
        ctx: ServerRequestContext[Any], params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        matched = match_resource(uri)
        if matched is None or matched[0].method not in reg.methods:
            known = ", ".join(r.template for r in RESOURCES if r.method in reg.methods)
            raise MCPError(_INVALID_PARAMS, f"unknown hpclab resource {uri!r}; known: {known}")
        route, groups = matched
        try:
            env = await call(route.method, route.params(groups))
        except HpcLabError as exc:
            raise MCPError(_INTERNAL_ERROR, f"{uri}: {exc.code}: {exc.message}", data=exc.details) from exc
        if not env.get("ok"):
            raise _resource_error(uri, env)
        mime = route.mime_type
        data = env.get("data")
        if route.text_field and isinstance(data, dict):
            text = str(data.get(route.text_field) or "") + _truncation_note(route, data)
            if route.name == "evidence-file":
                mime = _MIME_BY_SUFFIX.get(Path(groups.get("file", "")).suffix, "text/plain")
        else:
            text = json.dumps(env, ensure_ascii=False, separators=(",", ":"), default=str)
            mime = "application/json"
        return types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=uri, mime_type=mime, text=text)]
        )

    async def on_list_prompts(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[types.Prompt(name=n, description=d, arguments=a) for n, (_, d, a) in PROMPTS.items()]
        )

    async def on_get_prompt(
        ctx: ServerRequestContext[Any], params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        entry = PROMPTS.get(params.name)
        if entry is None:
            raise MCPError(_INVALID_PARAMS, f"unknown prompt {params.name!r}; known: {', '.join(PROMPTS)}")
        skill, desc, arg_specs = entry
        body = skill_text(skill) or f"(skill {skill} is not installed)"
        args = params.arguments or {}
        missing = [a.name for a in arg_specs if a.required and not args.get(a.name)]
        if missing:
            raise MCPError(_INVALID_PARAMS, f"prompt {params.name!r} needs: {', '.join(missing)}")
        header = "\n".join(f"{k}: {v}" for k, v in args.items() if v)
        text = f"{header}\n\nFollow this playbook:\n\n{body}" if header else body
        return types.GetPromptResult(
            description=desc,
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))],
        )

    return Server(
        "hpclab",
        version=__version__,
        title="HPC Lab",
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_list_resource_templates=on_list_resource_templates,
        on_read_resource=on_read_resource,
        on_list_prompts=on_list_prompts,
        on_get_prompt=on_get_prompt,
    )


async def serve_stdio(server: Server[Any] | None = None) -> None:
    srv = server or create_server()
    async with stdio_server() as (read_stream, write_stream):
        await srv.run(read_stream, write_stream, srv.create_initialization_options())


def main() -> None:
    # stdout is the protocol channel: all diagnostics go to stderr.
    logging.basicConfig(level=os.environ.get("HPCLAB_MCP_LOG_LEVEL", "WARNING"), stream=sys.stderr)
    anyio.run(serve_stdio)
