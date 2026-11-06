"""Rich renderers for CLI commands.

Every function takes the ``data`` dict of an Envelope (exactly the service's Data model dumped as
JSON) and returns a rich renderable; commands print it. Rules that keep the output honest:

* unknown is shown as ``unknown`` / ``?`` — never as 0, blank or "ok";
* user- and cluster-supplied strings go through ``Text`` (never parsed as rich markup);
* next steps are exact CLI commands the user can paste.

Nothing here talks to the controller or prints to stdout directly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Styles and small formatters
# ---------------------------------------------------------------------------

CODE_STYLE: dict[str, str] = {
    "V": "bold green",
    "R": "bold cyan",
    "Q": "yellow",
    "S": "magenta",
    "P": "dim",
    "E": "bold blue",
    "F": "bold red",
    "C!": "bold dark_orange",
    "U": "bold magenta",
    "X": "red",
    "?": "bold bright_black",
    "-": "dim",
}
CODE_LEGEND: dict[str, str] = {
    "V": "valid result",
    "R": "running",
    "Q": "queued",
    "S": "submitting",
    "P": "not dispatched",
    "E": "completed, evaluation pending — not a failure",
    "F": "failed",
    "C!": "completed, result not valid",
    "U": "submission uncertain / duplicate",
    "X": "cancelled / rejected / abandoned",
    "?": "state unknown now (observation too old)",
    "-": "no planned run",
}
# E (COMPLETED, evaluation pending) is none of these: it is expected to turn V on its own.
TERMINAL_CODES = frozenset({"V", "F", "C!", "X", "-"})
PROBLEM_CODES = frozenset({"F", "C!", "U", "X", "?"})
ACTIVE_CODES = frozenset({"R", "Q", "S", "U", "?"})

STATUS_STYLE: dict[str, str] = {
    "pass": "green",
    "ok": "green",
    "warn": "yellow",
    "fail": "bold red",
    "unchecked": "cyan",
    "pending": "cyan",
    "unavailable": "magenta",
    "unsupported": "magenta",
    "skipped": "dim",
    "partial": "yellow",
}
SEVERITY_STYLE: dict[str, str] = {
    "block": "bold red",
    "error": "bold red",
    "warn": "yellow",
    "warning": "yellow",
    "info": "dim",
}
_OBSERVED_PHASES = frozenset({"job_identified", "observing", "terminal"})
ENDED_PHASES = frozenset({"terminal", "submission_rejected", "cancelled_before_submit", "abandoned"})
RISK_STYLE = {"none": "green", "low": "green", "medium": "yellow", "high": "bold red", "unknown": "dim"}


def T(value: Any, style: str = "") -> Text:
    """A literal (never markup-parsed) text cell."""
    return Text("" if value is None else str(value), style=style)


def code_text(code: str | None, extra: str = "") -> Text:
    c = code or "?"
    out = Text(c, style=CODE_STYLE.get(c, ""))
    if extra:
        out.append(" " + extra)
    return out


def status_text(status: str | None) -> Text:
    s = status or "?"
    return Text(s, style=STATUS_STYLE.get(s, ""))


def severity_text(sev: str | None) -> Text:
    s = sev or "?"
    return Text(s, style=SEVERITY_STYLE.get(s, ""))


def gpuh(value: Any, *, unit: bool = True) -> str:
    """GPU-hours; ``None`` is *unknown*, never 0."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "unknown"
    if 0 < value < 0.005:
        s = "<0.01"
    else:
        s = f"{value:,.2f}"
    return f"{s} GPU-h" if unit else s


def dur(seconds: Any) -> str:
    """Compact duration: 45s, 12m05s, 3h07m, 2d04h. ``None`` → '?'."""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds):
        return "?"
    s = round(seconds)
    sign = "-" if s < 0 else ""
    s = abs(s)
    if s < 60:
        return f"{sign}{s}s"
    if s < 3600:
        return f"{sign}{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{sign}{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{sign}{s // 86400}d{(s % 86400) // 3600:02d}h"


def num(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) >= 1e6 or abs(value) < 10 ** (-decimals)):
        return f"{value:.{decimals}g}"
    return f"{value:.{decimals}f}"


def signed(value: Any, decimals: int = 4) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    s = num(value, decimals)
    return s if s.startswith("-") else "+" + s


def short_digest(d: Any, n: int = 12) -> str:
    if not d:
        return "?"
    s = str(d)
    algo, sep, hexpart = s.partition(":")
    return f"{algo}:{hexpart[:n]}" if sep else s[:n]


def size(n: Any) -> str:
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return "?"
    v = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or u == "TiB":
            return f"{int(v)} {u}" if u == "B" else f"{v:.1f} {u}"
        v /= 1024
    return f"{v:.1f} TiB"


def clip(value: Any, n: int = 80) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    s = " ".join(s.split()) if "\n" in s else s
    return s if len(s) <= n else s[: n - 1] + "…"


def cell_label(cell: Mapping[str, Any] | None, *, skip: Iterable[str] = ("seed",)) -> str:
    if not cell:
        return "(single cell)"
    skipped = set(skip)
    parts = [f"{k}={v}" for k, v in cell.items() if k not in skipped]
    return ", ".join(parts) or "(single cell)"


def kv(label: str, value: Any, style: str = "") -> Text:
    out = Text(f"{label}: ", style="bold")
    out.append_text(value if isinstance(value, Text) else Text("" if value is None else str(value), style))
    return out


def next_steps(commands: Sequence[str], title: str = "next") -> RenderableType:
    """Pasteable commands (deduplicated, order kept)."""
    seen: list[str] = []
    for c in commands:
        if c and c not in seen:
            seen.append(c)
    if not seen:
        return Text("")
    lines: list[RenderableType] = [Text(f"{title}:", style="bold")]
    lines.extend(Text("  " + c, style="cyan") for c in seen)
    return Group(*lines)


_ID_COLUMNS = frozenset({"run", "attempt", "evaluation", "entity", "file"})


def _is_id_column(name: str) -> bool:
    return name.endswith(" id") or name in _ID_COLUMNS


# hpclab ids are ~30 characters (+ " vN"); a longer id-like value (a sha256 entity, a path) is
# cut with an ellipsis rather than squeezing every other column to nothing on a narrow terminal.
_ID_MAX_WIDTH = 40


def table(*columns: str, title: str | None = None, **kwargs: Any) -> Table:
    """A table whose id-like columns never wrap (ids must stay copyable)."""
    t = Table(title=title, header_style="bold", title_justify="left", title_style="bold", **kwargs)
    for c in columns:
        if _is_id_column(c):
            t.add_column(c, no_wrap=True, overflow="ellipsis", max_width=_ID_MAX_WIDTH)
        else:
            t.add_column(c, overflow="fold")
    return t


def _blank(cell: Any) -> bool:
    text = cell.plain if isinstance(cell, Text) else str(cell)
    return text.strip() in ("", "-")


def dyn_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    keep: Iterable[str] = (),
    title: str | None = None,
) -> Table:
    """``table`` without the columns that are empty in every row (except ``keep``)."""
    keep_set = set(keep)
    idx = [i for i, c in enumerate(columns) if c in keep_set or any(not _blank(r[i]) for r in rows)]
    t = table(*[columns[i] for i in idx], title=title)
    for r in rows:
        t.add_row(*[r[i] if isinstance(r[i], Text) else T(r[i]) for i in idx])
    return t


def _g(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


# ---------------------------------------------------------------------------
# prepare / plans / experiments
# ---------------------------------------------------------------------------

_FINDING_ORDER = {"fail": 0, "warn": 1, "unchecked": 2, "pass": 3}
_SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}


def findings_table(
    findings: Sequence[Mapping[str, Any]], *, show_pass: bool = False, important_only: bool = False
) -> RenderableType:
    """Validation findings, worst first; passes are counted, not listed (unless ``show_pass``).
    ``important_only`` keeps failures and block/warn severities (info-level notes are counted)."""
    if important_only:
        hidden = sum(
            1
            for f in findings
            if f.get("status") != "pass"
            and not (f.get("status") == "fail" or f.get("severity") in ("block", "warn"))
        )
        findings = [
            f
            for f in findings
            if f.get("status") == "fail"
            or f.get("severity") in ("block", "warn")
            or f.get("status") == "pass"
        ]
        if hidden:
            return Group(
                findings_table(findings),
                Text(f"{hidden} informational finding(s) not shown", style="dim"),
            )
    items = sorted(
        (f for f in findings if show_pass or f.get("status") != "pass"),
        key=lambda f: (
            _FINDING_ORDER.get(str(f.get("status")), 9),
            _SEVERITY_ORDER.get(str(f.get("severity")), 9),
        ),
    )
    passed = sum(1 for f in findings if f.get("status") == "pass")
    if not items:
        return Text(f"all {passed} checks passed", style="green")
    t = table("status", "severity", "check", "finding", title="validation findings")
    for f in items:
        msg = Text(str(f.get("message") or ""))
        if f.get("remediation"):
            msg.append("\n→ " + str(f["remediation"]), style="yellow")
        if f.get("override_reason"):
            msg.append(f"\noverridden: {f['override_reason']}", style="magenta")
        t.add_row(status_text(f.get("status")), severity_text(f.get("severity")), T(f.get("check")), msg)
    parts: list[RenderableType] = [t]
    if not show_pass and passed:
        parts.append(Text(f"{passed} other check(s) passed (--all-findings to list them)", style="dim"))
    return Group(*parts)


def prepared_runs_table(
    runs: Sequence[Mapping[str, Any]], total: int | None = None, limit: int = 40
) -> RenderableType:
    t = table("label", "kind", "seed", "GPU-h", "run id", title="runs")
    for r in runs[:limit]:
        flags = []
        if r.get("added_post_hoc"):
            flags.append("post-hoc")
        if r.get("derived_from"):
            flags.append(f"derived from {r['derived_from']}")
        label = T(r.get("label"))
        if flags:
            label.append(f" ({', '.join(flags)})", style="yellow")
        t.add_row(
            label,
            T(r.get("kind")),
            T(r.get("seed") if r.get("seed") is not None else "-"),
            T(gpuh(r.get("requested_gpu_hours"), unit=False)),
            T(r.get("run_id") or "(not created yet)", "dim"),
        )
    out: list[RenderableType] = [t]
    shown = min(len(runs), limit)
    n = total if total is not None else len(runs)
    if n > shown:
        out.append(Text(f"… {n - shown} more run(s) not shown", style="dim"))
    return Group(*out)


def budget_line(budget: Mapping[str, Any] | None) -> Text:
    if not budget:
        return kv("budget", "unknown")
    req = budget.get("requested_gpu_hours_total")
    smoke = budget.get("smoke_gpu_hours")
    cap = budget.get("budget_gpu_hours")
    existing = budget.get("existing_planned_gpu_hours")
    line = Text("budget: ", style="bold")
    line.append(f"{gpuh(req)} requested (time limit × GPUs")
    if isinstance(smoke, (int, float)) and smoke:
        line.append(f", incl. smoke {gpuh(smoke, unit=False)}")
    line.append(")")
    if isinstance(existing, (int, float)) and existing:
        line.append(f" + {gpuh(existing)} already planned")
    if cap is None:
        line.append(" · no hpclab budget set (site quotas still apply)", style="dim")
    else:
        total = (req or 0) + (existing or 0)
        over = isinstance(req, (int, float)) and total > float(cap) + 1e-9
        line.append(f" of {gpuh(cap)} budget", style="bold red" if over else "")
        if over:
            line.append(" — OVER BUDGET", style="bold red")
    return line


def render_prepare(data: Mapping[str, Any], *, show_pass: bool = False) -> RenderableType:
    status = str(data.get("status") or "?")
    st_style = {"ready": "bold green", "preparing": "bold cyan", "invalid": "bold red"}.get(status, "")
    head = Text("plan ", style="bold")
    head.append(str(data.get("plan_id")), style="bold")
    head.append(" ")
    head.append(status, style=st_style)
    head.append(f" ({data.get('kind')}) · experiment {data.get('experiment_id')}")
    parts: list[RenderableType] = [head]
    val = data.get("validation") or {}
    counts = val.get("counts") or {}
    vline = Text("validation: ", style="bold")
    vline.append_text(status_text(val.get("status")))
    if counts:
        vline.append(
            " — " + ", ".join(f"{counts.get(k, 0)} {k}" for k in ("pass", "warn", "unchecked", "fail"))
        )
    blocking = val.get("blocking") or []
    if blocking:
        vline.append(f" · blocking: {', '.join(blocking)}", style="bold red")
    if val.get("overridden"):
        vline.append(f" · overridden: {', '.join(val['overridden'])}", style="magenta")
    parts.append(vline)
    parts.append(findings_table(val.get("findings") or [], show_pass=show_pass))
    for o in data.get("overrides") or []:
        style = "magenta" if o.get("applied") else "yellow"
        parts.append(
            T(
                f"override {o.get('check')}: {'applied' if o.get('applied') else 'NOT applied'}"
                f" — {o.get('reason')}" + (f" ({o['note']})" if o.get("note") else ""),
                style,
            )
        )
    advice = data.get("resource_advice") or {}
    for item in advice.get("items") or []:
        if item.get("verdict") in ("under_request", "over_request"):
            parts.append(
                T(
                    f"resource advice ({', '.join(item.get('labels') or [])[:80]}): {item['verdict']} — "
                    f"requested {item.get('requested_time')}, p90 elapsed {dur(item.get('p90_elapsed_s'))}, "
                    f"suggested {item.get('suggested_time') or '?'}"
                    + (f", mem {item['suggested_mem']}" if item.get("suggested_mem") else "")
                    + f" (n={item.get('n')})",
                    "yellow",
                )
            )
    parts.append(prepared_runs_table(data.get("runs") or [], total=data.get("runs_total")))
    parts.append(budget_line(data.get("budget")))
    snap = data.get("snapshot") or {}
    if snap:
        ref = snap.get("ref") or {}
        sline = Text("code snapshot: ", style="bold")
        if ref:
            sline.append(
                f"{short_digest(ref.get('digest'))} · {ref.get('file_count')} files, {size(ref.get('total_bytes'))}"
            )
            if ref.get("git_commit"):
                sline.append(
                    f" · git {str(ref['git_commit'])[:12]}{' (dirty)' if ref.get('git_dirty') else ''}"
                )
            else:
                sline.append(" · no git commit recorded", style="yellow")
        else:
            sline.append(f"{snap.get('mode')} (in progress)")
        if snap.get("excluded_count"):
            sline.append(f" · {snap['excluded_count']} file(s) excluded")
        parts.append(sline)
    if data.get("smoke_run_id"):
        parts.append(
            kv("smoke gate", f"{data['smoke_run_id']} runs first; the matrix is dispatched only if it passes")
        )
    return Group(*parts)


def prepare_next(data: Mapping[str, Any]) -> list[str]:
    status = data.get("status")
    pid = data.get("plan_id")
    if status == "ready":
        return [f"hpclab submit {pid}", f"hpclab plan {pid} --script    # review the exact sbatch script"]
    if status == "preparing":
        op = data.get("operation_id")
        return [f"hpclab ops {op} --wait" if op else f"hpclab plan {pid}", f"hpclab submit {pid}"]
    blocking = (data.get("validation") or {}).get("blocking") or []
    cmds = ["fix the blocking finding(s) above and run `hpclab prepare` again"]
    if blocking:
        flags = " ".join(f"--override {b}='<reason>'" for b in blocking)
        cmds.append(f"or, only if you accept the risk (recorded): hpclab prepare ... {flags}")
    return cmds


def render_plan(data: Mapping[str, Any], *, show_pass: bool = False) -> RenderableType:
    head = Text(f"plan {data.get('plan_id')} ", style="bold")
    head.append_text(status_text(data.get("status")))
    head.append(f" ({data.get('kind')}) · experiment {data.get('experiment_id')}")
    parts: list[RenderableType] = [head]
    if data.get("plan_hash"):
        parts.append(kv("plan hash", data["plan_hash"]))
    if data.get("snapshot_digest"):
        parts.append(kv("code snapshot", data["snapshot_digest"]))
    val = data.get("validation") or {}
    parts.append(
        kv(
            "validation",
            f"{val.get('status')} — "
            + ", ".join(
                f"{(val.get('counts') or {}).get(k, 0)} {k}" for k in ("pass", "warn", "unchecked", "fail")
            )
            + (f" · blocking: {', '.join(val['blocking'])}" if val.get("blocking") else ""),
        )
    )
    if val.get("findings"):
        parts.append(findings_table(val["findings"], show_pass=show_pass))
    cells = recovery_cells(data)
    if cells and not data.get("runs"):
        # a recovery plan freezes cells (existing runs + new execution plans), not new runs
        parts.append(recovery_cells_table(cells))
        parts.append(recovery_cost_line(cells))
        return Group(*parts)
    t = table("label", "kind", "seed", "status", "GPU-h", "resources", "run id", title="runs")
    for r in data.get("runs") or []:
        res = _g(r, "execution_plan", "resources", default={}) or {}
        res_s = (
            f"{res.get('gpus', '?')}×{res.get('gpu_type') or 'gpu'} {res.get('time', '?')} {res.get('partition') or ''}".strip()
            if res
            else "?"
        )
        t.add_row(
            T(r.get("label")),
            T(r.get("kind")),
            T(r.get("seed") if r.get("seed") is not None else "-"),
            T(r.get("status")),
            T(gpuh(r.get("requested_gpu_hours"), unit=False)),
            T(res_s),
            T(r.get("run_id"), "dim"),
        )
    parts.append(t)
    shown = len(data.get("runs") or [])
    total = data.get("runs_total")
    if isinstance(total, int) and total > shown:
        parts.append(T(f"… {total - shown} more run(s): --cursor for the next page", "dim"))
    parts.append(budget_line(data.get("budget")))
    return Group(*parts)


def recovery_cells(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The cells of a recovery plan (``plan_get.details.cells``); [] for other plans."""
    cells = _g(plan, "details", "cells", default=[]) or []
    return [c for c in cells if isinstance(c, Mapping)]


def recovery_totals(cells: Sequence[Mapping[str, Any]]) -> tuple[int, int, float | None]:
    """(eligible cells, ineligible cells, GPU-hours of the eligible ones — None if any is unknown)."""
    eligible = [c for c in cells if c.get("eligible")]
    hours = [c.get("requested_gpu_hours") for c in eligible]
    known = [h for h in hours if isinstance(h, (int, float)) and not isinstance(h, bool)]
    total = float(sum(known)) if len(known) == len(hours) else None
    return len(eligible), len(cells) - len(eligible), total


def recovery_cells_table(cells: Sequence[Mapping[str, Any]]) -> Table:
    t = table("label", "kind", "eligible", "strategy", "GPU-h", "diagnosis", "from attempt", title="cells")
    for c in cells:
        t.add_row(
            T(c.get("label")),
            T(c.get("kind")),
            T("yes" if c.get("eligible") else "no", "green" if c.get("eligible") else "red"),
            T(c.get("strategy") or "-"),
            T(gpuh(c.get("requested_gpu_hours"), unit=False)),
            T(c.get("diagnosis_code") or "-"),
            T(c.get("from_attempt_id") or "-", "dim"),
        )
    return t


def recovery_cost_line(cells: Sequence[Mapping[str, Any]]) -> Text:
    n, blocked, total = recovery_totals(cells)
    line = kv("GPU-hours", f"{gpuh(total)} at most for the {n} cell(s) to dispatch (time limit × GPUs)")
    if blocked:
        line.append(f" · {blocked} cell(s) not eligible (not dispatched)", style="yellow")
    return line


def render_experiments(data: Mapping[str, Any]) -> RenderableType:
    items = data.get("experiments") or []
    if not items:
        return Text("no experiments yet — prepare one: hpclab prepare hpclab.yaml", style="dim")
    t = table("name", "experiment id", "cluster", "status", "plans", "runs", "budget", "created")
    for e in items:
        t.add_row(
            T(e.get("name"), "bold"),
            T(e.get("experiment_id")),
            T(e.get("cluster_id")),
            T(e.get("status")),
            T(e.get("plan_count")),
            T(e.get("run_count")),
            T(gpuh(e.get("budget_gpu_hours")) if e.get("budget_gpu_hours") is not None else "-"),
            T(str(e.get("created_at") or "")[:19].replace("T", " ")),
        )
    return t


def counts_line(counts: Mapping[str, Any] | None, *, label: str = "cells") -> Text:
    c = dict(counts or {})
    planned = c.pop("planned", None)
    valid = c.pop("valid", None)
    out = Text(f"{label}: ", style="bold")
    if planned is not None:
        out.append(f"{valid if valid is not None else c.get('V', 0)}/{planned} valid", style="bold")
    first = planned is None
    for code in ("V", "R", "Q", "S", "P", "E", "U", "?", "F", "C!", "X"):
        n = c.pop(code, 0)
        if n:
            out.append("" if first else " · ")
            first = False
            out.append(code, style=CODE_STYLE.get(code, ""))
            out.append(f"={n}")
    for k, v in c.items():
        if v:
            out.append(f" · {k}={v}")
    return out


def render_experiment(data: Mapping[str, Any]) -> RenderableType:
    head = Text(f"experiment {data.get('name')} ", style="bold")
    head.append(f"({data.get('experiment_id')}) · {data.get('status')} · cluster {data.get('cluster_id')}")
    parts: list[RenderableType] = [head]
    if data.get("question"):
        parts.append(kv("question", data["question"]))
    if data.get("parent_experiment_id"):
        parts.append(kv("extends", data["parent_experiment_id"]))
    parts.append(counts_line(data.get("run_counts"), label="runs"))
    b = data.get("budget") or {}
    if b:
        line = Text("budget: ", style="bold")
        if b.get("budget") is None:
            line.append("no hpclab budget set")
        else:
            line.append(f"{gpuh(b.get('available'))} available of {gpuh(b.get('budget'))}")
        line.append(
            f" · reserved {gpuh(b.get('reserved'), unit=False)} · settled {gpuh(b.get('settled_consumed'), unit=False)}"
        )
        if b.get("unknown_count"):
            line.append(
                f" · {b['unknown_count']} attempt(s) with unknown consumption ({gpuh(b.get('unknown_reserved'))} kept reserved)",
                style="yellow",
            )
        parts.append(line)
    parts.append(kv("planned", f"{gpuh(data.get('planned_gpu_hours'))} over {data.get('runs_total')} run(s)"))
    t = table("plan id", "kind", "status", "runs", "validation", "created", title="plans")
    for p in data.get("plans") or []:
        t.add_row(
            T(p.get("plan_id")),
            T(p.get("kind")),
            status_text(p.get("status")),
            T(p.get("run_count")),
            status_text(p.get("validation_status")),
            T(str(p.get("created_at") or "")[:19].replace("T", " ")),
        )
    parts.append(t)
    eid = data.get("experiment_id")
    parts.append(next_steps([f"hpclab grid {eid}", f"hpclab compare {eid} --metric <name>"]))
    return Group(*parts)


# ---------------------------------------------------------------------------
# runs / grid / workflows
# ---------------------------------------------------------------------------


def render_runs(data: Mapping[str, Any]) -> RenderableType:
    rows = data.get("runs") or []
    if not rows:
        return Text("no runs match", style="dim")
    cols = ["label", "code", "job", "state", "left", "progress", "health", "reason", "run id"]
    body: list[list[Any]] = []
    for r in rows:
        la = r.get("latest_attempt") or {}
        state = la.get("scheduler_state") or ""
        if la.get("phase") and la.get("phase") not in ("observing", "terminal", "job_identified"):
            state = la["phase"]
        if la.get("observation_gap"):
            state += " (gap)"
        if la.get("cancel_state") and la.get("cancel_state") != "none":
            state += f" cancel:{la['cancel_state']}"
        reason = r.get("reason") or la.get("pending_reason") or ""
        code = code_text(r.get("code"))
        if r.get("stale"):
            code.append(" stale", style="bright_black")
        job = str(la.get("job_id") or "-")
        if (r.get("attempts") or 0) > 1:
            job += f" (#{la.get('seq') or r.get('attempts')})"
        body.append(
            [
                T(r.get("label")),
                code,
                T(job),
                T(state or "-"),
                T(dur(la.get("time_left_s")) if la.get("time_left_s") is not None else "-"),
                T(la.get("progress") or "-"),
                T(la.get("health") or "-"),
                T(clip(reason, 60), "dim"),
                T(r.get("run_id"), "dim"),
            ]
        )
    t = dyn_table(cols, body, keep=("label", "code", "run id"))
    parts: list[RenderableType] = [t, counts_line(data.get("counts"), label="runs")]
    parts.extend(foreign_lines(data.get("foreign_jobs") or []))
    return Group(*parts)


def foreign_lines(foreign: Sequence[Mapping[str, Any]]) -> list[Text]:
    out = []
    for f in foreign:
        if f.get("count"):
            out.append(
                T(
                    f"also on {f.get('cluster_id')}: {f['count']} job(s) of yours not managed by hpclab "
                    f"({f.get('running', 0)} running, {f.get('pending', 0)} pending, {f.get('gpus', 0)} GPU(s))"
                    + (" [stale]" if f.get("stale") else ""),
                    "dim",
                )
            )
        if f.get("unbound_hpclab_jobs"):
            out.append(
                T(
                    f"{f['unbound_hpclab_jobs']} job(s) with an hpclab name are not bound to any attempt on "
                    f"{f.get('cluster_id')} (reconciliation pending or duplicates) — see `hpclab status`",
                    "yellow",
                )
            )
    return out


def workflow_line(wf: Mapping[str, Any]) -> Text:
    state = str(wf.get("state") or "?")
    style = {
        "completed": "green",
        "gate_failed": "bold red",
        "paused": "bold yellow",
        "cancelled": "red",
    }.get(state, "cyan")
    line = Text("workflow ", style="bold")
    line.append(f"{wf.get('workflow_id')} ")
    line.append(state, style=style)
    line.append(f" · dispatch {wf.get('dispatch_state')}")
    gate = wf.get("gate") or {}
    if gate:
        gstate = str(gate.get("state"))
        line.append(" · gate ")
        line.append(gstate, style={"passed": "green", "failed": "bold red"}.get(gstate, "cyan"))
        if gate.get("reason"):
            line.append(f" ({gate['reason']})")
    if wf.get("max_parallel") is not None and wf.get("slots_used") is not None:
        line.append(f" · slots {wf.get('slots_used')}/{wf.get('max_parallel')}")
    elif wf.get("max_parallel") is not None:
        line.append(f" · at most {wf.get('max_parallel')} at once")
    if wf.get("dispatch_blocked_reason"):
        line.append(f" · blocked: {wf['dispatch_blocked_reason']}", style="yellow")
    return line


def render_grid(
    data: Mapping[str, Any],
    *,
    workflow: Mapping[str, Any] | None = None,
    title: str | None = None,
    max_problems: int = 12,
    target: str | None = None,
) -> RenderableType:
    """The grid; ``target`` (experiment or workflow id) makes the collect hint pasteable."""
    grid = data.get("grid") or {}
    parts: list[RenderableType] = []
    if title:
        parts.append(Text(title, style="bold"))
    if workflow:
        parts.append(workflow_line(workflow))
        breaker = workflow.get("breaker") or {}
        if breaker and breaker.get("state") not in (None, "closed", "cleared"):
            parts.append(
                T(
                    f"circuit breaker {breaker.get('state')}: {clip(breaker.get('fingerprint') or {}, 100)} — "
                    f"{breaker.get('resume_hint') or 'fix the environment, then hpclab workflow resume ' + str(workflow.get('workflow_id'))}",
                    "bold yellow",
                )
            )
    rows = grid.get("rows") or []
    metric = grid.get("metric")
    if not rows:
        parts.append(Text("no planned runs in this view", style="dim"))
        return Group(*parts)
    t = Table(header_style="bold", show_lines=False)
    row_axes = grid.get("row_axes") or []
    t.add_column(" / ".join(row_axes) or "cell", style="bold", overflow="fold")
    for c in grid.get("cols") or []:
        t.add_column(str(c), overflow="fold", no_wrap=True)
    present: set[str] = set()
    problems: list[Text] = []
    for r in rows:
        # an axis a row does not have (e.g. `variant` of a non-derived cell) is None: shown as —
        label = ", ".join("—" if v is None else str(v) for v in (r.get("values") or {}).values()) or str(
            r.get("key") or ""
        )
        cells: list[Text] = []
        for idx, c in enumerate(r.get("cells") or []):
            code = str(c.get("code") or "?")
            present.add(code)
            extra = ""
            if (c.get("runs") or 0) > 1:
                extra = f"×{c['runs']}"
            if code == "V" and metric and c.get("value") is not None:
                extra = (extra + " " if extra else "") + num(c.get("value"))
            cells.append(code_text(code, extra))
            if code in PROBLEM_CODES:
                col = (grid.get("cols") or [])[idx] if idx < len(grid.get("cols") or []) else "?"
                p = Text("  ")
                p.append(code, style=CODE_STYLE.get(code, ""))
                why = c.get("reason") or CODE_LEGEND.get(code, "")
                p.append(f" {label} {col}: {clip(why, 100)}")
                if c.get("run_id"):
                    p.append(f"  ({c['run_id']})", style="dim")
                problems.append(p)
        t.add_row(T(label), *cells)
    parts.append(t)
    total_rows = grid.get("total_rows")
    if isinstance(total_rows, int) and total_rows > len(rows):
        parts.append(
            T(
                f"showing rows {grid.get('offset', 0) + 1}–{grid.get('offset', 0) + len(rows)} of {grid['total_rows']} (--cursor for more)",
                "dim",
            )
        )
    legend = Text("legend: ", style="dim")
    first = True
    for code in CODE_STYLE:
        if code in present:
            if not first:
                legend.append(" · ", style="dim")
            first = False
            legend.append(code, style=CODE_STYLE[code])
            legend.append(f" {CODE_LEGEND.get(code, '')}", style="dim")
    parts.append(legend)
    if metric:
        parts.append(T(f"values: {metric} from the selected valid evaluation only (V cells)", "dim"))
    parts.append(counts_line(data.get("counts")))
    if problems:
        parts.append(Text("needs attention:", style="bold"))
        parts.extend(problems[:max_problems])
        if len(problems) > max_problems:
            parts.append(T(f"  … {len(problems) - max_problems} more", "dim"))
    # rows are paginated: the service's counts cover every run, the cells only this page
    pending_eval = max(grid_codes(data).count("E"), int((data.get("counts") or {}).get("E") or 0))
    if pending_eval:
        ref = target or (workflow or {}).get("workflow_id") or "<experiment or workflow id>"
        parts.append(
            T(
                f"{pending_eval} cell(s) completed, evaluation pending (automatic, or: hpclab collect {ref})"
                " — not a failure",
                CODE_STYLE["E"],
            )
        )
    for note in grid.get("notes") or []:
        parts.append(T(note, "dim"))
    if "?" in present:
        parts.append(
            T(
                "? = the last observation is too old to trust (controller paused, cluster unreachable or auth "
                "needed): check `hpclab status`",
                "bold bright_black",
            )
        )
    parts.extend(foreign_lines(data.get("foreign_jobs") or []))
    return Group(*parts)


def grid_codes(data: Mapping[str, Any]) -> list[str]:
    """Every cell code of a grid view, row by row."""
    grid = data.get("grid") or {}
    return [str(c.get("code") or "?") for r in grid.get("rows") or [] for c in r.get("cells") or []]


def render_workflow(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = [workflow_line(data)]
    parts.append(
        kv(
            "plan",
            f"{data.get('plan_id')} ({data.get('plan_kind')}) · experiment {data.get('experiment_id')} · version {data.get('version')}",
        )
    )
    parts.append(counts_line(data.get("codes"), label="cells"))
    if data.get("counts"):
        parts.append(kv("cell states", ", ".join(f"{k}={v}" for k, v in data["counts"].items() if v)))
    b = data.get("budget") or {}
    if b:
        line = kv(
            "GPU-hours",
            f"reserved {gpuh(b.get('reserved_gpu_hours'), unit=False)} · settled {gpuh(b.get('settled_gpu_hours'), unit=False)}",
        )
        if b.get("unknown_reservations"):
            line.append(
                f" · {b['unknown_reservations']} reservation(s) with unknown consumption", style="yellow"
            )
        parts.append(line)
    breaker = data.get("breaker") or {}
    if breaker:
        parts.append(
            kv(
                "circuit breaker",
                f"{breaker.get('state')}"
                + (f" since {breaker.get('tripped_at')}" if breaker.get("tripped_at") else "")
                + (
                    f" · fingerprint {clip(breaker.get('fingerprint'), 120)}"
                    if breaker.get("fingerprint")
                    else ""
                ),
                "yellow" if breaker.get("state") not in ("closed", "cleared") else "",
            )
        )
        if breaker.get("resume_hint"):
            parts.append(T("  " + breaker["resume_hint"], "yellow"))
    cancel = data.get("cancel") or {}
    if cancel and cancel.get("scope"):
        parts.append(
            kv(
                "cancel",
                f"{cancel.get('scope')} · operation {cancel.get('operation_id')} {cancel.get('operation_state') or ''}",
            )
        )
    if data.get("operation_id"):
        parts.append(
            kv(
                "dispatch operation",
                f"{data.get('operation_id')} {data.get('operation_state') or ''} {data.get('operation_stage') or ''}".strip(),
            )
        )
    t = table("#", "label", "kind", "code", "cell state", "attempt", "job", "state", "note")
    for c in data.get("cells") or []:
        state = c.get("scheduler_state") or c.get("phase") or ""
        if c.get("result_state") and c.get("result_state") not in ("unchecked",):
            state += f" / {c['result_state']}"
        if c.get("cancel_state") and c.get("cancel_state") != "none":
            state += f" / cancel {c['cancel_state']}"
        t.add_row(
            T(c.get("ordinal")),
            T(c.get("label")),
            T(c.get("kind")),
            code_text(c.get("code")),
            T(c.get("cell_state")),
            T(c.get("attempt_id") or "-", "dim"),
            T(c.get("job_id") or "-"),
            T(state or "-"),
            T(clip(c.get("note") or c.get("reason") or "", 60), "dim"),
        )
    parts.append(t)
    return Group(*parts)


def render_workflow_cancel(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = [
        kv(
            "workflow",
            f"{data.get('workflow_id')} → {data.get('state')} · dispatch {data.get('dispatch_state')} · scope {data.get('scope')}",
        )
    ]
    if data.get("counts"):
        parts.append(kv("cells", ", ".join(f"{k}={v}" for k, v in data["counts"].items() if v)))
    if data.get("budget_kept_gpu_hours"):
        parts.append(kv("GPU-hours still reserved", gpuh(data["budget_kept_gpu_hours"]), "yellow"))
    t = table("label", "cell state", "attempt", "phase", "cancel", "note")
    for c in data.get("cells") or []:
        t.add_row(
            T(c.get("label")),
            T(c.get("cell_state")),
            T(c.get("attempt_id") or "-", "dim"),
            T(c.get("phase") or "-"),
            T(c.get("cancel_state") or "-"),
            T(clip(c.get("note") or "", 60), "dim"),
        )
    if data.get("cells"):
        parts.append(t)
    for n in data.get("notes") or []:
        parts.append(T(n, "dim"))
    return Group(*parts)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def progress_bar(fraction: float | None, width: int = 24) -> str:
    if fraction is None:
        return "[" + "?" * width + "]"
    f = min(1.0, max(0.0, float(fraction)))
    full = round(f * width)
    return "[" + "█" * full + "·" * (width - full) + f"] {f * 100:5.1f}%"


UNRESOLVED_PHASES = frozenset({"submission_uncertain", "duplicate_detected"})


def submission_unresolved(attempt: Mapping[str, Any]) -> bool:
    """True while the submission outcome is still open (reconciling or a duplicate to decide)."""
    return attempt.get("phase") in UNRESOLVED_PHASES


def uncertainty_line(attempt: Mapping[str, Any]) -> Text:
    """The submission-uncertainty note: a warning while unresolved, history once resolved.

    ``uncertain_since`` is kept after reconciliation (or a human resolution) for the audit
    trail, so the phase decides whether the uncertainty is still live.
    """
    since = attempt.get("uncertain_since")
    if submission_unresolved(attempt):
        return T(
            f"submission outcome uncertain since {since}: hpclab is reconciling and will never resubmit on its own",
            "bold magenta",
        )
    job_id = (attempt.get("job") or {}).get("job_id")
    if job_id:
        until = f"until bound to job {job_id}"
        if attempt.get("identified_at"):
            until += f" at {attempt['identified_at']}"
    else:
        until = f"until resolved ({attempt.get('resolution') or attempt.get('phase')})"
    return T(f"submission was uncertain from {since} {until}", "dim")


def render_inspect(data: Mapping[str, Any]) -> RenderableType:
    run = data.get("run") or {}
    parts: list[RenderableType] = []
    head = Text("run ", style="bold")
    head.append(str(run.get("label")), style="bold")
    head.append(f" ({run.get('run_id')}) · {run.get('kind')} · ")
    head.append_text(code_text(run.get("code")))
    head.append(f" {run.get('status')}")
    if run.get("code_reason"):
        head.append(f" — {run['code_reason']}")
    parts.append(head)
    parts.append(
        kv(
            "cell",
            f"{cell_label(run.get('cell'))} · seed {run.get('seed') if run.get('seed') is not None else '-'}"
            f" · experiment {run.get('experiment_id')} · plan {run.get('plan_id')}",
        )
    )
    for flag, text in (
        ("derived_from", "derived from"),
        ("reproduction_of", "reproduction of"),
    ):
        if run.get(flag):
            parts.append(kv(text, run[flag]))
    if run.get("added_post_hoc"):
        parts.append(
            T("added after results existed (post-hoc seed/cell; disclosed in comparisons)", "yellow")
        )
    refs = data.get("refs") or {}
    res = refs.get("resources") or {}
    if refs:
        code = f"snapshot {short_digest(refs.get('snapshot_digest'))}"
        code += (
            f" · git {str(refs['git_commit'])[:12]}{' dirty' if refs.get('git_dirty') else ''}"
            if refs.get("git_commit")
            else " · no git commit"
        )
        parts.append(kv("code", code))
        if res:
            parts.append(
                kv(
                    "resources",
                    f"{res.get('gpus', '?')}× {res.get('gpu_type') or 'GPU'} · {res.get('time', '?')} · mem {res.get('mem') or '?'}"
                    f" · partition {res.get('partition') or '?'} · account {res.get('account') or '-'}",
                )
            )
    attempts = data.get("attempts") or []
    focus = data.get("focus_attempt_id")
    if attempts:
        att_cols = [
            "#",
            "attempt",
            "scheduler",
            "app",
            "result",
            "cancel",
            "job",
            "elapsed",
            "GPU-h",
            "diagnosis",
        ]
        att_rows: list[list[Any]] = []
        for a in attempts:
            job = a.get("job") or {}
            job_s = str(job.get("job_id") or "—")
            if job.get("node_list"):
                job_s += f" @{job['node_list']}"
            if job.get("restart_count"):
                job_s += f" (requeued ×{job['restart_count']})"
            sched = str(a.get("scheduler_state") or "?")
            if a.get("phase") not in _OBSERVED_PHASES:
                # before a job is identified (or after a human decision) the phase is the state
                sched = str(a.get("phase"))
            if a.get("raw_scheduler_state") and a.get("raw_scheduler_state") != sched:
                sched += f" ({a['raw_scheduler_state']})"
            if a.get("observation_gap"):
                sched += " gap"
            b = a.get("budget") or {}
            gh = b.get("settled_consumed") if b.get("state") == "settled" else b.get("reserved")
            gh_s = gpuh(gh, unit=False) + (
                f" {b.get('state')}" if b.get("state") and b.get("state") != "settled" else ""
            )
            if b.get("state") == "unknown":
                gh_s = "unknown"
            marker = "▶" if a.get("attempt_id") == focus else ""
            sched_style = {
                "COMPLETED": "green",
                "RUNNING": "cyan",
                "PENDING": "yellow",
                "FAILED": "red",
                "TIMEOUT": "red",
                "OUT_OF_MEMORY": "red",
                "NODE_FAIL": "red",
                "CANCELLED": "red",
                "PREEMPTED": "yellow",
            }.get(str(a.get("scheduler_state")), "")
            app_style = {
                "exited_ok": "green",
                "exited_error": "red",
                "killed": "red",
                "guard_failed": "red",
            }.get(str(a.get("app_state")), "")
            result_style = {"valid": "green", "invalid": "red", "incomplete": "yellow", "missing": "red"}.get(
                str(a.get("result_state")), ""
            )
            att_rows.append(
                [
                    T(f"{marker}{a.get('seq')}"),
                    T(a.get("attempt_id")),
                    T(sched, sched_style),
                    T(a.get("app_state"), app_style),
                    T(a.get("result_state"), result_style),
                    T(a.get("cancel_state") if a.get("cancel_state") != "none" else "-"),
                    T(job_s),
                    T(f"{dur(a.get('elapsed_s'))}/{dur(a.get('time_limit_s'))}"),
                    T(gh_s),
                    T(a.get("diagnosis_code") or "-", "red" if a.get("diagnosis_code") else ""),
                ]
            )
        parts.append(
            dyn_table(
                att_cols,
                att_rows,
                keep=("#", "attempt", "scheduler", "app", "result", "job"),
                title=f"attempts ({data.get('attempts_total', len(attempts))}) on {(attempts[0].get('job') or {}).get('cluster_id')}",
            )
        )
        fa = next((a for a in attempts if a.get("attempt_id") == focus), attempts[0])
        if fa.get("exit_code") not in (None, 0) or fa.get("exit_signal"):
            parts.append(
                kv(
                    "exit",
                    f"code {fa.get('exit_code')} signal {fa.get('exit_signal') or 0}"
                    + (f" · {fa['reason']}" if fa.get("reason") else ""),
                )
            )
        if fa.get("uncertain_since"):
            parts.append(uncertainty_line(fa))
    else:
        parts.append(T("no attempts yet (the run was never submitted)", "dim"))
    pending = data.get("pending")
    if pending:
        line = Text("pending: ", style="bold yellow")
        line.append(f"{pending.get('reason_code') or pending.get('raw') or '?'} [{pending.get('category')}] ")
        line.append(str(pending.get("explanation") or ""))
        parts.append(line)
        if pending.get("suggestion"):
            parts.append(T("  → " + str(pending["suggestion"]), "yellow"))
    prog = data.get("progress")
    focus_att = next((a for a in attempts if a.get("attempt_id") == focus), attempts[0] if attempts else {})
    # ETA, walltime left and time-limit risk only mean something while the job can still run
    ended = focus_att.get("phase") in ENDED_PHASES
    if prog:
        line = Text("progress: ", style="bold")
        if ended and prog.get("step") is None and not prog.get("records"):
            line.append("no metrics records were written", style="dim")
        else:
            line.append(progress_bar(prog.get("fraction")))
        if prog.get("step") is not None:
            line.append(
                f"  step {prog.get('step')}/{prog.get('total_steps') if prog.get('total_steps') is not None else '?'}"
            )
        if ended:
            line.append("  (attempt ended)", style="dim")
        else:
            risk = str(prog.get("timelimit_risk") or "unknown")
            line.append(
                f"  ETA {dur(prog.get('eta_s'))} vs walltime left {dur(prog.get('remaining_walltime_s'))}  risk "
            )
            line.append(risk, style=RISK_STYLE.get(risk, ""))
            if prog.get("stalled_suspect"):
                line.append("  STALLED?", style="bold red")
        if prog.get("final_seen"):
            line.append("  final record seen", style="green")
        parts.append(line)
        lm = prog.get("latest_metrics") or {}
        if lm:
            parts.append(
                T(
                    "  latest: "
                    + ", ".join(
                        f"{k}={num(v) if isinstance(v, float) else v}" for k, v in list(lm.items())[:8]
                    ),
                    "dim",
                )
            )
        if prog.get("last_record_age_s") is not None:
            parts.append(T(f"  last metrics record {dur(prog.get('last_record_age_s'))} ago", "dim"))
        for n in prog.get("notes") or []:
            parts.append(T("  " + n, "dim"))
    health = data.get("health")
    if health:
        if health.get("source") == "none":
            parts.append(T("health: no GPU telemetry yet (unknown, not idle)", "dim"))
        else:
            util = health.get("per_gpu_util_mean") or {}
            line = Text("health: ", style="bold")
            line.append(
                ("GPU util " + ", ".join(f"{k}:{v:.0f}%" for k, v in util.items()))
                if util
                else "GPU util unknown"
            )
            if health.get("peak_gpu_mem_gib") is not None:
                line.append(f" · peak mem {health['peak_gpu_mem_gib']:.1f} GiB")
            if health.get("heartbeat_age_s") is not None:
                line.append(f" · heartbeat {dur(health['heartbeat_age_s'])} ago")
            flags = health.get("flags") or []
            if flags:
                line.append(f" · flags: {', '.join(flags)} (advisory)", style="bold yellow")
            parts.append(line)
    probe = data.get("gpu_probe")
    if probe:
        line = Text("GPU probe: ", style="bold")
        line.append(str(probe.get("status")))
        if probe.get("detail"):
            line.append(f" — {probe['detail']}")
        if probe.get("next_allowed_at"):
            line.append(f" (next allowed {probe['next_allowed_at']})", style="dim")
        parts.append(line)
        for s in probe.get("samples") or []:
            parts.append(
                T(
                    f"  GPU{s.get('index')} {s.get('name') or ''}: util {num(s.get('util_percent'), 0)}% · mem "
                    f"{num(s.get('mem_used_mib'), 0)}/{num(s.get('mem_total_mib'), 0)} MiB · {num(s.get('power_w'), 0)} W"
                )
            )
    diag = data.get("diagnosis")
    if diag:
        parts.append(render_diagnosis(diag))
    env = data.get("env") or {}
    if env.get("drift"):
        parts.append(kv("environment drift", "; ".join(env["drift"]), "yellow"))
    arts = data.get("artifacts")
    if arts and not arts.get("total"):
        parts.append(kv("artifacts", f"none indexed yet · results {arts.get('result_state')}"))
    elif arts:
        line = kv(
            "artifacts",
            f"{arts.get('total')} ({', '.join(f'{k}={v}' for k, v in (arts.get('by_integrity') or {}).items())}) · results {arts.get('result_state')}",
        )
        if arts.get("required_missing"):
            line.append(f" · required missing: {', '.join(arts['required_missing'])}", style="red")
        parts.append(line)
    evals = data.get("evaluations") or []
    if evals:
        parts.append(
            kv(
                "evaluations",
                ", ".join(
                    f"{e.get('evaluation_id')} {e.get('status')}{' (selected)' if e.get('selected') else ''}"
                    for e in evals[:4]
                ),
            )
        )
    if data.get("stale"):
        parts.append(
            T("state may be outdated: the refresh timed out (showing the last known state)", "bold yellow")
        )
    parts.append(next_steps(inspect_next(data)))
    hints = [h for h in data.get("recommended_next") or [] if "(" not in h]
    for h in hints:
        parts.append(T("  note: " + h, "dim"))
    return Group(*parts)


def render_diagnosis(diag: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    items = diag.get("items") or []
    primary = diag.get("primary_code")
    if not primary and not items:
        return T(f"diagnosis: {diag.get('summary') or 'nothing found'}", "dim")
    head = Text("diagnosis: ", style="bold red" if primary else "bold")
    head.append(str(diag.get("summary") or primary or "none"))
    parts.append(head)
    for it in items[:3]:
        line = Text("  ")
        line.append(str(it.get("code")), style="bold red" if it.get("code") == primary else "bold")
        line.append(f" [{it.get('confidence')}] {it.get('title') or ''}")
        parts.append(line)
        if it.get("explanation"):
            parts.append(T("    " + clip(it["explanation"], 300)))
        for ev in (it.get("evidence") or [])[:5]:
            parts.append(T("    │ " + clip(ev, 200), "dim"))
        strategies = it.get("suggested_strategies") or []
        if strategies:
            parts.append(T("    suggested: " + ", ".join(strategies), "cyan"))
        if it.get("resource_hint"):
            parts.append(
                T(
                    "    resource hint: " + ", ".join(f"{k}={v}" for k, v in it["resource_hint"].items()),
                    "cyan",
                )
            )
    if diag.get("last_error_line"):
        parts.append(T("  last error: " + clip(diag["last_error_line"], 200), "red"))
    return Group(*parts)


def inspect_next(data: Mapping[str, Any]) -> list[str]:
    """CLI commands for the state shown (hints, never actions)."""
    run = data.get("run") or {}
    rid = run.get("run_id")
    attempts = data.get("attempts") or []
    focus = data.get("focus_attempt_id")
    a = next((x for x in attempts if x.get("attempt_id") == focus), attempts[0] if attempts else None)
    if a is None:
        return [f"hpclab submit {rid}"]
    aid = a.get("attempt_id")
    phase = a.get("phase")
    out: list[str] = []
    if data.get("stale"):
        out.append(f"hpclab inspect {rid} --refresh")
    if phase == "submission_uncertain":
        if a.get("operation_id"):
            out.append(f"hpclab ops {a['operation_id']} --wait")
    elif phase == "duplicate_detected":
        out.append(f"hpclab resolve {aid} --bind-job <job_id> --reason '<why>'   # human decision")
        out.append(f"hpclab resolve {aid} --abandon --reason '<why>'")
    elif phase in ("job_identified", "observing"):
        pending = data.get("pending") or {}
        if pending.get("category") == "never_starts":
            out.append(f"hpclab cancel {aid} --reason 'cannot start: {pending.get('reason_code')}'")
        if a.get("scheduler_state") == "RUNNING":
            out.append(f"hpclab logs {aid} --follow")
        health = data.get("health") or {}
        if health.get("flags"):
            out.append(f"hpclab inspect {rid} --gpu")
        prog = data.get("progress") or {}
        if prog.get("timelimit_risk") == "high":
            out.append(f"hpclab cancel {aid} --checkpoint   # only if the run declares checkpoint_and_exit")
        out.append(f"hpclab inspect {rid} --refresh")
    elif phase == "terminal":
        failed = a.get("scheduler_state") != "COMPLETED" or a.get("app_state") not in ("exited_ok", "unknown")
        if failed:
            out.append(f"hpclab logs {aid} --stderr --tail 200")
            diag = data.get("diagnosis") or {}
            strategies: list[str] = []
            for it in diag.get("items") or []:
                if it.get("code") == diag.get("primary_code"):
                    strategies = list(it.get("suggested_strategies") or [])
            rec = [s for s in strategies if s in ("retry", "resume", "adjust_resources")]
            if rec:
                out.append(f"hpclab recover {aid} --strategy {rec[0]}")
            elif "derive_run" in strategies:
                out.append(f"hpclab prepare --derive-from {rid} --set <param>=<value>")
            else:
                out.append(f"hpclab recover {aid}")
        elif a.get("result_state") in ("unchecked", "pending"):
            out.append(f"hpclab artifacts {aid}")
        elif a.get("result_state") in ("incomplete", "invalid", "missing"):
            out.append(f"hpclab artifacts {aid}")
            out.append(f"hpclab logs {aid} --tail 100")
        elif a.get("result_state") == "valid" and not any(
            e.get("attempt_id") == aid and e.get("status") == "valid" for e in data.get("evaluations") or []
        ):
            out.append(f"hpclab collect {aid}")
        elif a.get("result_state") == "valid" and run.get("experiment_id"):
            out.append(f"hpclab compare {run['experiment_id']}")
    elif phase in ("submission_rejected", "cancelled_before_submit", "abandoned"):
        out.append(f"hpclab events --entity {aid}")
    return out


# ---------------------------------------------------------------------------
# logs / artifacts / evaluations / recovery
# ---------------------------------------------------------------------------


def render_artifacts(data: Mapping[str, Any]) -> RenderableType:
    arts = data.get("artifacts") or []
    head = [str(data.get("attempt_id"))]
    for label, key in (("", "phase"), ("scheduler ", "scheduler_state"), ("results ", "result_state")):
        if data.get(key) is not None:
            head.append(f"{label}{data[key]}")
    parts: list[RenderableType] = [kv("attempt", " · ".join(head))]
    if not arts:
        parts.append(T("no declared outputs indexed yet (indexing runs when the attempt ends)", "dim"))
        return Group(*parts)
    t = table("name", "role", "kind", "integrity", "size", "digest", "req", "artifact id")
    for a in arts:
        integ = str(a.get("integrity"))
        style = {
            "finalized": "green",
            "mutable": "cyan",
            "incomplete": "yellow",
            "missing": "red",
            "unknown": "magenta",
        }.get(integ, "")
        name = str(a.get("name"))
        member = a.get("member_path")
        if member and not name.endswith(str(member)):
            name += f"#{member}"
        t.add_row(
            T(name),
            T(a.get("role")),
            T(a.get("kind")),
            T(integ, style),
            T(size(a.get("size_bytes"))),
            T(short_digest(a.get("digest"), 10) if a.get("digest") else a.get("digest_state") or "?"),
            T("yes" if a.get("required") else ""),
            T(a.get("artifact_id"), "dim"),
        )
    parts.append(t)
    return Group(*parts)


def render_artifact_read(data: Mapping[str, Any]) -> RenderableType | None:
    """Metadata line for `hpclab read` (the content itself is written raw by the command)."""
    a = data.get("artifact") or {}
    line = Text(f"{a.get('name')} ", style="bold")
    line.append(
        f"[{data.get('view')}] {data.get('bytes_read', 0)}/{size(data.get('size'))} · {a.get('integrity')} · "
        f"digest {'verified' if data.get('digest_verified') else 'not verified'} · source {data.get('source')}"
    )
    if data.get("truncated"):
        line.append(" · truncated", style="yellow")
    return line


def render_collect(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    if data.get("mode") == "single":
        ev = data.get("evaluation") or {}
        head = Text("evaluation ", style="bold")
        head.append(f"{ev.get('evaluation_id')} ")
        st = str(ev.get("status"))
        head.append(st, style={"valid": "bold green", "partial": "yellow", "invalid": "bold red"}.get(st, ""))
        head.append(
            f" · {'new' if data.get('created') else 'already existed'}"
            f" · {ev.get('evaluator_id')}@{ev.get('evaluator_version')}"
        )
        parts.append(head)
        t = table("metric", "value", "split", "step", "unit")
        for m in (ev.get("metrics") or [])[:40]:
            t.add_row(
                T(m.get("name")),
                T(num(m.get("value"))),
                T(m.get("split") or ""),
                T(m.get("step") if m.get("step") is not None else ""),
                T(m.get("unit") or ""),
            )
        if ev.get("metrics"):
            parts.append(t)
        if ev.get("metrics_truncated"):
            parts.append(T(f"… {ev.get('metrics_total')} metrics in total", "dim"))
        issues = (ev.get("validation") or {}).get("issues") or []
        for i in issues[:10]:
            parts.append(T(f"  {i.get('severity')}: {i.get('message')}", "yellow"))
        return Group(*parts)
    c = data.get("counts") or {}
    parts.append(
        kv(
            "collected",
            f"{c.get('collected', 0)} · skipped {data.get('skipped_total', c.get('skipped', 0))} · pending {data.get('pending', 0)}",
        )
    )
    if data.get("collected"):
        t = table("run", "attempt", "evaluation", "status", "new")
        for it in data["collected"][:40]:
            t.add_row(
                T(it.get("run_id")),
                T(it.get("attempt_id")),
                T(it.get("evaluation_id")),
                T(it.get("status")),
                T("yes" if it.get("created") else ""),
            )
        parts.append(t)
    skipped = [it for it in data.get("skipped") or [] if it.get("reason") != "already_evaluated"]
    already = sum(1 for it in data.get("skipped") or [] if it.get("reason") == "already_evaluated")
    if already:
        parts.append(T(f"{already} run(s) already had an evaluation (idempotent: nothing recomputed)", "dim"))
    if skipped:
        t = table("run", "attempt", "reason", "detail", title="skipped")
        for it in skipped[:40]:
            t.add_row(
                T(it.get("run_id") or "-"),
                T(it.get("attempt_id") or "-"),
                T(it.get("reason")),
                T(clip(it.get("detail") or "", 80), "dim"),
            )
        parts.append(t)
    if data.get("operation_id"):
        parts.append(
            T(f"still collecting in the background: hpclab ops {data['operation_id']} --wait", "cyan")
        )
    return Group(*parts)


def render_recover(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    diag = data.get("diagnosis") or {}
    if data.get("mode") == "batch":
        head = Text("recovery plan ", style="bold")
        head.append(
            f"{data.get('plan_id') or '(none)'} · {data.get('eligible_count', 0)}/{data.get('total', 0)} cell(s) eligible"
        )
        if data.get("source_workflow_id"):
            head.append(f" · from workflow {data['source_workflow_id']}")
        parts.append(head)
        t = table("label", "code", "eligible", "strategy", "GPU-h", "diagnosis", "blocking / probe")
        for c in data.get("cells") or []:
            t.add_row(
                T(c.get("label")),
                code_text(c.get("code")),
                T("yes" if c.get("eligible") else "no", "green" if c.get("eligible") else "red"),
                T(c.get("strategy") or "-"),
                T(gpuh(c.get("requested_gpu_hours"), unit=False)),
                T(c.get("diagnosis_code") or "-"),
                T(
                    clip(
                        c.get("blocking_reason")
                        or (f"probe run {c['probe_run_id']}" if c.get("probe_run_id") else ""),
                        80,
                    )
                ),
            )
        parts.append(t)
        parts.append(kv("GPU-hours", gpuh(data.get("requested_gpu_hours"))))
    else:
        head = Text("recovery of ", style="bold")
        head.append(f"{data.get('attempt_id')} (run {data.get('run_id')}): ")
        head.append(
            "eligible" if data.get("eligible") else "not eligible",
            style="bold green" if data.get("eligible") else "bold red",
        )
        if data.get("strategy"):
            head.append(f" · strategy {data['strategy']}")
        parts.append(head)
        if diag:
            line = Text("diagnosis: ", style="bold")
            line.append(f"{diag.get('primary_code') or 'none'}")
            if diag.get("title"):
                line.append(f" — {diag['title']}")
            if diag.get("confidence"):
                line.append(f" [{diag['confidence']}]")
            line.append(f" (source {diag.get('source')})", style="dim")
            parts.append(line)
            if diag.get("summary"):
                parts.append(T("  " + clip(diag["summary"], 300)))
            if diag.get("last_error_line"):
                parts.append(T("  last error: " + clip(diag["last_error_line"], 200), "red"))
            if diag.get("suggested_strategies"):
                parts.append(T("  suggested: " + ", ".join(diag["suggested_strategies"]), "cyan"))
        if data.get("resources"):
            r = data["resources"]
            parts.append(
                kv(
                    "resources",
                    f"{r.get('gpus', '?')}× {r.get('gpu_type') or 'GPU'} · {r.get('time')} · mem {r.get('mem') or '?'} · {r.get('partition')}",
                )
            )
        if data.get("resource_patch"):
            parts.append(
                kv("changes", ", ".join(f"{k}={v}" for k, v in data["resource_patch"].items()), "yellow")
            )
        if data.get("exclude_nodes"):
            parts.append(kv("excluding nodes", ", ".join(data["exclude_nodes"])))
        if data.get("suspect_nodes"):
            parts.append(kv("suspect nodes (not excluded)", ", ".join(data["suspect_nodes"]), "dim"))
        ck = data.get("checkpoint") or {}
        if ck:
            parts.append(
                kv(
                    "checkpoint",
                    f"{ck.get('artifact_id') or ck.get('remote_dir')} step {ck.get('step')} · {ck.get('status')} · {ck.get('continuation')}"
                    + (" · needs a load probe" if ck.get("needs_load_probe") else ""),
                )
            )
            for i in ck.get("issues") or []:
                parts.append(T("  " + i, "yellow"))
        probe = data.get("probe") or {}
        if probe:
            parts.append(
                kv(
                    "checkpoint probe",
                    f"{probe.get('state')} · run {probe.get('run_id')}"
                    + (f" — {probe['detail']}" if probe.get("detail") else ""),
                )
            )
        if data.get("requested_gpu_hours") is not None:
            parts.append(kv("GPU-hours", gpuh(data.get("requested_gpu_hours"))))
    checks = data.get("checks") or []
    if checks:
        t = table("status", "check", "message")
        for c in checks:
            t.add_row(status_text(c.get("status")), T(c.get("check")), T(c.get("message")))
        parts.append(t)
    for b in data.get("blocking") or []:
        parts.append(T("blocking: " + str(b), "red"))
    if data.get("mode") != "batch" and any("resource_patch" in str(b) for b in data.get("blocking") or []):
        parts.append(
            T(
                f"on the command line: hpclab recover {data.get('attempt_id')} --strategy adjust_resources "
                "--gpu-type <type> (or --mem / --time / --gpus / --partition / --exclude-nodes)",
                "cyan",
            )
        )
    for h in data.get("hints") or []:
        parts.append(T(f"hint ({h.get('strategy')}): {h.get('message')}", "cyan"))
    if data.get("next_step"):
        parts.append(T(str(data["next_step"]), "dim"))
    return Group(*parts)


def recover_next(data: Mapping[str, Any]) -> list[str]:
    if data.get("mode") == "batch":
        return (
            [f"hpclab submit {data['plan_id']}"] if data.get("plan_id") and data.get("eligible_count") else []
        )
    probe = data.get("probe") or {}
    if probe.get("run_id") and probe.get("state") not in ("passed",):
        cmd = f"hpclab submit {probe['run_id']}"
        if probe.get("exec_plan_hash"):
            cmd += f" --exec-plan {probe['exec_plan_hash']}"
        return [
            cmd + "   # checkpoint load probe first",
            f"hpclab recover {data.get('attempt_id')} --strategy resume   # again once the probe passed",
        ]
    if data.get("eligible") and data.get("exec_plan_hash") and data.get("run_id"):
        return [f"hpclab submit {data['run_id']} --exec-plan {data['exec_plan_hash']}"]
    return []


# ---------------------------------------------------------------------------
# compare / evidence / verify / trace
# ---------------------------------------------------------------------------


def _arrow(hib: Any) -> str:
    return " ↑" if hib is True else (" ↓" if hib is False else "")


def render_compare(data: Mapping[str, Any], *, decimals: int = 4) -> RenderableType:
    parts: list[RenderableType] = []
    exp = data.get("experiment") or {}
    rs = data.get("run_set") or {}
    head = Text("comparison ", style="bold")
    head.append(str(data.get("comparison_id")), style="bold")
    head.append(f" · {rs.get('kind')} {exp.get('name') or ''} ({rs.get('id')})")
    if data.get("reused"):
        head.append(" · identical to an archived comparison (reused)", style="dim")
    parts.append(head)
    info = {m.get("name"): m for m in data.get("metric_info") or []}
    metrics = list(data.get("metric_names") or [])
    t = Table(header_style="bold")
    t.add_column("group", style="bold")
    for m in metrics:
        mi = info.get(m) or {}
        unit = f" ({mi['unit']})" if mi.get("unit") else ""
        t.add_column(f"{m}{unit}{_arrow(mi.get('higher_is_better'))}", overflow="fold")
    for g in data.get("groups") or []:
        label = Text(str(g.get("label")))
        if g.get("is_baseline"):
            label.append(" (reference)", style="dim")
        row: list[Text] = [label]
        for m in metrics:
            st = (g.get("metrics") or {}).get(m) or {}
            n, en = st.get("n", 0), st.get("expected_n", 0)
            if st.get("mean") is None:
                cell = Text("—", style="dim")
            else:
                cell = Text(num(st.get("mean"), decimals))
                cell.append(" ± " + (num(st.get("std"), decimals) if st.get("std") is not None else "—"))
            cell.append(f" ({n}/{en})", style="bold yellow" if n < en else "dim")
            row.append(cell)
        t.add_row(*row)
    parts.append(t)
    parts.append(T("cells: mean ± sample std over seeds (included/expected); — = undefined", "dim"))
    diffs = data.get("differences") or []
    if diffs:
        parts.append(Text("differences vs baseline (descriptive; not a significance test):", style="bold"))
        for d in diffs:
            mi = info.get(d.get("metric")) or {}
            line = Text(
                f"  {d.get('group_label')} − {d.get('baseline_label')}  {d.get('metric')}{_arrow(mi.get('higher_is_better'))}: "
            )
            line.append(signed(d.get("mean_diff"), decimals), style="bold")
            if d.get("ci_low") is not None and d.get("ci_high") is not None:
                line.append(
                    f"  CI [{signed(d.get('ci_low'), decimals)}, {signed(d.get('ci_high'), decimals)}]"
                )
                if d["ci_low"] <= 0 <= d["ci_high"]:
                    line.append(" (includes 0)", style="dim")
            else:
                line.append("  CI undefined", style="dim")
            if d.get("kind") == "paired":
                line.append(f" · paired, {d.get('n')} seed(s)", style="dim")
            else:
                line.append(f" · unpaired, n={d.get('n')} vs {d.get('n_baseline')}", style="dim")
            if d.get("unpaired_group") or d.get("unpaired_baseline"):
                line.append(
                    f"  (unmatched seeds: group {d.get('unpaired_group') or []}, baseline {d.get('unpaired_baseline') or []})",
                    style="yellow",
                )
            parts.append(line)
        ci = _g(data, "contract", "ci_level")
        if ci:
            parts.append(T(f"  intervals are {ci:.0%} confidence intervals of the difference", "dim"))
    parts.append(coverage_block(data))
    inc = data.get("incomparable") or []
    for i in inc:
        parts.append(
            T(
                f"INCOMPARABLE: {i.get('reason')}"
                + (f" ({i['invariant']})" if i.get("invariant") else "")
                + (f" metric {i['metric']}" if i.get("metric") else "")
                + (f": {clip(i.get('details'), 160)}" if i.get("details") else ""),
                "bold red",
            )
        )
    parts.extend(integrity_lines(data.get("integrity") or {}))
    return Group(*parts)


def coverage_block(data: Mapping[str, Any]) -> RenderableType:
    counts = data.get("counts") or {}
    parts: list[RenderableType] = []
    line = Text("coverage: ", style="bold")
    line.append(
        f"{counts.get('included', 0)}/{counts.get('planned_runs', 0)} planned runs included · "
        f"missing {counts.get('missing', 0)} · failed {counts.get('failed', 0)} · excluded {counts.get('excluded', 0)}"
    )
    complete = counts.get("included", 0) == counts.get("planned_runs", 0)
    if complete:
        line.append("  complete", style="green")
    parts.append(line)
    for r in data.get("runs") or []:
        if r.get("status") == "included":
            continue
        style = {"missing": "yellow", "failed": "red", "excluded": "magenta"}.get(str(r.get("status")), "")
        why = r.get("exclusion_reason") or r.get("reason") or CODE_LEGEND.get(str(r.get("code")), "")
        rl = Text(f"  {r.get('status')}: ", style=style)
        rl.append(f"{r.get('group_label')} seed {r.get('seed') if r.get('seed') is not None else '-'} ")
        rl.append_text(code_text(r.get("code")))
        rl.append(f" {why}")
        rl.append(f"  ({r.get('run_id')})", style="dim")
        parts.append(rl)
    if data.get("runs_truncated"):
        parts.append(T(f"  … {data['runs_truncated']} more run(s) not listed", "dim"))
    return Group(*parts)


def integrity_lines(integ: Mapping[str, Any]) -> list[Text]:
    out: list[Text] = []
    for p in integ.get("post_hoc") or []:
        out.append(
            T(
                f"integrity: {p.get('group_label')} seed {p.get('seed')} was added after results existed ({p.get('status')})",
                "yellow",
            )
        )
    for e in integ.get("exclusions") or []:
        late = " — decided AFTER the value was observed" if e.get("observed_before_exclusion") else ""
        out.append(
            T(
                f"integrity: excluded {e.get('group_label')} seed {e.get('seed')}: {e.get('reason')}{late}",
                "magenta",
            )
        )
    for h in integ.get("heterogeneity") or []:
        if h.get("mixed"):
            out.append(
                T(
                    f"integrity: mixed {h.get('dimension')}: {', '.join(f'{k} ×{v}' for k, v in (h.get('values') or {}).items())}"
                    + (f" (+{h['unknown']} unknown)" if h.get("unknown") else "")
                    + (f" within {', '.join(h['within_groups'])}" if h.get("within_groups") else ""),
                    "yellow",
                )
            )
    for m in integ.get("multiple_valid_attempts") or []:
        out.append(
            T(
                f"integrity: {m.get('group_label')} seed {m.get('seed')} has {m.get('valid_attempts')} valid attempts; "
                f"rule {m.get('rule')} selected {m.get('selected_attempt_id')}",
                "yellow",
            )
        )
    for b in integ.get("below_min_n") or []:
        out.append(
            T(
                f"integrity: {b.get('group_label')}/{b.get('metric')} has n={b.get('n')} < min_n={b.get('min_n')}",
                "yellow",
            )
        )
    sel = integ.get("selection") or {}
    if sel.get("evaluation_selection_on_split"):
        out.append(T(f"integrity: selection uses split(s) {sel['evaluation_selection_on_split']}", "yellow"))
    return out


def render_evidence(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = [
        kv(
            "bundle",
            f"{data.get('bundle_id')} · comparison {data.get('comparison_id')} · {data.get('purpose')}",
        ),
    ]
    c = data.get("counts") or {}
    if c:
        parts.append(
            kv(
                "coverage",
                f"{c.get('included', 0)}/{c.get('planned_runs', 0)} planned runs included · missing {c.get('missing', 0)} · "
                f"failed {c.get('failed', 0)} · excluded {c.get('excluded', 0)}",
            )
        )
    if data.get("incomparable"):
        parts.append(T(f"{data['incomparable']} incomparable entr(y/ies) — see summary.md", "bold red"))
    for f in data.get("integrity_flags") or []:
        parts.append(T("integrity: " + str(f), "yellow"))
    return Group(*parts)


def render_written(data: Mapping[str, Any]) -> RenderableType:
    """Files of `evidence_write`: the directory once, then names (full paths do not fit a table)."""
    t = table("name", "bytes", "sha256", "status")
    for f in data.get("files") or []:
        t.add_row(
            T(f.get("name") or f.get("path")),
            T(f.get("bytes")),
            T(short_digest(f.get("sha256"), 12)),
            T("written" if f.get("written") else "unchanged"),
        )
    return Group(kv("written to", data.get("out_dir")), t)


def render_verify_findings(findings: Sequence[Mapping[str, Any]]) -> RenderableType:
    if not findings:
        return Text("no findings", style="green")
    t = table("severity", "code", "message", "where")
    for f in findings:
        t.add_row(
            severity_text(f.get("severity")),
            T(f.get("code"), "bold"),
            T(f.get("message")),
            T(", ".join(f.get("refs") or []), "dim"),
        )
    return t


def scanned_line(files: Sequence[str]) -> Text:
    """The manuscript files the verifier read (the given ones and what they \\input)."""
    if not files:
        return Text("scanned: no manuscript file could be read", style="yellow")
    return Text(f"scanned {len(files)} file(s):" + "".join(f"\n  {f}" for f in files), style="dim")


def not_checked_line() -> Text:
    return Text(
        "manuscript NOT checked: no \\hpcval tag in the scanned files (a pass needs tagged numbers; "
        "paste the bundle's table.tex, or \\input it)",
        style="bold red",
    )


def render_verify(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    ok = data.get("ok")
    head = Text("verify ", style="bold")
    head.append(f"{data.get('bundle_id')}: ")
    head.append("no errors" if ok else "ERRORS", style="bold green" if ok else "bold red")
    head.append(
        f" · {data.get('values_checked', 0)} value(s) recomputed · {data.get('tags_checked', 0)} tex tag(s) checked"
    )
    parts.append(head)
    parts.append(render_verify_findings(data.get("findings") or []))
    if data.get("findings_truncated"):
        parts.append(T(f"… {data['findings_truncated']} more finding(s)", "dim"))
    for rep in data.get("tex") or []:
        parts.append(
            T(
                f"tex {rep.get('source')}: {rep.get('tags')} tag(s), {rep.get('untagged_numbers')} untagged number(s), bundles {', '.join(rep.get('bundles') or []) or '-'}",
                "dim",
            )
        )
    if data.get("newer_comparison_ref"):
        parts.append(T(f"a newer comparison exists: {data['newer_comparison_ref']}", "yellow"))
    if data.get("ledger_changed"):
        parts.append(T("the ledger changed since this bundle was exported (see warnings)", "yellow"))
    parts.append(
        T(str(data.get("statement") or "record consistency only, not scientific correctness"), "dim")
    )
    return Group(*parts)


def render_trace(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    head = Text(f"{data.get('cell_ref')}: ", style="bold")
    head.append(f"{data.get('group_label')} / {data.get('metric')} {data.get('stat')} = ")
    head.append(num(data.get("value")), style="bold")
    head.append(f"  (n={data.get('n')}/{data.get('expected_n')})")
    head.append(
        "  chain complete" if data.get("complete") else "  chain INCOMPLETE",
        style="green" if data.get("complete") else "bold red",
    )
    parts.append(head)
    parts.append(
        kv(
            "bundle",
            f"{data.get('bundle_id')} · comparison {data.get('comparison_id') or '?'} · run set {data.get('run_set_ref') or '?'}",
        )
    )
    for s in data.get("seeds") or []:
        line = Text(f"seed {s.get('seed')}: ", style="bold")
        line.append(
            str(s.get("status")),
            style={"included": "green", "missing": "yellow", "failed": "red", "excluded": "magenta"}.get(
                str(s.get("status")), ""
            ),
        )
        if s.get("value") is not None:
            line.append(f" value {num(s.get('value'))}")
        if s.get("exclusion_reason"):
            line.append(f" — excluded: {s['exclusion_reason']}")
        parts.append(line)
        chain = s.get("chain") or {}
        ev = chain.get("evaluation") or {}
        art = chain.get("artifact") or {}
        att = chain.get("attempt") or {}
        job = chain.get("job") or {}
        run = chain.get("run") or {}
        code = chain.get("code") or {}
        env = chain.get("env") or {}
        env_bits = [
            str(env[k])
            for k in ("gpu", "torch", "cuda", "driver_version", "python", "host")
            if env.get(k) not in (None, "", [])
        ]
        links = [
            (
                "evaluation",
                ev.get("evaluation_id"),
                f"{ev.get('evaluator') or ''} {ev.get('status') or ''}".strip(),
            ),
            (
                "artifact",
                art.get("artifact_id"),
                f"{art.get('name') or ''} {short_digest(art.get('digest'))} {art.get('integrity') or ''}".strip(),
            ),
            (
                "attempt",
                att.get("attempt_id"),
                f"#{att.get('seq')} {att.get('scheduler_state') or att.get('phase') or ''} {att.get('result_state') or ''}".strip(),
            ),
            (
                "job",
                f"{job.get('cluster_id') or '?'}:{job.get('job_id') or '?'}" if job else None,
                f"owner {job.get('owner') or '?'} submitted {job.get('submit_time') or '?'}" if job else "",
            ),
            (
                "run",
                run.get("run_id"),
                f"{run.get('label') or ''} spec {short_digest(run.get('spec_hash'))}".strip(),
            ),
            (
                "code",
                short_digest(code.get("digest")) if code.get("digest") else None,
                (f"git {str(code['git_commit'])[:12]}" + (" dirty" if code.get("git_dirty") else ""))
                if code.get("git_commit")
                else "no git commit",
            ),
            ("env", " · ".join(env_bits) or None, ""),
        ]
        for name, ident, extra in links:
            if ident:
                parts.append(T(f"   {name:<10} {ident} {extra}".rstrip(), "dim"))
        for i in s.get("issues") or []:
            parts.append(T("   issue: " + i, "yellow"))
    for i in data.get("issues") or []:
        parts.append(T("issue: " + i, "yellow"))
    return Group(*parts)


# ---------------------------------------------------------------------------
# operations / usage / digest / vault / events / cluster
# ---------------------------------------------------------------------------


def render_operation(data: Mapping[str, Any]) -> RenderableType:
    state = str(data.get("state"))
    head = Text(f"operation {data.get('operation_id')} ", style="bold")
    head.append(f"({data.get('kind')}) ")
    head.append(
        state,
        style={"succeeded": "bold green", "failed": "bold red", "blocked": "bold yellow"}.get(state, "cyan"),
    )
    if data.get("stage"):
        head.append(f" at {data['stage']}")
    parts: list[RenderableType] = [head]
    if data.get("target_id"):
        parts.append(kv("target", f"{data.get('target_type')} {data.get('target_id')}"))
    if data.get("progress"):
        parts.append(kv("progress", ", ".join(f"{k}={v}" for k, v in data["progress"].items())))
    err = data.get("error") or {}
    if err:
        parts.append(kv("error", f"{err.get('code')}: {err.get('message')}", "red"))
        details = err.get("details") or {}
        if details.get("remediation"):
            parts.append(T("  → " + str(details["remediation"]), "yellow"))
    if data.get("result") not in (None, {}, []):
        parts.append(kv("result", clip(data.get("result"), 300)))
    events = sorted(data.get("recent_events") or [], key=lambda e: str(e.get("recorded_at") or ""))
    for e in events[-6:]:
        parts.append(
            T(
                f"  {str(e.get('recorded_at'))[11:19]}Z {e.get('event_type')} {clip(e.get('payload') or {}, 100)}",
                "dim",
            )
        )
    return Group(*parts)


def render_usage(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    vis = str(data.get("visibility"))
    head = Text("usage ", style="bold")
    who = f"account {data.get('account')}" if data.get("scope") == "account" else f"user {data.get('user')}"
    head.append(
        f"on {data.get('cluster_id')} · {who} · {str(data.get('since'))[:16]} → {str(data.get('until'))[:16]}"
    )
    parts.append(head)
    if vis == "self_only":
        parts.append(
            T(
                "visibility: only your jobs are visible (Slurm PrivateData hides other users' jobs) — "
                "no lab totals can be computed from here; ask an account coordinator for a lab-wide report",
                "bold yellow",
            )
        )
    elif vis == "unavailable":
        parts.append(T("visibility: accounting unavailable — numbers below are incomplete", "bold red"))
    else:
        parts.append(T(f"visibility: full (PrivateData={data.get('private_data') or 'none'})", "dim"))
    rows = data.get("rows") or []
    group_cols = list(data.get("group_by") or [])
    t = table(*group_cols, "jobs", "GPU-h", "by state", "waste", "queue wait (med)", "walltime used (med)")
    for r in rows:
        g = r.get("group") or {}
        by_state = ", ".join(
            f"{k} {gpuh(v, unit=False)}"
            for k, v in sorted((r.get("gpu_hours_by_state") or {}).items(), key=lambda kv_: -kv_[1])[:3]
        )
        waste = r.get("wasted_gpu_hours_by_cause")
        waste_s = (
            "unknown"
            if waste is None
            else (", ".join(f"{k} {gpuh(v, unit=False)}" for k, v in waste.items() if v) or "0")
        )
        gh = gpuh(r.get("gpu_hours"), unit=False)
        if r.get("gpu_count_unknown_jobs"):
            gh = "≥" + gh
        wa = r.get("walltime_accuracy")
        t.add_row(
            *[T(g.get(c, "?")) for c in group_cols],
            T(r.get("jobs")),
            T(gh),
            T(by_state or "-"),
            T(waste_s),
            T(dur(r.get("median_queue_wait_s"))),
            T(f"{wa * 100:.0f}%" if isinstance(wa, (int, float)) else "?"),
        )
    if rows:
        parts.append(t)
    else:
        parts.append(T("no jobs in this window", "dim"))
    if data.get("rows_truncated"):
        parts.append(T(f"… {data['rows_truncated']} more group(s) (--limit)", "dim"))
    totals = data.get("totals") or {}
    for key, tot in totals.items():
        if key == "account" and vis != "full":
            continue
        label = "your total" if key == "self" else f"{key} total"
        line = kv(label, f"{tot.get('jobs')} job(s) · {gpuh(tot.get('gpu_hours'))} allocated")
        if tot.get("gpu_count_unknown_jobs"):
            line.append(
                f" · {tot['gpu_count_unknown_jobs']} job(s) with unknown GPU count (lower bound)",
                style="yellow",
            )
        waste = tot.get("wasted_gpu_hours_by_cause")
        if waste:
            causes = ", ".join(f"{k} {gpuh(v, unit=False)}" for k, v in waste.items() if v)
            line.append(f" · waste {gpuh(sum(waste.values()))}" + (f" ({causes})" if causes else ""))
        parts.append(line)
    now = data.get("now") or {}
    if now:
        parts.append(
            kv(
                "now",
                f"{now.get('running_jobs', '?')} running ({now.get('running_gpus', '?')} GPUs) · {now.get('pending_jobs', '?')} pending ({now.get('pending_gpus', '?')} GPUs)"
                + (f" · {now['note']}" if now.get("note") else ""),
            )
        )
    det = data.get("details") or {}
    if det.get("window_truncated"):
        parts.append(
            T("partial report (time budget reached): re-run to continue; finished days are cached", "yellow")
        )
    for n in data.get("notes") or []:
        parts.append(T(n, "dim"))
    return Group(*parts)


def render_vault(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = [
        kv(
            "vault",
            f"{data.get('vault_dir')}"
            + (f" · experiment {data['experiment_id']}" if data.get("experiment_id") else ""),
        ),
        kv(
            "attempts",
            f"{data.get('attempts_in_scope', 0)} in scope · {data.get('fully_captured', 0)} fully captured · "
            f"{data.get('partially_captured', 0)} partial · {data.get('not_captured', 0)} not captured · "
            f"{data.get('total_items', 0)} item(s), {size(data.get('total_bytes'))}",
        ),
    ]
    rows = [a for a in data.get("attempts") or [] if a.get("missing_kinds") or not a.get("items")]
    if rows:
        t = table("attempt", "run", "phase", "items", "missing", "last capture", title="incomplete captures")
        for a in rows[:40]:
            t.add_row(
                T(a.get("attempt_id")),
                T(a.get("run_id"), "dim"),
                T(a.get("phase")),
                T(a.get("items")),
                T(", ".join(a.get("missing_kinds") or []) or "-"),
                T(a.get("last_captured_at") or "never"),
            )
        parts.append(t)
    return Group(*parts)


def render_events(data: Mapping[str, Any]) -> RenderableType:
    evs = data.get("events") or []
    if not evs:
        return Text("no events match", style="dim")
    t = table("seq", "recorded (UTC)", "event", "entity", "source", "payload")
    for e in evs:
        payload = (
            "(large payload omitted)" if e.get("payload_truncated") else clip(e.get("payload") or {}, 120)
        )
        ent = str(e.get("entity_id"))
        if ent.startswith("sha256:"):  # execution plans are identified by their digest
            ent = short_digest(ent, 16)
        if e.get("entity_version") is not None:
            ent += f" v{e['entity_version']}"
        t.add_row(
            T(e.get("seq")),
            T(str(e.get("recorded_at") or "")[:19].replace("T", " ")),
            T(e.get("event_type"), "bold"),
            T(ent),
            T(e.get("source")),
            T(payload, "dim"),
        )
    return t


def render_cluster_inspect(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = []
    head = Text(f"cluster {data.get('cluster_id')}: ", style="bold")
    head.append_text(status_text(data.get("status")))
    parts.append(head)
    shown: set[str] = set()
    for c in data.get("checks") or []:
        line = Text()
        st = str(c.get("status"))
        line.append(f"{st:>11}", style=STATUS_STYLE.get(st, ""))
        line.append(f"  {c.get('name')}", style="bold")
        line.append(f"  {c.get('summary') or ''}")
        parts.append(line)
        for f in (c.get("findings") or [])[:8]:
            fl = Text("             · ")
            fl.append_text(severity_text(f.get("severity")))
            fl.append(f" {f.get('code')}: {f.get('message')}")
            parts.append(fl)
            if f.get("remediation"):
                shown.add(str(f["remediation"]))
                parts.append(T(f"               → {f['remediation']}", "yellow"))
    for cand in data.get("candidates") or []:
        sh = cand.get("shape") or {}
        parts.append(
            T(
                f"start estimate {sh.get('gpus')}×{sh.get('gpu_type') or 'gpu'} {sh.get('time')} on {sh.get('partition')}: "
                f"{cand.get('est_start') or 'unknown'} ({cand.get('status')}, {cand.get('confidence')} confidence)",
                "dim",
            )
        )
    for r in data.get("remediation") or []:
        if str(r) not in shown:
            parts.append(T("→ " + str(r), "yellow"))
    return Group(*parts)


def render_notify_test(data: Mapping[str, Any]) -> RenderableType:
    chans = data.get("channels") or []
    if not chans:
        return T(data.get("note") or "no notification channels configured", "dim")
    t = table("channel", "kind", "target", "result")
    for c in chans:
        t.add_row(
            T(c.get("channel_id")),
            T(c.get("kind")),
            T(c.get("target")),
            T("ok" if c.get("ok") else f"FAILED: {c.get('error')}", "green" if c.get("ok") else "red"),
        )
    return t


def render_backup(data: Mapping[str, Any]) -> RenderableType:
    return Group(
        kv("backup", data.get("backup_dir")),
        kv(
            "ledger",
            f"{data.get('db_path')} ({size(data.get('db_bytes'))}, {short_digest(data.get('db_sha256'), 16)}) · integrity {data.get('integrity_check')}",
        ),
        kv(
            "vault",
            f"{data.get('vault_files')} file(s), {size(data.get('vault_bytes'))} · {data.get('snapshot_manifests')} snapshot manifest(s)",
        ),
        kv("manifest", f"{data.get('manifest_path')} ({short_digest(data.get('manifest_digest'), 16)})"),
    )


def render_submit_receipt(data: Mapping[str, Any]) -> RenderableType:
    return Group(
        kv("attempt", f"{data.get('attempt_id')} ({data.get('phase')}) for run {data.get('run_id')}"),
        kv("reserved", gpuh(data.get("reserved_gpu_hours"))),
        T(
            "this is a receipt, not proof that Slurm accepted the job: the controller stages and submits it in the background",
            "dim",
        ),
    )


def render_workflow_submit(data: Mapping[str, Any]) -> RenderableType:
    parts: list[RenderableType] = [workflow_line(data)]
    parts.append(
        kv(
            "cells",
            f"{data.get('planned_run_count')} planned"
            + (f" · {data['skipped_count']} skipped (already done)" if data.get("skipped_count") else "")
            + f" · at most {data.get('max_parallel')} at once",
        )
    )
    gate = data.get("gate") or {}
    if gate:
        parts.append(
            T(
                f"smoke gate {gate.get('run_id')} runs first; the other cells are dispatched only if it passes",
                "dim",
            )
        )
    if data.get("resumed"):
        parts.append(T("dispatch re-opened", "green"))
    return Group(*parts)
