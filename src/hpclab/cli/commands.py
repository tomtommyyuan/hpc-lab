"""Experiment, run and result commands (registered on ``hpclab.cli.main.app`` at import).

Every command goes through the controller (``call`` / ``fetch``), so the CLI, the MCP tools
and Python callers share one ledger and one set of safety checks. Conventions:

* every mutation sends a fresh idempotency key (``new_key()``);
* anything that spends GPU-hours or kills jobs asks first, unless ``--yes``; without a
  terminal (scripts, CI) ``--yes`` is required;
* ``--json`` prints the raw Envelope of the command's main call. Composite commands print one
  JSON object keyed by method name; ``logs --follow`` prints one compact Envelope per line;
* exit codes follow ``hpclab.cli.common`` (0 ok, 1 error/not ok, 2 usage, 3 controller
  unavailable, 4 auth required; 130 when a watch/follow loop is interrupted with Ctrl-C).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.text import Text

from hpclab.cli import views as V
from hpclab.cli.common import (
    EXIT_ERROR,
    EXIT_USAGE,
    STATE,
    auth_resume,
    call,
    console,
    err_console,
    exit_code_for,
    fetch,
    incomplete_reason,
    new_key,
    parse_csv_ints,
    parse_kv,
    print_json,
    print_json_error,
    show,
)
from hpclab.cli.main import app, sim_app, sim_root
from hpclab.domain.errors import HpcLabError
from hpclab.domain.ids import kind_of

EXIT_INTERRUPTED = 130

workflow_app = typer.Typer(
    help="Show, cancel or resume a workflow (a dispatched plan).", no_args_is_help=True
)
paper_app = typer.Typer(
    help="Check or strip the \\hpcval provenance tags in a manuscript.", no_args_is_help=True
)
notify_app = typer.Typer(help="Notification channels.", no_args_is_help=True)
attempt_app = typer.Typer(help="Attempt-level human decisions.", no_args_is_help=True)
evidence_app = typer.Typer(
    help="Evidence bundles: aliases of verify/trace (write a bundle with `hpclab export`).",
    no_args_is_help=True,
)
app.add_typer(workflow_app, name="workflow")
app.add_typer(paper_app, name="paper")
app.add_typer(notify_app, name="notify")
app.add_typer(attempt_app, name="attempt")
app.add_typer(evidence_app, name="evidence")

_ATTEMPT_DONE_PHASES = frozenset({"terminal", "submission_rejected", "cancelled_before_submit", "abandoned"})
_WF_DONE_STATES = frozenset({"completed", "cancelled", "gate_failed"})


# ---------------------------------------------------------------------------
# small plumbing (tests monkeypatch _interactive / _sleep / _now)
# ---------------------------------------------------------------------------


def _interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _now() -> float:
    return time.monotonic()


def out(renderable: RenderableType) -> None:
    console.print(renderable)


def write(text: str) -> None:
    """Raw data on stdout; a reader that went away (``| head``) ends the command quietly."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
        raise typer.Exit(0) from None


def usage_error(message: str) -> typer.Exit:
    if STATE.json_output:
        print_json_error("invalid_argument", message)
    err_console.print(f"[red bold]usage[/red bold]: {escape(message)}")
    return typer.Exit(EXIT_USAGE)


def fail(
    message: str,
    *,
    hint: str | None = None,
    code: int = EXIT_ERROR,
    error: str = "precondition_failed",
    json_error: bool = True,
) -> typer.Exit:
    """A domain error detected by the CLI (``error`` is its Envelope error code in --json mode;
    ``json_error=False`` when the command already printed its own JSON document)."""
    if STATE.json_output and json_error:
        print_json_error(error, message, {"hint": hint} if hint else None)
    err_console.print(f"[red bold]error[/red bold]: {escape(message)}")
    if hint:
        err_console.print(f"  [yellow]hint[/yellow]: {escape(hint)}")
    return typer.Exit(code)


def note(message: str, style: str = "dim") -> None:
    """Side information on stderr (keeps stdout clean for data and --json)."""
    err_console.print(Text(message, style=style))


def stderr_notes(env: dict[str, Any]) -> None:
    """Warnings and incomplete-coverage notes on stderr (when stdout carries raw data)."""
    for w in env.get("warnings") or []:
        note(f"warning: {w}", "yellow")
    cov = env.get("coverage") or {}
    if cov and not cov.get("complete", True):
        bad = [s for s in cov.get("sources") or [] if s.get("status") not in ("ok", "cached", "skipped")]
        note(
            "incomplete: "
            + (
                "; ".join(
                    f"{s.get('name')}={s.get('status')}" + (f" ({s['detail']})" if s.get("detail") else "")
                    for s in bad
                )
                or incomplete_reason(cov)
            ),
            "yellow",
        )


def confirm(question: str, *, yes: bool, details: RenderableType | None = None) -> None:
    """Ask before spending GPU-hours or killing jobs. Without a terminal (or with --json) the
    caller must pass --yes: a script must never block on a prompt or proceed silently."""
    if yes:
        return
    if STATE.json_output or not _interactive():
        if details is not None:
            err_console.print(details)
        raise usage_error(f"{question} — refusing without --yes (no interactive terminal)")
    if details is not None:
        console.print(details)
    if not typer.confirm(question, default=False):
        err_console.print("aborted: nothing was done")
        raise typer.Exit(EXIT_ERROR)


def local_time(iso_text: Any) -> str:
    """HH:MM:SS local time of an ISO timestamp ('' when absent or unparsable)."""
    if not iso_text:
        return ""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(iso_text).replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def abspath(p: str | Path) -> str:
    """Absolute path: the controller's working directory is not the caller's."""
    return str(Path(p).expanduser().resolve())


def csv_list(text: str | None) -> list[str] | None:
    if not text:
        return None
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items or None


def _scalar(text: str) -> Any:
    v: Any = parse_kv([f"x={text}"])["x"]
    return v


def pairs(values: list[str] | None, what: str) -> list[tuple[str, str]]:
    """``KEY=REASON`` options (``--override``, ``--exclude``)."""
    result = []
    for v in values or []:
        key, sep, rest = v.partition("=")
        if not sep or not key.strip() or not rest.strip():
            raise usage_error(f"{what} expects KEY=REASON, got {v!r}")
        result.append((key.strip(), rest.strip()))
    return result


def _watch_interval(interval: float) -> float:
    return max(0.2, float(interval))


@contextmanager
def interruptible(message: str) -> Iterator[None]:
    """Ctrl-C ends a watch/follow/wait loop cleanly: nothing is cancelled."""
    try:
        yield
    except KeyboardInterrupt:
        err_console.print(f"\n[yellow]stopped[/yellow] {escape(message)}")
        raise typer.Exit(EXIT_INTERRUPTED) from None


# ---------------------------------------------------------------------------
# id resolution
# ---------------------------------------------------------------------------


def experiment_id(ref: str) -> str:
    """An experiment id, or the newest experiment with exactly this name."""
    k = kind_of(ref)
    if k == "experiment":
        return ref
    if k is not None:
        raise usage_error(f"expected an experiment id (exp_...) or name, got a {k} id: {ref}")
    env = fetch("experiments_list", {"name": ref, "limit": 100})
    exact = [e for e in (env.get("data") or {}).get("experiments") or [] if e.get("name") == ref]
    if not exact:
        raise fail(
            f"no experiment named {ref!r}", hint="list them with `hpclab experiments`", error="not_found"
        )
    if len(exact) > 1:
        note(
            f"{len(exact)} experiments are named {ref!r}; using the newest ({exact[0]['experiment_id']}). "
            "Pass an id to choose another."
        )
    return str(exact[0]["experiment_id"])


def run_set(ref: str) -> tuple[str, str]:
    """(kind, id) for an experiment / plan / workflow id or an experiment name."""
    k = kind_of(ref)
    if k in ("experiment", "plan", "workflow"):
        return str(k), ref
    return "experiment", experiment_id(ref)


def inspect_data(*, run_id: str | None = None, attempt_id: str | None = None) -> dict[str, Any]:
    env = fetch("run_inspect", {"run_id": run_id, "attempt_id": attempt_id})
    return dict(env.get("data") or {})


def resolve_attempt(ref: str, *, active: bool = False) -> tuple[str, dict[str, Any]]:
    """Attempt id for an attempt or run id (a run means its latest, or its active, attempt)."""
    k = kind_of(ref)
    if k == "attempt":
        return ref, inspect_data(attempt_id=ref)
    if k == "run":
        data = inspect_data(run_id=ref)
        run = data.get("run") or {}
        if active:
            aid = run.get("claim_attempt_id")
            if not aid:
                raise fail(f"run {ref} has no active attempt", hint=f"see `hpclab inspect {ref}`")
            return str(aid), data
        aid = data.get("focus_attempt_id")
        if not aid:
            raise fail(f"run {ref} has no attempts yet", hint=f"submit it: hpclab submit {ref}")
        return str(aid), data
    raise usage_error(f"expected an attempt id (att_...) or run id (run_...), got {ref!r}")


def attempt_view(data: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    for a in data.get("attempts") or []:
        if a.get("attempt_id") == attempt_id:
            return dict(a)
    raise fail(
        f"attempt {attempt_id} is not in the run's attempt listing",
        hint="hpclab events --entity " + attempt_id,
        error="not_found",
    )


def default_cluster(cluster: str | None) -> str:
    if cluster:
        return cluster
    env = fetch("clusters_list", {})
    ids = [c.get("cluster_id") for c in (env.get("data") or {}).get("clusters") or []]
    if len(ids) == 1:
        return str(ids[0])
    if not ids:
        raise fail(
            "no cluster registered",
            hint="hpclab init --sim, or hpclab cluster add <id> ...",
            error="not_found",
        )
    raise usage_error(f"several clusters are registered ({', '.join(map(str, ids))}); pass --cluster")


# ---------------------------------------------------------------------------
# prepare / experiments / plans
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Experiments")
def prepare(
    spec: Annotated[
        Path | None, typer.Argument(help="ExperimentSpec YAML (not needed with --extends/--derive-from).")
    ] = None,
    cluster: Annotated[str | None, typer.Option("--cluster", help="Override spec.cluster.")] = None,
    experiment: Annotated[str | None, typer.Option("--experiment", help="Override spec.name.")] = None,
    extends: Annotated[
        str | None, typer.Option("--extends", help="Experiment to add cells/seeds to.")
    ] = None,
    add_seeds: Annotated[
        str | None, typer.Option("--add-seeds", help="With --extends: seeds to add, e.g. 4,5.")
    ] = None,
    add: Annotated[
        list[str] | None,
        typer.Option("--add", help="With --extends: KEY=v1,v2 values to add to a matrix axis."),
    ] = None,
    derive_from: Annotated[
        str | None, typer.Option("--derive-from", help="Run to derive a new run from.")
    ] = None,
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="With --derive-from: param k=v (repeatable).")
    ] = None,
    time_: Annotated[str | None, typer.Option("--time", help="With --derive-from: new time limit.")] = None,
    mem: Annotated[str | None, typer.Option("--mem", help="With --derive-from: new memory.")] = None,
    append_arg: Annotated[
        list[str] | None,
        typer.Option("--append-arg", help="With --derive-from: extra argv item appended (repeatable)."),
    ] = None,
    recapture_code: Annotated[
        bool, typer.Option("--recapture-code", help="With --derive-from: snapshot the current code.")
    ] = False,
    no_remote_checks: Annotated[
        bool, typer.Option("--no-remote-checks", help="Skip bounded checks on the cluster.")
    ] = False,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="CHECK=REASON: accept a blocking finding (recorded)."),
    ] = None,
    all_findings: Annotated[bool, typer.Option("--all-findings", help="Also list passing checks.")] = False,
) -> None:
    """Freeze code + config into runs and execution plans (nothing is submitted)."""
    if spec is None and not extends and not derive_from:
        raise usage_error("give a spec YAML, or --extends EXP, or --derive-from RUN")
    if (add_seeds or add) and not extends:
        raise usage_error("--add-seeds/--add need --extends EXP")
    if (set_ or time_ or mem or recapture_code or append_arg) and not derive_from:
        raise usage_error("--set/--time/--mem/--append-arg/--recapture-code need --derive-from RUN")
    params: dict[str, Any] = {
        "idempotency_key": new_key("prepare"),
        "cluster_id": cluster,
        "experiment": experiment,
        "validate_remote": not no_remote_checks,
    }
    if spec is not None:
        if not spec.is_file():
            raise usage_error(f"spec file not found: {spec}")
        params["spec_path"] = abspath(spec)
    if extends:
        params["extends"] = experiment_id(extends)
        seeds = parse_csv_ints(add_seeds)
        if seeds:
            params["add_seeds"] = seeds
        if add:
            matrix: dict[str, list[Any]] = {}
            for item in add:
                key, sep, vals = item.partition("=")
                if not sep or not key.strip() or not vals.strip():
                    raise usage_error(f"--add expects KEY=v1,v2, got {item!r}")
                matrix.setdefault(key.strip(), []).extend(_scalar(v) for v in vals.split(",") if v.strip())
            params["add_matrix"] = matrix
    if derive_from:
        res = {k: v for k, v in (("time", time_), ("mem", mem)) if v}
        params["derive_from"] = {
            "run_id": derive_from,
            "overrides": {
                "params": parse_kv(set_ or []),
                "resources": res,
                "argv_append": list(append_arg or []),
            },
            "recapture_code": recapture_code,
        }
    ov = pairs(override, "--override")
    if ov:
        params["overrides"] = [{"check": c, "reason": r} for c, r in ov]

    def render(env: dict[str, Any]) -> None:
        data = env.get("data") or {}
        out(V.render_prepare(data, show_pass=all_findings))
        out(V.next_steps(V.prepare_next(data)))

    env = call("experiment_prepare", params, render=render, quiet=True)
    if (env.get("data") or {}).get("status") == "invalid":
        raise typer.Exit(EXIT_ERROR)


@app.command(rich_help_panel="Experiments")
def experiments(
    name: Annotated[str | None, typer.Option("--name", help="Substring filter on the name.")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """List experiments, newest first."""
    call(
        "experiments_list",
        {"name": name, "limit": limit, "cursor": cursor},
        render=lambda env: out(V.render_experiments(env.get("data") or {})),
        quiet=True,
    )


@app.command("experiment", rich_help_panel="Experiments")
def experiment_cmd(ref: Annotated[str, typer.Argument(help="Experiment id or name.")]) -> None:
    """One experiment: question, plans, run counts by cell code, budget."""
    call(
        "experiment_get",
        {"experiment_id": experiment_id(ref)},
        render=lambda env: out(V.render_experiment(env.get("data") or {})),
        quiet=True,
    )


@app.command("plan", rich_help_panel="Experiments")
def plan_cmd(
    plan_id: str,
    script: Annotated[
        bool, typer.Option("--script", help="Print the exact sbatch script a run would submit.")
    ] = False,
    run: Annotated[
        str | None, typer.Option("--run", help="With --script: which run (default: the first).")
    ] = None,
    all_findings: Annotated[bool, typer.Option("--all-findings")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """A frozen plan: runs, execution plans, validation — or its job script."""
    if run and not script:
        raise usage_error("--run only applies with --script")
    if not script:
        call(
            "plan_get",
            {"plan_id": plan_id, "limit": limit, "cursor": cursor},
            render=lambda env: out(V.render_plan(env.get("data") or {}, show_pass=all_findings)),
            quiet=True,
        )
        return
    params: dict[str, Any] = {"plan_id": plan_id, "run_id": run}
    env = fetch("plan_script", params)
    if STATE.json_output:
        print_json(env)
        return
    data = env.get("data") or {}
    chunks = [str(data.get("script") or "")]
    nxt = env.get("next_cursor")
    pages = 0
    while nxt and pages < 64:  # pages are <= 64 KiB; scripts are small, this is a safety bound
        more = fetch("plan_script", {**params, "cursor": nxt})
        chunks.append(str((more.get("data") or {}).get("script") or ""))
        nxt = more.get("next_cursor")
        pages += 1
    note(
        f"# {data.get('label')} ({data.get('run_id')}) exec plan {data.get('exec_plan_hash')} · "
        f"attempt id / token are placeholders ({data.get('attempt_id_placeholder')})"
    )
    write("".join(chunks))
    if chunks and not chunks[-1].endswith("\n"):
        write("\n")
    for w in env.get("warnings") or []:
        note(f"warning: {w}", "yellow")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _hours(time_limit: Any) -> float | None:
    from hpclab.domain.clock import parse_slurm_duration

    try:
        s = parse_slurm_duration(str(time_limit)) if time_limit else None
    except (ValueError, TypeError):
        return None
    return s / 3600.0 if s else None


def _budget_lines(experiment: str | None) -> list[RenderableType]:
    if not experiment:
        return []
    env = fetch("experiment_get", {"experiment_id": experiment}, allow_error=True)
    b = (env.get("data") or {}).get("budget") if env.get("ok") else None
    if not b:
        return [V.T("budget: unknown (could not read the experiment's budget)", "yellow")]
    if b.get("budget") is None:
        return [V.T("budget: no hpclab budget set for this experiment (site quotas still apply)", "dim")]
    line = V.kv("budget", f"{V.gpuh(b.get('available'))} available of {V.gpuh(b.get('budget'))}")
    if b.get("reserved"):
        line.append(f" ({V.gpuh(b.get('reserved'), unit=False)} reserved by active attempts)")
    if b.get("unknown_count"):
        line.append(f" · {b['unknown_count']} attempt(s) with unknown consumption", style="yellow")
    return [line]


def _plan_cost(plan: dict[str, Any], extra_runs: list[dict[str, Any]]) -> list[RenderableType]:
    """Lines 'N runs × G GPU × T = X GPU-h' grouped by kind and shape."""
    groups: dict[tuple[str, Any, Any, Any], list[float | None]] = {}
    for r in [*(plan.get("runs") or []), *extra_runs]:
        res = ((r.get("execution_plan") or {}).get("resources")) or {}
        key = (str(r.get("kind")), res.get("gpus"), res.get("gpu_type"), res.get("time"))
        groups.setdefault(key, []).append(r.get("requested_gpu_hours"))
    lines: list[RenderableType] = []
    for (kind, gpus, gtype, tl), vals in sorted(
        groups.items(), key=lambda kv_: (kv_[0][0] != "smoke", kv_[0][0])
    ):
        known = [v for v in vals if isinstance(v, (int, float))]
        total = sum(known) if len(known) == len(vals) else None
        lines.append(
            V.T(
                f"  {kind}: {len(vals)} run(s) × {gpus if gpus is not None else '?'} GPU"
                f"{'' if gpus == 1 else 's'}{f' ({gtype})' if gtype else ''} × {tl or '?'} = {V.gpuh(total)}"
            )
        )
    b = plan.get("budget") or {}
    shown = len(plan.get("runs") or []) + len(extra_runs)
    total_runs = plan.get("runs_total")
    if isinstance(total_runs, int) and total_runs > shown:
        lines.append(V.T(f"  … {total_runs - shown} more run(s) not itemized", "dim"))
    lines.append(
        V.kv(
            "total",
            f"{V.gpuh(b.get('requested_gpu_hours_total'))} at most (time limit × GPUs; reserved as cells are "
            "admitted, settled from accounting when they end)",
        )
    )
    return lines


def _recovery_cost(cells: list[Any]) -> list[RenderableType]:
    """Lines 'strategy kind: N cell(s) = X GPU-h' for the eligible cells of a recovery plan."""
    groups: dict[tuple[str, str], list[Any]] = {}
    for c in cells:
        if c.get("eligible"):
            groups.setdefault((str(c.get("strategy") or "?"), str(c.get("kind") or "?")), []).append(c)
    lines: list[RenderableType] = []
    for (strategy, kind), group in sorted(groups.items(), key=lambda kv_: (kv_[0][1] != "smoke", kv_[0])):
        lines.append(
            V.T(f"  {strategy} {kind}: {len(group)} cell(s) = {V.gpuh(V.recovery_totals(group)[2])}")
        )
    _, blocked, total = V.recovery_totals(cells)
    if blocked:
        lines.append(V.T(f"  {blocked} cell(s) not eligible: not dispatched", "yellow"))
    lines.append(
        V.kv(
            "total",
            f"{V.gpuh(total)} at most (time limit × GPUs; reserved as cells are admitted, settled from "
            "accounting when they end)",
        )
    )
    return lines


def _plan_with_runs(plan_id: str, max_pages: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = fetch("plan_get", {"plan_id": plan_id, "limit": 100})
    plan = dict(env.get("data") or {})
    extra: list[dict[str, Any]] = []
    nxt = env.get("next_cursor")
    pages = 0
    while nxt and pages < max_pages:
        more = fetch("plan_get", {"plan_id": plan_id, "limit": 100, "cursor": nxt})
        extra.extend((more.get("data") or {}).get("runs") or [])
        nxt = more.get("next_cursor")
        pages += 1
    return plan, extra


def _pick_exec_plan(run_id: str, data: dict[str, Any]) -> tuple[str | None, str, str | None]:
    """The execution plan ``submit RUN`` uses: the run's own plan while unused, else the newest
    unused plan `hpclab recover` made for its latest attempt (from the audit trail).
    Returns (exec plan hash, where it came from, recovery plan id when known)."""
    run = data.get("run") or {}
    attempts = data.get("attempts") or []
    used = {a.get("exec_plan_hash") for a in attempts}
    cur = run.get("current_exec_plan_hash")
    if cur and cur not in used:
        return str(cur), "the run's execution plan", None
    latest = max(attempts, key=lambda a: int(a.get("seq") or 0), default=None)
    if latest is None:
        return None, "", None
    env = fetch(
        "events_list",
        {"entity_id": latest.get("attempt_id"), "event_type": "recovery.planned", "limit": 10},
        allow_error=True,
    )
    for e in (env.get("data") or {}).get("events") or [] if env.get("ok") else []:
        payload = e.get("payload") or {}
        h = payload.get("exec_plan_hash")
        if h and h not in used and payload.get("run_id") in (None, run_id):
            source = f"the {payload.get('strategy') or 'recovery'} plan from `hpclab recover {latest.get('attempt_id')}`"
            return str(h), source, (str(payload["plan_id"]) if payload.get("plan_id") else None)
    return None, "", None


def _recovery_cell_hours(plan_id: str | None, exec_plan_hash: str) -> float | None:
    """GPU-hours the recovery plan froze for this execution plan (None when unknown)."""
    if not plan_id:
        return None
    env = fetch("plan_get", {"plan_id": plan_id, "limit": 1}, allow_error=True)
    if not env.get("ok"):
        return None
    for c in V.recovery_cells(env.get("data") or {}):
        h = c.get("requested_gpu_hours")
        if c.get("exec_plan_hash") == exec_plan_hash and isinstance(h, (int, float)):
            return float(h)
    return None


def _submit_run(run_id: str, *, exec_plan: str | None, yes: bool, force: bool, reason: str | None) -> None:
    data = inspect_data(run_id=run_id)
    run = data.get("run") or {}
    if run.get("claim_attempt_id"):
        raise fail(
            f"run {run_id} already has an active attempt ({run['claim_attempt_id']})",
            hint=f"hpclab inspect {run_id}  (one execution per run; cancel or wait first)",
        )
    attempts = data.get("attempts") or []
    recovery_plan: str | None = None
    if exec_plan:
        target, source = exec_plan, "given with --exec-plan"
        used = next((a for a in attempts if a.get("exec_plan_hash") == target), None)
        if used is not None:
            raise fail(
                f"execution plan {V.short_digest(target, 16)} was already used by attempt #{used.get('seq')} "
                f"({used.get('attempt_id')}); an execution plan runs at most once",
                hint=f"create a new one: hpclab recover {used.get('attempt_id')}",
            )
    else:
        picked, source, recovery_plan = _pick_exec_plan(run_id, data)
        if picked is None:
            latest: dict[str, Any] = max(attempts, key=lambda a: int(a.get("seq") or 0), default={})
            raise fail(
                f"run {run_id} has no unused execution plan (attempt #{latest.get('seq')} used the last one)",
                hint=f"plan the next attempt first: hpclab recover {latest.get('attempt_id')}",
            )
        target = picked
    refs = data.get("refs") or {}
    lines: list[RenderableType] = [
        V.kv(
            "submit",
            f"run {run.get('label')} ({run_id}) · exec plan {V.short_digest(target, 16)} ({source})",
        )
    ]
    if refs.get("exec_plan_hash") == target and refs.get("resources"):
        res = refs["resources"]
        hours = _hours(res.get("time"))
        gpus = res.get("gpus")
        est = hours * gpus if hours is not None and isinstance(gpus, int) else None
        lines.append(
            V.T(
                f"  1 run × {gpus if gpus is not None else '?'} GPU(s){f' ({res.get("gpu_type")})' if res.get('gpu_type') else ''}"
                f" × {res.get('time') or '?'} = {V.gpuh(est)} at most (reserved at admission)"
            )
        )
    elif (hours := _recovery_cell_hours(recovery_plan, target)) is not None:
        lines.append(
            V.T(
                f"  1 run = {V.gpuh(hours)} at most (time limit × GPUs of the recovery plan {recovery_plan}; "
                "reserved at admission)"
            )
        )
    else:
        lines.append(
            V.T(
                "  GPU-hours: not shown here (this is a recovery/derived execution plan: see the `hpclab recover` "
                "output); the receipt reports the reservation",
                "yellow",
            )
        )
    lines.extend(_budget_lines(run.get("experiment_id")))
    if force:
        lines.append(V.T(f"bypassing the smoke gate: {reason}", "bold yellow"))
    confirm("Submit this run (spends GPU-hours)?", yes=yes, details=Group(*lines))
    if yes and not STATE.json_output:
        out(Group(*lines))

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_submit_receipt(d))
        out(
            V.next_steps(
                [
                    f"hpclab ops {d.get('operation_id')} --wait",
                    f"hpclab inspect {run_id}",
                    f"hpclab logs {d.get('attempt_id')} --follow",
                ],
                title="watch",
            )
        )

    call(
        "run_submit",
        {
            "run_id": run_id,
            "execution_plan_hash": target,
            "idempotency_key": new_key("submit"),
            "force_without_gate": force,
            "reason": reason,
        },
        render=render,
    )


def _submit_workflow(kind: str, ref: str, *, max_parallel: int | None, yes: bool) -> None:
    if kind == "experiment":
        exp = fetch("experiment_get", {"experiment_id": ref}).get("data") or {}
        # workflow_submit's own rule: the latest ready matrix/derived plan (a recovery or probe
        # plan from `hpclab recover` is submitted by its plan id, never picked implicitly)
        ready = [
            p
            for p in exp.get("plans") or []
            if p.get("status") == "ready" and p.get("plan_hash") and p.get("kind") in ("matrix", "derived")
        ]
        if not ready:
            states = (
                ", ".join(f"{p.get('plan_id')}={p.get('status')}" for p in exp.get("plans") or []) or "none"
            )
            raise fail(f"experiment {ref} has no ready plan ({states})", hint="hpclab prepare ... first")
        chosen = max(ready, key=lambda p: str(p.get("created_at") or ""))
        plan_id = str(chosen["plan_id"])
    else:
        plan_id = ref
    plan, extra = _plan_with_runs(plan_id)
    status = plan.get("status")
    if status != "ready" or not plan.get("plan_hash"):
        if status == "preparing":
            op = plan.get("operation_id")
            raise fail(
                f"plan {plan_id} is still preparing",
                hint=f"hpclab ops {op} --wait" if op else f"hpclab plan {plan_id}",
                error="not_ready",
            )
        raise fail(
            f"plan {plan_id} is {status}: it cannot be dispatched",
            hint=f"hpclab plan {plan_id} --all-findings",
        )
    cells = [] if plan.get("runs") else V.recovery_cells(plan)
    what = f"{V.recovery_totals(cells)[0]} cell(s)" if cells else f"{plan.get('runs_total')} run(s)"
    lines: list[RenderableType] = [
        V.kv(
            "submit",
            f"plan {plan_id} ({plan.get('kind')}) of experiment {plan.get('experiment_id')} · {what}",
        )
    ]
    lines.extend(_recovery_cost(cells) if cells else _plan_cost(plan, extra))
    lines.extend(_budget_lines(plan.get("experiment_id")))
    if plan.get("smoke_run_id") or any(c.get("kind") == "smoke" and c.get("eligible") for c in cells):
        lines.append(V.T("  the smoke run goes first; the matrix is dispatched only if it passes", "dim"))
    confirm("Dispatch this plan (spends GPU-hours)?", yes=yes, details=Group(*lines))
    if yes and not STATE.json_output:
        out(Group(*lines))
    params: dict[str, Any] = {
        "idempotency_key": new_key("workflow"),
        "plan_hash": plan["plan_hash"],
        "max_parallel": max_parallel,
    }
    if kind == "experiment":
        params["prepared_experiment_id"] = ref
    else:
        params["plan_id"] = plan_id

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_workflow_submit(d))
        wid = d.get("workflow_id")
        out(V.next_steps([f"hpclab grid {wid} --watch", f"hpclab workflow show {wid}"], title="watch"))

    call("workflow_submit", params, render=render, quiet=True)


@app.command(rich_help_panel="Runs")
def submit(
    target: Annotated[str, typer.Argument(help="run id, plan id, experiment id or experiment name.")],
    max_parallel: Annotated[
        int | None, typer.Option("--max-parallel", min=1, help="Plans: at most N cells at once.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
    exec_plan: Annotated[
        str | None,
        typer.Option(
            "--exec-plan",
            help="Runs: execution plan hash (default: the run's unused plan, else the newest from `hpclab recover`).",
        ),
    ] = None,
    force_without_gate: Annotated[
        bool, typer.Option("--force-without-gate", help="Runs: submit before the smoke gate passed.")
    ] = False,
    reason: Annotated[
        str | None, typer.Option("--reason", help="Why the gate is bypassed (recorded).")
    ] = None,
) -> None:
    """Submit one run, or dispatch a whole plan (smoke gate first, then the matrix)."""
    k = kind_of(target)
    if force_without_gate and not (reason or "").strip():
        raise usage_error("--force-without-gate needs --reason (it is recorded)")
    if k == "run":
        if max_parallel is not None:
            raise usage_error("--max-parallel applies to plans and experiments")
        _submit_run(target, exec_plan=exec_plan, yes=yes, force=force_without_gate, reason=reason)
        return
    if force_without_gate or exec_plan:
        raise usage_error("--force-without-gate/--exec-plan apply to single runs (submit a run id)")
    if k in ("plan", "experiment"):
        _submit_workflow(str(k), target, max_parallel=max_parallel, yes=yes)
        return
    if k is not None:
        raise usage_error(f"cannot submit a {k} id; pass a run, plan or experiment")
    _submit_workflow("experiment", experiment_id(target), max_parallel=max_parallel, yes=yes)


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------


def _workflow_version(workflow_id: str) -> dict[str, Any]:
    return dict(fetch("workflow_get", {"workflow_id": workflow_id, "limit": 1}).get("data") or {})


@workflow_app.command("show")
def workflow_show(
    workflow_id: str,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Dispatch state, gate, breaker, budget and per-cell states."""

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_workflow(d))
        cmds = [f"hpclab grid {workflow_id}"]
        if d.get("dispatch_state") == "paused":
            cmds.append(f"hpclab workflow resume {workflow_id}   # after fixing the cause")
        if d.get("state") not in _WF_DONE_STATES:
            cmds.append(f"hpclab workflow cancel {workflow_id}")
        out(V.next_steps(cmds))

    call(
        "workflow_get",
        {"workflow_id": workflow_id, "limit": limit, "cursor": cursor},
        render=render,
        quiet=True,
    )


def _with_version_retry(
    method: str, build: Callable[[int], dict[str, Any]], version: int, version_of: Callable[[], int]
) -> dict[str, Any]:
    """Call a versioned mutation; one automatic retry when the version moved underneath us
    (background observation/dispatch bumps versions all the time)."""
    for attempt in range(2):
        env = fetch(method, build(version), allow_error=True)
        err = env.get("error") or {}
        if env.get("ok") or err.get("code") != "version_conflict" or attempt == 1:
            return env
        cur = (err.get("details") or {}).get("current_version")
        version = int(cur) if isinstance(cur, int) else version_of()
    return env


@workflow_app.command("cancel")
def workflow_cancel(
    workflow_id: str,
    all_: Annotated[
        bool, typer.Option("--all", help="Also cancel queued/running jobs (identity-checked).")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Stop dispatching (default), or also cancel every active job of the workflow (--all)."""
    wf = _workflow_version(workflow_id)
    scope = "all_managed" if all_ else "queued_only"
    what = (
        "stop dispatch AND cancel every queued/running job of this workflow (each job's identity is re-verified "
        "right before scancel; mismatches are left alone)"
        if all_
        else "stop dispatching new cells (jobs already queued or running keep running)"
    )
    confirm(
        f"Cancel workflow {workflow_id}?",
        yes=yes,
        details=Group(V.workflow_line(wf), V.T(f"this will {what}", "bold yellow" if all_ else "")),
    )
    env = _with_version_retry(
        "workflow_cancel",
        lambda v: {
            "workflow_id": workflow_id,
            "expected_version": v,
            "scope": scope,
            "idempotency_key": new_key("wfcancel"),
        },
        int(wf.get("version") or 0),
        lambda: int(_workflow_version(workflow_id).get("version") or 0),
    )

    def render(e: dict[str, Any]) -> None:
        d = e.get("data") or {}
        out(V.render_workflow_cancel(d))
        if d.get("operation_id"):
            out(V.next_steps([f"hpclab ops {d['operation_id']} --wait", f"hpclab grid {workflow_id}"]))

    show(env, render=render)
    raise typer.Exit(exit_code_for(env))


@workflow_app.command("resume")
def workflow_resume(workflow_id: str) -> None:
    """Re-open dispatch paused by the circuit breaker (fix the cause first); on a gate_failed workflow, re-evaluate the smoke gate on the smoke run's latest attempt (e.g. after recovering the smoke run)."""
    wf = _workflow_version(workflow_id)
    if not STATE.json_output:
        out(V.workflow_line(wf))
    env = _with_version_retry(
        "workflow_submit",
        lambda v: {
            "workflow_id": workflow_id,
            "resume_dispatch": True,
            "expected_version": v,
            "idempotency_key": new_key("wfresume"),
        },
        int(wf.get("version") or 0),
        lambda: int(_workflow_version(workflow_id).get("version") or 0),
    )

    def render(e: dict[str, Any]) -> None:
        out(V.render_workflow_submit(e.get("data") or {}))
        out(V.next_steps([f"hpclab grid {workflow_id} --watch"]))

    show(env, render=render, quiet=True)
    raise typer.Exit(exit_code_for(env))


# ---------------------------------------------------------------------------
# runs / grid
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Runs")
def runs(
    experiment: Annotated[str | None, typer.Option("--experiment", help="Experiment id or name.")] = None,
    workflow: Annotated[str | None, typer.Option("--workflow")] = None,
    state: Annotated[
        list[str] | None, typer.Option("--state", help="Run status filter (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Runs with their latest attempt: code, scheduler state, time left, progress, health."""
    if bool(experiment) == bool(workflow):
        raise usage_error("pass exactly one of --experiment or --workflow")
    params: dict[str, Any] = {
        "experiment_id": experiment_id(experiment) if experiment else None,
        "workflow_id": workflow,
        "states": state or None,
        "limit": limit,
        "cursor": cursor,
    }
    call("runs_list", params, render=lambda env: out(V.render_runs(env.get("data") or {})), quiet=True)


def _grid_params(
    target: str, metric: str | None, rows: str | None, cols: str | None
) -> tuple[dict[str, Any], str | None]:
    kind, ident = run_set(target)
    params: dict[str, Any] = {
        "view": "grid",
        "metric": metric,
        "rows": csv_list(rows),
        "cols": csv_list(cols),
    }
    if kind == "workflow":
        params["workflow_id"] = ident
        return params, ident
    if kind == "plan":
        raise usage_error(
            "grid takes an experiment or workflow (a plan is dispatched as a workflow: see `hpclab submit`)"
        )
    params["experiment_id"] = ident
    return params, None


#: How long ``grid --watch`` keeps waiting, after the last change, once nothing is active any
#: more but cells may still change on their own: E cells (auto-evaluation normally follows within
#: seconds), a smoke run whose outputs are being checked and, for an experiment grid, P cells
#: some workflow might still dispatch.
_SETTLE_GRACE_S = 60.0
_WF_OPEN_STATES = frozenset({"gating", "dispatching", "draining"})
# the service's note on a smoke run that has not passed: "smoke run <label> (<run id>): <code> ..."
_SMOKE_NOTE_RE = re.compile(r"smoke run .*?\((run_\w+)\): (\S+)")
# the service's note on E runs nothing will evaluate: "...; <n> of them has/have no evaluation queued"
_UNQUEUED_E_RE = re.compile(r"run\(s\) are E: .*?; (\d+) of them (?:has|have) no evaluation queued")


@dataclass(frozen=True)
class _GridWatch:
    """What a grid snapshot means for ``--watch``: ``done`` (stop now, with ``reason``) or
    ``settling`` (nothing active; stop after the grace period unless it changes)."""

    done: bool
    settling: bool = False
    reason: str = ""
    unevaluated: int = 0
    undispatched: int = 0
    dispatch_blocked: bool = False
    smoke_checking: tuple[str, ...] = ()  # smoke runs completed, outputs not validated yet (E)
    key: tuple[Any, ...] = ()  # what changes when the grid progresses (restarts the grace)


def _watch_counts(data: dict[str, Any]) -> dict[str, int]:
    """Runs per code over the WHOLE grid: the service's ``counts`` cover every run, whatever the
    page shows (rows are paginated) and inside aggregated cells (a cell shows its worst code, so
    an F cell can hide a run still running). Falls back to the codes of the cells shown."""
    counts = data.get("counts")
    if isinstance(counts, dict):
        codes = {
            str(k): int(v)
            for k, v in counts.items()
            if k not in ("planned", "valid") and isinstance(v, int) and v > 0
        }
        if codes:
            return codes
    tally: dict[str, int] = {}
    for code in V.grid_codes(data):
        if code != "-":
            tally[code] = tally.get(code, 0) + 1
    return tally


def _smoke_states(notes: list[str]) -> list[tuple[str, str]]:
    """(run id, code) of the smoke runs that have not passed (the grid shows main runs only)."""
    out = []
    for n in notes:
        m = _SMOKE_NOTE_RE.match(n)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _unqueued_evaluations(notes: list[str]) -> int:
    """E runs with no evaluation queued (none declared, or the automatic one was skipped): they
    never turn V on their own."""
    for n in notes:
        m = _UNQUEUED_E_RE.search(n)
        if m:
            return int(m.group(1))
    return 0


def _p_cells_blocked(wf: dict[str, Any] | None, notes: list[str]) -> str | None:
    """Why no P cell can still be dispatched, or None if some might (or we cannot tell)."""
    if wf is not None:
        state = str(wf.get("state") or "")
        if state and state not in _WF_OPEN_STATES:
            return f"workflow {state}"
        if wf.get("dispatch_state") == "paused":
            return "dispatch paused (circuit breaker): needs a decision"
        if wf.get("dispatch_state") == "closed":
            return "workflow dispatch closed"
        if wf.get("dispatch_blocked_reason") and not wf.get("slots_used"):
            return f"dispatch blocked: {wf['dispatch_blocked_reason']}"
        return None
    for n in notes:  # experiment grid: the service says when a workflow stopped for good
        if "will not start on their own" in n:
            return n.partition(" - ")[0]
    return None


def _grid_watch_state(env: dict[str, Any], wf: dict[str, Any] | None) -> _GridWatch:
    data = env.get("data") or {}
    counts = _watch_counts(data)
    notes = [str(n) for n in (data.get("grid") or {}).get("notes") or []]
    smoke = _smoke_states(notes)
    key = (tuple(sorted(counts.items())), tuple(smoke))
    if (
        not counts
        or any(counts.get(c) for c in V.ACTIVE_CODES)
        or any(code in V.ACTIVE_CODES for _, code in smoke)  # the smoke run gating the P cells
    ):
        return _GridWatch(done=False, key=key)
    unevaluated = counts.get("E", 0)
    undispatched = counts.get("P", 0)
    if not unevaluated and not undispatched:
        return _GridWatch(done=True, reason="every cell is terminal", key=key)
    blocked = _p_cells_blocked(wf, notes)
    # P cells waiting for a smoke run whose outputs are still being checked: like an E cell
    checking = tuple(rid for rid, code in smoke if code == "E") if undispatched and not blocked else ()
    if undispatched and blocked is None and not checking:
        if wf is not None:
            return _GridWatch(done=False, key=key)  # the workflow can still dispatch them
        # an experiment grid cannot see its workflows: wait a while for a dispatch round
        return _GridWatch(
            done=False, settling=True, unevaluated=unevaluated, undispatched=undispatched, key=key
        )
    # E cells (and a smoke run being checked) get the grace period, unless none of them has an
    # evaluation queued (they would wait forever); blocked P cells never start
    evaluating = unevaluated > _unqueued_evaluations(notes)
    waiting = bool(evaluating or checking)
    if undispatched and blocked:
        reason = f"{blocked}: {undispatched} cell(s) will not be dispatched"
    else:
        reason = "" if waiting else "nothing is queued, running or being evaluated"
    return _GridWatch(
        done=not waiting,
        settling=waiting,
        reason=reason,
        unevaluated=unevaluated,
        undispatched=undispatched,
        dispatch_blocked=bool(undispatched and blocked),
        smoke_checking=checking,
        key=key,
    )


def _settled_notes(state: _GridWatch, ident: str, *, is_workflow: bool) -> list[str]:
    notes = []
    if state.unevaluated:
        notes.append(f"{state.unevaluated} cell(s) completed but not evaluated: run hpclab collect {ident}")
    if state.smoke_checking:
        for rid in state.smoke_checking:
            notes.append(
                f"the smoke run {rid} completed but its outputs are not validated yet, and the "
                f"{state.undispatched} planned cell(s) start only after it passes: hpclab inspect {rid}"
            )
    elif state.undispatched and not state.dispatch_blocked:
        where = f"hpclab workflow show {ident}" if is_workflow else f"hpclab submit {ident}"
        notes.append(
            f"{state.undispatched} cell(s) not dispatched and nothing is running: they start only through a "
            f"workflow ({where})"
        )
    return notes


@app.command(rich_help_panel="Runs")
def grid(
    target: Annotated[str, typer.Argument(help="Experiment id/name or workflow id.")],
    metric: Annotated[str | None, typer.Option("--metric", help="Show this metric in valid cells.")] = None,
    rows: Annotated[str | None, typer.Option("--rows", help="Row axes, e.g. method,lr.")] = None,
    cols: Annotated[str | None, typer.Option("--cols", help="Column axes (default seed).")] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch", "-w", help="Refresh until nothing more happens on its own (e.g. every cell terminal)."
        ),
    ] = False,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between refreshes.")] = 5.0,
    timeout: Annotated[
        float | None, typer.Option("--timeout", help="With --watch: give up after N seconds (exit 1).")
    ] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Method × seed grid with one code per cell (V R Q S P E F C! U X ?)."""
    params, wf_id = _grid_params(target, metric, rows, cols)
    ident = wf_id or str(params.get("experiment_id"))
    if cursor:
        params["cursor"] = cursor

    def once() -> tuple[dict[str, Any], dict[str, Any] | None]:
        env = fetch("runs_list", params)
        wf = None
        if wf_id:
            wenv = fetch("workflow_get", {"workflow_id": wf_id, "limit": 1}, allow_error=True)
            wf = wenv.get("data") if wenv.get("ok") else None
        return env, wf

    def renderable(env: dict[str, Any], wf: dict[str, Any] | None) -> RenderableType:
        stamp = local_time(env.get("observed_at"))
        title = f"{target}  ·  as of {stamp}" if stamp else target
        return V.render_grid(env.get("data") or {}, workflow=wf, title=title, target=ident)

    if not watch:
        env, wf = once()
        if STATE.json_output:
            print_json(env)
            return
        out(renderable(env, wf))
        show({**env, "summary": "", "data": None})  # warnings / incomplete coverage / cursor
        return

    step = _watch_interval(interval)
    started = _now()
    live_ok = console.is_terminal and not STATE.json_output
    settle_since: float | None = None
    settle_key: tuple[Any, ...] = ()

    def finished(env: dict[str, Any], wf: dict[str, Any] | None) -> _GridWatch | None:
        """The final state when the watch should stop now, else None."""
        nonlocal settle_since, settle_key
        st = _grid_watch_state(env, wf)
        if st.done:
            return st
        if not st.settling:
            settle_since = None
            return None
        if settle_since is None or st.key != settle_key:  # the grace counts from the last change
            settle_since = _now()
            settle_key = st.key
        if _now() - settle_since < _SETTLE_GRACE_S:
            return None
        return replace(st, done=True, reason=st.reason or "nothing is queued or running any more")

    final: _GridWatch | None = None
    with interruptible("watching (the jobs keep running)"):
        env, wf = once()
        if live_ok:
            from rich.live import Live

            with Live(renderable(env, wf), console=console, auto_refresh=False, transient=False) as live:
                while (final := finished(env, wf)) is None:
                    if timeout is not None and _now() - started >= timeout:
                        break
                    _sleep(step)
                    env, wf = once()
                    live.update(renderable(env, wf), refresh=True)
        else:
            while (final := finished(env, wf)) is None:
                if not STATE.json_output:
                    err_console.print(
                        V.counts_line((env.get("data") or {}).get("counts"), label=time.strftime("%H:%M:%S"))
                    )
                if timeout is not None and _now() - started >= timeout:
                    break
                _sleep(step)
                env, wf = once()
            if not STATE.json_output:
                out(renderable(env, wf))
    if STATE.json_output:
        print_json(env)
    else:
        show({**env, "summary": "", "data": None})
        if final is None:
            note(f"timed out after {timeout:g}s (the jobs keep running)", "yellow")
        else:
            if final.reason:
                note(f"done: {final.reason}", "bold")
            for line in _settled_notes(final, ident, is_workflow=wf_id is not None):
                note(line, "yellow")
    if final is None:
        raise typer.Exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# inspect / logs
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Runs")
def inspect(
    ref: Annotated[str, typer.Argument(help="Run id or attempt id.")],
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ask the cluster now (bounded) instead of the last observation.")
    ] = False,
    gpu: Annotated[
        bool,
        typer.Option("--gpu", help="Sample nvidia-smi inside the running job's allocation (rate-limited)."),
    ] = False,
) -> None:
    """Everything about one run: attempts (3 state axes + job), pending reason, progress/ETA, health, diagnosis."""
    k = kind_of(ref)
    if k not in ("run", "attempt"):
        raise usage_error(f"expected a run id (run_...) or attempt id (att_...), got {ref!r}")
    params = {
        "run_id": ref if k == "run" else None,
        "attempt_id": ref if k == "attempt" else None,
        "refresh": refresh or None,
        "probe": "gpu" if gpu else None,
    }
    call("run_inspect", params, render=lambda env: out(V.render_inspect(env.get("data") or {})), quiet=True)


@app.command(rich_help_panel="Runs")
def logs(
    ref: Annotated[str, typer.Argument(help="Attempt id, or run id (its latest attempt).")],
    stderr: Annotated[bool, typer.Option("--stderr", help="Read stderr instead of stdout.")] = False,
    tail: Annotated[int | None, typer.Option("--tail", "-n", min=1, help="Only the last N lines.")] = None,
    follow: Annotated[
        bool, typer.Option("--follow", "-f", help="Keep reading new output until the attempt ends.")
    ] = False,
    cursor: Annotated[str | None, typer.Option("--cursor", help="Continue from a previous read.")] = None,
    max_bytes: Annotated[int | None, typer.Option("--max-bytes", help="Bytes per read (<= 65536).")] = None,
    interval: Annotated[
        float, typer.Option("--interval", help="With --follow: seconds between reads at the end of the log.")
    ] = 2.0,
) -> None:
    """Bounded, cursor-based log reads (rotation- and UTF-8-safe)."""
    if tail and cursor:
        raise usage_error("--tail and --cursor are exclusive")
    attempt_id = ref if kind_of(ref) == "attempt" else resolve_attempt(ref)[0]
    stream = "stderr" if stderr else "stdout"
    base = {"attempt_id": attempt_id, "stream": stream, "max_bytes": max_bytes}

    last_note: list[str] = [""]

    def emit(env: dict[str, Any]) -> dict[str, Any]:
        d = env.get("data") or {}
        if STATE.json_output:
            print_json(env, compact=follow)
            return d
        if d.get("rotated"):
            note("log was rotated/replaced: reading restarted from its beginning", "yellow")
        text = str(d.get("text") or "")
        if text:
            write(text)
        if d.get("note") and d["note"] != last_note[0]:
            note(str(d["note"]))
            last_note[0] = str(d["note"])
        return d

    env = fetch("run_logs", {**base, "tail_lines": tail, "cursor": cursor})
    d = emit(env)
    if not follow:
        nxt = env.get("next_cursor")
        if nxt and not STATE.json_output and (d.get("truncated") or not d.get("eof")):
            note(f"more: hpclab logs {attempt_id}{' --stderr' if stderr else ''} --cursor {nxt}")
        return
    step = _watch_interval(interval)
    with interruptible("following the log (the job keeps running)"):
        while True:
            nxt = env.get("next_cursor")
            if not nxt:
                if not STATE.json_output:
                    note(f"[{attempt_id} has ended; end of {stream}]")
                return
            if d.get("eof") or not d.get("text"):
                _sleep(step)
            env = fetch("run_logs", {**base, "cursor": nxt})
            d = emit(env)


# ---------------------------------------------------------------------------
# cancel / recover / resolve
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Runs")
def cancel(
    ref: Annotated[str, typer.Argument(help="Attempt id, or run id (its active attempt).")],
    checkpoint: Annotated[
        bool,
        typer.Option(
            "--checkpoint", help="Ask the job to checkpoint and exit (checkpoint_and_exit runs only)."
        ),
    ] = False,
    reason: Annotated[str | None, typer.Option("--reason", help="Why (recorded).")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Identity-verified cancel of one attempt (nothing is cancelled on any mismatch)."""
    attempt_id, data = resolve_attempt(ref, active=True)
    a = attempt_view(data, attempt_id)
    run = data.get("run") or {}
    job = a.get("job") or {}
    desc = V.T(
        f"attempt {attempt_id} of run {run.get('label')} · {a.get('phase')} · {a.get('scheduler_state')}"
        f" · job {job.get('cluster_id')}:{job.get('job_id') or '(not identified yet)'}"
    )
    what = (
        "send the declared checkpoint signal (the job saves and exits)"
        if checkpoint
        else "cancel it (job id, owner, token and submit time are re-verified right before scancel)"
    )
    confirm(
        f"{'Checkpoint' if checkpoint else 'Cancel'} this attempt?",
        yes=yes,
        details=Group(desc, V.T("this will " + what)),
    )
    env = _with_version_retry(
        "run_cancel",
        lambda v: {
            "attempt_id": attempt_id,
            "expected_version": v,
            "idempotency_key": new_key("cancel"),
            "reason": reason,
            "mode": "request_checkpoint" if checkpoint else "cancel",
        },
        int(a.get("version") or 0),
        lambda: int(attempt_view(inspect_data(attempt_id=attempt_id), attempt_id).get("version") or 0),
    )

    def render(e: dict[str, Any]) -> None:
        d = e.get("data") or {}
        line = V.kv(
            "cancel",
            f"{d.get('cancel_state')} · phase {d.get('phase')} · mode {d.get('cancel_mode') or ('request_checkpoint' if checkpoint else 'cancel')}",
        )
        out(line)
        if d.get("note"):
            out(V.T(d["note"], "dim"))
        cmds = [f"hpclab ops {d['operation_id']} --wait"] if d.get("operation_id") else []
        cmds.append(f"hpclab inspect {d.get('run_id') or run.get('run_id')} --refresh")
        out(V.next_steps(cmds))

    show(env, render=render, quiet=True)
    raise typer.Exit(exit_code_for(env))


@app.command(rich_help_panel="Runs")
def recover(
    ref: Annotated[
        str | None, typer.Argument(help="Failed attempt id (or run id: its latest attempt).")
    ] = None,
    workflow: Annotated[
        str | None, typer.Option("--workflow", help="Plan recovery of a workflow's broken cells instead.")
    ] = None,
    missing: Annotated[
        bool, typer.Option("--missing", help="With --workflow: every cell without a valid result (default).")
    ] = False,
    diagnosis: Annotated[
        list[str] | None,
        typer.Option(
            "--diagnosis", help="With --workflow: only cells with this diagnosis code (repeatable)."
        ),
    ] = None,
    strategy: Annotated[
        str | None,
        typer.Option("--strategy", help="retry | resume | adjust_resources (default: from the diagnosis)."),
    ] = None,
    time_: Annotated[str | None, typer.Option("--time", help="New time limit (adjust_resources).")] = None,
    mem: Annotated[str | None, typer.Option("--mem", help="New memory (adjust_resources).")] = None,
    partition: Annotated[
        str | None, typer.Option("--partition", help="New partition (adjust_resources).")
    ] = None,
    gpu_type: Annotated[
        str | None,
        typer.Option("--gpu-type", help="New GPU type, e.g. a GPU with more memory after cuda_oom."),
    ] = None,
    gpus: Annotated[
        int | None, typer.Option("--gpus", min=0, max=64, help="New GPU count (adjust_resources).")
    ] = None,
    exclude_nodes: Annotated[
        str | None,
        typer.Option("--exclude-nodes", help="Nodes to avoid, comma-separated (e.g. after node_fail)."),
    ] = None,
    account: Annotated[str | None, typer.Option("--account", help="New Slurm account.")] = None,
    qos: Annotated[str | None, typer.Option("--qos", help="New QOS.")] = None,
    checkpoint: Annotated[
        str | None,
        typer.Option("--checkpoint", help="Checkpoint artifact to resume from (default: newest valid)."),
    ] = None,
) -> None:
    """Diagnosis-driven recovery plan (never submits): prints eligibility and the exact submit command."""
    if strategy and strategy not in ("retry", "resume", "adjust_resources"):
        raise usage_error("--strategy must be retry, resume or adjust_resources")
    patch: dict[str, Any] = {
        k: v
        for k, v in (
            ("time", time_),
            ("mem", mem),
            ("partition", partition),
            ("gpu_type", gpu_type),
            ("gpus", gpus),
            ("exclude_nodes", csv_list(exclude_nodes)),
            ("account", account),
            ("qos", qos),
        )
        if v is not None and v != ""
    }
    params: dict[str, Any] = {"idempotency_key": new_key("recover")}
    if workflow:
        if ref:
            raise usage_error("give an attempt OR --workflow, not both")
        if missing and diagnosis:
            raise usage_error("--missing and --diagnosis are exclusive")
        cells: Any = {"diagnosis_in": list(diagnosis)} if diagnosis else "missing_valid"
        params["selector"] = {"workflow_id": workflow, "cells": cells}
        if strategy or patch or checkpoint:
            raise usage_error("--strategy, --checkpoint and the resource options apply to one attempt")
    else:
        if not ref:
            raise usage_error("give a failed attempt id, or --workflow WF")
        if kind_of(ref) == "workflow":
            raise usage_error(
                f"{ref} is a workflow: plan the recovery of its broken cells with `hpclab recover --workflow {ref}`"
            )
        if missing or diagnosis:
            raise usage_error("--missing/--diagnosis need --workflow")
        params["attempt_id"] = ref if kind_of(ref) == "attempt" else resolve_attempt(ref)[0]
        params["strategy"] = strategy
        params["checkpoint_ref"] = checkpoint
        if patch:
            params["resource_patch"] = patch

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_recover(d))
        cmds = V.recover_next(d)
        if cmds:
            out(V.next_steps(cmds, title="submit with"))

    env = call("run_recover", params, render=render, quiet=True)
    d = env.get("data") or {}
    ok = d.get("eligible") if d.get("mode") != "batch" else bool(d.get("eligible_count"))
    if not ok:
        raise typer.Exit(EXIT_ERROR)


def _resolve(
    ref: str, abandon: bool, bind_job: str | None, reason: str, yes: bool, confirm_ended: bool = False
) -> None:
    if sum((abandon, bool(bind_job), confirm_ended)) != 1:
        raise usage_error("pass exactly one of --abandon, --bind-job JOB_ID or --confirm-ended")
    if not reason.strip():
        raise usage_error("--reason must not be empty (it is recorded in the audit log)")
    attempt_id = ref if kind_of(ref) == "attempt" else resolve_attempt(ref)[0]
    data = inspect_data(attempt_id=attempt_id)
    a = attempt_view(data, attempt_id)
    # "uncertain since" only while it is (an observing attempt in an observation gap can be
    # abandoned too); a resolved uncertainty is shown as history, as `hpclab inspect` does
    head = V.T(f"attempt {attempt_id} · phase {a.get('phase')}", "bold")
    if V.submission_unresolved(a):
        head.append(f" · uncertain since {a.get('uncertain_since') or '?'}")
    elif a.get("observation_gap"):
        head.append(" · observation gap (the job is not visible in squeue or sacct)")
    if a.get("uncertain_since") and not V.submission_unresolved(a):
        head = Text.assemble(head, "\n", V.uncertainty_line(a))
    if confirm_ended:
        risk = (
            "CONFIRM ENDED: you state that the attempt's original Slurm job has really ended and will not be "
            "requeued. hpclab re-checks squeue and sacct first and refuses if the job is still queued, running or was "
            "requeued. Only then does it lift the requeue watch, so the run can be recovered or submitted again."
        )
    elif abandon:
        risk = (
            "ABANDON: hpclab stops tracking this attempt and releases the run's execution claim, so the run can be "
            "submitted again. If Slurm did accept a job for it, that job may still be running and spending GPU-hours: "
            "hpclab will NOT cancel it (check `squeue --me` on the cluster and scancel it yourself). The budget "
            "reservation stays counted as unknown, never as zero. Submitting the run again while the lost job runs "
            "would duplicate the work and both could write results."
        )
    else:
        risk = (
            f"BIND job {bind_job} to this attempt: hpclab verifies the live identity (owner, submit token in the job "
            "name, submit time) before binding and refuses on any mismatch. If other jobs matched too, they stay "
            "untouched: cancel them yourself — hpclab never cancels suspected duplicates."
        )
    confirm(
        "This is a human decision that hpclab cannot verify for you. Proceed?",
        yes=yes,
        details=Group(head, Text(risk, style="bold red" if abandon else "bold yellow")),
    )
    env = _with_version_retry(
        "attempt_resolve",
        lambda v: {
            "attempt_id": attempt_id,
            "expected_version": v,
            "resolution": "confirm_ended" if confirm_ended else ("abandon" if abandon else "bind_job"),
            "reason": reason,
            "idempotency_key": new_key("resolve"),
            "job_id": bind_job,
        },
        int(a.get("version") or 0),
        lambda: int(attempt_view(inspect_data(attempt_id=attempt_id), attempt_id).get("version") or 0),
    )

    def render(e: dict[str, Any]) -> None:
        d = e.get("data") or {}
        out(
            V.kv(
                "resolved",
                f"{d.get('resolution')} → phase {d.get('phase')}"
                + (f" · job {d['job_id']}" if d.get("job_id") else ""),
            )
        )
        out(V.next_steps([f"hpclab inspect {d.get('run_id')} --refresh"]))

    show(env, render=render, quiet=True)
    raise typer.Exit(exit_code_for(env))


@app.command(rich_help_panel="Runs")
def resolve(
    ref: Annotated[str, typer.Argument(help="Attempt id (or run id).")],
    abandon: Annotated[
        bool, typer.Option("--abandon", help="Give up on the attempt (does not cancel any job).")
    ] = False,
    bind_job: Annotated[
        str | None, typer.Option("--bind-job", help="Bind this Slurm job id (identity-checked).")
    ] = None,
    confirm_ended: Annotated[
        bool,
        typer.Option(
            "--confirm-ended",
            help="State that a terminal attempt's original job really ended (lifts a requeue watch; re-checked live).",
        ),
    ] = False,
    reason: Annotated[str, typer.Option("--reason", help="Why (required; recorded).")] = "",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Human decision for an attempt hpclab cannot settle by itself: an unreconciled or duplicate
    submission, an unobservable attempt, or a requeue watch holding a run."""
    _resolve(ref, abandon, bind_job, reason, yes, confirm_ended)


@attempt_app.command("resolve")
def attempt_resolve(
    ref: Annotated[str, typer.Argument(help="Attempt id (or run id).")],
    abandon: Annotated[bool, typer.Option("--abandon")] = False,
    bind_job: Annotated[str | None, typer.Option("--bind-job")] = None,
    confirm_ended: Annotated[bool, typer.Option("--confirm-ended")] = False,
    reason: Annotated[str, typer.Option("--reason")] = "",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Same as `hpclab resolve`."""
    _resolve(ref, abandon, bind_job, reason, yes, confirm_ended)


# ---------------------------------------------------------------------------
# which / trace / ops
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Runs")
def which(
    job_id: Annotated[str, typer.Argument(help="Slurm job id.")],
    cluster: Annotated[str | None, typer.Option("--cluster")] = None,
    submit_time: Annotated[
        str | None, typer.Option("--submit-time", help="Disambiguates reused job ids (ISO time).")
    ] = None,
) -> None:
    """Which run/attempt a Slurm job id belongs to (job ids are not global identities)."""
    cid = default_cluster(cluster)
    params = {
        "job_ref": {
            "cluster_id": cid,
            "job_id": job_id,
            **({"submit_time": submit_time} if submit_time else {}),
        }
    }

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        run = d.get("run") or {}
        aid = d.get("focus_attempt_id")
        a: dict[str, Any] = next((x for x in d.get("attempts") or [] if x.get("attempt_id") == aid), {})
        line = Text(f"{cid}:{job_id} → ", style="bold")
        line.append(f"run {run.get('label')} ({run.get('run_id')}) · attempt #{a.get('seq')} {aid} · ")
        line.append_text(V.code_text(run.get("code")))
        line.append(f" {a.get('scheduler_state') or a.get('phase') or ''}")
        out(line)
        out(
            V.kv(
                "experiment",
                f"{run.get('experiment_id')} · plan {run.get('plan_id')} · cell {V.cell_label(run.get('cell'))} seed {run.get('seed')}",
            )
        )
        out(V.next_steps([f"hpclab inspect {run.get('run_id')}", f"hpclab logs {aid}"]))

    env = call("run_inspect", params, render=render, quiet=True, exit_on_error=False)
    err = env.get("error") or {}
    if not env.get("ok"):
        candidates = (err.get("details") or {}).get("candidates") or []
        if err.get("code") == "ambiguous" and candidates and not STATE.json_output:
            err_console.print("  [yellow]job ids are reused: pick one by its submit time[/yellow]")
            for c in candidates:
                err_console.print(
                    Text(
                        f"    hpclab which {job_id} --cluster {cid} --submit-time {c.get('submit_time')}"
                        f"   # run {c.get('run_id')}",
                        style="cyan",
                    )
                )
        raise typer.Exit(exit_code_for(env))


@app.command(rich_help_panel="Results")
def trace(
    cell_ref: Annotated[str, typer.Argument(help="<bundle_id>:<cell_id>, as in \\hpcval{...}.")],
) -> None:
    """The provenance chain behind one table cell: evaluation → artifact digest → attempt → job → run → code → env."""
    call(
        "trace",
        {"cell_ref": cell_ref},
        render=lambda env: out(V.render_trace(env.get("data") or {})),
        quiet=True,
    )


# A blocked operation in one of these situations waits for a person, not for time.
_HUMAN_BLOCK_CODES = frozenset({"auth_required", "submission_uncertain", "duplicate_detected"})
_HUMAN_BLOCK_STAGES = frozenset(
    {
        "unresolved",
        "waiting_for_resolution",
        "gate_failed",
        "paused_by_breaker",
        "budget_exhausted",
        "admission_denied",
    }
)


def needs_human(op: dict[str, Any]) -> bool:
    if op.get("state") != "blocked":
        return False
    code = (op.get("error") or {}).get("code")
    return code in _HUMAN_BLOCK_CODES or op.get("stage") in _HUMAN_BLOCK_STAGES


_AUTH_RECHECK_S = 30.0


def _auth_block_is_stale(op: dict[str, Any]) -> bool:
    """True when an operation is blocked on auth_required but the cluster's SSH master connection
    is open again (the user ran `hpclab auth`): the controller retries the blocked step on its
    own (within ~2 min), so waiting is right and asking to authenticate again would be wrong."""
    err = op.get("error") or {}
    cluster = (err.get("details") or {}).get("cluster_id")
    if op.get("state") != "blocked" or err.get("code") != "auth_required" or not cluster:
        return False
    env = fetch("auth_command", {"cluster_id": cluster}, allow_error=True)
    return bool(env.get("ok") and (env.get("data") or {}).get("master_alive") is True)


def wait_operation(operation_id: str, *, timeout: float | None, quiet: bool = False) -> dict[str, Any]:
    """Poll an operation until it succeeded/failed or is blocked on a human decision (auth,
    unresolved submission, breaker...), respecting poll_after_ms. Other blocked states (e.g.
    waiting for reconciliation) keep waiting: they clear by themselves, and so does an auth
    block whose connection is already open again."""
    started = _now()
    last = ""
    auth_checked_at: float | None = None
    resumed = False
    env = fetch("operation_get", {"operation_id": operation_id})
    with interruptible(f"waiting for {operation_id} (it continues in the background)"):
        while True:
            d = env.get("data") or {}
            state = str(d.get("state"))
            line = f"{state}{' at ' + str(d.get('stage')) if d.get('stage') else ''}"
            if line != last and not quiet and not STATE.json_output:
                err = (d.get("error") or {}).get("message")
                note(
                    f"{time.strftime('%H:%M:%S')} {operation_id}: {line}"
                    + (f" — {err}" if err and state == "blocked" else ""),
                    "cyan",
                )
                last = line
            if state in ("succeeded", "failed"):
                return env
            if needs_human(d):
                if (d.get("error") or {}).get("code") != "auth_required":
                    return env
                if auth_checked_at is not None and _now() - auth_checked_at < _AUTH_RECHECK_S:
                    pass  # the connection was open at the last check: keep waiting
                elif _auth_block_is_stale(d):
                    if not resumed:  # wake the blocked step now instead of at its next backoff
                        resumed = True
                        cluster = str(((d.get("error") or {}).get("details") or {}).get("cluster_id"))
                        _, message, _ = auth_resume(cluster)
                        if not quiet and not STATE.json_output:
                            note(f"the SSH connection is open again: {message}; still waiting", "cyan")
                    auth_checked_at = _now()
                else:
                    return env
            if timeout is not None and _now() - started >= timeout:
                return env
            _sleep(max(0.2, min(30.0, float(d.get("poll_after_ms") or 2000) / 1000.0)))
            env = fetch("operation_get", {"operation_id": operation_id})


@app.command(rich_help_panel="Runs")
def ops(
    operation_id: str,
    wait: Annotated[bool, typer.Option("--wait", help="Poll until it succeeds or fails.")] = False,
    timeout: Annotated[
        float | None, typer.Option("--timeout", help="With --wait: give up after N seconds (exit 1).")
    ] = None,
) -> None:
    """A background operation (submit, prepare, cancel, dispatch, collect ...)."""
    env = (
        wait_operation(operation_id, timeout=timeout)
        if wait
        else fetch("operation_get", {"operation_id": operation_id})
    )
    d = env.get("data") or {}

    def render(e: dict[str, Any]) -> None:
        out(V.render_operation(e.get("data") or {}))

    show(env, render=render, quiet=True)
    if wait and d.get("state") != "succeeded":
        if needs_human(d) and not STATE.json_output:
            note("this operation waits for a decision or action from you (see above)", "bold yellow")
        elif d.get("state") != "failed" and timeout is not None and not STATE.json_output:
            note(f"still {d.get('state')} after {timeout:g}s (it continues in the background)", "yellow")
        code = (d.get("error") or {}).get("code")
        raise typer.Exit(4 if code == "auth_required" else EXIT_ERROR)


# ---------------------------------------------------------------------------
# artifacts / read / collect
# ---------------------------------------------------------------------------


@app.command(rich_help_panel="Results")
def artifacts(
    ref: Annotated[str, typer.Argument(help="Attempt id (or run id: its latest attempt).")],
    role: Annotated[
        str | None,
        typer.Option("--role", help="metrics | checkpoint | log | prediction | figure | other | file | dir"),
    ] = None,
    members: Annotated[
        bool, typer.Option("--members", help="Also list members of directory outputs.")
    ] = False,
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-index from the cluster first.")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Declared outputs with size, digest and integrity (not a file browser)."""
    attempt_id = ref if kind_of(ref) == "attempt" else resolve_attempt(ref)[0]
    params = {
        "attempt_id": attempt_id,
        "kind": role,
        "include_members": members or None,
        "refresh": refresh or None,
        "limit": limit,
        "cursor": cursor,
    }
    call(
        "artifacts_list",
        params,
        render=lambda env: out(V.render_artifacts(env.get("data") or {})),
        quiet=True,
    )


@app.command(rich_help_panel="Results")
def read(
    artifact: Annotated[str, typer.Argument(help="Artifact id or hpclab://artifacts/<id>[#member].")],
    view: Annotated[str | None, typer.Option("--view", help="head | tail | text | json | jsonl_tail")] = None,
    max_bytes: Annotated[int | None, typer.Option("--max-bytes")] = None,
) -> None:
    """Bounded, stale-safe read of one artifact (content on stdout, details on stderr)."""
    if view and view not in ("head", "tail", "text", "json", "jsonl_tail"):
        raise usage_error("--view must be head, tail, text, json or jsonl_tail")
    env = fetch("artifact_read", {"artifact_ref": artifact, "view": view, "max_bytes": max_bytes})
    if STATE.json_output:
        print_json(env)
        return
    d = env.get("data") or {}
    meta = V.render_artifact_read(d)
    if meta is not None:
        err_console.print(meta)
    if d.get("binary"):
        dl = d.get("download") or {}
        note(
            "binary content is not printed"
            + (
                f"; download: hpclab ops {dl['operation_id']} --wait (cached by digest)"
                if dl.get("operation_id")
                else ""
            ),
            "yellow",
        )
    elif d.get("records") is not None:
        for rec in d["records"]:
            write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not d["records"] and d.get("truncated"):
            note(
                f"no complete JSONL record in the last {d.get('bytes_read')} byte(s): raise --max-bytes (max 65536)",
                "yellow",
            )
    elif d.get("value") is not None and (d.get("view") == "json"):
        write(json.dumps(d["value"], indent=2, ensure_ascii=False) + "\n")
    elif d.get("text") is not None:
        text = str(d["text"])
        write(text if text.endswith("\n") or not text else text + "\n")
    elif d.get("members") is not None:
        out(
            V.render_artifacts(
                {"artifacts": d["members"], "attempt_id": (d.get("artifact") or {}).get("attempt_id")}
            )
        )
    if d.get("note"):
        note(str(d["note"]))
    if d.get("truncated") and d.get("next_offset") is not None:
        note(f"truncated at byte {d['next_offset']} of {d.get('size')} (raise --max-bytes, max 65536)")
    for w in env.get("warnings") or []:
        note(f"warning: {w}", "yellow")


@app.command(rich_help_panel="Results")
def collect(
    ref: Annotated[
        str | None, typer.Argument(help="Attempt id (or run id); or a plan/workflow id for batch mode.")
    ] = None,
    experiment: Annotated[
        str | None, typer.Option("--experiment", help="Batch: every finished run of this experiment.")
    ] = None,
    evaluator: Annotated[
        str | None, typer.Option("--evaluator", help="Evaluator id (default: the run's declared one).")
    ] = None,
    config: Annotated[
        list[str] | None, typer.Option("--config", help="Evaluator config k=v (repeatable).")
    ] = None,
    wait: Annotated[bool, typer.Option("--wait", help="Wait for background batch work to finish.")] = False,
) -> None:
    """Extract metrics with a registered evaluator (idempotent; provenance-carrying)."""
    if bool(ref) == bool(experiment):
        raise usage_error("give an attempt/run (or plan/workflow) id, or --experiment EXP")
    params: dict[str, Any] = {
        "idempotency_key": new_key("collect"),
        "evaluator_id": evaluator,
        "config": parse_kv(config) if config else None,
    }
    if experiment:
        params["run_set_ref"] = experiment_id(experiment)
    else:
        assert ref is not None
        k = kind_of(ref)
        if k in ("plan", "workflow", "experiment"):
            params["run_set_ref"] = ref
        else:
            params["attempt_id"] = ref if k == "attempt" else resolve_attempt(ref)[0]
    env = call(
        "evaluation_collect", params, render=lambda e: out(V.render_collect(e.get("data") or {})), quiet=True
    )
    d = env.get("data") or {}
    if wait and d.get("operation_id") and d.get("pending"):
        final = wait_operation(str(d["operation_id"]), timeout=None)
        show(final, render=lambda e: out(V.render_operation(e.get("data") or {})), quiet=True)
        if (final.get("data") or {}).get("state") != "succeeded":
            raise typer.Exit(EXIT_ERROR)


# ---------------------------------------------------------------------------
# compare / export / verify / paper
# ---------------------------------------------------------------------------


def _contract_metrics(kind: str, ident: str) -> list[str]:
    exp_id = ident
    if kind == "plan":
        exp_id = str(
            (fetch("plan_get", {"plan_id": ident, "limit": 1}).get("data") or {}).get("experiment_id")
        )
    elif kind == "workflow":
        exp_id = str(
            (fetch("workflow_get", {"workflow_id": ident, "limit": 1}).get("data") or {}).get("experiment_id")
        )
    spec = (fetch("experiment_get", {"experiment_id": exp_id}).get("data") or {}).get("spec") or {}
    metrics = ((spec.get("comparison") or {}).get("metrics")) or []
    return [str(m.get("name")) for m in metrics if isinstance(m, dict) and m.get("name")]


@app.command(rich_help_panel="Results")
def compare(
    target: Annotated[
        str, typer.Argument(help="Experiment id/name, plan id or workflow id (the planned run set).")
    ],
    metric: Annotated[
        list[str] | None,
        typer.Option("--metric", "-m", help="Metric (repeatable; default: the spec's comparison metrics)."),
    ] = None,
    seeds: Annotated[
        str | None, typer.Option("--seeds", help="Expected seeds, e.g. 1,2,3 (checked against the plan).")
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="RUN=REASON: remove one run, disclosed (repeatable)."),
    ] = None,
    decimals: Annotated[int, typer.Option("--decimals", min=0, max=8)] = 4,
) -> None:
    """Planned-denominator comparison: mean ± std (n/expected), differences vs baseline, coverage."""
    kind, ident = run_set(target)
    metrics = list(metric or []) or _contract_metrics(kind, ident)
    if not metrics:
        raise usage_error("no --metric given and the experiment's spec declares no comparison metrics")
    params: dict[str, Any] = {
        "planned_run_set_ref": ident,
        "metric_names": metrics,
        "expected_seeds": parse_csv_ints(seeds),
        "exclusions": [{"run_id": r, "reason": why} for r, why in pairs(exclude, "--exclude")] or None,
        "idempotency_key": new_key("compare"),
    }

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_compare(d, decimals=decimals))
        cid = d.get("comparison_id")
        # E runs (completed, evaluation pending) count as missing: exporting now would publish seeds
        # that are merely not evaluated yet
        pending = sum(1 for r in d.get("runs") or [] if r.get("code") == "E")
        first = (
            [f"hpclab collect {ident}   # {pending} run(s) completed, evaluation pending; then compare again"]
            if pending
            else []
        )
        out(
            V.next_steps(
                [*first, f"hpclab export {cid} --out <dir>", f"hpclab grid {ident}" if kind != "plan" else ""]
            )
        )

    call("results_compare", params, render=render, quiet=True)


def _formats(text: str | None) -> list[str] | None:
    items = csv_list(text)
    if not items:
        return None
    alias = {"tex": "latex", "markdown": "md", "latex": "latex", "md": "md", "csv": "csv", "json": "json"}
    result = []
    for f in items:
        key = f.lower()
        if key not in alias:
            raise usage_error(f"unknown format {f!r} (md, latex, csv, json)")
        if alias[key] not in result:
            result.append(alias[key])
    return result


@app.command(rich_help_panel="Results")
def export(
    comparison: Annotated[
        str | None, typer.Argument(help="Comparison id (cmp_...) from `hpclab compare`.")
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Directory to write the bundle files into.")
    ] = Path("evidence"),
    experiment: Annotated[
        str | None, typer.Option("--experiment", help="Compare this experiment now and export it.")
    ] = None,
    bundle: Annotated[
        str | None,
        typer.Option("--bundle", help="Write an existing bundle's files again (evb_...); no new bundle."),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option("--metric", "-m", help="With --experiment: metrics (default: the spec's)."),
    ] = None,
    formats: Annotated[str | None, typer.Option("--format", help="md,latex,csv,json (default: all).")] = None,
    purpose: Annotated[
        str | None, typer.Option("--purpose", help="What the table is for, e.g. 'Table 2, NeurIPS draft'.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace files with different content.")
    ] = False,
) -> None:
    """Evidence bundle (Markdown summary, LaTeX, CSV, JSON manifest) written to --out."""
    if bundle is not None:
        if comparison or experiment:
            raise usage_error("--bundle writes an existing bundle: give no comparison id or --experiment")
        if metric or formats or purpose:
            raise usage_error(
                "--metric/--format/--purpose do not apply to --bundle (a bundle's files are fixed)"
            )
        bid = bundle.removeprefix("hpclab://evidence/")
        if kind_of(bid) != "evidence":
            raise usage_error(f"expected an evidence bundle id (evb_...), got {bundle!r}")
        _write_bundle(bid, out_dir, overwrite, exp_env=None)
        return
    if bool(comparison) == bool(experiment):
        raise usage_error("give a comparison id, or --experiment EXP (or --bundle BUNDLE to rewrite one)")
    if metric and not experiment:
        raise usage_error("--metric only applies with --experiment (a comparison already fixed its metrics)")
    params: dict[str, Any] = {
        "idempotency_key": new_key("export"),
        "formats": _formats(formats),
        "purpose": (purpose or f"hpclab export {comparison or experiment}")[:500],
    }
    if comparison:
        if kind_of(comparison) not in ("comparison", None) and not comparison.startswith(
            "hpclab://comparisons/"
        ):
            raise usage_error(f"expected a comparison id (cmp_...), got {comparison!r}")
        params["comparison_ref"] = comparison
    else:
        assert experiment is not None
        params["planned_run_set_ref"] = experiment_id(experiment)
        params["metric_names"] = list(metric) if metric else None
    exp_env = fetch("evidence_export", params)
    bundle_id = str((exp_env.get("data") or {}).get("bundle_id"))
    _write_bundle(bundle_id, out_dir, overwrite, exp_env=exp_env)


def _write_bundle(bundle_id: str, out_dir: Path, overwrite: bool, *, exp_env: dict[str, Any] | None) -> None:
    """evidence_write of one bundle, rendered (``exp_env``: the export that created it, if any)."""
    target = abspath(out_dir)
    wr_env = fetch(
        "evidence_write",
        {"bundle_id": bundle_id, "out_dir": target, "overwrite": overwrite or None},
        allow_error=True,
    )
    if STATE.json_output:
        doc: dict[str, Any] = {"ok": bool(wr_env.get("ok"))}
        if exp_env is not None:
            doc["evidence_export"] = exp_env
        doc["evidence_write"] = wr_env
        print_json(doc)
        raise typer.Exit(exit_code_for(wr_env))
    if exp_env is not None:
        show(exp_env, render=lambda e: out(V.render_evidence(e.get("data") or {})))
    if not wr_env.get("ok"):
        show(wr_env)
        if ((wr_env.get("error") or {}).get("details") or {}).get("conflicts"):
            err_console.print(
                "  [yellow]hint[/yellow]: re-run with --overwrite to replace those files, or choose another --out"
            )
        raise typer.Exit(exit_code_for(wr_env))
    wd = wr_env.get("data") or {}
    out(V.render_written(wd))
    tex = next((f.get("path") for f in wd.get("files") or [] if str(f.get("name")).endswith(".tex")), None)
    cmds = [f"hpclab verify {bundle_id}"]
    if tex:
        cmds.insert(0, f"\\input{{{tex}}} in your paper, then: hpclab paper check <paper>.tex")
    out(V.next_steps(cmds))


def _verify_checked(d: dict[str, Any]) -> bool:
    """Did verify find any \\hpcval tag in the manuscript (older controllers: tags_checked)?"""
    checked = d.get("checked")
    return bool(checked) if checked is not None else bool(d.get("tags_checked"))


@app.command(rich_help_panel="Results")
def verify(
    bundle: Annotated[str, typer.Argument(help="Evidence bundle id (evb_...).")],
    tex: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Manuscript .tex files to check against the bundle (\\input files are followed)."
        ),
    ] = None,
    recheck_remote: Annotated[
        bool, typer.Option("--recheck-remote", help="Re-hash small metrics files on the cluster.")
    ] = False,
) -> None:
    """Recompute a bundle from the ledger and check manuscript numbers (record consistency, not science)."""
    paths = [abspath(p) for p in tex or []]
    for p in paths:
        if not Path(p).is_file():
            raise usage_error(f"tex file not found: {p}")

    def render(env: dict[str, Any]) -> None:
        d = env.get("data") or {}
        out(V.render_verify(d))
        if paths:
            out(V.scanned_line(d.get("files_scanned") or []))
            if not _verify_checked(d):
                out(V.not_checked_line())

    env = call(
        "evidence_verify",
        {"bundle_ref": bundle, "tex_paths": paths or None, "recheck_remote": recheck_remote or None},
        render=render,
        quiet=True,
    )
    d = env.get("data") or {}
    if not d.get("ok", False) or (paths and not _verify_checked(d)):
        raise typer.Exit(EXIT_ERROR)


def _tex_tags(text: str) -> list[Any]:
    from hpclab.services.verify import parse_hpcval_tags

    return list(parse_hpcval_tags(text))


_MAX_PAPER_BUNDLES = 10


@paper_app.command("check")
def paper_check(
    tex: Annotated[
        list[Path], typer.Argument(help="Manuscript .tex files (\\input/\\include are followed).")
    ],
    bundle: Annotated[
        list[str] | None,
        typer.Option(
            "--bundle",
            help="Evidence bundle to verify against (repeatable); needed when the given files have no "
            "\\hpcval tag themselves (the tables come in through \\input).",
        ),
    ] = None,
    recheck_remote: Annotated[bool, typer.Option("--recheck-remote")] = False,
) -> None:
    """Check every \\hpcval-tagged number against its evidence bundle; exit 1 on any error or when nothing was checked."""
    paths: list[str] = []
    for p in tex:
        if not p.is_file():
            raise usage_error(f"tex file not found: {p}")
        paths.append(abspath(p))
    # Which bundles to verify: --bundle, then those tagged in the given files themselves. Every
    # verify call gets every file: the service follows \input and checks each tag whatever its
    # bundle, and reports the bundles it met in included files (tex[].bundles), verified next.
    explicit: list[str] = []
    for b in bundle or []:
        bid = b.removeprefix("hpclab://evidence/")
        if kind_of(bid) != "evidence":
            raise usage_error(f"--bundle expects an evidence bundle id (evb_...), got {b!r}")
        if bid not in explicit:
            explicit.append(bid)
    queue = list(explicit)
    malformed: list[str] = []
    for ap in paths:
        for tag in _tex_tags(Path(ap).read_text(encoding="utf-8", errors="replace")):
            bid = str(tag.ref).partition(":")[0].strip()
            if kind_of(bid) != "evidence":
                malformed.append(bid or str(tag.ref))
            elif bid not in queue:
                queue.append(bid)
    envs: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    failed: list[tuple[str, dict[str, Any]]] = []
    scanned: list[str] = []
    checked = False
    tags = 0

    def add(f: dict[str, Any]) -> None:
        key = (f.get("code"), f.get("severity"), f.get("message"), tuple(f.get("refs") or []))
        if key not in seen:
            seen.add(key)
            findings.append(f)

    i = 0
    while i < len(queue) and i < _MAX_PAPER_BUNDLES:
        bid = queue[i]
        i += 1
        env = fetch(
            "evidence_verify",
            {"bundle_ref": bid, "tex_paths": paths, "recheck_remote": recheck_remote or None},
            allow_error=True,
        )
        envs[bid] = env
        if not env.get("ok"):
            if (env.get("error") or {}).get("code") == "controller_unavailable":
                show(env)
                raise typer.Exit(exit_code_for(env))
            failed.append((bid, env))
            continue
        d = env.get("data") or {}
        checked = checked or _verify_checked(d)
        tags = max(tags, int(d.get("tags_checked") or 0))
        for f in d.get("files_scanned") or []:
            if f not in scanned:
                scanned.append(str(f))
        for f in d.get("findings") or []:
            add(f)
        for rep in d.get("tex") or []:
            for b in rep.get("bundles") or []:
                if b not in queue:
                    queue.append(str(b))
    skipped = queue[i:]
    verified_any = any(e.get("ok") for e in envs.values())
    for bid, env in failed:
        # a bad tag reference is reported by any successful call (it checks every tag); a named
        # --bundle, or a reference nothing could verify, is reported here
        if bid in explicit or not verified_any:
            err = env.get("error") or {}
            add(
                {
                    "code": "UNRESOLVED_REF",
                    "severity": "error",
                    "message": f"bundle {bid}: {err.get('code')}: {err.get('message')}",
                    "refs": paths,
                }
            )
    if not verified_any:
        for ref in dict.fromkeys(malformed):
            add(
                {
                    "code": "UNRESOLVED_REF",
                    "severity": "error",
                    "message": f"\\hpcval reference to {ref!r} is not an evidence bundle id",
                    "refs": paths,
                }
            )
    if not queue and not malformed:
        add(
            {
                "code": "NO_TAGS_FOUND",
                "severity": "error",
                "message": f"no \\hpcval tag in {', '.join(paths)} themselves, so there is no evidence bundle "
                "to verify against: if the tables come in through \\input, pass --bundle <evb_...> (the id is "
                "in their \\hpcval tags; the verifier follows \\input)",
                "refs": paths,
            }
        )
    order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: order.get(str(f.get("severity")), 9))
    errors = sum(1 for f in findings if f.get("severity") == "error")
    ok = errors == 0 and checked
    if STATE.json_output:
        print_json(
            {
                "ok": ok,
                "checked": checked,
                "tex_paths": paths,
                "files_scanned": scanned,
                "tags": tags,
                "bundles": envs,
                "bundles_not_verified": skipped,
                "findings": findings,
            }
        )
        raise typer.Exit(0 if ok else EXIT_ERROR)
    if not envs:
        raise fail(str(findings[0]["message"]), error=str(findings[0]["code"]).lower(), json_error=False)
    for bid, env in envs.items():
        d = env.get("data") or {}
        if env.get("ok"):
            out(V.T(str(d.get("summary") or bid), "dim"))
            for w in env.get("warnings") or []:
                out(V.T(f"  warning: {w}", "yellow"))
    if skipped:
        out(V.T(f"{len(skipped)} more bundle(s) referenced but not verified: {', '.join(skipped)}", "yellow"))
    out(V.scanned_line(scanned))
    out(V.render_verify_findings(findings))
    if verified_any and not checked:
        out(V.not_checked_line())
    head = Text(
        f"{tags} tag(s) in {len(scanned) or len(paths)} scanned file(s), {len(envs)} bundle(s): ",
        style="bold",
    )
    head.append(f"{errors} error(s)" if errors else "no errors", style="bold red" if errors else "bold green")
    out(head)
    out(V.T("record consistency only: this does not verify scientific correctness", "dim"))
    if not ok:
        raise typer.Exit(EXIT_ERROR)


_DEF_RE = re.compile(
    r"^[ \t]*\\(?:providecommand|newcommand|renewcommand|DeclareRobustCommand)\*?[ \t]*\{?[ \t]*\\hpcval[ \t]*\}?[^\n]*(?:\n|$)",
    re.M,
)
# hpclab's own comment explaining the tags (it would be stale once they are gone)
_TAG_COMMENT_RE = re.compile(r"^[ \t]*%[^\n]*\\hpcval[^\n]*(?:\n|$)", re.M)


def _brace_group(text: str, pos: int) -> tuple[int, int] | None:
    """(start, end) of the content of the balanced ``{...}`` at ``pos`` (after spaces)."""
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None
    depth = 0
    i = pos
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos + 1, i
        i += 1
    return None


def strip_hpcval(text: str) -> tuple[str, int]:
    """Replace every ``\\hpcval{ref}{value}`` (outside comments) with ``value``, drop the
    ``\\providecommand{\\hpcval}`` definition and comment lines that mention ``\\hpcval``.
    Returns (text, tags stripped)."""
    tags = _tex_tags(text)
    pieces: list[str] = []
    pos = 0
    n = 0
    for tag in tags:
        segment = text[tag.start : tag.end]
        first = _brace_group(segment, len("\\hpcval"))
        second = _brace_group(segment, first[1] + 1) if first else None
        if second is None:
            continue
        pieces.append(text[pos : tag.start])
        pieces.append(segment[second[0] : second[1]])
        pos = tag.end
        n += 1
    pieces.append(text[pos:])
    stripped = _TAG_COMMENT_RE.sub("", _DEF_RE.sub("", "".join(pieces)))
    return stripped, n


@paper_app.command("strip")
def paper_strip(
    tex: Annotated[Path, typer.Argument(help="Input .tex file.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here (default: stdout).")
    ] = None,
) -> None:
    """Camera-ready copy without provenance tags: \\hpcval{ref}{v} → v (local, no controller)."""
    if not tex.is_file():
        raise usage_error(f"tex file not found: {tex}")
    if output is not None and output.exists() and output.resolve() == tex.resolve():
        raise usage_error("refusing to overwrite the tagged source; write to another file")
    text, n = strip_hpcval(tex.read_text(encoding="utf-8"))
    if output is None:
        write(text)
    else:
        output.write_text(text, encoding="utf-8")
    note(f"stripped {n} \\hpcval tag(s)" + (f" → {output}" if output else ""))


# ---------------------------------------------------------------------------
# evidence aliases
# ---------------------------------------------------------------------------


@evidence_app.command("verify")
def evidence_verify_alias(
    bundle: str,
    tex: Annotated[list[Path] | None, typer.Argument()] = None,
    recheck_remote: Annotated[bool, typer.Option("--recheck-remote")] = False,
) -> None:
    """Same as `hpclab verify`."""
    verify(bundle, tex, recheck_remote)


@evidence_app.command("trace")
def evidence_trace_alias(cell_ref: str) -> None:
    """Same as `hpclab trace`."""
    trace(cell_ref)


# ---------------------------------------------------------------------------
# usage / digest / vault / backup / notify / events
# ---------------------------------------------------------------------------

_GROUP_BY = ("user", "account", "partition", "qos", "gpu_type", "state", "diagnosis_class")


@app.command(rich_help_panel="Lab")
def usage(
    cluster: Annotated[str | None, typer.Option("--cluster")] = None,
    account: Annotated[
        str | None, typer.Option("--account", help="Account-wide report (visibility permitting).")
    ] = None,
    mine: Annotated[bool, typer.Option("--mine", help="Only your jobs (default).")] = False,
    since: Annotated[str, typer.Option("--since", help="ISO date/time or 7d, 24h ...")] = "7d",
    until: Annotated[str | None, typer.Option("--until")] = None,
    by: Annotated[str | None, typer.Option("--by", help=f"Group by: {','.join(_GROUP_BY)}.")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    """GPU-hours by user/partition/state with waste by cause — honest about what Slurm lets you see."""
    if account and mine:
        raise usage_error("--account and --mine are exclusive")
    group_by = csv_list(by)
    for g in group_by or []:
        if g not in _GROUP_BY:
            raise usage_error(f"unknown --by {g!r} (choose from {', '.join(_GROUP_BY)})")
    params = {
        "cluster_id": default_cluster(cluster),
        "scope": "account" if account else "self",
        "account": account,
        "since": since,
        "until": until,
        "group_by": group_by,
        "limit": limit,
    }
    call("usage_report", params, render=lambda env: out(V.render_usage(env.get("data") or {})), quiet=True)


@app.command(rich_help_panel="Lab")
def digest(
    since: Annotated[str, typer.Option("--since")] = "7d",
    until: Annotated[str | None, typer.Option("--until")] = None,
    account: Annotated[
        str | None,
        typer.Option(
            "--account",
            help="Add a usage section for this account (the only remote call; visibility permitting).",
        ),
    ] = None,
    cluster: Annotated[
        str | None,
        typer.Option(
            "--cluster",
            help="Narrow the ledger sections (experiments, failures, GPU-hours, attention items, "
            "long-pending jobs) to this cluster; with --account, the cluster the account usage comes from.",
        ),
    ] = None,
    out_file: Annotated[Path | None, typer.Option("--out", "-o", help="Write the Markdown here.")] = None,
) -> None:
    """Weekly lab digest (Markdown) from the ledger: progress, failures, GPU-hours, attention items."""
    env = fetch("lab_digest", {"since": since, "until": until, "account": account, "cluster_id": cluster})
    if STATE.json_output:
        print_json(env)
        return
    md = str((env.get("data") or {}).get("markdown") or "")
    if out_file is not None:
        out_file.write_text(md if md.endswith("\n") else md + "\n", encoding="utf-8")
        note(f"wrote {out_file} ({len(md.encode())} bytes)")
    elif console.is_terminal:
        from rich.markdown import Markdown

        out(Markdown(md))
    else:
        write(md if md.endswith("\n") else md + "\n")
    stderr_notes(env)


@app.command(rich_help_panel="Lab")
def vault(
    experiment: Annotated[str | None, typer.Option("--experiment")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Local vault of raw job records (sacct, scripts, env, log head/tail, metrics) per attempt."""
    params = {
        "experiment_id": experiment_id(experiment) if experiment else None,
        "limit": limit,
        "cursor": cursor,
    }
    call("vault_status", params, render=lambda env: out(V.render_vault(env.get("data") or {})), quiet=True)


@app.command(rich_help_panel="Lab")
def backup(
    dest: Annotated[Path, typer.Argument(help="Backup directory (created).")],
    no_vault: Annotated[bool, typer.Option("--no-vault", help="Only the ledger and manifests.")] = False,
) -> None:
    """Consistent online backup of the ledger (+ vault) to DEST."""
    call(
        "backup",
        {
            "dest": abspath(dest),
            "idempotency_key": new_key("backup"),
            "include_vault": False if no_vault else None,
        },
        render=lambda env: out(V.render_backup(env.get("data") or {})),
        quiet=True,
    )


@notify_app.command("test")
def notify_test(cluster: Annotated[str | None, typer.Option("--cluster")] = None) -> None:
    """Send a test notification through every configured channel."""
    env = call(
        "notify_test",
        {"cluster_id": cluster},
        render=lambda e: out(V.render_notify_test(e.get("data") or {})),
        quiet=True,
    )
    if (env.get("data") or {}).get("failed"):
        raise typer.Exit(EXIT_ERROR)


@app.command(rich_help_panel="Lab")
def events(
    entity: Annotated[
        str | None, typer.Option("--entity", help="Exact id (attempt, run, workflow, plan ...).")
    ] = None,
    type_: Annotated[
        str | None, typer.Option("--type", help="Event type, or a prefix like 'attempt.'.")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
) -> None:
    """Audit trail: every state change, newest first."""
    call(
        "events_list",
        {"entity_id": entity, "event_type": type_, "limit": limit, "cursor": cursor},
        render=lambda env: out(V.render_events(env.get("data") or {})),
        quiet=True,
    )


# ---------------------------------------------------------------------------
# simulator reset / demo / onboarding
# ---------------------------------------------------------------------------


def make_writable(root: Path) -> None:
    """``chmod -R u+w`` without following symlinks (published snapshots are read-only)."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
                if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                    os.chmod(p, st.st_mode | stat.S_IWUSR)
            except OSError:
                continue
    try:
        st = os.lstat(root)
        os.chmod(root, st.st_mode | stat.S_IWUSR)
    except OSError:
        pass


@sim_app.command("reset")
def sim_reset(
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    cluster_id: Annotated[str, typer.Option("--cluster")] = "sim",
) -> None:
    """Kill simulated jobs and delete the simulator's state and files, then re-create it empty."""
    from hpclab.controller.files import ensure_private_dir
    from hpclab.sim.cluster import SimCluster

    root = sim_root(cluster_id)
    if not root.exists():
        raise fail(f"no simulator at {root}", hint="hpclab init --sim", error="not_found")
    if not root.is_dir() or root.is_symlink() or not (root / "state.db").is_file():
        # the root comes from the cluster overlay and may live outside the home: never delete a
        # directory that does not look like a simulator
        raise fail(
            f"{root} is not a simulator directory (no state.db): refusing to reset it",
            hint=f"check sim_root in the cluster's overlay (hpclab cluster show {cluster_id}), or hpclab init --sim",
        )
    confirm(
        f"Reset the simulator at {root}?",
        yes=yes,
        details=V.T(
            "All simulated jobs are killed and the simulated cluster's files (snapshots, attempt outputs, "
            "accounting) are deleted. The ledger keeps its history: attempts still active on the simulator will "
            "lose their jobs and show up as unknown/failed.",
            "bold yellow",
        ),
    )
    killed = SimCluster(root).shutdown(grace_s=2.0)
    make_writable(root)
    shutil.rmtree(root)
    # Re-created private (0700) and only as our own real directory: a relocated simulator lives
    # under a predictable name in shared /var/tmp, where another user could plant a symlink or
    # directory between the delete and the re-create.
    try:
        ensure_private_dir(root, what="simulator directory")
    except HpcLabError as exc:
        raise fail(exc.message, hint="then run: hpclab init --sim", error=exc.code) from None
    SimCluster.ensure(root)
    if cluster_id == "sim":
        fetch("init_sim", {})  # re-creates the remote root and keeps the registration current
    if STATE.json_output:
        print_json({"ok": True, "cluster_id": cluster_id, "sim_root": str(root), "jobs_stopped": len(killed)})
        return
    out(
        V.T(
            f"simulator {cluster_id} reset ({len(killed)} active job(s) stopped); fresh state at {root}",
            "green",
        )
    )


@app.command(rich_help_panel="Setup")
def demo(
    fast: Annotated[bool, typer.Option("--fast", help="Shorter jobs, fewer pauses.")] = False,
    keep: Annotated[
        bool, typer.Option("--keep", help="Keep the demo's experiments and files afterwards.")
    ] = False,
) -> None:
    """Guided tour on the local simulator: prepare → smoke gate → sweep → failure → diagnosis → recovery → evidence."""
    if STATE.json_output:
        raise usage_error("demo is interactive; --json is not supported")
    try:
        run_demo = importlib.import_module("hpclab.demo").run_demo
    except (ImportError, AttributeError) as exc:
        raise fail(
            f"the demo is not available in this installation ({exc})",
            hint="run the example by hand: hpclab examples copy tiny-classifier && hpclab init --sim "
            "&& hpclab prepare tiny-classifier/hpclab.yaml",
            error="unsupported",
        ) from exc
    # the demo runs in its own throwaway home and cleans up its controller, jobs and files itself
    with interruptible("the demo (it cleans up its own controller, simulated jobs and files)"):
        rc = run_demo(STATE.paths, console=console, fast=fast, keep=keep)
    raise typer.Exit(int(rc or 0))


_ONBOARD_SCRIPT = '''"""hpclab onboarding GPU smoke test (standard library only)."""
import json
import os
import socket
import subprocess
import sys
import time


def main() -> int:
    out = os.environ.get("HPCLAB_OUTPUT_DIR") or "."
    os.makedirs(out, exist_ok=True)
    print(f"host={socket.gethostname()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"python={sys.version.split()[0]} ({sys.executable})")
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        listing, rc = proc.stdout.strip(), proc.returncode
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
    except FileNotFoundError:
        listing, rc = "", 127
    except subprocess.TimeoutExpired:
        listing, rc = "", 124
    print("nvidia-smi -L:")
    print(listing or "(no output)")
    gpus = [line for line in listing.splitlines() if line.startswith("GPU ")]
    record = {
        "schema": "hpclab.metrics/v1",
        "step": 1,
        "total_steps": 1,
        "split": "smoke",
        "final": True,
        "time": time.time(),
        "metrics": {"gpus_visible": len(gpus), "nvidia_smi_ok": 1 if rc == 0 else 0},
    }
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    if rc != 0 or not gpus:
        print(f"no GPU visible to the job (nvidia-smi exit {rc})", file=sys.stderr)
        return 1
    return 0


sys.exit(main())
'''


def onboard_spec(
    cluster_id: str, project_root: str, *, partition: str | None, gpu_type: str | None
) -> dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", cluster_id)[:60] or "cluster"
    resources: dict[str, Any] = {"gpus": 1, "cpus": 1, "mem": "4G", "time": "00:10:00"}
    if partition:
        resources["partition"] = partition
    if gpu_type:
        resources["gpu_type"] = gpu_type
    return {
        "name": f"hpclab-onboard-{safe}",
        "question": f"Does a 1-GPU batch job start on {cluster_id}, see its GPU and write its outputs?",
        "cluster": cluster_id,
        "code": {"project_root": project_root},
        "entrypoint": {"argv": ["python3", "hpclab_gpu_smoke.py"]},
        "runtime": {"python": "python3"},
        "resources": resources,
        "outputs": [
            {
                "name": "metrics",
                "path": "{output_dir}/metrics.json",
                "role": "metrics",
                "format": "hpclab.metrics/v1",
                "required": True,
            }
        ],
        "evaluation": {
            "evaluator_id": "hpclab_metrics",
            "evaluator_version": "1",
            "output": "metrics",
            "config": {"split": "smoke"},
        },
        "tags": ["onboarding"],
    }


def _attempt_line(data: dict[str, Any]) -> str:
    aid = data.get("focus_attempt_id")
    a = next((x for x in data.get("attempts") or [] if x.get("attempt_id") == aid), None)
    if a is None:
        return "not submitted"
    parts = [str(a.get("phase"))]
    if a.get("scheduler_state") and a.get("scheduler_state") != "UNKNOWN":
        parts.append(str(a["scheduler_state"]))
    pending = data.get("pending") or {}
    if pending and a.get("scheduler_state") == "PENDING":
        parts.append(f"({pending.get('reason_code')}: {pending.get('explanation')})")
    if a.get("phase") == "terminal":
        parts.append(f"app {a.get('app_state')} · results {a.get('result_state')}")
    return " ".join(parts)


@app.command(rich_help_panel="Setup")
def onboard(
    cluster_id: Annotated[str, typer.Argument(help="Registered cluster id.")],
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not ask before spending the GPU time.")
    ] = False,
    partition: Annotated[
        str | None, typer.Option("--partition", help="Default: the cluster's default partition.")
    ] = None,
    gpu_type: Annotated[str | None, typer.Option("--gpu-type")] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the job (queueing included).")
    ] = 1800.0,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between status checks.")] = 5.0,
) -> None:
    """Inspect a cluster, then run a tiny 1-GPU smoke job end to end and report pass/fail."""
    insp = fetch("cluster_inspect", {"cluster_id": cluster_id})
    idata = insp.get("data") or {}
    if not STATE.json_output:
        out(V.render_cluster_inspect(idata))
        show({**insp, "summary": "", "data": None})
    if idata.get("status") == "fail":
        if STATE.json_output:
            print_json({"ok": False, "cluster_inspect": insp})
        raise fail(
            "the cluster check failed: fix the findings above before spending GPU time",
            hint=f"hpclab cluster inspect {cluster_id}",
            json_error=False,
        )
    confirm(
        f"Submit a 1-GPU, 10-minute smoke job to {cluster_id}?",
        yes=yes,
        details=V.T(
            f"onboarding job: 1 GPU{f' ({gpu_type})' if gpu_type else ''} × 00:10:00 on "
            f"{partition or 'the default partition'} (≤ 0.17 GPU-h); it prints the host, CUDA_VISIBLE_DEVICES and "
            "`nvidia-smi -L`, then exits"
        ),
    )
    result: dict[str, Any] = {"cluster_inspect": insp}
    with tempfile.TemporaryDirectory(prefix="hpclab-onboard-") as tmp:
        Path(tmp, "hpclab_gpu_smoke.py").write_text(_ONBOARD_SCRIPT, encoding="utf-8")
        prep = fetch(
            "experiment_prepare",
            {
                "idempotency_key": new_key("onboard"),
                "spec": onboard_spec(cluster_id, tmp, partition=partition, gpu_type=gpu_type),
            },
        )
        result["experiment_prepare"] = prep
        pdata = prep.get("data") or {}
        if pdata.get("status") == "preparing" and pdata.get("operation_id"):
            wait_operation(str(pdata["operation_id"]), timeout=300, quiet=STATE.json_output)
            pdata = {
                **pdata,
                **(fetch("plan_get", {"plan_id": pdata["plan_id"], "limit": 1}).get("data") or {}),
            }
    if not STATE.json_output:
        out(V.findings_table((pdata.get("validation") or {}).get("findings") or [], important_only=True))
    if pdata.get("status") != "ready":
        if STATE.json_output:
            print_json({"ok": False, **result})
        raise fail(
            f"the smoke plan is {pdata.get('status')}: {', '.join((pdata.get('validation') or {}).get('blocking') or []) or 'see findings'}",
            json_error=False,
        )
    main_runs = [r for r in pdata.get("runs") or [] if r.get("run_id")]
    if not main_runs:
        raise fail("the smoke plan has no run")
    run_id = str(main_runs[0]["run_id"])
    sub = fetch(
        "run_submit",
        {
            "run_id": run_id,
            "execution_plan_hash": main_runs[0].get("exec_plan_hash"),
            "idempotency_key": new_key("onboard"),
        },
    )
    result["run_submit"] = sub
    if not STATE.json_output:
        out(
            V.T(
                f"submitted run {run_id} (attempt {(sub.get('data') or {}).get('attempt_id')}); waiting for it…",
                "cyan",
            )
        )
    started = _now()
    last = ""
    data: dict[str, Any] = {}
    with interruptible(f"waiting (the job keeps running; follow it with `hpclab inspect {run_id}`)"):
        while True:
            env = fetch("run_inspect", {"run_id": run_id, "refresh": True}, allow_error=True)
            if env.get("ok"):
                data = dict(env.get("data") or {})
                line = _attempt_line(data)
                if line != last and not STATE.json_output:
                    note(f"{time.strftime('%H:%M:%S')} {line}", "cyan")
                    last = line
                aid = data.get("focus_attempt_id")
                a: dict[str, Any] = next(
                    (x for x in data.get("attempts") or [] if x.get("attempt_id") == aid), {}
                )
                settled = a.get("result_state") not in ("unchecked", "pending", None)
                if a.get("phase") in _ATTEMPT_DONE_PHASES and (settled or a.get("phase") != "terminal"):
                    break
            elif (env.get("error") or {}).get("code") == "auth_required":
                show(env)
                raise typer.Exit(4)
            if _now() - started >= timeout:
                if STATE.json_output:
                    print_json({"ok": False, **result, "run_inspect": env})
                raise fail(
                    f"the smoke job did not finish within {timeout:g}s (it keeps running)",
                    hint=f"hpclab inspect {run_id}",
                    json_error=False,
                )
            _sleep(_watch_interval(interval))
    result["run_inspect"] = {"ok": True, "data": data}
    aid = str(data.get("focus_attempt_id"))
    a = attempt_view(data, aid)
    passed = (
        a.get("scheduler_state") == "COMPLETED"
        and a.get("app_state") == "exited_ok"
        and a.get("result_state") == "valid"
    )
    logs_env = fetch("run_logs", {"attempt_id": aid, "tail_lines": 40}, allow_error=True)
    if STATE.json_output:
        print_json({"ok": passed, **result, "run_logs": logs_env})
        raise typer.Exit(0 if passed else EXIT_ERROR)
    text = str((logs_env.get("data") or {}).get("text") or "") if logs_env.get("ok") else ""
    if text:
        out(V.T("job output (tail):", "bold"))
        for line in text.rstrip("\n").splitlines()[-40:]:
            out(V.T("  " + line, "dim"))
    if passed:
        out(
            Text(
                f"PASS: {cluster_id} ran a 1-GPU job end to end (job {(a.get('job') or {}).get('job_id')})",
                style="bold green",
            )
        )
        out(
            V.next_steps(["hpclab prepare <your spec>.yaml   # with a smoke: block for your own environment"])
        )
        return
    out(
        Text(
            f"FAIL: the smoke job ended {a.get('scheduler_state')} / app {a.get('app_state')} / results {a.get('result_state')}",
            style="bold red",
        )
    )
    if data.get("diagnosis"):
        out(V.render_diagnosis(data["diagnosis"]))
    out(V.next_steps([f"hpclab logs {aid} --stderr --tail 100", f"hpclab inspect {run_id}"]))
    raise typer.Exit(EXIT_ERROR)
