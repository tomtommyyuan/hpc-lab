"""``hpclab demo``: a narrated tour of hpclab on the local Slurm simulator.

The tour runs the real product end to end, exactly as a user (or an agent over MCP) would:
a controller autostarted by the client, the tiny-classifier example prepared into immutable
runs, a 2x3 matrix behind a smoke gate, a node failure injected into one cell, the diagnosis,
a recovery plan, and a traceable LaTeX/Markdown table.

Isolation: everything happens in a throwaway hpclab home ``<home>/demo/<timestamp>/`` (or in the
temp directory when the home's path has characters job scripts cannot carry) with its own
controller, ledger and simulator, so the user's clusters, experiments and controller are never
touched. At the end -- also after Ctrl-C -- the demo's controller is stopped, its simulated jobs
are killed and the directory is removed, unless ``keep``, which leaves the home to explore with
``hpclab --home <dir> ...``.

Output goes to the given rich ``Console`` only (never ``print``).
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml
from rich.console import Console
from rich.markup import escape

from hpclab.bundled import bundled_path
from hpclab.config import Paths, get_paths
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.controller.files import tail_lines
from hpclab.domain.errors import HpcLabError
from hpclab.domain.ids import new_id

__all__ = ["DemoError", "run_demo"]

T = TypeVar("T")

EXAMPLE = "tiny-classifier"
STEPS = 8
EXIT_INTERRUPTED = 130  # like the CLI's Ctrl-C exit status
#: Paths in job scripts are restricted to these characters (services/jobscript.py); the
#: simulator's "remote" filesystem lives inside the demo home, so its path must match.
_SCRIPT_SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./@+,=:~-]*$")
# Simulator timings for a snappy tour (job lifecycles well under a second of overhead).
_SIM_TIMINGS: dict[str, Any] = {
    "min_job_age_s": 1.0,
    "kill_wait_s": 0.3,
    "poll_interval_s": 0.05,
    "fate_delay_s": 0.5,
}
_CODE_STYLE = {
    "V": "bold green",
    "R": "bold cyan",
    "Q": "yellow",
    "S": "yellow",
    "P": "dim",
    "E": "bold blue",
    "F": "bold red",
    "C!": "bold magenta",
    "U": "bold magenta",
    "X": "red",
    "?": "magenta",
}


class DemoError(Exception):
    """A step of the tour could not complete; ``hint`` says what to look at."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass
class _Tour:
    console: Console
    fast: bool
    paths: Paths  # the demo's own home
    client: HpcLabClient
    t0: float = field(default_factory=time.monotonic)
    n: int = 0
    faults_set: bool = False
    sim_root: Path | None = None  # where init_sim put the simulator (it may relocate it)

    # -- narration ---------------------------------------------------------------------
    def elapsed(self) -> str:
        s = int(time.monotonic() - self.t0)
        return f"{s // 60}:{s % 60:02d}"

    def step(self, title: str, why: str | None = None) -> None:
        self.n += 1
        self.console.print()
        self.console.rule(f"[bold][{self.n}/{STEPS}] {escape(title)}[/bold]", align="left", style="cyan")
        if why:
            self.console.print(f"[dim]{escape(why)}[/dim]")
        self.pause(1.5)

    def say(self, text: str, *, markup: bool = False) -> None:
        self.console.print("  " + (text if markup else escape(text)), highlight=False)

    def ok(self, text: str) -> None:
        self.console.print(f"  [green]✓[/green] {escape(text)}", highlight=False)

    def note(self, text: str) -> None:
        self.console.print(f"  [dim]{escape(text)}[/dim]", highlight=False)

    def pause(self, seconds: float) -> None:
        if not self.fast:
            time.sleep(seconds)

    # -- controller calls ----------------------------------------------------------------
    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """The Envelope of a successful call (``data`` plus lifted coverage/warnings)."""
        env = self.client.call(method, {k: v for k, v in params.items() if v is not None})
        if not env.get("ok"):
            err = env.get("error") or {}
            details = err.get("details") or {}
            raise DemoError(
                f"{method} failed: {err.get('code')}: {err.get('message')}",
                hint=details.get("remediation") or details.get("hint"),
            )
        return env

    def data(self, method: str, **params: Any) -> dict[str, Any]:
        data = self.call(method, **params).get("data")
        if not isinstance(data, dict):
            raise DemoError(f"{method} returned no data")
        return data

    def wait(self, what: str, pred: Callable[[], T | None], *, timeout: float, interval: float = 0.5) -> T:
        deadline = time.monotonic() + timeout
        while True:
            value = pred()
            if value:
                return value
            if time.monotonic() >= deadline:
                raise DemoError(
                    f"timed out after {timeout:.0f}s waiting for {what}",
                    hint="the simulator or the controller may be stuck; the controller log tail is below",
                )
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _codes(grid: dict[str, Any]) -> list[tuple[str, list[str]]]:
    rows = []
    for row in grid.get("rows") or []:
        label = ", ".join(str(v) for v in (row.get("values") or {}).values()) or str(row.get("key"))
        rows.append((label, [str(c.get("code")) for c in row.get("cells") or []]))
    return rows


def _styled(code: str) -> str:
    style = _CODE_STYLE.get(code, "")
    return f"[{style}]{escape(code)}[/{style}]" if style else escape(code)


def _grid_line(rows: list[tuple[str, list[str]]]) -> str:
    return "   ".join(f"{escape(label)}: {' '.join(_styled(c) for c in codes)}" for label, codes in rows)


def _short(digest: Any, n: int = 12) -> str:
    text = str(digest or "?")
    return text.split(":", 1)[-1][:n] if ":" in text else text[:n]


def _table_lines(markdown: str, heading: str) -> list[str]:
    """The first Markdown table under ``## <heading>``."""
    lines = markdown.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"## {heading}")
    except StopIteration:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            out.append(line)
        elif out:
            break
    return out


# ---------------------------------------------------------------------------
# The tour
# ---------------------------------------------------------------------------


def _intro(tour: _Tour, root: Path, keep: bool, *, relocated_from: Path | None = None) -> None:
    c = tour.console
    c.print()
    length = "about 30 seconds with --fast" if tour.fast else "about a minute"
    c.print(f"[bold]HPC Lab demo[/bold] — a guided tour on a simulated Slurm cluster ({length})")
    c.print(
        "You will see: prepare → smoke gate → a 2×3 sweep → a node failure → diagnosis → recovery → "
        "a paper table where every number traces back to a Slurm job."
    )
    c.print(
        f"[dim]It runs in its own throwaway hpclab home ({escape(str(root))}); your clusters, experiments and "
        f"controller are not touched. {'It is kept afterwards (--keep).' if keep else 'It is removed at the end (--keep keeps it).'}[/dim]"
    )
    if relocated_from is not None:
        c.print(
            f"[dim]It is in the temp directory because the path of your hpclab home ({escape(str(relocated_from))}) "
            "has characters that job scripts cannot use (a space or non-ASCII).[/dim]"
        )


def _setup(tour: _Tour) -> Any:
    """Controller + simulator; returns the SimCluster the controller really uses (``init_sim``
    places it outside the demo home when that path cannot appear in job scripts)."""
    from hpclab.services.clusters import effective_sim_root
    from hpclab.sim.cluster import SimCluster
    from hpclab.sim.config import SimConfig

    tour.step(
        "Start a controller and register the simulator",
        "One controller per hpclab home is the single writer of the ledger; the CLI, Python and every "
        "MCP server (your coding agent) are reconnectable clients of it.",
    )
    info = tour.client.ensure_running()
    tour.ok(f"controller running (pid {info.get('pid')}); it was autostarted by the first client call")
    data = tour.data("init_sim")
    cl = data.get("cluster") or {}
    tour.ok(
        f"cluster '{cl.get('cluster_id')}' registered: partitions {', '.join(cl.get('partitions') or [])}; "
        f"user {cl.get('user')}, account {cl.get('account')}"
    )
    tour.note("sbatch/squeue/sacct/scancel are simulated; jobs run as local processes on 'GPU nodes'.")
    tour.sim_root = effective_sim_root(tour.paths)
    return SimCluster.ensure(tour.sim_root, SimConfig().replace(**_SIM_TIMINGS))  # snappy timings


def _git_commit(project: Path) -> str | None:
    """Make the demo project a git repository with one commit (so the trace shows a commit).
    Best effort: without git, or if anything fails, the project is snapshotted as a plain tree."""
    git = shutil.which("git")
    if git is None:
        return None
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    base = [
        git,
        "-c",
        "user.name=hpclab demo",
        "-c",
        "user.email=demo@hpclab.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",  # the user's global hooks (e.g. pre-commit) must not veto it
    ]
    try:
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "tiny-classifier (hpclab demo)"]):
            subprocess.run([*base, *args], cwd=project, env=env, capture_output=True, timeout=30, check=True)
        out = subprocess.run(
            [*base, "rev-parse", "HEAD"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(project / ".git", ignore_errors=True)
        return None
    return out.stdout.strip() or None


def _project(tour: _Tour, root: Path, fast: bool) -> tuple[Path, dict[str, Any]]:
    tour.step(
        "The research project",
        "examples/tiny-classifier: a stdlib-only training script that logs metrics, checkpoints, resumes "
        "exactly and can fail on purpose — a stand-in for your own project.",
    )
    examples = bundled_path("examples")
    src = examples / EXAMPLE if examples else None
    if src is None or not (src / "hpclab.yaml").is_file():
        raise DemoError("the tiny-classifier example is not bundled with this installation")
    project = root / EXAMPLE
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("__pycache__", "outputs", "*.pyc"))
    spec: dict[str, Any] = yaml.safe_load((project / "hpclab.yaml").read_text(encoding="utf-8"))
    spec["code"] = {**(spec.get("code") or {}), "project_root": str(project)}
    if not fast:  # ~6 s jobs, so the grid shows them queued and running
        spec["entrypoint"]["argv"] = [*spec["entrypoint"]["argv"], "--sleep-per-step", "0.02"]
    matrix = spec.get("matrix") or {}
    seeds = spec.get("seeds") or []
    commit = _git_commit(project)
    tour.ok(f"copied to {project}" + (f" (git commit {commit[:10]})" if commit else ""))
    tour.say(f"[bold]question[/bold]: {escape(' '.join(str(spec.get('question', '')).split()))}", markup=True)
    axes = " × ".join(f"{k} {v}" for k, v in matrix.items())
    cells = max(1, len(seeds))
    for values in matrix.values():
        cells *= max(1, len(values))
    smoke = ", plus 1 smoke run that gates them" if spec.get("smoke") else ""
    tour.say(f"matrix: {axes} × seeds {seeds} = {cells} runs{smoke}")
    return project, spec


def _prepare(tour: _Tour, spec: dict[str, Any], root: Path) -> dict[str, Any]:
    tour.step(
        "Prepare: freeze code and config into immutable runs",
        "experiment_prepare snapshots the code (content-addressed, read-only on the cluster), expands the "
        "matrix and validates everything. Nothing is submitted yet.",
    )
    env = tour.call("experiment_prepare", idempotency_key=f"demo-{new_id('request')}", spec=spec)
    prep = env["data"]
    if prep.get("status") != "ready":
        validation = prep.get("validation") or {}
        blocking = [
            f
            for f in validation.get("findings") or []
            if f.get("status") == "fail" and f.get("severity") == "block"
        ]
        why = "; ".join(f"{f.get('check')}: {str(f.get('message'))[:400]}" for f in blocking[:2])
        raise DemoError(
            f"the plan is {prep.get('status')}: {why or validation.get('blocking')}",
            hint=next((str(f["remediation"]) for f in blocking if f.get("remediation")), None),
        )
    budget = prep.get("budget") or {}
    snap = (prep.get("snapshot") or {}).get("ref") or {}
    tour.ok(
        f"plan {prep['plan_id']} ready: {prep.get('runs_total')} runs, "
        f"{float(budget.get('requested_gpu_hours_total') or 0):.2f} GPU-h requested of a "
        f"{budget.get('budget_gpu_hours')} GPU-h budget (gpus × time limit, never optimistic runtime)"
    )
    if snap:
        tour.ok(
            f"code snapshot {_short(snap.get('digest'))}… — edits you make now never leak into queued jobs"
        )
    findings = (prep.get("validation") or {}).get("findings") or []
    counts: dict[str, int] = {}
    for f in findings:
        counts[str(f.get("status", "?"))] = counts.get(str(f.get("status", "?")), 0) + 1
    order = ["pass", "warn", "fail", "unchecked"]
    ordered = sorted(counts.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))
    tour.say(f"validation of {len(findings)} checks: " + ", ".join(f"{n} {s}" for s, n in ordered))
    shown = [f for f in findings if f.get("status") in ("fail", "warn")] + [
        f for f in findings if f.get("status") == "unchecked"
    ]
    for f in shown[:5]:
        color = {"fail": "red", "warn": "yellow"}.get(str(f.get("status")), "magenta")
        tour.say(
            f"  [{color}]{escape(str(f.get('status')))}[/{color}] {escape(str(f.get('check')))}: "
            f"{escape(str(f.get('message')).replace(str(root) + os.sep, '')[:140])}",
            markup=True,
        )
    tour.note("'unchecked' means hpclab could not verify it — it is never reported as a pass.")
    return prep


def _dispatch(tour: _Tour, prep: dict[str, Any], sim: Any) -> tuple[str, dict[str, Any]]:
    tour.step(
        "Submit the matrix: smoke gate first, at most 2 jobs at a time",
        "workflow_submit returns at once; the controller dispatches, observes and validates in the "
        "background (close the terminal and it keeps going).",
    )
    wf = tour.data(
        "workflow_submit",
        idempotency_key=f"demo-{new_id('request')}",
        plan_id=prep["plan_id"],
        plan_hash=prep["plan_hash"],
    )
    wid = str(wf["workflow_id"])
    tour.ok(f"workflow {wid}: {wf.get('planned_run_count')} runs, max_parallel {wf.get('max_parallel')}")
    tour.say(
        "grid codes: "
        + " ".join(
            f"{_styled(k)} {v}"
            for k, v in (
                ("V", "valid"),
                ("R", "running"),
                ("Q", "queued"),
                ("S", "submitting"),
                ("P", "planned"),
                ("E", "completed, evaluation pending (not a failure)"),
                ("F", "failed"),
                ("C!", "completed, result invalid"),
            )
        ),
        markup=True,
    )

    # The failure: once the smoke job has been accepted by Slurm, tell the simulator that the node
    # under the NEXT job will die. The next job is a main cell (they wait for the smoke gate).
    tour.wait("the smoke job to reach Slurm", lambda: len(sim.jobs()) >= 1, timeout=60, interval=0.1)
    sim.set_faults(node_fail_next=1)
    tour.faults_set = True
    tour.say(
        "[bold yellow]⚡ injected[/bold yellow]: the node under the next job will fail "
        "[dim](simulator fault node_fail_next=1)[/dim]",
        markup=True,
    )

    last: tuple[Any, ...] | None = None

    def progressed() -> dict[str, Any] | None:
        nonlocal last
        w = tour.data("workflow_get", workflow_id=wid, limit=1)
        grid = tour.data("runs_list", workflow_id=wid, view="grid").get("grid") or {}
        rows = _codes(grid)
        gate = (w.get("gate") or {}).get("state")
        key = (w.get("state"), gate, tuple((label, tuple(c)) for label, c in rows))
        if key != last:
            last = key
            gate_s = f" · smoke gate {gate}" if gate else ""
            tour.say(
                f"[dim]{tour.elapsed():>5}[/dim]  {escape(str(w.get('state')))}{escape(gate_s)}   {_grid_line(rows)}",
                markup=True,
            )
        if w.get("state") in ("completed", "cancelled", "gate_failed"):
            return w
        return None

    done = tour.wait("the matrix to finish", progressed, timeout=240, interval=0.5 if tour.fast else 1.0)
    if done.get("state") == "gate_failed":
        gate = done.get("gate") or {}
        diag = (
            (tour.data("run_inspect", run_id=gate["run_id"]).get("diagnosis") or {})
            if gate.get("run_id")
            else {}
        )
        raise DemoError(
            f"the smoke run failed ({diag.get('primary_code') or gate.get('reason') or 'see run_inspect'}), so the "
            "matrix was never dispatched -- the smoke gate doing its job",
            hint=str(diag.get("summary") or "") or None,
        )
    if done.get("state") != "completed":
        raise DemoError(f"the workflow ended {done.get('state')}")
    return wid, done


def _collect(tour: _Tour, run_set: str) -> None:
    """evaluation_collect over the run set: evaluations usually exist already (collected
    automatically when outputs validate); this makes sure, and waits for a background batch."""
    data = tour.data("evaluation_collect", idempotency_key=f"demo-{new_id('request')}", run_set_ref=run_set)
    counts = data.get("counts") or {}
    op_id = data.get("operation_id")
    if data.get("pending") and op_id:

        def finished() -> dict[str, Any] | None:
            op = tour.data("operation_get", operation_id=op_id)
            if op.get("state") == "failed":
                raise DemoError(f"evaluation_collect failed: {(op.get('error') or {}).get('message')}")
            return op if op.get("state") == "succeeded" else None

        tour.wait("the evaluations", finished, timeout=120)
    skipped = Counter(str(s_.get("reason")) for s_ in data.get("skipped") or [])
    tour.note(
        f"evaluation_collect: {counts.get('collected', 0)} collected now, "
        f"{skipped.get('already_evaluated', 0)} already evaluated automatically when their outputs validated"
    )


def _failure(tour: _Tour, wid: str, exp_id: str) -> dict[str, Any] | None:
    tour.step(
        "A cell failed — what hpclab did, and did not do",
        "run_inspect puts the scheduler's own account first (state, exit code, node), then matches the "
        "logs against known failure signatures; the diagnosis is deterministic, with its evidence.",
    )
    cells = tour.data("workflow_get", workflow_id=wid, limit=100).get("cells") or []
    failed = [c for c in cells if c.get("cell_state") == "failed"]
    if not failed:
        tour.say(
            "[yellow]no cell failed this time[/yellow] (the injected fault did not land); skipping recovery",
            markup=True,
        )
        return None
    cell = failed[0]
    insp = tour.data("run_inspect", run_id=cell["run_id"])
    att: dict[str, Any] = next(
        (a for a in insp.get("attempts") or [] if a.get("attempt_id") == cell.get("attempt_id")), {}
    )
    budget = att.get("budget") or {}
    tour.ok(
        f"failed cell: {cell.get('label')} — Slurm job {cell.get('job_id')} ended "
        f"{att.get('scheduler_state')} on {(att.get('job') or {}).get('node_list')}"
    )
    tour.say("hpclab recorded the failure as Slurm reported it and kept dispatching the other cells")
    tour.say("(infrastructure failures never trip the circuit breaker); it did NOT resubmit anything;")

    def gpu_h(value: Any) -> str:
        return f"{value:.2f} GPU-h" if isinstance(value, int | float) else "unknown GPU-h"

    tour.say(
        f"the attempt's GPU-hours stay on record ({budget.get('state')}: {gpu_h(budget.get('settled_consumed'))} "
        f"used of {gpu_h(budget.get('reserved'))} reserved)."
    )
    diag = insp.get("diagnosis") or {}
    item = (diag.get("items") or [{}])[0]
    tour.say(
        f"[bold]diagnosis[/bold] {escape(str(diag.get('primary_code')))} — {escape(str(item.get('title', '')))} "
        f"(confidence {escape(str(item.get('confidence')))})",
        markup=True,
    )
    if item.get("explanation"):
        tour.note(str(item["explanation"])[:300])
    for line in (item.get("evidence") or [])[:2]:
        tour.say(f"  [dim]evidence:[/dim] {escape(str(line)[:150])}", markup=True)
    if item.get("suggested_strategies"):
        tour.say(f"suggested: {', '.join(item['suggested_strategies'])}")

    _collect(tour, wid)
    cmp = tour.data(
        "results_compare",
        idempotency_key=f"demo-{new_id('request')}",
        planned_run_set_ref=exp_id,
        metric_names=["acc"],
    )
    parts = []
    for g in cmp.get("groups") or []:
        st = (g.get("metrics") or {}).get("acc") or {}
        gap = (
            f" (seed {', '.join(map(str, st.get('failed_seeds') or []))} failed)"
            if st.get("failed_seeds")
            else ""
        )
        parts.append(f"{g.get('label')} n={st.get('n')}/{st.get('expected_n')}{gap}")
    tour.say("the table right now: " + "; ".join(parts))
    tour.note("The failed seed stays in the denominator — it is never silently dropped from a mean.")
    return cell


def _recover(tour: _Tour, wid: str, exp_id: str) -> None:
    tour.step(
        "Recovery plan",
        "run_recover(selector=missing_valid) plans every cell without a valid result from its diagnosis: "
        "retry, resume from a validated checkpoint, or adjust resources. It never submits by itself.",
    )
    rec = tour.data(
        "run_recover",
        idempotency_key=f"demo-{new_id('request')}",
        selector={"workflow_id": wid, "cells": "missing_valid"},
    )
    cells = rec.get("cells") or []
    for c in cells:
        if c.get("eligible"):
            tour.ok(f"{c.get('label')}: {c.get('strategy')} (diagnosis {c.get('diagnosis_code')})")
        else:
            tour.say(
                f"[yellow]{escape(str(c.get('label')))}[/yellow]: not eligible — {escape(str(c.get('blocking_reason')))}",
                markup=True,
            )
    if not any(c.get("eligible") for c in cells):
        raise DemoError("no cell is eligible for recovery", hint=str(rec.get("next_step") or ""))
    wf2 = tour.data("workflow_submit", idempotency_key=f"demo-{new_id('request')}", plan_id=rec["plan_id"])
    wid2 = str(wf2["workflow_id"])
    tour.ok(f"recovery plan {rec['plan_id']} dispatched as workflow {wid2}")

    def complete() -> list[tuple[str, list[str]]] | None:
        w = tour.data("workflow_get", workflow_id=wid2, limit=1)
        if w.get("state") in ("cancelled", "gate_failed"):
            raise DemoError(f"the recovery workflow ended {w.get('state')}")
        grid = tour.data("runs_list", experiment_id=exp_id, view="grid").get("grid") or {}
        rows = _codes(grid)
        return rows if w.get("state") == "completed" and all(c == "V" for _, cs in rows for c in cs) else None

    rows = tour.wait("the recovered cell to validate", complete, timeout=120)
    tour.say(f"[dim]{tour.elapsed():>5}[/dim]  complete   {_grid_line(rows)}", markup=True)


def _evidence(tour: _Tour, exp_id: str) -> None:
    tour.step(
        "Evidence: a paper table where every number is traceable",
        "results_compare over the PLANNED runs, then evidence_export: LaTeX (booktabs), Markdown, CSV and a "
        "JSON manifest. Each LaTeX number is tagged \\hpcval{bundle:cell}{value} so `hpclab paper check` can "
        "later flag stale or hand-edited numbers.",
    )
    ev = tour.data(
        "evidence_export",
        purpose="hpclab demo",
        idempotency_key=f"demo-{new_id('request')}",
        planned_run_set_ref=exp_id,
        metric_names=["acc"],
    )
    bundle = str(ev["bundle_id"])
    counts = ev.get("counts") or {}
    tour.ok(f"bundle {bundle}: {counts.get('included')}/{counts.get('planned_runs')} planned runs included")
    tex = str(tour.data("evidence_get", bundle_id=bundle, file="table.tex").get("content") or "")
    md = str(tour.data("evidence_get", bundle_id=bundle, file="summary.md").get("content") or "")
    tour.console.print("\n  [bold]LaTeX[/bold] (table.tex)")
    body = [ln for ln in tex.splitlines() if ln and not ln.startswith("%")]
    for ln in body:
        tour.console.print(f"    {escape(ln)}", highlight=False, soft_wrap=True)
    tour.console.print("\n  [bold]Markdown[/bold] (summary.md, results and paired difference)")
    for ln in [*_table_lines(md, "Results"), "", *_table_lines(md, "Differences vs baseline (descriptive)")]:
        tour.console.print(f"    {escape(ln)}", highlight=False, soft_wrap=True)
    tour.note("Descriptive only: n, mean ± std and a 95% interval — hpclab does not bold a 'winner'.")

    cell_ref = f"{bundle}:mixup.acc.mean"
    tr = tour.data("trace", cell_ref=cell_ref)
    tour.console.print(
        f"\n  [bold]Trace of one number[/bold]: mixup acc mean = {tr.get('value')} ({cell_ref})"
    )
    seed = next((s for s in tr.get("seeds") or [] if s.get("chain")), None)
    if seed:
        ch = seed["chain"]
        ev_ = ch.get("evaluation") or {}
        art = ch.get("artifact") or {}
        att = ch.get("attempt") or {}
        job = ch.get("job") or {}
        run = ch.get("run") or {}
        code = ch.get("code") or {}
        env = ch.get("env") or {}
        metric = ev_.get("metric") or {}
        chain = [
            f"seed {seed.get('seed')}: value {seed.get('value')} ← evaluation {ev_.get('evaluation_id')} "
            f"({ev_.get('evaluator')}, {metric.get('split')} split, record {((metric.get('source') or {}).get('locator'))})",
            f"← artifact {art.get('name')} sha256 {_short(art.get('digest'))}… ({art.get('digest_source')}, {art.get('integrity')})",
            f"← attempt {att.get('attempt_id')} #{att.get('seq')} {att.get('scheduler_state')} on {att.get('node_list')}",
            f"← Slurm job {job.get('job_id')} on {job.get('cluster_id')} (owner {job.get('owner')}, submitted {job.get('submit_time')})",
            f"← run {run.get('label')} spec {_short(run.get('spec_hash'))}… ← code {_short(code.get('digest'))}…"
            + (f" (git {str(code.get('git_commit'))[:10]})" if code.get("git_commit") else ""),
            f"← environment: {env.get('gpu')}, python {env.get('python')}, driver {env.get('driver_version')}",
        ]
        for i, ln in enumerate(chain):
            tour.console.print(("    " if i == 0 else "      ") + escape(ln), highlight=False, soft_wrap=True)
    n_seeds = len(tr.get("seeds") or [])
    tour.note(f"The same chain exists for all {n_seeds} seeds of this cell and for every cell of the table.")


def _next_steps(tour: _Tour, keep: bool, exp_id: str, *, relocated_from: Path | None = None) -> None:
    tour.step("Where to go next")
    if relocated_from is not None:
        tour.console.print(
            f"  [yellow]note[/yellow]: your hpclab home ({escape(str(relocated_from))}) has characters job "
            "scripts cannot use (a space or non-ASCII): `hpclab init --sim` places the simulator in /var/tmp "
            "instead; a plain HPCLAB_HOME (or --home) such as ~/.hpclab avoids that.",
            highlight=False,
            soft_wrap=True,
        )
    lines = [
        (
            "Run it yourself",
            f"hpclab examples copy {EXAMPLE} my-project && hpclab init --sim && hpclab prepare my-project/hpclab.yaml",
        ),
        ("", "hpclab submit <plan id>  ·  hpclab grid <workflow id> --watch  ·  hpclab inspect <run id>"),
        (
            "Your cluster",
            "hpclab cluster profiles  ·  hpclab cluster add hai --profile hai --ssh-host <host> --user <you> --account <acct>",
        ),
        (
            "",
            "hpclab auth hai  (MFA in your own terminal)  ·  hpclab cluster inspect hai  ·  hpclab onboard hai",
        ),
        (
            "Your agent",
            "claude mcp add hpclab -- hpclab mcp serve   (`hpclab mcp config` for other MCP clients)",
        ),
        (
            "Skills",
            "hpclab skills install   (experiment design, run doctor, paper evidence, onboarding, lab overview)",
        ),
    ]
    for label, cmd in lines:
        tour.console.print(
            f"  [bold]{escape(label):<16}[/bold][cyan]{escape(cmd)}[/cyan]", highlight=False, soft_wrap=True
        )
    if keep:
        home = escape(shlex.quote(str(tour.paths.home)))  # pasteable even with spaces or quotes
        tour.console.print(
            f"\n  The demo home is kept: [cyan]hpclab --home {home} grid {exp_id}[/cyan] "
            f"(or export HPCLAB_HOME={home}); delete the directory when done.",
            soft_wrap=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _demo_root(paths: Paths) -> tuple[Path, Path | None]:
    """A fresh directory for the demo: ``<home>/demo/<stamp>``, or one in the temp directory when
    the home's path has characters job scripts cannot carry (the simulator's "remote" filesystem
    lives inside the demo home). Also returns ``<home>/demo`` (to remove once empty) or None."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{new_id('request')[-4:].lower()}"
    if not _SCRIPT_SAFE_PATH.match(str(paths.home / "demo" / name)):
        return Path(tempfile.mkdtemp(prefix=f"hpclab-demo-{name}-")), None
    if not paths.home.exists():  # `hpclab init` never ran: an hpclab home is private
        paths.home.mkdir(mode=0o700, parents=True)
    base = paths.home / "demo"
    base.mkdir(exist_ok=True)
    root = base / name
    root.mkdir(mode=0o700)
    return root, base


def _set_sigint(handler: Any) -> None:
    """Best effort: only the main thread can change signal handlers, and a handler installed
    outside Python (``None``) cannot be restored."""
    if handler is not None:
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGINT, handler)


@contextlib.contextmanager
def _sigint_deferred(original: Any) -> Iterator[None]:
    """Ignore Ctrl-C while cleaning up, so an impatient second Ctrl-C cannot orphan the demo's
    controller or leave its directory behind (cleanup is bounded: ~20 s at worst); then
    restore ``original``."""
    _set_sigint(signal.SIG_IGN)
    try:
        yield
    finally:
        _set_sigint(original)


def _make_writable(root: Path) -> None:
    """Published code snapshots are read-only; make everything deletable."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            p = Path(dirpath) / name
            with contextlib.suppress(OSError):
                st = p.lstat()
                if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                    p.chmod(st.st_mode | stat.S_IWUSR)


def _cleanup(
    tour: _Tour, demo_paths: Paths, root: Path, base: Path | None, keep: bool, console: Console
) -> list[str]:
    """Stop the demo's controller and simulated jobs; remove the directory unless ``keep``.
    Returns problems (reported, never raised: cleanup must not mask the demo's outcome)."""
    problems: list[str] = []
    tour.client.close()
    try:
        res = stop_controller(demo_paths, timeout=20.0, force=True)
        if not res.get("stopped", True):
            problems.append(f"the demo controller (pid {res.get('pid')}) did not stop")
    except Exception as exc:  # noqa: BLE001 - report and continue cleaning up
        problems.append(f"stopping the demo controller failed: {exc}")
    sim_root = tour.sim_root
    if sim_root is None:  # the tour stopped early: init_sim may still have placed a simulator
        try:
            from hpclab.services.clusters import effective_sim_root

            sim_root = effective_sim_root(demo_paths)
        except Exception:  # noqa: BLE001 - no overlay (or an unreadable one): the default place
            sim_root = demo_paths.sim_root("sim")
    if (sim_root / "state.db").exists():
        try:
            from hpclab.sim.cluster import SimCluster

            sim = SimCluster(sim_root)
            if tour.faults_set:
                sim.clear_faults()
            sim.shutdown(grace_s=2.0)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"stopping simulated jobs failed: {exc}")
    if not keep:
        _make_writable(root)
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            problems.append(f"could not remove {root}")
        if sim_root.exists() and not sim_root.is_relative_to(root):  # relocated by init_sim
            _make_writable(sim_root)
            shutil.rmtree(sim_root, ignore_errors=True)
            if sim_root.exists():
                problems.append(f"could not remove the demo's simulator {sim_root}")
        if base is not None:
            with contextlib.suppress(OSError):
                base.rmdir()  # <home>/demo, only if empty
    for p in problems:
        console.print(f"[yellow]cleanup[/yellow]: {escape(p)}")
    return problems


def run_demo(paths: Paths, *, console: Console, fast: bool = False, keep: bool = False) -> int:
    """Run the guided tour; 0 on success, 1 if a step failed (with the reason and a hint).

    ``paths`` is the user's hpclab home: the demo lives in a throwaway home below it
    (``<home>/demo/<stamp>``, or the temp directory if the home's path cannot appear in a job
    script). ``fast`` skips the narration pauses and uses the shortest jobs (for tests and
    impatient readers). ``keep`` keeps the demo home for exploring afterwards. Ctrl-C stops the
    tour, cleans up (a second Ctrl-C cannot interrupt that) and returns 130.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    try:
        root, base = _demo_root(paths)
        demo_paths = get_paths(root / "home").ensure()
    except OSError as exc:
        console.print(
            f"\n[red bold]demo stopped[/red bold]: cannot create the demo's home: {escape(str(exc))}"
        )
        console.print(
            "  [yellow]hint[/yellow]: HPCLAB_HOME (or --home) must be a directory you own and can write "
            f"(now: {escape(str(paths.home))})"
        )
        return 1
    tour = _Tour(console=console, fast=fast, paths=demo_paths, client=HpcLabClient(demo_paths, timeout=60.0))
    rc = 1
    try:
        relocated_from = paths.home if base is None else None
        _intro(tour, root, keep, relocated_from=relocated_from)
        sim = _setup(tour)
        _, spec = _project(tour, root, fast)
        prep = _prepare(tour, spec, root)
        wid, _done = _dispatch(tour, prep, sim)
        failed = _failure(tour, wid, prep["experiment_id"])
        if failed is not None:
            _recover(tour, wid, prep["experiment_id"])
        else:
            tour.n += 1  # keep the step numbering honest
        _evidence(tour, prep["experiment_id"])
        _next_steps(tour, keep, prep["experiment_id"], relocated_from=relocated_from)
        console.print(f"\n[bold green]demo complete[/bold green] in {tour.elapsed()}")
        rc = 0
    except (DemoError, HpcLabError) as exc:
        console.print(f"\n[red bold]demo stopped[/red bold]: {escape(exc.message)}")
        hint = exc.hint if isinstance(exc, DemoError) else (exc.details or {}).get("hint")
        if hint:
            console.print(f"  [yellow]hint[/yellow]: {escape(str(hint))}")
        log_tail = tail_lines(demo_paths.log, max_lines=12)
        if log_tail:
            console.print(f"  [dim]controller log ({escape(str(demo_paths.log))}), last lines:[/dim]")
            for ln in log_tail:
                console.print(f"    [dim]{escape(ln)}[/dim]", highlight=False, soft_wrap=True)
        if not keep:
            console.print(
                "  [dim]rerun with --keep to keep the demo's home (ledger, logs, job files) for a look[/dim]"
            )
        rc = 1
    except KeyboardInterrupt:
        _set_sigint(signal.SIG_IGN)  # first thing: a second Ctrl-C must not skip the cleanup
        console.print(
            "\n[yellow]demo interrupted[/yellow]: stopping its controller and simulated jobs "
            "(a few seconds) ...",
            highlight=False,
        )
        rc = EXIT_INTERRUPTED
    finally:
        with _sigint_deferred(original_sigint):
            problems = _cleanup(tour, demo_paths, root, base, keep, console)
    if rc == EXIT_INTERRUPTED and not problems:
        where = f"its home is kept at {root}" if keep else "its directory was removed"
        console.print(
            f"[dim]cleaned up: the demo's controller and simulated jobs are stopped; {escape(where)}. "
            "Nothing outside the demo was touched.[/dim]",
            highlight=False,
        )
    return rc
