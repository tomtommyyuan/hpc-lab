"""runs / grid / inspect / logs / cancel / recover / resolve / which / ops against a scripted controller."""

from __future__ import annotations

from typing import Any

import pytest
from cli_support import Cli, env, err, ok

EXP = "exp_01M3K9Q1CC33H1R96NRPJEDAVW"
WF = "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA"
FAILED_RUN = "run_01M3K9Q1D8QH3MCQSDR0S9N2WS"
FAILED_ATT = "att_01M3K9QD2FYB3TBD0RR59FXFBT"
GOOD_RUN = "run_01M3K9Q1D83ZECZ5F8ZVGB3VS7"
GOOD_ATT = "att_01M3K9Q7QY2C92358BRYJ7C2R7"
PENDING_RUN = "run_01M3K9QS0RSQ7AMQW9WZANQFMD"
PENDING_ATT = "att_01M3K9QS0T356550MDEFTXEAKN"


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def test_runs_lists_latest_attempts_and_counts(cli: Cli) -> None:
    cli.client.responses["runs_list"] = env("runs_list.list_done")
    r = cli.run("runs", "--experiment", EXP, "--state", "failed", "--state", "succeeded", "--limit", "5")
    assert r.exit_code == 0, r.text
    assert "method-mixup.seed-1" in r.stdout and "NODE_FAIL" in r.stdout and "node_fail" in r.stdout
    assert FAILED_RUN in r.stdout
    assert "runs: 5/7 valid" in r.stdout
    p = cli.client.params("runs_list")
    assert p == {"experiment_id": EXP, "states": ["failed", "succeeded"], "limit": 5}


def test_runs_needs_exactly_one_scope_and_json(cli: Cli) -> None:
    assert cli.run("runs").exit_code == 2
    assert cli.run("runs", "--experiment", EXP, "--workflow", WF).exit_code == 2
    cli.client.responses["runs_list"] = env("runs_list.list_running")
    r = cli.run("--json", "runs", "--workflow", WF)
    assert r.exit_code == 0 and r.json == env("runs_list.list_running")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def test_grid_shows_codes_values_legend_counts_and_problems(cli: Cli) -> None:
    cli.client.responses["runs_list"] = env("runs_list.grid_done")
    cli.client.responses["workflow_get"] = env("workflow_get.done")
    r = cli.run("grid", WF, "--metric", "acc", "--rows", "method", "--cols", "seed")
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert "V 0.8440" in out and "V 0.8700" in out
    assert "legend:" in out and "F failed" in out and "E completed, evaluation pending — not a failure" in out
    assert "cells: 4/6 valid" in out
    # the failed cell is listed with its diagnosis; the E cell (waiting for evaluation) is not a problem
    assert "F mixup seed=1: node_fail" in out
    assert "E mixup seed=3" not in out
    assert f"1 cell(s) completed, evaluation pending (automatic, or: hpclab collect {WF})" in out
    assert f"workflow {WF} completed" in out and "gate passed" in out
    p = cli.client.params("runs_list")
    assert p == {"view": "grid", "metric": "acc", "rows": ["method"], "cols": ["seed"], "workflow_id": WF}


def test_grid_counts_evaluation_pending_cells_beyond_this_page(cli: Cli) -> None:
    """Rows are paginated: the "completed, evaluation pending" line counts every run (the
    service's counts), not only the E cells of the page shown."""
    g = env("runs_list.grid_done")
    g["data"]["grid"]["total_rows"] = 9
    g["data"]["counts"] = {"V": 20, "F": 1, "E": 5, "planned": 26, "valid": 20}
    cli.client.responses["runs_list"] = g
    cli.client.responses["workflow_get"] = env("workflow_get.done")
    r = cli.run("grid", WF)
    assert r.exit_code == 0, r.text
    assert f"5 cell(s) completed, evaluation pending (automatic, or: hpclab collect {WF})" in r.stdout


def test_grid_experiment_target_and_plan_target(cli: Cli) -> None:
    cli.client.responses["runs_list"] = env("runs_list.grid_running")
    r = cli.run("grid", EXP)
    assert r.exit_code == 0
    assert cli.client.params("runs_list")["experiment_id"] == EXP
    assert "workflow_get" not in cli.client.methods()
    assert "Q" in r.stdout and "P" in r.stdout
    assert cli.run("grid", "pln_01M3K9Q1CC6RNE08MM569NQ42F").exit_code == 2


def test_grid_stale_cells_explain_the_question_mark(cli: Cli) -> None:
    g = env("runs_list.grid_running")
    g["data"]["grid"]["rows"][0]["cells"][0].update(code="?", reason="last observation 2h ago")
    cli.client.responses["runs_list"] = g
    r = cli.run("grid", EXP)
    assert "? baseline seed=1: last observation 2h ago" in r.stdout
    assert "observation is too old to trust" in r.stdout


def _all_valid(e: dict[str, Any]) -> dict[str, Any]:
    for row in e["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] == "E":
                c.update(code="V", reason=None, value=0.85)
    e["data"]["counts"] = {"V": 5, "F": 1, "planned": 6, "valid": 5}
    return e


def test_grid_watch_polls_until_every_cell_is_terminal(cli: Cli) -> None:
    done = _all_valid(env("runs_list.grid_done"))
    cli.client.responses["runs_list"] = [env("runs_list.grid_running"), env("runs_list.grid_done"), done]
    cli.client.responses["workflow_get"] = [env("workflow_get.running"), env("workflow_get.done")]
    r = cli.run("grid", WF, "--watch", "--interval", "3")
    assert r.exit_code == 0, r.text
    assert cli.client.methods().count("runs_list") == 3  # the E cell kept the watch going
    assert cli.sleeps == [3.0, 3.0]
    assert "done: every cell is terminal" in r.stderr
    assert "cells: 5/6 valid" in r.stdout
    assert "not evaluated" not in r.stderr


def test_grid_watch_gives_evaluation_60s_then_stops_with_a_collect_hint(cli: Cli) -> None:
    """Only E cells left (completed, evaluation pending): auto-evaluation normally follows within
    seconds; if it does not, stop after ~60 s and say how to evaluate them."""
    cli.client.responses["runs_list"] = env("runs_list.grid_done")
    cli.client.responses["workflow_get"] = env("workflow_get.done")
    r = cli.run("grid", WF, "--watch", "--interval", "5")
    assert r.exit_code == 0, r.text
    assert sum(cli.sleeps) == 60.0
    assert f"1 cell(s) completed but not evaluated: run hpclab collect {WF}" in r.stderr
    assert "timed out" not in r.stderr


def test_grid_watch_does_not_wait_for_an_evaluation_nobody_queued(cli: Cli) -> None:
    """E cells the service says have no evaluation queued (none declared, or the automatic one
    was skipped) never turn V on their own: stop at once with the collect hint, no 60 s wait."""
    g = env("runs_list.grid_done")
    g["data"]["grid"]["notes"] = [
        "1 run(s) are E: completed, evaluation pending (outputs being checked or not evaluated yet) - not "
        "a failure; they turn V once a valid evaluation exists; 1 of them has no evaluation queued (none "
        f'declared in the spec, or the automatic one was skipped): evaluation_collect(run_set_ref="{WF}", '
        'idempotency_key="<new key>")'
    ]
    cli.client.responses["runs_list"] = g
    cli.client.responses["workflow_get"] = env("workflow_get.done")
    r = cli.run("grid", WF, "--watch", "--interval", "5")
    assert r.exit_code == 0, r.text
    assert cli.sleeps == []
    assert f"1 cell(s) completed but not evaluated: run hpclab collect {WF}" in r.stderr


def test_grid_watch_stops_when_no_planned_cell_can_be_dispatched(cli: Cli) -> None:
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] in ("Q", "R"):
                c.update(code="F", reason="import_error")
    _recount(g)
    wf = env("workflow_get.running")
    wf["data"].update(state="gate_failed")
    cli.client.responses["runs_list"] = g
    cli.client.responses["workflow_get"] = wf
    r = cli.run("grid", WF, "--watch")
    assert r.exit_code == 0, r.text
    assert "done: workflow gate_failed" in r.stderr and "will not be dispatched" in r.stderr
    assert cli.sleeps == []


def test_grid_watch_keeps_going_while_the_workflow_can_dispatch(cli: Cli) -> None:
    """P cells of an open workflow (e.g. between two dispatch rounds): keep watching."""
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] in ("Q", "R"):
                c.update(code="V", reason=None)
    _recount(g)
    cli.client.responses["runs_list"] = [g, _all_valid(env("runs_list.grid_done"))]
    cli.client.responses["workflow_get"] = [env("workflow_get.running"), env("workflow_get.done")]
    r = cli.run("grid", WF, "--watch", "--interval", "2")
    assert r.exit_code == 0, r.text
    assert cli.sleeps == [2.0]
    assert "done: every cell is terminal" in r.stderr


def test_grid_watch_experiment_stops_when_notes_say_cells_will_not_start(cli: Cli) -> None:
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] in ("Q", "R"):
                c.update(code="F", reason="import_error")
    _recount(g)
    g["data"]["grid"]["notes"] = [
        f"workflow {WF}: state gate_failed, dispatch open - its undispatched cells will not start on their own"
    ]
    cli.client.responses["runs_list"] = g
    r = cli.run("grid", EXP, "--watch")
    assert r.exit_code == 0, r.text
    assert "will not be dispatched" in r.stderr and cli.sleeps == []


def test_grid_watch_experiment_with_idle_planned_cells_stops_after_a_grace(cli: Cli) -> None:
    """An experiment grid cannot see whether some workflow will still dispatch its P cells:
    nothing active for 60 s means nothing will start on its own."""
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] in ("Q", "R"):
                c.update(code="V", reason=None)
    _recount(g)
    cli.client.responses["runs_list"] = g
    r = cli.run("grid", EXP, "--watch", "--interval", "10")
    assert r.exit_code == 0, r.text
    assert sum(cli.sleeps) == 60.0
    assert "not dispatched and nothing is running" in r.stderr
    assert f"hpclab submit {EXP}" in r.stderr


def _recount(e: dict[str, Any]) -> dict[str, Any]:
    """Per-code counts consistent with the grid's cells (as the service computes them)."""
    counts: dict[str, int] = {}
    for row in e["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] != "-":
                counts[c["code"]] = counts.get(c["code"], 0) + 1
    counts["planned"] = sum(counts.values())
    counts["valid"] = counts.get("V", 0)
    e["data"]["counts"] = counts
    return e


def test_grid_watch_counts_the_rows_beyond_the_first_page(cli: Cli) -> None:
    """The grid is paginated (20 rows by default): cells still running on a later page must keep
    the watch going — the service's counts cover every run, the page only its rows."""
    page = _all_valid(env("runs_list.grid_done"))
    page["data"]["grid"]["total_rows"] = 12
    page["data"]["counts"] = {"V": 25, "F": 1, "R": 4, "Q": 2, "planned": 32, "valid": 25}
    final = _all_valid(env("runs_list.grid_done"))
    final["data"]["grid"]["total_rows"] = 12
    final["data"]["counts"] = {"V": 31, "F": 1, "planned": 32, "valid": 31}
    cli.client.responses["runs_list"] = [page, final]
    cli.client.responses["workflow_get"] = [env("workflow_get.running"), env("workflow_get.done")]
    r = cli.run("grid", WF, "--watch", "--interval", "3")
    assert r.exit_code == 0, r.text
    assert cli.client.methods().count("runs_list") == 2 and cli.sleeps == [3.0]
    assert "done: every cell is terminal" in r.stderr


def test_grid_watch_sees_a_running_run_inside_an_aggregated_cell(cli: Cli) -> None:
    """A cell that aggregates several runs shows its worst code (F): a run of it still running
    (counted per run by the service) keeps the watch going."""
    g = _all_valid(env("runs_list.grid_done"))
    failed = next(c for row in g["data"]["grid"]["rows"] for c in row["cells"] if c["code"] == "F")
    failed["runs"] = 2
    g["data"]["counts"] = {"V": 5, "F": 1, "R": 1, "planned": 7, "valid": 5}
    final = _all_valid(env("runs_list.grid_done"))
    final["data"]["counts"] = {"V": 6, "F": 1, "planned": 7, "valid": 6}
    cli.client.responses["runs_list"] = [g, final]
    cli.client.responses["workflow_get"] = [env("workflow_get.running"), env("workflow_get.done")]
    r = cli.run("grid", WF, "--watch", "--interval", "2")
    assert r.exit_code == 0, r.text
    assert cli.sleeps == [2.0] and "done: every cell is terminal" in r.stderr


def _main_cells_waiting_for_smoke(code: str, reason: str) -> dict[str, Any]:
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            c.update(code="P", reason="not dispatched")
    g["data"]["grid"]["notes"] = [
        f"smoke run smoke.baseline (run_01M3K9Q1CC6RNE08MM569NQ4SM): {code} {reason} - main cells of its "
        "workflow are dispatched only after a smoke run passes"
    ]
    return _recount(g)


def test_grid_watch_experiment_keeps_going_while_the_smoke_run_runs(cli: Cli) -> None:
    """The grid shows main runs only: a smoke run still queued/running (its note) is activity —
    the main cells start once it passes, so this is not "nothing is running"."""
    running = _main_cells_waiting_for_smoke("R", "RUNNING")
    done = _all_valid(env("runs_list.grid_done"))
    done["data"]["counts"] = {"V": 5, "F": 1, "planned": 6, "valid": 5}
    cli.client.responses["runs_list"] = [running] * 9 + [done]
    r = cli.run("grid", EXP, "--watch", "--interval", "10")
    assert r.exit_code == 0, r.text
    assert sum(cli.sleeps) == 90.0
    assert "nothing is running" not in r.stderr
    assert "done: every cell is terminal" in r.stderr


def test_grid_watch_smoke_outputs_never_validated_stops_after_the_grace(cli: Cli) -> None:
    """A gating workflow whose smoke run completed but whose outputs are still being checked (E):
    the main cells wait for it. If that does not change within ~60 s, stop and say where to look
    instead of watching forever."""
    g = _main_cells_waiting_for_smoke("E", "completed; outputs being checked (result_state=pending)")
    wf = env("workflow_get.running")
    wf["data"].update(state="gating")
    cli.client.responses["runs_list"] = g
    cli.client.responses["workflow_get"] = wf
    r = cli.run("grid", WF, "--watch", "--interval", "5", "--timeout", "600")
    assert r.exit_code == 0, r.text
    assert sum(cli.sleeps) == 60.0
    assert "timed out" not in r.stderr
    assert "smoke run" in r.stderr and "hpclab inspect run_01M3K9Q1CC6RNE08MM569NQ4SM" in r.stderr


def test_grid_watch_grace_restarts_while_evaluations_progress(cli: Cli) -> None:
    """Many E cells evaluated one after another: the 60 s grace counts from the last change, so a
    slow but progressing evaluation backlog is watched to the end."""

    def with_e(n: int) -> dict[str, Any]:
        e = env("runs_list.grid_done")
        cells = [c for row in e["data"]["grid"]["rows"] for c in row["cells"] if c["code"] == "V"]
        for c in cells[: n - 1]:
            c.update(code="E", reason="completed; evaluation pending", value=None)
        return _recount(e)

    snaps = (
        [with_e(4)] * 5
        + [with_e(3)] * 5
        + [with_e(2)] * 5
        + [_recount(_all_valid(env("runs_list.grid_done")))]
    )
    cli.client.responses["runs_list"] = snaps
    cli.client.responses["workflow_get"] = env("workflow_get.done")
    r = cli.run("grid", WF, "--watch", "--interval", "10")
    assert r.exit_code == 0, r.text
    assert sum(cli.sleeps) == 150.0
    assert "done: every cell is terminal" in r.stderr and "not evaluated" not in r.stderr


def test_grid_watch_timeout_and_ctrl_c(cli: Cli) -> None:
    cli.client.responses["runs_list"] = env("runs_list.grid_running")
    r = cli.run("grid", EXP, "--watch", "--interval", "5", "--timeout", "12")
    assert r.exit_code == 1
    assert "timed out after 12s" in r.stderr and "keep running" in r.stderr
    assert len(cli.sleeps) == 3

    calls = {"n": 0}

    def interrupt(p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return env("runs_list.grid_running")

    cli.client.responses["runs_list"] = interrupt
    r = cli.run("grid", EXP, "--watch")
    assert r.exit_code == 130
    assert "stopped watching" in r.stderr and "Traceback" not in r.text


def test_grid_watch_json_prints_only_the_final_envelope(cli: Cli) -> None:
    final = env("workflow_get.done")
    cli.client.responses["runs_list"] = [env("runs_list.grid_running"), env("runs_list.grid_done")]
    cli.client.responses["workflow_get"] = [env("workflow_get.running"), final]
    r = cli.run("--json", "grid", WF, "--watch", "--interval", "1", "--timeout", "30")
    # grid_done still has an E cell (evaluation pending): the 60 s grace outlasts the 30 s timeout
    assert r.json["data"]["view"] == "grid"
    assert r.stdout.count('"schema_version"') == 1


def test_grid_watch_stops_when_the_breaker_paused_dispatch(cli: Cli) -> None:
    g = env("runs_list.grid_running")
    for row in g["data"]["grid"]["rows"]:
        for c in row["cells"]:
            if c["code"] in ("Q", "R"):
                c.update(code="F", reason="import_error")
    _recount(g)
    wf = env("workflow_get.running")
    wf["data"].update(dispatch_state="paused", state="dispatching")
    cli.client.responses["runs_list"] = g
    cli.client.responses["workflow_get"] = wf
    r = cli.run("grid", WF, "--watch")
    assert r.exit_code == 0
    assert "dispatch paused (circuit breaker)" in r.stderr


# ---------------------------------------------------------------------------
# inspect / which
# ---------------------------------------------------------------------------


def test_inspect_failed_run_shows_axes_diagnosis_and_next_commands(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.failed")
    r = cli.run("inspect", FAILED_RUN)
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert FAILED_ATT in out and "NODE_FAIL" in out and "killed" in out and "incomplete" in out
    assert "diagnosis: Node failure" in out
    assert "│ scheduler: state=NODE_FAIL" in out  # evidence line
    assert "suggested: retry, resume, contact_support" in out
    assert f"hpclab logs {FAILED_ATT} --stderr --tail 200" in out
    assert f"hpclab recover {FAILED_ATT} --strategy retry" in out
    assert cli.client.params("run_inspect") == {"run_id": FAILED_RUN}


def test_inspect_pending_and_uncertain(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.pending")
    r = cli.run("inspect", PENDING_ATT, "--refresh", "--gpu")
    assert r.exit_code == 0
    assert "pending: Priority [cluster_busy]" in r.stdout
    assert "Nothing to fix" in r.stdout
    assert cli.client.params("run_inspect") == {"attempt_id": PENDING_ATT, "refresh": True, "probe": "gpu"}
    cli.client.responses["run_inspect"] = env("run_inspect.uncertain")
    r = cli.run("inspect", env("run_inspect.uncertain")["data"]["run"]["run_id"])
    assert "submission_uncertain" in r.stdout
    assert "will never resubmit on its own" in r.stdout


def test_inspect_succeeded_suggests_compare_and_progress_bar(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.succeeded")
    r = cli.run("inspect", GOOD_RUN)
    assert "100.0%" in r.stdout and "final record seen" in r.stdout
    assert f"hpclab compare {EXP}" in r.stdout
    assert cli.run("inspect", "bogus").exit_code == 2
    r = cli.run("--json", "inspect", GOOD_RUN)
    assert r.json == env("run_inspect.succeeded")


def test_which_uses_the_single_cluster_and_job_ref(cli: Cli) -> None:
    cli.client.responses["clusters_list"] = env("clusters_list")
    cli.client.responses["run_inspect"] = env("run_inspect.by_job")
    r = cli.run("which", "1002")
    assert r.exit_code == 0, r.text
    assert "sim:1002 →" in r.stdout and GOOD_RUN in r.stdout
    assert cli.client.params("run_inspect") == {"job_ref": {"cluster_id": "sim", "job_id": "1002"}}
    r = cli.run("which", "1002", "--cluster", "hai", "--submit-time", "2026-09-28T06:00:00Z")
    assert cli.client.params("run_inspect")["job_ref"] == {
        "cluster_id": "hai",
        "job_id": "1002",
        "submit_time": "2026-09-28T06:00:00Z",
    }


def test_which_needs_cluster_when_several_are_registered(cli: Cli) -> None:
    two = env("clusters_list")
    two["data"]["clusters"].append({**two["data"]["clusters"][0], "cluster_id": "hai"})
    cli.client.responses["clusters_list"] = two
    r = cli.run("which", "42")
    assert r.exit_code == 2 and "pass --cluster" in r.stderr
    cli.client.responses["run_inspect"] = err("not_found", "no hpclab attempt is bound to job 42 on sim")
    r = cli.run("which", "42", "--cluster", "sim")
    assert r.exit_code == 1 and "not_found" in r.stderr


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def test_logs_tail_writes_raw_text(cli: Cli) -> None:
    cli.client.responses["run_logs"] = env("run_logs")
    r = cli.run("logs", GOOD_ATT, "--tail", "4")
    assert r.exit_code == 0
    assert r.stdout == env("run_logs")["data"]["text"]
    assert "earlier output omitted" in r.stderr
    assert cli.client.params("run_logs") == {"attempt_id": GOOD_ATT, "stream": "stdout", "tail_lines": 4}


def test_logs_of_a_run_reads_its_latest_attempt_stderr(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.failed")
    cli.client.responses["run_logs"] = ok(
        {
            "attempt_id": FAILED_ATT,
            "stream": "stderr",
            "text": "Traceback: boom\n",
            "cursor": "c0",
            "next_cursor": None,
            "offset": 0,
            "next_offset": 16,
            "generation": 1,
            "size": 16,
            "truncated": False,
            "rotated": False,
            "eof": True,
            "note": None,
        }
    )
    r = cli.run("logs", FAILED_RUN, "--stderr")
    assert r.stdout == "Traceback: boom\n"
    assert cli.client.params("logs" and "run_logs")["attempt_id"] == FAILED_ATT
    assert cli.client.params("run_logs")["stream"] == "stderr"


def _chunk(
    text: str, nxt: str | None, *, eof: bool, note: str | None = None, rotated: bool = False
) -> dict[str, Any]:
    return ok(
        {
            "attempt_id": GOOD_ATT,
            "stream": "stdout",
            "text": text,
            "cursor": "c",
            "offset": 0,
            "next_offset": 0,
            "generation": 1,
            "size": 10,
            "truncated": False,
            "rotated": rotated,
            "eof": eof,
            "note": note,
        },
        next_cursor=nxt,
    )


def test_logs_follow_reads_until_the_attempt_ends(cli: Cli) -> None:
    cli.client.responses["run_logs"] = [
        _chunk("", "c0", eof=True, note="not started yet"),
        _chunk("", "c0", eof=True, note="not started yet"),
        _chunk("step 1\n", "c1", eof=False),
        _chunk("step 2\n", "c2", eof=True, rotated=True),
        _chunk("done\n", None, eof=True),
    ]
    r = cli.run("logs", GOOD_ATT, "--follow", "--interval", "2")
    assert r.exit_code == 0, r.text
    assert r.stdout == "step 1\nstep 2\ndone\n"
    assert r.stderr.count("not started yet") == 1  # a repeated note is printed once
    assert "rotated" in r.stderr and "has ended" in r.stderr
    cursors = [p.get("cursor") for m, p in cli.client.calls if m == "run_logs"]
    assert cursors == [None, "c0", "c0", "c1", "c2"]
    assert cli.sleeps == [2.0, 2.0, 2.0]  # sleeps only while at the end of the log


def test_logs_follow_json_is_one_envelope_per_line_and_ctrl_c(cli: Cli) -> None:
    import json

    cli.client.responses["run_logs"] = [_chunk("a\n", "c1", eof=False), _chunk("b\n", None, eof=True)]
    r = cli.run("--json", "logs", GOOD_ATT, "--follow")
    lines = [json.loads(line) for line in r.stdout.splitlines()]
    assert [x["data"]["text"] for x in lines] == ["a\n", "b\n"]

    def interrupt(p: dict[str, Any]) -> dict[str, Any]:
        if p.get("cursor"):
            raise KeyboardInterrupt
        return _chunk("x\n", "c1", eof=True)

    cli.client.responses["run_logs"] = interrupt
    r = cli.run("logs", GOOD_ATT, "-f")
    assert r.exit_code == 130 and "the job keeps running" in r.stderr


def test_logs_usage_errors(cli: Cli) -> None:
    assert cli.run("logs", GOOD_ATT, "--tail", "5", "--cursor", "c").exit_code == 2
    assert cli.run("logs", "exp_01M3K9Q1CC33H1R96NRPJEDAVW").exit_code == 2


# ---------------------------------------------------------------------------
# cancel / recover / resolve
# ---------------------------------------------------------------------------


def test_cancel_confirms_then_sends_version_and_mode(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.pending")
    cli.client.responses["run_cancel"] = env("run_cancel")
    r = cli.run("cancel", PENDING_RUN)
    assert r.exit_code == 2 and "run_cancel" not in cli.client.methods()
    assert "re-verified" in r.stderr
    r = cli.run("cancel", PENDING_RUN, "--yes", "--reason", "wrong lr")
    assert r.exit_code == 0, r.text
    p = cli.client.params("run_cancel")
    version = next(
        a for a in env("run_inspect.pending")["data"]["attempts"] if a["attempt_id"] == PENDING_ATT
    )["version"]
    assert p["attempt_id"] == PENDING_ATT and p["expected_version"] == version
    assert p["mode"] == "cancel" and p["reason"] == "wrong lr"
    assert "cancel: requested" in r.stdout and "hpclab ops op_" in r.stdout
    cli.run("cancel", PENDING_ATT, "--checkpoint", "--yes")
    assert cli.client.params("run_cancel")["mode"] == "request_checkpoint"


def test_cancel_interactive_and_refusals(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.pending")
    cli.client.responses["run_cancel"] = env("run_cancel")
    r = cli.run("cancel", PENDING_ATT, input="y\n", interactive=True)
    assert r.exit_code == 0 and "run_cancel" in cli.client.methods()
    cli.client.responses["run_inspect"] = env("run_inspect.succeeded")
    r = cli.run("cancel", GOOD_RUN, "--yes")
    assert r.exit_code == 1 and "no active attempt" in r.stderr
    cli.client.responses["run_inspect"] = env("run_inspect.pending")
    cli.client.responses["run_cancel"] = err(
        "precondition_failed", "the run does not declare checkpoint_and_exit"
    )
    assert cli.run("cancel", PENDING_ATT, "--checkpoint", "--yes").exit_code == 1


def test_recover_single_eligible_prints_the_exact_submit_command(cli: Cli) -> None:
    cli.client.responses["run_recover"] = env("run_recover.eligible")
    r = cli.run("recover", FAILED_ATT, "--strategy", "adjust_resources", "--time", "01:00:00", "--mem", "16G")
    assert r.exit_code == 0, r.text
    h = env("run_recover.eligible")["data"]["exec_plan_hash"]
    assert f"hpclab submit {FAILED_RUN} --exec-plan {h}" in r.stdout
    assert "eligible" in r.stdout and "node_fail" in r.stdout
    p = cli.client.params("run_recover")
    assert p["attempt_id"] == FAILED_ATT and p["strategy"] == "adjust_resources"
    assert p["resource_patch"] == {"time": "01:00:00", "mem": "16G"}
    assert p["idempotency_key"].startswith("recover-")


def test_recover_not_eligible_exits_1_and_run_ids_resolve(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.succeeded")
    cli.client.responses["run_recover"] = env("run_recover.not_eligible")
    r = cli.run("recover", GOOD_RUN)
    assert r.exit_code == 1
    assert "not eligible" in r.stdout and "nothing to recover" in r.stdout
    assert cli.client.params("run_recover")["attempt_id"] == GOOD_ATT


def test_recover_batch_selectors(cli: Cli) -> None:
    cli.client.responses["run_recover"] = env("run_recover.batch")
    r = cli.run("recover", "--workflow", WF, "--missing")
    assert r.exit_code == 0, r.text
    plan = env("run_recover.batch")["data"]["plan_id"]
    assert f"hpclab submit {plan}" in r.stdout and "1/1 cell(s) eligible" in r.stdout
    assert cli.client.params("run_recover")["selector"] == {"workflow_id": WF, "cells": "missing_valid"}
    cli.run("recover", "--workflow", WF, "--diagnosis", "node_fail", "--diagnosis", "timeout")
    assert cli.client.params("run_recover")["selector"]["cells"] == {"diagnosis_in": ["node_fail", "timeout"]}


@pytest.mark.parametrize(
    "args",
    [
        ("recover",),
        ("recover", FAILED_ATT, "--workflow", WF),
        ("recover", "--workflow", WF, "--missing", "--diagnosis", "x"),
        ("recover", "--workflow", WF, "--time", "01:00:00"),
        ("recover", FAILED_ATT, "--missing"),
        ("recover", FAILED_ATT, "--strategy", "pray"),
    ],
)
def test_recover_usage_errors(cli: Cli, args: tuple[str, ...]) -> None:
    assert cli.run(*args).exit_code == 2
    assert "run_recover" not in cli.client.methods()


def test_recover_a_workflow_id_points_at_the_workflow_option(cli: Cli) -> None:
    r = cli.run("recover", WF)
    assert r.exit_code == 2
    assert f"hpclab recover --workflow {WF}" in r.stderr
    assert cli.client.methods() == []


def test_resolve_is_a_loud_human_decision(cli: Cli) -> None:
    uncertain = env("run_inspect.uncertain")
    att = uncertain["data"]["focus_attempt_id"]
    cli.client.responses["run_inspect"] = uncertain
    cli.client.responses["attempt_resolve"] = env("attempt_resolve")
    r = cli.run("resolve", att, "--abandon", "--reason", "no job in sacct")
    assert r.exit_code == 2 and "attempt_resolve" not in cli.client.methods()
    assert "hpclab will NOT cancel it" in r.stderr and "unknown, never as zero" in r.stderr
    r = cli.run("resolve", att, "--abandon", "--reason", "no job in sacct", input="n\n", interactive=True)
    assert r.exit_code == 1 and "attempt_resolve" not in cli.client.methods()
    r = cli.run("attempt", "resolve", att, "--abandon", "--reason", "no job in sacct", "--yes")
    assert r.exit_code == 0, r.text
    p = cli.client.params("attempt_resolve")
    assert p["resolution"] == "abandon" and p["reason"] == "no job in sacct"
    assert p["expected_version"] == uncertain["data"]["attempts"][0]["version"]
    assert "phase abandoned" in r.stdout
    cli.run("resolve", att, "--bind-job", "1234", "--reason", "matches token", "--yes")
    p = cli.client.params("attempt_resolve")
    assert p["resolution"] == "bind_job" and p["job_id"] == "1234"


def test_resolve_header_says_uncertain_only_while_the_attempt_is_unresolved(cli: Cli) -> None:
    """The confirmation header must not call an attempt "uncertain" once it is not: abandoning an
    observing attempt in an observation gap (a human last resort) shows the gap, and a formerly
    uncertain attempt that was bound shows that as history (as `hpclab inspect` does)."""
    uncertain = env("run_inspect.uncertain")
    att = uncertain["data"]["focus_attempt_id"]
    a = uncertain["data"]["attempts"][0]
    since = a["uncertain_since"]
    cli.client.responses["run_inspect"] = uncertain
    r = cli.run("resolve", att, "--abandon", "--reason", "x")
    assert r.exit_code == 2
    assert f"uncertain since {since}" in r.stderr

    # never uncertain: an observing attempt whose job vanished from squeue and sacct
    gap = env("run_inspect.uncertain")
    ga = gap["data"]["attempts"][0]
    ga.update(phase="observing", uncertain_since=None, observation_gap=True, scheduler_state="RUNNING")
    ga["job"] = dict(ga["job"], job_id="4242")
    cli.client.responses["run_inspect"] = gap
    r = cli.run("resolve", att, "--abandon", "--reason", "x")
    assert r.exit_code == 2
    assert "uncertain" not in r.stderr, r.stderr
    assert "observation gap" in r.stderr

    # uncertain once, reconciled and bound since: history, not a live claim
    ga.update(uncertain_since=since, identified_at="2026-09-28T06:10:00Z")
    r = cli.run("resolve", att, "--abandon", "--reason", "x")
    assert r.exit_code == 2
    assert "uncertain since" not in r.stderr, r.stderr
    assert f"was uncertain from {since} until bound to job 4242" in " ".join(r.stderr.split())
    assert "attempt_resolve" not in cli.client.methods()


def test_resolve_usage_errors(cli: Cli) -> None:
    att = "att_01M3K9QT35YK3BRDF4M74CSDVT"
    assert cli.run("resolve", att, "--reason", "x").exit_code == 2
    assert cli.run("resolve", att, "--abandon", "--bind-job", "1", "--reason", "x").exit_code == 2
    assert cli.run("resolve", att, "--abandon", "--confirm-ended", "--reason", "x").exit_code == 2
    assert cli.run("resolve", att, "--abandon").exit_code == 2
    assert cli.client.calls == []


def test_resolve_confirm_ended_lifts_a_requeue_watch(cli: Cli) -> None:
    uncertain = env("run_inspect.uncertain")
    att = uncertain["data"]["focus_attempt_id"]
    cli.client.responses["run_inspect"] = uncertain
    cli.client.responses["attempt_resolve"] = env("attempt_resolve")
    r = cli.run("resolve", att, "--confirm-ended", "--reason", "sacct shows COMPLETED")
    assert r.exit_code == 2 and "re-checks squeue and sacct" in " ".join(r.stderr.split())
    r = cli.run("attempt", "resolve", att, "--confirm-ended", "--reason", "sacct shows COMPLETED", "--yes")
    assert r.exit_code == 0, r.text
    p = cli.client.params("attempt_resolve")
    assert (
        p["resolution"] == "confirm_ended" and p["reason"] == "sacct shows COMPLETED" and not p.get("job_id")
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


def _op(state: str, stage: str | None = None, **extra: Any) -> dict[str, Any]:
    e = env("operation_get.succeeded")
    e["data"].update(state=state, stage=stage, poll_after_ms=1500, **extra)
    return e


def test_ops_show_and_wait_until_final(cli: Cli) -> None:
    op = env("operation_get.succeeded")["data"]["operation_id"]
    cli.client.responses["operation_get"] = [
        _op("queued"),
        _op("running", "dispatching"),
        _op("succeeded", "completed"),
    ]
    r = cli.run("ops", op, "--wait")
    assert r.exit_code == 0, r.text
    assert cli.sleeps == [1.5, 1.5]
    assert "running at dispatching" in r.stderr
    assert f"operation {op}" in r.stdout and "succeeded" in r.stdout
    cli.client.responses["operation_get"] = env("operation_get.succeeded")
    r = cli.run("--json", "ops", op)
    assert r.json == env("operation_get.succeeded")


def test_ops_wait_stops_on_a_decision_and_on_failure(cli: Cli) -> None:
    blocked = env("operation_get.blocked")
    op = blocked["data"]["operation_id"]
    cli.client.responses["operation_get"] = [_op("running", "reconciling"), blocked]
    r = cli.run("ops", op, "--wait")
    assert r.exit_code == 1
    assert "waits for a decision" in r.stderr
    assert "hpclab attempt resolve" in r.stdout
    auth = _op(
        "blocked",
        "dispatching",
        error={
            "code": "auth_required",
            "message": "ssh master gone",
            "details": {"remediation": "hpclab auth hai"},
        },
    )
    cli.client.responses["operation_get"] = auth
    r = cli.run("ops", op, "--wait")
    assert r.exit_code == 4 and "hpclab auth hai" in r.stdout
    cli.client.responses["operation_get"] = _op(
        "failed", "rejected", error={"code": "submission_rejected", "message": "bad partition", "details": {}}
    )
    assert cli.run("ops", op, "--wait").exit_code == 1


def test_ops_wait_keeps_waiting_through_self_clearing_blocks_until_timeout(cli: Cli) -> None:
    waiting = _op(
        "blocked",
        "waiting_for_reconcile",
        error={"code": "unavailable", "message": "sacct lag", "details": {}},
    )
    cli.client.responses["operation_get"] = waiting
    r = cli.run("ops", "op_01M3K9Q1MHKZ8HWQR03AZ2GP5A", "--wait", "--timeout", "4")
    assert r.exit_code == 1 and "still blocked after 4s" in r.stderr
    assert len(cli.sleeps) == 3
