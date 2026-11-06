"""Regression tests for agent-facing friction found by driving the MCP server against the
simulator: each case is a result an agent actually received and would have acted on wrongly
(wait forever, report "OK" for an unchecked manuscript, export an empty table, ...)."""

from __future__ import annotations

from typing import Any

from hpclab.api.envelope import Coverage
from hpclab.mcp.server import render_text
from hpclab.services import app


def _ok(data: dict[str, Any], summary: str = "summary line", **env: Any) -> dict[str, Any]:
    return {"ok": True, "request_id": "r", "data": data, "summary": summary, "warnings": [], **env}


def _err(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "request_id": "r",
        "data": None,
        "summary": f"{code}: {message}",
        "error": {"code": code, "message": message, "retryable": False, "details": details},
    }


def _grid(cells: list[dict[str, Any]], notes: list[str] | None = None, **counts: int) -> dict[str, Any]:
    return {
        "view": "grid",
        "runs": [],
        "grid": {
            "row_axes": ["method"],
            "col_axes": ["seed"],
            "cols": [f"seed={i + 1}" for i in range(len(cells))],
            "rows": [{"key": "method=baseline", "values": {"method": "baseline"}, "cells": cells}],
            "total_rows": 1,
            "offset": 0,
            "metric": None,
            "legend": {},
            "notes": notes or [],
        },
        "total": len(cells),
        "counts": {**counts, "planned": len(cells)},
    }


def _next(text: str) -> str:
    return text.partition("\nNext:\n")[2]


# ---------------------------------------------------------------------------
# runs_list grid
# ---------------------------------------------------------------------------


def test_grid_of_a_never_dispatched_experiment_does_not_say_check_again_later() -> None:
    """Prepared, never submitted: the smoke run is P and no workflow exists. Waiting never helps."""
    notes = [
        "smoke run smoke.method-baseline.seed-1 (run_s): P not dispatched - main cells of its workflow are "
        "dispatched only after a smoke run passes"
    ]
    data = _grid([{"code": "P", "run_id": f"r{i}", "runs": 1} for i in range(3)], notes, P=3)
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    nxt = _next(text)
    assert "check again later" not in nxt, text
    assert 'workflow_submit(prepared_experiment_id="exp_1"' in nxt, text


def test_grid_while_the_smoke_run_is_active_says_check_again_later() -> None:
    notes = [
        "smoke run smoke.method-baseline.seed-1 (run_s): R RUNNING - main cells of its workflow are "
        "dispatched only after a smoke run passes"
    ]
    data = _grid([{"code": "P", "run_id": f"r{i}", "runs": 1} for i in range(3)], notes, P=3)
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "check again later" in _next(text) and "workflow_submit(" not in _next(text)


def test_completed_with_unchecked_outputs_is_not_called_a_failure() -> None:
    """``E`` (outputs being checked, then auto-evaluation) is the first state after every
    successful job; telling the agent 'why it failed' sends it hunting for a failure that is not there."""
    cells = [
        {"code": "V", "run_id": "a1", "runs": 1},
        {
            "code": "E",
            "run_id": "a2",
            "runs": 1,
            "reason": "completed; outputs being checked (result_state=unchecked)",
        },
        {
            "code": "E",
            "run_id": "a3",
            "runs": 1,
            "reason": "completed; outputs being checked (result_state=pending)",
        },
    ]
    data = _grid(cells, V=1)
    data["counts"]["E"] = 2
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "why it failed" not in text, text
    assert "completed, evaluation pending (not a failure)" in text
    assert "2 completed run(s) are not evaluated yet" in _next(text)
    assert "check again later" not in _next(text)  # nothing is queued or running


def test_smoke_run_completed_but_not_evaluated_is_not_a_failed_gate() -> None:
    notes = [
        "smoke run smoke.method-baseline.seed-1 (run_s): E completed; outputs being checked "
        "(result_state=unchecked) - main cells of its workflow are dispatched only after a smoke run passes"
    ]
    data = _grid([{"code": "P", "run_id": f"r{i}", "runs": 1} for i in range(2)], notes, P=2)
    nxt = _next(render_text("runs_list", _ok(data), {"experiment_id": "exp_1"}))
    assert "smoke run completed" in nxt and "check again shortly" in nxt, nxt
    assert "smoke gate failed" not in nxt
    # outputs that stay unchecked hold the whole matrix back: give the agent a way to look
    assert 'run_inspect(run_id="run_s")' in nxt, nxt


def test_smoke_run_with_invalid_outputs_failed_the_gate() -> None:
    notes = [
        "smoke run smoke.method-baseline.seed-1 (run_s): C! outputs invalid - main cells of its workflow "
        "are dispatched only after a smoke run passes"
    ]
    data = _grid([{"code": "P", "run_id": f"r{i}", "runs": 1} for i in range(2)], notes, P=2)
    nxt = _next(render_text("runs_list", _ok(data), {"experiment_id": "exp_1"}))
    assert "smoke gate failed" in nxt and 'run_inspect(run_id="run_s")' in nxt, nxt


def test_inspect_of_a_completed_unevaluated_run_does_not_suggest_recovery() -> None:
    data = _inspect([])
    data["run"].update(
        code="E", status="succeeded", code_reason="completed; evaluation pending (or run evaluation_collect)"
    )
    data["attempts"][0].update(phase="terminal", scheduler_state="COMPLETED", result_state="valid")
    text = render_text("run_inspect", _ok(data))
    nxt = _next(text)
    assert "run_recover(" not in nxt, text
    assert "evaluation pending (not a failure)" in text
    assert 'evaluation_collect(attempt_id="att_1"' in nxt, nxt


def test_compare_points_unevaluated_runs_at_collect_not_recovery() -> None:
    d = _compare("acc", "contract", 2)
    d["counts"].update(included=2, missing=1)
    d["runs"] = [
        {
            "run_id": "run_e",
            "label": "baseline.seed-3",
            "status": "missing",
            "code": "E",
            "latest_attempt_id": "att_e",
        },
        {
            "run_id": "run_r",
            "label": "mixup.seed-3",
            "status": "missing",
            "code": "R",
            "latest_attempt_id": "att_r",
        },
    ]
    nxt = _next(render_text("results_compare", _ok(d)))
    assert "run_recover(" not in nxt, nxt
    assert 'evaluation_collect(run_set_ref="exp_1"' in nxt and "compare again" in nxt, nxt
    d["runs"].append(
        {
            "run_id": "run_f",
            "label": "mixup.seed-2",
            "status": "failed",
            "code": "F",
            "latest_attempt_id": "att_f",
        }
    )
    nxt = _next(render_text("results_compare", _ok(d)))
    assert 'run_recover(attempt_id="att_f"' in nxt, nxt


def test_submitting_cells_point_at_what_blocks_them() -> None:
    """A submission blocked on SSH auth shows only as S in the grid; the hint must say where to look."""
    notes = [
        "smoke run smoke.method-baseline.seed-1 (run_s): S staging - main cells of its workflow are "
        "dispatched only after a smoke run passes"
    ]
    data = _grid([{"code": "P", "run_id": f"r{i}", "runs": 1} for i in range(2)], notes, P=2)
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert 'run_inspect(run_id="run_s")' in _next(text), text


def test_changes_listing_summary_does_not_contradict_itself() -> None:
    d = {"view": "list", "total": 0, "runs": [], "counts": {"V": 7}, "changes_cursor": "c"}
    s = app._s_runs_list(d)
    assert not s.startswith("0 run(s): V=7"), s
    assert s == "no run changed since the cursor (all 7 run(s): V=7)"
    d = {"view": "list", "total": 2, "runs": [{}, {}], "counts": {"V": 6, "R": 1}, "changes_cursor": "c"}
    assert app._s_runs_list(d) == "2 run(s) changed since the cursor (all 7 run(s): V=6, R=1)"


# ---------------------------------------------------------------------------
# run_inspect
# ---------------------------------------------------------------------------


def _inspect(ops: list[dict[str, Any]], rec: list[str] | None = None) -> dict[str, Any]:
    return {
        "run": {"run_id": "run_1", "label": "m.seed-1", "kind": "smoke", "status": "submitting", "code": "S"},
        "focus_attempt_id": "att_1",
        "attempts": [
            {
                "attempt_id": "att_1",
                "seq": 1,
                "version": 1,
                "phase": "staging",
                "scheduler_state": "UNKNOWN",
                "app_state": "unknown",
                "result_state": "unchecked",
                "job": {"cluster_id": "sim"},
            }
        ],
        "operations": ops,
        "recommended_next": rec or [],
    }


_AUTH_OP = {
    "operation_id": "op_1",
    "ref": "hpclab://operations/op_1",
    "kind": "submit",
    "state": "blocked",
    "stage": "auth_required",
    "target_type": "attempt",
    "target_id": "att_1",
    "progress": {},
    "error": {
        "code": "auth_required",
        "message": "SSH needs interactive authentication (ControlMaster expired)",
        "retryable": True,
        "details": {"cluster_id": "sim", "remediation": "hpclab auth sim", "blocked_step": "staging"},
    },
    "poll_after_ms": 2000,
    "updated_at": "t",
}


def test_inspect_shows_a_submission_blocked_on_authentication() -> None:
    data = _inspect([_AUTH_OP])
    summary = app._s_inspect(data)
    assert "blocked" in summary and "hpclab auth sim" in summary, summary
    text = render_text("run_inspect", _ok(data, summary))
    assert "operation op_1 (submit) blocked at auth_required" in text, text
    nxt = _next(text)
    assert "hpclab auth sim" in nxt and "their own terminal" in nxt, text
    assert "check again later" not in nxt


def test_inspect_ignores_finished_operations() -> None:
    done = {**_AUTH_OP, "state": "succeeded", "stage": "done", "error": None}
    text = render_text("run_inspect", _ok(_inspect([done])))
    assert "operation op_1" not in text


def test_recommended_calls_carry_the_required_idempotency_key() -> None:
    """The service's hints are shared with the CLI and omit idempotency_key; following one verbatim
    through MCP failed with 'idempotency_key: Field required'."""
    rec = [
        "read the error: run_logs(attempt_id='att_1', stream='stderr', tail_lines=200)",
        "recover: run_recover(attempt_id='att_1', strategy='adjust_resources')",
        "collect the metrics: evaluation_collect(attempt_id='att_1')",
        "the ETA exceeds the remaining walltime: run_cancel(attempt_id='att_1', mode='request_checkpoint', ...) stops it",
    ]
    text = render_text("run_inspect", _ok(_inspect([], rec)))
    nxt = _next(text)
    assert "run_logs(attempt_id='att_1', stream='stderr', tail_lines=200)" in nxt  # read-only: unchanged
    assert (
        "run_recover(attempt_id='att_1', strategy='adjust_resources', idempotency_key=\"<new key>\")" in nxt
    ), nxt
    assert "evaluation_collect(attempt_id='att_1', idempotency_key=\"<new key>\")" in nxt, nxt
    assert "idempotency_key" in next(line for line in nxt.splitlines() if "run_cancel(" in line), nxt


# ---------------------------------------------------------------------------
# evidence_verify: never "OK" for a manuscript that was not checked
# ---------------------------------------------------------------------------


def _verify(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "bundle_id": "evb_1",
        "ok": True,
        "findings": [],
        "counts_by_code": {},
        "counts_by_severity": {},
        "tex": [],
        "values_checked": 12,
        "tags_checked": 0,
    }
    d.update(over)
    return d


def test_verify_summary_is_not_ok_when_the_manuscript_could_not_be_read() -> None:
    cov = Coverage()
    cov.add("ledger", "ok")
    cov.add("tex:/p/missing.tex", "unavailable", detail="[Errno 2] No such file or directory")
    s = app._s_verify(_verify(coverage=cov))
    assert " OK " not in s and "INCOMPLETE" in s and "/p/missing.tex" in s, s


def test_verify_summary_is_not_ok_when_an_oversized_manuscript_was_skipped() -> None:
    cov = Coverage()
    cov.add("tex:/p/huge.tex", "skipped", detail="3000000 bytes exceeds 2000000")
    s = app._s_verify(_verify(coverage=cov))
    assert " OK " not in s and "/p/huge.tex" in s, s


def test_verify_summary_says_when_the_manuscript_has_no_tagged_numbers() -> None:
    """main.tex pulls the table in with \\input{table}: nothing in it was checked."""
    rep = {"source": "/p/main.tex", "tags": 0, "untagged_numbers": 0, "bundles": []}
    d = _verify(tex=[rep], coverage=Coverage())
    s = app._s_verify(d)
    assert " OK " not in s and "no \\hpcval" in s, s
    text = render_text("evidence_verify", _ok(d, s))
    assert "\\input" in _next(text), text


def test_verify_text_lists_the_files_scanned_through_input() -> None:
    reps = [
        {"source": "/p/main.tex", "tags": 0, "untagged_numbers": 0, "bundles": []},
        {"source": "/p/tables/t1.tex", "tags": 8, "untagged_numbers": 0, "bundles": ["evb_1"]},
    ]
    d = _verify(tex=reps, tags_checked=8, checked=True, files_scanned=["/p/main.tex", "/p/tables/t1.tex"])
    text = render_text("evidence_verify", _ok(d, "s"))
    assert "files scanned (2): /p/main.tex, /p/tables/t1.tex" in text, text
    assert "NOT checked" not in text and "not followed" not in text


def test_verify_without_tags_says_not_checked_and_that_includes_are_followed() -> None:
    no_tags = {
        "code": "NO_TAGS_FOUND",
        "severity": "error",
        "message": "no \\hpcval tag in the scanned manuscript files: it was NOT checked",
        "refs": ["/p/main.tex"],
    }
    d = _verify(
        ok=False,
        findings=[no_tags],
        counts_by_code={"NO_TAGS_FOUND": 1},
        checked=False,
        files_scanned=["/p/main.tex"],
        tex=[{"source": "/p/main.tex", "tags": 0, "untagged_numbers": 0}],
    )
    text = render_text("evidence_verify", _ok(d, "s"))
    assert "manuscript NOT checked" in text, text
    nxt = _next(text)
    assert "\\input" in nxt and "followed" in nxt and "not followed" not in nxt, nxt


def test_verify_unresolved_input_points_at_the_path() -> None:
    warn = {
        "code": "UNRESOLVED_INPUT",
        "severity": "warn",
        "message": "\\input{tables/t2} at /p/main.tex:12: /p/tables/t2.tex not found; the numbers in it were NOT checked",
        "refs": ["/p/main.tex:12", "/p/tables/t2.tex"],
    }
    d = _verify(
        findings=[warn],
        counts_by_code={"UNRESOLVED_INPUT": 1},
        checked=True,
        tags_checked=4,
        files_scanned=["/p/main.tex"],
    )
    cov = {
        "complete": False,
        "sources": [
            {"name": "tex:/p/main.tex", "status": "ok"},
            {
                "name": "tex:/p/tables/t2.tex",
                "status": "unavailable",
                "detail": "\\input{tables/t2} at /p/main.tex:12: not found",
            },
        ],
        "missing": [],
        "notes": [],
    }
    nxt = _next(render_text("evidence_verify", _ok(d, "s", coverage=cov)))
    assert "UNRESOLVED_INPUT" in nxt and "tex_paths" in nxt, nxt
    assert "not checked: /p/tables/t2.tex" not in nxt  # one hint for it, not two


def test_verify_summary_ok_when_tags_were_checked() -> None:
    rep = {"source": "/p/main.tex", "tags": 8, "untagged_numbers": 0, "bundles": ["evb_1"]}
    s = app._s_verify(_verify(tex=[rep], tags_checked=8, coverage=Coverage()))
    assert s.startswith("bundle evb_1: OK — 0 error(s), 0 warning(s)"), s


def test_verify_next_step_for_an_unreadable_manuscript() -> None:
    cov = {
        "complete": False,
        "sources": [{"name": "tex:/p/missing.tex", "status": "unavailable", "detail": "No such file"}],
        "missing": [],
        "notes": [],
    }
    text = render_text("evidence_verify", _ok(_verify(), "s", coverage=cov))
    assert "tex_paths" in _next(text) and "/p/missing.tex" in _next(text), text
    # the service also reports NO_TAGS_FOUND (nothing was read): one hint, about the path
    d = _verify(ok=False, checked=False, files_scanned=[], counts_by_code={"NO_TAGS_FOUND": 1})
    nxt = _next(render_text("evidence_verify", _ok(d, "s", coverage=cov)))
    assert "/p/missing.tex" in nxt and "no \\hpcval-tagged number was found" not in nxt, nxt


# ---------------------------------------------------------------------------
# results_compare with a metric nobody reports
# ---------------------------------------------------------------------------


def _compare(metric: str, source: str, n: int) -> dict[str, Any]:
    m = {"n": n, "expected_n": 3, "mean": 0.8 if n else None, "missing_seeds": [] if n else [1, 2, 3]}
    return {
        "comparison_id": "cmp_1",
        "experiment": {"name": "e"},
        "run_set": {"kind": "experiment", "id": "exp_1"},
        "counts": {"planned_runs": 3, "included": 3, "missing": 0, "failed": 0, "excluded": 0},
        "metric_names": [metric],
        "metric_info": [{"name": metric, "source": source}],
        "contract": {"metrics": [{"name": "acc"}, {"name": "loss"}]},
        "expected_seeds": [1, 2, 3],
        "groups": [{"label": "baseline", "is_baseline": True, "metrics": {metric: m}}],
        "runs": [],
    }


def test_compare_with_an_unknown_metric_says_so_and_does_not_suggest_exporting() -> None:
    text = render_text("results_compare", _ok(_compare("accuracy", "unknown", 0)))
    assert "UNKNOWN METRIC accuracy" in text and "acc, loss" in text, text
    nxt = _next(text)
    assert "evidence_export(" not in nxt, text
    assert 'results_compare(planned_run_set_ref="exp_1", metric_names=["acc", "loss"])' in nxt, text


def test_compare_refused_for_unknown_metrics_says_which_call_works() -> None:
    env = _err(
        "invalid_argument",
        "metric(s) ['accuracy'] are reported by none of the 6 evaluation(s) of hpclab://experiments/exp_1; "
        "known metrics: acc, loss",
        unknown_metrics=["accuracy"],
        known_metrics=["acc", "loss"],
        run_set="hpclab://experiments/exp_1",
        evaluations=6,
    )
    text = render_text("results_compare", env)
    assert 'what to do: results_compare(planned_run_set_ref="exp_1", metric_names=["acc", "loss"])' in text, (
        text
    )
    assert "evidence_export(" not in text


def test_compare_with_a_known_metric_suggests_exporting() -> None:
    text = render_text("results_compare", _ok(_compare("acc", "contract", 3)))
    assert "UNKNOWN METRIC" not in text and 'evidence_export(comparison_ref="cmp_1"' in _next(text)


# ---------------------------------------------------------------------------
# errors, recovery
# ---------------------------------------------------------------------------


def test_not_found_with_an_id_of_another_kind_names_the_right_parameter() -> None:
    env = _err("not_found", "run att_1X not found", run_id="att_1X")
    text = render_text("run_inspect", env)
    assert "'att_1X' is an attempt id (run_id takes run_… ids): pass it as attempt_id" in text, text
    env = _err("not_found", "attempt run_1X not found", attempt_id="run_1X")
    text = render_text("run_logs", env)
    assert "'run_1X' is a run id" in text and "run_inspect(run_id=" in text, text


def test_not_found_with_a_list_of_valid_values_says_to_pick_one() -> None:
    env = _err("not_found", "bundle evb_1 has no cell 'x.acc'", known_cells=["baseline.acc"])
    assert "use one of known_cells" in render_text(None, env)


def test_batch_recovery_of_environment_failures_says_fix_first() -> None:
    d = {
        "mode": "batch",
        "plan_id": "pln_r",
        "eligible_count": 2,
        "total": 6,
        "requested_gpu_hours": 0.33,
        "cells": [],
        "hints": [{"strategy": "fix_environment", "message": "fix it first, then retry"}],
    }
    nxt = _next(render_text("run_recover", _ok(d)))
    line = next(x for x in nxt.splitlines() if "workflow_submit(" in x)
    assert "only after" in line and "environment" in line, nxt


def test_cluster_inspect_prints_a_repeated_fix_once() -> None:
    fix = "run `hpclab auth sim` in a terminal"
    checks = [
        {
            "name": "auth",
            "status": "fail",
            "summary": "auth_required",
            "findings": [{"severity": "block", "code": "auth_required", "message": "m", "remediation": fix}],
        }
    ] + [
        {
            "name": n,
            "status": "skipped",
            "summary": "not run",
            "findings": [{"severity": "warn", "code": "blocked_by_auth", "message": "m", "remediation": fix}],
        }
        for n in ("identity", "scheduler", "storage", "runtime", "queue")
    ]
    text = render_text("cluster_inspect", _ok({"cluster_id": "sim", "status": "fail", "checks": checks}))
    assert text.count(fix) <= 2, text


def test_prepare_under_expired_auth_says_to_authenticate_before_dispatching() -> None:
    d = {
        "experiment_id": "exp_1",
        "plan_id": "pln_1",
        "plan_hash": "sha256:" + "a" * 64,
        "status": "ready",
        "smoke_run_id": None,
        "runs": [{"kind": "main", "label": "m", "run_id": "run_1", "exec_plan_hash": "sha256:b"}],
        "validation": {
            "status": "warn",
            "findings": [
                {
                    "check": "scheduler_test",
                    "status": "unchecked",
                    "severity": "info",
                    "message": "uploading the test script: SSH authentication required",
                    "remediation": "hpclab auth sim",
                }
            ],
        },
    }
    nxt = _next(render_text("experiment_prepare", _ok(d)))
    first = nxt.splitlines()[0]
    assert "`hpclab auth sim`" in first and "their own terminal" in first, nxt
    assert 'workflow_submit(plan_id="pln_1"' in nxt


def test_keyed_tools_match_the_schemas() -> None:
    """Hints get an idempotency_key for exactly the tools whose schema requires one."""
    from hpclab.mcp.catalog import TOOLS
    from hpclab.mcp.server import _KEYED_TOOLS
    from hpclab.services.app import build_registry

    reg = build_registry(strict=True)
    required = {
        t.name
        for t in TOOLS
        if (f := reg.get(t.method).params_model.model_fields.get("idempotency_key")) is not None
        and f.is_required()
    }
    assert set(_KEYED_TOOLS) == required


def test_recover_of_a_valid_attempt_has_no_none_diagnosis_and_keyed_hints() -> None:
    d = {
        "eligible": False,
        "strategy": None,
        "run_id": "run_1",
        "attempt_id": "att_1",
        "diagnosis": {
            "primary_code": None,
            "category": None,
            "confidence": None,
            "source": "computed",
            "summary": "Completed; no problems found in the scheduler state or log tails.",
        },
        "blocking": ["the attempt completed with valid results: nothing to recover"],
        "hints": [
            {
                "strategy": "derive_run",
                "message": "A new experiment from this Run is a derived Run.",
                "tool": "experiment_prepare",
                "arguments": {"derive_from": {"run_id": "run_1", "overrides": {}}},
            }
        ],
    }
    text = render_text("run_recover", _ok(d))
    assert "None" not in text, text
    assert "Completed; no problems found" in text
    assert (
        'experiment_prepare(derive_from={"run_id": "run_1", "overrides": {}}, idempotency_key="<new key>")'
        in (_next(text))
    ), text


def test_an_oversized_result_keeps_the_cursor_and_next_steps() -> None:
    """A wide grid with many failed cells exceeded the hard limit and the tail was cut: the page
    cursor and the Next: section disappeared from the text."""
    from hpclab.mcp.server import TEXT_HARD_LIMIT

    rows = []
    for r in range(20):
        cells = [
            {
                "code": "F" if (i + r) % 7 == 0 else "V",
                "run_id": f"run_{r:02d}{i:02d}" + "X" * 20,
                "runs": 1,
                "value": None if (i + r) % 7 == 0 else 0.8123456,
                "reason": "import_error" if (i + r) % 7 == 0 else None,
            }
            for i in range(64)
        ]
        rows.append(
            {"key": f"method=m{r}" + "y" * 30, "values": {"method": f"m{r}" + "y" * 30}, "cells": cells}
        )
    d = _grid([], P=0)
    d["grid"].update(cols=[f"seed={i}" for i in range(64)], rows=rows, total_rows=40, metric="acc")
    d["counts"] = {"V": 1100, "F": 183, "planned": 1283}
    text = render_text("runs_list", _ok(d, next_cursor="CUR"), {"experiment_id": "exp_1"})
    assert len(text.encode()) <= TEXT_HARD_LIMIT
    assert 'cursor="CUR"' in text and "\nNext:\n" in text and "why it failed" in _next(text), text[-1500:]
    assert text.splitlines()[0] == "summary line"


def test_compare_with_nothing_included_does_not_suggest_an_empty_table_and_hides_none() -> None:
    d = _compare("acc", "contract", 0)
    d["counts"].update(included=0, missing=2, failed=1)
    d["runs"] = [
        {
            "status": "failed",
            "label": "m.seed-1",
            "run_id": "run_1",
            "code": "F",
            "latest_attempt_id": "att_1",
        },
        {"status": "missing", "label": "m.seed-2", "run_id": "run_2", "code": "P", "latest_attempt_id": None},
    ]
    cov = {
        "complete": False,
        "sources": [],
        "missing": ["m.seed-1 [failed]", "m.seed-2 [missing]"],
        "notes": [],
    }
    text = render_text("results_compare", _ok(d, coverage=cov))
    assert "None" not in text, text
    assert "evidence_export(" not in _next(text) and "run_recover(" in _next(text), text
    assert text.count("m.seed-2") == 1, text  # listed once, not again under INCOMPLETE


def test_collect_skips_without_an_attempt_do_not_print_none() -> None:
    d = {
        "mode": "batch",
        "run_set": {"kind": "experiment", "id": "exp_1", "runs": 2},
        "counts": {"collected": 0, "skipped": 2},
        "collected": [],
        "skipped": [
            {"run_id": "run_1", "attempt_id": None, "reason": "not_dispatched"},
            {
                "run_id": "run_2",
                "attempt_id": "att_2",
                "reason": "scheduler_state_not_completed",
                "detail": "FAILED",
            },
        ],
    }
    text = render_text("evaluation_collect", _ok(d))
    assert "None" not in text and "att_2: FAILED" in text, text


def test_export_with_cells_lacking_values_warns_before_pasting() -> None:
    d = {"bundle_id": "evb_1", "comparison_id": "cmp_1", "counts": {}, "files": []}
    env = _ok(d, warnings=["all runs included but 2 metric cell(s) lack values: ['baseline/accuracy']"])
    nxt = _next(render_text("evidence_export", env))
    assert nxt.splitlines()[0].startswith("- do not paste yet"), nxt
