"""Text blocks of tool results (what an agent reads first) and the Envelope summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hpclab.mcp import server as srv
from hpclab.mcp.server import (
    TEXT_HARD_LIMIT,
    call_hint,
    compact_json,
    normalize_arguments,
    render_text,
)
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


def _grid(cells: list[list[dict[str, Any]]], notes: list[str] | None = None, **counts: int) -> dict[str, Any]:
    rows = [
        {"key": f"method={m}", "values": {"method": m}, "cells": row}
        for m, row in zip(("baseline", "mixup"), cells, strict=True)
    ]
    return {
        "view": "grid",
        "runs": [],
        "grid": {
            "row_axes": ["method"],
            "col_axes": ["seed"],
            "cols": ["seed=1", "seed=2"],
            "rows": rows,
            "total_rows": 2,
            "offset": 0,
            "metric": None,
            "legend": {},
            "notes": notes or [],
        },
        "total": 4,
        "counts": {**counts, "planned": 4},
    }


def test_invalid_arguments_show_each_validation_error() -> None:
    env = _err(
        "invalid_argument",
        "invalid parameters for runs_list: 2 error(s)",
        errors=[
            {"loc": "view", "msg": "Input should be 'list' or 'grid'", "type": "literal_error"},
            {"loc": "limit", "msg": "Input should be a valid integer", "type": "int_parsing"},
        ],
    )
    text = render_text("runs_list", env)
    lines = text.splitlines()
    assert lines[0] == "ERROR invalid_argument: invalid parameters for runs_list: 2 error(s)"
    assert (
        "- view: Input should be 'list' or 'grid'" in lines
        and "- limit: Input should be a valid integer" in lines
    )
    assert "retryable: no" in lines and any(x.startswith("what to do:") for x in lines)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            _err("version_conflict", "at version 6, expected 5", current_version=6),
            "retry with expected_version=6",
        ),
        (_err("not_found", "spec file /x.yaml not found", spec_path="/x.yaml"), "check the path"),
        (
            _err("not_found", "cluster 'hai' is not registered", known_clusters=["sim"]),
            "registered clusters: sim",
        ),
        (_err("not_found", "run run_x not found", run_id="run_x"), "ids are prefixed"),
        (
            _err("admission_denied", "plan already has its execution group", workflow_id="wf_1"),
            'runs_list(workflow_id="wf_1", view="grid")',
        ),
        (
            _err("auth_required", "run hpclab auth hai", remediation="hpclab auth hai"),
            "remediation: hpclab auth hai",
        ),
        (_err("idempotency_conflict", "key reused"), "use a new key"),
    ],
)
def test_errors_say_what_to_do(env: dict[str, Any], expected: str) -> None:
    assert expected in render_text("run_cancel", env)


def test_grid_is_a_matrix_with_attention_cells_and_hints() -> None:
    data = _grid(
        [
            [
                {"code": "F", "run_id": "run_a1", "runs": 1, "reason": "host_oom"},
                {"code": "V", "run_id": "run_a2", "runs": 1},
            ],
            [{"code": "V", "run_id": "run_b1", "runs": 1}, {"code": "R", "run_id": "run_b2", "runs": 1}],
        ],
        V=2,
        F=1,
        R=1,
    )
    text = render_text(
        "runs_list",
        _ok(data, "grid: 2/4 planned main run(s) valid (V=2, F=1, R=1)"),
        {"experiment_id": "exp_1"},
    )
    lines = text.splitlines()
    assert "method \\ seed  1  2" in lines
    assert "baseline       F  V" in lines and "mixup          V  R" in lines
    assert "- F method=baseline seed=1: host_oom (run_id run_a1)" in lines
    assert 'why it failed: run_inspect(run_id="run_a1")' in text
    assert "queued/running: check again later" in text
    assert len(text.encode()) < 1500


def test_completed_but_not_yet_evaluated_is_not_called_a_failure() -> None:
    """E = COMPLETED, evaluation pending; C! = COMPLETED with invalid outputs/evaluation."""
    pending = "completed; evaluation pending (or run evaluation_collect)"
    data = _grid(
        [
            [{"code": "V", "run_id": "a1", "runs": 1}, {"code": "V", "run_id": "a2", "runs": 1}],
            [
                {"code": "E", "run_id": "b1", "runs": 1, "reason": pending},
                {"code": "C!", "run_id": "b2", "runs": 1, "reason": "outputs invalid"},
            ],
        ],
        V=2,
    )
    data["counts"].update({"E": 1, "C!": 1})
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "completed, evaluation pending (not a failure)" in text
    assert f"- E method=mixup seed=1: {pending} (run_id b1)" in text
    assert "- C! method=mixup seed=2: outputs invalid (run_id b2)" in text
    assert 'why it failed: run_inspect(run_id="b2")' in text  # the genuinely invalid one
    assert "b1" not in next(ln for ln in text.splitlines() if "why it failed" in ln)
    assert "1 completed run(s) are not evaluated yet" in text
    assert 'evaluation_collect(run_set_ref="exp_1"' in text
    assert "results_compare(" not in text
    codes = next(ln for ln in text.splitlines() if ln.startswith("codes:"))
    assert "E completed, evaluation pending (not a failure)" in codes and "C! completed" in codes


def test_runs_not_evaluated_on_later_grid_pages_still_hold_back_the_comparison() -> None:
    """The grid is paginated; ``counts`` cover every run. E runs beyond this page are still not
    evaluated: say so, and do not suggest results_compare (it would count them missing)."""
    data = _grid(
        [
            [{"code": "V", "run_id": "a1", "runs": 1}, {"code": "V", "run_id": "a2", "runs": 1}],
            [{"code": "V", "run_id": "b1", "runs": 1}, {"code": "V", "run_id": "b2", "runs": 1}],
        ],
        V=4,
        E=3,
    )
    data["grid"]["total_rows"] = 5
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "3 completed run(s) are not evaluated yet" in text
    assert 'evaluation_collect(run_set_ref="exp_1"' in text
    assert "results_compare(" not in text


def test_c_bang_is_never_explained_away_as_pending() -> None:
    """The old workaround matched C! reasons; C! now always means a bad result."""
    data = _grid(
        [
            [{"code": "C!", "run_id": "a1", "runs": 1, "reason": "evaluation partial"}],
            [{"code": "V", "run_id": "b1", "runs": 1}],
        ],
        V=1,
    )
    data["grid"]["cols"] = ["seed=1"]
    data["counts"]["C!"] = 1
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "not a failure)" not in text.split("codes:")[0]
    assert 'why it failed: run_inspect(run_id="a1")' in text
    assert "evaluation_collect(" not in text


def test_grid_with_a_failed_smoke_gate_does_not_say_wait() -> None:
    notes = [
        "smoke run smoke.m-a.seed-1 (run_s): F host_oom - main cells of its workflow are dispatched only after a smoke run passes",
        "workflow wf_1: state gate_failed, dispatch open - its undispatched cells will not start on their own",
    ]
    data = _grid([[{"code": "P", "run_id": "r1", "runs": 1}] * 2] * 2, notes, P=4)
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "check again later" not in text
    assert 'run_inspect(run_id="run_s")' in text
    assert 'workflow_submit(workflow_id="wf_1", resume_dispatch=true' in text
    assert "note: smoke run smoke.m-a.seed-1" in text


def test_grid_of_undispatched_runs_without_workflow_says_nothing_is_running() -> None:
    data = _grid([[{"code": "P", "run_id": "r1", "runs": 1}] * 2] * 2, P=4)
    text = render_text("runs_list", _ok(data), {"experiment_id": "exp_1"})
    assert "nothing is running" in text and "check again later" not in text


def test_a_broken_renderer_falls_back_to_compact_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any) -> list[str]:
        raise KeyError("x")

    monkeypatch.setitem(srv.RENDERERS, "run_submit", boom)
    text = render_text("run_submit", _ok({"attempt_id": "att_1", "run_id": "run_1", "phase": "staging"}))
    assert text.splitlines()[0] == "summary line"
    assert '"attempt_id":"att_1"' in text


def test_compact_json_drops_noise_keeps_ids_and_fits_the_budget() -> None:
    h = "sha256:" + "a" * 64
    data = {
        "run_id": "run_1",
        "run_ref": "hpclab://runs/run_1",
        "exec_plan_hash": h,
        "nothing": None,
        "empty": [],
        "x": 0.123456789123,
        "rows": [{"i": i, "text": "y" * 1000} for i in range(200)],
    }
    out = compact_json(data, budget=4096)
    assert len(out.encode()) <= 4096
    parsed = json.loads(out)
    assert parsed["run_id"] == "run_1" and parsed["exec_plan_hash"] == h and "run_ref" not in parsed
    assert "nothing" not in parsed and "empty" not in parsed and parsed["x"] == 0.123457
    assert parsed["rows"][-1].startswith("… ")


def test_text_is_hard_limited() -> None:
    text = render_text(None, _ok({}, "s" * (TEXT_HARD_LIMIT * 2)))
    assert len(text.encode()) <= TEXT_HARD_LIMIT and "structuredContent" in text


def test_logs_are_raw_text_with_a_continue_hint() -> None:
    env = _ok(
        {
            "attempt_id": "att_1",
            "stream": "stderr",
            "text": 'Traceback (most recent call last):\n  File "t.py"\nRuntimeError: boom\n',
            "offset": 0,
            "next_offset": 60,
            "size": 900,
            "truncated": True,
            "rotated": False,
            "eof": False,
        },
        next_cursor="CUR",
    )
    text = render_text("run_logs", env)
    assert "RuntimeError: boom" in text and "\\n" not in text
    assert 'continue reading: run_logs(attempt_id="att_1", stream="stderr", cursor="CUR")' in text
    assert "more results:" not in text


def test_incomplete_coverage_is_loud() -> None:
    cov = {
        "complete": False,
        "sources": [{"name": "squeue:hai", "status": "unavailable", "detail": "auth required"}],
        "missing": ["queue state"],
        "notes": [],
    }
    text = render_text("usage_report", _ok({}, coverage=cov))
    assert "INCOMPLETE: squeue:hai=unavailable (auth required)" in text and "missing: queue state" in text


def test_call_hint_uses_json_literals() -> None:
    assert (
        call_hint("run_logs", attempt_id="a", tail_lines=40, stream=None, cols=["seed"], x=True)
        == 'run_logs(attempt_id="a", tail_lines=40, cols=["seed"], x=true)'
    )


def test_normalize_arguments_drops_nulls_and_resolves_paths(tmp_path: Path) -> None:
    out = normalize_arguments("runs_list", {"experiment_id": None, "view": None, "limit": 5}, tmp_path)
    assert out == {"limit": 5}
    out = normalize_arguments("experiment_prepare", {"spec_path": "exp/hpclab.yaml"}, tmp_path)
    assert out["spec_path"] == str((tmp_path / "exp/hpclab.yaml").resolve())
    out = normalize_arguments("evidence_verify", {"tex_paths": ["paper/main.tex"]}, tmp_path)
    assert out["tex_paths"] == [str((tmp_path / "paper/main.tex").resolve())]


def test_optional_parameters_are_plain_types_in_the_schema() -> None:
    prop = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "description": "d"}
    assert srv._collapse_optional(prop) == {"type": "string", "description": "d"}
    keep = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "x"}
    assert srv._collapse_optional(keep) == keep
    schema = {"properties": {"title": {"title": "Title", "anyOf": [{"type": "string"}, {"type": "integer"}]}}}
    srv._compact_schema(schema)
    assert schema == {"properties": {"title": {"type": ["string", "integer"]}}}


# ---------------------------------------------------------------------------
# Envelope summaries (services/app.py): the first line of every text block
# ---------------------------------------------------------------------------


def test_usage_summary_reads_the_scoped_totals() -> None:
    d = {
        "scope": "account",
        "cluster_id": "hai",
        "since": "a",
        "until": "b",
        "visibility": "self_only",
        "totals": {"self": {"gpu_hours": 12.5, "jobs": 3}},
    }
    s = app._s_usage(d)
    assert "12.5 allocated GPU-h (self, 3 jobs)" in s and "no lab total" in s


def test_runs_list_summary_counts_valid_cells_without_aggregate_keys() -> None:
    s = app._s_runs_list({"view": "grid", "total": 6, "counts": {"V": 5, "F": 1, "planned": 6, "valid": 5}})
    assert s == "grid: 5/6 planned main run(s) valid (V=5, F=1)"
    s = app._s_runs_list({"view": "list", "total": 30, "runs": [{}] * 20, "counts": {"V": 30}})
    assert s == "30 run(s) (showing 20): V=30"


def test_prepare_summary_names_blocking_checks() -> None:
    d = {
        "plan_id": "pln_1",
        "status": "invalid",
        "runs": [],
        "validation": {
            "status": "fail",
            "findings": [
                {"check": "env_guard_static", "severity": "block", "status": "fail", "override_reason": None}
            ],
        },
    }
    assert "BLOCKED by env_guard_static" in app._s_prepare(d)


def test_inspect_summary_leads_with_the_diagnosis() -> None:
    d = {
        "run": {"label": "m.seed-1", "status": "failed", "code": "F"},
        "focus_attempt_id": "att_1",
        "attempts": [{"attempt_id": "att_1", "seq": 1, "scheduler_state": "TIMEOUT", "job": {"job_id": "9"}}],
        "diagnosis": {
            "items": [{"code": "timeout", "title": "Walltime limit reached", "confidence": "high"}]
        },
    }
    assert app._s_inspect(d) == (
        "run m.seed-1: failed [F]; attempt #1 TIMEOUT (job 9); diagnosis timeout: Walltime limit reached (high)"
    )


def test_artifact_read_summary_for_directories_and_unknown_sizes() -> None:
    assert app._s_artifact_read({"artifact": {"name": "ckpt"}, "members": [], "members_total": 9}) == (
        "ckpt: directory with 9 member file(s)"
    )
    s = app._s_artifact_read({"artifact": {"name": "m"}, "view": "head", "bytes_read": 10, "size": None})
    assert "None" not in s


def test_verify_summary_counts_codes() -> None:
    d = {
        "bundle_id": "evb_1",
        "ok": False,
        "findings": [],
        "counts_by_severity": {"error": 2},
        "counts_by_code": {"STALE_TEX_NUMBER": 2},
    }
    assert app._s_verify(d).startswith("bundle evb_1: NOT OK — 2 error(s), 0 warning(s) (STALE_TEX_NUMBER×2)")
