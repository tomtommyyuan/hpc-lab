"""Renderers are honest about unknowns, and the captured Envelopes still match the services.

The CLI tests render real Envelopes captured from a controller + simulator run. If a service
renames or drops a field, ``test_corpus_matches_current_service_models`` fails, so the canned
data (and the renderers written against it) cannot silently drift from the services.
"""

from __future__ import annotations

from typing import Any

import pytest
from cli_support import corpus_names, env
from rich.console import Console

from hpclab.cli import views as V
from hpclab.services.app import build_registry

_LIFTED = ("coverage", "observed_at", "warnings", "next_cursor")


def _render(renderable: Any) -> str:
    c = Console(width=200, record=True, highlight=False)
    c.print(renderable)
    return c.export_text()


@pytest.fixture(scope="module")
def registry() -> Any:
    return build_registry(strict=True)


@pytest.mark.parametrize("name", corpus_names())
def test_corpus_matches_current_service_models(name: str, registry: Any) -> None:
    e = env(name)
    method = name.split(".")[0]
    model = registry.get(method).data_model
    assert model is not None, method
    data = dict(e["data"])
    fields = model.model_fields
    for key in _LIFTED:  # the registry lifts these into the Envelope; put them back
        if key not in fields:
            continue
        value = e.get(key)
        if value is None and not fields[key].is_required():
            value = fields[key].get_default(call_default_factory=True)
        if value is None and key == "coverage":
            value = {"complete": True, "sources": [], "missing": [], "notes": []}
        data[key] = value
    model.model_validate(data)


def test_unknown_values_are_never_rendered_as_zero() -> None:
    assert V.gpuh(None) == "unknown"
    assert V.gpuh(float("nan")) == "unknown"
    assert V.gpuh(0.0) == "0.00 GPU-h"
    assert V.gpuh(0.001) == "<0.01 GPU-h"
    assert V.gpuh(1234.5, unit=False) == "1,234.50"
    assert V.dur(None) == "?"
    assert V.dur(59) == "59s" and V.dur(61) == "1m01s" and V.dur(3 * 3600 + 7 * 60) == "3h07m"
    assert V.dur(2 * 86400 + 4 * 3600) == "2d04h"
    assert V.num(None) == "—" and V.num(0.123456) == "0.1235" and V.signed(-0.5) == "-0.5000"
    assert V.signed(0.25) == "+0.2500" and V.signed(None) == "—"
    assert V.size(None) == "?" and V.size(2048) == "2.0 KiB"
    assert V.short_digest("sha256:" + "a" * 64) == "sha256:" + "a" * 12


def test_usage_view_marks_lower_bounds_and_hides_account_totals_when_not_visible() -> None:
    d = env("usage_report.account_self_only")["data"]
    d["rows"][0]["gpu_count_unknown_jobs"] = 2
    d["totals"]["account"] = dict(d["totals"]["self"])
    text = _render(V.render_usage(d))
    assert "≥" in text  # rows with unknown GPU counts are a lower bound
    assert "account total" not in text
    d["visibility"] = "full"
    assert "account total" in _render(V.render_usage(d))


def test_usage_view_unknown_waste_is_unknown() -> None:
    d = env("usage_report.self")["data"]
    d["rows"][0]["wasted_gpu_hours_by_cause"] = None
    assert "unknown" in _render(V.render_usage(d))


def test_inspect_view_health_without_telemetry_is_unknown_not_idle() -> None:
    d = env("run_inspect.pending")["data"]
    d["health"] = {"source": "none", "flags": [], "per_gpu_util_mean": {}}
    text = _render(V.render_inspect(d))
    assert "no GPU telemetry yet (unknown, not idle)" in text
    d["health"] = {
        "source": "telemetry",
        "flags": ["idle_allocation"],
        "per_gpu_util_mean": {"0": 0.0},
        "heartbeat_age_s": 30,
    }
    text = _render(V.render_inspect(d))
    assert "idle_allocation" in text and "(advisory)" in text


def test_inspect_view_budget_unknown_and_observation_gap() -> None:
    d = env("run_inspect.failed")["data"]
    a = d["attempts"][0]
    a["budget"] = {
        "state": "unknown",
        "reserved": 0.17,
        "settled_consumed": None,
        "estimated_consumed": None,
        "note": None,
    }
    a["observation_gap"] = True
    text = _render(V.render_inspect(d))
    assert "unknown" in text and "gap" in text


def test_grid_view_multi_run_cells_and_empty_grid() -> None:
    d = env("runs_list.grid_done")["data"]
    d["grid"]["rows"][0]["cells"][0]["runs"] = 2
    text = _render(V.render_grid(d))
    assert "V ×2" in text
    d["grid"]["rows"] = []
    assert "no planned runs" in _render(V.render_grid(d))


def test_e_code_is_evaluation_pending_not_a_failure() -> None:
    """E = the job COMPLETED and its outputs are not known to be bad, no valid evaluation yet:
    it may still turn V, so it is neither terminal, nor active, nor a problem."""
    assert "E" in V.CODE_STYLE
    assert V.CODE_LEGEND["E"] == "completed, evaluation pending — not a failure"
    assert "E" not in V.TERMINAL_CODES | V.PROBLEM_CODES | V.ACTIVE_CODES
    assert "C!" in V.PROBLEM_CODES and "C!" in V.TERMINAL_CODES
    d = env("runs_list.grid_done")["data"]
    assert "E" in V.grid_codes(d)
    text = _render(V.render_grid(d))
    assert "E completed, evaluation pending — not a failure" in text  # legend
    assert "E mixup seed=3" not in text  # not listed under "needs attention"
    assert "F mixup seed=1: node_fail" in text
    assert "1 cell(s) completed, evaluation pending" in text and "hpclab collect" in text
    counts = _render(V.counts_line({"V": 1, "E": 2, "C!": 1, "planned": 4, "valid": 1}))
    assert "E=2" in counts and "C!=1" in counts
    assert counts.index("E=2") < counts.index("C!=1")


def test_c_bang_is_a_problem_with_its_reason() -> None:
    d = env("runs_list.grid_done")["data"]
    cell = d["grid"]["rows"][1]["cells"][2]
    cell.update(code="C!", reason="outputs incomplete")
    d["counts"].pop("E")  # the service's counts match the cells
    d["counts"]["C!"] = 1
    text = _render(V.render_grid(d))
    assert "C! mixup seed=3: outputs incomplete" in text
    assert "C! completed, result not valid" in text
    assert "evaluation pending" not in text


def test_compare_view_undefined_std_and_unpaired() -> None:
    d = env("results_compare")["data"]
    st = d["groups"][1]["metrics"]["acc"]
    st.update(n=1, std=None)
    d["differences"][0].update(kind="unpaired", ci_low=None, ci_high=None, n_baseline=3, unpaired_group=[1])
    text = _render(V.render_compare(d))
    assert "± —" in text and "(1/3)" in text
    assert "CI undefined" in text and "unpaired" in text and "unmatched seeds" in text


def test_recover_next_commands() -> None:
    assert V.recover_next(env("run_recover.not_eligible")["data"]) == []
    d = env("run_recover.eligible")["data"]
    assert V.recover_next(d) == [f"hpclab submit {d['run_id']} --exec-plan {d['exec_plan_hash']}"]
    d["probe"] = {"state": "pending", "run_id": "run_probe", "exec_plan_hash": "sha256:p"}
    cmds = V.recover_next(d)
    assert cmds[0].startswith("hpclab submit run_probe --exec-plan sha256:p")
    b = env("run_recover.batch")["data"]
    assert V.recover_next(b) == [f"hpclab submit {b['plan_id']}"]


def test_inspect_next_for_each_phase() -> None:
    d = env("run_inspect.uncertain")["data"]
    assert V.inspect_next(d)[0].startswith("hpclab ops op_")
    d["attempts"][0]["phase"] = "duplicate_detected"
    assert any("--bind-job" in c for c in V.inspect_next(d))
    d = env("run_inspect.pending")["data"]
    d["pending"] = {"category": "never_starts", "reason_code": "PartitionTimeLimit"}
    assert any(c.startswith("hpclab cancel") for c in V.inspect_next(d))
    d = env("run_inspect.failed")["data"]
    d["run"]["run_id"] = "run_x"
    d["attempts"] = []
    assert V.inspect_next(d) == ["hpclab submit run_x"]


def test_untrusted_strings_are_not_parsed_as_markup() -> None:
    d = env("run_inspect.failed")["data"]
    d["run"]["label"] = "[bold red]pwned[/] [link=http://x]x[/link]"
    text = _render(V.render_inspect(d))
    assert "[bold red]pwned[/]" in text


def test_inspect_says_uncertain_only_while_the_attempt_is_unresolved() -> None:
    d = env("run_inspect.uncertain")["data"]
    a = d["attempts"][0]
    since = a["uncertain_since"]
    text = _render(V.render_inspect(d))
    assert f"submission outcome uncertain since {since}" in text
    a["phase"] = "duplicate_detected"
    assert f"submission outcome uncertain since {since}" in _render(V.render_inspect(d))

    # reconciled: the controller found and bound the job; the uncertainty is history, not a warning
    a.update(phase="observing", scheduler_state="RUNNING", identified_at="2026-09-28T06:10:00Z")
    a["job"] = dict(a["job"], job_id="4242")
    text = _render(V.render_inspect(d))
    assert "uncertain since" not in text and "reconciling" not in text
    assert f"was uncertain from {since} until bound to job 4242" in text

    # a human abandoned it: no job to name, but the resolution is shown
    a.update(phase="abandoned", identified_at=None, resolution="abandon: squeue shows nothing")
    a["job"] = dict(a["job"], job_id=None)
    text = _render(V.render_inspect(d))
    assert "uncertain since" not in text
    assert f"was uncertain from {since}" in text and "abandon: squeue shows nothing" in text
