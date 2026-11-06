"""Regression tests for problems found by using the CLI against a real controller + simulator.

Each Envelope here is a real capture (``fixtures/envelopes.json``: ``plan_get.recovery_*``,
``events_list.*``, ``run_inspect.abandoned``, ``artifact_read.dir`` ...), so the scenarios are
the ones a user actually hits: a recovery plan after a gate failure, `submit RUN` after
`recover`, a cuda_oom diagnosis asking for another GPU type, 80-column terminals, `--json`
pipelines and an unusable HPCLAB_HOME.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from cli_support import Cli, env, ok

from hpclab.cli import views as V
from hpclab.cli.common import console, err_console

RECOVERY_PLAN = "pln_01M3KBK3P8ZJ14MDJ6R6ZYMA0G"  # batch recovery of a gate-failed workflow (7 cells)
SINGLE_RECOVERY_PLAN = "pln_01M3KBQMBX07R2YN30BRGH8RXZ"
ABANDONED_RUN = "run_01M3KBMVXVB850YHS1K3PYV501"
ABANDONED_ATT = "att_01M3KBN2WQ1BY031TBY69G15R5"
RETRY_PLAN_HASH = "sha256:063fce40be5ef0397acc9a6d1423a635814e983334bf4396e3bd1266f0d843c9"
FAILED_ATT = "att_01M3K9QD2FYB3TBD0RR59FXFBT"
GOOD_RUN = "run_01M3K9Q1D83ZECZ5F8ZVGB3VS7"


def _narrow(monkeypatch: pytest.MonkeyPatch, width: int = 80) -> None:
    monkeypatch.setattr(console, "_width", width)
    monkeypatch.setattr(err_console, "_width", width)


# ---------------------------------------------------------------------------
# recovery plans: the plan view and the pre-submit GPU-hours check
# ---------------------------------------------------------------------------


def test_plan_view_lists_the_cells_of_a_recovery_plan(cli: Cli) -> None:
    cli.client.responses["plan_get"] = env("plan_get.recovery_batch")
    r = cli.run("plan", RECOVERY_PLAN)
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert "smoke.method-baseline.seed-1" in out and "method-mixup.seed-3" in out
    assert "retry" in out and "initial" in out
    assert "7 cell(s)" in out and "1.25 GPU-h" in out
    assert "budget: unknown" not in out


def test_submit_recovery_plan_confirmation_counts_cells_and_gpu_hours(cli: Cli) -> None:
    cli.client.responses["plan_get"] = env("plan_get.recovery_batch")
    cli.client.responses["experiment_get"] = env("experiment_get")
    r = cli.run("submit", RECOVERY_PLAN)  # no terminal, no --yes: the cost is shown, nothing is sent
    assert r.exit_code == 2
    assert "0 run(s)" not in r.stderr
    assert "7 cell(s)" in r.stderr
    assert "1.25 GPU-h" in r.stderr and "total: unknown" not in r.stderr
    assert "workflow_submit" not in cli.client.methods()


def test_submit_run_after_recover_shows_the_gpu_hours_of_the_recovery_plan(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.abandoned")
    cli.client.responses["events_list"] = env("events_list.recovery_planned")
    cli.client.responses["plan_get"] = env("plan_get.recovery_single")
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["run_submit"] = env("run_submit")
    r = cli.run("submit", ABANDONED_RUN)
    assert r.exit_code == 2
    assert "retry plan" in r.stderr and ABANDONED_ATT in r.stderr
    assert "0.17 GPU-h" in r.stderr
    assert "not shown here" not in r.stderr
    assert "run_submit" not in cli.client.methods()
    r = cli.run("submit", ABANDONED_RUN, "--yes")
    assert r.exit_code == 0, r.text
    assert cli.client.params("run_submit")["execution_plan_hash"] == RETRY_PLAN_HASH


def test_submit_experiment_never_picks_a_recovery_plan(cli: Cli) -> None:
    """Like workflow_submit's own rule: the experiment's latest ready matrix/derived plan. A
    recovery plan (made by `hpclab recover`) is newer and 'ready', but is submitted by its id."""
    exp = env("experiment_get")
    matrix = exp["data"]["plans"][0]
    exp["data"]["plans"].append(
        {
            **matrix,
            "kind": "recovery",
            "plan_id": RECOVERY_PLAN,
            "plan_hash": "sha256:" + "f" * 64,
            "created_at": "2026-09-28T07:00:00.000Z",
            "run_count": 0,
            "smoke_run_id": None,
        }
    )
    cli.client.responses["experiment_get"] = exp
    cli.client.responses["plan_get"] = env("plan_get")
    cli.client.responses["workflow_submit"] = env("workflow_submit")
    r = cli.run("submit", exp["data"]["experiment_id"], "--yes")
    assert r.exit_code == 0, r.text
    assert cli.client.params("plan_get")["plan_id"] == matrix["plan_id"]
    assert cli.client.params("workflow_submit")["plan_hash"] == matrix["plan_hash"]


# ---------------------------------------------------------------------------
# recover: every resource field the diagnosis may ask for
# ---------------------------------------------------------------------------


def test_recover_can_change_gpu_type_as_the_cuda_oom_remediation_asks(cli: Cli) -> None:
    cli.client.responses["run_recover"] = env("run_recover.eligible")
    r = cli.run(
        "recover",
        FAILED_ATT,
        "--strategy",
        "adjust_resources",
        "--gpu-type",
        "h100",
        "--gpus",
        "2",
        "--exclude-nodes",
        "gpu-01,gpu-02",
        "--account",
        "lab2",
        "--qos",
        "high",
        "--mem",
        "8G",
    )
    assert r.exit_code == 0, r.text
    assert cli.client.params("run_recover")["resource_patch"] == {
        "gpu_type": "h100",
        "gpus": 2,
        "exclude_nodes": ["gpu-01", "gpu-02"],
        "account": "lab2",
        "qos": "high",
        "mem": "8G",
    }


def test_recover_resource_options_are_single_attempt_only(cli: Cli) -> None:
    r = cli.run("recover", "--workflow", "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA", "--gpu-type", "h100")
    assert r.exit_code == 2
    assert cli.client.calls == []


# ---------------------------------------------------------------------------
# narrow terminals: a long id-like value must not starve the other columns
# ---------------------------------------------------------------------------


def test_events_with_digest_entities_keep_every_column_readable_at_80_columns(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    _narrow(monkeypatch)
    cli.client.responses["events_list"] = env("events_list.long_entity")
    r = cli.run("events", "--type", "execution_plan.")
    assert r.exit_code == 0, r.text
    assert "worker" in r.stdout and "api" in r.stdout  # the source column is not squeezed to nothing
    assert "sha256:54b4583481e9cdb1 " in r.stdout  # a digest entity is shortened, not left to overflow


def test_export_lists_written_files_by_name_at_80_columns(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    _narrow(monkeypatch)
    cli.client.responses["evidence_export"] = env("evidence_export")
    wr = env("evidence_write")
    long_dir = "/very/long/path/to/a/paper/repository/checked/out/on/a/cluster/login/node/evidence"
    for f in wr["data"]["files"]:
        f["path"] = f"{long_dir}/{f['name']}"
    wr["data"]["out_dir"] = long_dir
    cli.client.responses["evidence_write"] = wr
    r = cli.run("export", "cmp_01M3K9QRZ5QK5S5CSQ2H9D0N2B", "--out", "/tmp/evidence-test-out")
    assert r.exit_code == 0, r.text
    for name in ("manifest.json", "summary.md", "table.csv", "table.tex"):
        assert name in r.stdout, name
    assert "written" in r.stdout


def test_export_conflict_names_the_cli_flag(cli: Cli) -> None:
    from cli_support import err

    cli.client.responses["evidence_export"] = env("evidence_export")
    cli.client.responses["evidence_write"] = err(
        "precondition_failed",
        "files with different content already exist; pass overwrite=true to replace them",
        conflicts=["/tmp/evidence-test-out/table.tex"],
    )
    r = cli.run("export", "cmp_01M3K9QRZ5QK5S5CSQ2H9D0N2B", "--out", "/tmp/evidence-test-out")
    assert r.exit_code == 1
    assert "--overwrite" in r.stderr


def test_which_on_a_reused_job_id_prints_the_commands_that_disambiguate(cli: Cli) -> None:
    from cli_support import err

    candidates = [
        {"attempt_id": "att_A", "run_id": "run_A", "submit_time": "2026-09-28T06:30:08.000Z"},
        {"attempt_id": "att_B", "run_id": "run_B", "submit_time": "2026-09-28T07:01:59.000Z"},
    ]
    cli.client.responses["run_inspect"] = err(
        "ambiguous", "2 attempts share job id 1000 on sim; give submit_time", candidates=candidates
    )
    r = cli.run("which", "1000", "--cluster", "sim")
    assert r.exit_code == 1
    assert "hpclab which 1000 --cluster sim --submit-time 2026-09-28T06:30:08.000Z" in r.stderr
    assert "hpclab which 1000 --cluster sim --submit-time 2026-09-28T07:01:59.000Z" in r.stderr


# ---------------------------------------------------------------------------
# --json: errors raised by the CLI itself are Envelopes too
# ---------------------------------------------------------------------------


def test_json_mode_prints_an_error_envelope_for_cli_side_errors(cli: Cli) -> None:
    cli.client.responses["experiments_list"] = ok({"experiments": []})
    r = cli.run("--json", "grid", "no-such-experiment")
    assert r.exit_code == 1
    doc = json.loads(r.stdout)
    assert doc["ok"] is False and doc["error"]["code"] == "not_found"
    assert "no-such-experiment" in doc["error"]["message"]
    assert doc["error"]["details"]["hint"]


def test_json_mode_refusal_without_yes_is_an_error_envelope(cli: Cli) -> None:
    cli.client.responses["plan_get"] = env("plan_get")
    cli.client.responses["experiment_get"] = env("experiment_get")
    r = cli.run("--json", "submit", "pln_01M3K9Q1CC6RNE08MM569NQ42F")
    assert r.exit_code == 2
    doc = json.loads(r.stdout)
    assert doc["ok"] is False and doc["error"]["code"] == "invalid_argument"
    assert "--yes" in doc["error"]["message"]
    assert "workflow_submit" not in cli.client.methods()


def test_json_mode_controller_unavailable_is_an_error_envelope(cli: Cli) -> None:
    cli.client.unavailable = True
    r = cli.run("--json", "inspect", GOOD_RUN)
    assert r.exit_code == 3
    doc = json.loads(r.stdout)
    assert doc["ok"] is False and doc["error"]["code"] == "controller_unavailable"


# ---------------------------------------------------------------------------
# an unusable home (permissions, expired AFS token) is a message, not a traceback
# ---------------------------------------------------------------------------


def test_unusable_home_is_reported_as_controller_unavailable(cli: Cli) -> None:
    def denied(_: dict[str, Any]) -> dict[str, Any]:
        raise PermissionError(13, "Permission denied", "/afs/home/alice/.hpclab/run/controller.lock")

    cli.client.responses["run_inspect"] = denied
    r = cli.run("inspect", GOOD_RUN)
    assert r.exit_code == 3
    assert "Permission denied" in r.stderr and "HPCLAB_HOME" in r.stderr
    r = cli.run("--json", "inspect", GOOD_RUN)
    assert r.exit_code == 3
    assert json.loads(r.stdout)["error"]["code"] == "controller_unavailable"


@pytest.mark.parametrize("command", ["init", "status"])
def test_init_and_status_with_an_unusable_home_exit_3_without_traceback(
    cli: Cli, command: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setenv("HPCLAB_HOME", str(blocker / "home"))
    r = cli.run(command)
    assert r.exit_code == 3, r.text
    assert "cannot use the hpclab home" in r.stderr and "HPCLAB_HOME" in r.stderr


# ---------------------------------------------------------------------------
# ops --wait after re-authenticating: an auth block that is already fixed is not a stop
# ---------------------------------------------------------------------------

OP = "op_01M3K9QT35C3NPYRVRGQGCXFW1"


def _auth_blocked() -> dict[str, Any]:
    e = env("operation_get.blocked")
    e["data"].update(
        stage="auth_required",
        error={
            "code": "auth_required",
            "message": "SSH needs interactive authentication (ControlMaster expired)",
            "retryable": True,
            "details": {"cluster_id": "hai", "remediation": "hpclab auth hai"},
        },
    )
    return e


def _auth_command(alive: bool | None) -> dict[str, Any]:
    return ok(
        {
            "cluster_id": "hai",
            "needed": not alive,
            "argv": ["ssh", "-M", "hai"] if alive is not None else [],
            "command": "ssh -M hai" if alive is not None else "",
            "master_alive": alive,
            "explanation": "…",
        }
    )


def test_ops_wait_keeps_waiting_when_the_connection_is_already_back(cli: Cli) -> None:
    """An older controller without auth_resume: keep waiting for the automatic retry."""
    cli.client.responses["operation_get"] = [_auth_blocked(), _auth_blocked(), env("operation_get.succeeded")]
    cli.client.responses["auth_command"] = _auth_command(True)
    r = cli.run("ops", OP, "--wait")
    assert r.exit_code == 0, r.text
    assert "retried automatically" in r.stderr
    assert cli.client.params("auth_command") == {"cluster_id": "hai"}
    assert cli.client.methods().count("auth_resume") == 1  # tried once, tolerated as unknown


def test_ops_wait_resumes_the_blocked_step_once_the_connection_is_back(cli: Cli) -> None:
    cli.client.responses["operation_get"] = [
        _auth_blocked(),
        _auth_blocked(),
        _auth_blocked(),
        env("operation_get.succeeded"),
    ]
    cli.client.responses["auth_command"] = _auth_command(True)
    cli.client.responses["auth_resume"] = ok({"woken": 1, "clusters": ["hai"], "note": None})
    r = cli.run("ops", OP, "--wait")
    assert r.exit_code == 0, r.text
    assert cli.client.methods().count("auth_resume") == 1
    assert cli.client.params("auth_resume") == {"cluster_id": "hai"}
    assert "resumed 1 waiting step(s)" in r.stderr


def test_ops_wait_stops_on_auth_when_login_is_still_needed(cli: Cli) -> None:
    cli.client.responses["operation_get"] = [_auth_blocked(), env("operation_get.succeeded")]
    cli.client.responses["auth_command"] = _auth_command(False)
    r = cli.run("ops", OP, "--wait")
    assert r.exit_code == 4
    assert "hpclab auth hai" in r.stdout
    assert cli.client.methods().count("operation_get") == 1


# ---------------------------------------------------------------------------
# paper check: one finding per bad tag
# ---------------------------------------------------------------------------

BUNDLE = "evb_01M3K9QS049T7A49DZQY7J7WJF"
UNKNOWN_BUNDLE = "evb_01M3KBAZD7GV8YQS4ERRPQFF5X"


def test_paper_check_reports_each_unresolvable_tag_once(cli: Cli, tmp_path: Any) -> None:
    """evidence_verify checks every tag of each tex file it is given (whatever its bundle), so
    unknown or malformed references in a file some verify call covered are already reported."""
    tex = tmp_path / "paper.tex"
    tex.write_text(
        f"A \\hpcval{{{BUNDLE}:mixup.acc.mean}}{{0.842}}\n"
        f"B \\hpcval{{{UNKNOWN_BUNDLE}:mixup.acc.mean}}{{0.842}}\n"
        "C \\hpcval{garbage}{1}\n"
    )
    where = str(tex.resolve())
    service_findings = [
        {
            "code": "UNRESOLVED_REF",
            "severity": "error",
            "message": f"\\hpcval references unknown bundle {UNKNOWN_BUNDLE!r}",
            "refs": [f"{where}:2"],
        },
        {
            "code": "UNRESOLVED_REF",
            "severity": "error",
            "message": "malformed \\hpcval reference 'garbage'",
            "refs": [f"{where}:3"],
        },
    ]

    def verify(p: dict[str, Any]) -> dict[str, Any]:
        if p["bundle_ref"] != BUNDLE:
            from cli_support import err

            return err("not_found", f"evidence bundle {p['bundle_ref']} not found")
        e = env("evidence_verify.clean")
        e["data"].update(ok=False, findings=service_findings)
        return e

    cli.client.responses["evidence_verify"] = verify
    r = cli.run("--json", "paper", "check", str(tex))
    assert r.exit_code == 1
    assert [f["message"] for f in r.json["findings"]] == [f["message"] for f in service_findings]
    r = cli.run("paper", "check", str(tex))
    assert r.exit_code == 1
    assert "2 error(s)" in r.stdout and "UNRESOLVED_REF" in r.stdout


def test_paper_check_still_reports_unknown_bundles_when_nothing_could_be_verified(
    cli: Cli, tmp_path: Any
) -> None:
    tex = tmp_path / "paper.tex"
    tex.write_text(f"B \\hpcval{{{UNKNOWN_BUNDLE}:mixup.acc.mean}}{{0.842}} C \\hpcval{{garbage}}{{1}}\n")
    from cli_support import err

    cli.client.responses["evidence_verify"] = err("not_found", f"evidence bundle {UNKNOWN_BUNDLE} not found")
    r = cli.run("--json", "paper", "check", str(tex))
    assert r.exit_code == 1
    messages = [f["message"] for f in r.json["findings"]]
    assert len(messages) == 2
    assert any(UNKNOWN_BUNDLE in m for m in messages) and any("garbage" in m for m in messages)


# ---------------------------------------------------------------------------
# honest rendering
# ---------------------------------------------------------------------------


def _text(renderable: Any, width: int = 200) -> str:
    from rich.console import Console

    c = Console(width=width, record=True, highlight=False)
    c.print(renderable)
    return c.export_text()


def test_inspect_of_an_ended_attempt_shows_no_eta_or_walltime_left() -> None:
    text = _text(V.render_inspect(env("run_inspect.succeeded")["data"]))
    assert "walltime left" not in text and "ETA" not in text
    assert "step 300/300" in text


def test_inspect_of_a_running_attempt_still_shows_eta_and_walltime_left() -> None:
    d = env("run_inspect.succeeded")["data"]
    for a in d["attempts"]:
        a.update(phase="observing", scheduler_state="RUNNING", result_state="unchecked")
    d["progress"].update(step=150, fraction=0.5, eta_s=120, remaining_walltime_s=480, final_seen=False)
    text = _text(V.render_inspect(d))
    assert "ETA 2m00s vs walltime left 8m00s" in text


def test_grid_rows_without_a_variant_do_not_print_none() -> None:
    d = env("runs_list.grid_done")["data"]
    grid = d["grid"]
    grid["row_axes"] = [*grid.get("row_axes", []), "variant"]
    for i, r in enumerate(grid["rows"]):
        r["values"] = {**(r.get("values") or {}), "variant": "d7ce120" if i == 0 else None}
    text = _text(V.render_grid(d))
    assert "None" not in text
    assert "d7ce120" in text


def test_read_jsonl_tail_without_a_complete_record_says_so(cli: Cli) -> None:
    cli.client.responses["artifact_read"] = env("artifact_read.jsonl_tail_empty")
    r = cli.run("read", "art_01M3KBA79C00A07FHR0WNSRX0C", "--view", "jsonl_tail", "--max-bytes", "300")
    assert r.exit_code == 0, r.text
    assert "no complete JSONL record" in r.stderr and "--max-bytes" in r.stderr


def test_read_of_a_directory_lists_members_without_placeholders(cli: Cli) -> None:
    cli.client.responses["artifact_read"] = env("artifact_read.dir")
    r = cli.run("read", "art_01M3KBA79CN1JH3E4FX2SVG0WH")
    assert r.exit_code == 0, r.text
    assert "None" not in r.text
    assert "checkpoints/ckpt-000000100/model.json" in r.stdout
    assert "#ckpt-000000100" not in r.stdout  # the member path is not repeated after the name


def test_usage_without_waste_has_no_empty_parentheses() -> None:
    text = _text(V.render_usage(env("usage_report.self_nowaste")["data"]))
    assert "()" not in text
    assert "waste 0.00 GPU-h" in text


def test_incomplete_coverage_note_explains_what_is_missing(cli: Cli) -> None:
    cli.client.responses["results_compare"] = env("results_compare")
    r = cli.run("compare", "exp_01M3K9Q1CC33H1R96NRPJEDAVW", "-m", "acc")
    assert r.exit_code == 0, r.text
    assert "some sources were not consulted" not in r.stdout
    assert "1 of 6 planned runs are not included" in r.stdout


# ---------------------------------------------------------------------------
# help texts say what the command really does
# ---------------------------------------------------------------------------


def _click(*path: str) -> Any:
    import typer

    from hpclab.cli.main import app

    cmd: Any = typer.main.get_command(app)
    for name in path:
        cmd = cmd.commands[name]
    return cmd


def _opt_help(cmd: Any, flag: str) -> str:
    return next(str(p.help or "") for p in cmd.params if flag in getattr(p, "opts", ()))


def test_help_texts_match_the_behaviour() -> None:
    digest = _click("digest")
    cluster_help = _opt_help(digest, "--cluster")
    assert "usage section" not in cluster_help.lower()  # only --account adds one
    assert "narrow" in cluster_help.lower() or "only" in cluster_help.lower()
    assert "usage section" in _opt_help(digest, "--account").lower()

    evidence = _click("evidence")
    assert "aliases of verify/trace" in str(evidence.help)  # there is no `evidence export`
    assert set(evidence.commands) == {"verify", "trace"}

    resume = " ".join(str(_click("workflow", "resume").help).split())
    assert "circuit breaker" in resume and "gate_failed" in resume and "latest attempt" in resume


def test_cli_commands_under_test_run_outside_the_repository(cli: Cli) -> None:
    """Commands with a cwd-relative default (`hpclab export` writes ./evidence, `examples copy`
    ./NAME) must never write into the checkout when a test forgets --out: the `cli` fixture
    runs every test in a scratch working directory."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    assert cwd != repo and repo not in cwd.parents, cwd

    cli.client.responses["evidence_export"] = env("evidence_export")
    cli.client.responses["evidence_write"] = env("evidence_write")
    assert cli.run("export", "cmp_01M3K9QRZ5QK5S5CSQ2H9D0N2B").exit_code == 0
    out_dir = Path(cli.client.params("evidence_write")["out_dir"])
    assert out_dir == cwd / "evidence"
