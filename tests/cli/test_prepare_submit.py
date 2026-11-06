"""prepare / experiments / plan / submit / workflow commands against a scripted controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cli_support import Cli, env, err, ok

PLAN = "pln_01M3K9Q1CC6RNE08MM569NQ42F"
EXP = "exp_01M3K9Q1CC33H1R96NRPJEDAVW"
WF = "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA"
FAILED_RUN = "run_01M3K9Q1D8QH3MCQSDR0S9N2WS"
FAILED_ATT = "att_01M3K9QD2FYB3TBD0RR59FXFBT"
GOOD_RUN = "run_01M3K9Q1D83ZECZ5F8ZVGB3VS7"


def _spec(tmp_path: Path) -> Path:
    p = tmp_path / "hpclab.yaml"
    p.write_text("name: x\n")
    return p


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_ready_shows_findings_runs_budget_and_next_command(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["experiment_prepare"] = env("experiment_prepare.ready")
    r = cli.run("prepare", str(_spec(tmp_path)))
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert f"plan {PLAN} ready" in out
    assert "validation findings" in out
    assert "unchecked" in out and "gpus_limits" in out  # unknown is shown as unchecked, never pass
    assert "other check(s) passed" in out
    assert "method-mixup.seed-3" in out and "smoke.method-baseline.seed-1" in out
    assert "budget: 1.25 GPU-h requested" in out and "of 5.00 GPU-h budget" in out
    assert f"hpclab submit {PLAN}" in out
    p = cli.client.params("experiment_prepare")
    assert p["spec_path"] == str(tmp_path / "hpclab.yaml") and Path(p["spec_path"]).is_absolute()
    assert p["validate_remote"] is True
    assert p["idempotency_key"].startswith("prepare-")


def test_prepare_all_findings_lists_passing_checks(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["experiment_prepare"] = env("experiment_prepare.ready")
    r = cli.run("prepare", str(_spec(tmp_path)), "--all-findings")
    assert "spec_schema" in r.stdout and "other check(s) passed" not in r.stdout


def test_prepare_invalid_exits_1_with_blocking_checks_and_override_hint(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["experiment_prepare"] = env("experiment_prepare.invalid")
    r = cli.run("prepare", str(_spec(tmp_path)))
    assert r.exit_code == 1
    assert "invalid" in r.stdout
    assert "blocking: budget, time_within_partition_max" in r.stdout
    assert "--override budget='<reason>' --override time_within_partition_max='<reason>'" in r.stdout
    assert "hpclab submit" not in r.stdout


def test_prepare_each_key_is_fresh(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["experiment_prepare"] = env("experiment_prepare.ready")
    spec = str(_spec(tmp_path))
    cli.run("prepare", spec)
    cli.run("prepare", spec)
    keys = [p["idempotency_key"] for m, p in cli.client.calls if m == "experiment_prepare"]
    assert len(set(keys)) == 2


def test_prepare_extend_derive_and_override_parameters(cli: Cli) -> None:
    cli.client.responses["experiment_prepare"] = env("experiment_prepare.ready")
    cli.client.responses["experiments_list"] = env("experiments_list")
    r = cli.run(
        "prepare",
        "--extends",
        "tiny-mixup-ablation",
        "--add-seeds",
        "4,5",
        "--add",
        "method=cutmix,mixup2",
        "--add",
        "lr=0.1",
        "--override",
        "budget=approved by PI",
        "--no-remote-checks",
    )
    assert r.exit_code == 0, r.text
    p = cli.client.params("experiment_prepare")
    assert p["extends"] == EXP  # the name was resolved to the newest experiment with that name
    assert cli.client.params("experiments_list")["name"] == "tiny-mixup-ablation"
    assert p["add_seeds"] == [4, 5]
    assert p["add_matrix"] == {"method": ["cutmix", "mixup2"], "lr": [0.1]}
    assert p["overrides"] == [{"check": "budget", "reason": "approved by PI"}]
    assert p["validate_remote"] is False
    assert "spec_path" not in p

    r = cli.run(
        "prepare",
        "--derive-from",
        GOOD_RUN,
        "--set",
        "lr=0.05",
        "--set",
        "method=mixup",
        "--time",
        "02:00:00",
        "--mem",
        "8G",
        "--append-arg",
        "--steps",
        "--append-arg",
        "20",
    )
    assert r.exit_code == 0, r.text
    d = cli.client.params("experiment_prepare")["derive_from"]
    assert d["run_id"] == GOOD_RUN
    assert d["overrides"] == {
        "params": {"lr": 0.05, "method": "mixup"},
        "resources": {"time": "02:00:00", "mem": "8G"},
        "argv_append": ["--steps", "20"],
    }


def test_prepare_usage_errors_exit_2_without_calling_the_controller(cli: Cli, tmp_path: Path) -> None:
    cases: list[tuple[str, ...]] = [
        ("prepare",),
        ("prepare", str(tmp_path / "missing.yaml")),
        ("prepare", str(_spec(tmp_path)), "--add-seeds", "4"),
        ("prepare", str(_spec(tmp_path)), "--set", "lr=1"),
        ("prepare", str(_spec(tmp_path)), "--override", "budget"),
        ("prepare", "--extends", EXP, "--add-seeds", "x,y"),
    ]
    for args in cases:
        r = cli.run(*args)
        assert r.exit_code == 2, (args, r.text)
    assert "experiment_prepare" not in cli.client.methods()


def test_prepare_json_prints_the_raw_envelope(cli: Cli, tmp_path: Path) -> None:
    e = env("experiment_prepare.ready")
    cli.client.responses["experiment_prepare"] = e
    r = cli.run("--json", "prepare", str(_spec(tmp_path)))
    assert r.exit_code == 0
    assert r.json == e


def test_prepare_domain_error_exit_1_and_controller_down_exit_3(cli: Cli, tmp_path: Path) -> None:
    cli.client.responses["experiment_prepare"] = err("invalid_argument", "spec is not valid", hint="see docs")
    r = cli.run("prepare", str(_spec(tmp_path)))
    assert r.exit_code == 1
    assert "invalid_argument" in r.stderr and "see docs" in r.stderr
    cli.client.unavailable = True
    r = cli.run("prepare", str(_spec(tmp_path)))
    assert r.exit_code == 3
    assert "controller unavailable" in r.stderr


# ---------------------------------------------------------------------------
# experiments / experiment / plan
# ---------------------------------------------------------------------------


def test_experiments_list_and_json(cli: Cli) -> None:
    cli.client.responses["experiments_list"] = env("experiments_list")
    r = cli.run("experiments", "--name", "tiny")
    assert r.exit_code == 0
    assert "tiny-mixup-ablation" in r.stdout and EXP in r.stdout
    assert cli.client.params("experiments_list")["name"] == "tiny"
    r = cli.run("--json", "experiments")
    assert r.json == env("experiments_list")


def test_experiment_by_name_resolves_and_renders(cli: Cli) -> None:
    cli.client.responses["experiments_list"] = env("experiments_list")
    cli.client.responses["experiment_get"] = env("experiment_get")
    r = cli.run("experiment", "tiny-mixup-ablation")
    assert r.exit_code == 0, r.text
    assert cli.client.params("experiment_get") == {"experiment_id": EXP}
    assert "question:" in r.stdout and PLAN in r.stdout
    assert "runs: 4/6 valid" in r.stdout
    assert f"hpclab grid {EXP}" in r.stdout


def test_experiment_unknown_name_and_wrong_id_kind(cli: Cli) -> None:
    cli.client.responses["experiments_list"] = ok({"experiments": [], "next_cursor": None})
    r = cli.run("experiment", "nope")
    assert r.exit_code == 1 and "no experiment named 'nope'" in r.stderr
    r = cli.run("experiment", PLAN)
    assert r.exit_code == 2 and "got a plan id" in r.stderr


def test_experiment_name_ambiguity_uses_newest_with_a_note(cli: Cli) -> None:
    listing = env("experiments_list")
    first = listing["data"]["experiments"][0]
    older = {**first, "experiment_id": "exp_01M3K9Q1CC33H1R96NRPJEDAAA"}
    listing["data"]["experiments"] = [first, older]
    cli.client.responses["experiments_list"] = listing
    cli.client.responses["experiment_get"] = env("experiment_get")
    r = cli.run("experiment", "tiny-mixup-ablation")
    assert r.exit_code == 0
    assert "2 experiments are named" in r.stderr
    assert cli.client.params("experiment_get")["experiment_id"] == first["experiment_id"]


def test_plan_show_and_script(cli: Cli) -> None:
    cli.client.responses["plan_get"] = env("plan_get")
    r = cli.run("plan", PLAN)
    assert r.exit_code == 0
    assert "plan hash" in r.stdout and "more run(s)" in r.stdout
    cli.client.responses["plan_script"] = env("plan_script")
    r = cli.run("plan", PLAN, "--script")
    assert r.exit_code == 0
    assert r.stdout.startswith("#!/bin/bash\n#SBATCH")  # raw script on stdout, nothing else
    assert "placeholders" in r.stderr
    r = cli.run("plan", PLAN, "--run", GOOD_RUN)
    assert r.exit_code == 2
    r = cli.run("--json", "plan", PLAN, "--script")
    assert r.json["data"]["script"].startswith("#!/bin/bash")


def test_plan_script_pages_are_concatenated(cli: Cli) -> None:
    first = env("plan_script")
    first["data"]["script"] = "#!/bin/bash\necho one\n"
    first["next_cursor"] = "c1"
    second = env("plan_script")
    second["data"]["script"] = "echo two\n"
    second["next_cursor"] = None
    cli.client.responses["plan_script"] = lambda p: second if p.get("cursor") == "c1" else first
    r = cli.run("plan", PLAN, "--script", "--run", GOOD_RUN)
    assert r.stdout == "#!/bin/bash\necho one\necho two\n"
    assert cli.client.params("plan_script", 0)["run_id"] == GOOD_RUN


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def _script_plan(cli: Cli) -> None:
    plan = env("plan_get")
    plan["next_cursor"] = None
    cli.client.responses["plan_get"] = plan
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["workflow_submit"] = env("workflow_submit")


def test_submit_plan_requires_yes_without_a_terminal(cli: Cli) -> None:
    _script_plan(cli)
    r = cli.run("submit", PLAN)
    assert r.exit_code == 2
    assert "refusing without --yes" in r.stderr
    assert "GPU-h" in r.stderr  # the cost is still shown
    assert "workflow_submit" not in cli.client.methods()


def test_submit_plan_interactive_no_aborts_yes_submits(cli: Cli) -> None:
    _script_plan(cli)
    r = cli.run("submit", PLAN, input="n\n", interactive=True)
    assert r.exit_code == 1 and "aborted" in r.stderr
    assert "workflow_submit" not in cli.client.methods()
    r = cli.run("submit", PLAN, input="y\n", interactive=True)
    assert r.exit_code == 0, r.text
    assert "workflow_submit" in cli.client.methods()


def test_submit_plan_with_yes_shows_cost_and_how_to_watch(cli: Cli) -> None:
    _script_plan(cli)
    r = cli.run("submit", PLAN, "--yes", "--max-parallel", "3")
    assert r.exit_code == 0, r.text
    out = r.stdout
    assert "main: 2 run(s) × 1 GPU (a100) × 00:10:00 = 0.33 GPU-h" in out
    assert "smoke: 1 run(s) × 1 GPU (a100) × 00:15:00 = 0.25 GPU-h" in out
    assert "more run(s) not itemized" in out
    assert "total: 1.25 GPU-h at most" in out
    assert "budget:" in out and "available of 5.00 GPU-h" in out
    assert f"hpclab grid {WF} --watch" in out
    p = cli.client.params("workflow_submit")
    assert p["plan_id"] == PLAN and p["max_parallel"] == 3
    assert p["plan_hash"] == env("plan_get")["data"]["plan_hash"]
    assert p["idempotency_key"].startswith("workflow-")


def test_submit_experiment_uses_latest_ready_plan_hash(cli: Cli) -> None:
    _script_plan(cli)
    r = cli.run("submit", EXP, "--yes")
    assert r.exit_code == 0, r.text
    p = cli.client.params("workflow_submit")
    assert p["prepared_experiment_id"] == EXP and "plan_id" not in p
    assert p["plan_hash"] == env("plan_get")["data"]["plan_hash"]
    assert cli.client.params("plan_get")["plan_id"] == PLAN


def test_submit_experiment_without_ready_plan_and_preparing_plan(cli: Cli) -> None:
    exp = env("experiment_get")
    for plan in exp["data"]["plans"]:
        plan["status"] = "invalid"
    cli.client.responses["experiment_get"] = exp
    r = cli.run("submit", EXP, "--yes")
    assert r.exit_code == 1 and "no ready plan" in r.stderr
    preparing = env("plan_get")
    preparing["data"].update(status="preparing", plan_hash=None, operation_id="op_01M3K9Q1MHKZ8HWQR03AZ2GP5A")
    cli.client.responses["plan_get"] = preparing
    r = cli.run("submit", PLAN, "--yes")
    assert r.exit_code == 1
    assert "still preparing" in r.stderr and "hpclab ops op_01M3K9Q1MHKZ8HWQR03AZ2GP5A --wait" in r.stderr
    assert "workflow_submit" not in cli.client.methods()


def test_submit_json_needs_yes_and_prints_the_envelope(cli: Cli) -> None:
    _script_plan(cli)
    r = cli.run("--json", "submit", PLAN)
    assert r.exit_code == 2 and r.json["ok"] is False and "--yes" in r.json["error"]["message"]
    assert "workflow_submit" not in cli.client.methods()
    r = cli.run("--json", "submit", PLAN, "--yes")
    assert r.exit_code == 0
    assert r.json == env("workflow_submit")


def test_submit_run_uses_its_unused_plan(cli: Cli) -> None:
    fresh = env("run_inspect.failed")
    fresh["data"]["attempts"] = []
    fresh["data"]["focus_attempt_id"] = None
    fresh["data"]["refs"]["exec_plan_hash"] = fresh["data"]["run"]["current_exec_plan_hash"]
    cli.client.responses["run_inspect"] = fresh
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["run_submit"] = env("run_submit")
    r = cli.run("submit", FAILED_RUN, "--yes")
    assert r.exit_code == 0, r.text
    assert "1 run × 1 GPU(s) (a100) × 00:10:00 = 0.17 GPU-h" in r.stdout
    assert "receipt, not proof" in r.stdout
    p = cli.client.params("run_submit")
    assert p["run_id"] == FAILED_RUN
    assert p["execution_plan_hash"] == fresh["data"]["run"]["current_exec_plan_hash"]
    assert p["force_without_gate"] is False


def test_submit_run_after_recover_picks_the_recovery_plan(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.failed")
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["run_submit"] = env("run_submit")
    new_hash = env("run_recover.eligible")["data"]["exec_plan_hash"]
    cli.client.responses["events_list"] = ok(
        {
            "events": [
                {
                    "seq": 9,
                    "event_id": "evt_1",
                    "entity_type": "attempt",
                    "entity_id": FAILED_ATT,
                    "entity_version": None,
                    "event_type": "recovery.planned",
                    "payload": {"strategy": "retry", "exec_plan_hash": new_hash, "run_id": FAILED_RUN},
                    "payload_truncated": False,
                    "source": "api",
                    "command_id": None,
                    "causation_id": None,
                    "observed_at": None,
                    "recorded_at": "2026-09-28T06:00:00.000Z",
                }
            ],
            "entity_id": FAILED_ATT,
            "event_type": "recovery.planned",
        }
    )
    r = cli.run("submit", FAILED_RUN, "--yes")
    assert r.exit_code == 0, r.text
    assert f"retry plan from `hpclab recover {FAILED_ATT}`" in r.stdout
    assert cli.client.params("run_submit")["execution_plan_hash"] == new_hash
    assert cli.client.params("events_list") == {
        "entity_id": FAILED_ATT,
        "event_type": "recovery.planned",
        "limit": 10,
    }


def test_submit_run_without_unused_plan_points_to_recover(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.failed")
    cli.client.responses["events_list"] = ok({"events": [], "entity_id": FAILED_ATT, "event_type": None})
    r = cli.run("submit", FAILED_RUN, "--yes")
    assert r.exit_code == 1
    assert f"hpclab recover {FAILED_ATT}" in r.stderr
    used = env("run_inspect.failed")["data"]["attempts"][0]["exec_plan_hash"]
    r = cli.run("submit", FAILED_RUN, "--yes", "--exec-plan", used)
    assert r.exit_code == 1 and "already used by attempt #1" in r.stderr
    assert "run_submit" not in cli.client.methods()


def test_submit_run_with_active_attempt_is_refused(cli: Cli) -> None:
    cli.client.responses["run_inspect"] = env("run_inspect.pending")
    r = cli.run("submit", env("run_inspect.pending")["data"]["run"]["run_id"], "--yes")
    assert r.exit_code == 1 and "already has an active attempt" in r.stderr


def test_submit_force_without_gate_needs_reason_and_only_for_runs(cli: Cli) -> None:
    fresh = env("run_inspect.failed")
    fresh["data"]["attempts"] = []
    cli.client.responses["run_inspect"] = fresh
    cli.client.responses["experiment_get"] = env("experiment_get")
    cli.client.responses["run_submit"] = env("run_submit")
    assert cli.run("submit", FAILED_RUN, "--yes", "--force-without-gate").exit_code == 2
    r = cli.run("submit", FAILED_RUN, "--yes", "--force-without-gate", "--reason", "gate env differs")
    assert r.exit_code == 0, r.text
    p = cli.client.params("run_submit")
    assert p["force_without_gate"] is True and p["reason"] == "gate env differs"
    assert cli.run("submit", PLAN, "--force-without-gate", "--reason", "x").exit_code == 2
    assert cli.run("submit", FAILED_RUN, "--max-parallel", "2").exit_code == 2
    assert cli.run("submit", WF).exit_code == 2


def test_submit_domain_errors_map_to_exit_codes(cli: Cli) -> None:
    _script_plan(cli)
    cli.client.responses["workflow_submit"] = err("budget_exceeded", "over budget", available=0.1)
    r = cli.run("submit", PLAN, "--yes")
    assert r.exit_code == 1 and "budget_exceeded" in r.stderr
    cli.client.responses["workflow_submit"] = err(
        "auth_required", "run hpclab auth", remediation="hpclab auth sim"
    )
    r = cli.run("submit", PLAN, "--yes")
    assert r.exit_code == 4 and "hpclab auth sim" in r.stderr


# ---------------------------------------------------------------------------
# workflow show / cancel / resume
# ---------------------------------------------------------------------------


def test_workflow_show(cli: Cli) -> None:
    cli.client.responses["workflow_get"] = env("workflow_get.running")
    r = cli.run("workflow", "show", WF)
    assert r.exit_code == 0
    assert f"workflow {WF}" in r.stdout and "gate passed" in r.stdout
    assert f"hpclab workflow cancel {WF}" in r.stdout
    r = cli.run("--json", "workflow", "show", WF)
    assert r.json == env("workflow_get.running")


def test_workflow_cancel_all_confirms_and_sends_version(cli: Cli) -> None:
    cli.client.responses["workflow_get"] = env("workflow_get.running")
    cli.client.responses["workflow_cancel"] = env("workflow_cancel")
    r = cli.run("workflow", "cancel", WF, "--all")
    assert r.exit_code == 2 and "workflow_cancel" not in cli.client.methods()
    assert "cancel every queued/running job" in r.stderr
    r = cli.run("workflow", "cancel", WF, "--all", "--yes")
    assert r.exit_code == 0, r.text
    p = cli.client.params("workflow_cancel")
    assert p["scope"] == "all_managed"
    assert p["expected_version"] == env("workflow_get.running")["data"]["version"]
    r = cli.run("workflow", "cancel", WF, "--yes")
    assert cli.client.params("workflow_cancel")["scope"] == "queued_only"


def test_workflow_cancel_retries_once_on_version_conflict(cli: Cli) -> None:
    cli.client.responses["workflow_get"] = env("workflow_get.running")
    cli.client.responses["workflow_cancel"] = [
        err("version_conflict", "stale", current_version=17),
        env("workflow_cancel"),
    ]
    r = cli.run("workflow", "cancel", WF, "--yes")
    assert r.exit_code == 0, r.text
    versions = [p["expected_version"] for m, p in cli.client.calls if m == "workflow_cancel"]
    assert versions[-1] == 17 and len(versions) == 2
    keys = {p["idempotency_key"] for m, p in cli.client.calls if m == "workflow_cancel"}
    assert len(keys) == 2  # a different request gets a different key


def test_workflow_resume_sends_only_resume_fields(cli: Cli) -> None:
    wf = env("workflow_get.running")
    wf["data"]["dispatch_state"] = "paused"
    cli.client.responses["workflow_get"] = wf
    resumed = env("workflow_submit")
    resumed["data"]["resumed"] = True
    cli.client.responses["workflow_submit"] = resumed
    r = cli.run("workflow", "resume", WF)
    assert r.exit_code == 0, r.text
    p = cli.client.params("workflow_submit")
    assert set(p) == {"workflow_id", "resume_dispatch", "expected_version", "idempotency_key"}
    assert p["resume_dispatch"] is True and p["expected_version"] == wf["data"]["version"]
    assert "dispatch re-opened" in r.stdout
    cli.client.responses["workflow_submit"] = err("precondition_failed", "dispatch is not paused")
    assert cli.run("workflow", "resume", WF).exit_code == 1


def _params_without_keys(p: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in p.items() if k != "idempotency_key"}
