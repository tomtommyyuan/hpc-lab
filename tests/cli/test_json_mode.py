"""--json prints exactly the Envelope of the command's main call (one JSON document on stdout)."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli_support import Cli, env

EXP = "exp_01M3K9Q1CC33H1R96NRPJEDAVW"
WF = "wf_01M3K9Q1MH6X7FJ7NA1J04TDXA"
PENDING_ATT = "att_01M3K9QS0T356550MDEFTXEAKN"
FAILED_ATT = "att_01M3K9QD2FYB3TBD0RR59FXFBT"
GOOD_ATT = "att_01M3K9Q7QY2C92358BRYJ7C2R7"
BUNDLE = "evb_01M3K9QS049T7A49DZQY7J7WJF"

# (argv after --json, {method: captured envelope name}, method whose envelope is printed)
CASES: list[tuple[tuple[str, ...], dict[str, str], str]] = [
    (("experiment", EXP), {"experiment_get": "experiment_get"}, "experiment_get"),
    (
        ("workflow", "cancel", WF, "--yes"),
        {"workflow_get": "workflow_get.running", "workflow_cancel": "workflow_cancel"},
        "workflow_cancel",
    ),
    (
        ("workflow", "resume", WF),
        {"workflow_get": "workflow_get.running", "workflow_submit": "workflow_submit"},
        "workflow_submit",
    ),
    (
        ("cancel", PENDING_ATT, "--yes"),
        {"run_inspect": "run_inspect.pending", "run_cancel": "run_cancel"},
        "run_cancel",
    ),
    (("recover", FAILED_ATT), {"run_recover": "run_recover.eligible"}, "run_recover"),
    (
        ("resolve", "att_01M3K9QT35YK3BRDF4M74CSDVT", "--abandon", "--reason", "gone", "--yes"),
        {"run_inspect": "run_inspect.uncertain", "attempt_resolve": "attempt_resolve"},
        "attempt_resolve",
    ),
    (("which", "1002", "--cluster", "sim"), {"run_inspect": "run_inspect.by_job"}, "run_inspect"),
    (("trace", f"{BUNDLE}:mixup.acc.mean"), {"trace": "trace"}, "trace"),
    (
        ("read", "art_01M3K9Q7QZ0000000000000000", "--view", "json"),
        {"artifact_read": "artifact_read.json"},
        "artifact_read",
    ),
    (("collect", GOOD_ATT), {"evaluation_collect": "evaluation_collect.single"}, "evaluation_collect"),
    (("verify", BUNDLE), {"evidence_verify": "evidence_verify.clean"}, "evidence_verify"),
    (("usage", "--cluster", "sim"), {"usage_report": "usage_report.self"}, "usage_report"),
    (("vault",), {"vault_status": "vault_status"}, "vault_status"),
    (("notify", "test"), {"notify_test": "notify_test"}, "notify_test"),
    (("events", "--limit", "4"), {"events_list": "events_list"}, "events_list"),
    (("workflow", "show", WF), {"workflow_get": "workflow_get.done"}, "workflow_get"),
    (("artifacts", GOOD_ATT), {"artifacts_list": "artifacts_list"}, "artifacts_list"),
    (("ops", "op_01M3K9Q1MHKZ8HWQR03AZ2GP5A"), {"operation_get": "operation_get.succeeded"}, "operation_get"),
]


@pytest.mark.parametrize(("argv", "responses", "main"), CASES, ids=[" ".join(c[0][:2]) for c in CASES])
def test_json_prints_the_main_envelope(
    cli: Cli, argv: tuple[str, ...], responses: dict[str, str], main: str
) -> None:
    for method, name in responses.items():
        cli.client.responses[method] = env(name)
    r = cli.run("--json", *argv)
    assert r.json == env(responses[main]), r.stderr
    assert r.exit_code in (0, 1)  # 1 only where the command's verdict is "not ok" (e.g. not eligible)


def test_json_backup_and_sim_reset(cli: Cli, tmp_path: Path, hpclab_home: Path) -> None:
    cli.client.responses["backup"] = env("backup")
    r = cli.run("--json", "backup", str(tmp_path / "b"))
    assert r.json == env("backup")
    from hpclab.sim.cluster import SimCluster

    SimCluster.ensure(hpclab_home / "sim" / "sim")
    cli.client.responses["init_sim"] = env("init_sim")
    r = cli.run("--json", "sim", "reset")
    assert r.exit_code == 2  # destructive: --json never implies --yes
    assert r.json["ok"] is False and r.json["error"]["code"] == "invalid_argument"
    assert "--yes" in r.json["error"]["message"]
    r = cli.run("--json", "sim", "reset", "--yes")
    assert r.json["ok"] is True and r.json["cluster_id"] == "sim"


def test_json_errors_are_envelopes_with_exit_codes(cli: Cli) -> None:
    from cli_support import err

    cli.client.responses["trace"] = err("not_found", "no such cell")
    r = cli.run("--json", "trace", f"{BUNDLE}:nope.acc.mean")
    assert r.exit_code == 1
    assert r.json["ok"] is False and r.json["error"]["code"] == "not_found"
