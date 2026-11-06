"""hpclab auth: open the SSH master, then wake the steps that waited for it (auth_resume)."""

from __future__ import annotations

from typing import Any

import pytest
from cli_support import Cli, ok

from hpclab.cli import main as cli_main


def _auth_command(alive: bool) -> dict[str, Any]:
    return ok(
        {
            "cluster_id": "hai",
            "needed": not alive,
            "argv": ["ssh", "-M", "hai"],
            "command": "ssh -M hai",
            "master_alive": alive,
            "explanation": "…",
        }
    )


def _resumed(n: int, note: str | None = None) -> dict[str, Any]:
    return ok({"woken": n, "clusters": ["hai"], "note": note})


@pytest.fixture
def ssh(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_call(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(cli_main.subprocess, "call", fake_call)
    return calls


def test_auth_when_already_connected_resumes_waiting_steps(cli: Cli, ssh: list[list[str]]) -> None:
    cli.client.responses["auth_command"] = _auth_command(True)
    cli.client.responses["auth_resume"] = _resumed(2)
    r = cli.run("auth", "hai")
    assert r.exit_code == 0, r.text
    assert "already connected" in r.stdout and "resumed 2 waiting step(s)" in r.stdout
    assert cli.client.params("auth_resume") == {"cluster_id": "hai"}
    assert ssh == []


def test_auth_after_a_successful_login_resumes_waiting_steps(cli: Cli, ssh: list[list[str]]) -> None:
    cli.client.responses["auth_command"] = _auth_command(False)
    cli.client.responses["auth_resume"] = _resumed(0, "nothing was waiting")
    r = cli.run("auth", "hai")
    assert r.exit_code == 0, r.text
    assert ssh == [["ssh", "-M", "hai"]]
    assert "connected" in r.stdout and "resumed 0 waiting step(s)" in r.stdout
    assert "nothing was waiting" in r.stdout
    assert cli.client.methods() == ["auth_command", "auth_resume"]


def test_auth_failed_login_resumes_nothing(cli: Cli, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main.subprocess, "call", lambda argv: 255)
    cli.client.responses["auth_command"] = _auth_command(False)
    r = cli.run("auth", "hai")
    assert r.exit_code == 1 and "ssh exited with 255" in r.stderr
    assert "auth_resume" not in cli.client.methods()


def test_auth_print_only_does_not_resume(cli: Cli, ssh: list[list[str]]) -> None:
    cli.client.responses["auth_command"] = _auth_command(False)
    r = cli.run("auth", "hai", "--print")
    assert r.exit_code == 0 and "ssh -M hai" in r.stdout
    assert cli.client.methods() == ["auth_command"] and ssh == []


def test_auth_with_a_controller_without_auth_resume_prints_a_hint(cli: Cli, ssh: list[list[str]]) -> None:
    """An older controller does not know the method: the login still succeeded (exit 0)."""
    cli.client.responses["auth_command"] = _auth_command(True)  # auth_resume is not scripted
    r = cli.run("auth", "hai")
    assert r.exit_code == 0, r.text
    assert "retried automatically" in r.text and "controller restart" in r.text
    assert "unknown method" not in r.text


def test_auth_json_keeps_stdout_one_json_document(cli: Cli, ssh: list[list[str]]) -> None:
    """`hpclab --json auth`: scripts parse stdout; the human lines (connected, resumed ...) go to
    stderr, so stdout stays the auth_command envelope alone."""
    cli.client.responses["auth_command"] = _auth_command(True)
    cli.client.responses["auth_resume"] = _resumed(2, "resumed cluster hai: 2 background item(s)")
    r = cli.run("--json", "auth", "hai")
    assert r.exit_code == 0, r.text
    assert r.json["data"]["master_alive"] is True  # parses: nothing else on stdout
    assert "already connected" in r.stderr and "resumed 2 waiting step(s)" in r.stderr

    cli.client.responses["auth_command"] = _auth_command(False)
    r = cli.run("--json", "auth", "hai")
    assert r.exit_code == 0, r.text
    assert r.json["data"]["master_alive"] is False
    assert "connected" in r.stderr and "running: ssh -M hai" in r.stderr

    r = cli.run("--json", "auth", "hai", "--print")
    assert r.exit_code == 0 and r.json["data"]["command"] == "ssh -M hai"
