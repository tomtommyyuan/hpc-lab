"""OpenSSHTransport command execution: quoting, ssh options, control master, failure classes."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from openssh_support import FakeSsh, MakeSsh

from hpclab.adapters import openssh
from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    ConnectionLost,
    Transport,
    TransportError,
    TransportTimeout,
)
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.openssh import OpenSSHTransport
from hpclab.config import Paths
from hpclab.domain.errors import ErrorCode, HpcLabError
from hpclab.profiles.schema import ClusterProfile, ResolvedCluster, UserOverlay
from hpclab.services.context import default_transport_factory

NASTY_ARGS = [
    "plain",
    "two words",
    "it's",
    'double "quotes"',
    "$(touch {sentinel})",
    "`touch {sentinel}`",
    "; touch {sentinel}",
    "&& touch {sentinel} ||",
    "line1\nline2",
    "tab\there",
    "ünïcødé 日本語 🚀",
    "*",
    "~",
    "$HOME",
    "\\",
    "",
    "-n",
    "--",
    "a|b>c<d",
]

ECHO_ARGV = [sys.executable, "-c", "import json, sys; sys.stdout.write(json.dumps(sys.argv[1:]))"]


def _pairs(argv: list[str]) -> list[tuple[str, str]]:
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 1)]


class TestWiring:
    def test_satisfies_the_transport_protocol(self, ssh: OpenSSHTransport) -> None:
        assert isinstance(ssh, Transport)

    def test_default_factory_builds_it_from_profile_and_overlay(
        self, tmp_path: Path, short_control_root: Path
    ) -> None:
        profile = ClusterProfile(
            cluster_id="cluster1", display_name="HAIC", transport="ssh", login_host="cluster1.example.edu"
        )
        overlay = UserOverlay(cluster_id="cluster1", user="alice", ssh_options=["ProxyJump=bastion"])

        transport = default_transport_factory(
            ResolvedCluster(profile=profile, overlay=overlay), Paths(tmp_path / "home")
        )

        assert isinstance(transport, OpenSSHTransport)
        command = transport.login_command()
        assert command[-2:] == ["cluster1.example.edu", "true"]
        assert ("-l", "alice") in _pairs(command) and ("-o", "ProxyJump=bastion") in _pairs(command)


class TestRun:
    def test_arguments_arrive_verbatim_and_are_never_interpreted(
        self, ssh: OpenSSHTransport, tmp_path: Path
    ) -> None:
        sentinel = tmp_path / "pwned"
        args = [a.replace("{sentinel}", str(sentinel)) for a in NASTY_ARGS]

        result = ssh.run([*ECHO_ARGV, *args])

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == args
        assert not sentinel.exists()

    def test_quoting_holds_under_every_posix_login_shell(
        self, ssh: OpenSSHTransport, login_shell: str, tmp_path: Path
    ) -> None:
        # sshd hands the joined command to the user's login shell: bash, zsh or dash/sh.
        sentinel = tmp_path / "pwned"
        args = [a.replace("{sentinel}", str(sentinel)) for a in NASTY_ARGS]

        result = ssh.run([*ECHO_ARGV, *args])
        script = ssh.run_script('printf "%s|" "$#" "$1" "$2"', args=["a b", "$(id)"])

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == args
        assert script.stdout == "2|a b|$(id)|"
        assert not sentinel.exists()

    def test_nonzero_exit_is_returned_not_raised(self, ssh: OpenSSHTransport) -> None:
        argv = ["sh", "-c", "echo out; echo err >&2; exit 3"]

        result = ssh.run(argv)

        assert (result.exit_code, result.stdout, result.stderr) == (3, "out\n", "err\n")
        assert result.argv == argv
        assert result.duration_s > 0

    def test_stdin_bytes_are_forwarded(self, ssh: OpenSSHTransport) -> None:
        payload = "line one\nünïcødé\n".encode()
        assert ssh.run(["cat"], stdin=payload).stdout.encode() == payload

    def test_stdin_is_never_inherited(self, ssh: OpenSSHTransport) -> None:
        # With stdin=None the remote side must see EOF at once (the MCP channel is our stdin).
        result = ssh.run(["cat"], timeout=10)
        assert (result.exit_code, result.stdout) == (0, "")

    def test_run_script_feeds_script_on_stdin_with_args(self, ssh: OpenSSHTransport) -> None:
        result = ssh.run_script('printf "%s|" "$#" "$1" "$2"', args=["a b", "$HOME"])
        assert result.stdout == "2|a b|$HOME|"

    def test_dash_arguments_cannot_become_ssh_options(self, ssh: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        result = ssh.run(["-oProxyCommand=touch /tmp/x"])

        assert result.exit_code == 127  # the remote shell tried to run it as a program
        argv = fake_ssh.calls()[-1]["argv"]
        assert argv[-3:] == ["--", "login.example.org", "'-oProxyCommand=touch /tmp/x'"]

    def test_path_arguments_are_accepted_like_local_transport(
        self, ssh: OpenSSHTransport, local: LocalTransport, tmp_path: Path
    ) -> None:
        target = tmp_path / "dir with space"
        target.mkdir()
        argv: list[object] = ["ls", "-d", target]

        remote = ssh.run(argv)  # type: ignore[arg-type]

        assert remote.exit_code == 0 and remote.stdout == f"{target}\n"
        assert remote.stdout == local.run(argv).stdout  # type: ignore[arg-type]

    @pytest.mark.parametrize("argv", [[], ["echo", "nul\x00byte"], ["echo", 3]])
    def test_invalid_argv_is_rejected_before_connecting(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, argv: list[object]
    ) -> None:
        with pytest.raises(HpcLabError) as exc:
            ssh.run(argv)  # type: ignore[arg-type]
        assert exc.value.code == ErrorCode.INVALID_ARGUMENT
        assert fake_ssh.calls() == []

    def test_remote_user_is_last_stdout_line_and_cached(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh
    ) -> None:
        fake_ssh.set(noise="Welcome to the cluster!\n")

        assert ssh.remote_user() == getpass.getuser()
        assert ssh.remote_user() == getpass.getuser()
        assert len(fake_ssh.calls()) == 1

    def test_remote_user_failure_is_a_command_failure_and_not_cached(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty-bin"
        empty.mkdir()
        fake_ssh.set(remote_path=str(empty))  # no `id` on the remote PATH

        with pytest.raises(CommandFailed) as exc:
            ssh.remote_user()
        assert exc.value.result.exit_code == 127

        fake_ssh.clear("remote_path")
        assert ssh.remote_user() == getpass.getuser()


class TestSshOptions:
    def test_batch_invocation_carries_the_mandatory_options(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh, tmp_path: Path
    ) -> None:
        transport = make_ssh(user="alice", connect_timeout=7, ssh_options=["ProxyJump=bastion"])

        transport.run(["true"])

        argv = fake_ssh.calls()[-1]["argv"]
        pairs = _pairs(argv)
        for option in (
            "BatchMode=yes",
            "ControlMaster=auto",
            f"ControlPath={transport.control_path}",
            "ControlPersist=30m",
            "ConnectTimeout=7",
            "ServerAliveInterval=15",
            "ServerAliveCountMax=3",
            "ProxyJump=bastion",
        ):
            assert ("-o", option) in pairs, option
        assert ("-l", "alice") in pairs
        assert {"-T", "-a", "-x"} <= set(argv)
        # hpclab's options come first: ssh keeps the first value it sees for each option.
        assert argv.index("ControlPersist=30m") < argv.index("ProxyJump=bastion")
        assert argv[-3:] == ["--", "login.example.org", "true"]
        assert transport.control_path.endswith("/ssh-%C")

    def test_login_command_is_interactive_but_shares_the_control_path(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh
    ) -> None:
        transport = make_ssh(user="alice")

        command = transport.login_command()

        assert "BatchMode=yes" not in command and "-T" not in command
        assert ("-o", f"ControlPath={transport.control_path}") in _pairs(command)
        assert ("-o", "ControlMaster=auto") in _pairs(command)
        assert ("-o", "ControlPersist=30m") in _pairs(command)
        assert command[-3:] == ["--", "login.example.org", "true"]
        control_dir = Path(transport.control_path).parent
        assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700
        assert fake_ssh.calls() == []  # building the command never connects

    def test_user_options_are_normalized_to_o_pairs(self, make_ssh: MakeSsh, fake_ssh: FakeSsh) -> None:
        transport = make_ssh(
            ssh_options=["ProxyJump=bastion", "-oCompression=yes", "-o Port=2222", "User alice"]
        )

        transport.run(["true"])

        pairs = _pairs(fake_ssh.calls()[-1]["argv"])
        for option in ("ProxyJump=bastion", "Compression=yes", "Port=2222", "User alice"):
            assert ("-o", option) in pairs

    def test_log_level_is_pinned_so_failures_stay_classifiable(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh
    ) -> None:
        # "LogLevel QUIET" in ~/.ssh/config (to hide cluster banners) would silence "Permission
        # denied": an expired MFA session would then look like a network drop. The command
        # line beats the config file.
        transport = make_ssh(ssh_options=["ProxyJump=bastion"])

        transport.run(["true"])

        for argv in (fake_ssh.calls()[-1]["argv"], transport.login_command()):
            assert ("-o", "LogLevel=INFO") in _pairs(argv)
            assert argv.index("ProxyJump=bastion") < argv.index("LogLevel=INFO")

    def test_explicit_log_level_in_overlay_options_wins(self, make_ssh: MakeSsh, fake_ssh: FakeSsh) -> None:
        transport = make_ssh(ssh_options=["LogLevel=DEBUG3"])

        transport.run(["true"])

        argv = fake_ssh.calls()[-1]["argv"]
        assert ("-o", "LogLevel=DEBUG3") in _pairs(argv) and "LogLevel=INFO" not in argv

    def test_o_flag_and_value_as_separate_items(self, make_ssh: MakeSsh, fake_ssh: FakeSsh) -> None:
        # The natural YAML spelling of "ssh -o ProxyJump=bastion": ["-o", "ProxyJump=bastion"].
        transport = make_ssh(ssh_options=["-o", "ProxyJump=bastion", "-o", "Port 2222"])

        transport.run(["true"])

        pairs = _pairs(fake_ssh.calls()[-1]["argv"])
        assert ("-o", "ProxyJump=bastion") in pairs and ("-o", "Port 2222") in pairs

    @pytest.mark.parametrize("options", [["-o"], ["ProxyJump=bastion", "-o"], ["-o", "-o"]])
    def test_dangling_o_flag_is_rejected(self, make_ssh: MakeSsh, options: list[str]) -> None:
        with pytest.raises(HpcLabError) as exc:
            make_ssh(ssh_options=options)
        assert exc.value.code == ErrorCode.INVALID_ARGUMENT

    @pytest.mark.parametrize(
        "option",
        [
            "BatchMode=no",
            "controlpath=/tmp/x",
            "ConnectTimeout=1",
            "-J bastion",
            "-i key",
            "NoValue",
            "Bad=a\nb",
        ],
    )
    def test_invalid_or_managed_options_are_rejected(self, make_ssh: MakeSsh, option: str) -> None:
        with pytest.raises(HpcLabError) as exc:
            make_ssh(ssh_options=[option])
        assert exc.value.code == ErrorCode.INVALID_ARGUMENT

    @pytest.mark.parametrize(
        ("field", "value"), [("host", "-oProxyCommand=x"), ("host", "a b"), ("host", ""), ("user", "-l")]
    )
    def test_host_and_user_cannot_inject_options(self, make_ssh: MakeSsh, field: str, value: str) -> None:
        with pytest.raises(HpcLabError) as exc:
            make_ssh(**{field: value})
        assert exc.value.code == ErrorCode.INVALID_ARGUMENT

    @pytest.mark.parametrize("timeout", [0, -1, True, 2.5])
    def test_connect_timeout_must_be_a_positive_int(self, make_ssh: MakeSsh, timeout: object) -> None:
        with pytest.raises(HpcLabError):
            make_ssh(connect_timeout=timeout)


class TestControlDirectory:
    def test_short_safe_directory_is_used_as_is(self) -> None:
        assert openssh._resolve_control_dir(Path("/home/a/.hpclab/run")) == Path("/home/a/.hpclab/run")

    @pytest.mark.parametrize("path", ["/" + "x" * 80, "/home/a b/.hpclab/run", "/home/100%/run"])
    def test_long_or_special_directory_falls_back_to_a_private_short_one(
        self, short_control_root: Path, path: str
    ) -> None:
        resolved = openssh._resolve_control_dir(Path(path))
        assert resolved.parent == short_control_root
        assert resolved.name.startswith(f"hpclab-{os.getuid()}-")
        assert openssh._resolve_control_dir(Path(path)) == resolved  # deterministic

    def test_fallback_directory_is_created_private(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        transport = make_ssh(control_dir=tmp_path / ("long" * 30))

        transport.run(["true"])

        control_dir = Path(transport.control_path).parent
        assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700

    def test_loose_permissions_are_tightened(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        transport = make_ssh(control_dir=tmp_path / ("long" * 30))
        control_dir = Path(transport.control_path).parent
        control_dir.mkdir(parents=True, mode=0o755)
        control_dir.chmod(0o755)

        transport.run(["true"])

        assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700

    def test_control_directory_is_recreated_if_it_disappears(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        # e.g. the fallback dir under /tmp removed by the OS tmp cleaner while the controller
        # runs: without it ssh silently stops multiplexing and login_command() cannot help.
        transport = make_ssh(control_dir=tmp_path / ("long" * 30))
        transport.run(["true"])
        control_dir = Path(transport.control_path).parent
        shutil.rmtree(control_dir)

        command = transport.login_command()

        assert control_dir.is_dir() and stat.S_IMODE(control_dir.stat().st_mode) == 0o700
        assert ("-o", f"ControlPath={transport.control_path}") in _pairs(command)
        shutil.rmtree(control_dir)
        transport.run(["true"])
        assert control_dir.is_dir()

    def test_uncreatable_control_directory_is_a_typed_precondition(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh, short_control_root: Path, tmp_path: Path
    ) -> None:
        short_control_root.write_text("a file where the control directory's parent should be")
        transport = make_ssh(control_dir=tmp_path / ("long" * 30))

        for call in (lambda: transport.run(["true"]), transport.login_command):
            with pytest.raises(TransportError) as exc:
                call()
            assert exc.value.code == ErrorCode.PRECONDITION_FAILED and not exc.value.retryable
        assert fake_ssh.calls() == []

    def test_symlinked_control_directory_is_refused(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        transport = make_ssh(control_dir=tmp_path / ("long" * 30))
        control_dir = Path(transport.control_path).parent
        control_dir.parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "elsewhere").mkdir()
        control_dir.symlink_to(tmp_path / "elsewhere")

        with pytest.raises(TransportError) as exc:
            transport.run(["true"])
        assert exc.value.code == ErrorCode.PRECONDITION_FAILED


class TestControlMaster:
    def test_check_master_reports_state(self, ssh: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        assert ssh.check_master() is False
        fake_ssh.set(master="1")
        assert ssh.check_master() is True
        argv = fake_ssh.calls()[-1]["argv"]
        assert argv[:2] == ["-O", "check"]
        assert ("-o", f"ControlPath={ssh.control_path}") in _pairs(argv)

    def test_stop_master(self, ssh: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        assert ssh.stop_master() is False
        fake_ssh.set(master="1")
        assert ssh.stop_master() is True
        assert fake_ssh.calls()[-1]["argv"][:2] == ["-O", "exit"]

    def test_check_master_never_raises(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        assert make_ssh(ssh_bin=str(tmp_path / "no-such-ssh")).check_master() is False

    def test_control_commands_do_not_wait_for_a_session_slot(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "-O check" talks to the local master socket and opens no remote session, so a long
        # upload holding every slot must not make the master look dead.
        monkeypatch.setattr(openssh, "MAX_CONCURRENT_SESSIONS", 1)
        transport = make_ssh()
        fake_ssh.set(sleep="1.5")
        busy = threading.Thread(target=transport.run, args=(["true"],))
        busy.start()
        time.sleep(0.3)
        try:
            fake_ssh.clear("sleep")
            fake_ssh.set(master="1")
            assert transport.check_master(timeout=0.8) is True
        finally:
            busy.join()

    def test_close_leaves_the_master_running(self, ssh: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        ssh.close()
        assert fake_ssh.calls() == []


AUTH_STDERR = [
    "alice@login.example.org: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,keyboard-interactive).",
    "Received disconnect from 10.0.0.1 port 22:2: Too many authentication failures",
    "Duo two-factor login for alice",
    "Control socket connect(/x/ssh-abc): No such file or directory",
    "alice@login.example.org: Permission denied (keyboard-interactive).",
]
PRECONNECT_STDERR = [
    "ssh: connect to host login.example.org port 22: Connection refused",
    "ssh: connect to host login.example.org port 22: Operation timed out",
    "ssh: Could not resolve hostname login.example.org: Name or service not known",
    "kex_exchange_identification: Connection closed by remote host",
    "mux_client_request_session: session request failed: Session open refused by peer",
    "channel 0: open failed: connect failed: No route to host\nstdio forwarding failed",
]
POSTCONNECT_STDERR = [
    # OpenSSH >= 9 prints this for a server-side close at any stage, including mid-command.
    "Connection closed by 10.0.0.1 port 22",
    "client_loop: send disconnect: Broken pipe",
    "Timeout, server login.example.org not responding.",
    "Connection to login.example.org closed by remote host.",
    "",
]
CONFIG_STDERR = [
    "command-line line 0: Bad configuration option: bogus",
    "Bad owner or permissions on /home/alice/.ssh/config",
    "Unable to negotiate with 10.0.0.1 port 22: no matching host key type found",
]


class TestFailureClassification:
    @pytest.mark.parametrize("stderr", AUTH_STDERR)
    def test_auth_failures_raise_auth_required_with_login_command(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, stderr: str
    ) -> None:
        fake_ssh.set(fail_stderr=stderr)

        with pytest.raises(AuthRequiredError) as exc:
            ssh.run(["sbatch", "job.sh"])

        err = exc.value
        assert err.code == ErrorCode.AUTH_REQUIRED and err.retryable
        assert err.maybe_executed is False
        assert err.details["login_command"] == ssh.login_command()
        assert err.details["reason"] == "auth"
        assert "Duo" in err.details["hint"] and "ssh" in err.details["hint"]
        assert err.details["login_command_str"] in err.details["hint"]

    def test_host_key_problems_need_the_user(self, ssh: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        fake_ssh.set(
            fail_stderr="@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\nHost key verification failed."
        )

        with pytest.raises(AuthRequiredError) as exc:
            ssh.run(["true"])
        assert exc.value.details["reason"] == "host_key"
        assert "admins" in exc.value.details["hint"]

    @pytest.mark.parametrize("stderr", PRECONNECT_STDERR)
    def test_connect_failures_did_not_execute(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, stderr: str
    ) -> None:
        fake_ssh.set(fail_stderr=stderr)

        with pytest.raises(ConnectionLost) as exc:
            ssh.run(["sbatch", "job.sh"])
        assert exc.value.maybe_executed is False
        assert exc.value.retryable and exc.value.code == ErrorCode.UNAVAILABLE
        assert exc.value.details["maybe_executed"] is False

    @pytest.mark.parametrize("stderr", POSTCONNECT_STDERR)
    def test_mid_session_failures_may_have_executed(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, stderr: str
    ) -> None:
        fake_ssh.set(fail_stderr=stderr)

        with pytest.raises(ConnectionLost) as exc:
            ssh.run(["sbatch", "job.sh"])
        assert exc.value.maybe_executed is True

    @pytest.mark.parametrize("stderr", CONFIG_STDERR)
    def test_config_errors_are_not_retryable(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, stderr: str
    ) -> None:
        fake_ssh.set(fail_stderr=stderr)

        with pytest.raises(TransportError) as exc:
            ssh.run(["true"])
        assert not isinstance(exc.value, AuthRequiredError | ConnectionLost)
        assert exc.value.code == ErrorCode.PRECONDITION_FAILED and not exc.value.retryable
        assert "/home/alice/.ssh" not in json.dumps(exc.value.to_dict())

    def test_drop_after_the_command_ran_is_maybe_executed_even_with_auth_words(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, tmp_path: Path
    ) -> None:
        marker = tmp_path / "ran"
        fake_ssh.set(drop_after="Permission denied (publickey).")

        with pytest.raises(ConnectionLost) as exc:
            ssh.run(["sh", "-c", f"touch {marker}; echo submitted"])
        assert exc.value.maybe_executed is True
        assert marker.exists()

    def test_remote_exit_255_is_treated_as_maybe_executed(self, ssh: OpenSSHTransport) -> None:
        with pytest.raises(ConnectionLost) as exc:
            ssh.run(["sh", "-c", "exit 255"])
        assert exc.value.maybe_executed is True

    def test_key_paths_and_sensitive_options_never_leak(self, make_ssh: MakeSsh, fake_ssh: FakeSsh) -> None:
        transport = make_ssh(ssh_options=["IdentityFile=/home/alice/keys/hpc_key"])
        fake_ssh.set(
            fail_stderr='Load key "/home/alice/.ssh/id_ed25519": bad permissions\nPermission denied (publickey).'
        )

        with pytest.raises(AuthRequiredError) as exc:
            transport.run(["true"])

        dumped = json.dumps(exc.value.to_dict()) + str(exc.value)
        for secret in ("/home/alice/.ssh", "id_ed25519", "/home/alice/keys/hpc_key"):
            assert secret not in dumped
        assert "IdentityFile=<redacted>" in exc.value.details["login_command"]
        # The unredacted command stays available to the local CLI that runs it for the user.
        assert "IdentityFile=/home/alice/keys/hpc_key" in transport.login_command()

    def test_ssh_client_warnings_in_successful_output_are_redacted(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh
    ) -> None:
        # A fresh connection prints key warnings; callers copy stderr into error details.
        fake_ssh.set(warn='Load key "/Users/alice/.ssh/id_old": invalid format')

        result = ssh.run(["sh", "-c", "echo 'sbatch: error: invalid partition' >&2; exit 1"])

        assert result.exit_code == 1
        assert "/Users/alice/.ssh" not in result.stderr and "id_old" not in result.stderr
        assert "sbatch: error: invalid partition" in result.stderr

    def test_timeout_raises_transport_timeout_promptly(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh
    ) -> None:
        fake_ssh.set(sleep="5")
        start = time.monotonic()

        with pytest.raises(TransportTimeout) as exc:
            ssh.run(["true"], timeout=0.5)

        assert time.monotonic() - start < 4
        assert exc.value.maybe_executed is True and exc.value.retryable
        assert exc.value.details["timeout_s"] == 0.5
        assert exc.value.details["maybe_executed"] is True

    def test_missing_ssh_client_is_unsupported(self, make_ssh: MakeSsh, tmp_path: Path) -> None:
        with pytest.raises(TransportError) as exc:
            make_ssh(ssh_bin=str(tmp_path / "no-such-ssh")).run(["true"])
        assert exc.value.code == ErrorCode.UNSUPPORTED and not exc.value.retryable

    def test_session_limit_times_out_without_executing(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(openssh, "MAX_CONCURRENT_SESSIONS", 1)
        transport = make_ssh()
        fake_ssh.set(sleep="1.5")
        busy = threading.Thread(target=transport.run, args=(["true"],))
        busy.start()
        time.sleep(0.3)
        try:
            with pytest.raises(TransportTimeout) as exc:
                transport.run(["touch", str(tmp_path / "should-not-exist")], timeout=0.3)
            assert exc.value.maybe_executed is False
            assert exc.value.details["maybe_executed"] is False
        finally:
            busy.join()
        assert not (tmp_path / "should-not-exist").exists()

    def test_waiting_for_a_session_counts_against_the_timeout(
        self, make_ssh: MakeSsh, fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openssh, "MAX_CONCURRENT_SESSIONS", 1)
        transport = make_ssh()
        fake_ssh.set(sleep="0.8")
        busy = threading.Thread(target=transport.run, args=(["true"],))
        busy.start()
        time.sleep(0.3)
        start = time.monotonic()
        try:
            # ~0.5 s waiting for the slot + >= 0.8 s running exceeds the 1.0 s budget.
            with pytest.raises(TransportTimeout) as exc:
                transport.run(["true"], timeout=1.0)
            assert exc.value.details["timeout_s"] == 1.0
            assert time.monotonic() - start < 1.8
        finally:
            busy.join()


@pytest.fixture
def real_ssh_control_dir() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="hlt-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skipif(shutil.which("ssh") is None, reason="OpenSSH client not installed")
class TestRealOpenSSH:
    """Against the real client: the options we pass are accepted and its errors classified."""

    def test_connection_refused_is_classified_as_not_executed(self, real_ssh_control_dir: Path) -> None:
        transport = OpenSSHTransport(
            "real", "127.0.0.1", ssh_options=["Port=1"], control_dir=real_ssh_control_dir, connect_timeout=3
        )

        with pytest.raises(TransportError) as exc:
            transport.run(["true"], timeout=20)

        assert isinstance(exc.value, ConnectionLost), exc.value.details
        assert exc.value.maybe_executed is False

    def test_config_warnings_do_not_turn_a_refusal_into_a_config_error(
        self, real_ssh_control_dir: Path
    ) -> None:
        # ssh warns 'Unsupported option "rsaauthentication"' (as it does for stale lines in
        # ~/.ssh/config) and carries on; the failure is still the connection refusal.
        transport = OpenSSHTransport(
            "real",
            "127.0.0.1",
            ssh_options=["RSAAuthentication=no", "Port=1"],
            control_dir=real_ssh_control_dir,
            connect_timeout=3,
        )

        with pytest.raises(TransportError) as exc:
            transport.run(["true"], timeout=20)

        assert isinstance(exc.value, ConnectionLost), exc.value.details
        assert exc.value.maybe_executed is False and exc.value.retryable

    def test_check_master_without_master_is_false(self, real_ssh_control_dir: Path) -> None:
        transport = OpenSSHTransport("real", "127.0.0.1", control_dir=real_ssh_control_dir)
        assert transport.check_master(timeout=10) is False
