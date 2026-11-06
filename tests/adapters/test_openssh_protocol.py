"""Unit tests for the remote-helper wire protocol and the ssh failure classifier.

These run the helper directly with local subprocesses (no ssh), so they are fast and can
inject faults — truncated uploads, garbage replies, old interpreters — at the byte level.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest

from hpclab.adapters.base import ConnectionLost
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.openssh import SshFailure, classify_ssh_failure, redact
from hpclab.adapters.remote_helper import (
    MARKER,
    REMOTE_HELPER,
    RawResult,
    RemoteFileOps,
    RemoteHelperError,
    _mtime,
    pathlib_form,
    split_after_marker,
)


def run_locally(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
    """A RemoteExecutor that behaves like a login shell on this machine (127 = not found)."""
    try:
        proc = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return RawResult(127, b"", f"bash: {argv[0]}: command not found\n".encode())
    return RawResult(proc.returncode, proc.stdout, proc.stderr)


def canned(stdout: bytes, stderr: bytes = b"", exit_code: int = 0) -> Callable[..., RawResult]:
    def execute(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
        return RawResult(exit_code, stdout, stderr)

    return execute


def available_interpreters() -> list[str]:
    """Every python3.x on PATH (old ones matter most: login nodes run the system python)."""
    names = ["python3", *(f"python3.{minor}" for minor in range(6, 15))]
    found = {os.path.realpath(p) for p in map(shutil.which, names) if p}
    return sorted(found) or [sys.executable]


def reply(payload: object) -> bytes:
    return MARKER + b"\n" + json.dumps(payload).encode() + b"\n"


class TestHelperSource:
    def test_parses_as_python_36(self) -> None:
        # RHEL 8 login nodes ship python3.6; the helper must not use newer syntax.
        ast.parse(REMOTE_HELPER, feature_version=(3, 6))

    def test_imports_only_stdlib(self) -> None:
        tree = ast.parse(REMOTE_HELPER)
        imported = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        assert imported <= {"base64", "errno", "hashlib", "json", "os", "sys", "tempfile"}
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]

    @pytest.mark.parametrize("python", available_interpreters())
    def test_runs_under_every_available_interpreter(self, python: str, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_bytes(b"abc")
        ops = RemoteFileOps(run_locally, python=python)
        local = LocalTransport("x")
        paths = [str(tmp_path / "f.txt"), str(tmp_path / "missing"), str(tmp_path)]

        assert ops.stat(paths, timeout=30) == local.stat(paths)
        assert ops.read_tail(paths[0], 2, timeout=30) == local.read_tail(paths[0], 2)
        assert ops.python_available is True


class TestFraming:
    def test_marker_splits_off_login_noise(self) -> None:
        assert split_after_marker(b"motd\n" + MARKER + b"\n{}\n") == b"{}\n"
        assert split_after_marker(b"no marker here") is None

    def test_payload_truncated_in_transit_is_never_written(self, tmp_path: Path) -> None:
        def truncating(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
            return run_locally(argv, stdin=(stdin or b"")[:-1], timeout=timeout)

        target = tmp_path / "out.bin"
        for python in (sys.executable, "hpclab-no-python"):
            with pytest.raises(ConnectionLost) as exc:
                RemoteFileOps(truncating, python=python).write_file(
                    str(target), b"0123456789", mode=0o644, timeout=10
                )
            assert exc.value.maybe_executed is False
            # to_dict() drops the attribute across the controller boundary; details keep it.
            assert exc.value.details["maybe_executed"] is False
            assert not target.exists()
            assert os.listdir(tmp_path) == []

    def test_no_reply_is_a_helper_error_with_stderr(self) -> None:
        ops = RemoteFileOps(canned(b"", b"Traceback: boom", exit_code=1), python="python3")

        with pytest.raises(RemoteHelperError) as exc:
            ops.stat(["/x"], timeout=5)

        assert exc.value.details["stderr"] == "Traceback: boom"

    @pytest.mark.parametrize(
        ("stderr", "exit_code"),
        [
            (b"bash: fork: retry: Resource temporarily unavailable", 254),
            (b"", 137),  # killed by the login node's per-user memory/CPU watchdog
        ],
    )
    def test_no_reply_is_retryable(self, stderr: bytes, exit_code: int) -> None:
        # Busy login nodes kill or refuse processes transiently; that is "unavailable", not a bug.
        with pytest.raises(RemoteHelperError) as exc:
            RemoteFileOps(canned(b"", stderr, exit_code=exit_code)).stat(["/x"], timeout=5)
        assert exc.value.retryable and exc.value.code == "unavailable"

    def test_malformed_reply_is_not_retryable(self) -> None:
        with pytest.raises(RemoteHelperError) as exc:
            RemoteFileOps(canned(MARKER + b"\nnot json\n")).stat(["/x"], timeout=5)
        assert not exc.value.retryable

    @pytest.mark.parametrize(
        "stdout",
        [
            MARKER + b"\nnot json\n",
            reply([1, 2]),  # a list, not a reply object
            reply({"ok": True, "result": [{"kind": "file"}]}),  # missing fields
            reply({"ok": True, "result": [None, None]}),  # wrong record count
            reply({"ok": False, "error": {"kind": "internal", "message": "KeyError: x"}}),
        ],
    )
    def test_malformed_or_failed_replies_are_helper_errors(self, stdout: bytes) -> None:
        with pytest.raises(RemoteHelperError):
            RemoteFileOps(canned(stdout)).stat(["/x"], timeout=5)

    def test_unknown_remote_errno_becomes_a_plain_oserror(self) -> None:
        stdout = reply({"ok": False, "error": {"kind": "os", "errno": "EWEIRD", "message": "odd failure"}})

        with pytest.raises(OSError) as exc:
            RemoteFileOps(canned(stdout)).read_range("/x", 0, 1, timeout=5)

        assert type(exc.value) is OSError and "odd failure" in str(exc.value)

    def test_remote_errno_names_map_to_local_subclasses(self) -> None:
        stdout = reply(
            {"ok": False, "error": {"kind": "os", "errno": "EACCES", "message": "Permission denied"}}
        )

        with pytest.raises(PermissionError):
            RemoteFileOps(canned(stdout)).read_range("/x", 0, 1, timeout=5)

    @pytest.mark.parametrize(
        "stderr", [b'  File "<hpclab-helper>", line 1\nSyntaxError: invalid syntax', b"Unknown option: -I"]
    )
    def test_too_old_python_switches_to_the_fallback(self, stderr: bytes) -> None:
        calls: list[list[str]] = []

        def execute(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
            calls.append(argv)
            if argv[0] == "python3":
                return RawResult(1, b"", stderr)
            return RawResult(0, MARKER + b"\nM\n", b"")

        ops = RemoteFileOps(execute, python="python3")

        assert ops.stat(["/x"], timeout=5)["/x"].kind == "missing"
        assert ops.python_available is False
        assert [c[0] for c in calls] == ["python3", "sh"]

    def test_non_executable_python_switches_to_the_fallback(self) -> None:
        def execute(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
            if argv[0] != "sh":
                return RawResult(126, b"", b"bash: /opt/py/bin/python3: Permission denied")
            return RawResult(0, MARKER + b"\nM\n", b"")

        ops = RemoteFileOps(execute, python="/opt/py/bin/python3")

        assert ops.stat(["/x"], timeout=5)["/x"].kind == "missing"
        assert ops.python_available is False

    def test_list_dir_retries_python_after_an_earlier_failed_probe(self, tmp_path: Path) -> None:
        # A module-provided python3 can be missing once (Lmod hiccup); list_dir has no fallback,
        # so it must not stay "unsupported" for the controller's lifetime.
        (tmp_path / "f.txt").write_bytes(b"abc")
        broken = {"python": True}

        def execute(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
            if argv[0] == sys.executable and broken["python"]:
                return RawResult(127, b"", b"bash: python3: command not found")
            return run_locally(argv, stdin=stdin, timeout=timeout)

        ops = RemoteFileOps(execute, python=sys.executable)
        assert ops.stat([str(tmp_path)], timeout=10)[str(tmp_path)].kind == "dir"
        assert ops.python_available is False
        broken["python"] = False

        listing = ops.list_dir(str(tmp_path), max_entries=10, timeout=10)

        assert listing == LocalTransport("x").list_dir(str(tmp_path), max_entries=10)
        assert ops.python_available is True

    def test_fallback_splits_long_argument_lists(self) -> None:
        calls: list[list[str]] = []

        def execute(argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
            calls.append(argv)
            if argv[0] != "sh":
                return RawResult(127, b"", b"not found")
            n_paths = len(argv) - 5
            return RawResult(0, MARKER + b"\n" + b"N\n" * n_paths, b"")

        paths = [f"/scratch/user/hpclab/snapshots/{i:06d}/" + "x" * 80 for i in range(2000)]

        result = RemoteFileOps(execute).sha256(paths, timeout=5)

        assert result == dict.fromkeys(paths)
        assert len(calls) > 2
        assert all(sum(len(a) for a in c) < 128 * 1024 for c in calls)


class TestPathAndTimeHelpers:
    @pytest.mark.parametrize(
        "path", ["", ".", "/", "//", "///", "a", "a/", "a//b/./c/", "//x/y", "./a/../b", "/a/."]
    )
    def test_pathlib_form_matches_pathlib(self, path: str) -> None:
        assert pathlib_form(path) == str(PurePosixPath(path))

    def test_mtime_is_rebuilt_exactly_like_cpython(self, tmp_path: Path) -> None:
        f = tmp_path / "t"
        f.write_bytes(b"x")
        os.utime(f, ns=(1_790_561_021_948_514_987, 1_790_561_021_948_514_987))
        st = os.stat(f)
        sec, nsec = divmod(st.st_mtime_ns, 10**9)

        assert _mtime(str(sec), f"{sec}.{nsec:09d}") == st.st_mtime  # BSD %Fm
        assert _mtime(str(sec), f"2026-09-27 10:00:{sec % 60:02d}.{nsec:09d} +0000") == st.st_mtime  # GNU %y
        assert _mtime("1790561021", "2026-09-27 10:00:00 +0000") == 1790561021.0


class TestClassifier:
    @pytest.mark.parametrize(
        ("stderr", "kind", "maybe"),
        [
            ("alice@h: Permission denied (publickey,keyboard-interactive).", "auth", False),
            ("Host key verification failed.", "host_key", False),
            ("command-line line 0: Bad configuration option: x", "config", False),
            ("ssh: connect to host h port 22: Connection timed out", "network", False),
            ("client_loop: send disconnect: Broken pipe", "network", True),
            ("something nobody has seen before", "network", True),
            # File-permission errors are not ssh auth failures.
            ('rsync: mkstemp "/x/.y" failed: Permission denied (13)', "network", True),
        ],
    )
    def test_classify(self, stderr: str, kind: str, maybe: bool) -> None:
        failure = classify_ssh_failure(stderr)
        assert (failure.kind, failure.maybe_executed) == (kind, maybe)

    # OpenSSH >= 9 reports a server-side close or reset through sshpkt_fatal ("Connection closed
    # by <ip> port <n>"), the same wording it uses during key exchange, and write errors as
    # "Connection to <ip> port <n>: <strerror>". None of these prove the command never started.
    @pytest.mark.parametrize(
        "stderr",
        [
            "Connection closed by 10.0.0.1 port 22",
            "Connection reset by 10.0.0.1 port 22",
            "client_loop: send disconnect: Network is unreachable",
            "ssh_packet_write_poll: Connection to 10.0.0.1 port 22: No route to host",
            "Connection to 10.0.0.1 port 22: Host is down",
            "Read from remote host login: Connection reset by peer",
        ],
    )
    def test_ambiguous_drops_may_have_executed(self, stderr: str) -> None:
        assert classify_ssh_failure(stderr) == SshFailure("network", maybe_executed=True)

    @pytest.mark.parametrize(
        "stderr",
        [
            "ssh: connect to host login port 22: Network is unreachable",
            "ssh: connect to host login port 22: No route to host",
            "kex_exchange_identification: Connection closed by remote host\n"
            "Connection closed by UNKNOWN port 65535",
            "channel 0: open failed: connect failed: No route to host\nstdio forwarding failed\n"
            "kex_exchange_identification: Connection closed by remote host",
            "Connection timed out during banner exchange",
        ],
    )
    def test_unambiguous_preconnect_failures_did_not_execute(self, stderr: str) -> None:
        assert classify_ssh_failure(stderr) == SshFailure("network", maybe_executed=False)

    # Warnings ssh prints and then carries on from must not decide the class of a later failure.
    @pytest.mark.parametrize(
        ("stderr", "maybe"),
        [
            ('command-line line 0: Unsupported option "rsaauthentication"\n'
             "ssh: connect to host h port 1: Connection refused", False),
            ('/Users/a/.ssh/config line 3: Unsupported option "gssapiauthentication"\n'
             "client_loop: send disconnect: Broken pipe", True),
            ('Load key "/Users/a/.ssh/id_old": invalid format\nConnection to login closed by remote host.', True),
            ("@@@ WARNING: UNPROTECTED PRIVATE KEY FILE! @@@\nclient_loop: send disconnect: Broken pipe", True),
            ('sign_and_send_pubkey: signing failed for ED25519 "k" from agent: agent refused operation\n'
             "Timeout, server login not responding.", True),
        ],
    )  # fmt: skip
    def test_warnings_do_not_classify_a_network_failure(self, stderr: str, maybe: bool) -> None:
        assert classify_ssh_failure(stderr) == SshFailure("network", maybe_executed=maybe)

    @pytest.mark.parametrize(
        "stderr",
        [
            'rsync: [sender] send_files failed to open "/proj/duo/x.py": Permission denied (13)',
            "error: cannot read /scratch/alice/duo.txt",
        ],
    )
    def test_the_word_duo_in_a_path_is_not_an_auth_failure(self, stderr: str) -> None:
        assert classify_ssh_failure(stderr).kind == "network"

    @pytest.mark.parametrize(
        "stderr",
        [
            "Duo two-factor login for alice",
            "Failed Duo login",
            "Autopushing login request to phone...\nDuo Push",
        ],
    )
    def test_duo_prompts_are_auth(self, stderr: str) -> None:
        assert classify_ssh_failure(stderr) == SshFailure("auth", maybe_executed=False)

    def test_remote_command_in_user_config_is_a_config_error(self) -> None:
        assert classify_ssh_failure("Cannot execute command-line and remote command.").kind == "config"

    def test_output_means_the_command_started(self) -> None:
        failure = classify_ssh_failure("Permission denied (publickey).", started=True)
        assert (failure.kind, failure.maybe_executed) == ("network", True)

    @pytest.mark.parametrize(
        "text",
        [
            'Load key "/home/alice/.ssh/id_ed25519": bad permissions',
            "Enter passphrase for key '/opt/keys/cluster':",
            "no such identity file /srv/keys/deploy: No such file",
            "Bad owner or permissions on /Users/alice/.ssh/config",
            "trying id_rsa_work",
        ],
    )
    def test_redact_removes_key_paths(self, text: str) -> None:
        cleaned = redact(text)
        for secret in (
            "/home/alice/.ssh",
            "id_ed25519",
            "/opt/keys/cluster",
            "/srv/keys/deploy",
            "id_rsa_work",
        ):
            assert secret not in cleaned
        assert "redacted" in cleaned
