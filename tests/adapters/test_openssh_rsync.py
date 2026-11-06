"""upload_dir / download over rsync: argument contract, mirroring semantics, failure classes.

The fake rsync (see ``openssh_support``) checks our flags and routes the remote side through
the ``-e`` ssh command, so ssh failures surface exactly as they would with real rsync. When a
real ``rsync`` is installed, the same round trip also runs through it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import time
from pathlib import Path

import pytest
from openssh_support import FakeSsh, MakeSsh

from hpclab.adapters.base import AuthRequiredError, CommandFailed, ConnectionLost, TransportTimeout
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.openssh import OpenSSHTransport
from hpclab.domain.errors import ErrorCode, HpcLabError


def snapshot_of(root: Path) -> dict[str, tuple[str, bytes | str]]:
    """Relative path -> ("file", bytes) | ("link", target), for comparing trees."""
    out: dict[str, tuple[str, bytes | str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            out[rel] = (
                ("link", os.readlink(full)) if os.path.islink(full) else ("file", Path(full).read_bytes())
            )
    return out


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "train.py").write_text("print('train')\n")
    (root / "pkg" / "model.py").write_text("W = 1\n")
    (root / "run.sh").write_text("#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    (root / "latest").symlink_to("train.py")
    return root


@pytest.fixture
def rsync_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    log = tmp_path / "fake-rsync.log"
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(log))
    return log


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def logged_argv(log: Path) -> list[str]:
    return json.loads(log.read_text().splitlines()[-1])["argv"]


class TestUpload:
    def test_mirrors_tree_and_creates_missing_parents(
        self, ssh: OpenSSHTransport, project: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "remote" / "hpclab" / "snapshots" / "abc123"

        ssh.upload_dir(str(project), str(dest))

        assert snapshot_of(dest) == snapshot_of(project)
        assert stat.S_IMODE((dest / "run.sh").stat().st_mode) == 0o755

    def test_reupload_deletes_files_removed_locally(
        self, ssh: OpenSSHTransport, project: Path, tmp_path: Path
    ) -> None:
        dest = tmp_path / "remote" / "snap"
        ssh.upload_dir(str(project), str(dest))
        (project / "pkg" / "model.py").unlink()
        (project / "new.py").write_text("x = 2\n")

        ssh.upload_dir(str(project), str(dest) + "/")

        assert snapshot_of(dest) == snapshot_of(project)

    def test_rsync_arguments(self, make_ssh: MakeSsh, rsync_log: Path, project: Path, tmp_path: Path) -> None:
        transport = make_ssh(user="alice")
        dest = str(tmp_path / "remote" / "snap")

        transport.upload_dir(str(project), dest)

        argv = logged_argv(rsync_log)
        assert argv[:2] == ["-a", "--delete-after"]
        shell = argv[argv.index("-e") + 1].split(" ")
        assert shell[0] == transport.ssh_bin
        assert "BatchMode=yes" in shell and f"ControlPath={transport.control_path}" in shell
        assert shell[-2:] == ["-l", "alice"]
        assert f"--rsync-path=mkdir -p {shlex.quote(str(tmp_path / 'remote'))} && rsync" in argv
        assert argv[-2:] == [f"{project}/", f"login.example.org:{dest}/"]

    def test_options_needing_quotes_go_through_a_private_wrapper(
        self, make_ssh: MakeSsh, rsync_log: Path, project: Path, tmp_path: Path
    ) -> None:
        # openrsync (macOS) splits -e on spaces only, so quoted options need a wrapper script.
        transport = make_ssh(ssh_options=["SetEnv=HPCLAB_TEST=a b"])
        dest = tmp_path / "remote" / "snap"

        transport.upload_dir(str(project), str(dest))

        wrapper = Path(logged_argv(rsync_log)[3])
        assert wrapper.parent == Path(transport.control_path).parent
        assert stat.S_IMODE(wrapper.stat().st_mode) == 0o700
        assert "'SetEnv=HPCLAB_TEST=a b'" in wrapper.read_text()
        assert snapshot_of(dest) == snapshot_of(project)

    def test_missing_local_directory(self, ssh: OpenSSHTransport, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ssh.upload_dir(str(tmp_path / "nope"), str(tmp_path / "dest"))

    @pytest.mark.parametrize(
        "dest", ["$USER/hpclab", "~/hpclab", "/scratch/a b", "-rf", "/x;rm -rf y", "/x/`id`", ":module/x"]
    )
    def test_unsafe_remote_paths_are_rejected(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, project: Path, dest: str
    ) -> None:
        with pytest.raises(HpcLabError) as exc:
            ssh.upload_dir(str(project), dest)
        assert exc.value.code == ErrorCode.INVALID_ARGUMENT
        assert fake_ssh.calls() == []

    def test_auth_failure_is_auth_required(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, project: Path, tmp_path: Path
    ) -> None:
        fake_ssh.set(fail_stderr="alice@login: Permission denied (keyboard-interactive).")

        with pytest.raises(AuthRequiredError) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "dest"))
        assert exc.value.details["login_command"] == ssh.login_command()

    def test_network_failure_is_connection_lost(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, project: Path, tmp_path: Path
    ) -> None:
        fake_ssh.set(fail_stderr="client_loop: send disconnect: Broken pipe")

        with pytest.raises(ConnectionLost) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "dest"))
        assert exc.value.maybe_executed is True and exc.value.retryable

    def test_timeout_kills_rsync_and_its_ssh_child(
        self, ssh: OpenSSHTransport, fake_ssh: FakeSsh, project: Path, tmp_path: Path
    ) -> None:
        pidfile = tmp_path / "ssh.pid"
        fake_ssh.set(sleep="30", pidfile=str(pidfile))

        with pytest.raises(TransportTimeout):
            ssh.upload_dir(str(project), str(tmp_path / "dest"), timeout=1.0)

        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.05)
        assert not _alive(pid)

    def test_file_errors_mentioning_duo_are_not_auth_failures(
        self, ssh: OpenSSHTransport, monkeypatch: pytest.MonkeyPatch, project: Path, tmp_path: Path
    ) -> None:
        # A research method or dataset called "Duo" is plausible; its paths show up in rsync errors.
        monkeypatch.setenv(
            "FAKE_RSYNC_FAIL_STDERR",
            'rsync: [sender] send_files failed to open "/proj/duo/x.py": Permission denied (13)\n'
            "rsync error: some files/attrs were not transferred (code 23)",
        )

        with pytest.raises(CommandFailed) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "remote" / "snap"))
        assert exc.value.details["exit_code"] == 23

    @pytest.mark.parametrize(
        ("stderr", "code"),
        [
            ('rsync: [receiver] write failed on "/scratch/a/snap/x": Disk quota exceeded (122)\n'
             "rsync: connection unexpectedly closed (812 bytes received so far) [sender]\n"
             "rsync error: error in rsync protocol data stream (code 12)", 12),
            ("mkdir: cannot create directory '/scratch/a/hpclab': No space left on device", 12),
        ],
    )  # fmt: skip
    def test_full_remote_storage_is_not_a_retryable_network_error(
        self,
        ssh: OpenSSHTransport,
        monkeypatch: pytest.MonkeyPatch,
        project: Path,
        tmp_path: Path,
        stderr: str,
        code: int,
    ) -> None:
        # Quota is the most common HPC upload failure; retrying it as a "network glitch" never ends.
        monkeypatch.setenv("FAKE_RSYNC_FAIL_STDERR", stderr)
        monkeypatch.setenv("FAKE_RSYNC_FAIL_CODE", str(code))

        with pytest.raises(CommandFailed) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "remote" / "snap"))
        assert exc.value.details["reason"] == "remote_storage_full"
        assert not exc.value.retryable

    def test_noisy_login_shell_is_a_precondition_with_a_hint(
        self, ssh: OpenSSHTransport, monkeypatch: pytest.MonkeyPatch, project: Path, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(
            "FAKE_RSYNC_FAIL_STDERR",
            "protocol version mismatch -- is your shell clean?\n"
            "(see the rsync manpage for an explanation)\n"
            "rsync error: protocol incompatibility (code 2) at compat.c(622) [sender=3.2.7]",
        )
        monkeypatch.setenv("FAKE_RSYNC_FAIL_CODE", "2")

        with pytest.raises(HpcLabError) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "remote" / "snap"))
        assert exc.value.code == ErrorCode.PRECONDITION_FAILED and not exc.value.retryable
        assert "bashrc" in exc.value.details["hint"]

    @pytest.mark.parametrize(
        "stderr",
        [
            "zsh:1: command not found: rsync",
            "bash: line 1: rsync: command not found",
            "sh: 1: rsync: not found",
        ],
    )
    def test_rsync_missing_on_the_login_node_is_unsupported(
        self,
        ssh: OpenSSHTransport,
        monkeypatch: pytest.MonkeyPatch,
        project: Path,
        tmp_path: Path,
        stderr: str,
    ) -> None:
        monkeypatch.setenv("FAKE_RSYNC_FAIL_STDERR", stderr + "\nrsync: connection unexpectedly closed")
        monkeypatch.setenv("FAKE_RSYNC_FAIL_CODE", "12")

        with pytest.raises(HpcLabError) as exc:
            ssh.upload_dir(str(project), str(tmp_path / "remote" / "snap"))
        assert exc.value.code == ErrorCode.UNSUPPORTED

    def test_remote_mkdir_failure_is_a_command_failure(
        self, ssh: OpenSSHTransport, project: Path, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "remote"
        blocker.write_text("a file where the parent directory should be")

        with pytest.raises(CommandFailed) as exc:
            ssh.upload_dir(str(project), str(blocker / "snap"))
        assert "mkdir" in exc.value.details["stderr"]


class TestDownload:
    def test_directory_is_mirrored(self, ssh: OpenSSHTransport, project: Path, tmp_path: Path) -> None:
        target = tmp_path / "local" / "copy"
        target.mkdir(parents=True)
        (target / "stale.txt").write_text("old")

        ssh.download(str(project), str(target))

        assert snapshot_of(target) == snapshot_of(project)

    def test_file_and_symlinked_file_match_local_transport(
        self, ssh: OpenSSHTransport, local: LocalTransport, project: Path, tmp_path: Path
    ) -> None:
        for name in ("train.py", "latest"):
            mine, theirs = tmp_path / "a" / name, tmp_path / "b" / name
            ssh.download(str(project / name), str(mine))
            local.download(str(project / name), str(theirs))
            assert not mine.is_symlink()
            assert mine.read_bytes() == theirs.read_bytes() == b"print('train')\n"

    def test_missing_remote_path(self, ssh: OpenSSHTransport, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ssh.download(str(tmp_path / "nope"), str(tmp_path / "out"))


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
class TestRealRsync:
    """The system rsync (GNU rsync or openrsync) through the fake ssh: our flags really work."""

    @pytest.fixture
    def real(self, make_ssh: MakeSsh) -> OpenSSHTransport:
        return make_ssh(rsync_bin=shutil.which("rsync"))

    def test_upload_and_download_round_trip(
        self, real: OpenSSHTransport, project: Path, tmp_path: Path
    ) -> None:
        remote = tmp_path / "remote" / "hpclab" / "snap"
        real.upload_dir(str(project), str(remote))
        assert snapshot_of(remote) == snapshot_of(project)

        (project / "train.py").unlink()
        real.upload_dir(str(project), str(remote))
        assert "train.py" not in snapshot_of(remote)

        back = tmp_path / "back"
        real.download(str(remote), str(back))
        assert snapshot_of(back) == snapshot_of(project)

    def test_auth_failure_through_real_rsync(
        self, real: OpenSSHTransport, fake_ssh: FakeSsh, project: Path, tmp_path: Path
    ) -> None:
        fake_ssh.set(fail_stderr="alice@login: Permission denied (publickey).")

        with pytest.raises(AuthRequiredError):
            real.upload_dir(str(project), str(tmp_path / "remote" / "snap"))
