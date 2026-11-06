"""File helpers over SSH must behave exactly like LocalTransport.

Every operation runs through the fake ssh (so the real quoting, framing and helper are used)
in two modes: with the embedded python helper and with the coreutils fallback that login
nodes without python3 get. Results are compared field by field with LocalTransport on the
same files.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
from openssh_support import FakeSsh, MakeSsh

from hpclab.adapters.base import TransportError
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.openssh import OpenSSHTransport
from hpclab.domain.errors import ErrorCode

MISSING_PYTHON = "hpclab-test-no-such-python"


@pytest.fixture(params=["python", "fallback"])
def remote(request: pytest.FixtureRequest, make_ssh: MakeSsh) -> OpenSSHTransport:
    python = sys.executable if request.param == "python" else MISSING_PYTHON
    return make_ssh(remote_python=python)


@pytest.fixture
def py_remote(make_ssh: MakeSsh) -> OpenSSHTransport:
    return make_ssh(remote_python=sys.executable)


def _raises(fn: object, *args: object, **kwargs: object) -> tuple[type[BaseException], int | None] | None:
    try:
        fn(*args, **kwargs)  # type: ignore[operator]
    except OSError as exc:
        return type(exc), exc.errno
    return None


class TestStat:
    def test_matches_local_for_files_dirs_links_and_missing(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_paths: list[str]
    ) -> None:
        got = remote.stat(sample_paths)

        assert got == local.stat(sample_paths)
        kinds = {Path(p).name: info.kind for p, info in got.items()}
        assert kinds["link"] == "symlink" and kinds["dangling"] == "missing" and kinds["sub"] == "dir"

    def test_empty_request_does_not_connect(self, remote: OpenSSHTransport, fake_ssh: FakeSsh) -> None:
        assert remote.stat([]) == {}
        assert remote.sha256([]) == {}
        assert fake_ssh.calls() == []

    def test_inode_changes_when_a_file_is_replaced(self, remote: OpenSSHTransport, sample_tree: Path) -> None:
        path = str(sample_tree / "a.txt")
        before = remote.stat([path])[path]
        replacement = sample_tree / "a.txt.new"
        replacement.write_bytes(b"rotated\n")
        os.replace(replacement, path)

        after = remote.stat([path])[path]

        assert before.inode != after.inode and after.size == len(b"rotated\n")


class TestReads:
    @pytest.mark.parametrize(
        ("offset", "length"),
        [(-5, 3), (0, 0), (0, 16), (100, 50), (4090, 100), (4096, 10), (5000, 5), (10, -1)],
    )
    def test_read_range_matches_local(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path, offset: int, length: int
    ) -> None:
        path = str(sample_tree / "bin.dat")
        assert remote.read_range(path, offset, length) == local.read_range(path, offset, length)

    @pytest.mark.parametrize("length", [-3, 0, 5, 4096, 10**6])
    def test_read_tail_matches_local(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path, length: int
    ) -> None:
        path = str(sample_tree / "bin.dat")
        assert remote.read_tail(path, length) == local.read_tail(path, length)

    def test_reads_of_empty_file_and_awkward_names(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path
    ) -> None:
        for name in ("empty", "new\nline.txt", "ünïcødé 日本.txt", "sp ace's.txt", "link"):
            path = str(sample_tree / name)
            assert remote.read_range(path, 0, 100) == local.read_range(path, 0, 100), name
            assert remote.read_tail(path, 3) == local.read_tail(path, 3), name

    @pytest.mark.parametrize("name", ["missing", "dangling", "sub"])
    def test_read_errors_match_local(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path, name: str
    ) -> None:
        path = str(sample_tree / name)
        expected = _raises(local.read_range, path, 0, 10)
        assert expected is not None
        assert _raises(remote.read_range, path, 0, 10) == expected
        assert _raises(remote.read_tail, path, 10) == _raises(local.read_tail, path, 10)

    def test_missing_file_raises_file_not_found(self, remote: OpenSSHTransport, sample_tree: Path) -> None:
        with pytest.raises(FileNotFoundError):
            remote.read_range(str(sample_tree / "nope.log"), 0, 10)


class TestSha256:
    def test_matches_local_including_none_for_missing_and_dirs(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_paths: list[str]
    ) -> None:
        got = remote.sha256(sample_paths)

        assert got == local.sha256(sample_paths)
        assert any(v is None for v in got.values()) and any(v for v in got.values())


class TestListDir:
    @pytest.mark.parametrize("suffix", ["", "/", "//sub/./"])
    @pytest.mark.parametrize("max_entries", [1, 3, 1000])
    def test_matches_local(
        self,
        py_remote: OpenSSHTransport,
        local: LocalTransport,
        sample_tree: Path,
        suffix: str,
        max_entries: int,
    ) -> None:
        root = str(sample_tree) + suffix
        assert py_remote.list_dir(root, max_entries=max_entries) == local.list_dir(
            root, max_entries=max_entries
        )

    @pytest.mark.parametrize("name", ["missing", "a.txt"])
    def test_non_directories_raise_like_local(
        self, py_remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path, name: str
    ) -> None:
        path = str(sample_tree / name)
        with pytest.raises(FileNotFoundError) as remote_exc:
            py_remote.list_dir(path)
        with pytest.raises(FileNotFoundError) as local_exc:
            local.list_dir(path)
        assert remote_exc.value.args == local_exc.value.args

    def test_needs_python_on_the_login_node(self, make_ssh: MakeSsh, sample_tree: Path) -> None:
        transport = make_ssh(remote_python=MISSING_PYTHON)

        with pytest.raises(TransportError) as exc:
            transport.list_dir(str(sample_tree))

        assert exc.value.code == ErrorCode.UNSUPPORTED and not exc.value.retryable
        assert transport.remote_python_available is False


class TestWriteFile:
    # Short ids: pytest exports the node id in PYTEST_CURRENT_TEST, i.e. into the ssh env.
    @pytest.mark.parametrize(
        "content", [b"", b"hello\n", bytes(range(256)) * 4096], ids=["empty", "small", "1MiB-binary"]
    )
    def test_matches_local_content_mode_and_parents(
        self, remote: OpenSSHTransport, local: LocalTransport, tmp_path: Path, content: bytes
    ) -> None:
        mine, theirs = tmp_path / "remote" / "a" / "b" / "f.bin", tmp_path / "local" / "a" / "b" / "f.bin"

        remote.write_file(str(mine), content, mode=0o640)
        local.write_file(str(theirs), content, mode=0o640)

        assert mine.read_bytes() == theirs.read_bytes() == content
        assert stat.S_IMODE(mine.stat().st_mode) == stat.S_IMODE(theirs.stat().st_mode) == 0o640
        assert sorted(os.listdir(mine.parent)) == ["f.bin"]  # no temp files left behind

    def test_overwrite_is_atomic_replace(self, remote: OpenSSHTransport, sample_tree: Path) -> None:
        path = sample_tree / "a.txt"
        old_inode = path.stat().st_ino

        remote.write_file(str(path), b"replaced", mode=0o600)

        assert path.read_bytes() == b"replaced"
        assert path.stat().st_ino != old_inode
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.parametrize("target", ["a.txt/child.txt", "sub"])
    def test_errors_match_local(
        self, remote: OpenSSHTransport, local: LocalTransport, sample_tree: Path, target: str
    ) -> None:
        path = str(sample_tree / target)
        expected = _raises(local.write_file, path, b"x")
        assert expected is not None
        assert _raises(remote.write_file, path, b"x") == expected

    @pytest.mark.parametrize("mode", [-1, 0o10000, True])
    def test_invalid_mode_is_rejected_locally(
        self, remote: OpenSSHTransport, fake_ssh: FakeSsh, tmp_path: Path, mode: int
    ) -> None:
        with pytest.raises(ValueError):
            remote.write_file(str(tmp_path / "x"), b"x", mode=mode)
        assert fake_ssh.calls() == []


def test_mkdirs_creates_nested_directories(remote: OpenSSHTransport, tmp_path: Path) -> None:
    targets = [tmp_path / "m" / "n" / "o", tmp_path / "with space"]

    remote.mkdirs([str(t) for t in targets])
    remote.mkdirs([str(t) for t in targets])  # idempotent

    assert all(t.is_dir() for t in targets)


def test_mkdirs_reports_os_errors(remote: OpenSSHTransport, sample_tree: Path) -> None:
    with pytest.raises(FileExistsError):
        remote.mkdirs([str(sample_tree / "a.txt")])


def test_login_shell_noise_on_stdout_is_ignored(
    remote: OpenSSHTransport,
    local: LocalTransport,
    fake_ssh: FakeSsh,
    sample_tree: Path,
    sample_paths: list[str],
) -> None:
    fake_ssh.set(noise="Welcome to the cluster!\nModule loaded: cuda/12.8\n")
    path = str(sample_tree / "bin.dat")

    assert remote.stat(sample_paths) == local.stat(sample_paths)
    assert remote.read_range(path, 7, 300) == local.read_range(path, 7, 300)
    assert remote.sha256([path]) == local.sha256([path])


def test_fallback_without_any_python_on_path(
    make_ssh: MakeSsh,
    local: LocalTransport,
    fake_ssh: FakeSsh,
    no_python_path: str,
    sample_tree: Path,
    sample_paths: list[str],
) -> None:
    """The real scenario: ``python3`` is simply absent from the login node's PATH."""
    transport = make_ssh()  # default remote_python="python3"
    fake_ssh.set(remote_path=no_python_path)
    path = str(sample_tree / "bin.dat")

    assert transport.stat(sample_paths) == local.stat(sample_paths)
    assert transport.remote_python_available is False
    assert transport.read_tail(path, 100) == local.read_tail(path, 100)
    assert transport.sha256(sample_paths) == local.sha256(sample_paths)
    transport.write_file(str(sample_tree / "out" / "w.txt"), b"written", mode=0o600)
    assert (sample_tree / "out" / "w.txt").read_bytes() == b"written"


def test_file_ops_match_local_under_every_login_shell(
    remote: OpenSSHTransport,
    local: LocalTransport,
    login_shell: str,
    sample_tree: Path,
    sample_paths: list[str],
) -> None:
    path = str(sample_tree / "bin.dat")
    target = sample_tree / "out dir" / "w.bin"

    assert remote.stat(sample_paths) == local.stat(sample_paths)
    assert remote.read_range(path, 5, 700) == local.read_range(path, 5, 700)
    assert remote.sha256(sample_paths) == local.sha256(sample_paths)
    remote.write_file(str(target), b"\x00\xffpayload\n", mode=0o600)
    assert target.read_bytes() == b"\x00\xffpayload\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_python_probe_happens_once(make_ssh: MakeSsh, fake_ssh: FakeSsh, sample_tree: Path) -> None:
    transport = make_ssh(remote_python=MISSING_PYTHON)
    path = str(sample_tree / "a.txt")

    transport.stat([path])
    transport.stat([path])

    commands = [call["command"] for call in fake_ssh.calls()]
    assert [c.startswith(MISSING_PYTHON) for c in commands] == [True, False, False]
    assert all(c.startswith("sh -c") for c in commands[1:])
