"""Fixtures for adapter tests (fake ssh/rsync wiring lives in ``openssh_support``)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from openssh_support import FAKE_RSYNC, FAKE_SSH, FakeSsh, MakeSsh, install_shim

from hpclab.adapters import openssh
from hpclab.adapters.local import LocalTransport
from hpclab.adapters.openssh import OpenSSHTransport


@pytest.fixture(scope="session")
def shim_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory holding the fake ``ssh`` and ``rsync`` executables."""
    root = tmp_path_factory.mktemp("shims")
    install_shim(root / "ssh", FAKE_SSH)
    install_shim(root / "rsync", FAKE_RSYNC)
    return root


_FAKE_KEYS = (
    "FAIL_STDERR",
    "FAIL_CODE",
    "DROP_AFTER",
    "SLEEP",
    "NOISE",
    "REMOTE_PATH",
    "MASTER",
    "LOG",
    "PIDFILE",
    "SHELL",
    "WARN",
)


@pytest.fixture
def fake_ssh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSsh:
    for key in _FAKE_KEYS:
        monkeypatch.delenv(f"FAKE_SSH_{key}", raising=False)
    for key in ("LOG", "FAIL_STDERR", "FAIL_CODE"):
        monkeypatch.delenv(f"FAKE_RSYNC_{key}", raising=False)
    log = tmp_path / "fake-ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    return FakeSsh(monkeypatch, log)


@pytest.fixture
def short_control_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep the fallback control directory inside the test's tmp dir instead of /tmp."""
    root = tmp_path / "ctlroot"
    monkeypatch.setattr(openssh, "_SHORT_CONTROL_ROOT", root)
    return root


@pytest.fixture
def make_ssh(shim_bin: Path, fake_ssh: FakeSsh, short_control_root: Path, tmp_path: Path) -> MakeSsh:
    """Factory for an OpenSSHTransport wired to the fake ssh/rsync."""

    def factory(**overrides: Any) -> OpenSSHTransport:
        kwargs: dict[str, Any] = {
            "control_dir": tmp_path / "ctl",
            "ssh_bin": str(shim_bin / "ssh"),
            "rsync_bin": str(shim_bin / "rsync"),
        }
        kwargs.update(overrides)
        host = kwargs.pop("host", "login.example.org")
        return OpenSSHTransport("testcluster", host, **kwargs)

    return factory


@pytest.fixture
def ssh(make_ssh: MakeSsh) -> OpenSSHTransport:
    return make_ssh()


@pytest.fixture
def local() -> LocalTransport:
    return LocalTransport("testcluster")


@pytest.fixture
def sample_tree(tmp_path: Path) -> Path:
    """Files, dirs, symlinks (file, dir, dangling) and awkward names, under one root."""
    root = tmp_path / "tree"
    (root / "sub" / "deeper").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"hello world\n")
    (root / "bin.dat").write_bytes(bytes(range(256)) * 16)
    (root / "empty").write_bytes(b"")
    (root / "sub" / "nested.txt").write_bytes(b"nested\n")
    (root / "sub" / "deeper" / "z.txt").write_bytes(b"z" * 100)
    (root / "sp ace's.txt").write_bytes(b"space")
    (root / "ünïcødé 日本.txt").write_bytes("ünï".encode())
    (root / "new\nline.txt").write_bytes(b"newline")
    (root / "link").symlink_to("a.txt")
    (root / "dirlink").symlink_to("sub")
    (root / "dangling").symlink_to("does-not-exist")
    return root


@pytest.fixture
def sample_paths(sample_tree: Path) -> list[str]:
    """Every interesting path under ``sample_tree`` plus missing and odd spellings."""
    names = [
        "a.txt",
        "bin.dat",
        "empty",
        "sub",
        "sub/nested.txt",
        "sp ace's.txt",
        "ünïcødé 日本.txt",
        "new\nline.txt",
        "link",
        "dirlink",
        "dangling",
        "missing",
        "sub/missing/x",
        "a.txt/child",
    ]
    root = sample_tree
    return [str(root / n) for n in names] + [str(root), f"{root}/sub/", f"{root}//sub/./nested.txt"]


LOGIN_SHELLS = [p for p in ("/bin/bash", "/bin/zsh", "/bin/dash") if Path(p).exists()]


@pytest.fixture(params=LOGIN_SHELLS, ids=lambda p: Path(p).name)
def login_shell(
    request: pytest.FixtureRequest, fake_ssh: FakeSsh, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> str:
    """Run the fake ssh's remote side under each POSIX login shell available here (the
    developer's own ~/.zshenv is kept out so results do not depend on the machine)."""
    monkeypatch.setenv("ZDOTDIR", str(tmp_path))
    monkeypatch.delenv("BASH_ENV", raising=False)
    fake_ssh.set(shell=request.param)
    return str(request.param)


@pytest.fixture
def no_python_path(tmp_path: Path) -> Iterator[str]:
    """A PATH with the coreutils the fallback needs but no python at all."""
    bindir = tmp_path / "nopython-bin"
    bindir.mkdir()
    tools = [
        "sh", "bash", "stat", "readlink", "realpath", "base64", "tr", "tail", "head", "mkdir",
        "mktemp", "cat", "wc", "chmod", "mv", "rm", "id", "sha256sum", "shasum", "perl", "env",
    ]  # fmt: skip
    for tool in tools:
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    if not (bindir / "sha256sum").exists() and not (bindir / "shasum").exists():
        pytest.skip("neither sha256sum nor shasum available")
    missing = [
        t for t in ("sh", "stat", "readlink", "base64", "tail", "head", "mktemp") if not (bindir / t).exists()
    ]
    if missing:
        pytest.skip(f"coreutils missing for the fallback test: {missing}")
    assert not any(p.name.startswith("python") for p in bindir.iterdir())
    yield str(bindir)
