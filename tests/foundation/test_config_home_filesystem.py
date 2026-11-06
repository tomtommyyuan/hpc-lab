"""HPCLAB_HOME must be on a local disk: a home on a network file system is detected and refused.

The ledger is SQLite in WAL mode (one host's shared memory) and the single-instance lock is a
per-host ``flock``. A home shared by two machines (an NFS home used from a workstation and a
login node) could run two controllers on one database, or let one machine signal a pid that
only means something on the other. Before the fix only comments said "must be on a local
disk"; nothing checked it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hpclab import config
from hpclab.config import (
    ALLOW_NETWORK_HOME_ENV,
    FileSystem,
    check_home_filesystem,
    filesystem_of,
    fstype_from_mounts,
    is_network_fstype,
)
from hpclab.domain.errors import ErrorCode, HpcLabError

MOUNTS = """\
/dev/sda1 / ext4 rw,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
nfs-home:/export/home /home nfs4 rw,relatime,vers=4.2 0 0
/dev/sdb1 /home/local ext4 rw 0 0
nfs-x:/x /home/with\\040space nfs rw 0 0
tmpfs /scratch tmpfs rw 0 0
nfs-y:/scratch /scratch nfs rw 0 0
"""


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/tom/.hpclab", ("nfs4", "/home")),
        ("/home", ("nfs4", "/home")),
        ("/home/local/tom/.hpclab", ("ext4", "/home/local")),
        ("/homer/tom", ("ext4", "/")),  # a sibling sharing the prefix is not under /home
        ("/home/with space/h", ("nfs", "/home/with space")),  # octal escapes in mount points
        ("/scratch/tom", ("nfs", "/scratch")),  # a later mount hides the one below it
        ("/", ("ext4", "/")),
    ],
)
def test_mount_table_longest_prefix_wins(path: str, expected: tuple[str, str]) -> None:
    assert fstype_from_mounts(path, MOUNTS) == expected


def test_mount_table_without_a_match_is_unknown() -> None:
    assert fstype_from_mounts("/home/x", "") is None
    assert fstype_from_mounts("/home/x", "garbage\n\n") is None


@pytest.mark.parametrize(
    "fstype",
    ["nfs", "nfs4", "NFS", "cifs", "smb3", "smbfs", "afs", "lustre", "gpfs", "beegfs", "ceph", "fuse.sshfs"],
)
def test_network_file_systems_are_recognised(fstype: str) -> None:
    assert is_network_fstype(fstype)


@pytest.mark.parametrize(
    "fstype", ["ext4", "xfs", "btrfs", "zfs", "apfs", "hfs", "tmpfs", "overlay", "fuseblk", "fuse.encfs"]
)
def test_local_file_systems_are_not_network(fstype: str) -> None:
    assert not is_network_fstype(fstype)


@pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="detection is implemented for macOS/Linux"
)
def test_a_temporary_directory_is_detected_as_local(tmp_path: Path) -> None:
    fs = filesystem_of(tmp_path)
    assert fs is not None and fs.fstype and not fs.network
    assert check_home_filesystem(tmp_path / "home") is None


@pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="detection is implemented for macOS/Linux"
)
def test_a_home_that_does_not_exist_yet_is_judged_by_its_nearest_ancestor(tmp_path: Path) -> None:
    assert filesystem_of(tmp_path / "not" / "yet" / "created") == filesystem_of(tmp_path)


def test_an_unreadable_mount_table_is_unknown_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setattr(config, "_PROC_MOUNTS", str(tmp_path / "no-such-mounts"))
    assert filesystem_of(tmp_path) is None
    assert check_home_filesystem(tmp_path) is None  # unknown is never refused


def test_linux_detection_reads_the_mount_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mounts = tmp_path / "mounts"
    real = str(tmp_path.resolve())
    mounts.write_text(f"/dev/sda1 / ext4 rw 0 0\nsrv:/h {real} nfs4 rw 0 0\n")
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setattr(config, "_PROC_MOUNTS", str(mounts))
    assert filesystem_of(tmp_path / "home") == FileSystem("nfs4", real, network=True)


@pytest.fixture
def nfs_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(ALLOW_NETWORK_HOME_ENV, raising=False)
    monkeypatch.setattr(config, "filesystem_of", lambda p: FileSystem("nfs4", "/home", network=True))
    return tmp_path / "home"


def test_a_network_home_is_refused_with_a_hint(nfs_home: Path) -> None:
    with pytest.raises(HpcLabError) as ei:
        check_home_filesystem(nfs_home)
    err = ei.value
    assert err.code == ErrorCode.PRECONDITION_FAILED and not err.retryable
    assert str(nfs_home) in err.message and "nfs4" in err.message and "network file system" in err.message
    assert err.details["fstype"] == "nfs4" and err.details["home"] == str(nfs_home)
    hint = err.details["hint"]
    assert "HPCLAB_HOME" in hint and "local disk" in hint and ALLOW_NETWORK_HOME_ENV in hint


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_a_network_home_can_be_allowed_explicitly(
    nfs_home: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ALLOW_NETWORK_HOME_ENV, value)
    warning = check_home_filesystem(nfs_home)
    assert warning is not None and "nfs4" in warning and ALLOW_NETWORK_HOME_ENV in warning


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_a_falsy_override_still_refuses(nfs_home: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ALLOW_NETWORK_HOME_ENV, value)
    with pytest.raises(HpcLabError):
        check_home_filesystem(nfs_home)


def test_a_local_home_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "filesystem_of", lambda p: FileSystem("apfs", "/", network=False))
    assert check_home_filesystem(tmp_path) is None
