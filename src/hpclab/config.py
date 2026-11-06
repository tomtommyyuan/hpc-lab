"""Local paths. One hpclab home = one controller = one ledger.

``HPCLAB_HOME`` (default ``~/.hpclab``) holds::

    hpclab.db            SQLite ledger (WAL) — must be on a local disk (see below)
    controller.sock      Unix socket (0600) inside a 0700 directory
    controller.lock      single-instance lock
    controller.log
    clusters/<id>.yaml   user overlays (private)
    profiles/<id>.yaml   locally installed/edited public profiles (override bundled ones)
    snapshots/           content-addressed code snapshots
    cache/               downloaded artifacts (by digest)
    evidence/            exported bundles (when the CLI exports to disk)
    sim/<cluster>/       simulator state for transport=sim clusters

The home must be on a local disk of the one machine that uses it: SQLite in WAL mode needs one
host's shared memory, and the controller's single-instance lock and pidfile only mean something
on one host. ``check_home_filesystem`` (run before a controller starts or is spawned) refuses a
home on a network file system (NFS, SMB, AFS, Lustre, GPFS, ...) unless
``HPCLAB_ALLOW_NETWORK_HOME=1`` says that only this machine ever uses it.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from hpclab.domain.errors import ErrorCode, HpcLabError


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def db(self) -> Path:
        return self.home / "hpclab.db"

    @property
    def socket(self) -> Path:
        p = self.home / "run" / "controller.sock"
        # AF_UNIX sun_path is limited in *bytes* (104 on macOS incl. NUL, 108 on Linux); a home
        # with non-ASCII names can be short in characters yet too long in bytes.
        if len(os.fsencode(str(p))) <= 100:
            return p
        # Fall back to a short per-user, per-home path (0700 dir, owner-checked by the controller).
        tag = hashlib.sha256(str(self.home).encode()).hexdigest()[:10]
        return Path("/tmp") / f"hpclab-{os.getuid()}-{tag}" / "controller.sock"  # noqa: S108 - 0700 dir, per-user

    @property
    def lock(self) -> Path:
        return self.home / "run" / "controller.lock"

    @property
    def pidfile(self) -> Path:
        return self.home / "run" / "controller.pid"

    @property
    def log(self) -> Path:
        return self.home / "logs" / "controller.log"

    @property
    def overlays(self) -> Path:
        return self.home / "clusters"

    @property
    def profiles(self) -> Path:
        return self.home / "profiles"

    @property
    def snapshots(self) -> Path:
        return self.home / "snapshots"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def evidence(self) -> Path:
        return self.home / "evidence"

    def sim_root(self, cluster_id: str) -> Path:
        return self.home / "sim" / cluster_id

    def ensure(self) -> Paths:
        for d in (
            self.home,
            self.home / "run",
            self.home / "logs",
            self.overlays,
            self.profiles,
            self.snapshots,
            self.cache,
            self.evidence,
        ):
            d.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        os.chmod(self.home / "run", 0o700)
        os.chmod(self.overlays, 0o700)  # private overlays (defence in depth; home is 0700 too)
        # A /tmp socket fallback directory is created by the controller when it binds (under its
        # instance lock) and removed on shutdown; creating it here would leak one per home.
        return self


def default_home() -> Path:
    env = os.environ.get("HPCLAB_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".hpclab").resolve()


def get_paths(home: str | Path | None = None) -> Paths:
    return Paths(Path(home).expanduser().resolve() if home else default_home())


# ---------------------------------------------------------------------------
# Which file system holds the home
# ---------------------------------------------------------------------------

ALLOW_NETWORK_HOME_ENV = "HPCLAB_ALLOW_NETWORK_HOME"

# File systems other machines can mount too (Linux /proc/mounts names; a FUSE mount appears as
# "fuse.<name>"). On macOS the kernel's MNT_LOCAL flag decides, these names only add to it.
NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb",
        "smb2",
        "smb3",
        "smbfs",
        "afs",
        "afpfs",
        "ncpfs",
        "coda",
        "davfs",
        "webdav",
        "sshfs",
        "ceph",
        "ceph-fuse",
        "glusterfs",
        "lustre",
        "gpfs",
        "beegfs",
        "panfs",
        "wekafs",
        "orangefs",
        "pvfs2",
        "ocfs2",
        "gfs2",
        "s3fs",
        "gcsfuse",
        "rclone",
        "juicefs",
    }
)

_PROC_MOUNTS = "/proc/self/mounts"
_MNT_LOCAL = 0x00001000  # macOS <sys/mount.h>
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class FileSystem:
    fstype: str
    mount_point: str
    network: bool


def is_network_fstype(fstype: str) -> bool:
    name = fstype.strip().lower()
    return name.removeprefix("fuse.") in NETWORK_FSTYPES


def fstype_from_mounts(path: str, mounts: str) -> tuple[str, str] | None:
    """``(fstype, mount point)`` of the mount holding the absolute, resolved ``path`` according
    to a ``/proc/mounts`` table; the longest matching mount point wins, and of equal ones the
    last (a later mount hides the one below it). None when nothing matches."""
    best: tuple[str, str] | None = None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mnt = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1])
        if not mnt.startswith("/"):
            continue
        covers = mnt == "/" or path == mnt or path.startswith(mnt.rstrip("/") + "/")
        if covers and (best is None or len(mnt) >= len(best[1])):
            best = (fields[2], mnt)
    return best


def _darwin_statfs(path: str) -> FileSystem | None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    try:  # x86_64 exports the 64-bit-inode layout under this name; arm64 has only that layout
        statfs = getattr(libc, "statfs$INODE64")
    except AttributeError:
        statfs = libc.statfs
    buf = ctypes.create_string_buffer(4096)  # sizeof(struct statfs) is 2168
    if statfs(os.fsencode(path), buf) != 0:
        return None
    raw = buf.raw
    (flags,) = struct.unpack_from("=I", raw, 64)
    fstype = raw[72:88].split(b"\0", 1)[0].decode("ascii", "replace")
    mount_point = os.fsdecode(raw[88:1112].split(b"\0", 1)[0])
    if not fstype:
        return None
    # an untriggered automount point is not itself a network file system
    network = (not flags & _MNT_LOCAL and fstype != "autofs") or is_network_fstype(fstype)
    return FileSystem(fstype, mount_point, network)


def _nearest_existing(path: Path) -> Path:
    p = Path(os.path.abspath(path))
    for candidate in (p, *p.parents):
        if os.path.exists(candidate):
            return candidate
    return Path("/")


def filesystem_of(path: Path) -> FileSystem | None:
    """The file system holding ``path`` (its nearest existing ancestor while it does not exist
    yet). None when it cannot be told: another platform, an unreadable mount table. Never
    raises."""
    target = _nearest_existing(path)
    try:
        if sys.platform == "darwin":
            return _darwin_statfs(str(target))
        if sys.platform.startswith("linux"):
            with open(_PROC_MOUNTS, encoding="utf-8", errors="surrogateescape") as fh:
                found = fstype_from_mounts(os.path.realpath(target), fh.read())
            return FileSystem(found[0], found[1], is_network_fstype(found[0])) if found else None
    except (OSError, AttributeError, ValueError, struct.error):
        return None
    return None


def check_home_filesystem(home: Path) -> str | None:
    """Refuse an hpclab home on a network file system.

    Raises ``HpcLabError`` (precondition_failed, not retryable) when ``home`` is on one, unless
    ``HPCLAB_ALLOW_NETWORK_HOME`` is set; then returns a warning for the caller to log. None
    for a local file system, or when the file system cannot be told."""
    fs = filesystem_of(home)
    if fs is None or not fs.network:
        return None
    problem = (
        f"HPCLAB_HOME {home} is on a network file system ({fs.fstype}, mounted at {fs.mount_point}); "
        "the ledger (SQLite in WAL mode) and the controller's single-instance lock need a local "
        "disk: machines sharing this home could run two controllers on one database, or stop "
        "each other's processes"
    )
    if os.environ.get(ALLOW_NETWORK_HOME_ENV, "").strip().lower() in _TRUTHY:
        return f"{problem}. Continuing because {ALLOW_NETWORK_HOME_ENV} is set: never use this home from another machine"
    raise HpcLabError(
        problem,
        code=ErrorCode.PRECONDITION_FAILED,
        retryable=False,
        details={
            "home": str(home),
            "fstype": fs.fstype,
            "mount_point": fs.mount_point,
            "hint": network_home_hint(),
        },
    )


def network_home_hint() -> str:
    return (
        "set HPCLAB_HOME (or --home) to a directory on this machine's local disk, one per machine; "
        f"if only this machine ever uses this home, set {ALLOW_NETWORK_HOME_ENV}=1 to use it anyway"
    )
