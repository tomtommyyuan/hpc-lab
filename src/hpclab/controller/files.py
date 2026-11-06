"""Controller runtime files shared by the daemon and its clients.

* ``paths.lock``    – ``flock``-ed exclusively by the one live controller for its whole lifetime.
                      The kernel drops the lock when the process dies, so "lock held" is the
                      authoritative liveness test (a pidfile alone can be stale or reused).
* ``paths.pidfile`` – the lock holder's pid (plain decimal on the first line, which is all older
                      clients read), then ``host=``, ``boot=`` and ``start=`` lines naming the
                      machine, its boot session and the process's start time; written after
                      locking. A home shared by two machines (an NFS home) shares this file and,
                      where locks are propagated, the lock, so a pid is only signalled when this
                      machine recorded it and the process is still the one that did (see
                      ``pid_record_problem``).
* socket directory  – must be a real directory owned by this user and not writable by others;
                      otherwise another local user could plant a fake controller socket.
* ``logs/controller-spawn.log`` – stdout/stderr of controllers spawned by clients, so startup
                      crashes (import errors, bad config) are visible to the user.

This module is deliberately stdlib-only so importing the client stays cheap.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import os
import re
import socket
import stat
import struct
import sys
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

from hpclab.config import Paths
from hpclab.domain.errors import ErrorCode, HpcLabError


class InstanceLock:
    """Exclusive, non-blocking ``flock`` on a lock file, held until ``release()``/process exit.

    The descriptor is non-inheritable (Python default), so subprocesses the controller starts
    (ssh masters, simulator job runners) never keep the lock alive after the controller dies.
    ``fork()`` *without* exec (multiprocessing's "fork" start method, daemonizing libraries)
    would still duplicate it — and an flock lives as long as any duplicate — so forked children
    close their copies right away (see ``_forget_locks_in_child``)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, wait_s: float = 0.0) -> bool:
        """Try to take the lock, retrying for up to ``wait_s`` seconds. Returns False if another
        process holds it."""
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + max(0.0, wait_s)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        return False
                    time.sleep(0.02)
                    continue
                self._fd = fd
                _held_locks.add(self)
                return True
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        _held_locks.discard(self)
        if fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


_held_locks: weakref.WeakSet[InstanceLock] = weakref.WeakSet()


def _forget_locks_in_child() -> None:
    """``os.register_at_fork`` hook: a forked child drops its duplicates of held lock descriptors.

    Only ``close`` — never ``LOCK_UN``, which would release the *parent's* lock (flock state is
    shared by all duplicates of the open file description)."""
    for lock in list(_held_locks):
        fd, lock._fd = lock._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
    _held_locks.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_locks_in_child)


def probe_lock(path: Path) -> bool | None:
    """True if some process (including this one, through another descriptor) holds ``path``,
    False if none does (or the file does not exist), None when it cannot be told because the
    file or a directory above it is not accessible (see ``home_problem``).

    Probes with a shared non-blocking lock that is released immediately; never creates files,
    never raises and never blocks (O_NONBLOCK: opening a FIFO planted there would otherwise wait
    for a writer forever)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:  # EACCES, ENOTDIR (a file where the home should be), ELOOP, EIO, ...
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def lock_is_held(path: Path) -> bool:
    """True only when some process is known to hold ``path``. Never raises: a lock that cannot
    be probed (unusable home) counts as not held — callers that must tell "free" from "unknown"
    use ``probe_lock``."""
    return probe_lock(path) is True


def wait_lock_free(path: Path, timeout: float, *, poll_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while lock_is_held(path):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
    return True


def read_pid(pidfile: Path) -> int | None:
    try:
        text = pidfile.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        pid = int(text.splitlines()[0]) if text else 0
    except ValueError:
        return None
    return pid if pid > 0 else None


def write_pidfile(pidfile: Path, pid: int) -> None:
    """Atomically replace the pidfile (tmp + rename, mode 0600) with ``pid`` and the identity of
    this machine and of the process (see ``read_pid_record``)."""
    lines = [str(pid), f"host={this_host()}"]
    boot = boot_id()
    if boot:
        lines.append(f"boot={boot}")
    start = process_start_token(pid)
    if start:
        lines.append(f"start={start}")
    tmp = pidfile.with_name(f".{pidfile.name}.{pid}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, ("\n".join(lines) + "\n").encode("ascii"))
    finally:
        os.close(fd)
    os.replace(tmp, pidfile)


# ---------------------------------------------------------------------------
# Which machine and which process a pidfile names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PidRecord:
    """A pidfile's contents. ``host``/``boot``/``start`` are None in a pidfile written by an older
    controller (pid only), or where this platform cannot tell them."""

    pid: int
    host: str | None = None
    boot: str | None = None
    start: str | None = None


def read_pid_record(pidfile: Path) -> PidRecord | None:
    """The pidfile's pid and the identity recorded with it; None when there is no valid pid.
    Never raises."""
    pid = read_pid(pidfile)
    if pid is None:
        return None
    try:
        lines = pidfile.read_text(encoding="ascii").splitlines()[1:]
    except (OSError, UnicodeDecodeError):
        return PidRecord(pid)
    fields: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and value.strip():
            fields[key.strip()] = value.strip()
    return PidRecord(pid, fields.get("host"), fields.get("boot"), fields.get("start"))


def this_host() -> str:
    """This machine's name as a pidfile records it (printable ASCII, no spaces)."""
    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    return re.sub(r"[^\x21-\x7e]", "?", name) or "unknown"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _darwin_sysctl(name: str) -> bytes | None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return None
    buf = ctypes.create_string_buffer(max(1, size.value))
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
        return None
    return buf.raw[: size.value]


@functools.cache
def boot_id() -> str | None:
    """An id of this machine's current boot session (a new one after every restart), or None
    where the platform cannot tell. Two machines never share one, unlike host names."""
    try:
        if sys.platform.startswith("linux"):
            text = (_read_text("/proc/sys/kernel/random/boot_id") or "").strip()
            return text or None
        if sys.platform == "darwin":
            raw = _darwin_sysctl("kern.bootsessionuuid")
            value = raw.split(b"\0", 1)[0].decode("ascii", "replace").strip() if raw else ""
            return value or None
    except (OSError, AttributeError, ValueError):
        return None
    return None


_PROC_PIDTBSDINFO = 3  # macOS <sys/proc_info.h>
_PROC_BSDINFO_SIZE = 136  # sizeof(struct proc_bsdinfo)


def process_start_token(pid: int) -> str | None:
    """When process ``pid`` of this machine started, as an opaque token (equal tokens mean the
    same process, not merely the same pid), or None when it is gone, not ours to inspect, or the
    platform cannot tell. Never raises."""
    if pid <= 0:
        return None
    try:
        if sys.platform.startswith("linux"):
            text = _read_text(f"/proc/{pid}/stat")
            if not text or ")" not in text:
                return None
            fields = text[text.rindex(")") + 1 :].split()
            return fields[19] if len(fields) > 19 else None  # field 22: starttime (clock ticks)
        if sys.platform == "darwin":
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            buf = ctypes.create_string_buffer(_PROC_BSDINFO_SIZE)
            n = libc.proc_pidinfo(
                ctypes.c_int(pid),
                ctypes.c_int(_PROC_PIDTBSDINFO),
                ctypes.c_uint64(0),
                buf,
                ctypes.c_int(_PROC_BSDINFO_SIZE),
            )
            if n != _PROC_BSDINFO_SIZE:
                return None
            (got_pid,) = struct.unpack_from("=I", buf.raw, 12)
            sec, usec = struct.unpack_from("=QQ", buf.raw, 120)
            return f"{sec}.{usec:06d}" if got_pid == pid else None
    except (OSError, AttributeError, ValueError, struct.error):
        return None
    return None


def recorded_elsewhere(rec: PidRecord) -> bool:
    """True when another machine (or this one before a restart) wrote ``rec``: its pid means
    nothing here. The boot session decides when both are known (a laptop's host name changes
    with the network); otherwise the host name. A pid-only record is trusted (older controller)."""
    mine = boot_id()
    if rec.boot and mine:
        return rec.boot != mine
    return rec.host is not None and rec.host != this_host()


def pid_record_problem(rec: PidRecord) -> str | None:
    """Why the pid in ``rec`` must not be signalled from this machine, or None when it may be:
    it was recorded by another machine (or boot), or it now belongs to a process started at
    another time than the one that recorded it (a reused pid)."""
    if recorded_elsewhere(rec):
        if rec.host and rec.host != this_host():
            return (
                f"controller pid {rec.pid} was recorded by host {rec.host}, another machine sharing this "
                f"hpclab home; that pid means nothing on this machine ({this_host()})"
            )
        return (
            f"controller pid {rec.pid} was recorded in another boot session of host {rec.host or '?'} "
            "(another machine of that name sharing this hpclab home, or this one before it restarted)"
        )
    if rec.start:
        now = process_start_token(rec.pid)
        if now is not None and now != rec.start:
            return (
                f"pid {rec.pid} from the pidfile now belongs to a process started after the controller "
                "that recorded it (the pid was reused)"
            )
    return None


def remove_pidfile(pidfile: Path, pid: int | None = None) -> None:
    """Remove the pidfile; with ``pid`` only if it still names that pid."""
    if pid is not None and read_pid(pidfile) != pid:
        return
    with contextlib.suppress(FileNotFoundError):
        pidfile.unlink()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _os_reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _ownership_problem(st: os.stat_result) -> str | None:
    """Why an ``lstat`` result is not a real directory of the current user (None if it is)."""
    if stat.S_ISLNK(st.st_mode):
        return "is a symlink"
    if not stat.S_ISDIR(st.st_mode):
        return "is not a directory"
    if st.st_uid != os.getuid():
        return f"is owned by uid {st.st_uid}, not by you (uid {os.getuid()})"
    return None


def check_private_dir(path: Path, *, what: str = "controller socket directory") -> None:
    """Raise ``HpcLabError`` unless ``path`` is a real directory owned by the current user and not
    writable by group/others (a spoofed socket could otherwise be planted there). Never raises
    ``OSError``: an inaccessible path is reported as an ``HpcLabError`` too."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise HpcLabError(
            f"{what} {path} does not exist",
            code=ErrorCode.PRECONDITION_FAILED,
            details={"path": str(path)},
        ) from None
    except OSError as exc:
        raise HpcLabError(
            f"cannot inspect {what} {path}: {_os_reason(exc)}",
            code=ErrorCode.PRECONDITION_FAILED,
            details={"path": str(path)},
        ) from None
    problem = _ownership_problem(st)
    if problem is None and st.st_mode & 0o022:
        problem = f"is writable by other users (mode {stat.S_IMODE(st.st_mode):o})"
    if problem:
        raise HpcLabError(
            f"refusing to use {what} {path}: it {problem}; remove it and retry",
            code=ErrorCode.PRECONDITION_FAILED,
            details={"path": str(path)},
        )


def ensure_private_dir(path: Path, *, what: str) -> Path:
    """Create ``path`` (mode 0700; its parent must exist) or accept an existing one, then insist
    that it is a real directory owned by the current user and tighten it to 0700.

    Meant for per-user directories under shared, world-writable parents (``/tmp``, ``/var/tmp``)
    whose names are predictable: a directory or symlink planted there by another user is refused
    (``HpcLabError``), never used. Never raises ``OSError``."""
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(path, 0o700)
        st = os.lstat(path)
        problem = _ownership_problem(st)
        if problem is None and stat.S_IMODE(st.st_mode) != 0o700:
            os.chmod(path, 0o700)
    except OSError as exc:
        raise HpcLabError(
            f"cannot create or use {what} {path}: {_os_reason(exc)}",
            code=ErrorCode.PRECONDITION_FAILED,
            details={"path": str(path)},
        ) from None
    if problem:
        raise HpcLabError(
            f"refusing to use {what} {path}: it {problem}; remove it and retry",
            code=ErrorCode.PRECONDITION_FAILED,
            details={"path": str(path)},
        )
    return path


def home_problem(home: Path) -> str | None:
    """Why ``home`` cannot serve as an hpclab home, or None if it looks usable. A home that does
    not exist yet is usable when its nearest existing ancestor is a directory we can write.

    Best effort and never raises: used to turn an ``OSError`` deep in the client into the hint
    "HPCLAB_HOME <path> is not usable: <reason>"."""
    try:
        st = os.stat(home)
    except (FileNotFoundError, NotADirectoryError):
        return _ancestor_problem(home)
    except PermissionError:
        return "permission denied: a directory above it cannot be entered"
    except OSError as exc:
        return f"cannot access it: {_os_reason(exc)}"
    if not stat.S_ISDIR(st.st_mode):
        return "it is not a directory (a file is in its place)"
    if not os.access(home, os.R_OK | os.W_OK | os.X_OK):
        return (
            "permission denied: you lack read, write or search permission on it "
            f"(mode {stat.S_IMODE(st.st_mode):o}, or a read-only file system)"
        )
    run = home / "run"
    try:
        st_run = os.stat(run)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"cannot access {run}: {_os_reason(exc)}"
    if not stat.S_ISDIR(st_run.st_mode):
        return f"{run} is not a directory"
    if not os.access(run, os.R_OK | os.W_OK | os.X_OK):
        return f"permission denied on {run} (mode {stat.S_IMODE(st_run.st_mode):o})"
    return None


def _ancestor_problem(home: Path) -> str | None:
    for parent in home.parents:
        try:
            st = os.stat(parent)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            return f"cannot access {parent}: {_os_reason(exc)}"
        if not stat.S_ISDIR(st.st_mode):
            return f"{parent} is not a directory, so the home cannot be created below it"
        if not os.access(parent, os.W_OK | os.X_OK):
            return f"permission denied: it does not exist and cannot be created in {parent}"
        return None
    return None


def unusable_home_hint(home: Path, problem: str) -> str:
    return (
        f"HPCLAB_HOME {home} is not usable: {problem}. HPCLAB_HOME (or --home) must be a directory you "
        "own and can write: fix its permissions, or point HPCLAB_HOME at another directory (on "
        "AFS/Kerberos homes, renew your token or use local or scratch storage)"
    )


_MAX_SOCKET_PATH_BYTES = 100  # AF_UNIX sun_path: 104 bytes on macOS/BSD, 108 on Linux, incl. NUL


def socket_path(paths: Paths) -> Path:
    """The controller socket path that daemon and clients actually use.

    Same as ``paths.socket`` except that the AF_UNIX limit is checked in *bytes*:
    ``Paths.socket`` measures characters, so a home with non-ASCII (e.g. CJK) directory names can
    look short enough yet fail to bind ("AF_UNIX path too long"). Such homes get the same
    per-user ``/tmp`` fallback ``Paths.socket`` uses for long ASCII homes."""
    p = paths.socket
    if len(os.fsencode(str(p))) <= _MAX_SOCKET_PATH_BYTES:
        return p
    tag = hashlib.sha256(str(paths.home).encode()).hexdigest()[:10]
    return Path("/tmp") / f"hpclab-{os.getuid()}-{tag}" / "controller.sock"  # noqa: S108 - 0700 dir, checked


def spawn_log_path(paths: Paths) -> Path:
    return paths.log.parent / "controller-spawn.log"


def tail_lines(path: Path, *, max_bytes: int = 4096, max_lines: int = 20) -> list[str]:
    """Last lines of a text file (best effort; [] when unreadable)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably cut
    return lines[-max_lines:]
