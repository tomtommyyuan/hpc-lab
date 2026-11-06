"""OpenSSH transport: the production ``Transport``, built on the system ``ssh`` and ``rsync``.

Design points (the failure taxonomy itself lives in :mod:`hpclab.adapters.base`):

* **Never interactive.** Every call runs with ``BatchMode=yes`` over a ``ControlMaster=auto``
  socket in ``control_dir``. The researcher opens the master once in a terminal with
  :meth:`OpenSSHTransport.login_command` (password + Duo/MFA); later calls multiplex over it
  and keep it alive (``ControlPersist=30m`` idle). When no master exists and non-interactive
  auth fails, calls raise ``AuthRequiredError`` whose details carry that command.
* **Exit status 255 is ssh's**, shared with the remote command. It is classified from stderr:
  auth / host key -> ``AuthRequiredError``; local configuration -> ``TransportError``
  (``precondition_failed``); network -> ``ConnectionLost`` whose ``maybe_executed`` is False
  only when ssh provably never started the remote command. Unrecognized 255s are treated as
  a lost connection that *may* have executed (safe for sbatch: it means "reconcile").
* **Output goes to temporary files, not pipes.** The first call may fork the background
  ``ControlPersist`` master; a ProxyJump/ProxyCommand child it started (and, on older
  OpenSSH, the master itself) keeps our stderr open, so waiting for pipe EOF would block
  for the master's lifetime.
* **stdin is a pipe or /dev/null, never inherited**: the MCP server's stdin is the protocol
  channel.
* **Remote commands are ``shlex.join(argv)``** after ``--`` and the host, so no argument can
  be parsed as an ssh option or re-split by the remote shell (which must be POSIX-like:
  bash/zsh/sh).
* **File helpers** go through :mod:`hpclab.adapters.remote_helper` and match LocalTransport.
* **Secrets**: key paths and sensitive option values are redacted from every error and log.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import posixpath
import re
import shlex
import signal
import stat as stat_mod
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hpclab.adapters.base import (
    AuthRequiredError,
    CommandFailed,
    CommandResult,
    ConnectionLost,
    RangeRead,
    StatInfo,
    TransportError,
    TransportTimeout,
)
from hpclab.adapters.remote_helper import RawResult, RemoteFileOps
from hpclab.domain.errors import ErrorCode, InvalidArgument

_log = logging.getLogger("hpclab.adapters.openssh")

CONTROL_PERSIST = "30m"
SERVER_ALIVE_INTERVAL = 15
SERVER_ALIVE_COUNT_MAX = 3
# Concurrent sessions per transport. sshd's default MaxSessions per connection is 10, and
# every hpclab process for this cluster (controller, CLI) multiplexes over the same master.
MAX_CONCURRENT_SESSIONS = 4

# AF_UNIX paths are limited to 104 (macOS) / 108 (Linux) bytes. ssh binds
# "<ControlPath>.<16 random chars>" before renaming it, and %C expands to 40 hex chars.
_SUN_PATH_BUDGET = 103
_CONTROL_NAME_LEN = len("/ssh-") + 40
_MASTER_TEMP_SUFFIX = 17
# Where a private short control directory is created when ``control_dir`` is too long or
# contains characters ssh/rsync would interpret (``%``, spaces). Patched in tests.
_SHORT_CONTROL_ROOT = Path("/tmp")  # noqa: S108 - dir is 0700 and ownership-checked before use
_SAFE_LOCAL_PATH = re.compile(r"[A-Za-z0-9_./+=@,-]+")
# Remote paths handed to rsync pass through the remote shell on rsync < 3.2.4, so only
# characters that are inert in every shell are accepted there.
_SAFE_REMOTE_PATH = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")
_OPTION_KEY = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_MANAGED_OPTIONS = frozenset(
    {
        "batchmode",
        "controlmaster",
        "controlpath",
        "controlpersist",
        "connecttimeout",
        "serveraliveinterval",
        "serveralivecountmax",
    }
)
_SENSITIVE_OPTION = re.compile(
    r"(?i)^(identityfile|certificatefile|identityagent|pkcs11provider|securitykeyprovider)(\s*=\s*|\s+)"
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'(?i)(load key\s+")[^"]*(")'), r"\1<redacted>\2"),
    (re.compile(r"(?i)(for key\s+')[^']*(')"), r"\1<redacted>\2"),
    (re.compile(r"(?i)((?:identity|certificate) file\s+)\S+"), r"\1<redacted>"),
    (re.compile(r"[^\s\"'=]*/\.ssh/[^\s\"':,]*"), "<redacted-path>"),
    (re.compile(r"[^\s\"'=]*\bid_(?:rsa|dsa|ecdsa|ed25519)(?:_sk)?[^\s\"':,]*"), "<redacted-key>"),
)

# rsync exit codes that mean the transport (not the transfer logic) failed.
_RSYNC_CONNECTION_CODES = frozenset({5, 10, 12, 30, 35, 255})
# rsync ran and reported per-file problems (23: partial transfer, 24: vanished source files):
# its stderr is about files, so it is never searched for ssh failure messages.
_RSYNC_PARTIAL_CODES = frozenset({23, 24})
# bash/sh/dash/tcsh ("rsync: command not found", "rsync: not found") and zsh wording.
_RSYNC_MISSING = re.compile(
    r"rsync: (?:command )?not found|command not found: rsync|rsync: no such file", re.IGNORECASE
)
# Deterministic remote-side failures that retrying cannot fix (checked before stream loss,
# because the remote rsync dying on them also closes the stream).
_RSYNC_STORAGE_FULL = re.compile(r"disk quota exceeded|no space left on device", re.IGNORECASE)
_RSYNC_SHELL_NOISE = re.compile(r"is your shell clean", re.IGNORECASE)
# ``--rsync-path`` runs ``mkdir -p <parent>`` first; its failure also surfaces as a closed stream.
_RSYNC_REMOTE_SETUP_FAILED = re.compile(r"^mkdir: ", re.MULTILINE)

FailureKind = Literal["auth", "host_key", "config", "network"]

# The classifier only trusts messages ssh prints when it gives up. ssh also prints *warnings*
# and carries on ('Unsupported option "x"', 'Load key ...: invalid format', unprotected key
# files, agent signing failures); if one of those decided the class, a later mid-session drop
# would be reported as "not executed". The remote command's stderr is part of the same stream,
# so patterns are specific to ssh's wording (a path containing "duo" is not a Duo prompt).
_HOST_KEY_PATTERNS = re.compile(
    r"host key verification failed|remote host identification has changed|host key for .* has changed",
    re.IGNORECASE,
)
_AUTH_PATTERNS = re.compile(
    "|".join(
        (
            r"permission denied \([a-z][a-z0-9,@.-]*\)",  # ssh lists methods; file errors say "(13)"
            r"too many authentication failures",
            r"no more authentication methods",
            r"authentication failed",
            r"keyboard-interactive",
            r"\bduo (?:two-factor|security|push|login|authentication|passcode)\b",
            r"\bfailed duo\b",
            r"control socket connect.*(?:no such file|connection refused)",
        )
    ),
    re.IGNORECASE,
)
_CONFIG_PATTERNS = re.compile(
    "|".join(
        (
            r"bad configuration option",  # also "terminating, N bad configuration options"
            r"bad owner or permissions",
            r"garbage at end of line",
            r"missing argument",
            r"can't open user config file",
            r"unable to negotiate",
            r"no matching (?:host key type|key exchange method|cipher|mac)",
            r"too long for unix domain socket",
            r"cannot execute command-line and remote command",  # RemoteCommand in ~/.ssh/config
        )
    ),
    re.IGNORECASE,
)
# Messages that can only come from failing *before* a session exists. ssh >= 9 reports a
# server-side close or reset during a session with the same words it uses during key
# exchange ("Connection closed by <ip> port 22", "Connection reset by ..."), and write errors
# as "Connection to <ip> port 22: No route to host", so those are deliberately absent: a VPN
# dropping mid-sbatch must stay "maybe executed".
_PRECONNECT_PATTERNS = re.compile(
    "|".join(
        (
            r"could not resolve hostname",
            r"name or service not known",
            r"nodename nor servname",
            r"temporary failure in name resolution",
            r"ssh: connect to host \S+ port \d+:",  # the direct TCP connect (any errno)
            r"connect failed: ",  # ProxyJump: the jump host could not reach the target
            r"stdio forwarding failed",
            r"banner exchange",
            r"kex_exchange_identification",
            r"ssh_exchange_identification",
            r"session open refused",  # mux: the server refused a new session (MaxSessions)
            r"session request failed",
        )
    ),
    re.IGNORECASE,
)
# Wording (samba rsync, openrsync, ssh) for "the ssh stream under rsync died".
_RSYNC_STREAM_LOST = re.compile(
    r"connection unexpectedly closed|unexpected end of file|broken pipe|connection (?:closed|reset) by|"
    r"not responding|closed by remote host|" + _PRECONNECT_PATTERNS.pattern,
    re.IGNORECASE,
)

_HINTS: dict[FailureKind, str] = {
    "auth": (
        "Run the login command in a terminal and complete the password/Duo prompt; hpclab reuses "
        "that connection (it stays open while hpclab polls, and for 30 minutes after the last use)."
    ),
    "host_key": (
        "The cluster's host key could not be verified (unknown or CHANGED). Run the login command in a "
        "terminal to inspect it; if it changed unexpectedly, confirm with your cluster admins before accepting."
    ),
    "config": "Fix the SSH configuration (ssh_options in the cluster overlay, or ~/.ssh/config), then retry.",
    "network": (
        "Could not keep a connection to the login node. Check your network/VPN (off-campus access often "
        "requires one) and retry."
    ),
}


@dataclass(frozen=True)
class SshFailure:
    """Classification of an ssh-level failure (exit 255)."""

    kind: FailureKind
    maybe_executed: bool


def redact(text: str) -> str:
    """Remove private key paths and similar secrets from ssh/rsync diagnostics."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def classify_ssh_failure(stderr: str, *, started: bool = False) -> SshFailure:
    """Classify an ssh exit-255 by its stderr.

    ``started`` means the remote side already produced stdout, so the command (or at least the
    login shell) ran: whatever the message, the failure happened afterwards.
    """
    if started:
        return SshFailure("network", maybe_executed=True)
    if _HOST_KEY_PATTERNS.search(stderr):
        return SshFailure("host_key", maybe_executed=False)
    if _AUTH_PATTERNS.search(stderr):
        return SshFailure("auth", maybe_executed=False)
    if _CONFIG_PATTERNS.search(stderr):
        return SshFailure("config", maybe_executed=False)
    if _PRECONNECT_PATTERNS.search(stderr):
        return SshFailure("network", maybe_executed=False)
    return SshFailure("network", maybe_executed=True)


def _redact_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        match = _SENSITIVE_OPTION.match(arg)
        out.append(arg[: match.end()] + "<redacted>" if match else redact(arg))
    return out


def _join_o_flags(options: list[str]) -> list[str]:
    """Merge a bare ``"-o"`` item with the item after it (``["-o", "Port=22"]``, the natural
    YAML spelling of ``ssh -o Port=22``). A trailing ``"-o"`` is left for validation to reject."""
    out: list[str] = []
    items = iter(options)
    for raw in items:
        if raw.strip() == "-o":
            value = next(items, None)
            out.append(raw if value is None or value.strip().startswith("-") else f"-o {value}")
        else:
            out.append(raw)
    return out


def _normalize_ssh_options(options: list[str]) -> list[str]:
    """Turn overlay ``ssh_options`` (``Key=Value``, ``Key Value``, ``-oKey=Value``,
    ``-o Key=Value`` or ``"-o", "Key=Value"`` as two items) into ``-o`` pairs.

    Other ssh flags are rejected (their arity is unknown), as are options hpclab manages
    itself: ssh uses the *first* value it sees, so they could not take effect anyway.
    """
    out: list[str] = []
    for raw in _join_o_flags(options):
        opt = raw.strip()
        if opt.startswith("-o"):
            opt = opt[2:].strip()
        key_match = _OPTION_KEY.match(opt)
        rest = opt[key_match.end() :] if key_match else ""
        if not key_match or not rest or rest[0] not in "= \t" or any(ord(c) < 32 and c != "\t" for c in opt):
            raise InvalidArgument(
                f"invalid ssh option {raw!r}: expected 'Key=Value' (e.g. 'ProxyJump=bastion')",
                details={"option": redact(raw)},
            )
        if key_match.group(0).lower() in _MANAGED_OPTIONS:
            raise InvalidArgument(
                f"ssh option {key_match.group(0)} is managed by hpclab and cannot be overridden",
                details={"option": key_match.group(0)},
            )
        out += ["-o", opt]
    return out


def _option_keys(normalized: list[str]) -> set[str]:
    """Lower-cased option names of ``_normalize_ssh_options`` output (``-o``/value pairs)."""
    return {m.group(0).lower() for opt in normalized[1::2] if (m := _OPTION_KEY.match(opt))}


def _check_word(name: str, value: str) -> str:
    if not value or value.startswith("-") or any(c.isspace() or ord(c) < 32 for c in value):
        raise InvalidArgument(f"invalid ssh {name} {value!r}", details={name: value})
    return value


def _resolve_control_dir(control_dir: Path) -> Path:
    """``control_dir`` if a socket path under it fits AF_UNIX and is inert for ssh/rsync
    (no ``%`` tokens, no spaces); otherwise a private short directory derived from it."""
    text = str(control_dir)
    fits = len(text) + _CONTROL_NAME_LEN + _MASTER_TEMP_SUFFIX <= _SUN_PATH_BUDGET
    if fits and _SAFE_LOCAL_PATH.fullmatch(text):
        return control_dir
    tag = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()[:12]
    return _SHORT_CONTROL_ROOT / f"hpclab-{os.getuid()}-{tag}"


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` 0700 and refuse to use it unless it is a real directory we own:
    a control socket in someone else's directory would let them ride our session."""

    def unusable(reason: str) -> TransportError:
        return TransportError(
            f"the SSH control directory is unusable: {reason}",
            code=ErrorCode.PRECONDITION_FAILED,
            retryable=False,
            details={"path": str(path)},
        )

    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(path)
        if not stat_mod.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise unusable("not a private directory owned by the current user")
        if st.st_mode & 0o077:
            os.chmod(path, 0o700)
    except OSError as exc:
        raise unusable(exc.strerror or type(exc).__name__) from exc


def _checked_argv(argv: list[str]) -> list[str]:
    """argv as strings. ``os.PathLike`` items are accepted like LocalTransport's subprocess
    does; anything else that is not a str (e.g. an int job id) is a caller bug."""
    if not argv:
        raise InvalidArgument("argv must not be empty")
    out: list[str] = []
    for arg in argv:
        text = os.fspath(arg) if isinstance(arg, os.PathLike) else arg
        if not isinstance(text, str):
            raise InvalidArgument(f"argv elements must be str, got {type(arg).__name__}")
        if "\x00" in text:
            raise InvalidArgument("argv elements must not contain NUL bytes")
        out.append(text)
    return out


def _rsync_remote_path(path: str) -> str:
    """Validate a remote path for rsync (see ``_SAFE_REMOTE_PATH``). ``~`` and ``$VAR`` are
    rejected rather than half-expanded: callers must resolve them first. A leading ``-``
    would be an option and a leading ``:`` makes ``host::path`` (rsync *daemon* syntax)."""
    if not _SAFE_REMOTE_PATH.fullmatch(path) or path.startswith(("-", ":")):
        raise InvalidArgument(
            f"remote path {path!r} contains characters that are unsafe for rsync; "
            "resolve '~'/'$USER' and avoid spaces or shell metacharacters",
            details={"path": path},
        )
    return path


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Kill the child's whole process group (rsync's ssh, ProxyJump helpers).

    A ``ControlPersist`` master is not affected: it ``daemon()``s into its own session.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    proc.wait()


class OpenSSHTransport:
    """``Transport`` over the system OpenSSH client (see the module docstring)."""

    def __init__(
        self,
        cluster_id: str,
        host: str,
        *,
        user: str | None = None,
        ssh_options: list[str] | None = None,
        control_dir: Path,
        ssh_bin: str = "ssh",
        rsync_bin: str = "rsync",
        connect_timeout: int = 10,
        remote_python: str = "python3",
    ) -> None:
        """``remote_python`` names the interpreter for the file helper on the login node
        (e.g. ``/usr/bin/python3``); if it is missing the coreutils fallback is used."""
        if isinstance(connect_timeout, bool) or not isinstance(connect_timeout, int) or connect_timeout <= 0:
            raise InvalidArgument(f"connect_timeout must be a positive int, got {connect_timeout!r}")
        self.cluster_id = cluster_id
        self.host = _check_word("host", host)
        self.user = _check_word("user", user) if user is not None else None
        self.ssh_bin = ssh_bin
        self.rsync_bin = rsync_bin
        self.connect_timeout = connect_timeout
        self._user_options = _normalize_ssh_options(list(ssh_options or []))
        self._user_sets_log_level = "loglevel" in _option_keys(self._user_options)
        self._control_dir = _resolve_control_dir(Path(control_dir).expanduser().absolute())
        self._sessions = threading.BoundedSemaphore(MAX_CONCURRENT_SESSIONS)
        self._remote_user: str | None = None
        self._files = RemoteFileOps(self._remote_exec, python=remote_python, label=cluster_id)

    # -- connection management ------------------------------------------------------
    @property
    def control_path(self) -> str:
        """The ``ControlPath`` template (``%C`` is expanded by ssh)."""
        return str(self._control_dir / "ssh-%C")

    @property
    def remote_python_available(self) -> bool | None:
        """Whether the login node ran the python file helper (None until the first file op).
        False means the coreutils fallback is in use and ``list_dir`` is unsupported."""
        return self._files.python_available

    def login_command(self) -> list[str]:
        """argv the *user* runs in a terminal to open the shared connection (password/Duo).

        It uses the same host, user and ControlPath as every hpclab call, so the master it
        leaves behind (``ControlPersist``) is the one hpclab multiplexes over. Creates the
        control directory as a side effect so the command works as printed.
        """
        self._ensure_control_dir()
        return [self.ssh_bin, *self._common_options(), "--", self.host, "true"]

    def check_master(self, *, timeout: float = 10.0) -> bool:
        """True if a shared connection is currently open (``ssh -O check``). Never raises."""
        return self._control_command("check", timeout=timeout)

    def stop_master(self, *, timeout: float = 10.0) -> bool:
        """Close the shared connection (``ssh -O exit``); True if one was closed. Never raises.

        The next call will need non-interactive auth or a new :meth:`login_command`.
        """
        return self._control_command("exit", timeout=timeout)

    def _control_command(self, command: str, *, timeout: float) -> bool:
        # Talks only to the local master socket and opens no remote session, so it does not
        # take a session slot: a long upload must not make the master look dead.
        what = f"ssh -O {command}"
        argv = [self.ssh_bin, "-O", command, *self._batch_options(), "--", self.host]
        try:
            raw = self._spawn_unlocked(argv, stdin=None, timeout=timeout, what=what, budget=timeout)
        except TransportError as exc:
            _log.debug("%s: %s failed: %s", self.cluster_id, what, exc.message)
            return False
        return raw.exit_code == 0

    def close(self) -> None:
        """Release this object. The shared master is deliberately left running: it holds the
        researcher's MFA session and will be reused by the next transport instance."""
        return None

    # -- commands --------------------------------------------------------------------------
    def remote_user(self) -> str:
        if self._remote_user is None:
            result = self.run(["id", "-un"], timeout=30.0)
            name = result.stdout.strip()
            if result.exit_code != 0 or not name:
                raise CommandFailed(
                    f"{self.cluster_id}: could not determine the remote user (id -un)", result
                )
            self._remote_user = name.splitlines()[-1].strip()
        return self._remote_user

    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        """See :class:`~hpclab.adapters.base.Transport`. ``stderr`` is :func:`redact`-ed: on a
        fresh connection it starts with ssh's own warnings (key paths), and callers copy it
        into error details."""
        words = _checked_argv(argv)
        start = time.monotonic()
        raw = self._ssh(shlex.join(words), stdin=stdin, timeout=timeout, what=words[0])
        return CommandResult(
            list(argv),
            raw.exit_code,
            raw.stdout.decode("utf-8", "replace"),
            redact(raw.stderr.decode("utf-8", "replace")),
            time.monotonic() - start,
        )

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        return self.run(["bash", "-s", "--", *(args or [])], timeout=timeout, stdin=script.encode("utf-8"))

    # -- file helpers (exact LocalTransport semantics, see remote_helper) -------------------------
    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        self._files.write_file(remote_path, content, mode=mode, timeout=timeout)

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        return self._files.read_range(remote_path, offset, length, timeout=timeout)

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        return self._files.read_tail(remote_path, length, timeout=timeout)

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        return self._files.stat(list(remote_paths), timeout=timeout)

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        return self._files.sha256(list(remote_paths), timeout=timeout)

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        return self._files.list_dir(remote_dir, max_entries=max_entries, timeout=timeout)

    def mkdirs(self, remote_dirs: list[str], *, timeout: float = 30.0) -> None:
        """``mkdir -p`` for each directory (not part of the Transport protocol)."""
        self._files.mkdirs(list(remote_dirs), timeout=timeout)

    # -- bulk transfer ----------------------------------------------------------------------
    def upload_dir(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        """Make ``remote_dir`` an exact copy of ``local_dir`` (``rsync -a --delete-after``),
        creating its parent directories. Not atomic: callers publish via staging + rename."""
        src = Path(local_dir)
        if not src.is_dir():
            raise FileNotFoundError(errno.ENOENT, "No such directory", local_dir)
        dest = _rsync_remote_path(remote_dir.rstrip("/") or "/")
        parent = posixpath.dirname(dest) or "."
        self._rsync(
            [
                "-a",
                "--delete-after",
                "-e",
                self._rsync_shell(),
                f"--rsync-path=mkdir -p {shlex.quote(parent)} && rsync",
                str(src.absolute()) + "/",
                f"{self._rsync_host()}:{dest.rstrip('/')}/",
            ],
            timeout=timeout,
            what="upload_dir",
        )

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        """Copy a remote file or directory tree to ``local_path`` (directories are mirrored,
        a file symlink is copied as its target, like LocalTransport). Missing -> FileNotFoundError."""
        source = _rsync_remote_path(remote_path)
        target = Path(local_path).absolute()
        remote = f"{self._rsync_host()}:{source}"
        if self._remote_is_dir(remote_path, timeout=min(timeout, 30.0)):
            args = ["-a", "--delete-after", "-e", self._rsync_shell(), remote.rstrip("/") + "/", f"{target}/"]
        else:
            args = ["-a", "-L", "-e", self._rsync_shell(), remote, str(target)]
        target.parent.mkdir(parents=True, exist_ok=True)
        self._rsync(args, timeout=timeout, what="download")

    def _remote_is_dir(self, remote_path: str, *, timeout: float) -> bool:
        """True for a directory (or a symlink to one); False for anything else that exists."""
        info = self.stat([remote_path], timeout=timeout)[remote_path]
        if not info.exists:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", remote_path)
        if info.kind == "symlink" and info.realpath:
            target = self.stat([info.realpath], timeout=timeout)[info.realpath]
            return target.kind == "dir"
        return info.kind == "dir"

    # -- ssh plumbing ----------------------------------------------------------------------------
    def _common_options(self) -> list[str]:
        opts = [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_path}",
            "-o", f"ControlPersist={CONTROL_PERSIST}",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", f"ServerAliveInterval={SERVER_ALIVE_INTERVAL}",
            "-o", f"ServerAliveCountMax={SERVER_ALIVE_COUNT_MAX}",
            *self._user_options,
        ]  # fmt: skip
        if not self._user_sets_log_level:
            # After the overlay's options (an explicit LogLevel there wins, e.g. DEBUG3 for
            # troubleshooting) but before ~/.ssh/config, where "LogLevel QUIET" (used to hide
            # banners) would silence the messages classify_ssh_failure depends on.
            opts += ["-o", "LogLevel=INFO"]
        if self.user:
            opts += ["-l", self.user]
        return opts

    def _batch_options(self) -> list[str]:
        # -T: no pty; -a/-x: never forward the agent or X11 to a shared login node.
        return ["-T", "-a", "-x", "-o", "BatchMode=yes", *self._common_options()]

    def _ensure_control_dir(self) -> None:
        # Checked on every call (a mkdir + lstat), not cached: if the directory vanishes while
        # the controller runs (an OS /tmp cleaner, a wiped ~/.hpclab/run), ssh would silently
        # stop multiplexing and login_command() could no longer create the master.
        _ensure_private_dir(self._control_dir)

    def _ssh(self, remote_command: str, *, stdin: bytes | None, timeout: float, what: str) -> RawResult:
        self._ensure_control_dir()
        argv = [self.ssh_bin, *self._batch_options(), "--", self.host, remote_command]
        raw = self._spawn(argv, stdin=stdin, timeout=timeout, what=what)
        if raw.exit_code == 255:
            stderr = raw.stderr.decode("utf-8", "replace")
            raise self._failure_error(classify_ssh_failure(stderr, started=bool(raw.stdout)), stderr, what)
        return raw

    def _remote_exec(self, argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult:
        return self._ssh(shlex.join(argv), stdin=stdin, timeout=timeout, what="file-helper")

    def _spawn(self, argv: list[str], *, stdin: bytes | None, timeout: float, what: str) -> RawResult:
        """Run a local ssh/rsync process within ``timeout`` seconds in total (waiting for a
        free session slot included). Raises only for local problems and timeouts."""
        deadline = time.monotonic() + timeout
        if not self._sessions.acquire(timeout=max(0.0, timeout)):
            busy = TransportTimeout(
                f"{self.cluster_id}: no free SSH session within {timeout:g}s ({what})",
                details={
                    "cluster_id": self.cluster_id,
                    "what": what,
                    "timeout_s": timeout,
                    "maybe_executed": False,
                },
            )
            busy.maybe_executed = False
            raise busy
        try:
            remaining = max(0.001, deadline - time.monotonic())
            return self._spawn_unlocked(argv, stdin=stdin, timeout=remaining, what=what, budget=timeout)
        finally:
            self._sessions.release()

    def _spawn_unlocked(
        self, argv: list[str], *, stdin: bytes | None, timeout: float, what: str, budget: float
    ) -> RawResult:
        start = time.monotonic()
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise TransportError(
                    f"{argv[0]!r} was not found on this machine",
                    code=ErrorCode.UNSUPPORTED,
                    retryable=False,
                    details={"program": argv[0]},
                ) from exc
            try:
                proc.communicate(input=stdin, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                _kill(proc)
                raise TransportTimeout(
                    f"{self.cluster_id}: {what} timed out after {budget:g}s",
                    details={
                        "cluster_id": self.cluster_id,
                        "what": what,
                        "timeout_s": budget,
                        "maybe_executed": True,
                    },
                ) from exc
            except BaseException:
                _kill(proc)
                raise
            out.seek(0)
            err.seek(0)
            raw = RawResult(proc.returncode, out.read(), err.read())
        _log.debug("%s: %s exit=%s in %.2fs", self.cluster_id, what, raw.exit_code, time.monotonic() - start)
        return raw

    def _failure_error(self, failure: SshFailure, stderr: str, what: str) -> TransportError:
        details: dict[str, Any] = {
            "cluster_id": self.cluster_id,
            "host": self.host,
            "what": what,
            "reason": failure.kind,
            "stderr": redact(stderr[-2000:]).strip(),
            "hint": _HINTS[failure.kind],
        }
        if failure.kind in ("auth", "host_key"):
            command = _redact_argv(self.login_command())
            details["login_command"] = command
            details["login_command_str"] = shlex.join(command)
            details["hint"] = f"{_HINTS[failure.kind]} Command: {shlex.join(command)}"
            return AuthRequiredError(
                f"{self.cluster_id}: SSH to {self.host} needs interactive authentication", details=details
            )
        if failure.kind == "config":
            return TransportError(
                f"{self.cluster_id}: SSH configuration error for {self.host}",
                code=ErrorCode.PRECONDITION_FAILED,
                retryable=False,
                details=details,
            )
        details["maybe_executed"] = failure.maybe_executed
        lost = ConnectionLost(
            f"{self.cluster_id}: connection to {self.host} failed ({what})", details=details
        )
        lost.maybe_executed = failure.maybe_executed
        return lost

    # -- rsync plumbing ---------------------------------------------------------------------------
    def _rsync_host(self) -> str:
        return f"[{self.host}]" if ":" in self.host else self.host

    def _rsync_shell(self) -> str:
        """The ``-e`` value: ssh with exactly our batch options.

        rsync implementations disagree on quoting inside ``-e`` (openrsync, the macOS default,
        splits on spaces only). When every token is shell-inert the plain command is used;
        otherwise a private wrapper script carrying the properly quoted command is.
        """
        self._ensure_control_dir()
        tokens = [self.ssh_bin, *self._batch_options()]
        if all(shlex.quote(t) == t for t in tokens):
            return " ".join(tokens)
        body = f'#!/bin/sh\nexec {shlex.join(tokens)} "$@"\n'
        tag = hashlib.sha256(body.encode("utf-8", "surrogateescape")).hexdigest()[:12]
        wrapper = self._control_dir / f"rsync-ssh-{tag}.sh"
        if not wrapper.exists():
            fd, tmp = tempfile.mkstemp(prefix=".rsync-ssh-", dir=str(self._control_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(tmp, 0o700)
            os.replace(tmp, wrapper)
        return str(wrapper)

    def _raise_rsync_setup_failure(self, stderr: str, what: str) -> None:
        """Raise for remote-environment problems rsync reports, which retrying cannot fix."""
        details = {"what": what, "stderr": redact(stderr[-2000:]).strip()}
        if _RSYNC_MISSING.search(stderr):
            raise TransportError(
                f"{self.cluster_id}: rsync is not available on {self.host}",
                code=ErrorCode.UNSUPPORTED,
                retryable=False,
                details=details,
            )
        if _RSYNC_SHELL_NOISE.search(stderr):
            raise TransportError(
                f"{self.cluster_id}: the login shell on {self.host} prints to stdout, which breaks rsync",
                code=ErrorCode.PRECONDITION_FAILED,
                retryable=False,
                details={
                    **details,
                    "hint": "A startup file (~/.bashrc, ~/.zshenv, ...) prints output in non-interactive "
                    "shells, e.g. an echo or 'module load'. Guard it with: [[ $- == *i* ]] || return",
                },
            )

    def _raise_rsync_transport_failure(self, exit_code: int, stderr: str, what: str) -> None:
        """Raise if an rsync failure was really ssh's; return for genuine transfer errors.

        Exit codes are not enough: openrsync exits 1 when its ssh child fails auth, so
        ssh-specific messages are recognized whatever the status. A partial transfer (23/24)
        means the connection worked and the messages are about files.
        """
        if exit_code in _RSYNC_PARTIAL_CODES:
            return
        failure = classify_ssh_failure(stderr)
        if failure.kind != "network":
            raise self._failure_error(failure, stderr, what)
        if exit_code in _RSYNC_CONNECTION_CODES or _RSYNC_STREAM_LOST.search(stderr):
            # A transfer that reached the network may have partially happened; both directions
            # are idempotent, so "maybe executed" only means "retrying is safe".
            raise self._failure_error(SshFailure("network", maybe_executed=True), stderr, what)

    def _rsync(self, args: list[str], *, timeout: float, what: str) -> None:
        argv = [self.rsync_bin, *args]
        start = time.monotonic()
        raw = self._spawn(argv, stdin=None, timeout=timeout, what=what)
        if raw.exit_code == 0:
            return
        stderr = raw.stderr.decode("utf-8", "replace")
        self._raise_rsync_setup_failure(stderr, what)
        storage_full = bool(_RSYNC_STORAGE_FULL.search(stderr))
        if not storage_full and not _RSYNC_REMOTE_SETUP_FAILED.search(stderr):
            self._raise_rsync_transport_failure(raw.exit_code, stderr, what)
        result = CommandResult(
            _redact_argv(argv),
            raw.exit_code,
            raw.stdout.decode("utf-8", "replace"),
            redact(stderr),
            time.monotonic() - start,
        )
        details: dict[str, Any] = {
            "what": what,
            "exit_code": raw.exit_code,
            "stderr": redact(stderr[-2000:]).strip(),
        }
        if storage_full:
            where = "on the cluster (check your quota)" if what == "upload_dir" else "on this machine"
            details["reason"] = "remote_storage_full" if what == "upload_dir" else "local_storage_full"
            details["hint"] = f"The destination filesystem {where} is full or over quota; free space first."
        raise CommandFailed(
            f"{self.cluster_id}: rsync failed during {what} (exit {raw.exit_code})",
            result,
            retryable=False if storage_full else None,
            details=details,
        )


__all__ = [
    "CONTROL_PERSIST",
    "MAX_CONCURRENT_SESSIONS",
    "OpenSSHTransport",
    "SshFailure",
    "classify_ssh_failure",
    "redact",
]
