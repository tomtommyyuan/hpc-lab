"""Remote file operations for transports that only have a shell on the login node.

Every file helper of the ``Transport`` contract (``stat``, ranged reads, hashing, listing,
atomic writes) must behave *exactly* like :class:`~hpclab.adapters.local.LocalTransport`,
because services are tested against the local/sim transports and then run over SSH. One
small stdlib-only Python program per call gives the exact semantics (``fstat`` on the open
file, ``realpath``, inode as generation marker) in a single round trip. A POSIX ``sh`` +
coreutils fallback covers login nodes without a usable ``python3``.

Wire protocol v1
----------------
Python path
    remote argv ``[python, "-I", "-S", "-c", BOOTSTRAP, "hpclab-helper"]``;
    stdin  = ``b"<len(helper)>\\n" + REMOTE_HELPER + header_json_line + payload``;
    header = ``{"v": 1, "op": ..., "args": {...}}`` (ASCII JSON, one line);
    stdout = ``<anything the login shell printed> + MARKER + b"\\n" + reply_json_line``;
    reply  = ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": {"kind": ..., ...}}``.

    The bootstrap reads a length-prefixed script from stdin, so the *same* stdin can carry
    arguments (no ARG_MAX limit, no quoting of user paths) and a raw binary payload for
    ``write_file`` after it. Reads come back base64-encoded inside the JSON reply.
    ``-I -S`` keep the user's ``PYTHONPATH``/``sitecustomize``/``.pth`` files (slow on NFS,
    sometimes chatty) out of the helper.

Fallback path (no python3)
    remote argv ``["sh", "-c", FALLBACK_SH, "hpclab-fallback", op, *args]``; stdin carries the
    payload for ``write_file``; stdout = ``<noise> + MARKER + b"\\n" + op records``.

Which operations need python3
    ``list_dir`` (an ordered, bounded, symlink-aware walk has no portable coreutils
    equivalent). ``stat``, ``read_range``, ``read_tail``, ``sha256``, ``write_file`` and
    ``mkdirs`` all work through the fallback, which accepts GNU coreutils and BSD ``stat``.

The ``MARKER`` line makes parsing immune to login-shell noise (``.bashrc`` banners, module
messages) printed to stdout before the helper runs.
"""

from __future__ import annotations

import base64
import binascii
import errno
import json
import logging
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from hpclab.adapters.base import ConnectionLost, RangeRead, StatInfo, TransportError
from hpclab.domain.errors import ErrorCode

_log = logging.getLogger("hpclab.adapters.remote_helper")

PROTOCOL_VERSION = 1
MARKER = b"@@HPCLAB-HELPER-V1@@"

# Ops that exist in the helper. ``FALLBACK_OPS`` work without python3 on the remote side.
PYTHON_OPS = frozenset({"stat", "read_range", "read_tail", "sha256", "list_dir", "write_file", "mkdirs"})
FALLBACK_OPS = frozenset({"stat", "read_range", "read_tail", "sha256", "write_file", "mkdirs"})

BOOTSTRAP = (
    "import sys;n=int(sys.stdin.buffer.readline());"
    "exec(compile(sys.stdin.buffer.read(n),'<hpclab-helper>','exec'))"
)

# The helper runs on login nodes that may only have an old system python (RHEL 8 ships 3.6):
# stdlib only, no f-strings, no dataclasses, no walrus. Paths are handled as bytes with
# utf-8/surrogateescape so non-UTF-8 names and C locales on old pythons both work.
REMOTE_HELPER = r"""# hpclab remote helper (protocol v1). Stdlib only; Python >= 3.5 (bytes paths in mkstemp).
import base64
import errno
import hashlib
import json
import os
import sys
import tempfile

MARK = b"@@HPCLAB-HELPER-V1@@"
PROTOCOL = 1


class HelperFailure(Exception):
    def __init__(self, kind, message):
        Exception.__init__(self, message)
        self.kind = kind
        self.message = message


def _b(text):
    return text.encode("utf-8", "surrogateescape")


def _s(raw):
    return raw.decode("utf-8", "surrogateescape")


def _kind(path):
    if os.path.islink(path):
        return "symlink"
    if os.path.isdir(path):
        return "dir"
    if os.path.isfile(path):
        return "file"
    return "other"


def _info(path, st, kind):
    return {
        "kind": kind,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "inode": st.st_ino,
        "realpath": _s(os.path.realpath(path)),
    }


def op_stat(args, payload):
    out = []
    for text in args["paths"]:
        path = _b(text)
        if not os.path.lexists(path):
            out.append(None)
            continue
        try:
            st = os.stat(path)
        except OSError:
            out.append(None)
            continue
        out.append(_info(path, st, _kind(path)))
    return out


def _read(path, offset, length):
    with open(path, "rb") as fh:
        st = os.fstat(fh.fileno())
        size = st.st_size
        off = max(0, min(offset, size))
        fh.seek(off)
        data = fh.read(max(0, length))
    return {
        "offset": off,
        "size": size,
        "mtime": st.st_mtime,
        "inode": st.st_ino,
        "eof": off + len(data) >= size,
        "data": base64.b64encode(data).decode("ascii"),
    }


def op_read_range(args, payload):
    return _read(_b(args["path"]), args["offset"], args["length"])


def op_read_tail(args, payload):
    size = os.stat(_b(args["stat_path"])).st_size
    return _read(_b(args["path"]), max(0, size - args["length"]), args["length"])


def op_sha256(args, payload):
    out = []
    for text in args["paths"]:
        try:
            digest = hashlib.sha256()
            with open(_b(text), "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
            out.append(digest.hexdigest())
        except OSError:
            out.append(None)
    return out


def op_list_dir(args, payload):
    root = _b(args["path"])
    limit = args["max_entries"]
    if not os.path.isdir(root):
        raise HelperFailure("not_a_directory", "not a directory")
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort(key=_s)
        for name in sorted(filenames, key=_s):
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            item = _info(full, st, "symlink" if os.path.islink(full) else "file")
            item["path"] = _s(full)
            out.append(item)
            if len(out) >= limit:
                return out
    return out


def op_write_file(args, payload):
    if len(payload) != args["size"]:
        raise HelperFailure(
            "truncated", "payload has %d bytes, expected %d" % (len(payload), args["size"])
        )
    path = _b(args["path"])
    parent = os.path.dirname(path) or b"."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=b".hpclab-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, args["mode"])
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"size": len(payload)}


def op_mkdirs(args, payload):
    for text in args["paths"]:
        os.makedirs(_b(text), exist_ok=True)
    return {}


OPS = {
    "stat": op_stat,
    "read_range": op_read_range,
    "read_tail": op_read_tail,
    "sha256": op_sha256,
    "list_dir": op_list_dir,
    "write_file": op_write_file,
    "mkdirs": op_mkdirs,
}


def _os_error(exc):
    name = errno.errorcode.get(exc.errno) if exc.errno else None
    return {"kind": "os", "errno": name, "message": exc.strerror or str(exc)}


def _handle(header, payload):
    if header.get("v") != PROTOCOL:
        return {"ok": False, "error": {"kind": "protocol", "message": "unsupported protocol version"}}
    handler = OPS.get(header.get("op"))
    if handler is None:
        return {"ok": False, "error": {"kind": "protocol", "message": "unknown op"}}
    try:
        return {"ok": True, "result": handler(header.get("args") or {}, payload)}
    except HelperFailure as exc:
        return {"ok": False, "error": {"kind": exc.kind, "message": exc.message}}
    except OSError as exc:
        return {"ok": False, "error": _os_error(exc)}
    except Exception as exc:  # report, never crash without a reply
        return {"ok": False, "error": {"kind": "internal", "message": "%s: %s" % (type(exc).__name__, exc)}}


def main():
    stdin = sys.stdin.buffer
    try:
        header = json.loads(stdin.readline().decode("ascii"))
    except ValueError:
        header = {}
    payload = stdin.read() if header.get("op") == "write_file" else b""
    reply = _handle(header, payload)
    body = json.dumps(reply, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    out = sys.stdout.buffer
    out.write(MARK + b"\n" + body + b"\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

# POSIX sh + coreutils fallback. Record formats (one line each, fields separated by one space):
#   stat:        "M" (missing) | "F <kind> <realpath-b64> <size> <mtime-sec> <inode> <raw-mtime>"
#   sha256:      "N" (unreadable) | "H <hex>"
#   read_*:      "E <ERRNO-NAME>" | "D <offset> <size> <mtime-sec> <inode> <raw-mtime>" + raw bytes
#   write_file:  "W <size>" | "T <size>" (truncated payload) | "E <stage> <message-b64>"
#   mkdirs:      "K" | "E <stage> <message-b64>"
# <raw-mtime> is GNU ``%y`` ("2026-09-27 10:00:00.123456789 +0000") or BSD ``%Fm``
# ("1790561021.123456789"); only its nanosecond fraction is used.
FALLBACK_SH = r"""
if stat -c %s / >/dev/null 2>&1; then GNU=1; else GNU=0; fi
meta() {
  if [ "$GNU" = 1 ]; then stat -L -c "%s %Y %i %y" -- "$1" 2>/dev/null
  else stat -L -f "%z %m %i %Fm" -- "$1" 2>/dev/null; fi
}
real() {
  r=$(readlink -f -- "$1" 2>/dev/null && printf x) || r=$(realpath -- "$1" 2>/dev/null && printf x) || return 1
  r=${r%x}
  r=${r%?}
  printf "%s" "$r" | base64 | tr -d "\n"
}
b64() { printf "%s" "$1" | base64 | tr -d "\n"; }
op=$1
shift
printf "%s\n" "@@HPCLAB-HELPER-V1@@"
case "$op" in
stat)
  for p in "$@"; do
    if [ -e "$p" ] && m=$(meta "$p") && r=$(real "$p"); then
      if [ -L "$p" ]; then k=symlink
      elif [ -d "$p" ]; then k=dir
      elif [ -f "$p" ]; then k=file
      else k=other; fi
      printf "F %s %s %s\n" "$k" "$r" "$m"
    else
      printf "M\n"
    fi
  done
  ;;
sha256)
  if command -v sha256sum >/dev/null 2>&1; then h=sha256sum; else h="shasum -a 256"; fi
  for p in "$@"; do
    if d=$( { $h < "$p"; } 2>/dev/null ); then printf "H %s\n" "${d%% *}"; else printf "N\n"; fi
  done
  ;;
read_range|read_tail)
  p=$1; a=$2; n=$3
  if [ ! -e "$p" ]; then printf "E ENOENT\n"; exit 0; fi
  if [ -d "$p" ]; then printf "E EISDIR\n"; exit 0; fi
  if [ ! -r "$p" ]; then printf "E EACCES\n"; exit 0; fi
  if ! m=$(meta "$p"); then printf "E ENOENT\n"; exit 0; fi
  size=${m%% *}
  if [ "$op" = read_tail ]; then a=$((size - n)); fi
  if [ "$a" -lt 0 ]; then a=0; fi
  if [ "$a" -gt "$size" ]; then a=$size; fi
  if [ "$n" -lt 0 ]; then n=0; fi
  printf "D %s %s\n" "$a" "$m"
  if [ "$n" -gt 0 ]; then tail -c +$((a + 1)) -- "$p" | head -c "$n"; fi
  ;;
write_file)
  p=$1; mode=$2; want=$3
  d=${p%/*}
  if [ "$d" = "$p" ]; then d=.; elif [ -z "$d" ]; then d=/; fi
  if ! e=$(mkdir -p -- "$d" 2>&1); then printf "E mkdir %s\n" "$(b64 "$e")"; exit 0; fi
  if [ -d "$p" ]; then printf "E rename %s\n" "$(b64 "Is a directory")"; exit 0; fi
  if ! t=$(mktemp "$d/.hpclab-XXXXXXXXXX" 2>&1); then printf "E mktemp %s\n" "$(b64 "$t")"; exit 0; fi
  if ! e=$(cat > "$t" 2>&1); then rm -f -- "$t"; printf "E write %s\n" "$(b64 "$e")"; exit 0; fi
  got=$(wc -c < "$t" | tr -d " ")
  if [ "$got" != "$want" ]; then rm -f -- "$t"; printf "T %s\n" "$got"; exit 0; fi
  if ! e=$( { chmod "$mode" "$t" && mv -f -- "$t" "$p"; } 2>&1 ); then
    rm -f -- "$t"; printf "E rename %s\n" "$(b64 "$e")"; exit 0
  fi
  printf "W %s\n" "$got"
  ;;
mkdirs)
  if e=$(mkdir -p -- "$@" 2>&1); then printf "K\n"; else printf "E mkdir %s\n" "$(b64 "$e")"; fi
  ;;
*)
  printf "X unknown-op\n"
  ;;
esac
"""

_HELPER_BYTES = REMOTE_HELPER.encode("ascii")
_STAT_KINDS = frozenset({"file", "dir", "symlink", "other"})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FRACTION = re.compile(r"\.(\d+)")
# Keep each fallback command line well below Linux MAX_ARG_STRLEN (128 KiB for one argv
# element; the whole remote command is one argument to the login shell).
_FALLBACK_ARG_BUDGET = 64 * 1024
# Shell arithmetic in the fallback is signed 64-bit; clamp offsets/lengths well inside it.
_MAX_SHELL_INT = 1 << 60
_MESSAGE_ERRNO = (
    ("permission denied", errno.EACCES),
    ("read-only file system", errno.EROFS),
    ("file exists", errno.EEXIST),
    ("not a directory", errno.ENOTDIR),
    ("is a directory", errno.EISDIR),
    ("no space left", errno.ENOSPC),
    ("quota exceeded", errno.EDQUOT),
    ("no such file", errno.ENOENT),
)

T = TypeVar("T")


@dataclass(frozen=True)
class RawResult:
    """Raw outcome of one remote command (exit status and undecoded output)."""

    exit_code: int
    stdout: bytes
    stderr: bytes


class RemoteExecutor(Protocol):
    """Runs ``argv`` on the remote side and returns its raw result for *any* exit status.

    Transport-level failures (auth, timeout, dropped connection) are raised as
    ``TransportError`` subclasses by the executor itself.
    """

    def __call__(self, argv: list[str], *, stdin: bytes | None, timeout: float) -> RawResult: ...


class RemoteHelperError(TransportError):
    """The remote helper misbehaved; the details carry the exit status and a stderr tail.

    A malformed or failed reply is a bug or version skew: not retryable. *No* reply at all is
    retryable: busy login nodes refuse forks and per-user watchdogs kill processes.
    """

    code = ErrorCode.UNAVAILABLE

    def __init__(
        self, message: str, *, retryable: bool = False, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, retryable=retryable, details=details)


class _PythonUnusable(Exception):
    """Internal signal: the remote python cannot run the helper; use the fallback."""


def pathlib_form(path: str) -> str:
    """Normalize like ``str(pathlib.PurePosixPath(path))``.

    LocalTransport passes some paths through ``pathlib.Path`` (dropping ``.`` components,
    duplicate and trailing slashes); the remote side must see the same path to match it.
    """
    if path.startswith("//") and not path.startswith("///"):
        root = "//"
    elif path.startswith("/"):
        root = "/"
    else:
        root = ""
    parts = [p for p in path.split("/") if p and p != "."]
    return (root + "/".join(parts)) or "."


def split_after_marker(stdout: bytes) -> bytes | None:
    """Return the bytes after the first ``MARKER`` line, or None if the marker is absent."""
    idx = stdout.find(MARKER + b"\n")
    if idx < 0:
        return None
    return stdout[idx + len(MARKER) + 1 :]


def _tail_text(data: bytes, limit: int = 2000) -> str:
    return data[-limit:].decode("utf-8", "replace")


def _mtime(sec: str, raw: str) -> float:
    """Rebuild ``st_mtime`` exactly as CPython does: ``sec + nsec * 1e-9``."""
    match = _FRACTION.search(raw)
    nsec = int(match.group(1)[:9].ljust(9, "0")) if match else 0
    return int(sec) + nsec * 1e-9


def _errno_from_message(message: str) -> int:
    lowered = message.lower()
    for needle, code in _MESSAGE_ERRNO:
        if needle in lowered:
            return code
    return errno.EIO


def _os_error(errno_name: str | None, message: str, path: str) -> OSError:
    """Build the local ``OSError`` subclass (FileNotFoundError, ...) for a remote errno *name*.

    Errno numbers differ between Linux and macOS, so the wire carries names.
    """
    code = getattr(errno, errno_name, None) if errno_name else None
    if isinstance(code, int):
        return OSError(code, message, path)
    return OSError(message)


def _b64decode(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise RemoteHelperError("remote helper returned malformed base64") from exc


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteHelperError("remote helper returned a malformed integer field")
    return value


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RemoteHelperError("remote helper returned a malformed number field")
    return float(value)


def _chunk_bounds(items: list[str], budget: int = _FALLBACK_ARG_BUDGET) -> list[tuple[int, int]]:
    """``[lo, hi)`` index ranges whose shell-quoted size stays under ``budget`` bytes."""
    bounds: list[tuple[int, int]] = []
    lo, used = 0, 0
    for i, item in enumerate(items):
        cost = len(shlex.quote(item).encode("utf-8", "surrogateescape")) + 1
        if i > lo and used + cost > budget:
            bounds.append((lo, i))
            lo, used = i, 0
        used += cost
    if lo < len(items):
        bounds.append((lo, len(items)))
    return bounds


def _check_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return max(-_MAX_SHELL_INT, min(value, _MAX_SHELL_INT))


class RemoteFileOps:
    """LocalTransport-equivalent file operations executed through a :class:`RemoteExecutor`.

    The first call probes the remote python implicitly; if it is missing or unusable the
    instance switches to the coreutils fallback for the rest of its life (logged once).
    ``list_dir`` then raises ``TransportError(code=unsupported)``.
    """

    def __init__(self, execute: RemoteExecutor, *, python: str = "python3", label: str = "remote") -> None:
        self._execute = execute
        self._python = python
        self._label = label
        self._python_ok: bool | None = None

    @property
    def python_available(self) -> bool | None:
        """True/False once known; None before the first helper call."""
        return self._python_ok

    # -- public operations (mirror LocalTransport) -----------------------------------
    def stat(self, paths: list[str], *, timeout: float) -> dict[str, StatInfo]:
        if not paths:
            return {}
        norm = [pathlib_form(p) for p in paths]

        def from_python(result: Any) -> dict[str, StatInfo]:
            records = self._expect_list(result, len(paths))
            return {p: self._stat_record(p, rec) for p, rec in zip(paths, records, strict=True)}

        return self._dispatch(
            "stat",
            {"paths": norm},
            timeout=timeout,
            convert=from_python,
            fallback=lambda: self._fb_stat(paths, norm, timeout),
        )

    def read_range(self, path: str, offset: int, length: int, *, timeout: float) -> RangeRead:
        offset = _check_int("offset", offset)
        length = _check_int("length", length)
        norm = pathlib_form(path)
        return self._dispatch(
            "read_range",
            {"path": norm, "offset": offset, "length": length},
            timeout=timeout,
            convert=lambda result: self._range_record(path, result),
            fallback=lambda: self._fb_read("read_range", path, norm, offset, length, timeout),
            error_path=path,
        )

    def read_tail(self, path: str, length: int, *, timeout: float) -> RangeRead:
        length = _check_int("length", length)
        norm = pathlib_form(path)
        return self._dispatch(
            "read_tail",
            # LocalTransport stats the raw path but opens the pathlib-normalized one.
            {"path": norm, "stat_path": path, "length": length},
            timeout=timeout,
            convert=lambda result: self._range_record(path, result),
            fallback=lambda: self._fb_read("read_tail", path, norm, 0, length, timeout),
            error_path=path,
        )

    def sha256(self, paths: list[str], *, timeout: float) -> dict[str, str | None]:
        if not paths:
            return {}

        def from_python(result: Any) -> dict[str, str | None]:
            hexes = self._expect_list(result, len(paths))
            return {p: self._digest(h) for p, h in zip(paths, hexes, strict=True)}

        return self._dispatch(
            "sha256",
            {"paths": list(paths)},
            timeout=timeout,
            convert=from_python,
            fallback=lambda: self._fb_sha256(paths, timeout),
        )

    def list_dir(self, path: str, *, max_entries: int, timeout: float) -> list[StatInfo]:
        max_entries = _check_int("max_entries", max_entries)

        def from_python(result: Any) -> list[StatInfo]:
            if not isinstance(result, list):
                raise RemoteHelperError("remote helper returned a malformed listing")
            return [self._listing_record(rec) for rec in result]

        return self._dispatch(
            "list_dir",
            {"path": pathlib_form(path), "max_entries": max_entries},
            timeout=timeout,
            convert=from_python,
            fallback=None,
            error_path=path,
        )

    def write_file(self, path: str, content: bytes, *, mode: int, timeout: float) -> None:
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
            raise ValueError(f"mode must be an int in [0, 0o7777], got {mode!r}")
        data = bytes(content)
        norm = pathlib_form(path)
        self._dispatch(
            "write_file",
            {"path": norm, "mode": mode, "size": len(data)},
            timeout=timeout,
            payload=data,
            convert=lambda result: None,
            fallback=lambda: self._fb_write(path, norm, data, mode, timeout),
            error_path=path,
        )

    def mkdirs(self, paths: list[str], *, timeout: float) -> None:
        if not paths:
            return
        self._dispatch(
            "mkdirs",
            {"paths": list(paths)},
            timeout=timeout,
            convert=lambda result: None,
            fallback=lambda: self._fb_mkdirs(paths, timeout),
            error_path=paths[0],
        )

    # -- dispatch ------------------------------------------------------------------
    def _dispatch(
        self,
        op: str,
        args: dict[str, Any],
        *,
        timeout: float,
        convert: Callable[[Any], T],
        fallback: Callable[[], T] | None,
        payload: bytes = b"",
        error_path: str = "",
    ) -> T:
        # An op without a fallback always tries python: an earlier failed probe (say, a module
        # system hiccup hiding python3) must not disable it for the controller's lifetime.
        if self._python_ok is not False or fallback is None:
            try:
                result = self._call_python(op, args, payload, timeout, error_path)
            except _PythonUnusable as exc:
                if self._python_ok is not False:  # log the transition once, not every retry
                    _log.warning(
                        "%s: remote %s is unusable (%s); using the coreutils fallback (list_dir unavailable)",
                        self._label,
                        self._python,
                        exc,
                    )
                self._python_ok = False
            else:
                self._python_ok = True
                return convert(result)
        if fallback is None:
            raise TransportError(
                f"{op} needs python3 on the login node, which is not available",
                code=ErrorCode.UNSUPPORTED,
                retryable=False,
                details={
                    "op": op,
                    "python": self._python,
                    "hint": "install python3 or set the remote python",
                },
            )
        return fallback()

    def _call_python(
        self, op: str, args: dict[str, Any], payload: bytes, timeout: float, error_path: str
    ) -> Any:
        header = json.dumps(
            {"v": PROTOCOL_VERSION, "op": op, "args": args}, ensure_ascii=True, separators=(",", ":")
        )
        stdin = b"%d\n" % len(_HELPER_BYTES) + _HELPER_BYTES + header.encode("ascii") + b"\n" + payload
        raw = self._execute(
            [self._python, "-I", "-S", "-c", BOOTSTRAP, "hpclab-helper"], stdin=stdin, timeout=timeout
        )
        body = split_after_marker(raw.stdout)
        if body is None:
            if _python_unusable(raw):
                raise _PythonUnusable(f"exit {raw.exit_code}: {_tail_text(raw.stderr, 200).strip()}")
            raise _no_reply(f"remote helper produced no reply for {op}", op, raw)
        try:
            reply = json.loads(body.split(b"\n", 1)[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RemoteHelperError(
                f"remote helper returned a malformed reply for {op}", details={"op": op}
            ) from exc
        if not isinstance(reply, dict):
            raise RemoteHelperError(f"remote helper returned a malformed reply for {op}", details={"op": op})
        if reply.get("ok") is True:
            return reply.get("result")
        raise self._helper_error(op, reply.get("error"), error_path)

    @staticmethod
    def _helper_error(op: str, error: Any, path: str) -> Exception:
        err = error if isinstance(error, dict) else {}
        kind = err.get("kind")
        message = str(err.get("message") or "remote error")
        if kind == "os":
            name = err.get("errno")
            return _os_error(name if isinstance(name, str) else None, message, path)
        if kind == "not_a_directory":
            return FileNotFoundError(path)  # exactly what LocalTransport.list_dir raises
        if kind == "truncated":
            return _truncated_upload(path, op)
        return RemoteHelperError(
            f"remote helper failed during {op}: {message}", details={"op": op, "kind": kind}
        )

    # -- python result conversion --------------------------------------------------------
    @staticmethod
    def _expect_list(result: Any, n: int) -> list[Any]:
        if not isinstance(result, list) or len(result) != n:
            raise RemoteHelperError("remote helper returned the wrong number of records")
        return result

    @staticmethod
    def _stat_record(path: str, rec: Any) -> StatInfo:
        if rec is None:
            return StatInfo(path=path, exists=False, kind="missing")
        if not isinstance(rec, dict) or rec.get("kind") not in _STAT_KINDS:
            raise RemoteHelperError("remote helper returned a malformed stat record")
        realpath = rec.get("realpath")
        return StatInfo(
            path=path,
            exists=True,
            kind=rec["kind"],
            size=_int(rec.get("size")),
            mtime=_float(rec.get("mtime")),
            inode=_int(rec.get("inode")),
            realpath=realpath if isinstance(realpath, str) else None,
        )

    @classmethod
    def _listing_record(cls, rec: Any) -> StatInfo:
        if not isinstance(rec, dict) or not isinstance(rec.get("path"), str):
            raise RemoteHelperError("remote helper returned a malformed listing record")
        return cls._stat_record(rec["path"], rec)

    @staticmethod
    def _range_record(path: str, rec: Any) -> RangeRead:
        if not isinstance(rec, dict) or not isinstance(rec.get("data"), str):
            raise RemoteHelperError("remote helper returned a malformed read record")
        return RangeRead(
            path=path,
            offset=_int(rec.get("offset")),
            data=_b64decode(rec["data"]),
            size=_int(rec.get("size")),
            mtime=_float(rec.get("mtime")),
            inode=_int(rec.get("inode")),
            eof=bool(rec.get("eof")),
        )

    @staticmethod
    def _digest(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise RemoteHelperError("remote helper returned a malformed digest")
        return "sha256:" + value

    # -- coreutils fallback ------------------------------------------------------------
    def _fallback(self, op: str, args: list[str], timeout: float, stdin: bytes | None = None) -> bytes:
        raw = self._execute(
            ["sh", "-c", FALLBACK_SH, "hpclab-fallback", op, *args], stdin=stdin, timeout=timeout
        )
        body = split_after_marker(raw.stdout)
        if body is None:
            raise _no_reply(f"coreutils fallback produced no reply for {op}", op, raw)
        return body

    def _fallback_lines(self, op: str, args: list[str], timeout: float, expected: int) -> list[str]:
        lines = self._fallback(op, args, timeout).decode("ascii", "replace").split("\n")
        if len(lines) < expected:
            raise RemoteHelperError(f"coreutils fallback returned too few records for {op}")
        return lines[:expected]

    def _fb_stat(self, paths: list[str], norm: list[str], timeout: float) -> dict[str, StatInfo]:
        out: dict[str, StatInfo] = {}
        for lo, hi in _chunk_bounds(norm):
            lines = self._fallback_lines("stat", norm[lo:hi], timeout, hi - lo)
            for path, line in zip(paths[lo:hi], lines, strict=True):
                out[path] = self._fb_stat_line(path, line)
        return out

    @staticmethod
    def _fb_stat_line(path: str, line: str) -> StatInfo:
        if line == "M":
            return StatInfo(path=path, exists=False, kind="missing")
        fields = line.split(" ", 6)
        if len(fields) != 7 or fields[0] != "F" or fields[1] not in _STAT_KINDS:
            raise RemoteHelperError("coreutils fallback returned a malformed stat record")
        _, kind, real_b64, size, sec, inode, raw_time = fields
        try:
            return StatInfo(
                path=path,
                exists=True,
                kind=kind,  # type: ignore[arg-type]
                size=int(size),
                mtime=_mtime(sec, raw_time),
                inode=int(inode),
                realpath=_b64decode(real_b64).decode("utf-8", "surrogateescape"),
            )
        except ValueError as exc:
            raise RemoteHelperError("coreutils fallback returned a malformed stat record") from exc

    def _fb_sha256(self, paths: list[str], timeout: float) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for lo, hi in _chunk_bounds(paths):
            for path, line in zip(
                paths[lo:hi], self._fallback_lines("sha256", paths[lo:hi], timeout, hi - lo), strict=True
            ):
                if line == "N":
                    out[path] = None
                elif line.startswith("H ") and _HEX64.fullmatch(line[2:]):
                    out[path] = "sha256:" + line[2:]
                else:
                    raise RemoteHelperError("coreutils fallback returned a malformed digest record")
        return out

    def _fb_read(self, op: str, path: str, norm: str, offset: int, length: int, timeout: float) -> RangeRead:
        body = self._fallback(op, [norm, str(offset), str(length)], timeout)
        header, _, data = body.partition(b"\n")
        fields = header.decode("ascii", "replace").split(" ", 5)
        if fields[0] == "E" and len(fields) == 2:
            raise _os_error(fields[1], _strerror(fields[1]), path)
        if fields[0] != "D" or len(fields) != 6:
            raise RemoteHelperError(f"coreutils fallback returned a malformed {op} record")
        _, off, size, sec, inode, raw_time = fields
        try:
            off_i, size_i = int(off), int(size)
            return RangeRead(
                path=path,
                offset=off_i,
                data=data,
                size=size_i,
                mtime=_mtime(sec, raw_time),
                inode=int(inode),
                eof=off_i + len(data) >= size_i,
            )
        except ValueError as exc:
            raise RemoteHelperError(f"coreutils fallback returned a malformed {op} record") from exc

    def _fb_write(self, path: str, norm: str, data: bytes, mode: int, timeout: float) -> None:
        body = self._fallback("write_file", [norm, format(mode, "o"), str(len(data))], timeout, stdin=data)
        line = body.split(b"\n", 1)[0].decode("ascii", "replace")
        if line.startswith("W "):
            return
        if line.startswith("T "):
            raise _truncated_upload(path, "write_file")
        raise self._fb_stage_error(line, path, "write_file")

    def _fb_mkdirs(self, paths: list[str], timeout: float) -> None:
        for lo, hi in _chunk_bounds(paths):
            line = (
                self._fallback("mkdirs", paths[lo:hi], timeout).split(b"\n", 1)[0].decode("ascii", "replace")
            )
            if line != "K":
                raise self._fb_stage_error(line, paths[lo], "mkdirs")

    @staticmethod
    def _fb_stage_error(line: str, path: str, op: str) -> Exception:
        fields = line.split(" ", 2)
        if fields[0] != "E" or len(fields) < 2:
            return RemoteHelperError(f"coreutils fallback returned a malformed {op} record")
        message = _b64decode(fields[2]).decode("utf-8", "replace").strip() if len(fields) == 3 else ""
        message = message or f"{fields[1]} failed"
        return OSError(_errno_from_message(message), message, path)


def _strerror(errno_name: str) -> str:
    code = getattr(errno, errno_name, None)
    return os.strerror(code) if isinstance(code, int) else errno_name


def _no_reply(message: str, op: str, raw: RawResult) -> RemoteHelperError:
    return RemoteHelperError(
        message,
        retryable=True,
        details={"op": op, "exit_code": raw.exit_code, "stderr": _tail_text(raw.stderr)},
    )


def _truncated_upload(path: str, op: str) -> ConnectionLost:
    """The payload arrived short, so the helper wrote nothing: safe to resend."""
    lost = ConnectionLost(
        f"upload of {path} was truncated in transit; nothing was written",
        details={"op": op, "maybe_executed": False},
    )
    lost.maybe_executed = False
    return lost


def _python_unusable(raw: RawResult) -> bool:
    """True when the remote python is missing (127), not executable (126) or too old to run
    the helper at all."""
    if raw.exit_code in (126, 127):
        return True
    stderr = raw.stderr.lower()
    return b"syntaxerror" in stderr or b"unknown option" in stderr


__all__ = [
    "BOOTSTRAP",
    "FALLBACK_OPS",
    "FALLBACK_SH",
    "MARKER",
    "PROTOCOL_VERSION",
    "PYTHON_OPS",
    "REMOTE_HELPER",
    "RawResult",
    "RemoteExecutor",
    "RemoteFileOps",
    "RemoteHelperError",
    "pathlib_form",
    "split_after_marker",
]
