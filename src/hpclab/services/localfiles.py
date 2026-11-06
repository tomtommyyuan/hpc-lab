"""Bounded reads of local files named by tool arguments (``spec_path``, ``tex_paths``).

Such a path can name anything the user can open: ``/dev/zero`` (``stat`` says 0 bytes, the
data never ends), a FIFO (``open`` waits for a writer forever), a terminal, a directory. The
controller is one long-lived process, so a read must neither block nor grow without bound:
open non-blocking, require a regular file on the *opened* descriptor (no stat/open race), and
read at most ``max_bytes + 1`` bytes (a regular file may grow after ``fstat``).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

__all__ = ["FileTooLarge", "NotRegularFile", "read_regular_file"]

_CHUNK = 1 << 20
_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOCTTY", 0)


class NotRegularFile(OSError):
    """The path exists but is not a regular file (device, FIFO, socket, directory)."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind  # "a FIFO", "a character device", ...


class FileTooLarge(OSError):
    """The file is larger than the caller's bound."""

    def __init__(self, message: str, max_bytes: int) -> None:
        super().__init__(message)
        self.max_bytes = max_bytes


def _kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "a directory"
    if stat.S_ISCHR(mode):
        return "a character device"
    if stat.S_ISBLK(mode):
        return "a block device"
    if stat.S_ISFIFO(mode):
        return "a FIFO"
    if stat.S_ISSOCK(mode):
        return "a socket"
    return "not a plain file"


def read_regular_file(path: str | os.PathLike[str], *, max_bytes: int) -> bytes:
    """The bytes of the regular file at ``path`` (symlinks followed), at most ``max_bytes``.

    Raises ``FileNotFoundError``/``PermissionError``/other ``OSError`` from ``open``,
    :class:`NotRegularFile` for anything but a regular file and :class:`FileTooLarge` when the
    file holds more than ``max_bytes`` bytes. Never blocks on a FIFO and never reads more than
    ``max_bytes + 1`` bytes.
    """
    shown = str(Path(path))
    fd = os.open(path, _OPEN_FLAGS)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            kind = _kind(st.st_mode)
            raise NotRegularFile(f"{shown} is not a regular file ({kind})", kind)
        if st.st_size > max_bytes:
            raise FileTooLarge(f"{shown} is larger than {max_bytes} bytes ({st.st_size} bytes)", max_bytes)
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, _CHUNK))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise FileTooLarge(f"{shown} is larger than {max_bytes} bytes", max_bytes)
        return data
    finally:
        os.close(fd)
