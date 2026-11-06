"""Transport that executes on this machine.

Used for clusters whose login node *is* this machine (``transport: local``) and as the base
of the simulator transport. ``run`` never uses a shell; file helpers use Python directly.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from hpclab.adapters.base import CommandResult, RangeRead, StatInfo, TransportTimeout


class LocalTransport:
    def __init__(
        self,
        cluster_id: str,
        *,
        env: dict[str, str] | None = None,
        path_prefix: list[str] | None = None,
        user: str | None = None,
    ) -> None:
        self.cluster_id = cluster_id
        self._extra_env = dict(env or {})
        self._path_prefix = list(path_prefix or [])
        self._user = user

    # -- environment -----------------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self._extra_env)
        if self._path_prefix:
            env["PATH"] = os.pathsep.join([*self._path_prefix, env.get("PATH", "")])
        return env

    def remote_user(self) -> str:
        if self._user is None:
            self._user = getpass.getuser()
        return self._user

    # -- commands -------------------------------------------------------------------
    def run(self, argv: list[str], *, timeout: float = 30.0, stdin: bytes | None = None) -> CommandResult:
        start = time.monotonic()
        try:
            # Never inherit our stdin: in the MCP server process it is the protocol channel.
            stdin_kw: dict[str, Any] = (
                {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
            )
            proc = subprocess.run(
                argv,
                **stdin_kw,
                capture_output=True,
                timeout=timeout,
                env=self._env(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransportTimeout(
                f"local command timed out after {timeout:.0f}s: {argv[0]}",
                details={"argv0": argv[0], "timeout_s": timeout},
            ) from exc
        except FileNotFoundError:
            return CommandResult(
                list(argv), 127, "", f"{argv[0]}: command not found", time.monotonic() - start
            )
        return CommandResult(
            list(argv),
            proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"),
            time.monotonic() - start,
        )

    def run_script(
        self, script: str, *, timeout: float = 30.0, args: list[str] | None = None
    ) -> CommandResult:
        return self.run(["bash", "-s", "--", *(args or [])], timeout=timeout, stdin=script.encode("utf-8"))

    # -- files ------------------------------------------------------------------------
    def upload_dir(self, local_dir: str, remote_dir: str, *, timeout: float = 600.0) -> None:
        src = Path(local_dir)
        dst = Path(remote_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)

    def write_file(
        self, remote_path: str, content: bytes, *, mode: int = 0o644, timeout: float = 30.0
    ) -> None:
        path = Path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".hpclab-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _range(self, remote_path: str, offset: int, length: int) -> RangeRead:
        path = Path(remote_path)
        with open(path, "rb") as fh:
            st = os.fstat(fh.fileno())
            size = st.st_size
            off = max(0, min(offset, size))
            fh.seek(off)
            data = fh.read(max(0, length))
        return RangeRead(
            path=remote_path,
            offset=off,
            data=data,
            size=size,
            mtime=st.st_mtime,
            inode=st.st_ino,
            eof=off + len(data) >= size,
        )

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        return self._range(remote_path, offset, length)

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        size = os.stat(remote_path).st_size
        return self._range(remote_path, max(0, size - length), length)

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        out: dict[str, StatInfo] = {}
        for p in remote_paths:
            path = Path(p)
            if not os.path.lexists(path):
                out[p] = StatInfo(path=p, exists=False, kind="missing")
                continue
            try:
                st = os.stat(path)
            except OSError:
                out[p] = StatInfo(path=p, exists=False, kind="missing", realpath=None)
                continue
            if path.is_symlink():
                kind = "symlink"
            elif path.is_dir():
                kind = "dir"
            elif path.is_file():
                kind = "file"
            else:
                kind = "other"
            out[p] = StatInfo(
                path=p,
                exists=True,
                kind=kind,  # type: ignore[arg-type]
                size=st.st_size,
                mtime=st.st_mtime,
                inode=st.st_ino,
                realpath=os.path.realpath(path),
            )
        return out

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for p in remote_paths:
            try:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                out[p] = "sha256:" + h.hexdigest()
            except OSError:
                out[p] = None
        return out

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        src = Path(remote_path)
        if src.is_dir():
            if Path(local_path).exists():
                shutil.rmtree(local_path)
            shutil.copytree(src, local_path, symlinks=True)
        else:
            shutil.copy2(src, local_path)

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        root = Path(remote_dir)
        if not root.is_dir():
            raise FileNotFoundError(remote_dir)
        out: list[StatInfo] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                kind = "symlink" if os.path.islink(full) else "file"
                out.append(
                    StatInfo(
                        path=full,
                        exists=True,
                        kind=kind,  # type: ignore[arg-type]
                        size=st.st_size,
                        mtime=st.st_mtime,
                        inode=st.st_ino,
                        realpath=os.path.realpath(full),
                    )
                )
                if len(out) >= max_entries:
                    return out
        return out

    def close(self) -> None:
        return None
