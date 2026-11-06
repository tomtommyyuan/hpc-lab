"""Helpers shared by the snapshot tests (imported by name: tests run in pytest's prepend mode)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def make_writable(root: Path) -> None:
    """Published snapshots are read-only; make them deletable for tmp cleanup."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for name in dirnames:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                os.chmod(full, 0o755)
        for name in filenames:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                os.chmod(full, 0o644)


class GitRepo:
    def __init__(self, root: Path) -> None:
        self.root = root

    def git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    def write(self, rel: str, content: str | bytes, *, executable: bool = False) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        if executable:
            path.chmod(0o755)
        return path

    def commit(self, message: str = "commit", *paths: str) -> str:
        self.git("add", "-A", *paths) if not paths else self.git("add", *paths)
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")
