"""Helpers for the cluster setup tests (imported by name, pytest prepend mode)."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
from pathlib import Path

from hpclab.sim.cluster import SimCluster


def relocated_sim_root(home: Path) -> Path:
    """Where init_sim places the simulator of a home with an unsafe path (the documented rule)."""
    tag = hashlib.sha256(os.fsencode(str(home))).hexdigest()[:10]
    return Path("/var/tmp").resolve() / f"hpclab-sim-{os.getuid()}-{tag}"


def make_writable(root: Path) -> None:
    """chmod -R u+w (published snapshots are read-only) so the tree can be deleted."""
    if not root.exists() or root.is_symlink():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            p = Path(dirpath) / name
            with contextlib.suppress(OSError):
                st = p.lstat()
                if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                    p.chmod(st.st_mode | stat.S_IWUSR)
    with contextlib.suppress(OSError):
        root.chmod(root.stat().st_mode | stat.S_IRWXU)


def remove_sim_root(root: Path) -> None:
    if root.is_symlink():
        root.unlink()
        return
    if not root.exists():
        return
    with contextlib.suppress(Exception):
        SimCluster(root).shutdown(grace_s=2.0)
    make_writable(root)
    shutil.rmtree(root, ignore_errors=True)
