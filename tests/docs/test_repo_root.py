"""No test leaves files in the repository root.

A ``:module/x`` directory (a copy of the rsync test's ``project`` fixture) once appeared in the
repo root: an rsync destination that looked like a remote spec was treated as a local path
relative to the working directory. The transport now rejects such destinations before calling
rsync (``tests/adapters/test_openssh_rsync.py::test_unsafe_remote_paths_are_rejected``). This
guard fails if anything like it comes back: a name with a colon is never part of the repository
(it is invalid on Windows and is how a ``host:path`` spec leaks onto the local disk).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_repo_root_has_no_remote_spec_leftovers() -> None:
    leaked = sorted(p.name for p in REPO.iterdir() if ":" in p.name)
    assert leaked == [], f"stray entries in the repo root (left by a test?): {leaked}"


def test_rsync_fixture_files_are_not_in_the_repo_root() -> None:
    # the rsync tests' project fixture: train.py + run.sh + pkg/model.py + a `latest` symlink
    assert not (REPO / "latest").is_symlink()
    assert not (REPO / "run.sh").exists() and not (REPO / "pkg").exists()
