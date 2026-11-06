"""Fixtures for snapshot tests: throwaway git repos, a local blob store, a fake "remote"."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from snapshot_support import GitRepo, make_writable

from hpclab.services.snapshot import SnapshotStore


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    root = tmp_path / "project"
    root.mkdir()
    r = GitRepo(root)
    r.git("init", "-q", "-b", "main")
    r.write("train.py", "print('train')\n")
    r.write("src/model.py", "class Model: ...\n")
    r.write("run.sh", "#!/bin/sh\necho run\n", executable=True)
    r.write(".gitignore", "ignored.log\n")
    r.commit("initial")
    return r


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "store")


@pytest.fixture
def remote_root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "remote" / "hpclab"
    yield root
    make_writable(tmp_path / "remote")


@pytest.fixture
def blob_exists(store: SnapshotStore) -> Callable[[bytes], bool]:
    def check(content: bytes) -> bool:
        return store.object_path(hashlib.sha256(content).hexdigest()).exists()

    return check
