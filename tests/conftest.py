"""Shared fixtures. Area-specific fixtures live in tests/<area>/conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpclab.config import Paths, get_paths
from hpclab.domain.clock import FakeClock
from hpclab.storage.db import Database


@pytest.fixture
def hpclab_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hpclab-home"
    monkeypatch.setenv("HPCLAB_HOME", str(home))
    return home


@pytest.fixture
def paths(hpclab_home: Path) -> Paths:
    return get_paths(hpclab_home).ensure()


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db(paths: Paths) -> Database:
    database = Database(paths.db)
    yield database
    database.close()
