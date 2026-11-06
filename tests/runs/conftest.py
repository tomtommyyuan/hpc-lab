"""Fixtures for runs/logs read-side tests; shares the fakes of tests/observe."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "observe"))

from observe_support import Env, make_env

from hpclab.services.observe import wait_for_background


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Env]:
    e = make_env(tmp_path)
    yield e
    if e.scheduler.block is not None:
        e.scheduler.block.set()
    assert wait_for_background(10)
    e.ctx.clusters.close()
    e.ctx.db.close_all()
