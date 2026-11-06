"""Fixtures for observe tests (fakes live in observe_support.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
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
