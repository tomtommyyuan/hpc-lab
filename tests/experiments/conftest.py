"""Fixtures for the prepare area (helpers live in prepare_support.py)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from prepare_support import Lab, make_lab

from hpclab.config import Paths


@pytest.fixture
def lab(paths: Paths, tmp_path: Path) -> Iterator[Lab]:
    obj, ctx = make_lab(paths, tmp_path)
    try:
        yield obj
    finally:
        ctx.clusters.close()
        ctx.db.close()
