"""Fixtures for evaluation tests; reuses the artifacts area's ``Lab`` helper."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from artifacts_support import Att, Lab


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    env = Lab(tmp_path)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def indexed(lab: Lab) -> Att:
    return lab.ready()
