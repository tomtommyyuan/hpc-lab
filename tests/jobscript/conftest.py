"""Fixtures for the job wrapper tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from jobscript_support import BASH, Harness, make_writable


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    if not Path(BASH).exists():
        pytest.skip("/bin/bash not available")
    h = Harness(tmp_path)
    yield h
    make_writable(tmp_path)
