"""Fixtures for coverage tests that go through the real index / evaluation path.

Reuses the artifacts area's ``Lab`` helper (a local "remote" filesystem and the wrapper's own
``outputs.json`` helper), as the evaluation tests do.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "artifacts"))

from artifacts_support import Lab


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    env = Lab(tmp_path)
    try:
        yield env
    finally:
        env.close()
