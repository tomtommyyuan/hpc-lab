"""Fixtures for the ops helpers (diagnosis, progress, checkpoint)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from ops_support import read_log


@pytest.fixture
def log() -> Callable[[str], str]:
    """``log("cuda_oom_torchrun.stderr.log")`` -> fixture text."""
    return read_log
