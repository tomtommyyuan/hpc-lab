"""Fixtures for the research-helper tests (evaluators, stats, render)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from research_bundle_support import build_bundle


@pytest.fixture
def bundle() -> dict[str, Any]:
    return build_bundle()


@pytest.fixture
def bundle_factory() -> Callable[[], dict[str, Any]]:
    return lambda: copy.deepcopy(build_bundle())
