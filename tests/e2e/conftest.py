"""Fixtures for the end-to-end suite (real controller subprocess + simulator; see e2e_support)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from e2e_support import Lab


@pytest.fixture
def make_lab(tmp_path: Path) -> Iterator[Callable[..., Lab]]:
    """``make_lab(sim={...}, policy={...})`` -> a started Lab; always closed after the test."""
    labs: list[Lab] = []

    def make(name: str = "lab", **kw: Any) -> Lab:
        lab = Lab.start(tmp_path / name, **kw)
        labs.append(lab)
        return lab

    yield make
    for lab in labs:
        lab.close()


@pytest.fixture
def lab(make_lab: Callable[..., Lab]) -> Lab:
    return make_lab()
