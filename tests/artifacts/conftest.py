"""Fixtures for the artifacts area (artifacts, vault)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from artifacts_support import Att, Lab

from hpclab.services.artifacts import index_attempt


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    env = Lab(tmp_path)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def indexed(lab: Lab) -> Att:
    """A terminal, COMPLETED attempt with valid outputs, indexed."""
    att = lab.attempt()
    lab.good_outputs(att)
    lab.manifest(att)
    res = index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert res.result_state == "valid", res
    return att
