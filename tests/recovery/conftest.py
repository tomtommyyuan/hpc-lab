from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# the workflows suite owns the shared ledger builders and the fake submission module
sys.path.insert(0, str(Path(__file__).parent.parent / "workflows"))

from wf_support import FakeSubmission, LabBuilder, install_fake_submission, make_ctx

from hpclab.domain.clock import FakeClock
from hpclab.services.context import ServiceContext


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ctx(tmp_path: Path, clock: FakeClock) -> Iterator[ServiceContext]:
    c = make_ctx(tmp_path, clock)
    yield c
    c.clusters.close()
    c.db.close_all()


@pytest.fixture
def lab(ctx: ServiceContext, tmp_path: Path) -> LabBuilder:
    return LabBuilder(ctx, str(tmp_path / "remote"))


@pytest.fixture
def sub(monkeypatch: pytest.MonkeyPatch) -> FakeSubmission:
    return install_fake_submission(monkeypatch)
