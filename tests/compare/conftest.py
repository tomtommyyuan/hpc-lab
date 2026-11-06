"""Fixtures for the comparison tests: a FakeClock ServiceContext (ledger only) and the Lab builder."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from lab_support import Lab

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.services.context import ServiceContext


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ctx(paths: Paths, clock: FakeClock) -> Iterator[ServiceContext]:
    context = ServiceContext.create(paths, clock=clock)
    yield context
    context.db.close_all()


@pytest.fixture
def lab(ctx: ServiceContext, clock: FakeClock) -> Lab:
    return Lab(ctx, clock)
