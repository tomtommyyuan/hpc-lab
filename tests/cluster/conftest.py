"""Fixtures for cluster_inspect unit tests (fakes live in ``cluster_fakes``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cluster_fakes import Env, FakeLogin, FakeSched

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.services.context import ServiceContext, Settings


@pytest.fixture
def env(paths: Paths) -> Iterator[Env]:
    clock = FakeClock()
    login, sched = FakeLogin(), FakeSched()
    ctx = ServiceContext.create(
        paths,
        clock=clock,
        transport_factory=lambda cluster, p: login,
        scheduler_factory=lambda transport, cluster, clk: sched,  # type: ignore[arg-type,return-value]
        settings=Settings(request_deadline_s=20.0),
    )
    e = Env(ctx=ctx, login=login, sched=sched, clock=clock)
    e.register()
    yield e
    ctx.db.close()
