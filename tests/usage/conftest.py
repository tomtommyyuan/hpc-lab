"""Fixtures for usage_report / lab_digest unit tests (fakes live in ``usage_fakes``)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from usage_fakes import NOW, FakeAcct, UsageEnv

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.profiles.schema import ClusterProfile, PartitionRule, UserOverlay
from hpclab.services.context import ServiceContext


@pytest.fixture
def uenv(paths: Paths) -> Iterator[UsageEnv]:
    clock = FakeClock(NOW)
    acct = FakeAcct(clock=clock)
    ctx = ServiceContext.create(
        paths,
        clock=clock,
        transport_factory=lambda cluster, p: acct,  # type: ignore[arg-type,return-value]
        scheduler_factory=lambda transport, cluster, clk: acct,  # type: ignore[arg-type,return-value]
    )
    profile = ClusterProfile(
        cluster_id="fake",
        display_name="Fake",
        transport="ssh",
        login_host="login",
        partitions={"gpu": PartitionRule(default=True), "interactive": PartitionRule(interactive_only=True)},
    )
    ctx.clusters.register(profile, UserOverlay(cluster_id="fake", user="alice", account="lab"))
    yield UsageEnv(ctx=ctx, acct=acct, clock=clock)
    ctx.db.close()
