"""Fixtures for the submission tests: a ServiceContext with scripted adapters and a FakeClock."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pytest
from submission_support import FakeScheduler, FakeTransport, UnitEnv

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, UserOverlay
from hpclab.services import submission
from hpclab.services.context import ServiceContext

if TYPE_CHECKING:
    from hpclab.adapters.base import Scheduler


@pytest.fixture
def unit(paths: Paths, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> Iterator[UnitEnv]:
    transport = FakeTransport("unit", user="alice")
    scheduler = FakeScheduler("unit", clock=fake_clock)
    ctx = ServiceContext.create(
        paths,
        clock=fake_clock,
        transport_factory=lambda cluster, p: transport,
        scheduler_factory=lambda t, cluster, clock: cast("Scheduler", scheduler),
    )
    ctx.principal = "tester@host"
    profile = ClusterProfile(
        cluster_id="unit",
        display_name="Unit cluster",
        transport="local",
        partitions={"gpu": PartitionRule(max_time="2-00:00:00", gpu_types=["a100"], default=True)},
        default_partition="gpu",
        policy=SitePolicy(reconcile_window_s=300.0, accounting_lag_s=60.0, submit_rate_per_min=100),
    )
    overlay = UserOverlay(cluster_id="unit", user="alice", account="lab", remote_root="/remote/hpclab")
    ctx.clusters.register(profile, overlay)
    env = UnitEnv(ctx=ctx, transport=transport, scheduler=scheduler, clock=fake_clock)

    def fake_publish(transport_: Any, store: Any, ref: Any, remote_root: str) -> str:
        """The real publish is covered by the snapshot suite and the sim tests."""
        if env.publish_error is not None:
            raise env.publish_error
        path = f"{remote_root.rstrip('/')}/snapshots/{ref.digest.split(':', 1)[1]}"
        env.published.append(path)
        return path

    monkeypatch.setattr(submission, "publish_snapshot", fake_publish)
    yield env
    ctx.clusters.close()
    ctx.db.close_all()
