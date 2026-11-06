"""Sim-backed ServiceContext for service tests.

    from hpclab.testing.simctx import sim_context

    def test_submit(tmp_path):
        with sim_context(tmp_path) as env:
            env.ctx          # ServiceContext with cluster "sim" registered (real SlurmScheduler + SimTransport)
            env.sim          # SimCluster: ground truth (env.sim.jobs()), faults (env.sim.set_faults(...))
            env.drain({...}) # run outbox handlers until nothing is due

The cluster is registered through the normal ``ClusterRegistry`` path, so services obtain the
transport/scheduler exactly as in production (``ctx.clusters.transport("sim")``). The
simulator is always shut down (live job process groups killed), even when the test fails.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpclab.adapters.base import Scheduler, Transport
from hpclab.config import get_paths
from hpclab.domain.clock import Clock
from hpclab.profiles.schema import ResolvedCluster
from hpclab.services.context import ServiceContext, Settings
from hpclab.services.outbox import Handler, OutboxRunner
from hpclab.sim.cluster import SimCluster
from hpclab.sim.profile import resolved_sim_cluster
from hpclab.sim.testing import sim_cluster


@dataclass
class SimEnv:
    ctx: ServiceContext
    sim: SimCluster
    cluster: ResolvedCluster

    @property
    def cluster_id(self) -> str:
        return self.cluster.cluster_id

    @property
    def transport(self) -> Transport:
        return self.ctx.clusters.transport(self.cluster_id)

    @property
    def scheduler(self) -> Scheduler:
        return self.ctx.clusters.scheduler(self.cluster_id)

    @property
    def remote_root(self) -> str:
        return self.cluster.remote_root()

    def runner(self, handlers: dict[str, Handler], *, lease_s: float = 60.0) -> OutboxRunner:
        r = OutboxRunner(self.ctx, lease_s=lease_s)
        r.register_all(handlers)
        return r

    def drain(self, handlers: dict[str, Handler], *, max_rounds: int = 200) -> int:
        """Run due outbox items (only kinds in ``handlers``) until none are due."""
        return self.runner(handlers).drain(max_rounds=max_rounds)


def wait_until(
    pred: Callable[[], Any], *, timeout: float = 20.0, interval: float = 0.1, what: str = "condition"
) -> Any:
    """Poll ``pred`` until truthy; return its value or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while True:
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(interval)


@contextmanager
def sim_context(
    tmp_path: Path,
    *,
    cluster_id: str = "sim",
    clock: Clock | None = None,
    settings: Settings | None = None,
    **sim_config: Any,
) -> Iterator[SimEnv]:
    """A fresh simulator + hpclab home under ``tmp_path`` with the sim cluster registered."""
    with sim_cluster(Path(tmp_path), **sim_config) as sim:
        paths = get_paths(Path(tmp_path) / "hpclab-home").ensure()
        ctx = ServiceContext.create(
            paths,
            clock=clock,
            settings=settings or Settings(min_poll_interval_s=0.2, request_deadline_s=15.0),
        )
        resolved = resolved_sim_cluster(cluster_id, sim)
        ctx.clusters.register(resolved.profile, resolved.overlay)
        try:
            yield SimEnv(ctx=ctx, sim=sim, cluster=ctx.clusters.get(cluster_id))
        finally:
            ctx.clusters.close()
            ctx.db.close()
