"""Reusable test helpers for suites that need a live simulated cluster.

    from hpclab.sim.testing import sim_cluster, sim_scheduler

    def test_something(tmp_path):
        with sim_cluster(tmp_path, min_job_age_s=0.5) as sim:
            transport, scheduler = sim_scheduler(sim)
            ...

``sim_cluster`` always shuts the simulator down (kills live job process groups), even when the
test fails. ``FAST_TIMINGS`` are applied unless overridden, keeping job lifecycles sub-second.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hpclab.adapters.slurm import SlurmScheduler
from hpclab.domain.clock import Clock
from hpclab.profiles.schema import ResolvedCluster
from hpclab.sim.cluster import SimCluster
from hpclab.sim.config import SimConfig
from hpclab.sim.profile import resolved_sim_cluster
from hpclab.sim.transport import SimTransport

FAST_TIMINGS: dict[str, Any] = {
    "min_job_age_s": 1.0,
    "kill_wait_s": 0.3,
    "poll_interval_s": 0.05,
    "fate_delay_s": 0.3,
}


@contextmanager
def sim_cluster(tmp_path: Path, **config: Any) -> Iterator[SimCluster]:
    """A fresh simulator under ``tmp_path/slurm-sim`` with ``SimConfig`` overrides ``config``."""
    cfg = SimConfig().replace(**{**FAST_TIMINGS, **config})
    cluster = SimCluster.ensure(Path(tmp_path) / "slurm-sim", cfg)
    try:
        yield cluster
    finally:
        cluster.shutdown(grace_s=2.0)


def sim_scheduler(
    cluster: SimCluster, *, cluster_id: str = "sim", clock: Clock | None = None
) -> tuple[SimTransport, SlurmScheduler, ResolvedCluster]:
    """The real ``SlurmScheduler`` wired to ``cluster`` through ``SimTransport``."""
    resolved = resolved_sim_cluster(cluster_id, cluster)
    transport = SimTransport(cluster_id, cluster.root)
    return transport, SlurmScheduler(transport, resolved, clock=clock), resolved
