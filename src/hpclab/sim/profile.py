"""Cluster profile + user overlay describing a simulator, so the rest of hpclab can register a
``transport: sim`` cluster exactly like a real one."""

from __future__ import annotations

from hpclab.domain.clock import format_slurm_duration
from hpclab.profiles.schema import (
    ClusterProfile,
    PartitionRule,
    ProfileSource,
    ResolvedCluster,
    SitePolicy,
    StorageRole,
    UserOverlay,
)
from hpclab.sim.cluster import SimCluster


def sim_profile(cluster_id: str, sim: SimCluster) -> ClusterProfile:
    """Public profile of the simulated cluster (fast poll intervals for tests and demos)."""
    cfg = sim.config
    partitions: dict[str, PartitionRule] = {}
    for p in cfg.partitions:
        types = sorted({n.gpu_type for n in cfg.nodes if n.name in p.nodes and n.gpu_type})
        max_gpus = max((n.gpus for n in cfg.nodes if n.name in p.nodes), default=0)
        partitions[p.name] = PartitionRule(
            max_time=format_slurm_duration(p.max_time_s) if p.max_time_s is not None else None,
            gpu_types=types,
            preemptible=p.preemptible,
            interactive_only=p.name == "interactive",
            max_gpus_per_job=max_gpus * len(p.nodes) or None,
            max_gpus_per_user=p.max_gpus_per_user,
            default=p.default,
        )
    default = cfg.default_partition()
    return ClusterProfile(
        cluster_id=cluster_id,
        display_name=f"Simulated Slurm cluster ({cfg.cluster_name})",
        transport="sim",
        partitions=partitions,
        default_partition=default.name if default else None,
        storage={
            "scratch": StorageRole(
                path=str(sim.fs_root / "scratch" / "{user}"), quota_hint="unlimited (local disk)"
            ),
            "home": StorageRole(path=str(sim.fs_root / "home" / "{user}")),
        },
        remote_root_template="{scratch}/hpclab",
        policy=SitePolicy(
            poll_min_interval_s=0.5,
            sacct_min_interval_s=0.5,
            reconcile_window_s=30.0,
            accounting_lag_s=max(1.0, cfg.accounting_lag_s),
        ),
        gpu_arch={"a100": "sm_80", "h100": "sm_90", "b200": "sm_100"},
        sources=[ProfileSource(title="hpclab Slurm simulator", notes=f"sim_root={sim.root}")],
        notes=["Local simulator: jobs run as local processes; GPUs and nvidia-smi are fake."],
    )


def sim_overlay(cluster_id: str, sim: SimCluster) -> UserOverlay:
    user = sim.user
    return UserOverlay(
        cluster_id=cluster_id,
        user=user,
        account=sim.config.default_account,
        qos=sim.config.default_qos,
        remote_root=str(sim.fs_root / "scratch" / user / "hpclab"),
        sim_root=str(sim.root),
    )


def resolved_sim_cluster(cluster_id: str, sim: SimCluster) -> ResolvedCluster:
    return ResolvedCluster(profile=sim_profile(cluster_id, sim), overlay=sim_overlay(cluster_id, sim))
