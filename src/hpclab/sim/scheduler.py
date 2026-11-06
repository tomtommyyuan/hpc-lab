"""FIFO scheduling for the simulator (pure: jobs + config in, decisions out).

Model (a simplified ``sched/builtin`` without backfill):

* Pending jobs are considered in submit order. A job blocked on ``Resources`` blocks every
  later job of the same partition (they pend with ``Priority``), which is what researchers see
  on a busy FIFO partition.
* Jobs held by dependencies, ``--hold``, an injected start delay, a partition time limit or a
  per-user GPU limit do not block others.
* Allocation is per node: CPUs, memory and concrete GPU indices, honoring ``--nodelist``,
  ``--exclude``, ``--constraint`` (node features) and admin state (drain/down nodes never
  receive jobs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hpclab.sim.config import NodeConfig, SimConfig
from hpclab.sim.formats import compress_hostlist
from hpclab.sim.state import SimJob


@dataclass
class NodeUse:
    cpus: int = 0
    mem_mb: int = 0
    gpus: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class Decision:
    job_id: int
    reason: str | None
    alloc: dict[str, dict[str, Any]] | None = None


def node_usage(jobs: list[SimJob]) -> dict[str, NodeUse]:
    use: dict[str, NodeUse] = {}
    for job in jobs:
        if job.state not in ("RUNNING", "COMPLETING", "SUSPENDED"):
            continue
        for node, a in job.alloc.items():
            u = use.setdefault(node, NodeUse())
            u.cpus += int(a.get("cpus", 0))
            u.mem_mb += int(a.get("mem", 0))
            u.gpus.update(int(g) for g in a.get("gpus", []))
    return use


def dependency_status(job: SimJob, by_id: Mapping[int, SimJob]) -> str | None:
    """None when satisfied, ``Dependency`` while waiting, ``DependencyNeverSatisfied`` when it
    can no longer be met (Slurm keeps such jobs pending unless kill_invalid_depend is set)."""
    if not job.dependency:
        return None
    waiting = False
    for clause in job.dependency.split(","):
        kind, _, ids = clause.partition(":")
        for raw in ids.split(":"):
            dep = _find(by_id, raw)
            if dep is None:
                continue  # purged dependency counts as satisfied, like Slurm
            if kind == "after":
                ok, never = dep.state != "PENDING", False
            elif kind == "afterany":
                ok, never = dep.terminal, False
            elif kind == "afterok":
                ok, never = dep.state == "COMPLETED", dep.terminal and dep.state != "COMPLETED"
            elif kind == "afternotok":
                ok, never = dep.terminal and dep.state != "COMPLETED", dep.state == "COMPLETED"
            else:
                ok, never = True, False
            if never:
                return "DependencyNeverSatisfied"
            waiting = waiting or not ok
    return "Dependency" if waiting else None


def _find(by_id: Mapping[int, SimJob], token: str) -> SimJob | None:
    base = token.split("_", 1)[0]
    if not base.isdigit():
        return None
    return by_id.get(int(base))


def _candidate_nodes(
    cfg: SimConfig, job: SimJob, admin: Mapping[str, tuple[str, str | None]]
) -> tuple[list[NodeConfig], list[str]]:
    """Nodes that could ever run the job (static fit) and those excluded only by admin state."""
    part = cfg.partition(job.partition)
    if part is None:
        return [], []
    fits: list[NodeConfig] = []
    unavailable: list[str] = []
    for name in part.nodes:
        node = cfg.node(name)
        if node is None or name in job.exclude:
            continue
        if job.req_nodelist and name not in job.req_nodelist:
            continue
        if job.constraint and job.constraint not in node.features:
            continue
        if job.gpu_type and node.gpu_type != job.gpu_type:
            continue
        if (
            node.gpus < job.gpus_per_node
            or node.cpus < job.cpus_per_node
            or node.mem_mb < job.mem_per_node_mb
        ):
            continue
        if admin.get(name, ("up", None))[0] != "up":
            unavailable.append(name)
            continue
        fits.append(node)
    return fits, unavailable


def _try_allocate(
    job: SimJob, nodes: list[NodeConfig], use: dict[str, NodeUse]
) -> dict[str, dict[str, Any]] | None:
    chosen: dict[str, dict[str, Any]] = {}
    for node in nodes:
        u = use.get(node.name, NodeUse())
        free_gpus = [i for i in range(node.gpus) if i not in u.gpus]
        if (
            node.cpus - u.cpus >= job.cpus_per_node
            and node.mem_mb - u.mem_mb >= job.mem_per_node_mb
            and len(free_gpus) >= job.gpus_per_node
        ):
            chosen[node.name] = {
                "gpus": free_gpus[: job.gpus_per_node],
                "cpus": job.cpus_per_node,
                "mem": job.mem_per_node_mb,
            }
            if len(chosen) == job.nodes:
                return chosen
    return None


def plan(
    cfg: SimConfig,
    jobs: list[SimJob],
    admin: Mapping[str, tuple[str, str | None]],
    *,
    now: float,
    pending_delay_s: float = 0.0,
) -> list[Decision]:
    """Decide which pending jobs start now and why the others wait."""
    by_id = {j.job_id: j for j in jobs}
    use = node_usage(jobs)
    running_gpus: dict[tuple[str, str], int] = {}
    for j in jobs:
        if j.state == "RUNNING":
            key = (j.user, j.partition)
            running_gpus[key] = running_gpus.get(key, 0) + j.total_gpus
    blocked: set[str] = set()
    decisions: list[Decision] = []
    for job in sorted((j for j in jobs if j.state == "PENDING"), key=lambda j: (j.submit_time, j.job_id)):
        part = cfg.partition(job.partition)
        reason: str | None
        if job.hold:
            reason = "JobHeldUser"
        elif (dep := dependency_status(job, by_id)) is not None:
            reason = dep
        elif now < job.submit_time + pending_delay_s:
            reason = "Priority"
        elif part is None or part.state.upper() != "UP":
            reason = "PartitionDown"
        elif part.max_time_s is not None and (job.time_limit_s is None or job.time_limit_s > part.max_time_s):
            reason = "PartitionTimeLimit"
        elif (
            part.max_gpus_per_user is not None
            and running_gpus.get((job.user, job.partition), 0) + job.total_gpus > part.max_gpus_per_user
        ):
            reason = "QOSMaxGRESPerUser"
        elif job.partition in blocked:
            reason = "Priority"
        else:
            nodes, unavailable = _candidate_nodes(cfg, job, admin)
            alloc = _try_allocate(job, nodes, use) if len(nodes) >= job.nodes else None
            if alloc is not None:
                for name, a in alloc.items():
                    u = use.setdefault(name, NodeUse())
                    u.cpus += a["cpus"]
                    u.mem_mb += a["mem"]
                    u.gpus.update(a["gpus"])
                key = (job.user, job.partition)
                running_gpus[key] = running_gpus.get(key, 0) + job.total_gpus
                decisions.append(Decision(job_id=job.job_id, reason=None, alloc=alloc))
                continue
            if len(nodes) < job.nodes and unavailable:
                reason = f"ReqNodeNotAvail, UnavailableNodes:{compress_hostlist(unavailable)}"
            else:
                reason = "Resources"
            blocked.add(job.partition)
        decisions.append(Decision(job_id=job.job_id, reason=reason))
    return decisions


def estimate_start(cfg: SimConfig, job: SimJob, jobs: list[SimJob], now: float) -> float | None:
    """Rough expected start for a pending job: when enough running jobs in its partition hit
    their time limits (what ``squeue --start`` shows; approximate by design)."""
    if job.state != "PENDING":
        return None
    part = cfg.partition(job.partition)
    if part is None:
        return None
    ends = sorted(
        (j.start_time or now) + (j.time_limit_s or 365 * 86400)
        for j in jobs
        if j.state == "RUNNING" and j.partition == job.partition
    )
    ahead = sum(
        1 for j in jobs if j.state == "PENDING" and j.partition == job.partition and j.job_id < job.job_id
    )
    if not ends:
        return now + 1
    idx = min(len(ends) - 1, ahead)
    return max(now + 1, ends[idx])
