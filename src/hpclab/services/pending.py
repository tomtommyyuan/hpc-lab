"""Decode Slurm PENDING reasons into something a researcher can act on.

``squeue`` reports one reason word per pending job (``Priority``, ``QOSMaxGRESPerUser``,
``ReqNodeNotAvail, UnavailableNodes:gpu-[01-02]`` ...). The same word means very different
things for the person waiting: "the cluster is busy, just wait" vs "you hit your own per-user
limit" vs "this job can never start as submitted". ``decode_pending`` is table driven and
pure; unknown words are reported as ``unknown`` (never guessed into a category).

Categories:

* ``cluster_busy``   - waiting for resources/priority; nothing to fix, only to wait (or pick a
  less busy partition/GPU type).
* ``self_limit``     - your own per-user QOS/association limit (running jobs, GPUs, ...): the job
  starts when one of *your* jobs ends.
* ``account_limit``  - a group/account-wide limit shared with your lab.
* ``never_starts``   - as submitted the job cannot start (time above the partition/QOS limit,
  impossible constraints, a dependency that can never be satisfied, a maintenance
  reservation overlapping the requested walltime). Cancel and resubmit with a fix.
* ``held``           - held by you or an admin (``scontrol release`` / contact support).
* ``begin_time``     - deliberately deferred (``--begin``) or waiting for a reservation window.
* ``dependency``     - waiting for another job (``--dependency``).
* ``unknown``        - no reason reported yet, or a reason hpclab does not know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

PendingCategory = Literal[
    "self_limit",
    "account_limit",
    "cluster_busy",
    "never_starts",
    "held",
    "begin_time",
    "dependency",
    "unknown",
]

__all__ = ["PendingCategory", "PendingInfo", "decode_pending"]


class PendingInfo(BaseModel):
    """Decoded PENDING reason. ``actionable`` = the researcher can do something now."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str | None
    category: PendingCategory
    explanation: str
    actionable: bool
    suggestion: str | None = None
    raw: str | None = None
    detail: str | None = None  # e.g. the unavailable node list of ReqNodeNotAvail


@dataclass(frozen=True)
class _Entry:
    category: PendingCategory
    explanation: str
    actionable: bool
    suggestion: str | None = None


_WAIT = "Nothing to fix: the job starts when resources free up."

# Exact reason words (Slurm job_reason strings as printed by squeue %r).
_EXACT: dict[str, _Entry] = {
    "Priority": _Entry(
        "cluster_busy",
        "Other jobs with higher priority are ahead in the queue.",
        False,
        _WAIT + " A smaller or shorter request (or another partition/GPU type) may start sooner.",
    ),
    "Resources": _Entry(
        "cluster_busy",
        "The job is next in line but the requested resources (GPUs/CPUs/memory) are all in use.",
        False,
        _WAIT,
    ),
    "PartitionDown": _Entry(
        "cluster_busy",
        "The partition is currently down (maintenance or outage).",
        False,
        "Check the cluster status page; the job starts when the partition comes back.",
    ),
    "PartitionInactive": _Entry(
        "cluster_busy", "The partition is inactive.", False, "Check the cluster status page."
    ),
    "NodeDown": _Entry(
        "cluster_busy",
        "A node the job needs is down.",
        False,
        "Wait, or resubmit excluding the node.",
    ),
    "Licenses": _Entry("cluster_busy", "Waiting for a license to become available.", False, _WAIT),
    "Cleaning": _Entry("cluster_busy", "Nodes are being cleaned up after a previous job.", False, _WAIT),
    "Prolog": _Entry("cluster_busy", "The job's prolog is running; it is about to start.", False, None),
    "Reservation": _Entry(
        "begin_time",
        "Waiting for its advanced reservation to become available.",
        False,
        "The job starts when the reservation window opens.",
    ),
    "BeginTime": _Entry(
        "begin_time",
        "The job was submitted with a start time (--begin) in the future, or is in a requeue delay.",
        False,
        None,
    ),
    "Dependency": _Entry(
        "dependency",
        "Waiting for the job(s) it depends on (--dependency) to finish.",
        False,
        None,
    ),
    "DependencyNeverSatisfied": _Entry(
        "never_starts",
        "A job this one depends on failed or was cancelled, so the dependency can never be satisfied.",
        True,
        "Cancel this job and resubmit without the failed dependency.",
    ),
    "BadConstraints": _Entry(
        "never_starts",
        "No node can satisfy the requested constraints/features.",
        True,
        "Cancel and resubmit with a valid --constraint / GPU type.",
    ),
    "PartitionTimeLimit": _Entry(
        "never_starts",
        "The requested walltime is longer than the partition's maximum.",
        True,
        "Cancel and resubmit with --time at or below the partition limit (or another partition).",
    ),
    "PartitionNodeLimit": _Entry(
        "never_starts",
        "The requested node count is outside the partition's limits.",
        True,
        "Cancel and resubmit with fewer nodes (or another partition).",
    ),
    "PartitionConfig": _Entry(
        "never_starts",
        "The request does not fit the partition's configuration (e.g. GPUs per node).",
        True,
        "Cancel and resubmit with a request the partition can hold.",
    ),
    "InvalidAccount": _Entry(
        "never_starts",
        "The job's account is invalid for this user/partition.",
        True,
        "Cancel and resubmit with a valid --account (see cluster_inspect limits).",
    ),
    "InvalidQOS": _Entry(
        "never_starts",
        "The job's QOS is invalid for this user/account.",
        True,
        "Cancel and resubmit with a valid --qos.",
    ),
    "QOSMaxWallDurationPerJobLimit": _Entry(
        "never_starts",
        "The requested walltime exceeds the QOS's per-job maximum.",
        True,
        "Cancel and resubmit with a shorter --time or another QOS.",
    ),
    "JobHeldUser": _Entry(
        "held",
        "The job was held by its owner (scontrol hold).",
        True,
        "Release it with 'scontrol release <jobid>' if that was not intended.",
    ),
    "JobHeldAdmin": _Entry(
        "held",
        "The job was held by a cluster administrator.",
        True,
        "Contact cluster support to learn why it was held.",
    ),
    "launch failed requeued held": _Entry(
        "held",
        "The job failed to launch on its node, was requeued and held by Slurm.",
        True,
        "Check the node/job logs, then 'scontrol release <jobid>' or cancel and resubmit.",
    ),
    "SystemFailure": _Entry(
        "cluster_busy",
        "A Slurm or node failure is delaying the job.",
        False,
        "Wait; contact support if it persists.",
    ),
    "JobArrayTaskLimit": _Entry(
        "self_limit", "The array's own running-task limit (%N) is reached.", False, _WAIT
    ),
}

# Prefix/pattern rules, first match wins (order matters: specific before generic).
_PATTERNS: tuple[tuple[re.Pattern[str], _Entry], ...] = (
    (
        re.compile(r"^QOSMax\w*PerJob"),
        _Entry(
            "never_starts",
            "The request exceeds a per-job limit of the QOS (e.g. GPUs, CPUs or nodes per job).",
            True,
            "Cancel and resubmit with a smaller request, or ask which QOS allows it.",
        ),
    ),
    (
        re.compile(r"^QOSMin\w*"),
        _Entry(
            "never_starts",
            "The request is below a per-job minimum of the QOS.",
            True,
            "Cancel and resubmit with a request that meets the QOS minimum.",
        ),
    ),
    (
        re.compile(r"^QOSMax\w*PerUser"),
        _Entry(
            "self_limit",
            "You reached a per-user limit of the QOS (running jobs, GPUs, CPUs ...).",
            False,
            "The job starts when one of your own jobs ends; cancel lower-priority jobs to start it sooner.",
        ),
    ),
    (
        re.compile(r"^QOSMax\w*PerAccount"),
        _Entry(
            "account_limit",
            "Your account reached a per-account limit of the QOS.",
            False,
            "Shared with your lab: it starts when the account's other jobs finish.",
        ),
    ),
    (
        re.compile(r"^QOSGrp\w*"),
        _Entry(
            "account_limit",
            "A group limit of the QOS (shared by everyone using it) is reached.",
            False,
            "Shared limit: it starts when other jobs under the QOS finish.",
        ),
    ),
    (
        re.compile(r"^QOS(NotAllowed|UsageThreshold|JobLimit|TimeLimit|ResourceLimit)"),
        _Entry(
            "account_limit",
            "A QOS usage limit is in effect.",
            False,
            "Check the QOS limits with cluster_inspect (limits).",
        ),
    ),
    (
        re.compile(r"^AssocGrp\w*"),
        _Entry(
            "account_limit",
            "Your account (association) reached a group limit (e.g. GPUs, GPU-minutes, running jobs).",
            False,
            "Shared with your lab/account: it starts when the account's usage drops; ask the PI if it is a budget cap.",
        ),
    ),
    (
        re.compile(r"^AssocMax\w*PerJob"),
        _Entry(
            "never_starts",
            "The request exceeds a per-job limit of your association.",
            True,
            "Cancel and resubmit with a smaller request.",
        ),
    ),
    (
        re.compile(r"^AssocMax\w*"),
        _Entry(
            "self_limit",
            "You reached a limit of your own association (e.g. max running jobs).",
            False,
            "The job starts when one of your own jobs ends.",
        ),
    ),
    (
        re.compile(r"^MaxJobs|^MaxSubmit|^MaxCpu|^MaxGRES", re.IGNORECASE),
        _Entry("self_limit", "A per-user job/resource limit is reached.", False, _WAIT),
    ),
    (
        re.compile(r"^Nodes_required_for_job_are_DOWN"),
        _Entry(
            "cluster_busy",
            "Nodes the job needs are down, drained or reserved.",
            False,
            "Wait, or resubmit excluding those nodes.",
        ),
    ),
    (
        re.compile(r"^JobHeld"),
        _Entry(
            "held", "The job is held.", True, "Release it with 'scontrol release <jobid>' or ask support."
        ),
    ),
)

_REQ_NODE_RE = re.compile(r"^ReqNodeNotAvail\b[,\s]*(?P<rest>.*)$", re.IGNORECASE)
_RESERVED_RE = re.compile(r"reserv|maint", re.IGNORECASE)


def _req_node_not_avail(raw: str, rest: str) -> PendingInfo:
    detail = rest.strip() or None
    if detail and _RESERVED_RE.search(detail):
        return PendingInfo(
            reason_code="ReqNodeNotAvail",
            category="never_starts",
            explanation=(
                "The nodes are reserved (e.g. a maintenance window) for a period that overlaps the "
                "requested walltime, so the job cannot start before the reservation ends."
            ),
            actionable=True,
            suggestion=(
                "Resubmit with a --time that ends before the reservation starts, or wait until the "
                "maintenance is over."
            ),
            raw=raw,
            detail=detail,
        )
    return PendingInfo(
        reason_code="ReqNodeNotAvail",
        category="cluster_busy",
        explanation="Some nodes the job could use are unavailable (down, drained or reserved).",
        actionable=False,
        suggestion="Wait; if it persists, check cluster_inspect (nodes) or resubmit excluding them.",
        raw=raw,
        detail=detail,
    )


def decode_pending(reason: str | None) -> PendingInfo:
    """Decode one squeue PENDING reason (raw text as printed; ``None``/``None`` = no reason yet)."""
    raw = (reason or "").strip()
    if not raw or raw.lower() in ("none", "(null)", "n/a"):
        return PendingInfo(
            reason_code=None,
            category="unknown",
            explanation="The scheduler has not evaluated the job yet (no pending reason reported).",
            actionable=False,
            suggestion="Wait for the next scheduling cycle.",
            raw=reason,
        )
    m = _REQ_NODE_RE.match(raw)
    if m:
        return _req_node_not_avail(raw, m.group("rest"))
    entry = _EXACT.get(raw)
    code = raw
    if entry is None:
        # "(Resources)" / "Priority, ..." forms: strip decoration and retry on the first word.
        word = re.split(r"[,\s(]", raw.strip("()"), maxsplit=1)[0]
        entry = _EXACT.get(word)
        code = word or raw
        if entry is None:
            entry = next((e for pat, e in _PATTERNS if pat.search(word)), None)
    if entry is None:
        return PendingInfo(
            reason_code=code,
            category="unknown",
            explanation=f"Slurm reports '{raw}', a pending reason hpclab does not know.",
            actionable=False,
            suggestion="See 'man squeue' (JOB REASON CODES) or ask cluster support.",
            raw=raw,
        )
    return PendingInfo(
        reason_code=code,
        category=entry.category,
        explanation=entry.explanation,
        actionable=entry.actionable,
        suggestion=entry.suggestion,
        raw=raw,
    )
