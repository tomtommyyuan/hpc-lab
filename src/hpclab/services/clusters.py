"""Cluster registration: public site profile + private user overlay -> ledger.

The overlay file (``$HPCLAB_HOME/clusters/<id>.yaml``, 0600) stays the user's editable source
of truth; ``cluster_register`` merges explicit fields into it, saves it, and snapshots the
resolved (profile, overlay) pair into the ledger so every ExecutionPlan can record which
rules it ran under (``profile_revision``).

``auth_resume`` is the "I just logged in again" signal: work parked on ``auth_required`` and
the observer's poll backoff are woken at once instead of after their pause.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hpclab.config import Paths
from hpclab.domain.clock import iso
from hpclab.domain.errors import HpcLabError, InvalidArgument, PreconditionFailed
from hpclab.profiles.loader import list_profiles, load_overlay, load_profile, save_overlay
from hpclab.profiles.schema import ResolvedCluster, UserOverlay
from hpclab.services.context import ServiceContext
from hpclab.storage.db import Tx, jload

log = logging.getLogger("hpclab.clusters")


class ClusterSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    display_name: str
    transport: str
    ssh_host: str | None
    user: str | None
    account: str | None
    qos: str | None
    remote_root: str  # "" when it cannot be resolved yet (see missing)
    default_partition: str | None
    partitions: list[str]
    profile_revision: str
    overlay_path: str | None = None
    missing: list[str] = []


class ClusterData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster: ClusterSummary
    created: bool
    next_steps: list[str]


class ClustersListData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clusters: list[ClusterSummary]


class ProfileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    cluster_id: str
    display_name: str
    source: str
    path: str
    error: str | None = None


class ProfilesListData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: list[ProfileEntry]


def _summary(rc: ResolvedCluster, overlay_file: Path | None = None) -> ClusterSummary:
    missing: list[str] = []
    if rc.profile.transport == "ssh" and not (rc.overlay.ssh_host or rc.profile.login_host):
        missing.append("ssh_host")
    if rc.profile.policy.require_account and not rc.overlay.account:
        missing.append("account")
    if not rc.overlay.user:
        missing.append("user")
    try:
        remote_root = rc.remote_root()
    except HpcLabError:
        remote_root = ""
        missing.append("remote_root")
    return ClusterSummary(
        cluster_id=rc.cluster_id,
        display_name=rc.profile.display_name,
        transport=rc.profile.transport,
        ssh_host=rc.overlay.ssh_host or rc.profile.login_host,
        user=rc.overlay.user,
        account=rc.overlay.account,
        qos=rc.overlay.qos,
        remote_root=remote_root,
        default_partition=rc.overlay.default_partition or rc.profile.default_partition,
        partitions=sorted(rc.profile.partitions),
        profile_revision=rc.revision,
        overlay_path=str(overlay_file) if overlay_file else None,
        missing=missing,
    )


def cluster_register(
    ctx: ServiceContext,
    *,
    profile: str,
    cluster_id: str | None = None,
    ssh_host: str | None = None,
    user: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    remote_root: str | None = None,
    default_partition: str | None = None,
    sim_root: str | None = None,
    ssh_options: list[str] | None = None,
    lab_users: list[str] | None = None,
) -> ClusterData:
    """Register (or update) a cluster from a site profile plus private overlay fields."""
    prof = load_profile(profile, ctx.paths)
    if cluster_id is not None and cluster_id != prof.cluster_id:
        raise InvalidArgument(
            f"profile {profile!r} is for cluster {prof.cluster_id!r}, not {cluster_id!r}",
            details={"hint": "copy the profile to $HPCLAB_HOME/profiles/<id>.yaml and change cluster_id"},
        )
    existing = load_overlay(prof.cluster_id, ctx.paths)
    base: dict[str, Any] = existing.model_dump() if existing else {"cluster_id": prof.cluster_id}
    updates = {
        "ssh_host": ssh_host,
        "user": user,
        "account": account,
        "qos": qos,
        "remote_root": remote_root,
        "default_partition": default_partition,
        "sim_root": sim_root,
        "ssh_options": ssh_options,
        "lab_users": lab_users,
    }
    base.update({k: v for k, v in updates.items() if v is not None})
    overlay = UserOverlay.model_validate(base)
    if default_partition is not None and default_partition not in prof.partitions:
        raise InvalidArgument(
            f"partition {default_partition!r} is not in profile {prof.cluster_id!r}",
            details={"partitions": sorted(prof.partitions)},
        )
    path = save_overlay(overlay, ctx.paths)
    created = prof.cluster_id not in ctx.clusters.ids()
    rc = ctx.clusters.register(prof, overlay)
    steps: list[str] = []
    summ = _summary(rc, path)
    if summ.missing:
        steps.append(
            f"set {', '.join(summ.missing)}: hpclab cluster add {rc.cluster_id} --profile {profile} ..."
        )
    if rc.profile.transport == "ssh":
        steps.append(f"authenticate once per session: hpclab auth {rc.cluster_id}")
    steps.append(f"check the cluster: hpclab cluster inspect {rc.cluster_id}")
    return ClusterData(cluster=summ, created=created, next_steps=steps)


def clusters_list(ctx: ServiceContext) -> ClustersListData:
    """Registered clusters with their effective settings and missing fields."""
    out = []
    for cid in ctx.clusters.ids():
        rc = ctx.clusters.get(cid)
        out.append(_summary(rc))
    return ClustersListData(clusters=out)


def cluster_get(ctx: ServiceContext, *, cluster_id: str) -> ClusterSummary:
    return _summary(ctx.clusters.get(cluster_id))


def profiles_list(ctx: ServiceContext) -> ProfilesListData:
    """Bundled and locally installed site profiles."""
    return ProfilesListData(
        profiles=[
            ProfileEntry(
                name=p.name,
                cluster_id=p.cluster_id,
                display_name=p.display_name,
                source=p.source,
                path=p.path,
                error=p.error,
            )
            for p in list_profiles(ctx.paths)
        ]
    )


# Where the simulator of a home whose path job scripts cannot carry is placed instead (first
# existing directory wins). /var/tmp survives reboots; /tmp is the fallback for systems without it.
_SIM_RELOCATION_BASES = ("/var/tmp", "/tmp")  # noqa: S108 - per-user 0700 dir, owner-checked


def _remote_path_ok(path: str) -> bool:
    """True if the job-script renderer accepts ``path`` as a remote root: the same check
    ``attempt_layout`` (every rendered script, and prepare's script_render) applies."""
    from hpclab.services.jobscript import attempt_layout

    try:
        attempt_layout(path, "att_probe")
    except InvalidArgument:
        return False
    return True


def _sim_root_ok(root: Path, user: str) -> bool:
    """True if a simulator can live at ``root``: job scripts accept its remote root (the
    renderer's own check), and ``root`` holds no PATH separator — the simulator puts
    ``<root>/bin`` (and ``<root>/gpu-bin`` for jobs) on PATH, where ':' would split the entry
    and ``sbatch`` would not be found."""
    return os.pathsep not in str(root) and _remote_path_ok(str(_sim_remote_root(root, user)))


def relocated_sim_root(home: Path) -> Path:
    """``<base>/hpclab-sim-<uid>-<sha256(home)[:10]>``: the simulator root of a home whose own
    path cannot appear in job scripts (persistent, one per home and user)."""
    tag = hashlib.sha256(os.fsencode(str(home))).hexdigest()[:10]
    name = f"hpclab-sim-{os.getuid()}-{tag}"
    for base in _SIM_RELOCATION_BASES:
        b = Path(base)
        if b.is_dir():
            return b.resolve() / name
    raise PreconditionFailed(
        f"cannot place the simulator: none of {', '.join(_SIM_RELOCATION_BASES)} exists",
        details={"hint": "set HPCLAB_HOME (or --home) to a plain path such as ~/.hpclab"},
    )


def _sim_remote_root(root: Path, user: str) -> Path:
    return root / "fs" / "scratch" / user / "hpclab"


def effective_sim_root(paths: Paths, cluster_id: str = "sim") -> Path:
    """The simulator root a registered sim cluster really uses: the overlay's ``sim_root``
    (``init_sim`` may have relocated it out of the home), else ``<home>/sim/<id>``.

    For callers without a ServiceContext (``hpclab sim fault|clear|reset``)."""
    overlay = load_overlay(cluster_id, paths)
    if overlay is not None and overlay.sim_root:
        return Path(overlay.sim_root)
    return paths.sim_root(cluster_id)


def init_sim(ctx: ServiceContext, *, cluster_id: Literal["sim"] = "sim") -> ClusterData:
    """Register the local Slurm simulator as a cluster (idempotent).

    The simulated "remote" filesystem lives under ``$HPCLAB_HOME/sim/<id>/fs``; jobs run as
    local processes owned by the current user. When the home's path has characters job scripts
    cannot carry (spaces, non-ASCII, shell metacharacters) or a ':' (which would split the
    simulator's PATH entries) the simulator is placed in
    ``/var/tmp/hpclab-sim-<uid>-<hash of the home>`` instead (0700, kept across calls and
    reboots), recorded in the overlay, and ``next_steps`` says so.
    """
    from hpclab.controller.files import ensure_private_dir
    from hpclab.sim.cluster import SimCluster

    user = getpass.getuser()
    default_root = ctx.paths.sim_root(cluster_id)
    root = default_root
    relocated = not _sim_root_ok(default_root, user)
    if relocated:
        root = relocated_sim_root(ctx.paths.home)
        try:
            ensure_private_dir(root, what="simulator directory")
        except HpcLabError as exc:
            # e.g. another user's directory or symlink under that predictable name: sticky /var/tmp
            # does not let us remove it, so the reliable way out is a home the simulator can use
            raise PreconditionFailed(
                exc.message,
                details={
                    **exc.details,
                    "hint": f"HPCLAB_HOME {ctx.paths.home} has characters job scripts cannot carry, so "
                    f"the simulator must live in {root}, which cannot be used. Remove {root} if it is "
                    "yours, or point HPCLAB_HOME (or --home) at a plain path such as ~/.hpclab "
                    "(letters, digits and ._-/ only)",
                },
            ) from None
    SimCluster.ensure(root)
    remote_root = _sim_remote_root(root, user)
    remote_root.mkdir(parents=True, exist_ok=True)
    data = cluster_register(
        ctx,
        profile="sim",
        user=user,
        account="lab",
        sim_root=str(root),
        remote_root=str(remote_root),
        default_partition="gpu",
    )
    steps: list[str] = []
    if relocated:
        steps.append(
            f"note: the simulator lives in {root} (private to you, kept across restarts), not under "
            f"HPCLAB_HOME {ctx.paths.home}: that path has characters (spaces, non-ASCII, shell "
            "metacharacters or ':') job scripts and the simulator's PATH cannot carry. If it is ever "
            "cleaned up, run `hpclab init --sim` again"
        )
    steps += [
        "copy the example project: hpclab examples copy tiny-classifier ./tiny-classifier",
        "then freeze it into a validated plan: hpclab prepare tiny-classifier/hpclab.yaml",
        "or take the guided tour first: hpclab demo",
    ]
    data.next_steps = steps
    return data


class AuthCommandData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    needed: bool
    argv: list[str]
    command: str
    master_alive: bool | None
    explanation: str


def auth_command(ctx: ServiceContext, *, cluster_id: str) -> AuthCommandData:
    """The interactive command that opens the SSH master connection hpclab reuses.

    The *user* runs it in their own terminal (MFA/Duo prompts go to them, never to hpclab);
    afterwards every controller call rides on that connection with ``BatchMode=yes``.
    """
    import shlex

    rc = ctx.clusters.get(cluster_id)
    if rc.profile.transport != "ssh":
        return AuthCommandData(
            cluster_id=cluster_id,
            needed=False,
            argv=[],
            command="",
            master_alive=None,
            explanation=f"cluster {cluster_id} uses transport {rc.profile.transport!r}; no SSH login needed",
        )
    transport = ctx.clusters.transport(cluster_id)
    login = getattr(transport, "login_command", None)
    check = getattr(transport, "check_master", None)
    if login is None:
        raise InvalidArgument(f"transport for {cluster_id} cannot produce a login command")
    argv = list(login())
    alive = bool(check()) if check is not None else None
    return AuthCommandData(
        cluster_id=cluster_id,
        needed=not alive,
        argv=argv,
        command=shlex.join(argv),
        master_alive=alive,
        explanation=(
            "an SSH master connection is already open; hpclab will reuse it"
            if alive
            else "run this in your terminal and complete MFA; hpclab reuses the connection (ControlPersist)"
        ),
    )


# ---------------------------------------------------------------------------
# auth_resume
# ---------------------------------------------------------------------------


class AuthResumeData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    woken: int  # auth-parked outbox items made due now
    clusters: list[str]  # clusters whose poll backoff was cleared
    note: str


_AUTH_PREFIX = "auth_required"
_MAX_EVENT_IDS = 50


def _cluster_of(tx: Tx, entity_id: str, *, follow_operation: bool = True) -> str | None:
    """The cluster an entity (by id prefix) belongs to, or None when it has none/is unknown."""
    prefix = entity_id.split("_", 1)[0]
    row = None
    if prefix == "att":
        row = tx.one("SELECT cluster_id FROM attempts WHERE attempt_id=?", (entity_id,))
    elif prefix == "art":
        row = tx.one(
            "SELECT a.cluster_id FROM artifacts x JOIN attempts a ON a.attempt_id=x.attempt_id "
            "WHERE x.artifact_id=?",
            (entity_id,),
        )
    elif prefix == "ev":
        row = tx.one(
            "SELECT a.cluster_id FROM evaluations e JOIN attempts a ON a.attempt_id=e.attempt_id "
            "WHERE e.evaluation_id=?",
            (entity_id,),
        )
    elif prefix == "run":
        row = tx.one(
            "SELECT e.cluster_id FROM runs r JOIN experiments e ON e.experiment_id=r.experiment_id "
            "WHERE r.run_id=?",
            (entity_id,),
        )
    elif prefix == "pln":
        row = tx.one(
            "SELECT e.cluster_id FROM plans p JOIN experiments e ON e.experiment_id=p.experiment_id "
            "WHERE p.plan_id=?",
            (entity_id,),
        )
    elif prefix == "wf":
        row = tx.one(
            "SELECT e.cluster_id FROM workflows w JOIN experiments e ON e.experiment_id=w.experiment_id "
            "WHERE w.workflow_id=?",
            (entity_id,),
        )
    elif prefix == "exp":
        row = tx.one("SELECT cluster_id FROM experiments WHERE experiment_id=?", (entity_id,))
    elif prefix == "op" and follow_operation:
        op = tx.one("SELECT target_id FROM operations WHERE operation_id=?", (entity_id,))
        if op is not None and op["target_id"]:
            return _cluster_of(tx, str(op["target_id"]), follow_operation=False)
    return str(row["cluster_id"]) if row is not None else None


def _item_clusters(tx: Tx, item: Any) -> set[str]:
    """Clusters an outbox item's target, payload ids and operation belong to."""
    ids: list[str] = []
    if item["target_id"]:
        ids.append(str(item["target_id"]))
    payload = jload(item["payload_json"], {}) or {}
    if isinstance(payload, dict):
        ids += [str(v) for k, v in payload.items() if k.endswith("_id") and isinstance(v, str) and v]
    if item["operation_id"]:
        ids.append(str(item["operation_id"]))
    out: set[str] = set()
    for entity_id in dict.fromkeys(ids):
        cid = _cluster_of(tx, entity_id)
        if cid is not None:
            out.add(cid)
    return out


def _poll_clusters(tx: Tx, cluster_id: str | None) -> Iterable[str]:
    if cluster_id is not None:
        return [cluster_id]
    rows = tx.all("SELECT DISTINCT cluster_id FROM poll_state ORDER BY cluster_id")
    return [str(r["cluster_id"]) for r in rows]


def auth_resume(ctx: ServiceContext, *, cluster_id: str | None = None) -> AuthResumeData:
    """Resume work that waits for a login, right after the user re-authenticated
    (``hpclab auth <cluster>``).

    Pending background items parked on ``auth_required`` (optionally only those of
    ``cluster_id``) become due now instead of after their pause, and the observer's poll
    backoff for the cluster (all clusters when omitted) is cleared so the next tick polls. If
    the login did not work, they simply park again. Nothing is submitted or re-run twice:
    every handler is crash-safe and re-entrant (invariants 2-4)."""
    if cluster_id is not None:
        ctx.clusters.get(cluster_id)  # NotFound for an unknown cluster
    now = iso(ctx.clock.now())
    with ctx.db.tx(source="human") as tx:
        rows = tx.all(
            "SELECT outbox_id, kind, operation_id, target_id, payload_json FROM outbox "
            "WHERE state='pending' AND substr(last_error, 1, ?) = ? AND not_before > ? ORDER BY outbox_id",
            (len(_AUTH_PREFIX), _AUTH_PREFIX, now),
        )
        woken = [r for r in rows if cluster_id is None or cluster_id in _item_clusters(tx, r)]
        for r in woken:
            tx.execute(
                "UPDATE outbox SET not_before=?, updated_at=? WHERE outbox_id=? AND state='pending'",
                (now, tx.now, r["outbox_id"]),
            )
        clusters = sorted(_poll_clusters(tx, cluster_id))
        cleared: list[dict[str, Any]] = []
        for cid in clusters:
            for ps in tx.all(
                "SELECT source, consecutive_failures, backoff_until FROM poll_state WHERE cluster_id=? "
                "AND (backoff_until IS NOT NULL OR consecutive_failures > 0)",
                (cid,),
            ):
                cleared.append(
                    {"cluster_id": cid, "source": ps["source"], "failures": ps["consecutive_failures"]}
                )
            tx.execute(
                "UPDATE poll_state SET backoff_until=NULL, consecutive_failures=0 WHERE cluster_id=?", (cid,)
            )
        tx.event(
            "cluster",
            cluster_id or "*",
            "cluster.auth_resumed",
            {
                "cluster_id": cluster_id,
                "woken": len(woken),
                "outbox_ids": [r["outbox_id"] for r in woken[:_MAX_EVENT_IDS]],
                "kinds": sorted({str(r["kind"]) for r in woken}),
                "clusters": clusters,
                "poll_backoff_cleared": cleared,
            },
        )
    where = f"cluster {cluster_id}" if cluster_id else "all clusters"
    if woken or cleared:
        note = (
            f"resumed {where}: {len(woken)} background item(s) waiting for login run now"
            + (f"; polling resumes ({len(cleared)} source(s) were backing off)" if cleared else "")
            + ". If the login did not work they wait again: run `hpclab auth "
            + (cluster_id or "<cluster>")
            + "`"
        )
    else:
        note = f"nothing was waiting for a login on {where}"
    log.info("auth_resume %s: woke %d item(s), cleared %d poll backoff(s)", where, len(woken), len(cleared))
    return AuthResumeData(woken=len(woken), clusters=clusters, note=note)
