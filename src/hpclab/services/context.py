"""Service context: the one object every service receives.

It carries the ledger, local paths, clock, the caller principal and a cluster registry that
builds (and caches) the Transport + Scheduler for a cluster. Tests swap factories to inject
fakes; production wires OpenSSH / Local / Sim transports and the Slurm scheduler.
"""

from __future__ import annotations

import getpass
import logging
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hpclab.adapters.base import Scheduler, Transport
from hpclab.config import Paths
from hpclab.domain.clock import Clock, SystemClock
from hpclab.domain.errors import HpcLabError, InvalidArgument, NotFound
from hpclab.domain.hashing import sha256_hex
from hpclab.profiles.schema import ClusterProfile, NotifyChannel, ResolvedCluster, UserOverlay
from hpclab.storage.db import Database, jdump, jload

log = logging.getLogger("hpclab.context")

TransportFactory = Callable[[ResolvedCluster, Paths], Transport]
SchedulerFactory = Callable[[Transport, ResolvedCluster, Clock], Scheduler]


# ---------------------------------------------------------------------------
# Notify channels in the ledger: ids, kinds and events only, never the target
# ---------------------------------------------------------------------------


def notify_channel_id(kind: str, target: str | None) -> str:
    """Secret-free, stable reference to a notify channel: ``<kind>:<sha256(target)[:12]>``."""
    return f"{kind}:{sha256_hex(target or '')[:12]}"


@dataclass(frozen=True)
class NotifyChannelRef:
    """A notify channel as ``clusters.overlay_json`` records it.

    The target (ntfy / webhook URL, command line) may embed a secret, and the ledger is the file
    users back up and attach to bug reports, so it stays in the 0600 overlay file and in the
    memory of the process that registered it (``ClusterRegistry.notify_channels``)."""

    channel_id: str
    kind: str
    events: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"channel_id": self.channel_id, "kind": self.kind, "events": list(self.events)}


def ledger_overlay(overlay: UserOverlay) -> dict[str, Any]:
    """The overlay as the ledger stores it: notify targets replaced by their channel ids."""
    data = overlay.model_dump(mode="json", exclude={"notify"})
    data["notify"] = [
        NotifyChannelRef(notify_channel_id(ch.kind, ch.target), ch.kind, tuple(ch.events)).to_json()
        for ch in overlay.notify
    ]
    return data


def _parse_stored_notify(
    overlay_data: Any, cluster_id: str
) -> list[tuple[NotifyChannelRef, NotifyChannel | None]]:
    """``(ref, legacy channel)`` for every valid ``notify`` entry of a stored overlay.

    The channel (with its target) is returned only for rows written before targets were kept
    out of the ledger (``{kind, target, events}``); invalid entries are skipped."""
    raw_list = overlay_data.get("notify") if isinstance(overlay_data, dict) else None
    out: list[tuple[NotifyChannelRef, NotifyChannel | None]] = []
    for raw in raw_list or []:
        try:
            if not isinstance(raw, dict):
                raise TypeError("not a mapping")
            if "channel_id" in raw:
                cid = raw["channel_id"]
                nc = NotifyChannel.model_validate({k: v for k, v in raw.items() if k != "channel_id"})
                if nc.target is not None or not isinstance(cid, str) or not cid.startswith(f"{nc.kind}:"):
                    raise ValueError("malformed channel reference")
                out.append((NotifyChannelRef(cid, nc.kind, tuple(nc.events)), None))
            else:
                nc = NotifyChannel.model_validate(raw)
                ref = NotifyChannelRef(notify_channel_id(nc.kind, nc.target), nc.kind, tuple(nc.events))
                out.append((ref, nc))
        except (TypeError, ValueError, ValidationError):  # never echo the entry: it may hold a secret
            log.warning("cluster %s: ignoring an invalid stored notify channel", cluster_id)
    return out


def notify_channel_refs(overlay_data: Any, cluster_id: str = "?") -> list[NotifyChannelRef]:
    """The notify channels of a stored ``overlay_json`` (parsed), without targets."""
    return [ref for ref, _ in _parse_stored_notify(overlay_data, cluster_id)]


def default_transport_factory(cluster: ResolvedCluster, paths: Paths) -> Transport:
    kind = cluster.profile.transport
    if kind == "ssh":
        from hpclab.adapters.openssh import OpenSSHTransport

        host = cluster.overlay.ssh_host or cluster.profile.login_host
        if not host:
            raise InvalidArgument(
                f"cluster {cluster.cluster_id}: no ssh_host in overlay and no login_host in profile"
            )
        return OpenSSHTransport(
            cluster.cluster_id,
            host,
            user=cluster.overlay.user,
            ssh_options=list(cluster.overlay.ssh_options),
            control_dir=paths.home / "run",
        )
    if kind == "local":
        from hpclab.adapters.local import LocalTransport

        return LocalTransport(cluster.cluster_id)
    if kind == "sim":
        from hpclab.sim.transport import SimTransport

        root = (
            Path(cluster.overlay.sim_root) if cluster.overlay.sim_root else paths.sim_root(cluster.cluster_id)
        )
        return SimTransport(cluster.cluster_id, root)
    raise InvalidArgument(f"unknown transport {kind!r}")


def default_scheduler_factory(transport: Transport, cluster: ResolvedCluster, clock: Clock) -> Scheduler:
    from hpclab.adapters.slurm import SlurmScheduler

    return SlurmScheduler(transport, cluster, clock=clock)


class ClusterRegistry:
    """Resolved clusters live in the ``clusters`` table (profile + overlay JSON).

    The stored overlay holds no notify targets (see ``NotifyChannelRef``); ``get`` puts them
    back from this registry's memory (targets it registered or has read) or, e.g. after a
    restart, from the cluster's 0600 overlay file, matching by channel id so a target is never
    guessed. A channel whose target neither source has any more is left out.
    """

    def __init__(
        self,
        db: Database,
        paths: Paths,
        clock: Clock,
        transport_factory: TransportFactory = default_transport_factory,
        scheduler_factory: SchedulerFactory = default_scheduler_factory,
    ) -> None:
        self.db = db
        self.paths = paths
        self.clock = clock
        self.transport_factory = transport_factory
        self.scheduler_factory = scheduler_factory
        self._transports: dict[str, Transport] = {}
        self._schedulers: dict[str, Scheduler] = {}
        self._lock = threading.Lock()
        # channel id -> target, for targets this process registered or read; never persisted
        self._notify_targets: dict[str, str | None] = {}
        # overlay file version (mtime, size, inode) last read per cluster: read once per version
        self._overlay_stamps: dict[str, tuple[int, int, int]] = {}

    # -- registration --------------------------------------------------------
    def register(self, profile: ClusterProfile, overlay: UserOverlay) -> ResolvedCluster:
        if overlay.cluster_id != profile.cluster_id:
            raise InvalidArgument("overlay.cluster_id must match profile.cluster_id")
        resolved = ResolvedCluster(profile=profile, overlay=overlay)
        self._remember_targets(overlay.notify)
        with self.db.tx(source="api") as tx:
            existing = tx.one("SELECT cluster_id FROM clusters WHERE cluster_id=?", (profile.cluster_id,))
            row = {
                "profile_json": jdump(profile.model_dump(mode="json", by_alias=True)),
                "overlay_json": jdump(ledger_overlay(overlay)),
                "profile_revision": resolved.revision,
            }
            if existing:
                tx.execute(
                    "UPDATE clusters SET profile_json=?, overlay_json=?, profile_revision=?, updated_at=? WHERE cluster_id=?",
                    (
                        row["profile_json"],
                        row["overlay_json"],
                        row["profile_revision"],
                        tx.now,
                        profile.cluster_id,
                    ),
                )
            else:
                tx.insert(
                    "clusters",
                    {"cluster_id": profile.cluster_id, **row, "created_at": tx.now, "updated_at": tx.now},
                )
            tx.event("cluster", profile.cluster_id, "cluster.registered", {"revision": resolved.revision})
        self.invalidate(profile.cluster_id)
        return resolved

    def invalidate(self, cluster_id: str) -> None:
        with self._lock:
            t = self._transports.pop(cluster_id, None)
            self._schedulers.pop(cluster_id, None)
        if t is not None:
            try:
                t.close()
            except Exception:  # closing a stale transport is best effort
                log.debug("closing transport for %s failed", cluster_id, exc_info=True)

    def ids(self) -> list[str]:
        return [r["cluster_id"] for r in self.db.all("SELECT cluster_id FROM clusters ORDER BY cluster_id")]

    def get(self, cluster_id: str) -> ResolvedCluster:
        row = self.db.one("SELECT * FROM clusters WHERE cluster_id=?", (cluster_id,))
        if row is None:
            raise NotFound(
                f"cluster {cluster_id!r} is not registered",
                details={"known_clusters": self.ids(), "hint": "hpclab cluster add <id> --profile <profile>"},
            )
        stored = jload(row["overlay_json"])
        if isinstance(stored, dict):
            notify = self._resolve_notify(cluster_id, _parse_stored_notify(stored, cluster_id))
            stored = {**stored, "notify": notify}
        return ResolvedCluster(
            profile=ClusterProfile.model_validate(jload(row["profile_json"])),
            overlay=UserOverlay.model_validate(stored),
        )

    def notify_channels(self, cluster_id: str) -> list[NotifyChannel]:
        """The cluster's notify channels (with targets) as of its last registration; [] for an
        unknown cluster. Channels whose target is no longer available are left out."""
        row = self.db.one("SELECT overlay_json FROM clusters WHERE cluster_id=?", (cluster_id,))
        if row is None:
            return []
        return self._resolve_notify(cluster_id, _parse_stored_notify(jload(row["overlay_json"]), cluster_id))

    # -- notify targets (kept out of the ledger) ---------------------------------
    def _remember_targets(self, channels: list[NotifyChannel]) -> None:
        with self._lock:
            for ch in channels:
                self._notify_targets[notify_channel_id(ch.kind, ch.target)] = ch.target

    def _lookup_target(self, ref: NotifyChannelRef) -> tuple[bool, str | None]:
        if ref.channel_id == notify_channel_id(ref.kind, None):
            return True, None  # a channel without a target (log, desktop)
        with self._lock:
            if ref.channel_id in self._notify_targets:
                return True, self._notify_targets[ref.channel_id]
        return False, None

    def _resolve_notify(
        self, cluster_id: str, parsed: list[tuple[NotifyChannelRef, NotifyChannel | None]]
    ) -> list[NotifyChannel]:
        refs = [ref for ref, _ in parsed]
        self._remember_targets([legacy for _, legacy in parsed if legacy is not None])
        missing = [ref.channel_id for ref in refs if not self._lookup_target(ref)[0]]
        if missing and self._read_overlay_file(cluster_id):
            missing = [ref.channel_id for ref in refs if not self._lookup_target(ref)[0]]
            if missing:  # logged once per overlay file version
                log.warning(
                    "cluster %s: notify channel(s) %s are not in the overlay file any more; they "
                    "cannot be delivered until the cluster is re-registered (hpclab cluster add %s ...)",
                    cluster_id,
                    ", ".join(missing),
                    cluster_id,
                )
        out: list[NotifyChannel] = []
        for ref in refs:
            ok, target = self._lookup_target(ref)
            if ok:
                out.append(
                    NotifyChannel.model_validate(
                        {"kind": ref.kind, "target": target, "events": list(ref.events)}
                    )
                )
        return out

    def _read_overlay_file(self, cluster_id: str) -> bool:
        """Learn the targets in the cluster's private overlay file; False when the file is
        missing or this version of it was read already."""
        from hpclab.profiles.loader import load_overlay, overlay_path

        try:
            st = overlay_path(cluster_id, self.paths).stat()
        except (OSError, HpcLabError):
            return False
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        with self._lock:
            if self._overlay_stamps.get(cluster_id) == stamp:
                return False
            self._overlay_stamps[cluster_id] = stamp
        try:
            overlay = load_overlay(cluster_id, self.paths)
        except Exception as exc:  # noqa: BLE001 - never break get(); log no values: the file holds secrets
            reason = exc.message if isinstance(exc, HpcLabError) else type(exc).__name__
            log.warning(
                "cluster %s: cannot read notify targets from the overlay file: %s", cluster_id, reason
            )
            return True
        if overlay is not None:
            self._remember_targets(overlay.notify)
        return True

    def redact_stored_notify_targets(self) -> int:
        """Rewrite ``clusters`` rows stored before notify targets were kept out of the ledger.

        Idempotent; returns the number of rows rewritten. The targets stay known to this
        process, and other processes read them from the 0600 overlay file."""
        todo: list[str] = []
        for row in self.db.all("SELECT cluster_id, overlay_json FROM clusters ORDER BY cluster_id"):
            if self._redacted(row["cluster_id"], row["overlay_json"]) is not None:
                todo.append(row["cluster_id"])
        if not todo:
            return 0
        changed = 0
        with self.db.tx(source="migration") as tx:
            for cid in todo:
                current = tx.one("SELECT overlay_json FROM clusters WHERE cluster_id=?", (cid,))
                new = self._redacted(cid, current["overlay_json"]) if current else None
                if new is not None:
                    tx.execute("UPDATE clusters SET overlay_json=? WHERE cluster_id=?", (new, cid))
                    changed += 1
        log.info("removed notify channel targets from %d stored cluster overlay(s)", changed)
        return changed

    def _redacted(self, cluster_id: str, overlay_json: str) -> str | None:
        """The stored overlay without notify targets, or None when it has none already."""
        data = jload(overlay_json)
        if not isinstance(data, dict) or not data.get("notify"):
            return None
        parsed = _parse_stored_notify(data, cluster_id)
        self._remember_targets([legacy for _, legacy in parsed if legacy is not None])
        new = jdump({**data, "notify": [ref.to_json() for ref, _ in parsed]})
        return None if new == overlay_json else new

    def transport(self, cluster_id: str) -> Transport:
        with self._lock:
            t = self._transports.get(cluster_id)
        if t is None:
            t = self.transport_factory(self.get(cluster_id), self.paths)
            with self._lock:
                t = self._transports.setdefault(cluster_id, t)
        return t

    def scheduler(self, cluster_id: str) -> Scheduler:
        with self._lock:
            s = self._schedulers.get(cluster_id)
        if s is None:
            s = self.scheduler_factory(self.transport(cluster_id), self.get(cluster_id), self.clock)
            with self._lock:
                s = self._schedulers.setdefault(cluster_id, s)
        return s

    def close(self) -> None:
        for cid in list(self._transports):
            self.invalidate(cid)


def default_principal() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


@dataclass
class Settings:
    """Runtime knobs; profile policy values take precedence where they exist."""

    request_deadline_s: float = 20.0  # max time a synchronous remote call may take in a request
    prepare_sync_max_bytes: int = 32 * 1024 * 1024  # snapshots above this size are prepared in the background
    worker_threads: int = 2
    tick_interval_s: float = 1.0
    min_poll_interval_s: float | None = None  # override profile poll interval (sim/tests)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceContext:
    db: Database
    paths: Paths
    clusters: ClusterRegistry
    clock: Clock = field(default_factory=SystemClock)
    principal: str = field(default_factory=default_principal)
    settings: Settings = field(default_factory=Settings)

    @classmethod
    def create(
        cls,
        paths: Paths,
        *,
        clock: Clock | None = None,
        transport_factory: TransportFactory = default_transport_factory,
        scheduler_factory: SchedulerFactory = default_scheduler_factory,
        settings: Settings | None = None,
        db: Database | None = None,
    ) -> ServiceContext:
        paths.ensure()
        clk = clock or SystemClock()
        database = db or Database(paths.db, clock=clk)
        registry = ClusterRegistry(database, paths, clk, transport_factory, scheduler_factory)
        try:
            registry.redact_stored_notify_targets()
        except Exception:  # a ledger we cannot rewrite now must not stop the process
            log.warning("could not remove notify targets from stored cluster overlays", exc_info=True)
        return cls(db=database, paths=paths, clusters=registry, clock=clk, settings=settings or Settings())


def load_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
