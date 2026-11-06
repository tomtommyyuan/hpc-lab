"""Cluster profile (public, shareable) + user overlay (private) = resolved cluster config.

* ``ClusterProfile``: site rules — partitions, storage roles, policies, runtime recipes,
  poll limits, known issues *as observations with source and expiry*. No usernames,
  accounts, private paths or credentials.
* ``UserOverlay``: SSH alias, remote user, account, QoS, remote root. Lives under the local
  hpclab home, never in a public profile. Contains *references* to existing SSH auth only.
* ``ResolvedCluster``: what services use; ``revision`` is a digest of the non-secret parts,
  stored in every ExecutionPlan so results can say which rules they ran under.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.domain.hashing import digest


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PartitionRule(Model):
    max_time: str | None = None
    gpu_types: list[str] = Field(default_factory=list)
    preemptible: bool = False
    interactive_only: bool = False
    max_gpus_per_job: int | None = None
    max_gpus_per_user: int | None = None
    default: bool = False
    notes: str | None = None


class StorageRole(Model):
    path: str  # may contain {user}
    quota_hint: str | None = None
    avoid_for: list[str] = Field(default_factory=list)  # e.g. ["outputs", "venv", "checkpoints"]
    shared_fs: bool = True
    notes: str | None = None


class SitePolicy(Model):
    login_node_compute: Literal["forbidden", "discouraged", "allowed"] = "forbidden"
    require_account: bool = True
    require_qos: bool = False
    poll_min_interval_s: float = 30.0  # minimum seconds between squeue batches per cluster
    sacct_min_interval_s: float = 60.0
    max_login_hash_bytes: int = 256 * 1024 * 1024  # larger files are hashed on compute nodes only
    reconcile_window_s: float = 900.0  # how long to keep looking for an uncertain submission
    accounting_lag_s: float = 120.0
    max_parallel_default: int = 4
    submit_rate_per_min: int = 20


class RuntimeRecipe(Model):
    description: str
    setup: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


class SiteObservation(Model):
    """A time-limited fact with provenance (e.g. a node with NFS trouble)."""

    key: str
    value: Any
    source: str
    observed_at: str
    expires_at: str | None = None
    severity: Literal["info", "warning", "critical"] = "info"


class ProfileSource(Model):
    title: str
    url: str | None = None
    retrieved_at: str | None = None
    notes: str | None = None


class ClusterProfile(Model):
    schema_: str = Field(default="hpclab.profile/v1", alias="schema")
    cluster_id: str
    display_name: str
    scheduler: Literal["slurm"] = "slurm"
    transport: Literal["ssh", "local", "sim"] = "ssh"
    login_host: str | None = None
    partitions: dict[str, PartitionRule] = Field(default_factory=dict)
    default_partition: str | None = None
    storage: dict[str, StorageRole] = Field(default_factory=dict)
    remote_root_template: str = "{scratch}/hpclab"
    policy: SitePolicy = Field(default_factory=SitePolicy)
    runtime_recipes: dict[str, RuntimeRecipe] = Field(default_factory=dict)
    gpu_arch: dict[str, str] = Field(default_factory=dict)  # gpu_type -> sm arch
    # Time-varying site facts with provenance, e.g. key "driver_cuda" value "12.8". Probes
    # (cluster_inspect) refresh them; a stale fact is reported with its observed_at.
    facts: list[SiteObservation] = Field(default_factory=list)
    known_issues: list[SiteObservation] = Field(default_factory=list)
    sources: list[ProfileSource] = Field(default_factory=list)
    support_contact: str | None = None
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class NotifyChannel(Model):
    kind: Literal["desktop", "ntfy", "slack_webhook", "command", "log"]
    target: str | None = None  # ntfy topic URL, webhook URL, or command path
    events: list[str] = Field(
        default_factory=lambda: [
            "run.succeeded",
            "run.failed",
            "attempt.uncertain",
            "workflow.finished",
            "budget.threshold",
        ]
    )


class UserOverlay(Model):
    cluster_id: str
    ssh_host: str | None = None  # ssh config alias or host name
    user: str | None = None  # expected remote Unix user (verified by cluster_inspect)
    account: str | None = None
    qos: str | None = None
    remote_root: str | None = None  # overrides profile remote_root_template
    default_partition: str | None = None
    ssh_options: list[str] = Field(default_factory=list)  # extra -o options; no secrets
    sim_root: str | None = None  # for transport=sim
    lab_users: list[str] = Field(default_factory=list)  # for PI usage reports
    notify: list[NotifyChannel] = Field(default_factory=list)


class ResolvedCluster(Model):
    profile: ClusterProfile
    overlay: UserOverlay

    @property
    def cluster_id(self) -> str:
        return self.profile.cluster_id

    @property
    def revision(self) -> str:
        public = self.profile.model_dump(mode="json", by_alias=True)
        private = self.overlay.model_dump(mode="json", exclude={"notify", "lab_users"})
        return digest({"profile": public, "overlay": private})

    def _substitute(self, text: str) -> str:
        """Replace the placeholders we can resolve; unknown ones stay visible (never "$USER"/"~",
        which sbatch --output/--chdir and quoted paths would not expand)."""
        out = text
        if self.overlay.user:
            out = out.replace("{user}", self.overlay.user)
        if self.overlay.sim_root:
            out = out.replace("{sim_fs}", self.overlay.sim_root.rstrip("/") + "/fs")
        return out

    def storage_path(self, role: str) -> str | None:
        r = self.profile.storage.get(role)
        if r is None:
            return None
        return self._substitute(r.path)

    def expand(self, text: str) -> str:
        """Expand ``{user}``, ``{scratch}``, ``{home}``, ``{sim_fs}``, ``{remote_root}``.

        Placeholders that cannot be resolved (e.g. ``{user}`` with no user in the overlay) are
        left in place so callers can detect and report them.
        """
        out = self._substitute(text)
        for role in ("scratch", "home"):
            p = self.storage_path(role)
            if p:
                out = out.replace("{" + role + "}", p)
        if "{remote_root}" in out:
            out = out.replace("{remote_root}", self.remote_root())
        return out

    def remote_root(self) -> str:
        """Absolute cluster path under which hpclab keeps snapshots and attempts.

        Raises ``PreconditionFailed`` when it cannot be resolved to a concrete absolute path.
        """
        from hpclab.domain.errors import PreconditionFailed

        if self.overlay.remote_root:
            root = self._substitute(self.overlay.remote_root)
        else:
            root = self.expand(self.profile.remote_root_template.replace("{remote_root}", ""))
        if "{" in root or "$" in root or not root.startswith("/"):
            raise PreconditionFailed(
                f"cannot resolve the remote root for cluster {self.cluster_id!r} (got {root!r})",
                details={
                    "cluster_id": self.cluster_id,
                    "template": self.overlay.remote_root or self.profile.remote_root_template,
                    "hint": f"set your cluster user and/or remote root: hpclab cluster add {self.cluster_id} "
                    "--profile <profile> --user <you> [--remote-root /abs/path]",
                },
            )
        return root.rstrip("/")
