"""Support for the prepare tests (imported by name, pytest prepend mode): a registered local "lab"
cluster whose *remote* filesystem is this machine (LocalTransport), a fake scheduler for
``sbatch --test-only``, and a small non-git project to snapshot."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpclab.adapters.base import SubmitResponse, TransportError
from hpclab.adapters.local import LocalTransport
from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.profiles.schema import (
    ClusterProfile,
    PartitionRule,
    ResolvedCluster,
    SiteObservation,
    SitePolicy,
    StorageRole,
    UserOverlay,
)
from hpclab.services.context import ServiceContext, Settings

TEST_ONLY_RAW = "sbatch: Job 4242 to start at 1788000000 using 2 processors on nodes n001 in partition gpu"


@dataclass
class FakeScheduler:
    """Records ``submit(test_only=True)`` calls; behaviour is set per test."""

    cluster_id: str = "lab"
    mode: str = "ok"  # ok | reject | reject_later | error | slow
    calls: list[dict[str, Any]] = field(default_factory=list)

    def submit(
        self, script_path: str, *, workdir: str, test_only: bool = False, receipt_path: str | None = None
    ) -> SubmitResponse:
        from hpclab.adapters.base import SubmissionRejected

        text = Path(script_path).read_text() if Path(script_path).exists() else ""
        self.calls.append(
            {"script_path": script_path, "workdir": workdir, "test_only": test_only, "script": text}
        )
        if not test_only:
            raise AssertionError("prepare must never submit a real job")
        if self.mode == "reject":
            raise SubmissionRejected(
                "sbatch refused: Invalid account or account/partition combination specified",
                details={"reason_code": "invalid_account", "retry_later": False, "stderr": "Invalid account"},
            )
        if self.mode == "reject_later":
            raise SubmissionRejected(
                "sbatch refused: QOSMaxSubmitJobPerUserLimit",
                details={"reason_code": "submit_limit", "retry_later": True},
            )
        if self.mode == "error":
            raise TransportError("sbatch --test-only could not be evaluated")
        if self.mode == "slow":
            time.sleep(3)
        return SubmitResponse(
            job_id="", cluster_name=None, raw=TEST_ONLY_RAW, submitted_at="2026-09-01T12:00:00Z"
        )


@dataclass
class Lab:
    ctx: ServiceContext
    root: Path  # fake cluster filesystem root
    project: Path
    scheduler: FakeScheduler
    clock: FakeClock

    @property
    def scratch(self) -> Path:
        return self.root / "scratch" / "alice"

    @property
    def remote_root(self) -> Path:
        return self.scratch / "hpclab"

    def spec(self, **updates: Any) -> dict[str, Any]:
        spec = copy.deepcopy(BASE_SPEC)
        spec["code"] = {"project_root": str(self.project)}
        for key, value in updates.items():
            if value is None:
                spec.pop(key, None)
            else:
                spec[key] = value
        return spec


BASE_SPEC: dict[str, Any] = {
    "name": "mixup-ablation",
    "question": "does mixup help?",
    "cluster": "lab",
    "entrypoint": {
        "argv": ["python", "train.py", "--method", "{method}", "--seed", "{seed}", "--out", "{output_dir}"]
    },
    "runtime": {"python": "python3"},
    "matrix": {"method": ["baseline", "mixup"]},
    "seeds": [1, 2, 3],
    "resources": {"gpus": 1, "gpu_type": "a100", "time": "01:00:00", "mem": "8G"},
    "outputs": [{"name": "metrics", "path": "{output_dir}/metrics.jsonl", "role": "metrics"}],
    "smoke": {"argv_append": ["--steps", "5"], "require_outputs": ["metrics"]},
    "budget": {"gpu_hours": 50, "max_parallel": 2},
    "evaluation": {"output": "metrics"},
}


def lab_profile(root: Path, **updates: Any) -> ClusterProfile:
    data: dict[str, Any] = {
        "cluster_id": "lab",
        "display_name": "Test lab cluster",
        "transport": "local",
        "partitions": {
            "gpu": PartitionRule(
                max_time="2-00:00:00", gpu_types=["a100", "b200"], max_gpus_per_job=8, default=True
            ),
            "interactive": PartitionRule(
                max_time="04:00:00",
                gpu_types=["a100"],
                interactive_only=True,
                max_gpus_per_job=1,
                max_gpus_per_user=1,
            ),
            "scavenge": PartitionRule(max_time="1-00:00:00", gpu_types=["h100"], preemptible=True),
            "cpu": PartitionRule(max_time="12:00:00"),
        },
        "default_partition": "gpu",
        "storage": {
            "home": StorageRole(
                path=str(root / "home" / "{user}"), avoid_for=["outputs", "checkpoints", "venv", "caches"]
            ),
            "scratch": StorageRole(path=str(root / "scratch" / "{user}")),
        },
        "remote_root_template": "{scratch}/hpclab",
        "policy": SitePolicy(require_account=True),
        "gpu_arch": {"a100": "sm_80", "b200": "sm_100", "h100": "sm_90"},
        "facts": [
            SiteObservation(
                key="driver_cuda",
                value="12.8",
                source="test",
                observed_at="2026-08-01T00:00:00Z",
                expires_at="2026-12-01T00:00:00Z",
            )
        ],
    }
    data.update(updates)
    return ClusterProfile(**data)


def lab_overlay(**updates: Any) -> UserOverlay:
    data: dict[str, Any] = {"cluster_id": "lab", "user": "alice", "account": "lab-acct"}
    data.update(updates)
    return UserOverlay(**data)


def write_project(project: Path) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    (project / "train.py").write_text("import sys\nprint('training', sys.argv)\n")
    (project / "uv.lock").write_text("version = 1\n")
    (project / "pyproject.toml").write_text('[project]\nname = "mypkg"\nversion = "0.1"\n')
    (project / "mypkg").mkdir(exist_ok=True)
    (project / "mypkg" / "__init__.py").write_text("")
    return project


def make_lab(paths: Paths, tmp_path: Path) -> tuple[Lab, ServiceContext]:
    clock = FakeClock()
    root = tmp_path / "cluster-fs"
    (root / "scratch" / "alice").mkdir(parents=True)
    (root / "home" / "alice").mkdir(parents=True)
    scheduler = FakeScheduler()
    ctx = ServiceContext.create(
        paths,
        clock=clock,
        transport_factory=lambda cluster, p: LocalTransport(cluster.cluster_id, user="alice"),
        scheduler_factory=lambda transport, cluster, clk: scheduler,  # type: ignore[arg-type,return-value]
        settings=Settings(request_deadline_s=10.0),
    )
    ctx.clusters.register(lab_profile(root), lab_overlay())
    project = write_project(tmp_path / "project")
    return Lab(ctx=ctx, root=root, project=project, scheduler=scheduler, clock=clock), ctx


def resolved(lab: Lab, **profile_updates: Any) -> ResolvedCluster:
    return ResolvedCluster(profile=lab_profile(lab.root, **profile_updates), overlay=lab_overlay())


# ---------------------------------------------------------------------------
# PlanContext builder for table-driven check tests
# ---------------------------------------------------------------------------

FAKE_DIGEST = "sha256:" + "ab" * 32


def manifest_of(files: dict[str, int] | None = None, *, digest: str = FAKE_DIGEST) -> dict[str, Any]:
    """A snapshot manifest; ``files`` maps path -> mode."""
    files = {"train.py": 0o644, "uv.lock": 0o644} if files is None else files
    entries = [
        {"path": p, "sha256": "sha256:" + format(i, "064x"), "size": 10, "mode": mode}
        for i, (p, mode) in enumerate(sorted(files.items()))
    ]
    return {
        "schema": "hpclab.snapshot/v1",
        "digest": digest,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": 10 * len(entries),
    }


def plan_context(
    spec: dict[str, Any],
    cluster: ResolvedCluster | None,
    *,
    manifest: dict[str, Any] | str | None = "default",
    remote_root: str | bool | None = True,
    **pc_kw: Any,
) -> Any:
    """Build a ``PlanContext`` the way experiment_prepare does (runs via the real builder)."""
    from hpclab.domain.models import CodeSnapshotRef, ExperimentSpec
    from hpclab.services import experiments as ex
    from hpclab.services.validation import PlanContext

    es = ExperimentSpec.model_validate(spec)
    man = manifest_of() if manifest == "default" else manifest
    rr: str | None
    if remote_root is True:
        rr = cluster.remote_root() if cluster is not None else None
    elif remote_root is False:
        rr = None
    else:
        rr = remote_root
    env = ex._Env(
        cluster, None if cluster else "cluster 'lab' is not registered", rr, None if rr else "no remote root"
    )
    digest = man["digest"] if isinstance(man, dict) else FAKE_DIGEST
    code = CodeSnapshotRef(digest=digest, file_count=1, total_bytes=1)
    target = ex._Target(mode="new", spec=es, experiment_id="exp_test", plan_kind="matrix")
    runs, smoke_error = ex._build_matrix_runs(target, env, code)
    return PlanContext(
        spec=es,
        cluster=cluster,
        runs=runs,
        now=FakeClock().now(),
        cluster_error=env.cluster_error,
        remote_root=rr,
        remote_root_error=env.remote_root_error,
        manifest=man if isinstance(man, dict) else None,
        smoke_error=pc_kw.pop("smoke_error", smoke_error),
        **pc_kw,
    )


def make_venv(
    path: Path,
    *,
    cfg: str | None = "home = /usr/bin\nversion_info = 3.12.3\ninclude-system-site-packages = false\n",
    torch_cuda: str | None = "absent",
    editable_dist: dict[str, bool] | None = None,
    pth: list[str] | None = None,
) -> Path:
    """A fake virtualenv on the fake cluster filesystem.

    ``torch_cuda``: "absent" (no torch), None (CPU build) or a CUDA version like "12.8".
    ``editable_dist``: dist name -> editable flag of its direct_url.json.
    """
    path.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (path / "pyvenv.cfg").write_text(cfg)
    site = path / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    if torch_cuda != "absent":
        (site / "torch").mkdir(exist_ok=True)
        cuda = "None" if torch_cuda is None else f"'{torch_cuda}'"
        tag = "cpu" if torch_cuda is None else "cu" + torch_cuda.replace(".", "")
        (site / "torch" / "version.py").write_text(
            "from typing import Optional\n\n__all__ = ['__version__']\n"
            f"__version__ = '2.8.0+{tag}'\ndebug = False\ncuda: Optional[str] = {cuda}\n"
        )
    for name, editable in (editable_dist or {}).items():
        d = site / f"{name}-0.1.dist-info"
        d.mkdir(exist_ok=True)
        (d / "direct_url.json").write_text(
            json.dumps({"url": f"file:///cluster/src/{name}", "dir_info": {"editable": editable}})
        )
    for name in pth or []:
        (site / name).write_text("/cluster/src\n")
    return path
