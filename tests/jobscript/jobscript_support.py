"""Harness for executing rendered job wrappers locally with /bin/bash (macOS: bash 3.2).

Imported by name (pytest prepend mode). Builds a fake published snapshot, renders
``job.sbatch`` with ``render_job_script`` and runs it the way Slurm would: cwd = attempt dir
(``--chdir``), stdout/stderr appended to the log files, SLURM_* variables set by hand.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpclab.domain.models import (
    CodeSnapshotRef,
    EntrypointSpec,
    ExecutionPlan,
    InputRef,
    OutputDecl,
    RecoverySource,
    ResilienceSpec,
    ResourceSpec,
    RunSpec,
    RuntimeSpec,
)
from hpclab.profiles.schema import ClusterProfile, ResolvedCluster, StorageRole, UserOverlay
from hpclab.services.jobscript import AttemptLayout, attempt_layout, job_name_for, render_job_script

BASH = "/bin/bash"
TOKEN = "k7m2p9q4r8s3"
RUN_ID = "run_01TEST"
ATTEMPT_ID = "att_01TEST"
SNAPSHOT_HEX = "ab" * 32

# A program whose behaviour is steered by PROG_MODE; it always dumps argv/env/cwd first.
PROG = r"""
import json, os, signal, sys, time
out = os.environ["HPCLAB_OUTPUT_DIR"]
os.makedirs(out, exist_ok=True)
mode = os.environ.get("PROG_MODE", "dump")
with open(os.path.join(out, "dump.json"), "w") as fh:
    json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ)}, fh)
def ready():
    open(os.path.join(out, "ready"), "w").close()
if mode == "exit":
    sys.exit(int(os.environ["PROG_EXIT"]))
if mode == "sleep":
    ready()
    time.sleep(float(os.environ.get("PROG_SLEEP", "30")))
if mode == "usr1":
    def on_usr1(signum, frame):
        with open(os.path.join(out, "got_usr1"), "w") as fh:
            fh.write("stop_file=%s" % os.path.exists(os.environ["HPCLAB_STOP_FILE"]))
        sys.exit(0)
    signal.signal(signal.SIGUSR1, on_usr1)
    ready()
    time.sleep(30)
    sys.exit(9)
if mode == "requeue":
    code = os.environ["HPCLAB_CODE_DIR"]
    marker = os.path.join(code, "segment-marker")
    if not os.path.exists(marker):
        open(marker, "w").write("first segment")
    ckpt = os.path.join(os.environ["HPCLAB_CHECKPOINT_DIR"], "step-5")
    os.makedirs(ckpt, exist_ok=True)
    open(os.path.join(ckpt, "model.bin"), "wb").write(b"weights")
    json.dump({"schema": "hpclab.checkpoint/v1", "step": 5, "complete": True, "resumable": True,
               "files": [{"path": "model.bin", "size": 7}]}, open(os.path.join(ckpt, "hpclab-ckpt.json"), "w"))
if mode == "outputs":
    open(os.path.join(out, "metrics.jsonl"), "w").write('{"step": 1}\n')
    os.makedirs(os.path.join(out, "preds", "sub"), exist_ok=True)
    open(os.path.join(out, "preds", "a.txt"), "w").write("A")
    open(os.path.join(out, "preds", "sub", "b.txt"), "w").write("B")
    open(os.path.join(os.environ["HPCLAB_CODE_DIR"], "results.txt"), "w").write("R")
"""

FAKE_NVIDIA_SMI = r"""#!/bin/bash
case "$*" in
  *--query-gpu=index,name*) echo "0, NVIDIA H100 80GB HBM3, 81559, 570.86.15, GPU-abc" ;;
  *--query-gpu=timestamp*) echo "2026/09/27 12:00:00.000, 0, GPU-abc, 97, 1000, 81559, 300.50" ;;
  -L) echo "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-abc)" ;;
  *) echo "| NVIDIA-SMI 570.86.15   Driver Version: 570.86.15   CUDA Version: 12.8 |" ;;
esac
"""

FAKE_TORCH = """
import sys
__version__ = "{version}"
class version:
    cuda = {cuda!r}
class _cudnn:
    @staticmethod
    def version():
        return 90100
class backends:
    cudnn = _cudnn
class cuda:
    @staticmethod
    def is_available():
        if not {available}:
            sys.stderr.write("UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080)\\n")
        return {available}
    @staticmethod
    def get_arch_list():
        return {arch_list!r} if {available} else []
    @staticmethod
    def device_count():
        return 1
    @staticmethod
    def get_device_name(i):
        return "{name}"
    @staticmethod
    def get_device_capability(i):
        return {cap!r}
"""


def make_writable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o755)
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o644)


def publish_fake_snapshot(remote_root: Path, files: dict[str, str], *, complete: bool = True) -> Path:
    """Mimic publish_snapshot's result: files + marker, everything read-only."""
    snap = remote_root / "snapshots" / SNAPSHOT_HEX
    for rel, content in files.items():
        p = snap / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    if complete:
        (snap / ".hpclab-complete").write_text('{"digest":"x"}\n')
    for dirpath, _dirs, filenames in os.walk(snap, topdown=False):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), 0o444)
        os.chmod(dirpath, 0o555)
    return snap


def fake_torch_dir(
    base: Path,
    *,
    version: str = "2.8.0+cu130",
    cuda: str | None = "13.0",
    available: bool = False,
    arch_list: list[str] | None = None,
    name: str = "NVIDIA H100 80GB HBM3",
    cap: tuple[int, int] = (9, 0),
) -> Path:
    pkg = base / "torch"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        FAKE_TORCH.format(
            version=version,
            cuda=cuda,
            available=available,
            arch_list=arch_list or ["sm_80", "sm_90"],
            name=name,
            cap=list(cap),
        )
    )
    return base


def fake_bin_dir(base: Path, *, nvidia_smi: bool = True) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if nvidia_smi:
        tool = base / "nvidia-smi"
        tool.write_text(FAKE_NVIDIA_SMI)
        tool.chmod(0o755)
    return base


@dataclass
class Harness:
    tmp: Path
    remote_root: Path = field(init=False)
    layout: AttemptLayout = field(init=False)
    snapshot: Path | None = None
    base_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.remote_root = self.tmp / "remote" / "hpclab"
        self.layout = attempt_layout(str(self.remote_root), ATTEMPT_ID)
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "HPCLAB_", "PROG_", "PYTHON"))
        }
        env.update(
            {
                "SLURM_JOB_ID": "4242",
                "SLURM_JOB_NAME": "hl.test.k7m2p9q4",
                "SLURM_JOB_PARTITION": "gpu",
                "SLURM_CPUS_PER_TASK": "2",
                "HPCLAB_PROBE_TIMEOUT_S": "20",
                "HPCLAB_TELEMETRY_INTERVAL_S": "60",
            }
        )
        env.pop("MASTER_PORT", None)
        self.base_env = env

    # -- building ------------------------------------------------------------
    def publish(self, files: dict[str, str] | None = None, *, complete: bool = True) -> Path:
        self.snapshot = publish_fake_snapshot(
            self.remote_root, {"prog.py": PROG, **(files or {})}, complete=complete
        )
        return self.snapshot

    @staticmethod
    def cluster(**overlay: Any) -> ResolvedCluster:
        return ResolvedCluster(
            profile=ClusterProfile(
                cluster_id="sim",
                display_name="Sim",
                transport="local",
                default_partition="gpu",
                storage={"scratch": StorageRole(path="/scratch/{user}")},
            ),
            overlay=UserOverlay(cluster_id="sim", **overlay),
        )

    def specs(
        self,
        *,
        argv: list[str] | None = None,
        runtime: dict[str, Any] | None = None,
        outputs: list[OutputDecl] | None = None,
        inputs: list[InputRef] | None = None,
        resources: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        seed: int | None = None,
        workdir: str = ".",
        recovery_path: str | None = None,
        resilience: ResilienceSpec | None = None,
        requeue: bool | None = None,
    ) -> tuple[RunSpec, ExecutionPlan]:
        if self.snapshot is None:
            self.publish()
        rt = {"python": sys.executable, **(runtime or {})}
        run_spec = RunSpec(
            experiment="exp",
            code=CodeSnapshotRef(digest="sha256:" + SNAPSHOT_HEX, file_count=1, total_bytes=1),
            entrypoint=EntrypointSpec(argv=["python", "prog.py"], workdir=workdir),
            argv=argv if argv is not None else [sys.executable, "prog.py"],
            params=params or {},
            seed=seed,
            inputs=inputs or [],
            runtime=RuntimeSpec(**rt),
            outputs=outputs or [],
        )
        recovery = (
            RecoverySource(
                from_attempt_id="att_prev", strategy="resume", checkpoint_remote_path=recovery_path
            )
            if recovery_path
            else None
        )
        plan = ExecutionPlan(
            run_id=RUN_ID,
            run_spec_hash=run_spec.spec_hash(),
            cluster_id="sim",
            profile_revision="sha256:rev",
            resources=ResourceSpec(**(resources or {})),
            remote_root=str(self.remote_root),
            snapshot_path=str(self.snapshot),
            recovery=recovery,
            resilience=resilience,
            requeue=requeue,
        )
        return run_spec, plan

    def render(self, **kw: Any) -> str:
        run_spec, plan = self.specs(**kw)
        return render_job_script(
            run_spec=run_spec,
            plan=plan,
            cluster=self.cluster(account="lab"),
            run_id=RUN_ID,
            attempt_id=ATTEMPT_ID,
            token=TOKEN,
            job_name=job_name_for("test", TOKEN),
            layout=self.layout,
        )

    def write(self, **kw: Any) -> Path:
        script = self.render(**kw)
        for d in self.layout.dirs_to_create():
            os.makedirs(d, exist_ok=True)
        path = Path(self.layout.script)
        if path.exists():
            path.chmod(0o644)
        path.write_text(script)
        return path

    # -- running -------------------------------------------------------------
    def _env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = dict(self.base_env)
        merged.update(env or {})
        return merged

    def start(self, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        out = open(self.layout.stdout, "ab")  # noqa: SIM115 - handed to the child, closed by GC
        err = open(self.layout.stderr, "ab")  # noqa: SIM115
        return subprocess.Popen(
            [BASH, self.layout.script], cwd=self.layout.dir, env=self._env(env), stdout=out, stderr=err
        )

    def run(self, env: dict[str, str] | None = None, timeout: float = 60.0) -> int:
        proc = self.start(env)
        return proc.wait(timeout=timeout)

    def wait_for(self, path: str | Path, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while not os.path.exists(path):
            if time.monotonic() > deadline:
                raise AssertionError(f"{path} did not appear; stderr:\n{self.stderr()}")
            time.sleep(0.05)

    def signal_when_ready(self, proc: subprocess.Popen[bytes], sig: signal.Signals) -> int:
        self.wait_for(Path(self.layout.output_dir) / "ready")
        proc.send_signal(sig)
        return proc.wait(timeout=30)

    # -- reading -------------------------------------------------------------
    def json(self, path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    def status(self) -> dict[str, Any]:
        return self.json(self.layout.status_json)

    def env_json(self) -> dict[str, Any]:
        return self.json(self.layout.env_json)

    def outputs(self) -> dict[str, Any]:
        return self.json(self.layout.outputs_manifest)

    def dump(self) -> dict[str, Any]:
        return self.json(os.path.join(self.layout.output_dir, "dump.json"))

    def stderr(self) -> str:
        return (
            Path(self.layout.stderr).read_text(errors="replace") if os.path.exists(self.layout.stderr) else ""
        )
