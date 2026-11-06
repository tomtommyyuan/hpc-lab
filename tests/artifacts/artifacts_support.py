"""Shared helpers for the artifacts / evaluations / vault tests (imported by name).

``Lab`` is a ServiceContext on a ``transport: local`` cluster whose "remote" filesystem is
``tmp_path/remote``, with a switchable failing transport and a fake scheduler for sacct.
``Lab.manifest(att)`` writes ``outputs.json`` with the *real* compute-node helper embedded in
the job wrapper (``jobscript.HELPER_SOURCE``), so the index is tested against the exact
format the wrapper produces.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpclab.adapters.base import (
    AuthRequiredError,
    CancelResponse,
    ConnectionLost,
    QueryResult,
    RangeRead,
    SchedulerCapabilities,
    StatInfo,
)
from hpclab.adapters.local import LocalTransport
from hpclab.config import get_paths
from hpclab.domain.clock import FakeClock, iso
from hpclab.domain.models import JobObservation
from hpclab.domain.states import SchedulerState
from hpclab.profiles.schema import ClusterProfile, SitePolicy, UserOverlay
from hpclab.services.context import ServiceContext, Settings
from hpclab.services.jobscript import HELPER_SOURCE
from hpclab.testing.factories import make_attempt, make_experiment, make_plan, make_run

DEFAULT_OUTPUTS: list[dict[str, Any]] = [
    {
        "name": "metrics",
        "path": "{output_dir}/metrics.jsonl",
        "role": "metrics",
        "format": "hpclab.metrics/v1",
        "required": True,
    },
    {"name": "summary", "path": "{output_dir}/metrics.json", "role": "metrics", "required": True},
    {
        "name": "checkpoints",
        "path": "{checkpoint_dir}",
        "kind": "dir",
        "role": "checkpoint",
        "format": "hpclab.checkpoint/v1",
        "required": False,
    },
]

FINAL = {
    "schema": "hpclab.metrics/v1",
    "step": 60,
    "total_steps": 60,
    "split": "val",
    "final": True,
    "time": 1790000000.0,
    "dataset": "synthetic-v1",
    "seed": 1,
    "metrics": {"acc": 0.9, "loss": 0.3},
    "units": {"acc": "fraction", "loss": "nats"},
    "higher_is_better": {"acc": True, "loss": False},
}


def metrics_jsonl(*, final: bool = True, acc: float = 0.9, nan: bool = False) -> str:
    lines = [
        {
            "schema": "hpclab.metrics/v1",
            "step": 10,
            "total_steps": 60,
            "split": "train",
            "metrics": {"loss": 0.7},
        },
        {
            "schema": "hpclab.metrics/v1",
            "step": 30,
            "total_steps": 60,
            "split": "val",
            "metrics": {"acc": 0.8},
        },
    ]
    if final:
        rec = dict(FINAL, metrics={"acc": acc, "loss": 0.3})
        if nan:
            rec["metrics"] = {"acc": acc, "loss": float("nan")}  # a diverged value, as json.dumps writes it
        lines.append(rec)
    return "".join(json.dumps(x) + "\n" for x in lines)


def load_helper() -> types.ModuleType:
    module = types.ModuleType("hpclab_wrapper_helper_for_tests")
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), module.__dict__)
    return module


_HELPER = load_helper()


class FlakyTransport(LocalTransport):
    """LocalTransport whose file operations can be switched to fail like an SSH outage."""

    def __init__(self, cluster_id: str) -> None:
        super().__init__(cluster_id, user="me")
        self.down: Exception | None = None
        self.calls: list[str] = []

    def _gate(self, name: str) -> None:
        self.calls.append(name)
        if self.down is not None:
            raise self.down

    def stat(self, remote_paths: list[str], *, timeout: float = 30.0) -> dict[str, StatInfo]:
        self._gate("stat")
        return super().stat(remote_paths, timeout=timeout)

    def read_range(self, remote_path: str, offset: int, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._gate("read_range")
        return super().read_range(remote_path, offset, length, timeout=timeout)

    def read_tail(self, remote_path: str, length: int, *, timeout: float = 30.0) -> RangeRead:
        self._gate("read_tail")
        return super().read_tail(remote_path, length, timeout=timeout)

    def sha256(self, remote_paths: list[str], *, timeout: float = 120.0) -> dict[str, str | None]:
        self._gate("sha256")
        return super().sha256(remote_paths, timeout=timeout)

    def list_dir(self, remote_dir: str, *, max_entries: int = 1000, timeout: float = 30.0) -> list[StatInfo]:
        self._gate("list_dir")
        return super().list_dir(remote_dir, max_entries=max_entries, timeout=timeout)

    def download(self, remote_path: str, local_path: str, *, timeout: float = 600.0) -> None:
        self._gate("download")
        super().download(remote_path, local_path, timeout=timeout)


class FakeScheduler:
    """Only what the vault needs: ``query_accounting`` for one job."""

    cluster_id = "lab"

    def __init__(self) -> None:
        self.jobs: dict[str, JobObservation] = {}
        self.down: Exception | None = None

    def probe_capabilities(self) -> SchedulerCapabilities:
        return SchedulerCapabilities(sacct_available=True)

    def query_accounting(self, *, job_ids: list[str] | None = None, **_: Any) -> QueryResult:
        if self.down is not None:
            raise self.down
        now = iso(datetime.now(UTC)) or ""
        return QueryResult(
            source="sacct", observed_at=now, jobs=[self.jobs[j] for j in job_ids or [] if j in self.jobs]
        )

    def cancel(self, job_id: str) -> CancelResponse:  # pragma: no cover - never used here
        raise AssertionError("the artifacts area must never cancel jobs")


def accounting_record(job_id: str) -> JobObservation:
    return JobObservation(
        job_id=job_id,
        source="sacct",
        observed_at="2026-09-01T12:00:00.000Z",
        state=SchedulerState.COMPLETED,
        raw_state="COMPLETED",
        name="hl.run.abcdefgh",
        user="me",
        account="lab",
        partition="gpu",
        elapsed_s=42,
        exit_code=0,
        alloc_gpus=1,
    )


@dataclass
class Att:
    attempt_id: str
    run_id: str
    experiment_id: str
    plan_id: str
    dir: Path

    def path(self, rel: str) -> Path:
        return self.dir / rel

    def write(self, rel: str, content: str | bytes) -> Path:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
        return p


class Lab:
    def __init__(self, tmp_path: Path, *, max_login_hash_bytes: int = 256 * 1024 * 1024) -> None:
        self.tmp = tmp_path
        self.remote_root = tmp_path / "remote"
        self.paths = get_paths(tmp_path / "home").ensure()
        self.clock = FakeClock()
        self.transport = FlakyTransport("lab")
        self.scheduler = FakeScheduler()
        self.ctx = ServiceContext.create(
            self.paths,
            clock=self.clock,
            transport_factory=lambda cluster, paths: self.transport,
            scheduler_factory=lambda transport, cluster, clock: self.scheduler,  # type: ignore[arg-type,return-value]
            settings=Settings(request_deadline_s=5.0),
        )
        self.ctx.principal = "tester@host"
        profile = ClusterProfile(
            cluster_id="lab",
            display_name="Lab",
            transport="local",
            policy=SitePolicy(max_login_hash_bytes=max_login_hash_bytes),
        )
        self.ctx.clusters.register(
            profile, UserOverlay(cluster_id="lab", user="me", remote_root=str(self.remote_root))
        )

    def close(self) -> None:
        self.ctx.db.close()

    # -- ledger ---------------------------------------------------------------
    def experiment(self, *, spec: dict[str, Any] | None = None) -> tuple[str, str]:
        with self.ctx.db.tx() as tx:
            exp = make_experiment(tx, cluster_id="lab", spec=spec)
            plan = make_plan(tx, exp)
        return exp, plan

    def attempt(
        self,
        *,
        outputs: list[dict[str, Any]] | None = None,
        phase: str = "terminal",
        scheduler_state: str = "COMPLETED",
        result_state: str = "unchecked",
        evaluation: dict[str, Any] | None = None,
        experiment: tuple[str, str] | None = None,
        seed: int = 1,
        job_id: str | None = "1001",
        run_id: str | None = None,
        spec_outputs: bool = True,
    ) -> Att:
        exp, plan = experiment or self.experiment()
        with self.ctx.db.tx() as tx:
            if run_id is None:
                spec: dict[str, Any] = {
                    "experiment": "exp",
                    "kind": "main",
                    "cell": {"seed": seed},
                    "seed": seed,
                    "argv": ["python", "train.py"],
                    "code": {
                        "digest": "sha256:" + "c" * 64,
                        "file_count": 1,
                        "total_bytes": 1,
                        "git_commit": "abc123",
                    },
                }
                if spec_outputs:
                    spec["outputs"] = DEFAULT_OUTPUTS if outputs is None else outputs
                if evaluation is not None:
                    spec["evaluation"] = evaluation
                run_id = make_run(tx, exp, plan, seed=seed, spec=spec, status="running")
            aid = make_attempt(
                tx,
                run_id,
                cluster_id="lab",
                phase=phase,
                scheduler_state=scheduler_state,
                result_state=result_state,
                job_id=job_id,
            )
            d = self.remote_root / "attempts" / aid
            terminal_at = iso(self.clock.now()) if phase == "terminal" else None
            tx.execute(
                "UPDATE attempts SET remote_dir=?, terminal_at=? WHERE attempt_id=?",
                (str(d), terminal_at, aid),
            )
        for sub in ("outputs", "checkpoints", "work", "logs"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return Att(aid, run_id, exp, plan, d)

    def ready(self, **kw: Any) -> Att:
        """A terminal COMPLETED attempt with the default good outputs, indexed."""
        from hpclab.services.artifacts import index_attempt

        att = self.attempt(**kw)
        self.good_outputs(att)
        self.manifest(att)
        index_attempt(self.ctx, attempt_id=att.attempt_id)
        return att

    def set_attempt(self, att: Att, **values: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in values)
        with self.ctx.db.tx() as tx:
            tx.execute(f"UPDATE attempts SET {cols} WHERE attempt_id=?", (*values.values(), att.attempt_id))

    def row(self, table: str, key: str, value: str) -> dict[str, Any]:
        r = self.ctx.db.one(f"SELECT * FROM {table} WHERE {key}=?", (value,))
        assert r is not None, (table, value)
        return r

    def artifact(self, att: Att, name: str) -> dict[str, Any]:
        r = self.ctx.db.one("SELECT * FROM artifacts WHERE attempt_id=? AND name=?", (att.attempt_id, name))
        assert r is not None, name
        return r

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return self.ctx.db.all("SELECT * FROM events WHERE event_type=? ORDER BY seq", (event_type,))

    # -- wrapper-side files -----------------------------------------------------
    def good_outputs(self, att: Att, *, checkpoint: bool = True) -> None:
        att.write("outputs/metrics.jsonl", metrics_jsonl())
        att.write("outputs/metrics.json", json.dumps(dict(FINAL, split="test")))
        if checkpoint:
            att.write("checkpoints/ckpt-000000020/model.bin", b"weights-20")
            att.write(
                "checkpoints/ckpt-000000020/hpclab-ckpt.json",
                '{"schema": "hpclab.checkpoint/v1", "step": 20}',
            )
            att.write("checkpoints/ckpt-000000060/model.bin", b"weights-60")
            att.write(
                "checkpoints/ckpt-000000060/hpclab-ckpt.json",
                '{"schema": "hpclab.checkpoint/v1", "step": 60}',
            )

    def manifest(
        self,
        att: Att,
        *,
        outputs: list[dict[str, Any]] | None = None,
        mode: str = "full",
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the wrapper's own ``outputs`` helper (as on the compute node)."""
        decls = DEFAULT_OUTPUTS if outputs is None else outputs
        static = {
            "{output_dir}": str(att.dir / "outputs"),
            "{checkpoint_dir}": str(att.dir / "checkpoints"),
            "{attempt_dir}": str(att.dir),
        }
        cfg_outputs = []
        for o in decls:
            path = o["path"]
            for k, v in static.items():
                path = path.replace(k, v)
            if not path.startswith("/"):
                path = f"{att.dir / 'work'}/{path}"
            cfg_outputs.append(
                {
                    "name": o["name"],
                    "path": path,
                    "kind": o.get("kind", "file"),
                    "role": o.get("role", "other"),
                    "required": o.get("required", True),
                }
            )
        cfg = {
            "attempt_id": attempt_id or att.attempt_id,
            "code_dir": str(att.dir / "work"),
            "outputs": cfg_outputs,
        }
        cfg_path = self.tmp / f"cfg-{att.attempt_id}.json"
        cfg_path.write_text(json.dumps(cfg))
        out = att.dir / "outputs.json"
        assert _HELPER.cmd_outputs([str(out), str(cfg_path), mode]) == 0
        doc: dict[str, Any] = json.loads(out.read_text())
        return doc

    def edit_manifest(self, att: Att, fn: Any) -> None:
        p = att.dir / "outputs.json"
        doc = json.loads(p.read_text())
        fn(doc)
        p.write_text(json.dumps(doc))


def keep_mtime(path: Path, content: bytes) -> None:
    """Rewrite a file with same-size content and restore its mtime (silent tampering)."""
    st = path.stat()
    path.write_bytes(content)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def auth_error() -> AuthRequiredError:
    return AuthRequiredError("simulated: MFA expired", details={"login_command": ["ssh", "-M", "lab"]})


def conn_lost() -> ConnectionLost:
    return ConnectionLost("simulated: connection dropped")
