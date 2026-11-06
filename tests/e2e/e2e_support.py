"""Harness for the end-to-end suite: a real controller *subprocess* (autostarted by
``HpcLabClient`` exactly as for a user), the simulator as external ground truth, and the
tiny-classifier example as the research project.

    lab = Lab.start(tmp_path)                  # fresh HPCLAB_HOME, sim registered via init_sim
    data = lab.call("experiment_prepare", ...)   # Envelope must be ok -> returns data
    err = lab.error("run_submit", ...)           # Envelope must be an error -> returns error
    lab.sim.jobs()                               # what sbatch really accepted
    lab.close()                                  # controller stopped, sim jobs killed, dirs writable

Speed: the controller polls the cluster every ~5 s (``observe.poll_all``); a local profile
override (``$HPCLAB_HOME/profiles/sim.yaml``, the documented way to adapt a site profile) only
shortens the reconcile window and accounting lag so fault scenarios settle within a test.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hpclab.bundled import bundled_path
from hpclab.config import Paths, get_paths
from hpclab.controller import client as client_mod
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.controller.files import lock_is_held, pid_alive, read_pid, socket_path
from hpclab.domain.ids import new_id
from hpclab.sim.cluster import SimCluster
from hpclab.sim.config import SimConfig

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "tiny-classifier"
CRASH_CONTROLLER = REPO / "benchmarks" / "crash_controller.py"

#: Simulator timings that keep job lifecycles short (same as hpclab.sim.testing.FAST_TIMINGS).
FAST_SIM: dict[str, Any] = {
    "min_job_age_s": 1.0,
    "kill_wait_s": 0.3,
    "poll_interval_s": 0.05,
    "fate_delay_s": 0.3,
}
#: Site-policy overrides for the local sim profile (see module docstring).
FAST_POLICY: dict[str, Any] = {"reconcile_window_s": 12, "accounting_lag_s": 1}

TERMINAL_WORKFLOW = ("completed", "cancelled", "gate_failed")
UNRESOLVED_PHASES = ("submission_in_flight", "submission_uncertain", "duplicate_detected")


def write_profile_override(paths: Paths, **policy: Any) -> Path:
    """Copy the bundled sim profile into the home with ``policy`` overrides."""
    root = bundled_path("profiles")
    assert root is not None, "bundled profiles not found"
    data = yaml.safe_load((root / "sim.yaml").read_text(encoding="utf-8"))
    data.setdefault("policy", {}).update(policy)
    out = paths.profiles / "sim.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def example_spec(project: Path, **changes: Any) -> dict[str, Any]:
    """The example's ExperimentSpec with ``code.project_root`` made absolute; ``changes`` replace
    top-level keys (``None`` deletes the key)."""
    spec: dict[str, Any] = yaml.safe_load((project / "hpclab.yaml").read_text(encoding="utf-8"))
    spec["code"] = {**(spec.get("code") or {}), "project_root": str(project)}
    for key, value in changes.items():
        if value is None:
            spec.pop(key, None)
        else:
            spec[key] = copy.deepcopy(value)
    return spec


def argv_with(spec: dict[str, Any], *extra: str) -> list[str]:
    return [*spec["entrypoint"]["argv"], *extra]


def run_example_locally(workdir: Path, *args: str) -> dict[str, Any]:
    """Run train.py uninterrupted outside hpclab; the final val record of metrics.jsonl."""
    out = workdir / "outputs"
    workdir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HPCLAB_", "SLURM_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE / "train.py"), *args, "--out", str(out)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return final_val_record(out / "metrics.jsonl")


def final_val_record(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    finals = [r for r in records if r.get("final") and r.get("split") == "val"]
    assert finals, f"no final val record in {path}"
    return finals[-1]


def make_writable(root: Path) -> None:
    """chmod u+w everything below ``root`` (published snapshots are read-only) so it can be deleted."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            p = Path(dirpath) / name
            with contextlib.suppress(OSError):
                st = p.lstat()
                if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                    p.chmod(st.st_mode | stat.S_IWUSR)
    with contextlib.suppress(OSError):
        root.chmod(root.stat().st_mode | stat.S_IWUSR)


@dataclass
class Lab:
    root: Path
    paths: Paths
    client: HpcLabClient
    project: Path
    extra_clients: list[HpcLabClient] = field(default_factory=list)
    _keys: Counter[str] = field(default_factory=Counter)

    # -- lifecycle ------------------------------------------------------------------------
    @classmethod
    def start(
        cls,
        root: Path,
        *,
        sim: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        controller_cmd: Callable[[Paths], list[str]] | None = None,
    ) -> Lab:
        """``controller_cmd(paths)`` replaces the argv the client autostarts (crash hooks)."""
        root.mkdir(parents=True, exist_ok=True)
        paths = get_paths(root / "home").ensure()
        write_profile_override(paths, **{**FAST_POLICY, **(policy or {})})
        SimCluster.ensure(paths.sim_root("sim"), SimConfig().replace(**{**FAST_SIM, **(sim or {})}))
        project = root / "project"
        shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
        client = HpcLabClient(paths, timeout=60, spawn_cmd=controller_cmd(paths) if controller_cmd else None)
        lab = cls(root=root, paths=paths, client=client, project=project)
        data = lab.call("init_sim")
        assert data["cluster"]["cluster_id"] == "sim", data
        return lab

    def close(self) -> None:
        for c in [self.client, *self.extra_clients]:
            with contextlib.suppress(Exception):
                c.close()
        if lock_is_held(self.paths.lock):
            with contextlib.suppress(Exception):
                stop_controller(self.paths, timeout=15.0, force=True)
        pid = read_pid(self.paths.pidfile)
        if pid and pid != os.getpid() and pid_alive(pid) and lock_is_held(self.paths.lock):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            self.sim.shutdown(grace_s=2.0)
        with client_mod._spawned_lock:  # reap controllers this process spawned (no zombies)
            procs = list(client_mod._spawned)
        for proc in procs:
            with contextlib.suppress(Exception):
                proc.poll()
        sock_dir = socket_path(self.paths).parent
        if sock_dir != self.paths.home / "run":
            shutil.rmtree(sock_dir, ignore_errors=True)
        make_writable(self.root)

    # -- calls ----------------------------------------------------------------------------
    def key(self, prefix: str) -> str:
        self._keys[prefix] += 1
        return f"{prefix}-{self._keys[prefix]}-{new_id('request')[-6:]}"

    def env(self, method: str, **params: Any) -> dict[str, Any]:
        clean = {k: v for k, v in params.items() if v is not None}
        return self.client.call(method, clean)

    def ok(self, method: str, **params: Any) -> dict[str, Any]:
        """The whole Envelope (coverage, warnings, summary are lifted out of ``data``); must be ok."""
        env = self.env(method, **params)
        assert env["ok"], f"{method} failed: {json.dumps(env.get('error'), default=str)[:3000]}"
        return env

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        env = self.ok(method, **params)
        data = env["data"]
        assert isinstance(data, dict)
        return data

    def error(self, method: str, **params: Any) -> dict[str, Any]:
        env = self.env(method, **params)
        assert not env["ok"], f"{method} unexpectedly succeeded: {env.get('summary')}"
        err = env["error"]
        assert isinstance(err, dict)
        return err

    # -- ground truth + ledger --------------------------------------------------------------
    @property
    def sim(self) -> SimCluster:
        return SimCluster(self.paths.sim_root("sim"))

    def ledger(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Read-only query of the ledger (the controller is the only writer)."""
        conn = sqlite3.connect(f"file:{self.paths.db}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def attempts(self, where: str = "1=1", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return self.ledger(
            "SELECT a.*, r.label, r.kind FROM attempts a JOIN runs r ON r.run_id=a.run_id "
            f"WHERE {where} ORDER BY a.created_at",
            params,
        )

    def jobs_by_name(self) -> Counter[str]:
        return Counter(str(j["name"]) for j in self.sim.jobs())

    def duplicate_jobs(self) -> dict[str, int]:
        """Job names (one per attempt) that Slurm accepted more than once."""
        return {name: n for name, n in self.jobs_by_name().items() if n > 1}

    def use_plain_controller(self) -> int:
        """Restart with the stock controller (``python -m hpclab.controller``); its pid."""
        self.extra_clients.append(self.client)
        self.client = HpcLabClient(self.paths, timeout=60)
        return self.controller_pid()

    def controller_pid(self) -> int:
        info = self.client.ensure_running()
        pid = info.get("pid")
        assert isinstance(pid, int)
        return pid

    def kill_controller(self) -> int:
        """SIGKILL the running controller (kill -9) and wait until its lock is free."""
        pid = self.controller_pid()
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while lock_is_held(self.paths.lock) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not lock_is_held(self.paths.lock), "killed controller still holds its lock"
        with client_mod._spawned_lock:
            procs = [p for p in client_mod._spawned if p.pid == pid]
        for p in procs:
            with contextlib.suppress(Exception):
                p.wait(timeout=5)
        return pid

    # -- experiment helpers -----------------------------------------------------------------
    def spec(self, **changes: Any) -> dict[str, Any]:
        return example_spec(self.project, **changes)

    def prepare(self, spec: dict[str, Any], **params: Any) -> dict[str, Any]:
        data = self.call("experiment_prepare", idempotency_key=self.key("prep"), spec=spec, **params)
        assert data["status"] == "ready", json.dumps(data["validation"], default=str)[:3000]
        return data

    def wait(
        self,
        pred: Callable[[], Any],
        *,
        timeout: float,
        what: str,
        interval: float = 0.5,
        dump: Callable[[], Any] | None = None,
    ) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            value = pred()
            if value:
                return value
            if time.monotonic() >= deadline:
                context = dump() if dump else self.snapshot()
                raise AssertionError(
                    f"timed out after {timeout:.0f}s waiting for {what}\n"
                    + json.dumps(context, indent=1, default=str)[:6000]
                )
            time.sleep(interval)

    def snapshot(self) -> dict[str, Any]:
        """Diagnostic dump for timeouts: attempts, workflows, open operations, sim jobs."""
        return {
            "attempts": [
                {
                    k: a[k]
                    for k in ("attempt_id", "label", "phase", "scheduler_state", "result_state", "job_id")
                }
                for a in self.attempts()
            ],
            "workflows": self.ledger("SELECT workflow_id, state, dispatch_state, version FROM workflows"),
            "operations": self.ledger(
                "SELECT kind, state, stage, error_json FROM operations WHERE state NOT IN ('succeeded')"
            ),
            "sim_jobs": [
                {k: j.get(k) for k in ("job_id", "name", "state", "exit_code")} for j in self.sim.jobs()
            ],
        }

    def workflow(self, workflow_id: str) -> dict[str, Any]:
        return self.call("workflow_get", workflow_id=workflow_id, limit=100)

    def wait_workflow(
        self, workflow_id: str, *, states: tuple[str, ...] = ("completed",), timeout: float = 60
    ) -> dict[str, Any]:
        def done() -> dict[str, Any] | None:
            w = self.workflow(workflow_id)
            if w["state"] in states:
                return w
            if w["state"] in TERMINAL_WORKFLOW and w["state"] not in states:
                raise AssertionError(f"workflow ended {w['state']}, expected {states}: {w}")
            return None

        return self.wait(done, timeout=timeout, what=f"workflow {workflow_id} in {states}")

    def wait_attempt(
        self, attempt_id: str, pred: Callable[[dict[str, Any]], bool], *, timeout: float, what: str
    ) -> dict[str, Any]:
        def check() -> dict[str, Any] | None:
            rows = self.attempts("a.attempt_id=?", (attempt_id,))
            return rows[0] if rows and pred(rows[0]) else None

        return self.wait(check, timeout=timeout, what=f"attempt {attempt_id}: {what}")

    def wait_settled(self, attempt_id: str, *, timeout: float = 45) -> dict[str, Any]:
        """Terminal with its outputs checked (result_state no longer pending/unchecked)."""
        return self.wait_attempt(
            attempt_id,
            lambda a: a["phase"] == "terminal" and a["result_state"] not in ("unchecked", "pending"),
            timeout=timeout,
            what="terminal and indexed",
        )

    def grid(self, **params: Any) -> dict[str, Any]:
        data = self.call("runs_list", view="grid", **params)
        grid = data["grid"]
        assert isinstance(grid, dict)
        return grid

    def wait_all_valid(self, *, timeout: float = 30, **scope: Any) -> dict[str, dict[str, str]]:
        """Every grid cell V (valid result AND its evaluation collected; right after a workflow
        completes a cell can briefly show C! until the automatic evaluation lands)."""

        def check() -> dict[str, dict[str, str]] | None:
            codes = grid_codes(self.grid(**scope))
            return codes if all(c == "V" for row in codes.values() for c in row.values()) else None

        return self.wait(check, timeout=timeout, what=f"all cells valid in {scope}", interval=0.5)

    def evaluated(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.call("run_inspect", run_id=run_id)["evaluations"] if e["status"] == "valid"]


def crash_controller_cmd(point: str, marker: Path, *, nth: int = 1) -> Callable[[Paths], list[str]]:
    """argv factory for ``Lab.start(controller_cmd=...)``: a controller that SIGKILLs itself at
    ``point`` of submit.dispatch (see benchmarks/crash_controller.py)."""

    def cmd(paths: Paths) -> list[str]:
        return [
            sys.executable,
            str(CRASH_CONTROLLER),
            "--home",
            str(paths.home),
            "--crash-at",
            point,
            "--nth",
            str(nth),
            "--marker",
            str(marker),
        ]

    return cmd


def grid_codes(grid: dict[str, Any]) -> dict[str, dict[str, str]]:
    """``{row key: {col: code}}`` of a runs_list grid."""
    return {
        row["key"]: dict(zip(grid["cols"], [c["code"] for c in row["cells"]], strict=True))
        for row in grid["rows"]
    }


@contextlib.contextmanager
def lab_session(root: Path, **kw: Any) -> Iterator[Lab]:
    lab = Lab.start(root, **kw)
    try:
        yield lab
    finally:
        lab.close()
