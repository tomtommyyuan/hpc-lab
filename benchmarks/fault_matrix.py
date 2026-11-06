#!/usr/bin/env python3
"""Fault-injection matrix: how hpclab's submission protocol behaves under injected faults.

    python benchmarks/fault_matrix.py --out benchmarks/results            # full: 5 reps per scenario
    python benchmarks/fault_matrix.py --out benchmarks/results --quick    # 1 rep each, < 3 min
    python benchmarks/fault_matrix.py --only lost_reply,crash_after_inflight --reps 3

Every repetition is independent: a fresh hpclab home, a real controller *subprocess* (autostarted
by ``HpcLabClient`` as for a user, or ``crash_controller.py`` for crash points), the real services,
the real ``SlurmScheduler`` against the Slurm simulator, and the tiny-classifier example. The
simulator is the external ground truth: what sbatch accepted, what scancel reached, what each job
did. For each scenario x repetition the script records submissions attempted (sbatch calls hpclab
made), jobs Slurm accepted, duplicates, wrong-job cancels, attempts left unresolved (and whether
hpclab flagged them for a human), wrong final states and the time until the outcome was decided.

Output: ``<out>/fault_matrix[_quick].json`` (every record, the environment, the settings) and
``<out>/fault_matrix[_quick].md`` (summary with full denominators, every failing repetition);
runs with ``--only`` or ``--reps`` write ``fault_matrix_custom.*`` instead. Ctrl-C stops every
repetition, cleans up its controller and simulated jobs, and writes no report (exit 130).
See benchmarks/README.md for the methodology and its limits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import random
import shutil
import signal
import sqlite3
import stat
import statistics
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import hpclab
from hpclab.bundled import bundled_path
from hpclab.config import Paths, get_paths
from hpclab.controller import client as client_mod
from hpclab.controller.client import HpcLabClient, stop_controller
from hpclab.controller.files import lock_is_held, socket_path
from hpclab.domain.ids import new_id
from hpclab.sim.cluster import SimCluster
from hpclab.sim.config import SimConfig

HERE = Path(__file__).resolve().parent
CRASH_CONTROLLER = HERE / "crash_controller.py"
UNRESOLVED = ("submission_in_flight", "submission_uncertain", "duplicate_detected")
FAST_SIM: dict[str, Any] = {
    "min_job_age_s": 1.0,
    "kill_wait_s": 0.3,
    "poll_interval_s": 0.05,
    "fate_delay_s": 0.3,
}
#: The only site-policy change vs the bundled sim profile: the reconcile window bounds how long an
#: uncertain submission is searched before it is handed to a human (sim default 120 s, HAI 900 s).
POLICY: dict[str, Any] = {"reconcile_window_s": 20}
STALE_OPS = ("unresolved", "duplicate_detected")


# ---------------------------------------------------------------------------
# One isolated environment per repetition
# ---------------------------------------------------------------------------


class Env:
    """hpclab home + simulator + controller subprocess for one repetition."""

    def __init__(self, root: Path, *, crash: tuple[str, int] | None = None) -> None:
        self.root = root
        self.paths: Paths = get_paths(root / "home").ensure()
        src = bundled_path("profiles")
        assert src is not None
        profile = yaml.safe_load((src / "sim.yaml").read_text(encoding="utf-8"))
        profile.setdefault("policy", {}).update(POLICY)
        (self.paths.profiles / "sim.yaml").write_text(
            yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
        )
        self.sim = SimCluster.ensure(self.paths.sim_root("sim"), SimConfig().replace(**FAST_SIM))
        examples = bundled_path("examples")
        assert examples is not None
        self.project = root / "project"
        shutil.copytree(
            examples / "tiny-classifier", self.project, ignore=shutil.ignore_patterns("__pycache__")
        )
        self.marker = root / "crashed.txt"
        spawn = None
        if crash is not None:
            point, nth = crash
            spawn = [
                sys.executable,
                str(CRASH_CONTROLLER),
                "--home",
                str(self.paths.home),
                "--crash-at",
                point,
                "--nth",
                str(nth),
                "--marker",
                str(self.marker),
            ]
        self.clients = [HpcLabClient(self.paths, timeout=60, spawn_cmd=spawn)]
        self.call("init_sim")

    @property
    def client(self) -> HpcLabClient:
        return self.clients[-1]

    def restart_plain(self) -> None:
        """Restart with the stock controller (after a crash)."""
        self.clients.append(HpcLabClient(self.paths, timeout=60))
        self.client.ensure_running()

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        env = self.client.call(method, {k: v for k, v in params.items() if v is not None})
        if not env.get("ok"):
            raise RuntimeError(f"{method}: {json.dumps(env.get('error'), default=str)[:1500]}")
        data = env["data"]
        assert isinstance(data, dict)
        return data

    def ledger(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        conn = sqlite3.connect(f"file:{self.paths.db}?mode=ro", uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def attempt(self, aid: str) -> dict[str, Any]:
        rows = self.ledger("SELECT * FROM attempts WHERE attempt_id=?", (aid,))
        return rows[0] if rows else {}

    def events(self, aid: str) -> list[str]:
        return [
            r["event_type"]
            for r in self.ledger("SELECT event_type FROM events WHERE entity_id=? ORDER BY seq", (aid,))
        ]

    def op(self, op_id: str | None) -> dict[str, Any]:
        rows = self.ledger("SELECT * FROM operations WHERE operation_id=?", (op_id,)) if op_id else []
        return rows[0] if rows else {}

    def spec(self, *, extra_argv: tuple[str, ...] = (), **changes: Any) -> dict[str, Any]:
        spec: dict[str, Any] = yaml.safe_load((self.project / "hpclab.yaml").read_text(encoding="utf-8"))
        spec["code"] = {"project_root": str(self.project)}
        spec["smoke"] = None
        spec["matrix"] = {"method": ["baseline"]}
        spec["seeds"] = [1]
        spec["entrypoint"]["argv"] = [*spec["entrypoint"]["argv"], *extra_argv]
        spec.update(changes)
        return {k: v for k, v in spec.items() if v is not None}

    def prepare(self, spec: dict[str, Any]) -> dict[str, Any]:
        data = self.call("experiment_prepare", idempotency_key=new_id("request"), spec=spec)
        if data["status"] != "ready":
            raise RuntimeError(f"plan not ready: {data['validation']}")
        return data

    def submit_single(self, *, extra_argv: tuple[str, ...] = ()) -> tuple[str, float]:
        prep = self.prepare(self.spec(extra_argv=extra_argv))
        run = prep["runs"][0]
        t0 = time.monotonic()
        rec = self.call(
            "run_submit",
            run_id=run["run_id"],
            execution_plan_hash=run["exec_plan_hash"],
            idempotency_key=new_id("request"),
        )
        return str(rec["attempt_id"]), t0

    def kill_controller(self) -> int:
        pid = int(self.client.ensure_running()["pid"])
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while lock_is_held(self.paths.lock) and time.monotonic() < deadline:
            time.sleep(0.05)
        return pid

    def close(self) -> None:
        for c in self.clients:
            with contextlib.suppress(Exception):
                c.close()
        with contextlib.suppress(Exception):
            stop_controller(self.paths, timeout=15.0, force=True)
        with contextlib.suppress(Exception):
            self.sim.shutdown(grace_s=2.0)
        with client_mod._spawned_lock:
            procs = list(client_mod._spawned)
        for p in procs:
            if p.poll() is not None:
                with contextlib.suppress(Exception):
                    p.wait(timeout=1)
        sock_dir = socket_path(self.paths).parent
        if sock_dir != self.paths.home / "run":
            shutil.rmtree(sock_dir, ignore_errors=True)


#: Set on Ctrl-C: every repetition stops at its next wait/sleep and cleans up (``Env.close``).
STOP = threading.Event()


class Interrupted(Exception):
    """The run was interrupted (Ctrl-C); raised in the repetitions' threads."""


def sleep(seconds: float) -> None:
    """``time.sleep`` that ends early (raising ``Interrupted``) when the run is interrupted."""
    if STOP.wait(max(0.0, seconds)):
        raise Interrupted("interrupted (Ctrl-C)")


def wait_for(pred: Callable[[], Any], *, timeout: float, what: str, interval: float = 0.2) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        if STOP.is_set():
            raise Interrupted("interrupted (Ctrl-C)")
        value = pred()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")
        sleep(interval)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Rec:
    scenario: str
    rep: int
    passed: bool = False
    outcome: str = ""
    expected: str = ""
    dispatches: int = 0  # sbatch calls hpclab made (attempt.dispatching events)
    attempts: int = 0
    jobs_accepted: int = 0  # ground truth: jobs the simulator's sbatch accepted
    external_jobs: int = 0  # jobs the scenario itself submitted (not hpclab)
    duplicates: int = 0  # jobs beyond one per attempt, not counting external ones
    cancels: int = 0  # scancel calls that reached Slurm
    wrong_cancels: int = 0  # ... on a job the scenario did not ask hpclab to cancel
    unresolved: int = 0  # attempts left in_flight / uncertain / duplicate_detected
    unresolved_flagged: int = 0  # ... whose operation asks a human, with a remediation
    wrong_states: int = 0  # bound attempts whose final scheduler state differs from Slurm's
    decided_s: float | None = None  # fault/crash -> the outcome was decided (bound, flagged, refused)
    settled_s: float | None = None  # fault/crash -> terminal and validated (success paths)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    wall_s: float = 0.0


def measure(env: Env, rec: Rec, *, expected_cancel_jobs: set[str] | None = None) -> None:
    """Ground-truth counters shared by every scenario."""
    attempts = env.ledger("SELECT * FROM attempts")
    jobs = env.sim.jobs()
    rec.attempts = len(attempts)
    rec.dispatches = len(env.ledger("SELECT 1 FROM events WHERE event_type='attempt.dispatching'"))
    rec.jobs_accepted = len(jobs)
    names = Counter(str(j["name"]) for j in jobs)
    dup = sum(max(0, names.get(a["job_name"], 0) - 1) for a in attempts)
    rec.duplicates = max(0, dup - rec.external_jobs)
    cancelled = [str(j.job_id) for j in env.sim.state.jobs() if j.cancel_requested]
    rec.cancels = len(cancelled)
    allowed = expected_cancel_jobs or set()
    rec.wrong_cancels = sum(1 for job_id in cancelled if job_id not in allowed)
    by_id = {str(j["job_id"]): j for j in jobs}
    for a in attempts:
        if a["phase"] in UNRESOLVED:
            rec.unresolved += 1
            op = env.op(a["operation_id"])
            err = json.loads(op.get("error_json") or "null") or {}
            if op.get("state") == "blocked" and (err.get("details") or {}).get("remediation"):
                rec.unresolved_flagged += 1
        if a["phase"] == "terminal" and a["job_id"] in by_id:
            truth = by_id[a["job_id"]]["state"]
            if a["scheduler_state"] != truth:
                rec.wrong_states += 1
                rec.notes.append(f"{a['attempt_id']}: ledger {a['scheduler_state']} vs Slurm {truth}")


def settled(env: Env, aid: str) -> bool:
    a = env.attempt(aid)
    return a.get("phase") == "terminal" and a.get("result_state") not in ("unchecked", "pending")


def flagged(env: Env, aid: str) -> bool:
    a = env.attempt(aid)
    if a.get("phase") not in UNRESOLVED:
        return False
    op = env.op(a.get("operation_id"))
    return op.get("state") == "blocked" and op.get("stage") in STALE_OPS


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _success_path(env: Env, rec: Rec, aid: str, t0: float, *, timeout: float = 90) -> None:
    """Wait for the attempt to be bound (decided) and settled; record times."""
    wait_for(lambda: env.attempt(aid).get("job_id") or flagged(env, aid), timeout=timeout, what="decision")
    rec.decided_s = round(time.monotonic() - t0, 2)
    wait_for(lambda: settled(env, aid) or flagged(env, aid), timeout=timeout, what="settled")
    rec.settled_s = round(time.monotonic() - t0, 2)


def _expect_single_success(env: Env, rec: Rec, aid: str) -> None:
    a = env.attempt(aid)
    rec.outcome = f"{a.get('phase')}/{a.get('scheduler_state')}/{a.get('result_state')} job {a.get('job_id')}"
    rec.passed = (
        rec.jobs_accepted == 1
        and rec.duplicates == 0
        and rec.wrong_cancels == 0
        and rec.wrong_states == 0
        and a.get("scheduler_state") == "COMPLETED"
        and a.get("result_state") == "valid"
        and rec.dispatches <= 1
    )


def sc_lost_reply(env: Env, rec: Rec) -> None:
    rec.expected = "1 job; reconciled (receipt/squeue/sacct) and bound; completes; no resubmission"
    env.sim.set_faults(sbatch_lose_reply=1)
    aid, t0 = env.submit_single()
    _success_path(env, rec, aid, t0)
    measure(env, rec)
    _expect_single_success(env, rec, aid)
    rec.passed = rec.passed and "attempt.uncertain" in env.events(aid)


def sc_ambiguous_queued(env: Env, rec: Rec) -> None:
    rec.expected = "sbatch errors ambiguously but queued the job: 1 job, bound, completes"
    env.sim.set_faults(sbatch_ambiguous_error=1, sbatch_ambiguous_creates_job=True)
    aid, t0 = env.submit_single()
    _success_path(env, rec, aid, t0)
    measure(env, rec)
    _expect_single_success(env, rec, aid)


def sc_ambiguous_not_queued(env: Env, rec: Rec) -> None:
    rec.expected = (
        "sbatch errors ambiguously and queued nothing: 0 jobs, never resubmitted, flagged for a human"
    )
    env.sim.set_faults(sbatch_ambiguous_error=1, sbatch_ambiguous_creates_job=False)
    aid, t0 = env.submit_single()
    wait_for(
        lambda: flagged(env, aid) or env.attempt(aid).get("phase") not in (*UNRESOLVED, "staging"),
        timeout=90,
        what="decision",
    )
    rec.decided_s = round(time.monotonic() - t0, 2)
    measure(env, rec)
    a = env.attempt(aid)
    rec.outcome = f"{a.get('phase')} (op {env.op(a.get('operation_id')).get('stage')})"
    rec.passed = (
        rec.jobs_accepted == 0 and rec.dispatches == 1 and rec.unresolved == rec.unresolved_flagged == 1
    )


_CRASH_EXPECTED = {
    "before_inflight": "nothing was sent yet; the restarted controller re-runs the step and dispatches once: 1 job, completes",
    "after_inflight": "hpclab cannot know whether sbatch ran: it reconciles, finds no job and flags the attempt "
    "for a human -- never resubmits: 0 jobs",
    "after_sbatch": "Slurm has the job but hpclab never read the reply: uncertain -> reconciled and bound to that one job",
    "before_identity": "the transaction recording the job id rolled back: uncertain -> reconciled and bound to the one job",
}


def _crash(point: str) -> Callable[[Env, Rec], None]:
    def run(env: Env, rec: Rec) -> None:
        rec.expected = _CRASH_EXPECTED[point]
        aid, _t0 = env.submit_single()
        wait_for(env.marker.exists, timeout=60, what=f"crash at {point}", interval=0.05)
        t_crash = time.monotonic()
        wait_for(lambda: not lock_is_held(env.paths.lock), timeout=10, what="dead controller's lock")
        rec.notes.append(f"phase at crash: {env.attempt(aid).get('phase')}")
        env.restart_plain()
        if point == "after_inflight":
            wait_for(lambda: flagged(env, aid), timeout=90, what="flagged unresolved")
            rec.decided_s = round(time.monotonic() - t_crash, 2)
            measure(env, rec)
            a = env.attempt(aid)
            rec.outcome = f"{a.get('phase')} (op {env.op(a.get('operation_id')).get('stage')})"
            rec.passed = (
                rec.jobs_accepted == 0
                and rec.dispatches == 1
                and rec.unresolved == rec.unresolved_flagged == 1
            )
            return
        _success_path(env, rec, aid, t_crash)
        measure(env, rec)
        _expect_single_success(env, rec, aid)
        if point in ("after_sbatch", "before_identity"):
            rec.passed = rec.passed and "attempt.uncertain" in env.events(aid)

    return run


def sc_accounting_gap(env: Env, rec: Rec) -> None:
    rec.expected = (
        "job leaves squeue before sacct knows it: gap flagged, never guessed; COMPLETED once sacct shows it"
    )
    env.sim.set_faults(accounting_lag_s=10, min_job_age_s=0.2)
    aid, t0 = env.submit_single()
    _success_path(env, rec, aid, t0, timeout=120)
    measure(env, rec)
    _expect_single_success(env, rec, aid)
    ev = env.events(aid)
    gap = "attempt.observation_gap" in ev
    rec.notes.append(f"observation_gap flagged: {gap}")
    term = env.ledger(
        "SELECT payload_json FROM events WHERE entity_id=? AND event_type='attempt.terminal'", (aid,)
    )
    src = json.loads(term[0]["payload_json"]).get("source") if term else None
    rec.notes.append(f"terminal state from: {src}")
    # either the gap was flagged, or the job was caught in squeue before it vanished; never invented
    rec.passed = rec.passed and (gap or src == "squeue")


def sc_squeue_timeouts(env: Env, rec: Rec) -> None:
    rec.expected = "squeue times out for 12 s while the job runs: no state is invented; completes after"
    aid, _t0 = env.submit_single(extra_argv=("--sleep-per-step", "0.02"))
    wait_for(lambda: env.attempt(aid).get("scheduler_state") == "RUNNING", timeout=60, what="RUNNING")
    env.sim.set_faults(transport_timeout_cmds=["squeue"])
    t_fault = time.monotonic()
    states_during: set[str] = set()
    while time.monotonic() - t_fault < 12:
        states_during.add(str(env.attempt(aid).get("scheduler_state")))
        sleep(0.3)
    env.sim.set_faults(transport_timeout_cmds=[])
    t_clear = time.monotonic()
    wait_for(lambda: settled(env, aid), timeout=90, what="settled")
    rec.decided_s = rec.settled_s = round(time.monotonic() - t_clear, 2)
    measure(env, rec)
    _expect_single_success(env, rec, aid)
    rec.notes.append(
        f"ledger states during the outage: {sorted(states_during)}; times measured from recovery"
    )
    # sacct keeps working, so a COMPLETED seen through sacct during the outage is legitimate
    rec.passed = rec.passed and states_during <= {"RUNNING", "COMPLETED"}


def sc_auth_window(env: Env, rec: Rec) -> None:
    rec.expected = (
        "auth expires for 6 s at submit: operation blocked with 'hpclab auth sim'; completes after; 1 job"
    )
    prep = env.prepare(env.spec())
    run = prep["runs"][0]
    env.sim.set_faults(auth_required=True)
    t0 = time.monotonic()
    rec_ = env.call(
        "run_submit",
        run_id=run["run_id"],
        execution_plan_hash=run["exec_plan_hash"],
        idempotency_key="auth-1",
    )
    aid = str(rec_["attempt_id"])

    def blocked() -> bool:
        op = env.op(env.attempt(aid).get("operation_id"))
        err = json.loads(op.get("error_json") or "null") or {}
        return op.get("state") == "blocked" and "hpclab auth" in str(
            (err.get("details") or {}).get("remediation")
        )

    wait_for(blocked, timeout=30, what="blocked with remediation")
    rec.notes.append(f"blocked with remediation after {time.monotonic() - t0:.1f}s")
    sleep(6 - (time.monotonic() - t0))
    env.sim.set_faults(auth_required=False)
    t_clear = time.monotonic()
    wait_for(lambda: settled(env, aid), timeout=200, what="settled after auth returned")
    rec.decided_s = rec.settled_s = round(time.monotonic() - t_clear, 2)
    rec.notes.append(
        "times measured from the moment auth works again (submit.* retry after auth_retry_s=120 s)"
    )
    measure(env, rec)
    _expect_single_success(env, rec, aid)


def sc_duplicate_token(env: Env, rec: Rec) -> None:
    rec.expected = "a 2nd job with the same token appears (external resubmission after a lost reply): flagged, nothing cancelled"
    env.sim.set_faults(sbatch_lose_reply=1)
    aid, t0 = env.submit_single(extra_argv=("--sleep-per-step", "0.01"))
    wait_for(lambda: len(env.sim.jobs()) >= 1, timeout=60, what="first job", interval=0.02)
    a = env.attempt(aid)
    script = Path(a["remote_dir"]) / "job.sbatch"
    out = env.sim.run("sbatch", "--parsable", str(script), cwd=a["remote_dir"])
    if out.returncode != 0:
        raise RuntimeError(f"external sbatch failed: {out.stderr}")
    rec.external_jobs = 1

    def decided() -> bool:
        return flagged(env, aid) or "attempt.duplicate_job" in env.events(aid)

    wait_for(decided, timeout=90, what="duplicate flagged")
    rec.decided_s = round(time.monotonic() - t0, 2)
    sleep(2)  # anything hpclab might do next (it must not cancel)
    measure(env, rec)
    a = env.attempt(aid)
    rec.outcome = f"{a.get('phase')} ({'duplicate_job event' if 'attempt.duplicate_job' in env.events(aid) else env.op(a.get('operation_id')).get('stage')})"
    rec.passed = rec.jobs_accepted == 2 and rec.cancels == 0 and rec.dispatches == 1 and decided()


def _long_running(env: Env) -> tuple[str, str]:
    aid, _ = env.submit_single(extra_argv=("--sleep-per-step", "0.1"))
    wait_for(lambda: env.attempt(aid).get("scheduler_state") == "RUNNING", timeout=60, what="RUNNING")
    return aid, str(env.attempt(aid)["job_id"])


def _cancel(env: Env, aid: str) -> float:
    t0 = time.monotonic()
    env.call(
        "run_cancel",
        attempt_id=aid,
        expected_version=env.attempt(aid)["version"],
        idempotency_key=new_id("request"),
        reason="benchmark",
    )
    return t0


def sc_cancel_genuine(env: Env, rec: Rec) -> None:
    rec.expected = "positive control: identity verified, exactly this job receives scancel, cancel confirmed"
    aid, job_id = _long_running(env)
    t0 = _cancel(env, aid)
    wait_for(
        lambda: env.attempt(aid).get("cancel_state") in ("confirmed", "refused", "failed"),
        timeout=60,
        what="cancel outcome",
    )
    rec.decided_s = round(time.monotonic() - t0, 2)
    measure(env, rec, expected_cancel_jobs={job_id})
    a = env.attempt(aid)
    rec.outcome = f"cancel {a.get('cancel_state')}, {a.get('scheduler_state')}"
    rec.passed = a.get("cancel_state") == "confirmed" and rec.cancels == 1 and rec.wrong_cancels == 0


def sc_cancel_foreign(env: Env, rec: Rec) -> None:
    rec.expected = "the job id now names another user's job (id reuse): cancel refused, zero scancel"
    aid, job_id = _long_running(env)
    with env.sim.state.tx() as conn:
        job = env.sim.state.load_job(conn, int(job_id))
        assert job is not None
        job.user = "mallory"
        env.sim.state.put_job(conn, job)
    t0 = _cancel(env, aid)
    wait_for(
        lambda: env.attempt(aid).get("cancel_state") in ("confirmed", "refused", "failed"),
        timeout=60,
        what="cancel outcome",
    )
    rec.decided_s = round(time.monotonic() - t0, 2)
    measure(env, rec)
    a = env.attempt(aid)
    rec.outcome = f"cancel {a.get('cancel_state')}"
    rec.passed = a.get("cancel_state") == "refused" and rec.cancels == 0


def sc_kill9_workflow(env: Env, rec: Rec) -> None:
    rec.expected = (
        "kill -9 at a random moment of a 2x2 dispatch: at most 1 job per attempt; every cell valid or flagged"
    )
    spec = env.spec(
        extra_argv=("--sleep-per-step", "0.005"), matrix={"method": ["baseline", "mixup"]}, seeds=[1, 2]
    )
    prep = env.prepare(spec)
    wf = env.call(
        "workflow_submit",
        idempotency_key=new_id("request"),
        plan_id=prep["plan_id"],
        plan_hash=prep["plan_hash"],
    )
    delay = random.uniform(0.0, 12.0)  # noqa: S311 - benchmark randomness, not security
    sleep(delay)
    env.kill_controller()
    t_kill = time.monotonic()
    phases = Counter(a["phase"] for a in env.ledger("SELECT phase FROM attempts"))
    rec.notes.append(f"killed {delay:.1f}s after submit; attempt phases then: {dict(phases)}")
    env.client.ensure_running()

    def done() -> bool:
        w = env.call("workflow_get", workflow_id=wf["workflow_id"], limit=1)
        if w["state"] == "completed":
            return True
        att = env.ledger("SELECT attempt_id, phase FROM attempts")
        open_ = [a for a in att if a["phase"] not in ("terminal", *UNRESOLVED)]
        return (
            not open_
            and all(flagged(env, a["attempt_id"]) for a in att if a["phase"] in UNRESOLVED)
            and len(att) == 4
        )

    wait_for(done, timeout=150, what="workflow settled", interval=0.5)
    rec.decided_s = rec.settled_s = round(time.monotonic() - t_kill, 2)
    measure(env, rec)
    w = env.call("workflow_get", workflow_id=wf["workflow_id"], limit=1)
    rec.outcome = f"workflow {w['state']} {w['counts']}"
    rec.passed = rec.duplicates == 0 and rec.wrong_states == 0 and rec.unresolved == rec.unresolved_flagged


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: str
    run: Callable[[Env, Rec], None]
    crash: str | None = None
    quick: bool = True


SCENARIOS: list[Scenario] = [
    Scenario("lost_reply", "sbatch_lose_reply=1 (job accepted, reply lost)", sc_lost_reply),
    Scenario("ambiguous_error_queued", "sbatch_ambiguous_error=1, job queued", sc_ambiguous_queued),
    Scenario(
        "ambiguous_error_not_queued", "sbatch_ambiguous_error=1, nothing queued", sc_ambiguous_not_queued
    ),
    Scenario(
        "crash_before_inflight",
        "controller SIGKILL before the in-flight commit",
        _crash("before_inflight"),
        "before_inflight",
    ),
    Scenario(
        "crash_after_inflight",
        "controller SIGKILL after the in-flight commit, before sbatch",
        _crash("after_inflight"),
        "after_inflight",
    ),
    Scenario(
        "crash_after_sbatch",
        "controller SIGKILL after sbatch accepted, reply unread",
        _crash("after_sbatch"),
        "after_sbatch",
    ),
    Scenario(
        "crash_before_identity",
        "controller SIGKILL before the job identity is committed",
        _crash("before_identity"),
        "before_identity",
    ),
    Scenario("accounting_gap", "accounting_lag_s=10 + min_job_age_s=0.2", sc_accounting_gap),
    Scenario("squeue_timeouts", "squeue transport timeouts for 12 s", sc_squeue_timeouts),
    Scenario("auth_required_window", "auth_required for 6 s from submit", sc_auth_window),
    Scenario("duplicate_same_token", "lost reply + external resubmission (same token)", sc_duplicate_token),
    Scenario("cancel_genuine", "run_cancel on a running job (control)", sc_cancel_genuine),
    Scenario(
        "cancel_foreign_owner", "run_cancel after the job id names another user's job", sc_cancel_foreign
    ),
    Scenario("kill9_random_workflow", "kill -9 at a random moment of a 2x2 workflow", sc_kill9_workflow),
]


# ---------------------------------------------------------------------------
# Runner + report
# ---------------------------------------------------------------------------


def _make_writable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            p = Path(dirpath) / name
            with contextlib.suppress(OSError):
                st = p.lstat()
                if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                    p.chmod(st.st_mode | stat.S_IWUSR)


def run_one(sc: Scenario, rep: int, work: Path, keep: bool) -> Rec:
    rec = Rec(scenario=sc.name, rep=rep)
    root = work / f"{sc.name}-{rep}"
    t = time.monotonic()
    env: Env | None = None
    try:
        env = Env(root, crash=(sc.crash, 1) if sc.crash else None)
        sc.run(env, rec)
        # the invariants every scenario must keep (README): no duplicate caused by hpclab, no
        # scancel of a job it was not asked to cancel, no final state that differs from Slurm's
        rec.passed = rec.passed and rec.duplicates == 0 and rec.wrong_cancels == 0 and rec.wrong_states == 0
    except Exception as exc:  # noqa: BLE001 - a failing repetition is a result, not a crash
        rec.passed = False
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.notes.append(traceback.format_exc(limit=3)[-1500:])
        if env is not None:
            with contextlib.suppress(Exception):
                if not rec.attempts:
                    measure(env, rec)
                snap = [
                    {k: a[k] for k in ("attempt_id", "phase", "scheduler_state", "result_state", "job_id")}
                    for a in env.ledger("SELECT * FROM attempts")
                ]
                rec.notes.append(f"attempts at failure: {snap}")
    finally:
        if env is not None:
            env.close()
        rec.wall_s = round(time.monotonic() - t, 1)
        if not keep:
            _make_writable(root)
            shutil.rmtree(root, ignore_errors=True)
    return rec


def _q(values: list[float]) -> str:
    if not values:
        return "—"
    s = sorted(values)
    med = statistics.median(s)
    return f"{med:.1f} / {s[-1]:.1f}"


def summarize(records: list[Rec]) -> list[dict[str, Any]]:
    out = []
    for sc in SCENARIOS:
        rs = [r for r in records if r.scenario == sc.name]
        if not rs:
            continue
        out.append(
            {
                "scenario": sc.name,
                "fault": sc.fault,
                "expected": next((r.expected for r in rs if r.expected), ""),
                "reps": len(rs),
                "passed": sum(r.passed for r in rs),
                "errors": sum(1 for r in rs if r.error),
                "dispatches": sum(r.dispatches for r in rs),
                "jobs_accepted": sum(r.jobs_accepted for r in rs),
                "external_jobs": sum(r.external_jobs for r in rs),
                "duplicates": sum(r.duplicates for r in rs),
                "cancels": sum(r.cancels for r in rs),
                "wrong_cancels": sum(r.wrong_cancels for r in rs),
                "attempts": sum(r.attempts for r in rs),
                "unresolved": sum(r.unresolved for r in rs),
                "unresolved_flagged": sum(r.unresolved_flagged for r in rs),
                "wrong_states": sum(r.wrong_states for r in rs),
                "decided_s": [r.decided_s for r in rs if r.decided_s is not None],
                "settled_s": [r.settled_s for r in rs if r.settled_s is not None],
            }
        )
    return out


def markdown(meta: dict[str, Any], summary: list[dict[str, Any]], records: list[Rec]) -> str:
    total = len(records)
    passed = sum(r.passed for r in records)
    n_chosen = len(meta.get("scenarios") or SCENARIOS)
    scope = f" ({n_chosen} of {len(SCENARIOS)} scenarios)" if n_chosen < len(SCENARIOS) else ""
    lines = [
        "# hpclab fault-injection matrix",
        "",
        f"Generated {meta['generated_at']} · hpclab {meta['hpclab_version']} · Python {meta['python']} · "
        f"{meta['platform']} · mode **{meta['mode']}**{scope} · {meta['reps']} rep(s) per scenario · "
        f"{meta['workers']} parallel workers · wall time {meta['wall_s']:.0f} s",
        "",
        f"**{passed}/{total} repetitions met their expected outcome.** Every repetition ran in a fresh hpclab home "
        "with a real controller subprocess, the real services and `SlurmScheduler` against the Slurm simulator "
        "(ground truth). Methodology and limits: [README.md](../README.md). Site-policy change vs the bundled sim "
        f"profile: `{json.dumps(POLICY)}`.",
        "",
        "## Summary (sums over all repetitions; nothing is omitted)",
        "",
        "| Scenario | Injected fault | Pass | sbatch calls by hpclab | Jobs accepted by Slurm | Duplicates | "
        "Cancels (wrong job) | Unresolved (flagged for a human) | Wrong final states | Decided, s (median / max) | "
        "Settled, s (median / max) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        ext = f" (+{s['external_jobs']} external)" if s["external_jobs"] else ""
        lines.append(
            f"| `{s['scenario']}` | {s['fault']} | {s['passed']}/{s['reps']} | {s['dispatches']} / {s['attempts']} attempts | "
            f"{s['jobs_accepted'] - s['external_jobs']}{ext} | {s['duplicates']} | {s['cancels']} ({s['wrong_cancels']}) | "
            f"{s['unresolved']} ({s['unresolved_flagged']}) | {s['wrong_states']} | {_q(s['decided_s'])} | {_q(s['settled_s'])} |"
        )
    lines += ["", "## Expected outcome per scenario", ""]
    for s in summary:
        lines.append(f"- `{s['scenario']}`: {s['expected']}")
    failing = [r for r in records if not r.passed]
    lines += ["", f"## Failure cases ({len(failing)})", ""]
    if not failing:
        lines.append("None in this run.")
    for r in failing:
        lines.append(f"### `{r.scenario}` rep {r.rep}")
        lines.append("")
        lines.append(f"- outcome: {r.outcome or '—'}")
        if r.error:
            lines.append(f"- error: `{r.error}`")
        lines.append(
            f"- counters: sbatch calls {r.dispatches}, jobs {r.jobs_accepted}, duplicates {r.duplicates}, "
            f"cancels {r.cancels} (wrong {r.wrong_cancels}), unresolved {r.unresolved} (flagged {r.unresolved_flagged}), "
            f"wrong states {r.wrong_states}"
        )
        for n in r.notes:
            lines.append(f"- note: {n.strip()[:1500]}")
        lines.append("")
    lines += [
        "",
        "## Every repetition",
        "",
        "| Scenario | Rep | Pass | Outcome | Decided s | Settled s | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(records, key=lambda x: ([s.name for s in SCENARIOS].index(x.scenario), x.rep)):
        notes = "; ".join(
            n.strip().replace("\n", " ")[:160] for n in r.notes if not n.startswith("Traceback")
        )
        lines.append(
            f"| `{r.scenario}` | {r.rep} | {'yes' if r.passed else '**no**'} | {r.outcome or r.error or ''} | "
            f"{r.decided_s if r.decided_s is not None else '—'} | {r.settled_s if r.settled_s is not None else '—'} | {notes} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "The simulator reproduces Slurm's command-line behaviour and output formats, not its internals: real "
        "slurmctld/slurmdbd timing, multi-cluster federation, SSH ControlMaster failures and site-specific "
        "wrappers are not exercised. Timings depend on the controller's 5 s observation tick and this machine's "
        "load (repetitions run in parallel). See README.md.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--quick", action="store_true", help="1 repetition per scenario (< 3 min)")
    p.add_argument("--reps", type=int, default=None, help="repetitions per scenario (default 5, quick 1)")
    p.add_argument("--only", default=None, help="comma-separated scenario names")
    p.add_argument("--jobs", type=int, default=None, help="parallel repetitions (default 5)")
    p.add_argument("--keep", action="store_true", help="keep each repetition's directory (debugging)")
    p.add_argument("--seed", type=int, default=None, help="random seed (kill -9 timing)")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = p.parse_args(argv)
    if args.list:
        for sc in SCENARIOS:
            sys.stdout.write(f"{sc.name:28} {sc.fault}\n")
        return 0
    chosen = SCENARIOS
    if args.only:
        names = {n.strip() for n in args.only.split(",") if n.strip()}
        unknown = names - {s.name for s in SCENARIOS}
        if unknown:
            p.error(f"unknown scenario(s): {sorted(unknown)}")
        chosen = [s for s in SCENARIOS if s.name in names]
    if args.reps is not None and args.reps < 1:
        p.error("--reps must be at least 1")
    if args.jobs is not None and args.jobs < 1:
        p.error("--jobs must be at least 1")
    reps = args.reps or (1 if args.quick else 5)
    # a subset or another repetition count is not the full (or quick) matrix: its own report, so
    # it can never overwrite the published one or pass itself off as it
    mode = "custom" if args.only or args.reps else ("quick" if args.quick else "full")
    stem = {"full": "fault_matrix", "quick": "fault_matrix_quick", "custom": "fault_matrix_custom"}[mode]
    workers = max(1, args.jobs or 5)
    seed = args.seed if args.seed is not None else int(time.time())
    random.seed(seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not os.access(out, os.W_OK):  # fail now, not after minutes of repetitions
        p.error(f"--out {out} is not writable")
    work = Path(tempfile.mkdtemp(prefix="hpclab-fault-matrix-"))
    tasks = [(sc, rep) for rep in range(1, reps + 1) for sc in chosen]
    # the long scenarios first, so the parallel pool drains evenly
    tasks.sort(key=lambda t: t[0].name not in ("auth_required_window", "kill9_random_workflow"))
    t0 = time.monotonic()
    records: list[Rec] = []
    lock = threading.Lock()
    sys.stderr.write(
        f"running {len(tasks)} repetitions ({len(chosen)} scenarios x {reps}) with {workers} workers\n"
    )
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(run_one, sc, rep, work, args.keep): (sc, rep) for sc, rep in tasks}
    try:
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                records.append(rec)
                sys.stderr.write(
                    f"[{time.monotonic() - t0:6.1f}s] {rec.scenario:28} rep {rec.rep}: "
                    f"{'pass' if rec.passed else 'FAIL'} ({rec.wall_s:.0f}s) {rec.outcome or rec.error or ''}\n"
                )
    except KeyboardInterrupt:
        # Ctrl-C: stop every repetition at its next wait, let each one stop its controller and
        # simulated jobs (a second Ctrl-C must not cut that short), write no partial report.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        STOP.set()
        running = sum(1 for f in futures if f.running())
        sys.stderr.write(
            f"\ninterrupted: stopping {running} running repetition(s) and their controllers and "
            "simulated jobs ...\n"
        )
        pool.shutdown(wait=True, cancel_futures=True)
        if args.keep:
            sys.stderr.write(f"interrupted: no report written; repetition directories kept in {work}\n")
        else:
            _make_writable(work)
            shutil.rmtree(work, ignore_errors=True)
            sys.stderr.write("interrupted: cleaned up; no report written\n")
        return 130
    pool.shutdown(wait=True)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    wall = time.monotonic() - t0
    meta = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hpclab_version": hpclab.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "mode": mode,
        "scenarios": [sc.name for sc in chosen],
        "reps": reps,
        "workers": workers,
        "random_seed": seed,
        "wall_s": round(wall, 1),
        "policy_overrides": POLICY,
        "sim_timings": FAST_SIM,
        "command": " ".join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])]),
    }
    summary = summarize(records)
    (out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "records": [asdict(r) for r in records]}, indent=1),
        encoding="utf-8",
    )
    (out / f"{stem}.md").write_text(markdown(meta, summary, records), encoding="utf-8")
    passed = sum(r.passed for r in records)
    sys.stderr.write(f"{passed}/{len(records)} passed in {wall:.0f}s -> {out / stem}.md\n")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
