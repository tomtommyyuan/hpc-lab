"""Simulator state: one SQLite file shared by every fake Slurm process.

Concurrency: each process opens short-lived connections; every mutation runs inside
``BEGIN IMMEDIATE`` (``SimState.tx``), so sbatch, scancel, the scheduler pass of each job runner
and admin commands serialize on the database lock. WAL lets readers (squeue, sacct) proceed
while a writer holds the lock.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpclab.sim.config import SimConfig, SimFaults

TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
STATE_COMPACT = {
    "PENDING": "PD",
    "RUNNING": "R",
    "COMPLETING": "CG",
    "COMPLETED": "CD",
    "FAILED": "F",
    "CANCELLED": "CA",
    "TIMEOUT": "TO",
    "OUT_OF_MEMORY": "OOM",
    "NODE_FAIL": "NF",
    "PREEMPTED": "PR",
    "BOOT_FAIL": "BF",
    "DEADLINE": "DL",
    "SUSPENDED": "S",
}
RUNNER_START_GRACE_S = 15.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS faults (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    name TEXT PRIMARY KEY,
    admin_state TEXT NOT NULL DEFAULT 'up',
    reason TEXT,
    since REAL
);
CREATE TABLE IF NOT EXISTS lost_replies (call_id TEXT PRIMARY KEY, job_id INTEGER, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    data_json TEXT NOT NULL,
    state TEXT NOT NULL,
    submit_time REAL NOT NULL,
    end_time REAL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
"""


@dataclass
class SimJob:
    """A simulated job (one array task = one row, like Slurm's job records)."""

    job_id: int
    name: str
    user: str
    uid: int
    account: str
    partition: str
    qos: str
    submit_time: float
    time_limit_s: int | None
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    cpus_per_node: int = 1
    mem_per_node_mb: int = 4096
    gpus_per_node: int = 0
    gpu_type: str | None = None
    state: str = "PENDING"
    reason: str | None = None
    raw_state: str | None = None  # e.g. "CANCELLED by 501" (sacct style)
    start_time: float | None = None
    end_time: float | None = None
    req_nodelist: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    constraint: str | None = None
    dependency: str | None = None
    comment: str | None = None
    submit_line: str = ""
    script_path: str = ""
    script_args: list[str] = field(default_factory=list)
    submit_dir: str = ""
    work_dir: str = ""
    stdout_pattern: str = ""
    stderr_pattern: str | None = None
    open_mode: str = "truncate"
    env: dict[str, str] = field(default_factory=dict)
    array_job_id: int | None = None
    array_task_id: int | None = None
    node_list: list[str] = field(default_factory=list)
    alloc: dict[str, dict[str, Any]] = field(default_factory=dict)  # node -> {"gpus": [...], "cpus", "mem"}
    exit_code: int | None = None
    exit_signal: int | None = None
    batch_exit_code: int | None = None
    batch_signal: int | None = None
    fate: str | None = None  # node_fail | preempt | oom | requeue (injected at submit)
    runner_pid: int | None = None
    pgid: int | None = None
    cancel_requested: bool = False
    cancelled_by: int | None = None
    restart_count: int = 0
    stdout_path: str | None = None
    stderr_path: str | None = None
    requeue: bool = False
    no_requeue: bool = False
    hold: bool = False
    # Earlier execution segments of a requeued job (oldest first): {"submit", "start", "end",
    # "state", "node_list", "exit_code", "exit_signal"} ("submit": the submit time that segment
    # ran under; a requeue restamps the job's); ``sacct --duplicates`` shows one record each.
    segments: list[dict[str, Any]] = field(default_factory=list)
    # Injected record (history, another user's job): never executed, scheduled or reaped.
    synthetic: bool = False

    # -- derived views ------------------------------------------------------------
    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def display_id(self) -> str:
        """``123`` or ``123_4`` for array tasks (squeue ``%i`` / sacct ``JobID``)."""
        if self.array_task_id is not None and self.array_job_id is not None:
            return f"{self.array_job_id}_{self.array_task_id}"
        return str(self.job_id)

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_node * self.nodes

    @property
    def total_cpus(self) -> int:
        return self.cpus_per_node * self.nodes

    def matches(self, token: str) -> bool:
        """Does a user-supplied id (``123``, ``123_4``, raw task id) select this job?"""
        base, sep, task = token.partition("_")
        if not base.isdigit():
            return False
        b = int(base)
        if sep:
            return self.array_job_id == b and task.isdigit() and self.array_task_id == int(task)
        return self.job_id == b or self.array_job_id == b

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SimJob:
        data = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def pid_alive(pid: int | None) -> bool:
    """True if ``pid`` exists and is not a zombie child of ours."""
    if not pid or pid <= 0:
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def safe_killpg(pgid: int | None, sig: int) -> bool:
    """Signal a job's process group, refusing anything that could hit ourselves or init."""
    if not pgid or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


class SimState:
    """Access to ``<root>/state.db`` and ``<root>/config.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.db_path = self.root / "state.db"
        self.config_path = self.root / "config.json"
        self._config: SimConfig | None = None

    # -- config ----------------------------------------------------------------
    @property
    def config(self) -> SimConfig:
        if self._config is None:
            self._config = SimConfig.load(self.config_path) if self.config_path.exists() else SimConfig()
        return self._config

    def reload_config(self) -> SimConfig:
        self._config = None
        return self.config

    # -- connections -------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Serializable write transaction (BEGIN IMMEDIATE)."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # -- faults ------------------------------------------------------------------
    def faults(self, conn: sqlite3.Connection | None = None) -> SimFaults:
        if conn is None:
            with self.reader() as c:
                return self.faults(c)
        rows = conn.execute("SELECT key, value_json FROM faults").fetchall()
        return SimFaults.from_dict({r["key"]: json.loads(r["value_json"]) for r in rows})

    def set_faults(self, **faults: Any) -> SimFaults:
        coerced = {k: SimFaults.coerce(k, v) for k, v in faults.items()}
        with self.tx() as conn:
            for key, value in coerced.items():
                conn.execute(
                    "INSERT INTO faults(key, value_json) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                    (key, json.dumps(value)),
                )
            self.bump_generation(conn)
            return self.faults(conn)

    def clear_faults(self) -> None:
        with self.tx() as conn:
            conn.execute("DELETE FROM faults")
            self.bump_generation(conn)

    @staticmethod
    def consume(conn: sqlite3.Connection, key: str) -> bool:
        """Decrement a counter fault inside the caller's transaction; True if one was consumed."""
        row = conn.execute("SELECT value_json FROM faults WHERE key=?", (key,)).fetchone()
        if row is None:
            return False
        value = int(json.loads(row["value_json"]) or 0)
        if value <= 0:
            return False
        conn.execute("UPDATE faults SET value_json=? WHERE key=?", (json.dumps(value - 1), key))
        return True

    def min_job_age_s(self, faults: SimFaults) -> float:
        return faults.min_job_age_s if faults.min_job_age_s is not None else self.config.min_job_age_s

    def accounting_lag_s(self, faults: SimFaults) -> float:
        return (
            faults.accounting_lag_s if faults.accounting_lag_s is not None else self.config.accounting_lag_s
        )

    def gpu_util(self, faults: SimFaults) -> float:
        return faults.gpu_util if faults.gpu_util is not None else self.config.gpu_util

    # -- job ids -------------------------------------------------------------------
    def allocate_job_id(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key='next_job_id'").fetchone()
        job_id = int(row["value"]) if row else self.config.first_job_id
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('next_job_id', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(job_id + 1),),
        )
        return job_id

    def peek_job_id(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key='next_job_id'").fetchone()
        return int(row["value"]) if row else self.config.first_job_id

    # -- jobs ---------------------------------------------------------------------
    @staticmethod
    def put_job(conn: sqlite3.Connection, job: SimJob) -> None:
        conn.execute(
            "INSERT INTO jobs(job_id, data_json, state, submit_time, end_time) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET data_json=excluded.data_json, state=excluded.state, "
            "end_time=excluded.end_time",
            (job.job_id, job.to_json(), job.state, job.submit_time, job.end_time),
        )
        SimState.bump_generation(conn)

    # -- scheduling generation --------------------------------------------------------
    @staticmethod
    def bump_generation(conn: sqlite3.Connection) -> None:
        """Record that something the scheduler depends on changed (jobs, nodes, faults)."""
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('sched_gen', '1') "
            "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER) + 1"
        )

    def generation(self) -> int:
        """Cheap read used by job runners to skip scheduling passes when nothing changed."""
        with self.reader() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='sched_gen'").fetchone()
        return int(row["value"]) if row else 0

    @staticmethod
    def load_job(conn: sqlite3.Connection, job_id: int) -> SimJob | None:
        row = conn.execute("SELECT data_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return SimJob.from_json(row["data_json"]) if row else None

    @staticmethod
    def load_jobs(conn: sqlite3.Connection, *, active_only: bool = False) -> list[SimJob]:
        sql = "SELECT data_json FROM jobs"
        if active_only:
            sql += " WHERE state IN ('PENDING', 'RUNNING', 'COMPLETING', 'SUSPENDED')"
        sql += " ORDER BY job_id"
        return [SimJob.from_json(r["data_json"]) for r in conn.execute(sql).fetchall()]

    def job(self, job_id: int) -> SimJob | None:
        with self.reader() as conn:
            return self.load_job(conn, job_id)

    def jobs(self) -> list[SimJob]:
        with self.reader() as conn:
            return self.load_jobs(conn)

    @staticmethod
    def finish(
        conn: sqlite3.Connection,
        job: SimJob,
        state: str,
        *,
        now: float,
        exit_code: int = 0,
        exit_signal: int = 0,
        batch_exit_code: int | None = None,
        batch_signal: int | None = None,
        raw_state: str | None = None,
        reason: str | None = None,
    ) -> SimJob:
        """Record a terminal state (no-op if the job is already terminal).

        A job that ends without ever starting gets ``start_time = end_time = now``, as
        slurmctld does for a cancelled pending job: squeue/sacct then show a Start for a job
        that never ran (no nodes, zero elapsed, empty AllocTRES).
        """
        if job.terminal:
            return job
        job.state = state
        job.raw_state = raw_state
        if job.start_time is None:
            job.start_time = now
        job.end_time = now
        job.exit_code = exit_code
        job.exit_signal = exit_signal
        job.batch_exit_code = batch_exit_code
        job.batch_signal = batch_signal
        job.reason = reason
        job.pgid = None if state != "RUNNING" else job.pgid
        SimState.put_job(conn, job)
        return job

    # -- nodes ---------------------------------------------------------------------
    def node_admin(self, conn: sqlite3.Connection | None = None) -> dict[str, tuple[str, str | None]]:
        if conn is None:
            with self.reader() as c:
                return self.node_admin(c)
        rows = conn.execute("SELECT name, admin_state, reason FROM nodes").fetchall()
        return {r["name"]: (r["admin_state"], r["reason"]) for r in rows}

    def set_node_state(self, name: str, state: str, reason: str | None = None) -> None:
        """Admin node state: ``up`` (resume), ``drain`` or ``down``."""
        if self.config.node(name) is None:
            raise ValueError(f"unknown node {name!r}")
        state = {"resume": "up", "idle": "up"}.get(state, state)
        if state not in ("up", "drain", "down"):
            raise ValueError(f"node state must be up/resume, drain or down, not {state!r}")
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO nodes(name, admin_state, reason, since) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET admin_state=excluded.admin_state, reason=excluded.reason, "
                "since=excluded.since",
                (name, state, reason if state != "up" else None, time.time()),
            )
            self.bump_generation(conn)

    # -- lost replies --------------------------------------------------------------
    @staticmethod
    def add_lost_reply(conn: sqlite3.Connection, call_id: str, job_id: int | None) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO lost_replies(call_id, job_id, created) VALUES (?, ?, ?)",
            (call_id, job_id, time.time()),
        )

    def pop_lost_reply(self, call_id: str) -> bool:
        """Consume the lost-reply marker for one transport call; True if sbatch recorded one."""
        with self.tx() as conn:
            cur = conn.execute("DELETE FROM lost_replies WHERE call_id=?", (call_id,))
            return cur.rowcount > 0

    # -- scheduling / reaping ----------------------------------------------------------
    def schedule(self, now: float | None = None) -> list[int]:
        """One FIFO scheduling pass; returns the job ids that were started."""
        from hpclab.sim.scheduler import plan

        t = time.time() if now is None else now
        with self.tx() as conn:
            # Injected pending records have no runner: they neither start nor block the queue.
            jobs = [j for j in self.load_jobs(conn) if not (j.synthetic and j.state == "PENDING")]
            if not any(j.state == "PENDING" for j in jobs):
                return []
            faults = self.faults(conn)
            decisions = plan(
                self.config, jobs, self.node_admin(conn), now=t, pending_delay_s=faults.pending_delay_s
            )
            by_id = {j.job_id: j for j in jobs}
            started: list[int] = []
            for d in decisions:
                job = by_id[d.job_id]
                if d.alloc is not None:
                    job.state = "RUNNING"
                    job.reason = None
                    job.start_time = t
                    job.alloc = d.alloc
                    job.node_list = list(d.alloc)
                    started.append(job.job_id)
                elif job.reason == d.reason:
                    continue
                else:
                    job.reason = d.reason
                self.put_job(conn, job)
            return started

    def reap(self, now: float | None = None) -> list[int]:
        """Jobs whose runner process vanished become NODE_FAIL (the "node" died with them)."""
        t = time.time() if now is None else now
        with self.reader() as conn:
            suspects = [
                j
                for j in self.load_jobs(conn, active_only=True)
                if not j.synthetic
                and (
                    (j.runner_pid and not pid_alive(j.runner_pid))
                    or (not j.runner_pid and t - j.submit_time > RUNNER_START_GRACE_S)
                )
            ]
        if not suspects:
            return []
        reaped: list[int] = []
        with self.tx() as conn:
            for stale in suspects:
                job = self.load_job(conn, stale.job_id)
                if job is None or job.terminal or job.runner_pid != stale.runner_pid:
                    continue
                safe_killpg(job.pgid, signal.SIGKILL)
                self.finish(conn, job, "NODE_FAIL", now=t, reason="NodeDown")
                reaped.append(job.job_id)
        return reaped
