"""Detached per-job supervisor: ``python -m hpclab.sim.jobrunner --root <sim_root> <job_id>``.

Plays slurmctld + slurmstepd for one job: waits for an allocation (running the shared FIFO
scheduling pass while pending), executes the batch script in its own session with the SLURM_*
environment, and records the terminal state:

* exit 0 -> COMPLETED ``0:0``; exit n -> FAILED ``n:0``; killed by signal s -> FAILED ``0:s``;
  processes the batch shell left behind are SIGKILLed when it exits (like proctrack/cgroup)
* time limit -> ``*** JOB n ON node CANCELLED AT t DUE TO TIME LIMIT ***``, SIGTERM, KillWait,
  SIGKILL -> TIMEOUT ``0:0``
* scancel -> ``*** JOB n ON node CANCELLED AT t ***`` -> CANCELLED (``CANCELLED by <uid>``) ``0:0``
* injected fates: PREEMPTED (TERM then KILL), NODE_FAIL (KILL, no message), OUT_OF_MEMORY
  (``Detected 1 oom-kill event(s)`` message, KILL, ``0:125``), preempt-and-requeue (TERM then
  KILL). A preempt-and-requeue or a node failure *requeues* a requeueable job (``--requeue``,
  or ``JobRequeue=1`` without ``--no-requeue``) like Slurm's batch_requeue: its segment (with
  the old submit time) is recorded for ``sacct --duplicates``, and it goes back to PENDING with
  ``restart_count + 1`` and a new submit time (now; one second after the old one when that is
  the same second, so accounting keeps separate records). A ``--no-requeue`` job ends
  PREEMPTED / NODE_FAIL.
* jobs on GPU nodes find the fake ``nvidia-smi`` (``<sim_root>/gpu-bin``) first on their PATH
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from hpclab.sim.formats import compress_hostlist, format_datetime_local
from hpclab.sim.state import SimJob, SimState, safe_killpg

log = logging.getLogger("hpclab.sim.jobrunner")

INTERNAL_ENV = ("HPCLAB_SIM_CALL_ID",)
RESCHEDULE_S = 0.5


@dataclass
class Outcome:
    state: str
    exit_code: int = 0
    exit_signal: int = 0
    batch_exit_code: int | None = None
    batch_signal: int | None = None
    raw_state: str | None = None
    reason: str | None = None
    # the controller requeues the job when it is requeueable (preempt-and-requeue, node failure)
    requeue: bool = False


def requeue_submit_time(previous: float, now: float) -> float:
    """The submit time Slurm stamps on a requeued job (batch_requeue_fini): now, or one second
    after the previous submission when that is the same second - every submission is its own
    accounting record, keyed by its (whole-second) submit time."""
    return now if int(now) > int(previous) else float(int(previous) + 1)


def expand_pattern(pattern: str, job: SimJob) -> str:
    """Slurm filename pattern substitution (%j %x %u %A %a %N %%)."""
    first_node = job.node_list[0] if job.node_list else "localhost"
    repl = {
        "j": str(job.job_id),
        "x": job.name,
        "u": job.user,
        "A": str(job.array_job_id if job.array_job_id is not None else job.job_id),
        "a": str(job.array_task_id if job.array_task_id is not None else 4294967294),
        "N": first_node,
        "n": "0",
        "t": "0",
        "%": "%",
    }
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "%" and i + 1 < len(pattern) and pattern[i + 1] in repl:
            out.append(repl[pattern[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def job_environment(job: SimJob, cfg_cluster: str, *, gpu_bin: str | None = None) -> dict[str, str]:
    """The environment a batch script sees: the submit environment plus SLURM_* variables.

    ``gpu_bin`` (the simulator's fake ``nvidia-smi`` directory) is prepended to PATH for jobs
    on GPU nodes.
    """
    env = {k: v for k, v in job.env.items() if k not in INTERNAL_ENV}
    first = job.node_list[0] if job.node_list else "localhost"
    gpus = [int(g) for g in job.alloc.get(first, {}).get("gpus", [])]
    cpus_on_node = int(job.alloc.get(first, {}).get("cpus", job.cpus_per_node))
    nodelist = compress_hostlist(job.node_list)
    slurm = {
        "SLURM_JOB_ID": str(job.job_id),
        "SLURM_JOBID": str(job.job_id),
        "SLURM_JOB_NAME": job.name,
        "SLURM_JOB_USER": job.user,
        "SLURM_JOB_UID": str(job.uid),
        "SLURM_JOB_ACCOUNT": job.account,
        "SLURM_JOB_PARTITION": job.partition,
        "SLURM_JOB_QOS": job.qos,
        "SLURM_JOB_NODELIST": nodelist,
        "SLURM_NODELIST": nodelist,
        "SLURM_JOB_NUM_NODES": str(job.nodes),
        "SLURM_NNODES": str(job.nodes),
        "SLURM_NTASKS": str(job.ntasks),
        "SLURM_NPROCS": str(job.ntasks),
        "SLURM_CPUS_ON_NODE": str(cpus_on_node),
        "SLURM_JOB_CPUS_PER_NODE": str(cpus_on_node),
        "SLURM_MEM_PER_NODE": str(job.mem_per_node_mb),
        "SLURM_SUBMIT_DIR": job.submit_dir,
        "SLURM_SUBMIT_HOST": socket.gethostname(),
        "SLURM_CLUSTER_NAME": cfg_cluster,
        "SLURMD_NODENAME": first,
        "SLURM_TOPOLOGY_ADDR": first,
        "SLURM_RESTART_COUNT": str(job.restart_count),
        "SLURM_JOB_START_TIME": str(int(job.start_time or time.time())),
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
        "SLURM_TASK_PID": str(os.getpid()),
        "ENVIRONMENT": "BATCH",
    }
    if job.time_limit_s is not None and job.start_time is not None:
        slurm["SLURM_JOB_END_TIME"] = str(int(job.start_time + job.time_limit_s))
    if job.cpus_per_task is not None:
        slurm["SLURM_CPUS_PER_TASK"] = str(job.cpus_per_task)
    if job.gpus_per_node:
        slurm["SLURM_GPUS_ON_NODE"] = str(len(gpus))
        slurm["SLURM_JOB_GPUS"] = ",".join(str(g) for g in gpus)
        slurm["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        slurm["GPU_DEVICE_ORDINAL"] = slurm["CUDA_VISIBLE_DEVICES"]
    if job.array_task_id is not None and job.array_job_id is not None:
        slurm["SLURM_ARRAY_JOB_ID"] = str(job.array_job_id)
        slurm["SLURM_ARRAY_TASK_ID"] = str(job.array_task_id)
    for key in [k for k in env if k.startswith("SLURM_") and k not in slurm and k != "SLURM_TIME_FORMAT"]:
        env.pop(key)
    env.update(slurm)
    if gpu_bin:
        env["PATH"] = os.pathsep.join([gpu_bin, env.get("PATH", "")]) if env.get("PATH") else gpu_bin
    return env


def gpu_bin_for(state: SimState, job: SimJob) -> str | None:
    """The fake nvidia-smi directory when the job's (first) node has GPUs, else None."""
    first = job.node_list[0] if job.node_list else None
    node = state.config.node(first) if first else None
    if node is None or not node.gpus:
        return None
    return str(state.root / "gpu-bin")


class JobRunner:
    def __init__(self, state: SimState, job_id: int) -> None:
        self.state = state
        self.job_id = job_id
        self.cfg = state.config
        self.poll = max(0.02, self.cfg.poll_interval_s)

    # -- phase 1: wait for an allocation -----------------------------------------
    def wait_for_allocation(self) -> SimJob | None:
        """Poll until the job is allocated (or ends while pending).

        A scheduling pass takes the database write lock, so it only runs when the scheduling
        generation changed (a job/node/fault changed) or ``RESCHEDULE_S`` elapsed (time-based
        conditions such as ``pending_delay_s``).
        """
        seen_gen = -1
        last_pass = 0.0
        while True:
            job = self.state.job(self.job_id)
            if job is None or job.terminal:
                return None
            if job.state == "RUNNING":
                return job
            gen = self.state.generation()
            now = time.monotonic()
            if gen != seen_gen or now - last_pass >= RESCHEDULE_S:
                seen_gen, last_pass = gen, now
                self.state.schedule()
                continue
            time.sleep(self.poll)

    # -- phase 2: run ---------------------------------------------------------------
    def run(self) -> int:
        while True:
            job = self.wait_for_allocation()
            if job is None:
                return 0
            outcome = self.execute(job)
            if not outcome.requeue or not self._requeue(outcome):
                break
        with self.state.tx() as conn:
            current = self.state.load_job(conn, self.job_id)
            if current is not None:
                self.state.finish(
                    conn,
                    current,
                    outcome.state,
                    now=time.time(),
                    exit_code=outcome.exit_code,
                    exit_signal=outcome.exit_signal,
                    batch_exit_code=outcome.batch_exit_code,
                    batch_signal=outcome.batch_signal,
                    raw_state=outcome.raw_state,
                    reason=outcome.reason,
                )
        # Freed resources: let queued jobs start without waiting for their own poll tick.
        self.state.schedule()
        return 0

    def _requeue(self, outcome: Outcome) -> bool:
        """Put the job back in the queue (Slurm requeue); False if it is not requeueable.

        The finished segment (state ``outcome.state``, its submit time) is kept for ``sacct
        --duplicates``; the job keeps its id, gets a new submit time (``requeue_submit_time``)
        and restarts from the top of its batch script with ``SLURM_RESTART_COUNT + 1``.
        """
        now = time.time()
        with self.state.tx() as conn:
            job = self.state.load_job(conn, self.job_id)
            if job is None or job.terminal or job.cancel_requested:
                return False
            if not (job.requeue or (self.cfg.job_requeue and not job.no_requeue)):
                return False
            # a preempted segment reads 0:15 in sacct, a node failure 0:0
            signal_no = 15 if outcome.state == "PREEMPTED" else outcome.exit_signal
            job.segments.append(
                {
                    "submit": job.submit_time,
                    "start": job.start_time,
                    "end": now,
                    "state": outcome.state,
                    "node_list": list(job.node_list),
                    "exit_code": outcome.exit_code,
                    "exit_signal": signal_no,
                }
            )
            job.submit_time = requeue_submit_time(job.submit_time, now)
            job.state, job.reason, job.raw_state = "PENDING", "BeginTime", None
            job.restart_count += 1
            job.start_time = job.end_time = None
            job.node_list, job.alloc, job.pgid, job.fate = [], {}, None, None
            self.state.put_job(conn, job)
        return True

    def _open_outputs(self, job: SimJob) -> tuple[IO[bytes], IO[bytes]] | None:
        mode = "ab" if job.open_mode == "append" else "wb"
        out_path = expand_pattern(job.stdout_pattern, job)
        err_path = expand_pattern(job.stderr_pattern, job) if job.stderr_pattern else out_path
        try:
            fo = open(out_path, mode)  # noqa: SIM115 - handed to the child process
        except OSError:
            return None
        if err_path == out_path:
            fe = fo
        else:
            try:
                fe = open(err_path, mode)  # noqa: SIM115
            except OSError:
                fo.close()
                return None
        job.stdout_path, job.stderr_path = out_path, err_path
        return fo, fe

    def _mark(self, **fields: object) -> SimJob | None:
        with self.state.tx() as conn:
            job = self.state.load_job(conn, self.job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            self.state.put_job(conn, job)
            return job

    def execute(self, job: SimJob) -> Outcome:
        streams = self._open_outputs(job)
        if streams is None:
            return Outcome("FAILED", exit_code=1, batch_exit_code=1, reason="NonZeroExitCode")
        fo, fe = streams
        try:
            self._mark(stdout_path=job.stdout_path, stderr_path=job.stderr_path)
            workdir = job.work_dir
            if not os.path.isdir(workdir):
                _note(
                    fe,
                    f"slurmstepd: error: couldn't chdir to `{workdir}': No such file or directory: going to /tmp instead",
                )
                workdir = "/tmp"
            fresh = self.state.job(self.job_id)
            if fresh is None or fresh.cancel_requested or fresh.terminal:
                return self._cancelled(fresh or job)
            try:
                proc = subprocess.Popen(
                    [job.script_path, *job.script_args],
                    cwd=workdir,
                    env=job_environment(job, self.cfg.cluster_name, gpu_bin=gpu_bin_for(self.state, job)),
                    stdin=subprocess.DEVNULL,
                    stdout=fo,
                    stderr=fe,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                _note(fe, f"slurmstepd: error: execve(): {os.path.basename(job.script_path)}: {exc.strerror}")
                return Outcome("FAILED", exit_code=1, batch_exit_code=1, reason="NonZeroExitCode")
            self._mark(pgid=proc.pid)
            return self.supervise(job, proc, fe)
        finally:
            fo.close()
            if fe is not fo:
                fe.close()

    def _cancelled(self, job: SimJob) -> Outcome:
        uid = job.cancelled_by if job.cancelled_by is not None else os.getuid()
        return Outcome("CANCELLED", batch_signal=15, batch_exit_code=0, raw_state=f"CANCELLED by {uid}")

    def _terminate(self, proc: subprocess.Popen[bytes], *, graceful: bool) -> None:
        if graceful:
            safe_killpg(proc.pid, signal.SIGCONT)
            safe_killpg(proc.pid, signal.SIGTERM)
            deadline = time.monotonic() + self.cfg.kill_wait_s
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.02)
        safe_killpg(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("job %s process group did not die after SIGKILL", self.job_id)

    def supervise(self, job: SimJob, proc: subprocess.Popen[bytes], fe: IO[bytes]) -> Outcome:
        start = job.start_time or time.time()
        deadline = start + job.time_limit_s if job.time_limit_s is not None else None
        fate_at = start + self.cfg.fate_delay_s if job.fate else None
        node = job.node_list[0] if job.node_list else "localhost"
        while proc.poll() is None:
            now = time.time()
            current = self.state.job(self.job_id)
            if current is not None and current.cancel_requested:
                _note(
                    fe,
                    f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT {format_datetime_local(now)} ***",
                )
                self._terminate(proc, graceful=True)
                return self._cancelled(current)
            if deadline is not None and now >= deadline:
                _note(
                    fe,
                    f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT "
                    f"{format_datetime_local(now)} DUE TO TIME LIMIT ***",
                )
                self._terminate(proc, graceful=True)
                return Outcome("TIMEOUT", batch_exit_code=0, batch_signal=15, reason="TimeLimit")
            if fate_at is not None and now >= fate_at:
                return self._apply_fate(job, proc, fe, node, now)
            time.sleep(self.poll)
        rc = proc.returncode
        # slurmstepd kills whatever is left in the step once the batch shell exits (proctrack):
        # background sidecars and orphaned workers must not outlive the job.
        safe_killpg(proc.pid, signal.SIGKILL)
        final = self.state.job(self.job_id)
        if final is not None and final.cancel_requested:
            _note(
                fe,
                f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT {format_datetime_local(time.time())} ***",
            )
            return self._cancelled(final)
        if rc == 0:
            return Outcome("COMPLETED", batch_exit_code=0, batch_signal=0)
        if rc > 0:
            return Outcome(
                "FAILED", exit_code=rc, batch_exit_code=rc, batch_signal=0, reason="NonZeroExitCode"
            )
        return Outcome(
            "FAILED",
            exit_code=0,
            exit_signal=-rc,
            batch_exit_code=0,
            batch_signal=-rc,
            reason="NonZeroExitCode",
        )

    def _apply_fate(
        self, job: SimJob, proc: subprocess.Popen[bytes], fe: IO[bytes], node: str, now: float
    ) -> Outcome:
        if job.fate == "requeue":
            _note(
                fe,
                f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT "
                f"{format_datetime_local(now)} DUE TO JOB REQUEUE ***",
            )
            self._terminate(proc, graceful=True)
            return Outcome("PREEMPTED", batch_exit_code=0, batch_signal=15, requeue=True)
        if job.fate == "preempt":
            _note(
                fe,
                f"slurmstepd: error: *** JOB {job.job_id} ON {node} CANCELLED AT "
                f"{format_datetime_local(now)} DUE TO PREEMPTION ***",
            )
            self._terminate(proc, graceful=True)
            return Outcome("PREEMPTED", batch_exit_code=0, batch_signal=15)
        if job.fate == "oom":
            _note(
                fe,
                f"slurmstepd: error: Detected 1 oom-kill event(s) in StepId={job.job_id}.batch. "
                "Some of your processes may have been killed by the cgroup out-of-memory handler.",
            )
            self._terminate(proc, graceful=False)
            return Outcome(
                "OUT_OF_MEMORY",
                exit_code=0,
                exit_signal=125,
                batch_exit_code=0,
                batch_signal=125,
                reason="OutOfMemory",
            )
        # node failure: no message (the node is gone); JobRequeue requeues a requeueable job
        self._terminate(proc, graceful=False)
        return Outcome("NODE_FAIL", batch_exit_code=0, batch_signal=9, reason="NodeDown", requeue=True)


def _note(stream: IO[bytes], line: str) -> None:
    try:
        stream.write((line + "\n").encode("utf-8"))
        stream.flush()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hpclab.sim.jobrunner")
    parser.add_argument("--root", required=True)
    parser.add_argument("job_id", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    state = SimState(Path(args.root))
    try:
        return JobRunner(state, args.job_id).run()
    except Exception:
        log.exception("job runner for %s crashed", args.job_id)
        with state.tx() as conn:
            job = state.load_job(conn, args.job_id)
            if job is not None:
                safe_killpg(job.pgid, signal.SIGKILL)
                state.finish(conn, job, "NODE_FAIL", now=time.time(), reason="NodeDown")
        return 1


if __name__ == "__main__":
    sys.exit(main())
