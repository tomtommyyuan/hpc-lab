"""An hpclab controller that SIGKILLs itself at a chosen point of ``submit.dispatch``.

Used by ``fault_matrix.py`` and ``tests/e2e`` to hit the crash windows of the submission
protocol *deterministically*, which an external ``kill -9`` at a random moment rarely does::

    python benchmarks/crash_controller.py --home <HPCLAB_HOME> --crash-at after_inflight \
        --nth 1 --marker <file>

It runs the normal controller (``hpclab.controller.daemon.Controller`` with the production
``default_app_factory``) after installing one hook. When the hook fires for the ``--nth`` time it
appends a line to ``--marker`` (fsynced) and sends SIGKILL to its own process: no ``finally``
blocks, no atexit, no SQLite commit of an open transaction -- the same as ``kill -9`` from
outside. If the marker already exists the hook never fires, so a client that autostarts this
script again after the crash gets an ordinary controller.

Crash points (in the order the dispatch handler reaches them):

``before_inflight``   the ``submit.dispatch`` handler starts for a STAGING attempt; the
                      in-flight intent is not committed yet.
``after_inflight``    ``phase=submission_in_flight`` is committed; sbatch has NOT run.
``after_sbatch``      the simulator's sbatch accepted the job and exited; its reply was never
                      read (the job exists, the ledger does not know its id).
``before_identity``   the reply was parsed and the job id is known in memory, inside the
                      transaction that would record it (never committed).

The hooks patch simulator/adapter/service internals (``SimTransport.run``,
``SlurmScheduler.submit``, ``submission.handle_dispatch``/``_bind_job``) and therefore only make
sense against the simulator. Nothing is printed to stdout.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

POINTS = ("before_inflight", "after_inflight", "after_sbatch", "before_identity")

log = logging.getLogger("hpclab.crash_controller")


class _Trigger:
    """Counts hook hits and kills the process on the n-th one (once per marker file)."""

    def __init__(self, point: str, nth: int, marker: Path) -> None:
        self.point = point
        self.nth = nth
        self.marker = marker
        self.hits = 0
        self.lock = threading.Lock()

    def armed(self) -> bool:
        return not self.marker.exists()

    def hit(self, detail: str) -> None:
        with self.lock:
            if not self.armed():
                return
            self.hits += 1
            if self.hits < self.nth:
                return
            line = f"{time.time():.6f} pid={os.getpid()} point={self.point} hit={self.hits} {detail}\n"
            fd = os.open(self.marker, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(10)  # never reached: SIGKILL cannot be caught


def install(point: str, trigger: _Trigger) -> None:
    """Install the hook for ``point`` (must run before the controller builds its app)."""
    from hpclab.adapters.slurm import SlurmScheduler
    from hpclab.domain.states import AttemptPhase
    from hpclab.services import submission
    from hpclab.sim.transport import SimTransport, command_name

    if point == "before_inflight":
        original_dispatch = submission.handle_dispatch

        @functools.wraps(original_dispatch)
        def dispatch(ctx: Any, item: Any) -> Any:
            aid = (item.payload or {}).get("attempt_id")
            row = ctx.db.one("SELECT phase FROM attempts WHERE attempt_id=?", (aid,)) if aid else None
            if row is not None and row["phase"] == AttemptPhase.STAGING:
                trigger.hit(f"attempt={aid}")
            return original_dispatch(ctx, item)

        # outbox_handlers() looks the name up at call time, i.e. when the controller starts
        submission.handle_dispatch = dispatch
    elif point == "after_inflight":
        original_submit = SlurmScheduler.submit

        @functools.wraps(original_submit)
        def submit(self: Any, script_path: str, **kw: Any) -> Any:
            if not kw.get("test_only"):
                trigger.hit(f"script={script_path}")
            return original_submit(self, script_path, **kw)

        SlurmScheduler.submit = submit  # type: ignore[method-assign]
    elif point == "after_sbatch":
        original_run = SimTransport.run

        @functools.wraps(original_run)
        def run(self: Any, argv: list[str], **kw: Any) -> Any:
            result = original_run(self, argv, **kw)
            if command_name(argv) == "sbatch" and "--test-only" not in argv:
                trigger.hit("sbatch returned")
            return result

        SimTransport.run = run  # type: ignore[method-assign]
    elif point == "before_identity":
        original_bind: Callable[..., Any] = submission._bind_job

        @functools.wraps(original_bind)
        def bind(tx: Any, cur: Any, **kw: Any) -> Any:
            if kw.get("source") == "sbatch_reply":
                trigger.hit(f"attempt={cur['attempt_id']} job={kw.get('job_id')}")
            return original_bind(tx, cur, **kw)

        submission._bind_job = bind  # type: ignore[assignment]
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(point)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--home", required=True)
    p.add_argument("--crash-at", choices=POINTS, required=True)
    p.add_argument("--nth", type=int, default=1, help="crash on the n-th hit of the hook (default 1)")
    p.add_argument("--marker", required=True, help="file recording the crash; its presence disarms the hook")
    args = p.parse_args(argv)

    from hpclab.config import get_paths
    from hpclab.controller.daemon import Controller, ControllerAlreadyRunning, default_app_factory

    paths = get_paths(args.home)
    install(args.crash_at, _Trigger(args.crash_at, max(1, args.nth), Path(args.marker)))
    controller = Controller(paths, default_app_factory)
    try:
        controller.serve_forever()
    except ControllerAlreadyRunning as exc:
        sys.stderr.write(f"crash_controller: {exc.message}\n")
        from hpclab.controller import protocol as proto

        return proto.EXIT_ALREADY_RUNNING
    return 0


if __name__ == "__main__":
    code = main()
    logging.shutdown()
    os._exit(code)
