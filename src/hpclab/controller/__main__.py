"""``python -m hpclab.controller [--home PATH] [--foreground] [--log-level LEVEL] [--workers N]``

Runs the controller for one hpclab home in this process until SIGTERM/SIGINT/SIGHUP or a
``controller.shutdown`` request (Ctrl-C again while it shuts down exits immediately). This is what ``HpcLabClient`` spawns (detached, in its own
session); ``--foreground`` additionally mirrors the log to stderr for interactive debugging.
Nothing is written to stdout (except argparse --help output).

Exit codes: 0 stopped cleanly, 1 startup/runtime error, 2 usage error, 3 another controller
already owns this home (``protocol.EXIT_ALREADY_RUNNING``).

When run as a script the process ends with ``os._exit`` as soon as the controller has stopped.
Threads that outlived the shutdown timeout (a handler blocked on a dead network, a service's
own non-daemon thread pool) therefore die at the same instant the kernel releases the instance
lock — never after it, when a successor could already be writing the ledger.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import logging
import os
import sys
from typing import NoReturn

from hpclab.config import get_paths
from hpclab.controller import protocol as proto
from hpclab.controller.daemon import Controller, ControllerAlreadyRunning, default_app_factory, log_formatter

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
WORKERS_ENV = "HPCLAB_WORKERS"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hpclab.controller",
        description="Run the hpclab controller (one per HPCLAB_HOME).",
    )
    p.add_argument("--home", help="hpclab home (default: $HPCLAB_HOME or ~/.hpclab)")
    p.add_argument("--foreground", action="store_true", help="also log to stderr (interactive debugging)")
    p.add_argument(
        "--log-level", default="INFO", choices=_LEVELS, type=str.upper, help="log level (default INFO)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=f"background worker threads (default: ${WORKERS_ENV} or 2); "
        "long transfers use at most N-1 of them",
    )
    return p


def _workers(arg: int | None) -> int | None:
    """``--workers``, else ``$HPCLAB_WORKERS`` (inherited from the client that spawns us)."""
    raw: str | int | None = arg if arg is not None else (os.environ.get(WORKERS_ENV) or None)
    if raw is None:
        return None
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{WORKERS_ENV} must be a whole number, got {raw!r}") from None
    if not 1 <= n <= 64:
        raise ValueError(f"the number of workers must be between 1 and 64, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse already wrote the usage message to stderr
        return proto.EXIT_USAGE if exc.code else proto.EXIT_OK
    try:
        workers = _workers(args.workers)
    except ValueError as exc:
        sys.stderr.write(f"hpclab controller: {exc}\n")
        return proto.EXIT_USAGE
    level = getattr(logging, args.log_level)
    faulthandler.enable()  # hard crashes leave a traceback in the spawn log (stderr)
    paths = get_paths(args.home)
    if args.foreground:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(log_formatter())
        handler.setLevel(level)
        logging.getLogger("hpclab").addHandler(handler)
    logging.captureWarnings(True)
    controller = Controller(paths, default_app_factory, log_level=level, worker_threads=workers)
    try:
        controller.serve_forever()
    except ControllerAlreadyRunning as exc:
        sys.stderr.write(f"hpclab controller: {exc.message}\n")
        return proto.EXIT_ALREADY_RUNNING
    except Exception as exc:
        logging.getLogger("hpclab.controller").exception("controller failed")
        sys.stderr.write(f"hpclab controller: fatal error: {type(exc).__name__}: {exc}\n")
        return proto.EXIT_ERROR
    if controller.finalize_pending:
        logging.getLogger("hpclab.controller").warning(
            "exiting with controller threads still running; they end with the process, which also "
            "releases the instance lock (their outbox items are released by the next controller)"
        )
    return proto.EXIT_OK


def _run_and_exit() -> NoReturn:
    code = main()
    logging.shutdown()  # flush the log files before the hard exit
    for stream in (sys.stderr, sys.stdout):
        with contextlib.suppress(Exception):
            stream.flush()
    os._exit(code)


if __name__ == "__main__":
    _run_and_exit()
