"""Entry point of the fake Slurm executables.

``python -m hpclab.sim.cli [--root <sim_root>] <command> [args...]`` where command is one of
``sbatch squeue sacct scancel sinfo sacctmgr sstat srun scontrol``, ``nvidia-smi`` (per job, via
``<sim_root>/gpu-bin``) or ``_admin``. The shims in
``<sim_root>/bin`` set ``HPCLAB_SIM_ROOT`` and call this module, so the real ``SlurmScheduler``
(including its bash snippets) runs unmodified against the simulator.

Admin (wrapped by ``hpclab sim ...`` later)::

    _admin set-fault key=value [key=value ...]   e.g. sbatch_lose_reply=1 transport_timeout_cmds=squeue,sacct
    _admin clear-faults | faults | jobs | info | shutdown
    _admin set-node <node> <up|drain|down> [reason words...]
    _admin wait-idle [timeout_s]
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import TextIO

from hpclab.sim.cmdutil import Ctx
from hpclab.sim.state import SimState

Handler = Callable[[Ctx, list[str]], int]

# command -> (module, function); imported lazily so each fake command loads only what it needs.
COMMANDS: dict[str, tuple[str, str]] = {
    "sbatch": ("hpclab.sim.cmd_sbatch", "main"),
    "squeue": ("hpclab.sim.cmd_squeue", "main"),
    "sacct": ("hpclab.sim.cmd_sacct", "main"),
    "sinfo": ("hpclab.sim.cmd_sinfo", "sinfo_main"),
    "scontrol": ("hpclab.sim.cmd_sinfo", "scontrol_main"),
    "scancel": ("hpclab.sim.cmd_misc", "scancel_main"),
    "sacctmgr": ("hpclab.sim.cmd_misc", "sacctmgr_main"),
    "sstat": ("hpclab.sim.cmd_misc", "sstat_main"),
    "srun": ("hpclab.sim.cmd_misc", "srun_main"),
    "nvidia-smi": ("hpclab.sim.cmd_misc", "nvidia_smi_main"),
}
SLURM_COMMANDS = tuple(COMMANDS)


def _handler(command: str) -> Handler | None:
    if command == "_admin":
        return admin_main
    target = COMMANDS.get(command)
    if target is None:
        return None
    module = import_module(target[0])
    handler: Handler = getattr(module, target[1])
    return handler


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> int:
    """Run one fake command. Streams/env/cwd are injectable for in-process tests."""
    args = list(sys.argv[1:] if argv is None else argv)
    environ = dict(os.environ if env is None else env)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    root = environ.get("HPCLAB_SIM_ROOT")
    if args[:1] == ["--root"] and len(args) >= 2:
        root, args = args[1], args[2:]
        environ["HPCLAB_SIM_ROOT"] = root
    if not args:
        err.write(__doc__ or "")
        return 2
    if not root:
        err.write("hpclab-sim: HPCLAB_SIM_ROOT is not set (use the shims in <sim_root>/bin or --root)\n")
        return 2
    state = SimState(Path(root))
    if not state.db_path.exists():
        err.write(f"hpclab-sim: no simulator at {root} (create it with SimCluster.ensure)\n")
        return 2
    command, rest = args[0], args[1:]
    handler = _handler(command)
    if handler is None:
        err.write(f"hpclab-sim: unknown command {command!r}\n")
        return 127
    ctx = Ctx(state=state, env=environ, cwd=cwd or os.getcwd(), out=out, err=err, stdin=stdin or sys.stdin)
    return handler(ctx, rest)


def _parse_assignments(words: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for word in words:
        key, eq, value = word.partition("=")
        if not eq or not key:
            raise ValueError(f"expected key=value, got {word!r}")
        out[key.strip()] = value
    return out


def admin_main(ctx: Ctx, argv: list[str]) -> int:
    """Fault/admin control plane of the simulator (JSON on stdout)."""
    from hpclab.sim.cluster import SimCluster

    if not argv:
        ctx.err.write("usage: _admin set-fault|clear-faults|faults|jobs|info|shutdown|set-node|wait-idle\n")
        return 2
    cluster = SimCluster(ctx.state.root)
    sub, rest = argv[0], argv[1:]
    try:
        if sub == "set-fault":
            faults = cluster.set_faults(**_parse_assignments(rest))
            _dump(ctx, faults.to_dict())
        elif sub == "clear-faults":
            cluster.clear_faults()
            _dump(ctx, cluster.faults().to_dict())
        elif sub == "faults":
            _dump(ctx, cluster.faults().to_dict())
        elif sub == "jobs":
            _dump(ctx, cluster.jobs())
        elif sub == "info":
            _dump(
                ctx,
                {"root": str(cluster.root), "bin": str(cluster.bin_dir), "config": cluster.config.to_dict()},
            )
        elif sub == "shutdown":
            _dump(ctx, {"killed": cluster.shutdown()})
        elif sub == "set-node":
            if len(rest) < 2:
                raise ValueError("usage: set-node <node> <up|drain|down> [reason...]")
            cluster.set_node_state(rest[0], rest[1], " ".join(rest[2:]) or None)
            _dump(ctx, {"node": rest[0], "state": rest[1]})
        elif sub == "wait-idle":
            timeout = float(rest[0]) if rest else 60.0
            idle = cluster.wait_idle(timeout)
            _dump(ctx, {"idle": idle})
            return 0 if idle else 1
        else:
            ctx.err.write(f"_admin: unknown subcommand {sub!r}\n")
            return 2
    except (ValueError, TypeError) as exc:
        ctx.err.write(f"_admin: error: {exc}\n")
        return 2
    return 0


def _dump(ctx: Ctx, value: object) -> None:
    ctx.out.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(main())
