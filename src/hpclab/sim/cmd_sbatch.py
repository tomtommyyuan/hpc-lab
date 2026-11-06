"""Fake ``sbatch``: option + ``#SBATCH`` parsing, controller-side validation with real error
texts, fault injection, job creation and runner spawn.

Real behaviors reproduced: CLI options override ``#SBATCH`` directives; directives stop at the
first command line; the script is copied at submit time (editing it later does not change the
queued job); relative ``--output`` paths resolve against the job's working directory; with
``--parsable`` stdout is ``<id>`` (``<id>;<cluster>`` in multi-cluster mode); ``--test-only``
prints ``sbatch: Job N to start at T using P processors on nodes X in partition Y`` to stderr
and queues nothing.

Deliberate deviations: time limits below one minute are honored to the second (real Slurm
rounds up to whole minutes; ``TimelimitRaw`` is still reported in rounded-up minutes) so TIMEOUT
paths can be tested quickly; ``--export=NONE`` keeps PATH/HOME/USER/PYTHONPATH so jobs can still
find the simulator; ``singleton`` dependencies are accepted and ignored.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace

from hpclab.sim.cmdutil import Ctx, Opt, Parsed, UsageError, flag, parse_args, split_list, sval
from hpclab.sim.config import PartitionConfig, SimConfig
from hpclab.sim.formats import compress_hostlist, expand_hostlist, parse_mem_mb, parse_time_limit, slurm_time
from hpclab.sim.scheduler import estimate_start, plan
from hpclab.sim.state import SimJob, SimState

PROG = "sbatch"

SBATCH_OPTS = [
    Opt("parsable", arg="none"),
    Opt("test-only", arg="none"),
    Opt("job-name", "J"),
    Opt("partition", "p"),
    Opt("account", "A"),
    Opt("qos", "q"),
    Opt("time", "t"),
    Opt("time-min"),
    Opt("nodes", "N"),
    Opt("ntasks", "n"),
    Opt("ntasks-per-node"),
    Opt("ntasks-per-gpu"),
    Opt("cpus-per-task", "c"),
    Opt("cpus-per-gpu"),
    Opt("mem"),
    Opt("mem-per-cpu"),
    Opt("mem-per-gpu"),
    Opt("gres"),
    Opt("gpus", "G"),
    Opt("gpus-per-node"),
    Opt("gpus-per-task"),
    Opt("gpu-bind"),
    Opt("output", "o"),
    Opt("error", "e"),
    Opt("input", "i"),
    Opt("chdir", "D"),
    Opt("open-mode"),
    Opt("comment"),
    Opt("dependency", "d"),
    Opt("exclude", "x"),
    Opt("nodelist", "w"),
    Opt("constraint", "C"),
    Opt("signal"),
    Opt("requeue", arg="none"),
    Opt("no-requeue", arg="none"),
    Opt("exclusive", arg="optional"),
    Opt("mail-type"),
    Opt("mail-user"),
    Opt("wrap"),
    Opt("export"),
    Opt("array", "a"),
    Opt("begin", "b"),
    Opt("hold", "H", arg="none"),
    Opt("nice", arg="optional"),
    Opt("reservation"),
    Opt("licenses", "L"),
    Opt("hint"),
    Opt("tmp"),
    Opt("switches"),
    Opt("distribution", "m"),
    Opt("wckey"),
    Opt("mincpus"),
    Opt("kill-on-invalid-dep"),
    Opt("propagate", arg="optional"),
    Opt("get-user-env", arg="optional"),
    Opt("quiet", "Q", arg="none"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
    Opt("usage", arg="none"),
]

FAILED = "Batch job submission failed:"
MSG_PARTITION = "Invalid partition name specified"
MSG_ACCOUNT = "Invalid account or account/partition combination specified"
MSG_TIME = "Requested time limit is invalid (missing or exceeds some limit)"
MSG_NODECONF = "Requested node configuration is not available"
MSG_GRES = "Invalid generic resource (gres) specification"
MSG_QOS = "Invalid qos specification"
MSG_QOS_POLICY = "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)"
MSG_DEPENDENCY = "Job dependency problem"
MSG_NODE_NAME = "Invalid node name specified"
MSG_FEATURE = "Invalid feature specification"
MSG_ARRAY = "Invalid job array specification"
MSG_SOCKET_TIMEOUT = "Socket timed out on send/recv operation"
MAX_ARRAY_TASKS = 1000


class Refusal(Exception):
    """Controller-side rejection: ``lines`` are printed as ``sbatch: error: <line>``."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[-1])
        self.lines = lines


@dataclass(frozen=True)
class GpuRequest:
    per_node: int
    gpu_type: str | None


def _read_script(ctx: Ctx, values: Parsed, positionals: list[str]) -> tuple[str, str, list[str]]:
    """Return (script text, display path, script args)."""
    wrap = sval(values, "wrap")
    if wrap is not None:
        if positionals:
            raise UsageError(f"{PROG}: error: Script arguments not permitted with --wrap option")
        return f"#!/bin/sh\n# This script was created by sbatch --wrap.\n\n{wrap}\n", "(wrap)", []
    if not positionals:
        text = ctx.stdin.read() if ctx.stdin is not None else ""
        if not text.strip():
            raise UsageError(f"{PROG}: error: Batch script is empty!")
        return text, "(stdin)", []
    path = positionals[0]
    full = path if os.path.isabs(path) else os.path.join(ctx.cwd, path)
    try:
        with open(full, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        raise UsageError(f"{PROG}: error: Unable to open file {path}") from None
    return text, full, positionals[1:]


def _directives(text: str) -> list[str]:
    """``#SBATCH`` arguments, stopping at the first command (real sbatch semantics)."""
    out: list[str] = []
    for line in text.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        if stripped.startswith("#SBATCH"):
            body = stripped[len("#SBATCH") :]
            if body and not body[0].isspace():
                continue
            body = re.split(r"\s+#", body, maxsplit=1)[0]
            try:
                out.extend(shlex.split(body))
            except ValueError:
                continue
    return out


def _gpu_request(values: Parsed, cfg: SimConfig, nodes: int, tasks_per_node: int) -> GpuRequest:
    def typed(spec: str) -> tuple[str | None, int]:
        parts = spec.split(":")
        try:
            if len(parts) == 1:
                return None, int(parts[0])
            if len(parts) == 2:
                return parts[0], int(parts[1])
        except ValueError:
            pass
        raise Refusal(f"{FAILED} {MSG_GRES}")

    gtype: str | None = None
    per_node = 0
    gres = sval(values, "gres")
    if gres and gres.lower() != "none":
        for item in gres.split(","):
            parts = [p for p in item.split(":") if p]
            name = parts[0].removeprefix("gres/") if parts else ""
            if name != "gpu":
                raise Refusal(f"{FAILED} {MSG_GRES}")
            try:
                if len(parts) == 1:
                    per_node += 1
                elif len(parts) == 2:
                    if parts[1].isdigit():
                        per_node += int(parts[1])
                    else:
                        gtype, per_node = parts[1], per_node + 1
                elif len(parts) == 3:
                    gtype, per_node = parts[1], per_node + int(parts[2])
                else:
                    raise ValueError(item)
            except ValueError:
                raise Refusal(f"{FAILED} {MSG_GRES}") from None
    if (spec := sval(values, "gpus-per-node")) is not None:
        gtype, per_node = typed(spec)
    if (spec := sval(values, "gpus-per-task")) is not None:
        gtype, n = typed(spec)
        per_node = n * tasks_per_node
    if (spec := sval(values, "gpus")) is not None:
        gtype, total = typed(spec)
        per_node = math.ceil(total / max(1, nodes))
    if gtype is not None and gtype not in cfg.gpu_types():
        raise Refusal(f"{FAILED} {MSG_GRES}")
    return GpuRequest(per_node=per_node, gpu_type=gtype)


def _int_opt(values: Parsed, key: str, default: int) -> int:
    raw = sval(values, key)
    if raw is None:
        return default
    first = raw.split("-", 1)[0]
    if not first.isdigit() or int(first) <= 0:
        raise UsageError(f'{PROG}: error: Invalid numeric value "{raw}" for --{key}.')
    return int(first)


def _array_tasks(spec: str) -> list[int]:
    body = spec.split("%", 1)[0]
    tasks: list[int] = []
    for part in body.split(","):
        rng, _, step_s = part.partition(":")
        step = int(step_s) if step_s.isdigit() else 1
        a, _, b = rng.partition("-")
        if not a.isdigit() or (b and not b.isdigit()) or step <= 0:
            raise Refusal(f"{FAILED} {MSG_ARRAY}")
        tasks.extend(range(int(a), (int(b) if b else int(a)) + 1, step))
    if not tasks or len(tasks) > MAX_ARRAY_TASKS:
        raise Refusal(f"{FAILED} {MSG_ARRAY}")
    return sorted(set(tasks))


@dataclass(frozen=True)
class Shape:
    """Per-node resource request derived from nodes/tasks/cpus/gpus/mem options."""

    nodes: int
    ntasks: int
    cpus_per_task: int | None
    cpus_per_node: int
    mem_per_node_mb: int
    gpus: GpuRequest


@dataclass(frozen=True)
class Placement:
    req_nodes: list[str]
    exclude: list[str]
    constraint: str | None


def _resolve_partition(cfg: SimConfig, values: Parsed) -> PartitionConfig:
    names = split_list(sval(values, "partition"))
    for name in names:
        if cfg.partition(name) is None:
            raise Refusal(f"invalid partition specified: {name}", f"{FAILED} {MSG_PARTITION}")
    partition = cfg.partition(names[0]) if names else cfg.default_partition()
    if partition is None:
        raise Refusal(f"{FAILED} No partition specified or system default partition")
    return partition


def _resolve_time_limit(cfg: SimConfig, partition: PartitionConfig, values: Parsed) -> int | None:
    raw = sval(values, "time")
    if raw is None:
        return partition.default_time_s or partition.max_time_s
    try:
        limit = parse_time_limit(raw)
    except ValueError:
        raise UsageError(f"{PROG}: error: Invalid --time specification") from None
    over = partition.max_time_s is not None and (limit is None or limit > partition.max_time_s)
    if over and cfg.enforce_part_limits:
        raise Refusal(f"{FAILED} {MSG_TIME}")
    return limit  # over the limit without EnforcePartLimits: pends with PartitionTimeLimit


def _resolve_shape(cfg: SimConfig, values: Parsed) -> Shape:
    nodes = _int_opt(values, "nodes", 1)
    if sval(values, "ntasks-per-node") is not None:
        tasks_per_node = _int_opt(values, "ntasks-per-node", 1)
        ntasks = tasks_per_node * nodes
    else:
        ntasks = _int_opt(values, "ntasks", nodes)
        tasks_per_node = math.ceil(ntasks / nodes)
    cpus_per_task = (
        _int_opt(values, "cpus-per-task", 1) if sval(values, "cpus-per-task") is not None else None
    )
    cpus_per_node = tasks_per_node * (cpus_per_task or 1)
    gpus = _gpu_request(values, cfg, nodes, tasks_per_node)
    if gpus.per_node and sval(values, "cpus-per-gpu") is not None:
        cpus_per_node = max(cpus_per_node, _int_opt(values, "cpus-per-gpu", 1) * gpus.per_node)
    if (mem := sval(values, "mem")) is not None:
        mem_mb = _parse_mem(mem)
    elif (per_cpu := sval(values, "mem-per-cpu")) is not None:
        mem_mb = _parse_mem(per_cpu) * cpus_per_node
    elif (per_gpu := sval(values, "mem-per-gpu")) is not None:
        mem_mb = _parse_mem(per_gpu) * max(1, gpus.per_node)
    else:
        mem_mb = cfg.def_mem_per_cpu_mb * cpus_per_node
    return Shape(nodes, ntasks, cpus_per_task, cpus_per_node, mem_mb, gpus)


def _check_placement(cfg: SimConfig, partition: PartitionConfig, values: Parsed, shape: Shape) -> Placement:
    """Refuse requests no node of the partition could ever satisfy (slurmctld's submit checks)."""
    req_nodes = expand_hostlist(sval(values, "nodelist") or "")
    exclude = expand_hostlist(sval(values, "exclude") or "")
    if any(cfg.node(name) is None for name in (*req_nodes, *exclude)):
        raise Refusal(f"{FAILED} {MSG_NODE_NAME}")
    constraint = sval(values, "constraint")
    if constraint and not any(constraint in n.features for n in cfg.nodes):
        raise Refusal(f"{FAILED} {MSG_FEATURE}")
    if any(n not in partition.nodes for n in req_nodes):
        raise Refusal(f"{FAILED} {MSG_NODECONF}")
    part_nodes = [n for n in (cfg.node(x) for x in partition.nodes) if n is not None]
    gpus = shape.gpus
    if gpus.gpu_type and gpus.per_node and not any(n.gpu_type == gpus.gpu_type for n in part_nodes):
        raise Refusal(f"{FAILED} {MSG_NODECONF}")
    fitting = [
        n
        for n in part_nodes
        if n.gpus >= gpus.per_node
        and n.cpus >= shape.cpus_per_node
        and (gpus.gpu_type is None or not gpus.per_node or n.gpu_type == gpus.gpu_type)
        and (not constraint or constraint in n.features)
        and n.name not in exclude
    ]
    if fitting and all(n.mem_mb < shape.mem_per_node_mb for n in fitting):
        raise Refusal("Memory specification can not be satisfied", f"{FAILED} {MSG_NODECONF}")
    if len(fitting) < max(shape.nodes, len(req_nodes)):
        raise Refusal(f"{FAILED} {MSG_NODECONF}")
    if partition.max_gpus_per_user is not None and gpus.per_node * shape.nodes > partition.max_gpus_per_user:
        raise Refusal(f"{FAILED} {MSG_QOS_POLICY}")  # DenyOnLimit semantics
    return Placement(req_nodes, exclude, constraint)


def _build_job(ctx: Ctx, values: Parsed, script_path: str, script_args: list[str], argv: list[str]) -> SimJob:
    """Validate the request like slurmctld would and build the job template (id assigned later)."""
    cfg = ctx.state.config
    partition = _resolve_partition(cfg, values)
    account = sval(values, "account") or cfg.default_account
    if account not in cfg.accounts:
        raise Refusal(f"{FAILED} {MSG_ACCOUNT}")
    qos = sval(values, "qos") or cfg.default_qos
    if qos not in cfg.qos:
        raise Refusal(f"{FAILED} {MSG_QOS}")
    time_limit = _resolve_time_limit(cfg, partition, values)
    shape = _resolve_shape(cfg, values)
    placement = _check_placement(cfg, partition, values, shape)
    dependency = sval(values, "dependency")
    if dependency:
        dependency = _validate_dependency(ctx, dependency.replace("?", ","))

    workdir_raw = sval(values, "chdir")
    workdir = os.path.normpath(os.path.join(ctx.cwd, workdir_raw)) if workdir_raw else ctx.cwd
    default_out = "slurm-%A_%a.out" if sval(values, "array") else "slurm-%j.out"
    err = sval(values, "error")
    open_mode = (sval(values, "open-mode") or "truncate").lower()
    if open_mode not in ("append", "truncate"):
        raise UsageError(f"{PROG}: error: Invalid --open-mode argument: {open_mode}")
    default_name = {"(wrap)": "wrap", "(stdin)": "sbatch"}.get(script_path, os.path.basename(script_path))
    return SimJob(
        job_id=0,
        name=sval(values, "job-name") or default_name,
        user=cfg.user,
        uid=os.getuid(),
        account=account,
        partition=partition.name,
        qos=qos,
        submit_time=ctx.now,
        time_limit_s=time_limit,
        nodes=shape.nodes,
        ntasks=shape.ntasks,
        cpus_per_task=shape.cpus_per_task,
        cpus_per_node=shape.cpus_per_node,
        mem_per_node_mb=shape.mem_per_node_mb,
        gpus_per_node=shape.gpus.per_node,
        gpu_type=shape.gpus.gpu_type,
        req_nodelist=placement.req_nodes,
        exclude=placement.exclude,
        constraint=placement.constraint,
        dependency=dependency,
        comment=sval(values, "comment"),
        submit_line=" ".join([PROG, *argv]),
        script_args=list(script_args),
        submit_dir=ctx.cwd,
        work_dir=workdir,
        stdout_pattern=os.path.join(workdir, sval(values, "output") or default_out),
        stderr_pattern=os.path.join(workdir, err) if err else None,
        open_mode=open_mode,
        env=_export_env(ctx, sval(values, "export")),
        requeue=flag(values, "requeue") and not flag(values, "no-requeue"),
        no_requeue=flag(values, "no-requeue"),
        hold=flag(values, "hold"),
    )


def _parse_mem(value: str) -> int:
    try:
        return parse_mem_mb(value)
    except ValueError:
        raise UsageError(f"{PROG}: error: Invalid --mem specification") from None


def _validate_dependency(ctx: Ctx, spec: str) -> str:
    jobs = ctx.state.jobs()
    clauses: list[str] = []
    for clause in spec.split(","):
        clause = clause.strip()
        if clause == "singleton":
            continue
        kind, sep, ids = clause.partition(":")
        if not sep or kind not in ("after", "afterany", "afterok", "afternotok", "aftercorr"):
            raise Refusal(f"{FAILED} {MSG_DEPENDENCY}")
        for token in ids.split(":"):
            token = token.split("+", 1)[0]
            if not any(j.matches(token) for j in jobs):
                raise Refusal(f"{FAILED} {MSG_DEPENDENCY}")
        clauses.append(f"{'afterok' if kind == 'aftercorr' else kind}:{ids}")
    return ",".join(clauses)


def _export_env(ctx: Ctx, export: str | None) -> dict[str, str]:
    base = dict(ctx.env)
    if export is None or export.upper().startswith("ALL"):
        env = base
        extra = export.split(",")[1:] if export else []
    elif export.upper().startswith("NONE"):
        env = {
            k: v for k, v in base.items() if k in ("HPCLAB_SIM_ROOT", "PATH", "HOME", "USER", "PYTHONPATH")
        }
        extra = export.split(",")[1:]
    else:
        env = dict(base)
        extra = export.split(",")
    for item in extra:
        key, eq, value = item.partition("=")
        if eq:
            env[key] = value
    env.pop("HPCLAB_SIM_CALL_ID", None)
    return env


def _spool_script(state: SimState, job_id: int, text: str) -> str:
    spool = state.root / "spool" / f"job{job_id:05d}"
    spool.mkdir(parents=True, exist_ok=True)
    path = spool / "slurm_script"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def spawn_runner(state: SimState, job_id: int) -> int:
    """Start the detached supervisor for ``job_id``; returns its pid.

    The runner must not inherit our stdout/stderr: callers capture sbatch's output through
    pipes and would otherwise block until the job ends.
    """
    logs = state.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "HPCLAB_SIM_CALL_ID"}
    with open(logs / f"runner-{job_id}.log", "ab") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hpclab.sim.jobrunner", "--root", str(state.root), str(job_id)],
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            cwd=str(state.root),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    return proc.pid


def _test_only(ctx: Ctx, template: SimJob) -> int:
    cfg = ctx.state.config
    with ctx.state.reader() as conn:
        jobs = ctx.state.load_jobs(conn)
        next_id = ctx.state.peek_job_id(conn)
        admin = ctx.state.node_admin(conn)
    probe = replace(template, job_id=next_id, state="PENDING")
    decisions = plan(cfg, [*jobs, probe], admin, now=ctx.now)
    mine = next((d for d in decisions if d.job_id == next_id), None)
    if mine is not None and mine.alloc is not None:
        start, nodes = ctx.now, list(mine.alloc)
    else:
        start = estimate_start(cfg, probe, jobs, ctx.now) or ctx.now
        part = cfg.partition(template.partition)
        nodes = list(part.nodes[: template.nodes]) if part else []
    ctx.err.write(
        f"{PROG}: Job {next_id} to start at {slurm_time(start, ctx.env)} using {template.total_cpus} "
        f"processors on nodes {compress_hostlist(nodes)} in partition {template.partition}\n"
    )
    return 0


def _print_refusal(ctx: Ctx, lines: tuple[str, ...]) -> int:
    for line in lines:
        ctx.err.write(f"{PROG}: error: {line}\n")
    return 1


def main(ctx: Ctx, argv: list[str]) -> int:
    try:
        cli, positionals = parse_args(PROG, argv, SBATCH_OPTS, stop_at_positional=True)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    if flag(cli, "version"):
        ctx.out.write(f"slurm {ctx.state.config.slurm_version}\n")
        return 0
    if flag(cli, "help") or flag(cli, "usage"):
        ctx.out.write("Usage: sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]\n")
        return 0
    try:
        text, script_path, script_args = _read_script(ctx, cli, positionals)
        if not text.startswith("#!"):
            raise UsageError(
                f"{PROG}: error: This does not look like a batch script.  The first\n"
                f"{PROG}: error: line must start with #! followed by the path to an interpreter.\n"
                f"{PROG}: error: For instance: #!/bin/sh"
            )
        if "\r\n" in text:
            raise UsageError(f"{PROG}: error: Batch script contains DOS line breaks (\\r\\n)")
        directives, _ = parse_args(PROG, _directives(text), SBATCH_OPTS)
        values: Parsed = {**directives, **cli}
        faults = ctx.state.faults()
        if faults.sbatch_reject:
            lines = tuple(
                ln.removeprefix(f"{PROG}: error: ") for ln in faults.sbatch_reject.splitlines() if ln.strip()
            )
            return _print_refusal(ctx, lines or (f"{FAILED} Unspecified error",))
        template = _build_job(ctx, values, script_path, script_args, argv)
        array = sval(values, "array")
        tasks = _array_tasks(array) if array else None
        if ctx.state.config.max_submit_jobs_per_user is not None:
            _check_submit_limit(ctx, len(tasks or [0]))
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    except Refusal as exc:
        return _print_refusal(ctx, exc.lines)
    if flag(values, "test-only"):
        return _test_only(ctx, template)
    return _submit(ctx, values, template, text, tasks)


def _check_submit_limit(ctx: Ctx, n_new: int) -> None:
    limit = ctx.state.config.max_submit_jobs_per_user
    active = [j for j in ctx.state.jobs() if not j.terminal and j.user == ctx.state.config.user]
    if limit is not None and len(active) + n_new > limit:
        raise Refusal("QOSMaxSubmitJobPerUserLimit", f"{FAILED} {MSG_QOS_POLICY}")


def _submit(ctx: Ctx, values: Parsed, template: SimJob, text: str, tasks: list[int] | None) -> int:
    state = ctx.state
    created: list[SimJob] = []
    ambiguous = lost = False
    with state.tx() as conn:
        if state.consume(conn, "sbatch_ambiguous_error"):
            ambiguous = True
            if not state.faults(conn).sbatch_ambiguous_creates_job:
                # The request died before the controller queued anything; the counter stays consumed.
                return _print_refusal(ctx, (f"{FAILED} {MSG_SOCKET_TIMEOUT}",))
        # Only a call made through SimTransport (which carries a call id) can lose its reply.
        lost = bool(ctx.call_id) and state.consume(conn, "sbatch_lose_reply")
        array_job_id: int | None = None
        for task in tasks if tasks is not None else [None]:
            job_id = state.allocate_job_id(conn)
            if tasks is not None and array_job_id is None:
                array_job_id = job_id
            fate = None
            for key, name in (
                ("node_fail_next", "node_fail"),
                ("preempt_next", "preempt"),
                ("oom_next", "oom"),
                ("requeue_next", "requeue"),
            ):
                if state.consume(conn, key):
                    fate = name
                    break
            job = replace(
                template,
                job_id=job_id,
                array_job_id=array_job_id,
                array_task_id=task,
                script_path=_spool_script(state, job_id, text),
                fate=fate,
            )
            state.put_job(conn, job)
            created.append(job)
        if lost and ctx.call_id:
            state.add_lost_reply(conn, ctx.call_id, created[0].job_id)
    pids = {job.job_id: spawn_runner(state, job.job_id) for job in created}
    with state.tx() as conn:
        for job_id, pid in pids.items():
            current = state.load_job(conn, job_id)
            if current is not None:
                current.runner_pid = pid
                state.put_job(conn, current)
    if ambiguous:
        return _print_refusal(ctx, (f"{FAILED} {MSG_SOCKET_TIMEOUT}",))
    shown = str(created[0].array_job_id or created[0].job_id)
    if flag(values, "parsable"):
        suffix = f";{state.config.cluster_name}" if state.config.parsable_cluster_suffix else ""
        ctx.out.write(f"{shown}{suffix}\n")
    else:
        ctx.out.write(f"Submitted batch job {shown}\n")
    return 0
