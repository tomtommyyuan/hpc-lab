"""Fake ``squeue``.

Emulated: ``-h``, ``-j/--jobs`` (all states shown for explicit ids), ``-u/--user``, ``--me``,
``-t/--states`` (``all`` included), ``-p``, ``-A``, ``-n``, ``-r/--array``, ``--start``,
``-O/--Format`` (width/justify/suffix semantics), ``-o/--format``, default format.
Finished jobs stay visible for ``MinJobAge`` (``min_job_age_s``). ``PrivateData=jobs`` hides
other users' jobs. A single unknown ``--jobs`` id
fails with ``slurm_load_jobs error: Invalid job id specified`` (exit 1), several ids just filter.
Deviation: pending array tasks are always listed one per line (as with ``--array``).
"""

from __future__ import annotations

from collections.abc import Callable

from hpclab.sim.cmdutil import Ctx, Opt, UsageError, flag, parse_args, split_list, sval
from hpclab.sim.config import SimConfig
from hpclab.sim.formats import (
    FieldSpec,
    compress_hostlist,
    format_mem,
    parse_long_format,
    parse_short_format,
    render_value,
    secs2time_str,
    slurm_time,
)
from hpclab.sim.scheduler import estimate_start
from hpclab.sim.state import STATE_COMPACT, SimJob

PROG = "squeue"
SQUEUE_OPTS = [
    Opt("noheader", "h", arg="none"),
    Opt("jobs", "j", arg="optional"),
    Opt("user", "u"),
    Opt("me", arg="none"),
    Opt("states", "t"),
    Opt("partition", "p"),
    Opt("account", "A"),
    Opt("name", "n"),
    Opt("qos", "q"),
    Opt("array", "r", arg="none"),
    Opt("start", arg="none"),
    Opt("Format", "O"),
    Opt("format", "o"),
    Opt("sort", "S"),
    Opt("long", "l", arg="none"),
    Opt("all", "a", arg="none"),
    Opt("steps", "s", arg="optional"),
    Opt("nodelist", "w"),
    Opt("clusters", "M"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", arg="none"),
    Opt("usage", arg="none"),
]
DEFAULT_FORMAT = "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
LONG_FORMAT = "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
START_FORMAT = "%.18i %.9P %.8j %.8u %.2t %.19S %.6D %20Y %R"
DEFAULT_STATES = frozenset({"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"})
LETTERS = {
    "i": "jobarrayid",
    "A": "jobid",
    "F": "arrayjobid",
    "K": "arraytaskid",
    "T": "state",
    "t": "statecompact",
    "j": "name",
    "u": "username",
    "U": "userid",
    "a": "account",
    "P": "partition",
    "q": "qos",
    "V": "submittime",
    "S": "starttime",
    "e": "endtime",
    "l": "timelimit",
    "M": "timeused",
    "L": "timeleft",
    "N": "nodelist",
    "r": "reason",
    "R": "reasonlist",
    "k": "comment",
    "Z": "workdir",
    "D": "numnodes",
    "C": "numcpus",
    "b": "tres-per-node",
    "Q": "priority",
    "E": "dependency",
    "m": "minmemory",
    "o": "command",
    "Y": "schednodes",
    "y": "nice",
}
HEADERS = {
    "jobarrayid": "JOBID",
    "jobid": "JOBID",
    "statecompact": "ST",
    "username": "USER",
    "timeused": "TIME",
    "numnodes": "NODES",
    "reasonlist": "NODELIST(REASON)",
    "numcpus": "CPUS",
    "tres-alloc": "TRES_ALLOC",
    "restartcnt": "RESTARTCNT",
    "schednodes": "SCHEDNODES",
}
USAGE = "Usage: squeue [-A account] [--clusters names] [-i seconds] [--job jobid] [-n name] [-o format]\n"


def job_gpu_type(job: SimJob, cfg: SimConfig) -> str | None:
    """Requested GPU type, or the type of the allocated node for untyped requests."""
    if job.gpu_type or not job.node_list:
        return job.gpu_type
    node = cfg.node(job.node_list[0])
    return node.gpu_type if node else None


def tres_items(job: SimJob, cfg: SimConfig, *, alloc: bool) -> list[tuple[str, str]]:
    """TRES of a job: allocated once it started, requested before (squeue ``tres-alloc``)."""
    items = [
        ("cpu", str(job.total_cpus)),
        ("mem", format_mem(job.mem_per_node_mb * job.nodes)),
        ("node", str(job.nodes)),
        ("billing", str(job.total_cpus)),
    ]
    if job.total_gpus and cfg.track_gpu_tres:
        items.append(("gres/gpu", str(job.total_gpus)))
        gtype = job_gpu_type(job, cfg) if alloc else job.gpu_type
        if gtype:
            items.append((f"gres/gpu:{gtype}", str(job.total_gpus)))
    return items


def tres_string(job: SimJob, cfg: SimConfig, *, alloc: bool) -> str:
    return ",".join(f"{k}={v}" for k, v in tres_items(job, cfg, alloc=alloc))


class View:
    """Per-invocation derived values (now, start estimates)."""

    def __init__(self, ctx: Ctx, jobs: list[SimJob], *, start_mode: bool) -> None:
        self.ctx = ctx
        self.jobs = jobs
        self.start_mode = start_mode

    def time(self, epoch: float | None, missing: str = "N/A") -> str:
        return slurm_time(epoch, self.ctx.env, missing=missing)

    def used(self, job: SimJob) -> float:
        if job.start_time is None:
            return 0.0
        end = job.end_time if job.end_time is not None else self.ctx.now
        return max(0.0, end - job.start_time)

    def value(self, name: str, job: SimJob) -> str:
        getter = FIELDS.get(name.lower())
        if getter is None:
            raise UsageError(f"Invalid job format specification: {name}")
        return getter(self, job)


def _start(v: View, j: SimJob) -> str:
    if j.state == "PENDING":
        return v.time(estimate_start(v.ctx.state.config, j, v.jobs, v.ctx.now)) if v.start_mode else "N/A"
    return v.time(j.start_time)


def _end(v: View, j: SimJob) -> str:
    if j.terminal:
        return v.time(j.end_time)
    if j.state == "RUNNING" and j.start_time is not None and j.time_limit_s is not None:
        return v.time(j.start_time + j.time_limit_s)
    return "N/A"


def _reason(j: SimJob) -> str:
    if j.state == "RUNNING":
        return "None"
    return j.reason or "None"


FIELDS: dict[str, Callable[[View, SimJob], str]] = {
    "jobarrayid": lambda v, j: j.display_id,
    "jobid": lambda v, j: str(j.job_id),
    "arrayjobid": lambda v, j: str(j.array_job_id or j.job_id),
    "arraytaskid": lambda v, j: str(j.array_task_id) if j.array_task_id is not None else "N/A",
    "state": lambda v, j: j.state,
    "statecompact": lambda v, j: STATE_COMPACT.get(j.state, j.state[:2]),
    "name": lambda v, j: j.name,
    "username": lambda v, j: j.user,
    "userid": lambda v, j: str(j.uid),
    "account": lambda v, j: j.account,
    "partition": lambda v, j: j.partition,
    "qos": lambda v, j: j.qos,
    "submittime": lambda v, j: v.time(j.submit_time),
    "starttime": _start,
    "endtime": _end,
    "timelimit": lambda v, j: secs2time_str(j.time_limit_s),
    "timeused": lambda v, j: secs2time_str(v.used(j)),
    "timeleft": lambda v, j: (
        secs2time_str(max(0.0, j.time_limit_s - v.used(j))) if j.time_limit_s is not None else "UNLIMITED"
    ),
    "nodelist": lambda v, j: compress_hostlist(j.node_list) if j.state != "PENDING" else "",
    "reason": lambda v, j: _reason(j),
    "reasonlist": lambda v, j: f"({_reason(j)})" if j.state == "PENDING" else compress_hostlist(j.node_list),
    "restartcnt": lambda v, j: str(j.restart_count),
    "tres-alloc": lambda v, j: tres_string(j, v.ctx.state.config, alloc=j.state != "PENDING"),
    "tres-per-node": lambda v, j: (
        f"gres/gpu:{j.gpu_type}:{j.gpus_per_node}"
        if j.gpus_per_node and j.gpu_type
        else (f"gres/gpu:{j.gpus_per_node}" if j.gpus_per_node else "N/A")
    ),
    "workdir": lambda v, j: j.work_dir,
    "comment": lambda v, j: j.comment if j.comment is not None else "(null)",
    "numnodes": lambda v, j: str(j.nodes),
    "numcpus": lambda v, j: str(j.total_cpus),
    "priority": lambda v, j: str(max(1, 100000 - j.job_id)),
    "dependency": lambda v, j: j.dependency or "(null)",
    "minmemory": lambda v, j: format_mem(j.mem_per_node_mb),
    "command": lambda v, j: j.script_path,
    "schednodes": lambda v, j: "(null)",
    "nice": lambda v, j: "0",
    "stdout": lambda v, j: j.stdout_path or j.stdout_pattern,
    "stderr": lambda v, j: j.stderr_path or (j.stderr_pattern or j.stdout_pattern),
}


def _render(view: View, specs: list[FieldSpec | str], job: SimJob) -> str:
    out: list[str] = []
    for spec in specs:
        if isinstance(spec, str):
            out.append(spec)
        else:
            out.append(render_value(view.value(spec.name, job), spec))
    return "".join(out)


def _header(specs: list[FieldSpec | str]) -> str:
    out: list[str] = []
    for spec in specs:
        if isinstance(spec, str):
            out.append(spec)
            continue
        key = spec.name.lower()
        title = HEADERS.get(key, key.upper())
        out.append(render_value(title, spec))
    return "".join(out)


def _short_specs(fmt: str) -> list[FieldSpec | str]:
    specs: list[FieldSpec | str] = []
    for item in parse_short_format(fmt):
        if isinstance(item, FieldSpec):
            name = LETTERS.get(item.name)
            if name is None:
                continue
            specs.append(FieldSpec(name=name, width=item.width, right=item.right, suffix=""))
        else:
            specs.append(item)
    return specs


def _state_filter(values: dict[str, str | bool]) -> frozenset[str] | None:
    raw = sval(values, "states")
    if raw is None:
        return None
    wanted: set[str] = set()
    long_by_short = {v: k for k, v in STATE_COMPACT.items()}
    for token in split_list(raw.upper()):
        if token == "ALL":
            return None
        wanted.add(long_by_short.get(token, token))
    return frozenset(wanted)


def main(ctx: Ctx, argv: list[str]) -> int:
    try:
        values, positionals = parse_args(PROG, argv, SQUEUE_OPTS)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    if flag(values, "version"):
        ctx.out.write(f"slurm {ctx.state.config.slurm_version}\n")
        return 0
    if flag(values, "help") or flag(values, "usage"):
        ctx.out.write(USAGE)
        return 0
    if ctx.state.faults().squeue_unavailable:
        ctx.err.write("slurm_load_jobs error: Unable to contact slurm controller (connect failure)\n")
        return 1
    ctx.state.reap(ctx.now)
    all_jobs = ctx.state.jobs()
    faults = ctx.state.faults()
    age = ctx.state.min_job_age_s(faults)
    in_memory = [j for j in all_jobs if not j.terminal or (j.end_time or 0) + age > ctx.now]
    if ctx.state.config.hides_other_jobs():
        in_memory = [j for j in in_memory if j.user == ctx.state.config.user]

    job_tokens = split_list(sval(values, "jobs")) + [p for p in positionals if p[:1].isdigit()]
    jobs = in_memory
    if job_tokens:
        jobs = [j for j in jobs if any(j.matches(t) for t in job_tokens)]
        if len(job_tokens) == 1 and not jobs:
            ctx.err.write("slurm_load_jobs error: Invalid job id specified\n")
            return 1
    users = split_list(sval(values, "user"))
    if flag(values, "me"):
        users.append(ctx.state.config.user)
    if users:
        jobs = [j for j in jobs if j.user in users or str(j.uid) in users]
    if parts := split_list(sval(values, "partition")):
        jobs = [j for j in jobs if j.partition in parts]
    if accounts := split_list(sval(values, "account")):
        jobs = [j for j in jobs if j.account in accounts]
    if names := split_list(sval(values, "name")):
        jobs = [j for j in jobs if j.name in names]
    start_mode = flag(values, "start")
    states = _state_filter(values)
    if start_mode:
        states = frozenset({"PENDING"})
    elif states is None and sval(values, "states") is None and not job_tokens:
        states = DEFAULT_STATES
    if states is not None:
        jobs = [j for j in jobs if j.state in states]

    view = View(ctx, all_jobs, start_mode=start_mode)
    specs: list[FieldSpec | str]
    try:
        if (long_fmt := sval(values, "Format")) is not None:
            specs = list(parse_long_format(long_fmt))
            for spec in specs:
                if isinstance(spec, FieldSpec) and spec.name.lower() not in FIELDS:
                    raise UsageError(f"Invalid job format specification: {spec.name}")
        else:
            fmt = sval(values, "format") or (
                START_FORMAT if start_mode else LONG_FORMAT if flag(values, "long") else DEFAULT_FORMAT
            )
            specs = _short_specs(fmt)
        lines = [_render(view, specs, j) for j in _sorted(view, jobs, start_mode)]
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return 1
    if not flag(values, "noheader"):
        ctx.out.write(_header(specs) + "\n")
    for line in lines:
        ctx.out.write(line + "\n")
    return 0


def _sorted(view: View, jobs: list[SimJob], start_mode: bool) -> list[SimJob]:
    if start_mode:
        cfg = view.ctx.state.config
        return sorted(jobs, key=lambda j: (estimate_start(cfg, j, view.jobs, view.ctx.now) or 0, j.job_id))
    return sorted(jobs, key=lambda j: j.job_id)
