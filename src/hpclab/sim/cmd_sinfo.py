"""Fake ``sinfo`` and ``scontrol`` (read-only ``show`` subcommands).

Node state follows Slurm: ``idle``/``mixed``/``allocated`` from CPU allocation, admin
``drain`` -> ``drained`` (``draining`` while jobs still run), ``down``. ``Gres`` prints like
``gpu:a100:4(S:0-1)`` and ``GresUsed`` like ``gpu:a100:2(IDX:0-1)`` / ``gpu:a100:0(IDX:N/A)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from hpclab.sim.cmd_squeue import tres_string
from hpclab.sim.cmdutil import Ctx, Opt, UsageError, flag, parse_args, split_list, sval
from hpclab.sim.config import NodeConfig, PartitionConfig, SimConfig
from hpclab.sim.formats import (
    FieldSpec,
    compress_hostlist,
    compress_ranges,
    elapsed_str,
    format_mem,
    parse_long_format,
    parse_short_format,
    render_value,
    secs2time_str,
    slurm_time,
)
from hpclab.sim.scheduler import node_usage
from hpclab.sim.state import SimJob

SINFO_OPTS = [
    Opt("noheader", "h", arg="none"),
    Opt("Node", "N", arg="none"),
    Opt("partition", "p"),
    Opt("nodes", "n"),
    Opt("Format", "O"),
    Opt("format", "o"),
    Opt("states", "t"),
    Opt("long", "l", arg="none"),
    Opt("summarize", "s", arg="none"),
    Opt("all", "a", arg="none"),
    Opt("list-reasons", "R", arg="none"),
    Opt("clusters", "M"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", arg="none"),
    Opt("usage", arg="none"),
]
SHORT = {
    "P": "partition",
    "R": "partitionname",
    "a": "available",
    "l": "time",
    "L": "defaulttime",
    "D": "nodes",
    "t": "statecompact",
    "T": "statelong",
    "N": "nodelist",
    "n": "nodehost",
    "G": "gres",
    "c": "cpus",
    "C": "cpusstate",
    "m": "memory",
    "e": "freemem",
    "E": "reason",
    "f": "features",
    "b": "features_act",
}
DEFAULT_FMT = "%9P %5a %10l %6D %6t %N"
NODE_FMT = "%12N %6D %9P %6t"
COMPACT = {
    "idle": "idle",
    "mixed": "mix",
    "allocated": "alloc",
    "drained": "drain",
    "draining": "drng",
    "down": "down",
}


@dataclass
class NodeStatus:
    node: NodeConfig
    state_long: str
    reason: str | None
    alloc_cpus: int
    alloc_mem: int
    gpus_used: list[int]


def node_statuses(ctx: Ctx, jobs: list[SimJob] | None = None) -> dict[str, NodeStatus]:
    cfg = ctx.state.config
    use = node_usage(jobs if jobs is not None else ctx.state.jobs())
    admin = ctx.state.node_admin()
    out: dict[str, NodeStatus] = {}
    for node in cfg.nodes:
        u = use.get(node.name)
        cpus, mem, gpus = (u.cpus, u.mem_mb, sorted(u.gpus)) if u else (0, 0, [])
        busy = bool(cpus or gpus)
        adm, reason = admin.get(node.name, ("up", None))
        if adm == "down":
            state = "down"
        elif adm == "drain":
            state = "draining" if busy else "drained"
        elif cpus >= node.cpus:
            state = "allocated"
        elif busy:
            state = "mixed"
        else:
            state = "idle"
        out[node.name] = NodeStatus(node, state, reason, cpus, mem, gpus)
    return out


def gres_used(ns: NodeStatus) -> str:
    node = ns.node
    if not node.gpus:
        return "(null)"
    idx = compress_ranges(ns.gpus_used) if ns.gpus_used else "N/A"
    typed = f"gpu:{node.gpu_type}:{len(ns.gpus_used)}" if node.gpu_type else f"gpu:{len(ns.gpus_used)}"
    return f"{typed}(IDX:{idx})"


def _time_limit(p: PartitionConfig) -> str:
    return "infinite" if p.max_time_s is None else secs2time_str(p.max_time_s)


def _cpus_state(ns: NodeStatus) -> str:
    total = ns.node.cpus
    alloc = ns.alloc_cpus
    other = total - alloc if ns.state_long in ("down", "drained", "draining") else 0
    idle = total - alloc - other
    return f"{alloc}/{idle}/{other}/{total}"


Getter = Callable[[PartitionConfig, list[NodeStatus]], str]


def _getters(cfg: SimConfig) -> dict[str, Getter]:
    def first(nodes: list[NodeStatus]) -> NodeStatus:
        return nodes[0]

    return {
        "nodelist": lambda p, ns: compress_hostlist([n.node.name for n in ns]),
        "nodehost": lambda p, ns: compress_hostlist([n.node.name for n in ns]),
        "nodeaddr": lambda p, ns: compress_hostlist([n.node.name for n in ns]),
        "partition": lambda p, ns: p.name + ("*" if p.default else ""),
        "partitionname": lambda p, ns: p.name,
        "available": lambda p, ns: p.state.lower(),
        "time": lambda p, ns: _time_limit(p),
        "defaulttime": lambda p, ns: secs2time_str(p.default_time_s) if p.default_time_s else "n/a",
        "nodes": lambda p, ns: str(len(ns)),
        "statelong": lambda p, ns: first(ns).state_long,
        "statecompact": lambda p, ns: COMPACT.get(first(ns).state_long, first(ns).state_long),
        "state": lambda p, ns: COMPACT.get(first(ns).state_long, first(ns).state_long),
        "gres": lambda p, ns: first(ns).node.gres(),
        "gresused": lambda p, ns: gres_used(first(ns)),
        "cpus": lambda p, ns: str(first(ns).node.cpus),
        "cpusstate": lambda p, ns: _cpus_state(first(ns)),
        "memory": lambda p, ns: str(first(ns).node.mem_mb),
        "allocmem": lambda p, ns: str(first(ns).alloc_mem),
        "freemem": lambda p, ns: str(first(ns).node.mem_mb - first(ns).alloc_mem),
        "features": lambda p, ns: ",".join(first(ns).node.features) or "(null)",
        "features_act": lambda p, ns: ",".join(first(ns).node.features) or "(null)",
        "reason": lambda p, ns: first(ns).reason or "none",
        "preemptmode": lambda p, ns: p.effective_preempt_mode(),
        "prioritytier": lambda p, ns: str(p.priority_tier),
    }


def sinfo_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "sinfo"
    try:
        values, _ = parse_args(prog, argv, SINFO_OPTS)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    cfg = ctx.state.config
    if flag(values, "version"):
        ctx.out.write(f"slurm {cfg.slurm_version}\n")
        return 0
    if flag(values, "help") or flag(values, "usage"):
        ctx.out.write("Usage: sinfo [OPTIONS]\n")
        return 0
    ctx.state.reap(ctx.now)
    statuses = node_statuses(ctx)
    getters = _getters(cfg)
    per_node = flag(values, "Node")
    try:
        specs = _specs(values, per_node, getters)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return 1
    # Unknown partitions simply match nothing (real sinfo prints no rows and exits 0).
    wanted_parts = split_list(sval(values, "partition"))
    wanted_nodes = set(split_list(sval(values, "nodes")))
    groups: list[tuple[PartitionConfig, list[NodeStatus]]] = []
    for part in cfg.partitions:
        if wanted_parts and part.name not in wanted_parts:
            continue
        members = [
            statuses[n] for n in part.nodes if n in statuses and (not wanted_nodes or n in wanted_nodes)
        ]
        if per_node:
            groups.extend((part, [m]) for m in members)
        else:
            by_state: dict[str, list[NodeStatus]] = {}
            for m in members:
                by_state.setdefault(m.state_long, []).append(m)
            groups.extend((part, ms) for ms in by_state.values())
    if not flag(values, "noheader"):
        ctx.out.write(_header(specs) + "\n")
    for part, members in groups:
        line = []
        for spec in specs:
            if isinstance(spec, str):
                line.append(spec)
            else:
                line.append(render_value(getters[spec.name.lower()](part, members), spec))
        ctx.out.write("".join(line) + "\n")
    return 0


def _specs(
    values: dict[str, str | bool], per_node: bool, getters: dict[str, Getter]
) -> list[FieldSpec | str]:
    long_fmt = sval(values, "Format")
    if long_fmt is not None:
        specs: list[FieldSpec | str] = []
        for spec in parse_long_format(long_fmt):
            if spec.name.lower() not in getters:
                raise UsageError(f"Invalid node format specification: {spec.name}")
            specs.append(spec)
        return specs
    out: list[FieldSpec | str] = []
    for item in parse_short_format(sval(values, "format") or (NODE_FMT if per_node else DEFAULT_FMT)):
        if isinstance(item, FieldSpec):
            name = SHORT.get(item.name)
            if name is not None:
                out.append(FieldSpec(name=name, width=item.width, right=item.right, suffix=""))
        else:
            out.append(item)
    return out


def _header(specs: list[FieldSpec | str]) -> str:
    titles = {
        "partitionname": "PARTITION",
        "statecompact": "STATE",
        "statelong": "STATE",
        "time": "TIMELIMIT",
    }
    out = []
    for spec in specs:
        if isinstance(spec, str):
            out.append(spec)
        else:
            key = spec.name.lower()
            out.append(render_value(titles.get(key, key.upper()), spec))
    return "".join(out)


# ---------------------------------------------------------------------------
# scontrol
# ---------------------------------------------------------------------------


def _partition_fields(cfg: SimConfig, p: PartitionConfig) -> list[tuple[str, str]]:
    nodes = [n for n in (cfg.node(x) for x in p.nodes) if n is not None]
    cpus = sum(n.cpus for n in nodes)
    mem = sum(n.mem_mb for n in nodes)
    gpus = sum(n.gpus for n in nodes)
    tres = f"cpu={cpus},mem={format_mem(mem)},node={len(nodes)},billing={cpus}" + (
        f",gres/gpu={gpus}" if gpus and cfg.track_gpu_tres else ""
    )
    return [
        ("PartitionName", p.name),
        ("AllowGroups", "ALL"),
        ("AllowAccounts", "ALL"),
        ("AllowQos", "ALL"),
        ("AllocNodes", "ALL"),
        ("Default", "YES" if p.default else "NO"),
        ("QoS", "N/A"),
        ("DefaultTime", elapsed_str(p.default_time_s) if p.default_time_s else "NONE"),
        ("DisableRootJobs", "NO"),
        ("ExclusiveUser", "NO"),
        ("GraceTime", str(p.grace_time_s)),
        ("Hidden", "NO"),
        ("MaxNodes", "UNLIMITED"),
        ("MaxTime", elapsed_str(p.max_time_s) if p.max_time_s is not None else "UNLIMITED"),
        ("MinNodes", "0"),
        ("LLN", "NO"),
        ("MaxCPUsPerNode", "UNLIMITED"),
        ("Nodes", compress_hostlist(p.nodes)),
        ("PriorityJobFactor", "1"),
        ("PriorityTier", str(p.priority_tier)),
        ("RootOnly", "NO"),
        ("ReqResv", "NO"),
        ("OverSubscribe", "NO"),
        ("OverTimeLimit", "NONE"),
        ("PreemptMode", p.effective_preempt_mode()),
        ("State", p.state.upper()),
        ("TotalCPUs", str(cpus)),
        ("TotalNodes", str(len(nodes))),
        ("SelectTypeParameters", "NONE"),
        ("JobDefaults", "(null)"),
        ("DefMemPerCPU", str(cfg.def_mem_per_cpu_mb)),
        ("MaxMemPerNode", "UNLIMITED"),
        ("TRES", tres),
    ]


def _job_fields(ctx: Ctx, job: SimJob) -> list[tuple[str, str]]:
    cfg = ctx.state.config

    def t(v: float | None) -> str:
        return slurm_time(v, ctx.env, missing="Unknown")

    run = 0.0
    if job.start_time is not None:
        run = (job.end_time or ctx.now) - job.start_time
    return [
        ("JobId", str(job.job_id)),
        ("JobName", job.name),
        ("UserId", f"{job.user}({job.uid})"),
        ("Account", job.account),
        ("QOS", job.qos),
        ("JobState", job.state),
        ("Reason", "None" if job.state == "RUNNING" else (job.reason or "None")),
        ("Dependency", job.dependency or "(null)"),
        ("Requeue", "1" if job.requeue else "0"),
        ("Restarts", str(job.restart_count)),
        ("BatchFlag", "1"),
        ("ExitCode", f"{job.exit_code or 0}:{job.exit_signal or 0}"),
        ("RunTime", elapsed_str(run)),
        ("TimeLimit", elapsed_str(job.time_limit_s)),
        ("SubmitTime", t(job.submit_time)),
        ("StartTime", t(job.start_time)),
        ("EndTime", t(job.end_time)),
        ("Partition", job.partition),
        ("NodeList", compress_hostlist(job.node_list) if job.node_list else "(null)"),
        ("NumNodes", str(job.nodes)),
        ("NumCPUs", str(job.total_cpus)),
        ("NumTasks", str(job.ntasks)),
        ("CPUs/Task", str(job.cpus_per_task or 1)),
        ("TRES", tres_string(job, cfg, alloc=job.state != "PENDING")),
        ("Command", job.script_path),
        ("WorkDir", job.work_dir),
        ("Comment", job.comment or "(null)"),
        ("StdErr", job.stderr_path or job.stderr_pattern or job.stdout_pattern),
        ("StdIn", "/dev/null"),
        ("StdOut", job.stdout_path or job.stdout_pattern),
    ]


def _node_fields(ns: NodeStatus) -> list[tuple[str, str]]:
    n = ns.node
    return [
        ("NodeName", n.name),
        ("CPUAlloc", str(ns.alloc_cpus)),
        ("CPUTot", str(n.cpus)),
        ("AvailableFeatures", ",".join(n.features) or "(null)"),
        ("Gres", n.gres()),
        ("GresUsed", gres_used(ns)),
        ("RealMemory", str(n.mem_mb)),
        ("AllocMem", str(ns.alloc_mem)),
        ("State", ns.state_long.upper()),
        ("Reason", ns.reason or "(null)"),
    ]


def _emit(ctx: Ctx, fields: list[tuple[str, str]], oneliner: bool) -> None:
    pairs = [f"{k}={v}" for k, v in fields]
    if oneliner:
        ctx.out.write(" ".join(pairs) + "\n")
        return
    ctx.out.write(pairs[0] + (" " + pairs[1] if len(pairs) > 1 else "") + "\n")
    rest = pairs[2:]
    for i in range(0, len(rest), 3):
        ctx.out.write("   " + " ".join(rest[i : i + 3]) + "\n")
    ctx.out.write("\n")


def _config_lines(ctx: Ctx) -> list[tuple[str, str]]:
    cfg = ctx.state.config
    faults = ctx.state.faults()
    return [
        (
            "AccountingStorageType",
            "accounting_storage/slurmdbd" if cfg.accounting_enabled else "accounting_storage/none",
        ),
        ("AccountingStoreFlags", "job_comment" if cfg.store_job_comment else "(null)"),
        (
            "AccountingStorageTRES",
            "cpu,mem,energy,node,billing,fs/disk,vmem,pages" + (",gres/gpu" if cfg.track_gpu_tres else ""),
        ),
        ("ClusterName", cfg.cluster_name),
        ("EnforcePartLimits", "ALL" if cfg.enforce_part_limits else "NO"),
        ("JobAcctGatherType", cfg.job_acct_gather_type),
        ("JobRequeue", "1" if cfg.job_requeue else "0"),
        ("KillWait", f"{cfg.kill_wait_s:g} sec"),
        ("MailProg", cfg.mail_prog or "(null)"),
        ("MinJobAge", f"{ctx.state.min_job_age_s(faults):g} sec"),
        ("PreemptMode", "CANCEL"),
        ("PreemptType", "preempt/partition_prio"),
        ("PrivateData", cfg.private_data or "none"),
        ("SchedulerType", "sched/builtin"),
        ("SelectType", "select/cons_tres"),
        ("SLURM_VERSION", cfg.slurm_version),
    ]


def scontrol_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "scontrol"
    args = list(argv)
    oneliner = False
    for opt in ("-o", "--oneliner"):
        while opt in args:
            args.remove(opt)
            oneliner = True
    if args and args[0] in ("-V", "--version", "version"):
        ctx.out.write(f"slurm {ctx.state.config.slurm_version}\n")
        return 0
    if len(args) < 2 or args[0] != "show":
        ctx.err.write(f"{prog}: error: hpclab-sim supports only 'scontrol show partition|job|node|config'\n")
        return 1
    what, target = args[1].lower(), (args[2] if len(args) > 2 else None)
    cfg = ctx.state.config
    if what in ("partition", "partitions"):
        parts = [p for p in cfg.partitions if target is None or p.name == target]
        if target is not None and not parts:
            ctx.err.write(f"Partition {target} not found\n")
            return 1
        for p in parts:
            _emit(ctx, _partition_fields(cfg, p), oneliner)
        return 0
    if what in ("job", "jobs"):
        ctx.state.reap(ctx.now)
        jobs = [j for j in ctx.state.jobs() if target is None or j.matches(target)]
        age = ctx.state.min_job_age_s(ctx.state.faults())
        jobs = [j for j in jobs if not j.terminal or (j.end_time or 0) + age > ctx.now]
        if target is not None and not jobs:
            ctx.err.write("slurm_load_jobs error: Invalid job id specified\n")
            return 1
        if not jobs:
            ctx.out.write("No jobs in the system\n")
        for j in jobs:
            _emit(ctx, _job_fields(ctx, j), oneliner)
        return 0
    if what in ("node", "nodes"):
        statuses = node_statuses(ctx)
        chosen = [s for name, s in statuses.items() if target is None or name == target]
        if target is not None and not chosen:
            ctx.err.write(f"Node {target} not found\n")
            return 1
        for s in chosen:
            _emit(ctx, _node_fields(s), oneliner)
        return 0
    if what == "config":
        ctx.out.write(f"Configuration data as of {slurm_time(ctx.now, ctx.env)}\n")
        for key, value in _config_lines(ctx):
            ctx.out.write(f"{key:<23} = {value}\n")
        return 0
    ctx.err.write(f"{prog}: error: invalid entity: {args[1]} for keyword: show\n")
    return 1
