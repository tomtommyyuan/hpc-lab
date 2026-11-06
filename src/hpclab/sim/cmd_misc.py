"""Fake ``scancel``, ``sacctmgr``, ``sstat``, ``srun --jobid --overlap`` and ``nvidia-smi``.

``scancel --signal`` follows slurmctld: a signal other than KILL is delivered only to a running
job (``--batch``: the batch shell alone; otherwise the whole job); a pending job answers
"Job is pending execution". The fake ``nvidia-smi`` (``<sim_root>/gpu-bin``, on the PATH of jobs
on GPU nodes, and behind ``srun --jobid --overlap nvidia-smi``) reports the job's allocated GPUs:
``-L``, the plain header table (driver and CUDA version) and ``--query-gpu=... --format=csv[,
noheader][,nounits]`` including ``timestamp``; utilization comes from the ``gpu_util`` fault.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from hpclab.sim.cmd_squeue import job_gpu_type
from hpclab.sim.cmdutil import Ctx, Opt, UsageError, flag, parse_args, split_list, sval
from hpclab.sim.config import SimConfig
from hpclab.sim.formats import elapsed_str, format_rss_k, parse_time_limit
from hpclab.sim.jobrunner import gpu_bin_for, job_environment
from hpclab.sim.state import SimJob, safe_killpg

# ---------------------------------------------------------------------------
# scancel
# ---------------------------------------------------------------------------

SCANCEL_OPTS = [
    Opt("signal", "s"),
    Opt("user", "u"),
    Opt("name", "n"),
    Opt("partition", "p"),
    Opt("account", "A"),
    Opt("state", "t"),
    Opt("batch", "b", arg="none"),
    Opt("full", "f", arg="none"),
    Opt("quiet", "Q", arg="none"),
    Opt("verbose", "v", arg="none"),
    Opt("interactive", "i", arg="none"),
    Opt("me", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
    Opt("usage", arg="none"),
]
_SIGNALS = {name.removeprefix("SIG"): int(sig) for name, sig in signal.Signals.__members__.items()}


def _signal_number(spec: str) -> int:
    s = spec.upper().removeprefix("SIG")
    if s.isdigit():
        return int(s)
    if s not in _SIGNALS:
        raise UsageError(f"scancel: error: Unknown job signal: {spec}")
    return _SIGNALS[s]


def _signal_batch_shell(pgid: int | None, sig: int) -> bool:
    """``scancel --batch``: only the batch script's shell (the process group leader)."""
    if not pgid or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.kill(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def scancel_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "scancel"
    try:
        values, ids = parse_args(prog, argv, SCANCEL_OPTS)
        raw_signal = sval(values, "signal")
        sig = _signal_number(raw_signal) if raw_signal else None
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    if flag(values, "version"):
        ctx.out.write(f"slurm {ctx.state.config.slurm_version}\n")
        return 0
    users = split_list(sval(values, "user")) + ([ctx.state.config.user] if flag(values, "me") else [])
    names = split_list(sval(values, "name"))
    if not ids and not users and not names:
        ctx.err.write(f"{prog}: error: No job identification provided\n")
        return 1
    rc = 0
    uid = os.getuid()
    with ctx.state.tx() as conn:
        jobs = ctx.state.load_jobs(conn)
        targets: list[tuple[str, list[SimJob]]] = [(t, [j for j in jobs if j.matches(t)]) for t in ids]
        if not ids:
            chosen = [
                j
                for j in jobs
                if not j.terminal and (not users or j.user in users) and (not names or j.name in names)
            ]
            targets = [(j.display_id, [j]) for j in chosen]
        for token, matched in targets:
            live = [j for j in matched if not j.terminal]
            if not matched:
                ctx.err.write(f"{prog}: error: Kill job error on job id {token}: Invalid job id specified\n")
                rc = 1
                continue
            if not live:
                ctx.err.write(
                    f"{prog}: error: Kill job error on job id {token}: Job/step already completing or completed\n"
                )
                rc = 1
                continue
            for job in live:
                if job.synthetic:  # injected record: no processes, the cancel lands at once
                    ctx.state.finish(conn, job, "CANCELLED", now=ctx.now, raw_state=f"CANCELLED by {uid}")
                    continue
                if sig is not None and sig != signal.SIGKILL:
                    if job.state != "RUNNING":
                        ctx.err.write(
                            f"{prog}: error: Kill job error on job id {token}: Job is pending execution\n"
                        )
                        rc = 1
                    elif flag(values, "batch") and not flag(values, "full"):
                        _signal_batch_shell(job.pgid, sig)
                    else:
                        safe_killpg(job.pgid, sig)
                    continue
                if job.state == "PENDING":
                    ctx.state.finish(conn, job, "CANCELLED", now=ctx.now, raw_state=f"CANCELLED by {uid}")
                    continue
                # The job's runner (slurmstepd) prints the CANCELLED banner, then SIGTERM/KillWait/SIGKILL.
                job.cancel_requested = True
                job.cancelled_by = uid
                ctx.state.put_job(conn, job)
    return rc


# ---------------------------------------------------------------------------
# sacctmgr
# ---------------------------------------------------------------------------

SACCTMGR_OPTS = [
    Opt("noheader", "n", arg="none"),
    Opt("parsable", "p", arg="none"),
    Opt("parsable2", "P", arg="none"),
    Opt("immediate", "i", arg="none"),
    Opt("quiet", "Q", arg="none"),
    Opt("readonly", "r", arg="none"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
]
ASSOC_DEFAULT_FORMAT = [
    "Cluster",
    "Account",
    "User",
    "Partition",
    "Share",
    "GrpJobs",
    "GrpTRES",
    "MaxJobs",
    "QOS",
]


def _assoc_row(cfg: SimConfig, account: str) -> dict[str, str]:
    """One user association as sacctmgr reports it (field names lower-cased)."""
    max_submit = str(cfg.max_submit_jobs_per_user or "")
    return {
        "cluster": cfg.cluster_name,
        "account": account,
        "user": cfg.user,
        "partition": "",
        "share": "1",
        "fairshare": "1",
        "grpjobs": "",
        "grptres": "",
        "maxjobs": "",
        "maxsubmit": max_submit,
        "maxsubmitjobs": max_submit,
        "maxtres": "",
        "maxwall": "",
        "qos": ",".join(cfg.qos),
        "defaultqos": cfg.default_qos,
        "def acct": cfg.default_account,
        "defaultaccount": cfg.default_account,
        "admin": "None",
        "parentname": "root",
    }


ASSOC_FIELDS = frozenset(_assoc_row(SimConfig(user="x"), "x"))
QOS_DEFAULT_FORMAT = [
    "Name",
    "Priority",
    "GraceTime",
    "Preempt",
    "PreemptMode",
    "Flags",
    "UsageFactor",
    "GrpTRES",
    "MaxTRES",
    "MaxWall",
    "MaxTRESPU",
    "MaxJobsPU",
    "MaxSubmitPU",
]


def _qos_row(cfg: SimConfig, name: str) -> dict[str, str]:
    """One QOS as ``sacctmgr show qos`` reports it (limits from the simulator config)."""
    return {
        "name": name,
        "priority": "0",
        "gracetime": "00:00:00",
        "preempt": "",
        "preemptexempttime": "",
        "preemptmode": "cluster",
        "flags": "",
        "usagethres": "",
        "usagefactor": "1.000000",
        "grptres": "",
        "grptresmins": "",
        "grptresrunmins": "",
        "grpjobs": "",
        "grpsubmit": "",
        "grpwall": "",
        "maxtres": "",
        "maxtrespernode": "",
        "maxtresmins": "",
        "maxwall": elapsed_str(cfg.qos_max_wall_s) if cfg.qos_max_wall_s is not None else "",
        "maxtrespu": f"gres/gpu={cfg.qos_max_gpus_per_user}" if cfg.qos_max_gpus_per_user is not None else "",
        "maxjobspu": str(cfg.qos_max_jobs_per_user) if cfg.qos_max_jobs_per_user is not None else "",
        "maxsubmitpu": str(cfg.max_submit_jobs_per_user or ""),
        "maxtrespa": "",
        "maxjobspa": "",
        "maxsubmitpa": "",
        "mintres": "",
    }


QOS_FIELDS = frozenset(_qos_row(SimConfig(user="x"), "x"))


def sacctmgr_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "sacctmgr"
    try:
        values, words = parse_args(prog, argv, SACCTMGR_OPTS)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    cfg = ctx.state.config
    if flag(values, "version"):
        ctx.out.write(f"slurm {cfg.slurm_version}\n")
        return 0
    if not cfg.accounting_enabled:
        ctx.err.write(
            f"{prog}: error: You are not running a supported accounting_storage plugin\n"
            "Only 'accounting_storage/slurmdbd' and 'accounting_storage/mysql' are supported.\n"
        )
        return 1
    if ctx.state.faults().sacct_unavailable:
        ctx.err.write(f"{prog}: error: Problem talking to the database: Connection refused\n")
        return 1
    if len(words) < 2 or words[0].lower() not in ("show", "list"):
        ctx.err.write(f" Unknown command: {' '.join(words[:1])}\n")
        return 1
    entity = words[1].lower()
    conditions: dict[str, str] = {}
    fmt: list[str] = []
    withassoc = False
    positional_user: str | None = None
    for word in words[2:]:
        key, eq, value = word.partition("=")
        k = key.lower()
        if not eq:
            if k in ("where", "withassoc", "withassocs"):
                withassoc = withassoc or k.startswith("withassoc")
            else:
                positional_user = key
            continue
        if k == "format":
            fmt = [f.split("%", 1)[0] for f in value.split(",") if f]
        else:
            conditions[k.rstrip("s")] = value
    if entity == "qos":
        return _show_qos(ctx, values, conditions, fmt, positional_user)
    if entity == "user" and positional_user and "user" not in conditions and "name" not in conditions:
        conditions["user"] = positional_user
    if entity == "user":
        conditions.setdefault("user", conditions.pop("name", "") or "")
        if not withassoc:
            fmt = fmt or ["User", "Def Acct", "Admin"]
    elif entity not in ("assoc", "association", "associations"):
        ctx.err.write(f" Unknown option: {words[1]}\n")
        return 1
    fmt = fmt or ASSOC_DEFAULT_FORMAT
    user_filter = split_list(conditions.get("user"))
    account_filter = split_list(conditions.get("account"))
    for name in fmt:
        if name.lower() not in ASSOC_FIELDS:
            ctx.err.write(f" Unknown field '{name}'\n")
            return 1
    rows = [
        _assoc_row(cfg, account)
        for account in cfg.accounts
        if (not account_filter or account in account_filter) and (not user_filter or cfg.user in user_filter)
    ]
    delim = "|"
    parsable = flag(values, "parsable") or flag(values, "parsable2")
    tail = delim if flag(values, "parsable") and not flag(values, "parsable2") else ""
    if not flag(values, "noheader"):
        if parsable:
            ctx.out.write(delim.join(fmt) + tail + "\n")
        else:
            ctx.out.write(" ".join(f"{n[:10]:>10}" for n in fmt) + " \n")
            ctx.out.write(" ".join("-" * 10 for _ in fmt) + " \n")
    for row in rows:
        cells = [row.get(n.lower(), "") for n in fmt]
        if parsable:
            ctx.out.write(delim.join(cells) + tail + "\n")
        else:
            ctx.out.write(" ".join(f"{c[:10]:>10}" for c in cells) + " \n")
    return 0


def _show_qos(
    ctx: Ctx,
    values: dict[str, str | bool],
    conditions: dict[str, str],
    fmt: list[str],
    positional: str | None,
) -> int:
    cfg = ctx.state.config
    fmt = fmt or QOS_DEFAULT_FORMAT
    for name in fmt:
        if name.lower() not in QOS_FIELDS:
            ctx.err.write(f" Unknown field '{name}'\n")
            return 1
    wanted = split_list(conditions.get("name") or positional)
    rows = [_qos_row(cfg, q) for q in cfg.qos if not wanted or q in wanted]
    _print_table(ctx, values, fmt, rows)
    return 0


def _print_table(ctx: Ctx, values: dict[str, str | bool], fmt: list[str], rows: list[dict[str, str]]) -> None:
    delim = "|"
    parsable = flag(values, "parsable") or flag(values, "parsable2")
    tail = delim if flag(values, "parsable") and not flag(values, "parsable2") else ""
    if not flag(values, "noheader"):
        if parsable:
            ctx.out.write(delim.join(fmt) + tail + "\n")
        else:
            ctx.out.write(" ".join(f"{n[:10]:>10}" for n in fmt) + " \n")
            ctx.out.write(" ".join("-" * 10 for _ in fmt) + " \n")
    for row in rows:
        cells = [row.get(n.lower(), "") for n in fmt]
        if parsable:
            ctx.out.write(delim.join(cells) + tail + "\n")
        else:
            ctx.out.write(" ".join(f"{c[:10]:>10}" for c in cells) + " \n")


# ---------------------------------------------------------------------------
# sstat
# ---------------------------------------------------------------------------

SSTAT_OPTS = [
    Opt("allsteps", "a", arg="none"),
    Opt("jobs", "j"),
    Opt("noheader", "n", arg="none"),
    Opt("parsable", "p", arg="none"),
    Opt("parsable2", "P", arg="none"),
    Opt("format", "o"),
    Opt("fields"),
    Opt("helpformat", "e", arg="none"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
]
SSTAT_FIELDS = ["JobID", "MaxRSS", "MaxRSSNode", "AveRSS", "AveCPU", "NTasks", "TRESUsageInTot", "MaxVMSize"]


def _sstat_row(job: SimJob, step: str, now: float) -> dict[str, str]:
    elapsed = int(now - (job.start_time or now))
    rss = (256 + 16 * min(elapsed, 3600)) * 1024 * 1024 if step == "batch" else 1024 * 1024
    cpu_s = int(elapsed * 0.9 * (job.cpus_per_task or 1)) if step == "batch" else 0
    node = job.node_list[0] if job.node_list else "localhost"
    return {
        "jobid": f"{job.display_id}.{step}",
        "maxrss": format_rss_k(rss),
        "maxrssnode": node,
        "averss": format_rss_k(rss),
        "avecpu": elapsed_str(cpu_s),
        "ntasks": "1",
        "tresusageintot": f"cpu={elapsed_str(cpu_s)},energy=0,fs/disk=0,mem={format_rss_k(rss)},pages=0,vmem={format_rss_k(rss * 2)}",
        "maxvmsize": format_rss_k(rss * 2),
    }


def sstat_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "sstat"
    try:
        values, positionals = parse_args(prog, argv, SSTAT_OPTS)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    if flag(values, "version"):
        ctx.out.write(f"slurm {ctx.state.config.slurm_version}\n")
        return 0
    if flag(values, "helpformat"):
        ctx.out.write("".join(f"{n:<20}" for n in SSTAT_FIELDS).rstrip() + "\n")
        return 0
    fields = [f.split("%", 1)[0] for f in split_list(sval(values, "format") or sval(values, "fields"))]
    fields = fields or ["JobID", "MaxRSS", "AveCPU", "NTasks"]
    for f in fields:
        if f.lower() not in {n.lower() for n in SSTAT_FIELDS}:
            ctx.err.write(f'{prog}: error: Invalid field requested: "{f}"\n')
            return 1
    tokens = split_list(sval(values, "jobs")) + positionals
    if not tokens:
        ctx.err.write(f"{prog}: error: You didn't give me any jobs to stat.\n")
        return 1
    jobs = ctx.state.jobs()
    rows: list[dict[str, str]] = []
    for token in tokens:
        base, _, step = token.partition(".")
        matched = [j for j in jobs if j.matches(base) and j.state == "RUNNING"]
        if not matched:
            ctx.err.write(f"{prog}: error: couldn't get steps for job {base}\n")
            return 1
        for job in matched:
            steps = [step] if step else (["extern", "batch"] if flag(values, "allsteps") else [])
            rows.extend(_sstat_row(job, s, ctx.now) for s in steps)
    parsable = flag(values, "parsable") or flag(values, "parsable2")
    tail = "|" if flag(values, "parsable") and not flag(values, "parsable2") else ""
    if not flag(values, "noheader"):
        ctx.out.write(("|".join(fields) + tail if parsable else " ".join(f"{f:>12}" for f in fields)) + "\n")
    for row in rows:
        cells = [row[f.lower()] for f in fields]
        ctx.out.write(("|".join(cells) + tail if parsable else " ".join(f"{c:>12}" for c in cells)) + "\n")
    return 0


# ---------------------------------------------------------------------------
# srun --jobid <id> --overlap <cmd>
# ---------------------------------------------------------------------------

SRUN_OPTS = [
    Opt("jobid"),
    Opt("overlap", arg="none"),
    Opt("nodes", "N"),
    Opt("ntasks", "n"),
    Opt("time", "t"),
    Opt("quiet", "Q", arg="none"),
    Opt("pty", arg="none"),
    Opt("gres"),
    Opt("gpus", "G"),
    Opt("mem"),
    Opt("nodelist", "w"),
    Opt("cpus-per-task", "c"),
    Opt("export"),
    Opt("verbose", "v", arg="none"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
]
GPU_NAMES = {"a100": "NVIDIA A100-SXM4-80GB", "h100": "NVIDIA H100 80GB HBM3", "b200": "NVIDIA B200"}
GPU_MEM_MIB = {"a100": 81920, "h100": 81559, "b200": 183359}


def _gpu_indices(job: SimJob, node: str | None) -> list[int]:
    target = node if node in job.alloc else (job.node_list[0] if job.node_list else "")
    return [int(g) for g in job.alloc.get(target, {}).get("gpus", [])]


def _nvidia_smi(ctx: Ctx, job: SimJob, argv: list[str]) -> int:
    cfg = ctx.state.config
    node = ctx.env.get("SLURMD_NODENAME") or (job.node_list[0] if job.node_list else None)
    gpus = _gpu_indices(job, node)
    if not gpus:
        ctx.out.write("No devices were found\n")
        return 6
    gtype = job_gpu_type(job, cfg) or "a100"
    name = GPU_NAMES.get(gtype, f"NVIDIA {gtype.upper()}")
    total = GPU_MEM_MIB.get(gtype, 81920)
    util = ctx.state.gpu_util(ctx.state.faults())
    uuid = {i: f"GPU-sim-{job.job_id:08d}-{i:04d}" for i in gpus}
    if "-L" in argv or "--list-gpus" in argv:
        for i in gpus:
            ctx.out.write(f"GPU {i}: {name} (UUID: {uuid[i]})\n")
        return 0
    query = next((a.split("=", 1)[1] for a in argv if a.startswith("--query-gpu=")), None)
    fmt = next((a.split("=", 1)[1] for a in argv if a.startswith("--format=")), "csv")
    if query is None:
        stamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(ctx.now))
        ctx.out.write(
            f"{stamp}\n"
            "+-----------------------------------------------------------------------------------------+\n"
            f"| NVIDIA-SMI {cfg.driver_version}   Driver Version: {cfg.driver_version}"
            f"   CUDA Version: {cfg.driver_cuda}  |\n"
            f"| hpclab-sim fake nvidia-smi: {len(gpus)} GPU(s) on {node} |\n"
            "+-----------------------------------------------------------------------------------------+\n"
        )
        for i in gpus:
            ctx.out.write(f"|   {i}  {name}  {int(total * 0.6)}MiB / {total}MiB  {util:.0f}% |\n")
        return 0
    nounits = "nounits" in fmt
    columns = [c.strip() for c in query.split(",")]
    if "noheader" not in fmt:
        ctx.out.write(", ".join(columns) + "\n")
    millis = int((ctx.now % 1) * 1000)
    timestamp = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(ctx.now)) + f".{millis:03d}"
    for i in gpus:
        values = {
            "timestamp": timestamp,
            "index": str(i),
            "name": name,
            "utilization.gpu": f"{util:.0f}" if nounits else f"{util:.0f} %",
            "memory.used": str(int(total * 0.6)) if nounits else f"{int(total * 0.6)} MiB",
            "memory.total": str(total) if nounits else f"{total} MiB",
            "power.draw": f"{80 + 3.5 * util:.2f}" if nounits else f"{80 + 3.5 * util:.2f} W",
            "temperature.gpu": "54",
            "uuid": uuid[i],
            "driver_version": cfg.driver_version,
            "pci.bus_id": f"00000000:{0x18 + i:02X}:00.0",
        }
        ctx.out.write(", ".join(values.get(c, "[N/A]") for c in columns) + "\n")
    return 0


NVIDIA_SMI_FAILED = (
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
    "Make sure that the latest NVIDIA driver is installed and running.\n"
)


def nvidia_smi_main(ctx: Ctx, argv: list[str]) -> int:
    """The per-job fake ``nvidia-smi``: identifies its job via ``SLURM_JOB_ID``."""
    job_id = ctx.env.get("SLURM_JOB_ID", "")
    job = next((j for j in ctx.state.jobs() if job_id and j.matches(job_id)), None)
    if job is None or job.state != "RUNNING":
        ctx.out.write(NVIDIA_SMI_FAILED)
        return 9
    return _nvidia_smi(ctx, job, argv)


def srun_main(ctx: Ctx, argv: list[str]) -> int:
    prog = "srun"
    try:
        values, command = parse_args(prog, argv, SRUN_OPTS, stop_at_positional=True)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    cfg = ctx.state.config
    if flag(values, "version"):
        ctx.out.write(f"slurm {cfg.slurm_version}\n")
        return 0
    jobid = sval(values, "jobid")
    if jobid is None:
        ctx.err.write(
            f"{prog}: error: hpclab-sim only supports steps inside an existing job (--jobid=<id>)\n"
        )
        return 1
    if not command:
        ctx.err.write(f"{prog}: fatal: No command given to execute.\n")
        return 1
    job = next((j for j in ctx.state.jobs() if j.matches(jobid)), None)
    if job is None:
        ctx.err.write(
            f"{prog}: error: Unable to confirm allocation for job {jobid}: Invalid job id specified\n"
            f"{prog}: Check SLURM_JOB_ID environment variable. Expired or invalid job {jobid}\n"
        )
        return 1
    if job.state == "PENDING":
        ctx.err.write(
            f"{prog}: error: Unable to confirm allocation for job {jobid}: Job is pending execution\n"
        )
        return 1
    if job.state != "RUNNING":
        ctx.err.write(
            f"{prog}: error: Unable to confirm allocation for job {jobid}: Job/step already completing or completed\n"
        )
        return 1
    if os.path.basename(command[0]) == "nvidia-smi":
        return _nvidia_smi(ctx, job, command[1:])
    timeout: float | None = None
    if (t := sval(values, "time")) is not None:
        try:
            timeout = parse_time_limit(t)
        except ValueError:
            ctx.err.write(f"{prog}: error: Invalid --time specification\n")
            return 1
    env = job_environment(job, cfg.cluster_name, gpu_bin=gpu_bin_for(ctx.state, job))
    env["SLURM_STEP_ID"] = "0"
    try:
        proc = subprocess.run(
            command,
            env=env,
            cwd=job.work_dir if os.path.isdir(job.work_dir) else None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        ctx.err.write(f"slurmstepd: error: execve(): {command[0]}: No such file or directory\n")
        ctx.err.write(
            f"{prog}: error: {job.node_list[0] if job.node_list else 'node'}: task 0: Exited with exit code 2\n"
        )
        return 2
    except subprocess.TimeoutExpired:
        ctx.err.write(
            f"{prog}: error: *** STEP {job.job_id}.0 CANCELLED AT {time.strftime('%Y-%m-%dT%H:%M:%S')} DUE TO TIME LIMIT ***\n"
        )
        return 1
    ctx.out.write(proc.stdout.decode("utf-8", "replace"))
    ctx.err.write(proc.stderr.decode("utf-8", "replace"))
    return proc.returncode
