"""Fake ``sacct``.

The accounting view lags the controller by ``accounting_lag_s``: sacct reports each job as it
was at ``now - lag`` (absent before its submit record arrives, then PENDING/RUNNING, the terminal
state only once the end record "reached slurmdbd"). Emulated flags: ``-X``, ``-n``, ``-P``/``-p``,
``--delimiter``, ``-j``, ``-u``, ``-a``, ``-A``, ``--name``, ``-S``/``-E`` (``now-N``, ISO local,
``MM/DD``), ``-s``, ``-o``/``--format`` (``Field%width``), ``-e``/``--helpformat``, ``-b``,
``-D``/``--duplicates``. Without ``-X`` each started job also has ``.batch`` and ``.extern`` step
lines. A requeued job is one record (its latest segment) unless ``--duplicates`` is given; then
every earlier segment that had ended by the accounting view time is listed first, with its own
Submit/Start/End/State/NodeList (same JobID; a requeue restamps the submit time). Until the
requeue reaches the (lagging) accounting view, the job shows its previous record.
``PrivateData=jobs`` hides other users' jobs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

from hpclab.sim.cmd_squeue import job_gpu_type
from hpclab.sim.cmdutil import Ctx, Opt, UsageError, flag, parse_args, split_list, sval
from hpclab.sim.formats import (
    compress_hostlist,
    elapsed_str,
    format_mem,
    format_rss_k,
    parse_timespec,
    slurm_time,
)
from hpclab.sim.state import STATE_COMPACT, SimJob

PROG = "sacct"
SACCT_OPTS = [
    Opt("allocations", "X", arg="none"),
    Opt("noheader", "n", arg="none"),
    Opt("parsable", "p", arg="none"),
    Opt("parsable2", "P", arg="none"),
    Opt("delimiter"),
    Opt("jobs", "j"),
    Opt("user", "u"),
    Opt("uid"),
    Opt("allusers", "a", arg="none"),
    Opt("accounts", "A"),
    Opt("name"),
    Opt("starttime", "S"),
    Opt("endtime", "E"),
    Opt("state", "s"),
    Opt("format", "o"),
    Opt("helpformat", "e", arg="none"),
    Opt("brief", "b", arg="none"),
    Opt("long", "l", arg="none"),
    Opt("truncate", "T", arg="none"),
    Opt("duplicates", "D", arg="none"),
    Opt("units"),
    Opt("clusters", "M"),
    Opt("allclusters", "L", arg="none"),
    Opt("partition", "r"),
    Opt("version", "V", arg="none"),
    Opt("help", "h", arg="none"),
    Opt("usage", arg="none"),
]
DEFAULT_FIELDS = ["JobID", "JobName", "Partition", "Account", "AllocCPUS", "State", "ExitCode"]
BRIEF_FIELDS = ["JobID", "State", "ExitCode"]


@dataclasses.dataclass
class Row:
    """One sacct line: a job as seen by accounting, or one of its steps."""

    job: SimJob
    step: str | None  # None = allocation line, else "batch"/"extern"
    state: str
    raw_state: str
    start: float | None
    end: float | None
    exit_code: str
    at: float  # time the view corresponds to

    @property
    def elapsed(self) -> int:
        if self.start is None:
            return 0
        end = self.end if self.end is not None else self.at
        return max(0, int(end - self.start))


def _segment_record(job: SimJob, seg: dict[str, Any], index: int) -> SimJob:
    """An earlier (requeued) segment of ``job`` as its own accounting record (its own submit
    time; segments recorded before submit times were kept share the job's)."""
    state = str(seg.get("state") or "REQUEUED")
    signal_no = int(seg.get("exit_signal") or 0)
    submit = seg.get("submit")
    return dataclasses.replace(
        job,
        submit_time=float(submit) if isinstance(submit, int | float) else job.submit_time,
        state=state,
        raw_state=state,
        start_time=seg.get("start"),
        end_time=seg.get("end"),
        node_list=list(seg.get("node_list") or []),
        exit_code=int(seg.get("exit_code") or 0),
        exit_signal=signal_no,
        batch_exit_code=0,
        batch_signal=signal_no,
        restart_count=index,
        segments=[],
    )


def _record_at(job: SimJob, at: float) -> SimJob | None:
    """The job's newest accounting record that had reached slurmdbd at ``at``.

    A requeued job has one record per submission: the new one (new submit time) arrives when
    the requeue happens (its submit stamp may be up to one second later), so a lagging view
    still shows the previous segment's record."""
    if not job.segments:
        return job if job.submit_time <= at else None
    arrival = min(job.submit_time, float(job.segments[-1].get("end") or job.submit_time))
    if arrival <= at:
        return job
    for index in range(len(job.segments) - 1, -1, -1):
        record = _segment_record(job, job.segments[index], index)
        prev_end = job.segments[index - 1].get("end") if index else None
        arrived = record.submit_time if prev_end is None else min(record.submit_time, float(prev_end))
        if arrived <= at:
            return dataclasses.replace(record, segments=[dict(s) for s in job.segments[:index]])
    return None


def accounting_view(job: SimJob, at: float) -> SimJob | None:
    """The job as slurmdbd knew it at time ``at`` (None: not recorded yet)."""
    found = _record_at(job, at)
    if found is None:
        return None
    job = found
    view = dataclasses.replace(job)
    if job.start_time is None or job.start_time > at:
        if job.terminal and job.end_time is not None and job.end_time <= at:
            return view  # ended while pending (cancelled before start)
        view.state, view.start_time, view.end_time, view.raw_state = "PENDING", None, None, None
        view.node_list, view.alloc, view.exit_code, view.exit_signal = [], {}, None, None
        return view
    if job.end_time is None or job.end_time > at:
        view.state, view.end_time, view.raw_state = "RUNNING", None, None
        view.exit_code, view.exit_signal = None, None
    return view


def segment_views(view: SimJob, at: float) -> list[SimJob]:
    """Earlier (requeued) segments of ``view`` as job records, oldest first."""
    out: list[SimJob] = []
    for i, seg in enumerate(view.segments):
        end = seg.get("end")
        if end is None or end > at:
            continue
        out.append(_segment_record(view, seg, i))
    return out


def _rows(view: SimJob, at: float, allocations_only: bool, duplicates: bool = False) -> list[Row]:
    if duplicates and view.segments:
        earlier: list[Row] = []
        for seg in segment_views(view, at):
            earlier.extend(_rows(seg, at, allocations_only))
        return earlier + _rows(view, at, allocations_only)
    exit_code = f"{view.exit_code or 0}:{view.exit_signal or 0}" if view.terminal else "0:0"
    rows = [
        Row(
            job=view,
            step=None,
            state=view.state,
            raw_state=view.raw_state or view.state,
            start=view.start_time,
            end=view.end_time,
            exit_code=exit_code,
            at=at,
        )
    ]
    if allocations_only or not view.node_list:  # never allocated: no step records
        return rows
    if view.terminal:
        bsig = view.batch_signal or 0
        bcode = view.batch_exit_code or 0
        bstate = {
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "OUT_OF_MEMORY": "OUT_OF_MEMORY",
        }.get(view.state, "CANCELLED")
        rows.append(Row(view, "batch", bstate, bstate, view.start_time, view.end_time, f"{bcode}:{bsig}", at))
        ext = "OUT_OF_MEMORY" if view.state == "OUT_OF_MEMORY" else "COMPLETED"
        rows.append(Row(view, "extern", ext, ext, view.start_time, view.end_time, "0:0", at))
    else:
        rows.append(Row(view, "batch", "RUNNING", "RUNNING", view.start_time, None, "0:0", at))
        rows.append(Row(view, "extern", "RUNNING", "RUNNING", view.start_time, None, "0:0", at))
    return rows


def _fake_rss(row: Row) -> str:
    if row.step != "batch":
        return "" if row.step is None else "0"
    return format_rss_k((256 + 16 * min(row.elapsed, 3600)) * 1024 * 1024)


class Fields:
    """Field name -> value getter; ``ctx`` supplies SLURM_TIME_FORMAT and config."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        cfg = ctx.state.config

        def t(value: float | None) -> str:
            return slurm_time(value, ctx.env, missing="Unknown")

        def jid(r: Row, raw: bool) -> str:
            base = str(r.job.job_id) if raw else r.job.display_id
            return f"{base}.{r.step}" if r.step else base

        def alloc_tres(r: Row) -> str:
            if not r.job.node_list:  # pending, or ended without an allocation
                return ""
            j = r.job
            items = [("billing", str(j.total_cpus)), ("cpu", str(j.total_cpus))]
            if j.total_gpus and cfg.track_gpu_tres:
                gtype = job_gpu_type(j, cfg)
                if gtype:
                    items.append((f"gres/gpu:{gtype}", str(j.total_gpus)))
                items.append(("gres/gpu", str(j.total_gpus)))
            items += [("mem", format_mem(j.mem_per_node_mb * j.nodes)), ("node", str(j.nodes))]
            return ",".join(f"{k}={v}" for k, v in items)

        def req_tres(r: Row) -> str:
            j = r.job
            items = [("billing", str(j.total_cpus)), ("cpu", str(j.total_cpus))]
            if j.total_gpus and cfg.track_gpu_tres:
                items.append(("gres/gpu", str(j.total_gpus)))
            items += [("mem", format_mem(j.mem_per_node_mb * j.nodes)), ("node", str(j.nodes))]
            return ",".join(f"{k}={v}" for k, v in items)

        def limit_min(r: Row) -> str:
            lim = r.job.time_limit_s
            return "UNLIMITED" if lim is None else str(math.ceil(lim / 60))

        self.getters: dict[str, tuple[str, Callable[[Row], str]]] = {}
        for name, getter in {
            "JobID": lambda r: jid(r, False),
            "JobIDRaw": lambda r: jid(r, True),
            "JobName": lambda r: r.step if r.step else r.job.name,
            "User": lambda r: "" if r.step else r.job.user,
            "UID": lambda r: "" if r.step else str(r.job.uid),
            "Group": lambda r: "" if r.step else r.job.user,
            "Account": lambda r: r.job.account,
            "Partition": lambda r: "" if r.step else r.job.partition,
            "QOS": lambda r: "" if r.step else r.job.qos,
            "State": lambda r: r.raw_state,
            "ExitCode": lambda r: r.exit_code,
            "DerivedExitCode": lambda r: "" if r.step else "0:0",
            "Submit": lambda r: t(r.job.submit_time),
            "Eligible": lambda r: t(r.job.submit_time),
            "Start": lambda r: t(r.start),
            "End": lambda r: t(r.end),
            "Elapsed": lambda r: elapsed_str(r.elapsed),
            "ElapsedRaw": lambda r: str(r.elapsed),
            "Timelimit": lambda r: "" if r.step else elapsed_str(r.job.time_limit_s),
            "TimelimitRaw": lambda r: "" if r.step else limit_min(r),
            "NodeList": lambda r: compress_hostlist(r.job.node_list) if r.job.node_list else "None assigned",
            "NNodes": lambda r: str(r.job.nodes),
            "NCPUS": lambda r: str(r.job.total_cpus),
            "AllocCPUS": lambda r: str(r.job.total_cpus) if r.job.node_list else "0",
            "ReqCPUS": lambda r: str(r.job.total_cpus),
            "AllocNodes": lambda r: str(r.job.nodes) if r.job.node_list else "0",
            "AllocTRES": alloc_tres,
            "ReqTRES": lambda r: "" if r.step else req_tres(r),
            "ReqMem": lambda r: format_mem(r.job.mem_per_node_mb),
            "Comment": lambda r: (r.job.comment or "") if cfg.store_job_comment and not r.step else "",
            "SubmitLine": lambda r: "" if r.step else r.job.submit_line,
            "WorkDir": lambda r: "" if r.step else r.job.work_dir,
            "Reason": lambda r: "" if r.step else (r.job.reason or "None"),
            "MaxRSS": _fake_rss,
            "CPUTimeRAW": lambda r: str(r.elapsed * r.job.total_cpus),
            "Priority": lambda r: "" if r.step else str(max(1, 100000 - r.job.job_id)),
            "Constraints": lambda r: r.job.constraint or "",
            "Cluster": lambda r: cfg.cluster_name,
            "Flags": lambda r: "",
        }.items():
            self.getters[name.lower()] = (name, getter)

    def resolve(self, spec: str) -> tuple[str, Callable[[Row], str], int | None]:
        name, _, width = spec.partition("%")
        entry = self.getters.get(name.lower())
        if entry is None:
            raise UsageError(f'{PROG}: error: Invalid field requested: "{name}"')
        return entry[0], entry[1], int(width) if width.isdigit() else None


def helpformat(ctx: Ctx) -> str:
    names = sorted(name for name, _ in Fields(ctx).getters.values())
    lines = []
    for i in range(0, len(names), 4):
        lines.append("".join(f"{n:<20}" for n in names[i : i + 4]).rstrip())
    return "\n".join(lines) + "\n"


def _window(ctx: Ctx, values: dict[str, str | bool], by_jobs: bool) -> tuple[float | None, float]:
    start_raw, end_raw = sval(values, "starttime"), sval(values, "endtime")
    try:
        end = parse_timespec(end_raw, ctx.now) if end_raw else ctx.now
        if start_raw:
            start: float | None = parse_timespec(start_raw, ctx.now)
        elif by_jobs:
            start = None
        else:
            midnight = datetime.fromtimestamp(ctx.now).replace(hour=0, minute=0, second=0, microsecond=0)
            start = midnight.timestamp()
    except ValueError as exc:
        raise UsageError(f"{PROG}: error: Invalid time specification: {exc}") from None
    return start, end


def main(ctx: Ctx, argv: list[str]) -> int:
    try:
        values, _ = parse_args(PROG, argv, SACCT_OPTS)
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return exc.rc
    cfg = ctx.state.config
    if flag(values, "version"):
        ctx.out.write(f"slurm {cfg.slurm_version}\n")
        return 0
    if flag(values, "helpformat"):
        ctx.out.write(helpformat(ctx))
        return 0
    if flag(values, "help") or flag(values, "usage"):
        ctx.out.write("Usage: sacct [options]\n")
        return 0
    if not cfg.accounting_enabled:
        ctx.err.write(f"{PROG}: error: Slurm accounting storage is disabled\n")
        return 1
    faults = ctx.state.faults()
    if faults.sacct_unavailable:
        ctx.err.write(f"{PROG}: error: Problem talking to the database: Connection refused\n")
        return 1
    ctx.state.reap(ctx.now)
    fields = Fields(ctx)
    raw_fields = split_list(sval(values, "format")) or (
        BRIEF_FIELDS if flag(values, "brief") else DEFAULT_FIELDS
    )
    try:
        specs = [fields.resolve(f) for f in raw_fields]
        job_tokens = split_list(sval(values, "jobs"))
        start, end = _window(ctx, values, bool(job_tokens))
    except UsageError as exc:
        ctx.err.write(exc.message + "\n")
        return 1

    at = ctx.now - ctx.state.accounting_lag_s(faults)
    views = [v for v in (accounting_view(j, at) for j in ctx.state.jobs()) if v is not None]
    if job_tokens:
        views = [v for v in views if any(v.matches(t) for t in job_tokens)]
    users = split_list(sval(values, "user"))
    if users:
        views = [v for v in views if v.user in users or str(v.uid) in users]
    elif not flag(values, "allusers") and not job_tokens:
        views = [v for v in views if v.user == cfg.user]
    if cfg.hides_other_jobs():  # PrivateData=jobs: a regular user sees only their own records
        views = [v for v in views if v.user == cfg.user]
    if accounts := split_list(sval(values, "accounts")):
        views = [v for v in views if v.account in accounts]
    if names := split_list(sval(values, "name")):
        views = [v for v in views if v.name in names]
    if parts := split_list(sval(values, "partition")):
        views = [v for v in views if v.partition in parts]
    if states := split_list((sval(values, "state") or "").upper()):
        long_by_short = {s: long for long, s in STATE_COMPACT.items()}
        wanted = {long_by_short.get(s, s) for s in states}
        views = [v for v in views if v.state in wanted]
    views = [
        v
        for v in views
        if v.submit_time <= end and (start is None or v.end_time is None or v.end_time >= start)
    ]

    rows: list[Row] = []
    for v in sorted(views, key=lambda x: x.job_id):
        rows.extend(_rows(v, at, flag(values, "allocations"), flag(values, "duplicates")))
    _print(ctx, values, specs, rows)
    return 0


def _print(
    ctx: Ctx,
    values: dict[str, str | bool],
    specs: list[tuple[str, Callable[[Row], str], int | None]],
    rows: list[Row],
) -> None:
    parsable2, parsable = flag(values, "parsable2"), flag(values, "parsable")
    header = not flag(values, "noheader")
    if parsable or parsable2:
        delim = sval(values, "delimiter") or "|"
        tail = delim if parsable and not parsable2 else ""
        if header:
            ctx.out.write(delim.join(name for name, _, _ in specs) + tail + "\n")
        for row in rows:
            ctx.out.write(delim.join(getter(row) for _, getter, _ in specs) + tail + "\n")
        return
    widths = [w or (12 if name == "JobID" else 10) for name, _, w in specs]
    if header:
        ctx.out.write(
            " ".join(name[:w].rjust(w) for (name, _, _), w in zip(specs, widths, strict=True)) + " \n"
        )
        ctx.out.write(" ".join("-" * w for w in widths) + " \n")
    for row in rows:
        cells = []
        for (_, getter, _), w in zip(specs, widths, strict=True):
            value = getter(row)
            cells.append((value[: w - 1] + "+") if len(value) > w else value.rjust(w))
        ctx.out.write(" ".join(cells) + " \n")
