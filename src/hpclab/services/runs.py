"""Read side of runs: ``runs_list`` (list + grid), ``run_inspect``, ``diagnose_attempt``,
``operation_get``.

Everything here is bounded (pages, capped lists, byte-bounded grids) and coverage-honest: cell
codes come from ``services.coverage`` (the single source of truth shared with results_compare and
evidence), scheduler freshness from ``poll_state``, and a source that could not be read is
reported as such - never as an empty success. Remote reads are limited to ``run_inspect``'s
focus attempt (metrics tail <= 16 KiB, telemetry tails, bounded log tails for diagnosis; the live
files are re-read at most once per poll interval unless ``refresh=True``) and the explicitly
requested refresh / GPU probe.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.api.envelope import Coverage, clamp_limit, decode_cursor, encode_cursor
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import Ambiguous, ErrorCode, HpcLabError, InvalidArgument, NotFound
from hpclab.domain.hashing import sha256_hex
from hpclab.domain.states import AttemptPhase, ResultState, RunStatus, SchedulerState
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import CellStatus, cell_statuses, select_evaluations, summarize_codes
from hpclab.services.diagnosis import Diagnosis, diagnose
from hpclab.services.health import Health, flag_char
from hpclab.services.observe import (
    LIVE_VIEW_TTL_S,
    OBSERVED_PHASES,
    LiveViews,
    attempt_facts,
    compute_live_views,
    identity_mismatch,
    layout_for,
    pick_observation,
    plan_dict,
    read_log_tails,
    refresh_attempts,
    store_live_views,
    sub_dict,
)
from hpclab.services.pending import PendingInfo, decode_pending
from hpclab.services.progress import Progress
from hpclab.storage.db import Row, jdump, jload

log = logging.getLogger("hpclab.runs")

__all__ = [
    "NEW_KEY",
    "JobRef",
    "OperationData",
    "RunInspectData",
    "RunsListData",
    "call",
    "diagnose_attempt",
    "operation_get",
    "run_inspect",
    "runs_list",
]

Scalar = str | int | float | bool | None

GRID_MAX_BYTES = 48 * 1024  # grid header + rows per page stay well under the 64 KiB response bound
LIST_MAX_BYTES = 48 * 1024  # list rows per page, same bound
MAX_GRID_RUNS = 4096
MAX_GRID_COLS = 64  # columns are not paginated: more is not a readable grid (MAX_SEEDS is 64)
MAX_LABEL = 80
MAX_COUNT_RUNS = 5000
COUNT_BATCH_RUNS = 200  # runs_list counts: runs whose statuses (and attempt rows) are held at once
MAX_ATTEMPTS_SHOWN = 20
MAX_TEXT = 300
GPU_PROBE_INTERVAL_S = 600.0
HEALTH_STALE_S = 3 * LIVE_VIEW_TTL_S  # cached health older than this is not shown as current
FOREIGN_STALE_S = 600.0
JOB_REF_SUBMIT_TOLERANCE_S = 2.0
# observations key (+ attempt id): when run_inspect last read the attempt's live files, and what
# it found. Later inspects reuse that read for one poll interval instead of re-reading.
LIVE_READ_PREFIX = "live:"
# worst first: an aggregated grid cell shows the most severe code of its runs. E (completed,
# evaluation pending) is not a failure: it only outranks V.
_CODE_SEVERITY = ("F", "C!", "U", "?", "X", "R", "Q", "S", "P", "E", "V")
LEGEND = {
    "V": "valid result",
    "E": "completed; evaluation pending (outputs being checked or not evaluated yet) - not a failure",
    "R": "running",
    "Q": "queued",
    "S": "submitting",
    "P": "prepared (not dispatched)",
    "U": "submission uncertain / duplicate",
    "F": "failed",
    "C!": "completed with incomplete/invalid outputs or a partial/invalid evaluation",
    "X": "cancelled / rejected / abandoned (incl. never dispatched by a cancelled workflow)",
    "?": "state unknown now (observation too old)",
    "-": "no planned run in this cell",
}


def _clip(value: Any, n: int = MAX_TEXT) -> Any:
    if isinstance(value, str) and len(value) > n:
        return value[:n] + "..."
    return value


NEW_KEY = "<new key>"  # idempotency_key placeholder: every mutation needs a fresh key


def call(tool: str, **args: Any) -> str:
    """A tool call as an MCP client sends it: ``tool(a="x", n=3, flag=true)`` (JSON literals;
    None arguments are left out). Mutations get ``idempotency_key=NEW_KEY`` from the caller."""
    parts = [f"{k}={json.dumps(v)}" for k, v in args.items() if v is not None]
    return f"{tool}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobRef(_M):
    """External job identity for reverse lookup (job ids are not global identities)."""

    cluster_id: str
    job_id: str
    submit_time: str | None = None


class AttemptBrief(_M):
    attempt_id: str
    seq: int
    phase: str
    scheduler_state: str
    job_id: str | None
    time_left_s: int | None
    pending_reason: str | None
    pending_category: str | None
    progress: str | None
    health: str  # one char: see services.health.flag_char
    observation_gap: bool
    cancel_state: str
    last_observed_at: str | None


class RunRow(_M):
    run_id: str
    label: str
    kind: str
    cell: dict[str, Scalar]
    seed: int | None
    status: str
    code: str
    reason: str | None
    attempts: int
    latest_attempt: AttemptBrief | None
    observed_at: str | None
    stale: bool


class ForeignJobs(_M):
    """The user's own non-hpclab jobs on a cluster (counts only; not actionable here)."""

    cluster_id: str
    count: int
    running: int
    pending: int
    gpus: int
    gpus_unknown_jobs: int
    partitions: dict[str, int]
    unbound_hpclab_jobs: int
    observed_at: str | None
    stale: bool


class GridCell(_M):
    code: str
    run_id: str | None = None
    runs: int = 1
    value: float | None = None
    reason: str | None = None


class GridRow(_M):
    key: str
    values: dict[str, Scalar]
    cells: list[GridCell]


class GridData(_M):
    row_axes: list[str]
    col_axes: list[str]
    cols: list[str]
    rows: list[GridRow]
    total_rows: int
    offset: int
    metric: str | None
    legend: dict[str, str]
    notes: list[str] = Field(default_factory=list)


class RunsListData(_M):
    view: Literal["list", "grid"]
    runs: list[RunRow] = Field(default_factory=list)
    grid: GridData | None = None
    total: int
    counts: dict[str, int]
    foreign_jobs: list[ForeignJobs] = Field(default_factory=list)
    changes_cursor: str | None = None
    coverage: Coverage
    observed_at: str | None = None
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        if self.view == "grid" and self.grid is not None:
            base = f"grid {len(self.grid.rows)}/{self.grid.total_rows} rows x {len(self.grid.cols)} cols"
        else:
            base = f"{len(self.runs)} of {self.total} runs"
        codes = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()) if k not in ("planned", "valid"))
        if self.counts.get("E"):
            codes += " (E = completed, evaluation pending; not failed)"
        return f"{base}; {codes}" if codes else base


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _poll_interval(ctx: ServiceContext, cluster_id: str) -> float:
    """The cluster's minimum scheduler poll interval (the settings override wins)."""
    if ctx.settings.min_poll_interval_s is not None:
        return float(ctx.settings.min_poll_interval_s)
    try:
        return float(ctx.clusters.get(cluster_id).profile.policy.poll_min_interval_s)
    except HpcLabError:
        return 30.0


def _stale_after(ctx: ServiceContext, cluster_id: str) -> float:
    return max(120.0, 4.0 * _poll_interval(ctx, cluster_id))


def _statuses(ctx: ServiceContext, runs: Sequence[Row]) -> dict[str, CellStatus]:
    by_cluster: dict[str, list[str]] = {}
    for r in runs:
        by_cluster.setdefault(r["cluster_id"], []).append(r["run_id"])
    out: dict[str, CellStatus] = {}
    now = ctx.clock.now()
    for cid, ids in by_cluster.items():
        out.update(cell_statuses(ctx.db, ids, stale_after_s=_stale_after(ctx, cid), now=now))
    return out


def _squeue_coverage(
    ctx: ServiceContext, cov: Coverage, cluster_ids: Sequence[str], active: set[str]
) -> str | None:
    """One source per cluster from poll_state; returns the oldest last-ok time of active clusters."""
    now = ctx.clock.now()
    oldest: str | None = None
    for cid in sorted(set(cluster_ids)):
        row = ctx.db.one("SELECT * FROM poll_state WHERE cluster_id=? AND source='squeue'", (cid,))
        name = f"squeue:{cid}"
        if cid not in active:
            cov.add(name, "skipped", row["last_ok_at"] if row else None, "no active attempts")
            continue
        last_ok = row["last_ok_at"] if row else None
        if row is None or last_ok is None:
            cov.add(name, "stale", None, "never polled successfully")
            continue
        if oldest is None or last_ok < oldest:
            oldest = last_ok
        if row["consecutive_failures"]:
            cov.add(name, "unavailable", last_ok, _clip(row["last_error"]))
            continue
        seen = parse_iso(last_ok)
        if seen is not None and (now - seen).total_seconds() > _stale_after(ctx, cid):
            cov.add(name, "stale", last_ok, "last successful poll is older than the freshness bound")
        else:
            cov.add(name, "ok", last_ok)
    return oldest


def _foreign(ctx: ServiceContext, cluster_ids: Sequence[str]) -> list[ForeignJobs]:
    out: list[ForeignJobs] = []
    now = ctx.clock.now()
    for cid in sorted(set(cluster_ids)):
        row = ctx.db.one(
            "SELECT value_json, observed_at FROM observations WHERE cluster_id=? AND obs_key='queue:foreign_jobs'",
            (cid,),
        )
        if row is None:
            continue
        v = jload(row["value_json"], {}) or {}
        seen = parse_iso(row["observed_at"])
        out.append(
            ForeignJobs(
                cluster_id=cid,
                count=int(v.get("count", 0)),
                running=int(v.get("running", 0)),
                pending=int(v.get("pending", 0)),
                gpus=int(v.get("gpus", 0)),
                gpus_unknown_jobs=int(v.get("gpus_unknown_jobs", 0)),
                partitions={str(k): int(n) for k, n in (v.get("partitions") or {}).items()},
                unbound_hpclab_jobs=int(v.get("unbound_hpclab_jobs", 0)),
                observed_at=row["observed_at"],
                stale=seen is None or (now - seen).total_seconds() > FOREIGN_STALE_S,
            )
        )
    return out


def _cached[M: BaseModel](model: type[M], raw: str | None) -> M | None:
    """A cached derived view (progress/health JSON) or None when absent/unreadable."""
    doc = jload(raw, None)
    if not isinstance(doc, dict):
        return None
    try:
        return model.model_validate(doc)
    except ValueError:
        return None


def _brief(a: Mapping[str, Any], now_epoch: float) -> AttemptBrief:
    progress = _cached(Progress, a.get("progress_json"))
    progress_text = progress.short_summary() if progress is not None else None
    health = _cached(Health, a.get("health_json"))
    if health is not None and (health.as_of is None or now_epoch - health.as_of > HEALTH_STALE_S):
        health = None  # an old "ok" is not a current "ok": show '-' (no current data)
    pending = a.get("pending_reason") if a.get("scheduler_state") == SchedulerState.PENDING.value else None
    active = a["phase"] in OBSERVED_PHASES
    return AttemptBrief(
        attempt_id=a["attempt_id"],
        seq=a["seq"],
        phase=a["phase"],
        scheduler_state=a["scheduler_state"],
        job_id=a.get("job_id"),
        time_left_s=a.get("time_left_s"),
        pending_reason=_clip(pending),
        pending_category=decode_pending(pending).category if active and pending is not None else None,
        progress=progress_text,
        health=flag_char(health) if active else "-",
        observation_gap=bool(a.get("observation_gap")),
        cancel_state=a.get("cancel_state") or "none",
        last_observed_at=a.get("last_observed_at"),
    )


def _cell(r: Mapping[str, Any]) -> dict[str, Scalar]:
    """Full matrix coordinates (for grouping; use ``_display_cell`` in responses)."""
    cell = jload(r.get("cell_key"), {}) or {}
    return {str(k): v for k, v in cell.items()} if isinstance(cell, dict) else {}


def _clip_label(value: Any, n: int = MAX_LABEL) -> str:
    """Display text of an axis value/label; long values keep a short digest so clipped labels of
    different values stay distinguishable."""
    text = str(value)
    if len(text) <= n:
        return text
    return f"{text[: n - 10]}~{sha256_hex(text.encode())[:8]}"


def _display_cell(r: Mapping[str, Any]) -> dict[str, Scalar]:
    return {
        k: (_clip_label(v, MAX_TEXT) if isinstance(v, str) else v) for k, v in list(_cell(r).items())[:32]
    }


def _json_size(model: BaseModel) -> int:
    """Serialized size as the transport sends it (ASCII-escaped JSON: the worst case)."""
    return len(json.dumps(model.model_dump(mode="json")))


def _byte_bounded[M: BaseModel](items: list[M], budget: int) -> list[M]:
    """The longest prefix of ``items`` whose JSON fits ``budget`` bytes (always at least one)."""
    out: list[M] = []
    used = 0
    for item in items:
        size = _json_size(item) + 1
        if out and used + size > budget:
            break
        out.append(item)
        used += size
    return out


# ---------------------------------------------------------------------------
# runs_list
# ---------------------------------------------------------------------------


def _parse_changed_since(value: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)
    data = decode_cursor(text)
    seq = data.get("seq")
    if data.get("k") != "chg" or not isinstance(seq, int) or seq < 0:
        raise InvalidArgument("changed_since must be an events sequence number or a changes cursor")
    return seq


def runs_list(
    ctx: ServiceContext,
    *,
    experiment_id: str | None = None,
    workflow_id: str | None = None,
    states: list[RunStatus] | None = None,
    cursor: str | None = None,
    limit: int | None = None,
    view: Literal["list", "grid"] = "list",
    rows: list[str] | None = None,
    cols: list[str] | None = None,
    metric: str | None = None,
    changed_since: str | None = None,
) -> RunsListData:
    """Runs managed by hpclab with their cell code, latest attempt and freshness (list), or a
    rows x cols grid of cell codes (+ the selected valid evaluation's metric). ``changed_since``
    (an events sequence number or a previous ``changes_cursor``) returns only runs that changed."""
    lim = clamp_limit(limit)
    where: list[str] = []
    params: list[Any] = []
    if experiment_id is not None:
        if ctx.db.one("SELECT 1 AS x FROM experiments WHERE experiment_id=?", (experiment_id,)) is None:
            raise NotFound(f"experiment {experiment_id} not found")
        where.append("r.experiment_id = ?")
        params.append(experiment_id)
    if workflow_id is not None:
        if ctx.db.one("SELECT 1 AS x FROM workflows WHERE workflow_id=?", (workflow_id,)) is None:
            raise NotFound(f"workflow {workflow_id} not found")
        where.append("r.run_id IN (SELECT run_id FROM workflow_cells WHERE workflow_id = ?)")
        params.append(workflow_id)
    if states:
        where.append(f"r.status IN ({','.join('?' for _ in states)})")
        params.extend(RunStatus(s).value for s in states)
    cur = decode_cursor(cursor)
    if cur.get("k") == "chg" and cursor is not None:
        # a changes cursor continues a changed_since listing: it supersedes the original
        # changed_since (clients pass next_cursor back with the same other parameters)
        changed_since, cursor, cur = cursor, None, {}
    if view == "grid":
        if experiment_id is None and workflow_id is None:
            raise InvalidArgument("the grid view needs experiment_id or workflow_id")
        if changed_since is not None:
            raise InvalidArgument("changed_since applies to the list view only")
        return _grid(ctx, where, params, rows, cols, metric, cur, lim, scope_ref=workflow_id or experiment_id)
    if rows or cols or metric:
        raise InvalidArgument("rows/cols/metric apply to view='grid' only")
    base = "SELECT r.*, e.cluster_id AS cluster_id FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id"
    cond = (" WHERE " + " AND ".join(where)) if where else ""
    warnings: list[str] = []
    changes_cursor: str | None = None
    next_cursor: str | None = None
    seqs: dict[str, int] = {}
    # The watermark is read BEFORE the rows: a change committed while this listing is being read
    # then has a higher seq and shows up in the next changed_since call instead of being lost.
    if changed_since is not None:
        if cur:
            raise InvalidArgument("cursor does not belong to this changed_since listing")
        since = _parse_changed_since(changed_since)
        watermark = _max_seq(ctx, since)
        page, seqs, full = _changed_page(ctx, base, where, params, since, watermark, lim)
    else:
        if cur and cur.get("k") != "list":
            raise InvalidArgument("cursor does not belong to this listing")
        watermark = _max_seq(ctx, 0)
        page_where = list(where)
        page_params = list(params)
        if cur:
            page_where.append("(r.created_at > ? OR (r.created_at = ? AND r.run_id > ?))")
            page_params.extend([cur.get("c"), cur.get("c"), cur.get("r")])
        page_cond = (" WHERE " + " AND ".join(page_where)) if page_where else ""
        page = ctx.db.all(
            f"{base}{page_cond} ORDER BY r.created_at, r.run_id LIMIT ?", [*page_params, lim + 1]
        )
        full = len(page) > lim
        page = page[:lim]
        count = ctx.db.one(f"SELECT COUNT(*) AS n FROM runs r{cond}", params)
        total = int(count["n"]) if count else 0
    statuses = _statuses(ctx, page)
    attempt_ids = [s.latest_attempt_id for s in statuses.values() if s.latest_attempt_id]
    attempts = _attempts_by_id(ctx, attempt_ids)
    now_epoch = ctx.clock.now().timestamp()
    out_rows = _byte_bounded(
        [_row(r, statuses[r["run_id"]], attempts, now_epoch) for r in page], LIST_MAX_BYTES
    )
    if len(out_rows) < len(page):
        full = True
        page = page[: len(out_rows)]
        statuses = {r["run_id"]: statuses[r["run_id"]] for r in page}
        warnings.append(f"page cut at {len(out_rows)} runs to stay under the response size bound")
    if changed_since is not None:
        if full and page:
            watermark = seqs[page[-1]["run_id"]]
        changes_cursor = encode_cursor({"k": "chg", "seq": watermark})
        next_cursor = changes_cursor if full else None
        total = len(page)
    else:
        changes_cursor = encode_cursor({"k": "chg", "seq": watermark})
        if full and page:
            last = page[-1]
            next_cursor = encode_cursor({"k": "list", "c": last["created_at"], "r": last["run_id"]})
    counts = _counts(ctx, cond, params, total, statuses, warnings)
    clusters = sorted({r["cluster_id"] for r in page}) or _clusters_for_scope(ctx, experiment_id)
    active = {r["cluster_id"] for r in page if statuses[r["run_id"]].phase in OBSERVED_PHASES}
    cov = Coverage()
    cov.add("ledger", "ok", iso(ctx.clock.now()))
    observed_at = _squeue_coverage(ctx, cov, clusters, active)
    stale_rows = sum(1 for s in statuses.values() if s.stale)
    if stale_rows:
        cov.notes.append(
            f"{stale_rows} run(s) show '?': their last observation is older than the freshness bound"
        )
    warnings.extend(_auth_notes(ctx, where, params))
    return RunsListData(
        view="list",
        runs=out_rows,
        total=total,
        counts=counts,
        foreign_jobs=_foreign(ctx, clusters),
        changes_cursor=changes_cursor,
        coverage=cov,
        observed_at=observed_at,
        next_cursor=next_cursor,
        warnings=warnings,
    )


def _clusters_for_scope(ctx: ServiceContext, experiment_id: str | None) -> list[str]:
    if experiment_id is not None:
        row = ctx.db.one("SELECT cluster_id FROM experiments WHERE experiment_id=?", (experiment_id,))
        return [row["cluster_id"]] if row else []
    return ctx.clusters.ids()


def _max_seq(ctx: ServiceContext, floor: int) -> int:
    row = ctx.db.one("SELECT MAX(seq) AS s FROM events")
    return max(floor, int(row["s"])) if row and row["s"] is not None else floor


def _changed_page(
    ctx: ServiceContext,
    base: str,
    where: list[str],
    params: list[Any],
    since: int,
    high: int,
    lim: int,
) -> tuple[list[Row], dict[str, int], bool]:
    """Runs with events in ``(since, high]`` (run, attempt, evaluation, artifact events), ordered
    by their newest change; returns (page, newest seq per run, page_full). Every event seq maps to
    one run, so the newest seq of the page's last run is a gap-free continuation watermark."""
    window = "e.seq > ? AND e.seq <= ?"
    changes = (
        f"SELECT e.entity_id AS run_id, e.seq FROM events e WHERE e.entity_type='run' AND {window} "
        "UNION ALL SELECT a.run_id, e.seq FROM events e JOIN attempts a ON a.attempt_id = e.entity_id "
        f"WHERE e.entity_type='attempt' AND {window} "
        "UNION ALL SELECT v.run_id, e.seq FROM events e JOIN evaluations v ON v.evaluation_id = e.entity_id "
        f"WHERE e.entity_type='evaluation' AND {window} "
        "UNION ALL SELECT t.run_id, e.seq FROM events e JOIN artifacts t ON t.artifact_id = e.entity_id "
        f"WHERE e.entity_type='artifact' AND {window}"
    )
    cond = (" AND " + " AND ".join(where)) if where else ""
    sql = (
        f"SELECT x.run_id AS changed_run, MAX(x.seq) AS last_seq FROM ({changes}) x "
        f"JOIN runs r ON r.run_id = x.run_id WHERE 1=1{cond} GROUP BY x.run_id ORDER BY last_seq LIMIT ?"
    )
    changed = ctx.db.all(sql, [*([since, high] * 4), *params, lim + 1])
    full = len(changed) > lim
    changed = changed[:lim]
    if not changed:
        return [], {}, False
    ids = [c["changed_run"] for c in changed]
    marks = ",".join("?" for _ in ids)
    rows = {r["run_id"]: r for r in ctx.db.all(f"{base} WHERE r.run_id IN ({marks})", ids)}
    page = [rows[i] for i in ids if i in rows]
    return page, {c["changed_run"]: int(c["last_seq"]) for c in changed}, full


def _attempts_by_id(ctx: ServiceContext, ids: list[str]) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        for a in ctx.db.all(f"SELECT * FROM attempts WHERE attempt_id IN ({marks})", chunk):
            out[a["attempt_id"]] = a
    return out


def _row(r: Row, st: CellStatus, attempts: Mapping[str, Row], now_epoch: float) -> RunRow:
    a = attempts.get(st.latest_attempt_id or "")
    return RunRow(
        run_id=r["run_id"],
        label=_clip(r["label"], 120),
        kind=r["kind"],
        cell=_display_cell(r),
        seed=r["seed"],
        status=r["status"],
        code=st.code,
        reason=_clip(st.reason),
        attempts=st.attempts,
        latest_attempt=_brief(a, now_epoch) if a else None,
        observed_at=st.observed_at,
        stale=st.stale,
    )


def _counts(
    ctx: ServiceContext,
    cond: str,
    params: list[Any],
    total: int,
    page_statuses: Mapping[str, CellStatus],
    warnings: list[str],
) -> dict[str, int]:
    """Codes of every run in scope (up to MAX_COUNT_RUNS). The scope can be the whole ledger, so
    only run ids are loaded for it and statuses are computed COUNT_BATCH_RUNS at a time: one
    batch's attempt rows (with their JSON blobs) are held at once, not every run's."""
    if total > MAX_COUNT_RUNS:
        warnings.append(f"counts cover this page only ({total} runs match; narrow the filter)")
        return summarize_codes(page_statuses.values())
    scope = ctx.db.all(
        "SELECT r.run_id, e.cluster_id AS cluster_id FROM runs r "
        f"JOIN experiments e ON e.experiment_id = r.experiment_id{cond}",
        params,
    )

    def batched() -> Iterator[CellStatus]:
        for i in range(0, len(scope), COUNT_BATCH_RUNS):
            yield from _statuses(ctx, scope[i : i + COUNT_BATCH_RUNS]).values()

    return summarize_codes(batched())


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def _sort_key(value: Scalar) -> tuple[int, float, str]:
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, bool):
        return (1, 0.0, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def _label(axes: list[str], values: tuple[Scalar, ...]) -> str:
    if not axes:
        return "all"
    return ",".join(f"{k}={v}" for k, v in zip(axes, values, strict=True))


def _grid(
    ctx: ServiceContext,
    where: list[str],
    params: list[Any],
    rows_axes: list[str] | None,
    cols_axes: list[str] | None,
    metric: str | None,
    cur: Mapping[str, Any],
    lim: int,
    scope_ref: str | None = None,
) -> RunsListData:
    if cur and cur.get("k") != "grid":
        raise InvalidArgument("cursor does not belong to this grid")
    offset = int(cur.get("o", 0)) if cur else 0
    if offset < 0:
        raise InvalidArgument("malformed grid cursor")
    cond = " AND ".join([*where, "r.kind = 'main'"])
    runs = ctx.db.all(
        "SELECT r.*, e.cluster_id AS cluster_id FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id"
        f" WHERE {cond} ORDER BY r.created_at, r.run_id LIMIT ?",
        [*params, MAX_GRID_RUNS + 1],
    )
    warnings: list[str] = []
    if len(runs) > MAX_GRID_RUNS:
        raise InvalidArgument(f"the grid is limited to {MAX_GRID_RUNS} runs; filter by workflow or state")
    cells = {r["run_id"]: _cell(r) for r in runs}
    axes: list[str] = sorted({k for c in cells.values() for k in c})
    varying = [k for k in axes if len({repr(c.get(k)) for c in cells.values()}) > 1]
    for name, given in (("rows", rows_axes), ("cols", cols_axes)):
        unknown = [a for a in (given or []) if a not in axes]
        if unknown:
            raise InvalidArgument(f"unknown {name} axes {unknown}", details={"axes": axes})
    col_axes = list(cols_axes) if cols_axes is not None else (["seed"] if "seed" in axes else [])
    if rows_axes is not None:
        row_axes = list(rows_axes)
    else:
        row_axes = [a for a in varying if a not in col_axes] or [a for a in axes if a not in col_axes]
    if set(row_axes) & set(col_axes):
        raise InvalidArgument("rows and cols axes must be disjoint")
    hidden = [a for a in varying if a not in row_axes and a not in col_axes]
    notes: list[str] = []
    if hidden:
        notes.append(f"axes {hidden} are not shown: cells aggregate runs that differ in them")
    notes.extend(_dispatch_notes(ctx, where, params))
    notes.extend(_auth_notes(ctx, where, params))
    statuses = _statuses(ctx, runs)
    awaiting = [s for s in statuses.values() if s.code == "E"]
    if awaiting:
        note = (
            f"{len(awaiting)} run(s) are E: completed, evaluation pending (outputs being checked or not "
            "evaluated yet) - not a failure; they turn V once a valid evaluation exists"
        )
        # outputs valid but nothing will evaluate them: no evaluation declared in the spec, or the
        # automatic one was skipped (never retried) - they would stay E forever
        ready = [s.latest_attempt_id for s in awaiting if s.result_state == ResultState.VALID.value]
        queued = _evaluation_queued(ctx) if ready else set()
        orphaned = [a for a in ready if a and a not in queued]
        if orphaned and scope_ref:
            note += (
                f"; {len(orphaned)} of them {'has' if len(orphaned) == 1 else 'have'} no evaluation queued "
                "(none declared in the spec, or the automatic one was skipped): "
                + call("evaluation_collect", run_set_ref=scope_ref, idempotency_key=NEW_KEY)
            )
        notes.append(note)
    col_keys = sorted(
        {tuple(cells[r["run_id"]].get(a) for a in col_axes) for r in runs},
        key=lambda t: tuple(_sort_key(v) for v in t),
    )
    if len(col_keys) > MAX_GRID_COLS:
        raise InvalidArgument(
            f"the grid would have {len(col_keys)} columns (at most {MAX_GRID_COLS}); put the axes with "
            "many values in rows (rows are paginated) or filter the runs",
            details={"cols": col_axes, "columns": len(col_keys), "axes": axes},
        )
    row_keys = sorted(
        {tuple(cells[r["run_id"]].get(a) for a in row_axes) for r in runs},
        key=lambda t: tuple(_sort_key(v) for v in t),
    )
    members: dict[tuple[tuple[Scalar, ...], tuple[Scalar, ...]], list[str]] = {}
    for r in runs:
        c = cells[r["run_id"]]
        key = (tuple(c.get(a) for a in row_axes), tuple(c.get(a) for a in col_axes))
        members.setdefault(key, []).append(r["run_id"])
    col_labels = [_clip_label(_label(col_axes, ck)) for ck in col_keys]
    out_rows: list[GridRow] = []
    used = len(json.dumps(col_labels)) + len(json.dumps(notes))  # the header shares the page budget
    for rk in row_keys[offset:]:
        grid_row = GridRow(
            key=_clip_label(_label(row_axes, rk), 2 * MAX_LABEL),
            values={
                a: (_clip_label(v) if isinstance(v, str) else v) for a, v in zip(row_axes, rk, strict=True)
            },
            cells=[_grid_cell(members.get((rk, ck), []), statuses, metric) for ck in col_keys],
        )
        size = _json_size(grid_row) + 1
        if out_rows and (len(out_rows) >= lim or used + size > GRID_MAX_BYTES):
            break
        out_rows.append(grid_row)
        used += size
    next_offset = offset + len(out_rows)
    next_cursor = encode_cursor({"k": "grid", "o": next_offset}) if next_offset < len(row_keys) else None
    if metric and not any(s.metrics.get(metric) is not None for s in statuses.values()):
        warnings.append(f"no selected valid evaluation reports metric {metric!r}")
    clusters = sorted({r["cluster_id"] for r in runs})
    active = {r["cluster_id"] for r in runs if statuses[r["run_id"]].phase in OBSERVED_PHASES}
    cov = Coverage()
    cov.add("ledger", "ok", iso(ctx.clock.now()))
    observed_at = _squeue_coverage(ctx, cov, clusters, active)
    return RunsListData(
        view="grid",
        grid=GridData(
            row_axes=row_axes,
            col_axes=col_axes,
            cols=col_labels,
            rows=out_rows,
            total_rows=len(row_keys),
            offset=offset,
            metric=metric,
            legend=LEGEND,
            notes=notes,
        ),
        total=len(runs),
        counts=summarize_codes(statuses.values()),
        foreign_jobs=_foreign(ctx, clusters),
        coverage=cov,
        observed_at=observed_at,
        next_cursor=next_cursor,
        warnings=warnings,
    )


_STOPPED_WORKFLOW_STATES = ("gate_failed", "cancelling", "cancelled")


def _dispatch_notes(ctx: ServiceContext, where: list[str], params: list[Any]) -> list[str]:
    """What the main-run grid cannot show on its own: smoke-gate runs of the same scope that are
    not valid (main cells wait for them) and workflows whose dispatch stopped or paused - without
    this, a grid of 'P' cells looks like work that is merely still queued."""
    notes: list[str] = []
    smoke_cond = " AND ".join([*where, "r.kind = 'smoke'"])
    smoke = ctx.db.all(
        "SELECT r.*, e.cluster_id AS cluster_id FROM runs r JOIN experiments e ON e.experiment_id = r.experiment_id"
        f" WHERE {smoke_cond} ORDER BY r.created_at, r.run_id LIMIT 5",
        params,
    )
    for r in smoke:
        st = _statuses(ctx, [r])[r["run_id"]]
        passed = st.code == "V" or (st.scheduler_state == "COMPLETED" and st.result_state == "valid")
        if not passed:
            why = f" {st.reason}" if st.reason else ""
            notes.append(
                f"smoke run {_clip(r['label'], 80)} ({r['run_id']}): {st.code}{why} - main cells of its workflow "
                "are dispatched only after a smoke run passes"
            )
    cond = " AND ".join(where) or "1=1"
    workflows = ctx.db.all(
        "SELECT DISTINCT w.workflow_id, w.state, w.dispatch_state FROM workflows w"
        " JOIN workflow_cells c ON c.workflow_id = w.workflow_id JOIN runs r ON r.run_id = c.run_id"
        f" WHERE {cond} AND (w.state IN ({','.join('?' for _ in _STOPPED_WORKFLOW_STATES)})"
        " OR w.dispatch_state = 'paused') ORDER BY w.created_at LIMIT 5",
        [*params, *_STOPPED_WORKFLOW_STATES],
    )
    for w in workflows:
        notes.append(
            f"workflow {w['workflow_id']}: state {w['state']}, dispatch {w['dispatch_state']} - "
            "its undispatched cells will not start on their own"
        )
    return notes


def _evaluation_queued(ctx: ServiceContext) -> set[str]:
    """Attempt ids with an evaluation waiting in the outbox (automatic, or a background batch)."""
    out: set[str] = set()
    for r in ctx.db.all(
        "SELECT kind, target_id, payload_json FROM outbox WHERE state IN ('pending', 'leased')"
        " AND kind IN ('evaluation.auto', 'evaluation.collect')"
    ):
        payload = jload(r["payload_json"], {}) or {}
        payload = payload if isinstance(payload, dict) else {}
        if r["kind"] == "evaluation.auto":
            out.add(str(payload.get("attempt_id") or r["target_id"] or ""))
        else:
            out.update(str(a) for a in payload.get("attempt_ids") or [])
    return out


_SUBMISSION_PHASES = (
    AttemptPhase.STAGING.value,
    AttemptPhase.SUBMISSION_IN_FLIGHT.value,
    AttemptPhase.SUBMISSION_UNCERTAIN.value,
)


def _auth_notes(ctx: ServiceContext, where: list[str], params: list[Any]) -> list[str]:
    """Submissions in scope whose submit operation is blocked on SSH authentication (MFA /
    expired ControlMaster), per cluster: without this a list of 'S' rows looks like progress."""
    cond = " AND ".join(where) or "1=1"
    marks = ",".join("?" for _ in _SUBMISSION_PHASES)
    # A changed host key is parked on stage auth_required as well, but not retryable
    # (submission._blocked): logging in again without verifying the key is how a MITM gets in.
    host_key = (
        "(json_extract(o.error_json, '$.retryable') = 0"
        " OR json_extract(o.error_json, '$.details.remediation') LIKE '%host key%')"
    )
    rows = ctx.db.all(
        f"SELECT a.cluster_id, {host_key} AS host_key, COUNT(DISTINCT a.attempt_id) AS n FROM attempts a"
        " JOIN runs r ON r.run_id = a.run_id"
        " JOIN operations o ON o.operation_id = a.operation_id"
        " OR (o.kind = 'submit' AND o.target_type = 'attempt' AND o.target_id = a.attempt_id)"
        f" WHERE {cond} AND a.phase IN ({marks}) AND o.state = 'blocked'"
        " AND (o.stage = 'auth_required' OR json_extract(o.error_json, '$.code') = 'auth_required')"
        " GROUP BY a.cluster_id, 2 ORDER BY a.cluster_id, 2",
        [*params, *_SUBMISSION_PHASES],
    )
    notes: list[str] = []
    for r in rows:
        cid = r["cluster_id"]
        if not r["n"]:
            continue
        if r["host_key"]:
            notes.append(
                f"{r['n']} submission(s) blocked because the SSH host key of {cid} changed: verify the new key "
                f"with your cluster admins before you run `hpclab auth {cid}`; nothing resumes until then"
            )
        else:
            notes.append(
                f"{r['n']} submission(s) waiting for SSH login on {cid}: run `hpclab auth {cid}`; "
                "they resume by themselves"
            )
    return notes


def _grid_cell(run_ids: list[str], statuses: Mapping[str, CellStatus], metric: str | None) -> GridCell:
    if not run_ids:
        return GridCell(code="-", runs=0)
    if len(run_ids) == 1:
        st = statuses[run_ids[0]]
        value = st.metrics.get(metric) if metric and st.code == "V" else None
        return GridCell(
            code=st.code,
            run_id=st.run_id,
            value=float(value) if isinstance(value, (int, float)) else None,
            reason=_clip(st.reason, 120),
        )
    codes = [statuses[r].code for r in run_ids]
    worst = min(codes, key=lambda c: _CODE_SEVERITY.index(c) if c in _CODE_SEVERITY else 0)
    tally = " ".join(
        f"{c}x{n}"
        for c, n in sorted(summarize_codes(statuses[r] for r in run_ids).items())
        if c not in ("planned", "valid")
    )
    return GridCell(code=worst, runs=len(run_ids), reason=_clip(f"{len(run_ids)} runs: {tally}", 120))


# ---------------------------------------------------------------------------
# run_inspect
# ---------------------------------------------------------------------------


class JobIdentity(_M):
    cluster_id: str
    job_id: str | None
    owner: str | None
    submit_time: str | None
    restart_count: int
    array_task_id: str | None
    node_list: str | None
    partition: str | None
    gpu_type: str | None


class SegmentView(_M):
    segment: int
    job_id: str | None
    start_time: str | None
    end_time: str | None
    state: str | None
    node_list: str | None


class BudgetView(_M):
    reserved: float | None
    estimated_consumed: float | None
    settled_consumed: float | None
    state: str
    note: str | None


class AttemptView(_M):
    attempt_id: str
    seq: int
    version: int
    phase: str
    scheduler_state: str
    raw_scheduler_state: str | None
    app_state: str
    result_state: str
    cancel_state: str
    cancel_mode: str | None
    cancel_reason: str | None
    job: JobIdentity
    segments: list[SegmentView]
    exit_code: int | None
    exit_signal: int | None
    reason: str | None
    start_time: str | None
    end_time: str | None
    elapsed_s: int | None
    time_limit_s: int | None
    time_left_s: int | None
    gpus: int
    dispatch_at: str | None
    identified_at: str | None
    terminal_at: str | None
    last_observed_at: str | None
    observation_gap: bool
    uncertain_since: str | None
    workflow_id: str | None
    operation_id: str | None
    exec_plan_hash: str
    created_reason: str | None
    resolution: str | None
    budget: BudgetView | None
    diagnosis_code: str | None


class OperationSummary(_M):
    operation_id: str
    ref: str
    kind: str
    state: str
    stage: str | None
    target_type: str | None
    target_id: str | None
    progress: dict[str, Any]
    error: dict[str, Any] | None
    poll_after_ms: int
    updated_at: str


class RunSummary(_M):
    run_id: str
    label: str
    kind: str
    experiment_id: str
    plan_id: str
    cell: dict[str, Scalar]
    seed: int | None
    status: str
    code: str
    code_reason: str | None
    spec_hash: str
    derived_from: str | None
    reproduction_of: str | None
    added_post_hoc: bool
    claim_attempt_id: str | None
    current_exec_plan_hash: str | None
    created_at: str


class SpecSummary(_M):
    argv: list[str]
    workdir: str | None
    params: dict[str, Any]
    outputs: list[dict[str, Any]]
    runtime: dict[str, Any]
    inputs: list[dict[str, Any]]
    evaluation: dict[str, Any] | None


class Refs(_M):
    plan_id: str
    plan_ref: str
    snapshot_digest: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    exec_plan_hash: str | None
    resources: dict[str, Any]
    profile_revision: str | None


class EnvView(_M):
    declared: dict[str, Any]
    observed: dict[str, Any] | None
    drift: list[str]
    source: Literal["env.json", "none"]


class ArtifactsSummary(_M):
    attempt_id: str
    total: int
    by_integrity: dict[str, int]
    required_missing: list[str]
    result_state: str


class EvaluationRef(_M):
    evaluation_id: str
    ref: str
    attempt_id: str
    evaluator: str
    status: str
    selected: bool
    created_at: str


class GpuSampleView(_M):
    index: int
    name: str | None
    util_percent: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    power_w: float | None


class GpuProbeView(_M):
    status: Literal["ok", "rate_limited", "refused", "not_running", "unavailable", "unsupported"]
    samples: list[GpuSampleView] = Field(default_factory=list)
    observed_at: str | None = None
    next_allowed_at: str | None = None
    detail: str | None = None


class RunInspectData(_M):
    run: RunSummary
    spec: SpecSummary
    refs: Refs
    focus_attempt_id: str | None
    attempts: list[AttemptView]
    attempts_total: int
    operations: list[OperationSummary]
    pending: PendingInfo | None = None
    progress: Progress | None = None
    health: Health | None = None
    diagnosis: Diagnosis | None = None
    env: EnvView | None = None
    artifacts: ArtifactsSummary | None = None
    evaluations: list[EvaluationRef] = Field(default_factory=list)
    gpu_probe: GpuProbeView | None = None
    recommended_next: list[str] = Field(default_factory=list)
    stale: bool = False
    coverage: Coverage
    observed_at: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        focus = next((a for a in self.attempts if a.attempt_id == self.focus_attempt_id), None)
        head = f"run {self.run.label} [{self.run.code}] {self.run.status}"
        if focus is not None:
            head += f"; attempt #{focus.seq} {focus.phase}/{focus.scheduler_state}"
            if focus.job.job_id:
                head += f" job {focus.job.job_id}"
        if self.diagnosis is not None and self.diagnosis.primary_code:
            head += f"; diagnosis {self.diagnosis.primary_code}"
        return head + (" (stale)" if self.stale else "")


def _resolve(
    ctx: ServiceContext, run_id: str | None, attempt_id: str | None, job_ref: JobRef | None
) -> tuple[str, str | None]:
    given = sum(x is not None for x in (run_id, attempt_id, job_ref))
    if given != 1:
        raise InvalidArgument("give exactly one of run_id, attempt_id or job_ref")
    if attempt_id is not None:
        a = ctx.db.one("SELECT run_id FROM attempts WHERE attempt_id=?", (attempt_id,))
        if a is None:
            raise NotFound(f"attempt {attempt_id} not found", details={"attempt_id": attempt_id})
        return a["run_id"], attempt_id
    if job_ref is not None:
        rows = ctx.db.all(
            "SELECT attempt_id, run_id, job_submit_time, restart_count FROM attempts WHERE cluster_id=? AND job_id=?"
            " ORDER BY created_at",
            (job_ref.cluster_id, job_ref.job_id),
        )
        if job_ref.submit_time is not None:
            try:
                want = parse_iso(job_ref.submit_time)
            except ValueError:
                want = None
            if want is None:
                raise InvalidArgument(
                    "job_ref.submit_time is not an ISO timestamp",
                    details={"submit_time": job_ref.submit_time, "example": "2026-09-01T11:00:00Z"},
                )
            rows = [
                r
                for r in rows
                if (t := parse_iso(r["job_submit_time"])) is not None
                and abs((t - want).total_seconds()) <= JOB_REF_SUBMIT_TOLERANCE_S
            ]
        if not rows:
            raise NotFound(
                f"no hpclab attempt is bound to job {job_ref.job_id} on {job_ref.cluster_id}",
                details={
                    "job_ref": job_ref.model_dump(),
                    "hint": "hpclab only tracks jobs it submitted; job ids are not global identities",
                },
            )
        if len(rows) > 1:
            raise Ambiguous(
                f"{len(rows)} attempts share job id {job_ref.job_id} on {job_ref.cluster_id}; give submit_time",
                details={
                    "candidates": [
                        {
                            "attempt_id": r["attempt_id"],
                            "run_id": r["run_id"],
                            "submit_time": r["job_submit_time"],
                        }
                        for r in rows[:20]
                    ]
                },
            )
        return rows[0]["run_id"], rows[0]["attempt_id"]
    assert run_id is not None
    return run_id, None


def run_inspect(
    ctx: ServiceContext,
    *,
    run_id: str | None = None,
    attempt_id: str | None = None,
    job_ref: JobRef | None = None,
    refresh: bool = False,
    probe: Literal["gpu"] | None = None,
) -> RunInspectData:
    """Everything known about one Run (focused on one attempt): spec + refs, attempt history with
    the three state axes and job identity, operations, pending reason, progress, health,
    diagnosis, environment drift, artifacts/evaluations and deterministic next-step hints.
    ``refresh=True`` does a bounded poll first (stale flag on timeout); ``probe='gpu'`` samples
    nvidia-smi inside a RUNNING job's allocation (identity-checked, at most every 10 minutes)."""
    rid, focus_id = _resolve(ctx, run_id, attempt_id, job_ref)
    run = ctx.db.one("SELECT * FROM runs WHERE run_id=?", (rid,))
    if run is None:
        raise NotFound(f"run {rid} not found", details={"run_id": rid})
    cov = Coverage()
    cov.add("ledger", "ok", iso(ctx.clock.now()))
    warnings: list[str] = []
    stale = False
    latest = ctx.db.one("SELECT * FROM attempts WHERE run_id=? ORDER BY seq DESC LIMIT 1", (rid,))
    focus_row = (
        ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (focus_id,))
        if focus_id is not None
        else latest
    )
    if refresh and focus_row is not None:
        deadline = max(1.0, min(ctx.settings.request_deadline_s, 15.0))
        rr = refresh_attempts(ctx, attempt_ids=[focus_row["attempt_id"]], deadline_s=deadline)
        stale = rr.stale
        warnings.extend(rr.notes)
        if rr.stale:
            cov.add(
                "refresh",
                "stale",
                rr.observed_at,
                "refresh timed out or failed; showing the last known state",
            )
        elif rr.refreshed:
            cov.add("refresh", "ok", rr.observed_at)
        else:
            cov.add("refresh", "cached", rr.observed_at, "rate limited or nothing to poll")
        focus_row = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (focus_row["attempt_id"],))
        # the poll may have moved the attempt (and released the claim / changed runs.status in
        # the same transaction): the summary must describe the refreshed state
        run = ctx.db.one("SELECT * FROM runs WHERE run_id=?", (rid,)) or run
    attempts = ctx.db.all(
        "SELECT * FROM attempts WHERE run_id=? ORDER BY seq DESC LIMIT ?", (rid, MAX_ATTEMPTS_SHOWN)
    )
    total_row = ctx.db.one("SELECT COUNT(*) AS n FROM attempts WHERE run_id=?", (rid,))
    cluster_row = ctx.db.one(
        "SELECT cluster_id FROM experiments WHERE experiment_id=?", (run["experiment_id"],)
    )
    cluster_id = cluster_row["cluster_id"] if cluster_row else (focus_row["cluster_id"] if focus_row else "")
    status = cell_statuses(ctx.db, [rid], stale_after_s=_stale_after(ctx, cluster_id), now=ctx.clock.now())[
        rid
    ]
    active = (
        {focus_row["cluster_id"]}
        if focus_row is not None and focus_row["phase"] in OBSERVED_PHASES
        else set()
    )
    observed_at = _squeue_coverage(ctx, cov, [cluster_id] if cluster_id else [], active)
    stale = stale or status.stale

    progress: Progress | None = None
    health: Health | None = None
    diagnosis: Diagnosis | None = None
    pending: PendingInfo | None = None
    env: EnvView | None = None
    artifacts: ArtifactsSummary | None = None
    gpu: GpuProbeView | None = None
    if focus_row is not None:
        progress, health, remote_ok = _live(ctx, focus_row, cov, refresh=refresh)
        if (
            focus_row["phase"] in OBSERVED_PHASES
            and focus_row["scheduler_state"] == SchedulerState.PENDING.value
        ):
            pending = decode_pending(focus_row["pending_reason"])
        diagnosis = _diagnosis_for(ctx, focus_row, cov, remote_ok=remote_ok)
        env = _env_view(ctx, run, focus_row, attempts)
        artifacts = _artifacts(ctx, focus_row)
        if probe == "gpu":
            gpu = _gpu_probe(ctx, focus_row)
    elif probe == "gpu":
        gpu = GpuProbeView(status="not_running", detail="the run has no attempt")
    evaluations = _evaluations(ctx, rid)
    views = [_attempt_view(ctx, a) for a in attempts]
    data = RunInspectData(
        run=_run_summary(run, status),
        spec=_spec_summary(run),
        refs=_refs(ctx, run, focus_row),
        focus_attempt_id=focus_row["attempt_id"] if focus_row else None,
        attempts=views,
        attempts_total=int(total_row["n"]) if total_row else len(views),
        operations=_operations(ctx, rid, attempts),
        pending=pending,
        progress=progress,
        health=health,
        diagnosis=diagnosis,
        env=env,
        artifacts=artifacts,
        evaluations=evaluations,
        gpu_probe=gpu,
        stale=stale,
        coverage=cov,
        observed_at=observed_at,
        warnings=warnings,
    )
    data.recommended_next = _recommend(
        data, focus_row, held=_workflow_hold(ctx, rid) if focus_row is None else None
    )
    return data


def _workflow_hold(ctx: ServiceContext, run_id: str) -> str | None:
    """Why a never-dispatched run is left to the workflow that still owns it (None when no open
    workflow will dispatch it): submitting it by hand would bypass that workflow's max_parallel."""
    from hpclab.services.recovery import workflow_hold_reason  # recovery imports this module

    for wf in ctx.db.all(
        "SELECT w.*, c.cell_state AS wf_cell_state FROM workflow_cells c JOIN workflows w"
        " ON w.workflow_id = c.workflow_id WHERE c.run_id = ? AND c.attempt_id IS NULL"
        " ORDER BY w.created_at DESC, w.workflow_id",
        (run_id,),
    ):
        why = workflow_hold_reason(ctx, wf, wf["wf_cell_state"])
        if why is not None:
            return why
    return None


def _run_summary(run: Row, st: CellStatus) -> RunSummary:
    return RunSummary(
        run_id=run["run_id"],
        label=_clip(run["label"], 120),
        kind=run["kind"],
        experiment_id=run["experiment_id"],
        plan_id=run["plan_id"],
        cell=_cell(run),
        seed=run["seed"],
        status=run["status"],
        code=st.code,
        code_reason=_clip(st.reason),
        spec_hash=run["spec_hash"],
        derived_from=run["derived_from"],
        reproduction_of=run.get("reproduction_of"),
        added_post_hoc=bool(run.get("added_post_hoc")),
        claim_attempt_id=run["claim_attempt_id"],
        current_exec_plan_hash=run["current_exec_plan_hash"],
        created_at=run["created_at"],
    )


def _spec_summary(run: Row) -> SpecSummary:
    spec = jload(run["spec_json"], {}) or {}
    spec = spec if isinstance(spec, dict) else {}
    entry = sub_dict(spec, "entrypoint")
    runtime = sub_dict(spec, "runtime")
    params = sub_dict(spec, "params")
    outputs = [o for o in (spec.get("outputs") or []) if isinstance(o, dict)]
    inputs = [i for i in (spec.get("inputs") or []) if isinstance(i, dict)]
    return SpecSummary(
        argv=[str(_clip(a, 200)) for a in (spec.get("argv") or [])[:40]],
        workdir=entry.get("workdir") if entry else None,
        params={str(k): _clip(v) for k, v in list(params.items())[:50]},
        outputs=[{k: o.get(k) for k in ("name", "role", "kind", "required", "path")} for o in outputs[:30]],
        runtime={
            "python": runtime.get("python"),
            "venv": runtime.get("venv"),
            "guard": runtime.get("guard"),
            "setup_lines": len(runtime.get("setup") or []),
            "env_keys": sorted((runtime.get("env") or {}).keys())[:50],
            "lockfiles": (runtime.get("lockfiles") or [])[:20],
        },
        inputs=[{k: _clip(i.get(k)) for k in ("name", "uri", "pin", "digest")} for i in inputs[:30]],
        evaluation=spec.get("evaluation") if isinstance(spec.get("evaluation"), dict) else None,
    )


def _refs(ctx: ServiceContext, run: Row, focus: Row | None) -> Refs:
    spec = jload(run["spec_json"], {}) or {}
    code = sub_dict(spec, "code")
    plan_row = ctx.db.one("SELECT snapshot_digest FROM plans WHERE plan_id=?", (run["plan_id"],))
    exec_hash = focus["exec_plan_hash"] if focus is not None else run["current_exec_plan_hash"]
    plan = plan_dict(ctx.db, exec_hash)
    res = sub_dict(plan, "resources")
    return Refs(
        plan_id=run["plan_id"],
        plan_ref=f"hpclab://plans/{run['plan_id']}",
        snapshot_digest=code.get("digest") or (plan_row["snapshot_digest"] if plan_row else None),
        git_commit=code.get("git_commit"),
        git_branch=code.get("git_branch"),
        git_dirty=code.get("git_dirty"),
        exec_plan_hash=exec_hash,
        resources={
            k: res.get(k)
            for k in ("partition", "account", "qos", "gpus", "gpu_type", "nodes", "time", "mem")
            if k in res
        },
        profile_revision=plan.get("profile_revision"),
    )


def _attempt_view(ctx: ServiceContext, a: Row) -> AttemptView:
    segs = ctx.db.all(
        "SELECT * FROM attempt_segments WHERE attempt_id=? ORDER BY segment LIMIT 50", (a["attempt_id"],)
    )
    b = ctx.db.one("SELECT * FROM budget_reservations WHERE attempt_id=?", (a["attempt_id"],))
    plan = plan_dict(ctx.db, a["exec_plan_hash"])
    diag = jload(a.get("diagnosis_json"), None)
    return AttemptView(
        attempt_id=a["attempt_id"],
        seq=a["seq"],
        version=a["version"],
        phase=a["phase"],
        scheduler_state=a["scheduler_state"],
        raw_scheduler_state=a["raw_scheduler_state"],
        app_state=a["app_state"],
        result_state=a["result_state"],
        cancel_state=a["cancel_state"],
        cancel_mode=a.get("cancel_mode"),
        cancel_reason=_clip(a["cancel_reason"]),
        job=JobIdentity(
            cluster_id=a["cluster_id"],
            job_id=a["job_id"],
            owner=a["job_owner"],
            submit_time=a["job_submit_time"],
            restart_count=int(a["restart_count"] or 0),
            array_task_id=a["array_task_id"],
            node_list=a["node_list"],
            partition=a.get("partition"),
            gpu_type=a.get("gpu_type"),
        ),
        segments=[
            SegmentView(
                segment=s["segment"],
                job_id=s["job_id"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                state=s["state"],
                node_list=s["node_list"],
            )
            for s in segs
        ],
        exit_code=a["exit_code"],
        exit_signal=a["exit_signal"],
        reason=_clip(a["reason"]),
        start_time=a["start_time"],
        end_time=a["end_time"],
        elapsed_s=a["elapsed_s"],
        time_limit_s=a["time_limit_s"],
        time_left_s=a.get("time_left_s"),
        gpus=int(a["gpus"] or 0),
        dispatch_at=a["dispatch_at"],
        identified_at=a["identified_at"],
        terminal_at=a["terminal_at"],
        last_observed_at=a["last_observed_at"],
        observation_gap=bool(a["observation_gap"]),
        uncertain_since=a["uncertain_since"],
        workflow_id=a["workflow_id"],
        operation_id=a["operation_id"],
        exec_plan_hash=a["exec_plan_hash"],
        created_reason=plan.get("created_reason"),
        resolution=_clip(a.get("resolution")),
        budget=BudgetView(
            reserved=b["reserved"],
            estimated_consumed=b["estimated_consumed"],
            settled_consumed=b["settled_consumed"],
            state=b["state"],
            note=_clip(b["note"]),
        )
        if b
        else None,
        diagnosis_code=diag.get("primary_code") if isinstance(diag, dict) else None,
    )


def _operations(ctx: ServiceContext, run_id: str, attempts: Sequence[Row]) -> list[OperationSummary]:
    ids = [run_id, *(a["attempt_id"] for a in attempts)]
    op_ids = [a["operation_id"] for a in attempts if a["operation_id"]]
    marks = ",".join("?" for _ in ids)
    sql = f"SELECT * FROM operations WHERE target_id IN ({marks})"
    params: list[Any] = list(ids)
    if op_ids:
        sql += f" OR operation_id IN ({','.join('?' for _ in op_ids)})"
        params.extend(op_ids)
    rows = ctx.db.all(sql + " ORDER BY created_at DESC LIMIT 20", params)
    return [_op_summary(r) for r in rows]


def _op_summary(r: Row) -> OperationSummary:
    return OperationSummary(
        operation_id=r["operation_id"],
        ref=f"hpclab://operations/{r['operation_id']}",
        kind=r["kind"],
        state=r["state"],
        stage=r["stage"],
        target_type=r["target_type"],
        target_id=r["target_id"],
        progress=jload(r["progress_json"], {}) or {},
        error=jload(r["error_json"], None),
        poll_after_ms=int(r["poll_after_ms"]),
        updated_at=r["updated_at"],
    )


def _ran(a: Mapping[str, Any]) -> bool:
    if a["phase"] in OBSERVED_PHASES:
        return a["scheduler_state"] not in (SchedulerState.PENDING.value, SchedulerState.UNKNOWN.value)
    if a["phase"] == AttemptPhase.TERMINAL.value:
        return bool(a.get("node_list")) or bool(a.get("elapsed_s"))
    return False


def _read_timeout(ctx: ServiceContext) -> float:
    """Per-read timeout for a synchronous request (several reads must fit its deadline)."""
    return max(3.0, min(10.0, ctx.settings.request_deadline_s / 4))


_NO_METRICS = "no metrics file yet"
_NO_TELEMETRY = "no GPU telemetry file (CPU-only job or sidecar not started)"


class _ReadSlot:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_READ_SLOTS: dict[str, _ReadSlot] = {}
_READ_SLOTS_GUARD = threading.Lock()


@contextmanager
def _live_read_slot(attempt_id: str, wait_s: float) -> Iterator[bool]:
    """One live read of an attempt's files at a time in this process: a concurrent inspect of
    the same attempt waits for it (and then reuses what it read). Yields False when the other
    read is still going after ``wait_s``."""
    with _READ_SLOTS_GUARD:
        slot = _READ_SLOTS.setdefault(attempt_id, _ReadSlot())
        slot.users += 1
    got = slot.lock.acquire(timeout=wait_s)
    try:
        yield got
    finally:
        if got:
            slot.lock.release()
        with _READ_SLOTS_GUARD:
            slot.users -= 1
            if slot.users == 0:
                _READ_SLOTS.pop(attempt_id, None)


def _live_state(a: Mapping[str, Any]) -> str:
    return f"{a['phase']}/{a['scheduler_state']}"


def _recent_live_read(ctx: ServiceContext, a: Mapping[str, Any], ttl: float) -> dict[str, Any] | None:
    """The last live read of this attempt's files if it is younger than ``ttl`` and the attempt
    is still in the state it was read in (a job that ended since is read again)."""
    row = ctx.db.one(
        "SELECT value_json, observed_at FROM observations WHERE cluster_id=? AND obs_key=?",
        (a["cluster_id"], f"{LIVE_READ_PREFIX}{a['attempt_id']}"),
    )
    if row is None:
        return None
    doc = jload(row["value_json"], None)
    seen = parse_iso(row["observed_at"])
    if not isinstance(doc, dict) or seen is None or doc.get("state") != _live_state(a):
        return None
    age = (ctx.clock.now() - seen).total_seconds()
    if not 0 <= age < ttl:  # expired, or the clock went back
        return None
    return {**doc, "observed_at": row["observed_at"], "age_s": age}


def _record_live_read(ctx: ServiceContext, a: Mapping[str, Any], views: LiveViews, ttl: float) -> None:
    doc = {
        "state": _live_state(a),
        "metrics": views.metrics_source,
        "telemetry": views.telemetry_source,
        "notes": [_clip(n) for n in views.notes[:4]],
    }
    expires = iso(ctx.clock.now() + timedelta(seconds=ttl))
    with ctx.db.tx(source="observer") as tx:
        tx.execute(
            "DELETE FROM observations WHERE cluster_id=? AND obs_key LIKE ? AND expires_at < ?",
            (a["cluster_id"], f"{LIVE_READ_PREFIX}%", tx.now),
        )
        tx.execute(
            "INSERT OR REPLACE INTO observations (cluster_id, obs_key, value_json, source, observed_at,"
            " expires_at) VALUES (?, ?, ?, 'run_inspect', ?, ?)",
            (a["cluster_id"], f"{LIVE_READ_PREFIX}{a['attempt_id']}", jdump(doc), tx.now, expires),
        )


def _replay_live_read(
    a: Mapping[str, Any], cov: Coverage, recent: Mapping[str, Any], ttl: float, *, running: bool
) -> tuple[Progress | None, Health | None, bool]:
    """The views a recent live read stored, with what that read found disclosed as of its time."""
    at = recent["observed_at"]
    reuse = (
        f"read {int(recent['age_s'])} s ago; an attempt's files are re-read at most every {ttl:g} s "
        "(refresh=true reads them now)"
    )
    notes = "; ".join(str(n) for n in recent.get("notes") or [])
    sources = [("metrics", recent.get("metrics"), _NO_METRICS)]
    if running:
        sources.append(("telemetry", recent.get("telemetry"), _NO_TELEMETRY))
    for name, found, missing in sources:
        if found == "unavailable":
            cov.add(name, "unavailable", at, _clip(f"{notes or 'the file could not be read'}; {reuse}"))
        else:
            cov.add(name, "cached", at, _clip(f"{missing}; {reuse}" if found == "missing" else reuse))
    progress = _cached(Progress, a.get("progress_json"))
    health = _cached(Health, a.get("health_json")) if running else None
    return progress, health, True


def _live(
    ctx: ServiceContext, a: Row, cov: Coverage, *, refresh: bool = False
) -> tuple[Progress | None, Health | None, bool]:
    """Progress (+ health while running) from bounded remote reads; cached views as fallback.
    A read is reused for one poll interval unless ``refresh`` (each read is several ssh execs on
    the login node, and agents poll inspect while they wait for a job); concurrent inspects of
    the attempt share one read. The flag is False when the cluster did not answer (callers skip
    further remote reads)."""
    running = a["phase"] in OBSERVED_PHASES and a["scheduler_state"] == SchedulerState.RUNNING.value
    if not _ran(a):
        cov.add("metrics", "skipped", None, "the job has not started")
        return None, None, True
    ttl = _poll_interval(ctx, a["cluster_id"])
    if not refresh:
        recent = _recent_live_read(ctx, a, ttl)
        if recent is not None:
            return _replay_live_read(a, cov, recent, ttl, running=running)
    wait = _read_timeout(ctx)
    with _live_read_slot(a["attempt_id"], wait) as got:
        if not got:
            cov.add(
                "metrics",
                "stale",
                None,
                f"another request has been reading this attempt's files for over {wait:g} s "
                "(the cluster is slow); showing the cached views",
            )
            health = _cached(Health, a.get("health_json")) if running else None
            return _cached(Progress, a.get("progress_json")), health, False
        if not refresh:  # another request may have read the files while this one waited
            current = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (a["attempt_id"],)) or a
            recent = _recent_live_read(ctx, current, ttl)
            if recent is not None:
                return _replay_live_read(current, cov, recent, ttl, running=running)
        return _read_live(ctx, a, cov, ttl, running=running)


def _read_live(
    ctx: ServiceContext, a: Row, cov: Coverage, ttl: float, *, running: bool
) -> tuple[Progress | None, Health | None, bool]:
    progress = _cached(Progress, a.get("progress_json"))
    health = _cached(Health, a.get("health_json")) if running else None
    try:
        views = compute_live_views(ctx, a, timeout=_read_timeout(ctx))
    except HpcLabError as exc:
        cov.add("metrics", "unavailable", None, _clip(f"{exc.code}: {exc.message}"))
        if progress is not None:
            cov.notes.append("progress/health are cached views from an earlier poll")
        return progress, health, False
    if views.unreachable:
        cov.add("metrics", "unavailable", None, _clip("; ".join(views.notes)))
        if progress is not None or health is not None:
            cov.notes.append("progress/health are cached views from an earlier poll")
        return progress, health, False
    now = iso(ctx.clock.now())
    cov.add(
        "metrics",
        "ok" if views.metrics_source in ("ok", "missing") else "unavailable",
        now,
        _NO_METRICS if views.metrics_source == "missing" else None,
    )
    if running:
        cov.add(
            "telemetry",
            "ok" if views.telemetry_source in ("ok", "missing") else "unavailable",
            now,
            _NO_TELEMETRY if views.telemetry_source == "missing" else None,
        )
    store_live_views(ctx, a["attempt_id"], views)
    _record_live_read(ctx, a, views, ttl)
    return views.progress or progress, views.health if running else None, True


def _failure_like(a: Mapping[str, Any]) -> bool:
    if a["phase"] != AttemptPhase.TERMINAL.value:
        return False
    if a["scheduler_state"] != SchedulerState.COMPLETED.value:
        return True
    return a["app_state"] not in ("exited_ok", "unknown") or a["result_state"] in (
        ResultState.INCOMPLETE.value,
        ResultState.INVALID.value,
    )


def _diagnosis_for(ctx: ServiceContext, a: Row, cov: Coverage, *, remote_ok: bool = True) -> Diagnosis | None:
    stored = jload(a.get("diagnosis_json"), None)
    if isinstance(stored, dict):
        try:
            diag = Diagnosis.model_validate(stored)
        except ValueError:
            diag = None
        # A COMPLETED attempt's result may have been judged after finalize: recompute then.
        if diag is not None and not (
            a["scheduler_state"] == SchedulerState.COMPLETED.value
            and diag.primary_code is None
            and _failure_like(a)
        ):
            return diag
    if not _failure_like(a):
        return None
    if not remote_ok:
        cov.add("logs", "skipped", None, "the cluster did not answer; diagnosis from scheduler facts only")
        return diagnose(attempt_facts(ctx.db, ctx, a), "", "")
    try:
        return diagnose_attempt(ctx, attempt_id=a["attempt_id"], fetch_logs=True)
    except HpcLabError as exc:
        cov.add("logs", "unavailable", None, _clip(f"{exc.code}: {exc.message}"))
        return diagnose(attempt_facts(ctx.db, ctx, a), "", "")


def _env_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    torch = sub_dict(doc, "torch")
    smi = sub_dict(doc, "nvidia_smi")
    py = sub_dict(doc, "python")
    slurm = sub_dict(doc, "slurm")
    guard = sub_dict(doc, "guard")
    return {
        "host": doc.get("host"),
        "python": {"version": py.get("version"), "executable": _clip(py.get("executable"))} if py else None,
        "packages_digest": doc.get("packages_digest"),
        "packages_tool": doc.get("packages_tool"),
        "torch": {k: torch.get(k) for k in ("version", "cuda", "cudnn", "cuda_available")} if torch else None,
        "gpu_names": sorted({str(n) for n in torch.get("device_names") or []})[:8]
        or sorted({str(g.get("name")) for g in smi.get("gpus") or [] if isinstance(g, dict)})[:8],
        "driver_version": smi.get("driver_version") if smi else None,
        "driver_cuda": smi.get("cuda_version") if smi else None,
        "lockfiles": doc.get("lockfiles") if isinstance(doc.get("lockfiles"), dict) else {},
        "slurm": {k: slurm.get(k) for k in ("partition", "nodelist", "gpus", "cuda_visible_devices")},
        "guard": {k: guard.get(k) for k in ("status", "failed")} if guard else None,
        "captured_at": doc.get("captured_at"),
        "degraded": bool(doc.get("degraded")),
        "errors": [_clip(e, 200) for e in (doc.get("errors") or [])[:5]],
    }


def _env_view(ctx: ServiceContext, run: Row, focus: Row, attempts: Sequence[Row]) -> EnvView:
    spec = jload(run["spec_json"], {}) or {}
    runtime = sub_dict(spec, "runtime")
    res = sub_dict(plan_dict(ctx.db, focus["exec_plan_hash"]), "resources")
    declared = {
        "python": runtime.get("python"),
        "venv": runtime.get("venv"),
        "lockfiles": runtime.get("lockfiles") or [],
        "gpu_type": focus.get("gpu_type") or res.get("gpu_type"),
        "gpus": res.get("gpus"),
    }
    doc = jload(focus.get("env_json"), None)
    if not isinstance(doc, dict):
        return EnvView(declared=declared, observed=None, drift=[], source="none")
    observed = _env_summary(doc)
    drift: list[str] = []
    gtype = declared.get("gpu_type")
    names = observed.get("gpu_names") or []
    if gtype and names and not any(str(gtype).lower() in n.lower() for n in names):
        drift.append(f"declared gpu_type {gtype!r} but the job saw {names}")
    for other in attempts:
        if other["attempt_id"] == focus["attempt_id"] or other["seq"] >= focus["seq"]:
            continue
        prev_doc = jload(other.get("env_json"), None)
        if not isinstance(prev_doc, dict):
            continue
        prev = _env_summary(prev_doc)
        for key, label in (
            ("packages_digest", "installed packages"),
            ("driver_version", "NVIDIA driver"),
            ("lockfiles", "lockfile digests"),
        ):
            if prev.get(key) and observed.get(key) and prev.get(key) != observed.get(key):
                drift.append(f"{label} differ from attempt #{other['seq']}")
        pt, ot = (prev.get("torch") or {}).get("version"), (observed.get("torch") or {}).get("version")
        if pt and ot and pt != ot:
            drift.append(f"torch {ot} (attempt #{other['seq']} had {pt})")
        py_prev, py_now = (
            (prev.get("python") or {}).get("version"),
            (observed.get("python") or {}).get("version"),
        )
        if py_prev and py_now and py_prev != py_now:
            drift.append(f"python {py_now} (attempt #{other['seq']} had {py_prev})")
        break  # compare with the closest earlier attempt that has an env record
    return EnvView(declared=declared, observed=observed, drift=drift[:10], source="env.json")


def _artifacts(ctx: ServiceContext, a: Row) -> ArtifactsSummary:
    by: dict[str, int] = {}
    total = 0
    for r in ctx.db.all(
        "SELECT integrity, COUNT(*) AS n FROM artifacts WHERE attempt_id=? GROUP BY integrity",
        (a["attempt_id"],),
    ):
        by[r["integrity"]] = int(r["n"])
        total += int(r["n"])
    missing = [
        r["name"]
        for r in ctx.db.all(
            "SELECT name FROM artifacts WHERE attempt_id=? AND required=1 AND integrity IN ('missing','incomplete')"
            " ORDER BY name LIMIT 20",
            (a["attempt_id"],),
        )
    ]
    return ArtifactsSummary(
        attempt_id=a["attempt_id"],
        total=total,
        by_integrity=by,
        required_missing=missing,
        result_state=a["result_state"],
    )


def _evaluations(ctx: ServiceContext, run_id: str) -> list[EvaluationRef]:
    chosen = select_evaluations(ctx.db, [run_id]).get(run_id)
    rows = ctx.db.all(
        "SELECT evaluation_id, attempt_id, evaluator_id, evaluator_version, status, created_at FROM evaluations"
        " WHERE run_id=? ORDER BY created_at DESC LIMIT 20",
        (run_id,),
    )
    return [
        EvaluationRef(
            evaluation_id=r["evaluation_id"],
            ref=f"hpclab://evaluations/{r['evaluation_id']}",
            attempt_id=r["attempt_id"],
            evaluator=f"{r['evaluator_id']}@{r['evaluator_version']}",
            status=r["status"],
            selected=chosen is not None and chosen["evaluation_id"] == r["evaluation_id"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def _gpu_probe(ctx: ServiceContext, a: Row) -> GpuProbeView:
    """nvidia-smi in the job's own allocation: RUNNING only, live identity check, 1 per 10 min."""
    if (
        a["phase"] not in OBSERVED_PHASES
        or a["scheduler_state"] != SchedulerState.RUNNING.value
        or not a["job_id"]
    ):
        return GpuProbeView(status="not_running", detail=f"attempt is {a['phase']}/{a['scheduler_state']}")
    key = f"probe.gpu:{a['attempt_id']}"
    now = ctx.clock.now()
    next_at = iso(now + timedelta(seconds=GPU_PROBE_INTERVAL_S))
    # Take the slot atomically BEFORE probing (check-then-act across concurrent requests).
    with ctx.db.tx(source="api") as tx:
        cached = tx.one("SELECT payload_json, created_at FROM report_cache WHERE cache_key=?", (key,))
        made = parse_iso(cached["created_at"]) if cached is not None else None
        if cached is not None and made is not None and made + timedelta(seconds=GPU_PROBE_INTERVAL_S) > now:
            prev = jload(cached["payload_json"], {}) or {}
            running = prev.get("status") == "in_progress"
            return GpuProbeView(
                status="rate_limited",
                samples=[GpuSampleView.model_validate(s) for s in prev.get("samples") or []],
                observed_at=prev.get("observed_at"),
                next_allowed_at=iso(made + timedelta(seconds=GPU_PROBE_INTERVAL_S)),
                detail="another GPU probe of this attempt is running now"
                if running
                else f"last probe ({prev.get('status')}) at {cached['created_at']}; at most one per 10 minutes",
            )
        tx.execute(
            "INSERT OR REPLACE INTO report_cache (cache_key, payload_json, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key, jdump({"status": "in_progress"}), tx.now, next_at),
        )
    view = GpuProbeView(status="unavailable", detail="the probe did not complete")
    try:
        view = _run_gpu_probe(ctx, a)
    except HpcLabError as exc:
        view = GpuProbeView(status="unavailable", detail=_clip(f"{exc.code}: {exc.message}"))
    finally:
        with ctx.db.tx(source="api") as tx:
            tx.execute(
                "UPDATE report_cache SET payload_json=? WHERE cache_key=?",
                (jdump(view.model_dump(mode="json")), key),
            )
    view.next_allowed_at = next_at
    return view


def _run_gpu_probe(ctx: ServiceContext, a: Row) -> GpuProbeView:
    scheduler = ctx.clusters.scheduler(a["cluster_id"])
    try:
        q = scheduler.query_queue(job_ids=[a["job_id"]])
    except HpcLabError as exc:
        return GpuProbeView(
            status="unavailable", detail=_clip(f"identity check failed: {exc.code}: {exc.message}")
        )
    obs = pick_observation(a, q.jobs)
    if obs is None:
        return GpuProbeView(
            status="not_running", detail="the job is no longer in the queue", observed_at=q.observed_at
        )
    why = identity_mismatch(a, obs)
    if why is not None:
        return GpuProbeView(
            status="refused", detail=f"live job identity does not match: {why}", observed_at=q.observed_at
        )
    if obs.state != SchedulerState.RUNNING:
        return GpuProbeView(
            status="not_running", detail=f"job is {obs.state.value}", observed_at=q.observed_at
        )
    timeout = max(5.0, min(20.0, ctx.settings.request_deadline_s))
    try:
        samples = scheduler.gpu_samples(a["job_id"], timeout=timeout)
    except HpcLabError as exc:
        status: Literal["not_running", "unsupported", "unavailable"]
        if exc.code in (ErrorCode.NOT_READY, ErrorCode.PRECONDITION_FAILED):
            status = "not_running"
        elif exc.code == ErrorCode.UNSUPPORTED:
            status = "unsupported"
        else:
            status = "unavailable"
        return GpuProbeView(status=status, detail=_clip(f"{exc.code}: {exc.message}"))
    return GpuProbeView(
        status="ok",
        samples=[
            GpuSampleView(
                index=s.index,
                name=s.name,
                util_percent=s.util_percent,
                mem_used_mib=s.mem_used_mib,
                mem_total_mib=s.mem_total_mib,
                power_w=s.power_w,
            )
            for s in samples[:16]
        ],
        observed_at=iso(ctx.clock.now()),
    )


def _recommend(data: RunInspectData, a: Row | None, *, held: str | None = None) -> list[str]:
    """Deterministic hints (never actions): the next tool call a human/agent would make, as a
    valid call (JSON literals, an ``idempotency_key`` placeholder on mutations). ``held``: why
    an open workflow still owns this never-dispatched run (then it is not submitted by hand)."""
    hints: list[str] = []
    run_id = data.run.run_id
    if a is None:
        if held is not None:
            return [held]
        plan = data.run.current_exec_plan_hash or f"<exec_plan_hash from hpclab://plans/{data.run.plan_id}>"
        hints.append(
            "submit the run: "
            + call("run_submit", run_id=run_id, execution_plan_hash=plan, idempotency_key=NEW_KEY)
        )
        return hints
    aid = a["attempt_id"]
    phase = a["phase"]
    if data.stale:
        hints.append("the state may be outdated: " + call("run_inspect", run_id=run_id, refresh=True))
    if phase == AttemptPhase.SUBMISSION_UNCERTAIN.value:
        hints.append(
            "submission outcome is unknown: hpclab is reconciling (receipt, squeue, sacct) and will never "
            "resubmit on its own; watch the submit operation"
        )
    elif phase == AttemptPhase.DUPLICATE_DETECTED.value:
        hints.append(
            "more than one Slurm job matches this attempt: a human must decide (CLI: hpclab attempt resolve); "
            "hpclab never cancels suspected duplicates automatically"
        )
    elif phase in OBSERVED_PHASES:
        if a["observation_gap"]:
            hints.append(
                "the job left the queue and accounting has no record yet (lag); hpclab keeps checking - do not resubmit"
            )
        if data.pending is not None and data.pending.category == "never_starts":
            hints.append(
                f"this job cannot start as submitted ({data.pending.reason_code}): {data.pending.suggestion} "
                "-> "
                + call("run_cancel", attempt_id=aid, expected_version=a["version"], idempotency_key=NEW_KEY)
            )
        elif data.pending is not None and data.pending.actionable and data.pending.suggestion:
            hints.append(data.pending.suggestion)
        if data.health is not None and data.health.flags:
            hints.append(
                f"advisory health flags {data.health.flags}: check with "
                + call("run_inspect", attempt_id=aid, probe="gpu")
                + " and "
                + call("run_logs", attempt_id=aid, tail_lines=100)
                + " before acting (flags are signals, not proof)"
            )
        if data.progress is not None and data.progress.timelimit_risk == "high":
            hints.append(
                "the ETA exceeds the remaining walltime: if the plan uses checkpoint_and_exit, "
                + call(
                    "run_cancel",
                    attempt_id=aid,
                    expected_version=a["version"],
                    mode="request_checkpoint",
                    idempotency_key=NEW_KEY,
                )
                + " stops it cleanly"
            )
    elif phase == AttemptPhase.TERMINAL.value:
        failed = a["scheduler_state"] != SchedulerState.COMPLETED.value or a["app_state"] not in (
            "exited_ok",
            "unknown",
        )
        if failed:
            hints.append(
                "read the error: " + call("run_logs", attempt_id=aid, stream="stderr", tail_lines=200)
            )
            diag = data.diagnosis
            strategies = diag.primary.suggested_strategies if diag is not None and diag.primary else []
            recoverable = [s for s in strategies if s in ("retry", "resume", "adjust_resources")]
            if recoverable:
                hints.append(
                    "recover: "
                    + call("run_recover", attempt_id=aid, strategy=recoverable[0], idempotency_key=NEW_KEY)
                )
            elif "derive_run" in strategies:
                hints.append(
                    "the fix changes the experiment (e.g. batch size): derive a new Run with the changed "
                    "params/resources in overrides: "
                    + call(
                        "experiment_prepare",
                        derive_from={"run_id": run_id, "overrides": {}, "recapture_code": False},
                        idempotency_key=NEW_KEY,
                    )
                )
            elif "fix_environment" in strategies or "fix_code" in strategies:
                hints.append(
                    "fix the code/environment, then prepare again (retrying unchanged would fail the same way)"
                )
        elif a["result_state"] in (ResultState.UNCHECKED.value, ResultState.PENDING.value):
            hints.append(
                "outputs are being indexed (artifacts.index); check "
                + call("artifacts_list", attempt_id=aid)
                + " shortly"
            )
        elif a["result_state"] in (ResultState.INCOMPLETE.value, ResultState.INVALID.value):
            hints.append("required outputs are missing/invalid: " + call("artifacts_list", attempt_id=aid))
        elif a["result_state"] == ResultState.VALID.value and not any(
            e.attempt_id == aid and e.status == "valid" for e in data.evaluations
        ):
            # newest first: a partial/invalid evaluation of this attempt makes it C!; collecting
            # again with the same inputs returns that same record (evaluation_collect is idempotent)
            bad = next((e for e in data.evaluations if e.attempt_id == aid), None)
            if bad is not None:
                hints.append(
                    f"evaluation {bad.evaluation_id} of this attempt is {bad.status}: read {bad.ref} (missing "
                    "fields / parse errors); the same inputs give the same verdict, so fix the evaluator "
                    "config first, then "
                    + call("evaluation_collect", attempt_id=aid, idempotency_key=NEW_KEY)
                    + " with the corrected evaluator_id/config"
                )
            else:
                hints.append(
                    "collect the metrics: "
                    + call("evaluation_collect", attempt_id=aid, idempotency_key=NEW_KEY)
                )
    return hints[:8]


# ---------------------------------------------------------------------------
# diagnose_attempt / operation_get
# ---------------------------------------------------------------------------


def diagnose_attempt(ctx: ServiceContext, *, attempt_id: str, fetch_logs: bool = True) -> Diagnosis:
    """Diagnosis of one attempt from ledger facts (+ bounded log tails when ``fetch_logs``).

    ``fetch_logs=False`` returns the stored diagnosis when there is one. A fresh diagnosis of a
    terminal attempt refreshes the cached ``diagnosis_json`` (a derived view)."""
    a = ctx.db.one("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,))
    if a is None:
        raise NotFound(f"attempt {attempt_id} not found", details={"attempt_id": attempt_id})
    stored_doc = jload(a.get("diagnosis_json"), None)
    stored: Diagnosis | None = None
    if isinstance(stored_doc, dict):
        try:
            stored = Diagnosis.model_validate(stored_doc)
        except ValueError:
            stored = None
    if not fetch_logs:
        return stored if stored is not None else diagnose(attempt_facts(ctx.db, ctx, a), "", "")
    transport = ctx.clusters.transport(a["cluster_id"])
    stdout, stderr, problems = read_log_tails(transport, layout_for(a), timeout=_read_timeout(ctx))
    if problems and stored is not None and stored.primary_code is not None:
        # The logs are gone (scratch purge) or unreadable now: a re-diagnosis would only know less
        # than the stored one computed while they were fresh (e.g. cuda_oom -> unknown_failure).
        return stored
    diag = diagnose(attempt_facts(ctx.db, ctx, a), stdout, stderr)
    unread = [p for p in problems if not p.endswith(" missing")]  # transient: do not cache
    if a["phase"] == AttemptPhase.TERMINAL.value and not unread:
        with ctx.db.tx(source="api") as tx:
            tx.update(
                "attempts",
                "attempt_id",
                attempt_id,
                {"diagnosis_json": jdump(diag.model_dump(mode="json"))},
                bump_version=False,
            )
    return diag


class OperationEvent(_M):
    event_type: str
    recorded_at: str
    payload: dict[str, Any]


class OperationData(_M):
    operation_id: str
    ref: str
    kind: str
    state: str
    stage: str | None
    target_type: str | None
    target_id: str | None
    target_ref: str | None
    progress: dict[str, Any]
    result: Any = None
    error: dict[str, Any] | None = None
    poll_after_ms: int
    version: int
    created_at: str
    updated_at: str
    finished_at: str | None
    recent_events: list[OperationEvent] = Field(default_factory=list)

    def summary(self) -> str:
        return f"operation {self.kind} {self.state}" + (f" ({self.stage})" if self.stage else "")


_TARGET_REFS = {
    "workflow": "hpclab://workflows/{}",
    "plan": "hpclab://plans/{}",
    "artifact": "hpclab://artifacts/{}",
    "evaluation": "hpclab://evaluations/{}",
    "comparison": "hpclab://comparisons/{}",
    "evidence": "hpclab://evidence/{}",
}
MAX_OPERATION_RESULT_BYTES = 32 * 1024


def operation_get(ctx: ServiceContext, *, operation_id: str) -> OperationData:
    """A background operation's state, stage, progress (``{}`` when unknown), result and error
    (resource ``hpclab://operations/{id}``)."""
    r = ctx.db.one("SELECT * FROM operations WHERE operation_id=?", (operation_id,))
    if r is None:
        raise NotFound(f"operation {operation_id} not found", details={"operation_id": operation_id})
    result: Any = None
    if r["result_json"] is not None:
        raw = r["result_json"]
        result = (
            {"truncated": True, "bytes": len(raw)} if len(raw) > MAX_OPERATION_RESULT_BYTES else jload(raw)
        )
    events = ctx.db.all(
        "SELECT event_type, recorded_at, payload_json FROM events WHERE entity_type='operation' AND entity_id=?"
        " ORDER BY seq DESC LIMIT 10",
        (operation_id,),
    )
    template = _TARGET_REFS.get(r["target_type"] or "")
    return OperationData(
        operation_id=r["operation_id"],
        ref=f"hpclab://operations/{r['operation_id']}",
        kind=r["kind"],
        state=r["state"],
        stage=r["stage"],
        target_type=r["target_type"],
        target_id=r["target_id"],
        target_ref=template.format(r["target_id"]) if template and r["target_id"] else None,
        progress=jload(r["progress_json"], {}) or {},
        result=result,
        error=jload(r["error_json"], None),
        poll_after_ms=int(r["poll_after_ms"]),
        version=int(r["version"]),
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        finished_at=r["finished_at"],
        recent_events=[
            OperationEvent(
                event_type=e["event_type"],
                recorded_at=e["recorded_at"],
                payload=jload(e["payload_json"], {}) or {},
            )
            for e in events
        ],
    )
