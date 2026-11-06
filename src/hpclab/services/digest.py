"""Lab digest (``lab_digest``): a Markdown summary of a period, built from the local ledger.

Sections: experiment progress (the shared coverage codes, so the digest and the grid agree),
failures by diagnosis cause with GPU-hours lost, settled GPU-hours, new evidence bundles, what
needs a human (uncertain/duplicate submissions, blocked operations) and long-pending jobs with
decoded reasons. Every number is read from a stored record; nothing is estimated.

The only remote call is the optional usage section (``account=...``), which is exactly
``usage_report(scope="account")`` for the same window, including its visibility label: under
``self_only`` the section says the numbers are the caller's own jobs, not lab totals.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hpclab.api.envelope import MAX_TEXT_BYTES, Coverage
from hpclab.domain.clock import iso, parse_iso
from hpclab.domain.errors import HpcLabError, InvalidArgument
from hpclab.services.context import ServiceContext
from hpclab.services.coverage import cell_statuses, summarize_codes
from hpclab.services.usage import UsageData, parse_when, usage_report
from hpclab.storage.db import jload

log = logging.getLogger("hpclab.digest")

MAX_DIGEST_DAYS = 366
MAX_EXPERIMENTS = 50
MAX_ITEMS = 50
LONG_PENDING_S = 4 * 3600
QUIET_LOOKBACK = timedelta(days=30)
# E (completed, evaluation pending) has no valid result yet: unfinished, but never a failure
UNFINISHED_CODES = ("P", "S", "Q", "R", "E", "U", "?")


class ExperimentProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    cluster_id: str
    planned: int
    counts: dict[str, int]  # coverage codes (V E R Q S P F C! X U ?)
    valid: int
    unfinished: int
    last_activity_at: str | None
    quiet: bool  # unfinished cells but no activity in the period


class FailureClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str  # diagnosis primary code, else the scheduler state
    category: str
    attempts: int
    gpu_hours_lost: float  # GPUs x elapsed of attempts whose elapsed time is known
    unknown_elapsed_attempts: int  # attempts not included in gpu_hours_lost (no accounting)
    # TIMEOUTs that left a finalized checkpoint: resumable progress, not lost (usage_report's rule)
    checkpointed_attempts: int = 0


class GpuHours(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settled_gpu_hours: float
    settled_attempts: int
    unknown_settlements: int  # settled without accounting: consumption unknown, not 0
    open_reserved_gpu_hours: float
    open_reservations: int


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    purpose: str
    run_set_ref: str | None
    created_at: str


class AttentionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["submission_uncertain", "duplicate_detected", "blocked_operation"]
    ref: str
    run_id: str | None = None
    label: str | None = None
    since: str | None = None
    detail: str | None = None


class LongPending(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    run_id: str
    label: str | None
    job_id: str | None
    pending_since: str | None
    pending_for_s: int | None
    reason: str | None
    category: str
    explanation: str | None


class DigestData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: str
    until: str
    cluster_id: str | None
    account: str | None
    markdown: str
    experiments: list[ExperimentProgress]
    failures: list[FailureClass]
    gpu_hours: GpuHours
    evidence: list[EvidenceItem]
    attention: list[AttentionItem]
    long_pending: list[LongPending]
    usage: UsageData | None = None
    usage_error: dict[str, Any] | None = None
    truncated: dict[str, int] = Field(default_factory=dict)
    coverage: Coverage
    observed_at: str

    def summary(self) -> str:
        fails = sum(f.attempts for f in self.failures)
        return (
            f"digest {self.since[:10]}..{self.until[:10]}: {len(self.experiments)} experiment(s), "
            f"{fails} failed attempt(s), {self.gpu_hours.settled_gpu_hours:.1f} settled GPU-h, "
            f"{len(self.attention)} item(s) need attention"
        )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _cluster_filter(cluster_id: str | None, column: str) -> tuple[str, list[Any]]:
    return (f" AND {column} = ?", [cluster_id]) if cluster_id else ("", [])


def _experiments(
    ctx: ServiceContext, *, start: str, end: str, cluster_id: str | None
) -> tuple[list[ExperimentProgress], int]:
    where, params = _cluster_filter(cluster_id, "e.cluster_id")
    rows = ctx.db.all(
        "SELECT e.experiment_id, e.name, e.cluster_id, e.created_at, "
        "MAX(MAX(COALESCE(a.updated_at, ''), COALESCE(r.updated_at, ''), e.updated_at)) AS activity "
        "FROM experiments e LEFT JOIN runs r ON r.experiment_id = e.experiment_id "
        "LEFT JOIN attempts a ON a.run_id = r.run_id "
        f"WHERE e.created_at <= ?{where} GROUP BY e.experiment_id ORDER BY activity DESC",
        [end, *params],
    )
    lookback = iso((parse_iso(start) or datetime.min) - QUIET_LOOKBACK) or ""
    out: list[ExperimentProgress] = []
    for r in rows:
        activity = r["activity"] or r["created_at"]
        active = start <= activity <= end or start <= r["created_at"] <= end
        if not active and activity < lookback:
            continue
        run_ids = [
            x["run_id"]
            for x in ctx.db.all(
                "SELECT run_id FROM runs WHERE experiment_id=? AND kind='main' ORDER BY run_id",
                (r["experiment_id"],),
            )
        ]
        # Same freshness bound as the runs_list grid: a Q/R cell whose last observation is too
        # old is '?' (state unknown now), never reported as still running.
        statuses = cell_statuses(
            ctx.db, run_ids, stale_after_s=_stale_after_s(ctx, r["cluster_id"]), now=ctx.clock.now()
        )
        counts = summarize_codes(statuses.values())
        unfinished = sum(counts.get(c, 0) for c in UNFINISHED_CODES)
        if not active and not unfinished:
            continue  # finished and untouched in the period: not news
        out.append(
            ExperimentProgress(
                experiment_id=r["experiment_id"],
                name=r["name"],
                cluster_id=r["cluster_id"],
                planned=counts.get("planned", 0),
                counts={k: v for k, v in counts.items() if k not in ("planned", "valid")},
                valid=counts.get("valid", 0),
                unfinished=unfinished,
                last_activity_at=activity,
                quiet=not active,
            )
        )
    return out[:MAX_EXPERIMENTS], max(0, len(out) - MAX_EXPERIMENTS)


def _stale_after_s(ctx: ServiceContext, cluster_id: str) -> float:
    """The grid's staleness bound (mirrors ``services.runs._stale_after``): 4 poll intervals,
    at least 2 minutes."""
    try:
        interval = float(ctx.clusters.get(cluster_id).profile.policy.poll_min_interval_s)
    except HpcLabError:
        interval = 30.0
    if ctx.settings.min_poll_interval_s is not None:
        interval = float(ctx.settings.min_poll_interval_s)
    return max(120.0, 4.0 * interval)


def _category(code: str) -> str:
    try:
        from hpclab.services.diagnosis import CODE_CATEGORY
    except ImportError:  # pragma: no cover
        return "unknown"
    return str(CODE_CATEGORY.get(code, "scheduler"))


def _failures(ctx: ServiceContext, *, start: str, end: str, cluster_id: str | None) -> list[FailureClass]:
    where, params = _cluster_filter(cluster_id, "cluster_id")
    rows = ctx.db.all(
        "SELECT attempt_id, scheduler_state, result_state, diagnosis_json, gpus, elapsed_s FROM attempts "
        "WHERE phase='terminal' AND terminal_at >= ? AND terminal_at <= ? "
        f"AND (scheduler_state != 'COMPLETED' OR result_state IN ('invalid', 'incomplete')){where}",
        [start, end, *params],
    )
    checkpointed = _finalized_checkpoints(
        ctx, [r["attempt_id"] for r in rows if r["scheduler_state"] == "TIMEOUT"]
    )
    acc: dict[str, dict[str, Any]] = {}
    for r in rows:
        diag = jload(r["diagnosis_json"], None) or {}
        code = diag.get("primary_code") if isinstance(diag, dict) else None
        cause = str(code or (r["scheduler_state"] or "unknown").lower())
        slot = acc.setdefault(cause, {"attempts": 0, "gh": 0.0, "unknown": 0, "ckpt": 0})
        slot["attempts"] += 1
        if r["attempt_id"] in checkpointed:
            slot["ckpt"] += 1  # stopped at the walltime with a finalized checkpoint: not lost
        elif r["elapsed_s"] is None:
            slot["unknown"] += 1
        else:
            slot["gh"] += int(r["gpus"] or 0) * int(r["elapsed_s"]) / 3600.0
    return [
        FailureClass(
            cause=cause,
            category=_category(cause),
            attempts=v["attempts"],
            gpu_hours_lost=round(v["gh"], 3),
            unknown_elapsed_attempts=v["unknown"],
            checkpointed_attempts=v["ckpt"],
        )
        for cause, v in sorted(acc.items(), key=lambda kv: (-kv[1]["gh"], kv[0]))
    ]


def _finalized_checkpoints(ctx: ServiceContext, attempt_ids: list[str]) -> set[str]:
    out: set[str] = set()
    for i in range(0, len(attempt_ids), 500):
        chunk = attempt_ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        out.update(
            r["attempt_id"]
            for r in ctx.db.all(
                f"SELECT DISTINCT attempt_id FROM artifacts WHERE attempt_id IN ({marks}) "
                "AND role='checkpoint' AND integrity='finalized'",
                chunk,
            )
        )
    return out


def _gpu_hours(ctx: ServiceContext, *, start: str, end: str, cluster_id: str | None) -> GpuHours:
    where, params = _cluster_filter(cluster_id, "a.cluster_id")
    base = "FROM budget_reservations b JOIN attempts a ON a.attempt_id = b.attempt_id WHERE 1=1" + where
    settled = ctx.db.one(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(b.settled_consumed), 0) AS gh {base} "
        "AND b.state='settled' AND b.settled_consumed IS NOT NULL AND b.updated_at >= ? AND b.updated_at <= ?",
        [*params, start, end],
    ) or {"n": 0, "gh": 0}
    unknown = ctx.db.one(
        f"SELECT COUNT(*) AS n {base} AND (b.state='unknown' OR (b.state='settled' AND b.settled_consumed IS NULL)) "
        "AND b.updated_at >= ? AND b.updated_at <= ?",
        [*params, start, end],
    ) or {"n": 0}
    open_ = ctx.db.one(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(b.reserved), 0) AS gh {base} AND b.state='reserved'", params
    ) or {"n": 0, "gh": 0}
    return GpuHours(
        settled_gpu_hours=round(float(settled["gh"] or 0), 3),
        settled_attempts=int(settled["n"] or 0),
        unknown_settlements=int(unknown["n"] or 0),
        open_reserved_gpu_hours=round(float(open_["gh"] or 0), 3),
        open_reservations=int(open_["n"] or 0),
    )


def _evidence(ctx: ServiceContext, *, start: str, end: str) -> tuple[list[EvidenceItem], int]:
    rows = ctx.db.all(
        "SELECT bundle_id, purpose, run_set_ref, created_at FROM evidence_bundles "
        "WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
        (start, end),
    )
    items = [EvidenceItem(**r) for r in rows[:MAX_ITEMS]]
    return items, max(0, len(rows) - MAX_ITEMS)


def _attention(ctx: ServiceContext, *, cluster_id: str | None) -> tuple[list[AttentionItem], int]:
    where, params = _cluster_filter(cluster_id, "a.cluster_id")
    items: list[AttentionItem] = []
    for r in ctx.db.all(
        "SELECT a.attempt_id, a.phase, a.run_id, a.uncertain_since, a.updated_at, a.job_id, r.label "
        "FROM attempts a JOIN runs r ON r.run_id = a.run_id "
        f"WHERE a.phase IN ('submission_uncertain', 'duplicate_detected'){where} ORDER BY a.updated_at",
        params,
    ):
        items.append(
            AttentionItem(
                kind=r["phase"],
                ref=r["attempt_id"],
                run_id=r["run_id"],
                label=r["label"],
                since=r["uncertain_since"] or r["updated_at"],
                detail="submission outcome unknown: hpclab keeps reconciling and never resubmits by itself"
                if r["phase"] == "submission_uncertain"
                else "more than one Slurm job matches this attempt: decide with `hpclab attempt resolve`",
            )
        )
    # The cluster a blocked operation waits on: the one named in its error, else its target's. A
    # digest for one cluster must not ask the user to log in to (or fix) another one.
    for r in ctx.db.all(
        "SELECT o.operation_id, o.kind, o.target_id, o.error_json, o.updated_at, "
        "COALESCE(a.cluster_id, er.cluster_id, ew.cluster_id, ee.cluster_id, ep.cluster_id) AS target_cluster "
        "FROM operations o "
        "LEFT JOIN attempts a ON o.target_type = 'attempt' AND a.attempt_id = o.target_id "
        "LEFT JOIN runs r ON o.target_type = 'run' AND r.run_id = o.target_id "
        "LEFT JOIN experiments er ON er.experiment_id = r.experiment_id "
        "LEFT JOIN workflows w ON o.target_type = 'workflow' AND w.workflow_id = o.target_id "
        "LEFT JOIN experiments ew ON ew.experiment_id = w.experiment_id "
        "LEFT JOIN experiments ee ON o.target_type = 'experiment' AND ee.experiment_id = o.target_id "
        "LEFT JOIN plans p ON o.target_type = 'plan' AND p.plan_id = o.target_id "
        "LEFT JOIN experiments ep ON ep.experiment_id = p.experiment_id "
        "WHERE o.state = 'blocked' ORDER BY o.updated_at"
    ):
        raw = jload(r["error_json"], None)
        err: dict[str, Any] = raw if isinstance(raw, dict) else {}
        details: dict[str, Any] = err["details"] if isinstance(err.get("details"), dict) else {}
        cid = details.get("cluster_id") or r["target_cluster"]
        if cluster_id and cid and cid != cluster_id:
            continue  # unknown cluster: kept (never hide something that needs a decision)
        where_ = f" on {cid}" if cid else ""
        fix = details.get("remediation")
        detail = f"{r['kind']} {r['target_id'] or ''}{where_}: {err.get('message', 'blocked')}" + (
            f" (fix: {fix})" if fix else ""
        )
        items.append(
            AttentionItem(
                kind="blocked_operation", ref=r["operation_id"], since=r["updated_at"], detail=detail[:300]
            )
        )
    return items[:MAX_ITEMS], max(0, len(items) - MAX_ITEMS)


def _decode(reason: str | None) -> tuple[str, str | None]:
    if not reason:
        return "unknown", None
    try:
        from hpclab.services.pending import decode_pending

        info = decode_pending(reason)
        return str(info.category), str(info.explanation)
    except ImportError:
        return "unknown", None
    except Exception:  # a decoder bug must not break the digest
        log.debug("decode_pending failed for %r", reason, exc_info=True)
        return "unknown", None


def _long_pending(
    ctx: ServiceContext, *, now: datetime, cluster_id: str | None
) -> tuple[list[LongPending], int]:
    where, params = _cluster_filter(cluster_id, "a.cluster_id")
    rows = ctx.db.all(
        "SELECT a.attempt_id, a.run_id, r.label, a.job_id, a.job_submit_time, a.identified_at, a.dispatch_at, "
        "a.pending_reason, a.reason FROM attempts a JOIN runs r ON r.run_id = a.run_id "
        f"WHERE a.phase IN ('job_identified', 'observing') AND a.scheduler_state='PENDING'{where}",
        params,
    )
    out: list[LongPending] = []
    for r in rows:
        since = r["job_submit_time"] or r["identified_at"] or r["dispatch_at"]
        started = parse_iso(since)
        waited = int((now - started).total_seconds()) if started else None
        if waited is not None and waited < LONG_PENDING_S:
            continue
        reason = r["pending_reason"] or r["reason"]
        category, explanation = _decode(reason)
        out.append(
            LongPending(
                attempt_id=r["attempt_id"],
                run_id=r["run_id"],
                label=r["label"],
                job_id=r["job_id"],
                pending_since=since,
                pending_for_s=waited,
                reason=reason,
                category=category,
                explanation=explanation,
            )
        )
    out.sort(key=lambda x: -(x.pending_for_s or 0))
    return out[:MAX_ITEMS], max(0, len(out) - MAX_ITEMS)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    return str(value if value is not None else "–").replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows)
    return out


def _hours(value: float) -> str:
    return f"{value:.1f}"


def render_markdown(d: DigestData) -> str:
    lines = [
        f"# HPC Lab digest: {d.since[:16]} → {d.until[:16]} UTC",
        "",
        f"_Generated {d.observed_at[:16]} UTC from the local ledger"
        + (f" (cluster {d.cluster_id})" if d.cluster_id else "")
        + ". Every number below is a stored record; nothing is estimated._",
        "",
        "## Experiments",
    ]
    if d.experiments:
        rows = []
        for e in d.experiments:
            c = e.counts
            rows.append(
                [
                    e.name,
                    e.planned,
                    e.valid,
                    c.get("R", 0),
                    c.get("Q", 0) + c.get("S", 0),
                    c.get("E", 0),
                    c.get("F", 0),
                    c.get("C!", 0),
                    c.get("U", 0) + c.get("?", 0),
                    c.get("P", 0),
                    # X: cancelled / rejected, incl. cells a cancelled workflow never dispatched
                    c.get("X", 0),
                    ("quiet since " if e.quiet else "") + (e.last_activity_at or "–")[:16],
                ]
            )
        lines += _table(
            [
                "experiment",
                "planned",
                "valid",
                "running",
                "queued",
                "evaluation pending",
                "failed",
                "completed, bad result",
                "unknown",
                "not dispatched",
                "cancelled",
                "last activity",
            ],
            rows,
        )
        if any(e.counts.get("E") for e in d.experiments):
            lines.append(
                "\n_evaluation pending: the job completed and its outputs are being checked or it is not "
                "evaluated yet - not a failure (if it persists, run evaluation_collect for the experiment). "
                "completed, bad result: incomplete/invalid outputs or a partial/invalid evaluation._"
            )
    else:
        lines.append("No experiment activity in this period.")
    lines += ["", "## Failures by cause"]
    if d.failures:
        lines += _table(
            [
                "cause",
                "category",
                "attempts",
                "GPU-hours lost",
                "attempts without accounting",
                "stopped with a checkpoint (not lost)",
            ],
            [
                [
                    f.cause,
                    f.category,
                    f.attempts,
                    _hours(f.gpu_hours_lost),
                    f.unknown_elapsed_attempts,
                    f.checkpointed_attempts,
                ]
                for f in d.failures
            ],
        )
        total = sum(f.gpu_hours_lost for f in d.failures)
        lines.append(
            f"\nTotal: {sum(f.attempts for f in d.failures)} attempt(s), {_hours(total)} GPU-hours lost."
        )
    else:
        lines.append("No failed attempts in this period.")
    g = d.gpu_hours
    lines += [
        "",
        "## GPU-hours (hpclab budget ledger)",
        f"- Settled: {_hours(g.settled_gpu_hours)} GPU-hours over {g.settled_attempts} attempt(s).",
        f"- Settled without accounting (consumption unknown, not zero): {g.unknown_settlements} attempt(s).",
        f"- Still reserved now: {_hours(g.open_reserved_gpu_hours)} GPU-hours over {g.open_reservations} attempt(s).",
        "",
        "## New evidence bundles",
    ]
    if d.evidence:
        lines += [
            f"- `{e.bundle_id}` ({e.purpose}) {e.run_set_ref or ''} {e.created_at[:16]}" for e in d.evidence
        ]
    else:
        lines.append("None.")
    lines += ["", "## Needs attention"]
    if d.attention:
        lines += [
            f"- **{a.kind}** `{a.ref}`" + (f" ({a.label})" if a.label else "") + f": {a.detail or ''}"
            for a in d.attention
        ]
    else:
        lines.append("Nothing needs a decision.")
    lines += ["", "## Long-pending jobs"]
    if d.long_pending:
        lines += _table(
            ["run", "job", "pending for", "reason", "category", "explanation"],
            [
                [
                    p.label or p.run_id,
                    p.job_id,
                    f"{(p.pending_for_s or 0) / 3600:.1f} h" if p.pending_for_s is not None else "?",
                    p.reason,
                    p.category,
                    p.explanation,
                ]
                for p in d.long_pending
            ],
        )
    else:
        lines.append("No job has been pending for more than 4 hours.")
    if d.account:
        lines += ["", f"## Usage: account {d.account}"]
        if d.usage is not None:
            u = d.usage
            if u.visibility != "full":
                lines.append(
                    f"**visibility: {u.visibility}** - Slurm showed only your own jobs; these are NOT lab totals."
                )
            key = "account" if "account" in u.totals else "self"
            t = u.totals[key]
            lines.append(
                f"- {'Account' if key == 'account' else 'Your'} allocated GPU-hours: {_hours(t.gpu_hours)} "
                f"over {t.jobs} job(s) (allocated, not utilization)."
            )
            if t.wasted_gpu_hours_by_cause:
                waste = ", ".join(f"{k} {_hours(v)}" for k, v in t.wasted_gpu_hours_by_cause.items())
                lines.append(f"- Waste by cause (GPU-hours): {waste}.")
            if u.rows:
                dims = u.group_by
                lines += [
                    "",
                    *_table(
                        [*dims, "jobs", "GPU-hours"],
                        [[*(r.group.get(dim) for dim in dims), r.jobs, _hours(r.gpu_hours)] for r in u.rows],
                    ),
                ]
            cov = "complete" if u.coverage.complete else "incomplete: " + "; ".join(u.coverage.missing)
            lines.append(f"\n_Coverage: {cov}; private data: {u.private_data or 'unknown'}._")
        else:
            err = d.usage_error or {}
            lines.append(f"Usage unavailable ({err.get('code', 'error')}): {err.get('message', '')}")
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        cut = text.encode("utf-8")[: MAX_TEXT_BYTES - 200].decode("utf-8", "ignore")
        text = cut + "\n\n_(truncated to 64 KiB; use the structured sections for everything)_\n"
    return text


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


def lab_digest(
    ctx: ServiceContext,
    *,
    since: str,
    until: str | None = None,
    account: str | None = None,
    cluster_id: str | None = None,
) -> DigestData:
    """Markdown digest of a period from the local ledger: experiment progress, failures by cause
    with GPU-hours lost, settled GPU-hours, new evidence bundles, items needing a decision and
    long-pending jobs. With ``account`` it adds the account's usage_report (the only remote call),
    labelled with its visibility. ``since``/``until``: ISO or relative (``7d``, ``24h``)."""
    now = ctx.clock.now()
    start = parse_when(since, now=now, what="since")
    end = parse_when(until, now=now, what="until") if until else now
    if end > now:
        end = now
    if start >= end:
        raise InvalidArgument("the digest period is empty: since must be before until")
    if end - start > timedelta(days=MAX_DIGEST_DAYS):
        raise InvalidArgument(f"the digest period may span at most {MAX_DIGEST_DAYS} days")
    if cluster_id is not None:
        ctx.clusters.get(cluster_id)
    s, e = iso(start) or "", iso(end) or ""
    experiments, more_exp = _experiments(ctx, start=s, end=e, cluster_id=cluster_id)
    failures = _failures(ctx, start=s, end=e, cluster_id=cluster_id)
    gpu = _gpu_hours(ctx, start=s, end=e, cluster_id=cluster_id)
    evidence, more_ev = _evidence(ctx, start=s, end=e)
    attention, more_att = _attention(ctx, cluster_id=cluster_id)
    pending, more_pend = _long_pending(ctx, now=now, cluster_id=cluster_id)
    coverage = Coverage()
    coverage.add(
        "ledger", "ok", observed_at=iso(now), detail="experiments, attempts, budget, evidence, operations"
    )
    usage: UsageData | None = None
    usage_error: dict[str, Any] | None = None
    if account:
        target = cluster_id
        if target is None:
            ids = ctx.clusters.ids()
            if len(ids) != 1:
                raise InvalidArgument(
                    "account usage needs cluster_id (several or no clusters are registered)",
                    details={"clusters": ids},
                )
            target = ids[0]
        try:
            usage = usage_report(ctx, cluster_id=target, scope="account", account=account, since=s, until=e)
            coverage.add(
                "usage_report",
                "ok" if usage.coverage.complete else "partial",
                observed_at=usage.observed_at,
                detail=f"visibility {usage.visibility}",
            )
        except HpcLabError as exc:
            usage_error = {"code": str(exc.code), "message": exc.message, "retryable": exc.retryable}
            coverage.add("usage_report", "unavailable", detail=exc.message[:300])
    truncated = {
        k: v
        for k, v in {
            "experiments": more_exp,
            "evidence": more_ev,
            "attention": more_att,
            "long_pending": more_pend,
        }.items()
        if v
    }
    data = DigestData(
        since=s,
        until=e,
        cluster_id=cluster_id,
        account=account,
        markdown="",
        experiments=experiments,
        failures=failures,
        gpu_hours=gpu,
        evidence=evidence,
        attention=attention,
        long_pending=pending,
        usage=usage,
        usage_error=usage_error,
        truncated=truncated,
        coverage=coverage,
        observed_at=iso(now) or "",
    )
    data.markdown = render_markdown(data)
    return data
