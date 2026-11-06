"""Fakes for usage_report / lab_digest unit tests (imported by name, pytest prepend mode):
scripted accounting behind a real ServiceContext (FakeClock at 2026-09-01T12:00:00Z)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from hpclab.adapters.base import CommandResult, QueryResult, SchedulerCapabilities
from hpclab.adapters.slurm import UsageSegment
from hpclab.domain.clock import FakeClock, iso, parse_iso
from hpclab.domain.models import JobObservation
from hpclab.services.context import ServiceContext

NOW = datetime.fromisoformat("2026-09-01T12:00:00+00:00")


def t(hours_ago: float) -> str:
    stamp = iso(NOW - timedelta(hours=hours_ago))
    assert stamp is not None
    return stamp


def seg(
    job_id: str,
    *,
    user: str = "alice",
    state: str = "COMPLETED",
    submit: float,
    start: float | None,
    end: float | None,
    gpus: int = 1,
    gpu_types: tuple[tuple[str, int], ...] = (("a100", 1),),
    account: str = "lab",
    partition: str = "gpu",
    limit_h: float | None = 4.0,
    name: str = "train",
    known: bool = True,
    ran: bool = True,
) -> UsageSegment:
    """A record with times given in hours before NOW (None = unknown / still running)."""
    return UsageSegment(
        job_id=job_id,
        user=user,
        account=account,
        partition=partition,
        state=state,
        gpus=gpus,
        elapsed_s=int((start - (end if end is not None else 0)) * 3600) if start is not None else 0,
        start_time=t(start) if start is not None else None,
        end_time=t(end) if end is not None else None,
        submit_time=t(submit),
        job_name=name,
        job_id_raw=job_id,
        qos="normal",
        raw_state=state,
        exit_code=0,
        exit_signal=0,
        time_limit_s=int(limit_h * 3600) if limit_h is not None else None,
        node_list="n1" if ran and start is not None else None,
        gpus_known=known,
        gpu_types=gpu_types,
    )


@dataclass
class FakeAcct:
    """Transport + scheduler in one: remote_user, capabilities, sacct windows, squeue."""

    clock: FakeClock
    cluster_id: str = "fake"
    user: str = "alice"
    private_data: str | None = "none"
    records: list[UsageSegment] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[Exception | None] = field(default_factory=list)  # per call, in order
    queue_jobs: list[JobObservation] = field(default_factory=list)
    account_jobs: list[JobObservation] = field(default_factory=list)
    remote_calls: int = 0

    # transport
    def remote_user(self) -> str:
        self.remote_calls += 1
        return self.user

    def run(self, argv: list[str], **_: Any) -> CommandResult:
        self.remote_calls += 1
        return CommandResult(argv, 0, "", "", 0.0)

    def close(self) -> None:
        return None

    # scheduler
    def probe_capabilities(self, *, refresh: bool = False) -> SchedulerCapabilities:
        return SchedulerCapabilities(
            sacct_available=True, private_data=self.private_data, cluster_name="fake"
        )

    def usage_segments(
        self, *, account: str | None, users: list[str] | None, since: str, until: str | None
    ) -> list[UsageSegment]:
        self.remote_calls += 1
        self.calls.append({"account": account, "users": users, "since": since, "until": until})
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        lo = parse_iso(since)
        hi = parse_iso(until) if until else self.clock.now()
        assert lo is not None and hi is not None
        out = []
        for r in self.records:
            if account and r.account != account:
                continue
            if users and r.user not in users:
                continue
            submit = parse_iso(r.submit_time)
            end = parse_iso(r.end_time)
            if submit is not None and submit > hi:
                continue
            if end is not None and end < lo:
                continue
            out.append(r)
        return out

    def query_queue(self, *, job_ids: list[str] | None = None, user: str | None = None) -> QueryResult:
        self.remote_calls += 1
        return QueryResult(
            source="squeue", observed_at=iso(self.clock.now()) or "", jobs=list(self.queue_jobs)
        )

    def query_account_queue(self, account: str) -> QueryResult:
        self.remote_calls += 1
        return QueryResult(
            source="squeue", observed_at=iso(self.clock.now()) or "", jobs=list(self.account_jobs)
        )


@dataclass
class UsageEnv:
    ctx: ServiceContext
    acct: FakeAcct
    clock: FakeClock
