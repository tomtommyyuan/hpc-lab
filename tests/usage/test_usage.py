"""usage_report against scripted accounting (FakeClock at 2026-09-01T12:00Z).

Covers: windows (ISO + relative), clipping, running jobs charged to now, requeue segments,
waste by cause (managed diagnosis vs foreign state, TIMEOUT with checkpoint), PrivateData
visibility (self_only has no account total), per-day finalized caching, partial failures,
grouping, medians, the "now" section and unknown GPU counts.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from usage_fakes import NOW, UsageEnv, seg

from hpclab.adapters.base import AuthRequiredError, CommandFailed, CommandResult, TransportError
from hpclab.domain.errors import ErrorCode, HpcLabError, InvalidArgument, LimitExceeded
from hpclab.domain.ids import new_id
from hpclab.domain.models import JobObservation
from hpclab.profiles.schema import UserOverlay
from hpclab.services.usage import floor_day, parse_when, resolve_window, usage_report
from hpclab.storage.db import jdump
from hpclab.testing import factories as f

# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_hours_ago"),
    [("7d", 168), ("24h", 24), ("30m", 0.5), ("2w", 336), ("45s", 45 / 3600), ("now", 0), ("1.5h", 1.5)],
)
def test_relative_durations(text: str, expected_hours_ago: float) -> None:
    assert parse_when(text, now=NOW) == NOW - timedelta(hours=expected_hours_ago)


def test_iso_dates_and_times() -> None:
    assert parse_when("2026-08-01", now=NOW).isoformat() == "2026-08-01T00:00:00+00:00"
    assert (
        parse_when("2026-08-01T10:00:00", now=NOW).isoformat() == "2026-08-01T10:00:00+00:00"
    )  # naive = UTC
    assert parse_when("2026-08-01T10:00:00+02:00", now=NOW).isoformat() == "2026-08-01T08:00:00+00:00"
    for bad in ("", "yesterday", "7 days", "2026-13-01", "-7d"):
        with pytest.raises(InvalidArgument):
            parse_when(bad, now=NOW)


def test_window_rules() -> None:
    start, end, notes = resolve_window(None, None, now=NOW)
    assert end == NOW and start == NOW - timedelta(days=7) and notes == []
    _, end, notes = resolve_window("1d", "2026-09-05", now=NOW)
    assert end == NOW and notes and "future" in notes[0]
    with pytest.raises(InvalidArgument):
        resolve_window("1h", "2h", now=NOW)  # since after until
    with pytest.raises(LimitExceeded):
        resolve_window("100d", None, now=NOW)
    assert floor_day(NOW).isoformat() == "2026-09-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# GPU-hours
# ---------------------------------------------------------------------------


def test_running_jobs_are_charged_to_now_and_records_clipped_to_the_window(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", state="RUNNING", submit=3, start=2, end=None, gpus=2, gpu_types=(("a100", 2),)),
        seg("2", user="bob", submit=200, start=170, end=166),  # started before the window: 2 of 4 h count
        seg("3", state="CANCELLED", submit=5, start=4, end=4, ran=False),  # cancelled while pending
    ]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="7d")
    total = data.totals["account"]
    assert total.gpu_hours == pytest.approx(6.0)  # 2 GPUs x 2 h (to now) + 1 GPU x 2 h (clipped)
    assert total.gpu_hours_by_state == {"COMPLETED": 2.0, "RUNNING": 4.0}
    assert total.jobs == 2  # the never-started job is not usage
    assert total.median_queue_wait_s == 3600.0  # only job 1 started inside the window
    assert data.since.startswith("2026-08-25T12:00") and data.until.startswith("2026-09-01T12:00")


def test_requeue_segments_all_count_and_typed_tres_is_not_doubled(uenv: UsageEnv) -> None:
    h100 = (("h100", 4),)
    uenv.acct.records = [
        seg("9", state="PREEMPTED", submit=11, start=10, end=9, gpus=4, gpu_types=h100),
        seg("9", state="COMPLETED", submit=11, start=8, end=6, gpus=4, gpu_types=h100),
    ]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="self", group_by=["gpu_type"], since="1d")
    total = data.totals["self"]
    assert total.gpu_hours == pytest.approx(12.0) and total.jobs == 1
    assert total.gpu_hours_by_state == {"COMPLETED": 8.0, "PREEMPTED": 4.0}
    assert total.wasted_gpu_hours_by_cause == {"preempted": 4.0}
    assert [r.group for r in data.rows] == [{"gpu_type": "h100"}]
    assert data.details.requeue_segments_included


def test_a_segment_seen_by_two_windows_is_counted_once(uenv: UsageEnv) -> None:
    # 22:00 Aug 31 -> 02:00 Sep 1: returned by the finalized-days call AND the live call
    uenv.acct.records = [seg("5", submit=15, start=14, end=10, gpus=2, gpu_types=(("a100", 2),))]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="self", since="2d")
    assert len(uenv.acct.calls) == 2
    assert data.totals["self"].gpu_hours == pytest.approx(8.0)


def test_cpu_only_and_unknown_gpu_counts(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("c", submit=3, start=2, end=1, gpus=0, gpu_types=()),
        seg("u", submit=3, start=2, end=1, gpus=0, gpu_types=(), known=False),
        seg("g", submit=3, start=2, end=1, gpus=1),
    ]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="self", since="1d")
    total = data.totals["self"]
    assert total.gpu_hours == pytest.approx(1.0)
    assert total.jobs == 2 and total.gpu_count_unknown_jobs == 1  # CPU job excluded, unknown one flagged
    assert any("unknown GPU count" in w for w in data.warnings)
    # a total that leaves out jobs whose GPUs could not be counted is a lower bound, not complete
    assert not data.coverage.complete and any("GPU count" in m for m in data.coverage.missing)


# ---------------------------------------------------------------------------
# waste by cause
# ---------------------------------------------------------------------------


def _attempt(
    uenv: UsageEnv, job_id: str, *, state: str, diagnosis: str | None, result: str = "unchecked"
) -> str:
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, cluster_id="fake")
        r = f.make_run(tx, e, f.make_plan(tx, e))
        return f.make_attempt(
            tx,
            r,
            cluster_id="fake",
            job_id=job_id,
            phase="terminal",
            scheduler_state=state,
            result_state=result,
            job_owner="alice",
            diagnosis_json=jdump({"primary_code": diagnosis}) if diagnosis else None,
        )


def test_waste_by_cause_managed_vs_foreign(uenv: UsageEnv) -> None:
    _attempt(uenv, "101", state="FAILED", diagnosis="cuda_oom")
    ckpt = _attempt(uenv, "102", state="TIMEOUT", diagnosis="timeout")
    _attempt(uenv, "106", state="COMPLETED", diagnosis="success_no_outputs", result="invalid")
    with uenv.ctx.db.tx() as tx:
        run_id = tx.one("SELECT run_id FROM attempts WHERE attempt_id=?", (ckpt,))["run_id"]
        tx.insert(
            "artifacts",
            {
                "artifact_id": new_id("artifact"),
                "attempt_id": ckpt,
                "run_id": run_id,
                "name": "checkpoints",
                "role": "checkpoint",
                "kind": "dir",
                "remote_path": "/x/checkpoints",
                "digest_state": "computed",
                "integrity": "finalized",
                "observed_at": tx.now,
            },
        )
    uenv.acct.records = [
        seg("101", state="FAILED", submit=10, start=9, end=8),
        seg("102", state="TIMEOUT", submit=10, start=9, end=7),  # stopped with a checkpoint: not waste
        seg("103", state="TIMEOUT", user="bob", submit=10, start=9, end=6),  # foreign: labelled by state
        seg(
            "104", state="CANCELLED", user="bob", submit=10, start=9, end=8, gpus=2, gpu_types=(("a100", 2),)
        ),
        seg("105", state="COMPLETED", user="bob", submit=10, start=9, end=8),
        seg("106", state="COMPLETED", submit=10, start=9, end=8),
    ]
    data = usage_report(
        uenv.ctx, cluster_id="fake", scope="account", group_by=["diagnosis_class"], since="1d"
    )
    total = data.totals["account"]
    assert total.wasted_gpu_hours_by_cause == {
        "cancelled": 2.0,
        "cuda_oom": 1.0,
        "success_no_outputs": 1.0,
        "timeout": 3.0,
    }
    assert total.managed_jobs == 3 and data.details.jobs_not_managed_by_hpclab == 3
    groups = {r.group["diagnosis_class"]: r for r in data.rows}
    assert set(groups) == {"foreign", "resources", "results"}
    assert groups["foreign"].jobs == 3


def test_waste_can_be_excluded(uenv: UsageEnv) -> None:
    uenv.acct.records = [seg("1", state="FAILED", submit=3, start=2, end=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="self", include=[], since="1d")
    total = data.totals["self"]
    assert total.wasted_gpu_hours_by_cause is None and total.median_queue_wait_s is None
    assert total.walltime_accuracy is None and data.now is None


# ---------------------------------------------------------------------------
# PrivateData honesty
# ---------------------------------------------------------------------------


def test_only_own_rows_with_unknown_private_data_is_self_only_without_account_total(uenv: UsageEnv) -> None:
    # "monkeypatched sacct returning only self": accounting answered, but only with my jobs
    uenv.acct.private_data = None
    uenv.acct.records = [seg("1", submit=3, start=2, end=1, gpus=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "self_only" and data.details.visibility == "self_only"
    assert set(data.totals) == {"self"} and "account" not in data.totals
    assert "account" not in data.model_dump(mode="json")["totals"]
    assert not data.coverage.complete and any("other users" in m for m in data.coverage.missing)
    assert any("NOT account totals" in n for n in data.notes)
    assert "your jobs only" in data.summary()


def test_private_data_jobs_is_self_only(uenv: UsageEnv) -> None:
    uenv.acct.private_data = "jobs,usage"
    uenv.acct.records = [seg("1", submit=3, start=2, end=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "self_only" and set(data.totals) == {"self"}
    assert data.private_data == "jobs,usage"


def test_visible_other_users_give_an_account_total(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", submit=3, start=2, end=1),
        seg("2", user="bob", submit=3, start=2, end=1, gpus=3),
    ]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "full" and set(data.totals) == {"account"}
    assert data.totals["account"].gpu_hours == pytest.approx(4.0)
    assert {r.group["user"] for r in data.rows} == {"alice", "bob"} and data.coverage.complete


def test_only_own_rows_are_self_only_even_when_slurm_conf_private_data_is_none(uenv: UsageEnv) -> None:
    # Spec §7.2: scope=account but only self rows returned -> self_only, NO account total.
    # `scontrol show config` shows slurm.conf's PrivateData, but sacct answers from slurmdbd,
    # whose own PrivateData may hide other users: "none" here proves nothing about sacct.
    uenv.acct.private_data = "none"
    uenv.acct.records = [seg("1", submit=3, start=2, end=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "self_only" and set(data.totals) == {"self"}
    assert not data.coverage.complete
    assert any("NOT account totals" in n for n in data.notes)
    assert "your jobs only" in data.summary()


def test_empty_account_window_is_not_an_account_total(uenv: UsageEnv) -> None:
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "self_only" and "account" not in data.totals


def test_now_section_trusts_slurm_conf_private_data_for_squeue(uenv: UsageEnv) -> None:
    # squeue is answered by slurmctld, which enforces slurm.conf's PrivateData: with "none"
    # known, seeing only my jobs in the queue is the full picture.
    uenv.acct.private_data = "none"
    uenv.acct.account_jobs = [
        JobObservation(job_id="1", source="squeue", observed_at="2026-09-01T12:00:00.000Z",
                       state="RUNNING", user="alice", alloc_gpus=2)  # type: ignore[arg-type]
    ]  # fmt: skip
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", include=["now"], since="1h")
    assert data.now is not None and data.now.visibility == "full" and data.now.running_gpus == 2
    assert data.visibility == "self_only"  # the accounting part could not see anyone else


def test_coordinator_sees_others_despite_private_data(uenv: UsageEnv) -> None:
    uenv.acct.private_data = "jobs"
    uenv.acct.records = [seg("2", user="bob", submit=3, start=2, end=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="1d")
    assert data.visibility == "full" and "account" in data.totals


def test_self_scope_queries_only_me(uenv: UsageEnv) -> None:
    uenv.acct.records = [seg("1", submit=3, start=2, end=1), seg("2", user="bob", submit=3, start=2, end=1)]
    data = usage_report(uenv.ctx, cluster_id="fake", since="1d")
    assert data.scope == "self" and set(data.totals) == {"self"} and data.visibility == "full"
    assert all(c["users"] == ["alice"] for c in uenv.acct.calls)
    assert data.totals["self"].gpu_hours == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# caching of finalized days
# ---------------------------------------------------------------------------


def test_finalized_days_are_cached(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", submit=120, start=119, end=100, gpus=1),
        seg("2", user="bob", submit=80, start=79, end=60, gpus=2, gpu_types=(("a100", 2),)),
        seg("3", submit=6, start=5, end=4),
    ]
    first = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="7d")
    assert first.details.sacct_calls == 2 and first.details.cached_days == 0
    uenv.clock.advance(600)
    uenv.acct.calls.clear()
    second = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="7d")
    assert len(uenv.acct.calls) == 1 and second.details.sacct_calls == 1  # only the live tail
    assert second.details.cached_days == 7
    assert second.totals["account"].gpu_hours == pytest.approx(first.totals["account"].gpu_hours)
    assert any(s.name == "report_cache" and s.status == "cached" for s in second.coverage.sources)


def test_days_with_running_jobs_are_not_cached(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", user="bob", submit=100, start=99, end=90),
        seg("2", state="RUNNING", submit=27, start=26, end=None),  # running since yesterday 10:00
    ]
    usage_report(uenv.ctx, cluster_id="fake", scope="account", since="7d")
    uenv.acct.calls.clear()
    uenv.clock.advance(60)
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="7d")
    assert data.details.cached_days == 6 and len(uenv.acct.calls) == 2  # yesterday re-read + live
    assert data.totals["account"].gpu_hours_by_state["RUNNING"] == pytest.approx(26 + 60 / 3600, abs=1e-3)


# ---------------------------------------------------------------------------
# failures are never an empty success
# ---------------------------------------------------------------------------


def _unavailable() -> CommandFailed:
    return CommandFailed(
        "sacct failed: Problem talking to the database",
        CommandResult([], 1, "", "", 0.0),
        code=ErrorCode.UNAVAILABLE,
    )


def test_partial_window_failure_is_reported(uenv: UsageEnv) -> None:
    uenv.acct.records = [seg("1", submit=100, start=99, end=90), seg("2", submit=3, start=2, end=1)]
    uenv.acct.errors = [_unavailable(), None]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="self", since="7d")
    assert not data.coverage.complete and data.details.window_truncated
    assert len(data.details.failed_windows) == 1 and data.warnings
    assert data.totals["self"].gpu_hours == pytest.approx(1.0)  # what was readable, labelled partial


def test_all_windows_failing_raises(uenv: UsageEnv) -> None:
    uenv.acct.errors = [_unavailable(), _unavailable()]
    with pytest.raises(HpcLabError) as exc:
        usage_report(uenv.ctx, cluster_id="fake", scope="self", since="7d")
    assert exc.value.code == ErrorCode.UNAVAILABLE


@pytest.mark.parametrize(
    "error",
    [
        TransportError("GPU usage is not recorded", code=ErrorCode.UNSUPPORTED),
        AuthRequiredError("needs Duo", details={"reason": "auth"}),
    ],
)
def test_unsupported_and_auth_errors_propagate(uenv: UsageEnv, error: Exception) -> None:
    uenv.acct.errors = [None, error]
    with pytest.raises(HpcLabError) as exc:
        usage_report(uenv.ctx, cluster_id="fake", scope="self", since="7d")
    assert exc.value is error


def test_account_scope_needs_an_account(uenv: UsageEnv) -> None:
    rc = uenv.ctx.clusters.get("fake")
    uenv.ctx.clusters.register(rc.profile, UserOverlay(cluster_id="fake", user="alice"))
    with pytest.raises(InvalidArgument):
        usage_report(uenv.ctx, cluster_id="fake", scope="account")
    assert (
        usage_report(uenv.ctx, cluster_id="fake", scope="account", account="lab", since="1h").account == "lab"
    )


# ---------------------------------------------------------------------------
# grouping, medians, now
# ---------------------------------------------------------------------------


def test_grouping_limit_and_interactive_hours(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", submit=5, start=4, end=2, partition="interactive"),
        seg("2", submit=5, start=4, end=3, name="bash"),
        seg("3", user="bob", submit=5, start=4, end=3),
    ]
    data = usage_report(
        uenv.ctx, cluster_id="fake", scope="account", group_by=["user", "partition"], limit=2, since="1d"
    )
    assert data.rows_total == 3 and len(data.rows) == 2 and data.rows_truncated == 1
    assert [r.group for r in data.rows] == [
        {"user": "alice", "partition": "gpu"},
        {"user": "alice", "partition": "interactive"},
    ]
    assert data.totals["account"].interactive_hours == pytest.approx(3.0)


def test_walltime_accuracy_and_queue_wait_medians(uenv: UsageEnv) -> None:
    uenv.acct.records = [
        seg("1", submit=10, start=9, end=7, limit_h=4),  # 2h of 4h, waited 1h
        seg("2", submit=10, start=7, end=4, limit_h=4),  # 3h of 4h, waited 3h
        seg("3", state="FAILED", submit=10, start=8, end=7, limit_h=4),  # not COMPLETED: excluded
    ]
    total = usage_report(uenv.ctx, cluster_id="fake", scope="self", since="1d").totals["self"]
    assert total.walltime_accuracy == pytest.approx(0.625)
    assert total.median_queue_wait_s == pytest.approx(7200.0)


def test_now_section(uenv: UsageEnv) -> None:
    def obs(job_id: str, state: str, gpus: int | None, user: str = "alice") -> JobObservation:
        return JobObservation(
            job_id=job_id, source="squeue", observed_at="2026-09-01T12:00:00.000Z", state=state, user=user,  # type: ignore[arg-type]
            alloc_gpus=gpus,
        )  # fmt: skip

    uenv.acct.account_jobs = [obs("1", "RUNNING", 2), obs("2", "PENDING", 4, "bob")]
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", include=["now"], since="1h")
    assert data.now is not None and data.now.visibility == "full"
    assert (data.now.running_gpus, data.now.pending_gpus, data.now.users_visible) == (2, 4, 2)
    uenv.acct.account_jobs = [obs("1", "RUNNING", None)]
    uenv.acct.private_data = "jobs"
    data = usage_report(uenv.ctx, cluster_id="fake", scope="account", include=["now"], since="1h")
    assert data.now is not None and data.now.running_gpus is None and data.now.visibility == "self_only"


def test_requeued_segment_is_waste_even_when_the_window_cuts_off_the_rerun(uenv: UsageEnv) -> None:
    # A REQUEUED record is by definition not the job's final run; whether the rerun falls inside
    # the window must not change how its hours are classified.
    both = [
        seg("7", user="bob", state="REQUEUED", submit=30, start=29, end=27),
        seg("7", user="bob", state="COMPLETED", submit=6, start=5, end=3),  # rerun (eligible again at -6h)
    ]
    uenv.acct.records = [seg("8", submit=30, start=29, end=28), *both]
    full = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="2d")
    cut = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="2d", until="10h")
    waste_full = full.totals["account"].wasted_gpu_hours_by_cause or {}
    waste_cut = cut.totals["account"].wasted_gpu_hours_by_cause or {}
    assert waste_full.get("requeued") == pytest.approx(2.0)
    assert waste_cut.get("requeued") == pytest.approx(2.0), waste_cut


def test_slow_accounting_is_cut_at_the_request_budget_and_resumed_from_the_cache(uenv: UsageEnv) -> None:
    uenv.ctx.settings.request_deadline_s = 20.0
    real = uenv.acct.usage_segments

    def slow(**kw):  # type: ignore[no-untyped-def]
        uenv.clock.advance(10.0)  # every sacct call takes 10 s on a busy slurmdbd
        return real(**kw)

    uenv.acct.usage_segments = slow  # type: ignore[method-assign]
    uenv.acct.records = [
        seg("1", submit=24 * 80, start=24 * 80 - 1, end=24 * 80 - 3),
        seg("2", user="bob", submit=24 * 40, start=24 * 40 - 1, end=24 * 40 - 2),
        seg("3", submit=3, start=2, end=1),
    ]
    first = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="90d")
    assert first.details.sacct_calls == 2  # 3 x 31-day chunks + live tail; budget = 17 s
    assert first.details.skipped_windows and not first.coverage.complete
    assert any("re-run" in w for w in first.warnings)
    second = usage_report(uenv.ctx, cluster_id="fake", scope="account", since="90d")
    assert second.details.skipped_windows == [] and second.coverage.complete
    assert second.details.cached_days >= 60
    assert second.totals["account"].gpu_hours == pytest.approx(2.0 + 1.0 + 1.0)
