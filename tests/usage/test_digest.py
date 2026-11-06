"""lab_digest: every number comes from the ledger (checked against direct SQL sums); the only
remote call is the optional usage section, labelled with its visibility."""

from __future__ import annotations

from datetime import timedelta

import pytest
from usage_fakes import UsageEnv, seg

from hpclab.domain.clock import iso
from hpclab.domain.errors import InvalidArgument
from hpclab.domain.ids import new_id
from hpclab.profiles.schema import ClusterProfile, UserOverlay
from hpclab.services.digest import lab_digest
from hpclab.services.usage import usage_report
from hpclab.storage.db import Tx, jdump
from hpclab.testing import factories as f


def _reserve(
    tx: Tx, experiment_id: str, attempt_id: str, *, state: str, reserved: float, settled: float | None
) -> None:
    tx.insert(
        "budget_reservations",
        {
            "reservation_id": new_id("reservation"),
            "experiment_id": experiment_id,
            "attempt_id": attempt_id,
            "reserved": reserved,
            "settled_consumed": settled,
            "state": state,
            "created_at": tx.now,
            "updated_at": tx.now,
        },
    )


@pytest.fixture
def ledger(uenv: UsageEnv) -> dict[str, str]:
    """One active experiment with a valid, a failed, an uncertain and a long-pending cell, plus an
    old finished experiment and an attempt on another cluster."""
    ids: dict[str, str] = {}
    past = iso(uenv.clock.now() - timedelta(hours=10))
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, name="mixup-ablation", cluster_id="fake")
        p = f.make_plan(tx, e)
        a, b, c, d = (f.make_run(tx, e, p, seed=s, label=f"seed-{s}") for s in (1, 2, 3, 4))
        f.make_run(tx, e, p, seed=9, kind="smoke", label="smoke")
        ok = f.make_attempt(tx, a, cluster_id="fake", phase="terminal", scheduler_state="COMPLETED",
                            result_state="valid", terminal_at=tx.now, gpus=1, elapsed_s=1800)  # fmt: skip
        f.make_evaluation(tx, ok, metrics={"acc": 0.9})
        bad = f.make_attempt(tx, b, cluster_id="fake", phase="terminal", scheduler_state="FAILED",
                             terminal_at=tx.now, gpus=2, elapsed_s=3600,
                             diagnosis_json=jdump({"primary_code": "cuda_oom"}))  # fmt: skip
        lost = f.make_attempt(tx, c, cluster_id="fake", phase="submission_uncertain", scheduler_state="UNKNOWN",
                              job_id=None, uncertain_since=tx.now)  # fmt: skip
        waiting = f.make_attempt(tx, d, cluster_id="fake", phase="observing", scheduler_state="PENDING",
                                 job_id="4711", job_submit_time=past, pending_reason="QOSMaxGRESPerUser",
                                 last_observed_at=iso(uenv.clock.now() + timedelta(hours=1)))  # fmt: skip
        _reserve(tx, e, ok, state="settled", reserved=4.0, settled=0.5)
        _reserve(tx, e, bad, state="unknown", reserved=4.0, settled=None)
        _reserve(tx, e, waiting, state="reserved", reserved=4.0, settled=None)
        # an old, finished experiment (not news) and noise on another cluster
        old = f.make_experiment(tx, name="old", cluster_id="fake", created_at="2026-01-01T00:00:00.000Z",
                                updated_at="2026-01-01T00:00:00.000Z")  # fmt: skip
        other = f.make_experiment(tx, name="elsewhere", cluster_id="other")
        other_run = f.make_run(tx, other, f.make_plan(tx, other))
        f.make_attempt(tx, other_run, cluster_id="other", phase="terminal", scheduler_state="TIMEOUT",
                       terminal_at=tx.now, gpus=8, elapsed_s=7200)  # fmt: skip
        tx.insert(
            "evidence_bundles",
            {
                "bundle_id": new_id("evidence"),
                "comparison_id": None,
                "run_set_ref": e,
                "purpose": "paper table 2",
                "manifest_json": "{}",
                "files_json": "{}",
                "coverage_json": "{}",
                "created_at": tx.now,
            },
        )
        op = tx.create_operation("submit", target_type="attempt", target_id=lost, state="blocked")
        tx.set_operation(op, error={"code": "auth_required", "message": "run hpclab auth fake"})
        ids.update(experiment=e, old=old, ok=ok, bad=bad, lost=lost, waiting=waiting, op=op)
    uenv.clock.advance(3600)
    return ids


def test_digest_numbers_match_the_ledger(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    d = lab_digest(uenv.ctx, since="7d", cluster_id="fake")
    db = uenv.ctx.db
    settled = db.one(
        "SELECT SUM(settled_consumed) AS s, COUNT(*) AS n FROM budget_reservations WHERE state='settled'"
    )
    assert (
        d.gpu_hours.settled_gpu_hours == pytest.approx(settled["s"])
        and d.gpu_hours.settled_attempts == settled["n"]
    )
    assert d.gpu_hours.unknown_settlements == 1
    assert d.gpu_hours.open_reserved_gpu_hours == pytest.approx(4.0) and d.gpu_hours.open_reservations == 1
    lost_row = db.one(
        "SELECT gpus * elapsed_s / 3600.0 AS gh FROM attempts WHERE attempt_id=?", (ledger["bad"],)
    )
    assert [(x.cause, x.category, x.attempts, x.gpu_hours_lost) for x in d.failures] == [
        ("cuda_oom", "resources", 1, pytest.approx(lost_row["gh"]))
    ]  # the TIMEOUT on another cluster is filtered out
    (exp,) = d.experiments
    assert exp.experiment_id == ledger["experiment"] and exp.planned == 4 and exp.valid == 1
    assert exp.counts["F"] == 1 and exp.counts["U"] == 1 and exp.counts["Q"] == 1 and not exp.quiet
    kinds = {(a.kind, a.ref) for a in d.attention}
    assert kinds == {("submission_uncertain", ledger["lost"]), ("blocked_operation", ledger["op"])}
    (pend,) = d.long_pending
    assert pend.attempt_id == ledger["waiting"] and pend.reason == "QOSMaxGRESPerUser"
    assert pend.pending_for_s == 11 * 3600
    assert len(d.evidence) == 1 and d.evidence[0].purpose == "paper table 2"
    md = d.markdown
    for part in ("## Experiments", "mixup-ablation", "## Failures by cause", "cuda_oom", "0.5 GPU-hours",
                 "## Needs attention", ledger["lost"], "## Long-pending jobs", "QOSMaxGRESPerUser"):  # fmt: skip
        assert part in md, part
    assert d.coverage.complete and d.usage is None
    assert uenv.acct.remote_calls == 0  # without an account: ledger only, no cluster calls


def test_digest_without_cluster_filter_sees_all_clusters(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    d = lab_digest(uenv.ctx, since="1d")
    assert {x.cause for x in d.failures} == {"cuda_oom", "timeout"}
    assert "elsewhere" in {e.name for e in d.experiments}


def test_digest_usage_section_equals_usage_report(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    uenv.acct.records = [seg("1", submit=5, start=4, end=2, gpus=2, gpu_types=(("a100", 2),)),
                         seg("2", user="bob", submit=5, start=4, end=3)]  # fmt: skip
    d = lab_digest(uenv.ctx, since="1d", account="lab", cluster_id="fake")
    assert d.usage is not None and d.usage.visibility == "full"
    direct = usage_report(
        uenv.ctx, cluster_id="fake", scope="account", account="lab", since=d.since, until=d.until
    )
    assert d.usage.totals == direct.totals
    assert "## Usage: account lab" in d.markdown and "5.0 over 2 job(s)" in d.markdown.replace(
        "GPU-hours: ", ""
    )


def test_self_only_usage_is_labelled_not_lab_totals(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    uenv.acct.private_data = None
    uenv.acct.records = [seg("1", submit=5, start=4, end=2)]
    d = lab_digest(uenv.ctx, since="1d", account="lab")  # single registered cluster: inferred
    assert d.usage is not None and d.usage.visibility == "self_only" and set(d.usage.totals) == {"self"}
    assert "visibility: self_only" in d.markdown and "NOT lab totals" in d.markdown
    assert not any(s.status == "ok" for s in d.coverage.sources if s.name == "usage_report")


def test_usage_failure_keeps_the_digest(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    from hpclab.adapters.base import AuthRequiredError

    uenv.acct.errors = [AuthRequiredError("needs Duo", details={"reason": "auth"})]
    d = lab_digest(uenv.ctx, since="1d", account="lab")
    assert d.usage is None and d.usage_error is not None and d.usage_error["code"] == "auth_required"
    assert "Usage unavailable (auth_required)" in d.markdown and not d.coverage.complete
    assert d.failures  # the ledger part is still there


def test_digest_input_validation(uenv: UsageEnv) -> None:
    with pytest.raises(InvalidArgument):
        lab_digest(uenv.ctx, since="last week")
    with pytest.raises(InvalidArgument):
        lab_digest(uenv.ctx, since="1h", until="2h")
    uenv.ctx.clusters.register(
        ClusterProfile(cluster_id="two", display_name="Two", transport="local"), UserOverlay(cluster_id="two")
    )
    with pytest.raises(InvalidArgument):
        lab_digest(uenv.ctx, since="1d", account="lab")  # which cluster?
    empty = lab_digest(uenv.ctx, since="1d")
    assert "No experiment activity" in empty.markdown and empty.failures == []


def test_stale_running_cells_are_unknown_like_the_grid(uenv: UsageEnv) -> None:
    """A RUNNING cell whose last observation is a day old is '?' in the runs_list grid; the
    digest uses the same coverage codes, so it must not report it as running."""
    from hpclab.services.runs import runs_list

    day_ago = iso(uenv.clock.now() - timedelta(days=1))
    fresh = iso(uenv.clock.now())
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, name="poller-died", cluster_id="fake")
        p = f.make_plan(tx, e)
        stale_run, live_run = (f.make_run(tx, e, p, seed=s, label=f"seed-{s}") for s in (1, 2))
        f.make_attempt(tx, stale_run, cluster_id="fake", phase="observing", scheduler_state="RUNNING",
                       job_id="11", last_observed_at=day_ago)  # fmt: skip
        f.make_attempt(tx, live_run, cluster_id="fake", phase="observing", scheduler_state="RUNNING",
                       job_id="12", last_observed_at=fresh)  # fmt: skip
    grid = runs_list(uenv.ctx, experiment_id=e)
    grid_codes = sorted(r.code for r in grid.runs)
    assert grid_codes == ["?", "R"]
    d = lab_digest(uenv.ctx, since="7d", cluster_id="fake")
    (exp,) = d.experiments
    assert exp.counts.get("R", 0) == 1 and exp.counts.get("?", 0) == 1, exp.counts


def test_timeout_with_a_finalized_checkpoint_is_not_lost_gpu_hours(uenv: UsageEnv) -> None:
    """Same rule as usage_report's waste: a walltime stop that left a finalized checkpoint is
    resumable progress, not lost GPU-hours (the attempt still counts as a TIMEOUT)."""
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, name="long-train", cluster_id="fake")
        p = f.make_plan(tx, e)
        r1, r2 = (f.make_run(tx, e, p, seed=s, label=f"seed-{s}") for s in (1, 2))
        kept = f.make_attempt(tx, r1, cluster_id="fake", phase="terminal", scheduler_state="TIMEOUT",
                              terminal_at=tx.now, gpus=4, elapsed_s=7200,
                              diagnosis_json=jdump({"primary_code": "timeout"}))  # fmt: skip
        f.make_attempt(tx, r2, cluster_id="fake", phase="terminal", scheduler_state="TIMEOUT",
                       terminal_at=tx.now, gpus=4, elapsed_s=3600,
                       diagnosis_json=jdump({"primary_code": "timeout"}))  # fmt: skip
        tx.insert(
            "artifacts",
            {"artifact_id": new_id("artifact"), "attempt_id": kept, "run_id": r1, "name": "checkpoints",
             "role": "checkpoint", "kind": "dir", "remote_path": "/x/ckpt", "digest_state": "computed",
             "integrity": "finalized", "observed_at": tx.now},
        )  # fmt: skip
    d = lab_digest(uenv.ctx, since="1d", cluster_id="fake")
    (timeout,) = [x for x in d.failures if x.cause == "timeout"]
    assert timeout.attempts == 2 and timeout.checkpointed_attempts == 1
    assert timeout.gpu_hours_lost == pytest.approx(4.0)  # only the attempt without a checkpoint
    assert "checkpoint" in d.markdown


def test_evaluation_pending_is_counted_separately_from_bad_results(uenv: UsageEnv) -> None:
    """E (completed, evaluation pending) is not a failure and not a bad result: its own column,
    and the cell is still unfinished (it has no valid result yet)."""
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, name="awaiting-eval", cluster_id="fake")
        p = f.make_plan(tx, e)
        r1, r2, r3 = (f.make_run(tx, e, p, seed=s, label=f"seed-{s}") for s in (1, 2, 3))
        for rid, rs in ((r1, "valid"), (r2, "unchecked"), (r3, "incomplete")):
            f.make_attempt(tx, rid, cluster_id="fake", phase="terminal", scheduler_state="COMPLETED",
                           result_state=rs, terminal_at=tx.now, gpus=1, elapsed_s=60)  # fmt: skip
    d = lab_digest(uenv.ctx, since="1d", cluster_id="fake")
    (exp,) = d.experiments
    assert exp.counts.get("E") == 2 and exp.counts.get("C!") == 1 and exp.counts.get("F", 0) == 0
    assert exp.unfinished == 2 and exp.valid == 0
    header = next(line for line in d.markdown.splitlines() if line.startswith("| experiment"))
    assert "evaluation pending" in header and "completed, unverified" not in header
    row = next(line for line in d.markdown.splitlines() if line.startswith("| awaiting-eval"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    vals = [c.strip() for c in row.strip("|").split("|")]
    by = dict(zip(cols, vals, strict=True))
    assert by["evaluation pending"] == "2" and by["completed, bad result"] == "1" and by["failed"] == "0"
    # the incomplete outputs are a failure class of their own; E is never listed as a failure
    assert sum(x.attempts for x in d.failures) == 1


def _workflow(tx: Tx, experiment_id: str, plan_id: str, cells: dict[str, str], *, state: str) -> str:
    wid = new_id("workflow")
    tx.insert(
        "workflows",
        {"workflow_id": wid, "experiment_id": experiment_id, "plan_id": plan_id, "plan_hash": "sha256:x",
         "max_parallel": 2, "dispatch_open": 0, "dispatch_state": "closed", "state": state,
         "created_at": tx.now, "updated_at": tx.now},
    )  # fmt: skip
    for i, (rid, cell_state) in enumerate(cells.items()):
        tx.insert(
            "workflow_cells",
            {"workflow_id": wid, "run_id": rid, "ordinal": i, "cell_state": cell_state,
             "dispatch_key": f"wf:{wid}:{rid}", "updated_at": tx.now},
        )  # fmt: skip
    return wid


def test_cells_of_a_cancelled_workflow_have_their_own_column(uenv: UsageEnv) -> None:
    """Never-dispatched cells of a cancelled workflow are X (not P): the table must still show
    them, so its columns add up to the planned total."""
    with uenv.ctx.db.tx() as tx:
        e = f.make_experiment(tx, name="stopped-sweep", cluster_id="fake")
        p = f.make_plan(tx, e)
        r1, r2, r3 = (f.make_run(tx, e, p, seed=s, label=f"seed-{s}") for s in (1, 2, 3))
        ok = f.make_attempt(tx, r1, cluster_id="fake", phase="terminal", scheduler_state="COMPLETED",
                            result_state="valid", terminal_at=tx.now, gpus=1, elapsed_s=60)  # fmt: skip
        f.make_evaluation(tx, ok, metrics={"acc": 0.9})
        _workflow(tx, e, p, {r1: "succeeded", r2: "cancelled", r3: "cancelled"}, state="cancelled")
    d = lab_digest(uenv.ctx, since="1d", cluster_id="fake")
    (exp,) = d.experiments
    assert exp.counts.get("X") == 2 and exp.counts.get("P", 0) == 0 and exp.planned == 3
    header = next(line for line in d.markdown.splitlines() if line.startswith("| experiment"))
    row = next(line for line in d.markdown.splitlines() if line.startswith("| stopped-sweep"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    by = dict(zip(cols, [c.strip() for c in row.strip("|").split("|")], strict=True))
    assert by["cancelled"] == "2" and by["not dispatched"] == "0"
    counted = [c for c in cols if c not in ("experiment", "planned", "last activity")]
    assert sum(int(by[c]) for c in counted) == 3, by  # every planned cell is in exactly one column


def test_blocked_operations_are_scoped_to_the_digest_cluster(uenv: UsageEnv, ledger: dict[str, str]) -> None:
    """A digest for one cluster must not ask the user to log in to another one."""
    with uenv.ctx.db.tx() as tx:
        other = f.make_experiment(tx, name="other-sweep", cluster_id="other")
        run = f.make_run(tx, other, f.make_plan(tx, other))
        att = f.make_attempt(
            tx, run, cluster_id="other", phase="staging", scheduler_state="UNKNOWN", job_id=None
        )
        op = tx.create_operation("submit", target_type="attempt", target_id=att, state="blocked")
        tx.set_operation(op, stage="auth_required", error={
            "code": "auth_required", "message": "ssh needs MFA", "retryable": True,
            "details": {"remediation": "hpclab auth other", "cluster_id": "other"}})  # fmt: skip
    fake = lab_digest(uenv.ctx, since="7d", cluster_id="fake")
    assert {a.ref for a in fake.attention if a.kind == "blocked_operation"} == {ledger["op"]}
    assert "hpclab auth other" not in fake.markdown
    everything = lab_digest(uenv.ctx, since="7d")
    item = next(a for a in everything.attention if a.ref == op)
    assert "other" in (item.detail or "") and "hpclab auth other" in (item.detail or ""), item
