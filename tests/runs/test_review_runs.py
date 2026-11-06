"""Adversarial review of the read side (runs_list, run_inspect, diagnose_attempt, run_logs): each
test pins down a defect found in review."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from observe_support import CLUSTER, Env, observation, seed

from hpclab.domain.errors import HpcLabError, InvalidArgument
from hpclab.services import runs
from hpclab.services.logs import run_logs
from hpclab.services.observe import apply_observation
from hpclab.services.registry import envelope_from_result
from hpclab.services.runs import JobRef, diagnose_attempt, run_inspect, runs_list
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_experiment, make_plan, make_run


def experiment(env: Env) -> str:
    with env.ctx.db.tx() as tx:
        return make_experiment(tx, cluster_id=CLUSTER)


def observe(env: Env, aid: str, state: str) -> None:
    env.clock.advance(1)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=aid, obs=observation(env, aid, state))


# ---------------------------------------------------------------------------
# runs_list(changed_since): pagination and watermarks
# ---------------------------------------------------------------------------


def test_changed_since_pages_advance_when_the_cursor_is_passed_back(env: Env) -> None:
    """The standard pagination protocol (same params + cursor=next_cursor) must advance, not
    return the first page again forever."""
    eid = experiment(env)
    changed: list[str] = []
    for i in range(5):
        rid, aid = seed(env, experiment_id=eid, cell={"i": i})
        observe(env, aid, "PENDING")
        changed.append(rid)
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page = runs_list(env.ctx, experiment_id=eid, changed_since="0", limit=2, cursor=cursor)
        seen += [r.run_id for r in page.runs]
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert sorted(seen) == sorted(changed) and len(seen) == 5


def _racing_watermark(env: Env, monkeypatch: pytest.MonkeyPatch, change: Any) -> None:
    """``change()`` commits (like a concurrent poll/prepare) just before the watermark is read."""
    original = runs._max_seq
    fired = {"done": False}

    def racing(ctx: Any, floor: int) -> int:
        if not fired["done"]:
            fired["done"] = True
            change()
        return original(ctx, floor)

    monkeypatch.setattr(runs, "_max_seq", racing)


def test_list_changes_cursor_never_skips_a_run_added_while_listing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every change is either in the page just returned or in the next changed_since call;
    otherwise the client never learns about it."""
    eid = experiment(env)
    seed(env, experiment_id=eid)
    added: list[str] = []

    def add_run() -> None:
        with env.ctx.db.tx() as tx:
            pid = make_plan(tx, eid)
            rid = make_run(tx, eid, pid, cell={"late": 1}, seed=None)
            tx.event("run", rid, "run.created", {})
        added.append(rid)

    _racing_watermark(env, monkeypatch, add_run)
    listing = runs_list(env.ctx, experiment_id=eid)
    monkeypatch.undo()
    later = runs_list(env.ctx, experiment_id=eid, changed_since=listing.changes_cursor)
    listed = {r.run_id for r in listing.runs} | {r.run_id for r in later.runs}
    assert added[0] in listed


def test_changed_watermark_never_skips_a_run_that_changed_while_listing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    eid = experiment(env)
    run_a, a_a = seed(env, experiment_id=eid)
    run_b, a_b = seed(env, experiment_id=eid)
    observe(env, a_a, "RUNNING")
    since = runs_list(env.ctx, experiment_id=eid).changes_cursor
    observe(env, a_a, "COMPLETED")
    _racing_watermark(env, monkeypatch, lambda: observe(env, a_b, "FAILED"))
    first = runs_list(env.ctx, experiment_id=eid, changed_since=since)
    monkeypatch.undo()
    later = runs_list(env.ctx, experiment_id=eid, changed_since=first.changes_cursor)
    listed = {r.run_id for r in first.runs} | {r.run_id for r in later.runs}
    assert listed == {run_a, run_b}


# ---------------------------------------------------------------------------
# grid: bounded output
# ---------------------------------------------------------------------------


def test_grid_with_too_many_columns_is_refused_not_unbounded(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for i in range(600):
            make_run(tx, eid, pid, cell={"lr": i}, seed=None)
    with pytest.raises(InvalidArgument) as exc:
        runs_list(env.ctx, experiment_id=eid, view="grid", rows=[], cols=["lr"])
    assert "columns" in exc.value.message
    # the default layout (the varying axis as rows) pages fine
    data = runs_list(env.ctx, experiment_id=eid, view="grid", limit=100)
    assert data.grid is not None and data.grid.total_rows == 600
    assert len(json.dumps(envelope_from_result(data, request_id="r")).encode()) <= 64 * 1024


def test_grid_with_long_axis_values_stays_under_64_kib(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for i in range(40):
            for s in range(64):
                make_run(tx, eid, pid, cell={"prompt": f"{i:03d}-" + "p" * 2000}, seed=s)
    cursor = None
    rows = 0
    for _ in range(50):
        data = runs_list(env.ctx, experiment_id=eid, view="grid", limit=100, cursor=cursor)
        assert len(json.dumps(envelope_from_result(data, request_id="r")).encode()) <= 64 * 1024
        assert data.grid is not None and len(data.grid.cols) == 64
        rows += len(data.grid.rows)
        cursor = data.next_cursor
        if cursor is None:
            break
    assert rows == 40
    # the long values as column labels (outside the per-row byte budget)
    flipped = runs_list(env.ctx, experiment_id=eid, view="grid", rows=["seed"], cols=["prompt"], limit=100)
    assert flipped.grid is not None and len(flipped.grid.cols) == 40
    assert len(set(flipped.grid.cols)) == 40  # clipped labels stay distinguishable
    assert len(json.dumps(envelope_from_result(flipped, request_id="r")).encode()) <= 64 * 1024


def test_list_rows_clip_long_cell_values(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for i in range(100):
            make_run(tx, eid, pid, cell={"prompt": f"{i}-" + "q" * 5000}, seed=None, label=f"r{i}")
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        data = runs_list(env.ctx, experiment_id=eid, limit=100, cursor=cursor)
        assert len(json.dumps(envelope_from_result(data, request_id="r")).encode()) <= 64 * 1024
        assert all(len(str(r.cell["prompt"])) <= 300 for r in data.runs)
        seen += [r.run_id for r in data.runs]
        cursor = data.next_cursor
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 100 and data.total == 100


def test_changed_pages_cut_by_size_continue_without_gaps(env: Env) -> None:
    eid = experiment(env)
    ids = []
    for i in range(60):
        rid, aid = seed(env, experiment_id=eid, cell={"prompt": f"{i}-" + "z" * 5000})
        env.clock.advance(1)
        with env.ctx.db.tx() as tx:
            obs = observation(env, aid, "PENDING", reason="ReqNodeNotAvail, UnavailableNodes:" + "n" * 900)
            apply_observation(tx, env.ctx, attempt_id=aid, obs=obs)
        ids.append(rid)
    seen: list[str] = []
    cursor = None
    pages = 0
    for _ in range(20):
        page = runs_list(env.ctx, experiment_id=eid, changed_since="0", limit=100, cursor=cursor)
        assert len(json.dumps(envelope_from_result(page, request_id="r")).encode()) <= 64 * 1024
        seen += [r.run_id for r in page.runs]
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break
    assert pages > 1 and sorted(seen) == sorted(ids) and len(seen) == 60


# ---------------------------------------------------------------------------
# run_inspect
# ---------------------------------------------------------------------------


def test_job_ref_with_a_malformed_submit_time_is_invalid_argument_not_internal(env: Env) -> None:
    seed(env, job_id="777", job_submit_time="2026-09-01T11:00:00.000Z")
    with pytest.raises(InvalidArgument):
        run_inspect(env.ctx, job_ref=JobRef(cluster_id=CLUSTER, job_id="777", submit_time="yesterday"))


def test_gpu_probe_rate_limit_holds_for_concurrent_requests(env: Env) -> None:
    """Check-then-act: two concurrent run_inspect(probe='gpu') must not both run srun."""
    run_id, aid = seed(env, last_observed_at=env.now())
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    env.scheduler.block = threading.Event()
    results: dict[str, Any] = {}

    def first() -> None:
        results["first"] = run_inspect(env.ctx, run_id=run_id, probe="gpu")

    t = threading.Thread(target=first)
    t.start()
    for _ in range(200):  # wait until the first probe is inside its identity check
        if env.scheduler.calls:
            break
        threading.Event().wait(0.01)
    second = run_inspect(env.ctx, run_id=run_id, probe="gpu")
    env.scheduler.block.set()
    t.join(10)
    assert second.gpu_probe is not None and second.gpu_probe.status == "rate_limited"
    assert results["first"].gpu_probe.status == "ok"
    squeue_calls = [c for c in env.scheduler.calls if c.startswith("squeue:")]
    assert len(squeue_calls) == 1 and len([c for c in env.scheduler.calls if c.startswith("gpu:")]) == 1


# ---------------------------------------------------------------------------
# diagnose_attempt: never overwrite a log-based diagnosis with a log-less one
# ---------------------------------------------------------------------------


def test_diagnosis_from_purged_logs_never_replaces_the_stored_one(env: Env) -> None:
    stored = {
        "items": [],
        "primary_code": "cuda_oom",
        "summary": "CUDA out of memory (from the logs at finalize time)",
        "last_error_line": "torch.OutOfMemoryError: CUDA out of memory.",
    }
    _, aid = seed(
        env,
        phase="terminal",
        scheduler_state="FAILED",
        exit_code=1,
        node_list="n1",
        elapsed_s=100,
        app_state="exited_error",
        diagnosis_json=jdump(stored),
    )
    # the scratch logs have been purged since finalize
    diag = diagnose_attempt(env.ctx, attempt_id=aid, fetch_logs=True)
    assert diag.primary_code == "cuda_oom"
    assert '"primary_code":"cuda_oom"' in env.attempt(aid)["diagnosis_json"]


def test_diagnosis_with_readable_logs_refreshes_the_cache(env: Env) -> None:
    _, aid = seed(
        env,
        phase="terminal",
        scheduler_state="FAILED",
        exit_code=1,
        node_list="n1",
        elapsed_s=100,
        app_state="exited_error",
        diagnosis_json=jdump({"items": [], "primary_code": None, "summary": "nothing yet"}),
    )
    d = env.attempt(aid)["remote_dir"]
    env.transport.put(f"{d}/logs/stdout.log", "starting\n")
    env.transport.put(
        f"{d}/logs/stderr.log",
        "Traceback (most recent call last):\nModuleNotFoundError: No module named 'timm'\n",
    )
    assert diagnose_attempt(env.ctx, attempt_id=aid).primary_code == "import_error"
    assert '"primary_code":"import_error"' in env.attempt(aid)["diagnosis_json"]


# ---------------------------------------------------------------------------
# run_logs: typed errors only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tail_lines", [None, 5])
def test_unreadable_log_is_a_typed_error(env: Env, tail_lines: int | None) -> None:
    _, aid = seed(env)
    env.transport.put(f"{env.attempt(aid)['remote_dir']}/logs/stdout.log", "x\n")
    method = "read_tail" if tail_lines else "read_range"
    env.transport.fail[method] = PermissionError(13, "Permission denied")
    with pytest.raises(HpcLabError) as exc:
        run_logs(env.ctx, attempt_id=aid, tail_lines=tail_lines)
    assert exc.value.code in ("unavailable", "unauthorized")


# ---------------------------------------------------------------------------
# bounded latency: run_inspect must not spend its deadline on reads that cannot succeed
# ---------------------------------------------------------------------------


def test_on_demand_refresh_does_not_spend_its_deadline_on_other_attempts_views(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """refresh=True polls squeue within the deadline; recomputing the cached progress/health of
    every running attempt on the cluster is the periodic poller's job, not the caller's."""
    import time

    from hpclab.services import observe as observe_mod

    _, aid = seed(env, last_observed_at=env.now())
    for _ in range(3):
        seed(env, job_id=None, phase="observing")  # other running attempts on the cluster

    def slow_views(*args: Any, **kwargs: Any) -> Any:
        time.sleep(1.5)
        return observe_mod.LiveViews()

    monkeypatch.setattr(observe_mod, "compute_live_views", slow_views)
    env.scheduler.queue = [observation(env, aid, "RUNNING", elapsed_s=5, time_limit_s=100)]
    res = observe_mod.refresh_attempts(env.ctx, attempt_ids=[aid], deadline_s=1.0)
    assert res.refreshed == [aid] and not res.stale, res.notes


def test_unreachable_cluster_short_circuits_the_remaining_remote_reads(env: Env) -> None:
    from hpclab.adapters.base import TransportTimeout

    run_id, _ = seed(env, last_observed_at=env.now(), node_list="n1", elapsed_s=50)
    env.transport.fail["read_tail"] = TransportTimeout("read timed out (NFS not responding)")
    env.transport.fail["stat"] = TransportTimeout("stat timed out")
    data = run_inspect(env.ctx, run_id=run_id)
    reads = [c for c in env.transport.calls if c[0] in ("read_tail", "read_range", "stat")]
    assert len(reads) == 1, reads  # the first timeout stops the others
    assert any(s.name == "metrics" and s.status == "unavailable" for s in data.coverage.sources)


def test_unreachable_cluster_skips_log_tails_for_an_on_demand_diagnosis(env: Env) -> None:
    from hpclab.adapters.base import TransportTimeout

    run_id, aid = seed(
        env, phase="terminal", scheduler_state="FAILED", exit_code=1, node_list="n1", elapsed_s=30
    )
    env.transport.fail["read_tail"] = TransportTimeout("read timed out")
    data = run_inspect(env.ctx, run_id=run_id)
    reads = [c for c in env.transport.calls if c[0] in ("read_tail", "read_range", "stat")]
    assert len(reads) == 1, reads
    assert data.diagnosis is not None  # from scheduler facts, disclosed in coverage
    assert any(s.name == "logs" and s.status != "ok" for s in data.coverage.sources)
    assert env.attempt(aid)["diagnosis_json"] is None  # a log-less diagnosis is not cached


def test_stale_cached_health_is_not_shown_as_ok_in_listings(env: Env) -> None:
    eid = experiment(env)
    old = env.clock.now().timestamp() - 3 * 3600
    seed(
        env,
        experiment_id=eid,
        last_observed_at=env.now(),
        health_json=jdump(
            {"source": "telemetry", "flags": [], "per_gpu_util_mean": {"0": 90.0}, "as_of": old}
        ),
    )
    brief = runs_list(env.ctx, experiment_id=eid).runs[0].latest_attempt
    assert brief is not None and brief.health == "-"
