"""runs_list: list rows, pagination, changed_since, grid (codes/metrics/64 KiB pages), coverage."""

from __future__ import annotations

import json
from typing import Any

import pytest
from observe_support import CLUSTER, Env, observation, seed

from hpclab.domain.errors import InvalidArgument, NotFound
from hpclab.domain.states import RunStatus
from hpclab.services.observe import apply_observation, poll_cluster
from hpclab.services.registry import envelope_from_result
from hpclab.services.runs import runs_list
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_attempt, make_evaluation, make_experiment, make_plan, make_run


def experiment(env: Env) -> str:
    with env.ctx.db.tx() as tx:
        return make_experiment(tx, cluster_id=CLUSTER)


def test_list_rows_carry_code_status_and_latest_attempt(env: Env) -> None:
    eid = experiment(env)
    run_q, _ = seed(
        env,
        experiment_id=eid,
        cell={"method": "a"},
        phase="observing",
        scheduler_state="PENDING",
        pending_reason="QOSMaxGRESPerUser",
        last_observed_at=env.now(),
    )
    run_r, a_r = seed(
        env, experiment_id=eid, cell={"method": "b"}, time_left_s=600, last_observed_at=env.now()
    )
    with env.ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET progress_json=?, health_json=? WHERE attempt_id=?",
            (
                jdump({"step": 40, "total_steps": 100, "fraction": 0.4, "timelimit_risk": "low"}),
                jdump(
                    {
                        "source": "telemetry",
                        "flags": ["idle_allocation"],
                        "per_gpu_util_mean": {"0": 0.0},
                        "as_of": env.clock.now().timestamp(),
                    }
                ),
                a_r,
            ),
        )
        pid = make_plan(tx, eid)
        run_p = make_run(tx, eid, pid, cell={"method": "c"}, seed=1)
    data = runs_list(env.ctx, experiment_id=eid)

    by_id = {r.run_id: r for r in data.runs}
    assert data.total == 3 and data.view == "list"
    assert by_id[run_q].code == "Q" and by_id[run_q].latest_attempt is not None
    brief_q = by_id[run_q].latest_attempt
    assert brief_q is not None
    assert brief_q.pending_reason == "QOSMaxGRESPerUser" and brief_q.pending_category == "self_limit"
    brief_r = by_id[run_r].latest_attempt
    assert by_id[run_r].code == "R" and brief_r is not None and brief_r.time_left_s == 600
    assert (
        brief_r.health == "I" and brief_r.progress is not None and brief_r.progress.startswith("step 40/100")
    )
    assert by_id[run_p].code == "P" and by_id[run_p].latest_attempt is None
    assert data.counts["planned"] == 3 and data.counts["Q"] == 1 and data.counts["P"] == 1
    assert data.changes_cursor is not None


def test_pagination_is_stable_and_bounded(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        ids = [make_run(tx, eid, pid, cell={"i": i}, seed=None) for i in range(7)]
    seen: list[str] = []
    cursor = None
    while True:
        page = runs_list(env.ctx, experiment_id=eid, limit=3, cursor=cursor)
        assert len(page.runs) <= 3 and page.total == 7
        seen += [r.run_id for r in page.runs]
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert sorted(seen) == sorted(ids) and len(seen) == 7
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, experiment_id=eid, limit=0)


def test_states_filter_and_unknown_scope(env: Env) -> None:
    eid = experiment(env)
    run_r, a_r = seed(env, experiment_id=eid)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=a_r, obs=observation(env, a_r, "RUNNING"))
    seed(env, experiment_id=eid, phase="terminal", scheduler_state="FAILED")
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE runs SET status='failed' WHERE run_id!=?", (run_r,))
    data = runs_list(env.ctx, experiment_id=eid, states=[RunStatus.RUNNING])
    assert [r.run_id for r in data.runs] == [run_r]
    with pytest.raises(NotFound):
        runs_list(env.ctx, experiment_id="exp_01NOTTHERE00000000000000000")
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, experiment_id=eid, rows=["method"])


def test_changed_since_returns_only_runs_with_new_events(env: Env) -> None:
    eid = experiment(env)
    seed(env, experiment_id=eid)
    run2, a2 = seed(env, experiment_id=eid)
    base = runs_list(env.ctx, experiment_id=eid)
    assert base.changes_cursor is not None
    nothing = runs_list(env.ctx, experiment_id=eid, changed_since=base.changes_cursor)
    assert nothing.runs == [] and nothing.next_cursor is None
    env.clock.advance(5)
    with env.ctx.db.tx() as tx:
        apply_observation(tx, env.ctx, attempt_id=a2, obs=observation(env, a2, "COMPLETED"))
    changed = runs_list(env.ctx, experiment_id=eid, changed_since=base.changes_cursor)
    assert [r.run_id for r in changed.runs] == [run2] and changed.runs[
        0
    ].code == "E"  # completed, not yet evaluated
    # the watermark moves: asking again with the new cursor returns nothing
    again = runs_list(env.ctx, experiment_id=eid, changed_since=changed.changes_cursor)
    assert again.runs == []
    # a raw events sequence number works too
    assert [r.run_id for r in runs_list(env.ctx, experiment_id=eid, changed_since="0").runs] == [run2]
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, experiment_id=eid, changed_since="not-a-cursor")


def test_coverage_reports_squeue_freshness_honestly(env: Env) -> None:
    eid = experiment(env)
    _, aid = seed(env, experiment_id=eid)
    never = runs_list(env.ctx, experiment_id=eid)
    assert not never.coverage.complete
    assert [s.status for s in never.coverage.sources if s.name == f"squeue:{CLUSTER}"] == ["stale"]
    env.scheduler.queue = [observation(env, aid, "RUNNING")]
    poll_cluster(env.ctx, cluster_id=CLUSTER)
    fresh = runs_list(env.ctx, experiment_id=eid)
    assert fresh.coverage.complete and fresh.observed_at == env.now()
    env.clock.advance(3600)
    old = runs_list(env.ctx, experiment_id=eid)
    assert not old.coverage.complete and old.runs[0].code == "?" and old.runs[0].stale


def test_foreign_jobs_summary_comes_from_the_last_squeue_snapshot(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        tx.execute(
            "INSERT INTO observations (cluster_id, obs_key, value_json, source, observed_at) VALUES (?,?,?,?,?)",
            (
                CLUSTER,
                "queue:foreign_jobs",
                jdump(
                    {
                        "count": 3,
                        "running": 2,
                        "pending": 1,
                        "gpus": 5,
                        "gpus_unknown_jobs": 0,
                        "partitions": {"gpu": 3},
                        "unbound_hpclab_jobs": 0,
                    }
                ),
                "squeue",
                env.now(),
            ),
        )
    data = runs_list(env.ctx, experiment_id=eid)
    assert len(data.foreign_jobs) == 1
    fj = data.foreign_jobs[0]
    assert (fj.count, fj.gpus, fj.partitions, fj.stale) == (3, 5, {"gpu": 3}, False)
    dumped = data.model_dump_json()
    assert "job_ids" not in dumped  # counts only; foreign ids are never exposed as actionable


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def grid_fixture(env: Env) -> tuple[str, dict[tuple[str, int], str]]:
    """methods x seeds with one run in every code state."""
    eid = experiment(env)
    runs: dict[tuple[str, int], str] = {}
    specs: list[tuple[str, int, dict[str, Any] | None]] = [
        (
            "base",
            1,
            {"phase": "terminal", "scheduler_state": "COMPLETED", "result_state": "valid", "eval": 0.91},
        ),
        ("base", 2, {"phase": "terminal", "scheduler_state": "TIMEOUT"}),
        ("mixup", 1, {"phase": "observing", "scheduler_state": "RUNNING"}),
        ("mixup", 2, {"phase": "observing", "scheduler_state": "PENDING"}),
        ("cutmix", 1, {"phase": "terminal", "scheduler_state": "COMPLETED", "result_state": "incomplete"}),
        ("cutmix", 2, {"phase": "submission_uncertain", "scheduler_state": "UNKNOWN"}),
        ("mixup", 3, {"phase": "cancelled_before_submit", "scheduler_state": "UNKNOWN"}),
        ("base", 3, None),  # prepared, never dispatched
    ]
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for method, s, att in specs:
            rid = make_run(tx, eid, pid, cell={"method": method}, seed=s)
            runs[(method, s)] = rid
            if att is None:
                continue
            att = dict(att)
            value = att.pop("eval", None)
            aid = make_attempt(tx, rid, cluster_id=CLUSTER, last_observed_at=tx.now, **att)
            if value is not None:
                make_evaluation(tx, aid, metrics={"acc": value})
    return eid, runs


def test_grid_codes_and_metric_values_from_valid_evaluations_only(env: Env) -> None:
    eid, runs = grid_fixture(env)
    data = runs_list(env.ctx, experiment_id=eid, view="grid", metric="acc")
    grid = data.grid
    assert grid is not None and data.runs == []
    assert grid.row_axes == ["method"] and grid.col_axes == ["seed"]
    assert grid.cols == ["seed=1", "seed=2", "seed=3"]
    table = {row.key: [c.code for c in row.cells] for row in grid.rows}
    assert table == {
        "method=base": ["V", "F", "P"],
        "method=cutmix": ["C!", "U", "-"],
        "method=mixup": ["R", "Q", "X"],
    }
    values = {(row.key, i): c.value for row in grid.rows for i, c in enumerate(row.cells)}
    assert values[("method=base", 0)] == pytest.approx(0.91)
    assert all(v is None for k, v in values.items() if k != ("method=base", 0))
    base_row = next(r for r in grid.rows if r.key == "method=base")
    assert base_row.cells[0].run_id == runs[("base", 1)] and base_row.cells[1].reason == "TIMEOUT"
    assert data.counts["planned"] == 8 and data.counts["valid"] == 1 and data.counts["F"] == 1
    assert grid.legend["C!"].startswith("completed")


def test_grid_axes_validation_and_aggregation(env: Env) -> None:
    eid, _ = grid_fixture(env)
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, view="grid")
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, experiment_id=eid, view="grid", rows=["nope"])
    with pytest.raises(InvalidArgument):
        runs_list(env.ctx, experiment_id=eid, view="grid", rows=["seed"], cols=["seed"])
    agg = runs_list(env.ctx, experiment_id=eid, view="grid", rows=["method"], cols=[])
    assert agg.grid is not None and agg.grid.cols == ["all"]
    base = next(r for r in agg.grid.rows if r.key == "method=base")
    assert base.cells[0].runs == 3 and base.cells[0].code == "F"  # worst code shown, all disclosed
    assert "3 runs" in (base.cells[0].reason or "")


def test_grid_pages_stay_under_64_kib(env: Env) -> None:
    eid = experiment(env)
    long = "x" * 180
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for i in range(60):
            for s in range(32):  # wide rows (long values are clipped, so width drives the size)
                rid = make_run(
                    tx, eid, pid, cell={"method": f"{long}-{i:03d}"}, seed=s, label=f"{long}-{i}-{s}"
                )
                if s % 2:
                    make_attempt(
                        tx,
                        rid,
                        cluster_id=CLUSTER,
                        phase="terminal",
                        scheduler_state="OUT_OF_MEMORY",
                        result_state="unchecked",
                    )
    keys: list[str] = []
    cursor = None
    pages = 0
    while True:
        data = runs_list(env.ctx, experiment_id=eid, view="grid", limit=100, cursor=cursor)
        env_dict = envelope_from_result(data, request_id="r1")
        assert len(json.dumps(env_dict).encode()) <= 64 * 1024
        assert data.grid is not None and data.grid.total_rows == 60
        keys += [r.key for r in data.grid.rows]
        pages += 1
        if data.next_cursor is None:
            break
        cursor = data.next_cursor
    assert pages > 1 and len(keys) == 60 and len(set(keys)) == 60


# ---------------------------------------------------------------------------
# E: completed, evaluation pending (not a failure)
# ---------------------------------------------------------------------------


def test_e_cells_are_awaiting_evaluation_not_failures(env: Env) -> None:
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        ids = {}
        for method, seed_v, att in (
            ("a", 1, {"result_state": "valid", "eval": 0.8}),  # V
            ("a", 2, {"result_state": "valid"}),  # E: outputs valid, evaluation pending
            ("b", 1, {"result_state": "unchecked"}),  # E: outputs being checked
            ("b", 2, {"result_state": "incomplete"}),  # C!
        ):
            rid = make_run(tx, eid, pid, cell={"method": method}, seed=seed_v)
            value = att.pop("eval", None)
            aid = make_attempt(tx, rid, cluster_id=CLUSTER, phase="terminal", scheduler_state="COMPLETED",
                               last_observed_at=tx.now, **att)  # fmt: skip
            if isinstance(value, float):
                make_evaluation(tx, aid, metrics={"acc": value})
            ids[(method, seed_v)] = rid
    data = runs_list(env.ctx, experiment_id=eid, view="grid")
    grid = data.grid
    assert grid is not None
    table = {row.key: [c.code for c in row.cells] for row in grid.rows}
    assert table == {"method=a": ["V", "E"], "method=b": ["E", "C!"]}
    assert data.counts["E"] == 2 and data.counts["C!"] == 1 and "F" not in data.counts
    assert "evaluation pending" in grid.legend["E"]
    assert "invalid" in grid.legend["C!"] or "incomplete" in grid.legend["C!"]
    assert any("2 run(s) are E" in n and "not a failure" in n for n in grid.notes), grid.notes
    # aggregated cells: E is less severe than anything unfinished or failed, worse than V only
    agg = runs_list(env.ctx, experiment_id=eid, view="grid", rows=["method"], cols=[])
    assert agg.grid is not None
    worst = {row.key: row.cells[0].code for row in agg.grid.rows}
    assert worst == {"method=a": "E", "method=b": "C!"}
    listed = runs_list(env.ctx, experiment_id=eid)
    assert "E=2" in listed.summary() and "evaluation pending" in listed.summary()


# ---------------------------------------------------------------------------
# submissions waiting for SSH authentication
# ---------------------------------------------------------------------------


def _blocked_on_auth(env: Env, eid: str, *, n: int, cluster_id: str = CLUSTER) -> list[str]:
    out = []
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        for i in range(n):
            rid = make_run(tx, eid, pid, cell={"method": f"auth{i}"}, seed=i)
            phase = "staging" if i % 2 == 0 else "submission_in_flight"
            aid = make_attempt(
                tx, rid, cluster_id=cluster_id, phase=phase, scheduler_state="UNKNOWN", job_id=None
            )
            op = tx.create_operation("submit", target_type="attempt", target_id=aid, stage="staging")
            tx.set_operation(
                op,
                state="blocked",
                stage="auth_required",
                error={"code": "auth_required", "message": "ssh needs MFA",
                       "details": {"remediation": f"hpclab auth {cluster_id}"}},
            )  # fmt: skip
            tx.execute("UPDATE attempts SET operation_id=? WHERE attempt_id=?", (op, aid))
            out.append(rid)
    return out


def test_submissions_blocked_on_ssh_auth_are_noted_in_list_and_grid(env: Env) -> None:
    eid = experiment(env)
    _blocked_on_auth(env, eid, n=3)
    # a submission blocked for another reason and one of another experiment are not counted
    with env.ctx.db.tx() as tx:
        rid = make_run(tx, eid, make_plan(tx, eid), cell={"method": "quota"}, seed=9)
        aid = make_attempt(
            tx, rid, cluster_id=CLUSTER, phase="staging", scheduler_state="UNKNOWN", job_id=None
        )
        op = tx.create_operation("submit", target_type="attempt", target_id=aid, state="blocked")
        tx.set_operation(op, stage="cluster_access", error={"code": "unavailable", "message": "quota"})
        tx.execute("UPDATE attempts SET operation_id=? WHERE attempt_id=?", (op, aid))
    other = experiment(env)
    _blocked_on_auth(env, other, n=1)
    want = f"3 submission(s) waiting for SSH login on {CLUSTER}: run `hpclab auth {CLUSTER}`; they resume by themselves"
    listed = runs_list(env.ctx, experiment_id=eid)
    assert want in listed.warnings, listed.warnings
    grid = runs_list(env.ctx, experiment_id=eid, view="grid").grid
    assert grid is not None
    assert want in grid.notes, grid.notes
    # scope: the whole ledger sees both experiments
    everything = runs_list(env.ctx)
    assert any(
        w.startswith(f"4 submission(s) waiting for SSH login on {CLUSTER}") for w in everything.warnings
    )


def test_no_auth_note_once_the_submissions_resumed(env: Env) -> None:
    eid = experiment(env)
    _blocked_on_auth(env, eid, n=1)
    with env.ctx.db.tx() as tx:
        tx.execute("UPDATE operations SET state='running', error_json=NULL, stage='staging'")
    data = runs_list(env.ctx, experiment_id=eid)
    assert not any("SSH login" in w for w in data.warnings)


def test_a_changed_host_key_is_not_presented_as_a_routine_login(env: Env) -> None:
    """submission._blocked parks a changed host key on stage auth_required too, but not retryable:
    the note must not tell the user to just log in again (that is how a MITM gets accepted)."""
    eid = experiment(env)
    _blocked_on_auth(env, eid, n=1)
    with env.ctx.db.tx() as tx:
        rid = make_run(tx, eid, make_plan(tx, eid), cell={"method": "hk"}, seed=7)
        aid = make_attempt(
            tx, rid, cluster_id=CLUSTER, phase="staging", scheduler_state="UNKNOWN", job_id=None
        )
        op = tx.create_operation("submit", target_type="attempt", target_id=aid, stage="staging")
        remediation = f"the host key of {CLUSTER} changed: verify it with your cluster admins, then run `hpclab auth {CLUSTER}`"
        tx.set_operation(
            op,
            state="blocked",
            stage="auth_required",
            error={"code": "auth_required", "message": "host key verification failed", "retryable": False,
                   "details": {"remediation": remediation, "cluster_id": CLUSTER}},
        )  # fmt: skip
        tx.execute("UPDATE attempts SET operation_id=? WHERE attempt_id=?", (op, aid))
    warnings = runs_list(env.ctx, experiment_id=eid).warnings
    login = [w for w in warnings if "waiting for SSH login" in w]
    assert login == [
        f"1 submission(s) waiting for SSH login on {CLUSTER}: run `hpclab auth {CLUSTER}`; they resume by themselves"
    ], warnings
    host = [w for w in warnings if "host key" in w]
    assert len(host) == 1 and host[0].startswith("1 submission(s)"), warnings
    assert "verify" in host[0] and "resume by themselves" not in host[0]


def test_e_cells_nothing_will_evaluate_say_how_to_evaluate_them(env: Env) -> None:
    """evaluation.auto is only enqueued when the spec declares an evaluation (and a skipped one is
    never retried): such E cells stay E forever unless someone runs evaluation_collect."""
    eid = experiment(env)
    with env.ctx.db.tx() as tx:
        pid = make_plan(tx, eid)
        aids = {}
        for name, rs in (("stuck", "valid"), ("queued", "valid"), ("checking", "unchecked")):
            rid = make_run(tx, eid, pid, cell={"method": name}, seed=1)
            aids[name] = make_attempt(tx, rid, cluster_id=CLUSTER, phase="terminal", scheduler_state="COMPLETED",
                                      result_state=rs, last_observed_at=tx.now)  # fmt: skip
        tx.enqueue("evaluation.auto", target_id=aids["queued"], payload={"attempt_id": aids["queued"]},
                   dedupe_key=f"evaluation.auto:{aids['queued']}")  # fmt: skip
    grid = runs_list(env.ctx, experiment_id=eid, view="grid").grid
    assert grid is not None
    note = next((n for n in grid.notes if "run(s) are E" in n), "")
    assert note.startswith("3 run(s) are E") and "not a failure" in note, grid.notes
    assert "1 of them has no evaluation queued" in note, note
    assert f'evaluation_collect(run_set_ref="{eid}", idempotency_key="<new key>")' in note, note
    # once every E cell has its evaluation queued (or is still being checked) there is nothing to do
    with env.ctx.db.tx() as tx:
        tx.enqueue("evaluation.auto", target_id=aids["stuck"], payload={"attempt_id": aids["stuck"]},
                   dedupe_key=f"evaluation.auto:{aids['stuck']}")  # fmt: skip
    grid = runs_list(env.ctx, experiment_id=eid, view="grid").grid
    assert grid is not None
    note = next((n for n in grid.notes if "run(s) are E" in n), "")
    assert "evaluation_collect" not in note and "no evaluation queued" not in note, note
