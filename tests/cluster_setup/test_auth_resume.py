"""auth_resume: after the user logs in again, work parked on ``auth_required`` runs at once.

Before, a re-authenticated user waited out every pause: outbox items parked for
``auth_retry_s`` and the observer's poll backoff (``AUTH_PAUSE_S``, or exponential backoff
after repeated failures). ``auth_resume`` makes those items due now and clears the backoff, for
one cluster or all of them, and records an event.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from hpclab.domain.clock import FakeClock, iso
from hpclab.domain.errors import NotFound
from hpclab.domain.ids import new_id
from hpclab.profiles.schema import ClusterProfile, PartitionRule, SitePolicy, UserOverlay
from hpclab.services.app import METHODS, build_registry
from hpclab.services.audit import events_list
from hpclab.services.clusters import AuthResumeData, auth_resume
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import HandlerResult, OutboxItem, OutboxRunner
from hpclab.storage.db import jdump
from hpclab.testing.factories import make_attempt, make_experiment, make_plan, make_run

MakeCtx = Callable[..., ServiceContext]
AUTH = "auth_required: ssh master is gone (BatchMode)"


def _register(ctx: ServiceContext, cluster_id: str) -> None:
    profile = ClusterProfile(
        cluster_id=cluster_id,
        display_name=cluster_id,
        transport="local",
        partitions={"gpu": PartitionRule(max_time="2-00:00:00", gpu_types=["a100"], default=True)},
        default_partition="gpu",
        policy=SitePolicy(),
    )
    ctx.clusters.register(profile, UserOverlay(cluster_id=cluster_id, user="alice", remote_root="/r/hpclab"))


@dataclass
class World:
    ctx: ServiceContext
    clock: FakeClock
    items: dict[str, int]

    def row(self, name: str) -> dict[str, Any]:
        r = self.ctx.db.one("SELECT * FROM outbox WHERE outbox_id=?", (self.items[name],))
        assert r is not None
        return dict(r)

    def due(self, name: str) -> bool:
        return self.row(name)["not_before"] <= iso(self.clock.now())

    def poll(self, cluster_id: str, source: str = "squeue") -> dict[str, Any]:
        r = self.ctx.db.one("SELECT * FROM poll_state WHERE cluster_id=? AND source=?", (cluster_id, source))
        assert r is not None
        return dict(r)


@pytest.fixture
def world(make_ctx: MakeCtx) -> World:
    clock = FakeClock()
    ctx = make_ctx("plain", clock=clock)
    _register(ctx, "alpha")
    _register(ctx, "beta")
    later = iso(clock.now() + timedelta(seconds=120))
    items: dict[str, int] = {}
    with ctx.db.tx(source="worker") as tx:
        exp_a = make_experiment(tx, cluster_id="alpha")
        plan_a = make_plan(tx, exp_a)
        run_a = make_run(tx, exp_a, plan_a)
        att_a = make_attempt(tx, run_a, cluster_id="alpha")
        exp_b = make_experiment(tx, cluster_id="beta")
        plan_b = make_plan(tx, exp_b)
        run_b = make_run(tx, exp_b, plan_b)
        att_b = make_attempt(tx, run_b, cluster_id="beta")
        wf_a = new_id("workflow")
        tx.insert(
            "workflows",
            {
                "workflow_id": wf_a,
                "experiment_id": exp_a,
                "plan_id": plan_a,
                "plan_hash": "sha256:x",
                "max_parallel": 1,
                "state": "dispatching",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        op_a = new_id("operation")
        tx.insert(
            "operations",
            {
                "operation_id": op_a,
                "kind": "cancel",
                "state": "running",
                "target_type": "attempt",
                "target_id": att_a,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )

        def add(name: str, kind: str, *, error: str | None, state: str = "pending", **cols: Any) -> None:
            cur = tx.execute(
                "INSERT INTO outbox (kind, operation_id, target_id, payload_json, state, not_before, last_error,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    cols.get("operation_id"),
                    cols.get("target_id"),
                    jdump(cols.get("payload") or {}),
                    state,
                    cols.get("not_before", later),
                    error,
                    tx.now,
                    tx.now,
                ),
            )
            items[name] = int(cur.lastrowid or 0)

        add("dispatch_a", "submit.dispatch", error=AUTH, target_id=att_a, payload={"attempt_id": att_a})
        add("workflow_a", "workflow.dispatch", error="auth_required", target_id=wf_a)
        add("cancel_a_by_op", "cancel.execute", error=AUTH, operation_id=op_a)
        add("dispatch_b", "submit.dispatch", error=AUTH, target_id=att_b, payload={"attempt_id": att_b})
        add("network_a", "submit.stage", error="unavailable: no route to host", target_id=att_a)
        add("leased_a", "observe.finalize", error=AUTH, state="leased", target_id=att_a)
        add("done_a", "artifacts.index", error=AUTH, state="done", target_id=att_a)
        add("notify", "notify.send", error=AUTH, target_id=new_id("notification"))
        tx.execute(
            "INSERT INTO poll_state (cluster_id, source, last_polled_at, consecutive_failures, backoff_until,"
            " last_error) VALUES (?, 'squeue', ?, 3, ?, 'auth_required: ssh master is gone'),"
            " (?, 'squeue', ?, 2, ?, 'unavailable: timeout'), (?, 'sacct', ?, 0, NULL, NULL)",
            ("alpha", tx.now, later, "beta", tx.now, later, "alpha", tx.now),
        )
    return World(ctx=ctx, clock=clock, items=items)


def test_resume_one_cluster_wakes_only_its_auth_parked_items(world: World) -> None:
    data = auth_resume(world.ctx, cluster_id="alpha")
    assert isinstance(data, AuthResumeData)
    assert data.woken == 3 and data.clusters == ["alpha"]
    assert "alpha" in data.note
    # attempt target, workflow -> experiment, and operation -> attempt all resolve to alpha
    assert world.due("dispatch_a") and world.due("workflow_a") and world.due("cancel_a_by_op")
    # other cluster, other errors, other states, no cluster at all: untouched
    for name in ("dispatch_b", "network_a", "notify"):
        assert not world.due(name), name
    assert world.row("leased_a")["state"] == "leased" and world.row("done_a")["state"] == "done"
    assert world.row("dispatch_a")["last_error"] == AUTH  # history kept, only the pause ends
    alpha = world.poll("alpha")
    assert alpha["backoff_until"] is None and alpha["consecutive_failures"] == 0
    beta = world.poll("beta")
    assert beta["backoff_until"] is not None and beta["consecutive_failures"] == 2


def test_resume_all_clusters(world: World) -> None:
    data = auth_resume(world.ctx)
    assert data.woken == 5  # every pending auth-parked item, whatever its cluster
    assert data.clusters == ["alpha", "beta"]
    for name in ("dispatch_a", "workflow_a", "cancel_a_by_op", "dispatch_b", "notify"):
        assert world.due(name), name
    assert not world.due("network_a")
    for cid in ("alpha", "beta"):
        row = world.poll(cid)
        assert row["backoff_until"] is None and row["consecutive_failures"] == 0


def test_resume_writes_an_event(world: World) -> None:
    auth_resume(world.ctx, cluster_id="alpha")
    events = events_list(world.ctx, event_type="cluster.auth_resumed").events
    assert len(events) == 1
    ev = events[0]
    assert ev.entity_id == "alpha" and ev.payload is not None
    assert ev.payload["woken"] == 3
    assert sorted(ev.payload["outbox_ids"]) == sorted(
        world.items[n] for n in ("dispatch_a", "workflow_a", "cancel_a_by_op")
    )
    assert ev.payload["poll_backoff_cleared"] == [{"cluster_id": "alpha", "source": "squeue", "failures": 3}]


def test_woken_items_run_on_the_next_worker_pass(world: World) -> None:
    ran: list[int] = []

    def handler(ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
        ran.append(item.outbox_id)
        return HandlerResult.done()

    runner = OutboxRunner(world.ctx)
    runner.register_all({"submit.dispatch": handler})
    assert runner.drain() == 0  # still paused
    auth_resume(world.ctx, cluster_id="alpha")
    assert runner.drain() == 1
    assert ran == [world.items["dispatch_a"]]


def test_second_resume_has_nothing_left_to_wake(world: World) -> None:
    assert auth_resume(world.ctx, cluster_id="alpha").woken == 3
    again = auth_resume(world.ctx, cluster_id="alpha")
    assert again.woken == 0 and again.clusters == ["alpha"]
    assert "nothing" in again.note


def test_unknown_cluster_is_not_found(world: World) -> None:
    with pytest.raises(NotFound):
        auth_resume(world.ctx, cluster_id="nope")
    assert not world.due("dispatch_a")


def test_registered_as_a_mutation_with_the_cli_contract(world: World) -> None:
    entry = next(e for e in METHODS if e.name == "auth_resume")
    assert entry.target == "hpclab.services.clusters:auth_resume" and entry.mutation is True
    reg = build_registry(strict=True)
    env = reg.dispatch(world.ctx, "auth_resume", {"cluster_id": "alpha"}, request_id=new_id("request"))
    assert env["ok"], env
    assert set(env["data"]) == {"woken", "clusters", "note"}
    assert env["data"]["woken"] == 3 and env["data"]["clusters"] == ["alpha"]
    env = reg.dispatch(world.ctx, "auth_resume", {}, request_id=new_id("request"))
    assert env["ok"] and env["data"]["clusters"] == ["alpha", "beta"], env


# -- the real parking path: a submission that met an expired login ----------------------------------


@pytest.mark.sim
@pytest.mark.timeout(120)
def test_a_submission_parked_on_an_expired_login_runs_right_after_auth_resume(make_ctx: MakeCtx) -> None:
    """End to end on the simulator: the staging step meets ``auth_required`` and parks for
    ``auth_retry_s``; after the login is back, ``auth_resume`` makes it run on the next worker pass
    instead of minutes later, and the job reaches the scheduler."""
    import shutil
    from pathlib import Path

    from hpclab.services.app import create_app
    from hpclab.services.clusters import effective_sim_root, init_sim
    from hpclab.services.experiments import experiment_prepare
    from hpclab.services.submission import run_submit
    from hpclab.sim.cluster import SimCluster

    ctx = make_ctx("plain")
    init_sim(ctx)
    sim = SimCluster(effective_sim_root(ctx.paths))
    project = ctx.paths.home.parent / "tiny-classifier"
    example = Path(__file__).resolve().parents[2] / "examples" / "tiny-classifier"
    shutil.copytree(example, project, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    prep = experiment_prepare(ctx, idempotency_key="prep-auth", spec_path=str(project / "hpclab.yaml"))
    assert prep.status == "ready"
    smoke = next(r for r in prep.runs if r.kind == "smoke")
    assert smoke.run_id and smoke.exec_plan_hash
    runner = OutboxRunner(ctx, lease_s=60.0)
    runner.register_all(create_app(ctx).outbox_handlers())

    sim.set_faults(auth_required=True)
    receipt = run_submit(
        ctx, run_id=smoke.run_id, execution_plan_hash=smoke.exec_plan_hash, idempotency_key="submit-auth"
    )
    runner.drain(max_rounds=20)
    parked = ctx.db.all(
        "SELECT * FROM outbox WHERE state='pending' AND substr(last_error, 1, 13)='auth_required'"
    )
    assert parked, ctx.db.all("SELECT kind, state, last_error FROM outbox")
    op = ctx.db.one("SELECT state, stage FROM operations WHERE operation_id=?", (receipt.operation_id,))
    assert op is not None and (op["state"], op["stage"]) == ("blocked", "auth_required")

    sim.clear_faults()  # the user logged in again
    assert runner.drain(max_rounds=5) == 0  # without auth_resume the step still waits
    assert sim.jobs() == []
    data = auth_resume(ctx, cluster_id="sim")
    assert data.woken == len(parked) and data.clusters == ["sim"]
    runner.drain(max_rounds=20)
    assert len(sim.jobs()) == 1
    op = ctx.db.one("SELECT state FROM operations WHERE operation_id=?", (receipt.operation_id,))
    assert op is not None and op["state"] != "blocked"
