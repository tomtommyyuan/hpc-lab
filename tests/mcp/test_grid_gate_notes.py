"""The main-run grid must say when main cells wait on a smoke gate or a stopped workflow.

Found while driving the MCP server like an agent: after a failed smoke gate the grid showed
``P`` in every cell forever (the smoke run is not a grid cell and the workflow state is not in
``runs_list``), so an agent kept "checking again later". ``runs_list(view='grid')`` now adds a
note naming the smoke run that is not valid and the workflow whose dispatch stopped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hpclab.config import get_paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.ids import new_id
from hpclab.services.context import ServiceContext
from hpclab.services.runs import runs_list
from hpclab.testing.factories import make_attempt, make_experiment, make_plan, make_run


def _ctx(tmp_path: Path) -> ServiceContext:
    return ServiceContext.create(get_paths(tmp_path / "home").ensure(), clock=FakeClock())


def _workflow(ctx: ServiceContext, eid: str, pid: str, runs: list[str], *, state: str, gate: str) -> str:
    wid = new_id("workflow")
    with ctx.db.tx() as tx:
        tx.insert(
            "workflows",
            {
                "workflow_id": wid,
                "experiment_id": eid,
                "plan_id": pid,
                "plan_hash": "sha256:x",
                "max_parallel": 2,
                "state": state,
                "gate_run_id": gate,
                "gate_state": "failed" if state == "gate_failed" else "pending",
                "version": 3,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        for i, rid in enumerate(runs):
            tx.insert(
                "workflow_cells",
                {
                    "workflow_id": wid,
                    "run_id": rid,
                    "ordinal": i,
                    "cell_state": "planned",
                    "dispatch_key": f"wf:{wid}:{rid}",
                    "updated_at": tx.now,
                },
            )
    return wid


def test_grid_notes_failed_smoke_gate_and_stopped_workflow(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    try:
        with ctx.db.tx() as tx:
            eid = make_experiment(tx)
            pid = make_plan(tx, eid)
            smoke = make_run(tx, eid, pid, kind="smoke", cell={"method": "a"}, label="smoke.method-a.seed-1")
            make_attempt(
                tx, smoke, phase="terminal", scheduler_state="OUT_OF_MEMORY", result_state="incomplete"
            )
            mains = [make_run(tx, eid, pid, cell={"method": m}, seed=1) for m in ("a", "b")]
        wid = _workflow(ctx, eid, pid, [smoke, *mains], state="gate_failed", gate=smoke)

        scopes: list[dict[str, Any]] = [{"experiment_id": eid}, {"workflow_id": wid}]
        for scope in scopes:
            data = runs_list(ctx, view="grid", **scope)
            assert data.grid is not None
            notes = " | ".join(data.grid.notes)
            assert smoke in notes and "smoke" in notes, notes
            assert wid in notes and "gate_failed" in notes, notes
            # the smoke run is still not a grid cell
            assert data.total == 2
    finally:
        ctx.db.close()


def test_grid_has_no_gate_note_when_the_smoke_run_passed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    try:
        with ctx.db.tx() as tx:
            eid = make_experiment(tx)
            pid = make_plan(tx, eid)
            smoke = make_run(tx, eid, pid, kind="smoke", cell={"method": "a"})
            make_attempt(tx, smoke, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
            make_run(tx, eid, pid, cell={"method": "a"}, seed=1)
        data = runs_list(ctx, experiment_id=eid, view="grid")
        assert data.grid is not None
        assert not any("smoke" in n for n in data.grid.notes)
    finally:
        ctx.db.close()
