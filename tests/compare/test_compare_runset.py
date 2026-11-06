"""resolve_run_set: the planned denominator for experiment / plan / workflow refs."""

from __future__ import annotations

import pytest
from lab_support import Lab

from hpclab.domain.errors import InvalidArgument, NotFound, PreconditionFailed
from hpclab.services.compare import resolve_run_set, strip_ref
from hpclab.services.context import ServiceContext


def _workflow(ctx: ServiceContext, experiment_id: str, plan_id: str, run_ids: list[str]) -> str:
    wf = "wf_01J00000000000000000000001"
    with ctx.db.tx() as tx:
        tx.insert(
            "workflows",
            {
                "workflow_id": wf,
                "experiment_id": experiment_id,
                "plan_id": plan_id,
                "plan_hash": "sha256:x",
                "max_parallel": 2,
                "state": "dispatching",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        for i, rid in enumerate(run_ids):
            tx.insert(
                "workflow_cells",
                {
                    "workflow_id": wf,
                    "run_id": rid,
                    "ordinal": i,
                    "cell_state": "planned",
                    "dispatch_key": f"wf:{wf}:{rid}",
                    "updated_at": tx.now,
                },
            )
    return wf


def test_experiment_run_set_has_main_runs_of_every_non_derived_plan(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment()
    p1 = lab.plan(exp)
    a = lab.run(exp, p1, method="erm", seed=1)
    lab.run(exp, p1, method="erm", seed=None, kind="smoke")
    p2 = lab.plan(exp)  # an extension plan (e.g. add_seeds)
    b = lab.run(exp, p2, method="erm", seed=2, added_post_hoc=True)
    derived_plan = lab.plan(exp, kind="derived")
    lab.run(exp, derived_plan, method="erm", seed=1)
    rs = resolve_run_set(ctx.db, exp)
    assert (rs.kind, rs.id, rs.experiment_id) == ("experiment", exp, exp)
    assert rs.run_ids == [a, b]
    assert rs.cells[1].added_post_hoc and rs.cells[0].treatment_key == {"method": "erm"}
    assert rs.cells[0].cell == {"method": "erm", "seed": 1}
    assert rs.ref == f"hpclab://experiments/{exp}"
    assert any("derived" in n for n in rs.notes)
    assert resolve_run_set(ctx.db, f"hpclab://experiments/{exp}").run_ids == [a, b]
    assert resolve_run_set(ctx.db, derived_plan).kind == "plan"


def test_plan_and_workflow_run_sets(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment()
    plan = lab.plan(exp)
    runs = [lab.run(exp, plan, method=m, seed=1) for m in ("erm", "mixup", "cutmix")]
    rs = resolve_run_set(ctx.db, f"hpclab://plans/{plan}")
    assert rs.kind == "plan" and rs.run_ids == runs
    wf = _workflow(ctx, exp, plan, [runs[2], runs[0]])
    ws = resolve_run_set(ctx.db, wf)
    assert (ws.kind, ws.run_ids, ws.experiment_id) == ("workflow", [runs[2], runs[0]], exp)
    assert ws.ref == f"hpclab://workflows/{wf}"
    # a recovery plan has no runs of its own: its workflow's cells are the set
    recovery = lab.plan(exp, kind="recovery")
    with ctx.db.tx() as tx:
        tx.execute("UPDATE workflows SET plan_id=? WHERE workflow_id=?", (recovery, wf))
    assert resolve_run_set(ctx.db, recovery).run_ids == [runs[2], runs[0]]


def test_bad_refs(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment()
    empty_plan = lab.plan(exp)
    with pytest.raises(PreconditionFailed):
        resolve_run_set(ctx.db, empty_plan)
    with pytest.raises(PreconditionFailed):
        resolve_run_set(ctx.db, exp)
    for missing in (
        "exp_01J0000000000000000000000Z",
        "pln_01J0000000000000000000000Z",
        "wf_01J0000000000000000000000Z",
    ):
        with pytest.raises(NotFound):
            resolve_run_set(ctx.db, missing)
    for bad in ("run_01J0000000000000000000000Z", "hello", ""):
        with pytest.raises(InvalidArgument):
            resolve_run_set(ctx.db, bad)
    with pytest.raises(InvalidArgument):
        strip_ref("hpclab://experiments/")
    assert strip_ref("hpclab://evidence/evb_1/table.tex") == "evb_1"
