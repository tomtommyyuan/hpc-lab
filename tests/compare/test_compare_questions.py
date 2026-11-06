"""results_compare: unknown metric names, the E / C! cell codes, and the request key.

* A metric that no evaluation of the planned run set reports is a typo, not a result: it is
  refused (with the known names) and never archived.
* ``E`` (COMPLETED, no valid evaluation *yet*) is missing — with its reason — never failed;
  ``C!`` (COMPLETED with bad outputs, or a partial/invalid evaluation) is failed.
* ``request_key`` identifies the question a comparison answers (run set, contract, exclusions,
  evaluation filter), independent of time and of the metric list.
"""

from __future__ import annotations

from typing import Any

import pytest
from lab_support import Lab, build_scenario

from hpclab.domain.errors import InvalidArgument
from hpclab.domain.ids import new_id
from hpclab.domain.models import ComparisonContract
from hpclab.services.compare import (
    ComparisonData,
    ExclusionSpec,
    comparison_request_key,
    results_compare,
)
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jload
from hpclab.testing.factories import make_attempt, make_evaluation


def cell(data: ComparisonData, group: str, metric: str = "acc") -> Any:
    return next(g for g in data.groups if g.label == group).metrics[metric]


def n_comparisons(ctx: ServiceContext) -> int:
    row = ctx.db.one("SELECT COUNT(*) AS n FROM comparisons")
    assert row is not None
    return int(row["n"])


# ---------------------------------------------------------------------------
# Unknown metric names
# ---------------------------------------------------------------------------


def test_a_metric_no_evaluation_reports_is_refused_and_not_archived(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    with pytest.raises(InvalidArgument) as exc:
        results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc", "accuracy"])
    assert exc.value.details["known_metrics"] == ["acc"]
    assert exc.value.details["unknown_metrics"] == ["accuracy"]
    assert "accuracy" in exc.value.message
    assert n_comparisons(ctx) == 0
    # the known metric alone is fine
    assert (
        results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"]).counts.included == 4
    )


def test_a_metric_reported_by_any_evaluation_is_known(ctx: ServiceContext, lab: Lab) -> None:
    """Known = reported by at least one evaluation of the run set (whatever its status): a metric
    only one seed reports is a gap in the other seeds, not a typo."""
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    r1, r2 = lab.run(exp, plan, seed=1), lab.run(exp, plan, seed=2)
    lab.valid(r1, metrics={"acc": 0.8})
    with ctx.db.tx() as tx:
        att = make_attempt(tx, r2, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, att, metrics={"acc": 0.7, "f1": 0.5}, status="partial")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc", "f1"])
    f1 = cell(data, "erm", "f1")
    assert (f1.n, f1.missing_seeds, f1.failed_seeds) == (0, [1], [2])


def test_evaluations_that_report_no_metric_at_all_are_diagnosed(ctx: ServiceContext, lab: Lab) -> None:
    """Evaluations exist but none reports any metric (e.g. a misconfigured evaluator): still refused
    (nothing to count), and the refusal points at the evaluations rather than at a typo."""
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    runs = [lab.run(exp, plan, seed=seed) for seed in (1, 2)]
    with ctx.db.tx() as tx:
        for rid in runs:
            att = make_attempt(tx, rid, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
            make_evaluation(tx, att, status="invalid", metrics_json="[]")
    with pytest.raises(InvalidArgument) as exc:
        results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert exc.value.details["known_metrics"] == []
    assert exc.value.details["evaluation_statuses"] == {"invalid": 2}
    assert "none of the 2 evaluation(s)" in exc.value.message and "reports any metric" in exc.value.message
    assert n_comparisons(ctx) == 0


def test_without_any_evaluation_the_comparison_proceeds_with_a_warning(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    for seed in (1, 2):
        lab.run(exp, plan, seed=seed)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert (data.counts.planned_runs, data.counts.missing) == (2, 2)
    assert any("no evaluation exists yet" in w and "acc" in w for w in data.warnings)


# ---------------------------------------------------------------------------
# E and C!
# ---------------------------------------------------------------------------


def test_e_is_missing_with_its_reason_and_c_bang_is_failed(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    runs = {s: lab.run(exp, plan, seed=s) for s in (1, 2, 3, 4, 5)}
    lab.valid(runs[1], acc=0.9)
    with ctx.db.tx() as tx:
        # E: outputs valid, evaluation pending
        make_attempt(tx, runs[2], phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        # E: outputs still being checked
        make_attempt(tx, runs[3], phase="terminal", scheduler_state="COMPLETED", result_state="unchecked")
        # C!: outputs incomplete
        make_attempt(tx, runs[4], phase="terminal", scheduler_state="COMPLETED", result_state="incomplete")
        # C!: the latest attempt's evaluation is partial
        att = make_attempt(tx, runs[5], phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, att, metrics={"acc": 0.5}, status="partial")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    by_run = {r.run_id: r for r in data.runs}
    assert [(by_run[runs[s]].code, by_run[runs[s]].status) for s in (2, 3, 4, 5)] == [
        ("E", "missing"),
        ("E", "missing"),
        ("C!", "failed"),
        ("C!", "failed"),
    ]
    assert "evaluation" in (by_run[runs[2]].reason or "")
    assert "outputs being checked" in (by_run[runs[3]].reason or "")
    assert by_run[runs[4]].reason == "completed but outputs incomplete"
    assert by_run[runs[5]].reason == "completed but evaluation partial"
    erm = cell(data, "erm")
    assert (erm.missing_seeds, erm.failed_seeds) == ([2, 3], [4, 5])
    assert (data.counts.missing, data.counts.failed, data.counts.by_code) == (
        2,
        2,
        {"C!": 2, "E": 2, "V": 1},
    )
    notes = " ".join(data.coverage.notes)
    assert "2 run(s) completed but have no valid evaluation yet (E)" in notes
    assert "counted as missing, not failed" in notes
    assert "2 run(s) completed with incomplete/invalid outputs or a partial/invalid evaluation (C!)" in notes


def test_a_never_dispatched_cell_of_a_cancelled_workflow_is_failed_x(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    r1, r2 = lab.run(exp, plan, seed=1), lab.run(exp, plan, seed=2)
    lab.valid(r1, acc=0.9)
    wf = new_id("workflow")
    with ctx.db.tx() as tx:
        tx.insert(
            "workflows",
            {
                "workflow_id": wf,
                "experiment_id": exp,
                "plan_id": plan,
                "plan_hash": "sha256:" + "0" * 64,
                "max_parallel": 2,
                "state": "cancelled",
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        for ordinal, (rid, state) in enumerate(((r1, "succeeded"), (r2, "cancelled"))):
            tx.insert(
                "workflow_cells",
                {
                    "workflow_id": wf,
                    "run_id": rid,
                    "ordinal": ordinal,
                    "cell_state": state,
                    "dispatch_key": f"k{ordinal}",
                    "updated_at": tx.now,
                },
            )
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    entry = next(r for r in data.runs if r.run_id == r2)
    assert (entry.code, entry.status) == ("X", "failed")
    assert "never dispatched" in (entry.reason or "")


# ---------------------------------------------------------------------------
# Request key
# ---------------------------------------------------------------------------


def _row(ctx: ServiceContext, cid: str) -> dict[str, Any]:
    row = ctx.db.one("SELECT * FROM comparisons WHERE comparison_id=?", (cid,))
    assert row is not None
    return row


def test_request_key_identifies_the_question_not_the_time_or_metric_list(
    ctx: ServiceContext, lab: Lab
) -> None:
    sc = build_scenario(lab)
    first = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    lab.valid(sc.runs[("mixup", 3)], metrics={"acc": 0.9, "loss": 0.2})
    lab.clock.advance(60)
    # same question, later, with the effective (spec default) contract spelled out and more metrics
    spelled = results_compare(
        ctx,
        planned_run_set_ref=f"hpclab://experiments/{sc.experiment_id}",
        metric_names=["loss", "acc"],
        comparison_contract=ComparisonContract(baseline={"method": "erm"}, treatment=["method"]),
    )
    exploratory = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=sc.runs[("erm", 1)], reason="looked odd")],
    )
    other_contract = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(
            baseline={"method": "erm"}, treatment=["method"], attempt_selection="first_valid", ci_level=0.9
        ),
    )
    filtered = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        evaluation_refs=[sc.evaluations[("erm", 1)]],
    )
    keys = {
        name: comparison_request_key(_row(ctx, d.comparison_id))
        for name, d in {
            "first": first,
            "spelled": spelled,
            "exploratory": exploratory,
            "other_contract": other_contract,
            "filtered": filtered,
        }.items()
    }
    assert keys["first"] == keys["spelled"]
    assert len({keys[k] for k in ("first", "exploratory", "other_contract", "filtered")}) == 4
    assert all(k is not None and k.startswith("sha256:") for k in keys.values())
    request = jload(_row(ctx, first.comparison_id)["request_json"])
    assert request["metric_names"] == ["acc"]  # the stored request itself is unchanged


def test_rerunning_a_request_reuses_its_archive_despite_exploratory_compares_in_between(
    ctx: ServiceContext, lab: Lab
) -> None:
    sc = build_scenario(lab)
    first = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    lab.clock.advance(60)
    results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=sc.runs[("erm", 1)], reason="exploring")],
    )
    lab.clock.advance(60)
    again = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert again.reused and again.comparison_id == first.comparison_id
    assert n_comparisons(ctx) == 2
    # once the ledger changes, the same request is archived anew
    lab.valid(sc.runs[("mixup", 3)], acc=0.9)
    changed = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert not changed.reused and n_comparisons(ctx) == 3


def test_an_included_value_whose_newer_attempt_awaits_evaluation_is_provisional(
    ctx: ServiceContext, lab: Lab
) -> None:
    """latest_valid counts the newest *evaluated* valid attempt: while a newer attempt of the same
    run COMPLETED and awaits evaluation (an E hidden behind the V), the value will change — the
    comparison says so. Under first_valid and pinned evaluation_refs the value cannot change: the
    newer attempt is disclosed, but nothing is provisional."""
    exp = lab.experiment(comparison={}, evaluation={"evaluator_id": "hpclab_metrics"})
    plan = lab.plan(exp)
    r1, r2 = lab.run(exp, plan, seed=1), lab.run(exp, plan, seed=2)
    _, _, ev1 = lab.valid(r1, acc=0.8)
    lab.valid(r2, acc=0.9)
    lab.valid(r2, acc=0.7, evaluator_id="other")  # evaluated, but not by the experiment's evaluator
    with ctx.db.tx() as tx:
        make_attempt(tx, r1, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert (data.counts.included, cell(data, "erm").values_by_seed) == (2, {"1": 0.8, "2": 0.9})
    (warning,) = [w for w in data.warnings if "newer attempt" in w]
    assert "2 included run(s)" in warning and "erm seed 1" in warning and "erm seed 2" in warning
    assert "provisional" in warning and "latest_valid" in warning
    first = results_compare(
        ctx,
        planned_run_set_ref=exp,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(attempt_selection="first_valid"),
    )
    assert [e.provisional for e in first.integrity.newer_attempts] == [False, False]
    assert not any("newer attempt" in w and "provisional" in w for w in first.warnings)
    (steady,) = [w for w in first.warnings if "newer attempt" in w]
    assert "COMPLETED, awaiting evaluation" in steady and "first_valid" in steady
    pinned = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=[ev1])
    assert [e.provisional for e in pinned.integrity.newer_attempts] == [False]
    assert not any("newer attempt" in w and "provisional" in w for w in pinned.warnings)
