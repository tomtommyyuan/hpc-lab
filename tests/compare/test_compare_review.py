"""Regression tests from the adversarial review of results_compare / evidence_export."""

from __future__ import annotations

from typing import Any

import pytest
from lab_support import Lab, build_scenario

from hpclab.services.compare import ComparisonData, ExclusionSpec, results_compare
from hpclab.services.context import ServiceContext
from hpclab.services.evidence import evidence_export
from hpclab.storage.db import Tx
from hpclab.testing.factories import make_attempt


def cell(data: ComparisonData, label: str, metric: str = "acc") -> Any:
    return next(g for g in data.groups if g.label == label).metrics[metric]


def count(ctx: ServiceContext, sql: str) -> int:
    row = ctx.db.one(sql)
    assert row is not None
    return int(row["n"])


def test_a_run_evaluated_by_another_evaluator_is_missing_not_failed(ctx: ServiceContext, lab: Lab) -> None:
    """The spec's evaluator decides what counts, but a run that succeeded and was evaluated with a
    different evaluator did not fail: a paper footnote must not call its seed 'failed'."""
    exp = lab.experiment(
        comparison={"baseline": {"method": "erm"}}, evaluation={"evaluator_id": "hpclab_metrics"}
    )
    plan = lab.plan(exp)
    for seed in (1, 2):
        lab.valid(lab.run(exp, plan, method="erm", seed=seed), acc=0.8 + seed / 100)
    other = lab.run(exp, plan, method="erm", seed=3)
    lab.valid(other, acc=0.9, evaluator_id="custom_eval")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    erm = cell(data, "erm")
    assert (erm.n, erm.missing_seeds, erm.failed_seeds) == (2, [3], [])
    run = next(r for r in data.runs if r.run_id == other)
    assert run.status == "missing" and run.reason is not None
    assert "hpclab_metrics" in run.reason and "evaluator" in run.reason
    # a genuinely failed run stays failed
    failed = lab.run(exp, plan, method="erm", seed=4)
    lab.failed(failed)
    again = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert cell(again, "erm").failed_seeds == [4]


def test_results_compare_archive_event_and_idempotency_record_commit_together(
    ctx: ServiceContext, lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc = build_scenario(lab)

    def boom(self: Tx, *args: object, **kwargs: object) -> None:
        raise RuntimeError("crash before the idempotency record")

    monkeypatch.setattr(Tx, "idem_store", boom)
    with pytest.raises(RuntimeError):
        results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], idempotency_key="k")
    assert count(ctx, "SELECT COUNT(*) AS n FROM comparisons") == 0
    assert count(ctx, "SELECT COUNT(*) AS n FROM events WHERE event_type='comparison.created'") == 0
    monkeypatch.undo()
    first = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], idempotency_key="k"
    )
    assert count(ctx, "SELECT COUNT(*) AS n FROM comparisons") == 1
    assert count(ctx, "SELECT COUNT(*) AS n FROM idempotency WHERE tool='results_compare'") == 1
    replay = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], idempotency_key="k"
    )
    assert replay.comparison_id == first.comparison_id


def test_evidence_bundle_event_and_idempotency_record_commit_together(
    ctx: ServiceContext, lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc = build_scenario(lab)
    cmp = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    real = Tx.idem_store

    def boom(self: Tx, principal: str, tool: str, *args: Any, **kwargs: Any) -> None:
        if tool == "evidence_export":
            raise RuntimeError("crash before the idempotency record")
        real(self, principal, tool, *args, **kwargs)

    monkeypatch.setattr(Tx, "idem_store", boom)
    with pytest.raises(RuntimeError):
        evidence_export(ctx, purpose="T1", idempotency_key="k", comparison_ref=cmp.comparison_ref)
    assert count(ctx, "SELECT COUNT(*) AS n FROM evidence_bundles") == 0
    assert count(ctx, "SELECT COUNT(*) AS n FROM events WHERE event_type='evidence.created'") == 0
    monkeypatch.undo()
    ev = evidence_export(ctx, purpose="T1", idempotency_key="k", comparison_ref=cmp.comparison_ref)
    again = evidence_export(ctx, purpose="T1", idempotency_key="k", comparison_ref=cmp.comparison_ref)
    assert again.bundle_id == ev.bundle_id
    assert count(ctx, "SELECT COUNT(*) AS n FROM evidence_bundles") == 1
    assert count(ctx, "SELECT COUNT(*) AS n FROM events WHERE event_type='evidence.created'") == 1


def test_exclusion_after_the_run_completed_is_late_even_before_evaluation(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Metrics are readable (artifact_read / run_inspect) as soon as a run COMPLETED with valid
    outputs; evaluation_collect is a separate, later step. Excluding in between is still a
    decision made after the result could be seen."""
    sc = build_scenario(lab)
    target = sc.runs[("mixup", 3)]
    lab.clock.advance(5)
    with ctx.db.tx() as tx:
        make_attempt(
            tx,
            target,
            phase="terminal",
            scheduler_state="COMPLETED",
            result_state="valid",
            terminal_at=tx.now,
        )
    lab.clock.advance(5)
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=target, reason="diverged")],
    )
    (rec,) = data.integrity.exclusions
    assert rec.observed_before_exclusion is True and rec.first_observed_at is not None
    # evaluating afterwards does not turn it into a pre-declared exclusion
    lab.valid(target, acc=0.5)
    later = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=target, reason="diverged")],
    )
    assert later.integrity.exclusions[0].observed_before_exclusion is True


def test_excluding_a_failed_run_is_not_seed_selection(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)  # mixup seed 2 failed with OOM: no value was ever observable
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=sc.runs[("mixup", 2)], reason="OOM on a bad node")],
    )
    assert data.integrity.exclusions[0].observed_before_exclusion is False
