"""results_compare: denominators, statistics against hand calculations, contract checks, archive."""

from __future__ import annotations

import math
from typing import Any

import pytest
from lab_support import CODE_B, ERM, MIXUP, Lab, build_scenario

from hpclab.domain.errors import (
    Ambiguous,
    IdempotencyConflict,
    InvalidArgument,
    NotFound,
    PolicyViolation,
    PreconditionFailed,
)
from hpclab.domain.models import ComparisonContract
from hpclab.services import stats
from hpclab.services.compare import (
    ComparisonData,
    ExclusionSpec,
    archive_content_hash,
    comparison_get,
    results_compare,
)
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jload
from hpclab.testing.factories import make_attempt

T_975_DF2 = 4.302652729911275  # Student-t 97.5% quantile, 2 degrees of freedom (tables)
T_975_DF1 = 12.706204736174698


def cell(data: ComparisonData, group: str, metric: str = "acc") -> Any:
    return next(g for g in data.groups if g.label == group).metrics[metric]


def run_status(data: ComparisonData, run_id: str) -> str:
    return next(r.status for r in data.runs if r.run_id == run_id)


# ---------------------------------------------------------------------------
# Denominators
# ---------------------------------------------------------------------------


def test_every_planned_run_is_included_missing_failed_or_excluded(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=sc.runs[("erm", 2)], reason="node gpu-07 flaky")],
    )
    c = data.counts
    assert (c.planned_runs, c.included, c.missing, c.failed, c.excluded) == (6, 3, 1, 1, 1)
    erm, mixup = cell(data, "erm"), cell(data, "mixup")
    assert erm.values_by_seed == {"1": 0.80, "3": 0.84}
    assert [(e.seed, e.reason) for e in erm.excluded] == [(2, "node gpu-07 flaky")]
    assert (mixup.n, mixup.missing_seeds, mixup.failed_seeds) == (1, [3], [2])
    for g in data.groups:
        for m in g.metrics.values():
            assert m.n + len(m.missing_seeds) + len(m.failed_seeds) + len(m.excluded) == m.expected_n == 3
    assert run_status(data, sc.runs[("mixup", 2)]) == "failed"
    assert run_status(data, sc.runs[("mixup", 3)]) == "missing"
    assert next(r for r in data.runs if r.run_id == sc.runs[("mixup", 2)]).code == "F"
    assert data.coverage.complete is False
    assert any("mixup" in m and "missing" in m for m in data.coverage.missing)
    assert data.summary.startswith("2 group(s) × 1 metric(s) over 6 planned runs: 3 included")


def test_running_and_prepared_runs_stay_in_the_denominator(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    lab.running(sc.runs[("mixup", 3)])
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert cell(data, "mixup").missing_seeds == [3]
    entry = next(r for r in data.runs if r.run_id == sc.runs[("mixup", 3)])
    assert (entry.code, entry.status) == ("R", "missing")


def test_metric_absent_or_non_finite_is_missing_or_failed_for_that_metric(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    r1, r2, r3 = (lab.run(exp, plan, method="erm", seed=s) for s in (1, 2, 3))
    lab.valid(r1, metrics={"acc": 0.8, "loss": 0.5})
    lab.valid(r2, metrics={"acc": 0.9})  # no loss reported
    lab.valid(r3, metrics={"acc": 0.7, "loss": None})  # a diverged loss is recorded as null
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc", "loss"])
    assert data.counts.included == 3
    loss = cell(data, "erm", "loss")
    assert (loss.n, loss.missing_seeds, loss.failed_seeds) == (1, [2], [3])
    assert cell(data, "erm", "acc").n == 3


# ---------------------------------------------------------------------------
# Statistics (hand calculations)
# ---------------------------------------------------------------------------


def test_group_statistics_match_hand_calculation(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True)
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    erm = cell(data, "erm")
    # erm: 0.80, 0.82, 0.84 -> mean 0.82, deviations ±0.02 -> s^2 = 0.0008/2 -> s = 0.02
    assert erm.mean == pytest.approx(0.82, abs=1e-12)
    assert erm.std == pytest.approx(0.02, rel=1e-9)
    assert erm.sem == pytest.approx(0.02 / math.sqrt(3), rel=1e-9)
    half = T_975_DF2 * 0.02 / math.sqrt(3)
    assert (erm.ci_low, erm.ci_high) == (
        pytest.approx(0.82 - half, rel=1e-9),
        pytest.approx(0.82 + half, rel=1e-9),
    )
    assert (erm.min, erm.max) == (0.80, 0.84)
    mixup = cell(data, "mixup")
    # mixup: 0.85, 0.86, 0.90 -> mean 0.87, squared deviations 4e-4 + 1e-4 + 9e-4 = 1.4e-3 -> s^2 = 7e-4
    assert mixup.mean == pytest.approx(0.87, abs=1e-12)
    assert mixup.std == pytest.approx(math.sqrt(7e-4), rel=1e-9)
    assert data.groups[0].is_baseline and not data.groups[1].is_baseline


def test_paired_differences_match_hand_calculation(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True)
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    (d,) = data.differences
    # per-seed differences mixup - erm: 0.05, 0.04, 0.06 -> mean 0.05, s = 0.01
    assert (d.kind, d.group_label, d.baseline_label, d.n, d.n_baseline) == ("paired", "mixup", "erm", 3, None)
    assert d.mean_diff == pytest.approx(0.05, rel=1e-9)
    assert d.std_diff == pytest.approx(0.01, rel=1e-6)
    half = T_975_DF2 * 0.01 / math.sqrt(3)
    assert d.ci_low == pytest.approx(0.05 - half, rel=1e-6)
    assert d.ci_high == pytest.approx(0.05 + half, rel=1e-6)
    assert d.paired_seeds == [1, 2, 3]


def test_unpaired_differences_use_welch(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True, comparison={"baseline": {"method": "erm"}, "pair_by_seed": False})
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    (d,) = data.differences
    va, vb = 7e-4 / 3, 4e-4 / 3  # var/n for mixup and erm
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / 2 + vb**2 / 2)
    half = stats.t_critical(df, 0.95) * se
    assert (d.kind, d.n, d.n_baseline) == ("unpaired", 3, 3)
    assert d.mean_diff == pytest.approx(0.05, rel=1e-9)
    assert d.se_diff == pytest.approx(se, rel=1e-9)
    assert (d.ci_low, d.ci_high) == (
        pytest.approx(0.05 - half, rel=1e-6),
        pytest.approx(0.05 + half, rel=1e-6),
    )


def test_single_seed_has_no_spread_and_no_interval(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    mixup = cell(data, "mixup")
    assert (mixup.n, mixup.mean) == (1, 0.85)
    assert mixup.std is None and mixup.sem is None and mixup.ci_low is None and mixup.ci_high is None
    (d,) = data.differences
    assert (d.n, d.ci_low, d.ci_high) == (1, None, None)
    assert d.mean_diff == pytest.approx(MIXUP[1] - ERM[1])
    assert d.unpaired_baseline == [2, 3]


def test_two_seed_interval_uses_t_with_one_degree_of_freedom(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[{"run_id": sc.runs[("erm", 2)], "reason": "x"}],  # type: ignore[list-item]
    )
    erm = cell(data, "erm")  # 0.80, 0.84
    std = math.sqrt(((0.80 - 0.82) ** 2 + (0.84 - 0.82) ** 2) / 1)
    assert erm.std == pytest.approx(std, rel=1e-9)
    assert erm.ci_high == pytest.approx(0.82 + T_975_DF1 * std / math.sqrt(2), rel=1e-9)


def test_multi_axis_baseline_pairs_with_the_same_other_axes(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={"baseline": {"method": "erm"}})
    plan = lab.plan(exp)
    values = {("erm", 0.1): 0.80, ("erm", 0.01): 0.70, ("mixup", 0.1): 0.85, ("mixup", 0.01): 0.78}
    for (method, lr), v in values.items():
        rid = lab.run(exp, plan, method=method, seed=1, extra_cell={"lr": lr})
        lab.valid(rid, acc=v)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    labels = [g.label for g in data.groups]
    assert labels == [
        "lr=0.1, method=erm",
        "lr=0.01, method=erm",
        "lr=0.1, method=mixup",
        "lr=0.01, method=mixup",
    ]
    diffs = {(d.group_label, d.baseline_label): d.mean_diff for d in data.differences}
    assert diffs == {
        ("lr=0.1, method=mixup", "lr=0.1, method=erm"): pytest.approx(0.05),
        ("lr=0.01, method=mixup", "lr=0.01, method=erm"): pytest.approx(0.08),
    }


def test_no_baseline_means_no_differences_with_a_warning(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True, comparison={})
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert data.differences == []
    assert any("no baseline" in w for w in data.warnings)
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(baseline={"method": "nope"}),
    )
    assert data.differences == [] and any("matches no group" in w for w in data.warnings)


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------


def test_expected_seeds_are_checked_against_the_plan(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    ok = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], expected_seeds=[1, 2, 3]
    )
    assert ok.expected_seeds == [1, 2, 3]
    with pytest.raises(PreconditionFailed) as exc:
        results_compare(
            ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], expected_seeds=[1, 2, 3, 4]
        )
    assert exc.value.details["groups"][0]["expected_but_not_planned"] == ["4"]
    with pytest.raises(PreconditionFailed) as exc:
        results_compare(
            ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], expected_seeds=[1, 2]
        )
    assert exc.value.details["groups"][0]["planned_but_not_expected"] == ["3"]


def test_exclusions_need_reasons_and_planned_runs(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    with pytest.raises(ValueError, match="non-empty reason"):
        ExclusionSpec(run_id=sc.runs[("erm", 1)], reason="   ")
    for bad in ({"run_id": sc.runs[("erm", 1)], "reason": ""}, {"run_id": sc.runs[("erm", 1)]}):
        with pytest.raises(InvalidArgument):
            results_compare(
                ctx,
                planned_run_set_ref=sc.experiment_id,
                metric_names=["acc"],
                exclusions=[bad],  # type: ignore[list-item]
            )
    with pytest.raises(InvalidArgument, match="not in the planned run set"):
        results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            exclusions=[ExclusionSpec(run_id="run_01J0000000000000000000000Z", reason="x")],
        )
    twice = [ExclusionSpec(run_id=sc.runs[("erm", 1)], reason="a")] * 2
    with pytest.raises(InvalidArgument, match="more than once"):
        results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], exclusions=twice)


def test_exclusion_after_observation_is_flagged_by_event_order(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    late = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=sc.runs[("erm", 2)], reason="looks off")],
    )
    (rec,) = late.integrity.exclusions
    assert rec.observed_before_exclusion is True and rec.code == "V"
    events = ctx.db.all("SELECT * FROM events WHERE event_type='run.excluded' AND entity_id=?", (rec.run_id,))
    assert len(events) == 1 and jload(events[0]["payload_json"])["comparison_id"] == late.comparison_id

    # declared while the run was still running, evaluated later: not a late exclusion
    target = sc.runs[("mixup", 3)]
    lab.running(target)
    lab.clock.advance(10)
    early = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=target, reason="preempted twice, config differs")],
    )
    assert early.integrity.exclusions[0].observed_before_exclusion is False
    lab.clock.advance(10)
    lab.valid(target, acc=0.95)
    later = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=target, reason="preempted twice, config differs")],
    )
    rec = later.integrity.exclusions[0]
    assert rec.observed_before_exclusion is False
    assert rec.first_declared_at is not None and rec.first_observed_at is not None
    assert rec.first_declared_at < rec.first_observed_at
    assert cell(later, "mixup").excluded[0].reason == "preempted twice, config differs"


def test_mixed_split_dataset_and_evaluator_are_incomparable(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={"baseline": {"method": "erm"}})
    plan = lab.plan(exp)
    runs = {(m, s): lab.run(exp, plan, method=m, seed=s) for m in ("erm", "mixup") for s in (1, 2)}
    lab.valid(runs[("erm", 1)], acc=0.8)
    lab.valid(runs[("erm", 2)], acc=0.81)
    lab.valid(runs[("mixup", 1)], acc=0.85, split="test")
    lab.valid(runs[("mixup", 2)], acc=0.86, dataset="cifar100", evaluator_version="2")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    by_inv = {i.invariant: i for i in data.incomparable}
    assert set(by_inv) == {"split", "dataset", "evaluator"}
    assert set(by_inv["split"].details["values"]) == {"val", "test"}
    assert by_inv["split"].metric == "acc"
    assert "mixup/seed 1" in by_inv["split"].details["values"]["test"]
    assert set(by_inv["evaluator"].details["values"]) == {"hpclab_metrics@1", "hpclab_metrics@2"}
    assert "1 incomparability" not in data.summary and "3 incomparability issue(s)" in data.summary


def test_report_split_and_declared_param_invariants(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    a = lab.run(exp, plan, method="erm", seed=1)
    b = lab.run(exp, plan, method="mixup", seed=1, params={"lr": 0.2})
    lab.valid(a, acc=0.8)
    lab.valid(b, acc=0.9)
    contract = ComparisonContract(
        invariants=["split", "lr", "not_a_param"],
        selection={"rule": "final", "report_split": "test"},
    )
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=contract)
    reasons = [i.reason for i in data.incomparable]
    assert any("report_split" in r for r in reasons)
    assert any(r == "parameter lr differs between included results" for r in reasons)
    assert any("'not_a_param' is not a recorded run parameter" in w for w in data.warnings)


def test_selection_leak_is_refused(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True)
    leak = {"rule": "best_by", "select_metric": "acc", "select_split": "val", "report_split": "val"}
    with pytest.raises(PolicyViolation) as exc:
        results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            comparison_contract={"selection": leak},  # type: ignore[arg-type]
        )
    assert exc.value.details["reason"] == "selection_leak"
    with pytest.raises(PolicyViolation):
        results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            comparison_contract={"selection": {"rule": "best_by", "select_split": "val"}},  # type: ignore[arg-type]
        )
    for ok in (
        {"rule": "best_by", "select_metric": "acc", "select_split": "val", "report_split": "test"},
        {"rule": "final", "select_split": "val", "report_split": "val"},
    ):
        data = results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            comparison_contract={"selection": ok, "invariants": ["evaluator"]},  # type: ignore[arg-type]
        )
        assert data.integrity.selection.rule == ok["rule"]


def test_evaluations_that_selected_on_the_reported_split_are_flagged(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    rid = lab.run(exp, plan, method="erm", seed=1)
    lab.valid(rid, acc=0.9, issues=[{"severity": "warning", "code": "selection_on_split", "message": "best"}])
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert data.integrity.selection.evaluation_selection_on_split == [rid]
    assert data.incomparable[0].reason.startswith("selection_leak")


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_integrity_reports_post_hoc_heterogeneity_and_multiple_attempts(
    ctx: ServiceContext, lab: Lab
) -> None:
    sc = build_scenario(lab, complete=True)
    extra = lab.run(
        sc.experiment_id, sc.plan_id, method="mixup", seed=4, added_post_hoc=True, code_digest=CODE_B
    )
    lab.valid(extra, acc=0.91, gpu="NVIDIA H100 80GB HBM3", torch="2.7.1")
    first, _, _ = lab.valid(sc.runs[("erm", 1)], acc=0.79)  # a second valid attempt of erm seed 1
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    integ = data.integrity
    assert [(p.run_id, p.seed, p.status) for p in integ.post_hoc] == [(extra, 4, "included")]
    het = {h.dimension: h for h in integ.heterogeneity}
    assert het["gpu"].mixed and het["gpu"].values == {"NVIDIA A100-SXM4-80GB": 6, "NVIDIA H100 80GB HBM3": 1}
    assert het["gpu"].within_groups == ["mixup"]
    assert het["torch"].mixed and het["code_digest"].within_groups == ["mixup"]
    (multi,) = integ.multiple_valid_attempts
    assert (multi.run_id, multi.valid_attempts, multi.rule) == (sc.runs[("erm", 1)], 2, "latest_valid")
    assert multi.selected_attempt_id == first
    assert cell(data, "erm").values_by_seed["1"] == 0.79  # latest valid attempt is used
    first_valid = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(baseline={"method": "erm"}, attempt_selection="first_valid"),
    )
    assert cell(first_valid, "erm").values_by_seed["1"] == 0.80
    assert cell(first_valid, "mixup").expected_n == 4  # the post-hoc seed joined the plan


def test_unknown_environment_is_counted_not_assumed(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    a, b = lab.run(exp, plan, seed=1), lab.run(exp, plan, seed=2, code_digest=None)
    lab.valid(a, acc=0.8)
    lab.valid(b, acc=0.8, gpu=None, torch=None)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    het = {h.dimension: h for h in data.integrity.heterogeneity}
    assert (het["gpu"].unknown, het["gpu"].mixed) == (1, False)
    assert het["code_digest"].unknown == 1


def test_below_min_n_is_warned(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(baseline={"method": "erm"}, min_n=2),
    )
    assert [(b.group_label, b.n) for b in data.integrity.below_min_n] == [("mixup", 1)]
    assert any("below the contract's min_n=2" in w for w in data.warnings)


def test_duplicate_runs_for_one_seed_are_ambiguous_until_one_is_excluded(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp = lab.experiment(comparison={})
    plan1, plan2 = lab.plan(exp), lab.plan(exp)
    a = lab.run(exp, plan1, seed=1)
    b = lab.run(exp, plan2, seed=1)
    lab.valid(a, acc=0.8)
    lab.valid(b, acc=0.9)
    with pytest.raises(Ambiguous) as exc:
        results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert set(exc.value.details["run_ids"]) == {a, b}
    data = results_compare(
        ctx,
        planned_run_set_ref=exp,
        metric_names=["acc"],
        exclusions=[ExclusionSpec(run_id=a, reason="rerun")],
    )
    assert cell(data, "erm").values_by_seed == {"1": 0.9}
    assert data.counts.excluded == 1 and data.counts.planned_runs == 2


def test_seedless_runs_use_the_none_seed(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={})
    plan = lab.plan(exp)
    rid = lab.run(exp, plan, method="erm", seed=None)
    lab.valid(rid, acc=0.7)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert cell(data, "erm").values_by_seed == {"none": 0.7}
    assert data.expected_seeds == ["none"]


def test_metric_units_and_directions_come_from_the_contract_or_the_evaluations(
    ctx: ServiceContext, lab: Lab
) -> None:
    sc = build_scenario(
        lab, complete=True, comparison={"metrics": [{"name": "acc", "unit": "%", "higher_is_better": True}]}
    )
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert data.metric_info[0].model_dump() == {
        "name": "acc",
        "unit": "%",
        "higher_is_better": True,
        "source": "contract",
    }
    data = results_compare(
        ctx,
        planned_run_set_ref=sc.experiment_id,
        metric_names=["acc"],
        comparison_contract=ComparisonContract(),
    )
    assert (data.metric_info[0].unit, data.metric_info[0].source) == ("fraction", "evaluations")


# ---------------------------------------------------------------------------
# Evaluation refs
# ---------------------------------------------------------------------------


def test_evaluation_refs_restrict_without_dropping_runs(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab, complete=True)
    keep = [sc.evaluations[("erm", s)] for s in (1, 2, 3)] + [
        f"hpclab://evaluations/{sc.evaluations[('mixup', 1)]}"
    ]
    data = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], evaluation_refs=keep
    )
    mixup = cell(data, "mixup")
    assert (mixup.n, mixup.missing_seeds) == (1, [2, 3])
    entry = next(r for r in data.runs if r.run_id == sc.runs[("mixup", 2)])
    assert entry.status == "missing" and "not among evaluation_refs" in (entry.reason or "")
    with pytest.raises(NotFound):
        results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            evaluation_refs=["ev_01J0000000000000000000000Z"],
        )
    other = build_scenario(lab, complete=True)
    with pytest.raises(InvalidArgument, match="outside the planned run set"):
        results_compare(
            ctx,
            planned_run_set_ref=sc.experiment_id,
            metric_names=["acc"],
            evaluation_refs=[other.evaluations[("erm", 1)]],
        )
    with pytest.raises(InvalidArgument):
        results_compare(
            ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], evaluation_refs=["run_x"]
        )


def test_metric_names_are_validated(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    for bad in ([], ["acc", "acc"], [""], [f"m{i}" for i in range(21)]):
        with pytest.raises(InvalidArgument):
            results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=bad)


def test_invalid_contract_in_spec_is_reported(ctx: ServiceContext, lab: Lab) -> None:
    exp = lab.experiment(comparison={"pair_by_seed": "maybe", "bogus": 1})
    plan = lab.plan(exp)
    lab.run(exp, plan)
    with pytest.raises(InvalidArgument, match="comparison contract is invalid"):
        results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def test_archive_is_immutable_content_addressed_and_reused_when_unchanged(
    ctx: ServiceContext, lab: Lab
) -> None:
    sc = build_scenario(lab)
    first = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    row = ctx.db.one("SELECT * FROM comparisons WHERE comparison_id=?", (first.comparison_id,))
    assert row is not None and row["run_set_ref"] == f"hpclab://experiments/{sc.experiment_id}"
    assert archive_content_hash(jload(row["result_json"])) == row["content_hash"] == first.content_hash
    assert first.comparison_ref == f"hpclab://comparisons/{first.comparison_id}"
    events = ctx.db.all(
        "SELECT * FROM events WHERE entity_id=? AND event_type='comparison.created'", (first.comparison_id,)
    )
    assert len(events) == 1

    lab.clock.advance(60)
    again = results_compare(
        ctx, planned_run_set_ref=f"hpclab://experiments/{sc.experiment_id}", metric_names=["acc"]
    )
    assert again.reused and again.comparison_id == first.comparison_id
    assert ctx.db.one("SELECT COUNT(*) AS n FROM comparisons")["n"] == 1  # type: ignore[index]

    lab.valid(sc.runs[("mixup", 3)], acc=0.9)
    changed = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert not changed.reused and changed.comparison_id != first.comparison_id
    assert cell(changed, "mixup").n == 2
    # the old archive is untouched
    old = comparison_get(ctx, comparison_id=first.comparison_id)
    assert cell(old, "mixup").n == 1 and old.content_hash == first.content_hash
    with pytest.raises(NotFound):
        comparison_get(ctx, comparison_id="cmp_01J0000000000000000000000Z")
    with pytest.raises(InvalidArgument):
        comparison_get(ctx, comparison_id=sc.experiment_id)


def test_idempotency_key_pins_the_response(ctx: ServiceContext, lab: Lab) -> None:
    sc = build_scenario(lab)
    first = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], idempotency_key="k"
    )
    lab.valid(sc.runs[("mixup", 3)], acc=0.9)  # state changes, but the key replays the stored response
    replay = results_compare(
        ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"], idempotency_key="k"
    )
    assert replay.comparison_id == first.comparison_id and cell(replay, "mixup").n == 1
    with pytest.raises(IdempotencyConflict):
        results_compare(
            ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc", "loss"], idempotency_key="k"
        )


def test_runs_listing_is_capped_but_the_archive_keeps_everything(
    ctx: ServiceContext, lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hpclab.services.compare as compare

    monkeypatch.setattr(compare, "MAX_RUNS_RETURNED", 4)
    sc = build_scenario(lab)
    data = results_compare(ctx, planned_run_set_ref=sc.experiment_id, metric_names=["acc"])
    assert (len(data.runs), data.runs_truncated) == (4, 2)
    full, chains = compare.load_archive(ctx.db, data.comparison_id)
    assert len(full.runs) == 6 and set(chains) == {r for k, r in sc.runs.items() if k != ("mixup", 3)}


def test_completed_but_unevaluated_is_missing_not_failed(ctx: ServiceContext, lab: Lab) -> None:
    """C! splits by output validity: valid outputs without an evaluation are merely not
    extracted yet (missing), incomplete/invalid outputs are a real failure."""
    exp = lab.experiment(comparison={"baseline": {"method": "erm"}, "treatment": ["method"]})
    plan = lab.plan(exp)
    r_ok = lab.run(exp, plan, method="erm", seed=1)
    r_unevaluated = lab.run(exp, plan, method="erm", seed=2)
    r_bad_outputs = lab.run(exp, plan, method="erm", seed=3)
    lab.valid(r_ok, acc=0.9)
    with ctx.db.tx() as tx:
        make_attempt(tx, r_unevaluated, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_attempt(
            tx, r_bad_outputs, phase="terminal", scheduler_state="COMPLETED", result_state="incomplete"
        )
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    by_run = {r.run_id: r for r in data.runs}
    assert by_run[r_unevaluated].status == "missing"
    assert "evaluation_collect" in (by_run[r_unevaluated].reason or "")
    assert by_run[r_bad_outputs].status == "failed"
    assert (data.counts.included, data.counts.missing, data.counts.failed) == (1, 1, 1)
