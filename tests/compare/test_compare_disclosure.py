"""Integrity disclosures of results_compare that need the ledger's history, not just its state.

* An exclusion counts as declared before the result was observed only while it has stayed in
  effect since then: a comparison (of any run set containing the run) that included the run in
  between revokes it, and a blanket exclusion of every planned run declares nothing about one run.
* A run the counted evaluator judged partial/invalid is *failed*, even when another evaluator's
  evaluation of the same attempt is valid.
* An included value that comes from an older attempt says what the run's newer attempt did
  (still running, failed, diverged), so the table never hides a re-run behind an old number.
"""

from __future__ import annotations

from typing import Any

from lab_support import Lab

from hpclab.domain.models import ComparisonContract
from hpclab.services.compare import ComparisonData, ExclusionSpec, results_compare
from hpclab.services.context import ServiceContext
from hpclab.services.evidence import evidence_export, evidence_get
from hpclab.services.verify import evidence_verify
from hpclab.testing.factories import make_attempt, make_evaluation

CONTRACT = {"baseline": {"method": "erm"}, "treatment": ["method"]}


def cell(data: ComparisonData, label: str, metric: str = "acc") -> Any:
    return next(g for g in data.groups if g.label == label).metrics[metric]


def run_of(data: ComparisonData, run_id: str) -> Any:
    return next(r for r in data.runs if r.run_id == run_id)


def md(ctx: ServiceContext, bundle_id: str) -> str:
    content = evidence_get(ctx, bundle_id=bundle_id, file="summary.md").content
    assert content is not None
    return content


def grid(lab: Lab, **kwargs: Any) -> tuple[str, str, dict[tuple[str, int], str]]:
    exp = lab.experiment(comparison=CONTRACT, **kwargs)
    plan = lab.plan(exp)
    runs = {(m, s): lab.run(exp, plan, method=m, seed=s) for m in ("erm", "mixup") for s in (1, 2, 3)}
    return exp, plan, runs


def results(lab: Lab, runs: dict[tuple[str, int], str]) -> None:
    for s, acc in {1: 0.80, 2: 0.82, 3: 0.84}.items():
        lab.valid(runs[("erm", s)], acc=acc)
    for s, acc in {1: 0.85, 2: 0.86, 3: 0.70}.items():
        lab.valid(runs[("mixup", s)], acc=acc)


def excl(run_id: str, reason: str) -> list[ExclusionSpec]:
    return [ExclusionSpec(run_id=run_id, reason=reason)]


# ---------------------------------------------------------------------------
# Exclusions: declared before the observation only while continuously in effect
# ---------------------------------------------------------------------------


def test_re_excluding_after_a_comparison_showed_the_value_is_late(ctx: ServiceContext, lab: Lab) -> None:
    """Declare early, include, look, exclude again: the exclusion in effect was decided after
    the value was seen, whatever an earlier (revoked) exclusion said."""
    exp, _plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    early = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert early.integrity.exclusions[0].observed_before_exclusion is False
    results(lab, runs)
    looked = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert cell(looked, "mixup").values_by_seed["3"] == 0.70
    lab.clock.advance(5)
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "looked unstable")
    )
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is True
    assert rec.first_declared_at == final.created_at  # this comparison's decision, not the revoked one
    assert rec.first_observed_at is not None and rec.first_observed_at < rec.first_declared_at
    assert rec.note is not None and looked.comparison_id in rec.note
    ev = evidence_export(ctx, purpose="Table 1", idempotency_key="k", comparison_ref=final.comparison_id)
    assert "late_exclusions" in ev.integrity_flags
    assert "Exclusions decided after the result was observed:** none" not in md(ctx, ev.bundle_id)
    report = evidence_verify(ctx, bundle_ref=ev.bundle_id)
    assert any(f.code == "SEED_SELECTION" and target in f.refs for f in report.findings)
    # re-running the same request is stable (and reused): the decision time does not move
    again = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "looked unstable")
    )
    assert again.reused and again.comparison_id == final.comparison_id


def test_an_exclusion_kept_in_effect_since_before_the_result_stays_pre_declared(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Honest pre-registration: excluded before the result, never re-included, excluded in the
    reported comparison. Excluding *other* runs in between does not revoke it."""
    exp, _plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    lab.running(target)
    first = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "config differs")
    )
    results(lab, runs)
    both = [*excl(target, "config differs"), *excl(runs[("erm", 1)], "bad node")]
    mid = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=both)
    by_run = {e.run_id: e for e in mid.integrity.exclusions}
    assert by_run[runs[("erm", 1)]].observed_before_exclusion is True
    assert by_run[target].observed_before_exclusion is False
    lab.clock.advance(5)
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "config differs")
    )
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is False and rec.note is None
    assert rec.first_declared_at == first.created_at


def test_an_exclusion_revoked_and_redeclared_before_the_result_is_still_pre_declared(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Only the exclusion in effect when the result became observable matters: re-including a run
    while it had no result, then excluding it again before the result, is still pre-declared —
    declared at the re-exclusion."""
    exp, _plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    lab.running(target)
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "a"))
    lab.clock.advance(1)
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])  # re-included, still running
    lab.clock.advance(1)
    redeclared = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "b")
    )
    results(lab, runs)
    final = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "b"))
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is False
    assert rec.first_declared_at == redeclared.created_at


def test_excluded_while_running_then_shown_then_excluded_again_is_late(ctx: ServiceContext, lab: Lab) -> None:
    exp, _plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    lab.running(target)
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "preempted"))
    results(lab, runs)
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])  # the value is shown
    lab.clock.advance(5)
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "looked unstable")
    )
    assert final.integrity.exclusions[0].observed_before_exclusion is True


def test_a_comparison_of_another_run_set_that_showed_the_run_revokes_its_exclusion(
    ctx: ServiceContext, lab: Lab
) -> None:
    """The value can be seen through any run set that contains the run (here: its plan)."""
    exp, plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    results(lab, runs)
    shown = results_compare(ctx, planned_run_set_ref=plan, metric_names=["acc"])
    assert cell(shown, "mixup").values_by_seed["3"] == 0.70
    lab.clock.advance(5)
    final = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert final.integrity.exclusions[0].observed_before_exclusion is True
    # a comparison of an unrelated experiment never revokes anything
    other_exp, _, other_runs = grid(lab)
    results_compare(ctx, planned_run_set_ref=other_exp, metric_names=["acc"], exclusions=[])
    kept = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert kept.integrity.exclusions[0].observed_before_exclusion is True
    assert other_runs


def test_a_blanket_exclusion_of_every_planned_run_declares_nothing(ctx: ServiceContext, lab: Lab) -> None:
    """Excluding every run before any result costs nothing and would pre-declare any later pick;
    it is not a decision about one run, so a later narrowing to the bad seed is late."""
    exp, _plan, runs = grid(lab)
    target = runs[("mixup", 3)]
    everything = [ExclusionSpec(run_id=r, reason="placeholder") for r in runs.values()]
    blanket = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=everything)
    assert blanket.counts.excluded == 6
    results(lab, runs)
    lab.clock.advance(5)
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "placeholder")
    )
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is True
    assert rec.note is not None and "every planned run" in rec.note
    ev = evidence_export(ctx, purpose="T", idempotency_key="k", comparison_ref=final.comparison_id)
    assert "late_exclusions" in ev.integrity_flags
    report = evidence_verify(ctx, bundle_ref=ev.bundle_id)
    assert any(f.code == "SEED_SELECTION" for f in report.findings)


# ---------------------------------------------------------------------------
# A result the counted evaluator judged bad is failed
# ---------------------------------------------------------------------------


def _two_by_two(lab: Lab) -> tuple[str, dict[tuple[str, int], str]]:
    exp = lab.experiment(comparison=CONTRACT, evaluation={"evaluator_id": "hpclab_metrics"})
    plan = lab.plan(exp)
    runs = {(m, s): lab.run(exp, plan, method=m, seed=s) for m in ("erm", "mixup") for s in (1, 2)}
    for s in (1, 2):
        lab.valid(runs[("erm", s)], acc=0.8 + s / 100)
    lab.valid(runs[("mixup", 1)], acc=0.85)
    return exp, runs


def test_a_partial_evaluation_by_the_counted_evaluator_is_failed_despite_another_valid_one(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, _ = lab.valid(target, metrics={"acc": 0.99}, evaluator_id="json_path")
    lab.clock.advance(1)
    with ctx.db.tx() as tx:  # the experiment's evaluator finds the run diverged (acc NaN)
        make_evaluation(tx, attempt, metrics={"acc": None}, status="partial")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("C!", "failed")
    assert run.reason is not None and run.reason.startswith("completed but evaluation partial")
    assert "hpclab_metrics" in run.reason and "collect one" not in run.reason
    mixup = cell(data, "mixup")
    assert (mixup.missing_seeds, mixup.failed_seeds) == ([], [2])
    assert (data.counts.failed, data.counts.missing) == (1, 0)
    assert any("(C!)" in n and "failed" in n for n in data.coverage.notes)


def test_a_valid_result_the_counted_evaluator_never_saw_stays_missing(ctx: ServiceContext, lab: Lab) -> None:
    """The latest attempt failed, but an older attempt COMPLETED with a valid result that only
    another evaluator evaluated: evaluating it with the counted evaluator can still make the run
    count, so it is missing (uncounted), not failed. Once the counted evaluator judged that
    attempt partial, it is failed."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    older, _, _ = lab.valid(target, acc=0.9, evaluator_id="json_path")
    lab.failed(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("F", "missing")
    assert run.reason is not None and run.reason.startswith("a valid evaluation exists")
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        make_evaluation(tx, older, metrics={"acc": None}, status="invalid")
    judged = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(judged, target)
    assert (run.code, run.status) == ("F", "failed")
    assert run.reason is not None and "invalid" in run.reason and older in run.reason
    assert cell(judged, "mixup").failed_seeds == [2]


# ---------------------------------------------------------------------------
# An included value from an older attempt discloses the newer attempt's outcome
# ---------------------------------------------------------------------------


def _older_value(ctx: ServiceContext, lab: Lab) -> tuple[str, str, str]:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, _ = lab.valid(target, acc=0.90)
    return exp, target, attempt


def test_an_included_value_whose_newer_attempt_is_running_is_provisional(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, target, counted = _older_value(ctx, lab)
    newer = lab.running(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.status, run.attempts, run.evaluation_attempt_id) == ("included", 2, counted)
    assert run.reason is not None and newer in run.reason  # the coverage reason is kept
    (entry,) = data.integrity.newer_attempts
    assert (entry.run_id, entry.value_attempt_id, entry.newer_attempt_id) == (target, counted, newer)
    assert (entry.outcome, entry.provisional) == ("active", True)
    assert entry.disclosure == "value from attempt #1; newer attempt #2 is in progress (RUNNING)"
    (warning,) = [w for w in data.warnings if "still in progress" in w]
    assert "mixup seed 2" in warning and "provisional" in warning and "latest_valid" in warning
    ev = evidence_export(ctx, purpose="T", idempotency_key="k", comparison_ref=data.comparison_id)
    assert warning in ev.warnings
    limitations = md(ctx, ev.bundle_id).split("## Limitations", 1)[1]
    assert "still in progress" in limitations and "mixup seed 2" in limitations


def test_an_included_value_whose_newer_attempt_failed_or_diverged_says_so(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, target, counted = _older_value(ctx, lab)
    oom = lab.failed(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert cell(data, "mixup").values_by_seed == {"1": 0.85, "2": 0.90}
    (entry,) = data.integrity.newer_attempts
    assert (entry.newer_attempt_id, entry.outcome, entry.provisional) == (oom, "failed", False)
    assert "OUT_OF_MEMORY" in entry.detail
    (warning,) = [w for w in data.warnings if "without a valid result" in w]
    assert "mixup seed 2" in warning and "OUT_OF_MEMORY" in warning and "provisional" not in warning

    lab.clock.advance(1)
    with ctx.db.tx() as tx:  # a re-run that completed but diverged: acc NaN, evaluation partial
        diverged = make_attempt(
            tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
        make_evaluation(tx, diverged, metrics={"acc": None}, status="partial")
    again = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    earlier, entry = again.integrity.newer_attempts  # every newer attempt, oldest first
    assert (earlier.newer_attempt_id, earlier.outcome) == (oom, "failed")
    assert (entry.newer_attempt_id, entry.outcome) == (diverged, "bad_result")
    assert "partial" in entry.detail
    run = run_of(again, target)
    assert (run.status, run.evaluation_attempt_id) == ("included", counted)
    assert any("without a valid result" in w and "partial" in w for w in again.warnings)
    ev = evidence_export(ctx, purpose="T", idempotency_key="k2", comparison_ref=again.comparison_id)
    assert "without a valid result" in md(ctx, ev.bundle_id).split("## Limitations", 1)[1]


def test_newer_attempt_disclosure_follows_the_selection_rule(ctx: ServiceContext, lab: Lab) -> None:
    """first_valid never changes with a newer attempt (nothing provisional), but the newer
    attempt is disclosed all the same — still running, or failed."""
    exp, target, _ = _older_value(ctx, lab)
    lab.running(target)
    first = ComparisonContract.model_validate({**CONTRACT, "attempt_selection": "first_valid"})
    running = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=first)
    (entry,) = running.integrity.newer_attempts
    assert (entry.outcome, entry.provisional) == ("active", False)
    (warning,) = [w for w in running.warnings if "newer attempt" in w]
    assert "first_valid" in warning and "provisional" not in warning
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        tx.execute(
            "UPDATE attempts SET phase='terminal', scheduler_state='FAILED' WHERE run_id=? AND seq=2",
            (target,),
        )
    failed = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=first)
    (entry,) = failed.integrity.newer_attempts
    assert (entry.outcome, entry.provisional) == ("failed", False)
    assert any("without a valid result" in w for w in failed.warnings)


def test_an_included_value_from_the_latest_attempt_discloses_nothing(ctx: ServiceContext, lab: Lab) -> None:
    exp, target, _ = _older_value(ctx, lab)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert data.integrity.newer_attempts == []
    assert run_of(data, target).reason is None
