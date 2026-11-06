"""Integrity of results_compare, from the final review.

* An exclusion is judged per *episode*: every comparison request is part of the run's history —
  also one answered by reusing an identical archived comparison — in the events table's order.
  The declaration is when the run's current exclusion became effective after the last request
  that included it; it is late when the result was observed before that.
* The counted evaluator's partial/invalid evaluation of the run's latest attempt makes the run
  *failed*, whatever other evaluators say about that attempt or about older attempts.
* An included value from an older attempt always names the newer attempt and its state
  (running, awaiting evaluation, failed, diverged), under every attempt selection rule.
* A counted evaluation whose protocol (evaluator id/version/config) differs from another valid
  evaluation of the same attempt is disclosed; only COMPLETED attempts count as valid attempts;
  a hand-picked (``evaluation_refs``) attempt is never attributed to the contract's rule.
"""

from __future__ import annotations

from typing import Any

from lab_support import Lab

from hpclab.domain.ids import new_id
from hpclab.domain.models import ComparisonContract
from hpclab.services.compare import (
    ComparisonData,
    ExclusionSpec,
    awaiting_evaluation,
    load_archive,
    results_compare,
)
from hpclab.services.context import ServiceContext
from hpclab.services.evidence import evidence_export
from hpclab.storage.db import jdump, jload
from hpclab.testing.factories import make_attempt, make_evaluation

CONTRACT = {"baseline": {"method": "erm"}, "treatment": ["method"]}
CONFIG = "sha256:" + "c" * 64  # the evaluator config hash lab.valid records
FIRST_VALID = ComparisonContract.model_validate({**CONTRACT, "attempt_selection": "first_valid"})


def cell(data: ComparisonData, label: str, metric: str = "acc") -> Any:
    return next(g for g in data.groups if g.label == label).metrics[metric]


def run_of(data: ComparisonData, run_id: str) -> Any:
    return next(r for r in data.runs if r.run_id == run_id)


def excl(run_id: str, reason: str) -> list[ExclusionSpec]:
    return [ExclusionSpec(run_id=run_id, reason=reason)]


def grid(lab: Lab) -> tuple[str, dict[tuple[str, int], str]]:
    exp = lab.experiment(comparison=CONTRACT)
    plan = lab.plan(exp)
    return exp, {(m, s): lab.run(exp, plan, method=m, seed=s) for m in ("erm", "mixup") for s in (1, 2, 3)}


def results(lab: Lab, runs: dict[tuple[str, int], str]) -> None:
    for s, acc in {1: 0.80, 2: 0.82, 3: 0.84}.items():
        lab.valid(runs[("erm", s)], acc=acc)
    for s, acc in {1: 0.85, 2: 0.86, 3: 0.70}.items():
        lab.valid(runs[("mixup", s)], acc=acc)


def comparison_events(ctx: ServiceContext) -> list[tuple[str, str]]:
    rows = ctx.db.all("SELECT event_type, entity_id FROM events WHERE entity_type='comparison' ORDER BY seq")
    return [(r["event_type"], r["entity_id"]) for r in rows]


# ---------------------------------------------------------------------------
# Exclusion episodes: every request counts, in events order
# ---------------------------------------------------------------------------


def test_declare_include_look_exclude_again_is_late_when_the_inclusion_reused_an_archive(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Declare early → include → look → exclude again. The inclusion request was answered by the
    identical archived comparison (nothing new was archived), but it still ended the first
    exclusion episode: the exclusion in effect was decided after the value was observed."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    shown = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])  # no result yet
    lab.clock.advance(1)
    early = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert early.integrity.exclusions[0].observed_before_exclusion is False
    lab.clock.advance(1)
    include = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert include.reused and include.comparison_id == shown.comparison_id
    results(lab, runs)  # look: the value is observable (and evaluated)
    lab.clock.advance(5)
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "looked unstable")
    )
    assert not final.reused
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is True
    assert rec.first_declared_at == final.created_at
    assert rec.first_observed_at is not None and rec.first_observed_at < rec.first_declared_at
    assert rec.note is not None and early.comparison_id in rec.note and shown.comparison_id in rec.note
    # every request is in the ledger, in the order it was made
    assert comparison_events(ctx) == [
        ("comparison.created", shown.comparison_id),
        ("comparison.created", early.comparison_id),
        ("comparison.reused", shown.comparison_id),
        ("comparison.created", final.comparison_id),
    ]
    ev = evidence_export(ctx, purpose="T", idempotency_key="k", comparison_ref=final.comparison_id)
    assert "late_exclusions" in ev.integrity_flags
    # asking the same question again is stable: reused, the decision time does not move
    again = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "looked unstable")
    )
    assert again.reused and again.comparison_id == final.comparison_id


def test_the_order_of_requests_is_the_events_order_not_the_clock(ctx: ServiceContext, lab: Lab) -> None:
    """Declaration and a reused inclusion recorded at the same instant: only the events table
    says which came last."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    shown = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    early = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    include = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert include.reused and shown.created_at == early.created_at  # same instant
    results(lab, runs)
    final = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert final.integrity.exclusions[0].observed_before_exclusion is True


def test_re_declaring_after_a_reused_inclusion_but_before_the_result_is_pre_declared(
    ctx: ServiceContext, lab: Lab
) -> None:
    """The episode in effect started with the re-declaration, which preceded the result."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    lab.running(target)
    shown = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    lab.clock.advance(1)
    results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    lab.clock.advance(1)
    assert (
        results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"]).comparison_id
        == shown.comparison_id
    )
    lab.clock.advance(1)
    redeclared = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x")
    )
    assert not redeclared.reused  # a new decision: the declaration time moved
    (rec,) = redeclared.integrity.exclusions
    assert rec.first_declared_at == redeclared.created_at and rec.observed_before_exclusion is False
    results(lab, runs)
    final = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is False
    assert rec.first_declared_at == redeclared.created_at


def test_repeating_an_exclusion_request_extends_the_episode(ctx: ServiceContext, lab: Lab) -> None:
    """A reused *exclusion* request is part of the same episode: it neither moves the declaration
    nor revokes it."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    lab.running(target)
    first = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    lab.clock.advance(1)
    again = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    assert again.reused and again.comparison_id == first.comparison_id
    results(lab, runs)
    final = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    (rec,) = final.integrity.exclusions
    assert rec.observed_before_exclusion is False and rec.first_declared_at == first.created_at
    assert rec.note is None


def _completed_and_evaluated(ctx: ServiceContext, run_id: str, acc: float) -> str:
    """A COMPLETED attempt whose end and evaluation are recorded as events, at the clock's now."""
    with ctx.db.tx() as tx:
        attempt = make_attempt(
            tx,
            run_id,
            phase="terminal",
            scheduler_state="COMPLETED",
            result_state="valid",
            terminal_at=tx.now,
        )
        tx.event("attempt", attempt, "attempt.terminal", {"scheduler_state": "COMPLETED"})
        ev = make_evaluation(tx, attempt, metrics={"acc": acc})
        tx.event("evaluation", ev, "evaluation.created", {"run_id": run_id, "attempt_id": attempt})
    return attempt


def test_a_result_recorded_after_the_declaration_at_the_same_instant_is_not_late(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Declaration and result share a timestamp (millisecond clock): the events table orders them."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    declared = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "planned ablation")
    )
    _completed_and_evaluated(ctx, target, 0.7)  # same instant, recorded after the declaration
    final = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "planned ablation")
    )
    (rec,) = final.integrity.exclusions
    assert rec.first_declared_at == declared.created_at == rec.first_observed_at
    assert rec.observed_before_exclusion is False


def test_a_result_recorded_before_the_declaration_at_the_same_instant_is_late(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    _completed_and_evaluated(ctx, target, 0.7)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "x"))
    (rec,) = data.integrity.exclusions
    assert rec.first_observed_at == data.created_at and rec.observed_before_exclusion is True


def _metrics_artifact(
    ctx: ServiceContext, attempt_id: str, *, role: str = "metrics", read: bool = True
) -> str:
    """A live (mutable) output of a running attempt; ``read``: artifact_read saw it grow."""
    with ctx.db.tx() as tx:
        run = tx.one("SELECT run_id FROM attempts WHERE attempt_id=?", (attempt_id,))
        assert run is not None
        art = new_id("artifact")
        tx.insert(
            "artifacts",
            {
                "artifact_id": art,
                "attempt_id": attempt_id,
                "run_id": run["run_id"],
                "name": role,
                "role": role,
                "kind": "file",
                "format": "jsonl",
                "remote_path": f"/remote/attempts/{attempt_id}/outputs/{role}.jsonl",
                "size_bytes": 10,
                "mtime": tx.now,
                "digest_state": "none",
                "integrity": "mutable",
                "required": 1,
                "observed_at": tx.now,
                "version": 1,
            },
        )
        if read:  # what artifacts._refresh_mutable records when a read finds the file changed
            tx.event("artifact", art, "artifact.observed", {"size_bytes": 20, "integrity": "mutable"})
    return art


def test_reading_the_live_metrics_before_excluding_a_run_is_an_observation(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Watching a seed's live metrics and then excluding it is an exclusion after observation,
    although no evaluation exists yet."""
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    live = lab.running(target)
    _metrics_artifact(ctx, live)
    read_at = ctx.db.one("SELECT recorded_at FROM events WHERE event_type='artifact.observed'")
    assert read_at is not None
    lab.clock.advance(1)
    data = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "diverging")
    )
    (rec,) = data.integrity.exclusions
    assert rec.observed_before_exclusion is True and rec.first_observed_at == read_at["recorded_at"]


def test_reads_of_other_outputs_or_after_the_declaration_are_not_observations_of_the_metric(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, runs = grid(lab)
    target = runs[("mixup", 3)]
    live = lab.running(target)
    _metrics_artifact(ctx, live, role="log")  # the log, not the metric
    _metrics_artifact(ctx, lab.running(runs[("mixup", 2)]))  # another run's metrics
    lab.clock.advance(1)
    declared = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "planned")
    )
    assert declared.integrity.exclusions[0].observed_before_exclusion is False
    lab.clock.advance(1)
    _metrics_artifact(ctx, live)  # read after the exclusion was declared
    lab.clock.advance(1)
    again = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], exclusions=excl(target, "planned")
    )
    (rec,) = again.integrity.exclusions
    assert rec.observed_before_exclusion is False and rec.first_declared_at == declared.created_at
    assert rec.first_observed_at is not None and rec.first_observed_at > declared.created_at


# ---------------------------------------------------------------------------
# The counted evaluator's verdict on the latest attempt decides
# ---------------------------------------------------------------------------


def _two_by_two(lab: Lab) -> tuple[str, dict[tuple[str, int], str]]:
    exp = lab.experiment(comparison=CONTRACT, evaluation={"evaluator_id": "hpclab_metrics"})
    plan = lab.plan(exp)
    runs = {(m, s): lab.run(exp, plan, method=m, seed=s) for m in ("erm", "mixup") for s in (1, 2)}
    for s in (1, 2):
        lab.valid(runs[("erm", s)], acc=0.8 + s / 100)
    lab.valid(runs[("mixup", 1)], acc=0.85)
    return exp, runs


def test_a_partial_latest_attempt_is_failed_although_an_older_attempt_has_another_evaluators_value(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    older, _, _ = lab.valid(target, acc=0.9, evaluator_id="json_path")  # never seen by hpclab_metrics
    lab.clock.advance(1)
    with ctx.db.tx() as tx:  # the re-run completed; the experiment's evaluator finds it diverged
        latest = make_attempt(tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, latest, metrics={"acc": None}, status="partial")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("C!", "failed")
    assert run.reason is not None and run.reason.startswith("completed but evaluation partial")
    assert "hpclab_metrics" in run.reason and "collect one" not in run.reason and older in run.reason
    mixup = cell(data, "mixup")
    assert (mixup.missing_seeds, mixup.failed_seeds) == ([], [2])
    assert (data.counts.failed, data.counts.missing) == (1, 0)


def test_an_invalid_latest_attempt_is_failed_under_pinned_evaluation_refs(
    ctx: ServiceContext, lab: Lab
) -> None:
    """With evaluation_refs the counted evaluations are the pinned ones: a pinned invalid
    evaluation of the latest attempt fails the run even if another (unpinned) one is valid."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    lab.valid(target, acc=0.9, evaluator_id="json_path")
    with ctx.db.tx() as tx:
        latest = make_attempt(tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        bad = make_evaluation(tx, latest, metrics={"acc": None}, status="invalid")
    pinned = [
        r["evaluation_id"]
        for r in ctx.db.all("SELECT evaluation_id, run_id FROM evaluations")
        if r["run_id"] != target or r["evaluation_id"] == bad
    ]
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=pinned)
    run = run_of(data, target)
    assert (run.code, run.status) == ("C!", "failed")
    assert run.reason is not None and "invalid" in run.reason and "evaluation_refs" in run.reason


def _older_attempt_judged_partial(ctx: ServiceContext, lab: Lab) -> tuple[str, str, str]:
    """mixup seed 2: attempt #1 valid for json_path, partial (diverged) for the experiment's
    evaluator hpclab_metrics."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    older, _, _ = lab.valid(target, acc=0.9, evaluator_id="json_path")
    with ctx.db.tx() as tx:
        make_evaluation(tx, older, metrics={"acc": None}, status="partial")
    return exp, target, older


def test_a_verdict_on_an_older_attempt_does_not_fail_a_run_whose_newer_attempt_is_running(
    ctx: ServiceContext, lab: Lab
) -> None:
    """Classified by the counted evaluator alone the run is running (missing): another evaluator's
    valid value of the diverged older attempt must not turn it into a failure."""
    exp, target, older = _older_attempt_judged_partial(ctx, lab)
    lab.running(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("R", "missing")
    assert run.reason is not None and run.reason.startswith("RUNNING") and older in run.reason
    assert "partial" in run.reason and "collect one" not in run.reason
    assert (cell(data, "mixup").missing_seeds, cell(data, "mixup").failed_seeds) == ([2], [])


def test_a_verdict_on_an_older_attempt_leaves_a_newer_completed_attempt_awaiting_evaluation(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, target, _ = _older_attempt_judged_partial(ctx, lab)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        make_attempt(tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("E", "missing")
    assert awaiting_evaluation(run)  # the run coverage and the summary agree that it awaits evaluation
    assert "1 of the missing completed and await evaluation" in data.summary
    ev = evidence_export(ctx, purpose="T", idempotency_key="k", comparison_ref=data.comparison_id)
    assert "awaiting_evaluation" in ev.integrity_flags


def test_a_verdict_on_an_older_attempt_still_fails_a_run_whose_newer_attempt_failed(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, target, older = _older_attempt_judged_partial(ctx, lab)
    lab.failed(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    run = run_of(data, target)
    assert (run.code, run.status) == ("F", "failed")
    assert run.reason is not None and older in run.reason and "partial" in run.reason


# ---------------------------------------------------------------------------
# Newer attempts are always disclosed
# ---------------------------------------------------------------------------


def _older_value(lab: Lab) -> tuple[str, str, str, str]:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, ev = lab.valid(target, acc=0.90)
    return exp, target, attempt, ev


def test_a_running_newer_attempt_is_disclosed_under_first_valid(ctx: ServiceContext, lab: Lab) -> None:
    exp, target, counted, _ = _older_value(lab)
    newer = lab.running(target)
    data = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=FIRST_VALID
    )
    assert cell(data, "mixup").values_by_seed == {"1": 0.85, "2": 0.90}
    (entry,) = data.integrity.newer_attempts
    assert (entry.run_id, entry.value_attempt_id, entry.newer_attempt_id) == (target, counted, newer)
    assert (entry.value_attempt_seq, entry.newer_attempt_seq) == (1, 2)
    assert (entry.outcome, entry.provisional) == ("active", False)
    assert entry.newer_state == "in progress (RUNNING)"
    assert entry.disclosure == "value from attempt #1; newer attempt #2 is in progress (RUNNING)"
    (warning,) = [w for w in data.warnings if "newer attempt" in w]
    assert "mixup seed 2: value from attempt #1; newer attempt #2 is in progress (RUNNING)" in warning
    assert "first_valid" in warning and "provisional" not in warning


def test_a_running_newer_attempt_is_disclosed_when_evaluations_are_pinned(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, target, counted, ev = _older_value(lab)
    newer = lab.running(target)
    refs = [r["evaluation_id"] for r in ctx.db.all("SELECT evaluation_id FROM evaluations")]
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=refs)
    assert ev in refs and run_of(data, target).status == "included"
    (entry,) = data.integrity.newer_attempts
    assert (entry.value_attempt_id, entry.newer_attempt_id, entry.provisional) == (counted, newer, False)
    assert any("pinned by evaluation_refs" in w and "newer attempt #2" in w for w in data.warnings)


def test_a_diverged_newer_attempt_names_its_evaluation(ctx: ServiceContext, lab: Lab) -> None:
    exp, target, counted, _ = _older_value(lab)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        diverged = make_attempt(
            tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
        make_evaluation(tx, diverged, metrics={"acc": None}, status="partial")
    for contract in (None, FIRST_VALID):
        data = results_compare(
            ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=contract
        )
        (entry,) = data.integrity.newer_attempts
        assert (entry.value_attempt_id, entry.newer_attempt_id, entry.outcome) == (
            counted,
            diverged,
            "bad_result",
        )
        assert entry.newer_state == "COMPLETED with a partial evaluation (hpclab_metrics)"
        assert entry.disclosure.startswith(
            "value from attempt #1; newer attempt #2 is COMPLETED with a partial"
        )


def test_a_failed_newer_attempt_states_how_it_ended(ctx: ServiceContext, lab: Lab) -> None:
    exp, target, _, _ = _older_value(lab)
    lab.failed(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    (entry,) = data.integrity.newer_attempts
    assert entry.newer_state == "failed (OUT_OF_MEMORY, cuda_oom)"
    assert entry.disclosure == "value from attempt #1; newer attempt #2 is failed (OUT_OF_MEMORY, cuda_oom)"


def test_every_newer_attempt_without_a_result_is_disclosed_not_only_the_latest(
    ctx: ServiceContext, lab: Lab
) -> None:
    """#1 valid, #2 diverged, #3 running: the diverged re-run is not hidden behind the running one."""
    exp, target, counted, _ = _older_value(lab)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        diverged = make_attempt(
            tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
        make_evaluation(tx, diverged, metrics={"acc": None}, status="partial")
    newest = lab.running(target)
    data = results_compare(
        ctx, planned_run_set_ref=exp, metric_names=["acc"], comparison_contract=FIRST_VALID
    )
    entries = data.integrity.newer_attempts
    assert [(e.value_attempt_id, e.newer_attempt_id, e.newer_attempt_seq) for e in entries] == [
        (counted, diverged, 2),
        (counted, newest, 3),
    ]
    assert [e.outcome for e in entries] == ["bad_result", "active"]
    assert entries[0].disclosure.startswith(
        "value from attempt #1; newer attempt #2 is COMPLETED with a partial"
    )
    ended = next(w for w in data.warnings if "ended without a valid result" in w)
    assert ended.startswith("1 included run(s)") and "newer attempt #2" in ended
    steady = next(w for w in data.warnings if "has no result yet" in w)
    assert steady.startswith("1 included run(s)") and "newer attempt #3" in steady


def test_an_intermediate_attempt_awaiting_evaluation_makes_the_value_provisional(
    ctx: ServiceContext, lab: Lab
) -> None:
    """#1 valid, #2 COMPLETED (not evaluated yet), #3 failed: under latest_valid the value becomes
    #2's once it is evaluated, whatever #3 did."""
    exp, target, counted, _ = _older_value(lab)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        pending = make_attempt(
            tx, target, phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
    failed = lab.failed(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    by_attempt = {e.newer_attempt_id: e for e in data.integrity.newer_attempts}
    assert set(by_attempt) == {pending, failed}
    assert (by_attempt[pending].outcome, by_attempt[pending].provisional) == ("awaiting_evaluation", True)
    assert (by_attempt[failed].outcome, by_attempt[failed].provisional) == ("failed", False)
    assert all(e.value_attempt_id == counted for e in by_attempt.values())
    assert any("provisional" in w and "newer attempt #2" in w for w in data.warnings)


def test_newer_attempt_warnings_count_runs_not_attempts(ctx: ServiceContext, lab: Lab) -> None:
    """#1 valid, #2 and #3 failed: one run, two disclosed re-runs."""
    exp, target, _, _ = _older_value(lab)
    second = lab.failed(target)
    third = lab.failed(target, state="NODE_FAIL")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert [(e.newer_attempt_id, e.outcome) for e in data.integrity.newer_attempts] == [
        (second, "failed"),
        (third, "failed"),
    ]
    (ended,) = [w for w in data.warnings if "ended without a valid result" in w]
    assert ended.startswith("1 included run(s)")
    assert (
        "newer attempt #2 is failed (OUT_OF_MEMORY" in ended
        and "newer attempt #3 is failed (NODE_FAIL" in ended
    )


def test_archives_with_the_earlier_newer_attempt_fields_still_load(ctx: ServiceContext, lab: Lab) -> None:
    exp, target, counted, _ = _older_value(lab)
    newer = lab.running(target)
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    row = ctx.db.one("SELECT result_json FROM comparisons WHERE comparison_id=?", (data.comparison_id,))
    assert row is not None
    archive = jload(row["result_json"])
    legacy = {
        "run_id": target,
        "group_label": "mixup",
        "seed": 2,
        "counted_attempt_id": counted,
        "latest_attempt_id": newer,
        "outcome": "active",
        "detail": f"attempt {newer} is observing (RUNNING)",
        "provisional": True,
    }
    archive["data"]["integrity"]["newer_attempts"] = [legacy]
    with ctx.db.tx() as tx:
        tx.execute(
            "UPDATE comparisons SET result_json=? WHERE comparison_id=?", (jdump(archive), data.comparison_id)
        )
    loaded, _ = load_archive(ctx.db, data.comparison_id)
    (entry,) = loaded.integrity.newer_attempts
    assert (entry.value_attempt_id, entry.newer_attempt_id) == (counted, newer)


# ---------------------------------------------------------------------------
# Protocol switches, COMPLETED-only valid attempts, hand-picked attempts
# ---------------------------------------------------------------------------


def test_a_protocol_switch_within_the_counted_attempt_is_disclosed(ctx: ServiceContext, lab: Lab) -> None:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, v1 = lab.valid(target, acc=0.90)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:  # re-evaluated under a new evaluator version: this one counts
        v2 = make_evaluation(tx, attempt, metrics={"acc": 0.93}, evaluator_version="2", config_hash=CONFIG)
        # another evaluator never counts for this experiment: not a switch of the counted protocol
        make_evaluation(tx, attempt, metrics={"acc": 0.5}, evaluator_id="json_path")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert cell(data, "mixup").values_by_seed["2"] == 0.93
    (switch,) = data.integrity.protocol_switches
    assert (switch.run_id, switch.attempt_id, switch.group_label, switch.seed) == (
        target,
        attempt,
        "mixup",
        2,
    )
    assert (switch.counted.evaluation_id, switch.counted.evaluator) == (v2, "hpclab_metrics@2")
    assert [(o.evaluation_id, o.evaluator, o.config_hash, o.values) for o in switch.others] == [
        (v1, "hpclab_metrics@1", CONFIG, {"acc": 0.90})
    ]
    assert switch.counted.values == {"acc": 0.93}
    (warning,) = [w for w in data.warnings if "protocol" in w]
    assert "mixup seed 2" in warning and "hpclab_metrics@2" in warning

    # the same protocol re-evaluated is not a switch
    other_exp, other_runs = _two_by_two(lab)
    a, _, _ = lab.valid(other_runs[("mixup", 2)], acc=0.9)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        make_evaluation(tx, a, metrics={"acc": 0.9}, config_hash=CONFIG)
    same_protocol = results_compare(ctx, planned_run_set_ref=other_exp, metric_names=["acc"])
    assert same_protocol.integrity.protocol_switches == []
    assert not any("protocol" in w for w in same_protocol.warnings)


def test_a_config_switch_is_disclosed_when_the_older_evaluation_is_pinned(
    ctx: ServiceContext, lab: Lab
) -> None:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, old = lab.valid(target, acc=0.90)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        new = make_evaluation(tx, attempt, metrics={"acc": 0.95}, config_hash="sha256:" + "e" * 64)
    rows = ctx.db.all("SELECT evaluation_id FROM evaluations")
    refs = [r["evaluation_id"] for r in rows if r["evaluation_id"] != new]
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=refs)
    assert cell(data, "mixup").values_by_seed["2"] == 0.90
    (switch,) = data.integrity.protocol_switches
    assert switch.counted.evaluation_id == old and [o.evaluation_id for o in switch.others] == [new]
    assert switch.others[0].config_hash == "sha256:" + "e" * 64 and switch.others[0].values == {"acc": 0.95}


def test_a_timeout_attempt_with_a_valid_evaluation_is_not_another_valid_attempt(
    ctx: ServiceContext, lab: Lab
) -> None:
    """A job killed at its walltime can leave finalized outputs and a valid ``select='last'``
    evaluation, but it is not a valid result (coverage never counts it)."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    completed, _, _ = lab.valid(target, acc=0.9)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        timeout = make_attempt(
            tx, target, phase="terminal", scheduler_state="TIMEOUT", result_state="valid", terminal_at=tx.now
        )
        make_evaluation(tx, timeout, metrics={"acc": 0.95})
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert run_of(data, target).evaluation_attempt_id == completed
    assert data.integrity.multiple_valid_attempts == []
    (entry,) = data.integrity.newer_attempts  # the TIMEOUT re-run is disclosed as such instead
    assert entry.outcome == "failed" and "TIMEOUT" in entry.newer_state


def test_hand_picked_attempts_are_attributed_to_evaluation_refs(ctx: ServiceContext, lab: Lab) -> None:
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    first, _, ev1 = lab.valid(target, acc=0.9)
    lab.valid(target, acc=0.8)
    refs = [
        r["evaluation_id"]
        for r in ctx.db.all("SELECT evaluation_id, run_id FROM evaluations")
        if r["run_id"] != target or r["evaluation_id"] == ev1
    ]
    pinned = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=refs)
    (multi,) = pinned.integrity.multiple_valid_attempts
    assert (multi.run_id, multi.selected_attempt_id, multi.rule) == (target, first, "evaluation_refs")
    ruled = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert ruled.integrity.multiple_valid_attempts[0].rule == "latest_valid"


def test_an_attempt_valid_only_for_another_evaluator_is_not_another_valid_attempt(
    ctx: ServiceContext, lab: Lab
) -> None:
    """The experiment counts hpclab_metrics only: a re-run that only json_path evaluated is a newer
    attempt awaiting evaluation, not a second valid attempt that latest_valid passed over."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    first, _, _ = lab.valid(target, acc=0.9)
    second, _, _ = lab.valid(target, acc=0.8, evaluator_id="json_path")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert run_of(data, target).evaluation_attempt_id == first
    assert data.integrity.multiple_valid_attempts == []
    (entry,) = data.integrity.newer_attempts
    assert (entry.newer_attempt_id, entry.outcome, entry.provisional) == (second, "awaiting_evaluation", True)
    # pinned evaluations: every valid attempt could have been pinned, so both count
    refs = [r["evaluation_id"] for r in ctx.db.all("SELECT evaluation_id FROM evaluations")]
    pinned = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"], evaluation_refs=refs)
    (multi,) = pinned.integrity.multiple_valid_attempts
    assert (multi.valid_attempts, multi.rule) == (2, "evaluation_refs")


def test_a_re_evaluation_of_the_counted_attempt_that_was_not_valid_is_disclosed(
    ctx: ServiceContext, lab: Lab
) -> None:
    """The experiment's evaluator, in a new version, judged the counted attempt partial after the
    old version found it valid: the older valid value is reported (only valid evaluations count),
    but the newer verdict is not hidden."""
    exp, runs = _two_by_two(lab)
    target = runs[("mixup", 2)]
    attempt, _, v1 = lab.valid(target, acc=0.90)
    lab.clock.advance(1)
    with ctx.db.tx() as tx:
        v2 = make_evaluation(tx, attempt, metrics={"acc": None}, status="partial", evaluator_version="2")
        # another evaluator never counts for this experiment: its verdict is no switch
        make_evaluation(tx, attempt, metrics={"acc": None}, status="invalid", evaluator_id="json_path")
    data = results_compare(ctx, planned_run_set_ref=exp, metric_names=["acc"])
    assert run_of(data, target).status == "included" and cell(data, "mixup").values_by_seed["2"] == 0.90
    (switch,) = data.integrity.protocol_switches
    assert (switch.counted.evaluation_id, switch.counted.status) == (v1, "valid")
    assert [(o.evaluation_id, o.evaluator, o.status, o.values) for o in switch.others] == [
        (v2, "hpclab_metrics@2", "partial", {"acc": None})
    ]
    (warning,) = [w for w in data.warnings if "protocol" in w]
    assert "hpclab_metrics@2" in warning and "(partial)" in warning

    # the same protocol judging it again (e.g. before its outputs were final) is no switch
    other_exp, other_runs = _two_by_two(lab)
    a, _, _ = lab.valid(other_runs[("mixup", 2)], acc=0.9)
    with ctx.db.tx() as tx:
        make_evaluation(tx, a, metrics={"acc": None}, status="partial", config_hash=CONFIG)
    assert (
        results_compare(ctx, planned_run_set_ref=other_exp, metric_names=["acc"]).integrity.protocol_switches
        == []
    )
