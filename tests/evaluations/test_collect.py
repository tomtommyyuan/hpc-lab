"""evaluation_collect: digest-verified inputs, idempotency, honest statuses, batch mode."""

from __future__ import annotations

import json
from typing import Any

import pytest
from artifacts_support import Att, Lab, auth_error, keep_mtime, metrics_jsonl

from hpclab.domain.errors import (
    IdempotencyConflict,
    InvalidArgument,
    NotFound,
    NotReady,
    PreconditionFailed,
    Stale,
)
from hpclab.services import evaluations, evaluators
from hpclab.services.artifacts import index_attempt
from hpclab.services.coverage import cell_statuses
from hpclab.services.evaluations import evaluation_collect, evaluation_get, handle_collect
from hpclab.services.outbox import OutboxItem, OutboxRunner
from hpclab.testing.factories import make_run


def count(lab: Lab, table: str) -> int:
    row = lab.ctx.db.one(f"SELECT COUNT(*) AS n FROM {table}")
    assert row is not None
    return int(row["n"])


def attempt_with(lab: Lab, content: str, **kw: object) -> Att:
    att = lab.attempt(**kw)  # type: ignore[arg-type]
    lab.good_outputs(att, checkpoint=False)
    att.write("outputs/metrics.jsonl", content)
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    return att


def test_single_collect_archives_metrics_with_provenance(lab: Lab, indexed: Att) -> None:
    res = evaluation_collect(lab.ctx, idempotency_key="k1", attempt_id=indexed.attempt_id)
    ev = res.evaluation
    assert res.mode == "single" and res.created is True and ev is not None
    assert ev.status == "valid" and ev.evaluator_id == "hpclab_metrics" and ev.evaluator_version == "1"
    metrics = {m["name"]: m for m in ev.metrics}
    assert (
        metrics["acc"]["value"] == 0.9
        and metrics["acc"]["split"] == "val"
        and metrics["acc"]["unit"] == "fraction"
    )
    assert (
        metrics["acc"]["step"] == 60
        and metrics["acc"]["seed"] == 1
        and metrics["acc"]["dataset"] == "synthetic-v1"
    )
    artifact = lab.artifact(indexed, "metrics")
    assert (
        ev.inputs[0]["digest"] == artifact["digest"]
        and ev.inputs[0]["artifact_id"] == artifact["artifact_id"]
    )
    prov = ev.provenance
    assert prov["attempt"]["job"]["job_id"] == "1001" and prov["run"]["code_digest"] == "sha256:" + "c" * 64
    assert prov["evaluator"]["config_hash"] == ev.config_hash and prov["collected_by"] == "tester@host"
    assert prov["selected_record"]["final"] is True
    assert ev.evaluation_ref == f"hpclab://evaluations/{ev.evaluation_id}"
    assert evaluation_get(lab.ctx, evaluation_id=ev.evaluation_ref) == ev

    statuses = cell_statuses(lab.ctx.db, [indexed.run_id])
    assert statuses[indexed.run_id].code == "V" and statuses[indexed.run_id].evaluation_id == ev.evaluation_id


def test_collect_is_idempotent_and_input_keyed(lab: Lab, indexed: Att) -> None:
    first = evaluation_collect(lab.ctx, idempotency_key="k1", attempt_id=indexed.attempt_id)
    replay = evaluation_collect(lab.ctx, idempotency_key="k1", attempt_id=indexed.attempt_id)
    assert replay == first
    other_key = evaluation_collect(lab.ctx, idempotency_key="k2", attempt_id=indexed.attempt_id)
    assert other_key.created is False and other_key.evaluation is not None and first.evaluation is not None
    assert other_key.evaluation.evaluation_id == first.evaluation.evaluation_id
    assert count(lab, "evaluations") == 1 and len(lab.events("evaluation.created")) == 1
    with pytest.raises(IdempotencyConflict):
        evaluation_collect(
            lab.ctx, idempotency_key="k1", attempt_id=indexed.attempt_id, config={"split": "test"}
        )
    explicit = evaluation_collect(
        lab.ctx, idempotency_key="k3", attempt_id=indexed.attempt_id, evaluator_id="hpclab_metrics", config={}
    )
    # an explicit {} normalizes (defaults applied) to the same protocol as the RunSpec default
    assert explicit.evaluation is not None and explicit.evaluation.config_hash == first.evaluation.config_hash


def test_a_different_protocol_is_a_different_evaluation(lab: Lab, indexed: Att) -> None:
    a = evaluation_collect(lab.ctx, idempotency_key="a", attempt_id=indexed.attempt_id)
    b = evaluation_collect(
        lab.ctx, idempotency_key="b", attempt_id=indexed.attempt_id, config={"split": "val", "select": "last"}
    )
    assert a.evaluation and b.evaluation and a.evaluation.evaluation_id != b.evaluation.evaluation_id
    assert a.evaluation.input_key != b.evaluation.input_key


def test_missing_final_record_is_partial_never_valid(lab: Lab) -> None:
    att = attempt_with(lab, metrics_jsonl(final=False))
    res = evaluation_collect(lab.ctx, idempotency_key="p", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "partial"
    codes = {i["code"] for i in res.evaluation.validation["issues"]}
    assert "no_final_record" in codes and any("partial" in w for w in res.warnings)
    assert cell_statuses(lab.ctx.db, [att.run_id])[att.run_id].code == "C!"


def test_non_finite_metric_is_partial(lab: Lab) -> None:
    att = attempt_with(lab, metrics_jsonl(nan=True))
    res = evaluation_collect(lab.ctx, idempotency_key="n", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "partial"
    loss = next(m for m in res.evaluation.metrics if m["name"] == "loss")
    assert loss["value"] is None


def test_evaluator_error_downgrades_a_valid_status(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = evaluators.run_evaluator

    def lying(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        issue = evaluators.EvalIssue(severity="error", code="made_up", message="an error the status ignored")
        return out.model_copy(update={"status": "valid", "issues": [*out.issues, issue]})

    monkeypatch.setattr(evaluators, "run_evaluator", lying)
    res = evaluation_collect(lab.ctx, idempotency_key="x", attempt_id=indexed.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "partial"
    assert res.evaluation.validation["evaluator_status"] == "valid"


def test_oversized_input_is_an_issue_without_download(
    lab: Lab, indexed: Att, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluators, "MAX_INPUT_BYTES", 16)
    lab.transport.calls.clear()
    res = evaluation_collect(lab.ctx, idempotency_key="big", attempt_id=indexed.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "invalid" and res.evaluation.metrics == []
    assert [i["code"] for i in res.evaluation.validation["issues"]] == ["input_too_large"]
    assert "download" not in lab.transport.calls
    again = evaluation_collect(lab.ctx, idempotency_key="big2", attempt_id=indexed.attempt_id)
    assert again.created is False


def test_running_attempt_is_not_evaluated(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    att.write("outputs/metrics.jsonl", metrics_jsonl())
    with pytest.raises(NotReady):
        evaluation_collect(lab.ctx, idempotency_key="r", attempt_id=att.attempt_id)
    assert count(lab, "evaluations") == 0 and count(lab, "idempotency") == 0


def test_unfinalized_inputs_are_refused(lab: Lab) -> None:
    att = lab.attempt()
    att.write("outputs/metrics.json", "{}")  # metrics.jsonl never written
    lab.manifest(att)
    with pytest.raises(PreconditionFailed) as err:
        evaluation_collect(lab.ctx, idempotency_key="m", attempt_id=att.attempt_id)  # indexes inline first
    assert err.value.details["integrity"] == "missing"
    no_manifest = lab.attempt()
    lab.good_outputs(no_manifest)
    with pytest.raises(PreconditionFailed):
        evaluation_collect(lab.ctx, idempotency_key="u", attempt_id=no_manifest.attempt_id)
    assert count(lab, "evaluations") == 0


def test_changed_input_is_stale_and_nothing_is_archived(lab: Lab, indexed: Att) -> None:
    path = indexed.path("outputs/metrics.jsonl")
    keep_mtime(path, path.read_bytes().replace(b'"acc": 0.9', b'"acc": 0.1'))
    with pytest.raises(Stale):
        evaluation_collect(lab.ctx, idempotency_key="s", attempt_id=indexed.attempt_id)
    assert count(lab, "evaluations") == 0
    assert lab.artifact(indexed, "metrics")["digest_state"] == "mismatch"
    assert lab.row("attempts", "attempt_id", indexed.attempt_id)["result_state"] == "invalid"


def test_request_errors_archive_nothing(lab: Lab, indexed: Att) -> None:
    with pytest.raises(NotFound):
        evaluation_collect(lab.ctx, idempotency_key="e1", attempt_id=indexed.attempt_id, evaluator_id="nope")
    with pytest.raises(InvalidArgument):
        evaluation_collect(lab.ctx, idempotency_key="e2", attempt_id=indexed.attempt_id, config={"bogus": 1})
    with pytest.raises(InvalidArgument):
        evaluation_collect(lab.ctx, idempotency_key="e3")
    with pytest.raises(InvalidArgument):
        evaluation_collect(
            lab.ctx, idempotency_key="e4", attempt_id=indexed.attempt_id, run_set_ref=indexed.experiment_id
        )
    assert count(lab, "evaluations") == 0 and count(lab, "idempotency") == 0
    ok = evaluation_collect(
        lab.ctx, idempotency_key="e2", attempt_id=indexed.attempt_id
    )  # key reusable after an error
    assert ok.evaluation is not None


def test_explicit_input_artifact_refs(lab: Lab, indexed: Att) -> None:
    summary = lab.artifact(indexed, "summary")
    res = evaluation_collect(
        lab.ctx,
        idempotency_key="sum",
        input_artifact_refs=[f"hpclab://artifacts/{summary['artifact_id']}"],
        config={"split": "test"},
    )
    assert res.evaluation is not None and res.evaluation.status == "valid"
    assert res.evaluation.attempt_id == indexed.attempt_id and res.evaluation.inputs[0]["name"] == "summary"
    other = lab.ready(seed=2)
    with pytest.raises(InvalidArgument):
        evaluation_collect(
            lab.ctx,
            idempotency_key="mixed",
            attempt_id=other.attempt_id,
            input_artifact_refs=[summary["artifact_id"]],
        )


def test_default_protocol_comes_from_the_run_spec(lab: Lab) -> None:
    att = lab.attempt(
        evaluation={
            "evaluator_id": "hpclab_metrics",
            "evaluator_version": "1",
            "output": "summary",
            "config": {"split": "test"},
        }
    )
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    res = evaluation_collect(lab.ctx, idempotency_key="d", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.inputs[0]["name"] == "summary"
    assert res.evaluation.provenance["evaluator"]["config"]["split"] == "test"


def test_batch_collects_valid_attempts_and_explains_the_rest(lab: Lab) -> None:
    exp = lab.experiment()
    good = lab.ready(experiment=exp, seed=1)
    running = lab.attempt(experiment=exp, seed=2, phase="observing", scheduler_state="RUNNING")
    failed = lab.ready(experiment=exp, seed=3, scheduler_state="FAILED")
    incomplete = lab.attempt(experiment=exp, seed=4)
    incomplete.write("outputs/metrics.jsonl", metrics_jsonl())
    lab.manifest(incomplete)
    index_attempt(lab.ctx, attempt_id=incomplete.attempt_id)
    with lab.ctx.db.tx() as tx:
        idle = make_run(tx, exp[0], exp[1], seed=5)

    res = evaluation_collect(lab.ctx, idempotency_key="b1", run_set_ref=f"hpclab://experiments/{exp[0]}")
    assert res.mode == "batch" and res.run_set == {"kind": "experiment", "id": exp[0], "runs": 5}
    assert [c.attempt_id for c in res.collected] == [good.attempt_id] and res.collected[0].status == "valid"
    reasons = {(s.attempt_id or s.run_id): s.reason for s in res.skipped}
    assert reasons == {
        running.attempt_id: "not_terminal",
        failed.attempt_id: "scheduler_state_not_completed",
        incomplete.attempt_id: "result_not_valid",
        idle: "not_dispatched",
    }
    assert res.counts["collected"] == 1 and res.counts["skipped"] == 4

    again = evaluation_collect(lab.ctx, idempotency_key="b2", run_set_ref=exp[0])
    assert again.collected == []
    done = next(s for s in again.skipped if s.attempt_id == good.attempt_id)
    assert done.reason == "already_evaluated" and done.evaluation_id == res.collected[0].evaluation_id
    assert (
        evaluation_collect(lab.ctx, idempotency_key="b1", run_set_ref=f"hpclab://experiments/{exp[0]}") == res
    )


def test_large_batch_becomes_a_resumable_operation(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluations, "INLINE_BATCH_MAX", 0)  # everything goes to the background
    monkeypatch.setattr(evaluations, "HANDLER_TIME_BUDGET_S", 0.0)  # one attempt per slice
    exp = lab.experiment()
    atts = [lab.ready(experiment=exp, seed=s) for s in (1, 2, 3)]
    res = evaluation_collect(lab.ctx, idempotency_key="big", run_set_ref=exp[1])  # plan id
    assert res.operation_id and res.pending == 3 and res.collected == []
    assert res.operation_ref == f"hpclab://operations/{res.operation_id}"
    assert evaluation_collect(lab.ctx, idempotency_key="big", run_set_ref=exp[1]) == res
    runner = OutboxRunner(lab.ctx)
    runner.register_all(evaluations.outbox_handlers())
    assert runner.run_once() == 1
    op = lab.row("operations", "operation_id", res.operation_id)
    assert op["state"] == "running" and '"done":1' in op["progress_json"]
    for _ in range(10):
        lab.clock.advance(1)
        runner.run_once()
    op = lab.row("operations", "operation_id", res.operation_id)
    assert op["state"] == "succeeded" and '"done":3' in op["progress_json"]
    assert count(lab, "evaluations") == 3
    assert {r["attempt_id"] for r in lab.ctx.db.all("SELECT attempt_id FROM evaluations")} == {
        a.attempt_id for a in atts
    }


def test_batch_handler_blocks_on_auth_and_resumes(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluations, "INLINE_BATCH_MAX", 0)
    exp = lab.experiment()
    lab.ready(experiment=exp, seed=1)
    res = evaluation_collect(lab.ctx, idempotency_key="auth", run_set_ref=exp[0])
    lab.transport.down = auth_error()
    row = lab.ctx.db.one("SELECT * FROM outbox WHERE kind='evaluation.collect'")
    assert row is not None

    def item(tries: int) -> OutboxItem:
        return OutboxItem(
            row["outbox_id"],
            row["kind"],
            row["operation_id"],
            row["target_id"],
            json.loads(row["payload_json"]),
            tries,
        )

    out = handle_collect(lab.ctx, item(1))
    assert out is not None and out.action == "retry" and out.delay_s == 120.0
    op = lab.row("operations", "operation_id", res.operation_id or "")
    assert op["state"] == "blocked" and "hpclab auth lab" in op["error_json"]
    lab.transport.down = None
    out = handle_collect(lab.ctx, item(2))
    assert out is not None and out.action == "done"
    assert lab.row("operations", "operation_id", res.operation_id or "")["state"] == "succeeded"


def test_run_set_ref_errors(lab: Lab) -> None:
    with pytest.raises(InvalidArgument):
        evaluation_collect(lab.ctx, idempotency_key="z1", run_set_ref="att_01J00000000000000000000000")
    with pytest.raises(NotFound):
        evaluation_collect(lab.ctx, idempotency_key="z2", run_set_ref="exp_01J00000000000000000000000")
    with pytest.raises(NotFound):
        evaluation_get(lab.ctx, evaluation_id="ev_01J00000000000000000000000")


def test_handlers_are_exported() -> None:
    assert set(evaluations.outbox_handlers()) == {"evaluation.collect", "evaluation.auto"}
    assert evaluations.periodic_tasks() == []


def test_single_collect_rechecks_a_stale_index_once(lab: Lab) -> None:
    att = lab.attempt()
    index_attempt(lab.ctx, attempt_id=att.attempt_id)  # indexed before the files were visible (NFS lag)
    assert lab.artifact(att, "metrics")["integrity"] == "missing"
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    res = evaluation_collect(lab.ctx, idempotency_key="late", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "valid" and not res.warnings
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "valid"


def test_attempts_that_never_ended_normally_are_refused(lab: Lab) -> None:
    att = lab.attempt(phase="abandoned", scheduler_state="UNKNOWN")
    with pytest.raises(PreconditionFailed):
        evaluation_collect(lab.ctx, idempotency_key="ab", attempt_id=att.attempt_id)


def test_small_batch_is_collected_inline_and_overflow_goes_background(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluations, "INLINE_BATCH_MAX", 2)
    exp = lab.experiment()
    for s in (1, 2, 3):
        lab.ready(experiment=exp, seed=s)
    res = evaluation_collect(lab.ctx, idempotency_key="mixed", run_set_ref=exp[1])
    assert len(res.collected) == 2 and res.pending == 1 and res.operation_id
    assert any("background" in w for w in res.warnings)
    runner = OutboxRunner(lab.ctx)
    runner.register_all(evaluations.outbox_handlers())
    runner.drain()
    assert count(lab, "evaluations") == 3


def test_valid_results_trigger_the_declared_evaluator_automatically(lab: Lab) -> None:
    """When a run declares an evaluator, outputs validating enqueues evaluation.auto; the
    handler creates the same record a manual evaluation_collect would (idempotent)."""
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    with lab.ctx.db.tx() as tx:
        spec = json.loads(tx.one("SELECT spec_json FROM runs WHERE run_id=?", (att.run_id,))["spec_json"])
        spec["evaluation"] = {
            "evaluator_id": "hpclab_metrics",
            "evaluator_version": "1",
            "output": "metrics",
            "config": {},
        }
        tx.execute("UPDATE runs SET spec_json=? WHERE run_id=?", (json.dumps(spec), att.run_id))
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    assert lab.ctx.db.one("SELECT COUNT(*) AS n FROM outbox WHERE kind='evaluation.auto'")["n"] == 1
    runner = OutboxRunner(lab.ctx)
    runner.register_all(evaluations.outbox_handlers())
    runner.drain()
    assert count(lab, "evaluations") == 1
    manual = evaluation_collect(lab.ctx, idempotency_key="later", attempt_id=att.attempt_id)
    assert manual.created is False and count(lab, "evaluations") == 1
