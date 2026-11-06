"""Which evaluation counts, end to end through index_attempt and evaluation_collect.

* A job that did not COMPLETE (killed at its walltime, OOM, ...) can still leave finalized
  outputs and a *valid* ``select='last'`` evaluation of a mid-training record; that value must
  never count as the run's result.
* Re-evaluating the counted attempt under another protocol keeps "newest valid evaluation
  counts", but the replaced evaluation is disclosed with the cell.
"""

from __future__ import annotations

import json

from artifacts_support import FINAL, Att, Lab, metrics_jsonl

from hpclab.services.artifacts import index_attempt
from hpclab.services.compare import results_compare
from hpclab.services.coverage import cell_statuses, select_evaluations
from hpclab.services.evaluations import evaluation_collect

LAST = {"split": "val", "select": "last"}


def _attempt(lab: Lab, content: str, **kw: object) -> Att:
    att = lab.attempt(**kw)
    lab.good_outputs(att, checkpoint=False)
    att.write("outputs/metrics.jsonl", content)
    lab.manifest(att)
    index_attempt(lab.ctx, attempt_id=att.attempt_id)
    return att


def test_timed_out_job_with_a_valid_last_record_evaluation_is_failed_not_valid(lab: Lab) -> None:
    exp = lab.experiment()
    done = lab.ready(experiment=exp, seed=1)
    ok = evaluation_collect(lab.ctx, idempotency_key="done", attempt_id=done.attempt_id)
    assert ok.evaluation is not None and ok.evaluation.status == "valid"
    # killed at step 30 of 60; the wrapper trapped TERM and still finalized the outputs
    timed_out = _attempt(lab, metrics_jsonl(final=False), experiment=exp, seed=2, scheduler_state="TIMEOUT")
    assert lab.row("attempts", "attempt_id", timed_out.attempt_id)["result_state"] == "valid"
    res = evaluation_collect(lab.ctx, idempotency_key="late", attempt_id=timed_out.attempt_id, config=LAST)
    ev = res.evaluation
    assert ev is not None and ev.status == "valid"
    assert [(m["name"], m["value"], m["step"]) for m in ev.metrics if m["name"] == "acc"] == [
        ("acc", 0.8, 30)
    ]
    assert any("TIMEOUT" in w and "does not count" in w for w in res.warnings), res.warnings

    assert timed_out.run_id not in select_evaluations(lab.ctx.db, [timed_out.run_id])
    cell = cell_statuses(lab.ctx.db, [timed_out.run_id])[timed_out.run_id]
    assert (cell.code, cell.reason, cell.evaluation_id, cell.metrics) == ("F", "TIMEOUT", None, {})

    cmp = results_compare(lab.ctx, planned_run_set_ref=exp[0], metric_names=["acc"])
    rows = {r.run_id: r for r in cmp.runs}
    assert rows[done.run_id].status == "included" and rows[done.run_id].values == {"acc": 0.9}
    late = rows[timed_out.run_id]
    assert (late.status, late.code, late.evaluation_id, late.values) == ("failed", "F", None, {})
    (group,) = cmp.groups
    assert group.metrics["acc"].n == 1 and group.metrics["acc"].values_by_seed == {"1": 0.9}


def test_re_evaluating_the_counted_attempt_under_another_protocol_is_disclosed(lab: Lab) -> None:
    final = dict(FINAL, metrics={"acc": 0.81, "loss": 0.3})
    ema = {k: v for k, v in final.items() if k != "final"} | {"metrics": {"acc": 0.86}}
    content = metrics_jsonl(final=False) + json.dumps(final) + "\n" + json.dumps(ema) + "\n"
    att = _attempt(lab, content)
    first = evaluation_collect(lab.ctx, idempotency_key="a", attempt_id=att.attempt_id)
    lab.clock.advance(3600)
    second = evaluation_collect(lab.ctx, idempotency_key="b", attempt_id=att.attempt_id, config=LAST)
    a, b = first.evaluation, second.evaluation
    assert a is not None and b is not None and a.config_hash != b.config_hash
    assert (a.status, b.status) == ("valid", "valid")

    cell = cell_statuses(lab.ctx.db, [att.run_id])[att.run_id]
    assert cell.code == "V" and cell.evaluation_id == b.evaluation_id and cell.metrics["acc"] == 0.86
    (old,) = cell.superseded
    assert (old.evaluation_id, old.config_hash, old.evaluator_id) == (
        a.evaluation_id,
        a.config_hash,
        "hpclab_metrics",
    )
    assert old.metrics["acc"] == 0.81 and (old.created_at or "") < (b.created_at or "")
    assert cell.reason is not None and a.evaluation_id in cell.reason
