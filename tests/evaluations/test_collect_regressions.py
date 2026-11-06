"""Regressions found in review: each test failed before its fix.

* collecting right after the job ended must not refuse "attempt active" because the
  artifacts were last indexed while it ran (the terminal index is still queued);
* an outage while collecting is a retryable ``unavailable`` / ``auth_required``, never a
  permanent ``precondition_failed`` that makes an agent give up on a good run.
"""

from __future__ import annotations

import pytest
from artifacts_support import Lab, auth_error, conn_lost, metrics_jsonl

from hpclab.domain.errors import ErrorCode, HpcLabError
from hpclab.services.artifacts import index_attempt
from hpclab.services.evaluations import evaluation_collect


def test_collect_right_after_the_job_ended_reindexes_mutable_inputs(lab: Lab) -> None:
    att = lab.attempt(phase="observing", scheduler_state="RUNNING")
    att.write("outputs/metrics.jsonl", metrics_jsonl(final=False))
    index_attempt(lab.ctx, attempt_id=att.attempt_id)  # someone listed outputs while it ran
    assert lab.artifact(att, "metrics")["integrity"] == "mutable"
    # the job finishes; observe marks the attempt terminal; artifacts.index is still queued
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    lab.set_attempt(att, phase="terminal", scheduler_state="COMPLETED", result_state="pending")
    res = evaluation_collect(lab.ctx, idempotency_key="now", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "valid"
    assert lab.artifact(att, "metrics")["integrity"] == "finalized"
    assert lab.row("attempts", "attempt_id", att.attempt_id)["result_state"] == "valid"


@pytest.mark.parametrize(
    ("error", "code"), [(conn_lost, ErrorCode.UNAVAILABLE), (auth_error, ErrorCode.AUTH_REQUIRED)]
)
def test_outage_while_collecting_is_retryable(lab: Lab, error: object, code: ErrorCode) -> None:
    att = lab.attempt()
    lab.good_outputs(att, checkpoint=False)
    lab.manifest(att)
    lab.transport.down = error()  # type: ignore[operator]
    with pytest.raises(HpcLabError) as err:
        evaluation_collect(lab.ctx, idempotency_key="down", attempt_id=att.attempt_id)
    assert err.value.code == code and err.value.retryable
    lab.transport.down = None
    res = evaluation_collect(lab.ctx, idempotency_key="down", attempt_id=att.attempt_id)
    assert res.evaluation is not None and res.evaluation.status == "valid"


def test_bad_batch_override_is_a_request_error_not_n_skips(lab: Lab) -> None:
    from hpclab.domain.errors import InvalidArgument, NotFound

    exp = lab.experiment()
    lab.ready(experiment=exp, seed=1)
    lab.ready(experiment=exp, seed=2)
    with pytest.raises(InvalidArgument):
        evaluation_collect(lab.ctx, idempotency_key="bad", run_set_ref=exp[0], config={"bogus": 1})
    with pytest.raises(NotFound):  # as in single mode: no such evaluator version
        evaluation_collect(lab.ctx, idempotency_key="bad2", run_set_ref=exp[0], evaluator_version="999")
    assert lab.ctx.db.one("SELECT COUNT(*) AS n FROM idempotency")["n"] == 0  # type: ignore[index]
    ok = evaluation_collect(lab.ctx, idempotency_key="bad", run_set_ref=exp[0])  # key still usable
    assert len(ok.collected) == 2
