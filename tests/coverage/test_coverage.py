from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from hpclab.domain.clock import iso
from hpclab.services.coverage import cell_statuses, select_evaluations, summarize_codes
from hpclab.storage.db import Database
from hpclab.testing.factories import make_attempt, make_evaluation, make_experiment, make_plan, make_run


def _setup(db: Database) -> dict[str, str]:
    ids: dict[str, str] = {}
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        ids["prepared"] = make_run(tx, e, p, seed=1)
        ids["valid"] = make_run(tx, e, p, seed=2)
        a = make_attempt(
            tx, ids["valid"], phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
        make_evaluation(tx, a, metrics={"acc": 0.91})
        ids["unverified"] = make_run(tx, e, p, seed=3)
        make_attempt(
            tx, ids["unverified"], phase="terminal", scheduler_state="COMPLETED", result_state="incomplete"
        )
        ids["evaluating"] = make_run(tx, e, p, seed=10)
        make_attempt(
            tx, ids["evaluating"], phase="terminal", scheduler_state="COMPLETED", result_state="valid"
        )
        ids["timeout"] = make_run(tx, e, p, seed=4)
        make_attempt(tx, ids["timeout"], phase="terminal", scheduler_state="TIMEOUT")
        ids["uncertain"] = make_run(tx, e, p, seed=5)
        make_attempt(
            tx, ids["uncertain"], phase="submission_uncertain", scheduler_state="UNKNOWN", job_id=None
        )
        ids["running"] = make_run(tx, e, p, seed=6)
        make_attempt(
            tx, ids["running"], phase="observing", scheduler_state="RUNNING", last_observed_at=tx.now
        )
        ids["queued"] = make_run(tx, e, p, seed=7)
        make_attempt(
            tx, ids["queued"], phase="observing", scheduler_state="PENDING", pending_reason="Priority"
        )
        ids["cancelled"] = make_run(tx, e, p, seed=8)
        make_attempt(tx, ids["cancelled"], phase="terminal", scheduler_state="CANCELLED")
        # an evaluation on an attempt whose outputs are NOT valid must never count
        ids["eval_on_invalid"] = make_run(tx, e, p, seed=9)
        a2 = make_attempt(
            tx, ids["eval_on_invalid"], phase="terminal", scheduler_state="COMPLETED", result_state="invalid"
        )
        make_evaluation(tx, a2, metrics={"acc": 0.99})
    return ids


def test_codes(db: Database) -> None:
    ids = _setup(db)
    st = cell_statuses(db, list(ids.values()))
    assert {k: st[v].code for k, v in ids.items()} == {
        "prepared": "P",
        "valid": "V",
        "unverified": "C!",
        "timeout": "F",
        "uncertain": "U",
        "running": "R",
        "queued": "Q",
        "cancelled": "X",
        "eval_on_invalid": "C!",  # outputs invalid: an evaluation of them never counts
        "evaluating": "E",
    }
    assert st[ids["valid"]].metrics == {"acc": 0.91}
    assert st[ids["timeout"]].reason == "TIMEOUT"
    assert st[ids["queued"]].reason == "Priority"
    counts = summarize_codes(st.values())
    assert counts["planned"] == 10 and counts["valid"] == 1


def test_stale_running_becomes_unknown(db: Database) -> None:
    ids = _setup(db)
    later = db.clock.now() + timedelta(hours=2)
    st = cell_statuses(db, [ids["running"]], stale_after_s=600, now=later)
    assert st[ids["running"]].code == "?" and st[ids["running"]].stale


def test_attempt_selection_rule(db: Database) -> None:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        a1 = make_attempt(tx, r, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        ev1 = make_evaluation(tx, a1, metrics={"acc": 0.5})
        a2 = make_attempt(tx, r, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        ev2 = make_evaluation(tx, a2, metrics={"acc": 0.7})
    assert select_evaluations(db, [r])[r]["evaluation_id"] == ev2
    assert select_evaluations(db, [r], attempt_selection="first_valid")[r]["evaluation_id"] == ev1
    assert select_evaluations(db, [r], evaluation_ids={ev1})[r]["evaluation_id"] == ev1
    # a valid earlier evaluation still counts while a newer attempt runs, with the reason disclosed
    with db.tx() as tx:
        make_attempt(
            tx, r, phase="observing", scheduler_state="RUNNING", last_observed_at=iso(tx.clock.now())
        )
    s = cell_statuses(db, [r])[r]
    assert s.code == "V" and s.evaluation_id == ev2 and "latest attempt" in (s.reason or "")


def test_just_identified_job_is_queued_not_stale(db: Database) -> None:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        make_attempt(tx, r, phase="job_identified", scheduler_state="UNKNOWN", identified_at=tx.now)
    now = db.clock.now()
    assert cell_statuses(db, [r], stale_after_s=600, now=now)[r].code == "Q"
    later = now + timedelta(hours=1)
    assert cell_statuses(db, [r], stale_after_s=600, now=later)[r].code == "?"


def test_partial_evaluation_of_another_evaluator_does_not_fail_the_cell(db: Database) -> None:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        a = make_attempt(tx, r, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, a, status="partial", evaluator_id="json_path")
    assert cell_statuses(db, [r], evaluator_id="hpclab_metrics")[r].code == "E"
    assert cell_statuses(db, [r])[r].code == "C!"  # unfiltered: the partial evaluation counts


# -- only a job that COMPLETED can produce a counted result ----------------------------------


@pytest.mark.parametrize("state", ["TIMEOUT", "FAILED", "OUT_OF_MEMORY", "PREEMPTED", "NODE_FAIL"])
def test_valid_evaluation_of_a_job_that_did_not_complete_never_counts(db: Database, state: str) -> None:
    """A job killed at its walltime can still write finalized outputs (the wrapper traps TERM),
    so result_state is valid, and an explicit select='last' evaluation of it is valid too. Its
    value is a mid-training number: the cell stays F and nothing counts for the run."""
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        a = make_attempt(tx, r, phase="terminal", scheduler_state=state, result_state="valid")
        make_evaluation(tx, a, metrics={"acc": 0.8})
    assert select_evaluations(db, [r]) == {}
    assert select_evaluations(db, [r], attempt_selection="first_valid") == {}
    s = cell_statuses(db, [r])[r]
    assert (s.code, s.reason) == ("F", state)
    assert s.evaluation_id is None and s.evaluation_attempt_id is None and s.metrics == {}


def test_valid_evaluation_of_a_cancelled_or_unfinished_job_never_counts(db: Database) -> None:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        cancelled = make_run(tx, e, p, seed=1)
        a = make_attempt(tx, cancelled, phase="terminal", scheduler_state="CANCELLED", result_state="valid")
        make_evaluation(tx, a)
        # COMPLETED is only final once the attempt is terminal (a requeue reopens it)
        reopened = make_run(tx, e, p, seed=2)
        b = make_attempt(tx, reopened, phase="observing", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, b)
    assert select_evaluations(db, [cancelled, reopened]) == {}
    st = cell_statuses(db, [cancelled, reopened])
    assert st[cancelled].code == "X" and st[reopened].code != "V"


def test_a_later_timed_out_attempt_does_not_replace_a_completed_result(db: Database) -> None:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        done = make_attempt(tx, r, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        ev_done = make_evaluation(tx, done, metrics={"acc": 0.9})
        late = make_attempt(tx, r, phase="terminal", scheduler_state="TIMEOUT", result_state="valid")
        make_evaluation(tx, late, metrics={"acc": 0.7})
    for rule in ("latest_valid", "first_valid"):
        chosen = select_evaluations(db, [r], attempt_selection=rule)
        assert chosen[r]["evaluation_id"] == ev_done
    s = cell_statuses(db, [r])[r]
    assert s.code == "V" and s.evaluation_id == ev_done and s.metrics == {"acc": 0.9}
    assert s.reason is not None and late in s.reason and "TIMEOUT" in s.reason


# -- several valid evaluations of the counted attempt ----------------------------------------


def _counted_attempt(db: Database) -> tuple[str, str]:
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        r = make_run(tx, e, p)
        a = make_attempt(tx, r, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
    return r, a


def test_re_evaluating_the_counted_attempt_under_another_protocol_is_disclosed(db: Database) -> None:
    r, a = _counted_attempt(db)
    with db.tx() as tx:
        ev_a = make_evaluation(
            tx,
            a,
            metrics={"acc": 0.80},
            config_hash="sha256:" + "a" * 64,
            created_at="2026-01-01T00:00:00.000Z",
        )
        ev_b = make_evaluation(
            tx,
            a,
            metrics={"acc": 0.86},
            config_hash="sha256:" + "b" * 64,
            created_at="2026-01-02T00:00:00.000Z",
        )
        # never valid, so never a result that the counted one replaced
        make_evaluation(tx, a, status="partial", created_at="2026-01-01T12:00:00.000Z")
    assert select_evaluations(db, [r])[r]["evaluation_id"] == ev_b  # the newest valid one still counts
    s = cell_statuses(db, [r])[r]
    assert s.code == "V" and s.evaluation_id == ev_b and s.metrics == {"acc": 0.86}
    assert [x.evaluation_id for x in s.superseded] == [ev_a]
    old = s.superseded[0]
    assert (old.evaluator_id, old.evaluator_version, old.config_hash, old.created_at) == (
        "hpclab_metrics",
        "1",
        "sha256:" + "a" * 64,
        "2026-01-01T00:00:00.000Z",
    )
    assert old.metrics == {"acc": 0.80}
    assert s.reason is not None and ev_a in s.reason and ev_b in s.reason
    # pinned evaluation_refs are the caller's explicit choice: only pinned ones are considered
    pinned = cell_statuses(db, [r], evaluation_ids={ev_b})[r]
    assert pinned.evaluation_id == ev_b and pinned.superseded == () and pinned.reason is None
    both = cell_statuses(db, [r], evaluation_ids={ev_a, ev_b})[r]
    assert [x.evaluation_id for x in both.superseded] == [ev_a]


def test_superseded_respects_the_evaluator_filter_and_other_attempts(db: Database) -> None:
    r, a = _counted_attempt(db)
    with db.tx() as tx:
        other = make_evaluation(tx, a, evaluator_id="json_path", created_at="2026-01-01T00:00:00.000Z")
        ev = make_evaluation(tx, a, created_at="2026-01-02T00:00:00.000Z")
        # a valid evaluation of an EARLIER attempt is the multiple-valid-attempts disclosure, not this one
        a0 = make_attempt(tx, r, seq=0, phase="terminal", scheduler_state="COMPLETED", result_state="valid")
        make_evaluation(tx, a0, created_at="2026-01-03T00:00:00.000Z")
    filtered = cell_statuses(db, [r], evaluator_id="hpclab_metrics")[r]
    assert filtered.evaluation_id == ev and filtered.superseded == () and filtered.reason is None
    unfiltered = cell_statuses(db, [r])[r]
    assert unfiltered.evaluation_id == ev and [x.evaluation_id for x in unfiltered.superseded] == [other]


def test_both_disclosures_share_the_reason_and_a_non_completed_evaluation_is_never_superseded(
    db: Database,
) -> None:
    """The counted attempt is not the latest (a later job timed out) and was re-evaluated: the
    reason names both. The timed-out attempt's valid evaluation is neither counted nor listed as
    superseded (it was never a result)."""
    r, a = _counted_attempt(db)
    with db.tx() as tx:
        ev_a = make_evaluation(
            tx,
            a,
            metrics={"acc": 0.80},
            config_hash="sha256:" + "a" * 64,
            created_at="2026-01-01T00:00:00.000Z",
        )
        ev_b = make_evaluation(
            tx,
            a,
            metrics={"acc": 0.86},
            config_hash="sha256:" + "b" * 64,
            created_at="2026-01-02T00:00:00.000Z",
        )
        late = make_attempt(tx, r, phase="terminal", scheduler_state="TIMEOUT", result_state="valid")
        ev_late = make_evaluation(tx, late, metrics={"acc": 0.5}, created_at="2026-01-03T00:00:00.000Z")
    s = cell_statuses(db, [r])[r]
    assert s.code == "V" and s.evaluation_id == ev_b and s.evaluation_attempt_id == a
    assert [x.evaluation_id for x in s.superseded] == [ev_a]
    assert s.reason is not None
    assert late in s.reason and "TIMEOUT" in s.reason and ev_a in s.reason and ev_b in s.reason
    assert ev_late not in s.reason


class _RecordingReader:
    """A reader that records every query (cell_statuses must not read attempt blobs)."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.queries: list[str] = []

    def all(self, sql: str, params: Any = ()) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return self.db.all(sql, params)


def test_cell_statuses_reads_only_the_attempt_columns_it_needs(db: Database) -> None:
    ids = _setup(db)
    with db.tx() as tx:
        e = make_experiment(tx)
        p = make_plan(tx, e)
        ids["oom"] = make_run(tx, e, p, seed=11)
        make_attempt(
            tx,
            ids["oom"],
            phase="terminal",
            scheduler_state="FAILED",
            diagnosis_json='{"primary_code": "oom_gpu"}',
            env_json='{"big": "' + "x" * 4096 + '"}',
            status_json='{"schema": "hpclab.status/v1"}',
            health_json='{"schema": "hpclab.health/v1"}',
        )
    reader = _RecordingReader(db)
    expected = cell_statuses(db, list(ids.values()))
    st = cell_statuses(reader, list(ids.values()))
    assert st == expected
    assert st[ids["oom"]].code == "F" and st[ids["oom"]].reason == "oom_gpu"  # diagnosis_json is read
    attempt_queries = [q for q in reader.queries if "FROM attempts" in q]
    assert attempt_queries
    for q in attempt_queries:
        assert "*" not in q, q
        for blob in ("env_json", "status_json", "health_json", "last_observation_json"):
            assert blob not in q, (blob, q)
