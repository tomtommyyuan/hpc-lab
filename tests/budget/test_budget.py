"""GPU-hour budget ledger (invariant 4.7): unknown is never 0, settled is never downgraded."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import BudgetExceeded, InvalidArgument, NotFound
from hpclab.services import budget
from hpclab.storage.db import Database
from hpclab.testing.factories import make_experiment


@pytest.fixture
def ledger(paths: Paths, fake_clock: FakeClock) -> Iterator[Database]:
    db = Database(paths.db, clock=fake_clock)
    yield db
    db.close_all()


def experiment(db: Database, budget_gpu_hours: float | None = 10.0) -> str:
    with db.tx() as tx:
        return make_experiment(tx, budget_gpu_hours=budget_gpu_hours)


def budget_events(db: Database, eid: str) -> list[str]:
    rows = db.all(
        "SELECT event_type FROM events WHERE entity_id=? AND event_type LIKE 'budget.%' ORDER BY seq", (eid,)
    )
    return [r["event_type"] for r in rows]


def test_reserve_counts_against_the_budget(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        rid = budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        assert budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0) == rid  # idempotent
    s = budget.summary(ledger, experiment_id=eid)
    assert (s.budget, s.reserved, s.available, s.active_count) == (10.0, 4.0, 6.0, 1)
    assert s.estimate_unknown_count == 1 and s.estimated_consumed is None  # unknown, not 0
    assert budget_events(ledger, eid) == ["budget.reserved"]
    with ledger.tx() as tx, pytest.raises(InvalidArgument):
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=5.0)


def test_settle_releases_the_difference(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        assert budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=1.5, note="sacct") == "settled"
    s = budget.summary(ledger, experiment_id=eid)
    assert (s.reserved, s.settled_consumed, s.available, s.settled_count) == (0.0, 1.5, 8.5, 1)
    payload = ledger.one("SELECT payload_json FROM events WHERE event_type='budget.settled'")
    assert payload is not None and '"released_gpu_hours":2.5' in payload["payload_json"]


def test_unknown_consumption_is_counted_at_its_reservation_never_zero(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        assert (
            budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=None, note="no accounting") == "unknown"
        )
    s = budget.summary(ledger, experiment_id=eid)
    assert s.unknown_count == 1 and s.unknown_reserved == 4.0
    assert s.settled_consumed == 0.0 and s.available == 6.0
    assert any("never as 0" in n for n in s.notes)
    with ledger.tx() as tx, pytest.raises(BudgetExceeded):
        budget.check_available(tx, experiment_id=eid, gpu_hours=6.5)
    # late accounting turns unknown into a real number
    with ledger.tx() as tx:
        assert budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=1.0, note="late sacct") == "settled"
    assert budget.summary(ledger, experiment_id=eid).available == 9.0


def test_settled_is_never_downgraded_to_unknown(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=2.0, note="sacct")
        assert budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=None, note="later poll") == "settled"
        assert budget.release(tx, attempt_id="att_a", note="oops") == "settled"
    s = budget.summary(ledger, experiment_id=eid)
    assert s.settled_consumed == 2.0 and s.unknown_count == 0
    assert budget_events(ledger, eid) == ["budget.reserved", "budget.settled"]


def test_release_frees_everything_once(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        assert budget.release(tx, attempt_id="att_a", note="rejected") == "released"
        assert budget.release(tx, attempt_id="att_a", note="again") == "released"
    s = budget.summary(ledger, experiment_id=eid)
    assert (s.reserved, s.available, s.released_count) == (0.0, 10.0, 1)
    assert budget_events(ledger, eid) == ["budget.reserved", "budget.released"]


def test_estimates_only_touch_active_reservations(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=4.0)
        budget.reserve(tx, experiment_id=eid, attempt_id="att_b", gpu_hours=2.0)
        assert budget.update_estimate(tx, attempt_id="att_a", estimated_gpu_hours=0.75) is True
        budget.release(tx, attempt_id="att_b", note="cancelled")
        assert budget.update_estimate(tx, attempt_id="att_b", estimated_gpu_hours=1.0) is False
    s = budget.summary(ledger, experiment_id=eid)
    assert s.estimated_consumed == 0.75 and s.estimate_unknown_count == 0
    assert s.reserved == 4.0  # the estimate never replaces the reservation while active


def test_no_budget_means_none_not_zero(ledger: Database) -> None:
    eid = experiment(ledger, budget_gpu_hours=None)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=400.0)
        budget.check_available(tx, experiment_id=eid, gpu_hours=10_000.0)
    s = budget.summary(ledger, experiment_id=eid)
    assert s.budget is None and s.available is None and s.reserved == 400.0


def test_check_available_boundary_and_details(ledger: Database) -> None:
    eid = experiment(ledger, budget_gpu_hours=4.0)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=3.0)
        budget.check_available(tx, experiment_id=eid, gpu_hours=1.0)  # exactly fits
        with pytest.raises(BudgetExceeded) as exc:
            budget.check_available(tx, experiment_id=eid, gpu_hours=1.01)
    assert exc.value.details["requested_gpu_hours"] == 1.01
    assert exc.value.details["summary"]["available"] == 1.0


def test_overdrawn_budget_is_reported_not_hidden(ledger: Database) -> None:
    eid = experiment(ledger, budget_gpu_hours=1.0)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=eid, attempt_id="att_a", gpu_hours=1.0)
        budget.settle(tx, attempt_id="att_a", consumed_gpu_hours=3.0, note="requeued twice")
    s = budget.summary(ledger, experiment_id=eid)
    assert s.available == -2.0 and any("overdrawn" in n for n in s.notes)


def test_invalid_amounts_and_missing_rows(ledger: Database) -> None:
    eid = experiment(ledger)
    with ledger.tx() as tx:
        for bad in (-1.0, float("nan"), float("inf")):
            with pytest.raises(InvalidArgument):
                budget.reserve(tx, experiment_id=eid, attempt_id="att_x", gpu_hours=bad)
        with pytest.raises(NotFound):
            budget.settle(tx, attempt_id="att_none", consumed_gpu_hours=1.0, note="n")
        assert budget.release(tx, attempt_id="att_none", note="n") is None
        assert budget.reservation(tx, attempt_id="att_none") is None
    with pytest.raises(NotFound):
        budget.summary(ledger, experiment_id="exp_missing")


def test_budgets_are_per_experiment(ledger: Database) -> None:
    a = experiment(ledger, budget_gpu_hours=2.0)
    b = experiment(ledger, budget_gpu_hours=2.0)
    with ledger.tx() as tx:
        budget.reserve(tx, experiment_id=a, attempt_id="att_a", gpu_hours=2.0)
        budget.check_available(tx, experiment_id=b, gpu_hours=2.0)
    assert budget.summary(ledger, experiment_id=b).available == 2.0
