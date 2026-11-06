"""Outbox lanes: latency-critical steps are not stuck behind long transfers.

Background work used to be leased strictly by ``not_before``: a ``cancel.execute`` enqueued
after a few long ``artifact.download`` items waited until every one of them finished on the
(two) controller workers, while the job kept burning GPU-hours. Now urgent kinds (cancel,
dispatch, reconcile) are leased first, and bulk kinds (transfers, indexing, capture) may only
occupy ``max_bulk_workers`` workers, so one worker is always free for everything else.
"""

from __future__ import annotations

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.services.context import ServiceContext
from hpclab.services.outbox import BULK_KINDS, URGENT_KINDS, OutboxItem, OutboxRunner

KINDS = ("artifact.download", "submit.stage", "observe.finalize", "cancel.execute", "workflow.cancel")


def _ctx(paths: Paths) -> ServiceContext:
    return ServiceContext.create(paths, clock=FakeClock())


def _runner(ctx: ServiceContext, worker_id: str, **kw: object) -> OutboxRunner:
    runner = OutboxRunner(ctx, worker_id=worker_id, **kw)  # type: ignore[arg-type]
    for kind in KINDS:
        runner.register(kind, lambda c, i: None)
    return runner


def _enqueue(ctx: ServiceContext, kind: str, n: int = 1) -> list[int]:
    out: list[int] = []
    with ctx.db.tx() as tx:
        for _ in range(n):
            oid = tx.enqueue(kind)
            assert oid is not None
            out.append(oid)
    return out


def test_kind_classes() -> None:
    assert {"cancel.execute", "workflow.cancel", "submit.dispatch", "submit.reconcile"} <= URGENT_KINDS
    assert {"artifact.download", "submit.stage", "artifacts.index", "vault.capture"} <= BULK_KINDS
    assert not URGENT_KINDS & BULK_KINDS


def test_kind_classes_name_real_kinds(paths: Paths) -> None:
    """A renamed kind would silently drop out of its lane: every class member is registered."""
    from hpclab.services.app import create_app

    registered = set(create_app(_ctx(paths)).outbox_handlers())
    assert not URGENT_KINDS - registered, URGENT_KINDS - registered
    assert not BULK_KINDS - registered, BULK_KINDS - registered


def test_workflow_dispatch_is_not_stuck_behind_a_finalize_backlog(paths: Paths) -> None:
    """A matrix wave that just finished enqueues hundreds of finalize items (ssh each); the
    ledger-only dispatch round that admits the next cells into the freed slots goes first."""
    ctx = _ctx(paths)
    _enqueue(ctx, "observe.finalize", 5)
    ctx.clock.advance(1)  # type: ignore[attr-defined]
    (dispatch,) = _enqueue(ctx, "workflow.dispatch")
    ctx.clock.advance(1)  # type: ignore[attr-defined]
    runner = _runner(ctx, "w0")
    runner.register("workflow.dispatch", lambda c, i: None)

    assert [i.outbox_id for i in runner._lease(1)] == [dispatch]


def test_cancel_is_leased_before_earlier_downloads(paths: Paths) -> None:
    ctx = _ctx(paths)
    _enqueue(ctx, "artifact.download", 4)
    _enqueue(ctx, "observe.finalize", 2)
    ctx.clock.advance(1)  # type: ignore[attr-defined]
    (cancel,) = _enqueue(ctx, "cancel.execute")
    ctx.clock.advance(1)  # type: ignore[attr-defined]

    leased = _runner(ctx, "w0")._lease(1)

    assert [i.outbox_id for i in leased] == [cancel]


def test_bulk_work_leaves_a_worker_free(paths: Paths) -> None:
    ctx = _ctx(paths)
    downloads = _enqueue(ctx, "artifact.download", 3)
    w0 = _runner(ctx, "w0", max_bulk_workers=1)
    w1 = _runner(ctx, "w1", max_bulk_workers=1)

    first = w0._lease(1)  # w0 is now busy with a (long) transfer
    assert [i.outbox_id for i in first] == [downloads[0]]
    assert w1._lease(1) == []  # the other worker does not start a second one ...

    stage = _enqueue(ctx, "submit.stage")[0]
    assert w1._lease(1) == []  # (staging is bulk as well)
    (cancel,) = _enqueue(ctx, "cancel.execute")
    finalize = _enqueue(ctx, "observe.finalize")[0]
    assert [i.outbox_id for i in w1._lease(1)] == [cancel]  # ... and is free for this at once
    assert [i.outbox_id for i in w1._lease(1)] == [finalize]

    w0._finish(first[0], "done")
    assert [i.outbox_id for i in w1._lease(1)] == [downloads[1]]  # the bulk slot is free again
    assert stage not in [i.outbox_id for i in w0._lease(1)]


def test_expired_bulk_leases_do_not_hold_the_slot(paths: Paths) -> None:
    ctx = _ctx(paths)
    _enqueue(ctx, "artifact.download", 2)
    w0 = _runner(ctx, "w0", max_bulk_workers=1, lease_s=30.0)
    w1 = _runner(ctx, "w1", max_bulk_workers=1, lease_s=30.0)
    assert len(w0._lease(1)) == 1
    assert w1._lease(1) == []
    ctx.clock.advance(31)  # type: ignore[attr-defined]  # w0 died: its lease expired
    assert len(w1._lease(1)) == 1


def test_without_a_cap_bulk_work_is_not_limited(paths: Paths) -> None:
    """One-shot runners (CLI drain, tests) keep their old behaviour."""
    ctx = _ctx(paths)
    ids = _enqueue(ctx, "artifact.download", 3)
    seen: list[int] = []

    def handler(c: ServiceContext, item: OutboxItem) -> None:
        seen.append(item.outbox_id)

    runner = OutboxRunner(ctx)
    runner.register("artifact.download", handler)
    assert runner.drain() == 3 and seen == ids
