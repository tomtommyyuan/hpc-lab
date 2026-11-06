"""services/audit.py: the read-only audit-trail listing behind `hpclab events`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from hpclab.config import Paths
from hpclab.domain.clock import FakeClock
from hpclab.domain.errors import InvalidArgument
from hpclab.services.app import build_registry
from hpclab.services.audit import MAX_PAYLOAD_CHARS, events_list
from hpclab.services.context import ServiceContext


@pytest.fixture
def ctx(paths: Paths, fake_clock: FakeClock) -> Iterator[ServiceContext]:
    c = ServiceContext.create(paths, clock=fake_clock)
    with c.db.tx(source="api") as tx:
        tx.event("attempt", "att_A", "attempt.created", {"seq": 1}, entity_version=1)
        tx.event("attempt", "att_A", "attempt.terminal", {"state": "COMPLETED"}, entity_version=2)
        tx.event("attempt", "att_B", "attempt.created", {"seq": 1})
        tx.event("run", "run_R", "run.status_changed", {"to": "running"})
        tx.event("workflow", "wf_W", "workflow_x.big", {"blob": "x" * (MAX_PAYLOAD_CHARS + 10)})
    yield c
    c.db.close()


def test_newest_first_with_payloads(ctx: ServiceContext) -> None:
    d = events_list(ctx)
    types = [e.event_type for e in d.events]
    assert types == [
        "workflow_x.big",
        "run.status_changed",
        "attempt.created",
        "attempt.terminal",
        "attempt.created",
    ]
    big = d.events[0]
    assert big.payload is None and big.payload_truncated  # bounded, and says so
    assert d.events[3].payload == {"state": "COMPLETED"} and d.events[3].entity_version == 2
    assert d.coverage.complete and d.next_cursor is None


def test_filters_by_entity_and_type_or_prefix(ctx: ServiceContext) -> None:
    assert [e.event_type for e in events_list(ctx, entity_id="att_A").events] == [
        "attempt.terminal",
        "attempt.created",
    ]
    assert len(events_list(ctx, event_type="attempt.created").events) == 2
    assert len(events_list(ctx, event_type="attempt.").events) == 3
    assert len(events_list(ctx, event_type="attempt.*").events) == 3
    # '_' in a prefix is literal, not a LIKE wildcard
    assert [e.event_type for e in events_list(ctx, event_type="workflow_x.").events] == ["workflow_x.big"]
    assert events_list(ctx, event_type="workflowXx.").events == []
    assert events_list(ctx, entity_id="att_nope").events == []


def test_cursor_pages_without_gaps(ctx: ServiceContext) -> None:
    first = events_list(ctx, limit=2)
    assert len(first.events) == 2 and first.next_cursor
    second = events_list(ctx, limit=2, cursor=first.next_cursor)
    third = events_list(ctx, limit=2, cursor=second.next_cursor)
    seqs = [e.seq for page in (first, second, third) for e in page.events]
    assert seqs == sorted(seqs, reverse=True) and len(set(seqs)) == 5
    assert third.next_cursor is None
    with pytest.raises(InvalidArgument):
        events_list(ctx, cursor="not-a-cursor")
    from hpclab.api.envelope import encode_cursor

    with pytest.raises(InvalidArgument):
        events_list(ctx, cursor=encode_cursor({"created_at": "x"}))


def test_registered_as_a_controller_method(ctx: ServiceContext) -> None:
    reg = build_registry(strict=True)
    method = reg.get("events_list")
    assert not method.mutation
    env = reg.dispatch(ctx, "events_list", {"entity_id": "run_R"}, request_id="req_1")
    assert env["ok"] and env["data"]["events"][0]["event_type"] == "run.status_changed"
    assert env["coverage"]["complete"] is True
