"""Read-only view of the audit trail (the append-only ``events`` table).

Every state change in the ledger writes an ``events`` row in the same transaction (spec §4.9),
so this listing answers "what happened to this attempt/run/workflow, when, and who did it".
Nothing here writes; payloads are bounded so one noisy event cannot blow up a page.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from hpclab.api.envelope import Coverage, clamp_limit, decode_cursor, encode_cursor
from hpclab.domain.clock import iso
from hpclab.domain.errors import InvalidArgument
from hpclab.services.context import ServiceContext
from hpclab.storage.db import jload

__all__ = ["EventItem", "EventsListData", "events_list"]

MAX_PAYLOAD_CHARS = 2000


class EventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    event_id: str
    entity_type: str
    entity_id: str
    entity_version: int | None
    event_type: str
    payload: dict[str, Any] | None
    payload_truncated: bool
    source: str
    command_id: str | None
    causation_id: str | None
    observed_at: str | None
    recorded_at: str


class EventsListData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventItem]
    entity_id: str | None
    event_type: str | None
    next_cursor: str | None
    coverage: Coverage


def _payload(raw: str | None) -> tuple[dict[str, Any] | None, bool]:
    if raw is None:
        return {}, False
    if len(raw) <= MAX_PAYLOAD_CHARS:
        value = jload(raw, {})
        return (value if isinstance(value, dict) else {"value": value}), False
    return None, True


def events_list(
    ctx: ServiceContext,
    *,
    entity_id: str | None = None,
    event_type: str | None = None,
    cursor: str | None = None,
    limit: int | None = None,
) -> EventsListData:
    """Audit events, newest first. ``entity_id`` = exact id (attempt, run, workflow, plan,
    cluster ...); ``event_type`` = exact type (e.g. ``attempt.terminal``) or a prefix ending in
    ``.`` / ``*`` (``attempt.`` or ``attempt.*``). Payloads above 2000 characters are omitted
    (``payload_truncated``)."""
    n = clamp_limit(limit)
    where: list[str] = []
    params: list[Any] = []
    if entity_id:
        where.append("entity_id = ?")
        params.append(entity_id)
    if event_type:
        prefix = event_type[:-1] if event_type.endswith("*") else event_type
        if prefix.endswith("."):
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("event_type LIKE ? ESCAPE '\\'")
            params.append(escaped + "%")
        else:
            where.append("event_type = ?")
            params.append(prefix)
    cur = decode_cursor(cursor)
    if cur:
        before = cur.get("seq")
        if not isinstance(before, int) or cur.get("k") != "events":
            raise InvalidArgument("malformed cursor (pass the next_cursor of a previous events_list call)")
        where.append("seq < ?")
        params.append(before)
    sql = "SELECT * FROM events" + (f" WHERE {' AND '.join(where)}" if where else "")
    rows = ctx.db.all(sql + " ORDER BY seq DESC LIMIT ?", (*params, n + 1))
    more = len(rows) > n
    rows = rows[:n]
    items = []
    for r in rows:
        payload, truncated = _payload(r["payload_json"])
        items.append(
            EventItem(
                seq=int(r["seq"]),
                event_id=r["event_id"],
                entity_type=r["entity_type"],
                entity_id=r["entity_id"],
                entity_version=r["entity_version"],
                event_type=r["event_type"],
                payload=payload,
                payload_truncated=truncated,
                source=r["source"],
                command_id=r["command_id"],
                causation_id=r["causation_id"],
                observed_at=r["observed_at"],
                recorded_at=r["recorded_at"],
            )
        )
    nxt = encode_cursor({"k": "events", "seq": int(rows[-1]["seq"])}) if more and rows else None
    cov = Coverage()
    cov.add("ledger", "ok", observed_at=iso(ctx.clock.now()))
    return EventsListData(
        events=items, entity_id=entity_id, event_type=event_type, next_cursor=nxt, coverage=cov
    )
