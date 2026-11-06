"""Notifications: batched, deduplicated, secret-free messages to the researcher's own channels.

Flow
----
* ``emit(tx, ...)`` runs inside the caller's transaction (e.g. ``observe.finalize``). For every
  configured channel (``notify:`` entries of the registered clusters' overlays) whose ``events``
  include the event type it writes one ``notifications`` row. No channels -> no rows.
* Action-required classes (``IMMEDIATE_EVENTS``: uncertain submission, auth required, walltime
  will time out, breaker tripped, budget threshold) or ``urgency="immediate"`` are queued at once
  (outbox ``notify.send``); everything else waits in ``pending`` and the periodic
  ``notify.flush`` sends one combined message per channel once the oldest pending row is older
  than the batching window (30 min default, ``settings.extra["notify_batch_window_s"]``).
* Dedupe: while a notification with the same key (default ``<event>:<entity>``) is unsent on a
  channel, a repeat is dropped; once sent, the same key may notify again.
* ``notify.send`` delivers (network outside any transaction; urllib/subprocess with timeouts),
  then records ``sent``, or retries with backoff and finally marks ``failed``. Delivery is
  at-least-once: a crash between delivery and the ledger update re-sends.

Row states: ``pending`` (waiting for the batch window) -> ``queued`` (handed to the outbox) ->
``sent`` | ``failed``.

Hygiene: payloads carry ids, event types, states and labels only. Titles and bodies are reduced
to one line; paths, URLs, secrets, float values (metrics) and long tokens are redacted. Channel
targets (webhook URLs may embed secrets) never enter the ledger, logs or errors: rows reference a
channel by ``<kind>:<hash>``, the stored cluster overlays hold only that id with the channel's kind and
events, and the target is looked up (``ClusterRegistry.notify_channels``: memory, else the 0600
overlay file) just before sending.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hpclab.domain.clock import iso
from hpclab.domain.errors import InvalidArgument
from hpclab.domain.ids import new_id
from hpclab.services.context import ServiceContext, notify_channel_id, notify_channel_refs
from hpclab.services.outbox import Handler, HandlerResult, OutboxItem
from hpclab.storage.db import Tx, jload

log = logging.getLogger("hpclab.notify")

SCHEMA = "hpclab.notification/v1"
DEFAULT_BATCH_WINDOW_S = 1800.0
FLUSH_INTERVAL_S = 60.0
SEND_TIMEOUT_S = 10.0
MAX_TRIES = 5
RETRY_BASE_S = 30.0
# A 'queued' row untouched this long (retries touch it every <= 4 min) whose outbox item is no
# longer pending/leased lost its carrier (the item died): flush hands it to a new item.
ORPHAN_AFTER_S = 3600.0
MAX_BATCH = 200
MAX_ITEMS_IN_MESSAGE = 50
TITLE_MAX = 120
BODY_MAX = 300

# Canonical event names that must reach the human now (everything else is batched).
IMMEDIATE_EVENTS = frozenset(
    {
        "attempt.uncertain",
        "auth.required",
        "walltime.will_timeout",
        "workflow.breaker_tripped",
        "budget.threshold",
    }
)
_ALIASES = {
    "submission_uncertain": "attempt.uncertain",
    "submission.uncertain": "attempt.uncertain",
    "attempt_uncertain": "attempt.uncertain",
    "attempt.submission_uncertain": "attempt.uncertain",
    "auth_required": "auth.required",
    "cluster.auth_required": "auth.required",
    "walltime_will_timeout": "walltime.will_timeout",
    "attempt.walltime_will_timeout": "walltime.will_timeout",
    "will_timeout": "walltime.will_timeout",
    "breaker_tripped": "workflow.breaker_tripped",
    "workflow_breaker_tripped": "workflow.breaker_tripped",
    "budget_threshold": "budget.threshold",
}

Urgency = Literal["batch", "immediate"]


class _Reader(Protocol):
    def all(self, sql: str, params: Any = ...) -> list[dict[str, Any]]: ...


def normalize_event(event_type: str) -> str:
    e = (event_type or "").strip().lower()
    return _ALIASES.get(e, e)


# ---------------------------------------------------------------------------
# Payload hygiene
# ---------------------------------------------------------------------------

_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(token|secret|password|passwd|api[_-]?key|authorization|bearer)\b\s*[=:]?\s*\S+", re.IGNORECASE
)
_ABS_PATH_RE = re.compile(r"(?<![\w.~])(?:~|\.{1,2})?/[^\s,;'\"()\[\]{}<>]+")
_REL_PATH_RE = re.compile(r"\b(?=[\w.-]*[A-Za-z])[\w.-]+(?:/[\w.-]+)+")
_FLOAT_KV_RE = re.compile(r"\b([A-Za-z_][\w.]*)\s*[=:]\s*[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
_FLOAT_RE = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?(?![\w.])")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")


def sanitize_text(text: Any, max_len: int) -> str:
    """One line, no paths/URLs/secrets/float values/long tokens, at most ``max_len`` chars."""
    t = _CTRL_RE.sub(" ", str(text if text is not None else ""))
    t = _URL_RE.sub("<url>", t)
    t = _SECRET_RE.sub(lambda m: f"{m.group(1)}=<redacted>", t)
    t = _ABS_PATH_RE.sub("<path>", t)
    t = _REL_PATH_RE.sub("<path>", t)
    t = _FLOAT_KV_RE.sub(lambda m: f"{m.group(1)}=<value>", t)
    t = _FLOAT_RE.sub("<num>", t)
    t = _LONG_TOKEN_RE.sub("<redacted>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    channel_id: str  # "<kind>:<hash of target>": the target itself never enters the ledger
    kind: str
    target: str | None
    events: tuple[str, ...]
    clusters: tuple[str, ...]

    def wants(self, event_type: str) -> bool:
        ev = normalize_event(event_type)
        for raw in self.events:
            pat = normalize_event(raw)
            if pat in ("*", "all") or pat == ev or (pat.endswith(".*") and ev.startswith(pat[:-1])):
                return True
        return False

    def target_hint(self) -> str:
        """What a human may see about the target (no secrets): URL host, command name."""
        if not self.target:
            return self.kind
        if self.kind in ("ntfy", "slack_webhook"):
            parts = urllib.parse.urlsplit(self.target)
            return f"{parts.scheme}://{parts.hostname or '?'}/…"
        if self.kind == "command":
            try:
                argv = shlex.split(self.target)
            except ValueError:
                return "command"
            return os.path.basename(argv[0]) if argv else "command"
        return self.kind


def channel_id(kind: str, target: str | None) -> str:
    return notify_channel_id(kind, target)


def configured_channels(reader: _Reader) -> list[Channel]:
    """Channels from every registered cluster overlay, merged by channel id (= kind + target).

    The ledger holds no targets: ``target`` is None here; ``with_target`` adds it to send."""
    merged: dict[str, Channel] = {}
    for row in reader.all("SELECT cluster_id, overlay_json FROM clusters ORDER BY cluster_id"):
        overlay = jload(row["overlay_json"], {}) or {}
        for ref in notify_channel_refs(overlay, row["cluster_id"]):
            cid = ref.channel_id
            prev = merged.get(cid)
            merged[cid] = Channel(
                channel_id=cid,
                kind=ref.kind,
                target=None,
                events=tuple(sorted(set(ref.events) | set(prev.events if prev else ()))),
                clusters=tuple(sorted({*(prev.clusters if prev else ()), row["cluster_id"]})),
            )
    return list(merged.values())


TARGET_UNAVAILABLE = (
    "the channel target is not in the cluster overlay file any more; fix the file or re-register "
    "the cluster (hpclab cluster add ...)"
)


def with_target(ctx: ServiceContext, channel: Channel) -> Channel | None:
    """``channel`` with its target from the cluster registry (never from the ledger), or None
    when no cluster that lists the channel can supply it any more."""
    for cluster_id in channel.clusters:
        for nc in ctx.clusters.notify_channels(cluster_id):
            if channel_id(nc.kind, nc.target) == channel.channel_id:
                return replace(channel, target=nc.target)
    return None


# ---------------------------------------------------------------------------
# emit (inside the caller's transaction)
# ---------------------------------------------------------------------------


def emit(
    tx: Tx,
    event_type: str,
    entity_id: str,
    title: str,
    body: str,
    urgency: Urgency = "batch",
    dedupe_key: str | None = None,
) -> list[str]:
    """Record a notification for every channel subscribed to ``event_type``; returns the new
    notification ids ([] when no channel wants it or it is a duplicate of an unsent one).

    Safe to call when nothing is configured. Writes only rows, events and outbox items in
    ``tx``; never performs I/O.
    """
    ev = normalize_event(event_type)
    if not ev or not str(entity_id or "").strip():
        raise InvalidArgument("notify.emit needs an event_type and an entity_id")
    if urgency not in ("batch", "immediate"):
        raise InvalidArgument(f"urgency must be 'batch' or 'immediate', not {urgency!r}")
    channels = [c for c in configured_channels(tx) if c.wants(ev)]
    if not channels:
        return []
    immediate = urgency == "immediate" or ev in IMMEDIATE_EVENTS
    key_base = dedupe_key or f"{ev}:{entity_id}"
    clean_title = sanitize_text(title, TITLE_MAX) or ev
    clean_body = sanitize_text(body, BODY_MAX)
    created: list[str] = []
    for ch in channels:
        dkey = f"{key_base}@{ch.channel_id}"
        existing = tx.one("SELECT notification_id, state FROM notifications WHERE dedupe_key=?", (dkey,))
        if existing is not None:
            if existing["state"] in ("pending", "queued"):
                continue  # the same news is already waiting to go out on this channel
            tx.execute(
                "UPDATE notifications SET dedupe_key=NULL WHERE notification_id=?",
                (existing["notification_id"],),
            )
        nid = new_id("notification")
        tx.insert(
            "notifications",
            {
                "notification_id": nid,
                "event_type": ev,
                "entity_id": str(entity_id),
                "channel": ch.channel_id,
                "title": clean_title,
                "body": clean_body,
                "state": "queued" if immediate else "pending",
                "tries": 0,
                "last_error": None,
                "dedupe_key": dkey,
                "created_at": tx.now,
                "updated_at": tx.now,
            },
        )
        tx.event(
            "notification",
            nid,
            "notification.created",
            {
                "event_type": ev,
                "entity_id": str(entity_id),
                "channel": ch.channel_id,
                "urgency": "immediate" if immediate else "batch",
            },
        )
        if immediate:
            tx.enqueue(
                "notify.send",
                target_id=nid,
                payload={"notification_ids": [nid], "urgency": "immediate"},
                dedupe_key=f"notify.send:{nid}",
            )
        created.append(nid)
    return created


# ---------------------------------------------------------------------------
# flush (periodic) and send (outbox)
# ---------------------------------------------------------------------------


def flush(ctx: ServiceContext, *, force: bool = False) -> int:
    """Queue one combined message per channel whose oldest pending notification has waited
    the batching window (``force``: regardless of age), and re-queue orphaned rows (see
    ``requeue_orphans``). Returns how many rows were queued."""
    queued = requeue_orphans(ctx)
    window = float(ctx.settings.extra.get("notify_batch_window_s", DEFAULT_BATCH_WINDOW_S))
    cutoff = iso(ctx.clock.now() - timedelta(seconds=window)) or ""
    due = [
        r["channel"]
        for r in ctx.db.all(
            "SELECT channel, MIN(created_at) AS oldest FROM notifications WHERE state='pending' GROUP BY channel"
        )
        if force or r["oldest"] <= cutoff
    ]
    for channel in due:
        with ctx.db.tx(source="worker") as tx:
            ids = [
                r["notification_id"]
                for r in tx.all(
                    "SELECT notification_id FROM notifications WHERE state='pending' AND channel=? "
                    "ORDER BY created_at, notification_id",
                    (channel,),
                )
            ]
            for i in range(0, len(ids), MAX_BATCH):
                batch = ids[i : i + MAX_BATCH]
                marks = ",".join("?" for _ in batch)
                tx.execute(
                    f"UPDATE notifications SET state='queued', updated_at=? WHERE state='pending' "
                    f"AND notification_id IN ({marks})",
                    (tx.now, *batch),
                )
                for nid in batch:
                    tx.event(
                        "notification", nid, "notification.batched", {"channel": channel, "batch": batch[0]}
                    )
                tx.enqueue(
                    "notify.send",
                    target_id=batch[0],
                    payload={"notification_ids": batch, "urgency": "batch"},
                    dedupe_key=f"notify.send:{batch[0]}",
                )
                queued += len(batch)
    return queued


def requeue_orphans(ctx: ServiceContext) -> int:
    """At-least-once: 'queued' rows whose ``notify.send`` item died (the runner gives up on a
    handler that crashed 3 times) would otherwise stay queued forever - never delivered, and
    their dedupe key would swallow every later notification of the same event. Rows still
    carried by a pending/leased item are left alone (no duplicates)."""
    cutoff = iso(ctx.clock.now() - timedelta(seconds=ORPHAN_AFTER_S)) or ""
    stale = ctx.db.all(
        "SELECT notification_id, channel, event_type FROM notifications WHERE state='queued' "
        "AND updated_at <= ? ORDER BY created_at, notification_id LIMIT ?",
        (cutoff, MAX_BATCH * 5),
    )
    if not stale:
        return 0
    carried: set[str] = set()
    for item in ctx.db.all(
        "SELECT payload_json FROM outbox WHERE kind='notify.send' AND state IN ('pending', 'leased')"
    ):
        payload = jload(item["payload_json"], {}) or {}
        carried.update(str(x) for x in payload.get("notification_ids") or [])
        if payload.get("notification_id"):
            carried.add(str(payload["notification_id"]))
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for r in stale:
        if r["notification_id"] not in carried:
            by_channel.setdefault(r["channel"], []).append(r)
    requeued = 0
    for channel, rows in by_channel.items():
        with ctx.db.tx(source="worker") as tx:
            for i in range(0, len(rows), MAX_BATCH):
                part = rows[i : i + MAX_BATCH]
                marks = ",".join("?" for _ in part)
                ids = [
                    x["notification_id"]
                    for x in tx.all(  # re-check inside the tx: a sender may have finished meanwhile
                        f"SELECT notification_id FROM notifications WHERE state='queued' AND updated_at <= ? "
                        f"AND notification_id IN ({marks}) ORDER BY created_at, notification_id",
                        (cutoff, *(x["notification_id"] for x in part)),
                    )
                ]
                if not ids:
                    continue
                urgent = any(normalize_event(x["event_type"]) in IMMEDIATE_EVENTS for x in part)
                tx.execute(
                    f"UPDATE notifications SET updated_at=? WHERE notification_id IN ({','.join('?' for _ in ids)})",
                    (tx.now, *ids),
                )
                for nid in ids:
                    tx.event("notification", nid, "notification.requeued", {"channel": channel})
                tx.enqueue(
                    "notify.send",
                    target_id=ids[0],
                    payload={"notification_ids": ids, "urgency": "immediate" if urgent else "batch"},
                    dedupe_key=f"notify.send:{ids[0]}",
                )
                requeued += len(ids)
    if requeued:
        log.warning("re-queued %d notification(s) whose send item was lost", requeued)
    return requeued


def build_message(rows: Sequence[dict[str, Any]], *, urgency: str, created_at: str | None) -> dict[str, Any]:
    """The JSON a channel receives: ids, event types, states and labels only."""
    items = [
        {
            "notification_id": r["notification_id"],
            "event_type": r["event_type"],
            "entity_id": r["entity_id"],
            "title": sanitize_text(r["title"], TITLE_MAX),
            "body": sanitize_text(r["body"], BODY_MAX),
            "created_at": r["created_at"],
        }
        for r in rows[:MAX_ITEMS_IN_MESSAGE]
    ]
    if len(rows) == 1:
        title, body = items[0]["title"], items[0]["body"]
    else:
        counts = Counter(r["event_type"] for r in rows)
        title = f"hpclab: {len(rows)} updates"
        body = ", ".join(f"{ev} x{n}" for ev, n in sorted(counts.items()))
    return {
        "schema": SCHEMA,
        "urgency": urgency,
        "count": len(rows),
        "title": title,
        "body": body,
        "created_at": created_at,
        "notification_ids": [r["notification_id"] for r in rows],
        "items": items,
        "items_truncated": max(0, len(rows) - MAX_ITEMS_IN_MESSAGE),
    }


def handle_send(ctx: ServiceContext, item: OutboxItem) -> HandlerResult:
    """Outbox ``notify.send`` {notification_ids, urgency}: deliver, then record the outcome."""
    ids = list(item.payload.get("notification_ids") or [])
    if not ids and item.payload.get("notification_id"):
        ids = [str(item.payload["notification_id"])]
    urgency = str(item.payload.get("urgency") or "batch")
    rows = []
    for nid in ids:
        row = ctx.db.one("SELECT * FROM notifications WHERE notification_id=?", (nid,))
        if row is not None and row["state"] == "queued":
            rows.append(row)
    if not rows:
        return HandlerResult.done("nothing left to send")
    channels = {c.channel_id: c for c in configured_channels(ctx.db)}
    channel = channels.get(rows[0]["channel"])
    tries = max(int(r["tries"]) for r in rows) + 1
    error: str | None = None
    if channel is None:
        error, tries = "the channel is no longer configured", MAX_TRIES
    elif (channel := with_target(ctx, channel)) is None:
        error = TARGET_UNAVAILABLE  # retried: the overlay file may be fixed meanwhile
    else:
        message = build_message(rows, urgency=urgency, created_at=iso(ctx.clock.now()))
        try:
            send_to_channel(ctx, channel, message)
        except NotifyError as exc:
            error = str(exc)
        except Exception as exc:  # a broken channel must not kill the worker
            log.exception("notification channel %s crashed", channel.channel_id)
            error = f"{type(exc).__name__} while sending"
    final = error is None or tries >= MAX_TRIES
    state = "sent" if error is None else ("failed" if final else "queued")
    with ctx.db.tx(source="worker") as tx:
        for r in rows:
            tx.execute(
                "UPDATE notifications SET state=?, tries=?, last_error=?, updated_at=? "
                "WHERE notification_id=? AND state='queued'",
                (state, tries, error, tx.now, r["notification_id"]),
            )
            tx.event(
                "notification",
                r["notification_id"],
                {"sent": "notification.sent", "failed": "notification.failed"}.get(
                    state, "notification.retry"
                ),
                {"channel": r["channel"], "tries": tries, "error": error},
            )
    if final:
        return HandlerResult.done(error)
    return HandlerResult.retry(RETRY_BASE_S * 2 ** (tries - 1), note=error)


# ---------------------------------------------------------------------------
# Channel senders (network/subprocess I/O; never inside a transaction)
# ---------------------------------------------------------------------------


class NotifyError(Exception):
    """Delivery failed; the message never contains the channel target."""


def send_to_channel(ctx: ServiceContext, channel: Channel, message: dict[str, Any]) -> None:
    if channel.kind == "log":
        _send_log(ctx, message)
    elif channel.kind == "command":
        _send_command(channel.target, message)
    elif channel.kind == "ntfy":
        _send_ntfy(channel.target, message)
    elif channel.kind == "slack_webhook":
        _send_slack(channel.target, message)
    elif channel.kind == "desktop":
        _send_desktop(message)
    else:
        raise NotifyError(f"unknown channel kind {channel.kind!r}")


def _send_log(ctx: ServiceContext, message: dict[str, Any]) -> None:
    path = ctx.paths.home / "logs" / "notifications.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, sort_keys=True) + "\n")
    except OSError as exc:
        raise NotifyError(f"cannot append to the notification log: {exc.strerror}") from exc
    log.info("notification: %s - %s", message["title"], message["body"])


def _send_command(target: str | None, message: dict[str, Any]) -> None:
    try:
        argv = shlex.split(target or "")
    except ValueError as exc:
        raise NotifyError("the command target cannot be parsed") from exc
    if not argv:
        raise NotifyError("the command channel has no command")
    try:
        proc = subprocess.run(
            argv,
            input=json.dumps(message, sort_keys=True).encode("utf-8"),
            capture_output=True,
            timeout=SEND_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise NotifyError(f"notification command not found: {os.path.basename(argv[0])}") from exc
    except subprocess.TimeoutExpired as exc:
        raise NotifyError(f"notification command timed out after {SEND_TIMEOUT_S:g}s") from exc
    except OSError as exc:
        raise NotifyError(f"notification command failed to start: {exc.strerror}") from exc
    if proc.returncode != 0:
        tail = sanitize_text(proc.stderr.decode("utf-8", "replace")[-300:], 200)
        raise NotifyError(f"notification command exited {proc.returncode}: {tail}")


def _post(url: str | None, data: bytes, headers: dict[str, str]) -> None:
    if not url:
        raise NotifyError("the channel has no URL")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise NotifyError("the channel URL must be http(s)://host/...")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")  # noqa: S310 - scheme checked
    try:
        with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as resp:  # noqa: S310 - scheme checked
            status = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raise NotifyError(f"HTTP {exc.code} from {parts.hostname}") from exc
    except urllib.error.URLError as exc:
        raise NotifyError(f"cannot reach {parts.hostname}: {sanitize_text(exc.reason, 120)}") from exc
    except (TimeoutError, OSError) as exc:
        raise NotifyError(f"cannot reach {parts.hostname}: {type(exc).__name__}") from exc
    if status >= 300:
        raise NotifyError(f"HTTP {status} from {parts.hostname}")


def _header(text: str) -> str:
    """HTTP headers are latin-1: keep them ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


def _send_ntfy(target: str | None, message: dict[str, Any]) -> None:
    _post(
        target,
        str(message["body"] or message["title"]).encode("utf-8"),
        {
            "Title": _header(message["title"]),
            "Priority": "high" if message["urgency"] == "immediate" else "default",
            "Tags": "hpclab",
            "Content-Type": "text/plain; charset=utf-8",
        },
    )


def _send_slack(target: str | None, message: dict[str, Any]) -> None:
    text = f"*{message['title']}*" + (f"\n{message['body']}" if message["body"] else "")
    _post(target, json.dumps({"text": text}).encode("utf-8"), {"Content-Type": "application/json"})


def _applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _send_desktop(message: dict[str, Any]) -> None:
    title, body = str(message["title"]), str(message["body"] or "")
    if sys.platform == "darwin" and shutil.which("osascript"):
        argv = [
            "osascript",
            "-e",
            f"display notification {_applescript(body)} with title {_applescript(title)}",
        ]
    elif shutil.which("notify-send"):
        argv = ["notify-send", "--app-name=hpclab", title, body]
    else:
        raise NotifyError("desktop notifications are not supported here (no osascript / notify-send)")
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=SEND_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NotifyError(f"desktop notification failed: {type(exc).__name__}") from exc
    if proc.returncode != 0:
        raise NotifyError(f"desktop notification exited {proc.returncode}")


# ---------------------------------------------------------------------------
# notify_test
# ---------------------------------------------------------------------------


class ChannelTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel_id: str
    kind: str
    target: str  # a hint only (host / command name), never the secret URL
    clusters: list[str]
    events: list[str]
    ok: bool
    error: str | None = None


class NotifyTestData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channels: list[ChannelTestResult] = Field(default_factory=list)
    sent: int = 0
    failed: int = 0
    note: str | None = None
    observed_at: str | None = None

    def summary(self) -> str:
        if not self.channels:
            return "no notification channels configured"
        return f"test notification: {self.sent} channel(s) ok, {self.failed} failed"


def notify_test(ctx: ServiceContext, *, cluster_id: str | None = None) -> NotifyTestData:
    """Send a test message to every configured channel (of ``cluster_id``, or all) right now
    and report per-channel success. The message carries no job data."""
    if cluster_id is not None:
        ctx.clusters.get(cluster_id)  # NotFound for an unknown cluster
    channels = [c for c in configured_channels(ctx.db) if cluster_id is None or cluster_id in c.clusters]
    now = iso(ctx.clock.now())
    if not channels:
        return NotifyTestData(
            note="no notification channels configured: add `notify:` entries (kind: desktop|ntfy|"
            "slack_webhook|command|log, target, events) to the cluster overlay",
            observed_at=now,
        )
    message = {
        "schema": SCHEMA,
        "urgency": "immediate",
        "count": 1,
        "title": "hpclab test notification",
        "body": "If you can read this, hpclab can reach this channel.",
        "created_at": now,
        "notification_ids": [],
        "items": [],
        "items_truncated": 0,
        "test": True,
    }
    results: list[ChannelTestResult] = []
    for ch in channels:
        error: str | None = None
        sendable = with_target(ctx, ch)
        try:
            if sendable is None:
                raise NotifyError(TARGET_UNAVAILABLE)
            send_to_channel(ctx, sendable, message)
        except NotifyError as exc:
            error = str(exc)
        except Exception as exc:  # report, do not crash the request
            log.exception("notification channel %s crashed", ch.channel_id)
            error = f"{type(exc).__name__} while sending"
        results.append(
            ChannelTestResult(
                channel_id=ch.channel_id,
                kind=ch.kind,
                target=(sendable or ch).target_hint(),
                clusters=list(ch.clusters),
                events=list(ch.events),
                ok=error is None,
                error=error,
            )
        )
    with ctx.db.tx(source="api") as tx:
        tx.event(
            "notification",
            cluster_id or "all",
            "notify.tested",
            {
                "channels": [r.channel_id for r in results],
                "failed": [r.channel_id for r in results if not r.ok],
            },
        )
    ok = sum(1 for r in results if r.ok)
    return NotifyTestData(channels=results, sent=ok, failed=len(results) - ok, observed_at=now)


# ---------------------------------------------------------------------------
# Controller wiring
# ---------------------------------------------------------------------------


def outbox_handlers() -> dict[str, Handler]:
    return {"notify.send": handle_send}


def periodic_tasks() -> list[Any]:
    from hpclab.services.periodic import PeriodicTask

    return [PeriodicTask("notify.flush", FLUSH_INTERVAL_S, flush_task, jitter_s=5.0)]


def flush_task(ctx: ServiceContext) -> None:
    """Periodic ``notify.flush``: send batches whose window has elapsed."""
    flush(ctx)
